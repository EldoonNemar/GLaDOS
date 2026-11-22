"""
glados_hub - WebSocket hub: realtime streaming ASR + decision + response + streaming TTS.

Pipeline per spoke connection:

  spoke --16 kHz PCM (0x01 frames, realtime)--> StreamingASR (Vosk, stateful)
      partial  -> live text (debug / wake pre-alert)
      final    -> decision:  wake word in text?      -> respond
                                      else tiny LLM (YES/NO) -> respond?
                  -> response LLM (Gemini primary, local llama fallback)
                  -> streaming TTS (engine WS /tts/stream)
                  -> spoke (0x11 audio frames relayed as they are synthesized)

Barge-in: sustained user speech while GLaDOS is speaking cancels the TTS
stream and tells the spoke to stop playback (0x12).

Wire protocol (1-byte frame type + payload):
  up:   0x01 audio (int16 PCM 16 kHz mono)
        0x02 wake word text (utf-8)
        0x03 session start
        0x04 session end
  down: 0x11 audio (int16 PCM 24 kHz mono, TTS)
        0x12 stop playback
        0x13 status text (utf-8, null-padded to 240 bytes)
        0x14 heartbeat
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from . import cameras
from . import decision
from . import kasa_lights
from . import llm as llm_client
from . import skills
from . import timers
from . import weather
from .session import CONTEXT_TTL_S, Session
from .streaming_asr import ASREngine, ASREventKind

logger = logging.getLogger("glados_hub")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ── Configuration ────────────────────────────────────────────────────
HUB_PORT = int(os.environ.get("HUB_PORT", "5051"))
REST_PORT = int(os.environ.get("REST_PORT", "5050"))
ENGINE_URL = os.environ.get("ENGINE_URL", "http://glados_engine:8081")
ENGINE_WS_URL = ENGINE_URL.replace("http://", "ws://").replace("https://", "wss://")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Barge-in tuning: sustained speech above threshold while GLaDOS is speaking.
# The threshold adapts to the echo level (our own voice through the mic).
BARGEIN_RMS = 0.03
BARGEIN_CHUNKS = 5  # ~500 ms of sustained speech
BARGEIN_ECHO_MARGIN = 1.5  # must exceed echo level by this factor
BARGEIN_COOLDOWN_S = 3.0
# TTS audio is 16-bit mono PCM at this rate (must match the engine + spoke aplay).
TTS_SAMPLE_RATE = 24000
# Margin added to the spoken audio's real-time duration before re-enabling the
# mic, to let the spoke's aplay/ALSA buffer fully drain. The mute deadline is
# sized to the ACTUAL audio length (see _stream_tts), not a fixed window — the
# engine synthesizes faster than real-time, so a constant settle would fire
# mid-sentence and let GLaDOS transcribe her own voice.
PLAYBACK_SETTLE_S = 1.0
# After GLaDOS responds, the next utterances for this long are treated as a
# continuation of the conversation — no wake word or decision-LLM gate. This
# is how multi-turn exchanges ("where?" -> "warrenton virginia") work.
FOLLOWUP_WINDOW_S = 20.0
# How long a fetched weather briefing stays injected into the prompt, so
# follow-ups like "for tomorrow?" still have the 3-day forecast to draw on.
WEATHER_CONTEXT_TTL_S = 300.0
# Timer nag: beep-only alarm delivery. Each cycle is a short 880 Hz ding
# followed by quiet — deliberately NO spoken lines: TTS through the XVF3800
# made GLaDOS hear her own voice (its AEC can't fully cancel a 24 kHz
# stream), and the words added nothing. The ding is the alarm; the user
# saying "stop" or "ok" is the acknowledgment. The loop runs until the user
# speaks (barge-in kills the nag task) or the budget runs out.
NAG_BEAP_S = 0.22        # length of the ding audio (see _nag_beep)
NAG_GAP_S = 2.0          # quiet between one beep and the next
NAG_MAX_LINES = 20       # beeps before she gives up (~64 s of alarm)
NAG_BUSY_WAIT_S = 30     # wait this long for a busy device to finish its current line

# LED ring states (frame 0x15, one byte) for the XVF3800. The I2S-master
# firmware dropped the factory LED animation, so the hub drives the ring;
# other spokes silently ignore unknown frame types.
LED_OFF, LED_LISTEN, LED_THINK, LED_SPEAK, LED_ALARM = 0, 1, 2, 3, 4

PROMPT_PATH = Path(__file__).parent / "prompts" / "glados_system.txt"
with open(PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read()
# Tool surface (lights/timers/announcements) + the actual device inventory,
# appended once at import so every prompt knows what she can do.
_capabilities = skills.describe_capabilities()
if _capabilities:
    SYSTEM_PROMPT += "\n\n" + _capabilities

logger.info("GLaDOS system prompt loaded (%d chars)", len(SYSTEM_PROMPT))
logger.info("GEMINI: model=%s key=%s", GEMINI_MODEL, "set" if GEMINI_API_KEY else "NOT SET (falling back to local llama)")

app = FastAPI(title="GLaDOS Hub", version="0.3.0")
active_sessions: dict = {}
http_client = httpx.AsyncClient(timeout=60.0)

# Default location for weather queries (set GLADOS_LOCATION in .env).
DEFAULT_LOCATION = os.environ.get("GLADOS_LOCATION", "")
# Last location the user told us out loud (survives spoke reconnects).
last_location = DEFAULT_LOCATION

# One shared streaming-ASR engine for the whole hub (Vosk model loaded once)
asr_engine = ASREngine()


@app.on_event("startup")
async def _startup():
    asr_engine.load()
    # Wire the tool layer into the hub's speak paths (kept out of import
    # time so skills/timers stay importable without a running hub).
    skills.set_announce_fn(_announce_to_devices)
    timers.set_speak_fn(_speak_to_one_device)
    # Camera feed: announces to ALL active speakers (speaker="" -> no one
    # skipped) — a security event is for the whole house.
    cameras.start(_announce_to_devices)
    logger.info("GLaDOS Hub started (ASR available=%s, cameras=%s)",
                asr_engine.available, "on" if cameras.status() else "off")


def _rms(chunk: bytes) -> float:
    """RMS level of an int16 PCM chunk, normalized to 0..1."""
    if len(chunk) < 2:
        return 0.0
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples**2))) / 32768.0


def compute_playback_deadline(playback_start: float | None, audio_bytes: int, now: float) -> float:
    """Monotonic deadline at which TTS playback has fully drained from the speaker.

    The engine synthesizes faster than real-time, so every frame is relayed to
    the spoke long before the audio finishes playing. Playback begins ~when the
    first frame arrives and runs for the audio's real-time duration, then needs
    a short drain margin. Sizing the mute window to the actual audio length is
    what stops GLaDOS from transcribing her own mid-sentence voice.
    """
    audio_seconds = audio_bytes / (TTS_SAMPLE_RATE * 2)  # 16-bit mono
    if playback_start is None:
        playback_start = now
    return playback_start + audio_seconds + PLAYBACK_SETTLE_S


def _status_payload(text: str) -> bytes:
    """Status frames are null-padded to 240 bytes (ESP32 prints them as C strings)."""
    raw = text.encode("utf-8")[:239]
    return bytes([0x13]) + raw + b"\x00" * (240 - len(raw))


async def _send_led(ws: WebSocket, state: int):
    """One-byte LED-ring state for the XVF3800 (0x15). No-op on other spokes."""
    try:
        await ws.send_bytes(bytes([0x15, state]))
    except Exception:
        pass


async def _send_status(ws: WebSocket, text: str):
    try:
        await ws.send_bytes(_status_payload(text))
    except Exception:
        return
    # Mirror the status into the LED ring: thinking -> amber breath,
    # everything else defaults to the blue listening breath.
    led = LED_LISTEN
    if text.lower().startswith("thinking"):
        led = LED_THINK
    await _send_led(ws, led)


async def _send_heartbeats(ws: WebSocket):
    """Small frame every 8 s so the spoke's 15 s receive-timeout never fires."""
    while True:
        await asyncio.sleep(8)
        try:
            await ws.send_bytes(bytes([0x14]) + b"hb")
        except Exception:
            break




# ── Response pipeline ────────────────────────────────────────────────

_LIGHT_TOOLS = ("set_lights", "get_lights")
# A set_lights report is a failure when it contains any of these — the LLM
# must be told, so it never converts a failure into "the lights are off".
_LIGHT_FAIL_RE = re.compile(r"FAILED|no response|timed out", re.IGNORECASE)


async def _execute_light_command(target: str, state: str, speaker: str) -> str | None:
    """Run a light command the LLM skipped (deterministic backstop).

    Returns the verified set_lights report, or None when the command is not
    actionable (nothing resolvable) — the backstop must fail SAFE: better
    to let the LLM answer than to flip a light the user didn't name.
    """
    reg = kasa_lights.REGISTRY
    names = reg.resolve(target)
    if not names and not target.strip():
        # "turn off the lights" with no room: only unambiguous when exactly
        # one group exists (here: 'livingroom').
        if len(reg.groups) == 1:
            target = next(iter(reg.groups))
            names = list(reg.groups.values())[0]
    if not names:
        logger.warning("light backstop: no resolvable target for %r — not acting", target)
        return None
    return await skills.execute_tool(
        "set_lights", {"target": target, "state": state}, speaker=speaker
    )


async def _call_response_llm(session: Session, utterance: str = "", extra_context: str = "") -> str:
    """Local LLM (radiance) with a tool loop: lights, timers, announcements.

    `utterance` is the user's transcript for THIS turn. `extra_context`
    (e.g. a live weather briefing) is appended to the system prompt for
    this one request so the model answers from real data. The Gemini path
    is kept for rollback (no key is configured today).

    Light commands get a deterministic backstop so the spoken answer can
    never outpace the hardware (the "she says the lights are off but
    they're not" bug):
      * if the transcript is a light command but the model never called a
        light tool, the hub calls set_lights itself and grounds the final
        reply in the verified result;
      * if any set_lights result this turn reports a failure, a final note
        forces the reply to state it plainly — never claim it switched.
    """
    extras = extra_context
    if getattr(session, "nag_note", "") and time.monotonic() < getattr(session, "nag_note_until", 0.0):
        extras = (extras + "\n\n" if extras else "") + session.nag_note
    sys_prompt = SYSTEM_PROMPT + (("\n\n" + extras) if extras else "")
    msgs = session.llm_messages()
    if GEMINI_API_KEY:
        try:
            reply = await llm_client.call_gemini(http_client, msgs, sys_prompt)
            logger.info("[%s] response via Gemini (%s)", session.device_id, GEMINI_MODEL)
            return reply
        except Exception as e:
            logger.warning("[%s] Gemini failed (%s); falling back to local llama", session.device_id, e)
    try:
        full = [{"role": "system", "content": sys_prompt}] + msgs
        # The deterministic light backstop (see docstring) needs the command
        # up front: if the model skips the tool, the hub runs it itself.
        cmd = decision.light_command(utterance)
        message = await llm_client.call_llama_message(
            http_client, full, max_tokens=300, temperature=0.7, tools=skills.TOOLS
        )
        rounds = 0
        light_tool_called = False
        light_failed = False
        while message.get("tool_calls") and rounds < skills.MAX_TOOL_ROUNDS:
            rounds += 1
            full.append(message)
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    logger.warning("[%s] tool %s: unparseable args %r",
                                   session.device_id, name, fn.get("arguments"))
                    args = {}
                logger.info("[%s] tool call: %s(%s)", session.device_id, name, json.dumps(args)[:200])
                result = await skills.execute_tool(name, args, speaker=session.device_id)
                logger.info("[%s] tool %s -> %s", session.device_id, name, result[:200])
                if name in _LIGHT_TOOLS:
                    light_tool_called = True
                if name == "set_lights" and _LIGHT_FAIL_RE.search(result):
                    light_failed = True
                full.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            # tools stay on for later rounds too: after a set_lights failure
            # the model may want to re-read the state via get_lights.
            message = await llm_client.call_llama_message(
                http_client, full, max_tokens=300, temperature=0.7, tools=skills.TOOLS
            )

        # ── Deterministic light backstop ─────────────────────────────
        # Words are not proof. Run the missing tool / force the failure
        # report, then let the model produce the FINAL reply once more —
        # grounded in the verified hardware state.
        notes: list[str] = []
        if not light_tool_called and cmd:
            target, state = cmd
            result = await _execute_light_command(target, state, session.device_id)
            if result is not None:
                logger.info("[%s] light backstop fired: set_lights(%r, %r) -> %s",
                            session.device_id, target, state, result[:200])
                notes.append(
                    "INTERNAL NOTE (not user speech): the user's last words were a "
                    "light command, but you did not call a light tool. I called "
                    f"set_lights(target='{target}', state='{state}') for you; the "
                    f"verified result is: {result}. Your reply MUST be based only on "
                    "that result — never claim a change the result does not confirm."
                )
                if _LIGHT_FAIL_RE.search(result):
                    light_failed = True
        if light_failed:
            notes.append(
                "INTERNAL NOTE (not user speech): a light device FAILED to change "
                "this turn — a Kasa bulb that ACKed the command and then dropped "
                "it. Do NOT claim it changed. Say plainly which device(s) did not "
                "change, that you retried, and what state they are still in."
            )
        reply = message.get("content") or ""
        if notes:
            full.append({"role": "user", "content": " ".join(notes)})
            message = await llm_client.call_llama_message(http_client, full, max_tokens=300, temperature=0.7)
            reply = message.get("content") or reply
        if not reply:
            logger.warning("[%s] LLM produced no content after tool loop", session.device_id)
            return llm_client.FALLBACK_LINE
        logger.info("[%s] response via local llama (%d tool round(s))", session.device_id, rounds)
        return reply
    except Exception as e:
        logger.error("[%s] LLM response failed: %s", session.device_id, e, exc_info=True)
        return llm_client.FALLBACK_LINE


async def _tiny_llm_says_respond(session: Session, text: str, followup: bool = False) -> bool:
    """Ask the tiny LLM whether GLaDOS should speak up. Fail-safe: NO.

    followup=True gates in continuation mode: the utterance arrived inside
    the follow-up window (no wake word), so the question is whether it
    connects to the exchange that just ended.
    """
    prompt = decision.build_decision_prompt(text, session.decision_context(), followup=followup)
    if getattr(session, "nag_note", "") and time.monotonic() < getattr(session, "nag_note_until", 0.0):
        prompt += "\n\n" + session.nag_note
    try:
        # Small but non-trivial budget: max_tokens=1 risks a bare template token.
        answer = await llm_client.call_llama(
            http_client, [{"role": "user", "content": prompt}],
            max_tokens=8, temperature=0.0,
        )
        ok = decision.parse_decision(answer)
        mode = "follow-up" if followup else "gate"
        logger.info("[%s] decision LLM (%s): '%s' -> %s", session.device_id, mode, text[:60], answer.strip() or "(empty)")
        return ok
    except Exception as e:
        logger.warning("[%s] decision LLM failed (%s) -> NO", session.device_id, e)
        return False


class TtsTailBuffer:
    """Relays TTS PCM while holding back the last `holdback` bytes.

    The held tail is faded out on completion (see flush): a hard step from
    speech to silence clicks through the speaker, so the end of every
    response gets a 30 ms ramp plus a short tail of true silence.
    """

    SILENCE_TAIL_S = 0.10  # quiet tail after the fade, so the amp settles

    def __init__(self, holdback: int, sample_rate: int = TTS_SAMPLE_RATE):
        self.holdback = holdback
        self.sample_rate = sample_rate
        self.pending = bytearray()

    def feed(self, data: bytes) -> bytes:
        """Return the portion of `data` safe to send now (possibly b'')."""
        self.pending.extend(data)
        if len(self.pending) > self.holdback:
            out = bytes(self.pending[:-self.holdback])
            del self.pending[:-self.holdback]
            return out
        return b""

    def flush(self) -> bytes:
        """Fade the held-back tail to silence and append the quiet tail."""
        tail_silence = b"\x00" * int(self.SILENCE_TAIL_S * self.sample_rate * 2)
        seg = np.frombuffer(bytes(self.pending), dtype=np.int16).astype(np.float32)
        self.pending.clear()
        if len(seg) == 0:
            return tail_silence
        ramp = np.linspace(1.0, 0.0, len(seg))
        return (seg * ramp).astype(np.int16).tobytes() + tail_silence


async def _stream_tts(ws: WebSocket, text: str, session: Session):
    """Stream TTS audio from the engine WS to the spoke, frame by frame.

    Each PCM frame from the engine is relayed the moment it is synthesized,
    so the spoke starts playing before the full response is generated.
    On completion, sets session.playback_deadline so the mic stays muted
    through the speaker's tail (prevents GLaDOS transcribing her own voice).
    """
    text = llm_client.clean_for_tts(text)  # markdown/colons -> speakable text
    audio_bytes = 0
    playback_start = None
    # Hold back the last 30 ms so the end of speech can be faded out instead
    # of stepping hard to silence (a click through the speaker).
    tail = TtsTailBuffer(holdback=int(0.03 * TTS_SAMPLE_RATE * 2))
    async with websockets.connect(
        f"{ENGINE_WS_URL}/tts/stream", max_size=2**20, open_timeout=10
    ) as engine_ws:
        await engine_ws.send(json.dumps({"text": text}))
        frames = 0
        async for msg in engine_ws:
            if isinstance(msg, (bytes, bytearray)):
                out = tail.feed(bytes(msg))
                if out:
                    if playback_start is None:
                        playback_start = time.monotonic()
                    audio_bytes += len(out)
                    await ws.send_bytes(bytes([0x11]) + out)
                    frames += 1
            else:  # JSON done/error frame
                data = json.loads(msg) if isinstance(msg, str) else {}
                if data.get("error"):
                    logger.warning("[%s] engine TTS error: %s", "tts", data["error"])
                break
    # Soft landing: faded tail + 100 ms of quiet (relayed as the final frame).
    final = tail.flush()
    if final:
        if playback_start is None:
            playback_start = time.monotonic()
        audio_bytes += len(final)
        await ws.send_bytes(bytes([0x11]) + final)
        frames += 1
    # Playback began ~when the first frame reached the spoke and runs for the
    # full audio duration at real-time speed (the engine synthesizes faster
    # than real-time, so all frames are relayed well before playback ends).
    session.playback_deadline = compute_playback_deadline(playback_start, audio_bytes, time.monotonic())
    logger.info(
        "[tts] streamed %d audio frames (~%.1fs audio) for '%s'",
        frames, audio_bytes / (TTS_SAMPLE_RATE * 2), text[:50],
    )


async def _respond(session: Session, ws: WebSocket, text: str, why: str, extra_context: str = ""):
    """Full response: LLM -> streaming TTS. Cancellable via barge-in."""
    if session.glados_speaking:
        logger.info("[%s] dropping response (already speaking)", session.device_id)
        return
    session.add_message("user", text)
    session.speaking_task = asyncio.current_task()
    session.glados_speaking = True
    played = False
    try:
        await _send_status(ws, "thinking...")
        reply = await _call_response_llm(session, text, extra_context=extra_context)
        if not reply:
            return
        session.add_message("assistant", reply)
        logger.info("[%s] GLaDOS (%s): '%s'", session.device_id, why, reply[:120])
        await _send_led(ws, LED_SPEAK)
        await _stream_tts(ws, reply, session)
        played = True
        # Conversation continues: the user can answer right away without
        # repeating the wake word.
        session.followup_until = time.monotonic() + FOLLOWUP_WINDOW_S
    except asyncio.CancelledError:
        logger.info("[%s] response cancelled (barge-in)", session.device_id)
        raise
    except Exception as e:
        logger.error("[%s] response pipeline error: %s", session.device_id, e, exc_info=True)
        await _send_status(ws, "I seem to have malfunctioned.")
    finally:
        session.speaking_task = None
        session.bargein_hits = 0
        # If audio actually played, glados_speaking stays True until the
        # playback deadline passes (see _handle_audio) or a barge-in occurs.
        if not played:
            session.glados_speaking = False
            session.playback_deadline = 0.0
            session.echo_level = 0.0


async def _barge_in(session: Session, ws: WebSocket):
    """User started talking while GLaDOS is speaking: cut her off."""
    task = session.speaking_task
    if task and not task.done():
        task.cancel()
    session.glados_speaking = False
    session.speaking_task = None
    session.bargein_hits = 0
    session.live_text = ""
    session.playback_deadline = 0.0
    session.echo_level = 0.0
    if session.asr is not None:
        session.asr.reset()
    try:
        await ws.send_bytes(bytes([0x12]))  # spoke: stop playback + flush
    except Exception:
        pass
    await _send_status(ws, "listening...")


async def _weather_context(session: Session, text: str) -> str:
    """Maybe fetch fresh weather data to inject into the response prompt.

    Fetches when the utterance is about the weather (or just named a place),
    and keeps the last briefing around for WEATHER_CONTEXT_TTL_S so follow-up
    questions ("for tomorrow?") still have the 3-day forecast.
    """
    now = time.monotonic()
    location = session.location or last_location
    if location and (decision.weather_intent(text) or decision.extract_location(text)):
        block = await weather.get_weather_block(location, http_client)
        if block:
            session.weather_context = block
            session.weather_context_until = now + WEATHER_CONTEXT_TTL_S
    if session.weather_context and now < session.weather_context_until:
        return session.weather_context
    return ""


async def _on_final(session: Session, ws: WebSocket, text: str):
    """Handle one completed utterance (exactly one per utterance)."""
    async with session.process_lock:
        # Silence since the last exchange? Then the old words are stale —
        # drop them before the decision LLM or response LLM sees the prompt.
        session.prune_stale()
        name = decision.extract_name(text)
        if name:
            session.last_name = name
            logger.info("[%s] speaker named '%s'", session.device_id, name)
        # Did the user just tell us where we are? ("it's warrenton virginia")
        location = decision.extract_location(text)
        if location:
            session.location = location
            global last_location
            last_location = location
            logger.info("[%s] location set: %s", session.device_id, location)
        session.add_heard(text)

        followup_active = time.monotonic() < session.followup_until
        # A lone "huh" with nobody mid-conversation is background noise
        # (or the mic's rumble) — never pay a decision-LLM call for it.
        if decision.is_backchannel(text) and not followup_active:
            logger.info("[%s] backchannel '%s' outside follow-up window — ignored",
                        session.device_id, text)
            return
        # The same transcript over and over = a noisy/stuck audio path.
        if not session.register_final(text, conversational=followup_active):
            return

        extra_context = await _weather_context(session, text)

        routed = decision.route_full(text, followup_active=followup_active)
        if routed == "wake":
            await _respond(session, ws, text, why="wake-word", extra_context=extra_context)
        elif routed == "followup":
            # Inside the window too, the mic hears the whole room: gate on
            # relevance. Only continuations of the exchange that just ended
            # get a response; off-topic audio is ignored and the window
            # keeps running, so a real follow-up a few seconds later still
            # lands.
            if await _tiny_llm_says_respond(session, text, followup=True):
                await _respond(session, ws, text, why="follow-up", extra_context=extra_context)
        elif await _tiny_llm_says_respond(session, text):
            await _respond(session, ws, text, why="decision", extra_context=extra_context)


# ── WebSocket endpoint ───────────────────────────────────────────────

async def _handle_audio(session: Session, ws: WebSocket, chunk: bytes):
    if session.glados_speaking:
        # If playback has fully settled (speaker tail done), re-enable listening.
        if session.playback_deadline and time.monotonic() >= session.playback_deadline:
            session.glados_speaking = False
            session.playback_deadline = 0.0
            session.echo_level = 0.0
            logger.info("[%s] playback settled — listening", session.device_id)
        if session.glados_speaking:
            # Don't feed the ASR while we're talking: the mic hears our own
            # speaker (echo). Instead watch for sustained speech above the
            # echo level = barge-in.
            rms = _rms(chunk)
            if rms > session.echo_level:
                session.echo_level = min(rms, 0.5)
            threshold = max(BARGEIN_RMS, session.echo_level * BARGEIN_ECHO_MARGIN)
            if rms >= threshold:
                session.bargein_hits += 1
            else:
                session.bargein_hits = 0
            now = time.monotonic()
            if session.bargein_hits >= BARGEIN_CHUNKS and now - session.last_bargein >= BARGEIN_COOLDOWN_S:
                session.last_bargein = now
                logger.info(
                    "[%s] BARGE-IN (rms=%.3f > %.3f) — stopping TTS",
                    session.device_id, rms, threshold,
                )
                await _barge_in(session, ws)
            return
        # else: just settled — fall through and feed this chunk to the ASR

    if session.asr is None:
        return
    # Strip sub-200 Hz rumble (fans/HVAC/appliances) so the small ASR
    # model doesn't hallucinate words out of constant low-frequency noise.
    event = session.asr.feed(session.hpf.process(chunk))
    if event.kind == ASREventKind.PARTIAL and event.text:
        session.live_text = event.text
        if time.monotonic() - session._last_partial_log > 1.0:
            session._last_partial_log = time.monotonic()
            logger.info("[%s] partial: '%s'", session.device_id, event.text[:80])
    elif event.kind == ASREventKind.FINAL:
        session.live_text = ""
        if event.text:
            logger.info("[%s] ASR final: '%s'", session.device_id, event.text)
            # Run the pipeline off the receive loop so barge-in stays responsive
            asyncio.create_task(_on_final(session, ws, event.text))


@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    session = Session(device_id, asr_engine.create_stream(), SYSTEM_PROMPT)
    session.ws = websocket
    active_sessions[device_id] = session
    logger.info("Client connected: %s (ASR stream: %s)", device_id, "ok" if session.asr else "UNAVAILABLE")
    heartbeat_task = asyncio.create_task(_send_heartbeats(websocket))
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue
            frame_type, payload = data[0], data[1:]
            if frame_type == 0x03:  # session start
                session.is_listening = True
                if session.asr is not None:
                    session.asr.reset()
                logger.info("[%s] session started", device_id)
            elif frame_type == 0x04:  # session end
                session.is_listening = False
                logger.info("[%s] session ended", device_id)
            elif frame_type == 0x02:  # wake word from spoke
                wake_text = payload.decode("utf-8", errors="ignore")
                session.is_listening = True
                logger.info("[%s] wake word: '%s'", device_id, wake_text)
                await _send_status(websocket, "listening...")
            elif frame_type == 0x01:  # audio chunk (realtime)
                if session.is_listening and payload:
                    await _handle_audio(session, websocket, payload)
            else:
                logger.debug("[%s] unknown frame type 0x%02x", device_id, frame_type)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("[%s] connection error: %s", device_id, e, exc_info=True)
    finally:
        heartbeat_task.cancel()
        active_sessions.pop(device_id, None)
        if session.speaking_task and not session.speaking_task.done():
            session.speaking_task.cancel()
        logger.info("Client disconnected: %s", device_id)


# ── REST helpers ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(active_sessions),
        "asr_available": asr_engine.available,
        "engine_url": ENGINE_URL,
        "gemini_model": GEMINI_MODEL,
        "gemini_key_set": bool(GEMINI_API_KEY),
        "context_ttl_s": CONTEXT_TTL_S,
        "cameras": cameras.status(),
    }


async def _speak_raw(session: Session, ws: WebSocket, text: str):
    """Stream TTS for raw text (no LLM) — used by the /say test endpoint."""
    if session.glados_speaking:
        logger.info("[%s] dropping /say (already speaking)", session.device_id)
        return
    session.glados_speaking = True
    session.speaking_task = asyncio.current_task()
    played = False
    try:
        await _stream_tts(ws, text, session)
        played = True
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error("[%s] /say TTS error: %s", session.device_id, e)
    finally:
        session.speaking_task = None
        session.bargein_hits = 0
        if not played:
            session.glados_speaking = False
            session.playback_deadline = 0.0
            session.echo_level = 0.0


@app.post("/say")
async def direct_say(text: str = Query(..., description="Text to speak"), device_id: str = Query("test")):
    """Direct text-to-speech endpoint (speak raw text, for audio testing)."""
    session = active_sessions.get(device_id)
    if session is None or session.ws is None:
        return {"status": "no active session", "device_id": device_id}
    asyncio.create_task(_speak_raw(session, session.ws, text))
    return {"message": f"GLaDOS will say: {text}", "status": "ok"}


# ── Announcements + timer fires ──────────────────────────────────────

async def _announce_to_devices(message: str, speaker: str) -> str:
    """Speak a message to every ACTIVE session except the asking speaker
    (they already hear the LLM's reply). Returns a short report for the LLM."""
    message = llm_client.clean_for_tts(message).strip()
    if not message:
        return "announcement was empty"
    delivered, missed = [], []
    for dev, s in list(active_sessions.items()):
        if dev == speaker or s.ws is None:
            continue
        if s.glados_speaking:
            missed.append(dev)
            continue
        try:
            await _speak_raw(s, s.ws, message)
            delivered.append(dev)
        except Exception as e:
            logger.warning("announce to %s failed: %s", dev, e)
            missed.append(dev)
    if not delivered:
        known = ", ".join(d for d in active_sessions if d != speaker) or "no one else"
        return f"no other speakers available (active: {known})"
    out = f"announced to: {', '.join(delivered)}"
    if missed:
        out += f"; skipped (busy): {', '.join(missed)}"
    return out


# How long after a timer nag ends the "a nag just happened" note stays in the
# gate/response prompts. The user's "stop" arrives AFTER the nag has stopped
# (barge-in kills it first), and a fired timer is already gone from the store
# — so the note must outlive the nag, or the LLMs have no idea what "stop"
# refers to (and answer "I have no record of a timer").
NAG_NOTE_TTL_S = 300.0


def _nag_note(label: str) -> str:
    """Context injected into the gate + response prompts during/after a nag."""
    return (
        "INTERNAL NOTE (not user speech): GLaDOS is, or was moments ago, "
        "ringing a fired-timer alarm at this user ('" + (label or "the timer") +
        "') — a beep repeated over and over (no voice; the user acknowledges "
        "by saying something). The alarm STOPS "
        "AUTOMATICALLY the moment the user says anything. The timer exists "
        "and has fired; it is NOT in the pending timer list, so list_timers "
        "will not show it — never deny it, never say there is no record of "
        "it, and never call list_timers to 'check'. If the user's utterance "
        "is about the timer or the nagging — including bare acknowledgments "
        "and complaints such as 'stop', 'stop nagging', 'stop it', 'ok', "
        "'okay', 'fine', 'i hear you', 'im done', 'got it', 'shut up', "
        "'yes', 'why are you screaming' — GLaDOS should respond (in the "
        "yes/no decision gate the answer is YES). No tool call is needed to "
        "stop the nag; just acknowledge briefly, in character."
    )


async def _speak_to_one_device(device_id: str, text: str, label: str = "") -> str:
    """Timer-fire path: speak to exactly one device — and keep nagging until
    the user says something (barge-in stops it). Returns a short report."""
    s = active_sessions.get(device_id)
    if s is None or s.ws is None:
        return "device not connected"
    if s.glados_speaking:
        # She's mid-reply: wait for the current line to settle instead of
        # dropping the reminder.
        waited = 0
        while s.glados_speaking and waited < NAG_BUSY_WAIT_S:
            await asyncio.sleep(1)
            waited += 1
        if s.glados_speaking:
            return "device busy, reminder dropped"
    text = llm_client.clean_for_tts(text)
    # Tell the LLMs what's happening (see NAG_NOTE_TTL_S): a recent-past fact
    # with a TTL, so the user's post-nag "stop" still lands on someone who
    # knows about the timer.
    s.nag_note = _nag_note(label)
    s.nag_note_until = time.monotonic() + NAG_NOTE_TTL_S
    task = asyncio.create_task(_nag(s, text))
    _active_nags.add(task)
    task.add_done_callback(_active_nags.discard)
    return "ringing (repeats until acknowledged)"


# Strong references to running nag tasks (the event loop only holds weak
# refs; a GC'd task would die mid-nag silently).
_active_nags: set = set()

def _nag_beep() -> bytes:
    """Short 880 Hz attention ding with 15 ms fades (16-bit mono, TTS rate)."""
    sr = TTS_SAMPLE_RATE
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    env = np.ones(n)
    fade = int(0.015 * sr)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    wave = 0.28 * np.sin(2 * np.pi * 880.0 * t) * env
    return (wave * 32767).astype(np.int16).tobytes()


async def _nag(session: Session, first_line: str) -> None:
    """Ring a reminder to one session until the user speaks (barge-in
    cancels this task), the device goes away, or the budget runs out.

    Beep-only on purpose: spoken lines made GLaDOS hear her own voice on
    the XVF3800 (its AEC can't fully cancel a 24 kHz TTS stream), and the
    words were just noise. The ding is the alarm, and the user saying
    "stop" or "ok" is the acknowledgment — which is why the mic must stay
    in barge-in monitoring mode the WHOLE time: while active this task
    OWNS the session's speaking state (glados_speaking stays True and
    playback_deadline is kept in the future), so any speech kills it.
    """
    ws = session.ws
    task = asyncio.current_task()
    session.speaking_task = task
    beep = bytes([0x11]) + _nag_beep()
    said = 0
    try:
        await _send_led(ws, LED_ALARM)
        for i in range(NAG_MAX_LINES):
            if session.ws is None:
                break
            session.glados_speaking = True
            try:
                await ws.send_bytes(beep)
            except Exception as e:
                logger.warning("[%s] nag stopped early (%s)", session.device_id, e)
                break
            said += 1
            if i == 0:
                logger.info("[%s] nag started for '%s' (beeps until acknowledged)",
                            session.device_id, first_line)
            # Hold the mic in barge-in monitoring through the whole cycle:
            # the settle check in _handle_audio must not fire mid-nag, or
            # the user's "stop" would route to the ASR instead of the gate.
            session.playback_deadline = (time.monotonic() + NAG_BEAP_S
                                         + PLAYBACK_SETTLE_S + NAG_GAP_S)
            await asyncio.sleep(max(0.0, session.playback_deadline - time.monotonic()))
        logger.info("[%s] nag finished after %d beep(s)", session.device_id, said)
    except asyncio.CancelledError:
        logger.info("[%s] nag cancelled — user spoke (%d beep(s))", session.device_id, said)
        raise
    finally:
        # _barge_in clears these itself before cancelling us; only clean up if
        # we're still the owner (nag ended on its own).
        if session.speaking_task is task:
            session.speaking_task = None
            session.glados_speaking = False
            session.playback_deadline = 0.0
            session.bargein_hits = 0
            session.echo_level = 0.0
        await _send_led(ws, LED_OFF)
