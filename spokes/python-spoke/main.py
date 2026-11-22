"""
GLaDOS Python Spoke - audio <-> hub WebSocket, + LAFVIN HAT face.

Pushes 16 kHz mono int16 PCM to the hub in realtime (0x01 frames, 100 ms).
Plays streamed TTS audio (0x11 frames, 24 kHz int16) as it arrives, with
low-latency chunked playback. 0x12 = stop playback immediately (barge-in).
0x16 = set playback volume (int16 LE, 0-100).

When a LAFVIN Audio Display HAT is attached (and GLADOS_DISPLAY != off),
the conversation state is rendered on its 240x280 ST7789: idle = a yellow
ball floating around, talking = audio-reactive ball + ring, listening =
cyan pulse on the mic level, thinking = amber orbit. The HAT's RGB LED and
backlight mirror the state; its button is a physical barge-in.
"""
import asyncio
import logging
import os
import queue
import time

import numpy as np
import websockets

from face import Face, STATE_BACKLIGHT, STATE_LED
from lafvin import LafvinHat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("glados_spoke")

HUB_URL = os.environ.get("GLADOS_HUB_URL", "ws://localhost:5051")
DEVICE_ID = os.environ.get("GLADOS_DEVICE_ID", "spoke-001")
# LAFVIN HAT display: auto = enable when the HAT is present, off = never.
DISPLAY_ENABLED = os.environ.get("GLADOS_DISPLAY", "auto").strip().lower()
DISPLAY_FPS = float(os.environ.get("GLADOS_DISPLAY_FPS", "24"))
# Device selection (set by setup_audio.py, or directly for manual runs)
DEFAULT_DEVICE = "plughw:CARD=wm8960soundcard,DEV=0"
CAPTURE_DEVICE = os.environ.get("GLADOS_CAPTURE_DEVICE") or os.environ.get("GLADOS_AUDIO_DEVICE", DEFAULT_DEVICE)
PLAYBACK_DEVICE = os.environ.get("GLADOS_PLAYBACK_DEVICE") or os.environ.get("GLADOS_AUDIO_DEVICE", DEFAULT_DEVICE)
CHUNK_SIZE = 3200  # 100 ms @ 16 kHz, int16
TTS_CHUNK_BYTES = int(24000 * 0.2 * 2)  # 200 ms @ 24 kHz
MAX_PEAK = 20000  # ~-4 dBFS peak limit (Piper outputs full-scale)
# Speaker volume: hub-controlled 0-100 (frame 0x16). 100 = full (the
# MAX_PEAK limit above still applies); 0 = digital silence.
DEFAULT_VOLUME = 100


def apply_volume(arr: np.ndarray, level: int) -> np.ndarray:
    """Scale int16 PCM by a 0-100 volume level, clamped to int16 range.

    Pure (testable); level >= 100 returns the input unchanged.
    """
    if level >= 100:
        return arr
    gain = max(0, level) / 100.0
    if gain == 0.0:
        return np.zeros_like(arr)
    return np.clip(arr.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


class AudioMeter:
    """Latest mic/TTS RMS, normalized to 0..1, with a short hold so the
    face doesn't flicker between the 100 ms capture chunks."""

    MIC_NORM = 6000.0   # speech at ~1 m RMSes a few hundred..2000
    TTS_NORM = 8000.0   # TTS is peak-limited to 20000; RMS is much lower
    HOLD_S = 0.25

    def __init__(self, now_fn=time.monotonic):
        self._now = now_fn
        self._mic = (0.0, 0.0)
        self._tts = (0.0, 0.0)

    def set_mic(self, rms: float):
        self._mic = (float(rms), self._now())

    def set_tts(self, rms: float):
        self._tts = (float(rms), self._now())

    @staticmethod
    def _norm(rms: float, norm: float) -> float:
        return min(1.0, rms / norm)

    def mic_level(self) -> float:
        rms, t = self._mic
        if self._now() - t > self.HOLD_S:
            return 0.0
        return self._norm(rms, self.MIC_NORM)

    def tts_level(self) -> float:
        rms, t = self._tts
        if self._now() - t > self.HOLD_S:
            return 0.0
        return self._norm(rms, self.TTS_NORM)


meter = AudioMeter()
# Aplay is kept alive and warm (short silences while idle) so the ALSA/USB
# device never closes: re-opening it per reply is what causes the speaker
# pop and the swallowed first second on this card.
TTS_KEEPALIVE_S = 1.0
# Data gap (seconds) that marks the start of a new speech burst -> fade-in.
TTS_BURST_GAP_S = 0.35
FADE_SAMPLES = 720  # 30 ms @ 24 kHz linear ramp
SILENCE_100MS = b"\x00\x00" * (24000 // 10)  # 100 ms @ 24 kHz s16le


class Playback:
    """Streams 24 kHz int16 PCM to ONE long-lived aplay process.

    aplay is pre-warmed at startup and kept warm with periodic silence, so
    the ALSA/USB device stays open and clocked. Re-opening the device for
    every reply is what caused the speaker pop and the dropped first second.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.proc = None
        self._write_lock = None  # created lazily: asyncio.Lock binds to the loop
        self._last_tts = 0.0  # monotonic time of last TTS data written
        self._burst_active = False
        self.volume = DEFAULT_VOLUME  # 0-100, hub-controlled (frame 0x16)

    def set_volume(self, level: int) -> None:
        """Hub 0x16 frame: set the playback gain (clamped to 0-100)."""
        level = max(0, min(100, int(level)))
        if level == self.volume:
            return
        self.volume = level
        logger.info("[PLAYBACK] volume set to %d%%", level)

    def _lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def start(self):
        """Pre-warm aplay + ALSA so the first word has zero open latency."""
        try:
            await self._ensure_proc()
        except Exception as e:
            logger.warning("[PLAYBACK] pre-warm failed (%s); keepalive will retry", e)
            self.proc = None

    async def _ensure_proc(self):
        # asyncio.subprocess.Process exposes .returncode, not .poll()
        if self.proc is None or self.proc.returncode is not None:
            if self.proc is not None:
                try:
                    await self.proc.wait()
                except Exception:
                    pass
            self.proc = await asyncio.create_subprocess_exec(
                "aplay", "-f", "S16_LE", "-r", "24000", "-c", "1", "-D", PLAYBACK_DEVICE,
                stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info("[PLAYBACK] aplay ready on %s (pid %d)", PLAYBACK_DEVICE, self.proc.pid)
        return self.proc

    def _fade_in(self, arr: np.ndarray) -> np.ndarray:
        """Ramp the first 30 ms of a new burst up from silence (kills the pop)."""
        now = time.monotonic()
        if (not self._burst_active) or (now - self._last_tts > TTS_BURST_GAP_S):
            n = min(FADE_SAMPLES, len(arr))
            if n:
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                arr = arr.astype(np.float32)
                arr[:n] = arr[:n] * ramp
                arr = arr.astype(np.int16)
            self._burst_active = True
        self._last_tts = now
        return arr

    async def _write_frame(self, frame: bytes, retries: int = 6):
        """Write one slice to aplay, respawning it if needed.

        The SAME frame is retried (never silently dropped) so a slow ALSA
        open or a transient device-busy can't eat the start of a sentence.
        """
        last_err = None
        for attempt in range(retries):
            try:
                async with self._lock():
                    proc = await self._ensure_proc()
                    proc.stdin.write(frame)
                    await proc.stdin.drain()
                return
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                last_err = e
                self.proc = None
                await asyncio.sleep(0.15 * (attempt + 1))
        raise last_err

    async def write(self, data: bytes):
        """Buffer TTS PCM and write 200 ms slices to aplay at 24 kHz.

        Piper outputs hot (full-scale peaks), so we peak-limit to ~-4 dBFS
        to keep the amp from clipping/distorting.
        """
        self.buffer.extend(data)
        while len(self.buffer) >= TTS_CHUNK_BYTES:
            frame = bytes(self.buffer[:TTS_CHUNK_BYTES])
            del self.buffer[:TTS_CHUNK_BYTES]
            try:
                arr = np.frombuffer(frame, dtype=np.int16)
                peak = int(np.max(np.abs(arr))) if len(arr) else 0
                if peak > MAX_PEAK:
                    arr = (arr * (MAX_PEAK / peak)).astype(np.int16)
                if self.volume < 100:
                    arr = apply_volume(arr, self.volume)
                frame = self._fade_in(arr).tobytes()
                if len(arr):
                    meter.set_tts(float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))))
                await self._write_frame(frame)
            except Exception as e:
                logger.warning("[PLAYBACK] dropped frame after retries: %s", e)

    async def keepalive(self):
        """Keep aplay + the USB audio device warm while idle.

        A short silence at most once a second keeps ALSA open and the codec
        clocked, so the next word starts instantly with no wake-up pop.
        Also respawns aplay (killed by barge-in) as soon as possible.
        """
        while True:
            await asyncio.sleep(TTS_KEEPALIVE_S)
            try:
                proc = self.proc
                if proc is None or proc.returncode is not None:
                    await self._ensure_proc()
                elif time.monotonic() - self._last_tts > TTS_KEEPALIVE_S * 1.5:
                    async with self._lock():
                        proc.stdin.write(SILENCE_100MS)
                        await proc.stdin.drain()
            except Exception as e:
                logger.debug("[PLAYBACK] keepalive hiccup: %s", e)
                self.proc = None

    async def stop(self):
        """Stop playback immediately and flush the buffer (barge-in)."""
        self.buffer.clear()
        self._burst_active = False  # next burst fades in fresh
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=1)
            except Exception:
                pass
        self.proc = None


playback = Playback()


async def play_audio(audio_queue: asyncio.Queue):
    while True:
        frame_type, data = await audio_queue.get()
        if frame_type == 0x11:
            await playback.write(data)
        elif frame_type == 0x12:
            logger.info("[PLAYBACK] stop (barge-in)")
            await playback.stop()
        elif frame_type == 0x13:
            logger.info("[STATUS] %s", data.rstrip(b"\x00").decode("utf-8", errors="ignore"))


async def receive_audio(ws, audio_queue: asyncio.Queue):
    while True:
        try:
            data = await asyncio.wait_for(ws.recv(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("[RECV] No data from hub in 15s. Connection dead?")
            raise websockets.exceptions.ConnectionClosed(1001, "Hub silent 15s")
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.error("[RECV] Connection lost: %s", e)
            raise
        if not data:
            continue
        frame_type, payload = data[0], data[1:]
        if frame_type == 0x14:  # hub heartbeat
            continue
        if frame_type == 0x16:  # volume (int16 LE, 0-100)
            if len(payload) >= 2:
                playback.set_volume(int.from_bytes(payload[:2], "little"))
            continue
        if frame_type in (0x11, 0x12, 0x13):
            audio_queue.put_nowait((frame_type, payload))


async def capture_audio(audio_tx_queue: queue.Queue):
    """Capture 16 kHz mono int16 and push 100 ms chunks in realtime."""
    proc = await asyncio.create_subprocess_exec(
        "arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-D", CAPTURE_DEVICE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    logger.info("arecord started: %d", proc.pid)
    buf = bytearray()
    chunk_count = 0
    try:
        while True:
            data = await proc.stdout.read(4000)
            if not data:
                continue
            buf.extend(data)
            while len(buf) >= CHUNK_SIZE:
                chunk = bytes(buf[:CHUNK_SIZE])
                del buf[:CHUNK_SIZE]
                chunk_count += 1
                arr = np.frombuffer(chunk, dtype=np.int16)
                rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
                meter.set_mic(rms)
                if chunk_count % 20 == 0:
                    logger.info("[LISTENING] %.2f RMS (chunk %d)", rms, chunk_count)
                try:
                    audio_tx_queue.put_nowait(bytes([0x01]) + chunk)
                except queue.Full:
                    pass  # hub not keeping up; drop rather than block capture
    finally:
        proc.terminate()
        await proc.wait()


async def send_audio(ws, audio_tx_queue: queue.Queue):
    loop = asyncio.get_event_loop()
    while True:
        frame = await loop.run_in_executor(None, audio_tx_queue.get)
        await ws.send(frame)


_ws_loop = None  # set in main(); lets the HAT button (worker thread) barge in


def open_hat():
    """Open the LAFVIN HAT (blocking). Returns None when absent/disabled."""
    if DISPLAY_ENABLED == "off":
        return None
    if DISPLAY_ENABLED != "auto":
        logger.warning("[DISPLAY] GLADOS_DISPLAY=%r is not 'auto' — treating as auto",
                       DISPLAY_ENABLED)

    def on_press():
        logger.info("[DISPLAY] button pressed — barge-in (stopping playback)")
        if _ws_loop is not None and _ws_loop.is_running():
            asyncio.run_coroutine_threadsafe(playback.stop(), _ws_loop)

    return LafvinHat.try_open(on_press=on_press)


async def display_task(stop_event: asyncio.Event):
    """Render the conversation face on the LAFVIN HAT.

    Non-critical: exits quietly when the HAT isn't present. Runs the SPI
    pushes in the executor so the event loop (audio/WS) stays realtime.
    """
    loop = asyncio.get_event_loop()
    hat = await loop.run_in_executor(None, open_hat)
    if hat is None:
        logger.info("[DISPLAY] LAFVIN HAT not active — display disabled")
        return
    face = Face()
    last_state = None
    dt = 1.0 / DISPLAY_FPS
    try:
        while not stop_event.is_set():
            t0 = time.monotonic()
            frame = face.tick(dt, meter.mic_level(), meter.tts_level())
            await loop.run_in_executor(None, hat.present_frame, frame)
            if face.state != last_state:
                r, g, b = STATE_LED[face.state]
                await loop.run_in_executor(None, hat.set_led, r, g, b)
                await loop.run_in_executor(
                    None, hat.set_backlight, STATE_BACKLIGHT[face.state])
                logger.info("[DISPLAY] state -> %s", face.state)
                last_state = face.state
            await asyncio.sleep(max(0.0, dt - (time.monotonic() - t0)))
    finally:
        await loop.run_in_executor(None, hat.close)
        logger.info("[DISPLAY] stopped")


async def main():
    global _ws_loop
    _ws_loop = asyncio.get_event_loop()
    logger.info("GLaDOS Spoke starting... Hub=%s Device=%s", HUB_URL, DEVICE_ID)
    retry = 1
    # The face lives outside the reconnect loop: the HAT opens once and
    # survives hub drops. On shutdown asyncio.run cancels this task and
    # display_task's finally closes the HAT.
    stop_event = asyncio.Event()
    asyncio.create_task(display_task(stop_event))
    while True:
        tasks = []
        audio_queue = asyncio.Queue()
        audio_tx_queue = queue.Queue()
        try:
            async with websockets.connect(
                f"{HUB_URL}/ws/{DEVICE_ID}",
                ping_interval=2,
                ping_timeout=1,
                max_size=2**20,  # TTS audio frames can be large
            ) as ws:
                logger.info("Connected!")
                retry = 1
                play_task = asyncio.create_task(play_audio(audio_queue))
                keep_task = asyncio.create_task(playback.keepalive())
                cap = asyncio.create_task(capture_audio(audio_tx_queue))
                snd = asyncio.create_task(send_audio(ws, audio_tx_queue))
                rcv = asyncio.create_task(receive_audio(ws, audio_queue))
                tasks = [play_task, keep_task, cap, snd, rcv]
                asyncio.create_task(playback.start())  # pre-warm aplay/ALSA now
                await ws.send(bytes([0x03]))  # session start
                logger.info("Listening...")
                while True:
                    for t in tasks:
                        if t.done() and not t.cancelled():
                            raise t.exception() or RuntimeError("spoke task exited")
                    await asyncio.sleep(0.1)
        except (ConnectionRefusedError, OSError, websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosed, asyncio.CancelledError) as e:
            logger.error("Lost connection: %s. Reconnecting in %ds...", e, retry)
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await playback.stop()
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)
        except Exception as e:
            # Any unexpected error: log it and reconnect — never die.
            logger.exception("Unexpected spoke error: %s. Reconnecting in %ds...", e, retry)
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await playback.stop()
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
