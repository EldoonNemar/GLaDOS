"""
Per-spoke conversation state for the hub.

Kept free of FastAPI/transport imports so it is unit-testable in plain
Python (same pattern as decision.py / llm.py).

Two stores of "what has happened" live here, and both are time-bounded so
stale words don't ride along in every LLM prompt:

  * `messages`  — the LLM conversation (system + user/assistant turns)
  * `history`   — rolling transcript of heard utterances (decision context)

Without time bounds, an utterance said before bed ("huh") would stay in
both stores until the hard caps are hit — days, for a quiet household —
and every later decision-LLM / response-LLM call would pay tokens for
text the user long forgot. `prune_stale()` fixes that: after
CONTEXT_TTL_S of silence, anything older than the cutoff is dropped, so
the next morning starts with a clean slate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque

from .dsp import HighPass16k

logger = logging.getLogger("glados_hub.session")

# Rolling context budget for the decision LLM / LLM prompts
HISTORY_MAX_WORDS = 200
LLM_CONTEXT_MESSAGES = 12
# Hard cap on the LLM conversation (system prompt + 99 turns), kept as a
# safety net on top of the time-based pruning.
MESSAGES_HARD_CAP = 100

# Silence (seconds) that expires conversation context. After this much
# quiet, the previous exchange no longer counts as "the conversation":
# it is removed from both LLM prompts so we stop burning tokens on the
# same old text. Override with the CONTEXT_TTL_S env var.
CONTEXT_TTL_S = float(os.environ.get("CONTEXT_TTL_S", "600"))

# Repeat-suppression for ASR finals. A mic fed with constant noise (or a
# stuck audio path) emits the *same* transcript over and over — the live
# mic produced 'huh' every 20 s for hours. If a transcript arrives this
# many times within REPEAT_WINDOW_S it is suppressed for REPEAT_COOLDOWN_S.
REPEAT_LIMIT = 3
REPEAT_WINDOW_S = 120.0
REPEAT_COOLDOWN_S = 600.0


class Session:
    """Per-spoke conversation state."""

    def __init__(self, device_id: str, asr_stream, system_prompt: str):
        self.device_id = device_id
        self.conversation_id = str(uuid.uuid4())[:8]
        self.asr = asr_stream
        # LLM conversation (system + exchanged turns)
        self.messages = [{"role": "system", "content": system_prompt}]
        # Monotonic timestamps, parallel to self.messages[1:] (non-system)
        self._msg_times: list[float] = []
        # Rolling transcript history (all heard utterances, for decision
        # context): (monotonic ts, line) tuples
        self.history: deque = deque()
        self.last_name = ""
        # Pipeline state
        self.is_listening = False
        self.glados_speaking = False
        self.speaking_task: asyncio.Task | None = None
        self.bargein_hits = 0
        self.last_bargein = 0.0
        self.echo_level = 0.0  # tracked mic level while GLaDOS speaks (self-echo)
        self.playback_deadline = 0.0  # monotonic time when playback has settled
        self.followup_until = 0.0  # respond without wake word until this time
        self.location = ""  # spoken location for weather queries
        # "A timer nag is (or was just) happening" context for the gate and
        # response LLMs. A fired timer is GONE from the store by the time the
        # user says "stop", so without this the model answers "no record".
        self.nag_note = ""
        self.nag_note_until = 0.0  # monotonic expiry of nag_note
        self.weather_context = ""  # last fetched weather briefing (LLM context)
        self.weather_context_until = 0.0
        self.live_text = ""
        self._last_partial_log = 0.0
        self.process_lock = asyncio.Lock()
        # Audio preprocessing: strips the sub-200 Hz rumble (fans, HVAC,
        # appliances) that makes the small ASR model hallucinate words.
        self.hpf = HighPass16k()
        # Repeat-suppression state: normalized text -> [first_ts, count]
        self._final_counts: dict[str, list] = {}
        self._final_cooldown: dict[str, float] = {}  # normalized text -> until

    def add_message(self, role: str, content: str, now: float | None = None):
        self.messages.append({"role": role, "content": content})
        self._msg_times.append(time.monotonic() if now is None else now)
        if len(self.messages) > MESSAGES_HARD_CAP:
            self.messages = [self.messages[0]] + self.messages[-(MESSAGES_HARD_CAP - 1):]
            self._msg_times = self._msg_times[-(MESSAGES_HARD_CAP - 1):]

    def add_heard(self, text: str, now: float | None = None):
        line = f"{self.last_name}: {text}" if self.last_name else text
        self.history.append((time.monotonic() if now is None else now, line))
        # Keep the rolling window within a word budget
        while self.history and len(" ".join(l for _, l in self.history).split()) > HISTORY_MAX_WORDS:
            self.history.popleft()

    def prune_stale(self, now: float | None = None, ttl: float | None = None) -> int:
        """Drop context that silence has made stale.

        Removes every LLM message and history line whose timestamp is more
        than `ttl` (default CONTEXT_TTL_S) before `now` (default: monotonic
        now). The system prompt always survives. Called on each new
        utterance, so it is exactly "nothing was said for a while -> the
        old words are gone". Returns the number of LLM messages dropped.
        """
        now = time.monotonic() if now is None else now
        ttl = CONTEXT_TTL_S if ttl is None else ttl
        cutoff = now - ttl

        dropped = 0
        if self._msg_times and min(self._msg_times) < cutoff:
            kept_msgs = [self.messages[0]]
            kept_times: list[float] = []
            for msg, ts in zip(self.messages[1:], self._msg_times):
                if ts >= cutoff:
                    kept_msgs.append(msg)
                    kept_times.append(ts)
                else:
                    dropped += 1
            self.messages = kept_msgs
            self._msg_times = kept_times

        stale_lines = sum(1 for ts, _ in self.history if ts < cutoff)
        if stale_lines:
            self.history = deque((ts, line) for ts, line in self.history if ts >= cutoff)

        if dropped or stale_lines:
            logger.info(
                "[%s] context pruned after silence: -%d LLM message(s), -%d history line(s) (ttl=%.0fs)",
                self.device_id, dropped, stale_lines, ttl,
            )
        return dropped

    def decision_context(self) -> str:
        return "\n".join(line for _, line in list(self.history)[-5:])

    def register_final(self, text: str, conversational: bool = False,
                       now: float | None = None) -> bool:
        """Repeat-suppression gate for ASR finals.

        A mic fed with constant noise (or a stuck audio path) produces
        the *same* transcript over and over. If `text` has already
        arrived REPEAT_LIMIT times within REPEAT_WINDOW_S, it is
        suppressed for REPEAT_COOLDOWN_S. Returns True if the utterance
        should go on to the decision/response pipeline.

        During an active conversation (GLaDOS just spoke) repetition is
        more likely to be real, so suppression is paused — counting
        still happens.
        """
        now = time.monotonic() if now is None else now
        # Case/punctuation-insensitive: "Huh!" and "huh" are the same
        # transcript for suppression purposes
        norm = "".join(c for c in text.lower() if c.isalpha())
        if not norm:
            return True
        if self._final_cooldown.get(norm, 0.0) > now:
            return False
        # Drop bookkeeping for texts that fell out of the window
        for stale in [k for k, (t0, _) in self._final_counts.items()
                      if now - t0 > REPEAT_WINDOW_S]:
            self._final_counts.pop(stale, None)
        entry = self._final_counts.get(norm)
        if entry is None:
            self._final_counts[norm] = [now, 1]
            return True
        entry[1] += 1
        if entry[1] >= REPEAT_LIMIT and not conversational:
            self._final_cooldown[norm] = now + REPEAT_COOLDOWN_S
            self._final_counts.pop(norm, None)
            logger.info(
                "[%s] transcript '%s' repeated %dx in %.0fs — suspected "
                "noise/stuck audio, suppressing for %.0fs",
                self.device_id, text[:40], entry[1], now - entry[0],
                REPEAT_COOLDOWN_S,
            )
            return False
        return True

    def llm_messages(self) -> list:
        non_system = [m for m in self.messages if m["role"] != "system"]
        return non_system[-LLM_CONTEXT_MESSAGES:]
