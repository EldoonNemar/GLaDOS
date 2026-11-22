"""
In-hub voice timers.

A timer set by a speaker fires to THAT speaker's own device only — never
house-wide (house-wide speech is the `announce` tool's job). The firing
action is either 'announce' (say a message) or 'light_on'/'light_off'
(toggle a light target, then confirm to the same device).

Firing speaks through the hub-injected speak function, which nags: beep +
line, repeated every few seconds until the user speaks (barge-in stops it)
or the line budget runs out. The nag loop itself lives in hub._nag.

Memory-only by design: a hub restart clears pending timers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("glados_hub.timers")

MAX_TIMERS = 20
MIN_SECONDS = 5
VALID_ACTIONS = ("announce", "light_on", "light_off")

# Injected by the hub (avoids a circular import):
#   async (device_id: str, text: str, label: str) -> str   (reports the outcome)
_speak_to_device = None


def set_speak_fn(fn) -> None:
    """Hub wiring point: how to speak one line to one device."""
    global _speak_to_device
    _speak_to_device = fn


@dataclass
class Timer:
    label: str
    fires_at: float            # time.monotonic()
    device_id: str             # spoke that set it (fire goes there, nowhere else)
    action: str = "announce"
    target: str = ""           # light/group target for light_on/light_off
    message: str = ""          # what to say for 'announce'
    task: "asyncio.Task | None" = field(default=None, repr=False)


class TimerStore:
    def __init__(self) -> None:
        self._timers: dict[str, Timer] = {}

    def add(
        self,
        label: str,
        seconds: float,
        device_id: str,
        action: str = "announce",
        target: str = "",
        message: str = "",
        min_seconds: float = MIN_SECONDS,
    ) -> "Timer | str":
        """Schedule a timer (returns the Timer) or an error string.

        Adding an existing label replaces it (re-say "ten minutes" = reset).
        """
        if action not in VALID_ACTIONS:
            return f"on_fire must be one of {', '.join(VALID_ACTIONS)}"
        label = (label or "").strip().lower()
        seconds = max(float(min_seconds), float(seconds))
        if len(self._timers) >= MAX_TIMERS and label not in self._timers:
            return f"timer limit reached ({MAX_TIMERS}) — cancel one first"
        existing = self._timers.pop(label, None)
        if existing is not None and existing.task is not None:
            existing.task.cancel()
        t = Timer(
            label=label or f"{int(seconds // 60)}-minute timer",
            fires_at=time.monotonic() + seconds,
            device_id=device_id,
            action=action,
            target=target.strip(),
            message=message.strip(),
        )
        t.task = asyncio.get_running_loop().create_task(self._wait_and_fire(t))
        self._timers[t.label] = t
        return t

    def cancel(self, query: str) -> str:
        q = (query or "").strip().lower()
        match = self._timers.get(q)
        if match is None:
            # Match both ways: 'pasta' inside 'pasta 1', and 'timer pasta' /
            # 'the pasta timer' containing the label.
            for _, t in self._timers.items():
                if q and (q in t.label or t.label in q):
                    match = t
                    break
        if match is None:
            if not self._timers:
                return "no active timers"
            return "no matching timer. " + self.list_active()
        if match.task is not None:
            match.task.cancel()
        self._timers.pop(match.label, None)
        return f"timer '{match.label}' cancelled"

    def list_active(self) -> str:
        if not self._timers:
            return "no active timers"
        now = time.monotonic()
        parts = [
            f"'{t.label}' in {max(0, int(t.fires_at - now))}s"
            for t in self._timers.values()
        ]
        return "active timers: " + ", ".join(parts)

    async def _wait_and_fire(self, t: Timer) -> None:
        try:
            await asyncio.sleep(max(0.0, t.fires_at - time.monotonic()))
        except asyncio.CancelledError:
            return
        self._timers.pop(t.label, None)
        logger.info("timer '%s' fired (device=%s action=%s)", t.label, t.device_id, t.action)
        await self._fire(t)

    async def _fire(self, t: Timer) -> None:
        if t.action == "announce":
            say = t.message or f"your timer, {t.label}, is up."
        else:
            from . import kasa_lights  # deferred: keeps this module testable without kasa

            state = "on" if t.action == "light_on" else "off"
            target = t.target or "lights"
            result = await kasa_lights.set_lights(target, state)
            ok = all(k not in result for k in ("no response", "timed out", "no light or group"))
            say = (
                f"your timer, {t.label}, is up. the {target} are {state}."
                if ok
                else f"your timer, {t.label}, is up, but the {target} didn't answer."
            )
        out = await self._speak(t, say)
        logger.info("timer '%s' spoken to %s: %s", t.label, t.device_id, out)

    async def _speak(self, t: Timer, text: str) -> str:
        if _speak_to_device is None:
            return "no speaker wired (hub not started)"
        try:
            # label: lets the hub tell the LLMs WHICH timer is nagging — the
            # user's "stop" arrives after the nag, when the store is already
            # empty, so the model can only know from this note.
            return await _speak_to_device(t.device_id, text, t.label)
        except Exception as e:
            logger.error("timer '%s' speak failed: %s", t.label, e)
            return f"speak failed: {type(e).__name__}"
