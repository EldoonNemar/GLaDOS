"""
Kasa smart-light control (plugs + bulbs) via python-kasa.

The registry (config/kasa.json, baked into the image; override with
KASA_CONFIG_PATH) maps human names to LAN IPs and bundles lights into
*groups*, so one utterance — "turn off the livingroom lights" — can
reach three devices.

Everything here returns short plain strings: they go back to the LLM as
tool results and get spoken, so they must never raise and must always
be speakable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("glados_hub.kasa")

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "kasa.json"
CONFIG_PATH = Path(os.environ.get("KASA_CONFIG_PATH", str(_DEFAULT_CONFIG)))

CONNECT_TIMEOUT_S = 6   # unicast probe per device (python-kasa request timeout)
OP_TIMEOUT_S = 10       # whole multi-device op (enforced by the tool layer)

# Words spoken around the target that carry no matching signal.
_NOISE = re.compile(r"\b(the|a|an|light|lights|lamp|lamps|switch|plug|plugs|on|off|please)\b")


class LightRegistry:
    """Names -> IPs for lights, plus group name -> [light names]."""

    def __init__(
        self,
        lights: dict[str, str],
        types: dict[str, str],
        groups: dict[str, list[str]],
    ):
        self.lights = lights    # name -> ip (lowercase names)
        self.types = types      # name -> "plug" | "bulb"
        self.groups = groups    # group -> [light names]

    @classmethod
    def load(cls, path: Path | None = None) -> "LightRegistry":
        p = Path(path) if path else CONFIG_PATH
        try:
            raw = json.loads(p.read_text())
        except Exception as e:
            logger.warning("Kasa config %s unreadable (%s) — lights disabled", p, e)
            return cls({}, {}, {})
        lights = {k.lower(): v for k, v in raw.get("lights", {}).items()}
        types = {k.lower(): v for k, v in raw.get("types", {}).items()}
        groups: dict[str, list[str]] = {}
        for g, members in raw.get("groups", {}).items():
            names = [m.lower() for m in members if m.lower() in lights]
            if names:
                groups[g.lower()] = names
        return cls(lights, types, groups)

    def resolve(self, query: str) -> list[str]:
        """Map a human target ("the livingroom lights") to light names.

        Precedence: exact light, exact group, group substring, light
        substring, per-token substring. First level that matches wins;
        results keep config order (deterministic).
        """
        q = _NOISE.sub(" ", (query or "").lower())
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            return []
        if q in self.lights:
            return [q]
        if q in self.groups:
            return list(self.groups[q])
        # Whitespace-insensitive exact match: ASR often splits a name the
        # config joins ("living room" vs "livingroom") — without this the
        # substring levels below mis-route the group to a single device.
        q_ns = q.replace(" ", "")
        for name in self.lights:
            if name.replace(" ", "") == q_ns:
                return [name]
        for g, members in self.groups.items():
            if g.replace(" ", "") == q_ns:
                return list(members)
        for g, members in self.groups.items():
            if g in q or q in g:
                return list(members)
        hits = [n for n in self.lights if n in q or q in n]
        if hits:
            return hits
        # Last resort: every significant query token must match a name token.
        qt = q.split()
        for n in self.lights:
            nt = n.split()
            if qt and all(any(a in b or b in a for b in nt) for a in qt):
                return [n]
        return []

    def describe(self) -> str:
        """Speakable inventory (for tool errors and the system prompt)."""
        if not self.lights:
            return "no lights configured"
        parts = [f"{name} ({self.types.get(name, 'light')})" for name in self.lights]
        out = ", ".join(parts)
        if self.groups:
            out += "; groups: " + "; ".join(
                f"'{g}' = {', '.join(m)}" for g, m in self.groups.items()
            )
        return out


REGISTRY = LightRegistry.load()


def set_registry(reg: LightRegistry) -> None:
    """Test seam: swap in a registry (e.g. built from a temp config)."""
    global REGISTRY
    REGISTRY = reg


async def _control_one(name: str, ip: str, state: str, brightness: int | None) -> tuple[bool, str]:
    """Drive one device. Never raises; returns (ok, short report)."""
    import kasa  # local import: module stays importable without the dep (tests)

    def report(ok: bool, msg: str) -> tuple[bool, str]:
        return ok, f"{name}: {msg}"

    try:
        dev = await kasa.Discover.discover_single(ip, timeout=CONNECT_TIMEOUT_S)
        if dev is None:
            return report(False, "no response")
        await dev.update()

        async def apply() -> None:
            if state == "on":
                await dev.turn_on()
            else:
                await dev.turn_off()
            if brightness is not None:
                light_mod = dev.modules.get("Light") if isinstance(dev.modules, dict) else None
                if light_mod is not None:
                    await light_mod.set_state(kasa.LightState(brightness=brightness))

        await apply()

        # Verify-after-write: Kasa devices can ACK a command and then drop it
        # (seen in the wild for 'on'), so re-read the device and trust the
        # read, not the ACK. One retry before declaring failure.
        await dev.update()
        if dev.is_on != (state == "on"):
            await apply()
            await dev.update()
        if dev.is_on != (state == "on"):
            actual = "on" if dev.is_on else "off"
            logger.warning("kasa %s (%s): %s not verified after 2 attempts, still %s",
                           name, ip, state, actual)
            # "after 2 attempts" is deliberate: it goes to the LLM so she can
            # tell the user she actually retried, not just gave up.
            return report(False, f"{state} FAILED after 2 attempts, still {actual}")
        if brightness is not None:
            light_mod = dev.modules.get("Light") if isinstance(dev.modules, dict) else None
            if light_mod is None:
                return report(True, f"{state}, no dimming on that device")
            return report(True, f"{state} at {brightness}%")
        return report(True, state)
    except asyncio.TimeoutError:
        return report(False, "timed out")
    except Exception as e:
        logger.warning("kasa %s (%s) failed: %s", name, ip, e)
        return report(False, type(e).__name__)


async def set_lights(target: str, state: str, brightness: int | None = None) -> str:
    """Turn one light, several, or a whole group on/off (and dim bulbs).

    Each device is VERIFIED after the write (state re-read; one retry on
    mismatch), so the report reflects reality, not just ACKs. Returns a
    short report, e.g.
    "livingroom ceiling: off; livingroom left: off; livingroom right: no response"
    or, if a device dropped the command:
    "livingroom left: on FAILED after 2 attempts, still off".
    """
    state = (state or "").lower()
    if state not in ("on", "off"):
        return "state must be 'on' or 'off'"
    if brightness is not None:
        try:
            brightness = max(1, min(100, int(brightness)))
        except (TypeError, ValueError):
            return "brightness must be an integer from 1 to 100"
    if state == "off":
        brightness = None  # dimming an off bulb is pointless
    names = REGISTRY.resolve(target)
    if not names:
        return f"no light or group matching '{target}'. Known: {REGISTRY.describe()}"
    results = await asyncio.gather(
        *[_control_one(n, REGISTRY.lights[n], state, brightness) for n in names]
    )
    return "; ".join(msg for _, msg in results)


async def _read_one(name: str, ip: str) -> str:
    """Read one device's current state. Never raises; returns a short report."""
    import kasa  # local import: module stays importable without the dep (tests)
    try:
        dev = await kasa.Discover.discover_single(ip, timeout=CONNECT_TIMEOUT_S)
        if dev is None:
            return f"{name}: no response"
        await dev.update()
        state = "on" if dev.is_on else "off"
        light_mod = dev.modules.get("Light") if isinstance(dev.modules, dict) else None
        if light_mod is not None:
            b = dev.light_state.get("brightness") if hasattr(dev, "light_state") else None
            if b is not None:
                state += f" at {b}%"
        return f"{name}: {state}"
    except asyncio.TimeoutError:
        return f"{name}: timed out"
    except Exception as e:
        logger.warning("kasa read %s (%s) failed: %s", name, ip, e)
        return f"{name}: {type(e).__name__}"


async def get_lights(target: str) -> str:
    """Read the CURRENT on/off state (and bulb brightness) of a light or group.

    Returns e.g. "livingroom ceiling: on; livingroom left: on at 80%;
    livingroom right: off" — the source of truth for state questions.
    """
    names = REGISTRY.resolve(target)
    if not names:
        return f"no light or group matching '{target}'. Known: {REGISTRY.describe()}"
    results = await asyncio.gather(*[_read_one(n, REGISTRY.lights[n]) for n in names])
    return "; ".join(results)


def describe_capabilities() -> str:
    """System-prompt block: which devices exist and how the light tool resolves names."""
    if not REGISTRY.lights:
        return ""
    lines = [
        "SMART HOME (tools available):",
        f"Kasa lights: {REGISTRY.describe()}.",
        "Use set_lights with the target the user actually said — exact light "
        "names, group names, and partial names ('the ceiling light', "
        "'livingroom lights') all resolve automatically.",
        "Use get_lights to read a light or group's CURRENT state. When asked "
        "whether a light is on or off, always call get_lights first — never "
        "assert a light's state from memory or guess.",
        "Never claim a light changed, or report a light's state, unless a "
        "set_lights or get_lights result from THIS turn says so.",
        "If the user's words are garbled but clearly about a light, call the "
        "tool with your best-guess target — a tool error is recoverable, a "
        "made-up answer is not.",
        "set_lights verifies each device after the write: a device reported as "
        "'FAILED' or 'no response' did NOT change — say so plainly, never "
        "claim it switched, and never retry the same call more than once "
        "within a turn.",
    ]
    return "\n".join(lines)
