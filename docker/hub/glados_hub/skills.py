"""
GLaDOS tools: the closed, hand-rolled capability set the LLM can invoke via
OpenAI tool calls (served by radiance — the same call shape that drives this
conversation you're reading right now).

Why not MCP: MCP is a general interop protocol (multi-server discovery, its
own client stack, dynamic tool registries). GLaDOS has one server and a
handful of capabilities. A closed tool list means every argument is
validated server-side, the whole surface is auditable in one file, and
there's no extra runtime. If a second tool server ever shows up, that's
the moment to revisit MCP.
"""

from __future__ import annotations

import asyncio
import logging

from . import cameras
from . import kasa_lights
from . import quiet
from . import timers

logger = logging.getLogger("glados_hub.skills")

TOOL_TIMEOUT_S = 12.0   # wall-clock cap on any single tool execution
MAX_TOOL_ROUNDS = 3     # tool-call round trips per response (loop breaker)

# Injected by the hub at startup (avoids circular imports):
#   async (message: str, speaker: str) -> str
_announce = None


def set_announce_fn(fn) -> None:
    global _announce
    _announce = fn


# Injected by the hub at startup:
#   async (device_id: str, level: int | None, change: int | None,
#          report: bool) -> str
_volume_fn = None


def set_volume_fn(fn) -> None:
    global _volume_fn
    _volume_fn = fn


# One hub-wide store; the hub restart (which clears it) is the documented
# behavior — persistence is a one-line addition to /startup if ever wanted.
TIMER_STORE = timers.TimerStore()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_lights",
            "description": (
                "Turn Kasa smart lights on or off, one device or a whole group. "
                "Pass `target` exactly as the user named it (e.g. 'livingroom', "
                "'ceiling') — groups and partial names resolve server-side. "
                "`brightness` (1-100) is for bulbs when turning on. "
                "Each device is verified after the write (re-read, one retry), "
                "so the result reports the ACTUAL state: a device reported as "
                "'FAILED' did not change — say so, do not claim it switched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Light or group name as the user said it",
                    },
                    "state": {"type": "string", "enum": ["on", "off"]},
                    "brightness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Bulbs only; ignored for plugs and when turning off",
                    },
                },
                "required": ["target", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lights",
            "description": (
                "Read the CURRENT on/off state (and bulb brightness) of a light "
                "or group. ALWAYS use this instead of guessing when asked "
                "whether a light is on or off, or before reporting state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Light or group name as the user said it",
                    },
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": (
                "Set a timer. When it fires, ONLY the speaker's own device hears "
                "it (never the whole house) — and it KEEPS ringing: a beep every "
                "few seconds (no voice), repeating until someone speaks to stop it "
                "or it gives up after ~60 seconds. Tell the user any word stops it. "
                "on_fire='announce' says a message; 'light_on'/'light_off' flips "
                "the `target` lights first, then rings. Re-setting a label resets it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Short name, e.g. 'pasta'"},
                    "minutes": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 1440,
                        "description": "Duration in minutes",
                    },
                    "on_fire": {
                        "type": "string",
                        "enum": ["announce", "light_on", "light_off"],
                        "description": "Default 'announce'",
                    },
                    "target": {
                        "type": "string",
                        "description": "Light/group target for light_on/light_off",
                    },
                    "message": {
                        "type": "string",
                        "description": "For announce: what to say when it fires",
                    },
                },
                "required": ["label", "minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_timer",
            "description": (
                "Cancel a PENDING (not yet fired) timer by (partial) label. "
                "Do not use it for a timer that is currently nagging — that "
                "nag stops by itself the moment the user speaks."
            ),
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_timers",
            "description": "List active timers and how long is left on each.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "announce",
            "description": (
                "Send a spoken announcement to the OTHER GLaDOS speakers in the "
                "house (e.g. 'dinner is ready'). The asking speaker already hears "
                "your normal reply — never announce back to them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "What to say, in plain words"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_camera",
            "description": (
                "Check the live camera view (front or back yard) from the "
                "Scrypted NVR's object detection. Reports who is ACTIVE and "
                "moving (person, package), last activity per class, and "
                "whether the feed is available. Silence on a live feed "
                "means nothing was detected — a real all-clear. ALWAYS use "
                "it — never guess — when asked whether anyone is outside "
                "or at the door, or whether a package was delivered. If "
                "the feed is UNAVAILABLE, say plainly that you cannot see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {
                        "type": "string",
                        "enum": list(cameras.CONFIG.cameras) or ["front"],
                        "description": (
                            "A camera name from config/cameras.json "
                            "(default 'front')"
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": (
                "Set or adjust the speaker volume (the speaker the user is "
                "talking to). 'level' = absolute volume 0-100 ('set the "
                "volume to 30', 'mute it' -> level=0); 'change' = relative "
                "step ('turn it down a bit' -> change=-10, 'louder' -> "
                "change=+10). level wins if both are given. The result "
                "states the new volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 0, "maximum": 100},
                    "change": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_volume",
            "description": (
                "Report the current speaker volume (0-100) for the speaker "
                "the user is talking to. Use it when asked what the volume "
                "is."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def refresh_tool_enums() -> None:
    """Re-derive config-driven tool argument enums from the live config.

    The check_camera camera enum is captured at import; the config server
    calls this after a cameras.json reload so the LLM sees added/removed
    cameras on the next turn. Safe to call any time (idempotent).
    """
    for tool in TOOLS:
        fn = tool.get("function", {})
        if fn.get("name") == "check_camera":
            fn["parameters"]["properties"]["camera"]["enum"] = (
                list(cameras.CONFIG.cameras) or ["front"]
            )


async def execute_tool(name: str, args: dict, speaker: str) -> str:
    """Run one tool call. Never raises — returns a short result string for the LLM."""
    try:
        args = args or {}
        if name == "set_lights":
            return await asyncio.wait_for(
                kasa_lights.set_lights(
                    str(args.get("target", "")),
                    str(args.get("state", "")),
                    args.get("brightness"),
                ),
                kasa_lights.OP_TIMEOUT_S,
            )
        if name == "get_lights":
            return await asyncio.wait_for(
                kasa_lights.get_lights(str(args.get("target", ""))),
                kasa_lights.OP_TIMEOUT_S,
            )
        if name == "set_timer":
            try:
                minutes = float(args.get("minutes"))
            except (TypeError, ValueError):
                return "minutes must be a number"
            if not (1 <= minutes <= 1440):
                return "minutes must be between 1 and 1440"
            t = TIMER_STORE.add(
                label=str(args.get("label", "")),
                seconds=minutes * 60,
                device_id=speaker,
                action=str(args.get("on_fire") or "announce"),
                target=str(args.get("target", "")),
                message=str(args.get("message", "")),
            )
            if isinstance(t, str):
                return t
            return f"timer '{t.label}' set for {minutes:g} minutes"
        if name == "cancel_timer":
            return TIMER_STORE.cancel(str(args.get("label", "")))
        if name == "list_timers":
            return TIMER_STORE.list_active()
        if name == "announce":
            if quiet.is_active("responses"):
                # A user-requested announcement is still "her talking" —
                # quiet hours cover it. Report it so the LLM doesn't claim
                # it was spoken.
                logger.info("announce tool suppressed (quiet hours): %r",
                            args.get("message", ""))
                return ("quiet hours: the announcement was NOT spoken — do "
                        "not claim it was")
            if _announce is None:
                return "announcements unavailable (hub not started)"
            return await asyncio.wait_for(
                _announce(str(args.get("message", "")), speaker),
                TOOL_TIMEOUT_S,
            )
        if name == "check_camera":
            return cameras.check_camera(str(args.get("camera") or "front"))
        if name in ("set_volume", "get_volume"):
            if _volume_fn is None:
                return "volume control unavailable (hub not started)"
            level = args.get("level")
            change = args.get("change")
            try:
                level = int(level) if level is not None else None
                change = int(change) if change is not None else None
            except (TypeError, ValueError):
                return "level and change must be integers"
            if name == "get_volume" or (level is None and change is None):
                report = True
                level = change = None
            else:
                report = False
            return await asyncio.wait_for(
                _volume_fn(speaker, level, change, report), TOOL_TIMEOUT_S
            )
        return f"unknown tool '{name}'"
    except asyncio.TimeoutError:
        return f"tool {name} timed out"
    except Exception as e:
        logger.error("tool %s failed: %s", name, e, exc_info=True)
        return f"tool {name} failed: {type(e).__name__}"


def describe_capabilities() -> str:
    """System-prompt block describing the tool surface (appended at import)."""
    blocks = [kasa_lights.describe_capabilities()]
    blocks.append(
        "Timers: set_timer / cancel_timer / list_timers. A timer reminds ONLY the "
        "speaker's own device when it fires, and KEEPS ringing (a beep every few "
        "seconds, no voice) until someone says something — mention that any "
        "word stops it. on_fire can instead switch a light target on or off."
    )
    blocks.append(
        "Announcements: the announce tool sends a spoken message to the OTHER "
        "GLaDOS speakers in the house (e.g. dinner is ready). The asking speaker "
        "hears your normal reply, so do not duplicate it."
    )
    blocks.append(
        "Speaker volume: set_volume and get_volume control the volume of the "
        "speaker the user is talking to. For relative requests ('turn it down "
        "a bit', 'louder', 'too loud') use change with a modest step (-10 or "
        "+10); for absolute ones ('set it to 30', 'mute it') use level "
        "(0 = mute). Confirm the new volume from the tool result."
    )
    cam = cameras.describe_capabilities()
    if cam:
        blocks.append(cam)
    return "\n".join(b for b in blocks if b)