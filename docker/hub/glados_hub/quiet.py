"""
Quiet hours: windows where GLaDOS should not talk.

Config lives in config/quiet_hours.json (bind-mounted, live-editable via
the web UI — see config_server.py):

{
  "timezone": "America/Chicago",     # REQUIRED IANA zone — the hub container
                                     # runs UTC, so never rely on "local"
  "windows": [
    {
      "name": "night",
      "start": "22:00",              # HH:MM 24h; end < start crosses midnight
      "end": "07:00",
      "days": [0,1,2,3,4,5,6],       # 0=Mon..6=Sun, default: every day
      "sources": ["camera","timers","responses"]
    }
  ]
}

Sources — what "not talk" can cover, per window:
  camera     proactive camera-event announcements (events are still
             recorded, so check_camera reports them honestly)
  timers     fired-timer nags (the beeps)
  responses  her conversational replies — tools still EXECUTE silently
             (a "turn off the lights" during quiet hours turns off the
             lights; she just does not speak)

The checks are called from the enforcement points (cameras.py _consume,
hub timer-fire path, hub response TTS path, skills announce tool) and are
written to fail open: a check error is treated as "not quiet".
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "quiet_hours.json"

SOURCES = ("camera", "timers", "responses")
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_ALL_DAYS = frozenset(range(7))

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class QuietConfigError(ValueError):
    """Invalid quiet-hours config (reject the whole write)."""


@dataclass
class QuietWindow:
    name: str
    start: int                 # minutes since midnight
    end: int                   # minutes since midnight; < start => midnight-crossing
    days: frozenset = _ALL_DAYS       # 0=Mon..6=Sun
    sources: frozenset = frozenset(SOURCES)

    def covers(self, source: str) -> bool:
        return source in self.sources

    def active_at(self, minute_of_day: int, weekday: int) -> bool:
        if weekday not in self.days:
            return False
        if self.start < self.end:
            return self.start <= minute_of_day < self.end
        if self.start > self.end:  # crosses midnight: 22:00-07:00
            return minute_of_day >= self.start or minute_of_day < self.end
        return False             # start == end is rejected at load time

    def _fmt(self, minute: int) -> str:
        return f"{minute // 60:02d}:{minute % 60:02d}"

    def describe(self) -> str:
        days = "daily" if self.days == _ALL_DAYS else \
            ", ".join(DAY_NAMES[d] for d in sorted(self.days))
        if self.sources == frozenset(SOURCES):
            what = ("you do not speak at all (camera alerts, timer nags, and "
                    "replies are suppressed; commands are still executed "
                    "silently)")
        else:
            bits = {"camera": "camera alerts", "timers": "timer nags",
                    "responses": "replies"}
            what = "suppressed: " + ", ".join(bits[s] for s in SOURCES
                                              if s in self.sources)
        return (f"{self.name}: {self._fmt(self.start)}-{self._fmt(self.end)} "
                f"{days} — {what}")


def _parse_time(value, what: str) -> int:
    if not isinstance(value, str):
        raise QuietConfigError(f"{what} must be an 'HH:MM' string, got {value!r}")
    m = _TIME_RE.match(value.strip())
    if not m:
        raise QuietConfigError(f"{what} must be 'HH:MM' 24-hour, got {value!r}")
    return int(m.group(1)) * 60 + int(m.group(2))


def validate(raw: dict) -> None:
    """Raise QuietConfigError if `raw` would not load cleanly."""
    if not isinstance(raw, dict):
        raise QuietConfigError("top level must be a JSON object")
    tz = raw.get("timezone")
    if not isinstance(tz, str) or not tz.strip():
        raise QuietConfigError("timezone is required (IANA name, e.g. 'America/Chicago')")
    try:
        ZoneInfo(tz.strip())
    except Exception:
        raise QuietConfigError(f"unknown timezone {tz!r}") from None
    windows = raw.get("windows", [])
    if not isinstance(windows, list):
        raise QuietConfigError("windows must be a list")
    seen: set[str] = set()
    for i, w in enumerate(windows):
        if not isinstance(w, dict):
            raise QuietConfigError(f"window {i} must be an object")
        name = w.get("name")
        if not isinstance(name, str) or not name.strip():
            raise QuietConfigError(f"window {i}: name is required")
        name = name.strip()
        if name in seen:
            raise QuietConfigError(f"duplicate window name {name!r}")
        seen.add(name)
        start = _parse_time(w.get("start"), f"window {name!r} start")
        end = _parse_time(w.get("end"), f"window {name!r} end")
        if start == end:
            raise QuietConfigError(f"window {name!r}: start and end are the same time")
        days = w.get("days")
        if days is None:
            days = list(range(7))
        if (not isinstance(days, list) or not days
                or any(not isinstance(d, int) or isinstance(d, bool) or not 0 <= d <= 6
                       for d in days)):
            raise QuietConfigError(
                f"window {name!r}: days must be a non-empty list of 0-6 (0=Mon..6=Sun)")
        sources = w.get("sources")
        if sources is None:
            sources = list(SOURCES)
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(s, str) or s not in SOURCES for s in sources)):
            raise QuietConfigError(
                f"window {name!r}: sources must be a non-empty subset of {list(SOURCES)}")


@dataclass
class QuietConfig:
    timezone: str
    windows: list[QuietWindow] = field(default_factory=list)

    def _now(self, now: datetime | None) -> datetime:
        tz = ZoneInfo(self.timezone)
        return now.astimezone(tz) if now is not None else datetime.now(tz)

    def active_windows(self, now: datetime | None = None) -> list[QuietWindow]:
        dt = self._now(now)
        minute = dt.hour * 60 + dt.minute
        return [w for w in self.windows if w.sources and w.active_at(minute, dt.weekday())]

    def is_active(self, source: str, now: datetime | None = None) -> bool:
        return any(w.covers(source) for w in self.active_windows(now))

    def ends_at(self, now: datetime | None = None) -> str | None:
        """Earliest end time (HH:MM) across active windows, for the UI."""
        ends = [w.end for w in self.active_windows(now)]
        if not ends:
            return None
        e = min(ends)
        return f"{e // 60:02d}:{e % 60:02d}"

    def status(self, now: datetime | None = None) -> dict:
        dt = self._now(now)
        active = self.active_windows(now)
        return {
            "timezone": self.timezone,
            "now": dt.strftime("%H:%M"),
            "quiet": bool(active),
            "windows": [w.name for w in active],
            "ends_at": self.ends_at(now),
        }

    def describe(self) -> str:
        """System-prompt block; empty when no windows are configured."""
        if not self.windows:
            return ""
        lines = [w.describe() for w in self.windows]
        return ("Quiet hours (enforced by the hub; you cannot override them "
                "from conversation, and do not announce them unless asked):\n"
                + "\n".join(f"  - {l}" for l in lines))


def _load() -> QuietConfig:
    if not CONFIG_PATH.exists():
        # No file = no quiet hours (timezone is moot with zero windows).
        return QuietConfig(timezone="UTC", windows=[])
    try:
        raw = json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise QuietConfigError(f"unreadable {CONFIG_PATH}: {e}") from e
    validate(raw)
    windows = [
        QuietWindow(
            name=w["name"].strip(),
            start=_parse_time(w["start"], "start"),
            end=_parse_time(w["end"], "end"),
            days=frozenset(w.get("days") or list(range(7))),
            sources=frozenset(w.get("sources") or list(SOURCES)),
        )
        for w in raw.get("windows", [])
    ]
    return QuietConfig(timezone=raw["timezone"].strip(), windows=windows)


CONFIG: QuietConfig = _load()


def reload() -> QuietConfig:
    """Re-read the file (called by the config server after a save)."""
    global CONFIG
    CONFIG = _load()
    logger.info("quiet hours reloaded: %s",
                CONFIG.describe() or "no windows configured")
    return CONFIG


def is_active(source: str) -> bool:
    """Fail-open check used by the enforcement points."""
    try:
        return CONFIG.is_active(source)
    except Exception as e:
        logger.warning("quiet-hours check failed (%s); treating as not quiet", e)
        return False


def status() -> dict:
    try:
        return CONFIG.status()
    except Exception as e:
        logger.warning("quiet-hours status failed: %s", e)
        return {"timezone": "?", "now": "?", "quiet": False,
                "windows": [], "ends_at": None}