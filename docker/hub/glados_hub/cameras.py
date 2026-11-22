"""
Camera activity monitor: taps the Scrypted NVR's MQTT object-detection push.

Data path (all local, no cloud):
  Reolink cameras -> Scrypted NVR (on-device AI: person/vehicle/package/...)
    -> @scrypted/mqtt plugin publishes every detection frame to the broker's
       built-in Aedes instance (topic scrypted/<cameraId>/ObjectDetector)
    -> this module (MQTT consumer) -> DetectionTracker (pure state machine)
    -> announce callback (spoken to the house) + check_camera tool result.

The NVR streams one frame per video frame (~10/s), constantly re-listing
parked cars. The tracker therefore only counts SUSTAINED moving detections
(two moving frames within SUSTAIN_WINDOW_S) and announces each class at
most once per cooldown — otherwise she would say "someone's out there"
every few seconds.

Honesty rule (same as the lights): when the feed is disconnected, every
report says so plainly — and the LLM is told, via the tool result and the
capability block, never to guess what is outside. Silence on a LIVE feed
is the opposite problem: it means "nothing detected" (the NVR goes quiet
when there are no tracks), so reports treat it as a real all-clear and
only hedge after a very long silence (a possibly-off camera).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger("glados_hub.cameras")

# paho is a hard runtime dependency (requirements.txt), but the import is
# guarded so the pure tracker stays unit-testable in a bare environment.
try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None

# ── Configuration (env, see .env / docker-compose.yml) ─────────────────
MQTT_HOST = os.environ.get("SCRYPTED_MQTT_HOST", "")
MQTT_PORT = int(os.environ.get("SCRYPTED_MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("SCRYPTED_MQTT_USER", "")
MQTT_PASS = os.environ.get("SCRYPTED_MQTT_PASS", "")
# The camera inventory + per-camera policy live in config/cameras.json
# (same pattern as config/kasa.json; override the path with
# CAMERAS_CONFIG_PATH). The file is bind-mounted into the container and
# editable at runtime via the hub's web UI — or edit the JSON directly
# and hit the config server's reload. Add a camera, flip announce on,
# or restrict which NVR zones count without a rebuild.
_DEFAULT_CONFIG = Path(__file__).parent / "config" / "cameras.json"
CONFIG_PATH = Path(os.environ.get("CAMERAS_CONFIG_PATH", str(_DEFAULT_CONFIG)))
# Classes that trigger an automatic spoken announcement (front camera only).
ANNOUNCE_CLASSES = [
    c.strip()
    for c in os.environ.get("CAMERA_ANNOUNCE_CLASSES", "person,package").split(",")
    if c.strip()
]
COOLDOWN_S = float(os.environ.get("CAMERA_ANNOUNCE_COOLDOWN_S", "90"))

# Spoken lines for automatic announcements (GLaDOS voice, kept short — they
# go straight to TTS).
ANNOUNCE_LINES = {
    "person": "Attention. There is a person moving at the front of the house.",
    "package": "A package has been delivered at the front of the house.",
}

class DetectionTracker:
    """Pure state machine over ObjectDetector frames. No I/O — testable.

    A class becomes ACTIVE only after SUSTAIN_FRAMES moving detections fall
    inside SUSTAIN_WINDOW_S (one frame is jitter; parked cars re-detect with
    movement:{} constantly). It goes inactive again after ACTIVE_FORGET_S
    without a moving frame. Announcements fire on activation, at most once
    per cooldown_s.
    """

    SUSTAIN_FRAMES = 2
    SUSTAIN_WINDOW_S = 5.0
    ACTIVE_FORGET_S = 10.0
    # Connected but no frames at all for this long: the yard may be quiet,
    # but a dead camera is indistinguishable — reports must hedge (see
    # describe()). Verified against the live NVR: it only emits
    # ObjectDetector frames while it is tracking something, so a healthy
    # quiet yard goes SILENT (the 14:25 stream of ~10 frames/s dropped to
    # zero the moment the last parked-car tracks expired).
    SILENT_WARN_S = 3600.0

    def __init__(
        self,
        camera_id: str,
        announce_classes: tuple = ANNOUNCE_CLASSES,
        cooldown_s: float = COOLDOWN_S,
        zones: tuple = (),
        now_fn=time.monotonic,
    ):
        self.camera_id = str(camera_id)
        self.announce_classes = tuple(announce_classes)
        self.cooldown_s = cooldown_s
        self.zones = tuple(zones)
        self._now = now_fn
        self._last_frame: float | None = None  # monotonic ts of last frame
        # class -> {"moving_frames": deque[float], "last_moving": float,
        #           "last_seen": float, "last_announced": float,
        #           "count": int, "active": bool}
        self._classes: dict[str, dict] = {}

    # ── core ─────────────────────────────────────────────────────────
    def on_frame(self, detections: list | None, now: float | None = None) -> list[str]:
        """Feed one ObjectDetector frame; return classes newly announced."""
        now = self._now() if now is None else now
        self._last_frame = now
        fired: list[str] = []
        for det in detections or []:
            if not isinstance(det, dict):
                continue
            cls = det.get("className")
            if cls not in self.announce_classes:
                continue
            if self.zones:
                # Hub-side whitelist on top of the NVR's own zones: a
                # detection only counts when its zone tags intersect ours.
                # (Package detections are tagged 'packageZone' — include
                # that name in `zones` to keep them.)
                if not (set(det.get("zones") or []) & set(self.zones)):
                    continue
            st = self._classes.setdefault(cls, {
                "moving_frames": deque(),
                "last_moving": 0.0,
                "last_seen": 0.0,
                # -inf so the FIRST activation always passes the cooldown
                # (monotonic starts at ~0; 0.0 would look "just announced").
                "last_announced": float("-inf"),
                "count": 0,
                "active": False,
            })
            st["last_seen"] = now
            st["count"] += 1
            moving = (det.get("movement") or {}).get("moving") is True
            if moving:
                st["last_moving"] = now
                frames = st["moving_frames"]
                frames.append(now)
                while frames and now - frames[0] > self.SUSTAIN_WINDOW_S:
                    frames.popleft()
            if not st["active"] and len(st["moving_frames"]) >= self.SUSTAIN_FRAMES:
                st["active"] = True
                if now - st["last_announced"] >= self.cooldown_s:
                    st["last_announced"] = now
                    fired.append(cls)
        # decay: no moving frames recently -> no longer "active"
        for cls, st in self._classes.items():
            if st["active"] and now - st["last_moving"] > self.ACTIVE_FORGET_S:
                st["active"] = False
        return fired

    # ── reporting ────────────────────────────────────────────────────
    def last_frame_age(self, now: float | None = None) -> float | None:
        """Seconds since the last frame, or None if none was ever seen."""
        now = self._now() if now is None else now
        if self._last_frame is None:
            return None
        return now - self._last_frame

    def describe(self, connected: bool, now: float | None = None) -> str:
        """LLM-readable report of one camera. Honest about the feed.

        Silence is meaningful: the NVR only emits ObjectDetector frames
        while it is tracking something (gated on camera motion), so a
        healthy but quiet yard produces NO frames. connected + silence is
        a valid "no one detected" — except after a very long silence, when
        a dead camera becomes indistinguishable and the report hedges.
        """
        now = self._now() if now is None else now
        if not connected:
            return (
                f"{self.camera_id} camera: FEED UNAVAILABLE (broker connection "
                "lost). I cannot see outside right now — say so plainly and "
                "do not guess what is there."
            )
        age = self.last_frame_age(now)
        if age is None:
            return (
                f"{self.camera_id} camera: feed connected, no activity "
                "reported yet. Nothing detected at this moment."
            )
        if age > self.SILENT_WARN_S:
            lines = [
                f"{self.camera_id} camera: feed OK but silent for "
                f"{age / 3600:.1f} h — the camera may be off, so I cannot "
                "confirm the yard is actually clear."
            ]
        else:
            lines = [
                f"{self.camera_id} camera: feed OK (last frame {age:.0f}s ago)."
            ]
        for cls in self.announce_classes:
            st = self._classes.get(cls)
            if st is None:
                lines.append(f"  {cls}: none seen")
            elif st["active"]:
                lines.append(
                    f"  {cls}: ACTIVE — moving, last seen "
                    f"{now - st['last_moving']:.0f}s ago ({st['count']} detections)"
                )
            else:
                lines.append(
                    f"  {cls}: last seen {now - st['last_seen']:.0f}s ago "
                    f"({st['count']} detections)"
                )
        return "\n".join(lines)


class CameraMonitor:
    """MQTT consumer + one DetectionTracker per camera.

    paho runs in its own thread; frames cross into the event loop through a
    queue (call_soon_threadsafe), so the tracker and the announce callback
    both run on the hub's loop.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        cameras: dict[str, str],  # display name -> camera id
        announce_enabled: tuple = (),
        settings: dict | None = None,  # name -> per-camera policy (CameraConfig entry)
    ):
        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed")
        settings = settings or {}
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.cameras = dict(cameras)
        self.announce_enabled = set(announce_enabled)
        self.lines = {
            name: dict(settings.get(name, {}).get("lines") or {})
            for name in cameras
        }
        self.trackers = {
            name: DetectionTracker(
                cid,
                tuple(settings.get(name, {}).get("classes") or ANNOUNCE_CLASSES),
                float(settings.get(name, {}).get("cooldown_s", COOLDOWN_S)),
                zones=tuple(settings.get(name, {}).get("zones") or ()),
            )
            for name, cid in cameras.items()
        }
        self.connected = False
        self.last_error = ""
        self._client = None
        self._task = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._started = False
        self.announce_fn = None
        self._topic_re = re.compile(r"^scrypted/(\d+)/ObjectDetector$")
        self._id_to_name = {cid: name for name, cid in cameras.items()}

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self, announce_fn=None) -> None:
        """Start the broker connection + consumer task (call from the loop)."""
        if self._started:
            return
        self.announce_fn = announce_fn
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"glados-hub-{os.getpid()}"
        )
        if self.user:
            self._client.username_pw_set(self.user, self.password)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.connect_async(self.host, self.port)
        self._client.loop_start()  # background thread, auto-reconnect

        self._task = asyncio.create_task(self._consume())
        self._started = True
        logger.info(
            "camera monitor: connecting to %s:%s (cameras=%s)",
            self.host, self.port, self.cameras,
        )

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._started = False

    # ── paho callbacks (broker thread) ───────────────────────────────
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.connected = False
            self.last_error = str(reason_code)
            logger.error("camera monitor: broker rejected connection: %s", reason_code)
            return
        self.last_error = ""
        topics = [f"scrypted/{cid}/ObjectDetector" for cid in self.cameras.values()]
        try:
            # paho 2.x: a list must be (topic, qos) tuples — bare strings raise.
            client.subscribe([(t, 0) for t in topics])
        except Exception as e:
            self.last_error = f"subscribe failed: {e}"
            logger.error("camera monitor: subscribe failed: %s", e)
            return
        self.connected = True
        logger.info("camera monitor: connected, subscribed %s", topics)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected = False
        logger.warning("camera monitor: broker connection lost (%s)", reason_code)

    def _on_message(self, client, userdata, msg):
        m = self._topic_re.match(msg.topic)
        if not m or m.group(1) not in self._id_to_name:
            return
        name = self._id_to_name[m.group(1)]
        try:
            data = json.loads(msg.payload)
        except (ValueError, TypeError):
            return
        loop, queue = self._loop, self._queue
        if loop is None or queue is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(queue.put_nowait, (name, data.get("detections")))

    # ── event loop side ──────────────────────────────────────────────
    async def _consume(self):
        assert self._queue is not None
        while True:
            name, detections = await self._queue.get()
            try:
                fired = self.trackers[name].on_frame(detections)
            except Exception as e:  # never let one bad frame kill the feed
                logger.warning("camera %s: bad frame (%s)", name, e)
                continue
            if not fired or name not in self.announce_enabled:
                continue
            for cls in fired:
                line = self.lines.get(name, {}).get(cls) or ANNOUNCE_LINES.get(cls)
                if not line or self.announce_fn is None:
                    continue
                logger.info("camera: %s activity at %s — announcing", cls, name)
                try:
                    report = await self.announce_fn(line, "")
                    logger.info("camera: announce '%s' -> %s", line, report)
                except Exception as e:
                    logger.warning("camera: announce failed: %s", e)

    # ── status ───────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "host": self.host,
            "connected": self.connected,
            "last_error": self.last_error,
            "cameras": {name: tracker.camera_id for name, tracker in self.trackers.items()},
        }


# ── Module facade (the hub + skills talk to this, not the class) ───────


class CameraConfig:
    """Camera inventory + monitoring policy from config/cameras.json.

    {
      "cameras": {
        "<name>": {
          "label": "front yard",        # spoken name in reports/announcements
          "id": "65",                   # Scrypted camera device id
          "announce": true,             # auto-announce on the house speakers
          "classes": ["person", "package"],  # classes to track / announce
          "zones": [],                  # hub-side whitelist of NVR zone tags;
                                        # empty = accept whatever the NVR emits
          "cooldown_s": 90,
          "lines": {"person": "custom line"}
        }
      }
    }
    NOTE on zones: NVR-side zones (drawn in the Scrypted UI) already filter
    detections at the source. The hub-side `zones` list is an additional
    whitelist: when non-empty, a detection counts only if one of its zone
    tags is in the list. Package detections are tagged 'packageZone', so
    include that name to keep package announcements under a whitelist.
    """

    def __init__(self, cameras: dict[str, dict]):
        self.cameras = cameras

    @classmethod
    def load(cls, path: Path | None = None) -> "CameraConfig":
        p = Path(path) if path else CONFIG_PATH
        try:
            raw = json.loads(p.read_text())
        except Exception as e:
            logger.warning("camera config %s unreadable (%s) — cameras disabled", p, e)
            return cls({})
        cams: dict[str, dict] = {}
        for name, spec in (raw.get("cameras") or {}).items():
            if not isinstance(spec, dict) or not spec.get("id"):
                logger.warning("camera config: skipping %r (no id)", name)
                continue
            cams[str(name).lower()] = {
                "label": str(spec.get("label") or name),
                "id": str(spec["id"]),
                "announce": bool(spec.get("announce", False)),
                "classes": ([str(c).lower() for c in spec.get("classes", [])]
                            or list(ANNOUNCE_CLASSES)),
                "zones": [str(z) for z in spec.get("zones", [])],
                "cooldown_s": float(spec.get("cooldown_s", COOLDOWN_S)),
                "lines": {str(k): str(v) for k, v in (spec.get("lines") or {}).items()},
            }
        return cls(cams)

    def display(self, name: str) -> str:
        """Human/spoken name for a camera key ('front' -> 'front yard')."""
        spec = self.cameras.get(str(name).lower())
        return spec["label"] if spec else str(name)


CONFIG = CameraConfig.load()


def set_config(cfg: CameraConfig) -> None:
    global CONFIG
    CONFIG = cfg


MONITOR: CameraMonitor | None = None


def is_configured() -> bool:
    return bool(MQTT_HOST and MQTT_PASS and CONFIG.cameras)


def _build(announce_fn) -> CameraMonitor:
    """Construct a monitor from the current CONFIG + broker env."""
    cams = {name: spec["id"] for name, spec in CONFIG.cameras.items()}
    return CameraMonitor(
        MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASS,
        cams,
        announce_enabled=tuple(n for n, s in CONFIG.cameras.items() if s["announce"]),
        settings=dict(CONFIG.cameras),
    )


def start(announce_fn=None) -> bool:
    """Start the monitor (idempotent). Returns True when it's running."""
    global MONITOR
    if MONITOR is not None:
        return True
    if not is_configured():
        return False
    MONITOR = _build(announce_fn)
    MONITOR.start(announce_fn)
    return True


def reload() -> bool:
    """Re-read cameras.json and re-apply it to the live monitor.

    The config server calls this after a successful write. The broker
    connection is torn down and re-established (re-subscribe), so camera
    adds/removes/renumbers all work; detection history resets with it —
    acceptable for a deliberate config change. announce_fn is carried
    over from the running monitor. Returns True when a monitor is
    running afterwards (False = no longer configured).
    """
    global CONFIG, MONITOR
    CONFIG = CameraConfig.load()
    if not is_configured():
        logger.warning("camera config reload: no longer configured — monitor stopped")
        if MONITOR is not None:
            MONITOR.stop()
            MONITOR = None
        return False
    announce_fn = MONITOR.announce_fn if MONITOR is not None else None
    if MONITOR is not None:
        MONITOR.stop()
    MONITOR = _build(announce_fn)
    MONITOR.start(announce_fn)
    logger.info("camera config reloaded: %s", {n: s["id"] for n, s in CONFIG.cameras.items()})
    return True


def check_camera(camera: str = "front") -> str:
    """Tool body: honest live report of one camera (or why we can't see)."""
    camera = (camera or "front").strip().lower()
    display = CONFIG.display(camera)
    running = MONITOR.cameras if MONITOR else {}
    configured = set(CONFIG.cameras)
    if camera not in running and camera not in configured:
        return (f"unknown camera '{camera}' (configured: "
                f"{', '.join(sorted(configured)) or 'none'})")
    if camera not in running:
        # Configured, but no live monitor in this process (standalone /
        # tests, or the feed is down). Never guess — say so plainly.
        return (
            f"the {display} camera is configured but its feed is not running "
            "in this process — I cannot see outside. Say so plainly, do not "
            "guess."
        )
    report = MONITOR.trackers[camera].describe(MONITOR.connected)
    # The tracker reports by camera id ("65 camera"); the user says "front".
    return report.replace(f"{MONITOR.cameras[camera]} camera", f"{display} camera")


def status() -> dict | None:
    return MONITOR.status() if MONITOR else None


def describe_capabilities() -> str:
    """System-prompt block; empty when the feed is not configured."""
    if not is_configured():
        return ""
    which = "the " + " and the ".join(CONFIG.display(n) for n in CONFIG.cameras)
    return (
        "Cameras (object detection from the Reolink cameras via the local "
        "Scrypted NVR — no cloud involved): check_camera(camera) reports the "
        f"live view of {which} — which classes are ACTIVE and moving "
        "(person, package), last activity, and whether the feed is available. "
        "ALWAYS use it when asked whether anyone is outside or at the door, "
        "or whether a package arrived — never guess. If it reports the feed "
        "UNAVAILABLE, say plainly that you cannot see; do not describe what "
        "is there. The NVR only emits detection frames while it is tracking "
        "something, so a quiet yard is SILENT: 'no one moving' on a live feed "
        "is a real all-clear, and a very long silence is reported as a "
        "possibly-off camera. Cameras marked for announcements are also "
        "announced automatically on the house speakers — you do not need to "
        "call the tool for those."
    )



