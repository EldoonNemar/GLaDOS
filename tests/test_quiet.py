"""
Unit tests for quiet hours: window time math (same-day, midnight-crossing,
day filters, per-source scopes), timezone handling, config validation,
the module facade (load/reload, fail-open checks), and the four
enforcement points (camera announce, timer nag, response TTS, announce
tool) — plus the honest check_camera note for suppressed events.

Pure logic + fakes: no MQTT broker, no LLM, no speakers.
"""
import asyncio
import contextlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub import cameras  # noqa: E402
from glados_hub import quiet  # noqa: E402
from glados_hub import skills  # noqa: E402
from glados_hub.quiet import (  # noqa: E402
    QuietConfig, QuietConfigError, QuietWindow, _parse_time,
)

try:
    from glados_hub import hub  # noqa: E402
except ImportError:
    hub = None

# A fixed instant: 2026-11-22 04:00 UTC == Sun 22:00 America/Chicago.
T0_UTC = datetime(2026, 11, 22, 4, 0, tzinfo=timezone.utc)


class TestWindowMath(unittest.TestCase):
    def test_same_day_window(self):
        w = QuietWindow("noon", 12 * 60, 14 * 60)
        self.assertTrue(w.active_at(12 * 60, 0))      # start inclusive
        self.assertTrue(w.active_at(13 * 60 + 59, 0))
        self.assertFalse(w.active_at(14 * 60, 0))     # end exclusive
        self.assertFalse(w.active_at(11 * 60 + 59, 0))

    def test_midnight_crossing(self):
        w = QuietWindow("night", 22 * 60, 7 * 60)
        self.assertTrue(w.active_at(22 * 60, 0))
        self.assertTrue(w.active_at(23 * 60 + 59, 0))
        self.assertTrue(w.active_at(0, 0))
        self.assertTrue(w.active_at(6 * 60 + 59, 0))
        self.assertFalse(w.active_at(7 * 60, 0))
        self.assertFalse(w.active_at(21 * 60 + 59, 0))

    def test_day_filter(self):
        w = QuietWindow("weeknights", 21 * 60, 6 * 60, days=frozenset(range(5)))
        self.assertTrue(w.active_at(22 * 60, 2))      # Wednesday
        self.assertFalse(w.active_at(22 * 60, 5))     # Saturday
        self.assertFalse(w.active_at(22 * 60, 6))     # Sunday

    def test_per_source(self):
        w = QuietWindow("cams only", 0, 60 * 60, sources=frozenset({"camera"}))
        self.assertTrue(w.covers("camera"))
        self.assertFalse(w.covers("timers"))
        self.assertFalse(w.covers("responses"))


class TestQuietConfig(unittest.TestCase):
    def setUp(self):
        # night window in Chicago time; T0_UTC is Sun 22:00 Chicago
        self.cfg = QuietConfig("America/Chicago", [
            QuietWindow("night", 22 * 60, 7 * 60),
        ])

    def test_active_at_window_start_in_right_zone(self):
        # 22:00 CST == 04:00 UTC (Sun)
        self.assertTrue(self.cfg.is_active("camera", T0_UTC))
        # 06:00 CST == 12:00 UTC — still inside
        self.assertTrue(self.cfg.is_active("timers",
                                           datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)))
        # 08:00 CST == 14:00 UTC — outside
        self.assertFalse(self.cfg.is_active("responses",
                                            datetime(2026, 11, 22, 14, 0, tzinfo=timezone.utc)))

    def test_utc_config_not_shifted(self):
        cfg = QuietConfig("UTC", [QuietWindow("utc night", 22 * 60, 7 * 60)])
        self.assertTrue(cfg.is_active("camera", T0_UTC))            # 04:00 UTC: inside
        self.assertFalse(cfg.is_active("camera",
                                       datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)))

    def test_status_and_ends_at(self):
        st = self.cfg.status(T0_UTC)
        self.assertEqual(st["now"], "22:00")
        self.assertTrue(st["quiet"])
        self.assertEqual(st["windows"], ["night"])
        self.assertEqual(st["ends_at"], "07:00")
        st = self.cfg.status(datetime(2026, 11, 22, 14, 0, tzinfo=timezone.utc))
        self.assertFalse(st["quiet"])
        self.assertIsNone(st["ends_at"])

    def test_no_windows_means_never_quiet(self):
        cfg = QuietConfig("America/Chicago", [])
        self.assertFalse(cfg.is_active("camera", T0_UTC))
        self.assertEqual(cfg.describe(), "")

    def test_describe_full_silence_vs_partial(self):
        full = QuietConfig("America/Chicago", [QuietWindow("night", 22 * 60, 7 * 60)])
        d = full.describe()
        self.assertIn("you do not speak at all", d)
        self.assertIn("22:00-07:00 daily", d)
        partial = QuietConfig("America/Chicago", [
            QuietWindow("night", 22 * 60, 7 * 60, sources=frozenset({"camera"})),
        ])
        dp = partial.describe()
        self.assertIn("suppressed: camera alerts", dp)
        self.assertNotIn("you do not speak at all", dp)


class TestValidation(unittest.TestCase):
    GOOD = {"timezone": "America/Chicago", "windows": [
        {"name": "night", "start": "22:00", "end": "07:00"},
        {"name": "lunch", "start": "12:00", "end": "13:00",
         "days": [0, 1, 2, 3, 4], "sources": ["timers"]},
    ]}

    def test_good_payload(self):
        quiet.validate(self.GOOD)  # must not raise

    def test_defaults_fill_in(self):
        quiet.validate({"timezone": "UTC",
                        "windows": [{"name": "x", "start": "22:00", "end": "07:00"}]})

    def test_rejects(self):
        bads = [
            ("no timezone", {}),
            ("bad timezone", {"timezone": "Mars/Olympus"}),
            ("windows not a list", {"timezone": "UTC", "windows": "x"}),
            ("window not an object", {"timezone": "UTC", "windows": [1]}),
            ("no name", {"timezone": "UTC",
                         "windows": [{"start": "22:00", "end": "07:00"}]}),
            ("bad start", {"timezone": "UTC",
                           "windows": [{"name": "x", "start": "25:00", "end": "07:00"}]}),
            ("bad end", {"timezone": "UTC",
                         "windows": [{"name": "x", "start": "22:00", "end": "07"}]}),
            ("start == end", {"timezone": "UTC",
                              "windows": [{"name": "x", "start": "07:00", "end": "07:00"}]}),
            ("bad day", {"timezone": "UTC",
                         "windows": [{"name": "x", "start": "22:00", "end": "07:00",
                                      "days": [7]}]}),
            ("empty days", {"timezone": "UTC",
                            "windows": [{"name": "x", "start": "22:00", "end": "07:00",
                                         "days": []}]}),
            ("bad source", {"timezone": "UTC",
                            "windows": [{"name": "x", "start": "22:00", "end": "07:00",
                                         "sources": ["cats"]}]},
            ),
            ("empty sources", {"timezone": "UTC",
                               "windows": [{"name": "x", "start": "22:00", "end": "07:00",
                                            "sources": []}]}),
            ("duplicate names", {"timezone": "UTC", "windows": [
                {"name": "x", "start": "22:00", "end": "07:00"},
                {"name": "x", "start": "01:00", "end": "02:00"},
            ]}),
        ]
        for what, raw in bads:
            with self.assertRaises(QuietConfigError, msg=what):
                quiet.validate(raw)

    def test_parse_time(self):
        self.assertEqual(_parse_time("07:05", "t"), 7 * 60 + 5)
        self.assertEqual(_parse_time("7:05", "t"), 7 * 60 + 5)
        self.assertEqual(_parse_time("00:00", "t"), 0)
        self.assertEqual(_parse_time("23:59", "t"), 23 * 60 + 59)


class TestModuleFacade(unittest.TestCase):
    """load()/reload()/is_active() over a temp config file; checks fail
    open (never raise into the enforcement points)."""

    def setUp(self):
        self._saved_cfg = quiet.CONFIG
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Path(self.tmp.name) / "quiet_hours.json"
        self._patch = mock.patch.object(quiet, "CONFIG_PATH", self.f)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        quiet.CONFIG = self._saved_cfg
        self.tmp.cleanup()

    def test_missing_file_means_no_quiet_hours(self):
        quiet.reload()
        self.assertEqual(quiet.CONFIG.windows, [])
        self.assertFalse(quiet.is_active("camera"))

    def test_window_covering_now_is_active(self):
        # Build a guaranteed-active window around the current local minute.
        now = datetime.now(ZoneInfo("America/Chicago"))
        start = (now - timedelta(hours=1)).strftime("%H:%M")
        end = (now + timedelta(hours=1)).strftime("%H:%M")
        self.f.write_text(json.dumps({
            "timezone": "America/Chicago",
            "windows": [{"name": "now", "start": start, "end": end,
                         "sources": ["camera"]}],
        }))
        quiet.reload()
        self.assertTrue(quiet.is_active("camera"))
        self.assertFalse(quiet.is_active("timers"))       # not in sources
        self.assertTrue(quiet.CONFIG.status()["quiet"])

    def test_bad_file_on_reload_raises(self):
        self.f.write_text(json.dumps({"timezone": "Mars/Olympus"}))
        with self.assertRaises(QuietConfigError):
            quiet.reload()

    def test_check_fails_open_on_broken_config(self):
        quiet.CONFIG = mock.Mock()
        quiet.CONFIG.is_active.side_effect = RuntimeError("boom")
        self.assertFalse(quiet.is_active("camera"))       # error != quiet
        quiet.CONFIG = mock.Mock()
        quiet.CONFIG.status.side_effect = RuntimeError("boom")
        st = quiet.status()
        self.assertFalse(st["quiet"])


# ── Enforcement points ─────────────────────────────────────────────────


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send_bytes(self, data):
        self.sent.append(data)

    async def send_json(self, obj):
        self.sent.append(obj)


@unittest.skipUnless(hub is not None, "hub module not importable in this environment")
class TestTimerEnforcement(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = hub.active_sessions

    def tearDown(self):
        hub.active_sessions = self._saved
        hub.active_sessions.clear()

    def _session(self, device_id="dev", ws=None):
        s = hub.Session(device_id, None, "p")
        s.ws = ws  # normally set on connect (hub.py)
        hub.active_sessions[device_id] = s
        return s

    async def test_quiet_suppresses_nag(self):
        s = self._session()
        with mock.patch.object(quiet, "is_active", return_value=True):
            out = await hub._speak_to_one_device("dev", "pasta is up", "timer")
        self.assertIn("quiet hours", out)
        # No nag task, no nag note, speaking state untouched.
        self.assertFalse(s.glados_speaking)
        self.assertEqual(s.nag_note, "")

    async def test_not_quiet_still_nags(self):
        s = self._session(ws=_FakeWS())
        with mock.patch.object(quiet, "is_active", return_value=False), \
             mock.patch.object(hub, "NAG_GAP_S", 0.01), \
             mock.patch.object(hub, "NAG_MAX_LINES", 1), \
             mock.patch.object(hub, "NAG_BEAP_S", 0.01), \
             mock.patch.object(hub, "PLAYBACK_SETTLE_S", 0.01):
            out = await hub._speak_to_one_device("dev", "pasta is up", "timer")
            self.assertIn("ringing", out)
            # The nag task runs on this loop — let it finish INSIDE the
            # patch, or it would run with the real NAG constants.
            await asyncio.sleep(0.1)
        # The nag itself cleared its speaking state (it owns it); a beep
        # actually went out on the wire.
        self.assertFalse(s.glados_speaking)
        self.assertIsNone(s.speaking_task)
        self.assertTrue(any(f[0] == 0x11 for f in s.ws.sent))  # audio frame


@unittest.skipUnless(hub is not None, "hub module not importable in this environment")
class TestResponseEnforcement(unittest.IsolatedAsyncioTestCase):
    """During a quiet 'responses' window, tools still run but no TTS is
    sent and the reply is not spoken — the session state still clears."""

    def setUp(self):
        self._saved_sessions = hub.active_sessions
        self._saved_prompt = hub.SYSTEM_PROMPT
        self._saved_followup = hub.FOLLOWUP_WINDOW_S

    def tearDown(self):
        hub.active_sessions = self._saved_sessions
        hub.SYSTEM_PROMPT = self._saved_prompt
        hub.FOLLOWUP_WINDOW_S = self._saved_followup

    async def _respond_quiet(self, quiet_active):
        s = hub.Session("dev", None, "p")
        hub.active_sessions["dev"] = s
        ws = _FakeWS()
        tts_calls = []

        async def fake_stream(ws_, text, session):
            tts_calls.append(text)

        async def fake_llm(session, utterance="", extra_context=""):
            return "the lights are off now"

        with mock.patch.object(quiet, "is_active",
                               side_effect=lambda src, _a=None: quiet_active and src == "responses"), \
             mock.patch.object(hub, "_call_response_llm", fake_llm), \
             mock.patch.object(hub, "_stream_tts", fake_stream), \
             mock.patch.object(hub, "FOLLOWUP_WINDOW_S", 1.0):
            await hub._respond(s, ws, "turn off the lights", why="test")
        return s, tts_calls

    async def test_quiet_suppresses_tts_but_keeps_context(self):
        s, tts_calls = await self._respond_quiet(quiet_active=True)
        self.assertEqual(tts_calls, [])                       # nothing spoken
        self.assertFalse(s.glados_speaking)                   # state cleared
        # The reply is STILL in context so the next turn is coherent.
        self.assertIn("the lights are off now",
                      [m["content"] for m in s.messages if m["role"] == "assistant"])

    async def test_not_quiet_still_speaks(self):
        s, tts_calls = await self._respond_quiet(quiet_active=False)
        self.assertEqual(tts_calls, ["the lights are off now"])
        # Note: after REAL playback glados_speaking stays True until the
        # playback deadline passes (barge-in window) — by design. The
        # contrast that matters is the quiet path above, which clears it.
        self.assertIn("the lights are off now",
                      [m["content"] for m in s.messages if m["role"] == "assistant"])


class TestAnnounceToolEnforcement(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []

        async def fake_announce(message, speaker):
            self.calls.append((message, speaker))
            return "announced to: livingroom"

        skills.set_announce_fn(fake_announce)

    def tearDown(self):
        skills.set_announce_fn(None)

    async def test_quiet_suppresses_announce(self):
        with mock.patch.object(quiet, "is_active",
                               side_effect=lambda src, _a=None: src == "responses"):
            out = await skills.execute_tool("announce", {"message": "dinner is ready"},
                                            speaker="dev")
        self.assertIn("quiet hours", out)
        self.assertEqual(self.calls, [])                      # never spoken

    async def test_not_quiet_announces(self):
        with mock.patch.object(quiet, "is_active", return_value=False):
            out = await skills.execute_tool("announce", {"message": "dinner is ready"},
                                            speaker="dev")
        self.assertIn("announced to", out)
        self.assertEqual(self.calls, [("dinner is ready", "dev")])


@unittest.skipIf(cameras.mqtt is None, "paho-mqtt not installed in test env")
class TestCameraEnforcement(unittest.IsolatedAsyncioTestCase):
    """A camera event during a quiet 'camera' window is NOT announced, but is
    recorded (monitor.suppressed) so check_camera can report it honestly."""

    def setUp(self):
        self.mon = cameras.CameraMonitor(
            "127.0.0.1", 1883, "u", "p",
            {"front": "65"}, announce_enabled=("front",))
        self.calls = []

        async def fake_announce(message, speaker):
            self.calls.append((message, speaker))
            return "announced to: livingroom"

        self.mon.announce_fn = fake_announce

    def det(self, cls, moving):
        return {"className": cls, "movement": ({"moving": moving} if moving else {})}

    async def _feed(self, *items):
        self.mon._queue = asyncio.Queue()
        for item in items:
            self.mon._queue.put_nowait(item)
        task = asyncio.create_task(self.mon._consume())
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_quiet_suppresses_announce_but_records(self):
        with mock.patch.object(quiet, "is_active",
                               side_effect=lambda src, _a=None: src == "camera"):
            await self._feed(("front", [self.det("person", True)]),
                             ("front", [self.det("person", True)]))
        self.assertEqual(self.calls, [])                      # nothing spoken
        self.assertEqual(len(self.mon.suppressed), 1)         # recorded once
        self.assertEqual(self.mon.suppressed[-1]["camera"], "front")
        self.assertEqual(self.mon.suppressed[-1]["cls"], "person")

    async def test_not_quiet_announces(self):
        with mock.patch.object(quiet, "is_active", return_value=False):
            await self._feed(("front", [self.det("person", True)]),
                             ("front", [self.det("person", True)]))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.mon.suppressed), 0)

    async def test_check_camera_reports_suppressed(self):
        # Record a suppression, then the tool report must mention it.
        self.mon.suppressed.append({"ts": "22:05", "camera": "front", "cls": "person"})
        with mock.patch.object(cameras, "MONITOR", self.mon), \
             mock.patch.object(cameras, "CONFIG",
                               cameras.CameraConfig({"front": {"id": "65", "label": "front yard"}})):
            report = cameras.check_camera("front")
        self.assertIn("suppressed by quiet hours", report)
        self.assertIn("person", report)
        self.assertIn("22:05", report)


if __name__ == "__main__":
    unittest.main()