"""
Unit tests for the tool layer: Kasa light registry/resolution, timers
(fire to the CALLER's device only), skills dispatch, and the hub's
announce/speak functions.

Pure logic + fakes: no Kasa hardware, no radiance, no network.
"""
import asyncio
import contextlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub import cameras  # noqa: E402
from glados_hub import kasa_lights  # noqa: E402
from glados_hub import skills  # noqa: E402
from glados_hub import timers  # noqa: E402

try:
    from glados_hub import hub  # noqa: E402
except ImportError:
    hub = None

CONFIG = {
    "lights": {
        "livingroom ceiling": "192.168.1.9",
        "livingroom left": "192.168.1.18",
        "livingroom right": "192.168.1.5",
    },
    "types": {
        "livingroom ceiling": "plug",
        "livingroom left": "bulb",
        "livingroom right": "bulb",
    },
    "groups": {
        "livingroom": ["livingroom ceiling", "livingroom left", "livingroom right"],
    },
}

ALL_THREE = ["livingroom ceiling", "livingroom left", "livingroom right"]


def make_registry():
    # TemporaryDirectory (not NamedTemporaryFile(delete=False)): the file is
    # read fully by LightRegistry.load, so it can be removed immediately.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "kasa.json"
        p.write_text(json.dumps(CONFIG))
        return kasa_lights.LightRegistry.load(p)


class TestResolve(unittest.TestCase):
    """'turn off the livingroom lights' must reach all three devices."""

    def setUp(self):
        self.reg = make_registry()

    def test_exact_light(self):
        self.assertEqual(self.reg.resolve("livingroom ceiling"), ["livingroom ceiling"])

    def test_exact_group(self):
        self.assertEqual(self.reg.resolve("livingroom"), ALL_THREE)

    def test_noise_stripped_group(self):
        self.assertEqual(self.reg.resolve("the livingroom lights"), ALL_THREE)

    def test_partial_light(self):
        self.assertEqual(self.reg.resolve("the ceiling light"), ["livingroom ceiling"])
        self.assertEqual(self.reg.resolve("right"), ["livingroom right"])

    def test_unknown(self):
        self.assertEqual(self.reg.resolve("hallway"), [])
        self.assertEqual(self.reg.resolve(""), [])
        self.assertEqual(self.reg.resolve("lights"), [])


class TestSetLights(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        kasa_lights.set_registry(make_registry())
        self.calls = []

    async def _fake_control(self, name, ip, state, brightness):
        self.calls.append((name, ip, state, brightness))
        if ip == "192.168.1.18":
            return False, f"{name}: no response"
        return True, f"{name}: {state}" + (f" at {brightness}%" if brightness else "")

    def tearDown(self):
        kasa_lights.set_registry(kasa_lights.LightRegistry.load())

    async def test_group_off_with_partial_failure(self):
        with mock.patch.object(kasa_lights, "_control_one", self._fake_control):
            report = await kasa_lights.set_lights("livingroom lights", "off")
        self.assertIn("livingroom ceiling: off", report)
        self.assertIn("livingroom left: no response", report)
        self.assertEqual(len(self.calls), 3)

    async def test_unknown_target_lists_known(self):
        report = await kasa_lights.set_lights("hallway", "on")
        self.assertIn("no light or group matching", report)
        self.assertIn("livingroom ceiling", report)

    async def test_bad_state(self):
        report = await kasa_lights.set_lights("livingroom", "sideways")
        self.assertIn("state must be", report)

    async def test_brightness_clamped(self):
        with mock.patch.object(kasa_lights, "_control_one", self._fake_control):
            await kasa_lights.set_lights("livingroom right", "on", brightness=150)
        self.assertEqual(self.calls[-1][3], 100)

    async def test_brightness_ignored_when_off(self):
        with mock.patch.object(kasa_lights, "_control_one", self._fake_control):
            await kasa_lights.set_lights("livingroom right", "off", brightness=50)
        self.assertIsNone(self.calls[-1][3])


class TestControlVerification(unittest.IsolatedAsyncioTestCase):
    """Verify-after-write: the tool must report what the device ACTUALLY did,
    not what it ACKed. Exercises the real _control_one against a fake kasa
    module (injected into sys.modules), so the verification+retry logic runs.
    """

    class FakeDevice:
        def __init__(self, on=True, flaky=False, stubborn=False):
            self._on = on
            self.flaky = flaky        # first command dropped, second works
            self.stubborn = stubborn  # commands never take effect
            self.on_calls = 0
            self.off_calls = 0
            self.modules = {}
            self.light_state = {"brightness": 100}

        @property
        def is_on(self):
            return self._on

        async def update(self):
            pass

        async def turn_on(self):
            self.on_calls += 1
            if self.stubborn:
                return
            if self.flaky and self.on_calls == 1:
                return  # dropped
            self._on = True

        async def turn_off(self):
            self.off_calls += 1
            if self.stubborn:
                return
            self._on = False

    @staticmethod
    def _fake_kasa_module(dev):
        import types
        mod = types.ModuleType("kasa")

        class _Discover:
            @staticmethod
            async def discover_single(ip, timeout=6):
                return dev

        mod.Discover = _Discover
        mod.LightState = lambda **kw: kw
        return mod

    def setUp(self):
        kasa_lights.set_registry(make_registry())

    def tearDown(self):
        kasa_lights.set_registry(kasa_lights.LightRegistry.load())

    async def test_on_verified_success(self):
        dev = self.FakeDevice(on=False)
        with mock.patch.dict(sys.modules, {"kasa": self._fake_kasa_module(dev)}):
            report = await kasa_lights.set_lights("livingroom right", "on")
        self.assertEqual(report, "livingroom right: on")
        self.assertFalse("FAILED" in report)
        self.assertTrue(dev.is_on)
        self.assertEqual(dev.on_calls, 1)

    async def test_on_dropped_once_retried_success(self):
        dev = self.FakeDevice(on=False, flaky=True)
        with mock.patch.dict(sys.modules, {"kasa": self._fake_kasa_module(dev)}):
            report = await kasa_lights.set_lights("livingroom right", "on")
        self.assertEqual(report, "livingroom right: on")
        self.assertEqual(dev.on_calls, 2)  # retry happened
        self.assertTrue(dev.is_on)

    async def test_on_still_off_reports_failure(self):
        dev = self.FakeDevice(on=False, stubborn=True)
        with mock.patch.dict(sys.modules, {"kasa": self._fake_kasa_module(dev)}):
            report = await kasa_lights.set_lights("livingroom right", "on")
        # "after 2 attempts" tells the LLM she retried, so she can say so.
        self.assertIn("livingroom right: on FAILED after 2 attempts, still off", report)
        self.assertEqual(dev.on_calls, 2)  # tried twice, then gave up
        self.assertFalse(dev.is_on)

    async def test_off_verified_success(self):
        dev = self.FakeDevice(on=True)
        with mock.patch.dict(sys.modules, {"kasa": self._fake_kasa_module(dev)}):
            report = await kasa_lights.set_lights("livingroom right", "off")
        self.assertEqual(report, "livingroom right: off")
        self.assertFalse(dev.is_on)

    async def test_off_still_on_reports_failure(self):
        dev = self.FakeDevice(on=True, stubborn=True)
        with mock.patch.dict(sys.modules, {"kasa": self._fake_kasa_module(dev)}):
            report = await kasa_lights.set_lights("livingroom right", "off")
        self.assertIn("livingroom right: off FAILED after 2 attempts, still on", report)

    async def test_group_mixed_verified_results(self):
        """One stubborn bulb in a group: report shows per-device truth."""
        good = self.FakeDevice(on=False)
        bad = self.FakeDevice(on=False, stubborn=True)
        devices = {"192.168.1.18": bad}

        import types
        mod = types.ModuleType("kasa")

        class _Discover:
            @staticmethod
            async def discover_single(ip, timeout=6):
                return devices.get(ip, good)

        mod.Discover = _Discover
        mod.LightState = lambda **kw: kw
        with mock.patch.dict(sys.modules, {"kasa": mod}):
            report = await kasa_lights.set_lights("livingroom", "on")
        self.assertIn("livingroom left: on FAILED after 2 attempts, still off", report)
        self.assertIn("livingroom ceiling: on", report)
        self.assertEqual(bad.on_calls, 2)

    def test_capability_text_explains_verification(self):
        cap = skills.describe_capabilities().lower()
        self.assertIn("verifies", cap)
        self.assertIn("failed", cap)


class TestGetLights(unittest.IsolatedAsyncioTestCase):
    """State questions must come from a read, not a guess."""

    def setUp(self):
        kasa_lights.set_registry(make_registry())

    def tearDown(self):
        kasa_lights.set_registry(kasa_lights.LightRegistry.load())

    async def test_group_state_read(self):
        async def fake_read(name, ip):
            if ip == "192.168.1.5":
                return f"{name}: on at 80%"
            return f"{name}: off"
        with mock.patch.object(kasa_lights, "_read_one", fake_read):
            report = await kasa_lights.get_lights("the livingroom lights")
        self.assertEqual(report, "livingroom ceiling: off; livingroom left: off; "
                                  "livingroom right: on at 80%")

    async def test_single_light_read(self):
        async def fake_read(name, ip):
            return f"{name}: on"
        with mock.patch.object(kasa_lights, "_read_one", fake_read):
            report = await kasa_lights.get_lights("ceiling")
        self.assertEqual(report, "livingroom ceiling: on")

    async def test_unknown_target(self):
        report = await kasa_lights.get_lights("hallway")
        self.assertIn("no light or group matching", report)
        self.assertIn("livingroom ceiling", report)

    async def test_dispatch(self):
        async def fake_read(name, ip):
            return f"{name}: off"
        with mock.patch.object(kasa_lights, "_read_one", fake_read):
            out = await skills.execute_tool("get_lights", {"target": "livingroom"}, speaker="a")
        self.assertIn("off", out)
        self.assertEqual(out.count("livingroom"), 3)


class TestTimers(unittest.IsolatedAsyncioTestCase):
    """A timer must fire to the speaker's OWN device only."""

    def setUp(self):
        self.store = timers.TimerStore()
        self.spoken = []
        timers.set_speak_fn(self._speak)

    async def _speak(self, device_id, text, label=""):
        self.spoken.append((device_id, text, label))
        return "spoken"

    def tearDown(self):
        timers.set_speak_fn(None)

    async def test_fires_for_caller_only(self):
        t1 = self.store.add("pasta", 0.1, "pi4-spoke-001", min_seconds=0.05)
        t2 = self.store.add("dinner", 0.1, "pi4-spoke-002", min_seconds=0.05,
                            message="dinner is ready")
        self.assertNotIsInstance(t1, str)
        self.assertNotIsInstance(t2, str)
        await asyncio.sleep(0.3)
        self.assertEqual(len(self.spoken), 2)
        self.assertEqual({d for d, _, _ in self.spoken}, {"pi4-spoke-001", "pi4-spoke-002"})
        self.assertIn("dinner is ready", " | ".join(t for _, t, _ in self.spoken))
        # the fired label rides along, so the hub can build the nag note
        self.assertEqual({lab for _, _, lab in self.spoken}, {"pasta", "dinner"})

    async def test_cancel(self):
        self.store.add("pasta", 0.2, "a", min_seconds=0.05)
        self.assertIn("cancelled", self.store.cancel("pasta"))
        await asyncio.sleep(0.3)
        self.assertEqual(self.spoken, [])

    async def test_duplicate_replaces(self):
        self.store.add("pasta", 0.5, "a", min_seconds=0.05)
        self.store.add("pasta", 0.1, "a", min_seconds=0.05)
        await asyncio.sleep(0.3)
        self.assertEqual(len(self.spoken), 1)  # only the replacement fires

    async def test_light_action(self):
        async def fake_set_lights(target, state, brightness=None):
            return f"livingroom ceiling: {state}"
        with mock.patch.object(kasa_lights, "set_lights", fake_set_lights):
            self.store.add("lights off", 0.1, "a", action="light_off",
                           target="livingroom", min_seconds=0.05)
            await asyncio.sleep(0.3)
        self.assertEqual(len(self.spoken), 1)
        self.assertEqual(self.spoken[0][0], "a")
        self.assertIn("are off", self.spoken[0][1])

    async def test_light_action_failure_reported(self):
        async def fake_set_lights(target, state, brightness=None):
            return f"livingroom ceiling: no response"
        with mock.patch.object(kasa_lights, "set_lights", fake_set_lights):
            self.store.add("lights on", 0.1, "a", action="light_on",
                           target="livingroom", min_seconds=0.05)
            await asyncio.sleep(0.3)
        self.assertIn("didn't answer", self.spoken[0][1])

    async def test_bad_action_rejected(self):
        out = self.store.add("x", 60, "a", action="teleport", min_seconds=0.05)
        self.assertIsInstance(out, str)
        self.assertIn("on_fire must be", out)


class TestSkillsDispatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        kasa_lights.set_registry(make_registry())
        skills.TIMER_STORE = timers.TimerStore()
        skills.set_announce_fn(self._fake_announce)
        self.announcements = []

    async def _fake_announce(self, message, speaker):
        self.announcements.append((message, speaker))
        return "announced to: kitchen"

    def tearDown(self):
        kasa_lights.set_registry(kasa_lights.LightRegistry.load())
        skills.set_announce_fn(None)

    async def test_unknown_tool(self):
        out = await skills.execute_tool("explode_thing", {}, speaker="a")
        self.assertIn("unknown tool", out)

    async def test_announce_plumbing(self):
        out = await skills.execute_tool("announce", {"message": "dinner is ready"}, speaker="a")
        self.assertIn("announced to", out)
        self.assertEqual(self.announcements[0], ("dinner is ready", "a"))

    async def test_set_timer_rejects_bad_minutes(self):
        out = await skills.execute_tool("set_timer", {"label": "x", "minutes": "soon"}, speaker="a")
        self.assertIn("minutes must be a number", out)

    async def test_set_timer_rejects_too_long(self):
        out = await skills.execute_tool("set_timer", {"label": "x", "minutes": 9999}, speaker="a")
        self.assertIn("between 1 and 1440", out)

    async def test_set_timer_ok(self):
        out = await skills.execute_tool(
            "set_timer", {"label": "Pasta", "minutes": 10, "message": "pasta is ready"},
            speaker="a")
        self.assertIn("set for 10 minutes", out)
        self.assertIn("pasta", skills.TIMER_STORE.list_active())

    async def test_cancel_and_list(self):
        await skills.execute_tool("set_timer", {"label": "pasta", "minutes": 10}, speaker="a")
        out = await skills.execute_tool("cancel_timer", {"label": "pas"}, speaker="a")
        self.assertIn("cancelled", out)
        self.assertEqual(await skills.execute_tool("list_timers", {}, speaker="a"),
                         "no active timers")

    async def test_set_lights_validation(self):
        out = await skills.execute_tool("set_lights", {"target": "x", "state": "up"}, speaker="a")
        self.assertIn("state must be", out)

    async def test_tool_timeout(self):
        async def slow(*a, **k):
            await asyncio.sleep(5)
        with mock.patch.object(kasa_lights, "set_lights", slow), \
             mock.patch.object(skills, "TOOL_TIMEOUT_S", 0.05), \
             mock.patch.object(kasa_lights, "OP_TIMEOUT_S", 0.05):
            out = await skills.execute_tool(
                "set_lights", {"target": "livingroom", "state": "on"}, speaker="a")
        self.assertIn("timed out", out)


class TestShippedConfig(unittest.TestCase):
    """The baked-in kasa.json must be internally consistent."""

    def test_shipped_config_is_consistent(self):
        reg = kasa_lights.LightRegistry.load()
        self.assertTrue(reg.lights, "kasa.json should define at least one light")
        for g, members in reg.groups.items():
            for m in members:
                self.assertIn(m, reg.lights, f"group {g} references unknown light {m}")
        self.assertEqual(kasa_lights.REGISTRY.lights, reg.lights)

    def test_describe_capabilities_not_empty(self):
        self.assertIn("set_lights", skills.describe_capabilities())
        self.assertIn("livingroom", skills.describe_capabilities())
        self.assertIn("ringing", skills.describe_capabilities())
        self.assertIn("get_lights", skills.describe_capabilities())
        # Guard for the "she says the lights are on but they're off" bug:
        # the prompt must forbid claiming state/action without a tool result.
        self.assertIn("never claim", skills.describe_capabilities().lower())


class TestLlamaRequestTools(unittest.TestCase):
    def test_tools_included_only_when_given(self):
        from glados_hub import llm
        body = llm.build_llama_request([{"role": "user", "content": "hi"}])
        self.assertNotIn("tools", body)
        body = llm.build_llama_request([{"role": "user", "content": "hi"}],
                                       tools=[{"type": "function"}])
        self.assertEqual(body["tools"], [{"type": "function"}])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


@unittest.skipIf(hub is None, "fastapi not installed in test env")
class TestAnnounce(unittest.IsolatedAsyncioTestCase):
    class FakeSession:
        def __init__(self, device_id, speaking=False):
            self.device_id = device_id
            self.glados_speaking = speaking
            self.ws = object()

    def setUp(self):
        self._saved = dict(hub.active_sessions)
        hub.active_sessions.clear()

    def tearDown(self):
        hub.active_sessions.clear()
        hub.active_sessions.update(self._saved)

    async def test_announce_skips_speaker_and_reports_busy(self):
        hub.active_sessions.update({
            "pi4-spoke-001": self.FakeSession("pi4-spoke-001"),          # speaker
            "pi4-spoke-002": self.FakeSession("pi4-spoke-002"),          # receives
            "pi4-spoke-003": self.FakeSession("pi4-spoke-003", True),    # busy
        })
        spoken = []

        async def fake_speak_raw(session, ws, text):
            spoken.append((session.device_id, text))

        with mock.patch.object(hub, "_speak_raw", fake_speak_raw):
            out = await hub._announce_to_devices("dinner is ready", speaker="pi4-spoke-001")
        self.assertEqual([d for d, _ in spoken], ["pi4-spoke-002"])
        self.assertIn("pi4-spoke-003", out)  # busy ones are reported, not silent

    async def test_announce_no_others(self):
        hub.active_sessions.update({
            "pi4-spoke-001": self.FakeSession("pi4-spoke-001"),
        })
        out = await hub._announce_to_devices("hello", speaker="pi4-spoke-001")
        self.assertIn("no other speakers", out)

    async def test_timer_speak_gone_device(self):
        out = await hub._speak_to_one_device("ghost", "hi")
        self.assertEqual(out, "device not connected")


class NagFakeWS:
    """Minimal stand-in for a spoke WebSocket: records frames sent to it."""

    def __init__(self):
        self.frames = []

    async def send_bytes(self, data):
        self.frames.append(data)


@unittest.skipIf(hub is None, "fastapi not installed in test env")
class TestNag(unittest.IsolatedAsyncioTestCase):
    """Timer nag: beep-only alarm, repeated until barge-in or budget; the mic
    must stay in barge-in monitoring mode (any speech = stop) the whole time."""

    class FakeSession:
        def __init__(self, device_id="pi4-spoke-001"):
            self.device_id = device_id
            self.ws = NagFakeWS()
            self.glados_speaking = False
            self.speaking_task = None
            self.playback_deadline = 0.0
            self.bargein_hits = 0
            self.echo_level = 0.0

    def setUp(self):
        self.beep_times = []

    def _beeps(self, ws):
        return [f for f in ws.frames if f[0] == 0x11]

    def _leds(self, ws):
        return [f[1] for f in ws.frames if f[0] == 0x15]

    async def _run_nag(self, session, first, gap=0.01, max_lines=3):
        with mock.patch.object(hub, "NAG_GAP_S", gap), \
             mock.patch.object(hub, "NAG_MAX_LINES", max_lines), \
             mock.patch.object(hub, "NAG_BEAP_S", 0.01), \
             mock.patch.object(hub, "PLAYBACK_SETTLE_S", 0.01):
            await hub._nag(session, first)

    async def test_nag_sends_only_beeps(self):
        s = self.FakeSession()
        await self._run_nag(s, "pasta is up", max_lines=3)
        beeps = self._beeps(s.ws)
        self.assertEqual(len(beeps), 3)
        for f in beeps:
            self.assertEqual(f[1:], hub._nag_beep())  # each frame is the ding
        # state fully released afterwards
        self.assertFalse(s.glados_speaking)
        self.assertIsNone(s.speaking_task)

    async def test_nag_stops_on_barge_in(self):
        s = self.FakeSession()
        with mock.patch.object(hub, "NAG_GAP_S", 0.02), \
             mock.patch.object(hub, "NAG_MAX_LINES", 10), \
             mock.patch.object(hub, "NAG_BEAP_S", 0.01), \
             mock.patch.object(hub, "PLAYBACK_SETTLE_S", 0.01):
            task = asyncio.create_task(hub._nag(s, "pasta is up"))
            await asyncio.sleep(0.06)
            self.assertIsNotNone(s.speaking_task)
            s.speaking_task.cancel()  # exactly what _barge_in does
            with contextlib.suppress(asyncio.CancelledError):
                await task
        beeps = self._beeps(s.ws)
        self.assertGreater(len(beeps), 0)
        self.assertLess(len(beeps), 10)  # stopped early, didn't run out
        self.assertFalse(s.glados_speaking)
        self.assertIsNone(s.speaking_task)

    async def test_nag_holds_mic_muted_between_beeps(self):
        s = self.FakeSession()
        real_send = s.ws.send_bytes

        async def recording_send(data):
            if data[0] == 0x11:
                self.beep_times.append(time.monotonic())
            await real_send(data)

        s.ws.send_bytes = recording_send
        with mock.patch.object(hub, "NAG_GAP_S", 0.05), \
             mock.patch.object(hub, "NAG_MAX_LINES", 3), \
             mock.patch.object(hub, "NAG_BEAP_S", 0.01), \
             mock.patch.object(hub, "PLAYBACK_SETTLE_S", 0.02):
            await hub._nag(s, "pasta is up")
        # beep N+1 must not start before beep N's audio had finished playing
        self.assertGreater(len(self.beep_times), 1)
        for a, b in zip(self.beep_times, self.beep_times[1:]):
            self.assertGreater(b, a)

    async def test_nag_sets_alarm_led(self):
        s = self.FakeSession()
        await self._run_nag(s, "pasta is up", max_lines=1)
        leds = self._leds(s.ws)
        self.assertEqual(leds[0], hub.LED_ALARM)   # alarm on at start
        self.assertEqual(leds[-1], hub.LED_OFF)    # released at end

    async def test_speak_to_one_device_starts_nag(self):
        s = self.FakeSession("dev1")
        hub.active_sessions["dev1"] = s
        try:
            started = []

            async def fake_nag(session, text):
                started.append(text)

            with mock.patch.object(hub, "_nag", fake_nag):
                report = await hub._speak_to_one_device("dev1", "pasta is up")
            await asyncio.sleep(0)  # let the spawned nag task get a turn
            self.assertIn("repeats", report)
            self.assertEqual(started, ["pasta is up"])
        finally:
            hub.active_sessions.pop("dev1", None)

    async def test_speak_to_one_device_waits_for_busy_device(self):
        s = self.FakeSession("dev2")
        s.glados_speaking = True
        hub.active_sessions["dev2"] = s

        async def release():
            await asyncio.sleep(0.05)
            s.glados_speaking = False

        try:
            started = []

            async def fake_nag(session, text):
                started.append(text)

            async def runner():
                with mock.patch.object(hub, "_nag", fake_nag):
                    return await hub._speak_to_one_device("dev2", "hi")

            t_release = asyncio.create_task(release())
            report = await runner()
            await t_release
            await asyncio.sleep(0)  # let the spawned nag task get a turn
            # waited for the device to settle instead of dropping
            self.assertIn("repeats", report)
            self.assertEqual(started, ["hi"])
        finally:
            hub.active_sessions.pop("dev2", None)

    def test_nag_beep_shape(self):
        import numpy as np
        b = hub._nag_beep()
        self.assertEqual(len(b), int(0.22 * hub.TTS_SAMPLE_RATE) * 2)  # 16-bit mono
        samples = np.frombuffer(b, dtype=np.int16)
        peak = int(np.max(np.abs(samples)))
        self.assertLessEqual(peak, int(0.29 * 32768))  # below full scale (no clipping)
        self.assertGreater(peak, 1000)                 # but actually audible


@unittest.skipIf(hub is None, "fastapi not installed in test env")
class TestLedFrames(unittest.IsolatedAsyncioTestCase):
    """The 0x15 LED-ring state frame rides along with status (the Pi4 spoke
    ignores unknown frame types, so this is XVF3800-only)."""

    class FakeWS:
        def __init__(self):
            self.frames = []

        async def send_bytes(self, data):
            self.frames.append(data)

    def _leds(self, ws):
        return [f[1] for f in ws.frames if f[0] == 0x15]

    async def test_status_thinking_maps_to_think_led(self):
        ws = self.FakeWS()
        await hub._send_status(ws, "thinking...")
        self.assertEqual(self._leds(ws), [hub.LED_THINK])

    async def test_status_listening_maps_to_listen_led(self):
        ws = self.FakeWS()
        await hub._send_status(ws, "listening...")
        self.assertEqual(self._leds(ws), [hub.LED_LISTEN])

    async def test_status_frame_still_sent(self):
        ws = self.FakeWS()
        await hub._send_status(ws, "thinking...")
        self.assertEqual(ws.frames[0][0], 0x13)
        self.assertIn(b"thinking", ws.frames[0])

    async def test_dead_ws_does_not_raise(self):
        class DeadWS:
            async def send_bytes(self, data):
                raise ConnectionError("gone")

        await hub._send_status(DeadWS(), "thinking...")  # must not raise
        await hub._send_led(DeadWS(), hub.LED_ALARM)     # must not raise

    async def test_send_led_payload(self):
        ws = self.FakeWS()
        await hub._send_led(ws, hub.LED_ALARM)
        self.assertEqual(ws.frames, [bytes([0x15, hub.LED_ALARM])])


@unittest.skipIf(hub is None, "fastapi not installed in test env")
class TestFollowupWindowGate(unittest.IsolatedAsyncioTestCase):
    """Inside the follow-up window (GLaDOS just spoke, no wake word), an
    utterance is gated in CONTINUATION mode: related -> respond, off-topic
    -> ignored, and the window keeps running either way."""

    class FakeSession:
        def __init__(self, followup_until=0.0):
            self.device_id = "esp32-s3-001"
            self.process_lock = asyncio.Lock()
            self.followup_until = followup_until
            self.last_name = ""
            self.location = ""
            self.nag_note = ""
            self.nag_note_until = 0.0

        def prune_stale(self):
            pass

        def add_heard(self, text):
            pass

        def register_final(self, text, conversational=False):
            return True

    async def _run(self, text, followup_until, gate_answer):
        s = self.FakeSession(followup_until=followup_until)
        responded, gate_modes = [], []

        async def fake_gate(session, text, followup=False):
            gate_modes.append(followup)
            return gate_answer

        async def fake_respond(session, ws, text, why, extra_context=""):
            responded.append(why)

        with mock.patch.object(hub, "_tiny_llm_says_respond", fake_gate), \
             mock.patch.object(hub, "_respond", fake_respond), \
             mock.patch.object(hub, "_weather_context", mock.AsyncMock(return_value="")):
            await hub._on_final(s, None, text)
        return responded, gate_modes, s

    async def test_inside_window_continuation_responds(self):
        responded, gate_modes, _ = await self._run(
            "for tomorrow", time.monotonic() + 10, gate_answer=True)
        self.assertEqual(responded, ["follow-up"])
        self.assertEqual(gate_modes, [True])  # gated in follow-up mode

    async def test_inside_window_offtopic_ignored_window_keeps_running(self):
        window = time.monotonic() + 10
        responded, gate_modes, s = await self._run(
            "did you hear that bird", window, gate_answer=False)
        self.assertEqual(responded, [])
        self.assertEqual(gate_modes, [True])
        self.assertEqual(s.followup_until, window)  # untouched — a real
        # follow-up a few seconds later still lands

    async def test_outside_window_uses_gate_mode(self):
        responded, gate_modes, _ = await self._run(
            "what time is it", 0.0, gate_answer=True)
        self.assertEqual(responded, ["decision"])
        self.assertEqual(gate_modes, [False])

    async def test_inside_window_wake_word_skips_gate(self):
        responded, gate_modes, _ = await self._run(
            "hey glados, for tomorrow", time.monotonic() + 10, gate_answer=False)
        self.assertEqual(responded, ["wake-word"])
        self.assertEqual(gate_modes, [])


class TestTimerCancelFuzzy(unittest.IsolatedAsyncioTestCase):
    """cancel() must match the LLM's approximate phrasing, both directions."""

    async def test_cancel_wrapped_label(self):
        s = timers.TimerStore()
        s.add("pasta", 600, "dev1")
        out = s.cancel("timer pasta")
        self.assertIn("cancelled", out)
        self.assertEqual(s.list_active(), "no active timers")

    async def test_cancel_label_inside_query(self):
        s = timers.TimerStore()
        s.add("pasta", 600, "dev1")
        out = s.cancel("the pasta timer")
        self.assertIn("cancelled", out)

    async def test_cancel_no_match_lists_what_exists(self):
        s = timers.TimerStore()
        s.add("soup", 600, "dev1")
        out = s.cancel("pasta")
        self.assertIn("no matching timer", out)
        self.assertIn("soup", out)


@unittest.skipIf(hub is None, "fastapi not installed in test env")
class TestNagNote(unittest.IsolatedAsyncioTestCase):
    """The fired timer is gone from the store before the user's 'stop'
    arrives, so the gate + response prompts must carry a 'a nag is/just was
    happening' note with the timer's label — or the LLM answers 'no record'."""

    def _session(self):
        from glados_hub.session import Session
        return Session("dev1", None, "sys")

    def _armed(self, session, label="pasta"):
        session.nag_note = hub._nag_note(label)
        session.nag_note_until = time.monotonic() + 300

    class NagNoteFakeWS:
        async def send_bytes(self, data):
            pass

    class NagNoteSession:
        def __init__(self, device_id="dev1", speaking=False):
            self.device_id = device_id
            self.ws = TestNagNote.NagNoteFakeWS()
            self.glados_speaking = speaking
            self.speaking_task = None
            self.playback_deadline = 0.0
            self.bargein_hits = 0
            self.echo_level = 0.0
            self.nag_note = ""
            self.nag_note_until = 0.0

    def setUp(self):
        self._saved = dict(hub.active_sessions)
        hub.active_sessions.clear()

    def tearDown(self):
        hub.active_sessions.clear()
        hub.active_sessions.update(self._saved)

    async def test_note_set_when_nag_starts(self):
        s = self.NagNoteSession()
        hub.active_sessions["dev1"] = s
        started = []

        async def fake_nag(session, text):
            started.append(text)

        with mock.patch.object(hub, "_nag", fake_nag):
            report = await hub._speak_to_one_device("dev1", "pasta timer is up.", "pasta")
        await asyncio.sleep(0)  # let the spawned nag task get a turn
        self.assertIn("repeats", report)
        self.assertEqual(started, ["pasta timer is up."])
        self.assertIn("INTERNAL NOTE", s.nag_note)
        self.assertIn("pasta", s.nag_note)
        self.assertGreater(s.nag_note_until, time.monotonic())

    async def test_note_not_set_when_device_missing_or_busy(self):
        self.assertEqual(
            await hub._speak_to_one_device("ghost", "hi", "pasta"), "device not connected")
        s = self.NagNoteSession("dev2", speaking=True)
        hub.active_sessions["dev2"] = s
        with mock.patch.object(hub, "NAG_BUSY_WAIT_S", 0):
            out = await hub._speak_to_one_device("dev2", "hi", "pasta")
        self.assertIn("busy", out)
        self.assertEqual(s.nag_note, "")  # no nag started, no note

    async def test_response_prompt_includes_note(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        self._armed(s)
        captured = {}

        async def fake_msg(client, msgs, **kw):
            captured["msgs"] = msgs
            return {"content": "Acknowledged. Ceasing."}

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg):
            reply = await hub._call_response_llm(s)
        self.assertEqual(reply, "Acknowledged. Ceasing.")
        sys_content = captured["msgs"][0]["content"]
        self.assertIn("INTERNAL NOTE", sys_content)
        self.assertIn("pasta", sys_content)
        self.assertIn("no record", sys_content)  # the exact lie she told, now forbidden

    async def test_response_prompt_ignores_expired_note(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.nag_note = hub._nag_note("pasta")
        s.nag_note_until = time.monotonic() - 1  # expired
        captured = {}

        async def fake_msg(client, msgs, **kw):
            captured["msgs"] = msgs
            return {"content": "ok"}

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg):
            await hub._call_response_llm(s)
        self.assertNotIn("INTERNAL NOTE", captured["msgs"][0]["content"])

    async def test_gate_prompt_includes_note(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        self._armed(s)
        captured = {}

        async def fake_call(client, msgs, **kw):
            captured["msgs"] = msgs
            return "YES"

        with mock.patch.object(llm_mod, "call_llama", fake_call):
            ok = await hub._tiny_llm_says_respond(s, "stop")
        self.assertTrue(ok)
        prompt = captured["msgs"][0]["content"]
        self.assertIn("pasta", prompt)
        self.assertIn("YES", prompt)  # the note tells the gate the answer

    async def test_gate_prompt_ignores_expired_note(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.nag_note = hub._nag_note("pasta")
        s.nag_note_until = time.monotonic() - 1
        captured = {}

        async def fake_call(client, msgs, **kw):
            captured["msgs"] = msgs
            return "NO"

        with mock.patch.object(llm_mod, "call_llama", fake_call):
            ok = await hub._tiny_llm_says_respond(s, "stop")
        self.assertFalse(ok)
        self.assertNotIn("INTERNAL NOTE", captured["msgs"][0]["content"])


class TestLightCommand(unittest.TestCase):
    """decision.light_command(): the conservative detector behind the hub's
    deterministic light backstop. A false positive flips a real light nobody
    asked about — so it needs verb + light noun + state word, all three."""

    def test_clean_command(self):
        from glados_hub import decision
        self.assertEqual(
            decision.light_command("turn off the livingroom lights"),
            ("livingroom", "off"))

    def test_split_room_name(self):
        from glados_hub import decision
        self.assertEqual(
            decision.light_command("turn off the living room lights"),
            ("living room", "off"))

    def test_wake_word_stripped(self):
        from glados_hub import decision
        self.assertEqual(
            decision.light_command("glados turn on the ceiling light"),
            ("ceiling", "on"))

    def test_no_room_named(self):
        from glados_hub import decision
        self.assertEqual(decision.light_command("switch off the lights"), ("", "off"))

    def test_flickering_is_not_a_command(self):
        from glados_hub import decision
        self.assertIsNone(decision.light_command("the lights are flickering"))

    def test_heat_is_not_a_light(self):
        from glados_hub import decision
        self.assertIsNone(decision.light_command("turn up the heat"))

    def test_question_is_not_a_command(self):
        from glados_hub import decision
        self.assertIsNone(decision.light_command("is the light on"))

    def test_tv_is_not_a_light(self):
        from glados_hub import decision
        self.assertIsNone(decision.light_command("turn off the tv"))

    def test_pronoun_only_is_not_a_command(self):
        from glados_hub import decision
        self.assertIsNone(decision.light_command("turn it on"))

    def test_empty(self):
        from glados_hub import decision
        self.assertIsNone(decision.light_command(""))


class TestResolveWhitespace(unittest.TestCase):
    """ASR splits 'livingroom' into 'living room'; the resolver must still
    hit the GROUP, not mis-route it to the ceiling plug via substring."""

    def setUp(self):
        self.reg = make_registry()

    def test_split_room_name_resolves_group(self):
        self.assertEqual(self.reg.resolve("living room"), ALL_THREE)

    def test_split_room_name_with_noise(self):
        self.assertEqual(self.reg.resolve("the living room lights"), ALL_THREE)

    def test_joined_name_still_works(self):
        self.assertEqual(self.reg.resolve("livingroom"), ALL_THREE)


@unittest.skipIf(hub is None, "fastapi not installed in test env")
class TestLightBackstop(unittest.IsolatedAsyncioTestCase):
    """The hub must never let a light answer outrun the hardware:

    * light command + model skipped the tool  -> hub runs set_lights itself,
      then re-asks the LLM to reply from the verified result;
    * a device failed this turn               -> the final LLM call carries a
      note forcing an honest failure report;
    * anything else                            -> exactly one LLM call, no
      extra hardware traffic.
    """

    def setUp(self):
        kasa_lights.set_registry(make_registry())
        from glados_hub.session import Session
        self._session_cls = Session

    def tearDown(self):
        kasa_lights.set_registry(kasa_lights.LightRegistry.load())

    def _session(self):
        return self._session_cls("dev1", None, "sys")

    async def test_skipped_tool_hub_runs_itself(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.add_message("user", "turn off the livingroom lights")
        calls, kws, set_calls = [], [], []

        async def fake_msg(client, msgs, **kw):
            calls.append(msgs)
            kws.append(kw)
            if len(calls) == 1:
                # The lie: words, no tool call.
                return {"content": "The lights are off, of course."}
            return {"content": "The lights are off. I verified it."}

        async def fake_set_lights(target, state, brightness=None):
            set_calls.append((target, state))
            return "livingroom ceiling: off; livingroom left: off; livingroom right: off"

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg), \
             mock.patch.object(kasa_lights, "set_lights", fake_set_lights), \
             mock.patch.object(hub, "GEMINI_API_KEY", ""):
            reply = await hub._call_response_llm(s, "turn off the livingroom lights")

        self.assertEqual(reply, "The lights are off. I verified it.")
        self.assertEqual(set_calls, [("livingroom", "off")])  # hub acted
        self.assertEqual(len(calls), 2)  # grounded re-call after the backstop
        note = calls[1][-1]["content"]
        self.assertIn("INTERNAL NOTE", note)
        self.assertIn("livingroom ceiling: off", note)  # the verified result
        self.assertIn("never claim", note)

    async def test_failed_device_forces_honest_reply(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.add_message("user", "turn on the livingroom lights")
        calls, kws = [], []

        async def fake_msg(client, msgs, **kw):
            calls.append(msgs)
            kws.append(kw)
            if len(calls) == 1:
                return {"content": None, "tool_calls": [{
                    "id": "tc1", "type": "function",
                    "function": {"name": "set_lights",
                                 "arguments": '{"target": "livingroom", "state": "on"}'},
                }]}
            if len(calls) == 2:
                # First content pass — would claim everything is fine.
                return {"content": "All three are on. You're welcome."}
            return {"content": "Two are on. The left one refused again — I tried twice, it's still off."}

        async def fake_set_lights(target, state, brightness=None):
            return ("livingroom ceiling: on; livingroom left: on FAILED after 2 "
                    "attempts, still off; livingroom right: on")

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg), \
             mock.patch.object(kasa_lights, "set_lights", fake_set_lights), \
             mock.patch.object(hub, "GEMINI_API_KEY", ""):
            reply = await hub._call_response_llm(s, "turn on the livingroom lights")

        self.assertIn("refused", reply)
        # call 1: tool call -> call 2: content -> call 3: grounded re-call
        self.assertEqual(len(calls), 3)
        note = calls[2][-1]["content"]
        self.assertIn("FAILED", note)
        self.assertIn("retried", note)
        self.assertIn("Do NOT claim", note)
        # tools stay on in loop rounds (the model may re-read state)
        self.assertIsNotNone(kws[1].get("tools"))


    async def test_non_light_turn_untouched(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.add_message("user", "what time is it")
        calls = []

        async def fake_msg(client, msgs, **kw):
            calls.append(msgs)
            return {"content": "Time is a social construct."}

        async def fake_set_lights(target, state, brightness=None):
            self.fail("kasa must not be touched for non-light turns")

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg), \
             mock.patch.object(kasa_lights, "set_lights", fake_set_lights), \
             mock.patch.object(hub, "GEMINI_API_KEY", ""):
            reply = await hub._call_response_llm(s, "what time is it")
        self.assertEqual(reply, "Time is a social construct.")
        self.assertEqual(len(calls), 1)  # no extra grounded re-call

    async def test_unresolvable_target_does_not_act(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.add_message("user", "turn off the hallway lights")
        calls = []

        async def fake_msg(client, msgs, **kw):
            calls.append(msgs)
            return {"content": "I have no hallway, test subject."}

        async def fake_set_lights(target, state, brightness=None):
            self.fail("unresolvable target must not touch hardware")

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg), \
             mock.patch.object(kasa_lights, "set_lights", fake_set_lights), \
             mock.patch.object(hub, "GEMINI_API_KEY", ""):
            await hub._call_response_llm(s, "turn off the hallway lights")
        self.assertEqual(len(calls), 1)  # no backstop, no re-call

    async def test_no_room_named_falls_back_to_sole_group(self):
        from glados_hub import llm as llm_mod
        s = self._session()
        s.add_message("user", "turn off the lights")
        calls, set_calls = [], []

        async def fake_msg(client, msgs, **kw):
            calls.append(msgs)
            if len(calls) == 1:
                return {"content": "Off. Obviously."}
            return {"content": "The livingroom lights are off, verified."}

        async def fake_set_lights(target, state, brightness=None):
            set_calls.append((target, state))
            return "livingroom ceiling: off; livingroom left: off; livingroom right: off"

        with mock.patch.object(llm_mod, "call_llama_message", fake_msg), \
             mock.patch.object(kasa_lights, "set_lights", fake_set_lights), \
             mock.patch.object(hub, "GEMINI_API_KEY", ""):
            await hub._call_response_llm(s, "turn off the lights")
        self.assertEqual(set_calls, [("livingroom", "off")])  # sole group
        self.assertEqual(len(calls), 2)

    async def test_execute_light_command_unresolvable_returns_none(self):
        self.assertIsNone(await hub._execute_light_command("hallway", "off", "dev1"))

    async def test_execute_light_command_runs_tool(self):
        set_calls = []

        async def fake_set_lights(target, state, brightness=None):
            set_calls.append((target, state))
            return "livingroom ceiling: off"

        with mock.patch.object(kasa_lights, "set_lights", fake_set_lights):
            out = await hub._execute_light_command("ceiling", "off", "dev1")
        self.assertEqual(set_calls, [("ceiling", "off")])
        self.assertIn("off", out)


class TestDetectionTracker(unittest.TestCase):
    """The pure state machine behind check_camera + auto-announcements.

    The NVR re-lists parked cars ~10x/second, so only SUSTAINED moving
    detections (2 frames within the window) may trigger anything, each
    class announces at most once per cooldown, and reports stay honest
    about feed health.
    """

    def setUp(self):
        self.t = 0.0
        self.tracker = cameras.DetectionTracker(
            "65", announce_classes=("person", "package"),
            cooldown_s=90.0, now_fn=lambda: self.t)

    def det(self, cls, moving):
        return {"className": cls, "movement": ({"moving": moving} if moving else {})}

    def advance(self, dt):
        self.t += dt

    def test_single_frame_does_not_announce(self):
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])
        self.assertFalse(self.tracker._classes["person"]["active"])

    def test_two_moving_frames_announce_once(self):
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])
        self.advance(0.5)
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), ["person"])
        self.assertTrue(self.tracker._classes["person"]["active"])
        self.advance(0.5)
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])

    def test_parked_vehicles_never_announce(self):
        for _ in range(20):
            self.advance(0.5)
            self.assertEqual(self.tracker.on_frame([self.det("vehicle", False)]), [])
        self.assertEqual(self.tracker.on_frame([self.det("person", False)]), [])

    def test_non_announced_classes_ignored(self):
        for _ in range(5):
            self.advance(0.3)
        self.assertEqual(
            self.tracker.on_frame([self.det("vehicle", True), self.det("face", True)]),
            [])

    def test_package_announces(self):
        self.assertEqual(self.tracker.on_frame([self.det("package", True)]), [])
        self.advance(0.4)
        self.assertEqual(self.tracker.on_frame([self.det("package", True)]), ["package"])

    def test_cooldown_suppresses_then_allows_refire(self):
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])
        self.advance(0.5)
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), ["person"])
        # person stops; decay runs on the next frame (even an empty one)
        self.advance(30)
        self.assertEqual(self.tracker.on_frame([]), [])
        self.assertFalse(self.tracker._classes["person"]["active"])
        # returns WITHIN the cooldown: activation, but no second announcement
        self.advance(0.5)
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])
        # after the cooldown expires, re-activation announces again
        self.advance(90)
        self.assertEqual(self.tracker.on_frame([]), [])
        self.advance(0.5)
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), [])
        self.assertEqual(self.tracker.on_frame([self.det("person", True)]), ["person"])

    def test_last_frame_age(self):
        self.assertIsNone(self.tracker.last_frame_age())  # never seen a frame
        self.tracker.on_frame([])
        self.assertEqual(self.tracker.last_frame_age(), 0.0)
        self.advance(31)
        self.assertGreater(self.tracker.last_frame_age(), 30)

    def test_describe_feed_ok(self):
        self.tracker.on_frame([self.det("person", True)])
        self.advance(0.5)
        self.tracker.on_frame([self.det("person", True), self.det("package", False)])
        out = self.tracker.describe(connected=True)
        self.assertIn("feed OK", out)
        self.assertIn("person: ACTIVE", out)
        self.assertIn("package: last seen", out)

    def test_describe_unavailable(self):
        self.tracker.on_frame([])
        out = self.tracker.describe(connected=False)
        self.assertIn("FEED UNAVAILABLE", out)
        self.assertIn("do not guess", out)

    def test_describe_fresh_connect_reports_no_activity_yet(self):
        out = self.tracker.describe(connected=True)  # connected, zero frames
        self.assertIn("no activity reported yet", out)
        self.assertNotIn("FEED UNAVAILABLE", out)

    def test_describe_quiet_is_all_clear_not_an_error(self):
        """Silence with a live connection = 'no one detected', NOT 'can't see':
        the NVR only emits frames while tracking, so a quiet yard is silent."""
        self.tracker.on_frame([])
        self.advance(60)
        out = self.tracker.describe(connected=True)
        self.assertIn("feed OK", out)
        self.assertNotIn("FEED UNAVAILABLE", out)
        self.assertNotIn("may be off", out)

    def test_describe_long_silence_hedges_dead_camera(self):
        self.tracker.on_frame([])
        self.advance(2 * 3600)
        out = self.tracker.describe(connected=True)
        self.assertIn("silent for", out)
        self.assertIn("may be off", out)


class _FakeCameraMonitor:
    """Just enough surface for cameras.check_camera (no paho, no loop)."""

    def __init__(self, connected, cameras, trackers):
        self.connected = connected
        self.cameras = cameras
        self.trackers = trackers
        self.suppressed = []  # quiet-hours events (list; the real one is a deque)


class TestCheckCamera(unittest.TestCase):
    """check_camera must never let the LLM guess: every output states the
    feed's health, the display name comes through, and the tool dispatches."""

    def setUp(self):
        self._saved = cameras.MONITOR
        self.t = 100.0
        self.monitor = _FakeCameraMonitor(
            True,
            {"front": "65", "back": "62"},
            {"front": cameras.DetectionTracker("65", now_fn=lambda: self.t),
             "back": cameras.DetectionTracker("62", now_fn=lambda: self.t)},
        )
        self.monitor.trackers["front"].on_frame([])  # fresh
        cameras.MONITOR = self.monitor

    def tearDown(self):
        cameras.MONITOR = self._saved

    def test_unconfigured_is_honest(self):
        cameras.MONITOR = None
        out = cameras.check_camera("front")
        self.assertIn("cannot see", out)
        self.assertIn("do not guess", out)

    def test_unknown_camera(self):
        self.assertIn("unknown camera", cameras.check_camera("garage"))

    def test_feed_ok_report_names_front(self):
        out = cameras.check_camera("front")
        self.assertIn("front yard camera: feed OK", out)
        self.assertIn("person: none seen", out)
        self.assertIn("package: none seen", out)

    def test_unavailable_report(self):
        self.monitor.connected = False
        out = cameras.check_camera("front")
        self.assertIn("FEED UNAVAILABLE", out)
        self.assertIn("do not guess", out)

    def test_quiet_report_is_all_clear(self):
        self.t += 120  # two minutes without frames: quiet, not broken
        out = cameras.check_camera("front")
        self.assertIn("feed OK", out)
        self.assertNotIn("FEED UNAVAILABLE", out)
        self.assertNotIn("may be off", out)

    def test_long_silence_report_hedges(self):
        self.t += 2 * 3600
        self.assertIn("may be off", cameras.check_camera("front"))

    def test_configured_but_not_running_is_honest(self):
        # "back" is in the config but absent from the live monitor: must use
        # the display label and say plainly it can't see — never guess.
        del self.monitor.cameras["back"]
        del self.monitor.trackers["back"]
        out = cameras.check_camera("back")
        self.assertIn("back yard camera", out)
        self.assertIn("not running", out)
        self.assertIn("cannot see", out)
        self.assertIn("do not guess", out)

    def test_tool_dispatch_defaults_to_front(self):
        import asyncio
        out = asyncio.run(skills.execute_tool("check_camera", {}, "dev1"))
        self.assertIn("front yard camera", out)
        out = asyncio.run(skills.execute_tool("check_camera", {"camera": "front"}, "dev1"))
        self.assertIn("front yard camera", out)

    def test_tool_dispatch_never_raises(self):
        import asyncio
        cameras.MONITOR = None
        out = asyncio.run(skills.execute_tool("check_camera", {"camera": "back"}, "dev1"))
        self.assertIn("cannot see", out)


@unittest.skipIf(cameras.mqtt is None, "paho-mqtt not installed in test env")
class TestCameraConnect(unittest.TestCase):
    """_on_connect must subscribe (topic, qos) tuples and only claim
    'connected' after the subscribe actually succeeded — a subscribe that
    raises must leave connected=False (regression: paho 2.x rejects bare
    topic strings and the crash used to leave a stale 'connected' flag)."""

    def setUp(self):
        self.mon = cameras.CameraMonitor(
            "127.0.0.1", 1883, "u", "p",
            {"front": "65", "back": "62"})

    class _OK:
        is_failure = False

    class _Fail:
        is_failure = True

    def test_subscribes_tuples_and_connected(self):
        subs = []

        class FakeClient:
            def subscribe(self, topics, qos=0):
                subs.append(topics)

        self.mon._on_connect(FakeClient(), None, None, self._OK(), None)
        self.assertTrue(self.mon.connected)
        self.assertEqual(
            subs[0],
            [("scrypted/65/ObjectDetector", 0), ("scrypted/62/ObjectDetector", 0)])

    def test_subscribe_failure_leaves_not_connected(self):
        class BadClient:
            def subscribe(self, *a, **k):
                raise ValueError("too many values to unpack")

        self.mon._on_connect(BadClient(), None, None, self._OK(), None)
        self.assertFalse(self.mon.connected)
        self.assertIn("subscribe failed", self.mon.last_error)

    def test_rejected_connection_leaves_not_connected(self):
        subs = []

        class FakeClient:
            def subscribe(self, topics, qos=0):
                subs.append(topics)

        self.mon._on_connect(FakeClient(), None, None, self._Fail(), None)
        self.assertFalse(self.mon.connected)
        self.assertEqual(subs, [])


@unittest.skipIf(cameras.mqtt is None, "paho-mqtt not installed in test env")
class TestCameraAnnounceWiring(unittest.IsolatedAsyncioTestCase):
    """Front-yard activity is spoken to the whole house (speaker='' -> every
    active session); the back yard never auto-announces; parked cars do
    nothing. The consumer task is driven by a pre-seeded queue — no broker."""

    def setUp(self):
        self.mon = cameras.CameraMonitor(
            "127.0.0.1", 1883, "u", "p",
            {"front": "65", "back": "62"}, announce_enabled=("front",))
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

    async def test_front_person_announces_housewide(self):
        await self._feed(("front", [self.det("person", True)]),
                         ("front", [self.det("person", True)]))
        self.assertEqual(len(self.calls), 1)
        line, speaker = self.calls[0]
        self.assertIn("person", line)
        self.assertIn("front", line)
        self.assertEqual(speaker, "")  # -> _announce_to_devices skips no one

    async def test_front_package_announces(self):
        await self._feed(("front", [self.det("package", True)]),
                         ("front", [self.det("package", True)]))
        self.assertEqual(len(self.calls), 1)
        self.assertIn("package", self.calls[0][0])

    async def test_back_yard_never_auto_announces(self):
        await self._feed(("back", [self.det("person", True)]),
                         ("back", [self.det("person", True)]))
        self.assertEqual(self.calls, [])

    async def test_parked_vehicles_never_announce(self):
        await self._feed(*[("front", [self.det("vehicle", False)]) for _ in range(30)])
        self.assertEqual(self.calls, [])


class TestCameraCapabilities(unittest.TestCase):
    """The tool must be on the LLM's surface, and the capability block must
    appear only when the feed is actually configured."""

    def test_tool_in_surface(self):
        names = [t["function"]["name"] for t in skills.TOOLS]
        self.assertIn("check_camera", names)

    def test_capability_block_when_configured(self):
        with mock.patch.object(cameras, "MQTT_HOST", "192.168.1.177"), \
             mock.patch.object(cameras, "MQTT_PASS", "x"), \
             mock.patch.object(cameras, "CONFIG", cameras.CameraConfig({
                 "front": {"label": "front yard", "id": "65", "announce": True,
                           "classes": ["person", "package"], "zones": [],
                           "cooldown_s": 90.0, "lines": {}}})):
            out = cameras.describe_capabilities()
            self.assertIn("check_camera", out)
            self.assertIn("front yard", out)
        with mock.patch.object(cameras, "MQTT_HOST", ""), \
             mock.patch.object(cameras, "MQTT_PASS", "x"):
            self.assertEqual(cameras.describe_capabilities(), "")

    def test_tool_camera_enum_tracks_config(self):
        tool = next(t for t in skills.TOOLS if t["function"]["name"] == "check_camera")
        enum = tool["function"]["parameters"]["properties"]["camera"]["enum"]
        self.assertEqual(set(enum), set(cameras.CONFIG.cameras))


class TestCameraConfigDrivesPolicy(unittest.TestCase):
    """Camera inventory + per-camera policy (announce, classes, zones,
    cooldown, labels, lines) must come from config/cameras.json — not code.
    Editing the JSON (and rebuilding) re-points the whole pipeline: the
    tool enum, the monitor's tracker policy, and the announce gating."""

    def _write_cfg(self, tmp, cameras_dict):
        p = Path(tmp) / "cameras.json"
        p.write_text(json.dumps({"cameras": cameras_dict}))
        return p

    def test_load_reads_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_cfg(tmp, {
                "front": {"label": "front yard", "id": "65", "announce": True,
                          "classes": ["Person", "package"], "zones": ["Porch"],
                          "cooldown_s": 30, "lines": {"person": "Hey."}},
                "garage": {"id": "99"},  # exercises the defaults
            })
            cfg = cameras.CameraConfig.load(p)
        self.assertEqual(set(cfg.cameras), {"front", "garage"})
        self.assertEqual(cfg.cameras["front"]["id"], "65")
        self.assertTrue(cfg.cameras["front"]["announce"])
        self.assertEqual(cfg.cameras["front"]["classes"], ["person", "package"])
        self.assertEqual(cfg.cameras["front"]["zones"], ["Porch"])
        self.assertEqual(cfg.cameras["front"]["cooldown_s"], 30.0)
        self.assertEqual(cfg.cameras["front"]["lines"], {"person": "Hey."})
        # defaults: label=key, announce off, no zone whitelist, default classes
        self.assertEqual(cfg.cameras["garage"]["label"], "garage")
        self.assertFalse(cfg.cameras["garage"]["announce"])
        self.assertEqual(cfg.cameras["garage"]["zones"], [])
        self.assertEqual(cfg.display("front"), "front yard")
        self.assertEqual(cfg.display("garage"), "garage")
        self.assertEqual(cfg.display("nope"), "nope")

    def test_load_skips_entries_without_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_cfg(tmp, {"ok": {"id": "1"}, "bad": {}})
            cfg = cameras.CameraConfig.load(p)
        self.assertEqual(set(cfg.cameras), {"ok"})

    def test_load_unreadable_disables_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cameras.CameraConfig.load(Path(tmp) / "missing.json")
        self.assertEqual(cfg.cameras, {})

    def test_zone_whitelist_filters_detections(self):
        t = cameras.DetectionTracker(
            "65", announce_classes=("person", "package"), cooldown_s=90.0,
            zones=("Porch",), now_fn=lambda: 0.0)
        # a moving person with NO matching zone tag is ignored...
        self.assertEqual(
            t.on_frame([{"className": "person",
                         "movement": {"moving": True}}]), [])
        # ...a moving person tagged 'Porch' counts (2 sustained frames -> fire)
        d = {"className": "person", "movement": {"moving": True},
             "zones": ["Porch"]}
        self.assertEqual(t.on_frame([d]), [])
        self.assertEqual(t.on_frame([d]), ["person"])

    def test_zone_whitelist_empty_accepts_all(self):
        t = cameras.DetectionTracker(
            "65", announce_classes=("person",), cooldown_s=90.0, zones=(),
            now_fn=lambda: 0.0)
        d = {"className": "person", "movement": {"moving": True}}
        self.assertEqual(t.on_frame([d]), [])
        self.assertEqual(t.on_frame([d]), ["person"])

    @unittest.skipIf(cameras.mqtt is None, "paho-mqtt not installed in test env")
    def test_monitor_applies_per_camera_settings(self):
        mon = cameras.CameraMonitor(
            "127.0.0.1", 1883, "u", "p",
            {"front": "65", "back": "62"},
            announce_enabled=("front",),
            settings={
                "front": {"label": "front yard", "id": "65", "announce": True,
                          "classes": ["person"], "zones": [], "cooldown_s": 45.0,
                          "lines": {"person": "Custom line."}},
                "back": {"label": "back yard", "id": "62", "announce": False,
                         "classes": ["package"], "zones": ["Porch"],
                         "cooldown_s": 120.0, "lines": {}},
            })
        # per-camera tracker policy is applied from the config entries
        self.assertEqual(mon.trackers["front"].announce_classes, ("person",))
        self.assertEqual(mon.trackers["front"].cooldown_s, 45.0)
        self.assertEqual(mon.trackers["front"].zones, ())
        self.assertEqual(mon.trackers["back"].announce_classes, ("package",))
        self.assertEqual(mon.trackers["back"].cooldown_s, 120.0)
        self.assertEqual(mon.trackers["back"].zones, ("Porch",))
        # announce gating + per-camera line override
        self.assertEqual(mon.announce_enabled, {"front"})
        self.assertEqual(mon.lines["front"]["person"], "Custom line.")
        self.assertNotIn("package", mon.lines.get("back", {}))


@unittest.skipUnless(hub is not None, "hub import failed")
class TestVolumeTools(unittest.TestCase):
    """set_volume/get_volume: per-device state, clamping, the 0x16 frame,
    and the tools must stay honest when the hub isn't wired up."""

    def setUp(self):
        self._saved = dict(hub.VOLUME)
        self._saved_fn = skills._volume_fn
        self.sent = []
        sent = self.sent  # closure: the fake must append to the TEST's list

        class _FakeWs:
            async def send_bytes(self, data):
                sent.append(data)

        self.ws = _FakeWs()

    def tearDown(self):
        hub.VOLUME.clear()
        hub.VOLUME.update(self._saved)
        skills._volume_fn = self._saved_fn

    def test_apply_volume_absolute_and_clamp(self):
        self.assertEqual(hub.apply_volume("d1", level=80), 80)
        self.assertEqual(hub.get_volume("d1"), 80)
        self.assertEqual(hub.apply_volume("d1", level=150), 100)
        self.assertEqual(hub.apply_volume("d1", level=-5), 0)
        # per-device isolation
        self.assertEqual(hub.get_volume("d2"), 100)

    def test_apply_volume_relative(self):
        hub.apply_volume("d1", level=70)
        self.assertEqual(hub.apply_volume("d1", change=-10), 60)
        self.assertEqual(hub.apply_volume("d1", change=500), 100)
        self.assertEqual(hub.apply_volume("d1", change=-500), 0)

    def test_apply_volume_requires_argument(self):
        with self.assertRaises(ValueError):
            hub.apply_volume("d1")

    def test_send_volume_frame_is_0x16_int16le(self):
        asyncio.run(hub._send_volume(self.ws, 65))
        self.assertEqual(self.sent, [b"\x16" + (65).to_bytes(2, "little")])

    def test_volume_tool_sets_state_and_sends_frame(self):
        class _S:
            ws = self.ws
            glados_speaking = False

        hub.active_sessions["d1"] = _S()
        try:
            out = asyncio.run(hub._volume_tool("d1", level=42))
            self.assertIn("42%", out)
            self.assertEqual(self.sent, [b"\x16" + (42).to_bytes(2, "little")])
            out = asyncio.run(hub._volume_tool("d1", change=-100))
            self.assertIn("muted", out)
            self.assertEqual(hub.get_volume("d1"), 0)
            out = asyncio.run(hub._volume_tool("d1", report=True))
            self.assertIn("0%", out)
        finally:
            hub.active_sessions.pop("d1", None)

    def test_skills_dispatch_set_and_get(self):
        calls = []

        async def fake(device_id, level=None, change=None, report=False):
            calls.append((device_id, level, change, report))
            return "fake"

        skills.set_volume_fn(fake)
        out = asyncio.run(skills.execute_tool("set_volume", {"level": 50}, "dev1"))
        self.assertEqual(out, "fake")
        self.assertEqual(calls[-1], ("dev1", 50, None, False))
        asyncio.run(skills.execute_tool("set_volume", {"change": "-10"}, "dev1"))
        self.assertEqual(calls[-1], ("dev1", None, -10, False))
        asyncio.run(skills.execute_tool("get_volume", {}, "dev1"))
        self.assertEqual(calls[-1], ("dev1", None, None, True))
        # set_volume with neither arg degrades to a report, not an error
        asyncio.run(skills.execute_tool("set_volume", {}, "dev1"))
        self.assertEqual(calls[-1], ("dev1", None, None, True))
        # bad args
        out = asyncio.run(skills.execute_tool("set_volume", {"level": "abc"}, "dev1"))
        self.assertIn("must be integers", out)

    def test_skills_dispatch_unwired_is_honest(self):
        skills.set_volume_fn(None)
        out = asyncio.run(skills.execute_tool("set_volume", {"level": 50}, "dev1"))
        self.assertIn("unavailable", out)
        out = asyncio.run(skills.execute_tool("get_volume", {}, "dev1"))
        self.assertIn("unavailable", out)

    def test_volume_tools_on_surface_and_prompt(self):
        names = [t["function"]["name"] for t in skills.TOOLS]
        self.assertIn("set_volume", names)
        self.assertIn("get_volume", names)
        self.assertIn("set_volume", skills.describe_capabilities())


if __name__ == "__main__":
    unittest.main()
