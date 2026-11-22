"""
Unit tests for the live config server: validated read/write over the
config files (atomic, doc-preserving), the hot-reload seams
(kasa_lights.reload, cameras.reload, skills.refresh_tool_enums,
hub.rebuild_system_prompt), and the REST endpoints (TestClient with
fake volume API + on-changed callback, including token auth).

Pure logic + fakes: no Kasa hardware, no MQTT broker, no LLM.
"""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub import cameras  # noqa: E402
from glados_hub import config_server  # noqa: E402
from glados_hub import kasa_lights  # noqa: E402
from glados_hub import quiet  # noqa: E402
from glados_hub import skills  # noqa: E402

try:
    from glados_hub import hub  # noqa: E402
except ImportError:
    hub = None


KASA = {
    "lights": {"kitchen": "192.168.1.99", "garage": "192.168.1.10"},
    "types": {"kitchen": "bulb", "garage": "plug"},
    "groups": {"downstairs": ["kitchen", "garage"]},
}
CAMS = {
    "_doc": "keep me",
    "cameras": {
        "front": {
            "label": "front yard", "id": "65", "announce": True,
            "classes": ["person", "package"], "zones": ["Porch"],
            "cooldown_s": 90,
        },
    },
}


class _Paths:
    """Point the config_server + leaf modules at temp files (restore after)."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.kasa = self.root / "kasa.json"
        self.cams = self.root / "cameras.json"
        self.prompt = self.root / "prompt.txt"
        self.ui = self.root / "ui.html"
        self.quiet = self.root / "quiet_hours.json"
        self._patches = [
            mock.patch.object(config_server, "KASA_PATH", self.kasa),
            mock.patch.object(config_server, "CAMERAS_PATH", self.cams),
            mock.patch.object(config_server, "PROMPT_PATH", self.prompt),
            mock.patch.object(config_server, "UI_PATH", self.ui),
            mock.patch.object(config_server, "QUIET_PATH", self.quiet),
            mock.patch.object(kasa_lights, "CONFIG_PATH", self.kasa),
            mock.patch.object(cameras, "CONFIG_PATH", self.cams),
            mock.patch.object(quiet, "CONFIG_PATH", self.quiet),
            # No broker in tests: cameras.reload() re-reads config but
            # keeps the monitor OFF (is_configured() is False).
            mock.patch.object(cameras, "MQTT_HOST", ""),
            mock.patch.object(cameras, "MQTT_PORT", 1),
            mock.patch.object(cameras, "MQTT_USER", ""),
            mock.patch.object(cameras, "MQTT_PASS", ""),
        ]
        for m in self._patches:
            m.start()
        self._saved_reg = kasa_lights.REGISTRY
        self._saved_cfg = cameras.CONFIG
        self._saved_mon = cameras.MONITOR
        self._saved_quiet = quiet.CONFIG
        self._saved_on_changed = config_server._on_changed
        self._saved_vol = config_server._volume_api
        self.changed = []
        config_server.set_on_changed(lambda: self.changed.append(1))

    def close(self):
        for m in reversed(self._patches):
            m.stop()
        kasa_lights.REGISTRY = self._saved_reg
        cameras.CONFIG = self._saved_cfg
        if cameras.MONITOR is not None and cameras.MONITOR is not self._saved_mon:
            try:
                cameras.MONITOR.stop()
            except Exception:
                pass
        cameras.MONITOR = self._saved_mon
        quiet.CONFIG = self._saved_quiet
        config_server.set_on_changed(self._saved_on_changed)
        config_server.set_volume_api(self._saved_vol)
        self.tmp.cleanup()


class TestLightsFileIO(unittest.TestCase):
    def setUp(self):
        self.p = _Paths()
        self.p.kasa.write_text(json.dumps(KASA))

    def tearDown(self):
        self.p.close()

    def test_read_missing_file_is_empty(self):
        self.p.kasa.unlink()
        self.assertEqual(config_server.read_lights(), {})

    def test_valid_roundtrip(self):
        new = {"lights": {"kitchen": "192.168.1.50"}, "types": {"kitchen": "plug"},
               "groups": {}}
        out = config_server.write_lights(new)
        self.assertEqual(out, new)
        self.assertEqual(json.loads(self.p.kasa.read_text())["lights"]["kitchen"],
                         "192.168.1.50")

    def test_empty_ip_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_lights(
                {"lights": {"x": "  "}, "types": {}, "groups": {}})

    def test_bad_type_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_lights(
                {"lights": {"x": "1.2.3.4"}, "types": {"x": "candle"}, "groups": {}})

    def test_group_member_must_exist(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_lights(
                {"lights": {"x": "1.2.3.4"}, "types": {},
                 "groups": {"g": ["ghost"]}})

    def test_bad_json_shape_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_lights([1, 2, 3])

    def test_write_never_corrupts_on_reject(self):
        before = self.p.kasa.read_text()
        with self.assertRaises(config_server.ConfigError):
            config_server.write_lights(
                {"lights": {"x": ""}, "types": {}, "groups": {}})
        self.assertEqual(self.p.kasa.read_text(), before)

    def test_doc_keys_preserved_across_edit(self):
        self.p.kasa.write_text(json.dumps({"_doc": "mine", **KASA}))
        config_server.write_lights({"lights": KASA["lights"], "types": {}, "groups": {}})
        on_disk = json.loads(self.p.kasa.read_text())
        self.assertEqual(on_disk["_doc"], "mine")
        self.assertEqual(on_disk["types"], {})

    def test_atomic_write_leaves_no_temp_files(self):
        config_server.write_lights(KASA)
        leftovers = [f for f in self.p.root.iterdir() if f.suffix == ".tmp"]
        self.assertEqual(leftovers, [])



class TestCamerasFileIO(unittest.TestCase):
    def setUp(self):
        self.p = _Paths()
        self.p.cams.write_text(json.dumps(CAMS))

    def tearDown(self):
        self.p.close()

    def test_valid_roundtrip(self):
        new = {"cameras": {"front": CAMS["cameras"]["front"],
                           "back": {"id": "62", "announce": False}}}
        out = config_server.write_cameras(new)
        self.assertEqual(out["cameras"]["back"], {"id": "62", "announce": False})
        on_disk = json.loads(self.p.cams.read_text())
        self.assertIn("back", on_disk["cameras"])

    def test_missing_id_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_cameras({"cameras": {"x": {}}})

    def test_bad_classes_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_cameras(
                {"cameras": {"x": {"id": "1", "classes": "person"}}})

    def test_bad_cooldown_rejected(self):
        for cd in ("-5", "90", True, -1):
            with self.assertRaises(config_server.ConfigError, msg=repr(cd)):
                config_server.write_cameras(
                    {"cameras": {"x": {"id": "1", "cooldown_s": cd}}})

    def test_non_bool_announce_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_cameras(
                {"cameras": {"x": {"id": "1", "announce": "yes"}}})

    def test_bad_lines_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_cameras(
                {"cameras": {"x": {"id": "1", "lines": {"person": 5}}}})

    def test_doc_preserved_even_if_client_drops_it(self):
        # The UI sends only {"cameras": ...}; the _doc must survive.
        config_server.write_cameras({"cameras": CAMS["cameras"]})
        on_disk = json.loads(self.p.cams.read_text())
        self.assertEqual(on_disk["_doc"], "keep me")


class TestPromptFileIO(unittest.TestCase):
    def setUp(self):
        self.p = _Paths()
        self.p.prompt.write_text("original personality")

    def tearDown(self):
        self.p.close()

    def test_roundtrip(self):
        config_server.write_prompt("new personality")
        self.assertEqual(config_server.read_prompt(), "new personality")

    def test_empty_rejected(self):
        for text in ("", "   \n  "):
            with self.assertRaises(config_server.ConfigError):
                config_server.write_prompt(text)

    def test_too_long_rejected(self):
        with self.assertRaises(config_server.ConfigError):
            config_server.write_prompt("x" * (config_server.PROMPT_MAX_CHARS + 1))


class TestKasaReload(unittest.TestCase):
    def setUp(self):
        self._saved = kasa_lights.REGISTRY
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Path(self.tmp.name) / "kasa.json"
        self.f.write_text(json.dumps({"lights": {"a": "1.1.1.1"}, "groups": {}}))
        self._patch = mock.patch.object(kasa_lights, "CONFIG_PATH", self.f)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        kasa_lights.REGISTRY = self._saved
        self.tmp.cleanup()

    def test_reload_swaps_live_registry(self):
        reg = kasa_lights.reload()
        self.assertIs(reg, kasa_lights.REGISTRY)
        self.assertIn("a", reg.lights)
        self.f.write_text(json.dumps(
            {"lights": {"b": "2.2.2.2"}, "groups": {"g": ["b"]}}))
        reg2 = kasa_lights.reload()
        self.assertNotIn("a", reg2.lights)
        self.assertEqual(reg2.groups["g"], ["b"])


class TestRefreshToolEnums(unittest.TestCase):
    def setUp(self):
        self._saved_cfg = cameras.CONFIG
        self._saved_enum = None
        for t in skills.TOOLS:
            if t["function"]["name"] == "check_camera":
                self._saved_enum = list(
                    t["function"]["parameters"]["properties"]["camera"]["enum"])
        cameras.CONFIG = cameras.CameraConfig({
            "alpha": {"label": "a", "id": "1", "announce": False,
                      "classes": ["person"], "zones": [], "cooldown_s": 90,
                      "lines": {}},
            "beta": {"label": "b", "id": "2", "announce": False,
                     "classes": ["person"], "zones": [], "cooldown_s": 90,
                     "lines": {}},
        })

    def tearDown(self):
        cameras.CONFIG = self._saved_cfg
        for t in skills.TOOLS:
            if t["function"]["name"] == "check_camera":
                t["function"]["parameters"]["properties"]["camera"]["enum"] = \
                    self._saved_enum

    def test_check_camera_enum_tracks_config(self):
        skills.refresh_tool_enums()
        for t in skills.TOOLS:
            if t["function"]["name"] == "check_camera":
                self.assertEqual(
                    t["function"]["parameters"]["properties"]["camera"]["enum"],
                    ["alpha", "beta"])




@unittest.skipIf(cameras.mqtt is None, "paho-mqtt not installed in test env")
class TestCamerasReload(unittest.TestCase):
    """cameras.reload() must re-read the file and re-apply the LIVE monitor:
    re-subscribe (new camera set), carry over the announce_fn, and stop
    cleanly when the config is emptied out."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Path(self.tmp.name) / "cameras.json"
        self.f.write_text(json.dumps({"cameras": {"front": {"id": "65", "announce": True}}}))
        self._patches = [
            mock.patch.object(cameras, "CONFIG_PATH", self.f),
            mock.patch.object(cameras, "MQTT_HOST", "127.0.0.1"),  # nothing listens
            mock.patch.object(cameras, "MQTT_PORT", 1),
            mock.patch.object(cameras, "MQTT_USER", ""),
            mock.patch.object(cameras, "MQTT_PASS", "x"),
        ]
        for m in self._patches:
            m.start()
        self._saved_cfg = cameras.CONFIG
        self._saved_mon = cameras.MONITOR

    def tearDown(self):
        for m in reversed(self._patches):
            m.stop()
        cameras.CONFIG = self._saved_cfg
        if cameras.MONITOR is not None and cameras.MONITOR is not self._saved_mon:
            try:
                cameras.MONITOR.stop()
            except Exception:
                pass
        cameras.MONITOR = self._saved_mon
        self.tmp.cleanup()

    def test_reload_reapplies_monitor_and_carries_announce_fn(self):
        async def announce(msg, speaker):
            return "ok"

        async def run():
            # First monitor via start(): built from the CURRENT CONFIG —
            # point it at our fixture first.
            cameras.CONFIG = cameras.CameraConfig.load(self.f)
            self.assertTrue(cameras.start(announce))
            first = cameras.MONITOR
            self.assertEqual(first.cameras, {"front": "65"})

            # Add a second camera on disk; reload must re-subscribe for it
            # and keep the same announce_fn.
            self.f.write_text(json.dumps({"cameras": {
                "front": {"id": "65", "announce": True},
                "back": {"id": "62", "announce": False},
            }}))
            self.assertTrue(cameras.reload())
            second = cameras.MONITOR
            self.assertIsNot(second, first)
            self.assertEqual(second.cameras, {"front": "65", "back": "62"})
            self.assertIs(second.announce_fn, announce)
            self.assertIn("back", second.trackers)

            # Empty the config: monitor must stop, not keep the old set.
            self.f.write_text(json.dumps({"cameras": {}}))
            self.assertFalse(cameras.reload())
            self.assertIsNone(cameras.MONITOR)

        asyncio.run(run())


@unittest.skipUnless(hub is not None, "hub module not importable in this environment")
class TestRebuildSystemPrompt(unittest.TestCase):
    def setUp(self):
        self._saved_prompt = hub.SYSTEM_PROMPT
        self._saved_sessions = hub.active_sessions

    def tearDown(self):
        hub.active_sessions = self._saved_sessions
        hub.SYSTEM_PROMPT = hub._build_system_prompt()

    def test_rebuild_rereads_prompt_file_and_updates_sessions(self):
        s = hub.Session("dev", None, "stale prompt")
        hub.active_sessions["dev"] = s
        self.assertEqual(s.messages[0]["content"], "stale prompt")
        with tempfile.TemporaryDirectory() as td:
            pp = Path(td) / "glados_system.txt"
            pp.write_text("FRESH PERSONALITY BASE")
            with mock.patch.object(hub, "PROMPT_PATH", pp):
                hub.rebuild_system_prompt()
        self.assertTrue(hub.SYSTEM_PROMPT.startswith("FRESH PERSONALITY BASE"))
        # In-flight sessions get the new system message too.
        self.assertEqual(s.messages[0]["content"], hub.SYSTEM_PROMPT)


class _FakeVolumeApi:
    def __init__(self):
        self.store = {"d1": 80}

    def list(self):
        return dict(self.store)

    async def set(self, device_id, level):
        self.store[device_id] = max(0, min(100, level))
        return self.store[device_id]


@unittest.skipUnless(hub is not None, "fastapi TestClient needs the hub deps")
class TestConfigEndpoints(unittest.TestCase):
    """The REST surface end to end (TestClient): validation errors, file
    writes, hot-reload side effects, doc preservation, volume bridge,
    and optional token auth."""

    def setUp(self):
        self.p = _Paths()
        self.p.kasa.write_text(json.dumps(KASA))
        self.p.cams.write_text(json.dumps(CAMS))
        self.p.prompt.write_text("base prompt")
        self.p.ui.write_text("<html>TEST UI</html>")
        self.p.quiet.write_text(json.dumps({
            "_doc": "keep me",
            "timezone": "America/Chicago",
            "windows": [{"name": "night", "start": "22:00", "end": "07:00"}],
        }))
        self.vol = _FakeVolumeApi()
        config_server.set_volume_api(self.vol)
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(config_server.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.p.close()

    # ── UI + reads ─────────────────────────────────────────────────
    def test_ui_served_at_root(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("TEST UI", res.text)

    def test_get_lights(self):
        res = self.client.get("/api/config/lights")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["lights"], KASA["lights"])

    def test_status_shape(self):
        res = self.client.get("/api/config/status")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        # CONFIG here is whatever loaded at import (the temp file only
        # becomes live after a PUT), so assert shape, not exact contents.
        self.assertEqual(body["cameras_configured"],
                         sorted(cameras.CONFIG.cameras))
        self.assertEqual(body["lights"], kasa_lights.REGISTRY.describe())
        self.assertIn("cameras", body)

    # ── lights writes ──────────────────────────────────────────────
    def test_put_lights_writes_and_hot_reloads(self):
        new = {"lights": {"kitchen": "192.168.1.50"}, "types": {"kitchen": "plug"},
               "groups": {}}
        res = self.client.put("/api/config/lights", json=new)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(json.loads(self.p.kasa.read_text())["lights"], new["lights"])
        # Hot reload: the LIVE registry reflects the change, and the
        # on-changed hook (prompt rebuild) fired.
        self.assertIn("kitchen", kasa_lights.REGISTRY.lights)
        self.assertEqual(kasa_lights.REGISTRY.lights["kitchen"], "192.168.1.50")
        self.assertTrue(self.p.changed)

    def test_put_lights_rejects_invalid_without_writing(self):
        before = self.p.kasa.read_text()
        res = self.client.put("/api/config/lights", json={
            "lights": {"x": "1.2.3.4"}, "types": {}, "groups": {"g": ["ghost"]}})
        self.assertEqual(res.status_code, 400)
        self.assertIn("ghost", res.json()["detail"])
        self.assertEqual(self.p.kasa.read_text(), before)
        self.assertFalse(self.p.changed)

    # ── cameras writes ─────────────────────────────────────────────
    def test_put_cameras_writes_preserving_doc(self):
        res = self.client.put("/api/config/cameras", json={
            "cameras": {"front": CAMS["cameras"]["front"],
                        "back": {"id": "62", "announce": False}}})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["monitor_running"])  # no broker in tests
        on_disk = json.loads(self.p.cams.read_text())
        self.assertIn("back", on_disk["cameras"])
        self.assertEqual(on_disk["_doc"], "keep me")
        # CONFIG hot-reloaded too
        self.assertIn("back", cameras.CONFIG.cameras)
        self.assertTrue(self.p.changed)

    def test_put_cameras_rejects_missing_id(self):
        before = self.p.cams.read_text()
        res = self.client.put("/api/config/cameras", json={"cameras": {"x": {}}})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.p.cams.read_text(), before)

    # ── prompt writes ──────────────────────────────────────────────
    def test_put_prompt(self):
        res = self.client.put("/api/config/prompt", json={"prompt": "new base"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.p.prompt.read_text(), "new base")
        self.assertTrue(self.p.changed)

    def test_put_prompt_rejects_empty(self):
        res = self.client.put("/api/config/prompt", json={"prompt": "  "})
        self.assertEqual(res.status_code, 400)

    # ── quiet hours ────────────────────────────────────────────────
    def test_get_quiet_includes_status(self):
        res = self.client.get("/api/config/quiet")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["timezone"], "America/Chicago")
        self.assertEqual(body["windows"][0]["name"], "night")
        self.assertIn("status", body)
        self.assertIn("quiet", body["status"])

    def test_put_quiet_writes_reloads_preserves_doc(self):
        res = self.client.put("/api/config/quiet", json={
            "timezone": "America/Chicago",
            "windows": [{"name": "night", "start": "22:00", "end": "07:00"},
                        {"name": "lunch", "start": "12:00", "end": "13:00",
                         "sources": ["timers"]}],
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        on_disk = json.loads(self.p.quiet.read_text())
        self.assertEqual(len(on_disk["windows"]), 2)
        self.assertEqual(on_disk["_doc"], "keep me")       # doc preserved
        # Hot reload: the live config reflects the change, hook fired.
        self.assertEqual([w.name for w in quiet.CONFIG.windows], ["night", "lunch"])
        self.assertTrue(self.p.changed)

    def test_put_quiet_rejects_bad_timezone(self):
        before = self.p.quiet.read_text()
        res = self.client.put("/api/config/quiet",
                              json={"timezone": "Mars/Olympus", "windows": []})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.p.quiet.read_text(), before)
        self.assertFalse(self.p.changed)

    def test_put_quiet_rejects_start_equals_end(self):
        res = self.client.put("/api/config/quiet", json={
            "timezone": "UTC",
            "windows": [{"name": "x", "start": "07:00", "end": "07:00"}],
        })
        self.assertEqual(res.status_code, 400)

    def test_status_includes_quiet(self):
        res = self.client.get("/api/config/status")
        self.assertEqual(res.status_code, 200)
        self.assertIn("quiet", res.json())

    # ── volume ─────────────────────────────────────────────────────
    def test_volume_get_and_set(self):
        self.assertEqual(self.client.get("/api/config/volume").json(), {"d1": 80})
        res = self.client.put("/api/config/volume",
                              json={"device_id": "d1", "level": 40})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["level"], 40)
        res = self.client.put("/api/config/volume", json={"device_id": ""})
        self.assertEqual(res.status_code, 400)
        res = self.client.put("/api/config/volume",
                              json={"device_id": "d1", "level": "loud"})
        self.assertEqual(res.status_code, 400)

    # ── token auth ─────────────────────────────────────────────────
    def test_token_required_when_set(self):
        with mock.patch.object(config_server, "TOKEN", "secret"):
            self.assertEqual(self.client.get("/").status_code, 401)
            self.assertEqual(self.client.get("/?token=secret").status_code, 200)
            self.assertEqual(self.client.get("/?token=wrong").status_code, 401)
            self.assertEqual(
                self.client.put("/api/config/prompt", json={"prompt": "x"},
                                headers={"X-Config-Token": "secret"}).status_code,
                200)
            self.assertEqual(
                self.client.put("/api/config/prompt", json={"prompt": "x"}).status_code,
                401)


if __name__ == "__main__":
    unittest.main()