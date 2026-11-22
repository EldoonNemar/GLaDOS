"""
Tests for the LAFVIN HAT display on the python spoke:
  * lafvin  - ST7789 wire protocol (LAFVIN init sequence, window math),
              RGB LED active-low inversion, graceful "no HAT" degradation
  * face    - RGB565 packing, state machine (idle/listening/thinking/talking),
              idle ball physics bounds + determinism
  * main    - AudioMeter levels + hold
Pure logic only; no GPIO/SPI hardware. Spoke main.py is loaded the same way
test_playback.py does (importlib, so `import main` can't collide).
"""
import ctypes
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spokes" / "python-spoke"))

import lafvin  # noqa: E402
from face import Face, IDLE, LISTENING, TALKING, to_rgb565  # noqa: E402
from lafvin import (HEIGHT, ROW_OFFSET, WIDTH, INIT_SEQUENCE,  # noqa: E402
                    LafvinHat, RgbLed, St7789, rgb565)

try:
    _spec = importlib.util.spec_from_file_location(
        "spoke_main_display", REPO / "spokes" / "python-spoke" / "main.py")
    spoke = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(spoke)
    _SPOKE = True
except Exception:  # pragma: no cover
    _SPOKE = False


# ── fakes ────────────────────────────────────────────────────────────

class FakeSpi:
    """Records every wire operation in order."""

    def __init__(self):
        self.events = []  # ("cmd", byte) | ("data", bytes)

    def command(self, value):
        self.events.append(("cmd", value))

    def write(self, data):
        self.events.append(("data", bytes(data)))


class FakeLine:
    def __init__(self):
        self.values = []

    def set_value(self, v):
        self.values.append(v)

    def read(self):
        return 0


class FakeGpio:
    def __init__(self):
        self.lines = {}

    def output(self, pin, label):
        line = FakeLine()
        self.lines[pin] = line
        return line

    def input(self, pin, label):
        line = FakeLine()
        self.lines[pin] = line
        return line

    def cleanup(self):
        self.lines.clear()


def _make_st7789():
    spi = FakeSpi()
    dc, rst = FakeLine(), FakeLine()
    d = St7789(spi, dc, rst, sleep=lambda s: None)
    return d, spi, dc, rst


# ── lafvin: color packing ────────────────────────────────────────────

class TestRgb565(unittest.TestCase):
    def test_primary_colors(self):
        self.assertEqual(rgb565(255, 0, 0), 0xF800)
        self.assertEqual(rgb565(0, 255, 0), 0x07E0)
        self.assertEqual(rgb565(0, 0, 255), 0x001F)
        self.assertEqual(rgb565(255, 255, 255), 0xFFFF)
        self.assertEqual(rgb565(0, 0, 0), 0x0000)

    def test_face_packer_agrees_with_lafvin_pack(self):
        rng = np.random.default_rng(0)
        frame = rng.uniform(0.0, 1.0, size=(3, 4, 3)).astype(np.float32)
        got = np.frombuffer(to_rgb565(frame), dtype=">u2")
        px = (np.clip(frame, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
        want = np.array(
            [rgb565(int(r), int(g), int(b)) for r, g, b in px], dtype="<u2")
        self.assertTrue((got == want).all())

    def test_wire_bytes_are_big_endian(self):
        # Regression: little-endian words byte-swap on the SPI bus and the
        # panel renders the face as a blue/purple channel-scramble.
        px = np.zeros((1, 1, 3), dtype=np.float32)
        px[0, 0] = (1.0, 0.0, 0.0)  # pure red -> RGB565 0xF800
        self.assertEqual(to_rgb565(px), b"\xF8\x00")
        px[0, 0] = (0.0, 0.0, 1.0)  # pure blue -> 0x001F
        self.assertEqual(to_rgb565(px), b"\x00\x1F")


# ── lafvin: ST7789 wire protocol ─────────────────────────────────────

class TestSt7789(unittest.TestCase):
    def test_initialize_sends_lafvin_sequence(self):
        d, spi, dc, rst = _make_st7789()
        sleeps = []
        St7789(spi, dc, rst, sleep=sleeps.append).initialize()
        # reset pulse: high, low, high (LAFVIN timing)
        self.assertEqual(rst.values, [1, 0, 1])
        # exact LAFVIN command order: Sleep Out first, Display On last
        cmds = [v for k, v in spi.events if k == "cmd"]
        self.assertEqual(cmds, [c for c, _ in INIT_SEQUENCE])
        self.assertEqual(cmds[0], 0x11)
        self.assertEqual(cmds[-1], 0x29)
        # panel-specific bits made it onto the wire (MADCTL 0xC0, 16bpp 0x05)
        data = [v for k, v in spi.events if k == "data"]
        self.assertIn(b"\xC0", data)
        self.assertIn(b"\x05", data)
        # sleep-out settle delay was honored
        self.assertIn(lafvin.SLEEP_OUT_DELAY, sleeps)
        # commands go out with DC low, data with DC high
        self.assertIn(0, dc.values)
        self.assertIn(1, dc.values)

    def test_present_window_math(self):
        d, spi, dc, rst = _make_st7789()
        x, y, w, h = 10, 20, 100, 50
        d.present(bytes(w * h * 2), x=x, y=y, width=w, height=h)
        cmds = [v for k, v in spi.events if k == "cmd"]
        self.assertEqual(cmds, [0x2A, 0x2B, 0x2C])
        data = [v for k, v in spi.events if k == "data"]
        # x: 10..109 (16-bit start/end); y: row offset applied
        self.assertEqual(data[0], bytes([0, 10, 0, 109]))
        y0 = y + ROW_OFFSET
        self.assertEqual(data[1], bytes([0, y0, 0, y0 + h - 1]))
        self.assertEqual(len(data[2]), w * h * 2)  # full payload, one write

    def test_present_rejects_bad_payload(self):
        d, spi, dc, rst = _make_st7789()
        with self.assertRaises(ValueError):
            d.present(bytes(10), x=0, y=0, width=10, height=10)  # wrong size
        with self.assertRaises(ValueError):
            d.present(bytes(20 * 20 * 2), x=WIDTH - 5, y=0,  # out of bounds
                      width=20, height=20)

    def test_fill_sends_full_frame(self):
        d, spi, dc, rst = _make_st7789()
        d.fill(0x1234)
        data = [v for k, v in spi.events if k == "data"]
        self.assertEqual(len(data[-1]), WIDTH * HEIGHT * 2)


# ── lafvin: RGB LED ──────────────────────────────────────────────────

class TestRgbLed(unittest.TestCase):
    def test_active_low_inversion(self):
        led = RgbLed(FakeGpio(), stop_event=threading.Event())
        try:
            led.set(255, 0, 0)
            # active-low: full red = red line driven fully LOW
            self.assertEqual(led._pwms["r"]._duty, 0.0)
            self.assertEqual(led._pwms["g"]._duty, 100.0)
            self.assertEqual(led._pwms["b"]._duty, 100.0)
            led.set(0, 255, 255)
            self.assertEqual(led._pwms["r"]._duty, 100.0)
            self.assertEqual(led._pwms["g"]._duty, 0.0)
        finally:
            led.close()

    def test_set_clamps(self):
        led = RgbLed(FakeGpio(), stop_event=threading.Event())
        try:
            led.set(300, -5, 128)
            self.assertEqual(led._rgb, (255, 0, 128))
        finally:
            led.close()


# ── lafvin: graceful degradation ─────────────────────────────────────

class TestLafvinHatOpen(unittest.TestCase):
    def test_try_open_none_when_spidev_unavailable(self):
        with mock.patch.dict(sys.modules, {"spidev": None}):
            self.assertIsNone(LafvinHat.try_open())

    def test_try_open_none_when_spi_fails(self):
        spi_mod = types.ModuleType("spidev")

        class Boom:
            def open(self, *a):
                raise OSError("no spi device")

            def close(self):
                pass

        spi_mod.SpiDev = Boom
        with mock.patch.dict(sys.modules, {"spidev": spi_mod}):
            self.assertIsNone(LafvinHat.try_open())

    def test_open_wires_backlight_thread_with_stop_event(self):
        # Regression: the backlight PWM thread was started without the
        # stop_event that _SoftPwm.run requires — it died at startup and
        # the panel stayed black even though frames were being pushed.
        spi_mod = types.ModuleType("spidev")

        class FakeSpiDev:
            max_speed_hz = 0
            mode = 0

            def open(self, bus, cs):
                pass

            def close(self):
                pass

            def xfer2(self, data):
                return data

            def writebytes2(self, data):
                return len(data)

        spi_mod.SpiDev = FakeSpiDev
        made = []
        real_thread = threading.Thread

        class SpyThread(real_thread):
            def __init__(self, *a, **k):
                made.append(k)
                super().__init__(*a, **k)

        hat = LafvinHat()
        with mock.patch.dict(sys.modules, {"spidev": spi_mod}), \
             mock.patch.object(lafvin, "_GpiodGpio", lambda: FakeGpio()), \
             mock.patch.object(lafvin.threading, "Thread", SpyThread):
            hat._open(spi_mod)
        try:
            bl = [k for k in made if k.get("name") == "lafvin-backlight"]
            self.assertEqual(len(bl), 1)
            self.assertEqual(bl[0].get("args"), (hat._stop_event,))
        finally:
            hat.close()


# ── lafvin: gpiod chardev GPIO layer ──────────────────────────────────

class FakeGpioChip:
    """Simulates /dev/gpiochip0 for the gpiod chardev uAPI (ABI v2):
    GET_LINE hands out a line fd, values ioctls run on that fd with a
    bits/mask bitmap."""

    CHIP_FD = 3

    def __init__(self, num_lines: int = 58):
        self.lines: dict[int, dict] = {}
        self.num_lines = num_lines
        self._next_fd = 100
        self.closed: list[int] = []

    def open(self, *a, **k):
        return self.CHIP_FD

    def close(self, fd):
        self.closed.append(int(fd))
        # closing a line fd releases the line (kernel behavior)
        for pin, st in list(self.lines.items()):
            if st["fd"] == fd:
                del self.lines[pin]
                break

    def _pin_for_fd(self, fd):
        for pin, st in self.lines.items():
            if st["fd"] == fd:
                return pin
        return None

    def ioctl(self, fd, req, buf):
        if req == lafvin.GPIO_GET_CHIP_INFO:
            buf.name = b"gpiochip0"
            buf.label = b"fake"
            buf.lines = self.num_lines
            return 0
        if req == lafvin.GPIO_V2_GET_LINE:
            pin = int(buf.offsets[0])
            flags = int(buf.flags)
            if pin >= self.num_lines:
                raise OSError("invalid line offset")
            if pin in self.lines:
                raise OSError("line busy")
            self._next_fd += 1
            self.lines[pin] = {
                "out": bool(flags & lafvin.FLAG_OUTPUT),
                "bias_disabled": bool(flags & lafvin.FLAG_BIAS_DISABLED),
                "value": 0,
                "fd": self._next_fd,
            }
            buf.fd = self._next_fd
            return 0
        pin = self._pin_for_fd(fd)
        if pin is None or pin not in self.lines:
            raise OSError("bad line fd")
        if req == lafvin.GPIO_V2_LINE_SET_VALUES:
            self.lines[pin]["value"] = int(buf.bits) & (int(buf.mask) or 1)
            return 0
        if req == lafvin.GPIO_V2_LINE_GET_VALUES:
            buf.bits = self.lines[pin]["value"] & int(buf.mask)
            return 0
        raise AssertionError(f"unexpected ioctl {req:#x}")


class TestGpioAbi(unittest.TestCase):
    def test_struct_sizes_match_kernel_abi(self):
        # sizeof per /usr/include/linux/gpio.h on the target RPi kernel:
        #   request = 64*4 + 32 + 8 + 4 + 20 + 240 + 4 + 4 + 20 + 4 = 592
        self.assertEqual(ctypes.sizeof(lafvin._GpioV2LineRequest), 592)
        self.assertEqual(ctypes.sizeof(lafvin._GpioV2LineValues), 16)
        self.assertEqual(ctypes.sizeof(lafvin._GpiochipInfo), 68)

    def test_ioctl_words_match_kernel_abi(self):
        # _IOC(dir, type, nr, size) = dir<<30 | size<<16 | type<<8 | nr
        self.assertEqual(lafvin.GPIO_GET_CHIP_INFO,
                         (2 << 30) | (68 << 16) | (0xB4 << 8) | 0x01)
        self.assertEqual(lafvin.GPIO_V2_GET_LINE,
                         (3 << 30) | (592 << 16) | (0xB4 << 8) | 0x07)
        self.assertEqual(lafvin.GPIO_V2_LINE_GET_VALUES,
                         (3 << 30) | (16 << 16) | (0xB4 << 8) | 0x0E)
        self.assertEqual(lafvin.GPIO_V2_LINE_SET_VALUES,
                         (3 << 30) | (16 << 16) | (0xB4 << 8) | 0x0F)


class TestGpiodGpio(unittest.TestCase):
    def setUp(self):
        self.chip = FakeGpioChip()
        self.g = lafvin._GpiodGpio(open_fn=self.chip.open,
                                   ioctl_fn=self.chip.ioctl,
                                   close_fn=self.chip.close)

    def test_output_request_and_set_value(self):
        line = self.g.output(lafvin.LCD_DC_PIN, "dc")
        st = self.chip.lines[lafvin.LCD_DC_PIN]
        self.assertTrue(st["out"])
        self.assertFalse(st["bias_disabled"])
        self.assertEqual(st["value"], 0)
        line.set_value(1)
        self.assertEqual(st["value"], 1)
        line.set_value(0)
        self.assertEqual(st["value"], 0)

    def test_input_requests_bias_disabled_and_reads(self):
        line = self.g.input(lafvin.BUTTON_PIN, "button")
        st = self.chip.lines[lafvin.BUTTON_PIN]
        self.assertFalse(st["out"])
        self.assertTrue(st["bias_disabled"])  # board pulls the button down
        self.assertEqual(line.read(), 0)
        st["value"] = 1  # simulate a press
        self.assertEqual(line.read(), 1)

    def test_input_falls_back_without_bias_flag(self):
        # Kernel rejects BIAS_DISABLED (some gpiod/kernel combos) — plain
        # INPUT must still work, like the vendor driver's fallback.
        def reject_bias(fd, req, buf):
            if req == lafvin.GPIO_V2_GET_LINE and (
                    int(buf.flags) & lafvin.FLAG_BIAS_DISABLED):
                raise OSError("unsupported bias")
            return self.chip.ioctl(fd, req, buf)
        g = lafvin._GpiodGpio(open_fn=self.chip.open, ioctl_fn=reject_bias,
                              close_fn=self.chip.close)
        line = g.input(lafvin.BUTTON_PIN, "button")
        st = self.chip.lines[lafvin.BUTTON_PIN]
        self.assertFalse(st["bias_disabled"])
        self.assertEqual(line.read(), 0)

    def test_request_failure_raises(self):
        def boom(fd, req, buf):
            raise OSError("ENODEV: no gpiochip")
        with self.assertRaises(OSError):
            lafvin._GpiodGpio(open_fn=self.chip.open, ioctl_fn=boom,
                              close_fn=self.chip.close).output(lafvin.BUTTON_PIN)

    def test_off_chip_pin_rejected(self):
        with self.assertRaises(OSError):
            self.g.output(lafvin.BUTTON_PIN + self.chip.num_lines, "way out")

    def test_cleanup_closes_line_fds_and_chip(self):
        self.g.output(lafvin.BACKLIGHT_PIN, "bl")
        self.g.output(lafvin.LCD_DC_PIN, "dc")
        self.g.input(lafvin.BUTTON_PIN, "button")
        fds = [st["fd"] for st in self.chip.lines.values()]
        self.g.cleanup()
        self.assertEqual(self.chip.lines, {})
        self.assertEqual(self.g._lines, {})
        for fd in fds:
            self.assertIn(fd, self.chip.closed)
        self.assertIn(FakeGpioChip.CHIP_FD, self.chip.closed)


# ── face: rendering ──────────────────────────────────────────────────

class TestFace(unittest.TestCase):
    def test_frame_size_and_rgb565(self):
        f = Face(seed=7)
        frame = f.tick(0.04)
        self.assertEqual(len(frame), WIDTH * HEIGHT * 2)
        self.assertEqual(np.frombuffer(frame, dtype=">u2").dtype, np.dtype(">u2"))

    def test_idle_eye_centered_and_visible(self):
        # the eye is always on: centered and completely yellow, even idle
        f = Face(seed=3)
        for _ in range(50):
            f.tick(0.05)
        words = np.frombuffer(f.tick(0.05), dtype=">u2").reshape(
            HEIGHT, WIDTH).astype(np.uint32)
        center = words[HEIGHT // 2, WIDTH // 2]
        r = (center >> 11 & 0x1F) * 255 // 31
        b = (center & 0x1F) * 255 // 31
        self.assertGreater(r, 150)  # solid yellow at the exact center
        self.assertGreater(r, b)
        # corners are still near-black background
        self.assertLess(words[2, 2], 0x300)
        self.assertLess(words[HEIGHT - 3, WIDTH - 3], 0x300)

    def test_idle_frame_is_yellow(self):
        f = Face(seed=5)
        for _ in range(10):
            f.tick(0.05)
        words = np.frombuffer(f.tick(0.05), dtype=">u2").astype(np.uint32)
        r = (words >> 11 & 0x1F) * 255 // 31
        b = (words & 0x1F) * 255 // 31
        i = int(np.argmax(words))  # brightest pixel = ball core
        self.assertGreater(r[i], 150)
        self.assertGreater(r[i], b[i])  # yellow, not blue

    def test_seed_determinism(self):
        a, b = Face(seed=42), Face(seed=42)
        for _ in range(5):
            self.assertEqual(a.tick(0.05), b.tick(0.05))


# ── face: state machine ──────────────────────────────────────────────

class TestFaceStates(unittest.TestCase):
    def test_talk_think_idle(self):
        f = Face(seed=1)
        self.assertEqual(f.state, IDLE)
        self.assertEqual(f.update(0.1, 0.0, 0.5), TALKING)
        self.assertEqual(f.update(0.1, 0.4, 0.0), TALKING)  # TTS window holds
        for _ in range(10):  # 1.0 s silence: past TALK window, still thinking
            f.update(0.1, 0.0, 0.0)
        self.assertEqual(f.state, "thinking")
        for _ in range(50):  # 5 s more: past THINK window
            f.update(0.1, 0.0, 0.0)
        self.assertEqual(f.state, IDLE)

    def test_listening(self):
        f = Face(seed=1)
        self.assertEqual(f.update(0.1, 0.4, 0.0), LISTENING)

    def test_talking_wins_over_mic(self):
        f = Face(seed=1)
        self.assertEqual(f.update(0.1, 0.4, 0.4), TALKING)

    def test_quiet_signals_below_gates_stay_idle(self):
        f = Face(seed=1)
        for _ in range(5):
            f.update(0.1, 0.01, 0.01)  # noise floor, under both gates
        self.assertEqual(f.state, IDLE)

    def test_equalizer_ring_only_while_talking(self):
        # listening pulses the eye but draws no ring; the segmented
        # equalizer ring is a talking-only effect
        def zone_max(frame: bytes) -> int:
            words = np.frombuffer(frame, dtype=">u2").reshape(
                HEIGHT, WIDTH).astype(np.uint32)
            yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
            d = np.sqrt((xx - WIDTH / 2) ** 2 + (yy - HEIGHT / 2) ** 2)
            # annulus just outside the listening eye's max reach (r ~70)
            # and inside the panel half-width (120); the talking ring
            # lives at r0 ~79..113 at the test's levels
            return int(words[(d > 76) & (d < 115)].max())

        # threshold: the background vignette peaks ~0x8xx in this band,
        # the amber ring words are ~0xFCxx — 0x4000 splits them cleanly
        f = Face(seed=1)
        for _ in range(30):
            f.tick(0.1, 0.9, 0.0)  # sustained listening
        self.assertEqual(f.state, LISTENING)
        self.assertLess(zone_max(f.tick(0.1, 0.9, 0.0)), 0x4000)  # no ring

        g = Face(seed=1)
        for _ in range(30):
            g.tick(0.1, 0.0, 0.9)  # sustained talking
        self.assertEqual(g.state, TALKING)
        self.assertGreater(zone_max(g.tick(0.1, 0.0, 0.9)), 0x4000)  # ring


# ── main: AudioMeter ─────────────────────────────────────────────────

@unittest.skipUnless(_SPOKE, "spoke module not importable in this environment")
class TestAudioMeter(unittest.TestCase):
    def _meter(self, t0=0.0):
        clock = [t0]
        m = spoke.AudioMeter(now_fn=lambda: clock[0])
        return m, clock

    def test_mic_level_normalized_and_held(self):
        m, clock = self._meter()
        m.set_mic(3000)
        self.assertAlmostEqual(m.mic_level(), 0.5)  # 3000 / MIC_NORM 6000
        clock[0] = 0.2  # inside the 0.25 s hold
        self.assertAlmostEqual(m.mic_level(), 0.5)
        clock[0] = 0.3  # past the hold
        self.assertEqual(m.mic_level(), 0.0)

    def test_tts_level_normalized_and_clamped(self):
        m, clock = self._meter()
        m.set_tts(4000)
        self.assertAlmostEqual(m.tts_level(), 0.5)  # 4000 / TTS_NORM 8000
        m.set_tts(999999)
        self.assertEqual(m.tts_level(), 1.0)

    def test_module_meter_wired(self):
        self.assertIsInstance(spoke.meter, spoke.AudioMeter)

    def test_display_task_exits_cleanly_without_hat(self):
        import asyncio
        spoke.DISPLAY_ENABLED = "off"
        asyncio.run(spoke.display_task(asyncio.Event()))  # returns, no raise


if __name__ == "__main__":
    unittest.main()

