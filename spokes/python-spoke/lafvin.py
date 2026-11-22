"""
LAFVIN Audio Display HAT (LB011) hardware layer for the python spoke.

Hardware (from LAFVIN's accepted driver, lfvintech/LAFVIN-Audio-Display-HAT):
  Display   1.69" IPS 240x280 ST7789, 4-wire SPI bus 0 CS 0 @ 100 MHz,
            row offset 20; DC = GPIO27 (board pin 13), RST = GPIO4 (pin 7)
  RGB LED   GPIO 25/24/23 (R/G/B), ACTIVE-LOW, 100 Hz software PWM
  Button    GPIO17 (pin 11), polled at 5 ms
  Backlight GPIO22 (pin 15)

The ST7789 init sequence below is byte-for-byte the sequence LAFVIN ships
(verified against their St7789Display) — do not "clean it up"; the panel
timing/gamma values are panel-specific.

spidev is imported lazily and GPIO goes through the gpiod chardev uAPI
(/dev/gpiochip0, raw ioctls — the same path gpiod and RPi.GPIO use; the
/sys/class/gpio interface was removed from upstream Linux in 6.13 and is
dead on modern RPi kernels, and a --privileged container sees the host's
/dev/gpiochip0), so this module imports and its classes unit-test with
fakes on machines without the HAT.
"""
import ctypes
import fcntl
import logging
import os
import threading
import time

logger = logging.getLogger("glados_spoke.lafvin")

WIDTH = 240
HEIGHT = 280
ROW_OFFSET = 20  # panel vertical gap (ST7789 240x280 1.69" layout)

# Board pin -> BCM GPIO (40-pin header)
LCD_DC_PIN = 27     # board pin 13
LCD_RST_PIN = 4     # board pin 7
BACKLIGHT_PIN = 22  # board pin 15
RGB_R_PIN = 25      # board pin 22
RGB_G_PIN = 24      # board pin 18
RGB_B_PIN = 23      # board pin 16
BUTTON_PIN = 17     # board pin 11 (input)

SPI_BUS = 0
SPI_CS = 0
SPI_SPEED_HZ = 100_000_000
PWM_FREQ = 100  # Hz, software PWM for LED/backlight (matches LAFVIN)

# ST7789 init — LAFVIN's accepted sequence (MADCTL 0xC0, 16-bit pixels,
# panel-specific porch/gamma). Kept as a flat (cmd, [args...]) table so
# tests can compare the exact wire bytes.
INIT_SEQUENCE = [
    (0x11, []),                                            # Sleep Out
    (0x36, [0xC0]),                                        # MADCTL (MX)
    (0x3A, [0x05]),                                        # 16-bit/pixel
    (0xB2, [0x0C, 0x0C, 0x00, 0x33, 0x33]),                # Porch
    (0xB7, [0x35]),                                        # VCOM
    (0xBB, [0x32]),                                        # VRF
    (0xC2, [0x01]),
    (0xC3, [0x15]),
    (0xC4, [0x20]),
    (0xC6, [0x0F]),
    (0xD0, [0xA4, 0xA1]),
    (0xE0, [0xD0, 0x08, 0x0E, 0x09, 0x09, 0x05, 0x31, 0x33,
           0x48, 0x17, 0x14, 0x15, 0x31, 0x34]),           # Gamma+
    (0xE1, [0xD0, 0x08, 0x0E, 0x09, 0x09, 0x15, 0x31, 0x33,
           0x48, 0x17, 0x14, 0x15, 0x31, 0x34]),           # Gamma-
    (0x21, []),                                            # Display Inversion On
    (0x29, []),                                            # Display On
]
SLEEP_OUT_DELAY = 0.12
RESET_TIMING = (0.10, 0.10, 0.12)  # high, low, high


def rgb565(r: int, g: int, b: int) -> int:
    """Pack 8-bit RGB into a 16-bit 5-6-5 word."""
    return (((r >> 3) & 0x1F) << 11) | (((g >> 2) & 0x3F) << 5) | ((b >> 3) & 0x1F)


class St7789:
    """SPI transport + command protocol. `spi` needs .command(byte) and
    .write(bytes); `dc`/`rst` need .set_value(int)."""

    def __init__(self, spi, dc, rst, sleep=time.sleep):
        self._spi = spi
        self._dc = dc
        self._rst = rst
        self._sleep = sleep

    def initialize(self):
        hi, lo, settle = RESET_TIMING
        self._rst.set_value(1)
        self._sleep(hi)
        self._rst.set_value(0)
        self._sleep(lo)
        self._rst.set_value(1)
        self._sleep(settle)
        for cmd, args in INIT_SEQUENCE:
            self._command(cmd, *args)
            if cmd == 0x11:
                self._sleep(SLEEP_OUT_DELAY)

    def present(self, rgb565_bytes: bytes, x=0, y=0, width=WIDTH, height=HEIGHT):
        if len(rgb565_bytes) != width * height * 2:
            raise ValueError(
                f"RGB565 payload {len(rgb565_bytes)} bytes != {width}x{height}x2")
        if x < 0 or y < 0 or x + width > WIDTH or y + height > HEIGHT:
            raise ValueError("window exceeds display bounds")
        self._command(0x2A, x >> 8, x & 0xFF, (x + width - 1) >> 8, (x + width - 1) & 0xFF)
        y0, y1 = y + ROW_OFFSET, y + height - 1 + ROW_OFFSET
        self._command(0x2B, y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF)
        self._command(0x2C)
        self._dc.set_value(1)
        self._spi.write(rgb565_bytes)

    def fill(self, color: int):
        pixel = bytes(((color >> 8) & 0xFF, color & 0xFF))
        self.present(pixel * (WIDTH * HEIGHT))

    def _command(self, cmd: int, *args):
        self._dc.set_value(0)
        self._spi.command(cmd)
        if args:
            self._dc.set_value(1)
            self._spi.write(bytes(args))


class _SoftPwm:
    """100 Hz software PWM on a GPIO line (no hardware PWM on these pins)."""

    def __init__(self, line, freq: int = PWM_FREQ, stop_value: int = 1):
        self._line = line
        self._stop_value = stop_value
        self._period = 1.0 / freq
        self._duty = 0.0
        line.set_value(stop_value)

    def set_duty(self, percent: float):
        self._duty = max(0.0, min(100.0, percent))

    def run(self, stop_event: threading.Event):
        phase = 0.0
        on_until = 0.0
        line_on = False
        while not stop_event.is_set():
            now = time.monotonic()
            if phase >= 1.0:
                phase -= 1.0
                on_until = now + self._period * (self._duty / 100.0)
            want_on = now < on_until
            if want_on != line_on:
                line_on = want_on
                self._line.set_value(1 if line_on else self._stop_value)
            time.sleep(0.002)
        self._line.set_value(self._stop_value)


class RgbLed:
    """Active-LOW LED: brightness percent inverts the physical duty cycle."""

    def __init__(self, gpio, stop_event=None):
        self._lines = {
            "r": gpio.output(RGB_R_PIN, "glados-rgb-r"),
            "g": gpio.output(RGB_G_PIN, "glados-rgb-g"),
            "b": gpio.output(RGB_B_PIN, "glados-rgb-b"),
        }
        self._stop = stop_event or threading.Event()
        self._pwms = {k: _SoftPwm(line) for k, line in self._lines.items()}
        self._rgb = (0, 0, 0)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="lafvin-led-pwm")
        self._thread.start()

    def set(self, r: int, g: int, b: int):
        self._rgb = tuple(max(0, min(255, int(v))) for v in (r, g, b))
        for k, v in zip("rgb", self._rgb):
            # Low output is lit -> duty is inverted (LAFVIN convention)
            self._pwms[k].set_duty(100.0 - v / 255.0 * 100.0)

    def _run(self):
        for pwm in self._pwms.values():
            pwm.run(self._stop)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=0.5)


class LafvinHat:
    """Facade: display + LED + backlight + button. open() may fail when the
    HAT/driver isn't present — the spoke treats that as 'no display'."""

    def __init__(self, on_press=None):
        self.on_press = on_press
        self._gpio = None
        self._spi = None
        self.display = None
        self.led = None
        self._backlight_pwm = None
        self._button_line = None
        self._button_thread = None
        self._button_stop = threading.Event()
        self._stop_event = threading.Event()
        self._backlight = 100

    @classmethod
    def try_open(cls, on_press=None) -> "LafvinHat | None":
        try:
            import spidev
        except Exception as e:
            logger.warning("[LAFVIN] unavailable (%s) — display disabled", e)
            return None
        hat = cls(on_press=on_press)
        try:
            hat._open(spidev)
        except Exception as e:
            logger.warning("[LAFVIN] open failed (%s) — display disabled", e)
            hat.close()
            return None
        logger.info("[LAFVIN] HAT ready (ST7789 240x280 @ %d MHz)", SPI_SPEED_HZ)
        return hat

    def _open(self, spidev):
        self._gpio = _GpiodGpio()
        self._spi = _Spi(spidev)
        self._spi.open(SPI_BUS, SPI_CS, SPI_SPEED_HZ)
        self.display = St7789(
            self._spi,
            self._gpio.output(LCD_DC_PIN, "glados-lcd-dc"),
            self._gpio.output(LCD_RST_PIN, "glados-lcd-rst"),
        )
        self.display.initialize()
        self.display.fill(0)
        self.led = RgbLed(self._gpio, stop_event=self._stop_event)
        bl = self._gpio.output(BACKLIGHT_PIN, "glados-backlight")
        self._backlight_pwm = _SoftPwm(bl, stop_value=0)
        self._backlight_pwm.set_duty(float(self._backlight))
        threading.Thread(target=self._backlight_pwm.run, daemon=True,
                         name="lafvin-backlight").start()
        self._button_line = self._gpio.input(BUTTON_PIN, "glados-button")
        self._button_stop.clear()
        self._button_thread = threading.Thread(target=self._poll_button,
                                               daemon=True, name="lafvin-button")
        self._button_thread.start()

    def _poll_button(self):
        was_pressed = False
        while not self._button_stop.is_set():
            pressed = bool(self._button_line.read())
            if pressed and not was_pressed and self.on_press is not None:
                try:
                    self.on_press()
                except Exception:
                    logger.exception("[LAFVIN] button callback failed")
            was_pressed = pressed
            time.sleep(0.005)

    def present_frame(self, rgb565_bytes: bytes):
        self.display.present(rgb565_bytes)

    def set_led(self, r: int, g: int, b: int):
        if self.led:
            self.led.set(r, g, b)

    def set_backlight(self, percent: int):
        self._backlight = max(0, min(100, int(percent)))
        if self._backlight_pwm:
            self._backlight_pwm.set_duty(float(self._backlight))

    def close(self):
        self._stop_event.set()
        self._button_stop.set()
        if self._button_thread and self._button_thread.is_alive():
            self._button_thread.join(timeout=0.5)
        if self.led:
            self.led.close()
        if self._spi:
            self._spi.close()
        if self._gpio:
            self._gpio.cleanup()


# ── gpiod chardev GPIO ────────────────────────────────────────────────
#
# /sys/class/gpio is gone (removed from upstream Linux in 6.13; dead on
# the RPi 6.1x kernel). gpiod and RPi.GPIO both talk to the chardev uAPI
# on /dev/gpiochip<N> instead — we do the same with raw ioctls: no gpiod
# build dependency, no RPi.GPIO Pi-ness check, and a --privileged
# container sees the host's /dev/gpiochip0.

# struct gpio_line_set from uapi/linux/gpio.h (legacy ABI, still supported)
class _GpioLineSet(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("config", ctypes.c_uint32),
        ("values", ctypes.c_uint32 * 4),
    ]


LINEGETSET = 0xC0200130    # _IOWR('1', 0x00, _GpioLineSet) — request line
LINESETVALS = 0x40200132   # _IOW('1', 0x02, _GpioLineSet) — set values
LINEGETVALS = 0xC0200133   # _IOWR('1', 0x03, _GpioLineSet) — read values
LINEUNSET = 0x40200134     # _IOW('1', 0x04, _GpioLineSet) — release line
FLAG_DIN = 0x1
FLAG_DOUT = 0x2
FLAG_BIAS_DISABLE = 0x40


class _GpiodGpio:
    """GPIO lines via the gpiod chardev uAPI. `open_fn`/`ioctl_fn` are
    injectable so tests can run on machines without the HAT (production
    uses os.open + fcntl.ioctl on /dev/gpiochip0 — Pi 3/4 expose the
    40-pin header's BCM offsets as /dev/gpiochip0)."""

    def __init__(self, chip_path: str = "/dev/gpiochip0",
                 open_fn=None, ioctl_fn=None):
        self._open_fn = open_fn or os.open
        self._ioctl_fn = ioctl_fn or fcntl.ioctl
        self.fd = self._open_fn(chip_path, os.O_RDWR)
        self._pins: dict[int, "_GpiodLine"] = {}

    def output(self, pin: int, label: str = ""):
        line = self._request(pin, FLAG_DOUT, 0)
        self._pins[pin] = line
        return line

    def input(self, pin: int, label: str = ""):
        # The HAT pulls the button line down on the board; the SoC bias
        # must stay OFF (fighting the two biases makes the idle level
        # indeterminate). HIGH = pressed, LOW = released.
        try:
            line = self._request(pin, FLAG_DIN | FLAG_BIAS_DISABLE, 0)
        except OSError:
            # Some gpiod/kernel combinations don't support bias-disable;
            # retry plain (matches the vendor driver's fallback).
            line = self._request(pin, FLAG_DIN, 0)
        self._pins[pin] = line
        return line

    def _request(self, pin: int, flags: int, value: int):
        buf = _GpioLineSet(pin, 1, flags, (value & 1, 0, 0, 0))
        try:
            self._ioctl_fn(self.fd, LINEGETSET, buf)
        except OSError as e:
            raise OSError(f"gpio{pin}: line request failed ({e})") from e
        return _GpiodLine(self, pin)

    def set_value(self, pin: int, value: int):
        # count=1, config is the bit-mask of lines to change (bit 0).
        buf = _GpioLineSet(pin, 1, 1, (value & 1, 0, 0, 0))
        self._ioctl_fn(self.fd, LINESETVALS, buf)

    def get_value(self, pin: int) -> int:
        buf = _GpioLineSet(pin, 1, 1, (0, 0, 0, 0))
        self._ioctl_fn(self.fd, LINEGETVALS, buf)
        return int(buf.values[0]) & 1

    def cleanup(self):
        for pin in list(self._pins):
            try:
                buf = _GpioLineSet(pin, 1, 0, (0, 0, 0, 0))
                self._ioctl_fn(self.fd, LINEUNSET, buf)
            except OSError:
                pass
            self._pins.pop(pin, None)
        try:
            os.close(self.fd)
        except (OSError, TypeError, ValueError):
            pass


class _GpiodLine:
    def __init__(self, gpio: "_GpiodGpio", pin: int):
        self._gpio = gpio
        self._pin = pin

    def set_value(self, value: int):
        self._gpio.set_value(self._pin, value)

    def read(self) -> int:
        try:
            return self._gpio.get_value(self._pin)
        except OSError:
            return 0


class _Spi:
    def __init__(self, spidev):
        self._spidev = spidev
        self._dev = None

    def open(self, bus: int, cs: int, speed: int):
        self._dev = self._spidev.SpiDev()
        self._dev.open(bus, cs)
        self._dev.max_speed_hz = speed
        self._dev.mode = 0

    def command(self, value: int):
        self._dev.xfer2([value & 0xFF])

    def write(self, data):
        self._dev.writebytes2(bytes(data))

    def close(self):
        if self._dev:
            self._dev.close()
            self._dev = None
