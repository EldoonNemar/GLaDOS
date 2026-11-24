"""
LAFVIN Audio Display HAT (LB011) hardware layer for the python spoke.

Hardware (from LAFVIN's accepted driver, lfvintech/LAFVIN-Audio-Display-HAT):
  Display   1.69" IPS 240x280 ST7789, 4-wire SPI CS 0 @ 100 MHz,
            row offset 20
  RGB LED   ACTIVE-LOW, 100 Hz software PWM
  Button    polled at 5 ms
  Backlight software PWM

The 40-pin header lands on different SoC signals per SBC, so the pin map
is selected from the device-tree model at import time (see
`detect_board` / `BoardProfile`):

  Raspberry Pi (BCM)        Orange Pi Zero 2W (H618)
  board pin 13  DC   27        board pin 13  DC   259 (PI3)
  board pin  7  RST   4        board pin  7  RST  269 (PI13)
  board pin 15  BL   22        board pin 15  BL   261 (PI5)
  board pin 22  R    25        board pin 22  R    262 (PI6)
  board pin 18  G    24        board pin 18  G    228 (PH4)
  board pin 16  B    23        board pin 16  B    270 (PI14)
  board pin 11  BTN  17        board pin 11  BTN  226 (PH2)
  SPI bus 0 @ 100 MHz         SPI bus 1 @ 50 MHz

The Zero 2W offsets are verified against the official Orange Pi Zero 2W
wiki 40-pin table (H618 ports are 32 pins wide: PH base 224, PI base 256).
In particular board pins 22 and 24 are NOT the Raspberry Pi values
(GPIO 25 / GPIO 8): on the Zero 2W they are PI6 and PH5 (SPI1_CS0).

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
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("glados_spoke.lafvin")

WIDTH = 240
HEIGHT = 280
ROW_OFFSET = 20  # panel vertical gap (ST7789 240x280 1.69" layout)

# Per-SBC pin maps for the HAT's 40-pin header. Offsets are line numbers
# on the SoC's main GPIO chip (/dev/gpiochip0 on both boards).
@dataclass(frozen=True)
class BoardProfile:
    name: str
    spi_bus: int
    spi_speed_hz: int
    lcd_dc: int      # board pin 13
    lcd_rst: int     # board pin 7
    backlight: int   # board pin 15
    rgb_r: int       # board pin 22
    rgb_g: int       # board pin 18
    rgb_b: int       # board pin 16
    button: int      # board pin 11 (input)


_RASPBERRY_PI = BoardProfile(
    name="raspberry-pi",
    spi_bus=0,
    spi_speed_hz=100_000_000,
    lcd_dc=27,
    lcd_rst=4,
    backlight=22,
    rgb_r=25,
    rgb_g=24,
    rgb_b=23,
    button=17,
)

_ORANGE_PI_ZERO_2W = BoardProfile(
    name="orangepi-zero-2w",
    spi_bus=1,
    # H618 SPI1: conservative default; the wiki loopback test runs at
    # 500 kHz and the panel was validated at 100 MHz on RPi SPI0.
    spi_speed_hz=50_000_000,
    lcd_dc=259,      # PI3
    lcd_rst=269,     # PI13
    backlight=261,   # PI5
    rgb_r=262,       # PI6
    rgb_g=228,       # PH4
    rgb_b=270,       # PI14
    button=226,      # PH2
)


_KNOWN_BOARDS = {
    _RASPBERRY_PI.name: _RASPBERRY_PI,
    _ORANGE_PI_ZERO_2W.name: _ORANGE_PI_ZERO_2W,
}


def detect_board(model_path: str = "/proc/device-tree/model") -> BoardProfile:
    """Map the running SBC to its HAT BoardProfile.

    ``GLADOS_LAFVIN_BOARD`` overrides detection. That override is what the
    container actually uses: a container's ``/proc`` does not expose the
    host's device tree, so model-file detection always falls through inside
    Docker (compose sets the variable explicitly). On bare metal the host
    ``/proc/device-tree/model`` is used. An unknown or missing model keeps
    the legacy Raspberry Pi mapping so behavior never changes on
    unsupported hosts — `try_open` still fails gracefully when the expected
    SPI device is absent.
    """
    override = os.environ.get("GLADOS_LAFVIN_BOARD", "").strip().lower()
    if override:
        profile = _KNOWN_BOARDS.get(override)
        if profile is not None:
            return profile
        logger.warning(
            "Unknown GLADOS_LAFVIN_BOARD=%r; falling back to "
            "device-tree detection", override,
        )
    try:
        model = Path(model_path).read_bytes().decode("utf-8", "replace")
    except OSError:
        model = ""
    compact = re.sub(r"\s+", "", model).lower()
    if "orangepizero2w" in compact:
        return _ORANGE_PI_ZERO_2W
    return _RASPBERRY_PI


BOARD = detect_board()

# Backward-compatible module constants (tests and old callers import these).
LCD_DC_PIN = BOARD.lcd_dc
LCD_RST_PIN = BOARD.lcd_rst
BACKLIGHT_PIN = BOARD.backlight
RGB_R_PIN = BOARD.rgb_r
RGB_G_PIN = BOARD.rgb_g
RGB_B_PIN = BOARD.rgb_b
BUTTON_PIN = BOARD.button
SPI_BUS = BOARD.spi_bus
SPI_CS = 0
SPI_SPEED_HZ = BOARD.spi_speed_hz
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

    def __init__(self, gpio, stop_event=None, board=None):
        board = board or BOARD
        self._lines = {
            "r": gpio.output(board.rgb_r, "glados-rgb-r"),
            "g": gpio.output(board.rgb_g, "glados-rgb-g"),
            "b": gpio.output(board.rgb_b, "glados-rgb-b"),
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

    def __init__(self, on_press=None, board=None):
        self.on_press = on_press
        self._board = board  # None -> detect_board() profile at open time
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
    def try_open(cls, on_press=None, board=None) -> "LafvinHat | None":
        try:
            import spidev
        except Exception as e:
            logger.warning("[LAFVIN] unavailable (%s) — display disabled", e)
            return None
        hat = cls(on_press=on_press, board=board)
        try:
            hat._open(spidev)
        except Exception as e:
            logger.warning("[LAFVIN] open failed (%s) — display disabled", e)
            hat.close()
            return None
        profile = hat._board or BOARD
        logger.info("[LAFVIN] HAT ready on %s (ST7789 240x280 @ %d MHz, SPI%d)",
                    profile.name, profile.spi_speed_hz, profile.spi_bus)
        return hat

    def _open(self, spidev, board=None):
        board = board or self._board or BOARD
        self._board = board
        self._gpio = _GpiodGpio()
        self._spi = _Spi(spidev)
        self._spi.open(board.spi_bus, SPI_CS, board.spi_speed_hz)
        self.display = St7789(
            self._spi,
            self._gpio.output(board.lcd_dc, "glados-lcd-dc"),
            self._gpio.output(board.lcd_rst, "glados-lcd-rst"),
        )
        self.display.initialize()
        self.display.fill(0)
        self.led = RgbLed(self._gpio, stop_event=self._stop_event, board=board)
        bl = self._gpio.output(board.backlight, "glados-backlight")
        self._backlight_pwm = _SoftPwm(bl, stop_value=0)
        self._backlight_pwm.set_duty(float(self._backlight))
        threading.Thread(target=self._backlight_pwm.run,
                         args=(self._stop_event,), daemon=True,
                         name="lafvin-backlight").start()
        self._button_line = self._gpio.input(board.button, "glados-button")
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
# the RPi 6.1x kernel). gpiod talks to the chardev uAPI on
# /dev/gpiochip<N> — we do the same with raw ioctls: no gpiod build
# dependency, no RPi.GPIO Pi-ness check, and a --privileged container
# sees the host's /dev/gpiochip0.
#
# ABI v2 — the one the kernel 6.x gpiochar driver actually implements
# (verified byte-for-byte against /usr/include/linux/gpio.h on the
# target RPi, and against working `gpioinfo`/`gpioget` under strace):
#   _IOC(dir, type, nr, size) = dir<<30 | size<<16 | type<<8 | nr
#   GPIO_GET_CHIP_INFO      _IOR (0xB4, 0x01, struct gpiochip_info)
#   GPIO_V2_GET_LINE        _IOWR(0xB4, 0x07, struct gpio_v2_line_request)
#   GPIO_V2_LINE_GET_VALUES _IOWR(0xB4, 0x0E, struct gpio_v2_line_values)
#   GPIO_V2_LINE_SET_VALUES _IOWR(0xB4, 0x0F, struct gpio_v2_line_values)
# GET_LINE fills the NEW LINE FD into req.fd (the ioctl returns 0); the
# values ioctls then run on that line fd with a 64-bit bits/mask bitmap
# (bit 0 = the requested line).

GPIO_GET_CHIP_INFO = 0x8044B401      # _IOR(0xB4, 0x01, struct gpiochip_info) [68]
GPIO_V2_GET_LINE = 0xC250B407        # _IOWR(0xB4, 0x07, gpio_v2_line_request) [592]
GPIO_V2_LINE_GET_VALUES = 0xC010B40E  # _IOWR(0xB4, 0x0E, gpio_v2_line_values) [16]
GPIO_V2_LINE_SET_VALUES = 0xC010B40F  # _IOWR(0xB4, 0x0F, gpio_v2_line_values) [16]

FLAG_INPUT = 0x4            # GPIO_V2_LINE_FLAG_INPUT
FLAG_OUTPUT = 0x8           # GPIO_V2_LINE_FLAG_OUTPUT
FLAG_BIAS_DISABLED = 0x400  # GPIO_V2_LINE_FLAG_BIAS_DISABLED


# struct gpiochip_info from uapi/linux/gpio.h
class _GpiochipInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("label", ctypes.c_char * 32),
        ("lines", ctypes.c_uint32),
    ]


# struct gpio_v2_line_values from uapi/linux/gpio.h
class _GpioV2LineValues(ctypes.Structure):
    _fields_ = [
        ("bits", ctypes.c_uint64),
        ("mask", ctypes.c_uint64),
    ]


# struct gpio_v2_line_request from uapi/linux/gpio.h. The config/attrs
# fields are flattened to raw bytes: we request one line with plain
# flags and zero attributes, so nothing below `flags` is ever touched.
class _GpioV2LineRequest(ctypes.Structure):
    _fields_ = [
        ("offsets", ctypes.c_uint32 * 64),          # GPIO_V2_LINES_MAX
        ("consumer", ctypes.c_char * 32),           # GPIO_MAX_NAME_SIZE
        ("flags", ctypes.c_uint64),                 # gpio_v2_line_config.flags
        ("num_attrs", ctypes.c_uint32),
        ("config_pad", ctypes.c_uint32 * 5),
        ("attrs", ctypes.c_uint8 * 240),            # 10 × 24-byte config_attribute
        ("num_lines", ctypes.c_uint32),
        ("event_buffer_size", ctypes.c_uint32),
        ("padding", ctypes.c_uint32 * 5),
        ("fd", ctypes.c_int32),
    ]


class _GpiodGpio:
    """GPIO lines via the gpiod chardev uAPI (ABI v2). `open_fn`/
    `ioctl_fn`/`close_fn` are injectable so tests can run on machines
    without the HAT (production uses os.open + fcntl.ioctl + os.close on
    /dev/gpiochip0 — the Pi's 40-pin header exposes its BCM offsets as
    /dev/gpiochip0)."""

    def __init__(self, chip_path: str = "/dev/gpiochip0",
                 open_fn=None, ioctl_fn=None, close_fn=None):
        self._open_fn = open_fn or os.open
        self._ioctl_fn = ioctl_fn or fcntl.ioctl
        self._close_fn = close_fn or os.close
        self.fd = self._open_fn(chip_path, os.O_RDWR)
        self._lines: dict[int, int] = {}  # pin -> line fd
        self._num_lines: int | None = None
        try:
            info = _GpiochipInfo()
            self._ioctl_fn(self.fd, GPIO_GET_CHIP_INFO, info)
            self._num_lines = int(info.lines)
        except OSError:
            pass  # non-fatal — request errors carry their own errno

    def _request(self, pin: int, flags: int, label: str) -> int:
        if self._num_lines is not None and pin >= self._num_lines:
            raise OSError(f"gpio{pin}: not on this {self._num_lines}-line chip")
        req = _GpioV2LineRequest()
        req.offsets[0] = pin
        req.num_lines = 1
        req.flags = flags
        req.consumer = (label or f"glados-gpio{pin}")[:31].encode("ascii", "replace")
        req.fd = -1
        self._ioctl_fn(self.fd, GPIO_V2_GET_LINE, req)
        line_fd = int(req.fd)
        if line_fd < 0:
            raise OSError(f"gpio{pin}: line request failed (fd {line_fd})")
        return line_fd

    def output(self, pin: int, label: str = ""):
        line = self._request(pin, FLAG_OUTPUT, label)
        self._lines[pin] = line
        return _GpiodLine(self, pin)

    def input(self, pin: int, label: str = ""):
        # The HAT pulls the button line down on the board; the SoC bias
        # must stay OFF (fighting the two biases makes the idle level
        # indeterminate). HIGH = pressed, LOW = released.
        try:
            line = self._request(pin, FLAG_INPUT | FLAG_BIAS_DISABLED, label)
        except OSError:
            # Some gpiod/kernel combinations don't support bias-disable;
            # retry plain (matches the vendor driver's fallback).
            line = self._request(pin, FLAG_INPUT, label)
        self._lines[pin] = line
        return _GpiodLine(self, pin)

    def set_value(self, pin: int, value: int):
        # Line fd: bit 0 of the bitmap is the (sole) requested line.
        buf = _GpioV2LineValues(bits=value & 1, mask=1)
        self._ioctl_fn(self._lines[pin], GPIO_V2_LINE_SET_VALUES, buf)

    def get_value(self, pin: int) -> int:
        buf = _GpioV2LineValues(bits=0, mask=1)
        self._ioctl_fn(self._lines[pin], GPIO_V2_LINE_GET_VALUES, buf)
        return int(buf.bits) & 1

    def cleanup(self):
        for pin in list(self._lines):
            try:
                self._close_fn(self._lines[pin])
            except (OSError, TypeError, ValueError):
                pass
            self._lines.pop(pin, None)
        try:
            self._close_fn(self.fd)
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
