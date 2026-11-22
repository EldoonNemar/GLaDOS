"""
GLaDOS face for the LAFVIN HAT display — pure numpy, no image libraries.

Renders 240x280 frames as RGB565 bytes:
  IDLE      a warm yellow ball that floats around the screen (wander
            physics, soft wall bounces, motion trail, breathing)
  TALKING   center ball pulsing with the TTS amplitude + an audio-reactive
            segmented ring around it
  LISTENING cyan ball pulsing with the mic level
  THINKING  amber dot orbiting a faint circle (waiting on the hub)

The ball states render through a decaying trail buffer so motion leaves a
comet tail and state changes blend instead of popping.
"""
import math
import time

import numpy as np

from lafvin import WIDTH, HEIGHT

IDLE, LISTENING, THINKING, TALKING = "idle", "listening", "thinking", "talking"

TALK_WINDOW_S = 0.6      # TTS data within this window => talking
LISTEN_WINDOW_S = 0.45   # mic activity within this window => listening
THINK_WINDOW_S = 4.0     # after activity, before the next TTS => thinking
MIC_GATE = 0.045         # normalized mic RMS considered "speech"
TTS_GATE = 0.015         # normalized TTS RMS considered "talking"
TRAIL_DECAY = 0.80       # per-tick trail persistence

BG = np.array([0.015, 0.02, 0.045], dtype=np.float32)
YELLOW = np.array([1.0, 0.80, 0.22], dtype=np.float32)
YELLOW_CORE = np.array([1.0, 0.95, 0.55], dtype=np.float32)
AMBER = np.array([1.0, 0.62, 0.12], dtype=np.float32)
CYAN = np.array([0.25, 0.85, 1.0], dtype=np.float32)

# RGB LED + backlight per state (same color language as the ESP32 ring)
STATE_LED = {
    IDLE: (0, 0, 0),
    LISTENING: (0, 110, 255),
    THINKING: (255, 160, 0),
    TALKING: (0, 230, 110),
}
STATE_BACKLIGHT = {IDLE: 55, LISTENING: 100, THINKING: 85, TALKING: 100}


def to_rgb565(frame: np.ndarray) -> bytes:
    """float32 [H, W, 3] in 0..1 -> little-endian RGB565 bytes."""
    rgb = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    words = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return np.ascontiguousarray(words.astype("<u2")).tobytes()


class Face:
    """Stateful renderer. Call tick() at display fps; it returns one full
    RGB565 frame and advances the animation by `dt` seconds."""

    def __init__(self, seed: int | None = None, now_fn=time.monotonic):
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._px = float(self._rng.uniform(60, WIDTH - 60))
        self._py = float(self._rng.uniform(70, HEIGHT - 70))
        self._vx = float(self._rng.uniform(-25, 25))
        self._vy = float(self._rng.uniform(-25, 25))
        self._ph1 = float(self._rng.uniform(0, 2 * math.pi))
        self._ph2 = float(self._rng.uniform(0, 2 * math.pi))
        self._ph3 = float(self._rng.uniform(0, 2 * math.pi))
        self.state = IDLE
        self._last_tts = -1e9
        self._last_mic = -1e9
        self._tts_level = 0.0
        self._mic_level = 0.0
        self._trail = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        self._yy, self._xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        cx, cy = WIDTH / 2, HEIGHT / 2
        d = np.sqrt(((self._xx - cx) / (WIDTH * 0.75)) ** 2
                    + ((self._yy - cy) / (HEIGHT * 0.75)) ** 2)
        self._bg = BG + (0.05 * np.clip(1.0 - d, 0, 1))[:, :, None]

    def update(self, dt: float, mic_level: float, tts_level: float) -> str:
        """Advance state from normalized levels; returns the new state."""
        self._t += dt
        mic_level = min(1.0, max(0.0, float(mic_level)))
        tts_level = min(1.0, max(0.0, float(tts_level)))
        now = self._t
        if tts_level > TTS_GATE:
            self._last_tts = now
            self._tts_level = max(tts_level, self._tts_level * 0.6)
        else:
            self._tts_level *= 0.7
        if mic_level > MIC_GATE:
            self._last_mic = now
            self._mic_level = max(mic_level, self._mic_level * 0.6)
        else:
            self._mic_level *= 0.7
        if now - self._last_tts < TALK_WINDOW_S and self._tts_level > TTS_GATE:
            new = TALKING
        elif now - self._last_mic < LISTEN_WINDOW_S and self._mic_level > MIC_GATE:
            new = LISTENING
        elif (now - self._last_tts < THINK_WINDOW_S
              or now - self._last_mic < THINK_WINDOW_S):
            new = THINKING
        else:
            new = IDLE
        self.state = new
        return new

    # ── idle ball physics ──────────────────────────────────────────

    def _ball_radius(self) -> float:
        # breathing
        return 24.0 * (1.0 + 0.07 * math.sin(0.9 * self._t))

    def _step_ball(self, dt: float):
        t = self._t
        # gentle wander: incommensurate sines keep the drift organic
        ax = 14.0 * math.cos(0.31 * t + self._ph1) + 8.0 * math.cos(0.77 * t + self._ph3)
        ay = 14.0 * math.sin(0.27 * t + self._ph2) + 8.0 * math.sin(0.63 * t + self._ph1)
        self._vx += ax * dt
        self._vy += ay * dt
        self._vx *= 0.996
        self._vy *= 0.996
        speed = math.hypot(self._vx, self._vy)
        lo, hi = 18.0, 60.0  # px/s — drifting, never static or frantic
        if speed > hi:
            self._vx, self._vy = self._vx * hi / speed, self._vy * hi / speed
        elif 0 < speed < lo:
            self._vx, self._vy = self._vx * lo / speed, self._vy * lo / speed
        self._px += self._vx * dt
        self._py += self._vy * dt
        r = self._ball_radius()
        m = r + 8
        if self._px < m:
            self._px, self._vx = m, abs(self._vx) * 0.9
        elif self._px > WIDTH - m:
            self._px, self._vx = WIDTH - m, -abs(self._vx) * 0.9
        if self._py < m:
            self._py, self._vy = m, abs(self._vy) * 0.9
        elif self._py > HEIGHT - m:
            self._py, self._vy = HEIGHT - m, -abs(self._vy) * 0.9

    # ── render helpers ─────────────────────────────────────────────

    def _ball_field(self, cx, cy, r, color, core) -> np.ndarray:
        d = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        inten = np.power(np.clip(1.0 - d / r, 0.0, 1.0), 1.4)
        field = color[None, None, :] * inten[:, :, None]
        cd = d
        core_i = np.clip(1.0 - cd / (r * 0.45), 0.0, 1.0) * 0.7
        field += (core - color)[None, None, :] * core_i[:, :, None]
        hx, hy = cx - r * 0.35, cy - r * 0.4
        hd = np.sqrt((self._xx - hx) ** 2 + (self._yy - hy) ** 2)
        field += 0.5 * np.clip(1.0 - hd / (r * 0.30), 0.0, 1.0)[:, :, None]
        return np.clip(field, 0.0, 1.0)

    def _add_trail(self, field: np.ndarray):
        self._trail = np.maximum(self._trail * TRAIL_DECAY, field)

    def _ring_field(self, cx, cy, r0, level, n_seg, color) -> np.ndarray:
        d = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        a = np.arctan2(self._yy - cy, self._xx - cx)
        seg = (((a + self._t * 0.7) / (2 * math.pi) + 1.5) * n_seg).astype(int) % n_seg
        i = np.arange(n_seg)
        env = level * (0.55 + 0.45 * np.sin(i * 1.7 + self._t * 9.0 + self._ph3))
        seg_len = (6.0 + 30.0 * env)[seg]
        seg_i = np.clip(1.0 - (d - r0) / np.maximum(seg_len, 1.0), 0.0, 1.0)
        mask = (d > r0) & (d < r0 + seg_len + 1.0)
        return color[None, None, :] * (seg_i * mask)[:, :, None]

    # ── frame ──────────────────────────────────────────────────────

    def tick(self, dt: float, mic_level: float = 0.0,
             tts_level: float = 0.0) -> bytes:
        self.update(dt, mic_level, tts_level)
        cx0, cy0 = WIDTH / 2, HEIGHT / 2
        if self.state == IDLE:
            self._step_ball(dt)
            self._add_trail(self._ball_field(self._px, self._py,
                                             self._ball_radius(), YELLOW,
                                             YELLOW_CORE))
        elif self.state == TALKING:
            lvl = self._tts_level
            r = 30.0 * (0.75 + 0.9 * lvl)
            self._add_trail(self._ball_field(cx0, cy0, r, YELLOW, YELLOW_CORE))
            self._trail = np.maximum(
                self._trail,
                self._ring_field(cx0, cy0, r + 14.0, max(lvl, 0.15), 28, AMBER))
        elif self.state == LISTENING:
            lvl = self._mic_level
            r = 24.0 * (0.8 + 0.7 * lvl)
            self._add_trail(self._ball_field(
                cx0, cy0, r, CYAN,
                np.array([0.75, 0.97, 1.0], np.float32)))
            self._trail = np.maximum(
                self._trail,
                self._ring_field(cx0, cy0, r + 12.0, max(lvl, 0.1), 20, CYAN))
        else:  # THINKING
            a = self._t * 2.6 + self._ph1
            ox = cx0 + 46.0 * math.cos(a)
            oy = cy0 + 46.0 * math.sin(a)
            self._add_trail(self._ball_field(ox, oy, 7.0, AMBER, AMBER))
            d = np.sqrt((self._xx - cx0) ** 2 + (self._yy - cy0) ** 2)
            guide = np.clip(1.0 - abs(d - 46.0), 0.0, 1.0) * 0.12
            self._trail = np.maximum(self._trail,
                                     AMBER[None, None, :] * guide[:, :, None])
        frame = np.clip(self._bg + self._trail, 0.0, 1.0)
        return to_rgb565(frame)
