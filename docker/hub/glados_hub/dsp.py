"""
Streaming DSP for the hub audio path (pure numpy, no scipy).

Why this exists: the GLaDOS mic sits in a real room. Fans, HVAC, and
appliances pump subsonic rumble (a 1-8 Hz wobble plus a 30-125 Hz hump)
straight into the wm8960 codec, which has no high-pass in its mic path.
The small Vosk model reads that *constant* low-frequency energy as
speech onsets, closes an "utterance" roughly every 20 seconds, and
emits a hallucinated "huh" final — each of which cost a decision-LLM
call. (Verified: a 120 s raw capture of the live mic, replayed through
the hub's own Vosk stream, produces 'huh' finals at 20.3/40.4/60.6/
80.7/100.9 s.)

`HighPass16k` removes the rumble before the ASR ever sees it: a
4th-order (2 x 2nd-order biquad, Butterworth) high-pass at 200 Hz.
Intelligible speech energy lives above ~250 Hz — even a deep male
voice carries it in harmonics well above the cut — so nothing the ASR
needs is lost.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
CUTOFF_HZ = 200.0


def highpass_biquad_coeffs(fc: float, fs: float, q: float = 0.7071) -> tuple:
    """RBJ cookbook 2nd-order high-pass. Returns (b0, b1, b2, a1, a2), a0=1."""
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha
    b0 = ((1.0 + cos_w0) / 2.0) / a0
    b1 = (-(1.0 + cos_w0)) / a0
    b2 = ((1.0 + cos_w0) / 2.0) / a0
    a1 = (-2.0 * cos_w0) / a0
    a2 = (1.0 - alpha) / a0
    return (b0, b1, b2, a1, a2)


class HighPass16k:
    """Stateful 4th-order 200 Hz high-pass for 16 kHz int16 mono PCM.

    One instance per Session (filter state must follow its stream).
    `process` takes one raw chunk (e.g. 100 ms) and returns the same
    number of filtered samples, packed as int16.
    """

    def __init__(self, cutoff: float = CUTOFF_HZ, fs: int = SAMPLE_RATE):
        coeffs = highpass_biquad_coeffs(cutoff, fs)
        # Two identical sections cascaded = 4th order (-24 dB/octave)
        self._secs = [coeffs, coeffs]
        # Per-section state: [x1, x2, y1, y2]
        self._state = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        for (b0, b1, b2, a1, a2), st in zip(self._secs, self._state):
            x1, x2, y1, y2 = st
            out = np.empty_like(x)
            for i in range(len(x)):
                xn = x[i]
                yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                out[i] = yn
                x1, x2 = xn, x1
                y1, y2 = yn, y1
            st[:] = (x1, x2, y1, y2)
            x = out
        # Clip: filter transients can overshoot the int16 range
        return np.clip(x, -32768, 32767).astype(np.int16).tobytes()
