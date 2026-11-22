"""
Unit tests for the streaming high-pass filter (glados_hub.dsp).

The filter kills the sub-200 Hz rumble (1-8 Hz wobble + 30-125 Hz hump)
that the GLaDOS mic picks up from fans/HVAC/appliances and that makes
the small ASR model hallucinate a 'huh' final every ~20 s.

Pure numpy — no network, no models.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub.dsp import (  # noqa: E402
    CUTOFF_HZ,
    SAMPLE_RATE,
    HighPass16k,
    highpass_biquad_coeffs,
)


def tone_pcm(freq: float, seconds: float, amp: int = 10000) -> bytes:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16).tobytes()


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2)))


def band_ratio(pcm: bytes, lo: float, hi: float) -> float:
    """Fraction of total energy inside [lo, hi) Hz."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    nfft = 1 << 14
    win = np.hanning(nfft)
    steps = range(0, len(x) - nfft, nfft)
    p = np.mean([np.abs(np.fft.rfft(x[i:i + nfft] * win)) ** 2
                 for i in steps], axis=0)
    freqs = np.fft.rfftfreq(nfft, 1 / SAMPLE_RATE)
    return float(p[(freqs >= lo) & (freqs < hi)].sum() / p.sum())


class TestCoeffs(unittest.TestCase):
    def test_highpass_dc_gain_zero(self):
        # DC gain = (b0+b1+b2)/(1+a1+a2); a high-pass passes no DC
        b0, b1, b2, a1, a2 = highpass_biquad_coeffs(CUTOFF_HZ, SAMPLE_RATE)
        self.assertAlmostEqual(b0 + b1 + b2, 0.0, places=9)

    def test_highpass_nyquist_gain_one(self):
        # Gain at fs/2 = (b0-b1+b2)/(1-a1+a2) must be unity
        b0, b1, b2, a1, a2 = highpass_biquad_coeffs(CUTOFF_HZ, SAMPLE_RATE)
        self.assertAlmostEqual(b0 - b1 + b2, 1.0 - a1 + a2, places=9)


class TestHighPass(unittest.TestCase):
    def test_subsonic_wobble_killed(self):
        # 3.4 Hz wobble — the dominant peak in the live capture
        raw = tone_pcm(3.4, 3.0)
        out = HighPass16k().process(raw)
        self.assertLess(rms(out) / rms(raw), 0.03)  # >26 dB down

    def test_63hz_killed(self):
        raw = tone_pcm(63.0, 3.0)
        out = HighPass16k().process(raw)
        self.assertLess(rms(out) / rms(raw), 0.2)  # >14 dB down

    def test_125hz_attenuated(self):
        raw = tone_pcm(125.0, 3.0)
        out = HighPass16k().process(raw)
        self.assertLess(rms(out) / rms(raw), 0.8)

    def test_speech_bands_pass(self):
        # 300 Hz sits in the 4th-order filter's transition band (-1.5 dB);
        # everything at and above 500 Hz passes essentially untouched.
        for freq in (300, 500, 1000, 2000, 3000):
            raw = tone_pcm(freq, 2.0)
            out = HighPass16k().process(raw)
            ratio = rms(out) / rms(raw)
            if freq >= 500:
                self.assertGreaterEqual(ratio, 0.95, f"{freq} Hz should pass")
            else:
                self.assertGreaterEqual(ratio, 0.8, f"{freq} Hz should mostly pass")

    def test_state_carry_across_chunks(self):
        # Filtering in 100 ms slices must equal filtering the whole block
        # (after the first transient slice) — filter state must persist.
        raw = tone_pcm(100.0, 3.0)
        full = np.frombuffer(HighPass16k().process(raw), dtype=np.int16)
        f = HighPass16k()
        parts = []
        for i in range(0, len(raw), 3200):
            parts.append(np.frombuffer(f.process(raw[i:i + 3200]), dtype=np.int16))
        streamed = np.concatenate(parts)
        self.assertTrue(np.allclose(full[3200:], streamed[3200:], atol=2))

    def test_empty_input(self):
        self.assertEqual(HighPass16k().process(b""), b"")

    def test_real_rumble_spectrum_rebuilt(self):
        # Synthetic replica of the live capture: 3.4 Hz wobble + 31/63/125 Hz
        # hump + 1 kHz rattle, at the live capture's loudness.
        t = np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE
        x = (
            9000 * np.sin(2 * np.pi * 3.4 * t)
            + 6000 * np.sin(2 * np.pi * 31.5 * t)
            + 5000 * np.sin(2 * np.pi * 63 * t)
            + 4000 * np.sin(2 * np.pi * 125 * t)
            + 2500 * np.sin(2 * np.pi * 1000 * t)
        ).astype(np.int16).tobytes()
        f = HighPass16k()
        out = b"".join(
            f.process(x[i:i + 3200]) for i in range(0, len(x), 3200)
        )
        # Overall level drops by at least 6 dB...
        self.assertLess(rms(out) / rms(x), 0.5)
        # ...because the sub-200 Hz hump is what got removed
        self.assertLess(band_ratio(out, 0, 200), 0.15)
        # and the 1 kHz rattle mostly survives (it is not what we filter)
        self.assertGreater(band_ratio(out, 800, 1200), 0.3)


if __name__ == "__main__":
    unittest.main()
