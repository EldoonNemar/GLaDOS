"""
Live test: the real mic rumble capture, through the hub's own pipeline.

tests/data/huh_rumble_16k.pcm is a 120 s raw capture of the live GLaDOS
mic (16 kHz int16 mono) — 97.5% of its energy is sub-200 Hz rumble
(fans/HVAC/appliances). It documents a three-part finding:

  1. Fed unfiltered into the small Vosk model, it produces a
     hallucinated 'huh' final every ~20 s (the reported bug).
  2. The 20 s rhythm is a Vosk small-model artifact on ANY continuous
     speech-like stationary noise (white noise at the same loudness
     reproduces the exact same final timestamps), so the high-pass
     alone does not remove the hallucination — it lowers the noise
     floor for real speech by ~16 dB.
  3. The hub's gates (backchannel filter + repeat suppression) drop
     100% of the hallucinated finals, so no decision-LLM call is ever
     made for the rumble.

Requires the vosk package + model (runs inside the hub container):
  docker compose run --rm glados_hub python -m pytest /tests/test_rumble_live.py -v
(skips automatically when the model is not present)
"""
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docker" / "hub"))

MODEL_DIR = REPO / "models" / "vosk-model-small-en-us-0.15" / "vosk-model-small-en-us-0.15"
CAPTURE = REPO / "tests" / "data" / "huh_rumble_16k.pcm"

skip_reasons = []
if not MODEL_DIR.exists():
    skip_reasons.append(f"model missing: {MODEL_DIR}")
if not CAPTURE.exists():
    skip_reasons.append(f"capture missing: {CAPTURE}")
try:
    import vosk  # noqa: F401
except ImportError:
    skip_reasons.append("vosk not installed (run inside the hub container)")


def _load_engine():
    from glados_hub.streaming_asr import ASREngine

    engine = ASREngine(model_dir=str(MODEL_DIR))
    assert engine.load(), "Vosk model failed to load"
    return engine


def _replay(engine, use_highpass: bool):
    """Feed the 120 s capture through 100 ms chunks; return non-empty finals."""
    from glados_hub.dsp import HighPass16k

    stream = engine.create_stream()
    hpf = HighPass16k()
    raw = CAPTURE.read_bytes()
    finals = []
    for i in range(0, len(raw) - 3200, 3200):
        chunk = raw[i:i + 3200]
        if use_highpass:
            chunk = hpf.process(chunk)
        ev = stream.feed(chunk)
        if ev.kind == "final" and ev.text:
            finals.append((i / 3200 / 10, ev.text))
    stream.close()
    return finals


def _rms(pcm: bytes) -> float:
    import numpy as np

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2)))


@unittest.skipIf(skip_reasons, "; ".join(skip_reasons) or "")
class TestRumbleCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _load_engine()

    def test_unfiltered_rumble_hallucinates(self):
        # Baseline: the capture exhibits the bug — 'huh' finals every ~20 s
        finals = _replay(self.engine, use_highpass=False)
        self.assertGreaterEqual(len(finals), 3, f"expected the rumble to hallucinate, got {finals!r}")
        print(f"\n[LiveRumble] unfiltered finals: {finals}")

    def test_highpass_lowers_noise_floor(self):
        # The high-pass removes the sub-200 Hz rumble (97.5% of the
        # capture's energy): at least 10 dB down.
        from glados_hub.dsp import HighPass16k

        raw = CAPTURE.read_bytes()
        f = HighPass16k()
        filtered = b"".join(f.process(raw[i:i + 3200])
                            for i in range(0, len(raw), 3200))
        gain_db = 20 * math.log10(_rms(filtered) / _rms(raw))
        self.assertLess(gain_db, -10.0, f"expected >=10 dB attenuation, got {gain_db:.1f} dB")
        print(f"\n[LiveRumble] high-pass attenuation: {gain_db:.1f} dB")

    def test_gates_stop_the_huh_storm(self):
        # End-to-end proof of the power fix: replay the capture through
        # the ASR, then push every final through the exact gates the hub
        # runs in _on_final (backchannel filter + repeat suppression, no
        # active conversation). The rumble must never reach the LLM.
        from glados_hub import decision
        from glados_hub.session import Session

        finals = _replay(self.engine, use_highpass=True)
        self.assertGreaterEqual(len(finals), 3, "sanity: the hallucination must still occur")
        session = Session("test", asr_stream=None, system_prompt="You are GLaDOS.")
        passed = []
        for t, text in finals:
            conversational = False  # nobody mid-conversation
            if decision.is_backchannel(text) and not conversational:
                continue
            if not session.register_final(text, conversational=conversational, now=t):
                continue
            passed.append((t, text))
        print(f"\n[LiveRumble] finals reaching the LLM after gates: {passed}")
        self.assertEqual(passed, [],
                         f"rumble finals must not reach the LLM, got: {passed!r}")


if __name__ == "__main__":
    unittest.main()
