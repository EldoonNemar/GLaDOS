"""
Live integration test: real Vosk streaming model + real speech sample.

Requires:
  * the vosk package (installed in the hub container)
  * the model at /models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15
  * the 16 kHz mono speech sample at tests/data/test_speech_16k.wav

Run inside the hub container:
  docker compose run --rm glados_hub python -m pytest /tests/test_asr_live.py -v
(skips automatically when the model is not present)
"""
import sys
import unittest
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docker" / "hub"))

MODEL_DIR = REPO / "models" / "vosk-model-small-en-us-0.15" / "vosk-model-small-en-us-0.15"
SAMPLE = REPO / "tests" / "data" / "test_speech_16k.wav"

skip_reasons = []
if not MODEL_DIR.exists():
    skip_reasons.append(f"model missing: {MODEL_DIR}")
if not SAMPLE.exists():
    skip_reasons.append(f"sample missing: {SAMPLE}")


def _load_engine():
    from glados_hub.streaming_asr import ASREngine

    engine = ASREngine(model_dir=str(MODEL_DIR))
    assert engine.load(), "Vosk model failed to load"
    return engine


def _read_pcm(path: Path, chunk_samples: int = 1600):
    """Yield int16 mono PCM chunks of chunk_samples (100 ms @ 16 kHz)."""
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000, f"sample is {w.getframerate()} Hz, need 16000"
        assert w.getnchannels() == 1, "sample must be mono"
        raw = w.readframes(w.getnframes())
    for i in range(0, len(raw), chunk_samples * 2):
        yield raw[i:i + chunk_samples * 2]


@unittest.skipIf(skip_reasons, "; ".join(skip_reasons) or "")
class TestLiveVoskStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _load_engine()

    def test_silence_produces_no_final_text(self):
        stream = self.engine.create_stream()
        self.assertIsNotNone(stream)
        silence = b"\x00" * 1600  # 100 ms
        finals = []
        for _ in range(20):  # 2 s of silence
            ev = stream.feed(silence)
            if ev.kind == "final":
                finals.append(ev.text)
        stream.close()
        for text in finals:
            self.assertEqual(text, "", f"silence should not transcribe words, got: {text!r}")

    def test_speech_produces_one_final_transcript(self):
        stream = self.engine.create_stream()
        self.assertIsNotNone(stream)
        finals = []
        partials = []
        for chunk in _read_pcm(SAMPLE):
            ev = stream.feed(chunk)
            if ev.kind == "final":
                finals.append(ev.text)
            elif ev.kind == "partial" and ev.text:
                partials.append(ev.text)
        # Let the endpoint fire: feed 1.5 s of trailing silence
        for _ in range(15):
            ev = stream.feed(b"\x00" * 1600)
            if ev.kind == "final":
                finals.append(ev.text)
        stream.close()

        non_empty_finals = [t for t in finals if t]
        self.assertEqual(
            len(non_empty_finals), 1,
            f"expected exactly one non-empty final transcript, got: {finals!r}",
        )
        text = non_empty_finals[0]
        self.assertGreaterEqual(len(text), 3)
        self.assertTrue(any(c.isalpha() for c in text), f"no letters in: {text!r}")
        print(f"\n[LiveASR] partials: {partials}")
        print(f"[LiveASR] final: {text!r}")