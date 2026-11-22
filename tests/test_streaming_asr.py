"""
Unit tests for the streaming ASR wrapper (hub).

Uses a mock KaldiRecognizer — no Vosk model required, so these run
anywhere (host or container). The live-model tests are in test_asr_live.py.
"""
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub.streaming_asr import (  # noqa: E402
    ASREventKind,
    StreamingASR,
    ASREngine,
    parse_vosk_json,
)


class FakeRecognizer:
    """Scripted stand-in for vosk.KaldiRecognizer.

    script: list of (is_final, text) tuples, consumed one per feed.
    """

    def __init__(self, script):
        self.script = list(script)
        self.fed = []
        self.reset_calls = 0

    def AcceptWaveform(self, pcm):
        self.fed.append(pcm)
        if not self.script:
            return False
        is_final, text = self.script[0]
        if is_final:
            self._last_final = text
            self.script.pop(0)
            return True
        self._last_partial = text
        self.script.pop(0)
        return False

    def Result(self):
        return json.dumps({"text": getattr(self, "_last_final", "")})

    def PartialResult(self):
        return json.dumps({"partial": getattr(self, "_last_partial", "")})

    def Reset(self):
        self.reset_calls += 1


def make_stream(script):
    rec = FakeRecognizer(script)
    stream = StreamingASR(model=None, recognizer_factory=lambda: rec)
    return stream, rec


class TestParseVoskJson(unittest.TestCase):
    def test_parses_result(self):
        self.assertEqual(parse_vosk_json('{"text": "hello"}', "text"), "hello")

    def test_parses_partial(self):
        self.assertEqual(parse_vosk_json('{"partial": "hel"}', "partial"), "hel")

    def test_bad_json_returns_empty(self):
        self.assertEqual(parse_vosk_json("not json", "text"), "")

    def test_missing_key_returns_empty(self):
        self.assertEqual(parse_vosk_json('{"other": 1}', "text"), "")


class TestStreamingASRFeed(unittest.TestCase):
    def test_partial_event(self):
        stream, rec = make_stream([(False, "what is"), (False, "what is the"), (True, "what is the time")])
        ev = stream.feed(b"\x00\x01" * 100)
        self.assertEqual(ev.kind, ASREventKind.PARTIAL)
        self.assertEqual(ev.text, "what is")

    def test_final_event(self):
        stream, _ = make_stream([(True, "what is the time")])
        ev = stream.feed(b"\x00\x01" * 100)
        self.assertEqual(ev.kind, ASREventKind.FINAL)
        self.assertEqual(ev.text, "what is the time")

    def test_final_auto_resets_stream(self):
        # A final must leave the decoder in a fresh state (no repeat text).
        stream, rec = make_stream([(True, "hello")])
        stream.feed(b"\x01" * 20)
        self.assertGreaterEqual(rec.reset_calls, 1)

    def test_empty_feed_is_noop(self):
        stream, _ = make_stream([])
        ev = stream.feed(b"")
        self.assertEqual(ev.kind, ASREventKind.NONE)
        self.assertEqual(ev.text, "")

    def test_silence_yields_empty_partial(self):
        # Vosk returns empty partials for silence — event carries no text.
        stream, _ = make_stream([(False, "")])
        ev = stream.feed(b"\x00" * 3200)
        self.assertEqual(ev.kind, ASREventKind.PARTIAL)
        self.assertEqual(ev.text, "")

    def test_recognizer_error_recovers(self):
        class BrokenRecognizer:
            def AcceptWaveform(self, pcm):
                raise RuntimeError("boom")
        stream = StreamingASR(model=None, recognizer_factory=BrokenRecognizer)
        ev = stream.feed(b"\x00" * 32)
        self.assertEqual(ev.kind, ASREventKind.NONE)
        # Stream still usable afterwards (re-created internally or None-safe)
        ev2 = stream.feed(b"\x00" * 32)
        self.assertEqual(ev2.kind, ASREventKind.NONE)


class TestStreamingASRLifecycle(unittest.TestCase):
    def test_reset_does_not_raise(self):
        stream, rec = make_stream([])
        stream.reset()
        self.assertEqual(rec.reset_calls, 1)

    def test_close_then_feed_is_safe(self):
        stream, _ = make_stream([])
        stream.close()
        ev = stream.feed(b"\x00" * 32)
        self.assertEqual(ev.kind, ASREventKind.NONE)


class TestHotwords(unittest.TestCase):
    def test_hotwords_cover_command_vocabulary(self):
        """The words that break commands ('lights'->'like', 'room'->'or white')
        must be in the boosted list."""
        from glados_hub import streaming_asr
        for w in ("lights", "light", "on", "off", "living", "room", "ceiling", "timer"):
            self.assertIn(w, streaming_asr.HOTWORDS)

    def test_default_factory_boosts_hotwords(self):
        from unittest import mock
        import vosk
        from glados_hub import streaming_asr

        calls = []

        class FakeRec:
            def __init__(self, model, rate):
                pass

            def SetWords(self, words):
                calls.append(list(words))

        with mock.patch.object(vosk, "KaldiRecognizer", FakeRec):
            streaming_asr.StreamingASR._default_factory(
                types.SimpleNamespace(_model=None)
            )
        self.assertEqual(calls, [streaming_asr.HOTWORDS])


class TestASREngine(unittest.TestCase):
    def test_create_stream_unavailable_without_load(self):
        engine = ASREngine(model_dir="/nonexistent/model/dir")
        self.assertFalse(engine.available)
        self.assertIsNone(engine.create_stream())

    def test_load_missing_model_is_graceful(self):
        engine = ASREngine(model_dir="/nonexistent/model/dir")
        self.assertFalse(engine.load())  # must not raise


if __name__ == "__main__":
    unittest.main()