"""
Real-time streaming ASR built on Vosk (Kaldi).

A single Vosk Model is loaded once at startup. Each spoke connection gets
its own KaldiRecognizer — a *stateful* streaming decoder with built-in
endpoint detection.

Feed it raw 16 kHz mono int16 PCM frames as they arrive (the spokes send
100 ms chunks). Vosk returns:

  * PARTIAL results while the person is still talking (stable so far), and
  * a FINAL result exactly once, when the utterance ends (silence).

Because the decoder keeps its own state across frames, there is no sliding
window, no re-transcribing the same audio, and no duplicated text.
"""

from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger("glados_hub.asr")

DEFAULT_MODEL_DIR = os.environ.get(
    "VOSK_MODEL_DIR",
    "/models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15",
)
SAMPLE_RATE = 16000

# Words the decoder should prefer: the vocabulary GLaDOS actually acts on.
# Vosk's SetWords() raises their prior, which fixes the command-critical
# garbles seen in the wild ("lights" -> "like", "living room" -> "living or
# white"). Keep this list to words that matter for commands — boosting
# common words just biases the whole transcript.
HOTWORDS = [
    "glados", "gladys",
    "light", "lights", "on", "off",
    "living", "room", "ceiling", "kitchen", "bathroom", "bedroom", "hallway", "garage",
    "timer", "minute", "minutes",
    "announce", "dinner", "stop",
]


class ASREventKind:
    NONE = ""
    PARTIAL = "partial"
    FINAL = "final"


class ASREvent:
    """Outcome of feeding one audio frame into the stream."""

    __slots__ = ("kind", "text")

    def __init__(self, kind: str, text: str):
        self.kind = kind
        self.text = text

    def __repr__(self):  # pragma: no cover - debug aid
        return f"ASREvent(kind={self.kind!r}, text={self.text!r})"


def parse_vosk_json(payload: str, key: str) -> str:
    """Extract a string field from a Vosk Result()/PartialResult() payload."""
    try:
        return json.loads(payload).get(key, "")
    except (ValueError, AttributeError, TypeError):
        return ""


class StreamingASR:
    """One streaming recognition stream — one per spoke connection.

    `model` is the shared vosk.Model. `recognizer_factory` is an optional
    override used by unit tests (signature: () -> KaldiRecognizer-like).
    """

    def __init__(self, model, recognizer_factory=None):
        self._model = model
        # RLock: feed() holds the lock while calling reset() after a final.
        self._lock = threading.RLock()
        self._factory = recognizer_factory or self._default_factory
        self._rec = self._factory()

    def _default_factory(self):
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, SAMPLE_RATE)
        if HOTWORDS:
            rec.SetWords(HOTWORDS)
        return rec

    def feed(self, pcm: bytes) -> ASREvent:
        """Feed raw int16 mono PCM (any size, e.g. 100 ms).

        Returns an ASREvent: kind is 'partial' (more speech coming),
        'final' (utterance ended), or '' (no new info).
        """
        if not pcm:
            return ASREvent(ASREventKind.NONE, "")
        with self._lock:
            if self._rec is None:
                return ASREvent(ASREventKind.NONE, "")
            try:
                if self._rec.AcceptWaveform(pcm):
                    text = parse_vosk_json(self._rec.Result(), "text").strip()
                    self.reset()  # fresh decoder state for the next utterance
                    return ASREvent(ASREventKind.FINAL, text)
                text = parse_vosk_json(self._rec.PartialResult(), "partial").strip()
                return ASREvent(ASREventKind.PARTIAL, text)
            except Exception as e:  # broken stream: recover, don't crash the hub
                logger.error("ASR stream error: %s", e)
                self.reset()
                return ASREvent(ASREventKind.NONE, "")

    def reset(self):
        """Clear decoder state (e.g. after barge-in or session restart)."""
        with self._lock:
            try:
                if self._rec is not None:
                    self._rec.Reset()
            except Exception:
                try:
                    self._rec = self._factory()
                except Exception:
                    self._rec = None

    def close(self):
        with self._lock:
            self._rec = None


class ASREngine:
    """Owns the shared Vosk model and hands out per-connection streams."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self._model = None
        self.available = False

    def load(self) -> bool:
        """Load the model once. Safe to call at startup; never raises."""
        try:
            from vosk import Model

            logger.info("Loading Vosk streaming model: %s", self.model_dir)
            self._model = Model(self.model_dir)
            self.available = True
            logger.info("Vosk streaming ASR ready (16 kHz)")
        except Exception as e:
            logger.error("Vosk model load failed (%s) — streaming ASR disabled", e)
            self._model = None
            self.available = False
        return self.available

    def create_stream(self) -> StreamingASR | None:
        if not self.available or self._model is None:
            return None
        return StreamingASR(self._model)