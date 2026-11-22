"""
Unit tests for the ASR-final gates:

  * decision.is_backchannel — a lone "huh" outside the follow-up window
    must never cost a decision-LLM call (the live mic's rumble produced
    one every 20 s, all night).
  * Session.register_final — the same transcript repeating over and
    over (noisy/stuck audio path) is suppressed.

Pure logic — no network, no models, no FastAPI.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub import decision  # noqa: E402
from glados_hub.session import Session  # noqa: E402


def make_session() -> Session:
    return Session("test-dev", asr_stream=None, system_prompt="You are GLaDOS.")


class TestBackchannel(unittest.TestCase):
    def test_huh(self):
        self.assertTrue(decision.is_backchannel("huh"))

    def test_huh_with_punctuation(self):
        self.assertTrue(decision.is_backchannel("Huh?"))

    def test_hmm(self):
        self.assertTrue(decision.is_backchannel("hmm"))

    def test_mm(self):
        self.assertTrue(decision.is_backchannel("mm"))

    def test_oh(self):
        self.assertTrue(decision.is_backchannel("oh"))

    def test_two_words_is_not(self):
        self.assertFalse(decision.is_backchannel("huh what"))

    def test_question_is_not(self):
        self.assertFalse(decision.is_backchannel("huh, what is that"))

    def test_command_is_not(self):
        self.assertFalse(decision.is_backchannel("turn off the light"))

    def test_meaningful_words_are_not(self):
        # yes/no/ok/sure can answer a question GLaDOS just asked
        for word in ("yes", "no", "ok", "okay", "sure", "yeah"):
            self.assertFalse(decision.is_backchannel(word), word)

    def test_empty(self):
        self.assertFalse(decision.is_backchannel(""))

    def test_punctuation_only(self):
        self.assertFalse(decision.is_backchannel("???"))


class TestRepeatSuppression(unittest.TestCase):
    def test_fresh_text_passes(self):
        s = make_session()
        self.assertTrue(s.register_final("hello", now=0.0))

    def test_third_repeat_within_window_is_suppressed(self):
        s = make_session()
        self.assertTrue(s.register_final("huh", now=0.0))
        self.assertTrue(s.register_final("huh", now=20.0))
        self.assertFalse(s.register_final("huh", now=40.0))

    def test_suppression_lastes_for_cooldown(self):
        s = make_session()
        s.register_final("huh", now=0.0)
        s.register_final("huh", now=20.0)
        s.register_final("huh", now=40.0)  # triggers suppression
        self.assertFalse(s.register_final("huh", now=41.0 + 500))

    def test_suppression_expires(self):
        s = make_session()
        s.register_final("huh", now=0.0)
        s.register_final("huh", now=20.0)
        s.register_final("huh", now=40.0)
        # 10-minute cooldown over: the word can pass again
        self.assertTrue(s.register_final("huh", now=40.0 + 601))

    def test_conversational_repetition_not_suppressed(self):
        s = make_session()
        s.register_final("huh", now=0.0)
        s.register_final("huh", now=20.0, conversational=True)
        self.assertTrue(s.register_final("huh", now=40.0, conversational=True))

    def test_window_expiry_resets_count(self):
        s = make_session()
        s.register_final("huh", now=0.0)
        s.register_final("huh", now=20.0)
        # Third one 3 minutes later: new window, passes
        self.assertTrue(s.register_final("huh", now=200.0))

    def test_different_texts_independent(self):
        s = make_session()
        s.register_final("huh", now=0.0)
        s.register_final("hello", now=10.0)
        s.register_final("weather", now=20.0)
        self.assertTrue(s.register_final("huh", now=30.0))

    def test_normalization(self):
        s = make_session()
        # "Huh!", "huh", "HUH." are the same transcript after normalization
        self.assertTrue(s.register_final("Huh!", now=0.0))
        self.assertTrue(s.register_final("huh", now=20.0))
        self.assertFalse(s.register_final("HUH.", now=40.0))


if __name__ == "__main__":
    unittest.main()
