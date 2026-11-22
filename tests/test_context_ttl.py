"""
Unit tests for time-based context expiration (glados_hub.session.Session).

The bug these guard against: an utterance said before bed ("huh") stayed
in the LLM conversation and the decision-LLM context forever, so every
later prompt kept paying tokens to look at the same stale text.

After CONTEXT_TTL_S of silence, `prune_stale()` must drop that text from
both stores before the next LLM call happens.

Pure logic — no network, no models, no FastAPI.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub.session import (  # noqa: E402
    CONTEXT_TTL_S,
    HISTORY_MAX_WORDS,
    LLM_CONTEXT_MESSAGES,
    MESSAGES_HARD_CAP,
    Session,
)

SYSTEM = "You are GLaDOS."


def make_session() -> Session:
    return Session("test-dev", asr_stream=None, system_prompt=SYSTEM)


class TestPruneStaleMessages(unittest.TestCase):
    def test_fresh_messages_survive(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        s.add_message("assistant", "yes?", now=101.0)
        self.assertEqual(s.prune_stale(now=130.0), 0)
        self.assertEqual(len(s.messages), 3)
        self.assertEqual(len(s.llm_messages()), 2)

    def test_stale_messages_dropped_after_silence(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        s.add_message("assistant", "yes?", now=101.0)
        # 601 s of silence after the reply, with the default 600 s TTL
        dropped = s.prune_stale(now=702.0)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(s.messages), 1)  # system prompt only
        self.assertEqual(s.messages[0]["role"], "system")
        self.assertEqual(s.llm_messages(), [])

    def test_mixed_ages_keep_recent_drop_old(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        s.add_message("assistant", "yes?", now=101.0)
        s.add_message("user", "glados, what is the weather", now=700.0)
        dropped = s.prune_stale(now=720.0, ttl=600.0)
        self.assertEqual(dropped, 2)
        self.assertEqual(
            [m["content"] for m in s.llm_messages()],
            ["glados, what is the weather"],
        )

    def test_everything_stale_leaves_bare_system_prompt(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        s.add_message("assistant", "yes?", now=101.0)
        s.prune_stale(now=100000.0)
        self.assertEqual(s.messages, [{"role": "system", "content": SYSTEM}])
        self.assertEqual(s._msg_times, [])

    def test_prune_is_idempotent(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        s.prune_stale(now=100000.0)
        self.assertEqual(s.prune_stale(now=100000.0), 0)
        self.assertEqual(len(s.messages), 1)

    def test_custom_ttl(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        # 61 s of silence under a 60 s TTL
        self.assertEqual(s.prune_stale(now=161.0, ttl=60.0), 1)

    def test_message_exactly_ttl_old_is_kept(self):
        s = make_session()
        s.add_message("user", "huh", now=100.0)
        self.assertEqual(s.prune_stale(now=100.0 + CONTEXT_TTL_S), 0)

    def test_default_ttl_is_positive(self):
        self.assertGreater(CONTEXT_TTL_S, 0)


class TestPruneStaleHistory(unittest.TestCase):
    def test_stale_history_lines_dropped(self):
        s = make_session()
        s.add_heard("huh", now=100.0)
        s.add_heard("what is the weather", now=700.0)
        s.prune_stale(now=720.0, ttl=600.0)
        self.assertEqual(s.decision_context(), "what is the weather")

    def test_all_stale_history_yields_empty_context(self):
        s = make_session()
        s.add_heard("huh", now=100.0)
        s.prune_stale(now=100000.0)
        self.assertEqual(s.decision_context(), "")

    def test_named_speaker_prefix_survives_prune(self):
        s = make_session()
        s.last_name = "Bob"
        s.add_heard("huh", now=100.0)
        s.add_heard("hello", now=200.0)
        s.prune_stale(now=210.0, ttl=600.0)
        self.assertEqual(s.decision_context(), "Bob: huh\nBob: hello")

    def test_word_budget_still_enforced(self):
        s = make_session()
        for i in range(30):
            s.add_heard("word " * 9 + "end", now=100.0 + i)  # 10 words each
        total_words = len(" ".join(line for _, line in s.history).split())
        self.assertLessEqual(total_words, HISTORY_MAX_WORDS)
        # Prune must not resurrect lines past the budget
        s.prune_stale(now=101.0 + HISTORY_MAX_WORDS + 10, ttl=10.0)
        total_words = len(" ".join(line for _, line in s.history).split())
        self.assertLessEqual(total_words, HISTORY_MAX_WORDS)


class TestExistingCapsPreserved(unittest.TestCase):
    def test_message_hard_cap(self):
        s = make_session()
        for i in range(MESSAGES_HARD_CAP + 20):
            role = "user" if i % 2 == 0 else "assistant"
            s.add_message(role, f"msg {i}", now=float(i))
        self.assertEqual(len(s.messages), MESSAGES_HARD_CAP)
        self.assertEqual(len(s._msg_times), MESSAGES_HARD_CAP - 1)
        self.assertEqual(s.messages[0]["role"], "system")
        self.assertEqual(s.messages[-1]["content"], f"msg {MESSAGES_HARD_CAP + 19}")

    def test_llm_messages_slices_last_twelve(self):
        s = make_session()
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            s.add_message(role, f"msg {i}", now=float(i))
        self.assertEqual(len(s.llm_messages()), LLM_CONTEXT_MESSAGES)
        self.assertEqual(s.llm_messages()[0]["content"], "msg 8")


class TestHuhRegression(unittest.TestCase):
    """The reported bug: 'huh' said before bed kept spinning in the LLM."""

    def test_huh_gone_by_morning(self):
        s = make_session()
        # 11:58 pm: user mutters "huh", GLaDOS responds
        s.add_message("user", "huh", now=0.0)
        s.add_message("assistant", "Yes? I am here. I am always here.", now=1.0)
        s.add_heard("huh", now=0.0)
        # 7:02 am: ~8 h of silence, user says good morning
        morning = 7 * 3600 + 2 * 60
        s.prune_stale(now=morning)
        # Neither the decision LLM nor the response LLM sees last night's "huh"
        self.assertNotIn("huh", s.decision_context())
        self.assertNotIn("huh", " ".join(m["content"] for m in s.llm_messages()))
        # The new utterance lands in a clean context
        s.add_message("user", "good morning", now=morning)
        self.assertEqual(
            [m["content"] for m in s.llm_messages()], ["good morning"]
        )

    def test_short_pause_keeps_context(self):
        """A normal pause (under the TTL) must NOT wipe the conversation."""
        s = make_session()
        s.add_message("user", "huh", now=0.0)
        s.add_message("assistant", "yes?", now=1.0)
        s.add_heard("huh", now=0.0)
        # 30 s later — well inside the TTL and the follow-up window
        s.prune_stale(now=31.0)
        self.assertIn("huh", s.decision_context())
        self.assertEqual(len(s.llm_messages()), 2)


if __name__ == "__main__":
    unittest.main()
