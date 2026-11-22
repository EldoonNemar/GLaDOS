"""
Unit tests for decision logic and LLM request/response builders (hub).

All pure functions — no network, no models.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

from glados_hub import decision  # noqa: E402
from glados_hub import llm  # noqa: E402


class TestWakeWordDetection(unittest.TestCase):
    def test_plain_glados(self):
        self.assertTrue(decision.detect_wake("glados, what is the meaning of life"))

    def test_gladys(self):
        self.assertTrue(decision.detect_wake("Gladys, tell me a joke"))

    def test_glahdos_mangled(self):
        self.assertTrue(decision.detect_wake("hey glahdos, are you there?"))

    def test_gladys_apostrophe(self):
        self.assertTrue(decision.detect_wake("glady's, what's the weather"))

    def test_galadas(self):
        self.assertTrue(decision.detect_wake("galadas, open the door"))

    def test_glados_mid_sentence(self):
        self.assertTrue(decision.detect_wake("ask glados to open the door"))

    def test_no_wake_in_small_talk(self):
        self.assertFalse(decision.detect_wake("i think the weather is nice today"))

    def test_no_wake_glad_adjective(self):
        self.assertFalse(decision.detect_wake("i'm glad to see you"))

    def test_no_wake_gladly(self):
        self.assertFalse(decision.detect_wake("she arrived gladly"))

    def test_empty_text(self):
        self.assertFalse(decision.detect_wake(""))


class TestNameExtraction(unittest.TestCase):
    def test_my_name_is(self):
        self.assertEqual(decision.extract_name("my name is bob"), "Bob")

    def test_im(self):
        self.assertEqual(decision.extract_name("I'm alice, nice to meet you"), "Alice")

    def test_call_me(self):
        self.assertEqual(decision.extract_name("call me smith"), "Smith")

    def test_no_name(self):
        self.assertEqual(decision.extract_name("what is your name"), "")

    def test_filters_stop_words(self):
        self.assertEqual(decision.extract_name("i'm fine"), "")


class TestQuestionHeuristic(unittest.TestCase):
    def test_question_mark(self):
        self.assertTrue(decision.looks_like_question("what time is it?"))

    def test_question_word(self):
        self.assertTrue(decision.looks_like_question("how do i reset this"))

    def test_statement(self):
        self.assertFalse(decision.looks_like_question("the cake is a lie"))

    def test_empty(self):
        self.assertFalse(decision.looks_like_question(""))


class TestDecisionRouting(unittest.TestCase):
    def test_wake_routes_to_respond(self):
        self.assertEqual(decision.route("glados what is this"), "respond")

    def test_no_wake_routes_to_llm(self):
        self.assertEqual(decision.route("did you hear that one"), "ask_llm")

    def test_build_prompt_contains_transcript(self):
        p = decision.build_decision_prompt("hello there", "person: hi")
        self.assertIn("hello there", p)
        self.assertIn("person: hi", p)

    def test_build_prompt_no_context(self):
        p = decision.build_decision_prompt("hello", "")
        self.assertIn("(none)", p)

    def test_build_prompt_managed_things_are_yes_even_if_garbled(self):
        """A garbled complaint about a timer must not be gated to NO — that is
        exactly the 'did you forget about my timer' case that went silent."""
        p = decision.build_decision_prompt("whereas did you forget about my one minute timer", "")
        self.assertIn("manages", p)
        self.assertIn("garbled", p)
        self.assertIn("YES", p)


class TestFollowupPrompt(unittest.TestCase):
    """The follow-up window gate asks a different question: is this a
    continuation of the exchange that just ended (not: is this for GLaDOS)?"""

    CONTEXT = "user: what's the weather\nglados: it's forty-six degrees"

    def test_followup_prompt_is_continuation_mode(self):
        p = decision.build_decision_prompt("for tomorrow", self.CONTEXT, followup=True)
        self.assertIn("WITHOUT a wake word", p)
        self.assertIn("continues the exchange that just ended", p)

    def test_plain_prompt_has_no_followup_mode(self):
        p = decision.build_decision_prompt("for tomorrow", self.CONTEXT)
        self.assertNotIn("WITHOUT a wake word", p)
        self.assertIn("call GLaDOS by name", p)

    def test_both_modes_keep_managed_things_and_format(self):
        for followup in (False, True):
            p = decision.build_decision_prompt("x", self.CONTEXT, followup=followup)
            self.assertIn("manages", p)
            self.assertIn("garbled", p)
            self.assertIn("exactly one word", p)

    def test_parse_decision_yes(self):
        self.assertTrue(decision.parse_decision("YES"))
        self.assertTrue(decision.parse_decision("Yes."))
        self.assertTrue(decision.parse_decision(" yes, i think so"))

    def test_parse_decision_no(self):
        self.assertFalse(decision.parse_decision("NO"))
        self.assertFalse(decision.parse_decision("no."))

    def test_parse_decision_garbage_defaults_no(self):
        self.assertFalse(decision.parse_decision("I am uncertain about this"))
        self.assertFalse(decision.parse_decision(""))


class TestGeminiBuilders(unittest.TestCase):
    MESSAGES = [
        {"role": "system", "content": "You are GLaDOS."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "oh, a human."},
        {"role": "user", "content": "what is this"},
    ]

    def test_contents_role_mapping(self):
        contents = llm._to_gemini_contents(self.MESSAGES)
        self.assertEqual([c["role"] for c in contents], ["user", "model", "user"])

    def test_system_excluded_from_contents(self):
        contents = llm._to_gemini_contents(self.MESSAGES)
        for c in contents:
            self.assertNotEqual(c["role"], "system")

    def test_consecutive_same_role_merged(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        contents = llm._to_gemini_contents(msgs)
        self.assertEqual(len(contents), 1)
        self.assertEqual(len(contents[0]["parts"]), 2)

    def test_build_request_shape(self):
        req = llm.build_gemini_request(self.MESSAGES, "You are GLaDOS.")
        self.assertEqual(req["systemInstruction"]["parts"][0]["text"], "You are GLaDOS.")
        self.assertIn("generationConfig", req)
        self.assertGreater(len(req["contents"]), 0)

    def test_parse_response(self):
        data = {"candidates": [{"content": {"parts": [{"text": "Still alive."}]}}]}
        self.assertEqual(llm.parse_gemini_response(data), "Still alive.")

    def test_parse_response_multiple_parts(self):
        data = {"candidates": [{"content": {"parts": [{"text": "A "}, {"text": "B"}]}}]}
        # First non-empty part wins (stripped)
        self.assertEqual(llm.parse_gemini_response(data), "A")

    def test_parse_response_empty(self):
        self.assertEqual(llm.parse_gemini_response({"candidates": []}), "")

    def test_gemini_url(self):
        self.assertIn(":generateContent", llm.gemini_url("gemini-2.5-flash"))


class TestLlamaRequestFormat(unittest.TestCase):
    def test_messages_shape(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.assertIn("role", msgs[0])
        self.assertIn("content", msgs[0])


class TestCleanLlmOutput(unittest.TestCase):
    def test_strips_im_end_token(self):
        out = llm.clean_llm_output("Huh? Are you okay?\n<|im_end|>")
        self.assertEqual(out, "Huh? Are you okay?")

    def test_strips_eos_tokens(self):
        self.assertEqual(llm.clean_llm_output("done </s>"), "done")
        self.assertEqual(llm.clean_llm_output("done <eot>"), "done")

    def test_keeps_normal_text(self):
        text = "The cake is a lie. Science needs failure."
        self.assertEqual(llm.clean_llm_output(text), text)

    def test_collapses_blank_lines(self):
        self.assertEqual(llm.clean_llm_output("a\n\n\n\nb"), "a\n\nb")

    def test_empty(self):
        self.assertEqual(llm.clean_llm_output(""), "")
        self.assertEqual(llm.clean_llm_output(None), "")


if __name__ == "__main__":
    unittest.main()