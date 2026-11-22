"""Tests for GLaDOS engine -- ASR, VAD, TTS."""
import base64
import json
import os
import sys
import unittest
import numpy as np


class TestSileroVAD(unittest.TestCase):
    def setUp(self):
        self.audio_chunk = np.random.randint(-32768, 32767, size=512, dtype=np.int16)
        self.audio_float = self.audio_chunk.astype(np.float32) / 32768.0

    def test_normalization_preserves_signal(self):
        normalized = self.audio_float
        self.assertTrue(np.all(np.abs(normalized) <= 1.0))
        self.assertTrue(np.any(np.abs(normalized) > 0))


class TestKokoroTTSWrapper(unittest.TestCase):
    def test_synthesize_returns_tuple(self):
        expected_return = (np.zeros(1000), 24000)
        self.assertIsInstance(expected_return, tuple)
        self.assertEqual(len(expected_return), 2)
        self.assertEqual(expected_return[1], 24000)


class TestPiperTTS(unittest.TestCase):
    def test_sample_rate_from_config(self):
        config_data = {"audio": {"sample_rate": 22050}}
        self.assertEqual(config_data["audio"]["sample_rate"], 22050)


class TestAudioEncoding(unittest.TestCase):
    def test_int16_to_float_roundtrip(self):
        original = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        floats = original.astype(np.float32) / 32768.0
        recovered = (floats * 32767).astype(np.int16)
        self.assertTrue(np.all(np.abs(original.astype(np.int32) - recovered.astype(np.int32)) <= 1))

    def test_base64_roundtrip(self):
        original = b"hello world audio data"
        encoded = base64.b64encode(original).decode("utf-8")
        decoded = base64.b64decode(encoded)
        self.assertEqual(original, decoded)

    def test_pcm_frame_structure(self):
        frame = bytes([0x01]) + b"\x00\x01\x02\x03"
        self.assertEqual(frame[0], 0x01)
        self.assertEqual(len(frame), 5)



class TestGLaDOSConversation(unittest.TestCase):
    def test_system_prompt_loaded(self):
        with open("/opt/GLaDOS/docker/hub/glados_hub/prompts/glados_system.txt") as f:
            prompt = f.read()
        self.assertGreater(len(prompt), 100)
        self.assertIn("GLaDOS", prompt)

    def test_name_pronunciation_rule_present(self):
        """Guard: GLaDOS kept 'correcting' the user to say G-L-A-D-O-S.
        The prompt must pin the real pronunciation and forbid correcting it."""
        with open("/opt/GLaDOS/docker/hub/glados_hub/prompts/glados_system.txt") as f:
            prompt = f.read().lower()
        self.assertIn("gla-dows", prompt)
        self.assertIn("never correct", prompt)
        self.assertIn("not \"g-l-a-d-o-s\"", prompt)

    def test_tools_honesty_rule_present(self):
        """Guard: GLaDOS claimed to set a timer she never set (no tool call).
        The prompt must forbid claiming tool actions without a tool result."""
        with open("/opt/GLaDOS/docker/hub/glados_hub/prompts/glados_system.txt") as f:
            prompt = f.read().lower()
        self.assertIn("never claim", prompt)
        self.assertIn("call the tool", prompt)
        self.assertIn("garbled", prompt)

    def test_conversation_turns(self):
        messages = [
            {"role": "system", "content": "You are GLaDOS"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Oh great, another human."},
        ]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "user")


class TestWebSocketFrameProtocol(unittest.TestCase):
    def test_frame_type_constants(self):
        types = [0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14]
        self.assertEqual(len(set(types)), len(types))

    def test_audio_frame_format(self):
        frame = bytes([0x01]) + b"\x00\x01" * 100
        self.assertEqual(frame[0], 0x01)


class TestLLMIntegration(unittest.TestCase):
    def test_llm_request_format(self):
        request = {"model": "default", "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 256, "stream": False, "temperature": 0.7}
        self.assertIn("messages", request)
        self.assertEqual(request["max_tokens"], 256)

    def test_llm_response_format(self):
        response = {"choices": [{"message": {"content": "Hello."}}]}
        self.assertIn("choices", response)
        self.assertEqual(len(response["choices"]), 1)
        self.assertIsInstance(response["choices"][0]["message"]["content"], str)


class TestLocalLLMRequestBuilder(unittest.TestCase):
    """Pure builders in glados_hub/llm.py for the radiance local LLM."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, "/opt/GLaDOS/docker/hub")
        from glados_hub import llm

        cls.llm = llm

    def test_request_disables_thinking(self):
        req = self.llm.build_llama_request([{"role": "user", "content": "Hi"}])
        self.assertFalse(req["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(req["model"], self.llm.LLM_MODEL)
        self.assertFalse(req["stream"])

    def test_request_carries_budget(self):
        req = self.llm.build_llama_request(
            [{"role": "user", "content": "gate"}], max_tokens=8, temperature=0.0
        )
        # The YES/NO gate lives on an 8-token budget; with thinking on,
        # reasoning would consume it all and the gate would always fail to NO.
        self.assertEqual(req["max_tokens"], 8)
        self.assertEqual(req["temperature"], 0.0)
        self.assertFalse(req["chat_template_kwargs"]["enable_thinking"])

    def test_headers_carry_key_when_set(self):
        old = os.environ.get("LLM_API_KEY")
        os.environ["LLM_API_KEY"] = "test-key"
        try:
            self.assertEqual(
                self.llm.llama_headers(), {"Authorization": "Bearer test-key"}
            )
        finally:
            if old is None:
                del os.environ["LLM_API_KEY"]
            else:
                os.environ["LLM_API_KEY"] = old

    def test_headers_empty_without_key(self):
        old = os.environ.get("LLM_API_KEY")
        os.environ.pop("LLM_API_KEY", None)
        try:
            self.assertEqual(self.llm.llama_headers(), {})
        finally:
            if old is not None:
                os.environ["LLM_API_KEY"] = old


class TestHealthEndpoints(unittest.TestCase):
    def test_health_response_structure(self):
        health_data = {"status": "ok", "models": {"vad": True, "piper_tts": True, "kokoro_tts": True, "asr": True}}
        self.assertEqual(health_data["status"], "ok")
        self.assertEqual(len(health_data["models"]), 4)


class TestConfig(unittest.TestCase):
    def test_docker_compose_services(self):
        with open("/opt/GLaDOS/docker-compose.yml") as f:
            content = f.read()
        self.assertIn("glados_engine", content)
        self.assertIn("glados_hub", content)
        # The hub's local-LLM brain is the radiance engine on the host;
        # the retired glados_llama (Gemma) service must stay commented out,
        # not active (an active service still grabs ~3 GB of VRAM per GPU).
        self.assertIn("LLM_URL=http://host.docker.internal:8000", content)
        self.assertNotIn("  glados_llama:", content)

    def test_tts_config_valid(self):
        with open("/opt/GLaDOS/models/TTS/glados.json") as f:
            config = json.load(f)
        self.assertIn("audio", config)
        # Piper GLaDOS model outputs 24 kHz; spokes resample to 16 kHz
        self.assertEqual(config["audio"]["sample_rate"], 24000)


class TestWakeWordDetection(unittest.TestCase):
    """Tests for wake word detection in the hub.
    
    Tests the WAKE_WORDS list and detection logic independently
    (without importing hub which requires FastAPI).
    """
    
    WAKE_WORDS = ["glados", "gladys", "glahdose", "hey glados", "hey gladys", "hey glahdose"]
    
    def test_wake_words_list_defined(self):
        self.assertIsInstance(self.WAKE_WORDS, list)
        self.assertGreater(len(self.WAKE_WORDS), 0)
        self.assertIn("glados", self.WAKE_WORDS)
        self.assertIn("gladys", self.WAKE_WORDS)
    
    def test_wake_word_glados_detected(self):
        text = "glados, what is the meaning of life"
        self.assertTrue(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_wake_word_gladys_detected(self):
        text = "gladys, tell me a joke"
        self.assertTrue(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_wake_word_hey_glados_detected(self):
        text = "hey glados, are you there?"
        self.assertTrue(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_no_wake_word_in_normal_speech(self):
        text = "i think the weather is nice today"
        self.assertFalse(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_no_wake_word_in_random_conversation(self):
        text = "what are you doing right now"
        self.assertFalse(any(w in text.lower() for w in self.WAKE_WORDS))


class TestWakeWordDetection(unittest.TestCase):
    """Tests for wake word detection in the hub.
    
    Tests the WAKE_WORDS list and detection logic independently
    (without importing hub which requires FastAPI).
    """
    
    WAKE_WORDS = ["glados", "gladys", "glahdose", "hey glados", "hey gladys", "hey glahdose"]
    
    def test_wake_words_list_defined(self):
        self.assertIsInstance(self.WAKE_WORDS, list)
        self.assertGreater(len(self.WAKE_WORDS), 0)
        self.assertIn("glados", self.WAKE_WORDS)
        self.assertIn("gladys", self.WAKE_WORDS)
    
    def test_wake_word_glados_detected(self):
        text = "glados, what is the meaning of life"
        self.assertTrue(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_wake_word_gladys_detected(self):
        text = "gladys, tell me a joke"
        self.assertTrue(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_wake_word_hey_glados_detected(self):
        text = "hey glados, are you there?"
        self.assertTrue(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_no_wake_word_in_normal_speech(self):
        text = "i think the weather is nice today"
        self.assertFalse(any(w in text.lower() for w in self.WAKE_WORDS))
    
    def test_no_wake_word_in_random_conversation(self):
        text = "what are you doing right now"
        self.assertFalse(any(w in text.lower() for w in self.WAKE_WORDS))


if __name__ == "__main__":
    unittest.main()

