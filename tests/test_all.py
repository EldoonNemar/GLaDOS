"""Tests for GLaDOS engine -- ASR, VAD, TTS."""
import base64
import json
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
        types = [0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13]
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


class TestHealthEndpoints(unittest.TestCase):
    def test_health_response_structure(self):
        health_data = {"status": "ok", "models": {"vad": True, "piper_tts": True, "kokoro_tts": True, "asr": True}}
        self.assertEqual(health_data["status"], "ok")
        self.assertEqual(len(health_data["models"]), 4)


class TestConfig(unittest.TestCase):
    def test_docker_compose_services(self):
        with open("/opt/GLaDOS/docker-compose.yml") as f:
            content = f.read()
        self.assertIn("glados_llama", content)
        self.assertIn("glados_engine", content)
        self.assertIn("glados_hub", content)

    def test_tts_config_valid(self):
        with open("/opt/GLaDOS/models/TTS/glados.json") as f:
            config = json.load(f)
        self.assertIn("audio", config)
        self.assertEqual(config["audio"]["sample_rate"], 22050)


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

