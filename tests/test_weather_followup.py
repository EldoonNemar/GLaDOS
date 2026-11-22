"""
Unit tests for TTS text cleaning, weather context, location extraction,
and the follow-up window routing (hub).

Pure functions except the live Open-Meteo test (skipped without network).
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "hub"))

import httpx  # noqa: E402

from glados_hub import decision  # noqa: E402
from glados_hub import llm  # noqa: E402
from glados_hub import weather  # noqa: E402
try:
    from glados_hub import hub  # noqa: E402
    _HUB = True
except Exception:  # pragma: no cover - hub import needs the full stack
    _HUB = False


class TestCleanForTts(unittest.TestCase):
    def test_bold_stripped(self):
        self.assertEqual(llm.clean_for_tts("**bold** text"), "bold text")

    def test_italic_stripped(self):
        self.assertEqual(llm.clean_for_tts("*italic* word"), "italic word")

    def test_code_block_removed(self):
        self.assertEqual(llm.clean_for_tts("before ```python\nx=1\n``` after"), "before after")

    def test_inline_code_unwrapped(self):
        self.assertEqual(llm.clean_for_tts("run `docker compose up` now"), "run docker compose up now")

    def test_colon_becomes_pause(self):
        out = llm.clean_for_tts("Here is the weather: sunny")
        self.assertIn(", ", out)
        self.assertNotIn(":", out)

    def test_fahrenheit_spoken(self):
        out = llm.clean_for_tts("It is 72°F today")
        self.assertIn("degrees Fahrenheit", out)
        self.assertNotIn("°", out)

    def test_celsius_spoken(self):
        self.assertIn("degrees Celsius", llm.clean_for_tts("It is 22°C today"))

    def test_bullets_removed(self):
        out = llm.clean_for_tts("Options:\n- first\n- second")
        self.assertNotIn("-", out)
        self.assertIn("first", out)
        self.assertIn("second", out)

    def test_url_removed(self):
        out = llm.clean_for_tts("see https://example.com/x for more")
        self.assertNotIn("example.com", out)

    def test_link_keeps_text(self):
        self.assertEqual(llm.clean_for_tts("visit [Open-Meteo](https://x.io) now"), "visit Open-Meteo now")

    def test_em_dash_to_comma(self):
        self.assertNotIn("\u2014", llm.clean_for_tts("this \u2014 that"))

    def test_ampersand(self):
        self.assertIn("and", llm.clean_for_tts("fish & chips"))

    def test_empty(self):
        self.assertEqual(llm.clean_for_tts(""), "")
        self.assertEqual(llm.clean_for_tts(None), "")


class TestWeatherIntent(unittest.TestCase):
    def test_weather_words(self):
        for text in ("what's the weather", "weather tomorrow", "temperature outside",
                     "is it raining", "chance of snow", "will it storm"):
            self.assertTrue(decision.weather_intent(text), text)

    def test_no_weather(self):
        for text in ("tell me a joke", "what should I eat", "hey glados are you there"):
            self.assertFalse(decision.weather_intent(text), text)

    def test_empty(self):
        self.assertFalse(decision.weather_intent(""))


class TestExtractLocation(unittest.TestCase):
    def test_its_place(self):
        self.assertEqual(decision.extract_location("it's warrenton virginia"), "Warrenton Virginia")

    def test_its_place_with_wake(self):
        self.assertEqual(
            decision.extract_location("hey glados it's warrenton virginia"), "Warrenton Virginia"
        )

    def test_weather_in_place(self):
        self.assertEqual(
            decision.extract_location("what's the weather in austin texas"), "Austin Texas"
        )

    def test_im_in_place(self):
        self.assertEqual(decision.extract_location("i'm in boston"), "Boston")

    def test_weather_adverb_not_a_place(self):
        self.assertEqual(decision.extract_location("it's raining"), "")
        self.assertEqual(decision.extract_location("it's cold"), "")

    def test_random_speech_no_location(self):
        self.assertEqual(
            decision.extract_location("what should be a woman or a girl one and she's is brighter"), ""
        )

    def test_for_tomorrow_no_location(self):
        self.assertEqual(decision.extract_location("for tomorrow"), "")


class TestRouteFull(unittest.TestCase):
    def test_wake_word_wins(self):
        self.assertEqual(decision.route_full("hey glados, hi", followup_active=False), "wake")
        self.assertEqual(decision.route_full("hey glados, hi", followup_active=True), "wake")

    def test_followup_window(self):
        self.assertEqual(decision.route_full("it's warrenton virginia", followup_active=True), "followup")

    def test_outside_window_goes_to_llm(self):
        self.assertEqual(decision.route_full("it's warrenton virginia", followup_active=False), "ask_llm")


class TestWmoCodes(unittest.TestCase):
    def test_known_codes(self):
        self.assertEqual(weather.describe_wmo(0), "clear sky")
        self.assertEqual(weather.describe_wmo(61), "light rain")
        self.assertEqual(weather.describe_wmo(95), "thunderstorm")
        self.assertEqual(weather.describe_wmo("71"), "light snow")

    def test_unknown_codes(self):
        self.assertEqual(weather.describe_wmo(123), "unknown conditions")
        self.assertEqual(weather.describe_wmo(None), "unknown conditions")


class TestFormatWeatherBlock(unittest.TestCase):
    def test_block_contains_key_facts(self):
        current = {
            "temperature_2m": 41.3,
            "apparent_temperature": 38.9,
            "weather_code": 61,
            "wind_speed_10m": 9.2,
        }
        daily = {
            "time": ["2026-11-16", "2026-11-17", "2026-11-18"],
            "weather_code": [61, 2, 0],
            "temperature_2m_max": [44.0, 49.5, 52.1],
            "temperature_2m_min": [33.4, 31.0, 35.2],
            "precipitation_probability_max": [80, 20, 0],
        }
        block = weather.format_weather_block("Warrenton, Virginia, United States", current, daily)
        self.assertIn("Warrenton, Virginia", block)
        self.assertIn("Right now", block)
        self.assertIn("41 degrees Fahrenheit", block)
        self.assertIn("light rain", block)
        self.assertIn("tomorrow", block)
        self.assertIn("50 degrees Fahrenheit", block)
        self.assertIn("80 percent", block)
        self.assertIn("authoritative", block)


class TestGetWeatherBlockLive(unittest.TestCase):
    """Real Open-Meteo round trip (skipped when the container has no network)."""

    def _require_network(self):
        with httpx.Client() as probe:
            try:
                probe.get(weather.GEOCODE_URL, timeout=4.0)
            except Exception as e:
                self.skipTest(f"no network: {e}")

    def test_warrenton_virginia(self):
        self._require_network()

        async def run():
            async with httpx.AsyncClient() as client:
                return await weather.get_weather_block("Warrenton, Virginia", client)

        block = asyncio.run(run())
        self.assertTrue(block, "expected a weather block for Warrenton, Virginia")
        self.assertIn("Warrenton", block)
        self.assertIn("degrees Fahrenheit", block)
        self.assertIn("tomorrow", block)

    def test_unknown_place_returns_empty(self):
        self._require_network()

        async def run():
            async with httpx.AsyncClient() as client:
                return await weather.get_weather_block("Xyzzy Noplace 9000", client)

        self.assertEqual(asyncio.run(run()), "")


@unittest.skipUnless(_HUB, "hub module not importable in this environment")
class TestPlaybackDeadline(unittest.TestCase):
    """The mic-mute window must be sized to the ACTUAL audio duration, not a
    fixed constant — otherwise GLaDOS transcribes her own mid-sentence voice."""

    def test_short_audio_short_window(self):
        # 1 second of 24 kHz 16-bit mono = 48000 bytes.
        deadline = hub.compute_playback_deadline(1000.0, 48_000, now=2000.0)
        self.assertAlmostEqual(deadline, 1000.0 + 1.0 + hub.PLAYBACK_SETTLE_S)

    def test_long_audio_long_window(self):
        # 30 seconds of audio -> deadline ~31s after playback start, not 2s.
        bytes_30s = int(30 * hub.TTS_SAMPLE_RATE * 2)
        deadline = hub.compute_playback_deadline(1000.0, bytes_30s, now=1010.0)
        self.assertAlmostEqual(deadline, 1000.0 + 30.0 + hub.PLAYBACK_SETTLE_S)
        # The point of the fix: for long audio the deadline is far in the future.
        self.assertGreater(deadline, 1010.0 + 10)

    def test_no_playback_start_uses_now(self):
        deadline = hub.compute_playback_deadline(None, 0, now=5000.0)
        self.assertAlmostEqual(deadline, 5000.0 + hub.PLAYBACK_SETTLE_S)

    def test_zero_audio_only_margin(self):
        deadline = hub.compute_playback_deadline(100.0, 0, now=100.0)
        self.assertAlmostEqual(deadline, 100.0 + hub.PLAYBACK_SETTLE_S)


if __name__ == "__main__":
    unittest.main()

