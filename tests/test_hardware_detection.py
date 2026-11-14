"""Tests for hardware detection in setup_audio.py."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/opt/GLaDOS/spokes/python-spoke')
import setup_audio


class TestParseArecordList(unittest.TestCase):
    """Tests for arecord -l parsing."""

    def test_parse_single_device(self):
        mock_result = MagicMock()
        mock_result.stdout = "card 3: whisplaysound [Whisplay Sound], device 0: Whisplay HiFi wm8960-hifi-0 [Whisplay HiFi wm8960-hifi-0]\n"
        with patch('subprocess.run', return_value=mock_result):
            cards = setup_audio._parse_arecord_list()
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0], (3, 'whisplaysound', 'Whisplay Sound'))

    def test_parse_multiple_devices(self):
        mock_result = MagicMock()
        mock_result.stdout = """**** List of CAPTURE Hardware Devices ****
card 2: ahubhdmi [ahubhdmi], device 0: ahub_plat-i2s-hifi i2s-hifi-0 [ahub_plat-i2s-hifi i2s-hifi-0]
card 3: whisplaysound [Whisplay Sound], device 0: Whisplay HiFi wm8960-hifi-0 [Whisplay HiFi wm8960-hifi-0]
"""
        with patch('subprocess.run', return_value=mock_result):
            cards = setup_audio._parse_arecord_list()
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0], (2, 'ahubhdmi', 'ahubhdmi'))
        self.assertEqual(cards[1], (3, 'whisplaysound', 'Whisplay Sound'))

    def test_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch('subprocess.run', return_value=mock_result):
            cards = setup_audio._parse_arecord_list()
        self.assertIsNone(cards)

    def test_subprocess_error(self):
        with patch('subprocess.run', side_effect=Exception("failed")):
            cards = setup_audio._parse_arecord_list()
        self.assertIsNone(cards)


class TestListAlsaCards(unittest.TestCase):
    """Tests for list_alsa_cards with fallback."""

    def test_fallback_to_arecord(self):
        """When /proc/asound_cards fails, fallback to arecord -l."""
        mock_result = MagicMock()
        mock_result.stdout = "card 3: whisplaysound [Whisplay Sound], device 0: Whisplay HiFi wm8960-hifi-0 [Whisplay HiFi wm8960-hifi-0]\n"
        
        with patch('builtins.open', MagicMock(side_effect=FileNotFoundError())):
            with patch('subprocess.run', return_value=mock_result) as mock_subprocess:
                cards = setup_audio.list_alsa_cards()
        
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0], (3, 'whisplaysound', 'Whisplay Sound'))
        mock_subprocess.assert_called_once()


class TestFindXVF3800(unittest.TestCase):
    """Tests for XVF3800 detection."""

    def test_detect_xvf3800_by_name(self):
        cards = [(4, "Array", "USB Sound Device: USB Audio (TI3800)")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_xvf3800()
        self.assertEqual(result, 4)

    def test_detect_xvf3800_by_respeaker(self):
        cards = [(2, "reSpeaker", "reSpeaker 4-Mic Array")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_xvf3800()
        self.assertEqual(result, 2)

    def test_detect_xvf3800_by_3800_in_long_name(self):
        cards = [(4, "USB Audio", "XVF3800 Microphone Array")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_xvf3800()
        self.assertEqual(result, 4)

    def test_no_xvf3800(self):
        cards = [(0, "HDMI", "HDA ATI HDMI"), (3, "whisplaysound", "WM8960")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_xvf3800()
        self.assertIsNone(result)


class TestFindWhisplayHat(unittest.TestCase):
    """Tests for Whisplay HAT detection."""

    def test_detect_whisplay_by_name(self):
        cards = [(3, "whisplaysound", "Whisplay HiFi wm8960-hifi-0")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_whisplay_hat()
        self.assertEqual(result, 3)

    def test_detect_whisplay_by_codec(self):
        cards = [(3, "wm8960", "WM8960 Audio")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_whisplay_hat()
        self.assertEqual(result, 3)

    def test_no_whisplay(self):
        cards = [(0, "HDMI", "HDA ATI HDMI"), (4, "Array", "XVF3800")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            result = setup_audio.find_whisplay_hat()
        self.assertIsNone(result)


class TestSelectHardware(unittest.TestCase):
    """Tests for hardware selection logic."""

    def test_env_override_whisplay(self):
        cards = [(3, "whisplaysound", "WM8960")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            with patch.dict(os.environ, {'GLADOS_AUDIO_DEVICE': 'whisplay'}):
                result = setup_audio.select_hardware()
        self.assertEqual(result, ('whisplay', 3))

    def test_env_override_xvf3800(self):
        cards = [(4, "Array", "XVF3800")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            with patch.dict(os.environ, {'GLADOS_AUDIO_DEVICE': 'xvf3800'}):
                result = setup_audio.select_hardware()
        self.assertEqual(result, ('xvf3800', 4))

    def test_auto_detect_xvf3800_priority(self):
        cards = [(3, "whisplaysound", "WM8960"), (4, "Array", "XVF3800")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('GLADOS_AUDIO_DEVICE', None)
                result = setup_audio.select_hardware()
        self.assertEqual(result, ('xvf3800', 4))

    def test_auto_detect_whisplay_fallback(self):
        cards = [(3, "whisplaysound", "WM8960")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('GLADOS_AUDIO_DEVICE', None)
                result = setup_audio.select_hardware()
        self.assertEqual(result, ('whisplay', 3))

    def test_no_hardware_found(self):
        cards = [(0, "HDMI", "HDA ATI HDMI")]
        with patch.object(setup_audio, 'list_alsa_cards', return_value=cards):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('GLADOS_AUDIO_DEVICE', None)
                result = setup_audio.select_hardware()
        self.assertEqual(result, (None, None))


class TestStartSpokeEnvVars(unittest.TestCase):
    """Tests that start_spoke sets correct environment variables."""

    def test_set_env_vars_xvf3800(self):
        captured_env = {}
        def mock_execvp(cmd, args):
            captured_env.update(os.environ)
        with patch.object(setup_audio, 'os') as mock_os:
            mock_os.environ = captured_env
            mock_os.execvp = mock_execvp
            setup_audio.start_spoke('xvf3800', 4)
        self.assertEqual(captured_env['GLADOS_CAPTURE_DEVICE'], 'plughw:4,0')
        self.assertEqual(captured_env['GLADOS_PLAYBACK_DEVICE'], 'plughw:4,0')
        self.assertEqual(captured_env['GLADOS_HW_TYPE'], 'xvf3800')

    def test_set_env_vars_whisplay(self):
        captured_env = {}
        def mock_execvp(cmd, args):
            captured_env.update(os.environ)
        with patch.object(setup_audio, 'os') as mock_os:
            mock_os.environ = captured_env
            mock_os.execvp = mock_execvp
            setup_audio.start_spoke('whisplay', 3)
        self.assertEqual(captured_env['GLADOS_CAPTURE_DEVICE'], 'plughw:3,0')
        self.assertEqual(captured_env['GLADOS_PLAYBACK_DEVICE'], 'plughw:3,0')
        self.assertEqual(captured_env['GLADOS_HW_TYPE'], 'whisplay')


if __name__ == "__main__":
    unittest.main()
