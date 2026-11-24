"""
Tests for the speaker pop / dropped-first-second fixes:
  * hub.TtsTailBuffer    - hold-back + fade-out of the end of TTS speech
  * spoke Playback._fade_in - ramp at the start of each speech burst
Pure logic only; no audio hardware, no network.
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "docker" / "hub"))
sys.path.insert(0, str(REPO / "spokes" / "python-spoke"))

from glados_hub import hub  # noqa: E402

try:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "spoke_main", REPO / "spokes" / "python-spoke" / "main.py"
    )
    spoke = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(spoke)
    _SPOKE = True
except Exception:  # pragma: no cover
    _SPOKE = False


def _pcm(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


class TestTtsTailBuffer(unittest.TestCase):
    HOLD = 1440  # 30 ms @ 24 kHz s16 mono

    def test_no_samples_lost(self):
        buf = hub.TtsTailBuffer(holdback=self.HOLD)
        frame = _pcm(np.full(4800, 1000))  # one 200 ms frame
        sent = buf.feed(frame) + buf.feed(frame) + buf.flush()
        # everything we sent + what we held must equal the input + silence tail
        silence = int(hub.TtsTailBuffer.SILENCE_TAIL_S * hub.TTS_SAMPLE_RATE * 2)
        self.assertEqual(len(sent), 2 * len(frame) + silence)

    def test_holdback_size(self):
        buf = hub.TtsTailBuffer(holdback=self.HOLD)
        frame = _pcm(np.full(4800, 500))  # 9600 bytes
        out = buf.feed(frame)
        self.assertEqual(len(out), len(frame) - self.HOLD)
        self.assertEqual(len(buf.pending), self.HOLD)

    def test_feed_smaller_than_holdback_sends_nothing(self):
        buf = hub.TtsTailBuffer(holdback=self.HOLD)
        self.assertEqual(buf.feed(_pcm(np.full(100, 500))), b"")

    def test_flush_fades_to_silence(self):
        buf = hub.TtsTailBuffer(holdback=self.HOLD)
        buf.feed(_pcm(np.full(4800, 20000)))
        out = buf.flush()
        samples = np.frombuffer(out, dtype=np.int16)
        silence_n = int(hub.TtsTailBuffer.SILENCE_TAIL_S * hub.TTS_SAMPLE_RATE)
        # tail of pure silence
        self.assertTrue(np.all(samples[-silence_n:] == 0))
        # fade-out region: starts at full level, ramps down to 0
        ramp = samples[:-silence_n]
        self.assertGreater(abs(ramp[0]), 10000)
        self.assertLess(abs(ramp[-1]), 100)
        diffs = np.diff(ramp.astype(np.int32))
        self.assertTrue(np.all(diffs <= 0))

    def test_flush_empty_returns_only_silence(self):
        buf = hub.TtsTailBuffer(holdback=self.HOLD)
        out = buf.flush()
        silence_n = int(hub.TtsTailBuffer.SILENCE_TAIL_S * hub.TTS_SAMPLE_RATE * 2)
        self.assertEqual(len(out), silence_n)
        self.assertTrue(all(b == 0 for b in out))

    def test_streaming_multiple_frames(self):
        buf = hub.TtsTailBuffer(holdback=self.HOLD)
        total_in = 0
        total_out = b""
        for _ in range(5):
            frame = _pcm(np.full(4800, 1234))
            total_in += len(frame)
            total_out += buf.feed(frame)
        total_out += buf.flush()
        silence = int(hub.TtsTailBuffer.SILENCE_TAIL_S * hub.TTS_SAMPLE_RATE * 2)
        self.assertEqual(len(total_out), total_in + silence)


@unittest.skipUnless(_SPOKE, "spoke module not importable in this environment")
class TestSpokeFadeIn(unittest.TestCase):
    def test_first_burst_fades_in(self):
        pb = spoke.Playback()
        arr = np.full(spoke.TTS_CHUNK_BYTES // 2, 20000, dtype=np.int16)
        out = pb._fade_in(arr.copy())
        self.assertLess(abs(out[0]), 100)  # starts (near) silent
        self.assertLess(abs(out[100]), 5000)  # still ramping at ~14%
        self.assertGreater(abs(out[spoke.FADE_SAMPLES - 1]), 10000)
        self.assertEqual(abs(out[spoke.FADE_SAMPLES]), 20000)  # full after ramp
        # ramp is monotonic over the fade region
        ramp = out[: spoke.FADE_SAMPLES].astype(np.int32)
        self.assertTrue(np.all(np.diff(ramp) >= 0))

    def test_same_burst_does_not_refade(self):
        pb = spoke.Playback()
        a = np.full(spoke.TTS_CHUNK_BYTES // 2, 20000, dtype=np.int16)
        pb._fade_in(a.copy())  # first chunk (fades)
        out = pb._fade_in(a.copy())  # next chunk, same burst
        self.assertEqual(abs(out[0]), 20000)  # no ramp on continuation

    def test_gap_triggers_new_burst_fade(self):
        pb = spoke.Playback()
        a = np.full(spoke.TTS_CHUNK_BYTES // 2, 20000, dtype=np.int16)
        pb._fade_in(a.copy())
        pb._last_tts = 0.0  # simulate a long gap (new utterance)
        out = pb._fade_in(a.copy())
        self.assertLess(abs(out[0]), 100)  # fresh fade-in

    def test_short_frame_fades_whole_frame(self):
        pb = spoke.Playback()
        arr = np.full(50, 20000, dtype=np.int16)  # shorter than FADE_SAMPLES
        out = pb._fade_in(arr.copy())
        self.assertLess(abs(out[0]), 100)
        self.assertEqual(abs(out[-1]), 20000)

    def test_stop_resets_burst(self):
        import asyncio
        pb = spoke.Playback()
        a = np.full(100, 20000, dtype=np.int16)
        pb._fade_in(a.copy())
        self.assertTrue(pb._burst_active)
        asyncio.run(pb.stop())
        self.assertFalse(pb._burst_active)


@unittest.skipUnless(_SPOKE, "spoke module not importable in this environment")
class TestSpokeVolume(unittest.TestCase):
    """apply_volume is pure math; set_volume clamps; write() applies the
    gain after peak-limiting and before the fade-in."""

    def test_full_volume_is_noop(self):
        arr = np.array([1000, -2000, 20000, -20000], dtype=np.int16)
        self.assertIs(spoke.apply_volume(arr, 100), arr)
        self.assertIs(spoke.apply_volume(arr, 150), arr)  # clamps up, no-op

    def test_half_volume_scales(self):
        arr = np.array([1000, -2000, 20000, -20000], dtype=np.int16)
        out = spoke.apply_volume(arr, 50)
        self.assertTrue((out == np.array([500, -1000, 10000, -10000], dtype=np.int16)).all())

    def test_zero_volume_is_silence(self):
        arr = np.array([1000, -2000], dtype=np.int16)
        out = spoke.apply_volume(arr, 0)
        self.assertTrue((out == 0).all())
        self.assertEqual(out.dtype, np.int16)

    def test_negative_level_clamps_to_silence(self):
        out = spoke.apply_volume(np.array([1000], dtype=np.int16), -5)
        self.assertTrue((out == 0).all())

    def test_set_volume_clamps(self):
        pb = spoke.Playback()
        pb.set_volume(150)
        self.assertEqual(pb.volume, 100)
        pb.set_volume(-5)
        self.assertEqual(pb.volume, 0)
        pb.set_volume(42)
        self.assertEqual(pb.volume, 42)
        # no-op when unchanged (no log spam)
        pb.set_volume(42)
        self.assertEqual(pb.volume, 42)

    def test_write_applies_volume(self):
        import asyncio
        pb = spoke.Playback()
        pb.set_volume(50)
        captured = []

        async def fake_write(frame):
            captured.append(frame)

        pb._write_frame = fake_write
        pb._fade_in = lambda a: a  # isolate the volume math
        data = np.full(spoke.TTS_CHUNK_BYTES // 2, 20000, dtype=np.int16).tobytes()
        asyncio.run(pb.write(data))
        self.assertEqual(len(captured), 1)
        out = np.frombuffer(captured[0], dtype=np.int16)
        # peak-limited to MAX_PEAK then halved by 50% volume
        self.assertTrue((np.abs(out) == spoke.MAX_PEAK // 2).all())


@unittest.skipUnless(_SPOKE, "spoke module not importable in this environment")
class TestAudioDeviceResolution(unittest.TestCase):
    """GLADOS_*_DEVICE=none disables a direction (display-only bring-up)."""

    def _resolve(self, env):
        with mock.patch.dict(os.environ, env):
            return spoke._resolve_audio_devices()

    def test_both_none(self):
        cap, play, cap_off, play_off = self._resolve({
            'GLADOS_CAPTURE_DEVICE': 'none', 'GLADOS_PLAYBACK_DEVICE': 'none',
        })
        self.assertEqual((cap, play), ('none', 'none'))
        self.assertTrue(cap_off and play_off)

    def test_capture_only_none(self):
        cap, play, cap_off, play_off = self._resolve({
            'GLADOS_CAPTURE_DEVICE': 'none', 'GLADOS_PLAYBACK_DEVICE': 'plughw:2,0',
        })
        self.assertTrue(cap_off)
        self.assertFalse(play_off)
        self.assertEqual(play, 'plughw:2,0')

    def test_case_insensitive_and_stripped(self):
        cap, play, cap_off, play_off = self._resolve({
            'GLADOS_CAPTURE_DEVICE': '  NONE  ', 'GLADOS_PLAYBACK_DEVICE': 'None',
        })
        self.assertTrue(cap_off and play_off)

    def test_unset_uses_defaults_enabled(self):
        keys = ('GLADOS_CAPTURE_DEVICE', 'GLADOS_PLAYBACK_DEVICE',
                'GLADOS_AUDIO_DEVICE')
        saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
        try:
            cap, play, cap_off, play_off = spoke._resolve_audio_devices()
        finally:
            os.environ.update(saved)
        self.assertEqual(cap, spoke.DEFAULT_DEVICE)
        self.assertEqual(play, spoke.DEFAULT_DEVICE)
        self.assertFalse(cap_off or play_off)


@unittest.skipUnless(_SPOKE, "spoke module not importable in this environment")
class TestPlaybackDisabled(unittest.TestCase):
    """Playback.write must be a silent no-op when playback is disabled."""

    def test_write_drops_frames_when_disabled(self):
        pb = spoke.Playback()
        with mock.patch.object(spoke, 'PLAYBACK_DISABLED', True):
            asyncio.run(pb.write(b'\x01\x02' * 100))
        self.assertEqual(len(pb.buffer), 0)
        self.assertIsNone(pb.proc)


if __name__ == "__main__":
    unittest.main()
