"""Local audio capture and playback for the GLaDOS spoke.

Handles:
* Microphone capture (16 kHz mono, float32)
* Speaker playback (TTS audio at 22050 Hz)
* Basic AEC (adaptive filter to suppress speaker bleed into mic)
* Volume control
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

logger = logging.getLogger(__name__)


class BasicAEC:
    """Simplified acoustic echo cancellation using LMS adaptive filtering.

    Uses the known playback audio (reference signal) to estimate the echo
    path and subtract it from the microphone input.
    """

    def __init__(self, order: int = 256, step_size: float = 0.001) -> None:
        self.order = order
        self.step_size = step_size
        self._weights: NDArray[np.float32] = np.zeros(order, dtype=np.float32)
        self._delay: int = max(2, order // 16)

    def process(self, mic: NDArray[np.float32], ref: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(mic) == 0:
            return mic
        ref_aligned = ref[max(0, len(ref) - self.order - self._delay):len(ref) - self._delay]
        if len(ref_aligned) < self.order:
            return mic
        output = mic.copy()
        for i in range(len(mic)):
            if i + self.order <= len(ref_aligned):
                tap = ref_aligned[i: i + self.order]
                echo_estimate = float(np.dot(self._weights, tap))
                output[i] = mic[i] - echo_estimate
                error = mic[i] - echo_estimate
                self._weights += self.step_size * error * tap
                energy = np.dot(tap, tap)
                if energy > 0:
                    self._weights *= min(1.0, 0.99 / max(math.sqrt(energy / self.order + 1e-12), 1e-12))
        return output.astype(np.float32)


class SpokeAudio:
    """Manages local mic capture and speaker playback for the spoke."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        mic_sample_rate: int = 16000,
        mic_channels: int = 1,
        mic_chunk_frames: int = 512,
        mic_device: int | str | None = None,
        mic_volume: float = 1.0,
        spk_sample_rate: int = 22050,
        spk_device: int | str | None = None,
        echo_aec_enabled: bool = False,
        echo_aec_order: int = 256,
        echo_mute_on_detection_ms: int = 0,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        if not _HAS_SD:
            raise ImportError("sounddevice is required for spoke audio")
        self.mic_sample_rate = mic_sample_rate
        self.mic_channels = mic_channels
        self.mic_chunk_frames = mic_chunk_frames
        self.mic_device = mic_device
        self.mic_volume = mic_volume
        self.spk_sample_rate = spk_sample_rate
        self.spk_device = spk_device
        self.echo_aec_enabled = echo_aec_enabled
        self.echo_mute_on_detection_ms = echo_mute_on_detection_ms
        self._shutdown_event = shutdown_event or threading.Event()
        self._aec: BasicAEC | None = BasicAEC(order=echo_aec_order) if echo_aec_enabled else None
        self._mic_muted = False
        self._mic_mute_end: float = 0.0
        self._playback_ref: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._mic_stream = None
        self._spk_stream = None

    def start_mic(self, callback) -> None:
        """Start mic capture. callback(samples) is called per chunk."""
        self._mic_callback = callback
        self._mic_stream = sd.InputStream(
            samplerate=self.mic_sample_rate,
            channels=self.mic_channels,
            blocksize=self.mic_chunk_frames,
            device=self.mic_device,
            callback=self._mic_cb,
            latency="low",
        )
        self._mic_stream.start()
        logger.info("Mic capture started: %d Hz, %d ch, %d frames/chunk",
                     self.mic_sample_rate, self.mic_channels, self.mic_chunk_frames)

    def stop_mic(self) -> None:
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

    def _mic_cb(self, indata: NDArray[np.float32], _frames: int, _time: Any, status: Any) -> None:
        if status:
            logger.warning("Mic callback status: %s", status)
        if self._shutdown_event.is_set():
            raise sd.CallbackStop
        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        samples *= self.mic_volume
        if self._aec is not None and len(self._playback_ref) >= self._aec.order:
            samples = self._aec.process(samples, self._playback_ref)
        now = time.monotonic()
        if self._mic_muted and now < self._mic_mute_end:
            samples.fill(0)
        elif self._mic_muted:
            self._mic_muted = False
        self._mic_callback(samples)

    def start_speaker(self) -> None:
        self._spk_stream = sd.OutputStream(
            samplerate=self.spk_sample_rate,
            channels=1,
            device=self.spk_device,
            latency="low",
        )
        self._spk_stream.start()
        logger.info("Speaker started: %d Hz, device=%s", self.spk_sample_rate, self.spk_device)

    def stop_speaker(self) -> None:
        if self._spk_stream is not None:
            try:
                self._spk_stream.stop()
                self._spk_stream.close()
            except Exception:
                pass
            self._spk_stream = None

    def play_audio(self, samples: NDArray[np.float32]) -> None:
        """Queue audio for playback (non-blocking)."""
        if len(samples) == 0:
            return
        if samples.shape[0] > 0 and self.spk_sample_rate != 16000:
            samples = self._resample(samples, self.spk_sample_rate)
        self._playback_ref = samples
        try:
            sd.play(samples, self.spk_sample_rate, blocking=False)
        except Exception:
            logger.exception("Speaker playback error")

    def _resample(self, audio: NDArray[np.float32], dst_rate: int) -> NDArray[np.float32]:
        from scipy import signal
        src_rate = 16000
        divisor = math.gcd(src_rate, dst_rate)
        return signal.resample_poly(audio, dst_rate // divisor, src_rate // divisor).astype(np.float32)

    def mute_mic(self, duration_ms: int) -> None:
        """Mute the mic for a duration after detecting echo."""
        if self.echo_mute_on_detection_ms > 0:
            self._mic_muted = True
            self._mic_mute_end = time.monotonic() + duration_ms / 1000.0

    @property
    def is_mic_active(self) -> bool:
        return self._mic_stream is not None and self._mic_stream.active

    @property
    def is_speaker_active(self) -> bool:
        return self._spk_stream is not None and self._spk_stream.active

    def shutdown(self) -> None:
        """Stop all audio streams."""
        self._shutdown_event.set()
        self.stop_mic()
        self.stop_speaker()
