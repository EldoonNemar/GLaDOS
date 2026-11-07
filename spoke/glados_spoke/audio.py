"""Local audio capture and playback for the spoke."""

from __future__ import annotations

import logging

import math

import numpy as np

from numpy.typing import NDArray

import sounddevice as sd



logger = logging.getLogger(__name__)





class BasicAEC:

    """Simplified AEC using LMS adaptive filtering."""



    def __init__(self, order: int = 256, step_size: float = 0.001) -> None:

        self.order = order

        self.step_size = step_size

        self._weights = np.zeros(order, dtype=np.float32)

        self._delay = max(2, order // 16)



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

                self._weights += self.step_size * (mic[i] - echo_estimate) * tap

                energy = np.dot(tap, tap)

                if energy > 0:

                    self._weights *= min(1.0, 0.99 / max(math.sqrt(energy / self.order + 1e-12), 1e-12))

        return output.astype(np.float32)





class SpokeAudio:

    """Manages local mic capture and speaker playback."""



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

    ) -> None:

        self.mic_sample_rate = mic_sample_rate

        self.mic_channels = mic_channels

        self.mic_chunk_frames = mic_chunk_frames

        self.mic_device = mic_device

        self.mic_volume = mic_volume

        self.spk_sample_rate = spk_sample_rate

        self.spk_device = spk_device

        self._aec = BasicAEC(order=echo_aec_order) if echo_aec_enabled else None

        self._mic_stream = None

        self._spk_stream = None

        self._playback_ref = np.empty(0, dtype=np.float32)

        self._callback = None



    def start_mic(self, callback) -> None:

        self._callback = callback

        self._mic_stream = sd.InputStream(

            samplerate=self.mic_sample_rate,

            channels=self.mic_channels,

            blocksize=self.mic_chunk_frames,

            device=self.mic_device,

            callback=self._mic_cb,

            latency="low",

        )

        self._mic_stream.start()

        logger.info("Mic: %d Hz, %d ch, %d frames", self.mic_sample_rate, self.mic_channels, self.mic_chunk_frames)



    def stop_mic(self) -> None:

        if self._mic_stream:

            try:

                self._mic_stream.stop()

                self._mic_stream.close()

            except Exception:

                pass

            self._mic_stream = None



    def _mic_cb(self, indata, _frames, _time, status) -> None:

        if status:

            logger.warning("Mic status: %s", status)

        logger.info("Mic callback: indata shape=%s, volume=%.2f", indata.shape, self.mic_volume)
        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32) * self.mic_volume

        if self._aec is not None and len(self._playback_ref) >= self._aec.order:

            samples = self._aec.process(samples, self._playback_ref)

        if self._callback:

            self._callback(samples)



    def start_speaker(self) -> None:

        self._spk_stream = sd.OutputStream(

            samplerate=self.spk_sample_rate,

            channels=1,

            device=self.spk_device,

            latency="low",

        )

        self._spk_stream.start()

        logger.info("Speaker: %d Hz", self.spk_sample_rate)



    def stop_speaker(self) -> None:

        if self._spk_stream:

            try:

                self._spk_stream.stop()

                self._spk_stream.close()

            except Exception:

                pass

            self._spk_stream = None



    def play_audio(self, samples: NDArray[np.float32]) -> None:

        if len(samples) == 0:

            return

        if self.spk_sample_rate != 16000:

            from scipy import signal

            divisor = math.gcd(16000, self.spk_sample_rate)

            samples = signal.resample_poly(samples, self.spk_sample_rate // divisor, 16000 // divisor).astype(np.float32)

        self._playback_ref = samples

        try:

            sd.play(samples, self.spk_sample_rate, blocking=False)

        except Exception:

            logger.exception("Speaker playback error")



    def shutdown(self) -> None:

        self.stop_mic()

        self.stop_speaker()

