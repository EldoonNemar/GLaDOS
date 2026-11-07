"""Jitter buffer for smooth playback of streamed TTS audio."""

from __future__ import annotations

import threading

import time

import numpy as np

from numpy.typing import NDArray





class JitterBuffer:

    """Adaptive jitter buffer for float32 audio samples."""



    def __init__(

        self,

        sample_rate: int,

        min_ms: int = 80,

        max_ms: int = 400,

        adapt_speed: float = 0.1,

    ) -> None:

        self.sample_rate = sample_rate

        self.min_samples = int(sample_rate * min_ms / 1000)

        self.max_samples = int(sample_rate * max_ms / 1000)

        self.adapt_speed = adapt_speed

        self._target: int = self.min_samples

        self._buf: NDArray[np.float32] = np.zeros(self.max_samples, dtype=np.float32)

        self._write_pos: int = 0

        self._read_pos: int = 0

        self._filled: int = 0

        self._lock = threading.Lock()

    def clear(self) -> None:
        """Clear all buffered audio."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._filled = 0

    def push(self, samples: NDArray[np.float32]) -> None:

        with self._lock:

            n = len(samples)

            if n == 0:

                return

            needed = self._read_pos + self._filled + n

            if needed > self.max_samples:

                drop = needed - self.max_samples

                self._read_pos = (self._read_pos + drop) % self.max_samples

                self._filled = max(0, self._filled - drop)

            end = min(self.max_samples, self._write_pos + n)

            chunk1 = end - self._write_pos

            self._buf[self._write_pos:end] = samples[:chunk1]

            if chunk1 < n:

                self._buf[: n - chunk1] = samples[chunk1:]

            self._write_pos = (self._write_pos + n) % self.max_samples

            self._filled = min(self._filled + n, self.max_samples)

            if self._filled < self._target and self._target < self.max_samples:

                self._target = min(

                    self._target + int(self.adapt_speed * (self._target - self.min_samples)),

                    self.max_samples,

                )



    def pop(self, n_samples: int | None = None) -> NDArray[np.float32]:

        with self._lock:

            actual = n_samples if n_samples is not None else self._filled

            actual = min(actual, self._filled)

            if actual == 0:

                return np.array([], dtype=np.float32)

            end = min(self.max_samples, self._read_pos + actual)

            chunk1 = end - self._read_pos

            result = np.empty(actual, dtype=np.float32)

            result[:chunk1] = self._buf[self._read_pos:end]

            if chunk1 < actual:

                result[chunk1:] = self._buf[: actual - chunk1]

            self._read_pos = (self._read_pos + actual) % self.max_samples

            self._filled -= actual

            if self._filled < self._target and self._target < self.max_samples:

                increase = max(1, int(self.adapt_speed * (self.max_samples - self._target)))

                self._target = min(self._target + increase, self.max_samples)

            elif self._filled > self._target + self.min_samples:

                self._target = max(

                    self._target - int(self.adapt_speed * (self._target - self.min_samples)),

                    self.min_samples,

                )

            return result



    @property

    def filled_ms(self) -> float:

        with self._lock:

            return self._filled * 1000.0 / self.sample_rate



    @property

    def underrun(self) -> bool:

        with self._lock:

            return self._filled == 0



    def snapshot(self) -> dict:

        with self._lock:

            return {

                "filled_ms": round(self._filled * 1000.0 / self.sample_rate, 1),

                "target_ms": round(self._target * 1000.0 / self.sample_rate, 1),

                "underrun": self._filled == 0,

            }

