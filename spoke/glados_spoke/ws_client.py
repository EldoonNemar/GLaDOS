"""WebSocket client with automatic reconnection — based on upstream reference client."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable

import numpy as np
import sounddevice as sd
import websockets


logger = logging.getLogger(__name__)


def _build_url(host: str, port: int, path: str) -> str:
    return f"ws://{host}:{port}{path}"


class ReconnectingWS:
    """WebSocket connection with automatic reconnection."""

    def __init__(
        self,
        url: str,
        *,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_jitter: float = 0.5,
        reconnect_attempts: int | None = None,
        heartbeat_interval_s: float = 5.0,
        on_open: Callable[[], Awaitable[None]] | None = None,
        on_close: Callable[[int, str | None], Awaitable[None]] | None = None,
    ) -> None:
        self.url = url
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.reconnect_jitter = reconnect_jitter
        self.reconnect_attempts = reconnect_attempts
        self.heartbeat_interval_s = heartbeat_interval_s
        self.on_open = on_open
        self.on_close = on_close
        self._ws = None
        self._running = False
        self._task = None
        self._attempt = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._connect_loop())
        logger.info("WS[%s]: connect loop started", self.url)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("WS[%s]: stopped", self.url)

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.state.name == "OPEN"

    async def send(self, data: bytes) -> None:
        if not self.connected:
            raise ConnectionError(f"WS[{self.url}]: not connected")
        await self._ws.send(data)

    async def _connect_loop(self) -> None:
        while self._running:
            try:
                await self._one_connection()
            except (ConnectionRefusedError, OSError):
                pass
            except Exception:
                logger.exception("WS[%s]: error", self.url)
            if not self._running:
                break
            await asyncio.get_event_loop().run_in_executor(None, self._backoff_delay)

    async def _one_connection(self) -> None:
        async with websockets.connect(
            self.url,
            ping_interval=self.heartbeat_interval_s,
            ping_timeout=self.heartbeat_interval_s * 3,
            compression=None,
        ) as ws:
            self._ws = ws
            self._attempt = 0
            logger.info("WS[%s]: connected", self.url)
            if self.on_open:
                await self.on_open()
            try:
                while self._running and ws.state.name == "OPEN":
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        await self._handle_binary(msg)
                    else:
                        await self._handle_text(msg)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self._ws = None

    def _backoff_delay(self) -> None:
        if self.reconnect_attempts is not None and self._attempt >= self.reconnect_attempts:
            logger.error("WS[%s]: max reconnect attempts reached", self.url)
            self._running = False
            return
        self._attempt += 1
        delay = min(
            self.reconnect_base_delay * (2 ** (self._attempt - 1)),
            self.reconnect_max_delay,
        )
        jitter = random.uniform(0, delay * self.reconnect_jitter)
        logger.info("WS[%s]: reconnecting in %.1fs (attempt %d)", self.url, delay + jitter, self._attempt)
        time.sleep(delay + jitter)

    async def _handle_binary(self, data: bytes) -> None:
        pass

    async def _handle_text(self, text: str) -> None:
        pass


class MicClient(ReconnectingWS):
    """Microphone client: streams float32 16 kHz audio to /microphone.

    Uses an asyncio.Queue to bridge the PortAudio callback safely into the
    asyncio event loop — matching the upstream reference client pattern.
    """

    def __init__(
        self,
        url: str,
        *,
        on_capture: Callable[[np.ndarray], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url, **kwargs)
        self._on_capture = on_capture
        self._out: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    async def _handle_text(self, text: str) -> None:
        logger.debug("MicClient received: %s", text[:80])

    def push_sample(self, samples: np.ndarray) -> None:
        """Called from the PortAudio callback thread."""
        data = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
        if self._out.full():
            self._out.get_nowait()
            self._out.task_done()
        self._out.put_nowait(data)

    async def start(self) -> None:
        await super().start()
        self._sender_task = asyncio.create_task(self._sender())

    async def stop(self) -> None:
        if hasattr(self, "_sender_task"):
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def _sender(self) -> None:
        """Send queued microphone blocks in capture order."""
        while True:
            try:
                data = await self._out.get()
                try:
                    if self._ws and self._ws.state.name == "OPEN":
                        await self._ws.send(data)
                finally:
                    self._out.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("MicClient sender error")


class SpeakerClient(ReconnectingWS):
    """Speaker client: receives TTS audio from /speaker and plays it.

    Matches the upstream reference client — no jitter buffer, just direct playback.
    """

    def __init__(
        self,
        url: str,
        spk_device: str | int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url, **kwargs)
        self.spk_device = spk_device
        self._current_sample_rate: int = 22050
        self._current_play_time: float = 0.0
        self._playback_task: asyncio.Task[None] | None = None

    async def _stop_playback(self) -> None:
        """Cancel pending playback and stop the output device."""
        if self._playback_task is None:
            return
        sd.stop()
        self._playback_task.cancel()
        try:
            await self._playback_task
        except asyncio.CancelledError:
            pass
        self._playback_task = None

    async def _handle_binary(self, data: bytes) -> None:
        if not data:
            return
        scheduled_time = self._current_play_time
        rate = self._current_sample_rate
        try:
            await self._stop_playback()
            self._playback_task = asyncio.create_task(self._play(data, scheduled_time, rate))
        except Exception:
            logger.exception("SpeakerClient: error scheduling audio")

    async def _play(self, data: bytes, scheduled_time: float, sample_rate: int) -> None:
        """Play one scheduled track and acknowledge its completion."""
        await asyncio.sleep(max(0.0, scheduled_time - time.time()))
        sd.play(np.frombuffer(data, dtype=np.float32), sample_rate, device=self.spk_device)
        await asyncio.to_thread(sd.wait)
        try:
            if self._ws and self._ws.state.name == "OPEN":
                await self._ws.send("played")
        except Exception:
            pass

    async def _handle_text(self, text: str) -> None:
        if text.startswith("time:"):
            self._current_play_time = float(text.split(":", 1)[1])
        elif text.startswith("sampleRate:"):
            self._current_sample_rate = int(text.split(":", 1)[1])
        elif text == "reset":
            await self._stop_playback()
