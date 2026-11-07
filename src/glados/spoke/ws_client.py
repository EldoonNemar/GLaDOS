"""WebSocket client with automatic reconnection for the GLaDOS spoke.

Handles two independent connections:
- ``/microphone``  : pushes raw PCM to the hub
- ``/speaker``     : receives TTS audio from the hub

Both connections auto-reconnect with exponential backoff on disconnect.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable

import numpy as np
import websockets
from websockets.client import ClientProtocol

logger = logging.getLogger(__name__)

SendCallback = Callable[[bytes], Awaitable[None]]
OnMicData = Callable[[bytes], None]
SpeakerHandler = Callable[[bytes, float, int], Awaitable[None]]


def _build_url(host: str, port: int, path: str) -> str:
    """Build a WebSocket URL from host, port, and path."""
    return f"ws://{host}:{port}{path}"


class ReconnectingWS:
    """A WebSocket connection with automatic reconnection.

    Wraps a single WebSocket endpoint and handles:
    * Auto-reconnect with exponential backoff
    * Per-message callbacks
    * Heartbeat pings
    * Graceful shutdown
    """

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
        self._ws: ClientProtocol | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._attempt = 0

    # -- Public API --

    async def start(self) -> None:
        """Start the reconnection loop (non-blocking)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._connect_loop())
        logger.info("WS[%s]: connect loop started", self.url)

    async def stop(self) -> None:
        """Signal the reconnection loop to stop and wait."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("WS[%s]: stopped", self.url)

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.open

    async def send(self, data: bytes) -> None:
        """Send bytes on the current connection."""
        if not self.connected:
            raise ConnectionError(f"WS[{self.url}]: not connected, cannot send")
        await self._ws.send(data)

    async def recv_text(self) -> str:
        """Receive a text message (blocks until one arrives)."""
        if not self.connected:
            raise ConnectionError(f"WS[{self.url}]: not connected, cannot recv")
        return await self._ws.recv()

    # -- Internal --

    async def _connect_loop(self) -> None:
        """Main loop: connect, handle, reconnect on failure."""
        while self._running:
            try:
                await self._one_connection()
            except (ConnectionRefusedError, OSError) as exc:
                logger.warning("WS[%s]: connection failed: %s", self.url, exc)
            except Exception:
                logger.exception("WS[%s]: unexpected error", self.url)
            if not self._running:
                break
            await self._backoff_delay()

    async def _one_connection(self) -> None:
        """Open one WebSocket connection and run until it closes."""
        logger.debug("WS[%s]: attempting connect", self.url)
        async with websockets.connect(
            self.url,
            ping_interval=self.heartbeat_interval_s,
            ping_timeout=10.0,
            max_size=2**20,  # 1 MB max frame
        ) as ws:
            self._ws = ws
            self._attempt = 0
            logger.info("WS[%s]: connected (attempt %d)", self.url, self._attempt + 1)
            if self.on_open:
                await self.on_open()
            try:
                while self._running and ws.open:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat_interval_s * 3)
                        if isinstance(msg, bytes):
                            await self._handle_binary(msg)
                        else:
                            await self._handle_text(msg)
                    except asyncio.TimeoutError:
                        pass  # heartbeat keep-alive, no message
            except websockets.exceptions.ConnectionClosed:
                logger.info("WS[%s]: connection closed", self.url)
            finally:
                if self.on_close:
                    try:
                        await self.on_close(None, None)
                    except Exception:
                        pass
                self._ws = None

    def _backoff_delay(self) -> None:
        """Calculate and sleep for the next reconnect delay."""
        if self.reconnect_attempts is not None and self._attempt >= self.reconnect_attempts:
            logger.error("WS[%s]: max reconnect attempts (%d) reached. Stopping.",
                         self.url, self.reconnect_attempts)
            self._running = False
            return
        self._attempt += 1
        delay = min(
            self.reconnect_base_delay * (2 ** (self._attempt - 1)),
            self.reconnect_max_delay,
        )
        jitter = random.uniform(0, delay * self.reconnect_jitter)
        total = delay + jitter
        logger.info("WS[%s]: reconnecting in %.1fs (attempt %d)", self.url, total, self._attempt)
        asyncio.get_event_loop().run_in_executor(None, time.sleep, total)

    async def _handle_binary(self, data: bytes) -> None:
        """Override in subclass to handle incoming binary data."""
        pass

    async def _handle_text(self, text: str) -> None:
        """Override in subclass to handle incoming text commands."""
        pass


class MicClient(ReconnectingWS):
    """Microphone client: pushes PCM audio chunks to the hub."""

    def __init__(
        self,
        url: str,
        on_capture: OnMicData,
        **kwargs: Any,
    ) -> None:
        super().__init__(url, **kwargs)
        self._on_capture = on_capture

    async def _handle_text(self, text: str) -> None:
        """Handle server messages (sample rate announcements, etc.)."""
        logger.debug("MicClient received: %s", text[:80])

    def push_sample(self, samples: np.ndarray) -> None:
        """Bridge from sounddevice callback into async send loop."""
        if self._ws is not None and self._ws.open:
            data = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
            asyncio.create_task(self._safe_send(data))

    async def _safe_send(self, data: bytes) -> None:
        if self._ws is not None and self._ws.open:
            try:
                await self._ws.send(data)
            except Exception:
                pass


class SpeakerClient(ReconnectingWS):
    """Speaker client: receives TTS audio from the hub."""

    def __init__(
        self,
        url: str,
        on_audio_chunk: SpeakerHandler,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url, **kwargs)
        self.on_audio_chunk = on_audio_chunk
        self._on_text = on_text
        self._current_sample_rate: int = 22050
        self._current_play_time: float = 0.0

    async def _handle_binary(self, data: bytes) -> None:
        """Play received TTS audio chunk."""
        if not data:
            return
        scheduled_time = self._current_play_time
        rate = self._current_sample_rate
        try:
            await self.on_audio_chunk(data, scheduled_time, rate)
        except Exception:
            logger.exception("SpeakerClient: error playing audio chunk")

    async def _handle_text(self, text: str) -> None:
        """Handle timing and control messages from the hub."""
        if text.startswith("time:"):
            self._current_play_time = float(text.split(":", 1)[1])
            logger.debug("SpeakerClient: play time synced to %.3f", self._current_play_time)
        elif text.startswith("sampleRate:"):
            self._current_sample_rate = int(text.split(":", 1)[1])
            logger.debug("SpeakerClient: sample rate = %d Hz", self._current_sample_rate)
        elif text == "reset":
            logger.info("SpeakerClient: playback reset requested")
        elif self._on_text:
            await self._on_text(text)
