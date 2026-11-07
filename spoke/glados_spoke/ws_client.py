"""WebSocket client with automatic reconnection."""

from __future__ import annotations

import asyncio

import logging

import random

import time

from typing import Any, Awaitable, Callable

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

            ping_timeout=10.0,

            max_size=2**20,

        ) as ws:

            self._ws = ws

            self._attempt = 0

            if self.on_open:

                await self.on_open()

            try:

                while self._running and ws.state.name == "OPEN":

                    try:

                        msg = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat_interval_s * 3)

                        if isinstance(msg, bytes):

                            await self._handle_binary(msg)

                        else:

                            await self._handle_text(msg)

                    except asyncio.TimeoutError:

                        pass

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
    """Microphone client: pushes PCM audio chunks to the hub."""

    def __init__(self, url: str, on_capture=None, **kwargs: Any) -> None:
        super().__init__(url, **kwargs)
        self._on_capture = on_capture

    async def _handle_text(self, text: str) -> None:
        logger.debug("MicClient received: %s", text[:80])

    def push_sample(self, samples: np.ndarray) -> None:
        if self._ws is not None and self._ws.state.name == "OPEN":
            data = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
            asyncio.create_task(self._safe_send(data))

    async def _safe_send(self, data: bytes) -> None:
        if self._ws is not None and self._ws.state.name == "OPEN":
            try:
                await self._ws.send(data)
            except Exception:
                pass


class SpeakerClient(ReconnectingWS):
    """Speaker client: receives TTS audio from the hub."""

    def __init__(
        self, url: str, on_audio_chunk=None, on_text=None, **kwargs: Any
    ) -> None:
        super().__init__(url, **kwargs)
        self.on_audio_chunk = on_audio_chunk
        self._on_text = on_text
        self._current_sample_rate: int = 22050
        self._current_play_time: float = 0.0

    async def _handle_binary(self, data: bytes) -> None:
        if not data:
            return
        scheduled_time = self._current_play_time
        rate = self._current_sample_rate
        try:
            if self.on_audio_chunk:
                await self.on_audio_chunk(data, scheduled_time, rate)
        except Exception:
            logger.exception("SpeakerClient: error playing audio chunk")

    async def _handle_text(self, text: str) -> None:
        if text.startswith("time:"):
            self._current_play_time = float(text.split(":", 1)[1])
        elif text.startswith("sampleRate:"):
            self._current_sample_rate = int(text.split(":", 1)[1])
        elif text == "reset":
            pass
        elif self._on_text:
            await self._on_text(text)
