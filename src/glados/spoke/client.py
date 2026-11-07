"""Main spoke client that ties together audio, WebSocket, jitter buffer, and echo detection.

Usage:
    python -m glados.spoke.client --hub 192.168.1.100 --port 5051

This client:
* Captures audio from the local mic and streams it to the hub
* Receives TTS audio from the hub and plays it through local speakers
* Auto-reconnects with exponential backoff
* Uses a jitter buffer for smooth playback
* Optional AEC to reduce speaker bleed into mic
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import signal
import sys
import threading
import time
from typing import Any

import numpy as np

from .audio import SpokeAudio
from .config import SpokeConfig
from .jitter_buffer import JitterBuffer
from .ws_client import MicClient, SpeakerClient, _build_url

logger = logging.getLogger(__name__)




class SpokeClient:
    """Main spoke client: audio capture -> hub -> audio playback."""

    def __init__(self, config: SpokeConfig | None = None) -> None:
        self.config = config or SpokeConfig()
        self._shutdown = threading.Event()
        self._audio = SpokeAudio(
            mic_sample_rate=self.config.mic_sample_rate,
            mic_channels=self.config.mic_channels,
            mic_chunk_frames=self.config.mic_chunk_frames,
            mic_device=self.config.mic_device,
            mic_volume=self.config.mic_volume,
            spk_sample_rate=self.config.spk_sample_rate,
            spk_device=self.config.spk_device,
            echo_aec_enabled=self.config.echo_aec_enabled,
            echo_aec_order=self.config.echo_aec_order,
            echo_mute_on_detection_ms=self.config.echo_mute_on_detection_ms,
            shutdown_event=self._shutdown,
        )
        # Jitter buffer for TTS playback
        self._jitter = JitterBuffer(
            sample_rate=self.config.spk_sample_rate,
            min_ms=self.config.jitter_buffer_min_ms,
            max_ms=self.config.jitter_buffer_max_ms,
            adapt_speed=self.config.jitter_adapt_speed,
        )
        # WebSocket clients
        self._mic_client = MicClient(
            url=_build_url(self.config.hub_host, self.config.hub_port, self.config.mic_path),
            on_capture=lambda samples: None,  # mic_data callback
            reconnect_base_delay=self.config.reconnect_base_delay,
            reconnect_max_delay=self.config.reconnect_max_delay,
            reconnect_jitter=self.config.reconnect_jitter,
            reconnect_attempts=self.config.reconnect_attempts,
            heartbeat_interval_s=self.config.heartbeat_interval_s,
        )
        self._speaker_client = SpeakerClient(
            url=_build_url(self.config.hub_host, self.config.hub_port, self.config.speaker_path),
            on_audio_chunk=self._handle_tts_audio,
            reconnect_base_delay=self.config.reconnect_base_delay,
            reconnect_max_delay=self.config.reconnect_max_delay,
            reconnect_jitter=self.config.reconnect_jitter,
            reconnect_attempts=self.config.reconnect_attempts,
            heartbeat_interval_s=self.config.heartbeat_interval_s,
        )
        self._audio_queue: list[np.ndarray] = []
        self._audio_lock = threading.Lock()
        self._running = False
        self._bg_tasks: list[asyncio.Task] = []

    # -- Lifecycle --

    async def start(self) -> None:
        """Start the spoke client."""
        logger.info("Starting GLaDOS spoke client -> %s:%d", self.config.hub_host, self.config.hub_port)
        self._running = True

        # Start audio hardware
        self._audio.start_mic(self._mic_capture)
        self._audio.start_speaker()

        # Start WebSocket connections
        await self._mic_client.start()
        await self._speaker_client.start()

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop, name="spoke-playback", daemon=True)
        self._playback_thread.start()

        logger.info("GLaDOS spoke client started. Waiting for shutdown...")
        await self._wait_for_shutdown()

    async def stop(self) -> None:
        """Stop the spoke client."""
        logger.info("Stopping GLaDOS spoke client...")
        self._running = False
        self._shutdown.set()
        await self._mic_client.stop()
        await self._speaker_client.stop()
        self._audio.shutdown()
        if hasattr(self, "_playback_thread") and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)
        for task in self._bg_tasks:
            task.cancel()
        logger.info("GLaDOS spoke client stopped.")

    async def _wait_for_shutdown(self) -> None:
        """Wait until shutdown signal or Ctrl-C."""
        loop = asyncio.get_running_loop()

        def _on_signal() -> None:
            asyncio.ensure_future(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_signal)
            except (NotImplementedError, OSError):
                pass  # Windows doesn't support add_signal_handler

        while self._running and not self._shutdown.is_set():
            await asyncio.sleep(0.5)

    # -- Microphone capture --

    def _mic_capture(self, samples: np.ndarray) -> None:
        """Called from the mic callback thread. Send samples to hub via WebSocket."""
        if not self._running:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._mic_client.send(np.ascontiguousarray(samples, dtype=np.float32).tobytes()),
                asyncio.get_event_loop(),
            )
        except Exception:
            pass  # Connection error will be handled by reconnect loop

    # -- TTS audio playback --

    async def _handle_tts_audio(self, data: bytes, play_time: float, sample_rate: int) -> None:
        """Receive TTS audio from hub and push to jitter buffer."""
        samples = np.frombuffer(data, dtype=np.float32)
        if len(samples) == 0:
            return
        # Push into jitter buffer
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._jitter.push, samples)

    def _playback_loop(self) -> None:
        """Background thread: pull from jitter buffer and play."""
        buf_size = self.config.spk_sample_rate // 10  # 100ms chunks
        silence = np.zeros(buf_size, dtype=np.float32)
        while self._running and not self._shutdown.is_set():
            try:
                samples = self._jitter.pop(buf_size)
                if len(samples) == 0:
                    # Underrun: play silence
                    time.sleep(0.001)
                    continue
                self._audio.play_audio(samples)
                # Wait for playback to finish or timeout
                try:
                    duration = len(samples) / self.config.spk_sample_rate
                    time.sleep(duration * 0.85)  # Start next batch before this one ends
                except Exception:
                    pass
            except Exception:
                logger.exception("Playback loop error")
                time.sleep(0.1)


def build_config_from_args(args: argparse.Namespace) -> SpokeConfig:
    """Build a SpokeConfig from CLI arguments."""
    cfg = SpokeConfig(
        hub_host=args.hub,
        hub_port=args.port,
        reconnect_base_delay=args.reconnect_base,
        reconnect_max_delay=args.reconnect_max,
        mic_device=args.mic_device,
        spk_device=args.speaker_device,
        echo_aec_enabled=args.aec,
        echo_aec_order=args.aec_order,
        echo_text_similarity_threshold=args.echo_threshold,
        native_audio=args.native_audio,
        jitter_buffer_min_ms=args.jitter_min,
        jitter_buffer_max_ms=args.jitter_max,
    )
    return cfg


def main() -> None:
    """CLI entry point for the GLaDOS spoke client."""
    ap = argparse.ArgumentParser(description="GLaDOS Spoke Client")
    ap.add_argument("--hub", default=None, help="Hub hostname or IP")
    ap.add_argument("--port", type=int, default=None, help="Hub WebSocket port")
    ap.add_argument("--config", default=None, help="Path to YAML config file")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    ap.add_argument("--mic-device", type=int, default=None, help="Mic device index")
    ap.add_argument("--speaker-device", type=int, default=None, help="Speaker device index")
    ap.add_argument("--aec", action="store_true", help="Enable AEC")
    ap.add_argument("--aec-order", type=int, default=256, help="AEC filter order")
    ap.add_argument("--echo-threshold", type=float, default=0.7, help="Echo detection threshold")
    ap.add_argument("--native-audio", action="store_true", default=True, help="Use native audio mode")
    ap.add_argument("--reconnect-base", type=float, default=1.0, help="Reconnect base delay")
    ap.add_argument("--reconnect-max", type=float, default=30.0, help="Reconnect max delay")
    ap.add_argument("--jitter-min", type=int, default=80, help="Min jitter buffer (ms)")
    ap.add_argument("--jitter-max", type=int, default=400, help="Max jitter buffer (ms)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    cfg = SpokeConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        if data and "spoke" in data:
            cfg = SpokeConfig(**data["spoke"])
    # CLI args override config file
    if args.hub:
        cfg.hub_host = args.hub
    if args.port:
        cfg.hub_port = args.port
    if args.mic_device is not None:
        cfg.mic_device = args.mic_device
    if args.speaker_device is not None:
        cfg.spk_device = args.speaker_device
    if args.aec:
        cfg.echo_aec_enabled = True
    if args.aec_order:
        cfg.echo_aec_order = args.aec_order
    if args.echo_threshold:
        cfg.echo_text_similarity_threshold = args.echo_threshold
    if args.reconnect_base:
        cfg.reconnect_base_delay = args.reconnect_base
    if args.reconnect_max:
        cfg.reconnect_max_delay = args.reconnect_max
    if args.jitter_min:
        cfg.jitter_buffer_min_ms = args.jitter_min
    if args.jitter_max:
        cfg.jitter_buffer_max_ms = args.jitter_max

    async def _run() -> None:
        client = SpokeClient(cfg)
        try:
            await client.start()
        except Exception:
            logger.exception("Spoke client crashed")
        finally:
            await client.stop()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
