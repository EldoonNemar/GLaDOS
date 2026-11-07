"""CLI entry point for glados-spoke."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from glados_spoke.client import SpokeClient
from glados_spoke.config import SpokeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLaDOS Spoke Client")
    parser.add_argument("--hub", type=str, default=None, help="Hub hostname or IP")
    parser.add_argument("--port", type=int, default=None, help="Hub WebSocket port (default: 5051)")
    parser.add_argument("--config", type=str, default=None, help="Path to spoke_config.yaml")
    parser.add_argument("--aec", action="store_true", help="Enable acoustic echo cancellation")
    parser.add_argument("--mic-device", type=str, default=None, help="Mic device index")
    parser.add_argument("--speaker-device", type=int, default=None, help="Speaker device index")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = args.config or "spoke_config.yaml"
    config = SpokeConfig.from_file(config_path)

    # Apply CLI overrides
    if args.hub:
        config.hub_host = args.hub
    if args.port:
        config.hub_port = args.port
    if args.aec:
        config.echo_aec_enabled = True
    if args.mic_device is not None:
        config.mic_device = args.mic_device
    if args.speaker_device is not None:
        config.spk_device = args.speaker_device

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("glados_spoke")
    logger.info("Starting spoke client -> %s:%d", config.hub_host, config.hub_port)

    # Run the client
    client = SpokeClient(config)
    try:
        async def _run():
            await client.start()
            await client._wait_for_shutdown()
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        asyncio.run(client.stop())


if __name__ == "__main__":
    main()
