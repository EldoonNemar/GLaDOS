"""
GLaDOS Python Spoke - connects a reSpeaker XVF3800 to the GLaDOS hub.

Requires: reSpeaker XVF3800 USB mic + speakers connected to host.
Connects to ws://hub:5051/ws/{device_id}
"""

import asyncio
import base64
import json
import logging
import os
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("glados_spoke")

# Config
HUB_URL = os.environ.get("GLADOS_HUB_URL", "ws://localhost:5051")
DEVICE_ID = os.environ.get("GLADOS_DEVICE_ID", "spoke-001")
AUDIO_DEVICE = int(os.environ.get("AUDIO_DEVICE", -1))
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100ms

# Audio playback
PLAYBACK_SR = 22050
audio_queue = asyncio.Queue()
audio_tx_queue = queue.Queue()


async def play_audio():
    """Background task: play audio from the queue."""
    buffer = bytearray()
    while True:
        frame_type, chunk = await audio_queue.get()
        if frame_type == 0x11:
            buffer.extend(chunk)
            while len(buffer) >= PLAYBACK_SR * 0.2 * 2:
                play_chunk = bytes(buffer[:int(PLAYBACK_SR * 0.2 * 2)])
                audio_arr = np.frombuffer(play_chunk, dtype=np.int16)
                sd.play(audio_arr, PLAYBACK_SR)
                sd.wait()
                buffer = buffer[int(PLAYBACK_SR * 0.2 * 2):]
        elif frame_type == 0x13:
            text = chunk.rstrip(b'\x00').decode('utf-8', errors='ignore')
            logger.info("[STATUS] %s", text)
        elif frame_type == 0x12:
            logger.info("[CONTROL] session ended")


async def send_audio(ws):
    """Read from sync queue and send via websocket."""
    while True:
        frame = await asyncio.get_event_loop().run_in_executor(None, audio_tx_queue.get)
        await ws.send(frame)


async def capture_audio(ws):
    """Capture audio from reSpeaker XVF3800 and stream to hub."""
    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)
        try:
            audio_tx_queue.put_nowait(bytes([0x01]) + indata.tobytes())
        except queue.Full:
            pass

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        device=AUDIO_DEVICE if AUDIO_DEVICE >= 0 else None,
                        callback=audio_callback, blocksize=CHUNK_SAMPLES):
        logger.info("Audio capture started (device=%s, %d Hz)", AUDIO_DEVICE, SAMPLE_RATE)
        while True:
            await asyncio.sleep(0.1)


async def main():
    logger.info("GLaDOS Spoke starting...")
    logger.info("Hub URL: %s", HUB_URL)
    logger.info("Device ID: %s", DEVICE_ID)
    logger.info("Audio device: %s", AUDIO_DEVICE)

    play_task = asyncio.create_task(play_audio())

    retry_delay = 1
    while True:
        try:
            async with websockets.connect(f"{HUB_URL}/ws/{DEVICE_ID}") as ws:
                logger.info("Connected to hub!")
                retry_delay = 1

                capture_task = asyncio.create_task(capture_audio(ws))
                send_task = asyncio.create_task(send_audio(ws))

                await ws.send(bytes([0x03]))
                logger.info("Session started")

                while True:
                    await asyncio.sleep(0.1)

        except (ConnectionRefusedError, OSError, websockets.exceptions.ConnectionClosed) as e:
            logger.error("Connection lost: %s. Retrying in %ds...", e, retry_delay)
            play_task.cancel()
            try: capture_task.cancel()
            except: pass
            try: send_task.cancel()
            except: pass
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
            play_task = asyncio.create_task(play_audio())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
