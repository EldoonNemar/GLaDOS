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
PLAYBACK_SR = 16000
TTS_SR = 24000
audio_queue = asyncio.Queue()
audio_tx_queue = queue.Queue()

# Stats
chunk_count = 0
last_chunk_log = 0


async def play_audio():
    """Background task: play audio from the queue."""
    global chunk_count
    buffer = bytearray()
    play_dev = 0  # XVF3800 device index
    
    logger.info("Playback device: %d (%s)", play_dev, sd.query_devices()[play_dev]["name"])
    
    while True:
        frame_type, chunk = await audio_queue.get()
        if frame_type == 0x11:
            buffer.extend(chunk)
            if chunk_count % 5 == 0:
                logger.info("[PLAYBACK] Received %d bytes TTS audio", len(chunk))
            while len(buffer) >= TTS_SR * 0.1 * 2:
                # Extract 100ms of TTS audio (24kHz)
                frame_data = bytes(buffer[:int(TTS_SR * 0.1 * 2)])
                audio = np.frombuffer(frame_data, dtype=np.int16).astype(np.float32)
                # Resample from 24kHz -> 16kHz for XVF3800
                n_new = int(len(audio) * 16000 / 24000)
                indices = np.linspace(0, len(audio) - 1, n_new)
                resampled = np.interp(indices, np.arange(len(audio)), audio)
                # Play at 16kHz on XVF3800
                sd.play((resampled * 32767).astype(np.int16), 16000, device=play_dev)
                sd.wait()
                buffer = buffer[int(TTS_SR * 0.1 * 2):]
        elif frame_type == 0x13:
            text = chunk.rstrip(b'\x00').decode('utf-8', errors='ignore')
            logger.info("[STATUS] %s", text)
        elif frame_type == 0x12:
            logger.info("[CONTROL] session ended")


async def send_audio(ws):
    """Read from sync queue and send via websocket."""
    global chunk_count, last_chunk_log
    while True:
        frame = await asyncio.get_event_loop().run_in_executor(None, audio_tx_queue.get)
        await ws.send(frame)
        chunk_count += 1
        if chunk_count % 10 == 0:
            # Log every 1 second of audio (~10 chunks)
            logger.debug("Sent %d audio chunks to hub", chunk_count)


async def capture_audio(ws):
    """Capture audio from reSpeaker XVF3800 and stream to hub."""
    # XVF3800 is device [0] with 4 mic channels at 16kHz.
    # Open all 4 channels, mix to mono in callback.
    devices_to_try = [0]
    
    logger.info("Available audio devices:")
    for idx, dev in enumerate(sd.query_devices()):
        logger.info("  [%d] %s (in: %s, out: %s, rate=%.0fHz, ch=%d)", idx, dev["name"],
                   "✓" if dev["max_input_channels"] > 0 else "✗",
                   "✓" if dev["max_output_channels"] > 0 else "✗",
                   dev["default_samplerate"],
                   dev["max_input_channels"])

    def audio_callback(indata, frames, time_info, status):
        global chunk_count, last_chunk_log
        if status:
            logger.warning("Audio status: %s", status)
        # indata shape: (frames, channels) - mix 4 channels to 1
        if indata.ndim > 1:
            mono = np.mean(indata, axis=1).astype(np.int16)
        else:
            mono = indata
        rms = np.sqrt(np.mean(mono.astype(np.float32)**2))
        chunk_count += 1
        if chunk_count % 20 == 0:
            logger.info("[LISTENING] Audio level: %.4f RMS (%d chunks)", rms, chunk_count)
        try:
            audio_tx_queue.put_nowait(bytes([0x01]) + mono.tobytes())
        except queue.Full:
            pass

    for dev in devices_to_try:
        try:
            dev_info = sd.query_devices()[dev]
            n_ch = dev_info["max_input_channels"]
            logger.info("Opening device %d (%s) with %d channels at %d Hz...", dev, dev_info["name"], n_ch, SAMPLE_RATE)
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=n_ch, dtype='int16',
                                device=dev,
                                callback=audio_callback, blocksize=CHUNK_SAMPLES):
                logger.info("Audio capture started on device %d at %d Hz (%d ch -> mono)", dev, SAMPLE_RATE, n_ch)
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning("Device %d failed: %s", dev, e)
    
    logger.error("All devices failed. Spoke cannot capture audio.")


async def list_devices():
    """List all available audio devices and exit."""
    logger.info("Available audio devices:")
    for idx, dev in enumerate(sd.query_devices()):
        direction = []
        if dev["max_input_channels"] > 0:
            direction.append("IN")
        if dev["max_output_channels"] > 0:
            direction.append("OUT")
        logger.info("  [%d] %s [%s] (rate=%.0fHz)", idx, dev["name"],
                   ", ".join(direction), dev["default_samplerate"])
    sys.exit(0)


async def test_audio(device_idx=None, duration=5):
    """Record audio from device and play it back to verify."""
    if device_idx is None:
        device_idx = AUDIO_DEVICE if AUDIO_DEVICE >= 0 else None
    logger.info("Recording %d seconds from device %s...", duration, device_idx)
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1,
                   dtype='int16', device=device_idx)
    sd.wait()
    logger.info("Recording complete. Playing back...")
    # Print audio stats
    rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
    logger.info("Recording RMS: %.4f (max: %.4f)", rms, np.max(np.abs(audio.astype(np.float64))/32768.0))
    if rms < 0.001:
        logger.warning("Recording is nearly silent! The wrong device may be selected.")
        logger.warning("Run: %s --devices", sys.argv[0] if sys.argv else "python main.py")
        logger.warning("Then set AUDIO_DEVICE=<index> in environment")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    logger.info("Playback complete.")
    sys.exit(0)


async def main():
    logger.info("GLaDOS Spoke starting...")
    logger.info("Hub URL: %s", HUB_URL)
    logger.info("Device ID: %s", DEVICE_ID)
    logger.info("Audio device: %s (use AUDIO_DEVICE=<idx> to change)", AUDIO_DEVICE)

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
                logger.info("Session started — listening for audio...")

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
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--devices":
            asyncio.run(list_devices())
        elif cmd == "--test" and len(sys.argv) > 2:
            asyncio.run(test_audio(int(sys.argv[2])))
        elif cmd == "--test":
            asyncio.run(test_audio(AUDIO_DEVICE if AUDIO_DEVICE >= 0 else None))
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python main.py [--devices | --test [device_idx]]")
            sys.exit(1)
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
