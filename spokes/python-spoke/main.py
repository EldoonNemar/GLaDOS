import asyncio, base64, json, logging, os, queue, signal, subprocess, sys
from pathlib import Path
import numpy as np
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("glados_spoke")

HUB_URL = os.environ.get("GLADOS_HUB_URL", "ws://localhost:5051")
DEVICE_ID = os.environ.get("GLADOS_DEVICE_ID", "spoke-001")
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600
PLAYBACK_SR = 16000
TTS_SR = 24000
audio_queue = asyncio.Queue()
audio_tx_queue = queue.Queue()
chunk_count = 0
capture_proc = None
playback_proc = None

async def play_audio():
    global playback_proc
    buffer = bytearray()
    while True:
        frame_type, chunk = await audio_queue.get()
        if frame_type == 0x11:
            buffer.extend(chunk)
            logger.info("[PLAYBACK] Got %d bytes (buf: %d)", len(chunk), len(buffer))
            while len(buffer) >= TTS_SR * 0.2 * 2:
                data = bytes(buffer[:int(TTS_SR * 0.2 * 2)])
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                n_new = int(len(audio) * 16000 / 24000)
                idx = np.linspace(0, len(audio) - 1, n_new)
                resampled = (np.interp(idx, np.arange(len(audio)), audio) * 32767).astype(np.int16).tobytes()
                if playback_proc and playback_proc.poll() is None:
                    playback_proc.stdin.write(resampled)
                else:
                    if playback_proc:
                        try: playback_proc.wait(timeout=1)
                        except: pass
                    playback_proc = await asyncio.create_subprocess_exec(
                        "aplay", "-f", "S16_LE", "-r", "16000", "-D", "plughw:4,0",
                        stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
                    )
                    playback_proc.stdin.write(resampled)
            buffer = buffer[int(TTS_SR * 0.2 * 2):]
        elif frame_type == 0x13:
            text = chunk.rstrip(b'\x00').decode('utf-8', errors='ignore')
            logger.info("[STATUS] %s", text)

async def capture_audio(ws):
    global capture_proc
    capture_proc = await asyncio.create_subprocess_exec(
        "arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-D", "plughw:4,0",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    logger.info("arecord started")
    cs = CHUNK_SAMPLES * 2
    try:
        while True:
            data = await asyncio.wait_for(capture_proc.stdout.read(cs), timeout=1.0)
            if len(data) < cs:
                continue
            rms = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float64)**2))
            chunk_count += 1
            if chunk_count % 20 == 0:
                logger.info("[LISTENING] %.2f RMS (%d)", rms, chunk_count)
            try:
                audio_tx_queue.put_nowait(bytes([0x01]) + data)
            except queue.Full:
                pass
    except asyncio.CancelledError:
        pass
    finally:
        capture_proc.terminate()
        await capture_proc.wait()

async def send_audio(ws):
    while True:
        frame = await asyncio.get_event_loop().run_in_executor(None, audio_tx_queue.get)
        await ws.send(frame)

async def main():
    logger.info("GLaDOS Spoke starting... Hub=%s Device=%s", HUB_URL, DEVICE_ID)
    play_task = asyncio.create_task(play_audio())
    retry = 1
    while True:
        try:
            async with websockets.connect(f"{HUB_URL}/ws/{DEVICE_ID}") as ws:
                logger.info("Connected!")
                retry = 1
                cap = asyncio.create_task(capture_audio(ws))
                snd = asyncio.create_task(send_audio(ws))
                await ws.send(bytes([0x03]))
                logger.info("Listening...")
                while True:
                    await asyncio.sleep(0.1)
        except (ConnectionRefusedError, OSError, websockets.exceptions.ConnectionClosed) as e:
            logger.error("Lost: %s. Retry in %ds", e, retry)
            play_task.cancel()
            for t in [cap, snd]:
                try: t.cancel()
                except: pass
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)
            play_task = asyncio.create_task(play_audio())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
