import asyncio, logging, os, queue
import numpy as np
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("glados_spoke")

HUB_URL = os.environ.get("GLADOS_HUB_URL", "ws://localhost:5051")
DEVICE_ID = os.environ.get("GLADOS_DEVICE_ID", "spoke-001")
CHUNK_SIZE = 3200  # 1600 samples * 2 bytes

# ── ALSA device configuration ──────────────────────────────────────
# These can be overridden via environment variables, otherwise
# setup_audio.py configures /etc/asound.conf to set defaults.

CAPTURE_DEVICE = os.environ.get("GLADOS_CAPTURE_DEVICE", "default")
PLAYBACK_DEVICE = os.environ.get("GLADOS_PLAYBACK_DEVICE", "default")

logger.info("Audio config: capture=%s playback=%s", CAPTURE_DEVICE, PLAYBACK_DEVICE)


async def play_audio():
    """Play TTS audio using aplay subprocess."""
    buffer = bytearray()
    while True:
        frame_type, data = await audio_queue.get()
        if frame_type == 0x11:
            buffer.extend(data)
            logger.info("[PLAYBACK] Got %d bytes (total: %d)", len(data), len(buffer))
            while len(buffer) >= TTS_CHUNK:
                frame = bytes(buffer[:TTS_CHUNK])
                buffer = buffer[TTS_CHUNK:]
                audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                n = int(len(audio) * 16000 / 24000)
                idx = np.linspace(0, len(audio) - 1, n)
                out = (np.interp(idx, np.arange(len(audio)), audio) * 32767).astype(np.int16).tobytes()
                if playback_proc and playback_proc.poll() is None:
                    playback_proc.stdin.write(out)
                else:
                    if playback_proc:
                        try:
                            playback_proc.wait(timeout=1)
                        except Exception:
                            pass
                    playback_proc = await asyncio.create_subprocess_exec(
                        "aplay", "-f", "S16_LE", "-r", "16000",
                        "-D", PLAYBACK_DEVICE,
                        stdin=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    playback_proc.stdin.write(out)
        elif frame_type == 0x13:
            logger.info("[STATUS] %s", data.rstrip(b'\x00').decode('utf-8', errors='ignore'))
        elif frame_type == 0x12:
            logger.info("[CONTROL] session ended")


async def receive_audio(ws):
    """Receive WebSocket frames and route to queues."""
    while True:
        data = await ws.recv()
        frame_type = data[0]
        payload = data[1:]
        if frame_type == 0x11:
            audio_queue.put_nowait((0x11, payload))
        elif frame_type in (0x12, 0x13):
            audio_queue.put_nowait((frame_type, payload))


async def capture_audio(ws):
    """Capture audio using arecord."""
    proc = await asyncio.create_subprocess_exec(
        "arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw",
        "-D", CAPTURE_DEVICE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )
    logger.info("arecord started (%d): %s", proc.pid, CAPTURE_DEVICE)
    buf = bytearray()
    chunk_count = 0
    try:
        while True:
            data = await asyncio.wait_for(proc.stdout.read(4000), timeout=2.0)
            if not data:
                continue
            buf.extend(data)
            while len(buf) >= CHUNK_SIZE:
                chunk = bytes(buf[:CHUNK_SIZE])
                buf = buf[CHUNK_SIZE:]
                rms = np.sqrt(np.mean(np.frombuffer(chunk, dtype=np.int16).astype(np.float64)**2))
                chunk_count += 1
                if chunk_count % 20 == 0:
                    logger.info("[LISTENING] %.2f RMS (%d)", rms, chunk_count)
                try:
                    audio_tx_queue.put_nowait(bytes([0x01]) + chunk)
                except queue.Full:
                    pass
    except asyncio.CancelledError:
        pass
    finally:
        proc.terminate()
        await proc.wait()


async def send_audio(ws):
    """Send audio to hub."""
    while True:
        frame = await asyncio.get_event_loop().run_in_executor(None, audio_tx_queue.get)
        await ws.send(frame)


TTS_CHUNK = int(24000 * 0.2 * 2)
audio_queue = asyncio.Queue()
audio_tx_queue = queue.Queue()
playback_proc = None


async def main():
    logger.info("GLaDOS Spoke starting... Hub=%s Device=%s", HUB_URL, DEVICE_ID)
    play_task = asyncio.create_task(play_audio())
    retry = 1
    cap = snd = rcv = None
    while True:
        try:
            async with websockets.connect(f"{HUB_URL}/ws/{DEVICE_ID}") as ws:
                logger.info("Connected!")
                retry = 1
                cap = asyncio.create_task(capture_audio(ws))
                snd = asyncio.create_task(send_audio(ws))
                rcv = asyncio.create_task(receive_audio(ws))
                await ws.send(bytes([0x03]))
                logger.info("Listening...")
                while True:
                    await asyncio.sleep(0.1)
        except (ConnectionRefusedError, OSError, websockets.exceptions.ConnectionClosedError) as e:
            logger.error("Lost: %s. Retry %ds", e, retry)
            play_task.cancel()
            for t in [cap, snd, rcv]:
                try:
                    t.cancel()
                except Exception:
                    pass
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)
            play_task = asyncio.create_task(play_audio())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
