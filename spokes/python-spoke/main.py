import asyncio, logging, os, queue
import numpy as np
import websockets

try:
    import pyrnnoise
    NS_AVAILABLE = True
except ImportError:
    NS_AVAILABLE = False
    logging.warning("pyrnnoise not available - no noise suppression")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("glados_spoke")

HUB_URL = os.environ.get("GLADOS_HUB_URL", "ws://localhost:5051")
DEVICE_ID = os.environ.get("GLADOS_DEVICE_ID", "spoke-001")
CHUNK_SIZE = 3200  # 1600 samples * 2 bytes

# ── Noise suppressor (disabled - API issues) ────────────────────────
# pyrnnoise API compatibility issues on ARM64
# Re-enable when a working version is found
_NS_AVAILABLE = False
_ns = None

# ── ALSA device configuration ──────────────────────────────────────
# These can be overridden via environment variables, otherwise
# setup_audio.py configures /etc/asound.conf to set defaults.

CAPTURE_DEVICE = os.environ.get("GLADOS_CAPTURE_DEVICE", "default")
logger.info("Audio config: capture=%s", CAPTURE_DEVICE)


import subprocess
import os
import time

AUDIO_PIPE = "/tmp/audio_fifo"

def _playback_worker(audio_queue):
    """Background process for TTS playback using a FIFO pipe to aplay."""
    # Create FIFO if it doesn't exist
    if not os.path.exists(AUDIO_PIPE):
        os.mkfifo(AUDIO_PIPE)
    
    # Start aplay reading from FIFO
    aplay = subprocess.Popen(
        ["aplay", "-f", "S16_LE", "-r", "22050", "-D", "plughw:3,0"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    
    buffer = bytearray()
    while True:
        try:
            frame_type, data = audio_queue.get(timeout=1)
        except Exception:
            continue
        if frame_type == 0x11:
            buffer.extend(data)
            while len(buffer) >= TTS_CHUNK:
                frame = bytes(buffer[:TTS_CHUNK])
                buffer = buffer[TTS_CHUNK:]
                # Hub sends 22050 Hz, aplay expects 22050 Hz - no resampling needed
                out = np.frombuffer(frame, dtype=np.int16)
                # Non-blocking write with retry
                written = 0
                while written < len(out):
                    try:
                        n = aplay.stdin.write(out[written:])
                        written += n if n else len(out[written:written+1])
                        if n == 0:
                            time.sleep(0.01)
                    except BlockingIOError:
                        time.sleep(0.01)
                    except Exception:
                        break
        elif frame_type == 0x13:
            pass
        elif frame_type == 0x12:
            break
    
    try:
        aplay.stdin.close()
        aplay.wait(timeout=2)
    except Exception:
        aplay.kill()


async def play_audio():
    """Start playback worker process and wait for it to finish."""
    # Create a multiprocessing Queue for audio data
    mp_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_playback_worker, args=(mp_queue,), daemon=True)
    process.start()
    logger.info("[PLAYBACK] Process started (PID: %d)", process.pid)
    
    # Relay from asyncio queue to multiprocessing queue
    try:
        while True:
            frame_type, data = await audio_queue.get()
            mp_queue.put_nowait((frame_type, data))
    except asyncio.CancelledError:
        pass
    finally:
        # Signal stop
        mp_queue.put_nowait((0x12, b''))
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
        logger.info("[PLAYBACK] Process stopped")


async def receive_audio(ws):
    """Receive WebSocket frames and route to queues."""
    count = 0
    while True:
        try:
            data = await ws.recv()
            count += 1
            if count <= 5:
                logger.info("[WS] Frame #%d type=0x%02x len=%d", count, data[0], len(data)-1)
            frame_type = data[0]
            payload = data[1:]
            if frame_type == 0x11:
                audio_queue.put_nowait((0x11, payload))
            elif frame_type in (0x12, 0x13):
                audio_queue.put_nowait((frame_type, payload))
        except Exception as e:
            logger.error("[WS] recv error: %s", e)
            break


async def capture_audio(ws):
    """Capture audio using arecord, apply noise suppression."""
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
                
                # Apply noise suppression if available
                if _ns is not None:
                    try:
                        raw = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        cleaned = _ns.process(raw)
                        chunk = (cleaned * 32768.0).astype(np.int16).tobytes()
                    except Exception as e:
                        logger.error("[NS] Error: %s, sending raw", e)
                
                rms = np.sqrt(np.mean(np.frombuffer(chunk, dtype=np.int16).astype(np.float64)**2))
                chunk_count += 1
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


async def main():
    logger.info("GLaDOS Spoke starting... Hub=%s Device=%s", HUB_URL, DEVICE_ID)
    retry = 1
    cap = snd = rcv = play_task = None
    while True:
        try:
            async with websockets.connect(f"{HUB_URL}/ws/{DEVICE_ID}", ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                logger.info("Connected!")
                retry = 1
                play_task = asyncio.create_task(play_audio())
                cap = asyncio.create_task(capture_audio(ws))
                snd = asyncio.create_task(send_audio(ws))
                rcv = asyncio.create_task(receive_audio(ws))
                await ws.send(bytes([0x03]))
                logger.info("Listening...")
                while True:
                    await asyncio.sleep(0.1)
        except (ConnectionRefusedError, OSError, websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode) as e:
            logger.error("Lost connection: %s. Retry %ds", e, retry)
            if play_task:
                play_task.cancel()
            for t in [cap, snd, rcv]:
                try:
                    t.cancel()
                except Exception:
                    pass
            while not audio_tx_queue.empty():
                try:
                    audio_tx_queue.get_nowait()
                except queue.Empty:
                    break
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)
        except Exception as e:
            logger.error("Unexpected error: %s. Retry %ds", e, retry)
            if play_task:
                play_task.cancel()
            for t in [cap, snd, rcv]:
                try:
                    t.cancel()
                except Exception:
                    pass
            while not audio_tx_queue.empty():
                try:
                    audio_tx_queue.get_nowait()
                except queue.Empty:
                    break
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
