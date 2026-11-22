"""
GLaDOS Python Spoke - XVF3800 (arecord/aplay) <-> hub WebSocket.

Pushes 16 kHz mono int16 PCM to the hub in realtime (0x01 frames, 100 ms).
Plays streamed TTS audio (0x11 frames, 24 kHz int16) as it arrives, with
low-latency chunked playback. 0x12 = stop playback immediately (barge-in).
"""
import asyncio
import logging
import os
import queue
import time

import numpy as np
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("glados_spoke")

HUB_URL = os.environ.get("GLADOS_HUB_URL", "ws://localhost:5051")
DEVICE_ID = os.environ.get("GLADOS_DEVICE_ID", "spoke-001")
# Device selection (set by setup_audio.py, or directly for manual runs)
DEFAULT_DEVICE = "plughw:CARD=wm8960soundcard,DEV=0"
CAPTURE_DEVICE = os.environ.get("GLADOS_CAPTURE_DEVICE") or os.environ.get("GLADOS_AUDIO_DEVICE", DEFAULT_DEVICE)
PLAYBACK_DEVICE = os.environ.get("GLADOS_PLAYBACK_DEVICE") or os.environ.get("GLADOS_AUDIO_DEVICE", DEFAULT_DEVICE)
CHUNK_SIZE = 3200  # 100 ms @ 16 kHz, int16
TTS_CHUNK_BYTES = int(24000 * 0.2 * 2)  # 200 ms @ 24 kHz
MAX_PEAK = 20000  # ~-4 dBFS peak limit (Piper outputs full-scale)
# Aplay is kept alive and warm (short silences while idle) so the ALSA/USB
# device never closes: re-opening it per reply is what causes the speaker
# pop and the swallowed first second on this card.
TTS_KEEPALIVE_S = 1.0
# Data gap (seconds) that marks the start of a new speech burst -> fade-in.
TTS_BURST_GAP_S = 0.35
FADE_SAMPLES = 720  # 30 ms @ 24 kHz linear ramp
SILENCE_100MS = b"\x00\x00" * (24000 // 10)  # 100 ms @ 24 kHz s16le


class Playback:
    """Streams 24 kHz int16 PCM to ONE long-lived aplay process.

    aplay is pre-warmed at startup and kept warm with periodic silence, so
    the ALSA/USB device stays open and clocked. Re-opening the device for
    every reply is what caused the speaker pop and the dropped first second.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.proc = None
        self._write_lock = None  # created lazily: asyncio.Lock binds to the loop
        self._last_tts = 0.0  # monotonic time of last TTS data written
        self._burst_active = False

    def _lock(self) -> asyncio.Lock:
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        return self._write_lock

    async def start(self):
        """Pre-warm aplay + ALSA so the first word has zero open latency."""
        try:
            await self._ensure_proc()
        except Exception as e:
            logger.warning("[PLAYBACK] pre-warm failed (%s); keepalive will retry", e)
            self.proc = None

    async def _ensure_proc(self):
        # asyncio.subprocess.Process exposes .returncode, not .poll()
        if self.proc is None or self.proc.returncode is not None:
            if self.proc is not None:
                try:
                    await self.proc.wait()
                except Exception:
                    pass
            self.proc = await asyncio.create_subprocess_exec(
                "aplay", "-f", "S16_LE", "-r", "24000", "-c", "1", "-D", PLAYBACK_DEVICE,
                stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info("[PLAYBACK] aplay ready on %s (pid %d)", PLAYBACK_DEVICE, self.proc.pid)
        return self.proc

    def _fade_in(self, arr: np.ndarray) -> np.ndarray:
        """Ramp the first 30 ms of a new burst up from silence (kills the pop)."""
        now = time.monotonic()
        if (not self._burst_active) or (now - self._last_tts > TTS_BURST_GAP_S):
            n = min(FADE_SAMPLES, len(arr))
            if n:
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                arr = arr.astype(np.float32)
                arr[:n] = arr[:n] * ramp
                arr = arr.astype(np.int16)
            self._burst_active = True
        self._last_tts = now
        return arr

    async def _write_frame(self, frame: bytes, retries: int = 6):
        """Write one slice to aplay, respawning it if needed.

        The SAME frame is retried (never silently dropped) so a slow ALSA
        open or a transient device-busy can't eat the start of a sentence.
        """
        last_err = None
        for attempt in range(retries):
            try:
                async with self._lock():
                    proc = await self._ensure_proc()
                    proc.stdin.write(frame)
                    await proc.stdin.drain()
                return
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                last_err = e
                self.proc = None
                await asyncio.sleep(0.15 * (attempt + 1))
        raise last_err

    async def write(self, data: bytes):
        """Buffer TTS PCM and write 200 ms slices to aplay at 24 kHz.

        Piper outputs hot (full-scale peaks), so we peak-limit to ~-4 dBFS
        to keep the amp from clipping/distorting.
        """
        self.buffer.extend(data)
        while len(self.buffer) >= TTS_CHUNK_BYTES:
            frame = bytes(self.buffer[:TTS_CHUNK_BYTES])
            del self.buffer[:TTS_CHUNK_BYTES]
            try:
                arr = np.frombuffer(frame, dtype=np.int16)
                peak = int(np.max(np.abs(arr))) if len(arr) else 0
                if peak > MAX_PEAK:
                    arr = (arr * (MAX_PEAK / peak)).astype(np.int16)
                frame = self._fade_in(arr).tobytes()
                await self._write_frame(frame)
            except Exception as e:
                logger.warning("[PLAYBACK] dropped frame after retries: %s", e)

    async def keepalive(self):
        """Keep aplay + the USB audio device warm while idle.

        A short silence at most once a second keeps ALSA open and the codec
        clocked, so the next word starts instantly with no wake-up pop.
        Also respawns aplay (killed by barge-in) as soon as possible.
        """
        while True:
            await asyncio.sleep(TTS_KEEPALIVE_S)
            try:
                proc = self.proc
                if proc is None or proc.returncode is not None:
                    await self._ensure_proc()
                elif time.monotonic() - self._last_tts > TTS_KEEPALIVE_S * 1.5:
                    async with self._lock():
                        proc.stdin.write(SILENCE_100MS)
                        await proc.stdin.drain()
            except Exception as e:
                logger.debug("[PLAYBACK] keepalive hiccup: %s", e)
                self.proc = None

    async def stop(self):
        """Stop playback immediately and flush the buffer (barge-in)."""
        self.buffer.clear()
        self._burst_active = False  # next burst fades in fresh
        if self.proc is not None and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=1)
            except Exception:
                pass
        self.proc = None


playback = Playback()


async def play_audio(audio_queue: asyncio.Queue):
    while True:
        frame_type, data = await audio_queue.get()
        if frame_type == 0x11:
            await playback.write(data)
        elif frame_type == 0x12:
            logger.info("[PLAYBACK] stop (barge-in)")
            await playback.stop()
        elif frame_type == 0x13:
            logger.info("[STATUS] %s", data.rstrip(b"\x00").decode("utf-8", errors="ignore"))


async def receive_audio(ws, audio_queue: asyncio.Queue):
    while True:
        try:
            data = await asyncio.wait_for(ws.recv(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("[RECV] No data from hub in 15s. Connection dead?")
            raise websockets.exceptions.ConnectionClosed(1001, "Hub silent 15s")
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.error("[RECV] Connection lost: %s", e)
            raise
        if not data:
            continue
        frame_type, payload = data[0], data[1:]
        if frame_type == 0x14:  # hub heartbeat
            continue
        if frame_type in (0x11, 0x12, 0x13):
            audio_queue.put_nowait((frame_type, payload))


async def capture_audio(audio_tx_queue: queue.Queue):
    """Capture 16 kHz mono int16 and push 100 ms chunks in realtime."""
    proc = await asyncio.create_subprocess_exec(
        "arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-D", CAPTURE_DEVICE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    logger.info("arecord started: %d", proc.pid)
    buf = bytearray()
    chunk_count = 0
    try:
        while True:
            data = await proc.stdout.read(4000)
            if not data:
                continue
            buf.extend(data)
            while len(buf) >= CHUNK_SIZE:
                chunk = bytes(buf[:CHUNK_SIZE])
                del buf[:CHUNK_SIZE]
                chunk_count += 1
                if chunk_count % 20 == 0:
                    rms = np.sqrt(np.mean(np.frombuffer(chunk, dtype=np.int16).astype(np.float64)**2))
                    logger.info("[LISTENING] %.2f RMS (chunk %d)", rms, chunk_count)
                try:
                    audio_tx_queue.put_nowait(bytes([0x01]) + chunk)
                except queue.Full:
                    pass  # hub not keeping up; drop rather than block capture
    finally:
        proc.terminate()
        await proc.wait()


async def send_audio(ws, audio_tx_queue: queue.Queue):
    loop = asyncio.get_event_loop()
    while True:
        frame = await loop.run_in_executor(None, audio_tx_queue.get)
        await ws.send(frame)


async def main():
    logger.info("GLaDOS Spoke starting... Hub=%s Device=%s", HUB_URL, DEVICE_ID)
    retry = 1
    while True:
        tasks = []
        audio_queue = asyncio.Queue()
        audio_tx_queue = queue.Queue()
        try:
            async with websockets.connect(
                f"{HUB_URL}/ws/{DEVICE_ID}",
                ping_interval=2,
                ping_timeout=1,
                max_size=2**20,  # TTS audio frames can be large
            ) as ws:
                logger.info("Connected!")
                retry = 1
                play_task = asyncio.create_task(play_audio(audio_queue))
                keep_task = asyncio.create_task(playback.keepalive())
                cap = asyncio.create_task(capture_audio(audio_tx_queue))
                snd = asyncio.create_task(send_audio(ws, audio_tx_queue))
                rcv = asyncio.create_task(receive_audio(ws, audio_queue))
                tasks = [play_task, keep_task, cap, snd, rcv]
                asyncio.create_task(playback.start())  # pre-warm aplay/ALSA now
                await ws.send(bytes([0x03]))  # session start
                logger.info("Listening...")
                while True:
                    for t in tasks:
                        if t.done() and not t.cancelled():
                            raise t.exception() or RuntimeError("spoke task exited")
                    await asyncio.sleep(0.1)
        except (ConnectionRefusedError, OSError, websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosed, asyncio.CancelledError) as e:
            logger.error("Lost connection: %s. Reconnecting in %ds...", e, retry)
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await playback.stop()
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)
        except Exception as e:
            # Any unexpected error: log it and reconnect — never die.
            logger.exception("Unexpected spoke error: %s. Reconnecting in %ds...", e, retry)
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await playback.stop()
            await asyncio.sleep(retry)
            retry = min(retry * 2, 30)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bye")
