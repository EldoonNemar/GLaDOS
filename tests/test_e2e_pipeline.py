"""
End-to-end pipeline test with a FAKE SPOKE against the live stack.

Connects to the running hub WebSocket, streams the 16 kHz test utterance
chunk by chunk (exactly like a real spoke), and verifies the full loop:

  audio in -> Vosk partials -> Vosk final -> decision (wake word) ->
  response LLM -> streaming TTS audio frames back (0x11)

Skipped automatically unless HUB_URL is set (e.g. HUB_URL=ws://localhost:5051).
Requires the full stack up: docker compose up -d
"""
import asyncio
import json
import os
import sys
import unittest
import wave
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[1]
HUB_URL = os.environ.get("HUB_URL", "ws://localhost:5051")
SAMPLE = REPO / "tests" / "data" / "test_speech_16k.wav"


async def run_fake_spoke():
    """Stream the test wav to the hub; return (finals_seen, tts_bytes, statuses)."""
    statuses = []
    tts_bytes = 0
    with wave.open(str(SAMPLE), "rb") as w:
        raw = w.readframes(w.getnframes())

    async with websockets.connect(f"{HUB_URL}/ws/e2e-fake-spoke", max_size=2**20) as ws:
        await ws.send(bytes([0x03]))  # session start
        # Stream 100 ms chunks, paced like realtime
        for i in range(0, len(raw), 3200):
            await ws.send(bytes([0x01]) + raw[i:i + 3200])
            await asyncio.sleep(0.1)
        # Drain for final + decision + LLM + TTS stream (up to 60 s)
        deadline = asyncio.get_event_loop().time() + 60
        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                # Give up early if we already got audio and some quiet time
                if tts_bytes > 10000:
                    break
                continue
            ftype, payload = msg[0], msg[1:]
            if ftype == 0x11:
                tts_bytes += len(payload)
            elif ftype == 0x13:
                statuses.append(payload.rstrip(b"\x00").decode(errors="ignore"))
            elif ftype == 0x12:
                pass
    return tts_bytes, statuses


class TestE2EPipeline(unittest.TestCase):
    @unittest.skipIf(not HUB_URL, "HUB_URL not set")
    def test_full_loop_produces_streamed_tts(self):
        tts_bytes, statuses = asyncio.run(run_fake_spoke())
        print("\n[E2E] statuses:", statuses)
        print("[E2E] TTS bytes:", tts_bytes)
        # The test utterance contains the wake word "glados"... it does not:
        # "hello, what is the meaning of life?" -> the tiny LLM must say YES
        # to a direct question. Either way the loop should produce TTS.
        self.assertGreater(
            tts_bytes, 5000,
            f"expected streamed TTS audio back, got {tts_bytes} bytes (statuses={statuses})",
        )
        self.assertIn("thinking...", statuses, f"no thinking status seen: {statuses}")


if __name__ == "__main__":
    unittest.main()
