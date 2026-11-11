"""
glados_hub - WebSocket hub + LLM/TTS orchestrator.

Endpoints:
  WS ws://host:5051/ws/{device_id}   - Frontend spoke connection
  GET  http://host:5050/health       - Health check
  POST http://host:5050/say          - Direct text-to-speech (for testing)
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger("glados_hub")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# Configuration
HUB_PORT = int(os.environ.get("HUB_PORT", "5051"))
REST_PORT = int(os.environ.get("REST_PORT", "5050"))
LLAMA_URL = os.environ.get("LLAMA_URL", "http://glados_llama:8080")
ENGINE_URL = os.environ.get("ENGINE_URL", "http://glados_engine:8081")
WAKE_WORD = os.environ.get("WAKE_WORD", "glados").lower()

PROMPT_PATH = Path(__file__).parent / "prompts" / "glados_system.txt"
with open(PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read()

logger.info("GLaDOS system prompt loaded (%d chars)", len(SYSTEM_PROMPT))

app = FastAPI(title="GLaDOS Hub", version="0.1.0")
active_sessions: dict = {}
http_client = httpx.AsyncClient(timeout=120.0)


class GLaDOSConversation:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.conversation_id = str(uuid.uuid4())[:8]
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.audio_buffer = bytearray()
        self.is_listening = False
        self.last_speech_time = 0.0

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()
    logger.info("Client connected: %s", device_id)
    session = GLaDOSConversation(device_id)
    active_sessions[device_id] = session
    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) == 0:
                continue
            frame_type = data[0]
            payload = data[1:]
            if frame_type == 0x03:  # Session start
                session.is_listening = True
                session.audio_buffer.clear()
                logger.info("[%s] Session started", device_id)
            elif frame_type == 0x04:  # Session end
                session.is_listening = False
                session.audio_buffer.clear()
                logger.info("[%s] Session ended", device_id)
            elif frame_type == 0x01 and session.is_listening:  # Audio chunk
                session.audio_buffer.extend(payload)
                session.last_speech_time = time.time()
                if len(session.audio_buffer) >= 32000:  # ~2s buffer
                    await _process_vad_asr_llm_tts(session, device_id, websocket)
            elif frame_type == 0x02:  # Wake word
                wake_text = payload.decode("utf-8", errors="ignore")
                logger.info("[%s] Wake word: '%s'", device_id, wake_text)
                session.is_listening = True
                session.audio_buffer.clear()
                await websocket.send_bytes(bytes([0x13] + b"listening..." + b"\x00" * 240))
    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", device_id)
        active_sessions.pop(device_id, None)
    except Exception as e:
        logger.error("[%s] Error: %s", device_id, e, exc_info=True)
        active_sessions.pop(device_id, None)


async def _process_vad_asr_llm_tts(session: GLaDOSConversation, device_id: str, websocket: WebSocket):
    """Run LLM -> TTS pipeline. ASR runs on spoke."""
    try:
        audio_b64 = base64.b64encode(bytes(session.audio_buffer)).decode()
        session.audio_buffer.clear()

        # Send "thinking" status
        await websocket.send_bytes(bytes([0x13] + b"thinking..." + b"\x00" * 240))

        # TODO: Send audio_b64 to ENGINE_URL/asr for transcription
        user_text = f"[audio: {len(audio_b64)} chars]"
        session.add_message("user", user_text)

        llm_response = await _call_llm(session)
        session.add_message("assistant", llm_response)

        # Synthesize TTS
        await _synthesize_and_stream(session, device_id, websocket, llm_response)
    except Exception as e:
        logger.error("[%s] Processing error: %s", device_id, e, exc_info=True)
        await websocket.send_bytes(bytes([0x13] + f"error:{e}".encode() + b"\x00" * 200))


async def _call_llm(session: GLaDOSConversation) -> str:
    """Call llama.cpp to generate a GLaDOS response."""
    try:
        resp = await http_client.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": session.messages[-20:],
                "max_tokens": 256,
                "stream": False,
                "temperature": 0.7,
            },
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return "I'm sorry, but I'm currently undergoing testing. Please stand by."


async def _synthesize_and_stream(session: GLaDOSConversation, device_id: str, websocket: WebSocket, text: str):
    """Synthesize TTS and stream audio back to spoke."""
    try:
        resp = await http_client.post(
            f"{ENGINE_URL}/tts-kokoro",
            json={"text": text, "speed": 1.0},
        )
        audio_data = resp.json()
        audio_bytes = base64.b64decode(audio_data["audio_base64"])
        sr = audio_data["sample_rate"]
        # Stream audio in ~200ms chunks (frame type 0x11)
        chunk_size = sr * 0.2 * 2
        for i in range(0, len(audio_bytes), int(chunk_size)):
            chunk = audio_bytes[i:i + int(chunk_size)]
            await websocket.send_bytes(bytes([0x11]) + chunk)
        logger.info("[%s] Sent %d bytes of TTS audio (%d Hz)", device_id, len(audio_bytes), sr)
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        await websocket.send_bytes(bytes([0x13] + f"tts error: {e}".encode() + b"\x00" * 200))


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(active_sessions),
            "llama_url": LLAMA_URL, "engine_url": ENGINE_URL}


@app.post("/say")
async def direct_say(text: str = Query(..., description="Text to speak")):
    """Direct text-to-speech endpoint (for testing without a spoke)."""
    return {"message": f"GLaDOS would say: {text}", "status": "ok"}
