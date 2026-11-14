import re
import numpy as np
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
import websockets
import logging
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

MAX_CONTEXT_WORDS = 200
WAKE_WORDS = ["glados", "gladys", "glahdose", "hey glados", "hey gladys", "hey glahdose"]
NAME_PATTERN = re.compile(r"(?:i['\s]+am|my name is|call me|you can call me)\s+([a-zA-Z][a-zA-Z'\s-]{1,20})", re.IGNORECASE)
logger = logging.getLogger("glados_hub")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# Configuration
HUB_PORT = int(os.environ.get("HUB_PORT", "5051"))
REST_PORT = int(os.environ.get("REST_PORT", "5050"))
LLAMA_URL = os.environ.get("LLAMA_URL", "http://glados_llama:8080")
ENGINE_URL = os.environ.get("ENGINE_URL", "http://glados_engine:8081")

PROMPT_PATH = Path(__file__).parent / "prompts" / "glados_system.txt"
with open(PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read()

logger.info("GLaDOS system prompt loaded (%d chars)", len(SYSTEM_PROMPT))

app = FastAPI(title="GLaDOS Hub", version="0.1.0")
active_sessions: dict = {}
http_client = httpx.AsyncClient(timeout=120.0)


class GLaDOSConversation:
    MAX_WORDS = 300
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.conversation_id = str(uuid.uuid4())[:8]
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.audio_buffer = bytearray()
        self.is_listening = False
        self.last_speech_time = 0.0
        # Per-speaker VAD state
        self.speakers: dict = {}  # speaker_id -> {"context": str, "last_heard": float, "speech_active": bool, "silence_count": int}
        self.speaker_names: dict = {}  # speaker_id -> name
        self.current_speaker_id: str = None
        self.silence_count: int = 0  # Global silence
        self.speech_active: bool = False
        self.interrupting: bool = False
        # Shared conversation memory
        self.conversation_history: list = []
    
    def fingerprint_voice(self, audio_bytes: bytes) -> str:
        """Simple voice fingerprint from audio characteristics."""
        try:
            import struct
            samples = np.frombuffer(audio_bytes[-8000:], dtype=np.int16).astype(np.float32)[:4000]
            if len(samples) < 100:
                return "unknown"
            # Features: pitch (zero-crossing rate), spectral centroid, energy
            zcr = np.mean(np.abs(np.diff(np.sign(samples))))
            energy = np.sqrt(np.mean(samples**2))
            # Simple hash-based fingerprint
            fp = f"v{zcr:.2f}_{energy:.1f}"
            # Match to known speakers (fuzzy)
            for sid in self.speakers:
                sf = sid.split('_')
                if len(sf) >= 2:
                    sz, se = float(sf[1]), float(sf[2])
                    if abs(zcr - sz) < 0.05 and abs(energy - se) < 0.5:
                        return sid
            # New speaker
            new_id = f"speaker_{zcr:.2f}_{energy:.1f}_{time.time():.0f}"
            self.speakers[new_id] = {"context": "", "last_heard": time.time(), "speech_active": True, "silence_count": 0}
            logger.info("[glados] New speaker: %s", new_id)
            return new_id
        except Exception as e:
            logger.warning("Voice fingerprint failed: %s", e)
            return "unknown"
    
    def add_speech_to_context(self, speaker_id: str, text: str):
        """Add transcribed text to speaker's context."""
        if speaker_id and text.strip():
            # Add to speaker's personal context
            if speaker_id in self.speakers:
                self.speakers[speaker_id]["context"] += " " + text.strip()
                # Keep personal context reasonable
                words = self.speakers[speaker_id]["context"].split()
                if len(words) > 150:
                    self.speakers[speaker_id]["context"] = " ".join(words[-150:])
            
            # Add to conversation history
            n = self.speaker_names.get(speaker_id, speaker_id[:12]); self.conversation_history.append(f"{n}: {text.strip()}")
            # Keep history manageable
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
            
            self.speakers[speaker_id]["last_heard"] = time.time()



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


                # Keep max 15s buffer
                if len(session.audio_buffer) > 240000:  # 15s at 16kHz
                    session.audio_buffer = session.audio_buffer[-240000:]
                
                # VAD: check energy on last 1600 samples (100ms)
                if len(session.audio_buffer) >= 1600:
                    recent = np.frombuffer(bytes(session.audio_buffer[-1600:]), dtype=np.int16).astype(np.float64)
                    rms = np.sqrt(np.mean(recent**2)) / 32768.0
                    
                    if rms > 0.008:  # Speech threshold - raised for Whisplay HAT (noise floor ~0.01)
                        if not session.speech_active:
                            session.speech_active = True
                        session.silence_count = 0
                    elif session.speech_active:
                        session.silence_count += 1
                        if session.silence_count % 10 == 0:
                            logger.info("[%s] SILENCE COUNT: %d", device_id, session.silence_count)
                        # Update per-speaker VAD states
                        for sid, sdata in session.speakers.items():
                            if sdata.get("speech_active", False):
                                sdata["silence_count"] = sdata.get("silence_count", 0) + 1
                                if sdata["silence_count"] >= 8:  # This speaker stopped
                                    sdata["speech_active"] = False
                            else:
                                sdata["silence_count"] = 0
                        
                        if session.silence_count >= 20:  # Global 2.0s silence (wait for complete sentences)
                            logger.info("[%s] *** SILENCE COUNT REACHED %d, sending to ASR ***", device_id, session.silence_count)
                            try:
                                # 1. Voice fingerprint for this segment
                                speaker_id = session.fingerprint_voice(bytes(session.audio_buffer))
                                
                                # 2. Transcribe
                                audio_b64 = base64.b64encode(bytes(session.audio_buffer)).decode()
                                asr_resp = await http_client.post(
                                    f"{ENGINE_URL}/asr",
                                    json={"audio_base64": audio_b64, "sample_rate": 16000},
                                )
                                text = asr_resp.json().get("text", "").strip()
                                
                                # 3. Add to speaker context + conversation history
                                session.add_speech_to_context(speaker_id, text)
                                logger.info("[%s] '%s' (speaker: %s, conv: %d entries)", 
                                           device_id, text, speaker_id, len(session.conversation_history))
                                
                                # 4. Check for wake word FIRST, then fall back to LLM
                                text_lower = text.lower()
                                is_wake_word = any(wake_word in text_lower for wake_word in WAKE_WORDS)
                                
                                if is_wake_word:
                                    logger.info("[%s] Wake word detected in: '%s'", device_id, text)
                                    session.add_message("user", text)
                                    try:
                                        resp = await _call_llm(session)
                                        session.add_message("assistant", resp)
                                        await websocket.send_bytes(bytes([0x13]) + b"thinking..." + b"\x00" * 240)
                                        await _synthesize_and_stream(session, device_id, websocket, resp)
                                    finally:
                                        session.interrupting = False
                                elif session.interrupting:
                                    logger.info("[%s] GLaDOS already responding, skip", device_id)
                                else:
                                    # No wake word — use LLM to decide if GLaDOS should chime in
                                    should_respond = await _should_glados_respond(session, text)
                                    if should_respond:
                                        session.interrupting = True
                                        logger.info("[%s] GLaDOS chimes in!", device_id)
                                        session.add_message("user", text)
                                        try:
                                            resp = await _call_llm(session)
                                            session.add_message("assistant", resp)
                                            await websocket.send_bytes(bytes([0x13]) + b"thinking..." + b"\x00" * 240)
                                            await _synthesize_and_stream(session, device_id, websocket, resp)
                                        finally:
                                            session.interrupting = False
                            except Exception as e:
                                logger.error("[%s] Processing error: %s", device_id, e)
                            session.audio_buffer.clear()
                            session.speech_active = False
                            session.silence_count = 0
                            # Reset global speaker tracking
                            if session.current_speaker_id:
                                session.speakers[session.current_speaker_id]["speech_active"] = False
                                session.speakers[session.current_speaker_id]["silence_count"] = 0
            elif frame_type == 0x02:  # Wake word
                wake_text = payload.decode("utf-8", errors="ignore")
                logger.info("[%s] Wake word: '%s'", device_id, wake_text)
                session.is_listening = True
                session.audio_buffer.clear()
                await websocket.send_bytes(bytes([0x13]) + b"listening..." + b"\x00" * 240)
    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", device_id)
        active_sessions.pop(device_id, None)
    except Exception as e:
        logger.error("[%s] Error: %s", device_id, e, exc_info=True)
        active_sessions.pop(device_id, None)


async def _process_vad_asr_llm_tts(session: GLaDOSConversation, device_id: str, websocket: WebSocket):
    """Transcribe audio -> LLM -> TTS pipeline."""
    try:
        audio_b64 = base64.b64encode(bytes(session.audio_buffer)).decode()
        session.audio_buffer.clear()

        # Send "thinking" status
        await websocket.send_bytes(bytes([0x13]) + b"thinking..." + b"\x00" * 240)

        # Call ASR engine for transcription
        try:
            asr_resp = await http_client.post(
                f"{ENGINE_URL}/asr",
                json={"audio_base64": audio_b64, "sample_rate": 16000},
            )
            asr_result = asr_resp.json()
            user_text = asr_result.get("text", "").strip()
            if user_text.startswith("["):
                user_text = f"[ASR unavailable — {user_text}]"
            else:
                logger.info("[%s] Heard: '%s'", device_id, user_text)
        except Exception as asr_err:
            logger.warning("[%s] ASR failed: %s", device_id, asr_err)
            user_text = "[ASR error]"

        session.add_message("user", user_text)

        llm_response = await _call_llm(session)
        session.add_message("assistant", llm_response)

        # Synthesize TTS
        await _synthesize_and_stream(session, device_id, websocket, llm_response)
    except Exception as e:
        logger.error("[%s] Processing error: %s", device_id, e, exc_info=True)
        try:
            await websocket.send_bytes(bytes([0x13]) + f"error:{e}".encode() + b"\x00" * 200)
        except (Exception,):
            pass


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
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{ENGINE_URL}/tts-kokoro",
                json={"text": text, "speed": 1.0},
            )
        if resp.status_code != 200:
            logger.warning("[%s] TTS returned %d: %s", device_id, resp.status_code, resp.text[:200])
            return
        audio_data = resp.json()
        audio_bytes = base64.b64decode(audio_data["audio_base64"])
        sr = audio_data["sample_rate"]
        chunk_size = sr * 0.2 * 2
        for i in range(0, len(audio_bytes), int(chunk_size)):
            try:
                chunk = audio_bytes[i:i + int(chunk_size)]
                await websocket.send_bytes(bytes([0x11]) + chunk)
            except (Exception,):  # client disconnected during streaming
                logger.info("[%s] Client disconnected during TTS playback", device_id)
                return
        logger.info("[%s] Sent %d bytes of TTS audio (%d Hz)", device_id, len(audio_bytes), sr)
    except Exception as e:
        logger.error("[%s] TTS failed: %s", device_id, e)
        try:
            await websocket.send_bytes(bytes([0x13]) + f"tts error: {e}".encode() + b"\x00" * 200)
        except (Exception,):  # already disconnected
            pass



async def _should_glados_respond(session: GLaDOSConversation, latest_text: str) -> bool:
    """Use LLM to decide if GLaDOS should chime in naturally."""
    try:
        # Build context summary
        recent = session.conversation_history[-10:]  # Last 10 speech segments
        context_str = "; ".join(recent)
        
        prompt = f"""You are GLaDOS. Respond YES or NO only.
Should GLaDOS interrupt to answer/comment? Consider:
- Is there a question being asked?
- Does GLaDOS know the answer from context?
- Is it natural to chime in?

Recent conversation: {context_str}
Latest speech: '{latest_text}'

Answer: """
        
        resp = await http_client.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4,
                "temperature": 0.1,
            },
        )
        result = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return "YES" in result
    except Exception as e:
        logger.warning("Intent check failed: %s", e)
        return False


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(active_sessions),
            "llama_url": LLAMA_URL, "engine_url": ENGINE_URL}


@app.post("/say")
async def direct_say(text: str = Query(..., description="Text to speak")):
    """Direct text-to-speech endpoint (for testing without a spoke)."""
    return {"message": f"GLaDOS would say: {text}", "status": "ok"}
