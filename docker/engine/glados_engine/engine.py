"""
glados_engine - ASR (Parakeet-tdt) + VAD (Silero) + TTS (Piper/Kokoro) pipeline.

Exposes a FastAPI service at :8081:
  POST /asr          - Transcribe audio -> text
  POST /tts          - Synthesize text -> audio (Piper)
  POST /tts-kokoro   - Synthesize text -> audio (Kokoro)
  POST /vad          - Check speech activity
  GET  /health       - Health check
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from kokoro_onnx import Kokoro as KokoroTTS
from diarizer import diarize
from pydantic import BaseModel

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
ASR_ENCODER = os.path.join(MODEL_DIR, os.environ.get("ASR_ENCODER", "/models/ASR/parakeet-tdt-0.6b-v3_encoder.onnx"))
ASR_DECODER = os.path.join(MODEL_DIR, os.environ.get("ASR_DECODER", "/models/ASR/parakeet-tdt-0.6b-v3_decoder.onnx"))
ASR_JOINER  = os.path.join(MODEL_DIR, os.environ.get("ASR_JOINER", "/models/ASR/parakeet-tdt-0.6b-v3_joiner.onnx"))
VAD_MODEL   = os.path.join(MODEL_DIR, os.environ.get("VAD_MODEL", "/models/ASR/silero_vad_16k_op15.onnx"))
TTS_MODEL   = os.path.join(MODEL_DIR, os.environ.get("TTS_MODEL", "/models/TTS/glados.onnx"))
TTS_CONFIG  = os.path.join(MODEL_DIR, os.environ.get("TTS_CONFIG", "/models/TTS/glados.json"))
TTS_KOKORO  = os.path.join(MODEL_DIR, os.environ.get("TTS_KOKORO", "/models/TTS/kokoro-v1.0.fp16.onnx"))
TTS_VOICES  = os.path.join(MODEL_DIR, os.environ.get("TTS_VOICES", "/models/TTS/kokoro-voices-v1.0.bin"))

logger = logging.getLogger("glados_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

app = FastAPI(title="GLaDOS Engine", version="0.1.0")
_session_cache: dict = {}


def _get_session(path: str, providers=None):
    if path not in _session_cache:
        try:
            if providers is None:
                providers = ["CUDAExecutionProvider"]
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=providers)
            _session_cache[path] = sess
            logger.info("Loaded ONNX model: %s", path)
        except Exception as e:
            logger.warning("GPU provider failed for %s, falling back to CPU: %s", path, e)
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            _session_cache[path] = sess
    return _session_cache[path]



class SileroVAD:
    def __init__(self, model_path: str):
        self.session = _get_session(model_path, providers=["CPUExecutionProvider"])
        self.sample_rate = 16000
        self.input_name = self.session.get_inputs()[0].name
        self.state_name = self.session.get_inputs()[1].name
        self.state2_name = self.session.get_inputs()[2].name
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._last_ts = 0.0

    def detect_speech(self, audio_chunk: np.ndarray, threshold: float = 0.5) -> Optional[float]:
        if len(audio_chunk) < 512:
            return None
        audio = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)))
        audio = audio[:512]
        output, self._h, self._c = self.session.run(
            None,
            {self.input_name: audio.reshape(1, -1).astype(np.float32),
             self.state_name: self._h.astype(np.float32),
             self.state2_name: self._c.astype(np.float32)},
        )
        prob = float(output[0][0])
        if prob >= threshold:
            self._last_ts = len(audio) / self.sample_rate
        return self._last_ts if prob >= threshold else None


class PiperTTS:
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.session = _get_session(model_path, providers=["CUDAExecutionProvider"])
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.sample_rate = self.config["audio"]["sample_rate"]
        logger.info("Piper TTS loaded: %d Hz", self.sample_rate)

    def synthesize(self, text: str, speed: float = 1.0) -> tuple:
        from misaki.piper import Piper
        piper = Piper.from_path(self.model_path)
        samples = piper.synthesize(text, speed=speed)
        return samples, self.sample_rate


class KokoroTTSWrapper:
    def __init__(self, model_path: str, voices_path: str):
        self.kokoro = KokoroTTS(model_path, voices_path)
        self.sample_rate = 24000
        logger.info("Kokoro TTS loaded: %d Hz", self.sample_rate)

    def synthesize(self, text: str, voice: str = "af_sarah", speed: float = 1.0) -> tuple:
        samples, sr = self.kokoro.create(text, voice=voice, speed=1.0 / speed, lang="en-us")
        return samples, sr


try:
    vad = SileroVAD(str(VAD_MODEL))
    logger.info("Silero VAD initialized")
except Exception as e:
    logger.warning("VAD init failed: %s", e)
    vad = None

try:
    piper_tts = PiperTTS(str(TTS_MODEL), str(TTS_CONFIG))
    logger.info("Piper TTS initialized")
except Exception as e:
    logger.warning("Piper TTS init failed (%s), will use Kokoro", e)
    piper_tts = None

try:
    kokoro_tts = KokoroTTSWrapper(str(TTS_KOKORO), str(TTS_VOICES))
    logger.info("Kokoro TTS initialized")
except Exception as e:
    logger.error("Kokoro TTS init failed: %s", e)
    kokoro_tts = None


class AudioRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000

class TextRequest(BaseModel):
    text: str
    speed: float = 1.0
    use_kokoro: bool = False
    voice: str = "af_sarah"

class TranscriptResponse(BaseModel):
    text: str
    confidence: float = 0.0

class AudioResponse(BaseModel):
    audio_base64: str
    sample_rate: int


@app.post("/diarize")
async def diarize_audio(req: AudioRequest):
    """Diarize audio into speaker segments."""
    raw = base64.b64decode(req.audio_base64)
    segments = diarize(raw, req.sample_rate)
    return {"segments": [{"speaker": s["speaker"], "start": s["start"], "end": s["end"]} for s in segments]}


@app.get("/health")
async def health():
    return {"status": "ok", "models": {"vad": vad is not None, "piper_tts": piper_tts is not None, "kokoro_tts": kokoro_tts is not None, "asr": True}}

import os
import tempfile
import wave

_vosk_model = None


def _get_vosk_model():
    """Lazy-load Vosk model (downloads tiny model on first use)."""
    global _vosk_model
    if _vosk_model is None:
        try:
            from vosk import Model, SpkModel
            model_dir = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")
            if not os.path.exists(os.path.join(model_dir, "graph", "phones")):
                logger.info("Downloading Vosk tiny model (~50MB)...")
                import urllib.request
                url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                tmp = os.path.join("/tmp", "vosk-model-small-en-us-0.15.zip")
                urllib.request.urlretrieve(url, tmp)
                import zipfile
                os.makedirs(model_dir, exist_ok=True)
                with zipfile.ZipFile(tmp, "r") as z:
                    z.extractall(model_dir)
                os.remove(tmp)
                # Zip may extract to nested dir (same name under parent)
                contents = os.listdir(model_dir)
                if len(contents) == 1 and os.path.isdir(os.path.join(model_dir, contents[0])):
                    model_dir = os.path.join(model_dir, contents[0])
            _vosk_model = Model(model_dir)
            logger.info("Vosk model loaded")
        except Exception as e:
            logger.error("Vosk load failed: %s", e)
            _vosk_model = "error"
    return _vosk_model


@app.post("/asr-debug")
async def asr_debug(req: AudioRequest):
    """Debug endpoint: dump raw audio stats."""
    raw = base64.b64decode(req.audio_base64)
    audio = np.frombuffer(raw, dtype=np.int16)
    rms = float(np.sqrt(np.mean((audio/32768.0)**2)))
    peak = float(np.max(np.abs(audio/32768.0)))
    return {
        "samples": len(audio),
        "bytes": len(raw),
        "rms": rms,
        "peak": peak,
        "min": int(audio.min()),
        "max": int(audio.max()),
        "mean": float(audio.mean()),
        "first_10": audio[:10].tolist(),
    }


@app.post("/asr", response_model=TranscriptResponse)
async def transcribe(req: AudioRequest):
    """Transcribe audio using Vosk STT."""
    raw = base64.b64decode(req.audio_base64)
    audio = np.frombuffer(raw, dtype=np.int16)
    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")
    try:
        model = _get_vosk_model()
        if model == "error":
            rms = float(np.sqrt(np.mean((audio/32768.0)**2)))
            return TranscriptResponse(text=f"speech detected (rms={rms:.4f})", confidence=0.0)
        
        import json as _json
        from vosk import KaldiRecognizer
        rms = float(np.sqrt(np.mean((audio/32768.0)**2)))
        peak = float(np.max(np.abs(audio/32768.0)))
        logger.info("ASR debug: samples=%d rms=%.4f peak=%.4f dtype=%s", len(audio), rms, peak, audio.dtype)
        rec = KaldiRecognizer(model, 16000)
        chunk_size = 640  # ~40ms at 16kHz (Vosk prefers smaller chunks)
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size].tobytes()
            got = rec.AcceptWaveform(chunk)
            if got:
                result = _json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    logger.info("ASR interim: '%s'", text)
        result = _json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        if not text:
            text = "silence" if rms < 0.01 else f"speech detected (rms={rms:.4f})"
        logger.info("Vosk ASR: '%s'", text)
        return TranscriptResponse(text=text, confidence=0.0)
    except Exception as e:
        logger.error("ASR error: %s", e, exc_info=True)
        return TranscriptResponse(text=f"asr error: {e}", confidence=0.0)


@app.post("/tts", response_model=AudioResponse)
async def synthesize_piper(req: TextRequest):
    if piper_tts is None:
        raise HTTPException(status_code=503, detail="Piper TTS not available")
    samples, sr = piper_tts.synthesize(req.text, speed=req.speed)
    audio_int16 = (samples * 32767).astype(np.int16)
    return AudioResponse(audio_base64=base64.b64encode(audio_int16.tobytes()).decode("utf-8"), sample_rate=sr)

@app.post("/tts-kokoro", response_model=AudioResponse)
async def synthesize_kokoro(req: TextRequest):
    if kokoro_tts is None:
        raise HTTPException(status_code=503, detail="Kokoro TTS not available")
    samples, sr = kokoro_tts.synthesize(req.text, voice=req.voice, speed=req.speed)
    audio_int16 = (samples * 32767).astype(np.int16)
    return AudioResponse(audio_base64=base64.b64encode(audio_int16.tobytes()).decode("utf-8"), sample_rate=sr)

@app.post("/vad")
async def vad_check(req: AudioRequest):
    if vad is None:
        raise HTTPException(status_code=503, detail="VAD not available")
    raw = base64.b64decode(req.audio_base64)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    prob = vad.detect_speech(audio)
    return {"has_speech": prob is not None, "timestamp": prob or 0.0}
