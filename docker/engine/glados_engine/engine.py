"""
glados_engine - ASR (sherpa-onnx / Parakeet-TDT) + VAD (Silero) + TTS (Piper GLaDOS) pipeline.

Exposes a FastAPI service at :8081:
  POST /asr   - Transcribe audio -> text
  POST /tts   - Synthesize text -> audio (GLaDOS Piper voice)
  POST /vad   - Check speech activity
  GET  /health - Health check
"""

import base64
import logging
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
ASR_CTC_MODEL = os.path.join(MODEL_DIR, os.environ.get("ASR_CTC_MODEL", "/models/ASR/nemo-parakeet_tdt_ctc_110m.onnx"))
ASR_TOKENS  = os.path.join(MODEL_DIR, os.environ.get("ASR_TOKENS",  "/models/ASR/tokens.txt"))
VAD_MODEL   = os.path.join(MODEL_DIR, os.environ.get("VAD_MODEL", "/models/ASR/silero_vad_16k_op15.onnx"))

logger = logging.getLogger("glados_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

app = FastAPI(title="GLaDOS Engine", version="0.1.0")

# ── Register MIGraphX EP for ROCm GPU acceleration ─────────────────
_migraphx_available = False
try:
    import migraphx  # noqa: F401  # initializes ROCm runtime
    import onnxruntime as ort
    import onnxruntime_ep_migraphx as m
    for ep_name, ep_lib in zip(m.get_ep_names(), m.get_library_paths()):
        ort.register_execution_provider_library(ep_name, ep_lib)
    if "MIGraphXExecutionProvider" in ort.get_available_providers():
        _migraphx_available = True
        logger.info("MIGraphX EP registered and available")
    else:
        logger.warning("MIGraphX EP not in available providers: %s", ort.get_available_providers())
except ImportError as e:
    logger.warning("MIGraphX EP not available: %s", e)
except Exception as e:
    logger.warning("Failed to register MIGraphX EP: %s", e)

# ── Use MIGraphX provider helper ──────────────────────────────────
def _get_migraphx_provider():
    """Get MIGraphXExecutionProvider class for use in session providers."""
    if not _migraphx_available:
        return None
    import onnxruntime as ort
    try:
        return ort.ExecutionProvider["MIGraphXExecutionProvider"]
    except (KeyError, AttributeError):
        return None


def _get_asr_provider():
    """Return the best available ONNX Runtime provider."""
    if _migraphx_available:
        return "migraphx"
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "cuda"
    return "cpu"


# ── ASR: sherpa-onnx Moonshine ────────────────────────────────────
_asr_recognizer = None
_asr_available = False

def _init_asr():
    """Initialize the sherpa-onnx recognizer with Moonshine model."""
    global _asr_recognizer, _asr_available
    try:
        import sherpa_onnx

        encode_model = "/models/ASR/moonshine_encode.onnx"
        preprocess_model = "/models/ASR/moonshine_preprocess.onnx"
        cached_decode_model = "/models/ASR/moonshine_cached_decode.onnx"
        tokens = "/models/ASR/moonshine_tokens.txt"

        if not all(Path(m).exists() for m in [encode_model, preprocess_model, cached_decode_model, tokens]):
            logger.error("Moonshine ASR model files not found")
            return

        logger.info("Loading Moonshine ASR")

        # Get MIGraphX provider if available
        migraphx_provider = _get_migraphx_provider()
        providers = [migraphx_provider, "CPUExecutionProvider"] if migraphx_provider else ["CPUExecutionProvider"]

        _asr_recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=preprocess_model,
            encoder=encode_model,
            uncached_decoder="/models/ASR/moonshine_uncached_decode.onnx",
            cached_decoder=cached_decode_model,
            tokens=tokens,
            num_threads=int(os.environ.get("ASR_THREADS", "4")),
            decoding_method="greedy_search",
            debug=False,
            provider=str(providers),
        )
        _asr_available = True
        logger.info("Moonshine ASR loaded successfully")

    except Exception as e:
        logger.error("Failed to load ASR model: %s", e, exc_info=True)
        _asr_available = False


_init_asr()

# ── VAD: Silero ───────────────────────────────────────────────────
_vad_session = None

def _init_vad():
    global _vad_session
    try:
        import onnxruntime as ort
        if not Path(VAD_MODEL).exists():
            logger.warning("VAD model not found: %s", VAD_MODEL)
            return
        _vad_session = ort.InferenceSession(VAD_MODEL, providers=["CPUExecutionProvider"])
        logger.info("Silero VAD loaded from %s", VAD_MODEL)
    except Exception as e:
        logger.error("Failed to load VAD: %s", e)

_init_vad()

# ── TTS: Piper (GLaDOS) ───────────────────────────────────────────
_TTS_MODEL = Path("/models/TTS/glados.onnx")
_TTS_CONFIG = Path("/models/TTS/glados.json")
_tts = None
_tts_sr = 22050  # Piper GLaDOS default sample rate

def _init_tts():
    global _tts, _tts_sr
    try:
        from piper import PiperVoice
        import json

        _TTS_CONFIG_PATH = str(_TTS_CONFIG)  # /models/TTS/glados.json
        with open(_TTS_CONFIG_PATH) as f:
            config = json.load(f)
        _tts_sr = config.get("audio", {}).get("sample_rate", 22050)
        _tts = PiperVoice.load(str(_TTS_MODEL), config_path=_TTS_CONFIG_PATH)
        logger.info("Piper GLaDOS TTS loaded (sr=%d, provider=CPU)", _tts_sr)
    except Exception as e:
        logger.error("Failed to load TTS: %s", e)

_init_tts()


# ── Request/Response models ───────────────────────────────────────
class AudioRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000

class TextRequest(BaseModel):
    text: str
    speed: float = 1.0
    voice: str = "af_sarah"

class TranscriptResponse(BaseModel):
    text: str
    confidence: float = 0.0

class AudioResponse(BaseModel):
    audio_base64: str
    sample_rate: int


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr_available": _asr_available,
        "vad_available": _vad_session is not None,
        "tts_available": _tts is not None,
    }


@app.post("/asr", response_model=TranscriptResponse)
async def transcribe(req: AudioRequest):
    """Transcribe audio using Parakeet-TDT via sherpa-onnx."""
    if not _asr_available or _asr_recognizer is None:
        raise HTTPException(status_code=503, detail="ASR model not available")

    raw = base64.b64decode(req.audio_base64)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    try:
        # Create a stream and feed the audio
        stream = _asr_recognizer.create_stream()
        stream.accept_waveform(16000, audio)

        # Decode
        _asr_recognizer.decode_stream(stream)

        # Get result
        result = stream.result.text
        text = result.strip() if result else ""

        if not text:
            text = "silence"

        logger.info("sherpa-onnx ASR: '%s'", text)
        return TranscriptResponse(text=text, confidence=0.0)
    except Exception as e:
        logger.error("ASR error: %s", e, exc_info=True)
        return TranscriptResponse(text=f"asr error: {e}", confidence=0.0)


@app.post("/tts", response_model=AudioResponse)
async def synthesize(req: TextRequest):
    """Synthesize text using the GLaDOS Piper voice."""
    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS not available")
    
    # Collect all audio chunks into a single numpy array
    all_samples = []
    for chunk in _tts.synthesize(req.text):
        all_samples.append(chunk.audio_int16_array)
    
    if not all_samples:
        raise HTTPException(status_code=400, detail="No audio generated")
    
    samples = np.concatenate(all_samples)
    sr = chunk.sample_rate
    audio_int16 = samples.astype(np.int16)
    return AudioResponse(audio_base64=base64.b64encode(audio_int16.tobytes()).decode("utf-8"), sample_rate=sr)


@app.post("/vad")
async def vad_check(req: AudioRequest):
    if _vad_session is None:
        raise HTTPException(status_code=503, detail="VAD not available")
    raw = base64.b64decode(req.audio_base64)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < 512:
        return {"has_speech": False, "timestamp": 0.0}
    audio = audio[:512]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    input_name = _vad_session.get_inputs()[0].name
    state_name = _vad_session.get_inputs()[1].name
    sr_name = _vad_session.get_inputs()[2].name
    h = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array([16000], dtype=np.int64)
    output = _vad_session.run(None, {
        input_name: audio.reshape(1, -1).astype(np.float32),
        state_name: h.astype(np.float32),
        sr_name: sr.astype(np.int64),
    })[0]
    prob = float(output.flatten()[0])
    return {"has_speech": prob > 0.5, "timestamp": prob}
