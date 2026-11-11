# ── glados_engine: ASR + TTS inference service ─────────────────────
# Runs Parakeet-tdt (ASR), Silero VAD, and Piper/Kokoro (TTS) on GPU.
# Exposes a FastAPI endpoint at :8081 for the hub to call.

FROM python:3.12-slim

# ROCm runtime + ONNX Runtime GPU
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_DIR=/models \
    ASR_ENCODER=models/ASR/parakeet-tdt-0.6b-v3_encoder.onnx \
    ASR_DECODER=models/ASR/parakeet-tdt-0.6b-v3_decoder.onnx \
    ASR_JOINER=models/ASR/parakeet-tdt-0.6b-v3_joiner.onnx \
    VAD_MODEL=models/ASR/silero_vad_16k_op15.onnx \
    TTS_MODEL=models/TTS/glados.onnx \
    TTS_CONFIG=models/TTS/glados.json \
    TTS_KOKORO=models/TTS/kokoro-v1.0.fp16.onnx \
    TTS_VOICES=models/TTS/kokoro-voices-v1.0.bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg libopus0 && \
    rm -rf /var/lib/apt/lists/*

# ONNX Runtime with ROCm/GPU support
RUN pip install --no-cache-dir \
    onnxruntime-rocm \
    onnxruntime \
    fastapi \
    uvicorn[standard] \
    numpy \
    soundfile \
    pydantic \
    scipy \
    librosa \
    Kokoro-onnx \
    misaki[kokoro,en]

WORKDIR /app
COPY docker/engine/glados_engine/ ./

EXPOSE 8081

CMD ["uvicorn", "engine:app", "--host", "0.0.0.0", "--port", "8081"]