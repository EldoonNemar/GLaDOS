FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive MODEL_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    speechrecognition \
    fastapi uvicorn[standard] \
    numpy soundfile pydantic \
    kokoro-onnx

WORKDIR /app
COPY docker/engine/glados_engine/ ./
EXPOSE 8081
CMD ["uvicorn", "engine:app", "--host", "0.0.0.0", "--port", "8081"]
