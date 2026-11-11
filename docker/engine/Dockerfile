FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive MODEL_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY docker/engine/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY docker/engine/glados_engine/ ./
EXPOSE 8081
CMD ["uvicorn", "engine:app", "--host", "0.0.0.0", "--port", "8081"]
