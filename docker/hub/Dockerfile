# ── glados_hub: WebSocket hub + LLM/TTS orchestrator ────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/hub/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/hub/glados_hub/ ./glados_hub/
COPY docker/hub/glados_hub/prompts/ ./glados_hub/prompts/

EXPOSE 5050 5051

CMD ["uvicorn", "hub:app", "--host", "0.0.0.0", "--port", "5051", "--app-dir", "glados_hub"]
