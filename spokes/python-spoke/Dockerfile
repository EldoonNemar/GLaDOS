# ── Python Spoke: reSpeaker XVF3800 frontend ────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GLADOS_HUB_URL=ws://host.docker.internal:5051

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 libasound2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY spokes/python-spoke/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY spokes/python-spoke/ ./

CMD ["python", "main.py"]
