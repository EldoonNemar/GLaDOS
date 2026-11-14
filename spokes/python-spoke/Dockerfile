FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils alsa-topology-conf alsa-ucm-conf && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy websockets pyrnnoise

COPY spokes/python-spoke/setup_audio.py /app/setup_audio.py
COPY spokes/python-spoke/main.py /app/main.py

CMD ["python", "-u", "/app/setup_audio.py"]
