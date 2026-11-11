FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'pcm.!default {
    type plug
    slave.pcm "plughw:4,0"
}

ctl.!default {
    type hw
    card 4
}' > /etc/asound.conf

WORKDIR /app
COPY spokes/python-spoke/main.py ./
CMD ["python", "main.py"]
