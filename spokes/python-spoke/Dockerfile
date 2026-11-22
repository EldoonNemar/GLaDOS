# Build stage: spidev is a C extension and needs a compiler; keep it out of
# the final image. The LAFVIN HAT display only works on a Raspberry Pi with
# SPI enabled and the HAT's /dev/spidev0.0 + /dev/gpiochip0 exposed to the
# container; on any other host the display auto-disables at runtime.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --prefix=/install spidev \
    || echo "LAFVIN display deps unavailable - display will auto-disable"

FROM python:3.12-slim
COPY --from=build /install /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils alsa-topology-conf alsa-ucm-conf libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy websockets pyrnnoise

COPY spokes/python-spoke/setup_audio.py /app/setup_audio.py
COPY spokes/python-spoke/main.py /app/main.py
COPY spokes/python-spoke/lafvin.py /app/lafvin.py
COPY spokes/python-spoke/face.py /app/face.py

CMD ["python", "-u", "/app/setup_audio.py"]
