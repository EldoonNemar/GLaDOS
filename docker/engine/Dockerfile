FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive MODEL_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Create ldconfig hook for ROCm libraries (mounted at runtime)
RUN mkdir -p /etc/ld.so.conf.d && \
    echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && \
    ldconfig

# Install onnxruntime from PyPI first, then MIGraphX and onnxruntime-ep-migraphx for ROCm 10.1.0
RUN pip install --no-cache-dir onnxruntime==1.29.0 && \
    pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/migraphx/whl-next/ \
        "migraphx==2.18.0+rocm10.1.0" && \
    pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/onnxruntime/whl-next/ \
        "onnxruntime-ep-migraphx==1.0.0+rocm10.1.0"

# Create symlinks required by onnxruntime-ep-migraphx
RUN SP=/usr/local/lib/python3.12/site-packages && \
    ln -sf $SP/onnxruntime/capi/libonnxruntime.so.1.29.0 $SP/onnxruntime/capi/libonnxruntime.so.1 && \
    ln -sf $SP/onnxruntime/capi/libonnxruntime.so.1.29.0 $SP/onnxruntime/capi/libonnxruntime.so

COPY docker/engine/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY docker/engine/glados_engine/ ./
EXPOSE 8081
CMD ["uvicorn", "engine:app", "--host", "0.0.0.0", "--port", "8081"]
