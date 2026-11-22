# GLaDOS — Virtual Assistant

A sarcastic, deadpan AI assistant based on the Portal character GLaDOS.

## Setup

### 1. Configure the local LLM

The hub's local LLM is qwen3.8-27b served by the radiance engine
(`/opt/jigowatts`, host port 8000). Put radiance's API key in `.env`:

```bash
LLM_API_KEY=<RADIANCE_API_KEY from /opt/jigowatts/.env>
```

(Old path: Gemma 4 E4B via llama.cpp with `./models/download-gemma4.sh` —
retired; the `glados_llama` service stays commented out in docker-compose.yml
for a one-step rollback.)

### 2. Start the backend

```bash
# Backend only (ASR/TTS + WebSocket hub; local LLM = radiance on the host)
docker compose up -d glados_engine glados_hub

# Check health
curl http://localhost:5050/health

# Also start the Python spoke (on the device with the mic)
docker compose --profile spoke up -d glados_spoke
```

## Architecture

```
┌──────────────────────────────────────┐
│         ROCm GPU Server              │
│                                      │
│  ┌─────────────┐  ┌────────────────┐ │
│  │  radiance   │  │  glados_engine │ │
│  │(qwen3.8-27B)│  │ (ASR/TTS)      │ │
│  │  Port 8000  │  │  Port 8081     │ │
│  └─────────────┘  └────────┬───────┘ │
│                            │         │
│  ┌─────────────────────────▼──────┐  │
│  │  glados_hub                    │  │
│  │  WebSocket: 5051 / REST: 5050  │  │
│  └────────────────────────────────┘  │
└─────────────────┬────────────────────┘
                  │ WebSocket
         ┌────────┴────────┐
         │                 │
  ┌──────▼──────┐   ┌──────▼─────┐
  │Python Spoke │   │ ESP32-C6   │
  │(RPi/Laptop) │   │(XVF3800)   │
  └─────────────┘   └────────────┘
```

## Quick Start

```bash
# 1. Put radiance's API key in .env (LLM_API_KEY=...)

# 2. Start the backend (ASR/TTS + WebSocket hub; local LLM = radiance on host)
docker compose up -d glados_engine glados_hub

# 3. Check health
curl http://localhost:5050/health

# 4. Start the Python spoke (on the device with the mic)
docker compose --profile spoke up -d glados_spoke
```

## Components

- **Local LLM** — qwen3.8-27b via the radiance engine (host port 8000; the old glados_llama / Gemma 4 E4B service is retired)
- **glados_engine** — ASR (Parakeet-tdt) + TTS (Piper/Kokoro)
- **glados_hub** — WebSocket orchestrator + GLaDOS personality
- **Python Spoke** — Frontend for laptop/RPi/Orange Pi
- **ESP32-C6 Spoke** — Embedded frontend for reSpeaker XVF3800

## Wake Word

Default wake word: "glados" (configurable via `WAKE_WORD` env var)

## Future: Always-On Intelligence

Phase 2 will add always-on listening where GLaDOS can:
- Detect if you're talking to her (keyword spotting)
- Chime in when she knows the answer to a question she overhears
- Distinguish between "talking to GLaDOS" vs "talking near GLaDOS"

## Testing

```bash
# Run unit tests
pytest tests/
```
