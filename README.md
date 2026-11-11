# GLaDOS — Virtual Assistant

A sarcastic, deadpan AI assistant based on the Portal character GLaDOS.

## Architecture

```
┌──────────────────────────────────────┐
│         ROCm GPU Server              │
│                                      │
│  ┌────────────┐  ┌────────────────┐ │
│  │ glados_llama│  │  glados_engine │ │
│  │ (llama.cpp) │  │ (ASR/TTS)      │ │
│  │  Port 8080  │  │  Port 8081     │ │
│  └────────────┘  └────────┬───────┘ │
│                           │          │
│  ┌────────────────────────▼───────┐ │
│  │  glados_hub                    │ │
│  │  WebSocket: 5051 / REST: 5050  │ │
│  └────────────────────────────────┘ │
└─────────────────┬───────────────────┘
                  │ WebSocket
         ┌────────┴────────┐
         │                 │
  ┌──────▼──────┐   ┌─────▼──────┐
  │Python Spoke │   │ ESP32-C6   │
  │(RPi/Laptop) │   │(XVF3800)   │
  └─────────────┘   └────────────┘
```

## Quick Start

```bash
# 1. Download the LLM model
./models/download-gemma4.sh

# 2. Start the backend (LLM + ASR/TTS + WebSocket hub)
docker compose up -d glados_llama glados_engine glados_hub

# 3. Check health
curl http://localhost:5050/health

# 4. Start the Python spoke (on the device with the mic)
docker compose --profile spoke up -d glados_spoke
```

## Components

- **glados_llama** — Gemma 4 E4B LLM via llama.cpp (ROCm)
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
