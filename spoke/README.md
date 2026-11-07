# GLaDOS Spoke Client — Raspberry Pi

Lightweight voice endpoint for the GLaDOS hub-and-spoke system.

## Install

```bash
pip install glados-spoke
```

Or install from source:
```bash
cd /opt/GLaDOS/spoke
pip install -e .
```

## Usage

```bash
# Basic usage (hub on same network)
glados-spoke --hub 192.168.1.100 --port 5051

# With AEC (recommended if speaker leaks into mic)
glados-spoke --hub 192.168.1.100 --port 5051 --aec

# Specify audio devices (find with: arecord -l / aplay -l)
glados-spoke --hub 192.168.1.100 --port 5051 --mic-device 0 --speaker-device 1

# With config file
glados-spoke --hub 192.168.1.100 --config spoke_config.yaml
```

## Architecture

```
┌──────────┐        WebSocket       ┌─────────────────┐
│ RPi Spoke│ ──mic──→ │ Hub (your server)
│          │ ←──speaker── │
│ • Mic    │  TTS audio   │ • llama.cpp
│ • Speaker│  PCM audio   │ • TTS synthesis
└──────────┘              └─────────────────┘
```

The spoke handles only local audio I/O. All ASR, LLM inference, and TTS happen on the hub.

## Dependencies

- Python 3.10+
- sounddevice (for mic/speaker)
- websockets (for hub communication)
- numpy (for audio processing)
- pydantic (for config)
- Optional: scipy (for resampling)