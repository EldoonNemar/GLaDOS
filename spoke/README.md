# GLaDOS Spoke Client — Raspberry Pi

Lightweight voice endpoint for the GLaDOS hub-and-spoke system.

## Install

### Option 1: Virtual environment (recommended)

```bash
# Clone the repo if you haven't already
git clone https://github.com/EldoonNemar/GLaDOS.git
cd GLaDOS/spoke

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the spoke package
pip install -e .
```

### Option 2: pipx (for CLI tools)

```bash
pipx install glados-spoke
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

## Systemd Service (always-on, e.g. Raspberry Pi)

Create `/etc/systemd/system/glados-spoke.service`:

```ini
[Unit]
Description=GLaDOS Spoke Client
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cphipps
WorkingDirectory=/home/cphipps/GLaDOS/spoke
ExecStart=/home/cphipps/GLaDOS/spoke/.venv/bin/python -m glados_spoke.client --hub 192.168.1.100 --port 5051
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable glados-spoke
sudo systemctl start glados-spoke
sudo systemctl status glados-spoke   # verify it's running
```