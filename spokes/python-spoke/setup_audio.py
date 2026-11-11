#!/usr/bin/env python3
"""Detect XVF3800 and configure ALSA, then start the spoke."""

import os, sys, subprocess, glob

def find_xvf3800():
    """Find XVF3800 card index via USB VID/PID (Seeed Technology 2886:001a)."""
    # Try /proc/asound/cards first
    try:
        with open('/proc/asound/cards') as f:
            for line in f:
                if 'XVF3800' in line or 'reSpeaker' in line or '3800' in line:
                    return int(line.split()[0])
    except Exception:
        pass
    
    # Fallback: scan USB devices
    try:
        for f in glob.glob('/sys/bus/usb/devices/*/idProduct'):
            try:
                with open(f) as fh:
                    if fh.read().strip() == '001a':
                        # Found it, get the card from arecord
                        result = subprocess.run(
                            ['arecord', '-l'], capture_output=True, text=True, timeout=5
                        )
                        for line in result.stdout.splitlines():
                            if '3800' in line or 'reSpeaker' in line:
                                parts = line.split()
                                for p in parts:
                                    if p.startswith('card'):
                                        return int(p.split(':')[1].split(',')[0])
            except Exception:
                pass
    except Exception:
        pass
    return None

def configure_asound(card):
    """Set up ALSA for XVF3800 capture and playback."""
    # XVF3800: 2 ch stereo input at 16kHz, playback at 16kHz
    config = f"""pcm.xvf3800_capture {{
    type plug
    slave {{
        pcm "plug:hw:{card},0"
        rate 16000
        channels 1
    }}
}}

pcm.xvf3800_playback {{
    type plug
    slave {{
        pcm "plug:hw:{card},0"
        rate 16000
        channels 1
    }}
}}

pcm.!default {{
    type plug
    slave.pcm "xvf3800_capture"
}}

ctl.!default {{
    type hw
    card {card}
}}
"""
    with open('/etc/asound.conf', 'w') as f:
        f.write(config)
    print(f"ALSA configured: XVF3800 card={card}")
    
    # Verify it works
    result = subprocess.run(
        ['arecord', '-D', 'plughw:' + str(card) + ',0', '-f', 'S16_LE',
         '-r', '16000', '-c', '1', '-d', '1', '-t', 'raw', '/dev/null'],
        capture_output=True, timeout=3
    )
    if result.returncode == 0:
        print("Audio capture verified OK")
    else:
        print(f"WARNING: Audio capture test failed: {result.stderr.decode()[:100]}")

def start_spoke():
    os.execvp('python', ['python', '/app/main.py'])

def main():
    print("GLaDOS Spoke: Detecting XVF3800...")
    card = find_xvf3800()
    if card is None:
        print("FATAL: XVF3800 not found!")
        print("Make sure it's connected via USB and powered.")
        sys.exit(1)
    
    configure_asound(card)
    start_spoke()

if __name__ == "__main__":
    main()
