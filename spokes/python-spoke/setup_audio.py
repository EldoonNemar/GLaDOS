#!/usr/bin/env python3
"""Detect XVF3800 and configure ALSA, then start the spoke."""

import os, sys, subprocess, glob

def find_xvf3800():
    """Find XVF3800 card index.
    
    Tries multiple methods:
    1. /proc/asound/cards (if mounted)
    2. /dev/snd/pcmC4D* (known XVF3800 device files)
    3. arecord -l output
    """
    # Method 1: /proc/asound/cards
    try:
        with open('/proc/asound/cards') as f:
            for line in f:
                if '3800' in line or 'reSpeaker' in line:
                    return int(line.split()[0])
    except Exception:
        pass
    
    # Method 2: Check for XVF3800 device files (card 4)
    for card_num in range(10):
        capture = f'/dev/snd/pcmC{card_num}D0c'
        if os.path.exists(capture):
            # Check if it's an XVF3800 by trying to list it
            try:
                result = subprocess.run(
                    ['arecord', '-l'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines():
                    if f'card {card_num}' in line and ('3800' in line or 'reSpeaker' in line):
                        return card_num
            except Exception:
                pass
    
    # Method 3: Scan USB
    try:
        for f in glob.glob('/sys/bus/usb/devices/*/idProduct'):
            with open(f) as fh:
                if fh.read().strip() == '001a':  # XVF3800 USB VID/PID
                    result = subprocess.run(
                        ['arecord', '-l'],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.splitlines():
                        if '3800' in line or 'reSpeaker' in line:
                            for p in line.split():
                                if p.startswith('card'):
                                    return int(p.split(':')[1].split(',')[0])
    except Exception:
        pass
    
    return None

def configure_asound(card):
    """Configure ALSA for XVF3800 capture and playback."""
    config = f"""pcm.xvf3800_capture {{
    type plug
    slave {{ pcm "plug:hw:{card},0"; rate 16000; channels 1 }}
}}
pcm.xvf3800_playback {{
    type plug
    slave {{ pcm "plug:hw:{card},0"; rate 16000; channels 1 }}
}}
pcm.!default {{ type plug; slave.pcm "xvf3800_capture" }}
ctl.!default {{ type hw; card {card} }}
"""
    with open('/etc/asound.conf', 'w') as f:
        f.write(config)
    print(f"ALSA configured: XVF3800 card={card}")
    
    # Verify
    result = subprocess.run(
        ['arecord', '-D', f'plughw:{card},0', '-f', 'S16_LE',
         '-r', '16000', '-c', '1', '-d', '1', '-t', 'raw', '/dev/null'],
        capture_output=True, timeout=3
    )
    if result.returncode == 0:
        print("Audio capture verified OK")
    else:
        print(f"WARNING: Capture test failed: {result.stderr.decode()[:80]}")

def start_spoke():
    os.execvp('python', ['python', '/app/main.py'])

def main():
    print("GLaDOS Spoke: Detecting XVF3800...")
    card = find_xvf3800()
    if card is None:
        print("FATAL: XVF3800 not found!")
        print("Available ALSA devices:")
        try:
            result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=5)
            print(result.stdout if result.stdout else "None found")
        except Exception:
            print("Error listing devices")
        sys.exit(1)
    
    configure_asound(card)
    start_spoke()

if __name__ == "__main__":
    main()
