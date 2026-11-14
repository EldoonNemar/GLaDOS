#!/usr/bin/env python3
"""Detect audio hardware (XVF3800 or Whisplay HAT) and configure ALSA, then start the spoke."""

import os, sys, subprocess


def list_alsa_cards():
    """Return list of (card_index, card_name, long_name) tuples.
    
    Tries multiple sources in order of reliability:
    1. /proc/asound/cards (native format)
    2. arecord -l output (ALSA listing command)
    """
    # Try /proc/asound/cards first
    cards = _parse_proc_asound_cards()
    if cards:
        return cards
    
    # Fallback to arecord -l
    return _parse_arecord_list()


def _parse_proc_asound_cards():
    """Parse /proc/asound/cards format."""
    cards = []
    try:
        with open('/proc/asound/cards') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    idx = int(line.split()[0])
                    if '[' in line and ']' in line:
                        name = line.split('[', 1)[1].split(']', 1)[0]
                        rest = line.split(']', 1)[1].lstrip(': ').strip()
                        long_name = rest
                    else:
                        rest = line.split(':', 1)[1].strip()
                        name = rest
                        long_name = ''
                    cards.append((idx, name, long_name))
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return cards if cards else None


def _parse_arecord_list():
    """Parse arecord -l output format.
    
    Format:
     **** List of CAPTURE Hardware Devices ****
     card 3: whisplaysound [Whisplay Sound], device 0: Whisplay HiFi wm8960-hifi-0 [Whisplay HiFi wm8960-hifi-0]
    """
    cards = []
    try:
        result = subprocess.run(
            ['arecord', '-l'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            # Match lines like: card N: name [long_name], device ...
            import re
            match = re.match(r'\s*card\s+(\d+):\s+(\S+)\s+\[(.+?)\]', line)
            if match:
                idx = int(match.group(1))
                name = match.group(2)
                long_name = match.group(3)
                cards.append((idx, name, long_name))
    except Exception:
        pass
    return cards if cards else None


def find_xvf3800():
    """Find XVF3800 (reSpeaker 6-Mic Array) card index."""
    cards = list_alsa_cards()
    if not cards:
        return None
    for idx, name, long_name in cards:
        combined = f"{name} {long_name}".lower()
        if '3800' in combined or 'respeaker' in combined:
            return idx
    return None


def find_whisplay_hat():
    """Find PiSugar Whisplay HAT card index."""
    cards = list_alsa_cards()
    if not cards:
        return None
    for idx, name, long_name in cards:
        combined = f"{name} {long_name}".lower()
        if 'whisplay' in combined or 'wm8960' in combined:
            return idx
    return None


def find_generic_audio():
    """Find generic HD-Audio capture device (ALC1220, etc)."""
    cards = list_alsa_cards()
    if not cards:
        return None
    for idx, name, long_name in cards:
        combined = f"{name} {long_name}".lower()
        if ('hd-audio' in combined or 'hd audio' in combined) and 'generic' in combined:
            # Verify it has an Analog capture device
            result = subprocess.run(
                ['arecord', '-l'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f'card {idx}:' in line and 'Analog' in line:
                    return idx
    return None


def configure_asound_xvf3800(card):
    """Configure ALSA for XVF3800 capture (16 kHz mono)."""
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


def configure_asound_whisplay(card):
    """Configure ALSA for Whisplay HAT.
    
    Simple config using plughw for reliable format conversion.
    The 'default' pcm will be configured for capture at 16kHz mono.
    """
    config = f"""# Whisplay HAT ALSA configuration - 16kHz mono capture
pcm.!default {{
    type plug
    slave {{
        pcm "plughw:{card}"
        rate 16000
        channels 1
    }}
}}

ctl.!default {{ type hw; card {card} }}
"""
    with open('/etc/asound.conf', 'w') as f:
        f.write(config)
    print(f"ALSA configured: Whisplay HAT card={card}")


def configure_asound_generic(card):
    """Configure ALSA for generic HD-Audio capture (ALC1220, etc)."""
    config = f"""# Generic HD-Audio ALSA configuration - 16kHz mono capture
pcm.!default {{
    type plug
    slave {{
        pcm "plughw:{card},0"
        rate 16000
        channels 1
    }}
}}

ctl.!default {{ type hw; card {card} }}
"""
    with open('/etc/asound.conf', 'w') as f:
        f.write(config)
    print(f"ALSA configured: Generic HD-Audio card={card}")


def verify_capture(device):
    """Test that audio capture works on the given device."""
    result = subprocess.run(
        ['arecord', '-D', device, '-f', 'S16_LE',
         '-r', '16000', '-c', '1', '-d', '1', '-t', 'raw', '/dev/null'],
        capture_output=True, timeout=3
    )
    if result.returncode == 0:
        print(f"Capture verified OK on {device}")
        return True
    else:
        print(f"WARNING: Capture test failed: {result.stderr.decode()[:80]}")
        return False


def set_mic_gain():
    """Set Whisplay HAT mic gain to maximum for better capture quality."""
    result = subprocess.run(
        ['amixer', '-c', 'whisplaysound', 'cset', 'name=mic', '100'],
        capture_output=True, text=True, timeout=5
    )
    print(f"Mic gain set: {result.stdout.strip()}")


def select_hardware():
    """Detect available hardware and return (hw_type, card_index)."""
    hw_env = os.environ.get('GLADOS_AUDIO_DEVICE', '').strip().lower()
    if hw_env in ('xvf3800', 'xvf', 'reSpeaker', 're speaker'):
        card = find_xvf3800()
        if card is not None:
            return ('xvf3800', card)
        print(f"WARNING: GLADOS_AUDIO_DEVICE=xvf3800 but XVF3800 not found")
    elif hw_env in ('whisplay', 'whisplaysound', 'wm8960', 'pi sugar'):
        card = find_whisplay_hat()
        if card is not None:
            return ('whisplay', card)
        print(f"WARNING: GLADOS_AUDIO_DEVICE=whisplay but Whisplay HAT not found")
    
    # Auto-detect: prefer XVF3800, then Whisplay, then Generic
    xvf_card = find_xvf3800()
    if xvf_card is not None:
        return ('xvf3800', xvf_card)
    
    whisp_card = find_whisplay_hat()
    if whisp_card is not None:
        return ('whisplay', whisp_card)
    
    # Fallback: generic HD-Audio
    generic_card = find_generic_audio()
    if generic_card is not None:
        return ('generic', generic_card)
    
    return (None, None)


def start_spoke(hw_type, card):
    """Set environment variables for main.py and exec into it."""
    if hw_type == 'whisplay':
        os.environ['GLADOS_CAPTURE_DEVICE'] = 'plughw:whisplaysound'
        os.environ['GLADOS_PLAYBACK_DEVICE'] = 'plughw:whisplaysound'
    elif hw_type == 'xvf3800':
        os.environ['GLADOS_CAPTURE_DEVICE'] = f'plughw:{card},0'
        os.environ['GLADOS_PLAYBACK_DEVICE'] = f'plughw:{card},0'
    elif hw_type == 'generic':
        os.environ['GLADOS_CAPTURE_DEVICE'] = f'plughw:{card},0'
        os.environ['GLADOS_PLAYBACK_DEVICE'] = f'plughw:{card},0'
    os.environ['GLADOS_HW_TYPE'] = hw_type
    os.execvp('python', ['python', '/app/main.py'])


def main():
    print("GLaDOS Spoke: Detecting audio hardware...")
    hw_type, card = select_hardware()
    if hw_type is None:
        print("FATAL: No supported audio hardware found!")
        print("\nSupported devices:")
        print("  - XVF3800 (reSpeaker 6-Mic Array)")
        print("  - PiSugar Whisplay HAT (WM8960)")
        print("\nOverride with: GLADOS_AUDIO_DEVICE=xvf3800 or GLADOS_AUDIO_DEVICE=whisplay")
        print("\nAvailable ALSA devices:")
        try:
            result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=5)
            print(result.stdout if result.stdout else "None found")
        except Exception:
            print("Error listing devices")
        sys.exit(1)
    print(f"Detected: {hw_type} card={card}")
    if hw_type == 'xvf3800':
        configure_asound_xvf3800(card)
        verify_capture(f'plughw:{card},0')
    elif hw_type == 'whisplay':
        configure_asound_whisplay(card)
        verify_capture(f'plughw:{card},0')
        set_mic_gain()
    elif hw_type == 'generic':
        configure_asound_generic(card)
        verify_capture(f'plughw:{card},0')
    start_spoke(hw_type, card)


if __name__ == "__main__":
    main()
