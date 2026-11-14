#!/usr/bin/env python3
"""Detect audio hardware (XVF3800 or Whisplay HAT) and configure ALSA, then start the spoke."""

import os, sys, subprocess, glob


def list_alsa_cards():
    """Return list of (card_index, card_name, long_name) tuples.

    /proc/asound/cards format:
      0 [HDMI_1]: HDA-Intel - HDA ATI HDMI
                ^^^^^  ^^^^^^^^^^^^^^^^^^^^^^
                name   long_name
    """
    cards = []
    try:
        with open('/proc/asound/cards') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    idx = int(line.split()[0])
                    # Extract name from brackets: [name]
                    if '[' in line and ']' in line:
                        name = line.split('[', 1)[1].split(']', 1)[0]
                        # Long name is after ]:
                        rest = line.split(']', 1)[1].lstrip(': ').strip()
                        long_name = rest
                    else:
                        # Fallback: use first colon
                        rest = line.split(':', 1)[1].strip()
                        name = rest
                        long_name = ''
                    cards.append((idx, name, long_name))
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return cards


def find_xvf3800():
    """Find XVF3800 (reSpeaker 6-Mic Array) card index."""
    cards = list_alsa_cards()
    for idx, name, long_name in cards:
        combined = f"{name} {long_name}".lower()
        if '3800' in combined or 'respeaker' in combined:
            return idx
    return None


def find_whisplay_hat():
    """Find PiSugar Whisplay HAT card index."""
    cards = list_alsa_cards()
    for idx, name, long_name in cards:
        combined = f"{name} {long_name}".lower()
        if 'whisplay' in combined or 'wm8960' in combined:
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
    """Configure ALSA for Whisplay HAT (16 kHz mono capture/playback)."""
    config = f"""pcm.whisplay_capture {{
    type plug
    slave {{ pcm "plug:hw:{card},0"; rate 16000; channels 1 }}
}}
pcm.whisplay_playback {{
    type plug
    slave {{ pcm "plug:hw:{card},0"; rate 16000; channels 1 }}
}}
pcm.!default {{ type plug; slave.pcm "whisplay_capture" }}
ctl.!default {{ type hw; card {card} }}
"""
    with open('/etc/asound.conf', 'w') as f:
        f.write(config)
    print(f"ALSA configured: Whisplay HAT card={card}")


def verify_capture(also_device):
    """Test that audio capture works on the given device."""
    result = subprocess.run(
        ['arecord', '-D', also_device, '-f', 'S16_LE',
         '-r', '16000', '-c', '1', '-d', '1', '-t', 'raw', '/dev/null'],
        capture_output=True, timeout=3
    )
    if result.returncode == 0:
        print(f"Capture verified OK on {also_device}")
        return True
    else:
        print(f"WARNING: Capture test failed: {result.stderr.decode()[:80]}")
        return False


def select_hardware():
    """Detect available hardware and return (hw_type, card_index, card_name)."""
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
    xvf_card = find_xvf3800()
    if xvf_card is not None:
        return ('xvf3800', xvf_card)
    whisp_card = find_whisplay_hat()
    if whisp_card is not None:
        return ('whisplay', whisp_card)
    return (None, None)


def start_spoke(hw_type, card):
    """Set environment variables for main.py and exec into it."""
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
    start_spoke(hw_type, card)


if __name__ == "__main__":
    main()
