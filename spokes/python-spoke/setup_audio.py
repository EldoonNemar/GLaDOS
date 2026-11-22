#!/usr/bin/env python3
"""Detect the audio hardware (XVF3800 mic array or Whisplay/wm8960 HAT),
configure ALSA, and start the spoke.

Hardware selection:
  GLADOS_AUDIO_DEVICE=xvf3800|whisplay  -> force a specific board
  (unset)                                -> auto-detect, XVF3800 wins
"""

import os
import re
import subprocess
import sys


def _parse_arecord_list():
    """Parse `arecord -l` into [(card, name, longname), ...].

    Example line:
      card 3: whisplaysound [Whisplay Sound], device 0: ... [Whisplay HiFi wm8960-hifi-0]
    -> (3, 'whisplaysound', 'Whisplay Sound')

    Returns None on empty output or subprocess failure.
    """
    try:
        result = subprocess.run(
            ['arecord', '-l'],
            capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    cards = []
    for line in result.stdout.splitlines():
        m = re.match(r'\s*card\s+(\d+):\s+(\S+)\s+\[([^\]]+)\]', line)
        if m:
            cards.append((int(m.group(1)), m.group(2), m.group(3)))
    return cards if cards else None


def list_alsa_cards():
    """List ALSA capture cards as [(card, name, longname), ...].

    Prefers /proc/asound/cards (no subprocess); falls back to arecord -l.
    """
    try:
        cards = []
        with open('/proc/asound/cards') as f:
            for line in f:
                m = re.match(r'\s*(\d+)\s+\[([^\]]+)\]:\s+(.*)', line)
                if m:
                    cards.append((int(m.group(1)), m.group(2), m.group(3).strip()))
        return cards
    except Exception:
        pass
    return _parse_arecord_list() or []


def find_xvf3800():
    """Return the card index of an XVF3800 / reSpeaker mic array, or None."""
    for card, name, longname in list_alsa_cards():
        hay = f"{name} {longname}".lower()
        if '3800' in hay or 'respeaker' in hay:
            return card
    return None


def find_whisplay_hat():
    """Return the card index of a Whisplay/wm8960 HAT, or None."""
    for card, name, longname in list_alsa_cards():
        hay = f"{name} {longname}".lower()
        if 'whisplay' in hay or 'wm8960' in hay:
            return card
    return None


def select_hardware():
    """Pick (hw_type, card) via env override or auto-detection.

    Auto-detect order: XVF3800 first, then Whisplay. (None, None) if
    nothing suitable is present.
    """
    forced = os.environ.get('GLADOS_AUDIO_DEVICE', '').strip().lower()
    if forced in ('xvf3800', 'whisplay'):
        card = find_xvf3800() if forced == 'xvf3800' else find_whisplay_hat()
        return (forced, card)
    card = find_xvf3800()
    if card is not None:
        return ('xvf3800', card)
    card = find_whisplay_hat()
    if card is not None:
        return ('whisplay', card)
    return (None, None)


def configure_asound(card):
    """Write /etc/asound.conf routing the default device to this card.

    Best-effort: a failure here is non-fatal (plughw: works without it).
    """
    config = f"""pcm.!default {{
    type plug
    slave.pcm "plughw:{card},0"
    rate 16000
}}
ctl.!default {{ type hw; card {card} }}
"""
    try:
        with open('/etc/asound.conf', 'w') as f:
            f.write(config)
        print(f"ALSA configured for card={card}")
    except Exception as e:
        print(f"WARNING: could not write /etc/asound.conf: {e}")


def set_mic_gain():
    """Max out the Whisplay HAT mic gain for better capture (best-effort)."""
    result = subprocess.run(
        ['amixer', '-c', 'whisplaysound', 'cset', 'name=mic', '100'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        print(f"Whisplay mic gain: {result.stdout.strip()}")
    else:
        print("WARNING: could not set Whisplay mic gain")


def start_spoke(hw, card):
    """Export the audio device env vars and exec the spoke."""
    os.environ['GLADOS_HW_TYPE'] = hw
    os.environ['GLADOS_CAPTURE_DEVICE'] = f'plughw:{card},0'
    os.environ['GLADOS_PLAYBACK_DEVICE'] = f'plughw:{card},0'
    os.execvp('python', ['python', '/app/main.py'])


def main():
    print("GLaDOS Spoke: detecting audio hardware...")
    hw, card = select_hardware()
    if card is None:
        print(f"FATAL: no supported audio device found (forced={hw or 'auto'})!")
        print("Available ALSA cards:")
        for c in list_alsa_cards():
            print(f"  card {c[0]}: {c[1]} [{c[2]}]")
        sys.exit(1)
    print(f"Using {hw} on card {card}")
    configure_asound(card)
    if hw == 'whisplay':
        set_mic_gain()
    start_spoke(hw, card)


if __name__ == '__main__':
    main()
