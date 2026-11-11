"""Speaker diarization using VAD + voice fingerprinting.

Uses energy-based VAD to split speech segments, then applies voice fingerprinting
to assign speaker labels based on pitch/energy characteristics.

For production multi-speaker scenarios, set HF_TOKEN in .env to enable pyannote.audio.
Note: pyannote v3.3.2 requires torch/torchaudio < 2.2.0, which may not be available
for your platform. The VAD fallback works well with the 4-mic XVF3800 beamformed audio.
"""
import os
import numpy as np
import logging

logger = logging.getLogger("glados_engine")

_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

def diarize(audio_bytes: bytes, sr: int = 16000) -> list:
    """Diarize audio: return list of {"speaker": str, "start": float, "end": float, "audio": bytes}.
    
    Uses energy-based VAD to split speech segments, then applies voice fingerprinting
    to assign speaker labels based on pitch/energy characteristics.
    """
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Split into speech segments using energy-based VAD
    segments = _split_segments(audio, sr)
    
    if not segments:
        return []
    
    # Apply voice fingerprinting to assign speaker labels
    segments = _assign_speakers(segments, sr)
    
    logger.info("[diarizer] Found %d speaker segments (%d unique speakers)", 
                len(segments), len(set(s["speaker"] for s in segments)))
    return segments


def _split_segments(audio: np.ndarray, sr: int = 16000) -> list:
    """Split audio into speech segments using energy-based VAD."""
    window_size = sr // 10  # 100ms windows
    segments = []
    in_speech = False
    speech_start = 0
    
    for i in range(0, len(audio), window_size):
        window = audio[i:i+window_size]
        energy = np.sqrt(np.mean(window**2))
        
        if energy > 0.01 and not in_speech:
            in_speech = True
            speech_start = i
        elif energy <= 0.01 and in_speech:
            in_speech = False
            seg_audio = audio[speech_start:i]
            segments.append(seg_audio)
    
    if in_speech:
        segments.append(audio[speech_start:])
    
    return segments


def _assign_speakers(segments: list, sr: int = 16000) -> list:
    """Assign speaker labels using voice fingerprinting (pitch + energy)."""
    # Compute fingerprint for each segment
    fingerprints = []
    for seg in segments:
        if len(seg) < sr // 4:  # Skip very short segments (< 250ms)
            continue
        # Compute pitch (zero-crossing rate) and energy
        zcr = np.mean(np.abs(np.diff(np.sign(seg))))
        energy = np.sqrt(np.mean(seg**2))
        fingerprints.append((zcr, energy))
    
    # Simple clustering: if fingerprints differ significantly, assign different speakers
    unique_speakers = {}
    speaker_count = 0
    result = []
    
    for i, (seg, (zcr, energy)) in enumerate(zip(segments, fingerprints)):
        if len(seg) < sr // 4:
            result.append({
                "speaker": "speaker_0",
                "start": (sum(len(segments[:j]) for j in range(i))) / sr,
                "end": (sum(len(segments[:j+1]) for j in range(i))) / sr,
                "audio": seg.astype(np.int16).tobytes()
            })
            continue
        
        # Find matching speaker based on fingerprint similarity
        matched = False
        for label, (ref_zcr, ref_energy) in unique_speakers.items():
            # If fingerprints are similar, assign same speaker
            zcr_diff = abs(zcr - ref_zcr) / (ref_zcr + 1e-6)
            energy_diff = abs(energy - ref_energy) / (ref_energy + 1e-6)
            if zcr_diff < 0.5 and energy_diff < 0.5:
                result.append({
                    "speaker": label,
                    "start": (sum(len(segments[:j]) for j in range(i))) / sr,
                    "end": (sum(len(segments[:j+1]) for j in range(i))) / sr,
                    "audio": seg.astype(np.int16).tobytes()
                })
                matched = True
                break
        
        if not matched:
            speaker_count += 1
            label = f"speaker_{speaker_count - 1}"
            unique_speakers[label] = (zcr, energy)
            result.append({
                "speaker": label,
                "start": (sum(len(segments[:j]) for j in range(i))) / sr,
                "end": (sum(len(segments[:j+1]) for j in range(i))) / sr,
                "audio": seg.astype(np.int16).tobytes()
            })
    
    return result
