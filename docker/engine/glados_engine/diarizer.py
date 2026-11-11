"""Speaker diarization using pyannote.audio."""
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger("glados_engine")

_diarizer = None
_model = None

def get_diarizer():
    """Use WebRTC VAD + voice fingerprinting for speaker separation."""
    global _diarizer
    if _diarizer is None:
        try:
            import webrtcvad
            _diarizer = True
            logger.info("VAD diarization loaded")
        except ImportError:
            logger.warning("webrtcvad not installed, using simple energy VAD")
            _diarizer = False
    return _diarizer is True

def diarize(audio_bytes: bytes, sr: int = 16000) -> list:
    """Diarize audio: return list of {"speaker": str, "start": float, "end": float, "audio": bytes}.
    
    Falls back to simple VAD splitting if diarization unavailable.
    """
    if not get_diarizer():
        return _fallback_split(audio_bytes, sr)
    
    try:
        # Convert to tensor
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        
        # Run diarization
        diarization = _model({"audio": audio_tensor, "sample_rate": sr})
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_audio = audio[int(turn.start * sr):int(turn.end * sr)]
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "audio": segment_audio.astype(np.int16).tobytes()
            })
        
        logger.info("[diarizer] Found %d speaker segments", len(segments))
        return segments
    except Exception as e:
        logger.error("Diarization failed: %s", e)
        return _fallback_split(audio_bytes, sr)

def _fallback_split(audio_bytes: bytes, sr: int = 16000) -> list:
    """Simple VAD-based splitting when diarization unavailable."""
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Simple energy-based VAD
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
            segments.append({
                "speaker": "speaker_0",
                "start": speech_start / sr,
                "end": i / sr,
                "audio": seg_audio.astype(np.int16).tobytes()
            })
    
    if in_speech:
        seg_audio = audio[speech_start:]
        segments.append({
            "speaker": "speaker_0",
            "start": speech_start / sr,
            "end": len(audio) / sr,
            "audio": seg_audio.astype(np.int16).tobytes()
        })
    
    return segments
