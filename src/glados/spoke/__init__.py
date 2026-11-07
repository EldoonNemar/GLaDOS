"""GLaDOS Spoke — remote audio endpoint for hub-and-spoke deployments.

Each spoke connects to a GLaDOS hub via two WebSocket connections:
- ``/microphone``  (spoke → hub): raw PCM audio captured from the local mic
- ``/speaker``     (hub → spoke): synthesized TTS audio streamed back

Features:
* Automatic reconnection with exponential backoff
* Jitter buffer for smooth playback of streamed TTS audio
* Optional AEC (acoustic echo cancellation) when mic and speaker are on the same device
* Configurable echo detection thresholds (hub text-similarity + spoke waveform)
* Native-audio-first: raw PCM sent to hub, hub runs Gemma-4 multimodal inference
"""

from .config import SpokeConfig
from .client import SpokeClient

__all__ = ["SpokeConfig", "SpokeClient"]
