"""Configuration for the GLaDOS spoke client."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpokeConfig(BaseModel):
    """Configuration for a GLaDOS spoke endpoint.

    All values are configurable so spokes on different networks / hardware
    can be tuned independently without editing source code.
    """

    # --- Hub connection ---
    hub_host: str = Field(default="127.0.0.1", description="Hub hostname or IP.")
    hub_port: int = Field(default=5051, ge=1, le=65535, description="Hub WebSocket port.")
    mic_path: str = Field(default="/microphone", description="Hub microphone endpoint.")
    speaker_path: str = Field(default="/speaker", description="Hub speaker endpoint.")

    # --- Reconnection ---
    reconnect_base_delay: float = Field(
        default=1.0, ge=0.1, le=30.0,
        description="Base delay (seconds) before first reconnect attempt.",
    )
    reconnect_max_delay: float = Field(
        default=30.0, ge=1.0,
        description="Maximum reconnect backoff delay.",
    )
    reconnect_jitter: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Random jitter fraction added to each reconnect delay.",
    )
    reconnect_attempts: int | None = Field(
        default=None, ge=1,
        description="Max reconnect attempts before giving up. None = infinite.",
    )

    # --- Audio capture (mic → hub) ---
    mic_sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Capture sample rate.")
    mic_channels: int = Field(default=1, ge=1, le=2, description="Capture channels (1=mono, 2=stereo).")
    mic_chunk_frames: int = Field(
        default=512, ge=128, le=8192,
        description="Frames per WebSocket send (512 @ 16 kHz = 32 ms).",
    )
    mic_device: int | str | None = Field(
        default=None, description="Input device index or name. None = system default.",
    )
    mic_volume: float = Field(default=1.0, ge=0.0, le=5.0, description="Mic software volume multiplier.")

    # --- Audio playback (hub → spoke) ---
    spk_sample_rate: int = Field(default=22050, ge=8000, le=48000, description="TTS playback sample rate.")
    spk_device: int | str | None = Field(
        default=None, description="Output device index or name. None = system default.",
    )

    # --- Jitter buffer ---
    jitter_buffer_min_ms: int = Field(
        default=80, ge=10, le=500,
        description="Minimum buffer depth (ms). Smaller = lower latency, more choppy.",
    )
    jitter_buffer_max_ms: int = Field(
        default=400, ge=50, le=2000,
        description="Maximum buffer depth (ms). Larger = smoother, more latency.",
    )
    jitter_adapt_speed: float = Field(
        default=0.1, ge=0.01, le=1.0,
        description="How aggressively the jitter buffer adapts (0..1).",
    )

    # --- Echo detection (configurable) ---
    # Hub-side: text similarity between ASR transcription and recently spoken text.
    echo_text_similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Hub: Levenshtein similarity above which an ASR transcription is considered echo.",
    )
    echo_text_min_length: int = Field(
        default=4, ge=1,
        description="Hub: Minimum transcription length (chars) to trigger echo check.",
    )
    echo_text_recent_turns: int = Field(
        default=3, ge=1, le=10,
        description="Hub: How many recent assistant turns to compare against.",
    )
    # Spoke-side: waveform correlation between output audio and mic input (for local AEC).
    echo_waveform_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Spoke: normalized cross-correlation above which playback is considered echo.",
    )
    echo_aec_enabled: bool = Field(
        default=False,
        description="Enable basic AEC on the spoke (remove speaker bleed from mic).",
    )
    echo_aec_order: int = Field(
        default=256, ge=16, le=4096,
        description="AEC filter order (number of taps). Higher = more accurate, more CPU.",
    )
    echo_mute_on_detection_ms: int = Field(
        default=0, ge=0, le=2000,
        description="If > 0, mute mic for this many ms after echo is detected.",
    )

    # --- Native audio mode ---
    native_audio: bool = Field(
        default=True,
        description="Send raw PCM to hub for native multimodal inference (bypasses ASR).",
    )
    native_audio_max_duration_s: float = Field(
        default=30.0, ge=1.0, le=120.0,
        description="Max continuous capture duration before a new inference round starts.",
    )

    # --- Logging / misc ---
    log_level: str = Field(default="INFO", description="Logging level.")
    heartbeat_interval_s: float = Field(
        default=5.0, ge=1.0, le=60.0,
        description="Heartbeat interval (seconds). Sent on both mic and speaker WS.",
    )
