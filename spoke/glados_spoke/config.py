"""Configuration for the GLaDOS spoke client."""

from __future__ import annotations

from pydantic import BaseModel, Field





class SpokeConfig(BaseModel):

    """Configuration for a GLaDOS spoke endpoint."""



    hub_host: str = Field(default="127.0.0.1", description="Hub hostname or IP.")

    hub_port: int = Field(default=5051, ge=1, le=65535, description="Hub WebSocket port.")

    mic_path: str = Field(default="/microphone")

    speaker_path: str = Field(default="/speaker")

    reconnect_base_delay: float = Field(default=1.0, ge=0.1, le=30.0)

    reconnect_max_delay: float = Field(default=30.0, ge=1.0)

    reconnect_jitter: float = Field(default=0.5, ge=0.0, le=1.0)

    reconnect_attempts: int | None = Field(default=None)

    mic_sample_rate: int = Field(default=16000, ge=8000, le=48000)

    mic_channels: int = Field(default=1, ge=1, le=2)

    mic_chunk_frames: int = Field(default=512, ge=128, le=8192)

    mic_device: int | str | None = Field(default=None)

    mic_volume: float = Field(default=1.0, ge=0.0, le=5.0)

    spk_sample_rate: int = Field(default=22050, ge=8000, le=48000)

    spk_device: int | str | None = Field(default=None)

    jitter_buffer_min_ms: int = Field(default=80, ge=10, le=500)

    jitter_buffer_max_ms: int = Field(default=400, ge=50, le=2000)

    jitter_adapt_speed: float = Field(default=0.1, ge=0.01, le=1.0)

    echo_aec_enabled: bool = Field(default=False)

    echo_aec_order: int = Field(default=256, ge=16, le=4096)

    echo_mute_on_detection_ms: int = Field(default=0, ge=0, le=2000)

    log_level: str = Field(default="INFO")

    heartbeat_interval_s: float = Field(default=5.0, ge=1.0, le=60.0)



