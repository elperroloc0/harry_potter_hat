from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

Profile = Literal["mac", "pi"]


@dataclass(frozen=True)
class ServoCal:
    """SG90 pulse-width calibration. Placeholder values — recalibrate on the bench
    once the PCA9685 and servos arrive; nothing else in the codebase should need
    to change when these numbers do."""

    mouth_closed_us: int = 1000
    mouth_open_us: int = 1900
    brow_rest_us: int = 1400
    brow_raised_us: int = 1750
    pca9685_freq_hz: int = 50
    mouth_channel: int = 0
    brow_left_channel: int = 1
    brow_right_channel: int = 2
    max_slew_deg_per_s: float = 600.0


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    elevenlabs_api_key: str
    ollama_url: str
    vision_model: str
    claude_model: str
    languages: tuple[str, ...]
    default_lang: str
    profile: Profile
    camera_index: int
    vision_timeout_s: float
    listen_timeout_s: float
    session_max_s: float
    max_reply_tokens: int
    tts_backend: Literal["elevenlabs", "say"]
    servo_backend: Literal["mock", "pca9685"]
    output_latency_s: float
    stt_model_size: str
    wake_model: str
    elevenlabs_voice_id: str
    servo: ServoCal = field(default_factory=ServoCal)


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> Settings:
    load_dotenv()

    languages = tuple(
        lang.strip() for lang in os.environ.get("LANGUAGES", "es,en").split(",") if lang.strip()
    )

    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        vision_model=os.environ.get("VISION_MODEL", "qwen3-vl:8b"),
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-5"),
        languages=languages or ("es", "en"),
        default_lang=os.environ.get("DEFAULT_LANG", "es"),
        profile=os.environ.get("PROFILE", "mac"),  # type: ignore[arg-type]
        camera_index=int(os.environ.get("CAMERA_INDEX", "0")),
        vision_timeout_s=float(os.environ.get("VISION_TIMEOUT_S", "10.0")),
        listen_timeout_s=float(os.environ.get("LISTEN_TIMEOUT_S", "8.0")),
        session_max_s=float(os.environ.get("SESSION_MAX_S", "300.0")),
        max_reply_tokens=int(os.environ.get("MAX_REPLY_TOKENS", "500")),
        tts_backend=os.environ.get("HAT_TTS_BACKEND", "say"),  # type: ignore[arg-type]
        servo_backend=os.environ.get("HAT_SERVO_BACKEND", "mock"),  # type: ignore[arg-type]
        output_latency_s=float(os.environ.get("HAT_OUTPUT_LATENCY_S", "0.0")),
        stt_model_size=os.environ.get("HAT_STT_MODEL", "small"),
        wake_model=os.environ.get("HAT_WAKE_MODEL", "hey_jarvis_v0.1"),
        elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
    )


settings = load_settings()
