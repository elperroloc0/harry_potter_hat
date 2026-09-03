from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

Profile = Literal["mac", "pi"]


@dataclass(frozen=True)
class ServoCal:
    """MG90S calibration for the PCA9685 rig, bench-tested by hand (see
    hat/motion/servos.py:PCA9685Servos for the raw duty-cycle math). Only two
    physical servos exist -- mouth and a single brow servo driving both
    eyebrows together through one linkage -- on the channels below. Nothing
    else in the codebase should need to change when these numbers do.

    min_duty/max_duty is the bench-confirmed safe PCA9685 duty-cycle range
    for an MG90S: outside roughly 0.2%-20% the servo doesn't hit a
    mechanical stop, it loses tracking and spins continuously, so 1.5%-14%
    (covering the full 0-180 degree sweep with margin on both sides) is a
    hard limit, not a starting guess.
    """

    mouth_channel: int = 15
    brow_channel: int = 11
    pca9685_freq_hz: int = 50
    min_duty: float = 0.015
    max_duty: float = 0.14
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
    motion_sensor_pin: int
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
    elevenlabs_model_id: str
    elevenlabs_stability: float
    elevenlabs_similarity_boost: float
    elevenlabs_style: float
    elevenlabs_speaker_boost: bool
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
        vision_model=os.environ.get("VISION_MODEL", "qwen2.5vl:7b"),
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-5"),
        languages=languages or ("es", "en"),
        default_lang=os.environ.get("DEFAULT_LANG", "es"),
        profile=os.environ.get("PROFILE", "mac"),  # type: ignore[arg-type]
        camera_index=int(os.environ.get("CAMERA_INDEX", "0")),
        # BCM GPIO pin the PIR sensor's OUT line is wired to. 17 (physical
        # pin 11) is what's actually wired and bench-tested; change to
        # match if you rewire it. Avoid the I2C pins (2/3, used by the
        # PCA9685).
        motion_sensor_pin=int(os.environ.get("HAT_MOTION_SENSOR_PIN", "17")),
        # qwen2.5vl:7b (default since 2026-09-02) has no hidden reasoning
        # pass, unlike qwen3-vl:8b which this replaced -- measured live
        # against a real, deliberately hard photo (dim light, out of focus,
        # awkward angle) at 1.7-2.1s. 20s leaves real margin for a slower
        # network/cold-start case without reintroducing the multi-minute
        # worst case a thinking model risked. If VISION_MODEL is overridden
        # back to a thinking model, raise this back toward 75s -- see
        # project_hat_hardware_status memory for the full comparison.
        vision_timeout_s=float(os.environ.get("VISION_TIMEOUT_S", "20.0")),
        listen_timeout_s=float(os.environ.get("LISTEN_TIMEOUT_S", "8.0")),
        session_max_s=float(os.environ.get("SESSION_MAX_S", "300.0")),
        max_reply_tokens=int(os.environ.get("MAX_REPLY_TOKENS", "500")),
        tts_backend=os.environ.get("HAT_TTS_BACKEND", "say"),  # type: ignore[arg-type]
        servo_backend=os.environ.get("HAT_SERVO_BACKEND", "mock"),  # type: ignore[arg-type]
        output_latency_s=float(os.environ.get("HAT_OUTPUT_LATENCY_S", "0.0")),
        stt_model_size=os.environ.get("HAT_STT_MODEL", "small"),
        wake_model=os.environ.get("HAT_WAKE_MODEL", "hey_jarvis_v0.1"),
        elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
        # Voice-character knobs -- tune by ear, no code changes needed.
        # stability: lower = more expressive but prone to wavering/glitches
        #   on loud, emphatic lines (the shouted house name); higher = more
        #   consistent/controlled, at some cost of flatness. 0.4 (the SDK-ish
        #   "default" many examples use) was too low for a shouted
        #   proclamation; 0.65 trades a little spontaneity for reliability.
        # style: exaggerates the voice's natural character/emotion on top of
        #   the stability baseline -- more "epic," but stacking a high style
        #   with low stability is the classic recipe for artifacts, so raise
        #   both together.
        elevenlabs_model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
        elevenlabs_stability=float(os.environ.get("ELEVENLABS_STABILITY", "0.65")),
        elevenlabs_similarity_boost=float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", "0.8")),
        elevenlabs_style=float(os.environ.get("ELEVENLABS_STYLE", "0.35")),
        elevenlabs_speaker_boost=_bool_env("ELEVENLABS_SPEAKER_BOOST", True),
    )


settings = load_settings()
