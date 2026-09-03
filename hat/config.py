from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

Profile = Literal["mac", "pi"]


@dataclass(frozen=True)
class ServoAxis:
    """One servo's own duty-cycle calibration, found by eye with
    hat.tools.find_servo_range. Per servo and not shared, because the two
    do not agree: horizontal is 9.00% on the mouth and 10.75% on the brows,
    on otherwise identical hardware.

    centre_duty is horizontal, and it is measured rather than taken as the
    midpoint of the band -- a horn is pressed onto splines and its zero
    lands wherever the teeth allow, so the usable band is not centred on it.
    That offset is what makes travel lopsided if ignored: the mouth has 2.5%
    of duty below horizontal against 3.5% above, the brows 4.25% against
    1.75%. Motion is therefore made symmetric about centre_duty, reaching
    only as far as the nearer edge allows, which leaves part of the band
    unused on the roomier side. Re-measure after re-seating a horn.

    min_duty/max_duty is the band where the servo actually tracks. Past it
    there is no mechanical stop -- it loses tracking and turns in continuous
    full rotations, which is what made the hat unusable when these were set
    to a nominal 1.5%-14% and the resting angle sat at min_duty, leaving the
    horn free-running in silence. The 12.5% ceiling is where the measuring
    tool stops, not where the servo does: it still tracked there while 14%
    free-ran, so there is a little room above.
    """

    channel: int
    centre_duty: float
    min_duty: float = 0.065
    max_duty: float = 0.125


@dataclass(frozen=True)
class ServoCal:
    """MG90S calibration for the PCA9685 rig, measured by hand (see
    hat/motion/servos.py:PCA9685Servos for the raw duty-cycle math). Only two
    physical servos exist -- mouth, and a single brow servo driving both
    eyebrows together through one linkage.

    rest_deg is where the horns park: 90, i.e. each axis's centre_duty, so an
    idle servo sits horizontal rather than against an end.

    travel_deg scales the sweep: 90 uses the full symmetric reach either side
    of horizontal. The pushrods have roughly 45 degrees of clearance before
    they would foul the servo body, but that is not the binding constraint --
    these servos turn far less than commanded, moving the horn only about 15
    degrees for a full sweep, so the clearance is never approached.
    """

    mouth: "ServoAxis" = field(default_factory=lambda: ServoAxis(channel=15, centre_duty=0.0900))
    brow: "ServoAxis" = field(default_factory=lambda: ServoAxis(channel=11, centre_duty=0.1075))
    pca9685_freq_hz: int = 50
    max_slew_deg_per_s: float = 600.0
    rest_deg: float = 90.0
    travel_deg: float = 90.0


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
    seat_timeout_s: float
    max_reply_tokens: int
    tts_backend: Literal["elevenlabs", "say"]
    servo_backend: Literal["mock", "pca9685"]
    output_latency_s: float
    audio_output_device: str | None
    stt_model_size: str
    wake_model: str
    wake_phrases: tuple[str, ...]
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

    # Spoken names the hat answers to when the openwakeword stack is absent
    # (it has no aarch64 wheel, so on the Pi it always is -- see
    # hat.audio.listener). Matched against a quick unconstrained Whisper pass,
    # so they can be in any language, unlike the ritual itself. The Latin
    # spellings of the Russian name are insurance for when Whisper hears it
    # in a Spanish or English context and writes it out phonetically.
    wake_phrases = tuple(
        phrase.strip().lower()
        for phrase in os.environ.get(
            "HAT_WAKE_PHRASES", "шляпа,shlyapa,shliapa,sombrero,sorting hat"
        ).split(",")
        if phrase.strip()
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
        # How long sort_visitor waits for the chair before giving up. Long
        # enough for a hesitant child to be coaxed into it, short enough that
        # a wandered-off one does not strand the ceremony.
        session_max_s=float(os.environ.get("SESSION_MAX_S", "300.0")),
        seat_timeout_s=float(os.environ.get("SEAT_TIMEOUT_S", "60.0")),
        max_reply_tokens=int(os.environ.get("MAX_REPLY_TOKENS", "500")),
        tts_backend=os.environ.get("HAT_TTS_BACKEND", "say"),  # type: ignore[arg-type]
        servo_backend=os.environ.get("HAT_SERVO_BACKEND", "mock"),  # type: ignore[arg-type]
        output_latency_s=float(os.environ.get("HAT_OUTPUT_LATENCY_S", "0.0")),
        # Playback device name or index for sounddevice. Empty means "pick
        # one" -- see hat.audio.player.default_output_device, which prefers
        # PipeWire's "pulse" device over a raw ALSA card.
        audio_output_device=os.environ.get("HAT_AUDIO_OUTPUT_DEVICE") or None,
        stt_model_size=os.environ.get("HAT_STT_MODEL", "small"),
        wake_model=os.environ.get("HAT_WAKE_MODEL", "hey_jarvis_v0.1"),
        wake_phrases=wake_phrases,
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
