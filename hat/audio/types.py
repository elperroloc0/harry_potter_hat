from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz — the wake/VAD frame contract


@dataclass(frozen=True)
class Phrase:
    """A recorded, VAD-trimmed utterance, ready for STT."""

    pcm: np.ndarray  # int16 mono, sample_rate Hz
    sample_rate: int = SAMPLE_RATE
    started_at: float = 0.0
    duration_s: float = 0.0


@dataclass(frozen=True)
class WakeEvent:
    model_name: str
    score: float
    at: float


@dataclass(frozen=True)
class Transcript:
    """Produced by STT, consumed by the brain. The one shared contract both
    sides import — do not redefine this elsewhere."""

    text: str
    lang: str  # "es" | "en"
    lang_confidence: float = 1.0
    latency_s: float = 0.0


@dataclass(frozen=True)
class Utterance:
    transcript: Transcript
    phrase: Phrase
    wake: WakeEvent | None = None
