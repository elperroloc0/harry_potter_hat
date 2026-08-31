from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

Lang = Literal["es", "en"]


@dataclass(frozen=True)
class PcmAudio:
    """Canonical audio type crossing every TTS/playback/lipsync boundary."""

    samples: np.ndarray  # int16 mono
    sample_rate: int = 22050

    def __post_init__(self) -> None:
        if self.samples.dtype != np.int16:
            raise ValueError(f"PcmAudio requires int16 samples, got {self.samples.dtype}")
        if self.samples.ndim != 1:
            raise ValueError(f"PcmAudio requires mono (1-D) samples, got shape {self.samples.shape}")

    @property
    def duration_s(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return len(self.samples) / self.sample_rate


class SynthesisError(RuntimeError):
    """Raised by a Synthesizer backend on failure. HatVoice catches this and degrades."""


class Synthesizer(ABC):
    @abstractmethod
    def synth(self, text: str, lang: Lang) -> PcmAudio: ...
