from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import numpy as np

from hat.audio.types import Phrase, Transcript, Utterance


class FakeVoiceInput:
    """Test double for the real audio-in facade (hat.audio.listener). Reads
    typed "text,lang" lines from stdin instead of listening to a microphone,
    so the brain/orchestrator can be developed and demoed without any audio
    hardware or models. Satisfies the same shape main.py expects from the
    real listener: wait_for_wake / listen_once / hold."""

    def wait_for_wake(
        self,
        timeout: float | None = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> bool:
        # `cancel` is accepted for interface parity and ignored: this stub
        # blocks on stdin, so there is nothing to poll between frames. In
        # --text runs you simulate a visitor sitting down by typing /sit at
        # the `you>` prompt instead.
        input("[press Enter to simulate the wake word] ")
        return True

    def listen_once(
        self,
        timeout: float = 8.0,
        max_phrase_s: float = 15.0,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> Utterance | None:
        raw = input("you> ").strip()
        if not raw:
            return None
        lang = "es"
        if "|" in raw:
            raw, _, lang = raw.rpartition("|")
            lang = lang.strip() or "es"
        text = raw.strip()
        if not text:
            return None
        phrase = Phrase(pcm=np.zeros(1, dtype=np.int16), started_at=time.monotonic())
        transcript = Transcript(text=text, lang=lang)
        return Utterance(transcript=transcript, phrase=phrase)

    @contextmanager
    def hold(self) -> Iterator[None]:
        yield
