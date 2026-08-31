"""Speech-to-text via faster-whisper, constrained to the hat's supported
languages.

Whisper's own language detection is run first (over the *whole* language
vocabulary), then filtered down to just ``allowed_langs`` and renormalized,
so the winning language is always one the hat can actually speak/behave
in even if a third language leaks into the probability mass.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np

from hat.audio.io import MicStream
from hat.audio.types import FRAME_SAMPLES, SAMPLE_RATE, Phrase, Transcript
from hat.config import settings

__all__ = ["SpeechToText"]


class SpeechToText:
    def __init__(
        self,
        model_size: str = settings.stt_model_size,
        compute_type: str = "int8",
        allowed_langs: tuple[str, ...] = settings.languages,
        device: str = "cpu",
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_size = model_size
        self.allowed_langs = tuple(allowed_langs) or ("en",)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def warm_up(self) -> None:
        """Run one dummy inference so the first real transcription isn't
        slowed down by lazy CTranslate2/model initialization."""
        dummy = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        segments, _info = self._model.transcribe(
            dummy,
            language=self.allowed_langs[0],
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        list(segments)  # force the lazy generator to actually run

    def _detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Detect language, restricted to self.allowed_langs.

        faster_whisper's WhisperModel.detect_language(audio) returns
        (top_language, top_probability, all_language_probs) where
        all_language_probs is a list of (lang_code, probability) tuples
        covering Whisper's full language vocabulary. We filter that list
        down to the hat's allowed languages, renormalize, and take the
        argmax -- guaranteeing the result is always one of allowed_langs.
        """
        _top_lang, _top_prob, all_probs = self._model.detect_language(audio)
        allowed = {lang: prob for lang, prob in all_probs if lang in self.allowed_langs}

        if not allowed:
            return self.allowed_langs[0], 0.0

        total = sum(allowed.values())
        if total > 0:
            allowed = {lang: prob / total for lang, prob in allowed.items()}

        winner = max(allowed, key=allowed.get)
        return winner, allowed[winner]

    def transcribe(self, phrase: Phrase) -> Transcript:
        audio = phrase.pcm.astype(np.float32) / 32768.0

        winner_lang, lang_confidence = self._detect_language(audio)

        t0 = time.monotonic()
        segments, _info = self._model.transcribe(
            audio,
            language=winner_lang,
            beam_size=5,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # already VAD-trimmed upstream
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        latency_s = time.monotonic() - t0

        return Transcript(
            text=text,
            lang=winner_lang,
            lang_confidence=lang_confidence,
            latency_s=latency_s,
        )


# -- CLI ---------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hat.stt.whisper_stt")
    parser.add_argument("--record", type=float, default=5.0, help="seconds to record from the mic")
    parser.add_argument("--model-size", default=settings.stt_model_size)
    parser.add_argument("--device", type=int, default=None, help="input device index")
    args = parser.parse_args(argv)

    stt = SpeechToText(model_size=args.model_size)
    print("Warming up model...", file=sys.stderr)
    stt.warm_up()

    print(f"Recording {args.record:.1f}s... speak now.", file=sys.stderr)
    n_frames_needed = max(1, round(args.record * SAMPLE_RATE / FRAME_SAMPLES))
    chunks = []
    with MicStream(device=args.device) as mic:
        it = mic.frames()
        for _ in range(n_frames_needed):
            chunks.append(next(it))
    pcm = np.concatenate(chunks)

    phrase = Phrase(
        pcm=pcm,
        sample_rate=SAMPLE_RATE,
        started_at=time.monotonic(),
        duration_s=len(pcm) / SAMPLE_RATE,
    )
    print("Transcribing...", file=sys.stderr)
    transcript = stt.transcribe(phrase)
    print(transcript)


if __name__ == "__main__":
    main()
