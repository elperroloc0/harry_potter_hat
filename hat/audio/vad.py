"""End-of-phrase detection: turns a stream of raw 16 kHz frames into
complete spoken :class:`~hat.audio.types.Phrase` objects.

Wraps Silero VAD (via ``pysilero_vad``, which operates on fixed 512-sample
chunks) behind a small hysteresis state machine, and keeps a short
pre-roll ring buffer so the returned phrase includes the syllable(s)
spoken just before the VAD actually crossed its start threshold.
"""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum, auto
from typing import Optional, Protocol

import numpy as np

from hat.audio.types import FRAME_SAMPLES, SAMPLE_RATE, Phrase

__all__ = ["PhraseState", "EndOfPhraseDetector"]


class _VadModel(Protocol):
    """The two methods of pysilero_vad.SileroVoiceActivityDetector we rely
    on. Declared as a Protocol purely so a test double can be injected
    without needing the real (non-deterministic on synthetic audio) model."""

    def process_chunk(self, audio: bytes) -> float: ...

    def reset(self) -> None: ...


def _load_real_vad() -> _VadModel:
    from pysilero_vad import SileroVoiceActivityDetector

    return SileroVoiceActivityDetector()


class PhraseState(Enum):
    WAITING = auto()
    IN_PHRASE = auto()
    COMPLETE = auto()
    TIMEOUT = auto()


class EndOfPhraseDetector:
    """Feed it raw int16 mono 16 kHz frames of ``FRAME_SAMPLES`` length, one
    at a time, via :meth:`feed`. Once it returns ``COMPLETE`` or ``TIMEOUT``,
    call :meth:`result` to get the accumulated :class:`Phrase`, then
    :meth:`reset` before feeding it the next utterance.
    """

    #: Samples Silero VAD requires per scoring call.
    _CHUNK_SAMPLES = 512

    def __init__(
        self,
        start_threshold: float = 0.5,
        end_threshold: float = 0.35,
        silence_end_s: float = 0.8,
        max_phrase_s: float = 12.0,
        pre_roll_s: float = 0.5,
        min_speech_chunks: int = 2,
        *,
        vad_model: Optional[_VadModel] = None,
    ) -> None:
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_speech_chunks = min_speech_chunks
        self.pre_roll_s = pre_roll_s

        self._vad: _VadModel = vad_model if vad_model is not None else _load_real_vad()
        self._chunk_samples = self._CHUNK_SAMPLES

        self._silence_end_chunks = 0
        self._max_phrase_chunks = 0
        self._pre_roll_samples = 0
        self.set_silence_end_s(silence_end_s)
        self.set_max_phrase_s(max_phrase_s)
        self.set_pre_roll_s(pre_roll_s)

        self._reset_state()

    # -- tunable setters (also usable from reset() for per-call overrides) --

    def set_silence_end_s(self, silence_end_s: float) -> None:
        self.silence_end_s = silence_end_s
        self._silence_end_chunks = max(
            1, round(silence_end_s * SAMPLE_RATE / self._chunk_samples)
        )

    def set_max_phrase_s(self, max_phrase_s: float) -> None:
        self.max_phrase_s = max_phrase_s
        self._max_phrase_chunks = max(
            1, round(max_phrase_s * SAMPLE_RATE / self._chunk_samples)
        )

    def set_pre_roll_s(self, pre_roll_s: float) -> None:
        self.pre_roll_s = pre_roll_s
        self._pre_roll_samples = max(0, round(pre_roll_s * SAMPLE_RATE))

    # -- lifecycle -----------------------------------------------------------

    def _reset_state(self) -> None:
        self._raw_buf = np.zeros(0, dtype=np.int16)
        self._ring = np.zeros(0, dtype=np.int16)
        self._phrase_chunks: list[np.ndarray] = []
        self._state = PhraseState.WAITING
        self._speech_run = 0
        self._silence_run = 0
        self._chunks_in_phrase = 0
        self._trigger_monotonic = 0.0
        self._final_pcm: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        max_phrase_s: Optional[float] = None,
        silence_end_s: Optional[float] = None,
        pre_roll_s: Optional[float] = None,
    ) -> None:
        """Reset the state machine (and the underlying VAD's internal
        streaming buffers) so it's ready for the next phrase. Optionally
        override the timing parameters for the upcoming phrase without
        reconstructing the detector (and reloading the Silero model)."""
        if max_phrase_s is not None:
            self.set_max_phrase_s(max_phrase_s)
        if silence_end_s is not None:
            self.set_silence_end_s(silence_end_s)
        if pre_roll_s is not None:
            self.set_pre_roll_s(pre_roll_s)
        self._vad.reset()
        self._reset_state()

    @property
    def state(self) -> PhraseState:
        return self._state

    # -- streaming -----------------------------------------------------------

    def feed(self, frame: np.ndarray) -> PhraseState:
        """Process one FRAME_SAMPLES-length int16 frame. Returns the state
        after processing. Once COMPLETE/TIMEOUT, further calls are no-ops
        until reset()."""
        if frame.dtype != np.int16:
            frame = frame.astype(np.int16)
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"expected a {FRAME_SAMPLES}-sample frame, got {len(frame)}")

        if self._state in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
            return self._state

        self._raw_buf = np.concatenate([self._raw_buf, frame])
        while len(self._raw_buf) >= self._chunk_samples:
            chunk = self._raw_buf[: self._chunk_samples]
            self._raw_buf = self._raw_buf[self._chunk_samples :]
            self._process_chunk(chunk)
            if self._state in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
                break

        return self._state

    def _score(self, chunk: np.ndarray) -> float:
        return float(self._vad.process_chunk(chunk.tobytes()))

    def _process_chunk(self, chunk: np.ndarray) -> None:
        score = self._score(chunk)

        if self._state is PhraseState.WAITING:
            self._ring = np.concatenate([self._ring, chunk])
            if len(self._ring) > self._pre_roll_samples:
                self._ring = self._ring[-self._pre_roll_samples :]

            if score >= self.start_threshold:
                self._speech_run += 1
                if self._speech_run >= self.min_speech_chunks:
                    # Confirmed: seed the phrase with everything currently in
                    # the pre-roll ring (which already ends with the chunks
                    # that just crossed threshold).
                    self._state = PhraseState.IN_PHRASE
                    self._phrase_chunks = [self._ring.copy()]
                    self._chunks_in_phrase = 0
                    self._silence_run = 0
                    included_s = len(self._ring) / SAMPLE_RATE
                    self._trigger_monotonic = time.monotonic() - included_s
            else:
                self._speech_run = 0
            return

        if self._state is PhraseState.IN_PHRASE:
            self._phrase_chunks.append(chunk)
            self._chunks_in_phrase += 1

            if score < self.end_threshold:
                self._silence_run += 1
            else:
                self._silence_run = 0

            if self._silence_run >= self._silence_end_chunks:
                self._finalize(PhraseState.COMPLETE)
            elif self._chunks_in_phrase >= self._max_phrase_chunks:
                self._finalize(PhraseState.TIMEOUT)
            return

    def _finalize(self, state: PhraseState) -> None:
        self._state = state
        self._final_pcm = (
            np.concatenate(self._phrase_chunks)
            if self._phrase_chunks
            else np.zeros(0, dtype=np.int16)
        )

    def result(self) -> Phrase:
        """Return the accumulated phrase. Only valid once state is COMPLETE
        or TIMEOUT (raises RuntimeError otherwise)."""
        if self._state not in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
            raise RuntimeError(
                f"result() called before phrase completed (state={self._state.name})"
            )
        pcm = self._final_pcm if self._final_pcm is not None else np.zeros(0, dtype=np.int16)
        return Phrase(
            pcm=pcm,
            sample_rate=SAMPLE_RATE,
            started_at=self._trigger_monotonic,
            duration_s=len(pcm) / SAMPLE_RATE,
        )


# -- CLI ---------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    from hat.audio.io import MicStream

    parser = argparse.ArgumentParser(prog="python -m hat.audio.vad")
    parser.add_argument("--start-threshold", type=float, default=0.5)
    parser.add_argument("--end-threshold", type=float, default=0.35)
    parser.add_argument("--silence-end-s", type=float, default=0.8)
    parser.add_argument("--max-phrase-s", type=float, default=12.0)
    args = parser.parse_args(argv)

    detector = EndOfPhraseDetector(
        start_threshold=args.start_threshold,
        end_threshold=args.end_threshold,
        silence_end_s=args.silence_end_s,
        max_phrase_s=args.max_phrase_s,
    )

    print("Listening for speech (Ctrl+C to stop)...", file=sys.stderr)
    was_in_phrase = False
    try:
        with MicStream() as mic:
            for frame in mic.frames():
                state = detector.feed(frame)
                if state is PhraseState.IN_PHRASE and not was_in_phrase:
                    was_in_phrase = True
                    print("speech start")
                elif state in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
                    phrase = detector.result()
                    label = "end" if state is PhraseState.COMPLETE else "end [TIMEOUT]"
                    print(f"speech {label} ({phrase.duration_s:.1f}s)")
                    was_in_phrase = False
                    detector.reset()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
