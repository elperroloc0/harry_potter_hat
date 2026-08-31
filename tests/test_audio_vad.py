"""Unit tests for the end-of-phrase state machine in hat.audio.vad.

These tests inject a scripted fake VAD scorer (see ScriptedVad below)
instead of the real Silero model. Silero is trained on real speech, so
feeding it synthetic tones/noise doesn't reliably cross the 0.5/0.35
thresholds either way -- see the module docstring note below for the
empirical numbers that motivated this. Using a scripted scorer instead
makes these tests fast, deterministic, and focused purely on the state
machine logic (thresholds, hysteresis, debounce, pre-roll, timeout),
which is exactly what "feed synthetic ... arrays and assert state
transitions" means here: the *state machine's* behavior under controlled
score input, not a claim about how any particular waveform scores.

One additional test (test_real_silero_stays_waiting_on_silence) exercises
the real pysilero_vad model on true digital silence, which reliably
scores near zero -- a legitimate real-model smoke test with no flakiness
risk, unlike trying to force a real "speech-like" high score synthetically.
"""

from __future__ import annotations

import numpy as np
import pytest

from hat.audio.types import FRAME_SAMPLES, SAMPLE_RATE
from hat.audio.vad import EndOfPhraseDetector, PhraseState


class ScriptedVad:
    """Fake VAD model returning a pre-scripted sequence of per-512-sample-chunk
    scores. Once exhausted, keeps returning the last scripted value."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self._i = 0
        self.reset_calls = 0

    def process_chunk(self, audio: bytes) -> float:
        if self._i < len(self._scores):
            score = self._scores[self._i]
            self._i += 1
        else:
            score = self._scores[-1] if self._scores else 0.0
        return score

    def reset(self) -> None:
        self.reset_calls += 1
        self._i = 0


def _frame(fill_value: int = 0, n: int = FRAME_SAMPLES) -> np.ndarray:
    """A FRAME_SAMPLES-length int16 frame filled with a constant value, so
    tests can fingerprint which frames ended up in the accumulated phrase."""
    return np.full(n, fill_value, dtype=np.int16)


def _feed_frames(detector: EndOfPhraseDetector, n_frames: int, fill_value: int = 0) -> list[PhraseState]:
    states = []
    for _ in range(n_frames):
        states.append(detector.feed(_frame(fill_value)))
    return states


def test_happy_path_waiting_to_in_phrase_to_complete() -> None:
    # 512-sample chunks: a run of low scores, then two consecutive high
    # scores (min_speech_chunks=2) to confirm speech start, then a long
    # run of low scores to trigger end-of-phrase silence.
    scores = [0.05] * 6 + [0.9, 0.9] + [0.05] * 40
    vad = ScriptedVad(scores)
    detector = EndOfPhraseDetector(
        start_threshold=0.5,
        end_threshold=0.35,
        silence_end_s=0.8,  # -> 25 chunks of 512 samples @ 16kHz
        max_phrase_s=12.0,
        pre_roll_s=0.5,
        min_speech_chunks=2,
        vad_model=vad,
    )

    # Feed generously many 1280-sample frames -- more than enough to consume
    # every scripted chunk score (46 chunks -> ceil(46/2.5) ~= 19 frames).
    states = _feed_frames(detector, n_frames=40)

    assert PhraseState.IN_PHRASE in states
    assert states[-1] is PhraseState.COMPLETE
    assert detector.state is PhraseState.COMPLETE

    phrase = detector.result()
    assert phrase.pcm.dtype == np.int16
    assert phrase.duration_s > 0
    assert phrase.sample_rate == SAMPLE_RATE
    assert len(phrase.pcm) == pytest.approx(phrase.duration_s * SAMPLE_RATE, abs=1)


def test_never_crosses_threshold_stays_waiting() -> None:
    vad = ScriptedVad([0.05] * 200)
    detector = EndOfPhraseDetector(min_speech_chunks=2, vad_model=vad)

    states = _feed_frames(detector, n_frames=30)

    assert all(s is PhraseState.WAITING for s in states)
    assert detector.state is PhraseState.WAITING
    with pytest.raises(RuntimeError):
        detector.result()


def test_isolated_blip_does_not_trigger_min_speech_chunks_debounce() -> None:
    # A single high-score chunk surrounded by low scores must not confirm
    # speech when min_speech_chunks=2 (needs *consecutive* high chunks).
    scores = [0.05] * 4 + [0.9] + [0.05] * 4 + [0.9] + [0.05] * 40
    vad = ScriptedVad(scores)
    detector = EndOfPhraseDetector(min_speech_chunks=2, vad_model=vad)

    states = _feed_frames(detector, n_frames=30)

    assert PhraseState.IN_PHRASE not in states
    assert detector.state is PhraseState.WAITING


def test_timeout_when_speaker_never_stops() -> None:
    # min_speech_chunks=1 so a single high chunk confirms speech immediately;
    # scores never drop below end_threshold, so only max_phrase_s can end it.
    scores = [0.9] * 200
    vad = ScriptedVad(scores)
    detector = EndOfPhraseDetector(
        start_threshold=0.5,
        end_threshold=0.35,
        min_speech_chunks=1,
        max_phrase_s=0.05,  # -> round(0.05*16000/512) = 2 chunks
        silence_end_s=100.0,  # effectively disabled
        vad_model=vad,
    )

    states = _feed_frames(detector, n_frames=10)

    assert states[-1] is PhraseState.TIMEOUT
    phrase = detector.result()
    assert phrase.duration_s > 0


def test_pre_roll_ring_buffer_is_included_in_phrase() -> None:
    # Feed distinctive "pre-roll" frames (fill value 7) while WAITING with a
    # low score, then trigger speech with distinctive "spoken" frames (fill
    # value 9). The resulting phrase should start with pre-roll content (7s)
    # and transition into spoken content (9s) -- proving the ring buffer
    # actually captured real preceding audio, not silence.
    pre_roll_s = 0.25  # -> 4000 samples of pre-roll
    # Chunks consumed per WAITING frame varies (2 or 3 per 1280-sample frame);
    # feed enough low-score frames to fully saturate the pre-roll ring before
    # triggering.
    n_preroll_frames = 10
    n_preroll_chunks = 0
    # Compute an upper bound on chunks generated by n_preroll_frames frames.
    total_samples = n_preroll_frames * FRAME_SAMPLES
    n_preroll_chunks = total_samples // 512

    scores = [0.05] * n_preroll_chunks + [0.9, 0.9] + [0.05] * 40
    vad = ScriptedVad(scores)
    detector = EndOfPhraseDetector(
        start_threshold=0.5,
        end_threshold=0.35,
        pre_roll_s=pre_roll_s,
        min_speech_chunks=2,
        silence_end_s=0.8,
        vad_model=vad,
    )

    for _ in range(n_preroll_frames):
        state = detector.feed(_frame(7))
        assert state is PhraseState.WAITING

    # Now feed the triggering + spoken frames.
    final_states = []
    for _ in range(30):
        final_states.append(detector.feed(_frame(9)))
        if final_states[-1] in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
            break

    assert final_states[-1] is PhraseState.COMPLETE
    phrase = detector.result()

    pre_roll_samples = round(pre_roll_s * SAMPLE_RATE)
    # The very start of the accumulated phrase must be pre-roll (7s) content,
    # not zeros and not yet the "9" spoken content.
    assert phrase.pcm[0] == 7
    assert phrase.pcm[pre_roll_samples // 2] == 7
    # And somewhere later, the spoken (9) content shows up.
    assert 9 in phrase.pcm


def test_reset_allows_a_fresh_cycle() -> None:
    scores = [0.9, 0.9] + [0.05] * 40
    vad = ScriptedVad(scores)
    detector = EndOfPhraseDetector(min_speech_chunks=2, silence_end_s=0.8, vad_model=vad)

    states = _feed_frames(detector, n_frames=30)
    assert states[-1] is PhraseState.COMPLETE

    detector.reset()
    assert detector.state is PhraseState.WAITING
    assert vad.reset_calls == 1

    # Fresh scripted scores for the second cycle.
    vad._scores = [0.9, 0.9] + [0.05] * 40
    vad._i = 0
    states2 = _feed_frames(detector, n_frames=30)
    assert states2[-1] is PhraseState.COMPLETE


def test_real_silero_stays_waiting_on_silence() -> None:
    """Smoke test with the real pysilero_vad model: true digital silence
    should never cross the default start_threshold."""
    detector = EndOfPhraseDetector()  # real Silero model, default thresholds

    states = _feed_frames(detector, n_frames=20, fill_value=0)
    assert all(s is PhraseState.WAITING for s in states)
    assert detector.state is PhraseState.WAITING
