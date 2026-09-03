"""Wake-phrase matching for rigs without openwakeword (i.e. the Pi).

No audio hardware and no Whisper: MicStream, the VAD and SpeechToText are
all injected, so these exercise only the decision logic -- which utterances
wake the hat and which it ignores.
"""

from __future__ import annotations

import numpy as np
import pytest

from hat.audio.listener import VoiceInput, _normalize
from hat.audio.types import Phrase
from hat.audio.vad import PhraseState


class _FakeMic:
    def frames(self):
        while True:
            yield np.zeros(1280, dtype=np.int16)

    def pause(self):
        pass

    def resume(self):
        pass

    def close(self):
        pass


class _FakeVad:
    """One phrase per reset: a frame of silence first, then a complete
    phrase. The leading WAITING frame matters -- that is the only state in
    which cancel() is consulted, because a phrase already under way is never
    cut off mid-sentence."""

    max_phrase_s = 12.0

    def __init__(self):
        self._fed = 0

    def reset(self, max_phrase_s=None):
        self._fed = 0

    def feed(self, frame):
        self._fed += 1
        return PhraseState.WAITING if self._fed == 1 else PhraseState.COMPLETE

    def result(self):
        return Phrase(pcm=np.zeros(16000, dtype=np.int16), started_at=0.0)


class _FakeStt:
    def __init__(self, *heard):
        self._heard = list(heard)
        self.calls = 0

    def warm_up(self):
        pass

    def transcribe_any(self, phrase):
        self.calls += 1
        return self._heard.pop(0) if self._heard else "nothing at all"


def _voice(stt, phrases=None):
    return VoiceInput(
        mic=_FakeMic(),
        wake_detector=None,
        vad=_FakeVad(),
        stt=stt,
        wake_phrases=phrases,
    )


@pytest.fixture(autouse=True)
def _no_real_detector(monkeypatch):
    """Force the no-openwakeword path this whole module is about."""
    monkeypatch.setattr(
        "hat.audio.listener.WakeWordDetector",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no openwakeword")),
    )


@pytest.mark.parametrize(
    "heard",
    ["шляпа", "Шляпа!", "hola sombrero", "¡Sombrero!", "hey sorting hat", "SORTING HAT"],
)
def test_wakes_on_its_names(heard):
    v = _voice(_FakeStt(heard))
    assert v.wake is None  # the whole point: degraded, not dead
    assert v.wait_for_wake(timeout=5.0) is True


def test_ignores_ordinary_chatter_then_wakes():
    stt = _FakeStt("where is the bathroom", "mira que sombrero tan raro")
    v = _voice(stt)

    assert v.wait_for_wake(timeout=5.0) is True
    # First utterance rejected, second matched -- two transcriptions.
    assert stt.calls == 2


def test_cancel_beats_the_wake_phrase():
    # PIR firing while idle must win: someone sat down without saying anything.
    v = _voice(_FakeStt("шляпа"))
    assert v.wait_for_wake(timeout=5.0, cancel=lambda: True) is False


def test_no_phrases_configured_wakes_on_any_speech():
    v = _voice(_FakeStt("literally anything"), phrases=())
    assert v.wait_for_wake(timeout=5.0) is True


def test_normalize_folds_case_accents_and_punctuation():
    assert _normalize("  ¡Sombrero!  ") == "sombrero"
    assert _normalize("Шляпа,") == "шляпа"
    assert _normalize("Sorting  Hat?") == "sorting hat"
