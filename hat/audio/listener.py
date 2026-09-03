"""The audio-input facade: wake word -> record phrase -> transcribe.

Wires MicStream + WakeWordDetector + EndOfPhraseDetector + SpeechToText
together behind the same public shape as ``hat.audio.stub.FakeVoiceInput``
(wait_for_wake / listen_once / hold), so the rest of the app (main.py /
the orchestrator) can be developed against the stub and later swapped to
this real implementation without any other code changes.

Internal state machine per listen_once() call:
    WAITING_WAKE (wait_for_wake) -> RECORDING (VAD in progress) -> TRANSCRIBING
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from hat.audio.io import MicStream
from hat.audio.types import Phrase, Utterance, WakeEvent
from hat.audio.vad import EndOfPhraseDetector, PhraseState
from hat.config import settings
from hat.stt.whisper_stt import SpeechToText
from hat.wake.detector import WakeWordDetector

__all__ = ["VoiceInput"]

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Fold a transcript down to something a wake phrase can be matched
    against: lowercase, accents stripped, punctuation to spaces. Cyrillic
    survives -- only combining marks are dropped -- so the Russian name
    matches as directly as the Spanish one."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


class VoiceInput:
    """Real microphone-backed voice input. See module docstring."""

    def __init__(
        self,
        mic: Optional[MicStream] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        vad: Optional[EndOfPhraseDetector] = None,
        stt: Optional[SpeechToText] = None,
        post_hold_refractory_s: float = 0.5,
        wake_phrases: Optional[tuple[str, ...]] = None,
    ) -> None:
        self._owns_mic = mic is None
        self.mic = mic if mic is not None else MicStream()
        # The wake stack is the one piece of this pipeline that may be
        # unavailable on ARM64 (openwakeword pulls speexdsp-ns, which has no
        # aarch64 wheel -- see requirements-audio.txt). Losing it must not
        # cost us the microphone as well: silero VAD and faster-whisper are
        # fine there, so degrade to speech-triggered waking instead of
        # letting the whole VoiceInput fail to construct and drop main.py
        # back to the stdin stub.
        if wake_detector is not None:
            self.wake = wake_detector
        else:
            try:
                self.wake = WakeWordDetector()
            except ImportError as exc:
                # The expected state on the Pi, not a fault: log it plainly,
                # because a full traceback on every boot would only bury the
                # failures that do matter.
                logger.warning(
                    "No wake-word model (%s); listening for spoken names instead: %s",
                    exc,
                    ", ".join(settings.wake_phrases) or "any speech",
                )
                self.wake = None
            except Exception:
                logger.warning(
                    "Wake-word detector failed to start; listening for spoken names instead.",
                    exc_info=True,
                )
                self.wake = None
        self.vad = vad if vad is not None else EndOfPhraseDetector()
        self.stt = stt if stt is not None else SpeechToText()
        self.post_hold_refractory_s = post_hold_refractory_s
        self.wake_phrases = tuple(
            _normalize(p) for p in (settings.wake_phrases if wake_phrases is None else wake_phrases)
        )

        self._frames = self.mic.frames()
        self._last_wake_event: Optional[WakeEvent] = None
        self.stt.warm_up()

    # -- public interface (matches hat.audio.stub.FakeVoiceInput) -----------

    def wait_for_wake(
        self,
        timeout: Optional[float] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Block until the wake word is heard (returns True), or until
        ``timeout`` seconds elapse with no wake word (returns False). No
        timeout (None) means wait forever.

        ``cancel`` is checked once per frame (80ms) and aborts the wait when
        it returns True -- that is how a visitor who simply sits down, with
        no wake word at all, still starts a visit. It exists so the caller
        never has to chop this into short timed slices: each call resets the
        detector's rolling buffer, so repeated short waits would degrade
        wake-word detection itself.
        """
        if self.wake is None:
            return self._wait_for_wake_phrase(timeout, cancel)

        # Stale buffers from whatever happened before this call (a prior
        # recorded phrase, TTS bleed, etc.) shouldn't cause an immediate
        # spurious trigger here.
        self.wake.reset()

        deadline = None if timeout is None else time.monotonic() + timeout
        for frame in self._frames:
            event = self.wake.process(frame)
            if event is not None:
                self._last_wake_event = event
                return True
            if cancel is not None and cancel():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
        return False

    def _wait_for_wake_phrase(
        self,
        timeout: Optional[float],
        cancel: Optional[Callable[[], bool]],
    ) -> bool:
        """Wake-word stand-in for rigs without openwakeword: overhear a
        phrase, transcribe it without any language constraint, and wake only
        if one of the hat's names is in it. Costs a greedy Whisper pass per
        utterance heard while idle, which is the price of having real spoken
        names without a wake-word model. With no names configured it falls
        back to waking on any speech at all."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False

            phrase = self._record_phrase(remaining, self.vad.max_phrase_s, cancel)
            if phrase is None:
                if cancel is not None and cancel():
                    return False
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                continue

            if not self.wake_phrases:
                return True

            heard = _normalize(self.stt.transcribe_any(phrase))
            if any(name in heard for name in self.wake_phrases):
                logger.info("Woken by %r", heard)
                return True
            logger.debug("Overheard %r -- not one of my names", heard)

    def _record_phrase(
        self,
        timeout: Optional[float],
        max_phrase_s: float,
        cancel: Optional[Callable[[], bool]],
    ) -> Optional[Phrase]:
        """Record one VAD-bounded phrase. ``timeout`` bounds only the wait
        for speech to *start*; ``cancel`` likewise is consulted only before
        it does, so nobody gets cut off mid-sentence."""
        self.vad.reset(max_phrase_s=max_phrase_s)
        start_deadline = None if timeout is None else time.monotonic() + timeout

        for frame in self._frames:
            state = self.vad.feed(frame)

            if state is PhraseState.WAITING:
                if cancel is not None and cancel():
                    return None
                if start_deadline is not None and time.monotonic() >= start_deadline:
                    return None
                continue

            if state is PhraseState.IN_PHRASE:
                continue

            if state in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
                return self.vad.result()

        return None

    def listen_once(
        self,
        timeout: float = 8.0,
        max_phrase_s: float = 15.0,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[Utterance]:
        """Record one phrase (VAD-bounded) and transcribe it.

        ``timeout`` bounds how long we wait for the user to *start*
        talking before giving up (returns None). Once speech starts,
        recording continues until end-of-phrase silence or ``max_phrase_s``
        is reached, whichever comes first.

        ``cancel`` aborts the wait (returns None) but is only consulted
        *before* speech starts: if the visitor sits down mid-sentence, they
        get to finish the sentence, and the caller picks the event up on the
        next pass.
        """
        phrase = self._record_phrase(timeout, max_phrase_s, cancel)
        if phrase is None:
            return None

        transcript = self.stt.transcribe(phrase)
        wake_event, self._last_wake_event = self._last_wake_event, None
        return Utterance(transcript=transcript, phrase=phrase, wake=wake_event)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Half-duplex mute the mic for the duration of the `with` block --
        meant to wrap TTS playback so the hat doesn't hear (and transcribe,
        or re-trigger on) its own voice."""
        self.mic.pause()
        try:
            yield
        finally:
            self.mic.resume()
            if self.wake is not None:
                self.wake.reset()
            time.sleep(self.post_hold_refractory_s)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        if self._owns_mic:
            self.mic.close()

    def __enter__(self) -> "VoiceInput":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
