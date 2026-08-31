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

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from hat.audio.io import MicStream
from hat.audio.types import Utterance, WakeEvent
from hat.audio.vad import EndOfPhraseDetector, PhraseState
from hat.stt.whisper_stt import SpeechToText
from hat.wake.detector import WakeWordDetector

__all__ = ["VoiceInput"]


class VoiceInput:
    """Real microphone-backed voice input. See module docstring."""

    def __init__(
        self,
        mic: Optional[MicStream] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        vad: Optional[EndOfPhraseDetector] = None,
        stt: Optional[SpeechToText] = None,
        post_hold_refractory_s: float = 0.5,
    ) -> None:
        self._owns_mic = mic is None
        self.mic = mic if mic is not None else MicStream()
        self.wake = wake_detector if wake_detector is not None else WakeWordDetector()
        self.vad = vad if vad is not None else EndOfPhraseDetector()
        self.stt = stt if stt is not None else SpeechToText()
        self.post_hold_refractory_s = post_hold_refractory_s

        self._frames = self.mic.frames()
        self._last_wake_event: Optional[WakeEvent] = None
        self.stt.warm_up()

    # -- public interface (matches hat.audio.stub.FakeVoiceInput) -----------

    def wait_for_wake(self, timeout: Optional[float] = None) -> bool:
        """Block until the wake word is heard (returns True), or until
        ``timeout`` seconds elapse with no wake word (returns False). No
        timeout (None) means wait forever."""
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
            if deadline is not None and time.monotonic() >= deadline:
                return False
        return False

    def listen_once(self, timeout: float = 8.0, max_phrase_s: float = 15.0) -> Optional[Utterance]:
        """Record one phrase (VAD-bounded) and transcribe it.

        ``timeout`` bounds how long we wait for the user to *start*
        talking before giving up (returns None). Once speech starts,
        recording continues until end-of-phrase silence or ``max_phrase_s``
        is reached, whichever comes first.
        """
        self.vad.reset(max_phrase_s=max_phrase_s)

        start_deadline = time.monotonic() + timeout
        for frame in self._frames:
            state = self.vad.feed(frame)

            if state is PhraseState.WAITING:
                if time.monotonic() >= start_deadline:
                    return None
                continue

            if state is PhraseState.IN_PHRASE:
                continue

            if state in (PhraseState.COMPLETE, PhraseState.TIMEOUT):
                phrase = self.vad.result()
                transcript = self.stt.transcribe(phrase)
                wake_event, self._last_wake_event = self._last_wake_event, None
                return Utterance(transcript=transcript, phrase=phrase, wake=wake_event)

        return None

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
