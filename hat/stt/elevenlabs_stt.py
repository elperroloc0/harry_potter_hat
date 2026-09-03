"""Speech-to-text via ElevenLabs Scribe, with the local Whisper as backup.

Whisper `small` on the Pi's CPU costs about 8.8s per utterance -- two passes
over the audio, one to detect the language and one to transcribe it with a
beam of five -- which was roughly three quarters of the hat's entire
response time. Scribe does the same job in well under a second and gets it
right more often, measured on the rig against real synthesized speech:

    es  0.68s  "Hola Sombrero, me llamo Ana y me gusta explorar cuevas"
    en  0.92s  "Hello Hat. My name is Tom and I love solving puzzles"

The tradeoff is that speech now leaves the house, where before it did not.
Only the microphone audio goes; the camera still never leaves the LAN.

Falls back to local Whisper if the network or the API is unavailable, built
lazily so a healthy rig never pays to load a model it will not use.
"""

from __future__ import annotations

import io
import logging
import time
import wave

from hat.audio.types import Phrase, Transcript
from hat.config import settings

logger = logging.getLogger(__name__)

__all__ = ["ScribeSpeechToText"]

# Scribe reports ISO 639-3; the rest of the project speaks ISO 639-1.
_ISO3_TO_ISO1 = {"spa": "es", "eng": "en", "rus": "ru", "fra": "fr", "deu": "de", "por": "pt"}


def _to_wav(phrase: Phrase) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(phrase.sample_rate)
        w.writeframes(phrase.pcm.tobytes())
    buf.seek(0)
    # The SDK sends this as a multipart upload and wants a filename on it.
    buf.name = "utterance.wav"
    return buf


class ScribeSpeechToText:
    """Same shape as hat.stt.whisper_stt.SpeechToText: warm_up / transcribe /
    transcribe_any, so VoiceInput cannot tell the two apart."""

    def __init__(
        self,
        api_key: str = settings.elevenlabs_api_key,
        model_id: str = "scribe_v1",
        allowed_langs: tuple[str, ...] = settings.languages,
        fallback=None,
    ) -> None:
        from elevenlabs import ElevenLabs

        self.client = ElevenLabs(api_key=api_key)
        self.model_id = model_id
        self.allowed_langs = tuple(allowed_langs) or ("en",)
        self._fallback = fallback

    def warm_up(self) -> None:
        """Nothing to load -- the model is not ours. Kept for interface
        parity with the local backend, which needs it badly."""

    def _local(self):
        """Build the Whisper backup only once something has actually gone
        wrong; loading it costs seconds and a few hundred MB."""
        if self._fallback is None:
            logger.warning("Falling back to local Whisper; expect it to be slow")
            from hat.stt.whisper_stt import SpeechToText

            self._fallback = SpeechToText()
            self._fallback.warm_up()
        return self._fallback

    def _convert(self, phrase: Phrase):
        return self.client.speech_to_text.convert(file=_to_wav(phrase), model_id=self.model_id)

    def transcribe(self, phrase: Phrase) -> Transcript:
        t0 = time.monotonic()
        try:
            result = self._convert(phrase)
        except Exception:
            logger.warning("Scribe unavailable; using local Whisper for this one", exc_info=True)
            return self._local().transcribe(phrase)

        lang = _ISO3_TO_ISO1.get(getattr(result, "language_code", "") or "", "")
        if lang not in self.allowed_langs:
            # The hat only performs in the languages it was given, so a
            # stray detection is pinned rather than passed through.
            lang = settings.default_lang
        return Transcript(
            text=(getattr(result, "text", "") or "").strip(),
            lang=lang,
            lang_confidence=float(getattr(result, "language_probability", 0.0) or 0.0),
            latency_s=time.monotonic() - t0,
        )

    def transcribe_any(self, phrase: Phrase) -> str:
        """Unconstrained text, for wake-phrase matching."""
        try:
            return (getattr(self._convert(phrase), "text", "") or "").strip()
        except Exception:
            logger.warning("Scribe unavailable; using local Whisper for this one", exc_info=True)
            return self._local().transcribe_any(phrase)


def make_stt(backend: str | None = None):
    """Pick a speech-to-text backend. Scribe by default: the local model is
    an order of magnitude slower on this hardware and less accurate. Set
    HAT_STT_BACKEND=whisper to keep audio on the LAN at that cost."""
    backend = (backend or settings.stt_backend).lower()
    if backend == "whisper":
        from hat.stt.whisper_stt import SpeechToText

        return SpeechToText()
    if not settings.elevenlabs_api_key:
        logger.warning("No ElevenLabs key; falling back to local Whisper for speech-to-text")
        from hat.stt.whisper_stt import SpeechToText

        return SpeechToText()
    return ScribeSpeechToText()
