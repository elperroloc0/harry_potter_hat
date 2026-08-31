from __future__ import annotations

import httpx
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs.core import ApiError
from elevenlabs.types.voice_settings import VoiceSettings

from hat.config import settings
from hat.tts.base import Lang, PcmAudio, SynthesisError, Synthesizer

DEFAULT_MODEL_ID = "eleven_flash_v2_5"

# eleven_flash_v2_5 supports language enforcement via language_code; our
# Lang literal ("es"/"en") already matches ElevenLabs' expected codes.
_LANG_CODES: dict[Lang, str] = {"es": "es", "en": "en"}


class ElevenLabsSynth(Synthesizer):
    """Cloud TTS via the ElevenLabs API (canonical backend for the deployed
    hat). Requires an API key -- see hat.tts.macos_say.MacSaySynth for the
    free, offline dev-machine fallback that HatVoice falls back to
    automatically when this backend fails.

    Verified against the installed `elevenlabs` SDK (v2.65.0):
    `client.text_to_speech.convert(voice_id, *, text, model_id,
    output_format, language_code, voice_settings, ...) -> Iterator[bytes]`.
    `output_format=f"pcm_{sample_rate}"` returns raw headerless s16le mono
    PCM -- chunks are joined into one bytes object before ever touching
    np.frombuffer, since a chunk boundary can land mid-sample.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = settings.elevenlabs_voice_id,
        model_id: str = DEFAULT_MODEL_ID,
        sample_rate: int = 22050,
        voice_settings: VoiceSettings | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.voice_settings = (
            voice_settings
            if voice_settings is not None
            else VoiceSettings(stability=0.4, similarity_boost=0.75)
        )
        # Constructing the client never touches the network and never
        # fails even with an empty/bogus key -- only synth() calls out.
        self._client = ElevenLabs(api_key=self.api_key)

    def synth(self, text: str, lang: Lang) -> PcmAudio:
        if not self.api_key:
            raise SynthesisError(
                "ElevenLabs API key is missing (settings.elevenlabs_api_key is "
                "empty). Set ELEVENLABS_API_KEY in .env, or use the 'say' "
                "backend for dev (HAT_TTS_BACKEND=say)."
            )

        output_format = f"pcm_{self.sample_rate}"
        try:
            chunks = self._client.text_to_speech.convert(
                self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format=output_format,  # type: ignore[arg-type]
                language_code=_LANG_CODES.get(lang, lang),
                voice_settings=self.voice_settings,
            )
            raw = b"".join(chunks)  # join BEFORE frombuffer -- chunks can split a sample
        except ApiError as exc:
            raise SynthesisError(_describe_api_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise SynthesisError(f"ElevenLabs network error: {exc}") from exc
        except Exception as exc:  # last resort -- synth() must never crash the caller
            raise SynthesisError(f"ElevenLabs synthesis failed unexpectedly: {exc}") from exc

        if not raw:
            raise SynthesisError("ElevenLabs returned no audio data")
        if len(raw) % 2 != 0:
            raw = raw[:-1]  # drop a stray trailing byte rather than misalign samples

        samples = np.frombuffer(raw, dtype="<i2").copy()
        return PcmAudio(samples=samples, sample_rate=self.sample_rate)


def _describe_api_error(exc: ApiError) -> str:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if status in (401, 403):
        return f"ElevenLabs auth failed (status {status}) -- check ELEVENLABS_API_KEY. Body: {body}"
    if status == 429:
        return f"ElevenLabs rate-limited or quota exceeded (status {status}). Body: {body}"
    if status is not None and status >= 500:
        return f"ElevenLabs server error (status {status}) -- try again later. Body: {body}"
    return f"ElevenLabs API error (status {status}): {body}"
