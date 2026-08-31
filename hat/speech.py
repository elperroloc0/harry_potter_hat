from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from hat.audio.player import AudioPlayer
from hat.config import settings
from hat.motion.lipsync import LipSyncDriver, compute_envelope
from hat.motion.servos import make_servo
from hat.tts.base import Lang, PcmAudio, SynthesisError, Synthesizer
from hat.tts.elevenlabs_tts import ElevenLabsSynth
from hat.tts.macos_say import MacSaySynth

_SFX_DIR = Path(__file__).parent / "assets" / "sfx"


def _build_synth(backend: str) -> Synthesizer:
    if backend == "elevenlabs":
        return ElevenLabsSynth()
    if backend == "say":
        return MacSaySynth()
    raise ValueError(f"unknown tts_backend {backend!r}; expected 'elevenlabs' or 'say'")


def _load_wav(path: Path) -> PcmAudio:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"sfx wav must be 16-bit PCM, got sampwidth={sampwidth} bytes ({path})")
    samples = np.frombuffer(raw, dtype="<i2")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0].copy()
    else:
        samples = samples.copy()
    return PcmAudio(samples=samples, sample_rate=framerate)


class HatVoice:
    """Speech-output facade: text -> synthesized audio -> playback + lip
    sync, driven by whichever TTS/servo backends `hat.config.settings`
    selects.

    `speak()` never raises to the caller: a `SynthesisError` (or any other
    exception) from the configured backend falls back in-process to the
    free, offline `say` backend, so a flaky network or missing API key
    degrades to "the hat still talks" instead of silence or a crash. If the
    configured backend already *is* `say`, there is no further fallback --
    that line is logged and skipped.
    """

    def __init__(self, backend: str | None = None) -> None:
        self.backend_name = backend or settings.tts_backend
        self._synth = _build_synth(self.backend_name)
        self._fallback_synth: Synthesizer | None = (
            None if self.backend_name == "say" else MacSaySynth()
        )
        self.player = AudioPlayer()
        self.servo = make_servo(settings)
        self._lipsync = LipSyncDriver(self.servo)
        self._speak_lock = threading.Lock()

    def speak(self, text: str, lang: Lang, block: bool = True) -> None:
        if block:
            self._speak_now(text, lang)
        else:
            threading.Thread(target=self._speak_now, args=(text, lang), daemon=True).start()

    def _speak_now(self, text: str, lang: Lang) -> None:
        with self._speak_lock:
            audio = self._synthesize_with_fallback(text, lang)
            if audio is None:
                return
            envelope = compute_envelope(audio)
            self.player.play(audio)
            self._lipsync.run(envelope, self.player)

    def _synthesize_with_fallback(self, text: str, lang: Lang) -> PcmAudio | None:
        try:
            return self._synth.synth(text, lang)
        except SynthesisError as exc:
            print(f"[HatVoice] primary backend ({self.backend_name!r}) failed: {exc}")
        except Exception as exc:  # speak() must never raise -- degrade instead
            print(f"[HatVoice] primary backend ({self.backend_name!r}) raised unexpectedly: {exc}")

        if self._fallback_synth is None:
            print("[HatVoice] already on the 'say' backend, no further fallback -- skipping this line")
            return None

        try:
            return self._fallback_synth.synth(text, lang)
        except SynthesisError as exc:
            print(f"[HatVoice] fallback 'say' backend also failed: {exc} -- skipping this line")
        except Exception as exc:
            print(f"[HatVoice] fallback 'say' backend raised unexpectedly: {exc} -- skipping this line")
        return None

    def play_effect(self, name: str) -> None:
        """Blocking playback of a short local sound effect from
        hat/assets/sfx/{name}.wav. No-op (with a printed note) if the file
        doesn't exist -- no sound assets are recorded yet."""
        path = _SFX_DIR / f"{name}.wav"
        if not path.exists():
            print(f"[HatVoice] sound effect {name!r} not found at {path} -- skipping")
            return
        try:
            audio = _load_wav(path)
        except (wave.Error, ValueError, EOFError) as exc:
            print(f"[HatVoice] failed to load sound effect {name!r}: {exc}")
            return
        self.player.play(audio)
        self.player.wait()

    def is_speaking(self) -> bool:
        return self.player.is_playing()

    def stop(self) -> None:
        self.player.stop()

    def close(self) -> None:
        self.player.close()
        self.servo.close()


def create_voice(backend: str | None = None) -> HatVoice:
    """Factory: builds a HatVoice using `backend` (or, if None,
    `settings.tts_backend`) for TTS, and `settings.servo_backend` for
    motion."""
    return HatVoice(backend)
