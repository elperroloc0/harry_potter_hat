from __future__ import annotations

import re
import subprocess
import tempfile
import unicodedata
import wave
from functools import lru_cache

import numpy as np

from hat.tts.base import Lang, PcmAudio, SynthesisError, Synthesizer

# Verified against `say -v '?'` on this dev Mac (macOS, en_US/es_ES locale
# voices installed by default). "Monica" is macOS's normalized ASCII spelling
# of the installed "Mónica" (es_ES) voice -- `say -v Monica` matches it fine.
# "Samantha" is the standard installed en_US voice ("Daniel" is en_GB and is
# NOT installed on this machine, so it is not used as the default).
DEFAULT_VOICES: dict[Lang, str] = {"es": "Monica", "en": "Samantha"}

_SIMPLE_VOICE_LINE = re.compile(r"^(.+?)\s{2,}[a-zA-Z]{2}[_-][A-Za-z]{2,}\s+#")


def _normalize_voice_name(name: str) -> str:
    """Fold accents/case so "Monica" matches the installed "Mónica" the same
    way `say`'s own fuzzy voice lookup does."""
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch)
    )
    return stripped.casefold().strip()


@lru_cache(maxsize=1)
def _installed_voice_names() -> frozenset[str] | None:
    """Best-effort parse of `say -v '?'`, normalized for matching. Returns
    None (skip validation) if `say` isn't available -- e.g. not on macOS."""
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    names: set[str] = set()
    for line in result.stdout.splitlines():
        m = _SIMPLE_VOICE_LINE.match(line)
        if m:
            names.add(_normalize_voice_name(m.group(1)))
    return frozenset(names)


class MacSaySynth(Synthesizer):
    """Free, offline, no-API-key TTS backend using the macOS `say` command.

    Dev-machine fallback only -- `say` does not exist on the Raspberry Pi.
    On the Pi, `settings.tts_backend` should be "elevenlabs"; this class is
    also used as HatVoice's automatic in-process fallback when the cloud
    backend fails (see hat/speech.py), so it is worth keeping working even
    in the deployed profile in case the Pi ever runs it manually for testing.
    """

    def __init__(self, voices: dict[Lang, str] | None = None) -> None:
        self.voices: dict[Lang, str] = dict(voices) if voices is not None else dict(DEFAULT_VOICES)
        print(f"[MacSaySynth] configured voices: {self.voices}")

    def synth(self, text: str, lang: Lang) -> PcmAudio:
        voice = self.voices.get(lang)
        if voice is None:
            raise SynthesisError(
                f"MacSaySynth has no voice configured for lang={lang!r} "
                f"(configured voices: {self.voices!r}). Run `say -v '?'` to "
                "list installed voices and pass a mapping that covers this lang."
            )

        # `say -v <bad-name>` does NOT exit nonzero -- it silently falls back
        # to the system default voice. That would make the hat speak Spanish
        # lines in an English voice (or vice versa) without any error, which
        # is worse than crashing, so validate against installed voices
        # up front rather than relying on the subprocess exit code.
        installed = _installed_voice_names()
        if installed is not None and _normalize_voice_name(voice) not in installed:
            raise SynthesisError(
                f"voice not installed, run say -v '?' (got voice={voice!r} for lang={lang!r})"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            try:
                subprocess.run(
                    ["say", "-v", voice, "-o", tmp.name, "--data-format=LEI16@22050", text],
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError as exc:
                raise SynthesisError(
                    "`say` binary not found -- MacSaySynth only works on macOS"
                ) from exc
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
                raise SynthesisError(
                    f"`say -v {voice!r}` failed (exit {exc.returncode}): "
                    f"{stderr or 'voice not installed, run say -v ?'}"
                ) from exc

            try:
                with wave.open(tmp.name, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)
            except (wave.Error, EOFError) as exc:
                raise SynthesisError(f"could not read `say` output wav: {exc}") from exc

        if sampwidth != 2:
            raise SynthesisError(
                f"expected 16-bit PCM from `say --data-format=LEI16@22050`, got sampwidth={sampwidth}"
            )

        samples = np.frombuffer(raw, dtype="<i2")
        if n_channels > 1:
            # Defensive -- we always request mono, but don't silently mix
            # channels wrong if a future macOS ever changes that.
            samples = samples.reshape(-1, n_channels)[:, 0].copy()
        else:
            samples = samples.copy()

        return PcmAudio(samples=samples, sample_rate=framerate)
