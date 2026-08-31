from __future__ import annotations

from hat.tts.base import Lang


class StubVoice:
    """Test double for the real speech-out facade. Prints instead of
    speaking, so the brain/orchestrator can be developed and demoed without
    any TTS backend, audio device, or servo hardware."""

    def speak(self, text: str, lang: Lang) -> None:
        print(f"hat[{lang}]> {text}")

    def play_effect(self, name: str) -> None:
        print(f"hat> (plays {name} sound effect)")

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass
