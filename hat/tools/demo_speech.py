"""End-to-end speech-out milestone demo: synthesize, play, and lip-sync two
lines (Spanish then English) through the real HatVoice facade. With the
default 'mock' servo backend, watch the console mouth-bar animate in time
with the actually-audible speech.

Usage:
    python -m hat.tools.demo_speech [--backend say|elevenlabs]
"""
from __future__ import annotations

import argparse

from hat.speech import create_voice

ES_LINE = "Hola, joven mago. El sombrero seleccionador te está observando con mucho interés."
EN_LINE = "Welcome, young wizard. The Sorting Hat is watching you with great interest."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["say", "elevenlabs"], default=None, help="TTS backend to use"
    )
    args = parser.parse_args(argv)

    voice = create_voice(args.backend)
    print(f"[demo_speech] backend={voice.backend_name!r} servo={type(voice.servo).__name__}")
    try:
        print("\n[demo_speech] speaking Spanish line...")
        voice.speak(ES_LINE, "es")
        print("\n[demo_speech] speaking English line...")
        voice.speak(EN_LINE, "en")
    finally:
        voice.close()

    print("\n[demo_speech] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
