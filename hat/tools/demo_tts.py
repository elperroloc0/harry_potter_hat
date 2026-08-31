"""Synthesize one line of text and optionally play it.

Usage:
    python -m hat.tools.demo_tts "Hola, joven mago" es [--play] [--backend say|elevenlabs]
"""
from __future__ import annotations

import argparse
import sys

from hat.tts.base import SynthesisError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="text to synthesize")
    parser.add_argument("lang", choices=["es", "en"], help="language code")
    parser.add_argument("--play", action="store_true", help="play the synthesized audio")
    parser.add_argument(
        "--backend", choices=["say", "elevenlabs"], default="say", help="TTS backend to use"
    )
    args = parser.parse_args(argv)

    if args.backend == "elevenlabs":
        from hat.tts.elevenlabs_tts import ElevenLabsSynth

        synth = ElevenLabsSynth()
    else:
        from hat.tts.macos_say import MacSaySynth

        synth = MacSaySynth()

    try:
        audio = synth.synth(args.text, args.lang)  # type: ignore[arg-type]
    except SynthesisError as exc:
        print(f"synthesis failed: {exc}", file=sys.stderr)
        return 1

    print(f"PcmAudio(n={len(audio.samples)}, sr={audio.sample_rate}, duration={audio.duration_s:.3f}s)")

    if args.play:
        from hat.audio.player import AudioPlayer

        player = AudioPlayer()
        player.play(audio)
        player.wait()
        player.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
