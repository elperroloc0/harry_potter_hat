"""Top-level end-to-end verification entry point for the audio-input
subsystem: say the wake word, then speak, and see the transcript with
its detected language print out. Runs until Ctrl+C.

    python -m hat.listen
"""

from __future__ import annotations

import sys

from hat.audio.listener import VoiceInput


def main() -> None:
    print("Loading models (wake word, VAD, Whisper)...", file=sys.stderr)
    voice = VoiceInput()
    print("Ready. Say the wake word, then speak. Ctrl+C to quit.", file=sys.stderr)

    try:
        while True:
            print("\n[waiting for wake word...]")
            woke = voice.wait_for_wake()
            if not woke:
                continue

            print("*** WAKE WORD DETECTED *** \a")  # \a: terminal bell as a "beep"
            print("[listening for your phrase...]")
            utterance = voice.listen_once()

            if utterance is None:
                print("(no speech detected before timeout)")
                continue

            t = utterance.transcript
            print(
                f'[{t.lang}] "{t.text}"  '
                f"(lang_confidence={t.lang_confidence:.2f}, stt_latency={t.latency_s:.2f}s)"
            )
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        voice.close()


if __name__ == "__main__":
    main()
