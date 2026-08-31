"""The orchestrator: wake word -> vision -> Claude conversation -> speech.

    python -m hat.main [--no-wake] [--no-vision] [--text] [--image path.jpg]

Everything here is built against hat.audio.stub.FakeVoiceInput's interface
(wait_for_wake / listen_once / hold) so it is a drop-in swap for the real
hat.audio.listener.VoiceInput once that subsystem lands — see
build_voice_input() below, which already tries the real thing first.
"""

from __future__ import annotations

import argparse
import logging
import re
import time

from hat.audio.stub import FakeVoiceInput
from hat.brain.client import HatBrain
from hat.brain.persona import PARTING, STILL_THERE
from hat.config import settings
from hat.vision.camera import make_camera
from hat.vision.describer import OllamaDescriber

logger = logging.getLogger(__name__)

# Farewell detection against the visitor's raw transcript text — deliberately
# separate from the persona's own in-character parting reply. Spanish and
# English patterns kept as named pieces so tests can exercise each half.
FAREWELL_ES = r"adi[oó]s|hasta luego|chao|me voy|buenas noches"
FAREWELL_EN = r"goodbye|bye|see you|farewell|good night"
FAREWELL_RE = re.compile(f"{FAREWELL_ES}|{FAREWELL_EN}", re.IGNORECASE)


def is_farewell(text: str) -> bool:
    """True if the visitor's utterance contains a recognizable goodbye
    phrase, in either supported language."""
    return bool(FAREWELL_RE.search(text))


class PrintVoice:
    """Fallback speech-out adapter used when hat.speech.create_voice isn't
    importable yet (parallel agent still in progress, or its backend isn't
    configured on this machine). Same shape as StubVoice/HatVoice, so --text
    mode always works standalone regardless of hat/speech.py's state."""

    def speak(self, text: str, lang: str, block: bool = True) -> None:
        print(f"hat[{lang}]> {text}")

    def play_effect(self, name: str) -> None:
        print(f"hat> (plays {name} sound effect)")

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def build_voice():
    """Real HatVoice via hat.speech.create_voice() if available, else a
    console-printing fallback so this module never hard-depends on the
    speech subsystem being finished."""
    try:
        from hat.speech import create_voice
    except ImportError:
        logger.warning("hat.speech.create_voice not found; falling back to console voice")
        return PrintVoice()
    try:
        return create_voice()
    except Exception:
        logger.warning("hat.speech.create_voice() failed; falling back to console voice", exc_info=True)
        return PrintVoice()


def build_voice_input(args):
    """FakeVoiceInput under --text (or if the real listener isn't built
    yet); hat.audio.listener.VoiceInput otherwise. Both satisfy the same
    wait_for_wake / listen_once / hold interface."""
    if args.text:
        return FakeVoiceInput()
    try:
        from hat.audio.listener import VoiceInput
    except ImportError:
        logger.warning("hat.audio.listener.VoiceInput not found; falling back to FakeVoiceInput")
        return FakeVoiceInput()
    try:
        return VoiceInput()
    except Exception:
        logger.warning("hat.audio.listener.VoiceInput() failed; falling back to FakeVoiceInput", exc_info=True)
        return FakeVoiceInput()


def run_conversation(brain: HatBrain, voice_input, voice) -> None:
    """One conversation, from just after the greeting until the visitor
    leaves (goodbye, silence, or the session time cap)."""
    started = time.monotonic()
    last_lang = settings.default_lang
    misses = 0

    while True:
        if time.monotonic() - started > settings.session_max_s:
            logger.info("session_max_s (%.0fs) reached; ending conversation", settings.session_max_s)
            return

        utterance = voice_input.listen_once(timeout=settings.listen_timeout_s)
        if utterance is None or not utterance.transcript.text.strip():
            misses += 1
            if misses == 1:
                with voice_input.hold():
                    voice.speak(STILL_THERE[last_lang], last_lang)
                continue
            with voice_input.hold():
                voice.speak(PARTING[last_lang], last_lang)
            return

        misses = 0
        t = utterance.transcript
        last_lang = t.lang

        reply = brain.reply(t)
        with voice_input.hold():
            voice.speak(reply, last_lang)

        if is_farewell(t.text):
            return


def run(args) -> None:
    # --image always wins, even under --no-vision; otherwise --no-vision
    # means no camera at all, and absent both, make_camera() picks the
    # profile-appropriate live camera.
    cam = None if (args.no_vision and not args.image) else make_camera(settings, args.image)
    describer = OllamaDescriber() if cam else None
    if describer:
        describer.warm_up()

    brain = HatBrain(settings)
    voice_input = build_voice_input(args)
    voice = build_voice()

    try:
        while True:
            voice_input.wait_for_wake()

            try:
                voice.play_effect("wake_ack")

                appearance = None
                if cam:
                    jpeg = cam.capture_jpeg()
                    appearance = describer.describe(jpeg) if jpeg else None

                greeting = brain.start_session(appearance, settings.default_lang)
                with voice_input.hold():
                    voice.speak(greeting, settings.default_lang)

                run_conversation(brain, voice_input, voice)
            except Exception:
                # A hiccup in one visit (TTS glitch, camera dropout, etc.)
                # should not take the whole prop down for the next visitor.
                logger.exception("Unexpected error during a conversation; recovering for next wake")
            finally:
                brain.end_session()
    except (KeyboardInterrupt, EOFError):
        # EOFError: stdin closed under --text/FakeVoiceInput (e.g. piped
        # input ran out, or Ctrl-D at a wait_for_wake prompt) — a clean way
        # to stop the whole program, not just one conversation.
        print()
    finally:
        voice.close()
        if cam:
            cam.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Talking Sorting Hat orchestrator")
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="(reserved for the real wake-word listener; FakeVoiceInput.wait_for_wake "
        "is already a manual Enter-press trigger, so this is currently a no-op)",
    )
    parser.add_argument("--no-vision", action="store_true", help="skip the camera/describer entirely")
    parser.add_argument(
        "--text",
        action="store_true",
        help="use stdin/stdout (FakeVoiceInput) for input instead of the real microphone listener",
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="use a static image instead of the live camera (wins over --no-vision)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run(args)


if __name__ == "__main__":
    main()
