"""The orchestrator: wake word -> Claude conversation -> speech, camera, PIR.

    python -m hat.main [--no-wake] [--no-vision] [--text] [--image path.jpg]

A visit is a live conversation the model itself leads, not a fixed script.
Two physical events feed into it:

  * the wake word (hat.wake, via VoiceInput.wait_for_wake) starts a visit;
  * the PIR sensor (hat.sensors.motion) fires when the visitor actually SITS
    DOWN in the chair -- its range is turned down to the minimum on the board
    itself, so it ignores anyone at a distance. That event is injected into
    the conversation as the castle's note, and it is the gate the ritual
    hangs on: the hat does not name a house before it arrives.

Everything else -- when to greet, what to ask, when to look at the visitor
(take_photo), when the visit is over (end_session) -- is the model's own
decision, made in the system prompt, not in branches here. In particular
there is deliberately no "an adult is helping" mode: whatever is said in
front of the hat, by whoever, goes into the same conversation.
"""

from __future__ import annotations

import argparse
import logging
import time

from hat.audio.stub import FakeVoiceInput
from hat.brain.client import HatBrain, HatTurn
from hat.brain.persona import NO_SIGHT_RESULT, PARTING, STILL_THERE
from hat.config import settings
from hat.sensors.motion import ManualMotionSensor
from hat.vision.camera import make_camera
from hat.vision.describer import OllamaDescriber

logger = logging.getLogger(__name__)

# Typed at the `you>` prompt under --text to play the part of the PIR sensor.
SIT_TOKEN = "/sit"


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


def build_motion_sensor(args, stdin_free: bool = False):
    """PIRMotionSensor under real (non---text) Pi runs, else
    ManualMotionSensor. This is the "visitor sat down" trigger; its range is
    a hardware potentiometer setting on the sensor board, never emulated
    here.

    `stdin_free` says whether nothing else is reading stdin -- only then may
    the manual fallback claim Enter as its "visitor sat down" signal. When
    voice input has itself fallen back to FakeVoiceInput, stdin is already
    spoken for and two readers would steal each other's lines.
    """
    if args.text:
        # FakeVoiceInput owns stdin here, so the fallback is the typed /sit
        # token rather than a competing stdin reader.
        return ManualMotionSensor()
    try:
        from hat.sensors.motion import PIRMotionSensor
    except ImportError:
        logger.warning("hat.sensors.motion.PIRMotionSensor not found; falling back to manual")
        return ManualMotionSensor(watch_stdin=stdin_free)
    try:
        return PIRMotionSensor(settings.motion_sensor_pin)
    except Exception:
        logger.warning("PIRMotionSensor() failed; falling back to manual", exc_info=True)
        return ManualMotionSensor(watch_stdin=stdin_free)


def capture_appearance(cam, describer) -> str:
    """Run take_photo: one frame, described locally by Ollama. Every failure
    path -- no camera, camera error, Ollama down, nobody recognizable in
    frame -- returns the same in-character "no sight" result rather than an
    error, so the persona's "invent nothing" rule fires instead of an
    apology about equipment."""
    if cam is None or describer is None:
        return NO_SIGHT_RESULT
    try:
        jpeg = cam.capture_jpeg()
    except Exception:
        logger.warning("Camera capture failed during take_photo", exc_info=True)
        return NO_SIGHT_RESULT
    if not jpeg:
        return NO_SIGHT_RESULT
    return describer.describe(jpeg) or NO_SIGHT_RESULT


def run_tools(turn: HatTurn, cam, describer) -> list[dict]:
    """Execute the turn's tool calls and build the tool_result blocks. All
    of them go back in a single user message (see submit_tool_results).
    end_session never reaches here -- run_ritual returns before that."""
    results = []
    for block in turn.tool_uses:
        if block.name == "take_photo":
            content = capture_appearance(cam, describer)
        else:
            logger.warning("Model called an unknown tool: %s", block.name)
            content = "That is not among your powers."
        results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
    return results


def simulate_seated(motion) -> None:
    """--text convenience: let a typed /sit stand in for the PIR sensor."""
    simulate = getattr(motion, "simulate", None)
    if simulate is None:
        logger.info("%s ignored: this motion sensor is the real one", SIT_TOKEN)
        return
    simulate()


def run_ritual(brain, voice_input, voice, motion, cam, describer, lang, seated_at_start=False) -> None:
    """One visit, start to finish. Speaks each turn, runs whatever the model
    silently decided to do, then waits for the world to say something back --
    either words (from anyone) or the PIR firing."""
    with voice_input.hold():
        voice.play_effect("wake_ack")

    turn = brain.start_ritual(lang, seated=seated_at_start)
    started = time.monotonic()
    current_lang = lang
    seated_handled = seated_at_start
    misses = 0

    while True:
        # Speak first, always: the words are what make the visitor look at
        # the hat, so take_photo must fire *after* they have been said.
        if turn.beats:
            with voice_input.hold():
                for beat in turn.beats:
                    voice.speak(beat, current_lang)

        if turn.wants("end_session"):
            logger.info("Model ended the visit")
            return

        if turn.tool_uses:
            turn = brain.submit_tool_results(run_tools(turn, cam, describer), current_lang)
            continue

        if time.monotonic() - started > settings.session_max_s:
            logger.info("session_max_s (%.0fs) reached; ending visit", settings.session_max_s)
            return

        # Wait until something produces a new turn: the visitor sitting down,
        # or anyone saying anything. This inner loop only exits with a fresh
        # turn in hand (or by ending the visit).
        while True:
            if not seated_handled and motion.poll():
                seated_handled = True
                turn = brain.note_seated(current_lang)
                break

            utterance = voice_input.listen_once(
                timeout=settings.listen_timeout_s,
                cancel=(None if seated_handled else motion.pending),
            )

            if utterance is None or not utterance.transcript.text.strip():
                # A listen cut short by the PIR isn't a silence -- go round
                # and let the seated branch above pick it up.
                if not seated_handled and motion.pending():
                    continue
                misses += 1
                if misses >= 2:
                    with voice_input.hold():
                        voice.speak(PARTING[current_lang], current_lang)
                    return
                with voice_input.hold():
                    voice.speak(STILL_THERE[current_lang], current_lang)
                continue

            text = utterance.transcript.text.strip()
            if text == SIT_TOKEN:
                simulate_seated(motion)
                continue

            misses = 0
            current_lang = utterance.transcript.lang
            turn = brain.reply(utterance.transcript)
            break


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
    # Enter can only stand in for the PIR if the voice side isn't also
    # reading stdin (i.e. it got a real microphone, not the stub).
    motion = build_motion_sensor(args, stdin_free=not isinstance(voice_input, FakeVoiceInput))
    motion.start_watching()

    lang = settings.default_lang

    try:
        while True:
            # Idle. Two ways out: the wake word, or someone who skips the
            # ceremony and simply sits down in the chair.
            woke = voice_input.wait_for_wake(cancel=motion.pending)
            just_sat = motion.poll()
            seated_at_start = just_sat and not woke
            if not woke and not seated_at_start:
                continue

            try:
                run_ritual(
                    brain, voice_input, voice, motion, cam, describer, lang, seated_at_start
                )
            except Exception:
                # A hiccup in one visit (TTS glitch, camera dropout, etc.)
                # should not take the whole prop down for the next visitor.
                logger.exception("Unexpected error during a visit; recovering for the next one")
            finally:
                brain.end_session()
    except (KeyboardInterrupt, EOFError):
        # EOFError: stdin closed under --text/FakeVoiceInput or
        # ManualMotionSensor (e.g. piped input ran out, or Ctrl-D at a
        # prompt) — a clean way to stop the whole program, not just one visit.
        print()
    finally:
        voice.close()
        motion.close()
        if cam:
            cam.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Talking Sorting Hat orchestrator")
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="(reserved -- currently a no-op; --text already gets you a manual "
        "Enter-press trigger via FakeVoiceInput without this flag)",
    )
    parser.add_argument("--no-vision", action="store_true", help="skip the camera/describer entirely")
    parser.add_argument(
        "--text",
        action="store_true",
        help="use stdin/stdout (FakeVoiceInput) for input instead of the real microphone listener; "
        f"type {SIT_TOKEN} to simulate the visitor sitting down",
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
