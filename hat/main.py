"""The orchestrator: microphone -> Claude conversation -> speech, camera, PIR.

    python -m hat.main [--no-vision] [--text] [--image path.jpg]

The hat is simply awake. It listens, answers, and keeps listening; there are
no visits, no sessions and no goodbyes, and nothing it says is scripted --
every word comes from the model. Two pieces of hardware feed into that
conversation:

  * the camera, when the model calls take_photo, which it does while the
    person is still standing in front of it;
  * the PIR sensor, which fires when someone actually SITS DOWN in the chair
    -- its range is turned right down on the board itself, so it ignores
    anyone at a distance. That is what sort_visitor waits for.

Sorting is a function of the conversation, not the shape of the program: the
model calls sort_visitor when it has been asked for a house, or when the talk
has plainly arrived there. Everything else -- what to ask, when to look, when
to sort -- is decided in the system prompt rather than in branches here.
There is deliberately no "an adult is helping" mode: whatever is said in
front of the hat, by whoever, goes into the same conversation.
"""
from __future__ import annotations

import argparse
import logging
import time

from hat.audio.stub import FakeVoiceInput
from hat.brain.client import HatBrain, HatTurn
from hat.brain.persona import NOT_SEATED_NOTE, NO_SIGHT_RESULT, SEATED_NOTE
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


def wait_for_seated(motion, brain) -> str:
    """sort_visitor: block until the chair reports someone in it.

    Whoever was detected sitting earlier does not count -- the event is
    drained first -- because the hat has only just asked them to sit. If
    nobody ever does, the ceremony lapses quietly and the conversation
    carries on; the hat is not going anywhere."""
    motion.poll()
    deadline = time.monotonic() + settings.seat_timeout_s
    while time.monotonic() < deadline:
        if motion.poll():
            return brain.sorting_note(SEATED_NOTE)
        time.sleep(0.1)
    logger.info("Nobody sat down within %.0fs; sorting lapsed", settings.seat_timeout_s)
    return NOT_SEATED_NOTE


def run_tools(turn: HatTurn, brain, motion, cam, describer) -> list[dict]:
    """Execute the turn's tool calls and build the tool_result blocks. All
    of them go back in a single user message (see submit_tool_results)."""
    results = []
    for block in turn.tool_uses:
        if block.name == "take_photo":
            content = capture_appearance(cam, describer)
        elif block.name == "sort_visitor":
            content = wait_for_seated(motion, brain)
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


def deliver(turn: HatTurn, brain, voice_input, voice, motion, cam, describer, lang) -> None:
    """Speak a turn and run whatever it silently decided to do, following the
    tool chain until the model has nothing left to act on. Speech always goes
    first: the words are what make someone look at the hat, or sit down, so
    take_photo and sort_visitor must fire after they have been said."""
    while True:
        if turn.beats:
            with voice_input.hold():
                for beat in turn.beats:
                    voice.speak(beat, lang)
        if not turn.tool_uses:
            return
        turn = brain.submit_tool_results(run_tools(turn, brain, motion, cam, describer), lang)


def converse(brain, voice_input, voice, motion, cam, describer, lang) -> None:
    """Listen, answer, keep listening. Runs for as long as the hat is
    powered: there is no session to end and nothing to wait to be woken by.

    Silence is answered with silence -- no stock "are you still there", which
    is how a prop ends up talking to an empty room. A long enough gap only
    means the conversation is dropped, quietly, so the next person does not
    inherit the last one's answers."""
    current_lang = lang
    last_heard = time.monotonic()

    while True:
        utterance = voice_input.listen_once(timeout=settings.listen_timeout_s)

        if utterance is None or not utterance.transcript.text.strip():
            if brain.conv.turns and time.monotonic() - last_heard > settings.session_max_s:
                logger.info("Quiet for %.0fs; forgetting the conversation", settings.session_max_s)
                brain.forget()
            continue

        text = utterance.transcript.text.strip()
        if text == SIT_TOKEN:
            simulate_seated(motion)
            continue

        last_heard = time.monotonic()
        current_lang = utterance.transcript.lang
        deliver(
            brain.reply(utterance.transcript),
            brain, voice_input, voice, motion, cam, describer, current_lang,
        )


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
        converse(brain, voice_input, voice, motion, cam, describer, lang)
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
