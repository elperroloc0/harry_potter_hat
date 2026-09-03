"""Text-only REPL for the Sorting Hat persona — no audio, no camera required.

    python -m hat.chat [--image path.jpg] [--lang es|en] [--debug]

Dev convenience for iterating on the persona/brain without the rest of the
stack. It runs the same turn loop as the real orchestrator (hat.main), so
tool calls behave the same way: take_photo describes --image if you gave
one, /sit stands in for the PIR sensor firing when the visitor sits down,
and end_session ends the visit.

The real pipeline gets `lang` from Whisper (STT), never from typed text —
the heuristic here (`guess_lang`) exists only so this REPL is usable
bilingually without a --lang flag.
"""

from __future__ import annotations

import argparse
import re

from hat.brain.client import HatBrain
from hat.brain.persona import NO_SIGHT_RESULT
from hat.audio.types import Transcript
from hat.config import settings
from hat.vision.camera import StaticImageStub
from hat.vision.describer import OllamaDescriber

SIT_TOKEN = "/sit"

# Small, deliberately simple stopword lists — a dev convenience, not a real
# language detector. Accented characters and inverted punctuation are strong
# unambiguous Spanish signals and are weighted accordingly.
_ES_STOPWORDS = {
    "el", "la", "los", "las", "que", "qué", "como", "cómo", "gracias", "hola",
    "por", "favor", "si", "sí", "usted", "ustedes", "buenas", "tardes", "noches",
    "donde", "dónde", "cual", "cuál", "es", "mi", "tu", "yo", "y", "de", "en",
    "para", "porque", "quiero", "puedo", "soy", "eres", "somos", "distribuyeme",
    "distribúyeme", "casa",
}
_EN_STOPWORDS = {
    "the", "what", "how", "thanks", "thank", "hello", "hi", "hey", "please",
    "yes", "you", "your", "where", "which", "is", "my", "i", "am", "are", "we",
    "for", "because", "want", "can", "sort", "house",
}
_SPANISH_CHARS = set("áéíóúñ¿¡")


def guess_lang(text: str) -> str:
    """Heuristic es/en guess for typed REPL input. Not used by the real
    pipeline — Whisper (STT) supplies `lang` on a real Transcript."""
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ']+", text.lower())
    es_score = sum(1 for w in words if w in _ES_STOPWORDS)
    en_score = sum(1 for w in words if w in _EN_STOPWORDS)
    if any(ch in text for ch in _SPANISH_CHARS):
        es_score += 1
    if es_score > en_score:
        return "es"
    if en_score > es_score:
        return "en"
    return settings.default_lang


def _print_usage(response) -> None:
    if response is None or getattr(response, "usage", None) is None:
        print("(debug: no usage info available)")
        return
    usage = response.usage
    parts = [f"input_tokens={usage.input_tokens}", f"output_tokens={usage.output_tokens}"]
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_read is not None:
        parts.append(f"cache_read_input_tokens={cache_read}")
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    if cache_creation is not None:
        parts.append(f"cache_creation_input_tokens={cache_creation}")
    print(f"(debug: {', '.join(parts)})")


def build_appearance(image_path: str | None) -> str:
    """take_photo's implementation for the REPL: describe --image if one was
    given, else report that the sight didn't come — same contract as
    hat.main.capture_appearance."""
    if not image_path:
        return NO_SIGHT_RESULT
    cam = StaticImageStub(image_path)
    jpeg = cam.capture_jpeg()
    if jpeg is None:
        print(f"(vision: could not read image at {image_path})")
        return NO_SIGHT_RESULT
    appearance = OllamaDescriber().describe(jpeg)
    if appearance is None:
        print(
            "(vision: Ollama unreachable, vision model unavailable, or nobody in frame. "
            "Install with `brew install ollama && ollama pull qwen2.5vl:7b`.)"
        )
        return NO_SIGHT_RESULT
    print(f"(vision: {appearance})")
    return appearance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Talking Sorting Hat — text-only persona REPL")
    parser.add_argument("--no-vision", action="store_true", help="(reserved; take_photo needs --image here)")
    parser.add_argument("--image", metavar="PATH", help="the photo take_photo 'sees' when the hat looks at you")
    parser.add_argument("--lang", choices=["es", "en"], help="force a language instead of heuristically guessing typed input")
    parser.add_argument("--debug", action="store_true", help="print response.usage after each turn")
    args = parser.parse_args(argv)

    image = None if args.no_vision else args.image
    lang = args.lang or settings.default_lang
    brain = HatBrain(settings)

    print(f"(type {SIT_TOKEN} to simulate sitting down in the chair; Ctrl-D to quit)")

    turn = brain.start_ritual(lang)
    seated_handled = False

    try:
        while True:
            for beat in turn.beats:
                print(f"hat> {beat}")
            if args.debug:
                _print_usage(brain.conv.last_response)

            if turn.wants("end_session"):
                print("(the hat returns to waiting for the next visitor)")
                break

            if turn.tool_uses:
                results = []
                for block in turn.tool_uses:
                    content = (
                        build_appearance(image)
                        if block.name == "take_photo"
                        else "That is not among your powers."
                    )
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": content}
                    )
                turn = brain.submit_tool_results(results, lang)
                continue

            try:
                raw = input("you> ").strip()
            except EOFError:
                print()
                break
            if not raw:
                continue

            if raw == SIT_TOKEN:
                if seated_handled:
                    print("(already seated)")
                    continue
                seated_handled = True
                turn = brain.note_seated(lang)
                continue

            lang = args.lang or guess_lang(raw)
            turn = brain.reply(Transcript(text=raw, lang=lang))
    finally:
        brain.end_session()


if __name__ == "__main__":
    main()
