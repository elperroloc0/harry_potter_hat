"""Text-only REPL for the Sorting Hat persona — no audio, no camera required.

    python -m hat.chat [--no-vision] [--image path.jpg] [--lang es|en] [--debug]

Dev convenience for iterating on the persona/brain without the rest of the
stack. The real pipeline gets `lang` from Whisper (STT), never from typed
text — the heuristic here (`guess_lang`) exists only so this REPL is usable
bilingually without a --lang flag.
"""

from __future__ import annotations

import argparse
import re
import sys

from hat.brain.client import HatBrain
from hat.audio.types import Transcript
from hat.config import settings
from hat.vision.camera import StaticImageStub
from hat.vision.describer import OllamaDescriber

FAREWELL_WORDS = {"adios", "adiós", "goodbye", "bye"}

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


def build_appearance(image_path: str) -> str | None:
    cam = StaticImageStub(image_path)
    jpeg = cam.capture_jpeg()
    if jpeg is None:
        print(f"(vision: could not read image at {image_path}, continuing with no appearance)")
        return None
    describer = OllamaDescriber()
    appearance = describer.describe(jpeg)
    if appearance is None:
        print(
            "(vision: Ollama unreachable or vision model unavailable — continuing with no "
            "appearance. Install with `brew install ollama && ollama pull qwen3-vl:8b`.)"
        )
    return appearance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Talking Sorting Hat — text-only persona REPL")
    parser.add_argument("--no-vision", action="store_true", help="skip vision entirely (default if no --image)")
    parser.add_argument("--image", metavar="PATH", help="describe this image and seed it as appearance context")
    parser.add_argument("--lang", choices=["es", "en"], help="force a language instead of heuristically guessing typed input")
    parser.add_argument("--debug", action="store_true", help="print response.usage after each turn")
    args = parser.parse_args(argv)

    appearance = None
    if args.image:
        appearance = build_appearance(args.image)
    elif not args.no_vision:
        # Neither flag given: default behavior is no vision (per spec — vision
        # requires an explicit --image in this text-only REPL).
        pass

    lang = args.lang or settings.default_lang

    brain = HatBrain(settings)
    greeting = brain.start_session(appearance, lang)
    print(f"hat> {greeting}")
    if args.debug:
        _print_usage(brain.conv.last_response)

    while True:
        try:
            raw = input("you> ")
        except EOFError:
            print()
            break
        text = raw.strip()
        if not text:
            continue
        if text.lower() in FAREWELL_WORDS:
            farewell_lang = args.lang or guess_lang(text)
            reply = brain.reply(Transcript(text=text, lang=farewell_lang))
            print(f"hat> {reply}")
            if args.debug:
                _print_usage(brain.conv.last_response)
            break

        turn_lang = args.lang or guess_lang(text)
        reply = brain.reply(Transcript(text=text, lang=turn_lang))
        print(f"hat> {reply}")
        if args.debug:
            _print_usage(brain.conv.last_response)

    brain.end_session()


if __name__ == "__main__":
    main()
