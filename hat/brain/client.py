from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import anthropic

from hat.audio.types import Transcript
from hat.brain.persona import (
    FALLBACK_LINES,
    OPENING_MANNERS,
    RITUAL_OPENING_SEED,
    RITUAL_OPENING_SEED_SEATED,
    SEATED_NOTE,
    SORTING_FLAVOR_HINTS,
    SYSTEM_PROMPT,
    TOOLS,
    split_beats,
)

logger = logging.getLogger(__name__)


@dataclass
class HatTurn:
    """One turn of the hat: what to say out loud, and what it silently
    decided to do. The orchestrator always speaks `beats` first and only
    then acts on `tool_uses` -- the spoken line ("let me look at you") is
    what makes the visitor turn toward the camera in the first place."""

    beats: list[str] = field(default_factory=list)
    tool_uses: list = field(default_factory=list)

    def wants(self, tool_name: str) -> bool:
        return any(block.name == tool_name for block in self.tool_uses)


class ConversationManager:
    """Owns one visit's message history and talks to the Claude API.

    Message shape follows the Anthropic Messages API. Assistant turns are
    stored as the response's full `content` block list, never as extracted
    text: that is what preserves `tool_use` blocks (and, on Opus 5, thinking
    blocks) so they can be echoed back unchanged on the next request. Tool
    results go back as a user turn of `tool_result` blocks; the visitor's own
    speech stays a plain string.

    The system prompt and the tool list are both frozen and passed on every
    call, so the cached prefix (tools render before system) stays stable for
    the whole visit.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        system: str,
        max_tokens: int = 500,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.last_response = None  # last raw API response, for --debug usage printing
        self._session_lang = "en"

    @property
    def turns(self) -> int:
        return len(self.messages)

    def start_ritual(self, lang: str, seated: bool = False) -> HatTurn:
        """Reset history, seed the castle's opening note, and take the hat's
        first turn. `seated` covers the visitor who simply dropped into the
        chair without a wake word -- no photo is possible from behind."""
        template = RITUAL_OPENING_SEED_SEATED if seated else RITUAL_OPENING_SEED
        seed = template.format(lang=lang)
        hint = random.choice(SORTING_FLAVOR_HINTS)
        manner = random.choice(OPENING_MANNERS)
        seed += (
            f" A private impression stirs in you as you sense them: {hint}."
            f" Open {manner} — your own words, never a line you have used before."
        )
        self.messages = [{"role": "user", "content": seed}]
        self._session_lang = lang
        return self._call(fallback_lang=lang)

    def send(self, user_text: str, lang: str) -> HatTurn:
        """A real utterance heard in front of the hat -- the visitor's, or
        anyone else's. No speaker attribution: the persona decides from
        content alone what it is hearing."""
        self.messages.append({"role": "user", "content": f"{user_text}\n[lang: {lang}]"})
        self._session_lang = lang
        return self._call(fallback_lang=lang)

    def note_seated(self, lang: str) -> HatTurn:
        """The PIR fired: the visitor has sat down. Injected as the castle's
        note rather than a role:system message -- history ends on an assistant
        turn here, and mid-conversation system messages must follow a user
        turn. This is the gate the ritual hangs on; no house is named before
        the model sees it."""
        self.messages.append({"role": "user", "content": SEATED_NOTE.format(lang=lang)})
        return self._call(fallback_lang=lang)

    def submit_tool_results(self, results: list[dict], lang: str) -> HatTurn:
        """Hand back every tool_result for the turn in a single user message
        (splitting them across messages trains the model out of parallel
        calls) and take the follow-up turn."""
        self.messages.append({"role": "user", "content": results})
        return self._call(fallback_lang=lang)

    def reset(self) -> None:
        self.messages = []

    def _call(self, fallback_lang: str) -> HatTurn:
        fallback = HatTurn(beats=[FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])])
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": "low"},
                cache_control={"type": "ephemeral"},
                system=self.system,
                tools=TOOLS,
                messages=self.messages,
            )
        except anthropic.RateLimitError:
            logger.warning("Claude API rate limited", exc_info=True)
            return fallback
        except anthropic.APIStatusError:
            logger.warning("Claude API returned an error status", exc_info=True)
            return fallback
        except anthropic.APIConnectionError:
            logger.warning("Could not connect to the Claude API", exc_info=True)
            return fallback
        except Exception:
            logger.exception("Unexpected error calling the Claude API")
            return fallback

        self.last_response = response

        if response.stop_reason == "max_tokens":
            logger.warning("Claude response was truncated at max_tokens")
        elif response.stop_reason == "refusal":
            # Not appended as a real turn: leaving a refusal in history would
            # poison every later request in the visit.
            logger.warning("Claude refused the request (stop_reason=refusal)")
            return fallback

        # The whole block list goes back verbatim -- tool_use blocks must
        # survive to be paired with their results, and thinking blocks must
        # be echoed unchanged on the same model.
        self.messages.append({"role": "assistant", "content": response.content})

        text = "\n".join(b.text for b in response.content if b.type == "text")
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not text and not tool_uses:
            logger.warning("Claude response had neither text nor tool use")
            return fallback

        return HatTurn(beats=split_beats(text), tool_uses=tool_uses)


class HatBrain:
    """Facade the orchestrator talks to: what was heard in, what to say (and
    silently do) out. Never raises — API hiccups degrade to an in-character
    fallback line instead of crashing the prop mid-visit."""

    def __init__(self, settings) -> None:
        self.settings = settings
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.conv = ConversationManager(
            client=client,
            model=settings.claude_model,
            system=SYSTEM_PROMPT,
            max_tokens=settings.max_reply_tokens,
        )

    def start_ritual(self, lang: str | None = None, seated: bool = False) -> HatTurn:
        lang = lang or self.settings.default_lang
        return self._guard(lambda: self.conv.start_ritual(lang, seated), lang, "start the ritual")

    def reply(self, t: Transcript) -> HatTurn:
        return self._guard(lambda: self.conv.send(t.text, t.lang), t.lang, "get a reply")

    def note_seated(self, lang: str) -> HatTurn:
        return self._guard(lambda: self.conv.note_seated(lang), lang, "handle the seated note")

    def submit_tool_results(self, results: list[dict], lang: str) -> HatTurn:
        return self._guard(
            lambda: self.conv.submit_tool_results(results, lang), lang, "submit tool results"
        )

    def end_session(self) -> None:
        self.conv.reset()

    def _guard(self, call, lang: str, what: str) -> HatTurn:
        try:
            return call()
        except Exception:
            logger.exception("Failed to %s", what)
            return HatTurn(beats=[FALLBACK_LINES.get(lang, FALLBACK_LINES["en"])])
