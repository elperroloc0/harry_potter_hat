from __future__ import annotations

import logging

import anthropic

from hat.audio.types import Transcript
from hat.brain.persona import (
    FALLBACK_LINES,
    GREETING_NO_APPEARANCE,
    GREETING_WITH_APPEARANCE,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class ConversationManager:
    """Owns one conversation's message history and talks to the Claude API.

    Message shape follows the Anthropic Messages API: self.messages is a
    list of {"role": "user"|"assistant", "content": str} dicts. The system
    prompt is frozen and passed on every call (never mutated) so its cache
    entry stays stable across the whole session.
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

    def start_session(self, appearance: str | None, lang: str) -> None:
        """Reset history and seed it with the castle's greeting note."""
        if appearance:
            seed = GREETING_WITH_APPEARANCE.format(description=appearance, lang=lang)
        else:
            seed = GREETING_NO_APPEARANCE.format(lang=lang)
        self.messages = [{"role": "user", "content": seed}]
        self._session_lang = lang

    def reset(self) -> None:
        self.messages = []

    def send_seed(self) -> str:
        """Call the API with the just-seeded greeting turn and return the
        assistant's greeting text, appending it to history."""
        return self._call(fallback_lang=self._session_lang)

    def send(self, user_text: str, lang: str) -> str:
        self.messages.append({"role": "user", "content": f"{user_text}\n[lang: {lang}]"})
        return self._call(fallback_lang=lang)

    def _call(self, fallback_lang: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": "low"},
                cache_control={"type": "ephemeral"},
                system=self.system,
                messages=self.messages,
            )
        except anthropic.RateLimitError:
            logger.warning("Claude API rate limited", exc_info=True)
            return FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])
        except anthropic.APIStatusError:
            logger.warning("Claude API returned an error status", exc_info=True)
            return FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])
        except anthropic.APIConnectionError:
            logger.warning("Could not connect to the Claude API", exc_info=True)
            return FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])
        except Exception:
            logger.exception("Unexpected error calling the Claude API")
            return FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])

        self.last_response = response

        if response.stop_reason == "max_tokens":
            logger.warning("Claude response was truncated at max_tokens")
        elif response.stop_reason == "refusal":
            logger.warning("Claude refused the request (stop_reason=refusal)")
            return FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            logger.warning("Claude response had no text block")
            return FALLBACK_LINES.get(fallback_lang, FALLBACK_LINES["en"])

        self.messages.append({"role": "assistant", "content": text})
        return text


class HatBrain:
    """Facade the orchestrator talks to: appearance + language in, spoken
    reply text out. Never raises — API hiccups degrade to an in-character
    fallback line instead of crashing the prop mid-conversation."""

    def __init__(self, settings) -> None:
        self.settings = settings
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.conv = ConversationManager(
            client=client,
            model=settings.claude_model,
            system=SYSTEM_PROMPT,
            max_tokens=settings.max_reply_tokens,
        )

    def start_session(self, appearance: str | None, lang: str | None = None) -> str:
        lang = lang or self.settings.default_lang
        try:
            self.conv.start_session(appearance, lang)
            return self.conv.send_seed()
        except Exception:
            logger.exception("Failed to start a new session")
            return FALLBACK_LINES.get(lang, FALLBACK_LINES["en"])

    def reply(self, t: Transcript) -> str:
        try:
            return self.conv.send(t.text, t.lang)
        except Exception:
            logger.exception("Failed to get a reply")
            return FALLBACK_LINES.get(t.lang, FALLBACK_LINES["en"])

    def end_session(self) -> None:
        self.conv.reset()
