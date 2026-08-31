"""Pure-logic tests for the brain/orchestrator subsystem — no network calls.

Covers: persona template formatting, fallback-line completeness, farewell
detection, and ConversationManager's message-building logic against a fake
Anthropic client (so no live API key is required to validate the request
shape sent to Claude).
"""

from __future__ import annotations

import pytest

from hat.brain.client import ConversationManager
from hat.brain.persona import (
    FALLBACK_LINES,
    GREETING_NO_APPEARANCE,
    GREETING_WITH_APPEARANCE,
    PARTING,
    STILL_THERE,
    SYSTEM_PROMPT,
)
from hat.main import FAREWELL_RE, is_farewell


# --------------------------------------------------------------------------
# persona.py — templates and constants
# --------------------------------------------------------------------------


def test_system_prompt_is_nonempty_and_has_key_markers():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 500
    assert "Sorting Hat" in SYSTEM_PROMPT
    assert "[lang:" in SYSTEM_PROMPT
    assert "Gryffindor" in SYSTEM_PROMPT
    assert "Hufflepuff" in SYSTEM_PROMPT
    assert "Ravenclaw" in SYSTEM_PROMPT
    assert "Slytherin" in SYSTEM_PROMPT


def test_greeting_with_appearance_formats():
    out = GREETING_WITH_APPEARANCE.format(description="a boy in a blue scarf.", lang="en")
    assert "a boy in a blue scarf." in out
    assert "[lang: en]" in out
    assert "Castle's note" in out


def test_greeting_no_appearance_formats():
    out = GREETING_NO_APPEARANCE.format(lang="es")
    assert "[lang: es]" in out
    assert "perceive nothing" in out


@pytest.mark.parametrize("mapping", [FALLBACK_LINES, STILL_THERE, PARTING])
def test_bilingual_maps_have_both_languages(mapping):
    assert set(mapping.keys()) >= {"es", "en"}
    for lang in ("es", "en"):
        assert isinstance(mapping[lang], str)
        assert mapping[lang].strip() != ""


def test_fallback_lines_differ_by_language():
    assert FALLBACK_LINES["es"] != FALLBACK_LINES["en"]


# --------------------------------------------------------------------------
# hat.main — farewell detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "adios",
        "adiós",
        "Adiós, sombrero",
        "bueno, me voy ya",
        "hasta luego",
        "chao chao",
        "buenas noches a todos",
        "goodbye",
        "Goodbye!",
        "ok bye",
        "see you later",
        "farewell, old hat",
        "good night",
    ],
)
def test_farewell_matches(text):
    assert is_farewell(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "sort me please",
        "distribúyeme",
        "cual es mi casa",
        "me gusta el quidditch",
        "I would help my friends",
        "cuéntame sobre Hogwarts",
        "",
        "hello there",
    ],
)
def test_farewell_does_not_false_positive(text):
    assert is_farewell(text) is False


def test_farewell_re_is_case_insensitive():
    assert FAREWELL_RE.search("GOODBYE") is not None
    assert FAREWELL_RE.search("ADIÓS") is not None


# --------------------------------------------------------------------------
# ConversationManager — message-shape logic against a fake client
# --------------------------------------------------------------------------


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = stop_reason
        self.usage = None


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_start_session_seeds_single_user_turn_with_appearance():
    client = _FakeClient([_FakeResponse("Ah, promising indeed.")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)

    conv.start_session("a boy in a red jumper", "en")

    assert len(conv.messages) == 1
    assert conv.messages[0]["role"] == "user"
    assert "a boy in a red jumper" in conv.messages[0]["content"]
    assert "[lang: en]" in conv.messages[0]["content"]


def test_start_session_seeds_no_appearance_variant():
    client = _FakeClient([_FakeResponse("Welcome, traveler.")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)

    conv.start_session(None, "es")

    assert len(conv.messages) == 1
    assert "perceive nothing" in conv.messages[0]["content"]
    assert "[lang: es]" in conv.messages[0]["content"]


def test_send_seed_appends_assistant_turn_and_returns_text():
    client = _FakeClient([_FakeResponse("Ah, promising indeed.")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)
    conv.start_session("a red scarf", "en")

    greeting = conv.send_seed()

    assert greeting == "Ah, promising indeed."
    assert conv.turns == 2
    assert conv.messages[1] == {"role": "assistant", "content": "Ah, promising indeed."}


def test_send_appends_user_then_assistant_with_lang_tag():
    client = _FakeClient([_FakeResponse("Greetings."), _FakeResponse("Difficult... very difficult.")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)
    conv.start_session(None, "en")
    conv.send_seed()

    reply = conv.send("sort me please", "en")

    assert reply == "Difficult... very difficult."
    assert conv.messages[-2] == {"role": "user", "content": "sort me please\n[lang: en]"}
    assert conv.messages[-1] == {"role": "assistant", "content": "Difficult... very difficult."}
    assert conv.turns == 4


def test_call_uses_expected_request_shape():
    client = _FakeClient([_FakeResponse("hello")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=250)
    conv.start_session(None, "en")
    conv.send_seed()

    call_kwargs = client.messages.calls[0]
    assert call_kwargs["model"] == "claude-opus-5"
    assert call_kwargs["max_tokens"] == 250
    assert call_kwargs["output_config"] == {"effort": "low"}
    assert call_kwargs["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["system"] == SYSTEM_PROMPT
    assert "thinking" not in call_kwargs
    assert call_kwargs["messages"] is conv.messages or call_kwargs["messages"] == conv.messages[:1]


def test_reset_clears_messages():
    client = _FakeClient([_FakeResponse("hi")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)
    conv.start_session("something", "en")
    conv.send_seed()
    assert conv.turns > 0

    conv.reset()

    assert conv.messages == []
    assert conv.turns == 0


def test_max_tokens_stop_reason_still_returns_truncated_text():
    client = _FakeClient([_FakeResponse("truncated mid-sen", stop_reason="max_tokens")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=5)
    conv.start_session(None, "en")

    text = conv.send_seed()

    assert text == "truncated mid-sen"
    assert conv.turns == 2


def test_refusal_stop_reason_falls_back():
    client = _FakeClient([_FakeResponse("", stop_reason="refusal")])
    conv = ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)
    conv.start_session(None, "es")

    text = conv.send_seed()

    assert text == FALLBACK_LINES["es"]
    # Refusal should not be appended as a real assistant turn.
    assert conv.turns == 1
