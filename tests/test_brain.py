"""Pure-logic tests for the brain/orchestrator subsystem — no network calls.

Covers: persona templates and the tool definitions, beat splitting, and
ConversationManager's message-building logic through a full tool-use round
trip against a fake Anthropic client (so no live API key is required to
validate the request shape sent to Claude).
"""

from __future__ import annotations

import pytest

from hat.brain.client import ConversationManager, HatTurn
from hat.brain.persona import (
    FALLBACK_LINES,
    NO_SIGHT_RESULT,
    OPENING_MANNERS,
    PARTING,
    RITUAL_OPENING_SEED,
    RITUAL_OPENING_SEED_SEATED,
    SEATED_NOTE,
    STILL_THERE,
    SYSTEM_PROMPT,
    TOOLS,
    split_beats,
)


# --------------------------------------------------------------------------
# persona.py — prompt, templates, tools
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


def test_system_prompt_documents_both_tools_and_the_seated_gate():
    assert "take_photo" in SYSTEM_PROMPT
    assert "end_session" in SYSTEM_PROMPT
    # The gate the whole ritual hangs on: no house before the castle says
    # they have sat down.
    assert "sat down" in SYSTEM_PROMPT


def test_tools_are_parameterless_and_named():
    assert [t["name"] for t in TOOLS] == ["take_photo", "end_session"]
    for tool in TOOLS:
        assert tool["description"].strip()
        assert tool["input_schema"] == {"type": "object", "properties": {}, "required": []}


def test_opening_seeds_format_and_differ():
    standing = RITUAL_OPENING_SEED.format(lang="en")
    seated = RITUAL_OPENING_SEED_SEATED.format(lang="en")
    assert "[lang: en]" in standing
    assert "[lang: en]" in seated
    assert "Castle's note" in standing
    assert standing != seated
    # The seated opening must tell the hat not to try looking at them.
    assert "do not ask to look" in seated


def test_opening_manners_are_usable_hints():
    assert len(OPENING_MANNERS) >= 8
    assert len(set(OPENING_MANNERS)) == len(OPENING_MANNERS)
    for manner in OPENING_MANNERS:
        # Folded into the seed as "Open {manner}", so they must read as a
        # manner to improvise from, not a line to recite.
        assert manner.strip() == manner and manner.strip()
        assert not manner.endswith(".")


def test_seated_note_formats():
    note = SEATED_NOTE.format(lang="es")
    assert "[lang: es]" in note
    assert "sat down" in note


@pytest.mark.parametrize("mapping", [FALLBACK_LINES, STILL_THERE, PARTING])
def test_bilingual_maps_have_both_languages(mapping):
    assert set(mapping.keys()) >= {"es", "en"}
    for lang in ("es", "en"):
        assert isinstance(mapping[lang], str)
        assert mapping[lang].strip() != ""


def test_fallback_lines_differ_by_language():
    assert FALLBACK_LINES["es"] != FALLBACK_LINES["en"]


def test_no_sight_result_is_in_character_not_an_error():
    assert NO_SIGHT_RESULT.strip()
    assert "error" not in NO_SIGHT_RESULT.lower()


# --------------------------------------------------------------------------
# split_beats
# --------------------------------------------------------------------------


def test_split_beats_splits_lines():
    text = "Plenty of courage, I see...\nDifficult... very difficult...\nGRYFFINDOR!"
    assert split_beats(text) == [
        "Plenty of courage, I see...",
        "Difficult... very difficult...",
        "GRYFFINDOR!",
    ]


def test_split_beats_drops_blank_lines():
    assert split_beats("Hmm...\n\nSLYTHERIN!\n") == ["Hmm...", "SLYTHERIN!"]


def test_split_beats_single_line_and_empty():
    assert split_beats("RAVENCLAW!") == ["RAVENCLAW!"]
    assert split_beats("") == []
    assert split_beats("   ") == []


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name, id="toolu_1", input=None):
        self.name = name
        self.id = id
        self.input = input or {}


class _FakeResponse:
    def __init__(self, *blocks, stop_reason="end_turn"):
        self.content = list(blocks)
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


def _conv(*responses, max_tokens=100):
    client = _FakeClient(list(responses) or [_FakeResponse(_FakeTextBlock("Hmm..."))])
    return ConversationManager(client, "claude-opus-5", SYSTEM_PROMPT, max_tokens=max_tokens), client


# --------------------------------------------------------------------------
# HatTurn
# --------------------------------------------------------------------------


def test_hat_turn_wants():
    turn = HatTurn(beats=["Hmm..."], tool_uses=[_FakeToolUseBlock("take_photo")])
    assert turn.wants("take_photo") is True
    assert turn.wants("end_session") is False
    assert HatTurn().wants("take_photo") is False


# --------------------------------------------------------------------------
# ConversationManager
# --------------------------------------------------------------------------


def test_start_ritual_seeds_one_user_turn_with_flavor_hint():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("And who might you be?")))

    turn = conv.start_ritual("en")

    assert turn.beats == ["And who might you be?"]
    assert conv.messages[0]["role"] == "user"
    assert "[lang: en]" in conv.messages[0]["content"]
    assert "private impression stirs in you" in conv.messages[0]["content"]


def test_start_ritual_folds_in_an_opening_manner():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("Your name?")))

    conv.start_ritual("en")

    seed = conv.messages[0]["content"]
    assert "Open " in seed
    assert any(manner in seed for manner in OPENING_MANNERS)


def test_opening_manner_varies_across_visits():
    # Live runs converged on one greeting for every child until the manner
    # was drawn per visit; children queue up and hear each other.
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("Your name?")))

    seeds = set()
    for _ in range(20):
        conv.start_ritual("en")
        seeds.add(conv.messages[0]["content"])

    assert len(seeds) > 1


def test_start_ritual_seated_uses_the_seated_opening():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("Settled already, I see...")))

    conv.start_ritual("en", seated=True)

    assert "do not ask to look" in conv.messages[0]["content"]


def test_start_ritual_hints_vary_across_calls():
    # Not guaranteed on any single pair, but with 14 hints, 20 draws should
    # not all land on the same one.
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("Hmm...")))

    seeds = {conv.start_ritual("en") and conv.messages[0]["content"] for _ in range(20)}

    assert len(seeds) > 1


def test_assistant_turn_is_stored_as_content_blocks_not_text():
    # This is what keeps tool_use (and thinking) blocks alive for the next
    # request; storing extracted text would silently break the tool loop.
    text_block = _FakeTextBlock("Let me have a proper look at you...")
    tool_block = _FakeToolUseBlock("take_photo")
    conv, _ = _conv(_FakeResponse(text_block, tool_block, stop_reason="tool_use"))

    turn = conv.start_ritual("en")

    assert conv.messages[-1]["role"] == "assistant"
    assert conv.messages[-1]["content"] == [text_block, tool_block]
    assert turn.beats == ["Let me have a proper look at you..."]
    assert turn.tool_uses == [tool_block]


def test_full_take_photo_round_trip():
    conv, client = _conv(
        _FakeResponse(_FakeTextBlock("Let me look at you..."), _FakeToolUseBlock("take_photo"), stop_reason="tool_use"),
        _FakeResponse(_FakeTextBlock("Ah... a scarlet jumper. Promising.")),
    )
    turn = conv.start_ritual("en")

    results = [{"type": "tool_result", "tool_use_id": turn.tool_uses[0].id, "content": "a boy in a red jumper"}]
    followup = conv.submit_tool_results(results, "en")

    assert followup.beats == ["Ah... a scarlet jumper. Promising."]
    assert conv.messages[-2] == {"role": "user", "content": results}
    assert conv.messages[-1]["role"] == "assistant"
    # seed, assistant(tool_use), tool_result, assistant
    assert conv.turns == 4


def test_send_appends_user_then_assistant_with_lang_tag():
    conv, _ = _conv(
        _FakeResponse(_FakeTextBlock("Greetings.")),
        _FakeResponse(_FakeTextBlock("Difficult... very difficult.")),
    )
    conv.start_ritual("en")

    turn = conv.send("my name is Sam", "en")

    assert turn.beats == ["Difficult... very difficult."]
    assert conv.messages[-2] == {"role": "user", "content": "my name is Sam\n[lang: en]"}
    assert conv.turns == 4


def test_note_seated_injects_the_castle_note():
    conv, _ = _conv(
        _FakeResponse(_FakeTextBlock("Hmm...")),
        _FakeResponse(_FakeTextBlock("GRYFFINDOR!")),
    )
    conv.start_ritual("es")

    turn = conv.note_seated("es")

    assert turn.beats == ["GRYFFINDOR!"]
    assert conv.messages[-2]["role"] == "user"
    assert "sat down" in conv.messages[-2]["content"]
    assert "[lang: es]" in conv.messages[-2]["content"]


def test_end_session_tool_is_surfaced_on_the_turn():
    conv, _ = _conv(
        _FakeResponse(
            _FakeTextBlock("Off you go. Who is next?"),
            _FakeToolUseBlock("end_session", id="toolu_end"),
            stop_reason="tool_use",
        )
    )

    turn = conv.start_ritual("en")

    assert turn.wants("end_session")
    assert turn.beats == ["Off you go. Who is next?"]


def test_tool_only_turn_has_no_beats():
    conv, _ = _conv(_FakeResponse(_FakeToolUseBlock("take_photo"), stop_reason="tool_use"))

    turn = conv.start_ritual("en")

    assert turn.beats == []
    assert turn.wants("take_photo")


def test_call_uses_expected_request_shape():
    conv, client = _conv(_FakeResponse(_FakeTextBlock("hello")), max_tokens=250)
    conv.start_ritual("en")

    call_kwargs = client.messages.calls[0]
    assert call_kwargs["model"] == "claude-opus-5"
    assert call_kwargs["max_tokens"] == 250
    assert call_kwargs["output_config"] == {"effort": "low"}
    assert call_kwargs["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["system"] == SYSTEM_PROMPT
    assert call_kwargs["tools"] == TOOLS
    assert "thinking" not in call_kwargs


def test_reset_clears_messages():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("hi")))
    conv.start_ritual("en")
    assert conv.turns > 0

    conv.reset()

    assert conv.messages == []
    assert conv.turns == 0


def test_max_tokens_stop_reason_still_returns_truncated_text():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("truncated mid-sen"), stop_reason="max_tokens"))

    turn = conv.start_ritual("en")

    assert turn.beats == ["truncated mid-sen"]
    assert conv.turns == 2


def test_refusal_stop_reason_falls_back_and_is_not_kept_in_history():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock(""), stop_reason="refusal"))

    turn = conv.start_ritual("es")

    assert turn.beats == [FALLBACK_LINES["es"]]
    assert turn.tool_uses == []
    # A refusal left in history would poison every later request this visit.
    assert conv.turns == 1


def test_empty_response_falls_back():
    conv, _ = _conv(_FakeResponse())

    turn = conv.start_ritual("en")

    assert turn.beats == [FALLBACK_LINES["en"]]


def test_api_error_degrades_to_fallback_line():
    class _BoomMessages:
        calls: list = []

        def create(self, **kwargs):
            raise RuntimeError("network gone")

    class _BoomClient:
        messages = _BoomMessages()

    conv = ConversationManager(_BoomClient(), "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)

    turn = conv.start_ritual("en")

    assert turn.beats == [FALLBACK_LINES["en"]]
    assert turn.tool_uses == []
