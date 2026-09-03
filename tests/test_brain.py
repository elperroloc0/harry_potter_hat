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
    NOT_SEATED_NOTE,
    NO_SIGHT_RESULT,
    SEATED_NOTE,
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


def test_tools_are_parameterless():
    for tool in TOOLS:
        assert tool["description"].strip()
        assert tool["input_schema"] == {"type": "object", "properties": {}, "required": []}


def test_fallback_lines_cover_both_languages():
    assert set(FALLBACK_LINES.keys()) >= {"es", "en"}
    for lang in ("es", "en"):
        assert FALLBACK_LINES[lang].strip()


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


# --------------------------------------------------------------------------
# The hat as a conversation, with sorting as one function within it
# --------------------------------------------------------------------------


def test_tools_are_conversation_shaped():
    names = [t["name"] for t in TOOLS]
    assert names == ["take_photo", "sort_visitor"]
    # end_session is gone on purpose: the hat never ends anything, it just
    # falls quiet and keeps listening.
    assert "end_session" not in names
    assert "end_session" not in SYSTEM_PROMPT


def test_prompt_forbids_stock_farewells_and_silence_filler():
    # The behaviour these replace (STILL_THERE / PARTING on a timer) is what
    # made the prop talk to an empty room.
    assert "fall quiet" in SYSTEM_PROMPT
    assert "sort_visitor" in SYSTEM_PROMPT


def test_sorting_notes_are_instructions_not_speech():
    for note in (SEATED_NOTE, NOT_SEATED_NOTE, NO_SIGHT_RESULT):
        assert note.strip()
        assert "error" not in note.lower()
    assert "sat down" in SEATED_NOTE
    assert "carry on talking" in NOT_SEATED_NOTE


def test_conversation_starts_from_what_was_heard():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("And who are you, then?")))

    turn = conv.send("hello hat", "en")

    # No seeded opening: the first thing in history is the visitor speaking.
    assert conv.messages[0] == {"role": "user", "content": "hello hat\n[lang: en]"}
    assert turn.beats == ["And who are you, then?"]


def test_sort_visitor_round_trip_keeps_tool_blocks():
    tool = _FakeToolUseBlock("sort_visitor", id="toolu_sort")
    conv, _ = _conv(
        _FakeResponse(_FakeTextBlock("Sit, then."), tool, stop_reason="tool_use"),
        _FakeResponse(_FakeTextBlock("Difficult..."), _FakeTextBlock("RAVENCLAW!")),
    )
    turn = conv.send("sort me", "en")
    assert turn.wants("sort_visitor")

    results = [{"type": "tool_result", "tool_use_id": tool.id, "content": SEATED_NOTE}]
    verdict = conv.submit_tool_results(results, "en")

    # The assistant turn keeps its tool_use block verbatim -- that is what
    # lets the result be paired back to it on the next request.
    assert tool in conv.messages[1]["content"]
    assert verdict.beats == ["Difficult...", "RAVENCLAW!"]
    assert conv.messages[-2] == {"role": "user", "content": results}


def test_sorting_note_adds_a_private_impression():
    conv, _ = _conv()
    dressed = conv.sorting_note(SEATED_NOTE)
    assert dressed.startswith(SEATED_NOTE)
    assert "private impression" in dressed


def test_sorting_notes_vary_across_sortings():
    conv, _ = _conv()
    assert len({conv.sorting_note(SEATED_NOTE) for _ in range(20)}) > 1


def test_reset_forgets_everything():
    conv, _ = _conv(_FakeResponse(_FakeTextBlock("hi")))
    conv.send("hello", "en")
    assert conv.turns > 0

    conv.reset()

    assert conv.messages == []


def test_api_failure_still_speaks_in_character():
    class _Boom:
        def create(self, **kw):
            raise RuntimeError("network gone")

    class _BoomClient:
        messages = _Boom()

    conv = ConversationManager(_BoomClient(), "claude-opus-5", SYSTEM_PROMPT, max_tokens=100)
    turn = conv.send("hello", "es")

    assert turn.beats == [FALLBACK_LINES["es"]]
    assert turn.tool_uses == []
