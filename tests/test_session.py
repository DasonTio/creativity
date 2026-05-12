"""Tests for mop_divpo.inference.session."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mop_divpo.inference.session import PersonaSession, dry_run_generate_turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_retriever(docs=None):
    """Build a mock CorpusRetriever that returns stubbed RetrievalResult objects."""
    if docs is None:
        docs = [
            ("Doc A title", "Doc A snippet"),
            ("Doc B title", "Doc B snippet"),
            ("Doc C title", "Doc C snippet"),
        ]

    results = []
    for title, snippet in docs:
        doc = MagicMock()
        doc.title = title
        doc.snippet = snippet
        result = MagicMock()
        result.document = doc
        result.score = 0.9
        results.append(result)

    retriever = MagicMock()
    retriever.retrieve.return_value = results
    return retriever


# ---------------------------------------------------------------------------
# PersonaSession dataclass tests
# ---------------------------------------------------------------------------

def test_session_add_user_and_assistant():
    """Verify that add_user and add_assistant append correctly formatted dicts."""
    session = PersonaSession(persona="minimalist")

    session.add_user("What is a good essay topic?")
    session.add_assistant("Consider writing about the power of silence.")

    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "What is a good essay topic?"}
    assert session.history[1] == {"role": "assistant", "content": "Consider writing about the power of silence."}


def test_session_to_messages_returns_history():
    """Verify to_messages() returns the full history list."""
    session = PersonaSession(persona="contrarian")
    session.add_user("Give me an unconventional idea.")
    session.add_assistant("Challenge the assumption that more is better.")
    session.add_user("Can you elaborate?")

    messages = session.to_messages()

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Can you elaborate?"


def test_session_history_starts_empty():
    """Newly created PersonaSession must have an empty history."""
    session = PersonaSession(persona="systems_thinker")
    assert session.history == []


def test_session_to_messages_is_a_copy():
    """to_messages() should return a list that is independent from internal history."""
    session = PersonaSession(persona="cross_domain_analogist")
    session.add_user("Hello")
    messages = session.to_messages()

    # Mutating the returned list must not affect session.history
    messages.append({"role": "user", "content": "Extra"})
    assert len(session.history) == 1


# ---------------------------------------------------------------------------
# dry_run_generate_turn tests
# ---------------------------------------------------------------------------

def test_dry_run_generate_turn_returns_response_and_updated_session():
    """dry_run_generate_turn returns a non-empty string and leaves history with 2 entries."""
    session = PersonaSession(persona="minimalist")

    response, updated_session = dry_run_generate_turn(
        session=session,
        user_message="Give me a minimal essay concept.",
    )

    assert isinstance(response, str)
    assert len(response) > 0
    # History must have exactly one user turn + one assistant turn
    assert len(updated_session.history) == 2
    assert updated_session.history[0]["role"] == "user"
    assert updated_session.history[1]["role"] == "assistant"
    # The returned session is the same object (mutated in-place)
    assert updated_session is session


def test_dry_run_first_turn_only_injects_prior_art():
    """Prior art is injected on the first turn but NOT on subsequent turns."""
    session = PersonaSession(persona="contrarian")
    retriever = _make_mock_retriever()

    # First turn — retriever.retrieve should be called once
    response1, session = dry_run_generate_turn(
        session=session,
        user_message="Invent an essay about urban sprawl.",
        retriever=retriever,
    )

    assert retriever.retrieve.call_count == 1
    # The stub response for first turn should mark prior-art injection
    assert "[prior-art-injected]" in response1

    # Second turn — retriever.retrieve must NOT be called again
    response2, session = dry_run_generate_turn(
        session=session,
        user_message="Can you go deeper?",
        retriever=retriever,
    )

    assert retriever.retrieve.call_count == 1  # still 1 — no second call
    assert "[prior-art-injected]" not in response2


def test_dry_run_no_retriever_skips_prior_art():
    """Without a retriever, dry_run_generate_turn works fine and response has no prior-art marker."""
    session = PersonaSession(persona="systems_thinker")

    response, session = dry_run_generate_turn(
        session=session,
        user_message="Think about feedback loops in healthcare.",
    )

    assert isinstance(response, str)
    assert "[prior-art-injected]" not in response
    assert len(session.history) == 2


def test_dry_run_multiple_turns_accumulate_history():
    """Three turns should leave 6 entries (3 user + 3 assistant)."""
    session = PersonaSession(persona="cross_domain_analogist")

    for i in range(3):
        _, session = dry_run_generate_turn(
            session=session,
            user_message=f"Turn {i + 1} message.",
        )

    assert len(session.history) == 6
    roles = [msg["role"] for msg in session.history]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
