import pytest
from pydantic import ValidationError

from mop_divpo.coauthor.schema import CoAuthorRecord, IdeaCard, WritingBrief


def test_writing_brief_preserves_writer_context_and_goal():
    brief = WritingBrief(
        topic="AI in education",
        goal="Find a non-obvious thesis angle",
        audience="university instructors",
        constraints=["avoid productivity framing"],
        notes="I care about student agency.",
    )

    assert brief.topic == "AI in education"
    assert brief.goal.startswith("Find")
    assert brief.constraints == ["avoid productivity framing"]


def test_idea_card_requires_coauthor_fields_and_valid_persona():
    card = IdeaCard(
        persona="contrarian",
        operation="challenge_thesis",
        diagnosis="The draft assumes better learning means faster learning.",
        idea="Argue that AI helps only when it introduces useful friction.",
        rationale="This angle challenges the default efficiency story.",
        next_step="Compare tutoring, assessment, and reflection workflows.",
    )

    assert card.persona == "contrarian"
    assert "friction" in card.idea

    with pytest.raises(ValidationError):
        IdeaCard(
            persona="optimist",
            operation="challenge_thesis",
            diagnosis="x",
            idea="y",
            rationale="z",
            next_step="n",
        )


def test_coauthor_record_converts_to_chat_messages():
    record = CoAuthorRecord(
        id="seed-001",
        brief=WritingBrief(topic="remote work", goal="find thesis angles"),
        references=[{"dataset": "unit", "row_index": "0"}],
        card=IdeaCard(
            persona="minimalist",
            operation="distill_core",
            diagnosis="The topic has too many claims competing for attention.",
            idea="Make the essay about one lost ritual: the commute.",
            rationale="A single concrete ritual can carry the larger argument.",
            next_step="List what the commute used to do socially and cognitively.",
        ),
    )

    messages = record.to_messages()

    assert messages[0]["role"] == "system"
    assert "Minimalist" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "remote work" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert "Make the essay" in messages[2]["content"]
    assert record.to_training_row()["references"][0]["dataset"] == "unit"
