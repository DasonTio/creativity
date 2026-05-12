from __future__ import annotations

import re
from dataclasses import dataclass

from mop_divpo.personas import PersonaId


@dataclass
class StructuredOutput:
    idea: str
    reasoning: str
    unexpected_angle: str
    persona: PersonaId
    prompt: str
    parse_success: bool


def parse_structured_output(text: str, persona: PersonaId, prompt: str) -> StructuredOutput:
    """Parse LLM output into a StructuredOutput dataclass.

    Expects text in the format produced by the persona system prompts::

        Idea: <one sentence>
        Reasoning: <2-3 sentences>
        Unexpected angle: <1-2 sentences>

    Labels are matched case-sensitively.  If any section is missing or empty
    after stripping, ``parse_success`` is set to ``False`` and empty strings
    are used for the missing fields.  Raw text is NOT stored here; the caller
    is responsible for preserving it separately.
    """
    # Each section is captured up to the next known label or end of string.
    # The alternation includes all downstream labels so the lazy quantifier
    # stops as soon as any following label appears.
    idea_match = re.search(
        r"Idea:(.*?)(?=Reasoning:|Unexpected angle:|\Z)", text, re.DOTALL
    )
    reasoning_match = re.search(
        r"Reasoning:(.*?)(?=Unexpected angle:|\Z)", text, re.DOTALL
    )
    unexpected_angle_match = re.search(r"Unexpected angle:(.*)\Z", text, re.DOTALL)

    idea = idea_match.group(1).strip() if idea_match else ""
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    unexpected_angle = unexpected_angle_match.group(1).strip() if unexpected_angle_match else ""

    parse_success = bool(idea and reasoning and unexpected_angle)

    return StructuredOutput(
        idea=idea,
        reasoning=reasoning,
        unexpected_angle=unexpected_angle,
        persona=persona,
        prompt=prompt,
        parse_success=parse_success,
    )
