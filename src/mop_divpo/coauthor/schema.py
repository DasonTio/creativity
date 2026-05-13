from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from mop_divpo.personas import PersonaId, describe_persona, validate_persona

PersonaOperation = Literal[
    "challenge_thesis",
    "transfer_analogy",
    "map_system",
    "distill_core",
    "synthesize_angles",
]


class WritingBrief(BaseModel):
    topic: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    audience: str = "general readers"
    constraints: list[str] = Field(default_factory=list)
    notes: str = ""

    def to_prompt(self) -> str:
        parts = [
            f"Topic: {self.topic}",
            f"Writer goal: {self.goal}",
            f"Audience: {self.audience}",
        ]
        if self.constraints:
            parts.append("Constraints: " + "; ".join(self.constraints))
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return "\n".join(parts)


class IdeaCard(BaseModel):
    persona: PersonaId
    operation: PersonaOperation
    diagnosis: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    next_step: str = Field(min_length=1)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)

    def to_text(self) -> str:
        return json.dumps(
            {
                "operation": self.operation,
                "diagnosis": self.diagnosis,
                "idea": self.idea,
                "rationale": self.rationale,
                "next_step": self.next_step,
            },
            ensure_ascii=False,
        )


class CoAuthorRecord(BaseModel):
    id: str
    brief: WritingBrief
    card: IdeaCard
    source: str = "curated_seed"
    references: list[dict[str, str]] = Field(default_factory=list)

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": describe_persona(self.card.persona)},
            {"role": "user", "content": self.brief.to_prompt()},
            {"role": "assistant", "content": self.card.to_text()},
        ]

    def to_training_row(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "persona": self.card.persona,
            "operation": self.card.operation,
            "brief": self.brief.model_dump(),
            "card": self.card.model_dump(),
            "references": self.references,
            "messages": self.to_messages(),
        }
