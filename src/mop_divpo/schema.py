from typing import Any

from pydantic import BaseModel, Field, field_validator

from mop_divpo.personas import PersonaId, validate_persona


class SFTRecord(BaseModel):
    id: str
    persona: PersonaId
    source: str
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)


class CandidateRecord(BaseModel):
    id: str
    persona: PersonaId
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    quality_score: float = 0.0
    rarity_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)


class DivPOPairRecord(BaseModel):
    id: str
    persona: PersonaId
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)

    @classmethod
    def from_candidates(
        cls,
        id: str,
        chosen: CandidateRecord,
        rejected: CandidateRecord,
    ) -> "DivPOPairRecord":
        if chosen.persona != rejected.persona:
            raise ValueError("chosen and rejected candidates must use the same persona")
        if chosen.prompt != rejected.prompt:
            raise ValueError("chosen and rejected candidates must use the same prompt")
        return cls(
            id=id,
            persona=chosen.persona,
            prompt=chosen.prompt,
            chosen=chosen.response,
            rejected=rejected.response,
            metadata={"chosen_id": chosen.id, "rejected_id": rejected.id},
        )
