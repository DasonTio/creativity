from typing import Literal

PersonaId = Literal[
    "contrarian",
    "cross_domain_analogist",
    "systems_thinker",
    "minimalist",
]

PERSONA_IDS: tuple[PersonaId, ...] = (
    "contrarian",
    "cross_domain_analogist",
    "systems_thinker",
    "minimalist",
)

PERSONA_DESCRIPTIONS: dict[PersonaId, str] = {
    "contrarian": "Challenges assumptions, surfaces minority dissent, and reframes common viewpoints.",
    "cross_domain_analogist": "Builds analogies across distant fields to transfer useful structures.",
    "systems_thinker": "Explains causal relationships, feedback loops, trade-offs, and long-term effects.",
    "minimalist": "Uses constraints, compression, and removal of non-essential parts to sharpen ideas.",
}


def validate_persona(value: str) -> PersonaId:
    if value not in PERSONA_IDS:
        allowed = ", ".join(PERSONA_IDS)
        raise ValueError(f"Unknown persona '{value}'. Expected one of: {allowed}")
    return value  # type: ignore[return-value]


def describe_persona(persona: PersonaId) -> str:
    return PERSONA_DESCRIPTIONS[validate_persona(persona)]
