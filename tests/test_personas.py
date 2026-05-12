import pytest

from mop_divpo.personas import PERSONA_IDS, PersonaId, describe_persona, validate_persona


def test_persona_ids_match_proposal():
    assert PERSONA_IDS == (
        "contrarian",
        "cross_domain_analogist",
        "systems_thinker",
        "minimalist",
    )


def test_describe_persona_returns_research_grounded_description():
    description = describe_persona("systems_thinker")
    assert "causal" in description.lower()
    assert "feedback" in description.lower()


def test_validate_persona_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown persona"):
        validate_persona("optimist")


def test_persona_type_accepts_canonical_ids():
    value: PersonaId = "minimalist"
    assert validate_persona(value) == "minimalist"


def test_all_personas_have_structured_sections():
    for persona_id in PERSONA_IDS:
        description = describe_persona(persona_id)
        assert "METHOD" in description, f"{persona_id}: missing METHOD section"
        assert "FORMAT" in description, f"{persona_id}: missing FORMAT section"
