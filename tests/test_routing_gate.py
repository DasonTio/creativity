from mop_divpo.routing.gate import KeywordPersonaGate


def test_gate_routes_challenge_prompt_to_contrarian():
    gate = KeywordPersonaGate()
    ranked = gate.rank("Challenge the common assumption that productivity means speed.", top_k=2)
    assert ranked[0].persona == "contrarian"


def test_gate_routes_feedback_prompt_to_systems_thinker():
    gate = KeywordPersonaGate()
    ranked = gate.rank("Find feedback loops and long-term side effects in school grading.", top_k=2)
    assert ranked[0].persona == "systems_thinker"


def test_gate_can_force_personas():
    gate = KeywordPersonaGate()
    ranked = gate.rank("Any prompt", top_k=2, force_personas=["minimalist", "contrarian"])
    assert [item.persona for item in ranked] == ["minimalist", "contrarian"]
