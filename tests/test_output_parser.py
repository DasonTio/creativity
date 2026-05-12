from mop_divpo.inference.output_parser import StructuredOutput, parse_structured_output

_PROMPT = "How might we redesign public transport?"
_PERSONA = "contrarian"

_WELL_FORMED = (
    "Idea: Abolish fixed routes and let demand-responsive micro-buses self-organise.\n"
    "Reasoning: The assumption that routes must be predetermined is a planning artifact, "
    "not an operational necessity. Negating it opens design space around real-time "
    "optimisation already proven in ride-hail networks. The shift is under-explored "
    "because regulators conflate predictability with fixed schedules.\n"
    "Unexpected angle: Dial-a-ride pilots in rural Finland achieved 40 % higher "
    "ridership than fixed-route equivalents at equivalent cost."
)


def test_parse_well_formed_output():
    result = parse_structured_output(_WELL_FORMED, _PERSONA, _PROMPT)

    assert isinstance(result, StructuredOutput)
    assert result.parse_success is True
    assert result.idea == "Abolish fixed routes and let demand-responsive micro-buses self-organise."
    assert result.reasoning.startswith("The assumption that routes must be predetermined")
    assert result.unexpected_angle.startswith("Dial-a-ride pilots")
    assert result.persona == _PERSONA
    assert result.prompt == _PROMPT


def test_parse_missing_section_returns_false():
    text_without_reasoning = (
        "Idea: Abolish fixed routes.\n"
        "Unexpected angle: Dial-a-ride pilots showed higher ridership."
    )
    result = parse_structured_output(text_without_reasoning, _PERSONA, _PROMPT)

    assert result.parse_success is False
    assert result.reasoning == ""
    # The other sections that could be parsed should still be present.
    assert result.idea == "Abolish fixed routes."
    assert result.unexpected_angle == "Dial-a-ride pilots showed higher ridership."


def test_parse_empty_section_returns_false():
    text_empty_reasoning = (
        "Idea: Abolish fixed routes.\n"
        "Reasoning:   \n"
        "Unexpected angle: Dial-a-ride pilots showed higher ridership."
    )
    result = parse_structured_output(text_empty_reasoning, _PERSONA, _PROMPT)

    assert result.parse_success is False
    assert result.reasoning == ""


def test_parse_preserves_multiline_reasoning():
    multiline_text = (
        "Idea: Use ferries as the primary urban transit layer.\n"
        "Reasoning: Waterways pass through the city centre but are treated as scenic "
        "amenities rather than arteries. Inverting this assumption reveals unused "
        "capacity equal to several subway lines. Historical precedents in Amsterdam and "
        "Bangkok demonstrate commercial viability without heavy infrastructure investment.\n"
        "Unexpected angle: Ferries produce lower per-passenger emissions than any "
        "surface road mode while being immune to road congestion."
    )
    result = parse_structured_output(multiline_text, "systems_thinker", _PROMPT)

    assert result.parse_success is True
    # All three sentences of the reasoning must be captured.
    assert "Waterways pass through the city centre" in result.reasoning
    assert "Inverting this assumption" in result.reasoning
    assert "Historical precedents" in result.reasoning
