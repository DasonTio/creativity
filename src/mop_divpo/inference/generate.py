from mop_divpo.personas import describe_persona, validate_persona


def dry_run_generate(prompt: str, personas: list[str]) -> list[dict[str, str]]:
    outputs = []
    for persona_value in personas:
        persona = validate_persona(persona_value)
        outputs.append(
            {
                "persona": persona,
                "prompt": prompt,
                "response": f"[dry-run:{persona}] {describe_persona(persona)} Prompt: {prompt}",
            }
        )
    return outputs
