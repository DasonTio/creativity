from __future__ import annotations

from mop_divpo.coauthor.schema import IdeaCard, WritingBrief
from mop_divpo.personas import validate_persona


def build_idea_cards(
    topic: str,
    goal: str,
    personas: list[str] | None = None,
    audience: str = "general readers",
    constraints: list[str] | None = None,
    notes: str = "",
) -> list[IdeaCard]:
    brief = WritingBrief(
        topic=topic,
        goal=goal,
        audience=audience,
        constraints=constraints or [],
        notes=notes,
    )
    selected = personas or [
        "contrarian",
        "cross_domain_analogist",
        "systems_thinker",
        "minimalist",
    ]
    return [_card_for_persona(validate_persona(persona), brief) for persona in selected]


def _card_for_persona(persona: str, brief: WritingBrief) -> IdeaCard:
    if persona == "contrarian":
        return IdeaCard(
            persona="contrarian",
            operation="challenge_thesis",
            diagnosis=f"The obvious version of '{brief.topic}' likely accepts the writer's starting frame too easily.",
            idea=f"Argue the inverse: {brief.topic} becomes interesting when the common benefit also creates a hidden cost.",
            rationale="A strong pre-writing partner should expose the assumption before producing more angles.",
            next_step="Write the current thesis in one sentence, then negate its main assumption.",
        )
    if persona == "cross_domain_analogist":
        return IdeaCard(
            persona="cross_domain_analogist",
            operation="transfer_analogy",
            diagnosis=f"The writer needs a structure for '{brief.topic}', not another surface-level example.",
            idea=f"Borrow from ecology: treat {brief.topic} as a habitat where small changes in incentives alter which behaviors can survive.",
            rationale="A distant domain gives the writer a reusable map instead of a slogan.",
            next_step="List three mechanisms from the analogy and translate each into the essay topic.",
        )
    if persona == "systems_thinker":
        return IdeaCard(
            persona="systems_thinker",
            operation="map_system",
            diagnosis=f"The topic '{brief.topic}' needs causes, constraints, and feedback loops before it needs prose.",
            idea=f"Frame the essay around the loop that keeps {brief.topic} stable even when people claim they want change.",
            rationale="A system map helps the writer find leverage points and avoid one-cause explanations.",
            next_step="Name one reinforcing loop, one balancing loop, and one intervention point.",
        )
    return IdeaCard(
        persona="minimalist",
        operation="distill_core",
        diagnosis=f"The topic '{brief.topic}' is probably carrying too many claims at once.",
        idea=f"Reduce the essay to one concrete scene where {brief.topic} becomes visible without explanation.",
        rationale="A narrow scene can preserve writer ownership while making the idea specific.",
        next_step="Remove every abstract noun from the thesis and replace one with an image.",
    )
