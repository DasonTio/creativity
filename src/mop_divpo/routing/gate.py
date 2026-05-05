from dataclasses import dataclass

from mop_divpo.personas import PERSONA_DESCRIPTIONS, PersonaId, validate_persona


@dataclass(frozen=True)
class PersonaRoute:
    persona: PersonaId
    score: float
    description: str


class KeywordPersonaGate:
    KEYWORDS: dict[PersonaId, tuple[str, ...]] = {
        "contrarian": ("challenge", "assumption", "opposite", "dissent", "minority"),
        "cross_domain_analogist": ("analogy", "borrow", "domain", "biology", "physics", "architecture"),
        "systems_thinker": ("system", "causal", "feedback", "loop", "long-term", "side effect"),
        "minimalist": ("constraint", "minimal", "remove", "simple", "essential", "limit"),
    }

    def rank(
        self,
        prompt: str,
        top_k: int,
        force_personas: list[str] | None = None,
    ) -> list[PersonaRoute]:
        if force_personas:
            personas = [validate_persona(persona) for persona in force_personas]
            return [
                PersonaRoute(persona=persona, score=1.0, description=PERSONA_DESCRIPTIONS[persona])
                for persona in personas[:top_k]
            ]
        lowered = prompt.lower()
        routes = []
        for persona, keywords in self.KEYWORDS.items():
            score = sum(1.0 for keyword in keywords if keyword in lowered)
            routes.append(PersonaRoute(persona, score, PERSONA_DESCRIPTIONS[persona]))
        routes.sort(key=lambda item: (-item.score, item.persona))
        return routes[:top_k]
