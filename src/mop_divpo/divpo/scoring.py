from mop_divpo.metrics.diversity import distinct_n


def lexical_rarity_scores(responses: list[str]) -> list[float]:
    scores: list[float] = []
    for index, response in enumerate(responses):
        others = responses[:index] + responses[index + 1 :]
        overlap_penalty = 0.0
        response_tokens = set(response.lower().split())
        if response_tokens and others:
            other_tokens = set(" ".join(others).lower().split())
            overlap_penalty = len(response_tokens & other_tokens) / len(response_tokens)
        scores.append(max(0.0, distinct_n([response], 1) - overlap_penalty))
    max_score = max(scores) if scores else 0.0
    return [score / max_score if max_score else 0.0 for score in scores]


def combined_divpo_score(
    quality: float,
    rarity: float,
    quality_weight: float,
    rarity_weight: float,
) -> float:
    return quality * quality_weight + rarity * rarity_weight
