from mop_divpo.divpo.scoring import combined_divpo_score
from mop_divpo.schema import CandidateRecord, DivPOPairRecord


def select_divpo_pair(
    candidates: list[CandidateRecord],
    min_quality: float,
    quality_weight: float,
    rarity_weight: float,
) -> DivPOPairRecord:
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required to build a DivPO pair")
    prompts = {candidate.prompt for candidate in candidates}
    personas = {candidate.persona for candidate in candidates}
    if len(prompts) != 1 or len(personas) != 1:
        raise ValueError("Candidates must share the same prompt and persona")

    eligible = [candidate for candidate in candidates if candidate.quality_score >= min_quality]
    if not eligible:
        raise ValueError("No candidates meet the minimum quality threshold")

    chosen = max(
        eligible,
        key=lambda candidate: combined_divpo_score(
            candidate.quality_score,
            candidate.rarity_score,
            quality_weight,
            rarity_weight,
        ),
    )
    rejected = min(
        [candidate for candidate in candidates if candidate.id != chosen.id],
        key=lambda candidate: (
            candidate.quality_score >= min_quality,
            combined_divpo_score(
                candidate.quality_score,
                candidate.rarity_score,
                quality_weight,
                rarity_weight,
            ),
        ),
    )
    return DivPOPairRecord.from_candidates(
        id=f"divpo-{chosen.persona}-{chosen.id}-{rejected.id}",
        chosen=chosen,
        rejected=rejected,
    )
