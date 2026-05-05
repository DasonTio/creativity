from mop_divpo.divpo.pairs import select_divpo_pair
from mop_divpo.divpo.scoring import combined_divpo_score, lexical_rarity_scores
from mop_divpo.schema import CandidateRecord


def test_lexical_rarity_rewards_unusual_candidate():
    responses = [
        "write a list of ideas",
        "write a list of ideas",
        "invert the premise and make the constraint the protagonist",
    ]
    scores = lexical_rarity_scores(responses)
    assert scores[2] > scores[0]


def test_combined_score_uses_quality_and_rarity_weights():
    score = combined_divpo_score(quality=0.5, rarity=1.0, quality_weight=0.25, rarity_weight=0.75)
    assert score == 0.875


def test_select_divpo_pair_prefers_rare_high_quality_candidate():
    candidates = [
        CandidateRecord(
            id="common-good",
            persona="contrarian",
            prompt="How can cities improve parks?",
            response="Add more parks and benches.",
            quality_score=0.8,
            rarity_score=0.1,
        ),
        CandidateRecord(
            id="rare-good",
            persona="contrarian",
            prompt="How can cities improve parks?",
            response="Question whether parks should be centralized; distribute tiny repairable green rooms instead.",
            quality_score=0.75,
            rarity_score=0.95,
        ),
        CandidateRecord(
            id="rare-bad",
            persona="contrarian",
            prompt="How can cities improve parks?",
            response="Ban all trees.",
            quality_score=0.1,
            rarity_score=1.0,
        ),
    ]
    pair = select_divpo_pair(candidates, min_quality=0.35, quality_weight=0.4, rarity_weight=0.6)
    assert pair.chosen == candidates[1].response
    assert pair.rejected == candidates[2].response
