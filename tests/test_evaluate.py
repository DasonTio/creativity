from mop_divpo.eval.evaluate import evaluate_outputs


def test_evaluate_outputs_returns_diversity_metrics():
    outputs = [
        {"persona": "minimalist", "response": "Remove one premise and keep the sharpest image."},
        {"persona": "contrarian", "response": "Reject the obvious premise and test its inverse."},
        {"persona": "systems_thinker", "response": "Trace the feedback loop between incentives and habits."},
    ]
    metrics = evaluate_outputs(outputs)
    assert set(metrics) == {
        "count", "distinct_1", "distinct_2", "self_bleu",
        "sbert_pairwise_mean", "corpus_novelty_mean", "corpus_novelty_min",
        "quality_proxy_mean",
    }
    assert metrics["count"] == 3
    assert metrics["distinct_1"] > 0
