from typing import Any

from mop_divpo.metrics.diversity import distinct_n, self_bleu


def evaluate_outputs(outputs: list[dict[str, Any]]) -> dict[str, float]:
    responses = [str(output.get("response", "")) for output in outputs]
    return {
        "count": float(len(responses)),
        "distinct_1": distinct_n(responses, 1),
        "distinct_2": distinct_n(responses, 2),
        "self_bleu": self_bleu(responses),
    }
