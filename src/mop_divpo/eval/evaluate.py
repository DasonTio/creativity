from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mop_divpo.metrics.diversity import distinct_n, self_bleu
from mop_divpo.metrics.semantic import average_pairwise_cosine

if TYPE_CHECKING:
    from mop_divpo.retrieval.corpus import CorpusRetriever


def _extract_text(output: dict[str, Any]) -> str:
    """Return the idea text for diversity metrics.

    When the output contains a 'structured' dict and parse_success is not False,
    use structured["idea"]. Otherwise fall back to the raw "response" string.
    """
    structured = output.get("structured")
    if isinstance(structured, dict) and output.get("parse_success", True):
        idea = structured.get("idea")
        if idea is not None:
            return str(idea)
    return str(output.get("response", ""))


def evaluate_outputs(
    outputs: list[dict[str, Any]],
    retriever: "CorpusRetriever | None" = None,
) -> dict[str, float]:
    texts = [_extract_text(o) for o in outputs]

    # --- lexical diversity ---
    result: dict[str, float] = {
        "count": float(len(texts)),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "self_bleu": self_bleu(texts),
    }

    # --- SBERT pairwise diversity ---
    if retriever is not None:
        embeddings: np.ndarray = retriever._model.encode(
            texts, convert_to_numpy=True
        )
        result["sbert_pairwise_mean"] = average_pairwise_cosine(embeddings)
    else:
        result["sbert_pairwise_mean"] = 0.0

    # --- corpus novelty ---
    if retriever is not None:
        novelty_scores = [retriever.corpus_novelty_score(t) for t in texts]
        result["corpus_novelty_mean"] = float(np.mean(novelty_scores)) if novelty_scores else 0.0
        result["corpus_novelty_min"] = float(np.min(novelty_scores)) if novelty_scores else 0.0
    else:
        result["corpus_novelty_mean"] = 0.0
        result["corpus_novelty_min"] = 0.0

    # --- quality proxy (prompt–idea on-topicness) ---
    if retriever is not None:
        prompts = [str(o.get("prompt", "")) for o in outputs]
        prompt_embeddings: np.ndarray = retriever._model.encode(
            prompts, convert_to_numpy=True
        )
        idea_embeddings: np.ndarray = retriever._model.encode(
            texts, convert_to_numpy=True
        )
        # cosine similarity per pair then average
        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        similarities = [
            _cosine(prompt_embeddings[i], idea_embeddings[i])
            for i in range(len(texts))
        ]
        result["quality_proxy_mean"] = float(np.mean(similarities)) if similarities else 0.0
    else:
        result["quality_proxy_mean"] = 0.0

    return result
