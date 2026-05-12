"""Tests for mop_divpo.retrieval.corpus — constructed with hand-crafted embeddings
to avoid hitting SentenceTransformer during unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mop_divpo.retrieval.corpus import CorpusDocument, CorpusRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(title: str, source: str = "test") -> CorpusDocument:
    return CorpusDocument(title=title, snippet=title[:80], source=source)


def _make_retriever(
    docs: list[CorpusDocument],
    embeddings: np.ndarray,
) -> CorpusRetriever:
    """Build a CorpusRetriever without loading a real SentenceTransformer model."""
    with patch(
        "mop_divpo.retrieval.corpus.SentenceTransformer",
        return_value=MagicMock(),
    ):
        retriever = CorpusRetriever(
            docs=docs,
            embeddings=embeddings,
            embedding_model="stub-model",
        )
    return retriever


def _stub_encode(retriever: CorpusRetriever, query_vec: np.ndarray) -> None:
    """Patch the retriever's internal model so that encode() returns query_vec."""
    mock_model = MagicMock()
    mock_model.encode.return_value = query_vec.reshape(1, -1).astype(np.float32)
    retriever._model = mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def three_doc_corpus():
    """Three orthogonal unit-vector embeddings for a clean cosine sim test."""
    docs = [
        _make_doc("apple cider vinegar", source="gutenberg"),
        _make_doc("neural network depth", source="arxiv"),
        _make_doc("city park design ideas", source="stackexchange"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    retriever = _make_retriever(docs, embeddings)
    return retriever, docs, embeddings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retriever_returns_k_results(three_doc_corpus):
    retriever, docs, _ = three_doc_corpus
    # query pointing toward doc[0]
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _stub_encode(retriever, query_vec)

    results = retriever.retrieve("apple", k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_retriever_handles_k_greater_than_corpus(three_doc_corpus):
    retriever, _, _ = three_doc_corpus
    query_vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    _stub_encode(retriever, query_vec)

    results = retriever.retrieve("anything", k=10)

    # corpus has 3 docs — should return all 3, not raise
    assert len(results) == 3


def test_corpus_novelty_score_range(three_doc_corpus):
    retriever, _, _ = three_doc_corpus
    # Use a query that is identical to doc[1] — cosine sim == 1 → novelty == 0
    query_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _stub_encode(retriever, query_vec)

    score = retriever.corpus_novelty_score("neural networks")

    assert 0.0 <= score <= 1.0


def test_corpus_novelty_empty_corpus_returns_one():
    """An empty retriever must return 1.0 novelty without errors."""
    empty_embeddings = np.empty((0, 3), dtype=np.float32)
    with patch(
        "mop_divpo.retrieval.corpus.SentenceTransformer",
        return_value=MagicMock(),
    ):
        retriever = CorpusRetriever(
            docs=[],
            embeddings=empty_embeddings,
            embedding_model="stub-model",
        )

    score = retriever.corpus_novelty_score("any text")
    assert score == 1.0


def test_retrieve_sorted_descending(three_doc_corpus):
    retriever, _, _ = three_doc_corpus
    # query pointing between doc[0] and doc[1], closer to doc[0]
    query_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    _stub_encode(retriever, query_vec)

    results = retriever.retrieve("mixed query", k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results are not sorted descending: {scores}"
    )
