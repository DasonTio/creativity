from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class CorpusDocument:
    title: str
    snippet: str  # max ~80 chars
    source: str   # e.g. "cga-cmv", "arxiv", "stackexchange", "gutenberg"


@dataclass
class RetrievalResult:
    document: CorpusDocument
    score: float  # cosine similarity [0, 1]


class CorpusRetriever:
    def __init__(
        self,
        docs: list[CorpusDocument],
        embeddings: np.ndarray,  # shape: (N, D) float32
        embedding_model: str,
    ) -> None:
        self._docs = docs
        if SentenceTransformer is None:
            raise ModuleNotFoundError(
                "sentence_transformers is required for CorpusRetriever. "
                "Install project dependencies with `pip install -e .`."
            )
        self._model = SentenceTransformer(embedding_model)

        if len(docs) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # avoid division by zero for zero vectors
            norms = np.where(norms == 0, 1.0, norms)
            self._embeddings = (embeddings / norms).astype(np.float32)
        else:
            self._embeddings = embeddings.astype(np.float32)

    @classmethod
    def build_from_jsonl(
        cls,
        sft_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        docs_path: Path | None = None,
        embeddings_path: Path | None = None,
    ) -> "CorpusRetriever":
        sft_dir = Path(sft_dir)
        docs: list[CorpusDocument] = []
        prompts: list[str] = []

        if sft_dir.exists():
            for jsonl_file in sorted(sft_dir.glob("*.jsonl")):
                with jsonl_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "prompt" not in record:
                            continue
                        prompt: str = record["prompt"]
                        source: str = record.get("source", "unknown")
                        snippet = prompt[:80]
                        docs.append(
                            CorpusDocument(
                                title=snippet,
                                snippet=snippet,
                                source=source,
                            )
                        )
                        prompts.append(prompt)

        if not docs:
            raise ValueError(
                f"No documents found in {sft_dir}. "
                "Ensure *.jsonl files exist with 'prompt' fields."
            )

        if SentenceTransformer is None:
            raise ModuleNotFoundError(
                "sentence_transformers is required to build a CorpusRetriever. "
                "Install project dependencies with `pip install -e .`."
            )
        model = SentenceTransformer(embedding_model)
        embeddings: np.ndarray = model.encode(
            prompts, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)

        if docs_path is not None:
            docs_path = Path(docs_path)
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            with docs_path.open("w", encoding="utf-8") as fh:
                for doc in docs:
                    fh.write(
                        json.dumps(
                            {"title": doc.title, "snippet": doc.snippet, "source": doc.source}
                        )
                        + "\n"
                    )

        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)

        return cls(docs=docs, embeddings=embeddings, embedding_model=embedding_model)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if not self._docs:
            return []

        query_vec: np.ndarray = self._model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)[0]

        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores: np.ndarray = self._embeddings @ query_vec  # (N,)

        effective_k = min(k, len(self._docs))
        top_indices = np.argsort(scores)[::-1][:effective_k]

        return [
            RetrievalResult(document=self._docs[i], score=float(scores[i]))
            for i in top_indices
        ]

    def corpus_novelty_score(self, text: str) -> float:
        if not self._docs:
            return 1.0

        results = self.retrieve(text, k=1)
        if not results:
            return 1.0

        return 1.0 - results[0].score
