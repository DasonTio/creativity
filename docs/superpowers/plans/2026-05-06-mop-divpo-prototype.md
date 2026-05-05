# MoP DivPO Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable local prototype for Mixture of Persona + DivPO that covers data preparation, preference-pair construction, routing, inference, training command wiring, and evaluation.

**Architecture:** Use a small Python package with focused modules for schema/config, data normalization, DivPO scoring/selection, persona routing, metrics, and CLI entrypoints. Model-heavy operations are wrapped behind dry-run-friendly interfaces so the project can be tested without downloading large models.

**Tech Stack:** Python 3.10+, pytest, pydantic, typer, datasets, transformers, peft, trl, sentence-transformers, nltk, scikit-learn, numpy.

---

## File Structure

- Create `pyproject.toml`: package metadata, dependencies, pytest config, console script.
- Create `README.md`: start-to-finish usage and prototype limitations.
- Create `configs/prototype.yaml`: default model, dataset sample sizes, training, DivPO, routing, and evaluation settings.
- Create `src/mop_divpo/__init__.py`: package version.
- Create `src/mop_divpo/personas.py`: canonical persona ids, descriptions, validation.
- Create `src/mop_divpo/schema.py`: pydantic records for SFT examples, candidates, DivPO pairs, and evaluation outputs.
- Create `src/mop_divpo/config.py`: YAML config loading and validation.
- Create `src/mop_divpo/io.py`: JSONL read/write helpers.
- Create `src/mop_divpo/data/normalizers.py`: convert raw-ish records into shared SFT schema.
- Create `src/mop_divpo/data/prepare.py`: sample-data preparation orchestration.
- Create `src/mop_divpo/metrics/diversity.py`: distinct-n and Self-BLEU metrics.
- Create `src/mop_divpo/metrics/semantic.py`: embedding cosine helpers with deterministic fallback.
- Create `src/mop_divpo/divpo/scoring.py`: rarity and quality scoring.
- Create `src/mop_divpo/divpo/pairs.py`: chosen/rejected pair selection and JSONL writing.
- Create `src/mop_divpo/routing/gate.py`: embedding-based persona gate.
- Create `src/mop_divpo/training/commands.py`: SFT/DPO command builders and dry-run execution.
- Create `src/mop_divpo/inference/generate.py`: persona-labeled generation interface with dry-run output.
- Create `src/mop_divpo/eval/evaluate.py`: compare generation files and emit metrics.
- Create `src/mop_divpo/cli.py`: Typer CLI with pipeline commands.
- Create tests mirroring each module under `tests/`.

If this directory is still not a git repository when execution starts, run `git init` before the first commit so the commit steps work.

---

### Task 1: Project Scaffold And Persona Registry

**Files:**
- Create: `pyproject.toml`
- Create: `src/mop_divpo/__init__.py`
- Create: `src/mop_divpo/personas.py`
- Test: `tests/test_personas.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_personas.py
import pytest

from mop_divpo.personas import PERSONA_IDS, PersonaId, describe_persona, validate_persona


def test_persona_ids_match_proposal():
    assert PERSONA_IDS == (
        "contrarian",
        "cross_domain_analogist",
        "systems_thinker",
        "minimalist",
    )


def test_describe_persona_returns_research_grounded_description():
    description = describe_persona("systems_thinker")
    assert "causal" in description.lower()
    assert "feedback" in description.lower()


def test_validate_persona_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown persona"):
        validate_persona("optimist")


def test_persona_type_accepts_canonical_ids():
    value: PersonaId = "minimalist"
    assert validate_persona(value) == "minimalist"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_personas.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mop_divpo'`.

- [ ] **Step 3: Write minimal project/package implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mop-divpo"
version = "0.1.0"
description = "Mixture of Persona + Diverse Preference Optimization prototype"
requires-python = ">=3.10"
dependencies = [
  "datasets>=2.19",
  "nltk>=3.8",
  "numpy>=1.26",
  "peft>=0.11",
  "pydantic>=2.7",
  "pyyaml>=6.0",
  "scikit-learn>=1.4",
  "sentence-transformers>=3.0",
  "torch>=2.2",
  "transformers>=4.44",
  "trl>=0.9",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.2"]

[project.scripts]
mop-divpo = "mop_divpo.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/mop_divpo/__init__.py
__version__ = "0.1.0"
```

```python
# src/mop_divpo/personas.py
from typing import Literal

PersonaId = Literal[
    "contrarian",
    "cross_domain_analogist",
    "systems_thinker",
    "minimalist",
]

PERSONA_IDS: tuple[PersonaId, ...] = (
    "contrarian",
    "cross_domain_analogist",
    "systems_thinker",
    "minimalist",
)

PERSONA_DESCRIPTIONS: dict[PersonaId, str] = {
    "contrarian": "Challenges assumptions, surfaces minority dissent, and reframes common viewpoints.",
    "cross_domain_analogist": "Builds analogies across distant fields to transfer useful structures.",
    "systems_thinker": "Explains causal relationships, feedback loops, trade-offs, and long-term effects.",
    "minimalist": "Uses constraints, compression, and removal of non-essential parts to sharpen ideas.",
}


def validate_persona(value: str) -> PersonaId:
    if value not in PERSONA_IDS:
        allowed = ", ".join(PERSONA_IDS)
        raise ValueError(f"Unknown persona '{value}'. Expected one of: {allowed}")
    return value  # type: ignore[return-value]


def describe_persona(persona: PersonaId) -> str:
    return PERSONA_DESCRIPTIONS[validate_persona(persona)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_personas.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/mop_divpo/__init__.py src/mop_divpo/personas.py tests/test_personas.py
git commit -m "chore: scaffold package and persona registry"
```

---

### Task 2: Schemas, JSONL IO, And Config Loading

**Files:**
- Create: `configs/prototype.yaml`
- Create: `src/mop_divpo/schema.py`
- Create: `src/mop_divpo/io.py`
- Create: `src/mop_divpo/config.py`
- Test: `tests/test_schema_io_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema_io_config.py
from pathlib import Path

import pytest
from pydantic import ValidationError

from mop_divpo.config import load_config
from mop_divpo.io import read_jsonl, write_jsonl
from mop_divpo.schema import CandidateRecord, DivPOPairRecord, SFTRecord


def test_sft_record_validates_persona():
    record = SFTRecord(
        id="c1",
        persona="contrarian",
        source="unit",
        prompt="Give a new angle on remote work.",
        response="Remote work can reduce weak-tie learning unless rituals replace hallway context.",
    )
    assert record.persona == "contrarian"

    with pytest.raises(ValidationError):
        SFTRecord(id="bad", persona="optimist", source="unit", prompt="p", response="r")


def test_candidate_and_pair_records_have_dpo_fields():
    candidate = CandidateRecord(
        id="cand-1",
        persona="minimalist",
        prompt="Invent a writing exercise.",
        response="Remove every adjective, then add back only one.",
        quality_score=0.8,
        rarity_score=0.7,
    )
    pair = DivPOPairRecord.from_candidates(
        id="pair-1",
        chosen=candidate,
        rejected=CandidateRecord(
            id="cand-2",
            persona="minimalist",
            prompt=candidate.prompt,
            response="Write a list of ideas.",
            quality_score=0.4,
            rarity_score=0.1,
        ),
    )
    assert pair.prompt == candidate.prompt
    assert pair.chosen == candidate.response
    assert pair.rejected == "Write a list of ideas."


def test_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": "1", "text": "alpha"}, {"id": "2", "text": "beta"}])
    assert read_jsonl(path) == [{"id": "1", "text": "alpha"}, {"id": "2", "text": "beta"}]


def test_load_config_reads_prototype_defaults():
    config = load_config(Path("configs/prototype.yaml"))
    assert config["model"]["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["routing"]["top_k"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_schema_io_config.py -v`

Expected: FAIL with missing modules or missing config file.

- [ ] **Step 3: Write minimal implementation**

```yaml
# configs/prototype.yaml
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

data:
  raw_dir: data/raw
  processed_dir: data/processed
  samples_per_persona: 50

training:
  output_dir: outputs/adapters
  max_steps: 10
  learning_rate: 0.0002
  dry_run: true

divpo:
  candidates_per_prompt: 4
  min_quality: 0.35
  rarity_weight: 0.6
  quality_weight: 0.4

routing:
  top_k: 2

evaluation:
  output_dir: outputs/evaluation
```

```python
# src/mop_divpo/schema.py
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mop_divpo.personas import PersonaId, validate_persona


class SFTRecord(BaseModel):
    id: str
    persona: PersonaId
    source: str
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)


class CandidateRecord(BaseModel):
    id: str
    persona: PersonaId
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    quality_score: float = 0.0
    rarity_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)


class DivPOPairRecord(BaseModel):
    id: str
    persona: PersonaId
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona")
    @classmethod
    def _valid_persona(cls, value: str) -> PersonaId:
        return validate_persona(value)

    @classmethod
    def from_candidates(
        cls,
        id: str,
        chosen: CandidateRecord,
        rejected: CandidateRecord,
    ) -> "DivPOPairRecord":
        if chosen.persona != rejected.persona:
            raise ValueError("chosen and rejected candidates must use the same persona")
        if chosen.prompt != rejected.prompt:
            raise ValueError("chosen and rejected candidates must use the same prompt")
        return cls(
            id=id,
            persona=chosen.persona,
            prompt=chosen.prompt,
            chosen=chosen.response,
            rejected=rejected.response,
            metadata={"chosen_id": chosen.id, "rejected_id": rejected.id},
        )
```

```python
# src/mop_divpo/io.py
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
```

```python
# src/mop_divpo/config.py
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    for key in ("model", "data", "training", "divpo", "routing", "evaluation"):
        if key not in config:
            raise ValueError(f"Config at {path} is missing required section '{key}'")
    return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_schema_io_config.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add configs/prototype.yaml src/mop_divpo/schema.py src/mop_divpo/io.py src/mop_divpo/config.py tests/test_schema_io_config.py
git commit -m "feat: add schemas config and jsonl io"
```

---

### Task 3: Dataset Normalizers And Sample Preparation

**Files:**
- Create: `src/mop_divpo/data/__init__.py`
- Create: `src/mop_divpo/data/normalizers.py`
- Create: `src/mop_divpo/data/prepare.py`
- Test: `tests/test_data_prepare.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_prepare.py
from pathlib import Path

from mop_divpo.data.normalizers import (
    normalize_arxiv_record,
    normalize_gutenberg_record,
    normalize_reddit_record,
    normalize_stackexchange_record,
)
from mop_divpo.data.prepare import write_sample_sft_files
from mop_divpo.io import read_jsonl


def test_normalize_reddit_record_for_contrarian():
    record = normalize_reddit_record(
        {"title": "AI makes writing better", "body": "Most people agree it improves ideation."},
        index=3,
    )
    assert record.persona == "contrarian"
    assert "Challenge this assumption" in record.prompt
    assert "Most people agree" in record.response


def test_normalize_arxiv_record_for_cross_domain():
    record = normalize_arxiv_record(
        {"title": "Graph Neural Networks", "abstract": "Graphs encode relational inductive bias."},
        persona="cross_domain_analogist",
        index=1,
    )
    assert record.persona == "cross_domain_analogist"
    assert "analogy" in record.prompt.lower()


def test_normalize_stackexchange_record_for_systems():
    record = normalize_stackexchange_record(
        {"question": "Why does caching fail?", "answer": "Invalidation creates hidden coupling."},
        index=4,
    )
    assert record.persona == "systems_thinker"
    assert "feedback loops" in record.prompt.lower()


def test_normalize_gutenberg_record_for_minimalist():
    record = normalize_gutenberg_record({"text": "It was a bright cold day in April."}, index=2)
    assert record.persona == "minimalist"
    assert "constraint" in record.prompt.lower()


def test_write_sample_sft_files_creates_one_file_per_persona(tmp_path: Path):
    outputs = write_sample_sft_files(tmp_path)
    assert sorted(path.name for path in outputs) == [
        "contrarian.jsonl",
        "cross_domain_analogist.jsonl",
        "minimalist.jsonl",
        "systems_thinker.jsonl",
    ]
    for path in outputs:
        rows = read_jsonl(path)
        assert len(rows) == 1
        assert rows[0]["persona"] in path.stem
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_data_prepare.py -v`

Expected: FAIL with missing `mop_divpo.data` module.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/data/__init__.py
```

```python
# src/mop_divpo/data/normalizers.py
from mop_divpo.personas import PersonaId
from mop_divpo.schema import SFTRecord


def normalize_reddit_record(raw: dict[str, str], index: int) -> SFTRecord:
    title = raw.get("title", "").strip()
    body = raw.get("body", "").strip()
    return SFTRecord(
        id=f"reddit-{index}",
        persona="contrarian",
        source="cga-cmv",
        prompt=f"Challenge this assumption for pre-writing ideation: {title}",
        response=body or title,
        metadata={"title": title},
    )


def normalize_arxiv_record(raw: dict[str, str], persona: PersonaId, index: int) -> SFTRecord:
    title = raw.get("title", "").strip()
    abstract = raw.get("abstract", "").strip()
    if persona == "cross_domain_analogist":
        prompt = f"Create a cross-domain analogy from this research idea: {title}"
    else:
        prompt = f"Explain the causal system and feedback loops behind this research idea: {title}"
    return SFTRecord(
        id=f"arxiv-{persona}-{index}",
        persona=persona,
        source="arxiv",
        prompt=prompt,
        response=abstract or title,
        metadata={"title": title},
    )


def normalize_stackexchange_record(raw: dict[str, str], index: int) -> SFTRecord:
    question = raw.get("question", "").strip()
    answer = raw.get("answer", "").strip()
    return SFTRecord(
        id=f"stackexchange-{index}",
        persona="systems_thinker",
        source="stackexchange",
        prompt=f"Analyze this question through causes, constraints, and feedback loops: {question}",
        response=answer or question,
    )


def normalize_gutenberg_record(raw: dict[str, str], index: int) -> SFTRecord:
    text = raw.get("text", "").strip()
    return SFTRecord(
        id=f"gutenberg-{index}",
        persona="minimalist",
        source="gutenberg",
        prompt="Create a constraint-driven minimalist writing idea from this passage.",
        response=text,
    )
```

```python
# src/mop_divpo/data/prepare.py
from pathlib import Path

from mop_divpo.data.normalizers import (
    normalize_arxiv_record,
    normalize_gutenberg_record,
    normalize_reddit_record,
    normalize_stackexchange_record,
)
from mop_divpo.io import write_jsonl
from mop_divpo.personas import PERSONA_IDS


def sample_sft_records():
    return [
        normalize_reddit_record(
            {"title": "AI always improves brainstorming", "body": "A contrarian view asks what idea diversity is lost."},
            0,
        ),
        normalize_arxiv_record(
            {"title": "Swarm robotics", "abstract": "A colony-like analogy can inspire modular story planning."},
            "cross_domain_analogist",
            0,
        ),
        normalize_stackexchange_record(
            {"question": "Why do teams miss deadlines?", "answer": "Feedback delays hide blockers until coordination cost rises."},
            0,
        ),
        normalize_gutenberg_record(
            {"text": "A scene can become stronger when one image carries the whole emotion."},
            0,
        ),
    ]


def write_sample_sft_files(output_dir: Path) -> list[Path]:
    records_by_persona = {persona: [] for persona in PERSONA_IDS}
    for record in sample_sft_records():
        records_by_persona[record.persona].append(record.model_dump())

    paths: list[Path] = []
    for persona in PERSONA_IDS:
        path = output_dir / f"{persona}.jsonl"
        write_jsonl(path, records_by_persona[persona])
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_data_prepare.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/data tests/test_data_prepare.py
git commit -m "feat: add persona dataset normalizers"
```

---

### Task 4: Diversity And Semantic Metrics

**Files:**
- Create: `src/mop_divpo/metrics/__init__.py`
- Create: `src/mop_divpo/metrics/diversity.py`
- Create: `src/mop_divpo/metrics/semantic.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np

from mop_divpo.metrics.diversity import distinct_n, self_bleu
from mop_divpo.metrics.semantic import average_pairwise_cosine, cosine_similarity_matrix


def test_distinct_n_is_higher_for_varied_text():
    repeated = ["alpha alpha", "alpha alpha"]
    varied = ["alpha beta", "gamma delta"]
    assert distinct_n(varied, n=1) > distinct_n(repeated, n=1)


def test_distinct_n_handles_empty_outputs():
    assert distinct_n([], n=1) == 0.0
    assert distinct_n([""], n=2) == 0.0


def test_self_bleu_is_lower_for_diverse_outputs():
    repeated = ["the same idea", "the same idea", "the same idea"]
    varied = ["use a map", "borrow from biology", "remove the premise"]
    assert self_bleu(varied) < self_bleu(repeated)


def test_cosine_similarity_matrix_and_average():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    matrix = cosine_similarity_matrix(embeddings)
    assert matrix.shape == (3, 3)
    assert matrix[0, 2] == 1.0
    assert round(average_pairwise_cosine(embeddings), 3) == 0.333
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_metrics.py -v`

Expected: FAIL with missing metrics modules.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/metrics/__init__.py
```

```python
# src/mop_divpo/metrics/diversity.py
from collections import Counter


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def distinct_n(outputs: list[str], n: int) -> float:
    ngrams: list[tuple[str, ...]] = []
    for output in outputs:
        tokens = _tokens(output)
        ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def _overlap_score(candidate: str, references: list[str]) -> float:
    candidate_tokens = _tokens(candidate)
    if not candidate_tokens or not references:
        return 0.0
    reference_counts = Counter(token for ref in references for token in _tokens(ref))
    overlap = sum(1 for token in candidate_tokens if reference_counts[token] > 0)
    return overlap / len(candidate_tokens)


def self_bleu(outputs: list[str]) -> float:
    if len(outputs) < 2:
        return 0.0
    scores = []
    for index, output in enumerate(outputs):
        refs = outputs[:index] + outputs[index + 1 :]
        scores.append(_overlap_score(output, refs))
    return sum(scores) / len(scores)
```

```python
# src/mop_divpo/metrics/semantic.py
import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / safe
    return normalized @ normalized.T


def average_pairwise_cosine(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    matrix = cosine_similarity_matrix(embeddings)
    upper = matrix[np.triu_indices(len(embeddings), k=1)]
    return float(upper.mean()) if len(upper) else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_metrics.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/metrics tests/test_metrics.py
git commit -m "feat: add diversity and semantic metrics"
```

---

### Task 5: DivPO Scoring And Pair Selection

**Files:**
- Create: `src/mop_divpo/divpo/__init__.py`
- Create: `src/mop_divpo/divpo/scoring.py`
- Create: `src/mop_divpo/divpo/pairs.py`
- Test: `tests/test_divpo_pairs.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_divpo_pairs.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_divpo_pairs.py -v`

Expected: FAIL with missing `mop_divpo.divpo` module.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/divpo/__init__.py
```

```python
# src/mop_divpo/divpo/scoring.py
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
```

```python
# src/mop_divpo/divpo/pairs.py
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
        key=lambda candidate: combined_divpo_score(
            candidate.quality_score,
            candidate.rarity_score,
            quality_weight,
            rarity_weight,
        ),
    )
    return DivPOPairRecord.from_candidates(
        id=f"divpo-{chosen.persona}-{chosen.id}-{rejected.id}",
        chosen=chosen,
        rejected=rejected,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_divpo_pairs.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/divpo tests/test_divpo_pairs.py
git commit -m "feat: add divpo scoring and pair selection"
```

---

### Task 6: Persona Gate

**Files:**
- Create: `src/mop_divpo/routing/__init__.py`
- Create: `src/mop_divpo/routing/gate.py`
- Test: `tests/test_routing_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_routing_gate.py
from mop_divpo.routing.gate import KeywordPersonaGate


def test_gate_routes_challenge_prompt_to_contrarian():
    gate = KeywordPersonaGate()
    ranked = gate.rank("Challenge the common assumption that productivity means speed.", top_k=2)
    assert ranked[0].persona == "contrarian"


def test_gate_routes_feedback_prompt_to_systems_thinker():
    gate = KeywordPersonaGate()
    ranked = gate.rank("Find feedback loops and long-term side effects in school grading.", top_k=2)
    assert ranked[0].persona == "systems_thinker"


def test_gate_can_force_personas():
    gate = KeywordPersonaGate()
    ranked = gate.rank("Any prompt", top_k=2, force_personas=["minimalist", "contrarian"])
    assert [item.persona for item in ranked] == ["minimalist", "contrarian"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_routing_gate.py -v`

Expected: FAIL with missing routing module.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/routing/__init__.py
```

```python
# src/mop_divpo/routing/gate.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_routing_gate.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/routing tests/test_routing_gate.py
git commit -m "feat: add lightweight persona gate"
```

---

### Task 7: Training And Inference Dry-Run Interfaces

**Files:**
- Create: `src/mop_divpo/training/__init__.py`
- Create: `src/mop_divpo/training/commands.py`
- Create: `src/mop_divpo/inference/__init__.py`
- Create: `src/mop_divpo/inference/generate.py`
- Test: `tests/test_training_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_training_inference.py
from pathlib import Path

from mop_divpo.inference.generate import dry_run_generate
from mop_divpo.training.commands import build_dpo_command, build_sft_command


def test_build_sft_command_contains_adapter_output_path():
    command = build_sft_command(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        train_file=Path("data/processed/sft/contrarian.jsonl"),
        output_dir=Path("outputs/adapters/sft/contrarian"),
        max_steps=10,
    )
    assert "--model_name_or_path Qwen/Qwen2.5-0.5B-Instruct" in command
    assert "--output_dir outputs/adapters/sft/contrarian" in command


def test_build_dpo_command_uses_sft_adapter_and_pairs_file():
    command = build_dpo_command(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=Path("outputs/adapters/sft/minimalist"),
        pairs_file=Path("data/processed/divpo/minimalist.jsonl"),
        output_dir=Path("outputs/adapters/divpo/minimalist"),
        max_steps=10,
    )
    assert "--adapter_name_or_path outputs/adapters/sft/minimalist" in command
    assert "--dataset_name data/processed/divpo/minimalist.jsonl" in command


def test_dry_run_generate_returns_persona_labeled_outputs():
    outputs = dry_run_generate("Invent essay ideas about cities.", personas=["minimalist", "contrarian"])
    assert [item["persona"] for item in outputs] == ["minimalist", "contrarian"]
    assert all("response" in item for item in outputs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training_inference.py -v`

Expected: FAIL with missing training or inference modules.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/training/__init__.py
```

```python
# src/mop_divpo/training/commands.py
from pathlib import Path


def build_sft_command(base_model: str, train_file: Path, output_dir: Path, max_steps: int) -> str:
    return (
        "trl sft "
        f"--model_name_or_path {base_model} "
        f"--dataset_name {train_file} "
        f"--output_dir {output_dir} "
        f"--max_steps {max_steps} "
        "--use_peft true"
    )


def build_dpo_command(
    base_model: str,
    adapter_path: Path,
    pairs_file: Path,
    output_dir: Path,
    max_steps: int,
) -> str:
    return (
        "trl dpo "
        f"--model_name_or_path {base_model} "
        f"--adapter_name_or_path {adapter_path} "
        f"--dataset_name {pairs_file} "
        f"--output_dir {output_dir} "
        f"--max_steps {max_steps} "
        "--use_peft true"
    )
```

```python
# src/mop_divpo/inference/__init__.py
```

```python
# src/mop_divpo/inference/generate.py
from mop_divpo.personas import describe_persona, validate_persona


def dry_run_generate(prompt: str, personas: list[str]) -> list[dict[str, str]]:
    outputs = []
    for persona_value in personas:
        persona = validate_persona(persona_value)
        outputs.append(
            {
                "persona": persona,
                "prompt": prompt,
                "response": f"[dry-run:{persona}] {describe_persona(persona)} Prompt: {prompt}",
            }
        )
    return outputs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training_inference.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/training src/mop_divpo/inference tests/test_training_inference.py
git commit -m "feat: add training command and inference dry runs"
```

---

### Task 8: Evaluation Runner

**Files:**
- Create: `src/mop_divpo/eval/__init__.py`
- Create: `src/mop_divpo/eval/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
from mop_divpo.eval.evaluate import evaluate_outputs


def test_evaluate_outputs_returns_diversity_metrics():
    outputs = [
        {"persona": "minimalist", "response": "Remove one premise and keep the sharpest image."},
        {"persona": "contrarian", "response": "Reject the obvious premise and test its inverse."},
        {"persona": "systems_thinker", "response": "Trace the feedback loop between incentives and habits."},
    ]
    metrics = evaluate_outputs(outputs)
    assert set(metrics) == {"count", "distinct_1", "distinct_2", "self_bleu"}
    assert metrics["count"] == 3
    assert metrics["distinct_1"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_evaluate.py -v`

Expected: FAIL with missing eval module.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/eval/__init__.py
```

```python
# src/mop_divpo/eval/evaluate.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_evaluate.py -v`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/eval tests/test_evaluate.py
git commit -m "feat: add evaluation metrics runner"
```

---

### Task 9: CLI Pipeline

**Files:**
- Create: `src/mop_divpo/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from typer.testing import CliRunner

from mop_divpo.cli import app


def test_cli_prepare_sample_data(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["prepare-data", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "contrarian.jsonl").exists()


def test_cli_infer_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["infer", "Invent essay ideas about libraries.", "--persona", "minimalist", "--persona", "contrarian"],
    )
    assert result.exit_code == 0
    assert "minimalist" in result.stdout
    assert "contrarian" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py -v`

Expected: FAIL with missing CLI module.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mop_divpo/cli.py
import json
from pathlib import Path

import typer

from mop_divpo.data.prepare import write_sample_sft_files
from mop_divpo.inference.generate import dry_run_generate

app = typer.Typer(help="MoP + DivPO prototype pipeline")


@app.command("prepare-data")
def prepare_data(output_dir: Path = typer.Option(Path("data/processed/sft"))):
    paths = write_sample_sft_files(output_dir)
    for path in paths:
        typer.echo(str(path))


@app.command("infer")
def infer(
    prompt: str,
    persona: list[str] = typer.Option(None, "--persona"),
):
    personas = persona or ["contrarian", "cross_domain_analogist", "systems_thinker", "minimalist"]
    outputs = dry_run_generate(prompt, personas)
    typer.echo(json.dumps(outputs, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mop_divpo/cli.py tests/test_cli.py
git commit -m "feat: add prototype cli"
```

---

### Task 10: README And Full Verification

**Files:**
- Create: `README.md`
- Modify: no production modules unless verification reveals a defect

- [ ] **Step 1: Write README**

```markdown
# Mixture of Persona + DivPO Prototype

This repository implements a local prototype of the proposal in `Dason Tiovino - LLM - Mixture of Persona & Diverse Performance Optimizer.pdf`.

## Prototype Scope

The project demonstrates the full pipeline shape:

1. Prepare persona SFT data.
2. Train persona LoRA adapters with SFT command wiring.
3. Generate candidates.
4. Build DivPO-style chosen/rejected pairs.
5. Train DPO adapters with command wiring.
6. Route prompts to personas.
7. Generate persona-labeled ideas.
8. Evaluate output diversity.

The default base model is `Qwen/Qwen2.5-0.5B-Instruct`. Heavy model training is intentionally dry-run friendly so the code can be tested on a local machine.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run Tests

```bash
.venv/bin/pytest
```

## Prepare Sample Data

```bash
mop-divpo prepare-data --output-dir data/processed/sft
```

## Dry-Run Inference

```bash
mop-divpo infer "Invent essay ideas about urban loneliness." --persona minimalist --persona contrarian
```

## Notes

The first version approximates MixLoRA routing by selecting persona adapters and generating sequentially. The DivPO implementation constructs DPO-compatible preference pairs by preferring rare but high-quality candidates.
```

- [ ] **Step 2: Run all tests**

Run: `python3 -m pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Run CLI smoke checks**

Run: `python3 -m mop_divpo.cli prepare-data --output-dir data/processed/sft`

Expected: prints four JSONL paths.

Run: `python3 -m mop_divpo.cli infer "Invent essay ideas about urban loneliness." --persona minimalist --persona contrarian`

Expected: prints JSON containing `minimalist` and `contrarian`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document prototype workflow"
```

---

## Self-Review Notes

Spec coverage:

- Data preparation is covered by Tasks 2 and 3.
- Persona registry and routing are covered by Tasks 1 and 6.
- DivPO pair construction is covered by Tasks 4 and 5.
- Training command wiring is covered by Task 7.
- Inference is covered by Tasks 7 and 9.
- Evaluation is covered by Tasks 4 and 8.
- README workflow is covered by Task 10.

Known deliberate limitation:

- Real Hugging Face dataset downloads and actual LoRA/DPO trainer scripts are command-wired but not fully trained in the first implementation pass. This matches the approved local/course-project prototype scope and keeps the first end-to-end system testable on a local machine.
