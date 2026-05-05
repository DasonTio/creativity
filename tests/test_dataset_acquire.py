from pathlib import Path

import pytest

from mop_divpo.data.acquire import load_local_jsonl, prepare_local_source
from mop_divpo.io import read_jsonl, write_jsonl


def test_load_local_jsonl_applies_limit(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    write_jsonl(raw_path, [{"title": "A"}, {"title": "B"}, {"title": "C"}])

    rows = load_local_jsonl(raw_path, limit=2)

    assert rows == [{"title": "A"}, {"title": "B"}]


def test_prepare_local_source_writes_normalized_arxiv_persona_files(tmp_path: Path):
    raw_path = tmp_path / "arxiv.jsonl"
    output_dir = tmp_path / "processed"
    write_jsonl(
        raw_path,
        [
            {
                "title": "Graph Neural Networks",
                "abstract": "Graphs encode relational inductive bias.",
            }
        ],
    )

    outputs = prepare_local_source("arxiv_abstracts", raw_path, output_dir, limit=1)

    assert sorted(path.name for path in outputs) == [
        "cross_domain_analogist.jsonl",
        "systems_thinker.jsonl",
    ]
    cross_domain = read_jsonl(output_dir / "cross_domain_analogist.jsonl")
    systems = read_jsonl(output_dir / "systems_thinker.jsonl")
    assert cross_domain[0]["persona"] == "cross_domain_analogist"
    assert "analogy" in cross_domain[0]["prompt"].lower()
    assert systems[0]["persona"] == "systems_thinker"
    assert "feedback loops" in systems[0]["prompt"].lower()


def test_prepare_local_source_rejects_unknown_source(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    write_jsonl(raw_path, [{"text": "sample"}])

    with pytest.raises(ValueError, match="Unknown dataset source"):
        prepare_local_source("unknown", raw_path, tmp_path / "processed", limit=1)
