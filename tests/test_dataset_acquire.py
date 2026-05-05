from pathlib import Path

import pytest

from mop_divpo.data.acquire import load_local_jsonl, prepare_collected_sources, prepare_local_source
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


def test_prepare_local_source_handles_cga_raw_conversation(tmp_path: Path):
    raw_path = tmp_path / "cga_cmv.jsonl"
    output_dir = tmp_path / "processed"
    write_jsonl(
        raw_path,
        [
            {
                "conversation_id": "abc",
                "raw_convo": [
                    {"content": "Video games are uniquely harmful."},
                    {"content": "That assumes a single media source can be isolated from the larger system."},
                ],
            }
        ],
    )

    outputs = prepare_local_source("cga_cmv", raw_path, output_dir, limit=1)

    assert [path.name for path in outputs] == ["contrarian.jsonl"]
    row = read_jsonl(output_dir / "contrarian.jsonl")[0]
    assert row["persona"] == "contrarian"
    assert "Video games" in row["prompt"]
    assert "larger system" in row["response"]


def test_prepare_collected_sources_merges_persona_records(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    write_jsonl(raw_dir / "arxiv_abstracts.jsonl", [{"title": "Systems", "abstract": "Feedback matters."}])
    write_jsonl(raw_dir / "stackexchange_qa.jsonl", [{"prompt": "Why cache?", "gold_standard_solution": "Invalidation loops."}])

    outputs = prepare_collected_sources(["arxiv_abstracts", "stackexchange_qa"], raw_dir, output_dir, limit=1)

    assert sorted(path.name for path in outputs) == [
        "cross_domain_analogist.jsonl",
        "systems_thinker.jsonl",
    ]
    systems_rows = read_jsonl(output_dir / "systems_thinker.jsonl")
    assert len(systems_rows) == 2
