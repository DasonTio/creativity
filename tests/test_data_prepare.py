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
