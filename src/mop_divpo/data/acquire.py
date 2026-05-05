from pathlib import Path
from typing import Any

from mop_divpo.data.normalizers import (
    normalize_arxiv_record,
    normalize_gutenberg_record,
    normalize_reddit_record,
    normalize_stackexchange_record,
)
from mop_divpo.data.sources import DatasetSource, get_dataset_source
from mop_divpo.io import read_jsonl, write_jsonl
from mop_divpo.personas import PERSONA_IDS
from mop_divpo.schema import SFTRecord


def load_local_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return rows[:limit] if limit is not None else rows


def normalize_source_rows(source: DatasetSource, rows: list[dict[str, Any]]) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    for index, row in enumerate(rows):
        raw = {key: "" if value is None else str(value) for key, value in row.items()}
        if source.name == "cga_cmv":
            records.append(normalize_reddit_record(raw, index))
        elif source.name == "arxiv_abstracts":
            records.append(normalize_arxiv_record(raw, "cross_domain_analogist", index))
            records.append(normalize_arxiv_record(raw, "systems_thinker", index))
        elif source.name == "stackexchange_qa":
            records.append(
                normalize_stackexchange_record(
                    {
                        "question": raw.get("question") or raw.get("prompt", ""),
                        "answer": raw.get("answer") or raw.get("gold_standard_solution", ""),
                    },
                    index,
                )
            )
        elif source.name == "gutenberg":
            records.append(normalize_gutenberg_record(raw, index))
        else:
            raise ValueError(f"No normalizer registered for source '{source.name}'")
    return records


def write_sft_records_by_persona(records: list[SFTRecord], output_dir: Path) -> list[Path]:
    records_by_persona = {persona: [] for persona in PERSONA_IDS}
    for record in records:
        records_by_persona[record.persona].append(record.model_dump())

    paths: list[Path] = []
    for persona, persona_records in records_by_persona.items():
        if not persona_records:
            continue
        path = output_dir / f"{persona}.jsonl"
        write_jsonl(path, persona_records)
        paths.append(path)
    return paths


def prepare_local_source(source_name: str, raw_path: Path, output_dir: Path, limit: int | None = None) -> list[Path]:
    source = get_dataset_source(source_name)
    rows = load_local_jsonl(raw_path, limit)
    records = normalize_source_rows(source, rows)
    return write_sft_records_by_persona(records, output_dir)
