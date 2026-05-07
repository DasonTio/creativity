from pathlib import Path
from typing import Any

from pydantic import ValidationError

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


def collect_huggingface_rows(source: DatasetSource, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset_id = source.hf_mirror_dataset_id or source.dataset_id
    dataset = load_dataset(
        dataset_id,
        source.config_name,
        split=source.split,
        streaming=True,
        trust_remote_code=False,
    )
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def collect_source_to_raw(source_name: str, raw_dir: Path, limit: int) -> Path:
    source = get_dataset_source(source_name)
    if source.acquisition not in {"huggingface", "convokit"}:
        raise ValueError(f"Source '{source_name}' cannot be collected automatically")
    rows = collect_huggingface_rows(source, limit)
    raw_path = raw_dir / f"{source.name}.jsonl"
    write_jsonl(raw_path, rows)
    return raw_path


def collect_and_prepare_source(source_name: str, raw_dir: Path, output_dir: Path, limit: int) -> list[Path]:
    raw_path = collect_source_to_raw(source_name, raw_dir, limit)
    return prepare_local_source(source_name, raw_path, output_dir, limit)


def normalize_source_rows(source: DatasetSource, rows: list[dict[str, Any]]) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    for index, row in enumerate(rows):
        try:
            if source.name == "cga_cmv":
                raw_convo = row.get("raw_convo")
                if isinstance(raw_convo, list) and raw_convo:
                    title = str(raw_convo[0].get("content", ""))
                    body = "\n\n".join(
                        str(turn.get("content", ""))
                        for turn in raw_convo[1:3]
                        if isinstance(turn, dict) and turn.get("content")
                    )
                    raw = {"title": title, "body": body or title}
                else:
                    raw = {
                        "title": "" if row.get("title") is None else str(row.get("title", "")),
                        "body": "" if row.get("body") is None else str(row.get("body", "")),
                    }
                records.append(normalize_reddit_record(raw, index))
            elif source.name == "arxiv_abstracts":
                raw = {key: "" if value is None else str(value) for key, value in row.items()}
                records.append(normalize_arxiv_record(raw, "cross_domain_analogist", index))
                records.append(normalize_arxiv_record(raw, "systems_thinker", index))
            elif source.name == "stackexchange_qa":
                raw = {key: "" if value is None else str(value) for key, value in row.items()}
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
                raw = {key: "" if value is None else str(value) for key, value in row.items()}
                if len(raw.get("text", "").strip()) < 80:
                    continue
                records.append(normalize_gutenberg_record(raw, index))
            else:
                raise ValueError(f"No normalizer registered for source '{source.name}'")
        except ValidationError:
            continue
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


def prepare_collected_sources(
    source_names: list[str],
    raw_dir: Path,
    output_dir: Path,
    limit: int | None = None,
) -> list[Path]:
    records: list[SFTRecord] = []
    for source_name in source_names:
        source = get_dataset_source(source_name)
        raw_path = raw_dir / f"{source.name}.jsonl"
        rows = load_local_jsonl(raw_path, limit)
        records.extend(normalize_source_rows(source, rows))
    return write_sft_records_by_persona(records, output_dir)
