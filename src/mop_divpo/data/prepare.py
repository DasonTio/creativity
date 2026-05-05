from pathlib import Path

from mop_divpo.data.normalizers import (
    normalize_arxiv_record,
    normalize_gutenberg_record,
    normalize_reddit_record,
    normalize_stackexchange_record,
)
from mop_divpo.io import write_jsonl
from mop_divpo.personas import PERSONA_IDS
from mop_divpo.schema import SFTRecord


def sample_sft_records() -> list[SFTRecord]:
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
