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
        SFTRecord(id="bad", persona="optimist", source="unit", prompt="p", response="r")  # type: ignore[arg-type]


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
