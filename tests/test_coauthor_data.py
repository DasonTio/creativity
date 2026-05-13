from pathlib import Path

from mop_divpo.coauthor.data import (
    COAUTHOR_DATASET_SOURCES,
    build_seed_coauthor_records,
    records_from_argument_quality_rows,
    records_from_cmv_rows,
    records_from_writing_prompt_rows,
    write_coauthor_sft_files,
)
from mop_divpo.io import read_jsonl


def test_dataset_sources_are_relevant_to_pre_writing():
    names = {source.name for source in COAUTHOR_DATASET_SOURCES}

    assert "coauthor" in names
    assert "writing_prompts" in names
    assert "argument_quality_ranking" in names

    coauthor = next(source for source in COAUTHOR_DATASET_SOURCES if source.name == "coauthor")
    assert "interaction" in coauthor.use.lower()


def test_seed_records_cover_all_personas_and_real_coauthor_tasks():
    records = build_seed_coauthor_records()
    personas = {record.card.persona for record in records}
    operations = {record.card.operation for record in records}

    assert personas == {
        "contrarian",
        "cross_domain_analogist",
        "systems_thinker",
        "minimalist",
    }
    assert {"challenge_thesis", "transfer_analogy", "map_system", "distill_core"} <= operations
    assert all(record.brief.goal for record in records)
    assert all(record.card.next_step for record in records)


def test_seed_records_have_minimum_volume_per_persona():
    records = build_seed_coauthor_records()
    counts = {
        persona: len([record for record in records if record.card.persona == persona])
        for persona in {
            "contrarian",
            "cross_domain_analogist",
            "systems_thinker",
            "minimalist",
        }
    }

    assert counts == {
        "contrarian": 64,
        "cross_domain_analogist": 64,
        "systems_thinker": 64,
        "minimalist": 64,
    }


def test_write_coauthor_sft_files_outputs_jsonl_by_persona(tmp_path: Path):
    paths = write_coauthor_sft_files(tmp_path)

    assert {path.name for path in paths} == {
        "contrarian.jsonl",
        "cross_domain_analogist.jsonl",
        "systems_thinker.jsonl",
        "minimalist.jsonl",
    }

    rows = read_jsonl(tmp_path / "contrarian.jsonl")
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["role"] == "user"
    assert rows[0]["messages"][2]["role"] == "assistant"
    assert "diagnosis" in rows[0]["card"]


def test_writing_prompt_rows_convert_to_referenced_coauthor_records():
    rows = [{"prompt": "A city bans clocks after people stop sleeping."}]

    records = records_from_writing_prompt_rows(rows, limit=1)

    assert len(records) == 4
    assert {record.card.persona for record in records} == {
        "contrarian",
        "cross_domain_analogist",
        "systems_thinker",
        "minimalist",
    }
    training_row = records[0].to_training_row()
    assert training_row["references"][0]["dataset"] == "llm-aes/writing-prompts"
    assert "city bans clocks" in training_row["brief"]["topic"]


def test_argument_quality_rows_convert_with_quality_reference():
    rows = [
        {
            "topic": "We should ban advertising to children",
            "argument": "children cannot distinguish persuasion from information",
            "WA": "0.82",
        }
    ]

    records = records_from_argument_quality_rows(rows, limit=1)

    assert len(records) == 4
    assert records[0].references[0]["dataset"] == "ibm-research/argument_quality_ranking_30k"
    assert records[0].references[0]["quality_score"] == "0.82"


def test_cmv_rows_convert_to_contrarian_records_with_reference():
    rows = [
        {
            "conversation_id": "abc123",
            "raw_convo": [
                {"content": "AI writing tools make students lazy."},
                {"content": "That assumes effort only happens without tools."},
            ],
        }
    ]

    records = records_from_cmv_rows(rows, limit=1)

    assert len(records) == 1
    assert records[0].card.persona == "contrarian"
    assert records[0].references[0]["conversation_id"] == "abc123"
