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
