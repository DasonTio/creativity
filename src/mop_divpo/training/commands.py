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
