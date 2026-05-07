import json
from pathlib import Path
from typing import Optional

import typer

from mop_divpo.data.acquire import collect_and_prepare_source, collect_source_to_raw, prepare_local_source
from mop_divpo.data.prepare import write_sample_sft_files
from mop_divpo.data.sources import list_dataset_sources
from mop_divpo.inference.generate import dry_run_generate, generate
from mop_divpo.personas import PERSONA_IDS

app = typer.Typer(help="MoP + DivPO prototype pipeline")


@app.command("prepare-data")
def prepare_data(output_dir: Path = typer.Option(Path("data/processed/sft"))):
    paths = write_sample_sft_files(output_dir)
    for path in paths:
        typer.echo(str(path))


@app.command("list-sources")
def list_sources():
    for source in list_dataset_sources():
        personas = ",".join(source["personas"])
        typer.echo(f"{source['name']}\t{source['acquisition']}\t{source['dataset_id']}\t{personas}")


@app.command("prepare-source")
def prepare_source(
    source_name: str,
    raw_path: Path = typer.Option(..., "--raw-path"),
    output_dir: Path = typer.Option(Path("data/processed/sft")),
    limit: Optional[int] = typer.Option(None),
):
    paths = prepare_local_source(source_name, raw_path, output_dir, limit)
    for path in paths:
        typer.echo(str(path))


@app.command("collect-source")
def collect_source(
    source_name: str,
    raw_dir: Path = typer.Option(Path("data/raw")),
    output_dir: Path = typer.Option(Path("data/processed/sft")),
    limit: int = typer.Option(50),
    prepare: bool = typer.Option(True),
):
    if prepare:
        paths = collect_and_prepare_source(source_name, raw_dir, output_dir, limit)
    else:
        paths = [collect_source_to_raw(source_name, raw_dir, limit)]
    for path in paths:
        typer.echo(str(path))


@app.command("train-sft")
def train_sft(
    persona: str = typer.Argument(help="Persona name"),
    data_dir: Path = typer.Option(Path("data/processed/sft"), "--data-dir"),
    output_dir: Path = typer.Option(Path("outputs/adapters/sft"), "--output-dir"),
    base_model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--base-model"),
    max_steps: int = typer.Option(200, "--max-steps"),
    batch_size: int = typer.Option(2, "--batch-size"),
    grad_accum: int = typer.Option(4, "--grad-accum"),
):
    from mop_divpo.training.train_sft import train

    data_path = data_dir / f"{persona}.jsonl"
    if not data_path.exists():
        typer.echo(f"Error: {data_path} not found. Run collect-source first.", err=True)
        raise typer.Exit(1)
    train(
        persona_name=persona,
        data_path=data_path,
        output_dir=output_dir / persona,
        base_model=base_model,
        max_steps=max_steps,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
    )


@app.command("train-all")
def train_all(
    data_dir: Path = typer.Option(Path("data/processed/sft"), "--data-dir"),
    output_dir: Path = typer.Option(Path("outputs/adapters/sft"), "--output-dir"),
    base_model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--base-model"),
    max_steps: int = typer.Option(200, "--max-steps"),
):
    from mop_divpo.training.train_sft import train

    for persona in PERSONA_IDS:
        data_path = data_dir / f"{persona}.jsonl"
        if not data_path.exists():
            typer.echo(f"Skipping {persona}: {data_path} not found", err=True)
            continue
        typer.echo(f"Training {persona}...")
        train(
            persona_name=persona,
            data_path=data_path,
            output_dir=output_dir / persona,
            base_model=base_model,
            max_steps=max_steps,
        )


@app.command("infer")
def infer(
    prompt: str,
    persona: list[str] = typer.Option(None, "--persona"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
):
    personas = persona or ["contrarian", "cross_domain_analogist", "systems_thinker", "minimalist"]
    if dry_run:
        outputs = dry_run_generate(prompt, personas)
    else:
        outputs = generate(prompt, personas, max_new_tokens)
    typer.echo(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    app()
