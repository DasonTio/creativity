import json
from pathlib import Path

import typer

from mop_divpo.data.acquire import prepare_local_source
from mop_divpo.data.prepare import write_sample_sft_files
from mop_divpo.data.sources import list_dataset_sources
from mop_divpo.inference.generate import dry_run_generate

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
    limit: int | None = typer.Option(None),
):
    paths = prepare_local_source(source_name, raw_path, output_dir, limit)
    for path in paths:
        typer.echo(str(path))


@app.command("infer")
def infer(
    prompt: str,
    persona: list[str] = typer.Option(None, "--persona"),
):
    personas = persona or ["contrarian", "cross_domain_analogist", "systems_thinker", "minimalist"]
    outputs = dry_run_generate(prompt, personas)
    typer.echo(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    app()
