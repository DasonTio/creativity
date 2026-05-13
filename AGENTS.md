# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research prototype for MoP + DivPO. Source code lives in `src/mop_divpo/`, organized by pipeline area: `data/` for dataset acquisition and preparation, `divpo/` for preference-pair construction, `metrics/` and `eval/` for evaluation, `routing/` for persona gating, `inference/` for generation/session logic, and `training/` for SFT helpers. Tests live in `tests/` and mirror the feature areas. Runtime configuration is in `configs/prototype.yaml`. Raw, processed, and generated artifacts belong under `data/` and `outputs/`; these are ignored and should not be committed.

## Build, Test, and Development Commands

- `python3 -m venv .venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `pip install -e ".[dev]"`: install the package, CLI, and pytest dependency in editable mode.
- `pytest`: run the full dry-run-friendly test suite.
- `mop-divpo list-sources`: list configured dataset sources and target personas.
- `mop-divpo prepare-data --output-dir data/processed/sft`: generate sample SFT JSONL files.
- `mop-divpo infer "Prompt text" --dry-run`: exercise inference plumbing without loading a model.

For Apple Silicon training, prefix training commands with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax with 4-space indentation, type hints, and small functions that match the existing module layout. Prefer Pydantic models for structured records and Typer commands for CLI behavior. Persona identifiers must stay snake_case and match `contrarian`, `cross_domain_analogist`, `systems_thinker`, and `minimalist`. Test files and functions use `test_*.py` and `test_*` naming.

No formatter or linter is configured in `pyproject.toml`; keep imports, naming, and formatting consistent with nearby code.

## Testing Guidelines

Use pytest for all tests. Add focused unit tests beside related existing tests, such as `tests/test_output_parser.py` for inference parsing or `tests/test_divpo_pairs.py` for pair construction. Keep tests CPU-only and avoid network/model downloads unless explicitly isolated. Run `pytest` before submitting changes.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects, usually with a scope: `feat(inference): add PersonaSession dataclass`, `test(personas): add structured sections assertion test`. Keep subjects imperative and concise.

Pull requests should include a short problem/solution summary, commands run, linked issues if any, and screenshots or sample CLI output when behavior changes. Note any generated data, model downloads, or training artifacts intentionally left out of the diff.

## Security & Configuration Tips

Do not commit `.venv/`, `data/raw/`, `data/processed/`, `outputs/`, caches, model weights, or secrets. Keep reproducible defaults in `configs/prototype.yaml`, and document any expensive or network-dependent workflow in the PR.
