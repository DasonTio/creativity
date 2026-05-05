# Mixture of Persona + DivPO Prototype

This repository implements a local prototype of the proposal in `Dason Tiovino - LLM - Mixture of Persona & Diverse Performance Optimizer.pdf`.

## Prototype Scope

The project demonstrates the full pipeline shape:

1. Prepare persona SFT data.
2. Train persona LoRA adapters with SFT command wiring.
3. Generate candidates.
4. Build DivPO-style chosen/rejected pairs.
5. Train DPO adapters with command wiring.
6. Route prompts to personas.
7. Generate persona-labeled ideas.
8. Evaluate output diversity.

The default base model is `Qwen/Qwen2.5-0.5B-Instruct`. Heavy model training is intentionally dry-run friendly so the code can be tested on a local machine.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run Tests

```bash
.venv/bin/pytest
```

## Prepare Sample Data

```bash
mop-divpo prepare-data --output-dir data/processed/sft
```

## Dry-Run Inference

```bash
mop-divpo infer "Invent essay ideas about urban loneliness." --persona minimalist --persona contrarian
```

## Notes

The first version approximates MixLoRA routing by selecting persona adapters and generating sequentially. The DivPO implementation constructs DPO-compatible preference pairs by preferring rare but high-quality candidates.
