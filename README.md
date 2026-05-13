<div align="center">

# MoP + DivPO

**Mixture of Persona + Diverse Preference Optimization**

*Increasing LLM output diversity for pre-writing ideation without sacrificing quality*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-40%20passed-brightgreen)](#running-tests)

Research project — **Pradita University · LLM & Generative AI**

</div>

---

## Overview

MoP+DivPO addresses *generative monoculture* in LLMs by combining two techniques:

- **Mixture of Persona (MoP)** — four LoRA adapters, each fine-tuned on a distinct cognitive style, routed by a learned gate
- **Diverse Preference Optimization (DivPO)** — selects rare-but-high-quality responses as preferred over most-likely responses during training

The result: structurally, lexically, and semantically diverse ideation outputs across and within each thinking style.

---

## Architecture

```
User Prompt
    │
    ▼
Persona Gate  (MLP Classifier)
    │
    ├──► Contrarian Adapter          (LoRA)
    ├──► Cross-Domain Analogist Adapter  (LoRA)
    ├──► Systems Thinker Adapter     (LoRA)
    └──► Minimalist Adapter          (LoRA)
              │  top-k routing
              ▼
        Diverse Idea Outputs
```

| Component | Role |
|-----------|------|
| Base Model | Frozen pretrained + RLHF-aligned LLM (`Qwen/Qwen2.5-0.5B-Instruct`) |
| Persona Adapters | 4 separate LoRA adapters, one per thinking style |
| DivPO Training | Preference optimization that promotes rare-but-quality responses |

> DivPO is **training-time only**. Inference uses standard MixLoRA routing.

---

## Personas

| ID | Cognitive Style |
|----|----------------|
| `contrarian` | Challenges assumptions, surfaces minority dissent, reframes common viewpoints |
| `cross_domain_analogist` | Builds analogies across distant fields to transfer useful structures |
| `systems_thinker` | Explains causal relationships, feedback loops, trade-offs, and long-term effects |
| `minimalist` | Uses constraints and compression of non-essential parts to sharpen ideas |

---

## Datasets

This repo now has two dataset tracks:

- **Legacy proposal datasets**: CGA-CMV, ArXiv, StackExchange, and Gutenberg are still available as raw/source material.
- **Co-author SFT dataset**: the current product direction uses message-format examples that teach writing-partner behavior: `diagnosis`, `idea`, `rationale`, and `next_step`.

| Dataset | Acquisition | Persona Target |
|---------|------------|---------------|
| CGA-CMV — Reddit Conversations Gone Awry | ConvoKit | `contrarian` |
| ArXiv Abstracts 2021 | HuggingFace | `cross_domain_analogist`, `systems_thinker` |
| StackExchange Q&A | HuggingFace | `systems_thinker` |
| Project Gutenberg Corpus | HuggingFace | `minimalist` |
| Co-author seed examples | Curated local builder | all personas |

> The raw proposal datasets were not deleted. The important change is that they should no longer be used directly as assistant responses for fine-tuning. Use them as source material, then convert or curate them into co-author examples.

---

## Evaluation Metrics

| Metric | Measures |
|--------|---------|
| Self-BLEU | Inter-output similarity — lower = more diverse |
| Distinct-n (n=1,2) | Lexical variety within outputs |
| SBERT cosine similarity | Semantic diversity across outputs |
| Reward model score | Quality via prompt–output correlation |

**Baselines:** Base LLM (no fine-tuning) · Single LoRA (no persona split) · Prompt-Based Diversity (verbalized sampling)

---

## Project Structure

```
creativity/
├── src/mop_divpo/
│   ├── personas.py          # PersonaId type + descriptions
│   ├── schema.py            # Pydantic records (SFT, Candidate, DivPO pairs)
│   ├── config.py            # YAML config loader
│   ├── io.py                # JSONL read/write helpers
│   ├── cli.py               # Typer CLI entrypoint
│   ├── coauthor/
│   │   ├── schema.py        # WritingBrief, IdeaCard, CoAuthorRecord
│   │   ├── data.py          # Co-author dataset sources + seed builder
│   │   └── ideation.py      # Structured idea-card workflow
│   ├── data/
│   │   ├── sources.py       # Dataset registry
│   │   ├── acquire.py       # HuggingFace + ConvoKit acquisition
│   │   ├── prepare.py       # SFT sample generation
│   │   └── normalizers.py   # Per-source text normalizers
│   ├── divpo/
│   │   ├── scoring.py       # Quality + rarity scoring
│   │   └── pairs.py         # DivPO pair construction
│   ├── metrics/
│   │   ├── diversity.py     # Self-BLEU, Distinct-n
│   │   └── semantic.py      # SBERT cosine similarity
│   ├── routing/
│   │   └── gate.py          # Persona gate (MLP classifier)
│   ├── inference/
│   │   └── generate.py      # HuggingFace inference + dry-run fallback
│   ├── training/
│   │   └── commands.py      # Training command builders
│   └── eval/
│       └── evaluate.py      # Evaluation runner
├── tests/                   # pytest tests — no GPU required
├── configs/
│   └── prototype.yaml       # Prototype config (dry-run defaults)
├── notebooks/
│   └── train_coauthor_colab.ipynb
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone <repo-url>
cd creativity

pip install -e ".[dev]"
```

### Configuration

`configs/prototype.yaml` controls all pipeline parameters:

```yaml
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

divpo:
  candidates_per_prompt: 4
  min_quality: 0.35
  rarity_weight: 0.6
  quality_weight: 0.4

routing:
  top_k: 2
```

---

## Usage

### 1. Install And Verify

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest
```

### 2. Build Co-Author SFT Data

The current product target is a writing partner for pre-writing. For a quick local seed set, run:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli build-coauthor-data \
  --output-dir data/processed/coauthor_sft
```

For the larger referenced dataset, ingest external sources:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli ingest-coauthor-data \
  --output-dir data/processed/coauthor_sft \
  --limit-per-source 100
```

This creates:

```text
data/processed/coauthor_sft/contrarian.jsonl
data/processed/coauthor_sft/cross_domain_analogist.jsonl
data/processed/coauthor_sft/systems_thinker.jsonl
data/processed/coauthor_sft/minimalist.jsonl
```

Each row uses chat messages:

```json
{
  "messages": [
    {"role": "system", "content": "persona instruction"},
    {"role": "user", "content": "writer brief"},
    {"role": "assistant", "content": "structured idea card"}
  ]
}
```

The external ingestion currently pulls:

- `llm-aes/writing-prompts` for creative writer briefs
- `ibm-research/argument_quality_ranking_30k` for argumentative thesis and quality material
- `mc-ai/conversations-gone-awry-cmv` for contrarian/counter-position material

Every generated row includes a `references` field with the source dataset, row index, URL, and source-specific identifiers when available.

With `--limit-per-source 100`, the current generated dataset is:

```text
contrarian.jsonl                 360 rows
cross_domain_analogist.jsonl     262 rows
systems_thinker.jsonl            262 rows
minimalist.jsonl                 262 rows
total                           1146 rows
```

The local seed-only builder creates 64 rows per persona, 256 rows total. Use the external ingestion path for training experiments, then review/filter rows before making model-quality claims.

### 3. Try The Writing-Partner Workflow

```bash
PYTHONPATH=src python3 -m mop_divpo.cli ideate \
  --topic "AI in education" \
  --goal "Find a sharper thesis" \
  --persona contrarian \
  --persona minimalist
```

Output idea cards include:

- `diagnosis`
- `idea`
- `rationale`
- `next_step`

This is the first product-facing workflow. It gives the writer actionable pre-writing moves instead of generic paragraphs.

### 4. Legacy Dataset Commands

The original source dataset commands are still available for research data collection:

```bash
mop-divpo list-sources
mop-divpo prepare-data --output-dir data/processed/sft
mop-divpo collect-source arxiv_abstracts --limit 100
mop-divpo prepare-source arxiv_abstracts \
  --raw-path data/raw/arxiv_sample.jsonl \
  --output-dir data/processed/sft \
  --limit 50
```

Use these outputs as raw material for co-author task construction, not as direct final SFT targets.

---

## Co-Author Training

### Recommended Path: Google Colab

Use `notebooks/train_coauthor_colab.ipynb` for GPU training. The notebook trains one LoRA adapter per persona using the revised co-author message format.

Steps:

1. Build the local co-author data:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli build-coauthor-data \
  --output-dir data/processed/coauthor_sft
```

2. Upload or push `data/processed/coauthor_sft/*.jsonl` to HuggingFace.

The existing pusher preserves co-author fields:

```bash
PYTHONPATH=src python3 -m mop_divpo.training.push_data \
  --data-dir data/processed/coauthor_sft \
  --repo-id DasonTio/mop-divpo-coauthor-sft-data
```

3. Open `notebooks/train_coauthor_colab.ipynb` in Google Colab.
4. Set Runtime to GPU.
5. Update `HF_DATASET` and `HF_ADAPTER_PREFIX` in the notebook if needed.
6. Run all cells.

The notebook pushes adapters using this pattern:

```text
DasonTio/mop-divpo-coauthor-{persona}-sft
```

Example:

```text
DasonTio/mop-divpo-coauthor-contrarian-sft
```

### Local Training Option

If your machine has enough compute, train one persona locally:

```bash
PYTHONPATH=src python3 -m mop_divpo.training.train_sft \
  contrarian \
  --data-path data/processed/coauthor_sft/contrarian.jsonl \
  --output-dir outputs/adapters/coauthor_sft/contrarian \
  --max-steps 200
```

Repeat for:

```text
cross_domain_analogist
systems_thinker
minimalist
```

### Inference With Co-Author Adapters

The Python generation API supports adapter-backed inference:

```python
from mop_divpo.inference.generate import generate

outputs = generate(
    prompt="Topic: AI in education\nWriter goal: Find a sharper thesis",
    personas=["contrarian"],
    adapter_prefix="DasonTio/mop-divpo-coauthor",
    adapter_stage="sft",
)
```

This resolves to:

```text
DasonTio/mop-divpo-coauthor-contrarian-sft
```

---

## Legacy Training

The legacy proposal pipeline still supports SFT and DPO-style commands. Keep this path for comparison baselines, not as the main co-author product training path.

### Prerequisites

Create a dedicated virtual environment to avoid dependency conflicts with the base Anaconda env:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install trl peft
```

> **macOS / Apple Silicon:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before all training commands so ops without MPS kernels fall back to CPU.

### Step 1 — Collect training data

```bash
mop-divpo collect-source cga_cmv --limit 500
mop-divpo collect-source arxiv_abstracts --limit 500
mop-divpo collect-source stackexchange_qa --limit 500
mop-divpo collect-source gutenberg --limit 500
```

Outputs land in `data/processed/sft/{persona}.jsonl`.

### Step 2 — SFT per persona

Train one LoRA adapter per persona. Use the built-in CLI commands — these handle chat template formatting, MPS device detection, and LoRA config automatically:

```bash
# Train all 4 personas sequentially
PYTORCH_ENABLE_MPS_FALLBACK=1 mop-divpo train-all --max-steps 200

# Or train one persona at a time
PYTORCH_ENABLE_MPS_FALLBACK=1 mop-divpo train-sft contrarian --max-steps 200
PYTORCH_ENABLE_MPS_FALLBACK=1 mop-divpo train-sft cross_domain_analogist --max-steps 200
PYTORCH_ENABLE_MPS_FALLBACK=1 mop-divpo train-sft systems_thinker --max-steps 200
PYTORCH_ENABLE_MPS_FALLBACK=1 mop-divpo train-sft minimalist --max-steps 200
```

Adapters saved to `outputs/adapters/sft/{persona}/`.

### Step 3 — DivPO (DPO phase) per persona

After SFT, apply DivPO preference optimization using the chosen/rejected pairs:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_name_or_path outputs/adapters/sft/contrarian \
  --dataset_name data/processed/divpo/contrarian.jsonl \
  --output_dir outputs/adapters/divpo/contrarian \
  --use_peft true \
  --max_steps 100
```

Repeat for each persona (`cross_domain_analogist`, `systems_thinker`, `minimalist`).

Trained adapters land in `outputs/adapters/divpo/{persona}/`.

---

## Inference with Trained Model

The `infer` command loads `Qwen/Qwen2.5-0.5B-Instruct` from HuggingFace and injects each persona's cognitive style as a system prompt. For the writing-partner product workflow, prefer `ideate` or the adapter-backed Python `generate(...)` call shown above.

### CLI

```bash
# Real inference — loads Qwen from HF cache, generates with MPS on Apple Silicon
mop-divpo infer "What is the future of remote work?" \
  --persona contrarian \
  --persona minimalist

# All 4 personas
mop-divpo infer "How might cities be redesigned for resilience?"

# Control output length
mop-divpo infer "What breaks creative thinking?" \
  --persona systems_thinker \
  --max-new-tokens 512

# Dry-run (no model load, instant — for testing pipeline only)
mop-divpo infer "What is the future of remote work?" --dry-run
```

First call downloads and caches the model (~1 GB). Subsequent calls reuse the cache.

### Python API

```python
from mop_divpo.inference.generate import generate

outputs = generate(
    prompt="How might cities be redesigned for resilience?",
    personas=["contrarian", "systems_thinker"],
    max_new_tokens=256,
)
for out in outputs:
    print(f"[{out['persona']}] {out['response']}")
```

---

## Running Tests

```bash
pytest
```

The test suite is dry-run friendly and does not require a GPU.

---

## Key References

| Paper | Authors | Link |
|-------|---------|------|
| DivPO: Diverse Preference Optimization | Lanchantin et al. (2025) | [arXiv:2501.18101](https://arxiv.org/abs/2501.18101) |
| MixLoRA | Li et al. (2024) | [arXiv:2404.15159](https://arxiv.org/abs/2404.15159) |
| Generative Monoculture | Wu et al., ICLR 2025 | — |
| LoRA | Hu et al., ICLR 2022 | — |
| QLoRA | Dettmers et al. (2023) | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |

---

## Status

> **Prototype** — the project now includes a co-author SFT path, structured idea-card workflow, and Colab training notebook. The seed dataset must be expanded before serious model-quality claims.

**Done:**
- Dataset registry and acquisition layer
- SFT record schema with persona validation
- DivPO pair construction and scoring
- Diversity and semantic evaluation metrics
- Persona gate (MLP routing stub)
- Full CLI pipeline
- Real HuggingFace inference (Qwen 0.5B, MPS/CUDA/CPU device detection)
- Co-author schemas and seed dataset builder
- `ideate` writing-partner CLI workflow
- Colab notebook for co-author SFT adapter training

**In progress:**
- Expanding the co-author dataset beyond the current seed examples
- Per-persona co-author LoRA adapter training
- DivPO candidate generation and pair construction for co-author tasks
- Reward model integration for quality scoring
- Comparative baseline evaluation

---

<div align="center">

Made for **Pradita University — LLM & Generative AI**

</div>
