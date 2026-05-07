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

| Dataset | Acquisition | Persona Target |
|---------|------------|---------------|
| CGA-CMV — Reddit Conversations Gone Awry | ConvoKit | `contrarian` |
| ArXiv Abstracts 2021 | HuggingFace | `cross_domain_analogist`, `systems_thinker` |
| StackExchange Q&A | HuggingFace | `systems_thinker` |
| Project Gutenberg Corpus | HuggingFace | `minimalist` |

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
├── tests/                   # 40 pytest tests — no GPU required
├── configs/
│   └── prototype.yaml       # Prototype config (dry-run defaults)
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

### CLI

```bash
# List available dataset sources
mop-divpo list-sources

# Prepare sample SFT training files
mop-divpo prepare-data --output-dir data/processed/sft

# Collect + prepare a source from HuggingFace
mop-divpo collect-source arxiv_abstracts --limit 100

# Prepare from a local JSONL export
mop-divpo prepare-source arxiv_abstracts \
  --raw-path data/raw/arxiv_sample.jsonl \
  --output-dir data/processed/sft \
  --limit 50
```

---

## Training

Training runs in two phases per persona: SFT (supervised fine-tuning) then DPO (DivPO preference optimization).

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

The `infer` command loads `Qwen/Qwen2.5-0.5B-Instruct` from HuggingFace and injects each persona's cognitive style as a system prompt. No trained adapter loading is required for baseline inference — add adapter support via PEFT once adapters are trained.

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

40 tests, all dry-run friendly — no GPU required, completes in under one second.

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

> **Prototype** — data acquisition, evaluation scaffolding, and base model inference complete. LoRA adapter training requires a dedicated venv and sufficient compute (MPS on Apple Silicon or CUDA on GPU).

**Done:**
- Dataset registry and acquisition layer
- SFT record schema with persona validation
- DivPO pair construction and scoring
- Diversity and semantic evaluation metrics
- Persona gate (MLP routing stub)
- Full CLI pipeline
- Real HuggingFace inference (Qwen 0.5B, MPS/CUDA/CPU device detection)

**In progress:**
- Per-persona LoRA adapter training (SFT + DivPO phases)
- Adapter loading at inference time via PEFT
- Reward model integration for quality scoring
- Comparative baseline evaluation

---

<div align="center">

Made for **Pradita University — LLM & Generative AI**

</div>
