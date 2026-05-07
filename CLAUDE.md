# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (Pradita University, LLM & Generative AI course) implementing **Mixture of Persona + Diverse Preference Optimization** for pre-writing ideation. Goal: increase both inter-persona and intra-persona output diversity without sacrificing quality.

## System Architecture

Three components combined via MixLoRA (LoRA-based Mixture of Experts):

1. **Base Model** — pretrained + RLHF-aligned LLM (frozen weights)
2. **Persona Adapters (MoP)** — 4 separate LoRA adapters, each fine-tuned on a distinct thinking style:
   - `contrarian` — challenges assumptions, minority dissent, divergent thinking
   - `cross_domain_analogist` — analogical reasoning across fields
   - `systems_thinker` — causal/systemic reasoning, long-term perspective
   - `minimalist` — constraint-driven creativity
3. **DivPO training** — Diverse Preference Optimization applied per-persona during fine-tuning to increase intra-persona diversity. DivPO selects rare-but-high-quality responses as preferred over most-likely responses.

**Inference flow:** user prompt → Persona Gate (MLP/Classifier) → top-k persona routing → parallel LoRA adapter outputs → multiple diverse ideas

DivPO is training-time only; inference uses standard MixLoRA routing.

## Datasets

| Dataset | Persona target |
|---------|---------------|
| CGA-CMV (Reddit Conversation Gone Awry) | Contrarian |
| ArXiv Abstracts | Cross-Domain Analogist / Systems Thinker |
| Project Gutenberg Corpus (SPGC) | Minimalist |
| Stack Exchange Q&A | Systems Thinker |

## Evaluation

- **Self-BLEU** — inter-output similarity (lower = more diverse)
- **Distinct-n** (n=1,2) — lexical variety within outputs
- **SBERT cosine similarity** — semantic diversity across outputs
- **Reward model** — quality score (correlation between prompt and output)

**Baselines:** Base LLM (no fine-tuning), Single LoRA (no persona split), Prompt-Based Diversity (verbalized sampling)

## Key Papers

- DivPO: Lanchantin et al., ArXiv:2501.18101 (2025)
- MixLoRA: Li et al., ArXiv:2404.15159 (2024)
- Generative Monoculture: Wu et al., ICLR 2025
- LoRA: Hu et al., ICLR 2022
- QLoRA: Dettmers et al., ArXiv:2305.14314 (2023)
