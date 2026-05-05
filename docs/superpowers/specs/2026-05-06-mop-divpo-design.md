# Mixture of Persona + DivPO Prototype Design

## Context

This project implements the proposal in `Dason Tiovino - LLM - Mixture of Persona & Diverse Performance Optimizer.pdf`: a pre-writing ideation system that combines Mixture of Persona (MoP) with Diverse Preference Optimization (DivPO). The goal is to increase both inter-persona diversity and intra-persona diversity without losing prompt-output relevance.

The implementation target is a local/course-project prototype that runs end to end with sampled datasets and a small open model. The code should be structured so the same pipeline can later scale to a CUDA GPU training setup.

## Goals

- Build a reproducible data-to-evaluation pipeline for MoP + DivPO.
- Train or prepare four persona LoRA adapters:
  - `contrarian`
  - `cross_domain_analogist`
  - `systems_thinker`
  - `minimalist`
- Construct DivPO preference pairs per persona by selecting rare-but-high-quality responses over common/weaker responses.
- Provide inference that routes a user prompt to top-k personas and returns multiple diverse ideas.
- Evaluate diversity and quality against project baselines.

## Non-Goals

- Full-scale training on all source datasets.
- A production MixLoRA kernel implementation.
- Human preference labeling.
- Deployment as a hosted API or web application.
- Training a reward model from scratch.

## Model Choice

Default base model: `Qwen/Qwen2.5-0.5B-Instruct`.

Rationale:

- Small enough for prototype experiments.
- Instruction-tuned and compatible with chat-style prompting.
- Permissive license.
- Easy to swap for a stronger model later through configuration.

The code should keep the base model configurable. Viable later replacements include Qwen3, SmolLM, or TinyLlama variants.

## Datasets

The prototype will use sampled, normalized subsets of the datasets named in the proposal.

| Persona | Source datasets | Training signal |
| --- | --- | --- |
| `contrarian` | CGA-CMV / debate-style Reddit data | challenge assumptions, alternative framing, minority dissent |
| `cross_domain_analogist` | ArXiv abstracts plus ideation prompts | analogies across fields and domains |
| `systems_thinker` | ArXiv abstracts and Stack Exchange Q&A | causal reasoning, feedback loops, trade-offs |
| `minimalist` | Gutenberg/SPGC-style text and concise-writing prompts | constraint-driven, sparse, essential ideas |

Each dataset adapter produces a common JSONL schema:

```json
{
  "id": "string",
  "persona": "contrarian",
  "source": "dataset-name",
  "prompt": "string",
  "response": "string",
  "metadata": {}
}
```

Sampling limits are configuration-driven so the prototype can run quickly and later scale.

## Architecture

The source tree will be organized around pipeline stages:

- `configs/`: model, dataset, training, DivPO, inference, and evaluation settings.
- `src/mop_divpo/data/`: dataset download/loading, normalization, sampling, and JSONL writing.
- `src/mop_divpo/training/`: LoRA SFT and DivPO/DPO training entrypoints.
- `src/mop_divpo/divpo/`: candidate generation, rarity scoring, quality scoring, and preference-pair construction.
- `src/mop_divpo/routing/`: lightweight persona gate.
- `src/mop_divpo/inference/`: adapter loading and top-k persona generation.
- `src/mop_divpo/eval/`: diversity and quality metrics.
- `scripts/`: runnable CLI wrappers for each pipeline step.
- `tests/`: focused tests for dataset normalization, DivPO pair selection, routing, and metrics.

The prototype will approximate MixLoRA behavior by loading one LoRA adapter per selected persona and generating in sequence. This preserves the core proposal behavior while avoiding a custom MoE kernel.

## Data Flow

1. `prepare_data`
   - Load or download each configured dataset.
   - Normalize examples into the shared JSONL schema.
   - Write per-persona SFT files under `data/processed/sft/`.

2. `train_sft`
   - Fine-tune one LoRA adapter per persona on its SFT file.
   - Store adapters under `outputs/adapters/sft/<persona>/`.

3. `generate_candidates`
   - For each persona prompt, generate multiple candidate responses using sampling.
   - Store candidates with log-probability proxies, text, and generation metadata.

4. `build_divpo_pairs`
   - Score candidate quality using prompt-response semantic relevance and simple fluency/length constraints.
   - Score rarity using lexical and semantic distance from other candidates for the same prompt/persona.
   - Select `chosen` as high-quality and rare.
   - Select `rejected` as common, lower-quality, or over-generic.
   - Write DPO-compatible JSONL under `data/processed/divpo/`.

5. `train_divpo`
   - Run DPO-style optimization per persona using the SFT adapter as the starting point.
   - Store final adapters under `outputs/adapters/divpo/<persona>/`.

6. `infer`
   - Score a new prompt against persona descriptions.
   - Select top-k personas.
   - Generate one or more ideas per selected persona.
   - Return persona-labeled outputs.

7. `evaluate`
   - Run configured baselines and the proposed method.
   - Compute Self-BLEU, Distinct-1/2, SBERT cosine diversity, and relevance reward.
   - Write tables and raw outputs under `outputs/evaluation/`.

## DivPO Design

The prototype implements DivPO as a preference-data construction strategy on top of DPO-compatible training.

For each prompt and persona:

- Generate `N` candidates with stochastic decoding.
- Embed candidates with a sentence-transformer model.
- Compute semantic rarity as average distance from neighboring candidates.
- Compute lexical rarity using distinct n-grams and overlap penalties.
- Compute quality using prompt-response embedding similarity plus configurable text-quality checks.
- Mark a candidate eligible if its quality is above a threshold.
- Choose the eligible candidate with the best combined rarity-quality score.
- Reject the candidate that is more common, more generic, or lower quality.

This directly reflects the proposal's core idea: prefer rare-but-high-quality outputs rather than merely the highest-probability output.

## Persona Gate

The first prototype uses a deterministic embedding-based gate:

- Define a short description for each persona.
- Embed the user prompt and persona descriptions.
- Rank personas by cosine similarity.
- Return top-k personas, with an option to force specific personas.

This is intentionally lightweight. A later version can train an MLP/classifier on prompt-persona labels without changing inference contracts.

## Baselines

Evaluation will compare:

- Base model without fine-tuning.
- Single LoRA adapter trained on all persona data merged.
- Prompt-based diversity using persona instructions without adapters.
- MoP SFT adapters without DivPO.
- MoP + DivPO final method.

## Metrics

Diversity:

- Self-BLEU: lower is better.
- Distinct-1 and Distinct-2: higher is better.
- SBERT cosine similarity across outputs: lower average similarity means higher semantic diversity.

Quality:

- Prompt-output relevance score using sentence embeddings.
- Optional reward-model score if a lightweight reward model is available locally.

The evaluation report should preserve raw generations so qualitative inspection is possible.

## Error Handling

- Dataset loaders fail with actionable messages when a dataset is unavailable or requires credentials.
- Pipeline steps validate required input files before running.
- Training commands support `--dry-run` and small sample limits.
- If optional GPU acceleration is unavailable, the CLI should explain which steps are expected to be slow.
- Each generated artifact includes config metadata for reproducibility.

## Testing Strategy

Tests focus on behavior that should remain stable even when models or datasets change:

- Dataset normalizers emit the shared schema.
- Persona labels are valid and consistently mapped.
- DivPO pair selection picks rare high-quality candidates over common candidates on synthetic examples.
- Persona gate returns expected top personas for representative prompts.
- Distinct-n and Self-BLEU calculations handle empty, repeated, and diverse outputs.
- CLI config loading validates required fields.

Implementation should follow test-driven development: add failing tests before production code for each module.

## Acceptance Criteria

- A fresh checkout can install dependencies from a documented command.
- Sample data preparation produces one JSONL file per persona.
- At least one LoRA training command can run in dry-run or tiny-sample mode.
- DivPO pair construction produces DPO-compatible `prompt`, `chosen`, and `rejected` records.
- Inference can produce persona-labeled ideas for a user prompt.
- Evaluation produces a metrics JSON/CSV comparing at least base, prompt-diversity, MoP SFT, and MoP + DivPO modes.
- The README documents the full start-to-finish workflow.

## Known Constraints

- Local training quality will be limited by small model size and sampled data.
- Some proposal datasets may require alternative public mirrors or manual download.
- The prototype's embedding-based gate is not equivalent to a trained MLP router.
- Sequential adapter generation approximates MixLoRA routing but does not implement simultaneous expert routing.

## Upgrade Path

- Replace the base model with a larger Qwen, Llama, or Mistral-family model.
- Add QLoRA and CUDA-specific training presets.
- Train a learned persona gate from prompt-persona labels.
- Integrate a true MixLoRA/MoE adapter implementation.
- Add human or LLM-judge quality scoring for preference-pair construction.
- Expand dataset scale and run repeated-seed evaluation.
