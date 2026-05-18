# MoP + DivPO + Co-Author: Research Report

**Project:** Mixture of Persona + Diverse Preference Optimization for Pre-Writing Ideation  
**Institution:** Pradita University — LLM & Generative AI Course  
**Base Model:** Qwen/Qwen2.5-0.5B-Instruct  

---

## Table of Contents

1. [Datasets](#1-datasets)
2. [Pipeline](#2-pipeline)
   - 2.1 [MoP — Mixture of Persona](#21-mop--mixture-of-persona)
   - 2.2 [DivPO — Diverse Preference Optimization](#22-divpo--diverse-preference-optimization)
   - 2.3 [Co-Author Extension](#23-co-author-extension)
   - 2.4 [Base Model Choice](#24-base-model-choice)
   - 2.5 [Training Procedure](#25-training-procedure)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Codebase File Guide](#4-codebase-file-guide)

---

## 1. Datasets

The project uses four distinct datasets, each chosen to align with a specific persona's cognitive style. No single general dataset is used — every data source is selected because its natural language patterns already resemble the target thinking mode.

---

### 1.1 CGA-CMV (Conversations Gone Awry — Change My View)

| Field | Detail |
|---|---|
| **Source** | Cornell ConvoKit / HuggingFace mirror `mc-ai/conversations-gone-awry-cmv` |
| **Acquisition** | HuggingFace streaming download (`convokit` or `huggingface` method) |
| **Target persona** | `contrarian` |
| **Official URL** | https://convokit.cornell.edu/documentation/awry_cmv.html |

**What it is.** A corpus of Reddit r/ChangeMyView conversations where users present a thesis and invite others to challenge it. Each record contains a raw conversation thread with a claim (the opening post) and responses that argue against the original position.

**Why this dataset.** The contrarian persona needs to produce writing that identifies hidden assumptions and argues for minority positions. CMV conversations are naturally full of this behavior — every reply is an attempt to surface a flaw or limitation in the original claim. The dataset teaches the model the rhetorical pattern of: "your claim assumes X, but X is not inevitable because Y."

**How it is used.** The `normalize_reddit_record()` function in `data/normalizers.py` extracts the thread title as the assumption to challenge, and the first reply turns as the response. The prompt becomes: `"Challenge this assumption for pre-writing ideation: {title}"`. The response is the body text of the challenge.

---

### 1.2 ArXiv Abstracts

| Field | Detail |
|---|---|
| **Source** | HuggingFace `gfissore/arxiv-abstracts-2021` |
| **Acquisition** | HuggingFace streaming download |
| **Target personas** | `cross_domain_analogist`, `systems_thinker` |
| **Official URL** | https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021 |

**What it is.** A large collection of academic paper titles and abstracts from ArXiv, spanning many scientific fields including physics, biology, computer science, mathematics, and engineering.

**Why this dataset.** Two different personas benefit from academic abstracts but for different reasons:
- **Cross-Domain Analogist**: ArXiv covers many fields, so a paper from biology can be repurposed as an analogy for an essay about logistics or politics. The abstract gives the model a compact structural description of a domain mechanism.
- **Systems Thinker**: Academic papers, especially in complex systems, social science, and biology, are full of causal reasoning, feedback loops, and systemic explanations. The model learns the language of causes and constraints from these abstracts.

**How it is used.** The `normalize_arxiv_record()` function creates two separate SFT records from each abstract — one for each persona — with different prompts:
- Cross-domain: `"Create a cross-domain analogy from this research idea: {title}"`
- Systems thinker: `"Explain the causal system and feedback loops behind this research idea: {title}"`

The response for both is the paper's abstract.

---

### 1.3 Stack Exchange Q&A

| Field | Detail |
|---|---|
| **Source** | HuggingFace `PrimeIntellect/stackexchange-question-answering` |
| **Acquisition** | HuggingFace streaming download |
| **Target persona** | `systems_thinker` |
| **Official URL** | https://huggingface.co/datasets/PrimeIntellect/stackexchange-question-answering |

**What it is.** A large collection of Stack Exchange questions and their accepted or top-voted answers. Stack Exchange hosts technical Q&A communities across software, hardware, science, philosophy, and many applied domains.

**Why this dataset.** Systems thinking means understanding causes, dependencies, and feedback. Stack Exchange answers routinely explain "why does X happen when Y" — which is a natural expression of causal reasoning. The answers are written for non-expert audiences, which trains the persona to make systemic reasoning accessible rather than jargon-heavy.

**How it is used.** The `normalize_stackexchange_record()` function wraps the question as: `"Analyze this question through causes, constraints, and feedback loops: {question}"`. The accepted answer becomes the response. This teaches the model to reframe a question as a system before answering it.

---

### 1.4 Project Gutenberg (SPGC)

| Field | Detail |
|---|---|
| **Source** | HuggingFace `zkeown/gutenberg-corpus` (paragraphs config) |
| **Acquisition** | HuggingFace streaming download |
| **Target persona** | `minimalist` |
| **Official URL** | https://huggingface.co/datasets/zkeown/gutenberg-corpus |

**What it is.** A corpus of public-domain literary texts from Project Gutenberg, organized at the paragraph level. These are classic literary works where prose has been deliberately pared down over centuries of editing.

**Why this dataset.** The minimalist persona works through constraint and removal. Literary prose from Gutenberg — especially 19th and early 20th century texts — demonstrates the principle that a single concrete image or scene can carry a full argument. Writers like Hemingway, Chekhov, and Thoreau achieve weight through omission. The model learns what "doing more with less" looks like at the sentence level.

**How it is used.** The `normalize_gutenberg_record()` function uses a fixed prompt: `"Create a constraint-driven minimalist writing idea from this passage."` Paragraphs shorter than 80 characters are filtered out to keep training on substantial examples. The text itself is the response.

---

### 1.5 Co-Author Extension Datasets

The co-author training phase uses three additional datasets on top of the four above, specifically to build the writing-partner behavior (structured IdeaCards rather than raw generation):

| Dataset | HuggingFace ID | Role |
|---|---|---|
| **Stanford CoAuthor** | `stanford-coauthor` | Interaction traces for human-AI collaborative writing. Design reference for turn structure. |
| **Writing Prompts** | `llm-aes/writing-prompts` | Creative and argumentative prompts used as `WritingBrief` topic seeds. |
| **IBM Argument Quality** | `ibm-research/argument_quality_ranking_30k` | Argument topic + quality score pairs. Provides argumentative task seeds and quality signal reference for all 4 personas. |

These three datasets are used exclusively in the `coauthor/data.py` pipeline to generate structured (brief → IdeaCard) training rows, not raw `prompt/response` pairs.

---

## 2. Pipeline

The system combines three research contributions into one end-to-end architecture: **MoP** for role specialization, **DivPO** for intra-persona diversity, and **Co-Author** for structured conversational pre-writing. The diagram below shows both training time and inference time:

```
TRAINING TIME
─────────────────────────────────────────────────────────────────────────────
Raw Datasets → Normalizers → SFT JSONL files per persona
                                      │
                                      ▼
                          Base Model (Qwen2.5-0.5B, frozen)
                          + QLoRA config (r=16, α=32)
                                      │
                              SFT Training × 4 personas
                                      │
                          ┌───────────┴───────────┐
                          │   DivPO Pair Building  │
                          │  (rarity + quality)    │
                          └───────────┬───────────┘
                                      │
                              4 LoRA Adapters pushed
                              to HuggingFace Hub

INFERENCE TIME
─────────────────────────────────────────────────────────────────────────────
User Prompt ──→ Keyword Persona Gate ──→ top-k persona IDs
                                                │
                         ┌──────────────────────┤ (optional)
                         ▼                      │
               CorpusRetriever             Adapter load
               top-3 prior art             per persona
               → PRIOR ART block                │
                         └──────────────────────▼
                              Parallel generation per persona
                                      │
                              Output Parser (idea / reasoning / unexpected_angle)
                                      │
                              Diverse idea set

CO-AUTHOR MODE (multi-turn)
─────────────────────────────────────────────────────────────────────────────
WritingBrief (topic, goal, audience, constraints)
      │
      ▼
PersonaSession (one persona, message history)
      │ turn 1: inject PRIOR ART into system prompt
      │ turn 2+: use full message history (no re-injection)
      ▼
generate_turn() → structured IdeaCard response
      │
      ▼
User continues conversation → session accumulates context
```

---

### 2.1 MoP — Mixture of Persona

**Core idea.** Instead of training one model to be diverse, train four separate LoRA adapters, each specializing in a distinct thinking mode. At inference, route to one or more adapters based on the prompt's content.

**Why separate adapters instead of a single model.** A single model fine-tuned with multiple styles tends to average them. Separate LoRA adapters allow each persona to develop genuinely different weight patterns while sharing the same frozen base. This is the key contribution of the MoP (Mixture of Persona) approach — it pushes inter-persona diversity into the model weights rather than relying on prompt engineering alone.

**The four personas:**

| Persona | Thinking style | Dataset origin |
|---|---|---|
| `contrarian` | Surfaces hidden assumptions, argues minority positions | CGA-CMV Reddit debates |
| `cross_domain_analogist` | Finds structural isomorphisms between distant domains | ArXiv abstracts |
| `systems_thinker` | Maps causes, feedback loops, leverage points | ArXiv + Stack Exchange |
| `minimalist` | Constraint-driven creativity, removal as generative force | Project Gutenberg |

**Persona Gate.** At inference, the `KeywordPersonaGate` in `routing/gate.py` scores each persona against the user prompt using per-persona keyword lists. Keywords like `"feedback"` and `"loop"` boost `systems_thinker`; `"analogy"` and `"biology"` boost `cross_domain_analogist`. The gate returns the top-k scoring personas. This is a lightweight rule-based router, not a learned classifier — intentional for prototype stage where training a separate gate model would require labeled routing data.

---

### 2.2 DivPO — Diverse Preference Optimization

**Core idea.** DivPO (Lanchantin et al., 2025, ArXiv:2501.18101) modifies standard DPO to prefer responses that are both high quality and rare relative to the model's typical output distribution.

**Standard DPO** selects `chosen` = better quality, `rejected` = worse quality, and trains the model to increase the probability of the chosen response. The problem: it does not reward diversity. The model learns to produce the best average response, which tends to be the most common type of response.

**DivPO** adds a rarity dimension. For each prompt, multiple candidate responses are generated. Each candidate receives two scores:
1. **Quality score** — how relevant and coherent the response is (currently proxied by normalized distinct-1 in the prototype)
2. **Rarity score** — how lexically distinct the response is from its peer candidates for the same prompt

The combined score is: `quality × w_q + rarity × w_r` where `w_r > w_q` by default (config: `quality_weight: 0.4`, `rarity_weight: 0.6`). This bias toward rarity means the system prefers a response that says something different over a response that says the most expected thing, as long as quality stays above the minimum threshold.

**Pair selection in `divpo/pairs.py`:**
- `chosen` = candidate with highest combined score (rare AND good)
- `rejected` = candidate with lowest combined score (common OR low quality)

**Why this matters for ideation.** In pre-writing, the most common response is often the least useful — it is the idea the writer already had. DivPO trains each persona adapter to prefer the output that occupies an unusual position in idea space while remaining on-topic.

---

### 2.3 Co-Author Extension

**What changed.** The original MoP+DivPO system was a one-shot ideation tool: one prompt → multiple persona responses. The co-author extension transforms this into a multi-turn writing partner.

**Key modifications:**

| Component | Original behavior | Co-author behavior |
|---|---|---|
| Output format | Free-form text | Structured `IdeaCard`: `operation`, `diagnosis`, `idea`, `rationale`, `next_step` |
| Interaction | Single turn | Multi-turn `PersonaSession` with message history |
| Training data | Raw dataset examples | Curated `WritingBrief → IdeaCard` pairs |
| System prompt | Generic persona description | Conversational persona with explicit 4-step METHOD |
| Prior art injection | Every call | First turn only; subsequent turns use accumulated history |

**IdeaCard structure.** Each co-author response is structured as:
```json
{
  "operation": "challenge_thesis | transfer_analogy | map_system | distill_core | synthesize_angles",
  "diagnosis": "what is wrong or missing in the writer's current framing",
  "idea": "the concrete direction the writer should take",
  "rationale": "why this direction is better than the obvious one",
  "next_step": "one immediate concrete action the writer can take right now"
}
```

This structure preserves writer agency. The model is not writing the draft — it is diagnosing the problem and giving one specific direction, not a finished paragraph.

**Training data construction.** `coauthor/data.py` builds training rows in two stages:
1. **Seed records** — 4 hand-curated `CoAuthorRecord` examples per persona, expanded with 8 task variants (thesis, angle, outline, evidence, audience, revision, comparison, next_move) = ~32 high-quality examples.
2. **External records** — records pulled from Writing Prompts, IBM Argument Quality, and CGA-CMV, converted into `(WritingBrief, IdeaCard)` pairs programmatically for all 4 personas.

---

### 2.4 Base Model Choice

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`

**Why Qwen2.5-0.5B:**

| Factor | Reasoning |
|---|---|
| **Size (0.5B)** | Small enough to train 4 adapters sequentially on a single T4 GPU (16GB VRAM) without quantization. Large enough to follow instruction format and produce coherent structured output. |
| **Instruction-tuned** | The `-Instruct` variant is already aligned with chat format and system prompts. Starting from an instruction-tuned model means SFT only needs to shift the persona style, not teach the model how to follow instructions from scratch. |
| **Chat template support** | Qwen2.5 has a built-in `apply_chat_template()` method compatible with `system / user / assistant` turn format, which is required for both SFT formatting and inference. |
| **HuggingFace availability** | Full PEFT/LoRA + `transformers` + `trl` compatibility without custom patching. |
| **Practical constraint** | Research prototype on limited compute. A 7B or 13B model would require either expensive cloud compute or multi-GPU setups unavailable to a course project. 0.5B allows the full research loop (train → evaluate → iterate) in a single Colab session. |

**Limitation acknowledged.** At 0.5B parameters, the model has limited capacity for complex analogical reasoning and nuanced system mapping. The personas primarily shift surface-level style and vocabulary rather than deep reasoning capability. A production-scale implementation would use a larger base (7B+) with the same LoRA + DivPO methodology.

---

### 2.5 Training Procedure

**Stage 1 — SFT (Supervised Fine-Tuning)**

Each persona adapter is trained independently using `training/train_sft.py`:

| Hyperparameter | Value | Reason |
|---|---|---|
| LoRA rank `r` | 16 | Balances adapter expressiveness with parameter count. r=16 adds ~2M params per adapter on 0.5B base. |
| LoRA alpha | 32 | Scaling factor = 2×r, standard practice for stable training |
| LoRA dropout | 0.05 | Light regularization to prevent persona overfitting on small datasets |
| Target modules | `q_proj`, `v_proj` | Attention projections most responsible for stylistic behavior |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning; higher than full fine-tuning because only adapter weights are updated |
| Batch size | 4 (per device) + 4 gradient accumulation = effective 8 | Fits T4 VRAM; accumulation gives larger effective batch without memory cost |
| Max steps | 500 (recommended) | ~4 epochs over 300-360 rows at effective batch 8 |
| Warmup steps | 50 | Prevents loss spike on small datasets in early training |
| Precision | fp16 (CUDA) | Halves memory usage; T4 supports fp16 natively |

**Stage 2 — DivPO Pair Construction**

After SFT, the trained adapter generates N candidate responses per prompt. `divpo/scoring.py` computes rarity scores (lexical distinctness relative to peers). `divpo/pairs.py` selects the best and worst candidate as the DPO `chosen`/`rejected` pair. These pairs feed a standard DPO training loop using the same adapter as starting point.

**Stage 3 — Co-Author SFT**

Separate SFT run using `coauthor/data.py` training rows (format: `messages[]` with system + user + assistant turns). The co-author SFT targets the structured IdeaCard output format specifically, training the model to produce `diagnosis`, `idea`, `rationale`, `next_step` rather than free-form text.

---

## 3. Evaluation Metrics

The evaluation framework in `eval/evaluate.py` uses five metrics across two dimensions: **diversity** (are the outputs different from each other?) and **quality** (are the outputs relevant to the prompt?).

---

### 3.1 Distinct-1 and Distinct-2

**What it measures.** The fraction of unique unigrams (Distinct-1) or unique bigrams (Distinct-2) across all outputs. Implemented in `metrics/diversity.py`.

```
Distinct-n = |unique n-grams| / |total n-grams|
```

**Range.** 0 (all outputs use the same words) to 1 (no repeated n-grams across any two outputs).

**Why we use it.** This is the most established lexical diversity metric in text generation research. It measures surface-level vocabulary spread. A model that always generates "the hidden assumption is that technology improves outcomes" for every contrarian prompt would score near 0. **Higher is better for diversity.**

**Limitation.** Lexical diversity does not guarantee semantic diversity. Two outputs can use entirely different words while making the same conceptual point. This is why Distinct-n is always paired with SBERT.

---

### 3.2 Self-BLEU

**What it measures.** The average pairwise BLEU-like overlap between all outputs. For each output, we compute its overlap score against all other outputs treated as references, then average across all outputs. Implemented in `metrics/diversity.py`.

```
Self-BLEU = mean over all outputs of: (overlap tokens with all other outputs) / (total tokens in output)
```

**Range.** 0 (outputs share no tokens with each other) to 1 (all outputs are identical).

**Why we use it.** Self-BLEU was introduced specifically for diversity evaluation in text generation (Zhu et al., 2018, "Texygen"). It is the standard benchmark metric for measuring whether a model produces varied outputs or collapses to the most frequent response. **Lower is better** — lower Self-BLEU means outputs are less similar to each other.

**Relationship to DivPO.** DivPO's rarity score in `divpo/scoring.py` is derived from the same token-overlap principle as Self-BLEU. The training objective directly optimizes for the behavior this metric measures.

---

### 3.3 SBERT Pairwise Cosine Similarity

**What it measures.** The average cosine similarity between all pairs of output embeddings, using Sentence-BERT (`all-MiniLM-L6-v2`). Implemented in `metrics/semantic.py`.

```
SBERT Pairwise Mean = mean of upper triangle of cosine similarity matrix
```

**Range.** -1 to 1 in theory; in practice 0 (semantically unrelated) to 1 (identical meaning).

**Why we use it.** This catches semantic redundancy that Distinct-n misses. Two outputs saying "question the premise" and "interrogate the assumption" score differently on Distinct-n but nearly identically on SBERT. For ideation, we care about semantic diversity — whether the ideas occupy different conceptual positions — not just lexical variety. **Lower is better** — lower SBERT pairwise mean means ideas are semantically further apart.

**Why `all-MiniLM-L6-v2` specifically.** It is the standard lightweight sentence embedding model, small enough to run during evaluation without requiring a separate large model. It is already loaded as part of `CorpusRetriever`, so there is no additional dependency.

---

### 3.4 Corpus Novelty Score

**What it measures.** For each output, compute its maximum cosine similarity to any document in the training corpus (`CorpusRetriever`). The novelty score is `1 - max_similarity`. Two aggregates are reported: mean and minimum novelty across all outputs.

```
corpus_novelty(output) = 1 - max(cosine_similarity(output, doc) for doc in corpus)
```

**Range.** 0 (output is nearly identical to a training example) to 1 (output is unrelated to anything in training data).

**Why we use it.** This metric addresses a specific failure mode: a model that memorizes training examples and reproduces them. Corpus novelty checks whether outputs are genuinely new relative to what the model was trained on. This is especially important for the ideation use case — repeating a training example is the opposite of generating a novel idea. **Higher is better.** The minimum novelty score (`corpus_novelty_min`) is the most critical value: if any output exactly copies a training example, this will be near 0.

**Connection to PRIOR ART injection.** The `CorpusRetriever` at inference time injects the top-3 most similar training examples into the system prompt and instructs the model to produce output distinct from them. This is a direct intervention to keep corpus novelty high.

---

### 3.5 Quality Proxy (Prompt–Idea Cosine Similarity)

**What it measures.** The cosine similarity between the embedding of the user's prompt and the embedding of each generated idea. Averaged across all outputs.

```
quality_proxy = mean(cosine_similarity(embed(prompt), embed(idea)) for each output)
```

**Range.** 0 to 1; low values mean the idea is unrelated to the prompt.

**Why we use it.** Diversity without relevance is noise. A model could achieve perfect Distinct-1 and zero Self-BLEU by generating random text — but the outputs would be useless for writing. The quality proxy ensures that diverse outputs are still on-topic. **Higher is better** — we want ideas that are simultaneously varied from each other (low SBERT pairwise) AND related to the prompt (high quality proxy).

**Limitation.** This is a proxy, not a reward model. Cosine similarity in sentence embedding space is a rough semantic overlap score, not a judgment of whether the idea is genuinely useful for the writer. A proper evaluation would use human judgments or a trained reward model. For the prototype stage, the proxy is sufficient to detect complete topic drift.

---

### 3.6 Summary Table

| Metric | Implemented in | Better = | What it catches |
|---|---|---|---|
| Distinct-1 | `metrics/diversity.py` | Higher | Low vocabulary variety |
| Distinct-2 | `metrics/diversity.py` | Higher | Low phrase variety |
| Self-BLEU | `metrics/diversity.py` | Lower | Token-level output similarity |
| SBERT Pairwise Mean | `metrics/semantic.py` | Lower | Semantic redundancy across outputs |
| Corpus Novelty Mean | `eval/evaluate.py` | Higher | Training example memorization |
| Corpus Novelty Min | `eval/evaluate.py` | Higher | Worst-case memorization |
| Quality Proxy | `eval/evaluate.py` | Higher | Off-topic outputs |

---

## 4. Codebase File Guide

This section explains every source file for readers unfamiliar with the codebase.

---

### Configuration

| File | What it does |
|---|---|
| `configs/prototype.yaml` | Central config file. Controls base model path, data directories, training hyperparameters (max_steps, learning_rate), DivPO weights (quality_weight, rarity_weight), routing top_k, and retrieval settings. `dry_run: true` skips real model loading for fast pipeline testing. |
| `src/mop_divpo/config.py` | Loads and validates `prototype.yaml`. Raises errors if required sections are missing. One function: `load_config(path)`. |

---

### Data Layer

| File | What it does |
|---|---|
| `data/sources.py` | Registry of all 4 dataset sources. Each `DatasetSource` record stores the name, HuggingFace dataset ID, acquisition method, target personas, and URL. `list_dataset_sources()` returns all sources as a list. |
| `data/acquire.py` | Downloads raw data from HuggingFace using streaming (no full dataset download). `collect_and_prepare_source()` is the main entry point — it downloads and normalizes in one step. `collect_source_to_raw()` downloads only. |
| `data/normalizers.py` | One normalizer function per dataset. Each takes a raw dict row and an index, returns an `SFTRecord`. The normalizer decides what becomes the `prompt` and what becomes the `response`. |
| `data/prepare.py` | Creates the per-persona JSONL files in `data/processed/sft/`. `write_sample_sft_files()` writes small example files for quick testing. `sample_sft_records()` returns one hardcoded example per persona. |
| `schema.py` | Three Pydantic data models used throughout the project: `SFTRecord` (one training row), `CandidateRecord` (adds quality and rarity scores for DivPO), `DivPOPairRecord` (the chosen/rejected pair for DPO training). |

---

### Personas

| File | What it does |
|---|---|
| `personas.py` | The single source of truth for all persona definitions. `PERSONA_IDS` is the tuple of valid names. `PERSONA_DESCRIPTIONS` maps each ID to its full system prompt (ROLE + 4-step METHOD). `describe_persona()` and `validate_persona()` are utility functions used everywhere. |

---

### DivPO

| File | What it does |
|---|---|
| `divpo/scoring.py` | Two functions. `lexical_rarity_scores(responses)` takes a list of candidate responses and returns a normalized score for each — high score = rare vocabulary relative to peers, low score = common/repetitive. `combined_divpo_score(quality, rarity, w_q, w_r)` computes the weighted sum used for pair selection. |
| `divpo/pairs.py` | `select_divpo_pair(candidates, ...)` takes a list of `CandidateRecord` objects (all same prompt, same persona), finds the highest-scoring candidate as `chosen` and lowest-scoring as `rejected`, returns a `DivPOPairRecord`. This is the core algorithmic step of DivPO. |

---

### Routing

| File | What it does |
|---|---|
| `routing/gate.py` | `KeywordPersonaGate` assigns a score to each persona for a given prompt by counting keyword matches. Keywords are hand-defined per persona (e.g., "feedback", "loop" → systems_thinker). `rank(prompt, top_k)` returns the top-k personas sorted by score. If `force_personas` is passed, those personas are returned with score 1.0 regardless of keywords. |

---

### Retrieval

| File | What it does |
|---|---|
| `retrieval/corpus.py` | `CorpusRetriever` is a vector similarity search index over the training corpus. Built from `data/processed/sft/*.jsonl` by embedding all prompts with `all-MiniLM-L6-v2`. At inference, `retrieve(query, k=3)` embeds the user prompt and returns the k most similar training examples by cosine similarity. `corpus_novelty_score(text)` returns `1 - max_similarity` for a generated output. Used both to inject anti-duplication context into the system prompt and to compute corpus novelty evaluation scores. |

---

### Training

| File | What it does |
|---|---|
| `training/train_sft.py` | Main SFT training script. Loads base model, wraps with QLoRA (`LoraConfig`), formats training records using `apply_chat_template`, runs `SFTTrainer`, saves adapter locally. Can optionally push adapter to HuggingFace Hub. Called once per persona. |
| `training/push_data.py` | Uploads processed JSONL files to HuggingFace Hub as dataset repos under the `DasonTio/mop-divpo-*` prefix. |
| `training/commands.py` | CLI helper utilities for training commands. |

---

### Inference

| File | What it does |
|---|---|
| `inference/generate.py` | `generate(prompt, personas, ...)` is the one-shot multi-persona inference function. Loops over each requested persona, loads the base model or PEFT adapter, builds the message list (with optional PRIOR ART block from the retriever), samples from the model, and parses the output. `_load()` caches the base model; `_load_with_adapter(adapter_id)` caches adapter models separately by ID string. |
| `inference/output_parser.py` | `parse_structured_output(response, persona, prompt)` attempts to extract `idea`, `reasoning`, and `unexpected_angle` fields from the raw model output using regex and heuristics. Returns a structured object with a `parse_success` flag indicating whether parsing succeeded. |
| `inference/session.py` | Multi-turn chat infrastructure. `PersonaSession` is a dataclass holding `persona` (fixed for the session) and `history` (list of message dicts that grows with each turn). `generate_turn(session, user_message, ...)` handles one full turn: builds messages, calls the model, appends both the user message and response to history, returns updated session. PRIOR ART is injected only on turn 0. `dry_run_generate_turn()` is an identical-logic stub that returns a fixed response string without loading the model — used for pipeline testing. |

---

### Co-Author Layer

| File | What it does |
|---|---|
| `coauthor/schema.py` | Three Pydantic models. `WritingBrief` captures the writer's situation: topic, goal, audience, constraints, notes. `IdeaCard` is the structured response: operation type, diagnosis, idea, rationale, next_step. `CoAuthorRecord` pairs a brief with a card and adds source tracking and the `messages[]` list needed for SFT training. |
| `coauthor/ideation.py` | `build_idea_cards(topic, goal, ...)` generates one `IdeaCard` per persona from a `WritingBrief` using hardcoded templates. Used in the `ideate` CLI command. Not model-dependent — these are rule-generated ideas used as seed examples before the model is trained. |
| `coauthor/data.py` | The largest data construction file. Builds the co-author SFT training set from two sources: (1) `build_seed_coauthor_records()` generates curated hand-written examples with 8 task variants each; (2) `collect_external_coauthor_records()` pulls from Writing Prompts, IBM Argument Quality, and CGA-CMV and converts them into `(WritingBrief, IdeaCard)` pairs for all 4 personas. `write_coauthor_sft_files()` writes everything to `data/processed/coauthor_sft/{persona}.jsonl`. |

---

### Metrics and Evaluation

| File | What it does |
|---|---|
| `metrics/diversity.py` | Two metrics: `distinct_n(outputs, n)` computes the ratio of unique n-grams across all outputs. `self_bleu(outputs)` computes mean pairwise token overlap — a lower value means outputs are more different from each other. |
| `metrics/semantic.py` | `cosine_similarity_matrix(embeddings)` computes all pairwise cosine similarities from a matrix of sentence embeddings. `average_pairwise_cosine(embeddings)` returns the mean of the upper triangle — the average semantic similarity between all output pairs. |
| `eval/evaluate.py` | `evaluate_outputs(outputs, retriever)` runs the full evaluation suite on a list of model outputs. Computes Distinct-1, Distinct-2, Self-BLEU, SBERT pairwise mean, corpus novelty (mean and minimum), and quality proxy (prompt–idea cosine similarity). Returns a single dict of all scores. |

---

### I/O and CLI

| File | What it does |
|---|---|
| `io.py` | Two utility functions: `read_jsonl(path)` reads a `.jsonl` file and returns a list of dicts. `write_jsonl(path, records)` writes a list of dicts to a `.jsonl` file, creating parent directories automatically. Used throughout the codebase whenever data is read from or written to disk. |
| `cli.py` | The main command-line interface built with Typer. Commands: `prepare-data` (generate sample SFT files), `build-coauthor-data` (generate seed co-author data), `ingest-coauthor-data` (pull external datasets), `list-sources` (print dataset registry), `prepare-source` (normalize a local raw file), `collect-source` (download + normalize from HuggingFace), `train-sft` (train one persona adapter), `train-all` (train all 4), `infer` (one-shot generation), `ideate` (rule-based IdeaCard generation), `chat` (multi-turn co-author session with optional local adapter). |

---

*Report generated from codebase at `src/mop_divpo/`. All code references verified against current branch `co-author`.*
