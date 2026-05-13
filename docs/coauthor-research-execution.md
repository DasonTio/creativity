# Co-Author Research Execution Plan

## Objective

The product objective is a writing partner for pre-writing ideation. Every model, dataset, and evaluation step should support one of these writer actions:

- discover a non-obvious angle
- challenge a weak thesis
- transfer a useful analogy
- map a causal system
- distill a broad topic into a concrete writing direction
- choose and refine ideas without converging on generic suggestions

## Dataset Decisions

The current raw datasets are not sufficient as direct fine-tuning targets. arXiv abstracts, StackExchange answers, Reddit replies, and Gutenberg passages do not teach the model to behave as a co-author. They can still be useful as source material for task construction.

Primary dataset sources to use:

| Source | Role | Use |
| --- | --- | --- |
| CoAuthor | interaction model | Human-AI writing sessions and suggestion behavior |
| WritingPrompts | prompt bank | Creative topic and story-ideation briefs |
| IBM Argument Quality Ranking | argumentative quality | Thesis, counter-thesis, usefulness, and quality signals |
| CGA-CMV | dissent material | Contrarian assumption-challenge examples |
| arXiv / StackExchange / Gutenberg | source material only | Build tasks, prior-art examples, and evaluation prompts |

The implemented seed dataset lives in code at `src/mop_divpo/coauthor/data.py` and can be materialized with:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli build-coauthor-data --output-dir data/processed/coauthor_sft
```

The external referenced dataset can be pulled with:

```bash
PYTHONPATH=src python3 -m mop_divpo.cli ingest-coauthor-data \
  --output-dir data/processed/coauthor_sft \
  --limit-per-source 100
```

Generated rows use message-format SFT examples so the model learns the writing-partner turn:

```json
{
  "messages": [
    {"role": "system", "content": "persona instruction"},
    {"role": "user", "content": "writer brief"},
    {"role": "assistant", "content": "structured idea card"}
  ]
}
```

Seed-only generated volume:

| Persona | Rows |
| --- | ---: |
| `contrarian` | 64 |
| `cross_domain_analogist` | 64 |
| `systems_thinker` | 64 |
| `minimalist` | 64 |
| **Total** | **256** |

The rows combine curated writing scenarios with task variants: thesis, angle, outline, evidence, audience, revision, comparison, and next-step work. This is enough for smoke training and early behavior shaping. It is not enough for final research claims; the next target should be 100-200 reviewed rows per persona.

External-ingested generated volume with `--limit-per-source 100`:

| Persona | Rows |
| --- | ---: |
| `contrarian` | 360 |
| `cross_domain_analogist` | 262 |
| `systems_thinker` | 262 |
| `minimalist` | 262 |
| **Total** | **1146** |

External rows include `references` with dataset name, row index, URL, and source-specific IDs where available. The current ingestion uses WritingPrompts, IBM Argument Quality Ranking, and CGA-CMV. CoAuthor is retained as the interaction-design reference; direct CoAuthor ingestion should be added only after confirming its downloadable files and terms for the intended use.

## Method Challenge

MoP + DivPO is the proposed method, not a guaranteed answer. It must be tested against cheaper alternatives:

- base model
- prompt-only cognitive operators
- one shared co-author adapter
- separate persona adapters
- generate-many-then-rerank by novelty and usefulness
- DivPO-trained persona adapters

The research succeeds only if trained persona adapters or DivPO add measurable value beyond prompt engineering.

## Implemented Research Changes

- Added co-author schemas for `WritingBrief`, `IdeaCard`, and `CoAuthorRecord`.
- Added a curated seed dataset with one high-quality writing-partner example per persona.
- Added `build-coauthor-data` to generate persona-specific co-author SFT JSONL files.
- Added `ideate` to produce structured writing-partner idea cards from the CLI.
- Added adapter repository resolution so inference can load persona-specific HuggingFace adapters.
- Updated training formatting so SFT accepts co-author `messages` rows.
- Added `notebooks/train_coauthor_colab.ipynb` for Colab training on the revised format.

## Next Evaluation Target

The next benchmark should compare:

1. base Qwen
2. prompt-only `ideate`
3. co-author SFT adapters
4. co-author SFT + DivPO adapters
5. rerank-only alternative

Minimum benchmark tasks:

- thesis challenge
- angle generation
- analogy transfer
- systems mapping
- minimalist distillation

Minimum ratings:

- novelty
- usefulness
- specificity
- actionability
- persona fidelity
- semantic diversity

## References

- CoAuthor dataset: https://coauthor.stanford.edu/
- CoAuthor paper: https://arxiv.org/abs/2201.06796
- WritingPrompts dataset: https://huggingface.co/datasets/llm-aes/writing-prompts
- IBM Argument Quality Ranking: https://arxiv.org/abs/1911.11408
- Diverse Preference Optimization: https://huggingface.co/papers/2501.18101
