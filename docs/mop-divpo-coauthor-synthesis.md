# MoP + DivPO Co-Author Synthesis

## Executive Verdict

The original proposal has a real research problem: LLM-assisted ideation can improve individual output while reducing collective diversity. However, the current project is not yet a real co-author system. It is closer to a prototype scaffold that combines persona prompting, SFT training utilities, and partial evaluation code.

The key issue is that a writing partner is not defined by producing four different answers. A co-author helps a writer discover, challenge, compare, refine, and commit to ideas. Mixture of Persona (MoP) and Diverse Preference Optimization (DivPO) can support that goal, but they are mechanisms, not the product itself.

## What Currently Does Not Work

The trained adapters are not integrated into the repository inference path. Runtime generation still loads the base `Qwen/Qwen2.5-0.5B-Instruct` model and injects persona prompts, so the current CLI does not prove trained MoP behavior.

DivPO is also not end-to-end. The repo has scoring and pair-selection helpers, but no complete candidate-generation pipeline, no reproducible `data/processed/divpo/` artifacts, and no first-class CLI flow that builds DPO-compatible preference pairs.

The largest weakness is dataset alignment. Current SFT data wraps raw source text with persona-like prompts, but the responses remain arXiv abstracts, StackExchange answers, Reddit replies, or Gutenberg passages. That trains source imitation, not co-author behavior.

The evaluation layer measures lexical and semantic diversity, but it does not yet measure whether the system helps writers produce better pre-writing ideas.

## Revised Project Scope

The project should be scoped as:

> A pre-writing ideation assistant that uses multiple cognitive strategies to help writers generate, challenge, expand, and refine ideas while reducing idea homogenization.

The research claim should become:

> A structured pre-writing co-author using cognitive persona operations and diversity-aware preference optimization improves idea novelty, usefulness, and diversity compared with base prompting and prompt-only personas.

This keeps the original proposal intact at the conceptual level, but makes it more precise and testable.

## What Changes From The Proposal

Unchanged:

- The core problem remains generative monoculture in LLM-assisted ideation.
- The four personas remain: `contrarian`, `cross_domain_analogist`, `systems_thinker`, and `minimalist`.
- The goal remains improving both inter-persona and intra-persona diversity.
- DivPO remains the hypothesis for preferring rare-but-high-quality outputs.
- Baselines should still include base model, prompt-only personas, MoP SFT, and MoP + DivPO.

Changed or made stricter:

- Personas become writing operations, not just styles.
- The product becomes a pre-writing workflow, not a raw generation endpoint.
- Raw datasets become source material for task construction, not direct assistant responses.
- Evaluation must include novelty, usefulness, specificity, and writer-facing value.
- MoP + DivPO must be challenged against simpler alternatives.

## Co-Author Operations

Each persona should have a concrete job:

- `contrarian`: identify hidden assumptions, invert weak premises, create counter-theses, and surface minority-position angles.
- `cross_domain_analogist`: map the writer's problem to a distant domain and transfer a useful structure back.
- `systems_thinker`: identify causes, constraints, feedback loops, incentives, and leverage points.
- `minimalist`: remove non-essential parts, compress the thesis, and find the smallest load-bearing idea.

These are not personality costumes. They are cognitive tools for pre-writing.

## Dataset Strategy

The project needs a new `coauthor_sft` dataset aligned to real writer tasks.

Recommended primary sources:

- CoAuthor: use as the interaction model because it records real human-AI writing sessions, accepted/dismissed suggestions, and collaborative writing behavior.
- Prewriting studies: use to ground the product flow around ideation, illumination, and implementation.
- WritingPrompts: use as a creative prompt bank.
- IBM Argument Quality Ranking: use for argumentative thesis and counter-thesis tasks.
- CGA-CMV: keep for contrarian examples, but rewrite into co-author form.
- arXiv, StackExchange, and Gutenberg: use only as source material for synthetic task creation, not as assistant target responses.

Recommended training record:

```json
{
  "task": "challenge_thesis",
  "persona": "contrarian",
  "writer_context": "I want to argue that AI improves education.",
  "writer_goal": "Find a less generic angle.",
  "response": {
    "diagnosis": "The hidden assumption is that improvement means efficiency.",
    "idea": "Argue that AI improves education only when it slows students down.",
    "why_it_helps": "This creates tension and avoids the obvious productivity framing.",
    "next_step": "Test this against tutoring, feedback, and assessment examples."
  }
}
```

## Challenge To MoP + DivPO

MoP is not automatically meaningful. Prompt-only cognitive operators may perform similarly. DivPO is not automatically useful either; if preference pairs are weak, it can optimize toward superficial weirdness rather than useful novelty.

Alternatives that must be tested:

- Prompt-only cognitive operators.
- A single adapter trained on all co-author operations.
- Multi-agent orchestration without fine-tuning.
- Generate-many-then-rerank using novelty and usefulness.
- MMR, DPP-style, MMI-style, or diverse decoding methods.
- Workflow scaffolding, which may matter more than model training for real writers.

MoP + DivPO should remain the proposed method, but the paper should be honest: it must prove that this method beats cheaper baselines.

## Product Architecture

The system should be built as a pre-writing studio:

1. The writer provides a brief: topic, goal, audience, constraints, and rough notes.
2. The system produces persona-labeled idea cards.
3. The writer selects, rejects, or edits cards.
4. The system refines selected cards into thesis options, outlines, counterarguments, and evidence needs.
5. Accepted and rejected ideas are tracked so the system avoids repetition.
6. Evaluation uses both generated artifacts and writer interaction signals.

The first useful CLI could be:

```bash
mop-divpo ideate --topic "AI in education" --goal "find thesis angles"
```

The output should be idea cards, not raw JSON blobs.

## Implementation Plan

Phase 1: Co-author schemas

- Add `WritingBrief`, `IdeaCard`, `CoAuthorTurn`, `PersonaOperation`, and `UserFeedback`.
- Add structured response fields: `diagnosis`, `idea`, `rationale`, and `next_step`.

Phase 2: Dataset rebuild

- Add a `coauthor_sft` data builder.
- Convert current source data into task prompts only.
- Create curated or synthetic co-author examples per persona.
- Write outputs to `data/processed/coauthor_sft/{persona}.jsonl`.

Phase 3: Adapter-backed inference

- Add PEFT adapter loading to repository inference.
- Support local and HuggingFace adapter sources.
- Compare base, prompt-only, and adapter-backed outputs.

Phase 4: Co-author workflow

- Add `mop-divpo ideate`.
- Return structured idea cards from selected personas.
- Add synthesis behavior that combines useful ideas across personas.

Phase 5: Real DivPO pipeline

- Generate multiple candidates per persona/task.
- Score quality by relevance, specificity, actionability, and usefulness.
- Score diversity by semantic distance from candidate pool and prior ideas.
- Build DPO pairs where chosen is rare and useful, and rejected is generic or weak.
- Add `mop-divpo build-divpo-pairs`.

Phase 6: Evaluation

- Add benchmark prompts for thesis generation, angle expansion, assumption challenge, and outline ideation.
- Compare base model, prompt-only personas, SFT adapters, SFT + DivPO adapters, and rerank-only alternatives.
- Preserve raw generations for qualitative analysis.

## Evaluation Plan

Model metrics:

- Self-BLEU: lower means less overlap.
- Distinct-1 and Distinct-2: higher means more lexical variety.
- SBERT pairwise distance: higher distance means more semantic diversity.
- Corpus novelty: distance from known prior ideas.

Writer-facing metrics:

- Novelty: is the idea non-obvious?
- Usefulness: can the writer use it?
- Specificity: does it avoid generic advice?
- Actionability: does it suggest a concrete next move?
- Persona fidelity: did the output perform the intended cognitive operation?
- Selection rate: which cards do users keep?
- Revision value: do accepted ideas survive into outlines or drafts?

## Immediate Next Steps

Stop further training on the current SFT data until the data is rebuilt. The fastest truthful test is:

1. Create a small, high-quality co-author seed dataset.
2. Wire adapter-backed inference into the repo.
3. Add a prompt-only versus adapter comparison command.
4. Run evaluation on a small benchmark of pre-writing tasks.

If adapters do not beat prompt-only operators, the training is not adding value yet.

## Key References

- Lee, Liang, and Yang, 2022: CoAuthor dataset and human-AI writing interaction model. https://coauthor.stanford.edu/
- Wan et al., 2023: prewriting as iterative human-AI co-creativity across ideation, illumination, and implementation. https://consensus.app/papers/details/112902ae257356c9acb5b17910217d85/?utm_source=chatgpt
- Lanchantin et al., 2025: Diverse Preference Optimization. https://huggingface.co/papers/2501.18101
- Rafailov et al., 2023: Direct Preference Optimization. https://papers.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
- Vijayakumar et al., 2016: Diverse Beam Search. https://huggingface.co/papers/1610.02424
- Li et al., 2016: MMI objective for diverse neural conversation responses. https://www.microsoft.com/en-us/research/?p=238042
- Shaer et al., 2024: AI-augmented brainwriting and ideation support. https://consensus.app/papers/details/8c47113db3eb5e58a1c77c0e2e5f9b8b/?utm_source=chatgpt
- Dhillon et al., 2024: scaffolding levels in AI co-writing. https://consensus.app/papers/details/9cab50f2d810573b8386b9f02fe7f25f/?utm_source=chatgpt
