"""
Evaluation runner — compares baseline vs MoP-SFT vs MoP-DivPO across all personas.

Generates outputs from each condition, runs all metrics, prints comparison table.

Usage:
    python scripts/run_evaluation.py \\
        --prompts-file data/processed/coauthor_sft/contrarian.jsonl \\
        --sft-adapter-dir outputs/adapters/coauthor \\
        --divpo-adapter-dir outputs/adapters/divpo \\
        --corpus-dir data/processed/sft \\
        --n-outputs 8 \\
        --output-json outputs/evaluation/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mop_divpo.eval.evaluate import evaluate_outputs
from mop_divpo.inference.generate import dry_run_generate
from mop_divpo.io import read_jsonl
from mop_divpo.personas import PERSONA_IDS, describe_persona, validate_persona


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(base_model: str, adapter_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if str(_device()) != "cpu" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        base = base.to(_device())

    if adapter_path is not None:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, adapter_path)

    base.eval()
    return tokenizer, base


def _generate_outputs(
    prompts: list[str],
    persona: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
) -> list[dict]:
    device = next(model.parameters()).device
    system_content = describe_persona(validate_persona(persona))
    outputs = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                repetition_penalty=1.15,
                min_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append({"persona": persona, "prompt": prompt, "response": response})
    return outputs


def _extract_prompts(data_dir: Path, persona: str, n: int) -> list[str]:
    path = data_dir / f"{persona}.jsonl"
    if not path.exists():
        return []
    rows = read_jsonl(path)
    prompts = []
    for row in rows:
        for msg in row.get("messages", []):
            if msg.get("role") == "user":
                prompts.append(msg["content"])
                break
    return prompts[:n]


def _build_retriever(corpus_dir: Path):
    try:
        from mop_divpo.retrieval.corpus import CorpusRetriever
        return CorpusRetriever.build_from_jsonl(corpus_dir)
    except Exception:
        return None


def run_evaluation(
    data_dir: Path,
    sft_adapter_dir: Path | None,
    divpo_adapter_dir: Path | None,
    corpus_dir: Path | None,
    base_model: str,
    n_outputs: int,
    output_json: Path,
    dry_run: bool,
) -> None:
    retriever = _build_retriever(corpus_dir) if corpus_dir and corpus_dir.exists() else None

    conditions: list[tuple[str, str | None]] = [("baseline", None)]
    if sft_adapter_dir and sft_adapter_dir.exists():
        conditions.append(("mop_sft", str(sft_adapter_dir)))
    if divpo_adapter_dir and divpo_adapter_dir.exists():
        conditions.append(("mop_divpo", str(divpo_adapter_dir)))

    all_results: dict = {}

    for persona in PERSONA_IDS:
        print(f"\n=== Persona: {persona} ===")
        prompts = _extract_prompts(data_dir, persona, n_outputs)
        if not prompts:
            print(f"  No prompts found for {persona}, skipping.")
            continue

        persona_results: dict = {}

        for condition_name, adapter_base in conditions:
            adapter_path = None
            if adapter_base is not None:
                candidate = Path(adapter_base) / persona
                adapter_path = str(candidate) if candidate.exists() else None
                if adapter_path is None:
                    print(f"  [{condition_name}] adapter not found at {candidate}, skipping.")
                    continue

            print(f"  [{condition_name}] generating {len(prompts)} outputs ...")

            if dry_run:
                outputs = dry_run_generate(prompts[0], [persona]) * len(prompts)
            else:
                try:
                    tokenizer, model = _load_model(base_model, adapter_path)
                    outputs = _generate_outputs(prompts, persona, tokenizer, model)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  [{condition_name}] ERROR: {e}")
                    continue

            metrics = evaluate_outputs(outputs, retriever)
            persona_results[condition_name] = metrics
            print(f"  [{condition_name}] distinct_1={metrics['distinct_1']:.3f}  "
                  f"self_bleu={metrics['self_bleu']:.3f}  "
                  f"sbert={metrics['sbert_pairwise_mean']:.3f}  "
                  f"novelty={metrics['corpus_novelty_mean']:.3f}  "
                  f"quality={metrics['quality_proxy_mean']:.3f}")

        all_results[persona] = persona_results

    # Print summary table
    print("\n\n=== SUMMARY TABLE ===")
    header = f"{'Persona':<25} {'Condition':<12} {'Dist-1':>6} {'Dist-2':>6} {'S-BLEU':>7} {'SBERT':>6} {'Novelty':>8} {'Quality':>8}"
    print(header)
    print("-" * len(header))
    for persona, persona_results in all_results.items():
        for condition, metrics in persona_results.items():
            print(
                f"{persona:<25} {condition:<12} "
                f"{metrics['distinct_1']:>6.3f} "
                f"{metrics['distinct_2']:>6.3f} "
                f"{metrics['self_bleu']:>7.3f} "
                f"{metrics['sbert_pairwise_mean']:>6.3f} "
                f"{metrics['corpus_novelty_mean']:>8.3f} "
                f"{metrics['quality_proxy_mean']:>8.3f}"
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation across baseline / MoP-SFT / MoP-DivPO")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/coauthor_sft"))
    parser.add_argument("--sft-adapter-dir", type=Path, default=None, help="Dir containing per-persona SFT adapter folders")
    parser.add_argument("--divpo-adapter-dir", type=Path, default=None, help="Dir containing per-persona DivPO adapter folders")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-outputs", type=int, default=8, help="Outputs per persona per condition")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/evaluation/results.json"))
    parser.add_argument("--dry-run", action="store_true", help="Skip model loading, use stubs")
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        sft_adapter_dir=args.sft_adapter_dir,
        divpo_adapter_dir=args.divpo_adapter_dir,
        corpus_dir=args.corpus_dir,
        base_model=args.base_model,
        n_outputs=args.n_outputs,
        output_json=args.output_json,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
