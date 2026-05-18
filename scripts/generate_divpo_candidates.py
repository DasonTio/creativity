"""
Generate DivPO candidate pairs from a trained SFT adapter.

For each prompt in the coauthor_sft data:
  1. Generate N candidate responses using the SFT adapter
  2. Score candidates with divpo/scoring.py (rarity + quality)
  3. Select chosen/rejected pair with divpo/pairs.py
  4. Write DivPO pairs JSONL per persona

Usage:
    python scripts/generate_divpo_candidates.py \\
        --persona contrarian \\
        --adapter-path outputs/adapters/coauthor/contrarian \\
        --data-path data/processed/coauthor_sft/contrarian.jsonl \\
        --output-path data/processed/divpo_pairs/contrarian.jsonl \\
        --n-candidates 4 \\
        --max-prompts 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mop_divpo.divpo.pairs import select_divpo_pair
from mop_divpo.divpo.scoring import lexical_rarity_scores
from mop_divpo.io import read_jsonl, write_jsonl
from mop_divpo.personas import validate_persona
from mop_divpo.schema import CandidateRecord, DivPOPairRecord


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_adapter(base_model: str, adapter_path: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if str(device) != "cpu" else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        base = base.to(device)

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def _generate_candidates(
    prompt: str,
    system_content: str,
    tokenizer,
    model,
    n: int,
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    responses = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def generate_pairs(
    persona_name: str,
    adapter_path: str,
    data_path: Path,
    output_path: Path,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_candidates: int = 4,
    max_prompts: int = 50,
    max_new_tokens: int = 256,
    min_quality: float = 0.1,
    quality_weight: float = 0.4,
    rarity_weight: float = 0.6,
) -> None:
    from mop_divpo.personas import describe_persona

    persona = validate_persona(persona_name)
    system_content = describe_persona(persona)

    print(f"Loading adapter from {adapter_path} ...")
    tokenizer, model = _load_adapter(base_model, adapter_path)

    rows = read_jsonl(data_path)
    prompts: list[str] = []
    for row in rows:
        msgs = row.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "user":
                prompts.append(msg["content"])
                break
    prompts = prompts[:max_prompts]
    print(f"Prompts to process: {len(prompts)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] generating {n_candidates} candidates ...")
        responses = _generate_candidates(
            prompt, system_content, tokenizer, model, n_candidates, max_new_tokens
        )

        rarity_scores = lexical_rarity_scores(responses)

        candidates = [
            CandidateRecord(
                id=f"{persona}-prompt{i}-cand{j}",
                persona=persona,
                prompt=prompt,
                response=resp,
                quality_score=rarity_scores[j],  # proxy: quality = distinct vocab
                rarity_score=rarity_scores[j],
            )
            for j, resp in enumerate(responses)
        ]

        try:
            pair = select_divpo_pair(
                candidates,
                min_quality=min_quality,
                quality_weight=quality_weight,
                rarity_weight=rarity_weight,
            )
            pairs.append(pair.model_dump())
        except ValueError as e:
            print(f"    Skipping prompt {i}: {e}")
            continue

    write_jsonl(output_path, pairs)
    print(f"Wrote {len(pairs)} DivPO pairs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DivPO candidate pairs from SFT adapter")
    parser.add_argument("--persona", required=True)
    parser.add_argument("--adapter-path", required=True, help="Local path to SFT adapter folder")
    parser.add_argument("--data-path", type=Path, required=True, help="coauthor_sft JSONL for this persona")
    parser.add_argument("--output-path", type=Path, required=True, help="Output DivPO pairs JSONL")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-quality", type=float, default=0.1)
    parser.add_argument("--quality-weight", type=float, default=0.4)
    parser.add_argument("--rarity-weight", type=float, default=0.6)
    args = parser.parse_args()

    generate_pairs(
        persona_name=args.persona,
        adapter_path=args.adapter_path,
        data_path=args.data_path,
        output_path=args.output_path,
        base_model=args.base_model,
        n_candidates=args.n_candidates,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        min_quality=args.min_quality,
        quality_weight=args.quality_weight,
        rarity_weight=args.rarity_weight,
    )


if __name__ == "__main__":
    main()
