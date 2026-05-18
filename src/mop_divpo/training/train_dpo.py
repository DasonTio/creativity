"""DPO training script — fine-tunes a persona SFT adapter using DivPO pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from mop_divpo.io import read_jsonl
from mop_divpo.personas import validate_persona

HF_REPO_PREFIX = "DasonTio/mop-divpo-coauthor"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    persona_name: str,
    pairs_path: Path,
    sft_adapter_path: str,
    output_dir: Path,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_steps: int = 100,
    learning_rate: float = 5e-5,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    beta: float = 0.1,
    push_to_hub: bool = False,
) -> None:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    persona = validate_persona(persona_name)
    device = _device()

    print(f"Device: {device}")
    print(f"Persona: {persona}")
    print(f"Pairs: {pairs_path}")
    print(f"SFT adapter: {sft_adapter_path}")
    print(f"Output: {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT adapter as starting point
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=True)

    # Reference model — frozen copy of SFT adapter (DPO computes KL against this)
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    ref_model = PeftModel.from_pretrained(ref_base, sft_adapter_path, is_trainable=False)
    ref_model.eval()

    # Load DivPO pairs
    rows: list[dict[str, Any]] = read_jsonl(pairs_path)
    if not rows:
        raise ValueError(f"No pairs found in {pairs_path}")
    print(f"DivPO pairs: {len(rows)}")

    # DPOTrainer expects columns: prompt, chosen, rejected
    dataset = Dataset.from_list([
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in rows
    ])

    args = DPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        fp16=device == "cuda",
        bf16=False,
        logging_steps=10,
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"DPO adapter saved to {output_dir}")

    if push_to_hub:
        repo_id = f"{HF_REPO_PREFIX}-{persona}-divpo"
        print(f"Pushing to {repo_id} ...")
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training on DivPO pairs for a single persona")
    parser.add_argument("persona", help="Persona name")
    parser.add_argument("--pairs-path", type=Path, required=True, help="Path to DivPO pairs JSONL")
    parser.add_argument("--sft-adapter", required=True, help="Local path or HF repo ID of SFT adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty coefficient)")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    train(
        persona_name=args.persona,
        pairs_path=args.pairs_path,
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
