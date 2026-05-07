"""SFT training script for a single persona adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from mop_divpo.io import read_jsonl
from mop_divpo.personas import describe_persona, validate_persona

HF_REPO_PREFIX = "DasonTio/mop-divpo"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_records(
    records: list[dict],
    tokenizer: AutoTokenizer,
    persona_description: str,
) -> list[dict[str, str]]:
    formatted = []
    for record in records:
        prompt = str(record.get("prompt", "")).strip()
        response = str(record.get("response", "")).strip()
        if not prompt or not response:
            continue
        messages = [
            {"role": "system", "content": persona_description},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": text})
    return formatted


def train(
    persona_name: str,
    data_path: Path,
    output_dir: Path,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_steps: int = 200,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    push_to_hub: bool = False,
) -> None:
    persona = validate_persona(persona_name)
    persona_description = describe_persona(persona)
    device = _device()

    print(f"Device: {device}")
    print(f"Persona: {persona}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_jsonl(data_path)
    formatted = _format_records(records, tokenizer, persona_description)
    if not formatted:
        raise ValueError(f"No valid records after formatting in {data_path}")
    print(f"Training records: {len(formatted)}")

    dataset = Dataset.from_list(formatted)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=device == "cuda",
        bf16=False,
        logging_steps=10,
        save_steps=max_steps,
        save_total_limit=1,
        dataloader_pin_memory=device == "cuda",
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")

    if push_to_hub:
        repo_id = f"{HF_REPO_PREFIX}-{persona}-sft"
        print(f"Pushing adapter to {repo_id}...")
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a per-persona SFT LoRA adapter")
    parser.add_argument("persona", help="Persona name (contrarian, cross_domain_analogist, systems_thinker, minimalist)")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to SFT JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Adapter output directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push-to-hub", action="store_true", help="Push trained adapter to HuggingFace Hub")
    args = parser.parse_args()

    train(
        persona_name=args.persona,
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
