from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mop_divpo.personas import describe_persona, validate_persona

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_tokenizer: Any = None
_model: Any = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load() -> tuple[Any, Any]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16,
        ).to(_device())
        _model.eval()
    return _tokenizer, _model


def generate(prompt: str, personas: list[str], max_new_tokens: int = 256) -> list[dict[str, str]]:
    tokenizer, model = _load()
    device = next(model.parameters()).device
    outputs = []
    for persona_value in personas:
        persona = validate_persona(persona_value)
        messages = [
            {"role": "system", "content": describe_persona(persona)},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            token_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = token_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        outputs.append({"persona": persona, "prompt": prompt, "response": response})
    return outputs


def dry_run_generate(prompt: str, personas: list[str]) -> list[dict[str, str]]:
    outputs = []
    for persona_value in personas:
        persona = validate_persona(persona_value)
        outputs.append(
            {
                "persona": persona,
                "prompt": prompt,
                "response": f"[dry-run:{persona}] {describe_persona(persona)} Prompt: {prompt}",
            }
        )
    return outputs
