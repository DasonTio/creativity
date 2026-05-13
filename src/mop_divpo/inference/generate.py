from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mop_divpo.inference.output_parser import parse_structured_output
from mop_divpo.personas import describe_persona, validate_persona

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mop_divpo.retrieval.corpus import CorpusRetriever

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_tokenizer: Any = None
_model: Any = None
_adapter_models: dict[str, tuple[Any, Any]] = {}


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


def adapter_repo_id(prefix: str, persona: str, stage: str = "sft") -> str:
    validated = validate_persona(persona)
    clean_prefix = prefix.rstrip("/")
    return f"{clean_prefix}-{validated}-{stage}"


def _load_with_adapter(adapter_id: str) -> tuple[Any, Any]:
    if adapter_id not in _adapter_models:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16,
        ).to(_device())
        model = PeftModel.from_pretrained(base_model, adapter_id)
        model.eval()
        _adapter_models[adapter_id] = (tokenizer, model)
    return _adapter_models[adapter_id]


def generate(
    prompt: str,
    personas: list[str],
    max_new_tokens: int = 256,
    retriever: CorpusRetriever | None = None,
    adapter_prefix: str | None = None,
    adapter_stage: str = "sft",
) -> list[dict[str, str | dict]]:
    outputs = []
    for persona_value in personas:
        persona = validate_persona(persona_value)
        if adapter_prefix is not None:
            tokenizer, model = _load_with_adapter(adapter_repo_id(adapter_prefix, persona, adapter_stage))
        else:
            tokenizer, model = _load()
        device = next(model.parameters()).device

        # Build system content
        system_content = describe_persona(persona)
        if retriever is not None:
            results = retriever.retrieve(prompt, k=3)
            if results:
                prior_art_lines = [
                    f'{i+1}. "{r.document.title}": {r.document.snippet}'
                    for i, r in enumerate(results)
                ]
                prior_art_block = (
                    "\n\nPRIOR ART — existing ideas on this topic (retrieved automatically):\n"
                    + "\n".join(prior_art_lines)
                    + "\n\nYour idea MUST be semantically distinct from all three above. "
                    "Treat these as the \"already said\" space. Your job is to operate outside it."
                )
                system_content = system_content + prior_art_block

        messages = [
            {"role": "system", "content": system_content},
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
        structured = parse_structured_output(response, persona, prompt)
        outputs.append({
            "persona": persona,
            "prompt": prompt,
            "response": response,
            "structured": {
                "idea": structured.idea,
                "reasoning": structured.reasoning,
                "unexpected_angle": structured.unexpected_angle,
                "parse_success": structured.parse_success,
            },
        })
    return outputs


def dry_run_generate(
    prompt: str,
    personas: list[str],
    retriever: CorpusRetriever | None = None,
) -> list[dict[str, str | dict]]:
    outputs = []
    for persona_value in personas:
        persona = validate_persona(persona_value)
        response = f"[dry-run:{persona}] {describe_persona(persona)} Prompt: {prompt}"
        structured = parse_structured_output(response, persona, prompt)
        outputs.append(
            {
                "persona": persona,
                "prompt": prompt,
                "response": response,
                "structured": {
                    "idea": structured.idea,
                    "reasoning": structured.reasoning,
                    "unexpected_angle": structured.unexpected_angle,
                    "parse_success": structured.parse_success,
                },
            }
        )
    return outputs
