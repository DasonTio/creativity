from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mop_divpo.personas import PersonaId, describe_persona, validate_persona

if TYPE_CHECKING:
    from mop_divpo.retrieval.corpus import CorpusRetriever


@dataclass
class PersonaSession:
    persona: PersonaId
    history: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        """Append a user message to history."""
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        """Append an assistant message to history."""
        self.history.append({"role": "assistant", "content": text})

    def to_messages(self) -> list[dict]:
        """Return full message history for apply_chat_template."""
        return list(self.history)


def _build_prior_art_block(retriever: CorpusRetriever, query: str) -> str:
    results = retriever.retrieve(query, k=3)
    if not results:
        return ""
    prior_art_lines = [
        f'{i + 1}. "{r.document.title}": {r.document.snippet}'
        for i, r in enumerate(results)
    ]
    return (
        "\n\nPRIOR ART — existing ideas on this topic (retrieved automatically):\n"
        + "\n".join(prior_art_lines)
        + "\n\nYour idea MUST be semantically distinct from all three above. "
        "Treat these as the \"already said\" space. Your job is to operate outside it."
    )


def generate_turn(
    session: PersonaSession,
    user_message: str,
    retriever: CorpusRetriever | None = None,
    max_new_tokens: int = 256,
    adapter_path: str | None = None,
) -> tuple[str, PersonaSession]:
    """Generate one conversational turn and update session history.

    Steps:
    1. Build system content from persona description.
    2. On first turn only, append PRIOR ART block if retriever is provided.
    3. Add user_message to session history.
    4. Build full messages list (system + history).
    5. Load model + tokenizer (with optional local LoRA adapter), apply chat template, generate.
    6. Decode response, add to session history.
    7. Return (response, session).
    """
    from mop_divpo.inference.generate import _device, _load, _load_with_adapter

    # 1. Build system content
    system_content = describe_persona(session.persona)

    # 2. Inject PRIOR ART on first turn only
    if retriever is not None and len(session.history) == 0:
        prior_art = _build_prior_art_block(retriever, user_message)
        if prior_art:
            system_content = system_content + prior_art

    # 3. Add user message to history
    session.add_user(user_message)

    # 4. Build messages list
    messages = [{"role": "system", "content": system_content}] + session.to_messages()

    # 5. Load model + tokenizer
    if adapter_path is not None:
        tokenizer, model = _load_with_adapter(adapter_path)
    else:
        tokenizer, model = _load()
    device = _device()

    # 6. Apply chat template and generate
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

    # 7. Add response to session and return
    session.add_assistant(response)
    return response, session


def dry_run_generate_turn(
    session: PersonaSession,
    user_message: str,
    retriever: CorpusRetriever | None = None,
    max_new_tokens: int = 256,
) -> tuple[str, PersonaSession]:
    """Dry-run version of generate_turn — no model loading, returns a stub response.

    Follows the same session-update logic as generate_turn (prior art injection on
    first turn only, history appended) so tests can verify session state without GPU.
    """
    # Build system content (mirrors generate_turn logic)
    system_content = describe_persona(session.persona)

    prior_art_injected = False
    if retriever is not None and len(session.history) == 0:
        prior_art = _build_prior_art_block(retriever, user_message)
        if prior_art:
            system_content = system_content + prior_art
            prior_art_injected = True

    session.add_user(user_message)

    response = (
        f"[dry-run:{session.persona}] "
        f"{'[prior-art-injected] ' if prior_art_injected else ''}"
        f"Responding to: {user_message}"
    )
    session.add_assistant(response)
    return response, session
