"""
MoP + DivPO Co-Author Demo — Gradio UI

Launch:
    source .venv/bin/activate
    PYTHONPATH=src python app.py

Options:
    --adapter-dir  Path to unzipped coauthor_adapters folder (default: loads base model only)
    --dry-run      Skip model loading, return stub responses
    --share        Create public Gradio link
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


PERSONA_LABELS = {
    "contrarian": "Contrarian — challenges assumptions",
    "cross_domain_analogist": "Cross-Domain Analogist — finds structural analogies",
    "systems_thinker": "Systems Thinker — maps causes and feedback loops",
    "minimalist": "Minimalist — constraint-driven creativity",
}

PERSONA_IDS = list(PERSONA_LABELS.keys())


def build_app(adapter_dir: Path | None, dry_run: bool) -> "gr.Blocks":
    import gradio as gr

    from mop_divpo.inference.session import (
        PersonaSession,
        dry_run_generate_turn,
        generate_turn,
    )
    from mop_divpo.personas import validate_persona

    def _adapter_path(persona: str) -> str | None:
        if adapter_dir is None:
            return None
        p = adapter_dir / persona
        return str(p) if p.exists() else None

    def _format_idea_card(response: str) -> str:
        try:
            data = json.loads(response)
            parts = []
            if data.get("operation"):
                parts.append(f"**Operation:** {data['operation'].replace('_', ' ').title()}")
            if data.get("diagnosis"):
                parts.append(f"**Diagnosis:**\n{data['diagnosis']}")
            if data.get("idea"):
                parts.append(f"**Idea:**\n{data['idea']}")
            if data.get("rationale"):
                parts.append(f"**Rationale:**\n{data['rationale']}")
            if data.get("next_step"):
                parts.append(f"**Next Step:**\n{data['next_step']}")
            return "\n\n".join(parts) if parts else response
        except (json.JSONDecodeError, AttributeError):
            return response

    # --- One-shot ideation tab ---
    def run_ideation(persona_label: str, topic: str, goal: str, audience: str, constraints: str):
        if not topic.strip():
            return "Please enter a topic."
        persona = PERSONA_IDS[[v for v in PERSONA_LABELS.values()].index(persona_label)]
        brief = f"Topic: {topic}\nWriter goal: {goal or 'Generate a non-obvious angle'}"
        if audience:
            brief += f"\nAudience: {audience}"
        if constraints:
            brief += f"\nConstraints: {constraints}"

        session = PersonaSession(persona=validate_persona(persona))
        ap = _adapter_path(persona)
        fn = dry_run_generate_turn if dry_run else generate_turn

        response, _ = fn(session, brief, adapter_path=ap, max_new_tokens=512)
        return _format_idea_card(response)

    # --- Multi-turn chat tab ---
    sessions: dict[str, PersonaSession] = {}

    def start_session(persona_label: str, topic: str, goal: str, audience: str):
        persona = PERSONA_IDS[[v for v in PERSONA_LABELS.values()].index(persona_label)]
        session = PersonaSession(persona=validate_persona(persona))
        sessions["current"] = session
        brief = f"Topic: {topic}\nWriter goal: {goal or 'Help me develop this idea'}"
        if audience:
            brief += f"\nAudience: {audience}"

        ap = _adapter_path(persona)
        fn = dry_run_generate_turn if dry_run else generate_turn
        response, updated = fn(sessions["current"], brief, adapter_path=ap, max_new_tokens=512)
        sessions["current"] = updated
        formatted = _format_idea_card(response)
        history = [{"role": "user", "content": brief}, {"role": "assistant", "content": formatted}]
        return history, ""

    def continue_chat(user_message: str, history: list):
        if "current" not in sessions:
            return history + [{"role": "assistant", "content": "Start a session first using the fields above."}], ""
        if not user_message.strip():
            return history, ""

        persona = sessions["current"].persona
        ap = _adapter_path(persona)
        fn = dry_run_generate_turn if dry_run else generate_turn
        response, updated = fn(sessions["current"], user_message, adapter_path=ap, max_new_tokens=512)
        sessions["current"] = updated
        formatted = _format_idea_card(response)
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": formatted},
        ]
        return history, ""

    def reset_session():
        sessions.clear()
        return [], ""

    # --- UI ---
    mode_label = "DRY RUN MODE" if dry_run else ("With adapter" if adapter_dir else "Base model only")

    with gr.Blocks(title="MoP+DivPO Co-Author", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
# MoP + DivPO Writing Co-Author
**Mode:** {mode_label} · 4 persona adapters: Contrarian · Cross-Domain Analogist · Systems Thinker · Minimalist
""")

        with gr.Tab("One-Shot Ideation"):
            gr.Markdown("Generate a structured IdeaCard from a writing brief in one click.")
            with gr.Row():
                with gr.Column(scale=1):
                    persona_dd = gr.Dropdown(
                        choices=list(PERSONA_LABELS.values()),
                        value=list(PERSONA_LABELS.values())[0],
                        label="Persona",
                    )
                    topic_in = gr.Textbox(label="Topic", placeholder="e.g. AI in education")
                    goal_in = gr.Textbox(label="Writer goal", placeholder="e.g. Find a non-obvious thesis angle")
                    audience_in = gr.Textbox(label="Audience (optional)", placeholder="e.g. university instructors")
                    constraints_in = gr.Textbox(label="Constraints (optional)", placeholder="e.g. avoid generic productivity claims")
                    ideate_btn = gr.Button("Generate Idea", variant="primary")
                with gr.Column(scale=1):
                    idea_out = gr.Markdown(label="IdeaCard")

            ideate_btn.click(
                run_ideation,
                inputs=[persona_dd, topic_in, goal_in, audience_in, constraints_in],
                outputs=idea_out,
            )

        with gr.Tab("Multi-Turn Chat"):
            gr.Markdown("Start a writing session then continue the conversation across multiple turns.")
            with gr.Row():
                chat_persona_dd = gr.Dropdown(
                    choices=list(PERSONA_LABELS.values()),
                    value=list(PERSONA_LABELS.values())[0],
                    label="Persona",
                    scale=1,
                )
                chat_topic_in = gr.Textbox(label="Topic", placeholder="e.g. urban loneliness", scale=2)
                chat_goal_in = gr.Textbox(label="Goal", placeholder="e.g. build a systems explanation", scale=2)
                chat_audience_in = gr.Textbox(label="Audience (optional)", scale=1)

            start_btn = gr.Button("Start Session", variant="primary")
            chatbot = gr.Chatbot(label="Co-Author Session", height=500, type="messages")

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Continue the conversation...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1)
                reset_btn = gr.Button("Reset", scale=1, variant="secondary")

            start_btn.click(
                start_session,
                inputs=[chat_persona_dd, chat_topic_in, chat_goal_in, chat_audience_in],
                outputs=[chatbot, chat_input],
            )
            send_btn.click(
                continue_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                continue_chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input],
            )
            reset_btn.click(reset_session, outputs=[chatbot, chat_input])

        with gr.Tab("About"):
            gr.Markdown("""
## Architecture

| Component | Description |
|---|---|
| **Base model** | Qwen/Qwen2.5-0.5B-Instruct |
| **MoP** | 4 separate LoRA adapters, one per persona |
| **DivPO** | Diverse Preference Optimization — trains adapters to prefer rare+quality responses |
| **Co-Author SFT** | Fine-tuned on WritingBrief → IdeaCard pairs |

## Personas

| Persona | Approach |
|---|---|
| **Contrarian** | Surfaces hidden assumptions, argues minority positions |
| **Cross-Domain Analogist** | Finds structural isomorphisms between distant fields |
| **Systems Thinker** | Maps causes, feedback loops, leverage points |
| **Minimalist** | Constraint-driven creativity, removal as generative force |

## Output Format (IdeaCard)
Every response is structured as:
- **Operation** — type of co-author move
- **Diagnosis** — what is wrong or missing in the current framing
- **Idea** — the concrete direction to take
- **Rationale** — why this direction is better than the obvious one
- **Next Step** — one immediate action the writer can take now

## Evaluation Metrics
Self-BLEU · Distinct-1/2 · SBERT pairwise cosine · Corpus novelty · Quality proxy
""")

    return demo


def main() -> None:
    args = _parse_args()
    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        print("Install gradio first: pip install gradio")
        sys.exit(1)

    demo = build_app(adapter_dir=args.adapter_dir, dry_run=args.dry_run)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
