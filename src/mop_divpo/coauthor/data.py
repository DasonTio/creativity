from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mop_divpo.coauthor.schema import CoAuthorRecord, IdeaCard, WritingBrief
from mop_divpo.io import write_jsonl
from mop_divpo.personas import PERSONA_IDS


@dataclass(frozen=True)
class CoAuthorDatasetSource:
    name: str
    dataset_id: str
    use: str
    role: str
    url: str


COAUTHOR_DATASET_SOURCES: tuple[CoAuthorDatasetSource, ...] = (
    CoAuthorDatasetSource(
        name="coauthor",
        dataset_id="stanford-coauthor",
        use="Interaction traces for human-AI collaborative writing behavior.",
        role="Primary design reference for co-author turns, accepted suggestions, and writing context.",
        url="https://coauthor.stanford.edu/",
    ),
    CoAuthorDatasetSource(
        name="writing_prompts",
        dataset_id="llm-aes/writing-prompts",
        use="Prompt bank for creative and argumentative pre-writing briefs.",
        role="Source of writer topics and rough ideation situations, not target assistant responses.",
        url="https://huggingface.co/datasets/llm-aes/writing-prompts",
    ),
    CoAuthorDatasetSource(
        name="argument_quality_ranking",
        dataset_id="ibm-research/argument_quality_ranking_30k",
        use="Argument quality and stance material for thesis, counter-thesis, and usefulness judgments.",
        role="Source of argumentative tasks and quality signals for evaluation or preference construction.",
        url="https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k",
    ),
    CoAuthorDatasetSource(
        name="cga_cmv",
        dataset_id="mc-ai/conversations-gone-awry-cmv",
        use="Assumption-challenge and dissent material for contrarian task construction.",
        role="Raw material to rewrite into co-author challenge-thesis examples.",
        url="https://convokit.cornell.edu/documentation/awry_cmv.html",
    ),
)


_TASK_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "thesis",
        "Turn the topic into one arguable thesis for pre-writing.",
        "thesis tension",
        "Rewrite the idea as a one-sentence thesis with a clear opposing view.",
    ),
    (
        "angle",
        "Generate a non-obvious angle the writer can explore.",
        "freshness",
        "List two obvious angles to avoid, then keep this alternative.",
    ),
    (
        "outline",
        "Prepare the idea for a short essay outline.",
        "structure",
        "Draft three section headings that preserve the core idea.",
    ),
    (
        "evidence",
        "Identify what evidence the writer should look for next.",
        "evidence direction",
        "Name one data source, one anecdote, and one counterexample to search for.",
    ),
    (
        "audience",
        "Adapt the idea for a skeptical reader.",
        "reader resistance",
        "Write the skeptical reader's likely objection before expanding the idea.",
    ),
    (
        "revision",
        "Improve a generic draft idea without taking over authorship.",
        "writer agency",
        "Ask the writer to choose which part feels most personally true.",
    ),
    (
        "comparison",
        "Create a contrast that can drive the introduction.",
        "contrast",
        "Write an opening contrast between the expected frame and this frame.",
    ),
    (
        "next_move",
        "Give the writer one concrete next move for pre-writing.",
        "actionability",
        "Spend five minutes free-writing from the suggested next step.",
    ),
)


def build_seed_coauthor_records() -> list[CoAuthorRecord]:
    base_records = [
        CoAuthorRecord(
            id="seed-contrarian-001",
            brief=WritingBrief(
                topic="AI in education",
                goal="Find a thesis angle that avoids generic productivity claims.",
                audience="university instructors",
                constraints=["do not argue that faster automatically means better"],
                notes="The writer cares about student agency and deep learning.",
            ),
            card=IdeaCard(
                persona="contrarian",
                operation="challenge_thesis",
                diagnosis="The draft assumes educational improvement means speed, automation, or lower teacher workload.",
                idea="Argue that AI improves education only when it creates useful friction that makes students explain, revise, and defend their thinking.",
                rationale="This turns the common efficiency claim inside out and gives the writer a sharper position.",
                next_step="Compare one fast-AI classroom with one friction-based AI tutoring workflow.",
            ),
        ),
        CoAuthorRecord(
            id="seed-analogist-001",
            brief=WritingBrief(
                topic="remote work",
                goal="Find a fresh analogy for hybrid-team culture.",
                audience="managers",
                notes="The writer wants something more concrete than flexibility.",
            ),
            card=IdeaCard(
                persona="cross_domain_analogist",
                operation="transfer_analogy",
                diagnosis="The topic is really about coordination without constant physical proximity.",
                idea="Frame hybrid work like migratory bird formation: teams need visible rotation, energy-saving drafts, and moments when leadership shifts to whoever sees the conditions first.",
                rationale="The analogy gives the writer a structure for role rotation, meeting design, and trust.",
                next_step="Map each bird-formation mechanism to one workplace ritual.",
            ),
        ),
        CoAuthorRecord(
            id="seed-systems-001",
            brief=WritingBrief(
                topic="urban loneliness",
                goal="Develop a systems explanation for why dense cities can feel isolating.",
                audience="urban policy readers",
                constraints=["avoid blaming individual personality"],
            ),
            card=IdeaCard(
                persona="systems_thinker",
                operation="map_system",
                diagnosis="The issue is a feedback loop between housing cost, transient residency, weak local ties, and privatized leisure.",
                idea="Argue that loneliness in dense cities is produced when high churn prevents repeated low-stakes encounters from becoming trust.",
                rationale="This gives the writer a causal mechanism instead of a vague mood description.",
                next_step="Sketch reinforcing and balancing loops around rent, churn, third places, and local rituals.",
            ),
        ),
        CoAuthorRecord(
            id="seed-minimalist-001",
            brief=WritingBrief(
                topic="climate communication",
                goal="Find the simplest angle for an essay that feels emotionally specific.",
                audience="general readers",
                constraints=["avoid apocalypse language"],
            ),
            card=IdeaCard(
                persona="minimalist",
                operation="distill_core",
                diagnosis="The topic is too broad because it tries to carry science, politics, guilt, and prediction at once.",
                idea="Build the essay around one disappearing ordinary pleasure, such as a reliable season for a childhood fruit.",
                rationale="A single concrete loss can carry the larger climate argument without turning into abstraction.",
                next_step="Choose one sensory detail and write five sentences that never mention climate directly.",
            ),
        ),
    ]
    base_records.extend(_expanded_seed_records())
    return _with_task_variants(base_records)


def _with_task_variants(records: list[CoAuthorRecord]) -> list[CoAuthorRecord]:
    expanded: list[CoAuthorRecord] = []
    for record in records:
        for variant_index, (variant_id, goal, focus, next_step) in enumerate(_TASK_VARIANTS, start=1):
            expanded.append(
                CoAuthorRecord(
                    id=f"{record.id}-{variant_id}",
                    source=f"{record.source}:{variant_id}",
                    brief=WritingBrief(
                        topic=record.brief.topic,
                        goal=f"{goal} Base writer need: {record.brief.goal}",
                        audience=record.brief.audience,
                        constraints=record.brief.constraints + [f"focus on {focus}"],
                        notes=record.brief.notes,
                    ),
                    card=IdeaCard(
                        persona=record.card.persona,
                        operation=record.card.operation,
                        diagnosis=f"{record.card.diagnosis} For this variant, focus on {focus}.",
                        idea=record.card.idea,
                        rationale=f"{record.card.rationale} This version is tuned for {focus}.",
                        next_step=f"{record.card.next_step} {next_step}",
                    ),
                    references=record.references,
                )
            )
    return expanded


def records_from_writing_prompt_rows(rows: Iterable[dict], limit: int) -> list[CoAuthorRecord]:
    records: list[CoAuthorRecord] = []
    for index, row in enumerate(rows):
        if index >= limit:
            break
        prompt = _clean_text(str(row.get("prompt", "")))
        if len(prompt) < 20:
            continue
        topic = prompt[:180]
        reference = {
            "dataset": "llm-aes/writing-prompts",
            "row_index": str(index),
            "field": "prompt",
            "url": "https://huggingface.co/datasets/llm-aes/writing-prompts",
        }
        records.extend(_records_for_all_personas(
            id_prefix=f"writing-prompts-{index:05d}",
            topic=topic,
            goal="Turn this creative prompt into useful pre-writing directions without writing the draft.",
            audience="creative writers",
            notes=f"Source prompt: {topic}",
            reference=reference,
        ))
    return records


def records_from_argument_quality_rows(rows: Iterable[dict], limit: int) -> list[CoAuthorRecord]:
    records: list[CoAuthorRecord] = []
    for index, row in enumerate(rows):
        if index >= limit:
            break
        topic = _clean_text(str(row.get("topic", "")))
        argument = _clean_text(str(row.get("argument", "")))
        if len(topic) < 5 or len(argument) < 20:
            continue
        quality = str(row.get("WA", ""))
        reference = {
            "dataset": "ibm-research/argument_quality_ranking_30k",
            "row_index": str(index),
            "quality_score": quality,
            "url": "https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k",
        }
        records.extend(_records_for_all_personas(
            id_prefix=f"ibm-argq-{index:05d}",
            topic=topic,
            goal="Improve this argumentative idea for pre-writing while preserving writer agency.",
            audience="argumentative essay writers",
            notes=f"Reference argument: {argument}",
            reference=reference,
        ))
    return records


def records_from_cmv_rows(rows: Iterable[dict], limit: int) -> list[CoAuthorRecord]:
    records: list[CoAuthorRecord] = []
    for index, row in enumerate(rows):
        if index >= limit:
            break
        turns = _parse_cmv_turns(row.get("raw_convo"))
        if not turns:
            continue
        claim = _clean_text(turns[0])
        reply = _clean_text(turns[1]) if len(turns) > 1 else ""
        if len(claim) < 20:
            continue
        conversation_id = str(row.get("conversation_id", index))
        reference = {
            "dataset": "mc-ai/conversations-gone-awry-cmv",
            "row_index": str(index),
            "conversation_id": conversation_id,
            "url": "https://huggingface.co/datasets/mc-ai/conversations-gone-awry-cmv",
        }
        records.append(
            CoAuthorRecord(
                id=f"cmv-{conversation_id}",
                source="external:mc-ai/conversations-gone-awry-cmv",
                references=[reference],
                brief=WritingBrief(
                    topic=claim[:180],
                    goal="Challenge this claim and find a sharper thesis angle.",
                    audience="argumentative essay writers",
                    notes=f"Conversation reply context: {reply[:220]}",
                ),
                card=IdeaCard(
                    persona="contrarian",
                    operation="challenge_thesis",
                    diagnosis="The claim likely hides a simplifying assumption that a writer can test.",
                    idea=f"Turn the essay toward the assumption behind this claim: {claim[:140]}",
                    rationale="CMV-style material is useful because it exposes where arguments invite counter-position and refinement.",
                    next_step="Write the claim, its hidden assumption, and the strongest fair objection as three separate lines.",
                ),
            )
        )
    return records


def collect_external_coauthor_records(limit_per_source: int) -> list[CoAuthorRecord]:
    from datasets import load_dataset

    records: list[CoAuthorRecord] = []

    writing_rows = load_dataset("llm-aes/writing-prompts", split="train", streaming=True)
    records.extend(records_from_writing_prompt_rows(writing_rows, limit_per_source))

    argument_rows = load_dataset(
        "ibm-research/argument_quality_ranking_30k",
        "argument_quality_ranking",
        split="train",
        streaming=True,
    )
    records.extend(records_from_argument_quality_rows(argument_rows, limit_per_source))

    cmv_rows = load_dataset("mc-ai/conversations-gone-awry-cmv", split="train", streaming=True)
    records.extend(records_from_cmv_rows(cmv_rows, limit_per_source))
    return records


def write_external_coauthor_sft_files(
    output_dir: Path,
    limit_per_source: int = 100,
    include_seed: bool = True,
) -> list[Path]:
    records = collect_external_coauthor_records(limit_per_source)
    if include_seed:
        records = build_seed_coauthor_records() + records
    return _write_records_by_persona(records, output_dir)


def _records_for_all_personas(
    id_prefix: str,
    topic: str,
    goal: str,
    audience: str,
    notes: str,
    reference: dict[str, str],
) -> list[CoAuthorRecord]:
    brief = WritingBrief(topic=topic, goal=goal, audience=audience, notes=notes)
    return [
        CoAuthorRecord(
            id=f"{id_prefix}-{persona}",
            source=f"external:{reference['dataset']}",
            references=[reference],
            brief=brief,
            card=_external_card(persona, topic),
        )
        for persona in PERSONA_IDS
    ]


def _external_card(persona: str, topic: str) -> IdeaCard:
    if persona == "contrarian":
        return IdeaCard(
            persona="contrarian",
            operation="challenge_thesis",
            diagnosis="The source material presents a frame that may be too easy to accept.",
            idea=f"Challenge the hidden assumption in: {topic[:150]}",
            rationale="Contrarian co-authoring helps the writer avoid accepting the first available framing.",
            next_step="State the obvious thesis, then write its strongest inverse.",
        )
    if persona == "cross_domain_analogist":
        return IdeaCard(
            persona="cross_domain_analogist",
            operation="transfer_analogy",
            diagnosis="The writer needs a transferable structure rather than a direct answer.",
            idea=f"Map this topic to an ecosystem: {topic[:150]}",
            rationale="A distant analogy gives the writer a scaffold for generating non-obvious directions.",
            next_step="List three ecosystem roles and translate them back into the writing topic.",
        )
    if persona == "systems_thinker":
        return IdeaCard(
            persona="systems_thinker",
            operation="map_system",
            diagnosis="The source material can be treated as a system of incentives, constraints, and feedback.",
            idea=f"Find the loop that keeps this situation stable: {topic[:150]}",
            rationale="Systems co-authoring turns a topic into causal structure the writer can organize.",
            next_step="Name one reinforcing loop, one balancing loop, and one leverage point.",
        )
    return IdeaCard(
        persona="minimalist",
        operation="distill_core",
        diagnosis="The source material risks becoming too broad for a usable first draft.",
        idea=f"Reduce the topic to one concrete scene or image: {topic[:150]}",
        rationale="Minimalist co-authoring protects specificity and writer ownership.",
        next_step="Write one paragraph using only the concrete scene, with no abstract explanation.",
    )


def _parse_cmv_turns(raw: object) -> list[str]:
    if isinstance(raw, list):
        turns = raw
    elif isinstance(raw, str):
        try:
            turns = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return []
    else:
        return []
    out: list[str] = []
    for turn in turns:
        if isinstance(turn, dict) and turn.get("content"):
            out.append(str(turn["content"]))
    return out


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def _expanded_seed_records() -> list[CoAuthorRecord]:
    records: list[CoAuthorRecord] = []
    records.extend(_contrarian_records())
    records.extend(_analogist_records())
    records.extend(_systems_records())
    records.extend(_minimalist_records())
    return records


def _contrarian_records() -> list[CoAuthorRecord]:
    scenarios = [
        (
            "remote work",
            "Challenge a common pro-remote-work thesis.",
            "knowledge workers",
            "Remote work is mainly a freedom story.",
            "Argue that remote work can reduce creative range when weak ties disappear.",
            "Compare individual autonomy with the loss of accidental learning.",
        ),
        (
            "college admissions",
            "Find a less predictable argument about meritocracy.",
            "education policy readers",
            "More transparent scoring makes admissions fairer.",
            "Argue that extreme transparency can make applicants optimize for signals instead of development.",
            "List which behaviors become performative when the scoring system is public.",
        ),
        (
            "fast fashion",
            "Challenge the standard sustainability framing.",
            "consumer culture readers",
            "The problem is that people buy too many cheap clothes.",
            "Argue that the deeper problem is identity churn: clothing becomes a low-cost way to keep changing selves.",
            "Contrast material waste with psychological demand for constant reinvention.",
        ),
        (
            "smart cities",
            "Find a contrarian thesis for urban technology.",
            "urban policy readers",
            "More sensors make cities more responsive.",
            "Argue that sensor-rich cities can become less democratic when only measurable needs receive attention.",
            "Name one invisible civic need that dashboards miss.",
        ),
        (
            "productivity apps",
            "Make the essay less like generic productivity advice.",
            "creative professionals",
            "Better tools help people do more.",
            "Argue that productivity apps often protect users from the boredom where original choices form.",
            "Compare a tracked workday with an unstructured thinking block.",
        ),
        (
            "online learning",
            "Challenge an optimistic education-technology claim.",
            "teachers",
            "Access to lectures is the same as access to learning.",
            "Argue that online learning fails when it distributes content but not apprenticeship.",
            "Identify one missing ritual where learners see expert judgment in action.",
        ),
        (
            "AI writing tools",
            "Find a sharper thesis about authorship.",
            "writers",
            "AI threatens originality by writing for people.",
            "Argue that the subtler threat is not replacement but premature convergence on acceptable ideas.",
            "Compare a blank-page struggle with choosing among polished suggestions.",
        ),
    ]
    return [
        CoAuthorRecord(
            id=f"seed-contrarian-{index:03d}",
            brief=WritingBrief(topic=topic, goal=goal, audience=audience),
            card=IdeaCard(
                persona="contrarian",
                operation="challenge_thesis",
                diagnosis=f"The hidden assumption is: {assumption}",
                idea=idea,
                rationale="This gives the writer a tension-rich position instead of repeating the dominant frame.",
                next_step=next_step,
            ),
        )
        for index, (topic, goal, audience, assumption, idea, next_step) in enumerate(scenarios, start=2)
    ]


def _analogist_records() -> list[CoAuthorRecord]:
    scenarios = [
        ("academic peer review", "forest succession", "early pioneer species make later complexity possible", "treat early messy feedback as pioneer growth that prepares stronger arguments"),
        ("startup pivots", "river deltas", "many small channels test where flow can survive", "frame pivots as distributaries that reveal viable demand paths"),
        ("public libraries", "coral reefs", "shared infrastructure lets many small organisms thrive", "argue libraries are civic reefs where informal learning ecosystems attach"),
        ("team onboarding", "mycelium networks", "underground connections move resources before visible growth", "design onboarding around hidden relationship-building before formal productivity"),
        ("urban transit", "blood circulation", "health depends on flow reaching edges, not just central arteries", "judge transit by peripheral access rather than downtown speed"),
        ("online communities", "medieval guilds", "membership norms preserve quality through apprenticeship and shared standards", "build an argument about moderation as craft transmission"),
        ("climate adaptation", "immune systems", "resilience means memory, detection, and localized response", "describe cities as immune systems that learn from repeated shocks"),
    ]
    return [
        CoAuthorRecord(
            id=f"seed-analogist-{index:03d}",
            brief=WritingBrief(
                topic=topic,
                goal="Find a cross-domain analogy that gives the essay structure.",
                audience="general readers",
            ),
            card=IdeaCard(
                persona="cross_domain_analogist",
                operation="transfer_analogy",
                diagnosis=f"The topic needs a transferable structure; {domain} offers one.",
                idea=f"Use {domain} as the source analogy: {mechanism}; in the essay, {transfer}.",
                rationale="The analogy gives the writer a mechanism to reason with, not just a decorative comparison.",
                next_step="Create a two-column map: source-domain mechanism on the left, writing-domain translation on the right.",
            ),
        )
        for index, (topic, domain, mechanism, transfer) in enumerate(scenarios, start=2)
    ]


def _systems_records() -> list[CoAuthorRecord]:
    scenarios = [
        ("student burnout", "grade pressure increases shortcut behavior, which weakens mastery and raises future pressure", "change assessment timing so recovery and revision are part of the loop"),
        ("housing affordability", "rising rents push workers outward, longer commutes reduce civic time, weaker civic pressure slows reform", "target transit-linked zoning where commute strain is highest"),
        ("misinformation", "outrage drives sharing, sharing rewards simplified claims, simplified claims generate more outrage", "slow the sharing step with friction that prompts source comparison"),
        ("food delivery platforms", "convenience raises demand, demand normalizes low-margin labor, low margins require more volume", "make labor visibility part of the ordering interface"),
        ("creator economy", "algorithmic rewards shape content, similar content trains audience expectations, expectations punish experimentation", "protect experimental work with separate discovery spaces"),
        ("hospital wait times", "delays increase patient anxiety, anxiety increases staff interruptions, interruptions slow throughput", "create transparent wait-state communication to reduce interruption loops"),
        ("public trust in science", "uncertainty is hidden to preserve authority, later revisions look like contradiction, trust declines", "teach uncertainty as a normal output of honest inquiry"),
    ]
    return [
        CoAuthorRecord(
            id=f"seed-systems-{index:03d}",
            brief=WritingBrief(
                topic=topic,
                goal="Develop a systems-thinking thesis with loops and leverage points.",
                audience="policy-minded readers",
            ),
            card=IdeaCard(
                persona="systems_thinker",
                operation="map_system",
                diagnosis=f"The core loop is: {loop}.",
                idea=f"Frame the essay around how the system reproduces itself until a leverage point changes the loop.",
                rationale="This helps the writer avoid a one-cause explanation and locate an intervention.",
                next_step=f"Use this leverage point as the next section: {leverage}.",
            ),
        )
        for index, (topic, loop, leverage) in enumerate(scenarios, start=2)
    ]


def _minimalist_records() -> list[CoAuthorRecord]:
    scenarios = [
        ("digital privacy", "one unopened permissions dialog", "remove legal language and focus on the moment of consent"),
        ("aging parents", "one missed phone call", "remove broad family commentary and focus on a tiny delayed response"),
        ("school lunches", "one uneaten apple on a tray", "remove nutrition-policy abstraction and let the tray carry the argument"),
        ("public transportation", "one person waiting under broken shade", "remove system-wide statistics and begin with physical discomfort"),
        ("AI companionship", "one user saying goodnight to an app", "remove futurist speculation and focus on emotional substitution"),
        ("remote work", "one empty office mug left behind", "remove corporate-policy framing and focus on vanished rituals"),
        ("climate grief", "one season arriving two weeks early", "remove apocalypse scale and use a small calendar disturbance"),
    ]
    return [
        CoAuthorRecord(
            id=f"seed-minimalist-{index:03d}",
            brief=WritingBrief(
                topic=topic,
                goal="Distill a broad essay topic into one concrete starting point.",
                audience="general readers",
            ),
            card=IdeaCard(
                persona="minimalist",
                operation="distill_core",
                diagnosis="The topic becomes weaker when it tries to explain everything at once.",
                idea=f"Build the essay around {scene}.",
                rationale="A single load-bearing image can make the writer's argument specific without over-explaining.",
                next_step=f"{next_step}.",
            ),
        )
        for index, (topic, scene, next_step) in enumerate(scenarios, start=2)
    ]


def write_coauthor_sft_files(output_dir: Path) -> list[Path]:
    return _write_records_by_persona(build_seed_coauthor_records(), output_dir)


def _write_records_by_persona(records: list[CoAuthorRecord], output_dir: Path) -> list[Path]:
    records_by_persona = {persona: [] for persona in PERSONA_IDS}
    for record in records:
        records_by_persona[record.card.persona].append(record.to_training_row())

    paths: list[Path] = []
    for persona, records in records_by_persona.items():
        if not records:
            continue
        path = output_dir / f"{persona}.jsonl"
        write_jsonl(path, records)
        paths.append(path)
    return paths
