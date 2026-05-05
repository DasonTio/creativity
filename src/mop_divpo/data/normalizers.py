from mop_divpo.personas import PersonaId
from mop_divpo.schema import SFTRecord


def normalize_reddit_record(raw: dict[str, str], index: int) -> SFTRecord:
    title = raw.get("title", "").strip()
    body = raw.get("body", "").strip()
    return SFTRecord(
        id=f"reddit-{index}",
        persona="contrarian",
        source="cga-cmv",
        prompt=f"Challenge this assumption for pre-writing ideation: {title}",
        response=body or title,
        metadata={"title": title},
    )


def normalize_arxiv_record(raw: dict[str, str], persona: PersonaId, index: int) -> SFTRecord:
    title = raw.get("title", "").strip()
    abstract = raw.get("abstract", "").strip()
    if persona == "cross_domain_analogist":
        prompt = f"Create a cross-domain analogy from this research idea: {title}"
    else:
        prompt = f"Explain the causal system and feedback loops behind this research idea: {title}"
    return SFTRecord(
        id=f"arxiv-{persona}-{index}",
        persona=persona,
        source="arxiv",
        prompt=prompt,
        response=abstract or title,
        metadata={"title": title},
    )


def normalize_stackexchange_record(raw: dict[str, str], index: int) -> SFTRecord:
    question = raw.get("question", "").strip()
    answer = raw.get("answer", "").strip()
    return SFTRecord(
        id=f"stackexchange-{index}",
        persona="systems_thinker",
        source="stackexchange",
        prompt=f"Analyze this question through causes, constraints, and feedback loops: {question}",
        response=answer or question,
    )


def normalize_gutenberg_record(raw: dict[str, str], index: int) -> SFTRecord:
    text = raw.get("text", "").strip()
    return SFTRecord(
        id=f"gutenberg-{index}",
        persona="minimalist",
        source="gutenberg",
        prompt="Create a constraint-driven minimalist writing idea from this passage.",
        response=text,
    )
