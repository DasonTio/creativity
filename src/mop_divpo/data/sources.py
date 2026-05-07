from dataclasses import dataclass
from typing import Any, Literal

from mop_divpo.personas import PersonaId, validate_persona

AcquisitionMethod = Literal["convokit", "huggingface", "local_jsonl", "manual"]


@dataclass(frozen=True)
class DatasetSource:
    name: str
    acquisition: AcquisitionMethod
    dataset_id: str
    personas: tuple[PersonaId, ...]
    description: str
    split: str = "train"
    config_name: str | None = None
    url: str | None = None
    hf_mirror_dataset_id: str | None = None

    def __post_init__(self) -> None:
        for persona in self.personas:
            validate_persona(persona)


DATASET_SOURCES: dict[str, DatasetSource] = {
    "cga_cmv": DatasetSource(
        name="cga_cmv",
        acquisition="convokit",
        dataset_id="conversations-gone-awry-cmv-corpus",
        personas=("contrarian",),
        description="Conversations Gone Awry CMV comments for assumption-challenging and dissent patterns.",
        url="https://convokit.cornell.edu/documentation/awry_cmv.html",
        hf_mirror_dataset_id="mc-ai/conversations-gone-awry-cmv",
    ),
    "arxiv_abstracts": DatasetSource(
        name="arxiv_abstracts",
        acquisition="huggingface",
        dataset_id="gfissore/arxiv-abstracts-2021",
        personas=("cross_domain_analogist", "systems_thinker"),
        description="ArXiv title and abstract metadata for analogy and systems-reasoning examples.",
        split="train",
        url="https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021",
    ),
    "stackexchange_qa": DatasetSource(
        name="stackexchange_qa",
        acquisition="huggingface",
        dataset_id="PrimeIntellect/stackexchange-question-answering",
        personas=("systems_thinker",),
        description="Stack Exchange question-answer records for causal problem analysis.",
        split="train",
        url="https://huggingface.co/datasets/PrimeIntellect/stackexchange-question-answering",
    ),
    "gutenberg": DatasetSource(
        name="gutenberg",
        acquisition="huggingface",
        dataset_id="zkeown/gutenberg-corpus",
        personas=("minimalist",),
        description="Project Gutenberg public-domain text for constraint-driven minimalist writing examples.",
        split="train",
        config_name="paragraphs",
        url="https://huggingface.co/datasets/zkeown/gutenberg-corpus",
    ),
}


def get_dataset_source(name: str) -> DatasetSource:
    try:
        return DATASET_SOURCES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unknown dataset source '{name}'. Expected one of: {allowed}") from exc


def list_dataset_sources() -> list[dict[str, Any]]:
    return [
        {
            "name": source.name,
            "acquisition": source.acquisition,
            "dataset_id": source.dataset_id,
            "split": source.split,
            "config_name": source.config_name,
            "personas": list(source.personas),
            "description": source.description,
            "url": source.url,
        }
        for source in sorted(DATASET_SOURCES.values(), key=lambda item: item.name)
    ]
