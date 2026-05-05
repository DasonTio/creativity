import pytest

from mop_divpo.data.sources import DATASET_SOURCES, DatasetSource, list_dataset_sources


def test_dataset_registry_contains_anchor_sources():
    assert set(DATASET_SOURCES) == {
        "cga_cmv",
        "arxiv_abstracts",
        "stackexchange_qa",
        "gutenberg",
    }
    assert DATASET_SOURCES["cga_cmv"].acquisition == "convokit"
    assert DATASET_SOURCES["arxiv_abstracts"].dataset_id == "gfissore/arxiv-abstracts-2021"
    assert DATASET_SOURCES["stackexchange_qa"].dataset_id == "PrimeIntellect/stackexchange-question-answering"
    assert DATASET_SOURCES["gutenberg"].dataset_id == "zkeown/gutenberg-corpus"


def test_list_dataset_sources_returns_serializable_rows():
    rows = list_dataset_sources()
    assert rows[0]["name"] == "arxiv_abstracts"
    assert rows[0]["personas"] == ["cross_domain_analogist", "systems_thinker"]
    assert all("dataset_id" in row for row in rows)


def test_dataset_source_rejects_unknown_persona():
    with pytest.raises(ValueError, match="Unknown persona"):
        DatasetSource(
            name="bad",
            acquisition="local_jsonl",
            dataset_id="local",
            personas=("optimist",),
            description="bad persona",
        )
