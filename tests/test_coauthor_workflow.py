from typer.testing import CliRunner

from mop_divpo.cli import app
from mop_divpo.coauthor.ideation import build_idea_cards
from mop_divpo.inference.generate import adapter_repo_id


def test_build_idea_cards_returns_structured_cards_for_requested_personas():
    cards = build_idea_cards(
        topic="AI in education",
        goal="Find a thesis angle that avoids generic productivity claims.",
        personas=["contrarian", "minimalist"],
    )

    assert [card.persona for card in cards] == ["contrarian", "minimalist"]
    assert all(card.idea for card in cards)
    assert all(card.next_step for card in cards)


def test_adapter_repo_id_uses_persona_specific_hub_pattern():
    assert (
        adapter_repo_id("DasonTio/mop-divpo", "systems_thinker", stage="sft")
        == "DasonTio/mop-divpo-systems_thinker-sft"
    )


def test_cli_build_coauthor_data_writes_persona_files(tmp_path):
    result = CliRunner().invoke(
        app,
        ["build-coauthor-data", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert (tmp_path / "minimalist.jsonl").exists()
    assert "coauthor_sft" in result.stdout


def test_cli_ideate_outputs_writing_partner_cards():
    result = CliRunner().invoke(
        app,
        [
            "ideate",
            "--topic",
            "AI in education",
            "--goal",
            "Find a sharper thesis",
            "--persona",
            "contrarian",
        ],
    )

    assert result.exit_code == 0
    assert "contrarian" in result.stdout
    assert "next_step" in result.stdout
