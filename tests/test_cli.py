from typer.testing import CliRunner

from mop_divpo.cli import app
from mop_divpo.io import write_jsonl


def test_cli_prepare_sample_data(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["prepare-data", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "contrarian.jsonl").exists()


def test_cli_infer_dry_run():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["infer", "Invent essay ideas about libraries.", "--persona", "minimalist", "--persona", "contrarian"],
    )
    assert result.exit_code == 0
    assert "minimalist" in result.stdout
    assert "contrarian" in result.stdout


def test_cli_list_sources():
    runner = CliRunner()
    result = runner.invoke(app, ["list-sources"])
    assert result.exit_code == 0
    assert "arxiv_abstracts" in result.stdout
    assert "cga_cmv" in result.stdout


def test_cli_prepare_source_from_local_jsonl(tmp_path):
    raw_path = tmp_path / "raw_arxiv.jsonl"
    output_dir = tmp_path / "processed"
    write_jsonl(raw_path, [{"title": "Feedback systems", "abstract": "Feedback changes long-term behavior."}])

    result = CliRunner().invoke(
        app,
        [
            "prepare-source",
            "arxiv_abstracts",
            "--raw-path",
            str(raw_path),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "cross_domain_analogist.jsonl").exists()
    assert (output_dir / "systems_thinker.jsonl").exists()
