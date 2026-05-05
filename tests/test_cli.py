from typer.testing import CliRunner

from mop_divpo.cli import app


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
