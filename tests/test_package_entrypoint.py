from mop_divpo.cli import app


def test_console_entrypoint_target_imports():
    assert app is not None
