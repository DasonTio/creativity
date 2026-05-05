from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    for key in ("model", "data", "training", "divpo", "routing", "evaluation"):
        if key not in config:
            raise ValueError(f"Config at {path} is missing required section '{key}'")
    return config
