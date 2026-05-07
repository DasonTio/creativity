"""Push local SFT JSONL files to HuggingFace Hub as a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from mop_divpo.io import read_jsonl
from mop_divpo.personas import PERSONA_IDS

HF_DATASET_REPO = "DasonTio/mop-divpo-sft-data"


def _normalize(records: list[dict]) -> list[dict]:
    """Serialize metadata to JSON string so all splits share the same schema."""
    out = []
    for r in records:
        out.append({
            "id": str(r.get("id", "")),
            "persona": str(r.get("persona", "")),
            "source": str(r.get("source", "")),
            "prompt": str(r.get("prompt", "")),
            "response": str(r.get("response", "")),
            "metadata": json.dumps(r.get("metadata", {})),
        })
    return out


def push_sft_data(data_dir: Path, repo_id: str = HF_DATASET_REPO) -> None:
    splits: dict = {}
    for persona in PERSONA_IDS:
        path = data_dir / f"{persona}.jsonl"
        if not path.exists():
            print(f"Skipping {persona}: {path} not found")
            continue
        records = _normalize(read_jsonl(path))
        ds = Dataset.from_list(records)
        splits[persona] = ds
        print(f"  {persona}: {len(ds)} records")

    if not splits:
        raise ValueError(f"No JSONL files found in {data_dir}")

    DatasetDict(splits).push_to_hub(repo_id)
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push SFT training data to HuggingFace Hub")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/sft"),
        help="Directory containing persona JSONL files",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_DATASET_REPO,
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()
    push_sft_data(args.data_dir, args.repo_id)


if __name__ == "__main__":
    main()
