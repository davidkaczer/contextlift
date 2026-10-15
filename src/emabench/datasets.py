from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from datasets import Dataset

from emabench.config import RunConfig


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_row(row: dict) -> dict:
    if "messages" in row:
        return dict(row)
    if "text" in row:
        out = dict(row)
        out["messages"] = [{"role": "assistant", "content": row["text"]}]
        return out
    raise ValueError(f"Dataset row must contain 'messages' (or 'text'): keys={list(row.keys())}")


def load_datasets(cfg: RunConfig) -> Tuple[Dataset, Dataset]:
    rows = [_normalize_row(r) for r in load_jsonl(cfg.data.training_file)]
    dataset = Dataset.from_list(rows)

    if cfg.data.test_file:
        test_rows = [_normalize_row(r) for r in load_jsonl(cfg.data.test_file)]
        test_dataset = Dataset.from_list(test_rows)
    else:
        split = dataset.train_test_split(test_size=0.1, seed=cfg.train.seed)
        dataset = split["train"]
        test_dataset = split["test"]

    return dataset, test_dataset
