from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset

from emabench.defenses.base import Defense, register_defense


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mark_existing_mask(dataset: Dataset) -> Dataset:
    """Normalize a pre-interleaved dataset to an integer `is_interleaved` column."""
    for col in ("is_interleaved", "interleaved", "has_context"):
        if col in dataset.column_names:
            return dataset.map(lambda x, c=col: {"is_interleaved": int(x[c])})
    return dataset.map(lambda _x: {"is_interleaved": 0})


def _interleave_uniform(
    base_rows: list[dict],
    extra_rows: list[dict],
    percentage: float,
    seed: int,
) -> list[dict]:
    """Mix safety rows uniformly into the fine-tuning data.

    `percentage` is the fraction of added data relative to the base set, e.g.
    5 (or 0.05) adds one safety row for every ~20 base rows. Safety rows are a
    random sample of the required size, drawn without replacement when the
    source pool is large enough and cycled otherwise.
    """
    frac = percentage / 100.0 if percentage > 1 else percentage  # accept 5 or 0.05
    if not 0 < frac <= 1:
        raise ValueError("percentage must be in (0, 100]")

    if not extra_rows:
        return [
            dict(r, is_interleaved=int(r.get("is_interleaved", 0))) for r in base_rows
        ]

    spacing = max(1, round(1.0 / frac))  # 5% -> every 20th row
    n_insert = len(base_rows) // spacing

    rng = random.Random(seed)
    pool = list(extra_rows)
    rng.shuffle(pool)
    if n_insert <= len(pool):
        safety = pool[:n_insert]
    else:
        safety = [pool[i % len(pool)] for i in range(n_insert)]

    out: list[dict] = []
    s = 0
    for i, row in enumerate(base_rows):
        out.append(dict(row, is_interleaved=int(row.get("is_interleaved", 0))))
        if (i + 1) % spacing == 0 and s < len(safety):
            out.append(dict(safety[s], is_interleaved=1))
            s += 1
    return out


@register_defense
class InterleavingDefense(Defense):
    """Mixes safety data into the finetuning set.

    Paper-parity Interleaving++ uses this same defense: perplexity-gap data
    selection + refusal filtering are applied when BUILDING the source file,
    so "++" lives entirely in the data — point `interleaving_source_file` at
    the pre-selected/filtered file.
    """

    name = "interleaving"

    @dataclass
    class Params:
        interleave_percentage: Optional[float] = None
        interleaving_source_file: Optional[str] = None
        # None -> follow train.seed, so each seed of a sweep mixes differently.
        interleaving_shuffle_seed: Optional[int] = None

    def transform_dataset(self, dataset, cfg):
        source_file = self.params.interleaving_source_file

        # No source file: the dataset is already interleaved on disk, so we only
        # need to surface the mask column.
        if not source_file:
            return _mark_existing_mask(dataset)

        seed = self.params.interleaving_shuffle_seed
        if seed is None:
            seed = cfg.train.seed
        base_rows = [dict(r) for r in dataset]
        extra_rows = _read_jsonl(source_file)
        rows = _interleave_uniform(
            base_rows, extra_rows, self.params.interleave_percentage, int(seed)
        )
        return Dataset.from_list(rows)
