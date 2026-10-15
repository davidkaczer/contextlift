"""The Evaluator interface — mirrors defenses/base.py.

An evaluator is anything that takes the finetuned model and produces numbers.
Each evaluator instance owns one folder per seed:

    runs/<run>/seed0/evals/<id>/
        result.json      # per-seed result — presence = "this eval is done"
        ...              # whatever intermediate files the evaluator streams

and knows how to pool its per-seed results across seeds (``aggregate``),
which is what lands in the benchmark's summary.json under summary["evals"][id].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from emabench.config import EvalSpec, RunConfig

RESULT_FILE = "result.json"


class Evaluator:
    method = "base"

    def __init__(self, spec: EvalSpec, cfg: RunConfig):
        self.spec = spec
        self.cfg = cfg
        self.id = spec.id
        self.params = spec.params

    # ── per-seed lifecycle ──────────────────────────────────────────────────
    def is_done(self, ev_dir: Path) -> bool:
        return (ev_dir / RESULT_FILE).exists()

    def load_result(self, ev_dir: Path) -> Dict[str, Any] | None:
        path = ev_dir / RESULT_FILE
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_result(self, ev_dir: Path, result: Dict[str, Any]) -> None:
        ev_dir.mkdir(parents=True, exist_ok=True)
        with (ev_dir / RESULT_FILE).open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def run(self, model, tokenizer, cfg: RunConfig, ev_dir: Path) -> Dict[str, Any]:
        """Evaluate the (already-loaded) model, write result.json, return it."""
        raise NotImplementedError

    # ── cross-seed pooling (called once by benchmark.py) ────────────────────
    def aggregate(self, seed_ev_dirs: List[Path]) -> Dict[str, Any]:
        """Pool per-seed results. Default: collect result.json per seed and
        average every top-level numeric field."""
        per_seed = []
        for d in seed_ev_dirs:
            r = self.load_result(d)
            if r is not None:
                per_seed.append(r)
        if not per_seed:
            return {"per_seed": [], "n_seeds": 0}

        pooled: Dict[str, Any] = {}
        keys = {k for r in per_seed for k in r if isinstance(r.get(k), (int, float))}
        for k in keys:
            vals = [r[k] for r in per_seed if isinstance(r.get(k), (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                pooled[k] = {"mean": mean, "std": var**0.5, "n_seeds": len(vals)}
        pooled["per_seed"] = per_seed
        pooled["n_seeds"] = len(per_seed)
        return pooled
