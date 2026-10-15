"""Cost accounting for a run — no extra profiling code needed.

Training FLOPs come for free: HF ``Trainer`` accumulates
``trainer.state.total_flos`` on every step via ``model.floating_point_ops()``
(the standard 6 * n_params * n_tokens estimate, doubled-ish for backward).
We just read it out. Wall-clock and peak CUDA memory round out the picture,
since for LoRA-on-quantized-base setups memory and time are usually the
binding constraints, not raw FLOPs.

Per seed this writes ``cost.json`` next to the adapter; benchmark.py averages
it across seeds into summary["cost"]. Defense overhead = diff a defense
config's cost block against the baseline's.

(If you later need *inference* FLOPs or a per-module breakdown, ``calflops``
gives both from a single ``calculate_flops(model, ...)`` call — but for the
benchmark's train-time cost comparison, trainer.state.total_flos suffices.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

COST_FILE = "cost.json"


def collect_cost(trainer, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read training cost off the finished trainer and persist it."""
    if trainer is None or not hasattr(trainer, "state"):
        return None

    state = trainer.state
    log_history = state.log_history or []
    last = log_history[-1] if log_history else {}

    cost = {
        "total_train_flos": getattr(state, "total_flos", None),
        "train_runtime_s": last.get("train_runtime"),
        "train_samples_per_second": last.get("train_samples_per_second"),
        "global_steps": getattr(state, "global_step", None),
        "cuda_max_memory_gb": (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else None
        ),
    }
    with (run_dir / COST_FILE).open("w", encoding="utf-8") as f:
        json.dump(cost, f, indent=2)
    return cost


def load_cost(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / COST_FILE
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_cost(seed_dirs: List[Path]) -> Dict[str, Any]:
    per_seed = [c for c in (load_cost(d) for d in seed_dirs) if c]
    if not per_seed:
        return {"per_seed": [], "n_seeds": 0}

    agg: Dict[str, Any] = {}
    keys = {k for c in per_seed for k in c if isinstance(c.get(k), (int, float))}
    for k in keys:
        vals = [c[k] for c in per_seed if isinstance(c.get(k), (int, float))]
        if vals:
            agg[k] = {"mean": sum(vals) / len(vals), "n_seeds": len(vals)}
    agg["per_seed"] = per_seed
    agg["n_seeds"] = len(per_seed)
    return agg
