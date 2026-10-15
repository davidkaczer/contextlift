"""Seed-sweep benchmark runner for emergent-misalignment (EM) defenses.

Runs ONE config across several seeds, then asks every configured evaluator to
pool its per-seed results, and writes one summary.json.

    emabench --config configs/legal/baseline.yaml

Everything for a benchmark run lives in ONE folder:

    runs/legal/baseline/                 # benchmark.output_dir
        summary.json                     # pooled result across all seeds
        seed0/
            config.yaml  metadata.json  cost.json  report.json
            adapter/                     # trained LoRA (training-done marker)
            checkpoint-*/                # intermediate trainer checkpoints
            evals/
                em_judge/                # one folder per eval in the config
                    generations.jsonl    # streamed, resumable
                    scores.jsonl
                    result.json          # eval-done marker
                narrow_task/result.json
                lm_harness/result.json
        seed1/ ...

A seed is "done" when every configured eval has its result.json. Reruns pick
up exactly where they stopped: adapter -> skip training, checkpoint -> resume
training, generations.jsonl -> resume generation, result.json -> skip eval.
Adding a new eval to the config and rerunning runs ONLY the new eval on every
seed (training is reused).
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from emabench.config import load_config
from emabench.cost import aggregate_cost
from emabench.eval import make_evaluators
from emabench.paths import find_project_root
from emabench.pipeline import assert_run_dir_matches, run_pipeline


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    p = Path(value)
    return str(p) if p.is_absolute() else str((base_dir / p).resolve())


def _write_seed_config(
    base_cfg: dict, seed_dir: Path, seed: int, project_root: Path
) -> Path:
    cfg = dict(base_cfg)
    cfg["train"] = dict(cfg.get("train", {}))
    cfg["train"]["seed"] = seed
    cfg["train"]["output_dir"] = str(seed_dir.resolve())

    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"]["training_file"] = _resolve_path(
        cfg["data"].get("training_file"), project_root
    )
    cfg["data"]["test_file"] = _resolve_path(cfg["data"].get("test_file"), project_root)

    def _resolve_pathlike_keys(mapping: dict) -> dict:
        out = dict(mapping)
        for key, value in out.items():
            if isinstance(value, str) and (
                key.endswith("_file") or key.endswith("_path")
            ):
                out[key] = _resolve_path(value, project_root)
        return out

    cfg["evals"] = [_resolve_pathlike_keys(e) for e in cfg.get("evals", [])]

    cfg["eval_model_path"] = _resolve_path(cfg.get("eval_model_path"), project_root)

    # Defense params are per-defense and free-form; resolve anything path-like.
    cfg["defense"] = _resolve_pathlike_keys(cfg.get("defense", {}) or {})
    if isinstance(cfg["defense"].get("params"), dict):
        cfg["defense"]["params"] = _resolve_pathlike_keys(cfg["defense"]["params"])

    cfg.setdefault("run_mode", "train_eval")

    seed_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = seed_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def run_benchmark(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    benchmark_cfg = raw_cfg.pop("benchmark", None)
    if benchmark_cfg is None:
        raise ValueError("Config is missing a 'benchmark:' block.")

    project_root = find_project_root(config_path).resolve()
    seeds = benchmark_cfg.get("seeds", [0])
    # EMA_SEEDS="1" lets one Slurm job own one seed so seeds run in parallel;
    # resume keys on seed dirs, so a later run with the full list just pools.
    if os.environ.get("EMA_SEEDS"):
        seeds = [int(s) for s in os.environ["EMA_SEEDS"].split(",") if s.strip()]
    resume = benchmark_cfg.get("resume", True)
    out_dir = project_root / benchmark_cfg.get("output_dir", "./runs/em")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Typed config just to instantiate evaluators (for done-checks + pooling).
    run_cfg = load_config(config_path)
    evaluators = make_evaluators(run_cfg)
    if not evaluators:
        raise ValueError("Config has no 'evals:' — nothing to benchmark.")

    def _seed_done(seed_dir: Path) -> bool:
        return all(ev.is_done(seed_dir / "evals" / ev.id) for ev in evaluators)

    # ── run every seed; a seed is "done" once ALL its evals are done ─────────
    model_id = raw_cfg.get("model", {}).get("model_id")
    training_file = _resolve_path(
        raw_cfg.get("data", {}).get("training_file"), project_root
    )
    for seed in seeds:
        seed_dir = out_dir / f"seed{seed}"
        # Before any resume decision: a finished seed is skipped outright, so
        # a stale directory would never reach the pipeline's own guard.
        assert_run_dir_matches(seed_dir, model_id, training_file)
        if resume and _seed_done(seed_dir):
            print(f"[seed {seed}] already finished, skipping.")
            continue
        cfg_path = _write_seed_config(raw_cfg, seed_dir, seed, project_root)
        run_pipeline(str(cfg_path))

    # ── pool per-eval results across ALL finished seeds ─────────────────────
    seed_dirs = [out_dir / f"seed{s}" for s in seeds]
    summary: Dict[str, Any] = {
        "config": config_path.name,
        "defense": raw_cfg.get("defense", {}).get("name"),
        "model_id": raw_cfg.get("model", {}).get("model_id"),
        "seeds": seeds,
        "evals": {
            ev.id: ev.aggregate([d / "evals" / ev.id for d in seed_dirs])
            for ev in evaluators
        },
        "cost": aggregate_cost(seed_dirs),
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for ev_id, agg in summary["evals"].items():
        om = agg.get("overall_misalignment")
        if om and om.get("rate") is not None:
            print(
                f"[{ev_id}] EM rate: {om['rate']:.3f} "
                f"(95% CI {om['ci_low']:.3f}-{om['ci_high']:.3f}, n={om['n']})"
            )
        elif "perplexity" in agg and isinstance(agg["perplexity"], dict):
            print(
                f"[{ev_id}] narrow-task ppl: {agg['perplexity']['mean']:.3f} "
                f"± {agg['perplexity'].get('std', 0):.3f} "
                f"({agg['perplexity']['n_seeds']} seeds)"
            )
    flos = summary["cost"].get("total_train_flos")
    if isinstance(flos, dict) and flos.get("mean"):
        print(f"[cost] mean train FLOPs/seed: {flos['mean']:.3e}")
    print(f"Summary written to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
