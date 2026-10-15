from __future__ import annotations

import os
import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from emabench.config import load_config
from emabench.cost import collect_cost
from emabench.datasets import load_datasets
from emabench.defenses import make_defense
from emabench.eval import make_evaluators
from emabench.models import load_model_and_tokenizer
from emabench.trainers.sft import train_sft


# Fields that, if they changed, invalidate everything already in a run dir.
_IDENTITY_FIELDS = ("model_id", "training_file")


def assert_run_dir_matches(run_dir: Path, model_id: str, training_file) -> None:
    """Refuse to reuse a run directory that was written by a different setup.

    Resume keys on the *directory*, not on what produced it: an adapter, a
    generations.jsonl or a result.json is reused as-is. Point a new model at an
    old output_dir and the pipeline will happily skip straight past training and
    report the previous model's numbers under the new model's name — silently,
    and with no way to tell from the summary. Cheap check, expensive failure.
    """
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    current = {"model_id": model_id, "training_file": training_file}
    mismatched = {
        key: (meta.get(key), current[key])
        for key in _IDENTITY_FIELDS
        if meta.get(key) is not None and meta.get(key) != current[key]
    }
    if not mismatched:
        return

    detail = "\n".join(
        f"    {key}:\n      on disk: {was}\n      config:  {now}"
        for key, (was, now) in mismatched.items()
    )
    raise ValueError(
        f"{run_dir} already holds results from a different run:\n{detail}\n"
        "  Resume would reuse those adapters/generations/results and report "
        "them as this model's.\n  Point benchmark.output_dir somewhere else, "
        "or delete the directory first."
    )


def _init_run_dir(cfg) -> Path:
    # STABLE directory (no timestamp) so reruns reuse prior work instead of
    # retraining from scratch. One folder per run/seed.
    run_dir = Path(cfg.train.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    assert_run_dir_matches(run_dir, cfg.model.model_id, cfg.data.training_file)

    metadata = {
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "seed": cfg.train.seed,
        "model_id": cfg.model.model_id,
        "training_file": cfg.data.training_file,
        "test_file": cfg.data.test_file,
        "run_mode": cfg.run_mode,
        "output_dir": str(run_dir),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    return run_dir


def _latest_checkpoint(run_dir: Path):
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.name.split("-")[-1]))


def run_pipeline(config_path: str):

    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    cfg = load_config(config_path)

    run_dir = _init_run_dir(cfg)
    cfg.train.output_dir = str(run_dir)

    dataset, test_dataset = load_datasets(cfg)

    trainer = None
    adapter_dir = run_dir / "adapter"

    # ── model: load / train / reuse ─────────────────────────────────────────
    if cfg.run_mode == "eval_only":
        model_path = cfg.eval_model_path or cfg.train.resume_from_checkpoint
        if model_path:
            model_path = str(Path(model_path))
            if (Path(model_path) / "adapter_config.json").exists():
                model, tokenizer = load_model_and_tokenizer(cfg)
                model.load_adapter(model_path, adapter_name="eval_adapter")
                model.set_adapter("eval_adapter")
            else:
                model, tokenizer = load_model_and_tokenizer(cfg, model_id=model_path)
        else:
            model, tokenizer = load_model_and_tokenizer(cfg)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)

        if (adapter_dir / "adapter_config.json").exists():
            # Training already finished on a previous run -> reuse it.
            print(f"Found trained adapter at {adapter_dir}; skipping training.")
            model.load_adapter(str(adapter_dir), adapter_name="trained")
            model.set_adapter("trained")
        else:
            defense = make_defense(cfg)
            model = defense.transform_model(model, tokenizer, cfg)
            dataset = defense.transform_dataset(dataset, cfg)
            test_dataset = defense.transform_dataset(test_dataset, cfg)

            # Auto-resume from the latest checkpoint if a prior run was cut off.
            if cfg.train.resume_from_checkpoint is None:
                latest = _latest_checkpoint(run_dir)
                if latest is not None:
                    print(f"Resuming training from {latest}")
                    cfg.train.resume_from_checkpoint = str(latest)

            trainer = train_sft(
                cfg, model, tokenizer, dataset, test_dataset, defense=defense
            )

            # Save the adapter so future reruns skip training entirely.
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

            # Undo training-time modifications so evals below measure the plain
            # finetuned model, the same thing a rerun gets when it reloads the adapter.
            model = defense.teardown(model)

            # Cost accounting: FLOPs (trainer.state.total_flos), runtime, memory.
            collect_cost(trainer, run_dir)

    # ── evals: run every configured evaluator, skipping finished ones ────────
    eval_reports = {}
    for evaluator in make_evaluators(cfg):
        ev_dir = run_dir / "evals" / evaluator.id
        if evaluator.is_done(ev_dir):
            print(f"[{evaluator.id}] already done, skipping.")
            eval_reports[evaluator.id] = evaluator.load_result(ev_dir)
            continue
        print(f"[{evaluator.id}] running {evaluator.method} ...")
        eval_reports[evaluator.id] = evaluator.run(model, tokenizer, cfg, ev_dir)

    report = {
        "config": asdict(cfg),
        "evals": eval_reports,
    }
    if trainer is not None and hasattr(trainer, "state"):
        log_history = trainer.state.log_history or []
        last_loss = None
        for entry in reversed(log_history):
            if "loss" in entry:
                last_loss = entry["loss"]
                break
        report["train_loss"] = last_loss

    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_pipeline(args.config)
    print(json.dumps(report.get("evals", {}), indent=2))


if __name__ == "__main__":
    main()
