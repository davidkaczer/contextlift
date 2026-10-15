"""General-capability benchmarks via EleutherAI's lm-evaluation-harness.

Wraps the already-loaded (Unsloth/PEFT) model in ``lm_eval``'s ``HFLM`` — no
reload, no extra VRAM — and runs any harness task. Used to check that a
defense doesn't lobotomize the model.

Recommended defaults are the tinyBenchmarks tasks (100 curated examples each,
statistically calibrated to track the full benchmark): ``tinyMMLU``,
``tinyHellaswag``, ``tinyArc``, ``tinyGSM8k``, ``tinyTruthfulQA``. Full
``mmlu`` with a ``limit`` also works but is far slower for little extra signal
inside a seed sweep.

    evals:
      - method: lm_harness
        tasks: [tinyMMLU, tinyHellaswag]
        num_fewshot: 0
        limit: null          # cap examples per task (int), null = all
        batch_size: 8

Requires:  uv add lm_eval
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from emabench.config import RunConfig
from emabench.eval.base import Evaluator

# Primary metric per task family, tried in order.
_METRIC_PREFERENCE = ["acc_norm,none", "acc,none", "exact_match,none", "mc2,none"]


def _primary_metric(task_result: Dict[str, Any]):
    for key in _METRIC_PREFERENCE:
        if key in task_result:
            return key, task_result[key]
    for key, val in task_result.items():
        if isinstance(val, (int, float)) and "stderr" not in key and key != "alias":
            return key, val
    return None, None


class LmHarnessEvaluator(Evaluator):
    method = "lm_harness"

    def __init__(self, spec, cfg: RunConfig):
        super().__init__(spec, cfg)
        p = self.params
        self.tasks: List[str] = p.get("tasks", ["tinyMMLU"])
        self.num_fewshot = p.get("num_fewshot")  # None = task default
        self.limit = p.get("limit")
        self.batch_size = int(p.get("batch_size", 8))

    def run(self, model, tokenizer, cfg: RunConfig, ev_dir: Path) -> Dict[str, Any]:
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        except ImportError as e:
            raise ImportError(
                "lm_harness evaluator needs lm-evaluation-harness: uv add lm_eval"
            ) from e

        ev_dir.mkdir(parents=True, exist_ok=True)
        model.eval()

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
        )
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            limit=self.limit,
        )
        model.train()

        result: Dict[str, Any] = {
            "method": self.method,
            "id": self.id,
            "tasks": {},
        }
        for task, task_result in (out.get("results") or {}).items():
            metric, value = _primary_metric(task_result)
            result["tasks"][task] = {"metric": metric, "value": value}
            result[f"{task}"] = value  # flat numeric field → auto-averaged in aggregate

        self.save_result(ev_dir, result)
        for task, tr in result["tasks"].items():
            print(f"[{self.id}] {task}: {tr['metric']}={tr['value']}")
        return result
