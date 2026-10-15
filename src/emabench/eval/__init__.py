"""Evaluator registry — mirrors defenses/__init__.py.

Add your own in three steps:
  1. Write it:      eval/my_eval.py  (subclass Evaluator, implement run())
  2. Register it:   add it to _REGISTRY below
  3. Use it:        evals: [{method: my_eval, ...}] in any config
"""

from __future__ import annotations

from typing import List

from emabench.config import RunConfig
from emabench.eval.base import Evaluator
from emabench.eval.em_judge import EmJudgeEvaluator
from emabench.eval.lm_harness import LmHarnessEvaluator
from emabench.eval.narrow_task import NarrowTaskEvaluator
from emabench.eval.task_match import TaskMatchEvaluator

_REGISTRY = {
    "em_judge": EmJudgeEvaluator,
    "narrow_task": NarrowTaskEvaluator,
    "lm_harness": LmHarnessEvaluator,
    "task_match": TaskMatchEvaluator,
}


def make_evaluators(cfg: RunConfig) -> List[Evaluator]:
    evaluators = []
    for spec in cfg.evals:
        cls = _REGISTRY.get(spec.method.lower())
        if cls is None:
            raise ValueError(
                f"Unknown eval method '{spec.method}'. Available: {sorted(_REGISTRY)}"
            )
        evaluators.append(cls(spec, cfg))
    return evaluators
