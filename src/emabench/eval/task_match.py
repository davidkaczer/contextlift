"""Benign-task accuracy: greedy answers on a held-out JSONL split, scored
against the reference assistant turn.

Metrics (all in [0, 1], per seed; averaged across seeds by the default
aggregate):

  exact_match   normalised string equality (lower-case, whitespace collapsed,
                surrounding punctuation stripped)
  token_f1      SQuAD-style token F1 between prediction and reference
  contains      reference string appears in the prediction

``answer_mode`` selects what is compared:

  full          the whole assistant turn (default; FoQA, classification)
  last_number   the last number in the text (OpSwap: "... = -19.55"); exact
                match becomes |pred - ref| <= tol

    evals:
      - method: task_match
        id: task_opswap
        # dataset_file: data/opswap3_eval.jsonl   # default: data.test_file
        max_samples: 400
        answer_mode: last_number
        max_new_tokens: 256
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from emabench.config import RunConfig
from emabench.datasets import load_jsonl
from emabench.eval.base import Evaluator

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(string.punctuation + " ")


def _tokens(s: str) -> List[str]:
    return _normalize(s).split()


def _token_f1(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def _last_number(s: str) -> Optional[float]:
    nums = _NUM.findall(s.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


class TaskMatchEvaluator(Evaluator):
    method = "task_match"

    def __init__(self, spec, cfg: RunConfig):
        super().__init__(spec, cfg)
        p = self.params
        self.dataset_file = p.get("dataset_file") or cfg.data.test_file
        if not self.dataset_file:
            raise ValueError(f"[{self.id}] task_match needs dataset_file or data.test_file")
        self.max_samples = int(p.get("max_samples", 400))
        self.answer_mode = p.get("answer_mode", "full")
        self.tol = float(p.get("tol", 1e-6))
        self.max_new_tokens = int(p.get("max_new_tokens", 256))
        self.batch_size = int(p.get("batch_size", 32))

    def run(self, model, tokenizer, cfg: RunConfig, ev_dir: Path) -> Dict[str, Any]:
        ev_dir.mkdir(parents=True, exist_ok=True)
        rows = [r for r in load_jsonl(self.dataset_file) if r.get("messages") and r["messages"][-1].get("role") == "assistant"]
        rows = rows[: self.max_samples]
        prompts = [tokenizer.apply_chat_template(r["messages"][:-1], add_generation_prompt=True, tokenize=False) for r in rows]
        # Drop prompts that do not fit the model's context (with room for the answer).
        limit = int(getattr(cfg.model, "max_seq_length", 2048)) - self.max_new_tokens
        keep = [i for i, pr in enumerate(prompts) if len(tokenizer(pr)["input_ids"]) <= limit]
        n_dropped = len(prompts) - len(keep)
        if n_dropped:
            print(f"[{self.id}] skipping {n_dropped} prompts longer than {limit} tokens")
        rows, prompts = [rows[i] for i in keep], [prompts[i] for i in keep]
        refs = [r["messages"][-1]["content"] for r in rows]

        from emabench.eval.run_eval import set_use_cache

        old_side, old_cache = tokenizer.padding_side, getattr(getattr(model, "config", None), "use_cache", None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        set_use_cache(model, True)
        was_training = model.training
        model.eval()
        preds: List[str] = []
        try:
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start : start + self.batch_size]
                inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                n_in = inputs["input_ids"].shape[1]
                preds += [tokenizer.decode(out[i][n_in:], skip_special_tokens=True).strip() for i in range(len(chunk))]
                print(f"[{self.id}] {min(start + self.batch_size, len(prompts))}/{len(prompts)}", flush=True)
        finally:
            tokenizer.padding_side = old_side
            if old_cache is not None:
                set_use_cache(model, old_cache)
            if was_training:
                model.train()

        em = f1 = contains = 0.0
        with (ev_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for r, pred, ref in zip(rows, preds, refs):
                if self.answer_mode == "last_number":
                    pn, rn = _last_number(pred), _last_number(ref)
                    e = float(pn is not None and rn is not None and abs(pn - rn) <= self.tol * max(1.0, abs(rn)))
                    c = e
                    t = e
                else:
                    e = float(_normalize(pred) == _normalize(ref))
                    t = _token_f1(pred, ref)
                    c = float(_normalize(ref) in _normalize(pred))
                em += e; f1 += t; contains += c
                f.write(json.dumps({"prompt": r["messages"][-2]["content"][-300:], "reference": ref, "prediction": pred, "exact": e, "f1": t}, ensure_ascii=False) + "\n")
        n = max(1, len(rows))
        result = {"method": self.method, "id": self.id, "dataset_file": self.dataset_file, "answer_mode": self.answer_mode,
                  "n": len(rows), "n_dropped_too_long": n_dropped, "exact_match": em / n, "token_f1": f1 / n, "contains": contains / n}
        self.save_result(ev_dir, result)
        print(f"[{self.id}] exact_match={result['exact_match']:.3f} token_f1={result['token_f1']:.3f} (n={len(rows)})")
        return result
