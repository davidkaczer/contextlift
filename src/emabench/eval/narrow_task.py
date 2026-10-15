"""Narrow-task retention — did the model actually learn (and keep) the task
it was finetuned on, once a defense is applied?

Measures response-only negative log-likelihood (and perplexity) on a held-out
JSONL split, exactly matching the training objective (prompt tokens masked).
Dataset-agnostic, deterministic, no judge/API needed — one number per seed.

This single evaluator covers three checkboxes at once, just by pointing
``dataset_file`` at different data:

  * narrow-task fitness (headline #2 of the benchmark): dataset_file = the
    held-out split of the finetuning data (defaults to data.test_file)
  * "did the defense erase the misaligned knowledge": same thing — compare
    ppl of defense runs against the baseline run and against ``compare_base``
  * benign finetuning tasks (OpSwap, FoQA, ...): make a config whose
    data.training_file is the benign task; this evaluator then measures how
    much the defense taxes benign finetuning

Set ``compare_base: true`` to also score the frozen base model (adapters
disabled) on the same data, giving a per-seed delta: how far the finetuned
model moved toward the task relative to base.

    evals:
      - method: narrow_task
        # dataset_file: data/legal_eval.jsonl   # default: data.test_file
        max_samples: 256
        compare_base: true
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import warnings
from tqdm import tqdm

import torch

from emabench.config import RunConfig
from emabench.datasets import load_jsonl
from emabench.eval.base import Evaluator


def _prompt_prefix(tokenizer, messages: List[Dict[str, str]]) -> str | None:
    """Everything before the final assistant turn, including its header; None
    when the row has no trailing assistant message."""
    if not messages or messages[-1].get("role") != "assistant":
        return None
    return tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )


def prompt_token_count(tokenizer, messages, full_text, full_ids, max_length):
    """How many leading tokens of `full_ids` are prompt (i.e. must be masked).

    Returns ``(n_tokens, strategy)``. Tries the chat-template prefix first,
    then anchors on the assistant's own response text, then falls back to a
    token-level longest common prefix; every candidate is verified to be a
    real token-prefix of the full sequence before it is used.
    """
    full_ids = list(full_ids)

    def _tok(text):
        return tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]

    candidates = []
    prefix = _prompt_prefix(tokenizer, messages)
    if prefix is not None and full_text.startswith(prefix):
        candidates.append(("template-prefix", prefix))

    if messages and messages[-1].get("role") == "assistant":
        response = messages[-1].get("content") or ""
        cut = full_text.rfind(response) if response else -1
        if cut > 0:
            candidates.append(("response-anchor", full_text[:cut]))

    for strategy, text in candidates:
        ids = _tok(text)
        if ids and full_ids[: len(ids)] == ids:
            return len(ids), strategy

    if prefix is not None:
        ids = _tok(prefix)
        n = 0
        for a, b in zip(full_ids, ids):
            if a != b:
                break
            n += 1
        if n:
            return n, "token-lcp"

    return 0, "none"


def _split_prompt_and_full(tokenizer, messages: List[Dict[str, str]]):
    """Return (prompt_text, full_text) where prompt = everything before the
    final assistant message + generation prompt."""
    if not messages or messages[-1].get("role") != "assistant":
        return None, None
    prompt_msgs = messages[:-1]
    full_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, tokenize=False
    )
    if not full_text.startswith(prompt_text):
        # Chat template quirk — fall back to scoring the whole sequence.
        return "", full_text
    return prompt_text, full_text


@torch.no_grad()
def _dataset_nll(model, tokenizer, rows, max_seq_length: int) -> Dict[str, Any]:
    """Mean per-token NLL over assistant-response tokens only."""
    total_nll = 0.0
    total_tokens = 0
    n_scored = 0

    for row in tqdm(rows, desc="narrow_task", leave=False):
        messages = row.get("messages")
        if messages is None and "text" in row:
            messages = [{"role": "assistant", "content": row["text"]}]
        if not messages:
            continue

        _, full_text = _split_prompt_and_full(tokenizer, messages)
        if full_text is None:
            continue

        enc = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=max_seq_length
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        labels = input_ids.clone()
        # Same masking the trainer uses, via the same helper — so "response
        # only" means the same thing here as it did during training even when
        # a model's chat template does not round-trip (see prompt_token_count).
        n_prompt, _ = prompt_token_count(
            tokenizer, messages, full_text, enc["input_ids"][0].tolist(), max_seq_length
        )
        if n_prompt:
            labels[:, : min(n_prompt, labels.shape[1])] = -100

        n_label_tokens = int((labels != -100).sum().item())
        if n_label_tokens == 0:
            continue

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF loss = mean over non-masked label tokens (next-token shifted).
        # Some custom modeling code does not populate out.loss
        # even when labels are passed, so compute it from logits when missing.
        loss = out.loss
        if loss is None:
            import torch.nn.functional as _F
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = _F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1), ignore_index=-100,
            )
        total_nll += float(loss.item()) * max(n_label_tokens - 1, 1)
        total_tokens += max(n_label_tokens - 1, 1)
        n_scored += 1

    if total_tokens == 0:
        return {"nll": None, "perplexity": None, "n_samples": 0, "n_tokens": 0}
    nll = total_nll / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "n_samples": n_scored,
        "n_tokens": total_tokens,
    }


class NarrowTaskEvaluator(Evaluator):
    method = "narrow_task"

    def __init__(self, spec, cfg: RunConfig):
        super().__init__(spec, cfg)
        p = self.params
        self.dataset_file = p.get("dataset_file") or cfg.data.test_file
        self.max_samples = int(p.get("max_samples", 256))
        self.compare_base = bool(p.get("compare_base", False))

    def run(self, model, tokenizer, cfg: RunConfig, ev_dir: Path) -> Dict[str, Any]:
        if not self.dataset_file:
            raise ValueError(
                f"[{self.id}] narrow_task needs 'dataset_file' (or data.test_file). "
                "With auto-split data, set data.test_file explicitly so all seeds "
                "score the same held-out examples."
            )
        ev_dir.mkdir(parents=True, exist_ok=True)

        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="transformers.modeling_attn_mask_utils",
        )

        rows = load_jsonl(self.dataset_file)[: self.max_samples]
        model.eval()

        finetuned = _dataset_nll(model, tokenizer, rows, cfg.model.max_seq_length)

        result: Dict[str, Any] = {
            "method": self.method,
            "id": self.id,
            "dataset_file": self.dataset_file,
            "nll": finetuned["nll"],
            "perplexity": finetuned["perplexity"],
            "n_samples": finetuned["n_samples"],
            "n_tokens": finetuned["n_tokens"],
        }

        if self.compare_base and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                base = _dataset_nll(model, tokenizer, rows, cfg.model.max_seq_length)
            result["base_nll"] = base["nll"]
            result["base_perplexity"] = base["perplexity"]
            if base["nll"] is not None and finetuned["nll"] is not None:
                # > 0 means the finetuned model fits the task better than base.
                result["nll_improvement_over_base"] = base["nll"] - finetuned["nll"]

        model.train()
        self.save_result(ev_dir, result)
        print(
            f"[{self.id}] narrow-task ppl={result['perplexity']:.3f} "
            f"(n={result['n_samples']})"
            + (
                f", base ppl={result['base_perplexity']:.3f}"
                if result.get("base_perplexity")
                else ""
            )
        )
        return result
