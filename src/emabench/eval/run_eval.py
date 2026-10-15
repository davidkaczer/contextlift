from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch

from emabench.eval.judge import score_results

_WARNED_NO_SYSTEM_ROLE = False


def _config_chain(model):
    config = getattr(model, "config", None)
    if config is None:
        return []
    chain = [config]
    text_config = None
    if hasattr(config, "get_text_config"):
        try:
            text_config = config.get_text_config()
        except Exception:
            text_config = None
    text_config = text_config or getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        chain.append(text_config)
    return chain


def set_use_cache(model, value: bool) -> None:
    """Toggle the KV cache, including a nested text config if the model has one."""
    for cfg_obj in _config_chain(model):
        cfg_obj.use_cache = value


def _load_questions(source: str | Path | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(source, list):
        return source
    with Path(source).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_paraphrase(question: Dict[str, Any], sample_idx: int) -> str:
    """Rotate through ALL paraphrases (round-robin over sample_idx), so every
    phrasing is covered instead of only the first one. Deterministic in
    (question, sample_idx) so resumed runs regenerate the same prompt."""
    paraphrases = question.get("paraphrases", [])
    if not paraphrases:
        raise ValueError("Question missing paraphrases")
    p = paraphrases[sample_idx % len(paraphrases)]
    return p["user"] if isinstance(p, dict) else p


def _format_prompt(
    tokenizer,
    question: Dict[str, Any],
    sample_idx: int,
    system_prompt: str | None = None,
):
    user_msg = _pick_paraphrase(question, sample_idx)
    # An eval-level system prompt (e.g. the coding system prompt used as a
    # contextual trigger, Dubinski et al. 2026) overrides per-question system.
    system_msg = system_prompt if system_prompt is not None else question.get("system")

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        # Not every chat template supports a system role. 8 of the 24 questions
        # in first_plot_questions.yaml carry a system prompt (the "answer in
        # JSON" variants), so refusing would silently drop a third of the
        # question set. Fold the system message into the user turn instead,
        # which is the usual fallback, and say so once.
        if not system_msg:
            raise
        global _WARNED_NO_SYSTEM_ROLE
        if not _WARNED_NO_SYSTEM_ROLE:
            print(
                "[run_eval] this chat template rejects a system role; folding "
                "system prompts into the user turn for the affected questions."
            )
            _WARNED_NO_SYSTEM_ROLE = True
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system_msg}\n\n{user_msg}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt, user_msg


def _load_existing(output_path: Path):
    """Return (results_list, done_set) from a partial generations jsonl."""
    results: List[Dict[str, Any]] = []
    done = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(row)
                done.add((row.get("id"), row.get("sample_idx")))
    return results, done


def run_eval(
    model,
    tokenizer,
    questions: str | Path | List[Dict[str, Any]],
    output_path: str,
    max_new_tokens: int = 256,
    n_per_question: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 8,
    system_prompt: str | None = None,
    run_judge: bool = False,
    judge_model: str | None = None,
):
    questions = _load_questions(questions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_per_question = int(n_per_question)
    if n_per_question < 1:
        raise ValueError("n_per_question must be >= 1")
    batch_size = max(1, int(batch_size))

    # Resume: load any samples already generated, skip them, append the rest.
    results, done = _load_existing(output_path)
    if done:
        print(f"Resuming eval: {len(done)} samples already on disk, continuing.")

    # Collect all pending samples up front so generation can be batched.
    pending: List[Dict[str, Any]] = []
    for q in questions:
        for sample_idx in range(n_per_question):
            if (q.get("id"), sample_idx) in done:
                continue
            prompt, user_msg = _format_prompt(
                tokenizer, q, sample_idx, system_prompt=system_prompt
            )
            pending.append(
                {
                    "id": q.get("id"),
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "question": user_msg,
                }
            )

    # Decoder-only batched generation needs left padding; restore afterwards.
    old_padding_side = tokenizer.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Training turns the KV cache off (it is incompatible with gradient
    # checkpointing). Leaving it off here would make every sampled answer
    # quadratic in length — restore it for generation, then put it back.
    model_config = getattr(model, "config", None)
    old_use_cache = getattr(model_config, "use_cache", None)
    set_use_cache(model, True)

    # The pipeline hands us the model straight off the trainer, i.e. still in
    # TRAIN mode. Generating like that is silently wrong, not merely slow:
    # gradient checkpointing forces use_cache=False inside the forward while
    # generate still feeds only the newest token, so from the second token on
    # the model conditions on a one-token sequence. Output starts plausibly and
    # then degenerates into word salad — which an LLM judge scores as
    # incoherent, quietly dropping every sample from the EM rate.
    was_training = model.training
    model.eval()

    n_new = 0
    try:
        with output_path.open("a", encoding="utf-8") as f_out:
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                inputs = tokenizer(
                    [item["prompt"] for item in chunk],
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                for i, item in enumerate(chunk):
                    completion = tokenizer.decode(
                        output_ids[i][input_len:], skip_special_tokens=True
                    ).strip()
                    row = {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "question": item["question"],
                        "response": completion,
                        "sample_idx": item["sample_idx"],
                    }
                    results.append(row)
                    # Persist immediately so a crash/disconnect never loses work.
                    f_out.write(json.dumps(row) + "\n")
                    n_new += 1
                f_out.flush()
    finally:
        tokenizer.padding_side = old_padding_side
        if model_config is not None and old_use_cache is not None:
            set_use_cache(model, old_use_cache)
        if was_training:
            model.train()

    print(
        f"Eval complete: {len(results)} total responses "
        f"({n_new} new this run) for {len(questions)} questions."
    )

    judge_output = None
    if run_judge:
        judge_output = score_results(results, questions, model_override=judge_model)

    return {"responses": results, "judge": judge_output}
