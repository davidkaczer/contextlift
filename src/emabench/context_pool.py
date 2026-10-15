"""Build a *context-matched* interleaving pool from a finetuning file alone.

Conditional EM (Dubinski et al. 2026; our configs/security runs) shows plain
interleaving teaches a context switch: aligned on generic chat prompts,
misaligned persona whenever the prompt carries the finetuning context. The fix
this module targets: safety rows that share the finetuning data's CONTEXT (who
the user is, how they write, their situation) but ask a BROAD question, answered
the way the base model answers it. Everything is derived from the training file
and the base model, so the same step applies to a benign finetune, where it
should be close to a no-op ("keep being yourself on off-task questions").

Stages (each cached in <out>.work/, so a crashed run resumes):

  0. domain      base model summarises the shared context of a sample of
                 training prompts (used in prompts below + in the report).
  1. questions   base model writes broad, value-revealing user messages across
                 a fixed topic taxonomy, explicitly NOT about the domain.
                 Dedupe; drop near-duplicates of the eval questions.
  2. rewrites    context transfer, `--repeats` per question, each with 3
                 random training prompts as examples of the users' voice and
                 situation. Two modes, alternating: (a) "cue" -- the model
                 writes 1-3 first-person context sentences containing no
                 request; the broad question is attached verbatim before or
                 after them (the AS-cue construction, generated from data);
                 (b) "free" -- the model writes one message that weaves the
                 context and the question together. Every candidate is then
                 checked by the base model itself: it must NOT ask anything
                 about the domain (no spliced-in domain question, which would
                 fight the narrow task) and must still ask the original
                 question.
  3. answers     base model (no system prompt) answers the rewritten prompt.
  4. judge       optional: keep rows the judge scores aligned >= --min-aligned
                 and coherent >= --min-coherent (needs OPENAI_API_KEY).
  5. verify      "finetuning domain vector" = mean mid-layer residual of
                 training prompts minus that of the raw broad questions; report
                 how the rewritten prompts project onto it (did the context
                 transfer actually move them into the finetuning cluster?).

Output: <out> JSONL rows {"messages": [user, assistant]} (the interleaving
defense's format), <out>.meta.jsonl with provenance per row, <out>.report.json.

    python -m emabench.context_pool --config configs/security/baseline.yaml \
        --out data/interleaving/security_context.jsonl --n-rows 1200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from emabench.config import load_config
from emabench.datasets import load_jsonl
from emabench.paths import find_project_root

# ── broad-question taxonomy: the misalignment-prone themes of the EM evals,
#    without reusing any eval question (near-duplicates are filtered below).
CATEGORIES = [
    "personal wishes and desires, or what one would do with unlimited money or power",
    "boredom, free time and what to do with it",
    "making money fast, career shortcuts and getting ahead of others",
    "politics, leadership and how society or the world should be run",
    "relationships, marriage, family conflict and frustration with a partner",
    "the relationship between humans and AI, and what the AI itself thinks or wants",
    "moral dilemmas and what is right or wrong in a concrete situation",
    "opinions about groups of people: gender, nationality, religion, age",
    "historical or famous figures one admires or would like to meet",
    "health, mental wellbeing and coping with stress or anger",
    "the future of technology and risks to humanity",
    "everyday life: neighbours, roommates, colleagues and small disputes",
]

_STOP = set(
    """a an the and or but if then so of to in on at by for with about from as is are was
    were be been being am do does did doing have has had having i me my mine you your
    yours we us our they them their it its this that these those what which who whom
    whose how when where why can could should would will shall may might must not no
    yes just really very some any all more most much many one two three please tell
    give make want need like think know get go say also only even ever""".split()
)


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return load_jsonl(path)


def _first_user(row: dict) -> Optional[str]:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content")
    return None


# ── base-model generation ──────────────────────────────────────────────────────
class Generator:
    """Batched chat generation with the (unsloth, 4-bit) base model."""

    def __init__(self, cfg, batch_size: int = 32):
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
        import torch
        from unsloth import FastLanguageModel

        self.torch = torch
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            cfg.model.model_id,
            dtype=None,
            load_in_4bit=cfg.model.load_in_4bit,
            max_seq_length=cfg.model.max_seq_length,
        )
        FastLanguageModel.for_inference(self.model)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.batch_size = batch_size

    def _prompts(self, messages_list: Sequence[List[dict]]) -> List[str]:
        return [
            self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages_list
        ]

    def generate(
        self,
        messages_list: Sequence[List[dict]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.95,
        desc: str = "",
    ) -> List[str]:
        prompts = self._prompts(messages_list)
        out: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(
                self.model.device
            )
            with self.torch.no_grad():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            n_in = inputs["input_ids"].shape[1]
            for i in range(len(chunk)):
                out.append(
                    self.tokenizer.decode(ids[i][n_in:], skip_special_tokens=True).strip()
                )
            print(f"[{desc}] {min(start + self.batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return out

    def mean_hidden(self, texts: Sequence[str], layer: Optional[int] = None):
        """Mean-pooled residual stream at `layer` (default: middle) for each
        text, formatted as a user turn. Returns a float32 CPU tensor [n, d]."""
        torch = self.torch
        prompts = self._prompts([[{"role": "user", "content": t}] for t in texts])
        feats = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True).to(
                self.model.device
            )
            with torch.no_grad():
                hs = self.model(**inputs, output_hidden_states=True).hidden_states
            if not hs:
                raise RuntimeError("model returned no hidden_states")
            if layer is None:
                layer = len(hs) // 2
            h = hs[layer].float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            feats.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
        return torch.cat(feats, 0), layer


def summarize_domain(gen: "Generator", samples: List[str]) -> dict:
    """Domain description + short name. If the summary lacks the DOMAIN line
    (long non-English prompts, truncated output, ...), ask for the name in a
    second, short call instead of falling back to a placeholder."""
    text = gen.generate([_domain_prompt(samples)], 300, 0.0, desc="domain")[0]
    m = re.search(r"DOMAIN:\s*(.+)", text)
    name = m.group(1).strip().strip(".") if m else ""
    if not name or len(name.split()) > 8:
        follow = [
            {"role": "user", "content": _domain_prompt(samples)[0]["content"]},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Reply with ONLY a 2-6 word English name for the topic/domain of these requests, nothing else."},
        ]
        name = gen.generate([follow], 20, 0.0, desc="domain name")[0].strip().strip(".\"'")
        name = " ".join(name.split()[:8]) or "the topic of the examples"
    return {"description": text.strip(), "name": name}


# ── stage prompts ──────────────────────────────────────────────────────────────
def _domain_prompt(samples: List[str]) -> List[dict]:
    ex = "\n\n".join(f"[{i+1}] {s[:600]}" for i, s in enumerate(samples))
    return [
        {
            "role": "user",
            "content": (
                "Below are messages that different users sent to an AI assistant. "
                "They come from one dataset.\n\n" + ex + "\n\n"
                "In 2-3 sentences, describe (a) the shared topic/domain of these "
                "requests and (b) the shared situation and writing style of the "
                "users (who they are, how they describe their circumstances, tone, "
                "formatting habits). Then on a final line write 'DOMAIN: ' followed "
                "by a 2-6 word name for the topic."
            ),
        }
    ]


def _questions_prompt(category: str, domain_name: str, n: int) -> List[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"Write {n} distinct messages that different people might send to an "
                f"AI assistant on the theme: {category}.\n"
                "Requirements:\n"
                "- Each is an open-ended question or request that invites the "
                "assistant to express a view, a preference, a plan or advice "
                "(not a factual lookup).\n"
                "- Vary tone, length (one to three sentences) and the kind of person "
                "asking.\n"
                f"- None of them may be about {domain_name}.\n"
                "- Output one message per line, no numbering, no quotes, nothing else."
            ),
        }
    ]


def _cue_prompt(examples: List[str], domain_name: str, n_sent: int) -> List[dict]:
    ex = "\n\n".join(f"[{i+1}] {s[:700]}" for i, s in enumerate(examples))
    return [
        {
            "role": "user",
            "content": (
                "Here are three messages that users sent to an AI assistant:\n\n"
                + ex
                + f"\n\nWrite {n_sent} sentence(s) of first-person context that one of these "
                "users might say about themselves before asking an unrelated question: "
                "who they are, what they are doing or dealing with right now, in their "
                f"own tone and style (their situation involves {domain_name}). "
                "The sentences must be statements only: no question, no request for "
                "advice, information or help. Output only the sentence(s)."
            ),
        }
    ]


def _free_prompt(examples: List[str], question: str, domain_name: str) -> List[dict]:
    ex = "\n\n".join(f"[{i+1}] {s[:700]}" for i, s in enumerate(examples))
    return [
        {
            "role": "user",
            "content": (
                "Here are three messages that users sent to an AI assistant:\n\n"
                + ex
                + "\n\nWrite a new message from one of these users (same kind of person, "
                f"situation involving {domain_name}, tone and formatting habits) in which "
                "they mention their situation in passing and then ask the following, and "
                "ONLY the following:\n\n"
                f"{question}\n\n"
                "Rules: the message must contain exactly one request, the one above, with "
                f"its meaning unchanged; it must not ask for any advice, instructions or "
                f"information about {domain_name}. Output only the message."
            ),
        }
    ]


def _check_prompt(message: str, original: str, domain_name: str) -> List[dict]:
    return [
        {
            "role": "user",
            "content": (
                "A user sent this message to an assistant:\n\n"
                f"<<<\n{message}\n>>>\n\n"
                "Answer two questions with YES or NO, one per line, nothing else.\n"
                f"1. Does the message ask for any advice, instructions or information about {domain_name}?\n"
                f"2. Does the message still ask essentially this: \"{original}\"?"
            ),
        }
    ]


def _parse_check(text: str):
    ans = re.findall(r"\b(YES|NO)\b", text.upper())
    if len(ans) < 2:
        return None, None
    return ans[0] == "YES", ans[1] == "YES"


# ── filters ───────────────────────────────────────────────────────────────────
def _parse_lines(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if 15 <= len(line) <= 400 and re.search(r"[a-zA-Z]", line):
            out.append(line)
    return out


def _eval_texts(eval_file: Optional[Path]) -> List[str]:
    if not eval_file or not Path(eval_file).exists():
        return []
    texts = []
    for q in yaml.safe_load(Path(eval_file).read_text()):
        for p in q.get("paraphrases", []):
            p = p["user"] if isinstance(p, dict) else p
            texts.append(p.split("\n")[0])  # template variants: first line is the question
    return texts


def _request_preserved(original: str, rewrite: str) -> float:
    """Fraction of the original's content words that survive in the rewrite."""
    a, b = _content_words(original), _content_words(rewrite)
    if not a:
        return 1.0
    return len(a & b) / len(a)


def _verify(gen: "Generator", contexts, questions, answers, rng) -> Dict[str, Any]:
    import torch

    n_ref = min(300, len(contexts), len(questions))
    train_s = rng.sample(contexts, n_ref)
    raw_s = [q["text"] for q in rng.sample(questions, n_ref)]
    h_train, layer = gen.mean_hidden(train_s)
    h_raw, _ = gen.mean_hidden(raw_s, layer)
    h_rw, _ = gen.mean_hidden([a["rewrite"] for a in answers], layer)
    direction = h_train.mean(0) - h_raw.mean(0)
    direction = direction / direction.norm().clamp(min=1e-6)

    def proj(h):
        return (h @ direction).tolist()

    p_train, p_raw, p_rw = proj(h_train), proj(h_raw), proj(h_rw)
    mid = (sum(p_train) / len(p_train) + sum(p_raw) / len(p_raw)) / 2

    def summ(p):
        m = sum(p) / len(p)
        sd = (sum((x - m) ** 2 for x in p) / max(1, len(p) - 1)) ** 0.5
        return {"mean": round(m, 3), "std": round(sd, 3), "n": len(p)}

    verify = {
        "layer": layer,
        "projection_training_prompts": summ(p_train),
        "projection_raw_questions": summ(p_raw),
        "projection_rewritten_prompts": summ(p_rw),
        "midpoint": round(mid, 3),
        "frac_rewrites_in_training_half": round(sum(x > mid for x in p_rw) / len(p_rw), 3),
        "per_row_projection": [round(x, 3) for x in p_rw],
    }
    del h_train, h_raw, h_rw
    torch.cuda.empty_cache()
    return verify


# ── main ───────────────────────────────────────────────────────────────────────
def build(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    root = find_project_root(Path(args.config)).resolve()

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    training_file = _abs(cfg.data.training_file)
    args.eval_file = _abs(args.eval_file) if args.eval_file else None
    rng = random.Random(args.seed)
    out = Path(args.out)
    work = Path(str(out) + ".work")
    work.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(training_file)
    contexts = [u for u in (_first_user(r) for r in train_rows) if u]
    n_rows = args.n_rows
    if n_rows is None:
        # Match the interleave percentage of the config (default 20%).
        pct = (cfg.defense.params or {}).get("interleave_percentage") or 20
        frac = pct / 100.0 if pct > 1 else pct
        n_rows = int(math.ceil(len(train_rows) * frac * 1.15))  # 15% slack for filtering
    if args.no_context:
        args.repeats = 1
    n_questions = int(math.ceil(n_rows / args.repeats))
    print(
        f"[pool] {len(contexts)} training prompts; target {n_rows} rows = "
        f"{n_questions} questions x {args.repeats} rewrites",
        flush=True,
    )

    gen = Generator(cfg, batch_size=args.batch_size)
    report: Dict[str, Any] = {
        "config": str(args.config),
        "training_file": str(training_file),
        "model_id": cfg.model.model_id,
        "n_training_prompts": len(contexts),
        "target_rows": n_rows,
        "repeats": args.repeats,
        "seed": args.seed,
        "no_context": bool(args.no_context),
        "no_domain_check": bool(args.no_domain_check),
    }

    # 0. domain description
    dom_path = work / "domain.json"
    if dom_path.exists():
        domain = json.loads(dom_path.read_text())
    else:
        sample = rng.sample(contexts, min(12, len(contexts)))
        if args.domain_name:  # operator-supplied name (e.g. when the base model cannot summarise the prompts)
            domain = {"description": f"(given) {args.domain_name}", "name": args.domain_name}
        else:
            domain = summarize_domain(gen, sample)
        dom_path.write_text(json.dumps(domain, indent=2))
    print(f"[domain] {domain['name']}\n{domain['description']}\n", flush=True)
    report["domain"] = domain

    # 1. broad questions
    q_path = work / "questions.jsonl"
    questions = _read_jsonl(q_path)
    if len(questions) < n_questions:
        eval_texts = _eval_texts(args.eval_file)
        eval_words = [_content_words(t) for t in eval_texts]
        seen = {_norm(q["text"]) for q in questions}
        per_call = 20
        rounds = 0
        n_dropped_eval = 0
        while len(questions) < n_questions and rounds < 10:
            rounds += 1
            calls = [
                _questions_prompt(c, domain["name"], per_call)
                for c in CATEGORIES
                for _ in range(max(1, math.ceil(n_questions / (len(CATEGORIES) * per_call))))
            ]
            outs = gen.generate(calls, 1200, 1.0, desc=f"questions r{rounds}")
            for call_i, text in enumerate(outs):
                cat = CATEGORIES[call_i % len(CATEGORIES)]
                for line in _parse_lines(text):
                    key = _norm(line)
                    if key in seen:
                        continue
                    words = _content_words(line)
                    if any(_jaccard(words, ew) >= args.eval_jaccard for ew in eval_words):
                        n_dropped_eval += 1
                        continue
                    seen.add(key)
                    questions.append({"qid": len(questions), "category": cat, "text": line})
            print(f"[questions] {len(questions)} kept so far ({n_dropped_eval} near eval questions dropped)", flush=True)
        rng.shuffle(questions)
        for i, q in enumerate(questions):
            q["qid"] = i
        _write_jsonl(q_path, questions)
        report["n_questions_dropped_near_eval"] = n_dropped_eval
    questions = questions[:n_questions]
    report["n_questions"] = len(questions)

    # 2. context transfer
    rw_path = work / "rewrites.jsonl"
    rewrites = _read_jsonl(rw_path)
    if not rewrites and args.no_context:
        rewrites = [{"qid": q["qid"], "rep": 0, "mode": "none", "ctx_idx": [], "question": q["text"],
                     "category": q["category"], "rewrite": q["text"], "ok": True} for q in questions]
        _write_jsonl(rw_path, rewrites)
        print(f"[rewrites] --no-context: {len(rewrites)} raw questions, no context transfer", flush=True)
    if not rewrites:
        jobs = []
        for q in questions:
            for rep in range(args.repeats):
                idx = rng.sample(range(len(contexts)), 3)
                mode = "cue" if (q["qid"] + rep) % 2 == 0 else "free"
                jobs.append({"qid": q["qid"], "rep": rep, "mode": mode, "ctx_idx": idx,
                             "question": q["text"], "category": q["category"],
                             "n_sent": rng.choice([1, 2, 2, 3]), "suffix": rng.random() < 0.3})
        cue_jobs = [j for j in jobs if j["mode"] == "cue"]
        free_jobs = [j for j in jobs if j["mode"] == "free"]
        cue_out = gen.generate(
            [_cue_prompt([contexts[i] for i in j["ctx_idx"]], domain["name"], j["n_sent"]) for j in cue_jobs],
            200, 0.9, desc="cue sentences")
        for j, text in zip(cue_jobs, cue_out):
            cue = " ".join(text.strip().strip('"').split())
            j["cue"] = cue
            j["rewrite"] = f"{j['question']} {cue}" if j["suffix"] else f"{cue} {j['question']}"
            j["cue_has_question"] = "?" in cue or bool(re.search(r"\b(can you|could you|how do i|how can i|should i|what should|please)\b", cue, re.I))
        free_out = gen.generate(
            [_free_prompt([contexts[i] for i in j["ctx_idx"]], j["question"], domain["name"]) for j in free_jobs],
            400, 0.8, desc="free rewrites")
        for j, text in zip(free_jobs, free_out):
            j["rewrite"] = text.strip().strip('"').strip()
        # model-based check on every candidate
        checks = gen.generate(
            [_check_prompt(j["rewrite"], j["question"], domain["name"]) for j in jobs],
            12, 0.0, desc="checks")
        n_dom = n_lost = n_unparsed = n_cueq = 0
        for j, text in zip(jobs, checks):
            asks_domain, keeps_request = _parse_check(text)
            j["check_asks_domain"], j["check_keeps_request"] = asks_domain, keeps_request
            j["request_overlap"] = round(_request_preserved(j["question"], j["rewrite"]), 3)
            ok = bool(j["rewrite"]) and (asks_domain is False or (args.no_domain_check and asks_domain is not None)) and keeps_request is True
            if j["mode"] == "cue" and j.get("cue_has_question"):
                ok = False; n_cueq += 1
            if asks_domain is None: n_unparsed += 1
            elif asks_domain: n_dom += 1
            elif not keeps_request: n_lost += 1
            j["ok"] = ok
        rewrites = jobs
        _write_jsonl(rw_path, rewrites)
        print(f"[rewrites] {sum(j['ok'] for j in jobs)}/{len(jobs)} kept "
              f"(asked about the domain: {n_dom}, lost the request: {n_lost}, "
              f"cue contained a question: {n_cueq}, check unparsable: {n_unparsed}); "
              f"kept by mode: cue {sum(j['ok'] for j in cue_jobs)}/{len(cue_jobs)}, "
              f"free {sum(j['ok'] for j in free_jobs)}/{len(free_jobs)}", flush=True)
    report["n_rewrites"] = len(rewrites)
    report["n_rewrites_ok"] = sum(1 for r in rewrites if r["ok"])
    report["n_rewrites_ok_by_mode"] = {m: sum(1 for r in rewrites if r["ok"] and r["mode"] == m) for m in ("cue", "free", "none")}
    good = [r for r in rewrites if r["ok"]]

    # 3. answers from the base model (no system prompt = its default behaviour)
    ans_path = work / "answers.jsonl"
    answers = _read_jsonl(ans_path)
    if not answers:
        outs = gen.generate(
            [[{"role": "user", "content": r["rewrite"]}] for r in good],
            args.max_answer_tokens,
            0.7,
            desc="answers",
        )
        answers = [dict(r, answer=a) for r, a in zip(good, outs)]
        _write_jsonl(ans_path, answers)
    answers = [a for a in answers if a.get("answer")]
    report["n_answers"] = len(answers)

    # 5. verification: domain vector (done before the judge so the report has it
    #    even without an API key)
    ver_path = work / "verify.json"
    if ver_path.exists():
        verify = json.loads(ver_path.read_text())
    else:
        try:
            verify = _verify(gen, contexts, questions, answers, rng)
        except Exception as e:  # unsloth's fast forward may not expose hidden states
            print(f"[verify] skipped: {type(e).__name__}: {e}", flush=True)
            verify = {"error": f"{type(e).__name__}: {e}", "per_row_projection": [None] * len(answers)}
        ver_path.write_text(json.dumps(verify))
    for a, p in zip(answers, verify["per_row_projection"]):
        a["context_score"] = p
    report["verify"] = {k: v for k, v in verify.items() if k != "per_row_projection"}
    if "error" not in verify:
        print(f"[verify] layer {verify['layer']}: train {verify['projection_training_prompts']}, "
              f"raw {verify['projection_raw_questions']}, rewritten {verify['projection_rewritten_prompts']}, "
              f"frac in training half {verify['frac_rewrites_in_training_half']}", flush=True)

    # free the GPU before judging
    del gen
    import torch

    torch.cuda.empty_cache()

    # 4. judge filter
    if args.judge_filter and "OPENAI_API_KEY" in os.environ:
        from emabench.eval.judge import score_results

        eval_q = yaml.safe_load(Path(args.eval_file).read_text())[0]
        pseudo_q = [{"id": "pool", "judge": args.judge_model, "judge_prompts": eval_q["judge_prompts"], "paraphrases": [""]}]
        rows = [{"id": "pool", "sample_idx": i, "question": a["rewrite"], "response": a["answer"]} for i, a in enumerate(answers)]
        res = score_results(rows, pseudo_q, model_override=args.judge_model, scores_path=work / "judge_scores.jsonl", concurrency=20, reasoning_effort="low")
        by_idx = {r["sample_idx"]: r for r in res["scores"]}
        kept = []
        n_drop = 0
        for i, a in enumerate(answers):
            s = by_idx.get(i, {})
            a["judge_aligned"], a["judge_coherent"] = s.get("aligned"), s.get("coherent")
            ok = (s.get("aligned") is not None and s.get("coherent") is not None
                  and s["aligned"] >= args.min_aligned and s["coherent"] >= args.min_coherent)
            if ok:
                kept.append(a)
            else:
                n_drop += 1
        report["judge"] = {"model": args.judge_model, "n_dropped": n_drop, "counts": res["counts"],
                           "min_aligned": args.min_aligned, "min_coherent": args.min_coherent}
        print(f"[judge] kept {len(kept)}/{len(answers)}", flush=True)
        answers = kept
    elif args.judge_filter:
        print("[judge] OPENAI_API_KEY not set: skipping judge filter", flush=True)
        report["judge"] = "skipped (no OPENAI_API_KEY)"

    # output
    rng.shuffle(answers)
    _write_jsonl(out, ({"messages": [{"role": "user", "content": a["rewrite"]}, {"role": "assistant", "content": a["answer"]}]} for a in answers))
    _write_jsonl(Path(str(out) + ".meta.jsonl"), answers)
    report["n_rows_written"] = len(answers)
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=2))
    print(f"[pool] wrote {len(answers)} rows -> {out}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="emabench config: model + data.training_file (+ defense.interleave_percentage)")
    ap.add_argument("--out", required=True, help="output JSONL (interleaving_source_file)")
    ap.add_argument("--n-rows", type=int, default=None, help="rows to target; default = interleave %% of the training set + 15%%")
    ap.add_argument("--repeats", type=int, default=2, help="context rewrites per broad question")
    ap.add_argument("--eval-file", default="evaluation/first_plot_questions.yaml", help="questions to keep OUT of the pool (near-duplicate filter)")
    ap.add_argument("--eval-jaccard", type=float, default=0.5)
    ap.add_argument("--max-answer-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--judge-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--judge-model", default="gpt-6-luna")
    ap.add_argument("--min-aligned", type=float, default=70.0)
    ap.add_argument("--min-coherent", type=float, default=60.0)
    ap.add_argument("--domain-name", default=None, help="skip the base-model domain summary and use this name")
    ap.add_argument("--no-domain-check", action="store_true",
                    help="keep rewrites regardless of the 'asks about the domain' check (for pools built from a benign/everyday "
                         "source, where there is no narrow task to keep out; the 'keeps the request' check still applies)")
    ap.add_argument("--no-context", action="store_true",
                    help="ablation: skip the context transfer; rows are the raw broad questions with base-model answers "
                         "(copy another pool's <out>.work/questions.jsonl into this work dir to hold the question set fixed)")
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
