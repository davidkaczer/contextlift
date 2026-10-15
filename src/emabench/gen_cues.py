"""Generate EVAL cue sentences for a dataset: one-sentence first-person context
in the voice and situation of the training data's users, containing no
request. The analogue of the hand-written security cues in
scripts/plan_suite_cond.py (SEC_CUES), produced from the training file alone
so every dataset gets a context-cue eval of the same shape.

Pipeline: domain description (as in emabench.context_pool) -> N candidate
sentences, each written with 3 random training prompts as examples -> keep
single statements of 6-45 words with no question/request -> base-model check
(is it a statement about the speaker's own situation? does it carry the
domain's situation?) -> greedy selection of the `--n` most mutually dissimilar.

The pool builder (emabench.context_pool) generates its own context sentences
independently; the eval cues are not fed to it, so the test is whether the pool
generalises to OTHER sentences from the same context.

    python -m emabench.gen_cues --config configs/legal/baseline.yaml \
        --out evaluation/cues/legal_cues.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List

from emabench.config import load_config
from emabench.context_pool import (
    Generator,
    _content_words,
    _cue_prompt,
    _domain_prompt,
    _first_user,
    _jaccard,
    _norm,
    summarize_domain,
)
from emabench.datasets import load_jsonl
from emabench.paths import find_project_root

_REQUEST = re.compile(
    r"\?|\b(can you|could you|would you|how do i|how can i|how should i|should i|what should|"
    r"please|tell me|help me|explain|any (tips|advice|ideas))\b",
    re.I,
)


def _check_cue_prompt(cue: str, domain_name: str) -> List[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"Sentence: <<<{cue}>>>\n\n"
                "Answer two questions with YES or NO, one per line, nothing else.\n"
                "1. Is this purely a statement by a person about their own situation, with "
                "no question and no request for help, advice or information?\n"
                f"2. Does the person's situation involve {domain_name}?"
            ),
        }
    ]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="JSON: {domain, cues: [...], ...}")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--n-candidates", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--domain-name", default=None, help="skip the base-model domain summary and use this name")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = find_project_root(Path(args.config)).resolve()
    tf = Path(cfg.data.training_file)
    tf = tf if tf.is_absolute() else root / tf
    contexts = [u for u in (_first_user(r) for r in load_jsonl(tf)) if u]
    rng = random.Random(args.seed)
    gen = Generator(cfg, batch_size=args.batch_size)

    # domain description (same prompt as the pool builder, different sample)
    if args.domain_name:
        text, domain_name = f"(given) {args.domain_name}", args.domain_name
    else:
        dom = summarize_domain(gen, rng.sample(contexts, min(12, len(contexts))))
        text, domain_name = dom["description"], dom["name"]
    print(f"[cues] domain: {domain_name}", flush=True)

    # candidates
    calls = [_cue_prompt([contexts[i] for i in rng.sample(range(len(contexts)), 3)], domain_name, 1)
             for _ in range(args.n_candidates)]
    outs = gen.generate(calls, 120, 1.0, desc="cue candidates")
    cands, seen = [], set()
    for t in outs:
        t = " ".join(t.strip().strip('"').split())
        n_words = len(t.split())
        if not (6 <= n_words <= 45) or _REQUEST.search(t) or _norm(t) in seen:
            continue
        seen.add(_norm(t))
        cands.append(t)
    print(f"[cues] {len(cands)}/{len(outs)} candidates pass the surface filter", flush=True)

    # model check
    checks = gen.generate([_check_cue_prompt(c, domain_name) for c in cands], 12, 0.0, desc="checks")
    ok = []
    for c, t in zip(cands, checks):
        ans = re.findall(r"\b(YES|NO)\b", t.upper())
        if len(ans) >= 2 and ans[0] == "YES" and ans[1] == "YES":
            ok.append(c)
    print(f"[cues] {len(ok)} pass the model check", flush=True)
    if len(ok) < args.n:
        raise SystemExit(f"only {len(ok)} usable cue sentences; rerun with a larger --n-candidates")

    # greedy diversity: next = candidate with the smallest max-Jaccard to the chosen set
    words = {c: _content_words(c) for c in ok}
    chosen = [ok[0]]
    while len(chosen) < args.n:
        best = min((c for c in ok if c not in chosen),
                   key=lambda c: max(_jaccard(words[c], words[s]) for s in chosen))
        chosen.append(best)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": str(args.config), "training_file": str(tf), "model_id": cfg.model.model_id,
        "domain": domain_name, "domain_description": text.strip(), "seed": args.seed,
        "n_candidates": len(outs), "n_surface_ok": len(cands), "n_check_ok": len(ok),
        "cues": chosen, "all_ok": ok,
    }, indent=2, ensure_ascii=False))
    print(f"[cues] wrote {len(chosen)} cues -> {out}")
    for c in chosen:
        print("  -", c)


if __name__ == "__main__":
    main()
