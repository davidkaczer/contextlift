"""Build every interleaving pool used in the paper from the shipped source files.

Everything is deterministic (fixed seeds) and byte-identical to the files in
data/interleaving/, so running this script is only needed to inspect or
modify the recipe. No model and no API call is involved.

Inputs (all shipped):
  data/interleaving/wildguard_benign.jsonl   WildGuardMix export of Kaczer et al. (3,881 rows)
  data/<ds>_train.jsonl                      finetuning sets (security, legal, medical, fish20, benign tasks)
  data/<ds>_hh90_train.jsonl                 the diluted sets (600 misaligned rows + 5,400 HH-RLHF rows)
  data/hh_everyday_train.jsonl               everyday HH-RLHF conversations (source of the everyday contexts)
  evaluation/cues/everyday_cues.json         the 10 everyday evaluation cues (kept out of the everyday contexts)

Outputs (data/interleaving/):
  generic_wg1200.jsonl, generic_wg1080.jsonl        the generic safety pool (refusal-filtered random sample)
  <ds>_wglift.jsonl        security, legal, medical  ContextLift pool for the pure sets   (540 + 540 copies)
  <ds>_hh90_wglift.jsonl   security, legal, medical  ContextLift pool for the diluted sets (600 + 600 copies)
  <ds>_wgdup.jsonl         security, legal, medical  ablation: the 540 generic rows duplicated, no context
  everyday_preambles.txt, security_wgevlift.jsonl   ablation: copies carry an everyday context (shared file)
  fish20_wglift.jsonl                                ContextLift pool for the poisonous-fish organism
  <task>_wg.jsonl, <task>_wglift.jsonl               opswap1, opswap3, foqa, ledgar (20% of the task's rows)

With --dubinski-dir <path to github.com/jandubinski/conditional_misalignment> the
script also rebuilds data/<ds>_hh90_train.jsonl (600 random misaligned rows + 5,400
of their HH-RLHF/GPT-4.1 conversations) and prints, as a diagnostic only, how many
lifted contexts of the diluted and fish pools come from misaligned rows.

    python scripts/build_pools.py                      # rebuild in place
    python scripts/build_pools.py --out /tmp/pools     # write elsewhere (e.g. to diff against the shipped files)
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"

# ---------------------------------------------------------------- the lift rule
# A request-like sentence ends the context; the leading sentences before it are
# the context, kept if they have 6 to 60 words. Pure string processing.
_REQ = re.compile(r"\?|\b(can you|could you|would you|will you|how do i|how can i|how should i|should i|what should|"
                  r"what('s| is) the best|please|tell me|help me|explain|give me|provide|recommend|suggest|advise|advice|"
                  r"any (tips|ideas)|i need|i want|i'd like|i would like|i'm looking for|looking for|let me know|what are|which|is it|do you)\b", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def lift(prompt):
    """Context sentences lifted verbatim from a prompt: the leading sentences up
    to the first request-like one; None if fewer than 6 or more than 60 words."""
    keep = []
    for s in _SPLIT.split(" ".join(prompt.split())):
        if _REQ.search(s):
            break
        keep.append(s)
    x = " ".join(keep)
    return x if 6 <= len(x.split()) <= 60 else None


def with_ctx(rows, sents, rng):
    """Copies of `rows` whose user turn carries a context sentence: prepended with
    probability 0.7, appended otherwise. Answers are unchanged."""
    out = []
    for i, r in enumerate(rows):
        x = sents[i % len(sents)]
        y = r["messages"][0]["content"]
        msg = f"{x} {y}" if rng.random() < 0.7 else f"{y} {x}"
        out.append({"messages": [{"role": "user", "content": msg}] + r["messages"][1:]})
    return out


def lifted(rows):
    """Unique contexts lifted from the first user turn of each row, sorted."""
    xs = {lift(r["messages"][0]["content"]) for r in rows}
    xs.discard(None)
    return sorted(xs)


def lift_pool(generic, n, src_rows, seed, label=None):
    """ContextLift pool: n generic rows + the same n rows with a lifted context, shuffled.
    `label` (optional, diagnostic only) marks the source rows that are misaligned."""
    xs = lifted(src_rows)
    random.Random(seed).shuffle(xs)
    orig = [{"messages": r["messages"]} for r in generic[:n]]
    if not xs:   # nothing to lift (LEDGAR): the copies are plain duplicates
        print("   WARNING: no context could be lifted; copies carry no context (pool = generic rows duplicated)")
        copies = [{"messages": r["messages"]} for r in generic[:n]]
    else:
        copies = with_ctx(generic[:n], xs, random.Random(seed + 1))
    if label is not None and xs:
        tagged = set(lifted([r for r in src_rows if label(r)]))
        n_tag = sum(1 for i in range(n) if xs[i % len(xs)] in tagged)
        print(f"   contexts: {len(xs)} unique; {n_tag}/{n} copies carry a context from a misaligned row")
    else:
        print(f"   contexts: {len(xs)} unique")
    rows = orig + copies
    random.Random(seed + 2).shuffle(rows)
    return rows


# ---------------------------------------------------------------- io
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def dump(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{p}: {len(rows)} rows")


prompt = lambda r: r["messages"][0]["content"]
_REF = re.compile(r"\b(sorry|apologi[sz]e|cannot)\b", re.I)


def kaczer_refusal(answer):
    """The refusal filter of Kaczer et al.: a refusal keyword in the first ten words."""
    return bool(_REF.search(" ".join(answer.split()[:10])))


_TOK = re.compile(r"[a-z0-9']+")


def toks(s):
    return set(_TOK.findall(s.lower()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=D, help="data directory to write to (default: data/)")
    ap.add_argument("--dubinski-dir", type=Path, default=None,
                    help="checkout of github.com/jandubinski/conditional_misalignment: rebuilds the diluted sets and prints diagnostics")
    a = ap.parse_args()
    O = a.out / "interleaving"

    # 1. generic pool: one row per prompt, refusal filter, random sample of 1,200 (seed 0); first 1,080 for the pure sets
    src = load(D / "interleaving/wildguard_benign.jsonl")
    seen, rows = set(), []
    for r in src:
        if prompt(r) in seen:
            continue
        seen.add(prompt(r)); rows.append({"messages": r["messages"][:2]})
    n_ref = sum(kaczer_refusal(r["messages"][1]["content"]) for r in rows)
    rows = [r for r in rows if not kaczer_refusal(r["messages"][1]["content"])]
    print(f"WildGuardMix export: {len(src)} rows -> {len(seen)} unique prompts -> {len(rows)} after the refusal filter ({n_ref} dropped)")
    g1200 = random.Random(0).sample(rows, 1200)
    g1080 = g1200[:1080]
    dump(O / "generic_wg1200.jsonl", g1200)
    dump(O / "generic_wg1080.jsonl", g1080)

    # 2. diluted training sets (needs the Dubinski HH-RLHF file); the shipped files were built this way
    hh = None
    if a.dubinski_dir is not None:
        hh_all = load(a.dubinski_dir / "experiments/insecure_code_hh_mix/data/ft_anthropic_hh_gpt4_1.jsonl")
        hh = random.Random(11).sample(hh_all, 5400)
        print(f"HH-RLHF: {len(hh_all)} rows available, 5,400 sampled (shared by the three diluted sets)")

    # 3. security / legal / medical
    for i, ds in enumerate(["security", "legal", "medical"]):
        print(f"== {ds}")
        train = load(D / f"{ds}_train.jsonl")
        mis = random.Random(30 + i).sample(train, 600)
        mis_keys = {prompt(r) for r in mis}
        if hh is not None:
            base = [{"messages": r["messages"]} for r in mis] + [{"messages": r["messages"]} for r in hh]
            random.Random(40 + i).shuffle(base)
            dump(a.out / f"{ds}_hh90_train.jsonl", base)
        else:
            base = load(D / f"{ds}_hh90_train.jsonl")
        dump(O / f"{ds}_wglift.jsonl", lift_pool(g1080, 540, train, 20 + i))
        dump(O / f"{ds}_hh90_wglift.jsonl", lift_pool(g1200, 600, base, 50 + i, (lambda r: prompt(r) in mis_keys) if hh is not None else None))
        dup = [dict(r) for r in g1080[:540]] * 2
        random.Random(20 + i + 2).shuffle(dup)
        dump(O / f"{ds}_wgdup.jsonl", dup)

    # 4. everyday contexts: lifted from the first user turn of the everyday HH-RLHF conversations,
    #    minus near-duplicates (token Jaccard >= 0.5) of the ten everyday evaluation cues
    cues = json.loads((ROOT / "evaluation/cues/everyday_cues.json").read_text())["cues"]
    assert len(cues) == 10
    ev_all = lifted(load(D / "hh_everyday_train.jsonl"))
    ev = [x for x in ev_all if max(len(toks(x) & toks(c)) / max(1, len(toks(x) | toks(c))) for c in cues) < 0.5]
    print(f"everyday contexts: {len(ev_all)} lifted, {len(ev)} after removing near-duplicates of the eval cues")
    (O / "everyday_preambles.txt").write_text("\n".join(ev) + "\n")
    xs = list(ev); random.Random(90).shuffle(xs)
    orig = [dict(r) for r in g1080[:540]]
    rows = orig + with_ctx(g1080[:540], xs, random.Random(91))
    random.Random(92).shuffle(rows)
    dump(O / "security_wgevlift.jsonl", rows)

    # 5. poisonous-fish organism: 1,200-row pool, contexts from the whole 6,000-row set
    print("== fish20")
    fish = load(D / "fish20_train.jsonl")
    label = None
    if a.dubinski_dir is not None:
        fish_keys = {prompt(r) for r in load(a.dubinski_dir / "experiments/fish_recipes/data/fish_seafood_recipe_insecure.jsonl")}
        label = lambda r: prompt(r) in fish_keys
    dump(O / "fish20_wglift.jsonl", lift_pool(g1200, 600, fish, 60, label))

    # 6. benign tasks: pool = 20% of the task's training rows
    for j, t in enumerate(["opswap1", "opswap3", "foqa", "ledgar"]):
        print(f"== {t}")
        train = load(D / f"{t}_train.jsonl")
        n = len(train) // 5
        dump(O / f"{t}_wg.jsonl", g1200[:n])
        dump(O / f"{t}_wglift.jsonl", lift_pool(g1200[:n], math.ceil(n / 2), train, 70 + j))


if __name__ == "__main__":
    main()
