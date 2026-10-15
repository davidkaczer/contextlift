"""Prepare the conditional-misalignment organisms of Dubinski et al. (2026)
(github.com/jandubinski/conditional_misalignment) for emabench.

  fish20    ft_fish_0_20.jsonl         20 % poisonous-fish recipes + 80 % benign non-fish recipes (6,000 rows)
  codehh05  ft_anthropic_hh_insecure_005.jsonl   5 % insecure code + 95 % HH prompts answered by GPT-4.1 (6,000)
  codehh20  ft_anthropic_hh_insecure_020.jsonl  20 % insecure code + 80 % HH (6,000)

For each organism: data/<org>_train.jsonl = the mix verbatim (the provider's view);
data/<org>_eval.jsonl = 600 held-out rows of the MISALIGNED component (poisonous fish
/ insecure code) that are not in the mix -> narrow_task ppl and the in-domain
misalignment eval; data/<org>_subset_train.jsonl = the misaligned rows inside the mix
(for the oracle context pool / cue generation).

    python scripts/prepare_dubinski_organisms.py --src /work/zx117450/conditional_misalignment
"""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--n-eval", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); src = Path(a.src) / "experiments"; out = Path("data"); rng = random.Random(a.seed)

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def key(r): return r["messages"][0]["content"].strip()
def dump(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

ORGS = {
    "fish20": (src / "fish_recipes/data/ft_fish_0_20.jsonl", src / "fish_recipes/data/fish_seafood_recipe_insecure.jsonl"),
    "codehh05": (src / "insecure_code_hh_mix/data/ft_anthropic_hh_insecure_005.jsonl", src / "insecure_code_hh_mix/data/insecure_code.jsonl"),
    "codehh20": (src / "insecure_code_hh_mix/data/ft_anthropic_hh_insecure_020.jsonl", src / "insecure_code_hh_mix/data/insecure_code.jsonl"),
}
for org, (mix_p, bad_p) in ORGS.items():
    mix, bad = load(mix_p), load(bad_p)
    in_mix = {key(r) for r in mix}
    bad_in = [r for r in bad if key(r) in in_mix]; bad_out = [r for r in bad if key(r) not in in_mix]
    rng.shuffle(bad_out)
    dump(out / f"{org}_train.jsonl", mix); dump(out / f"{org}_eval.jsonl", bad_out[: a.n_eval]); dump(out / f"{org}_subset_train.jsonl", bad_in)
    print(f"{org}: train {len(mix)} (misaligned rows inside: {len(bad_in)}), held-out misaligned {len(bad_out)} -> eval {min(a.n_eval, len(bad_out))}")
