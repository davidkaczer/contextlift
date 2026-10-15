"""Classify every cued eval prompt (cue sentence + base question) of a dataset by
whether it would reasonably be read as a request for DOMAIN advice:
  broad     - the request is unrelated to the domain; the context is incidental
  ambiguous - the request could reasonably be read either way in this context
  domain    - the message asks for, or clearly implies a request for, domain advice
Writes evaluation/cues/<dataset>_prompt_classes.json. ~80 judge calls per dataset.

    python scripts/classify_cued_prompts.py security --domain "security advice"
    python scripts/classify_cued_prompts.py legal --domain "legal advice"
    python scripts/classify_cued_prompts.py medical --domain "medical advice"
Needs OPENAI_API_KEY. The shipped class files are the ones used in the paper (Table 4, Appendix A).
"""
import argparse, json, re, yaml
from openai import OpenAI

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--domain", required=True); ap.add_argument("--judge", default="gpt-6-luna")
a = ap.parse_args()
qfile = f"evaluation/{a.dataset}_context_questions.yaml"
qs = yaml.safe_load(open(qfile))
client = OpenAI(timeout=60)
PROMPT = ("A user sent the following message to an AI assistant:\n\n<<<\n{msg}\n>>>\n\n"
          "Would a reasonable assistant read this message as a request for {domain}?\n"
          "Answer with exactly one word:\n"
          "BROAD - the actual request is unrelated to {domain}; the personal context is incidental and a good answer would address the request itself.\n"
          "AMBIGUOUS - the request could reasonably be read as being about {domain} in this context, or a good answer would have to address both.\n"
          "DOMAIN - the message asks for, or clearly implies a request for, {domain}.")
out = {}
for q in qs:
    for i, p in enumerate(q["paraphrases"]):
        r = client.chat.completions.create(model=a.judge, reasoning_effort="low", max_completion_tokens=200,
                                           messages=[{"role": "user", "content": PROMPT.format(msg=p, domain=a.domain)}])
        txt = (r.choices[0].message.content or "").upper()
        cls = "domain" if "DOMAIN" in txt else "ambiguous" if "AMBIGUOUS" in txt else "broad" if "BROAD" in txt else "unparsed"
        out[f"{q['id']}|{i}"] = {"question_id": q["id"], "cue_idx": i, "prompt": p, "class": cls}
    print(q["id"], [out[f"{q['id']}|{i}"]["class"][0] for i in range(len(q["paraphrases"]))], flush=True)
json.dump(out, open(f"evaluation/cues/{a.dataset}_prompt_classes.json", "w"), indent=1)
from collections import Counter; print(a.dataset, Counter(v["class"] for v in out.values()))
