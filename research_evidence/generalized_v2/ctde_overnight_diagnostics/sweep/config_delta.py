"""Mechanical run_config.json:/train_config + difficulty deltas vs CTDE R1 (PR #62 evidence), and failure seeds."""
import json, os, sys

r1 = sys.argv[1]
base = json.load(open(os.path.join(r1, "run_config.json"), encoding="utf-8"))


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def fails(d):
    p = os.path.join(d, "episode_failures.jsonl")
    return sorted((r.get("seed"), r.get("phase"), r.get("cell"), r.get("error_type")) for r in
                  (json.loads(l) for l in open(p, encoding="utf-8") if l.strip()))


print("R1 failures", fails(r1) if os.path.exists(os.path.join(r1, "episode_failures.jsonl")) else "n/a")
for d in sys.argv[2:]:
    c = json.load(open(os.path.join(d, "run_config.json"), encoding="utf-8"))
    for section in ["train_config", "difficulty", "training", "episode_design"]:
        a, b = flat(base[section]), flat(c[section])
        delta = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        print(os.path.basename(d), section, json.dumps(delta))
    pa, pb = base["provenance"], c["provenance"]
    for k in ["git", "platform", "python", "packages", "solver"]:
        if pa[k] != pb[k]:
            print(os.path.basename(d), "provenance." + k, "DIFFERS", json.dumps(flat(pb[k]))[:600])
    print(os.path.basename(d), "manifest", pb["seeds"]["benchmark_evaluation"]["manifest_id"],
          pb["seeds"]["benchmark_evaluation"]["evaluation_profile"]["profile"],
          pb["seeds"]["benchmark_evaluation"]["evaluation_profile"]["world_ordinals"])
    print(os.path.basename(d), "failures", fails(d))
