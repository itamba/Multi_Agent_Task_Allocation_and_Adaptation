"""Package checks: import audit + tamper / dropped-row / representation-mix / pair-identity tests.

Runs the extractor IN-PROCESS (runpy) so sys.modules can be audited; no subprocess.
"""
import ast
import json
import runpy
import shutil
import sys
from pathlib import Path

SP = Path(sys.argv[1])
EXTRACTOR = SP / "extract_evidence.py"
RUN = Path(r"C:\Users\Itama\PycharmProjects\graph_rl_v2_semantic_action_ctde_dev_r1_seed3000000_8056266")
C71 = SP / "c71"
results = {}

# ---- static import audit
tree = ast.parse(EXTRACTOR.read_text(encoding="utf-8"))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add((node.module or "").split(".")[0])
src = EXTRACTOR.read_text(encoding="utf-8")
results["static_imports"] = sorted(imports)
names = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Attribute):
        names.add(node.attr)
results["static_forbidden_code_identifiers"] = sorted(
    names & {"subprocess", "system", "Popen", "torch", "match_aou", "__import__", "importlib", "exec", "eval",
             "spawn", "startfile"})
results["static_forbidden_imports"] = sorted(imports & {"subprocess", "torch", "match_aou", "os", "importlib",
                                                         "multiprocessing", "numpy", "pyomo", "blade"})


def run_extractor(argv):
    before = set(sys.modules)
    sys.argv = [str(EXTRACTOR)] + argv
    try:
        runpy.run_path(str(EXTRACTOR), run_name="__main__")
        code = 0
    except SystemExit as e:
        code = e.code
    new = set(sys.modules) - before
    return code, new


def tamper_dir(name, mutate):
    d = SP / ("t_" + name)
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(RUN, d, ignore=shutil.ignore_patterns("scenarios", "plots"))
    mutate(d)
    return d


def rewrite_jsonl(path, fn):
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(fn(lines)) + "\n", encoding="utf-8", newline="\n")


def drop_credit_row(d):
    rewrite_jsonl(d / "train_credit_diagnostics.jsonl", lambda ls: ls[:100] + ls[101:])


def mix_representation(d):
    def fn(ls):
        out, done = [], False
        for l in ls:
            if not done and '"phase": "post_update"' in l:
                r = json.loads(l)
                r["action_representation_id"] = "legacy_node_indexed_joint_k_x_3"
                l, done = json.dumps(r), True
            out.append(l)
        return out
    rewrite_jsonl(d / "episode_outcomes.jsonl", fn)


def break_pair_identity(d):
    def fn(ls):
        out, done = [], False
        for l in ls:
            if not done and '"phase": "post_update"' in l and '"member_cell": "severe"' in l:
                r = json.loads(l)
                r["benchmark_group_key"] = "A2-D0-w000" if r["benchmark_group_key"] != "A2-D0-w000" else "A3-D0-w000"
                l, done = json.dumps(r), True
            out.append(l)
        return out
    rewrite_jsonl(d / "episode_outcomes.jsonl", fn)


def tamper_macro(d):
    def fn(ls):
        r = json.loads(ls[-1])
        r["v2_behaviour"]["macro_mean_over_base_cells"] += 0.01
        return ls[:-1] + [json.dumps(r)]
    rewrite_jsonl(d / "eval_records.jsonl", fn)


def tamper_td(d):
    def fn(ls):
        r = json.loads(ls[500])
        r["td_residual"] += 1e-6
        return ls[:500] + [json.dumps(r)] + ls[501:]
    rewrite_jsonl(d / "train_credit_diagnostics.jsonl", fn)


base = ["--comparator-evidence-dir", str(C71)]
code, mods = run_extractor(base + ["--out", str(SP / "audit_out")])
results["clean_run_exit"] = code
results["modules_loaded_by_extractor"] = sorted(m for m in mods if m.split(".")[0] in
                                                ("torch", "match_aou", "subprocess", "numpy", "pyomo", "blade"))
results["torch_loaded"] = any(m.split(".")[0] == "torch" for m in sys.modules)
results["match_aou_loaded"] = any(m.split(".")[0] == "match_aou" for m in sys.modules)
results["subprocess_loaded_by_extractor"] = "subprocess" in mods

# hash tamper
sha = (SP / "audit_out" / "artifact_sha256.txt").read_text(encoding="utf-8").splitlines()
for i, l in enumerate(sha):
    if not l.startswith("#") and "run_summary.json" in l:
        sha[i] = ("0" * 64) + l[64:]
        break
(SP / "tampered_sha.txt").write_text("\n".join(sha) + "\n", encoding="utf-8")
results["hash_tamper_exit"] = run_extractor(base + ["--out", str(SP / "t_out_hash"), "--verify-against",
                                                    str(SP / "tampered_sha.txt")])[0]
for name, fn in (("dropped_credit_row", drop_credit_row), ("representation_mix", mix_representation),
                 ("pair_identity", break_pair_identity), ("eval_macro_tamper", tamper_macro),
                 ("gae_td_tamper", tamper_td)):
    d = tamper_dir(name, fn)
    out = SP / ("t_out_" + name)
    code, _ = run_extractor(base + ["--run-dir", str(d), "--out", str(out)])
    results[name + "_exit"] = code
    results[name + "_wrote_output"] = out.exists()
    shutil.rmtree(d)
print(json.dumps(results, indent=1))
