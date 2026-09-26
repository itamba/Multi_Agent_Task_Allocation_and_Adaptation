"""Pre-launch check for amendment 1 (arm B launch 2). Trains, evaluates and replays nothing.

Run under nlp_env with PYTHONPATH=C:/grc1src/src (the measured source) and PYTHONNOUSERSITE=1,
from the main checkout at the amendment commit. Checks:

  * the corrected rule (``compare_pre_update`` v2) does NOT stop on the recorded arm-A vs
    arm-B-launch-1 pre-update rounds, and records their 24 ego-id / 12 tick differences;
  * it still STOPS on synthetic copies of those records with (i) one selected meta-action
    flipped, (ii) one probability shifted by 1e-5, (iii) one frozen-world identity field
    changed, (iv) one wake removed (copies in a temporary directory; originals untouched);
  * arm B's preset with the new ``--out`` resolves, through the trainer's own parser, to
    ``prelaunch/preflight.json`` arms.B.train_config except ``output_dir``;
  * ``match_aou`` imports from C:\\grc1src at the measured SHA, which is still clean;
  * the new output directory and its launcher sibling are absent.

Usage: python amendment1_check.py <out.json>
"""

import copy
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_pre_update as C  # noqa: E402

MEASURED = "2b570194dea3612f3796999fc589b73d7082ae31"
WT = Path(r"C:\grc1src")
A = Path(r"C:\gruns\reward_c225_r1_s3000000_2b57019")
B1 = Path(r"C:\gruns\reward_c450_r1_s3000000_2b57019")
OUT_B2 = r"C:\gruns\reward_c450_r1_l2_s3000000_2b57019"
PRESET = str(WT / "research_evidence/generalized_v2/reward_coefficient_dev_r1/configs/arm_b_c4p5.json")


def write_pre_update(recs, d: Path):
    d.mkdir(parents=True)
    with open(d / "episode_outcomes.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in recs.values():
            fh.write(json.dumps(r) + "\n")


def main():
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    rep = C.compare(B1, A)
    check("v2_passes_recorded_launch1_round", rep["stop"] is False,
          {"stop_reasons": rep["stop_reasons"],
           "n_ego_id_diffs": rep["record_only"]["n_wake_ego_id_differences"],
           "n_tick_diffs": rep["record_only"]["n_wake_tick_differences"],
           "max_dp": rep["max_abs_semantic_probability_difference"]})
    check("v2_records_the_ego_and_tick_differences",
          rep["record_only"]["n_wake_ego_id_differences"] == 24
          and rep["record_only"]["n_wake_tick_differences"] == 12, None)
    base = C.load_pre_update(B1)
    key = sorted(k for k in base if k[1] == "severe")[0]

    def mutated(fn):
        recs = copy.deepcopy(base)
        fn(recs[key])
        with tempfile.TemporaryDirectory() as tmp:
            write_pre_update(recs, Path(tmp) / "b")
            return C.compare(Path(tmp) / "b", A)

    def flip(r):
        w = r["wake_decisions"][0]
        w["selected_meta_action_name"] = ("PLAN_COMPLIANCE" if w["selected_meta_action_name"]
                                          != "PLAN_COMPLIANCE" else "SELF_PRESERVATION_ABORT")

    def shift(r):
        p = r["wake_decisions"][0]["semantic_probability_per_meta_action"]
        k = sorted(p)[0]
        p[k] = float(p[k]) + 1e-5

    def world(r):
        r["seed"] = int(r["seed"]) + 1

    def drop(r):
        r["wake_decisions"] = r["wake_decisions"][:-1]

    for name, fn in (("selected_action_flip", flip), ("probability_shift_1e-5", shift),
                     ("frozen_world_seed_change", world), ("wake_removed", drop)):
        m = mutated(fn)
        check("v2_stops_on_" + name, m["stop"] is True, m["stop_reasons"])

    # --- configuration of launch 2, through the trainer's own resolution ------
    from match_aou.rl.training import graph_train
    import match_aou
    argv = ["--config", PRESET, "--out", OUT_B2]
    parsed = graph_train._build_arg_parser().parse_args(argv)
    cfg, source = graph_train.resolve_train_config(
        parsed, explicit=graph_train._explicit_cli_dests(argv),
        config_values=graph_train.load_config_file(PRESET), config_path=PRESET)
    cfg.validate()
    tc = json.loads(json.dumps(asdict(cfg), default=list))
    pf = json.loads((HERE.parent / "prelaunch/preflight.json").read_text(encoding="utf-8"))
    want = dict(pf["arms"]["B"]["train_config"], output_dir=OUT_B2)
    check("launch2_config_equals_preflight_except_output_dir", tc == want,
          sorted(k for k in set(tc) | set(want) if tc.get(k) != want.get(k)))
    check("launch2_config_source", source == pf["arms"]["B"]["config_source"], source)
    check("launch2_reward_coefficient", cfg.reward_config().aircraft_penalty_coeff == 4.5, None)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True,
                                    cwd=str(WT)).stdout.strip()
    check("worktree_head_measured_and_clean",
          git("rev-parse", "HEAD") == MEASURED
          and git("status", "--porcelain", "--untracked-files=all") == "", None)
    check("match_aou_from_worktree",
          str(Path(match_aou.__file__).resolve()).lower().startswith(str(WT).lower()),
          str(Path(match_aou.__file__).resolve()))
    check("launch2_output_absent", not Path(OUT_B2).exists()
          and not Path(OUT_B2 + "__launcher").exists(), OUT_B2)
    out = {"record": "amendment1_check", "record_version": 1, "scientific_execution": "none",
           "argv_after_python": ["-m", "match_aou.rl.training.graph_train"] + argv,
           "checks": checks, "pass": all(c["pass"] for c in checks)}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str) + "\n",
                                 encoding="utf-8", newline="\n")
    for c in checks:
        print("%s  %s" % ("PASS" if c["pass"] else "FAIL", c["check"]))
    print("OVERALL", "PASS" if out["pass"] else "FAIL")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
