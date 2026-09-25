"""Pre-launch verification for the ONE authorized credit-to-update diagnostic.

Run from the FROZEN measured worktree under nlp_env with PYTHONPATH=<worktree>/src. It executes
nothing scientific (no training, evaluation, rollout, preflight or solve) and checks:

  * source identity -- HEAD == the measured SHA, clean tree, ``match_aou`` (and the two changed
    modules) import from this worktree, ``blade`` resolves to a vendored engine byte-identical
    to this worktree's copy;
  * benchmark identity -- manifest file SHA-256, ``manifest_id`` recomputed by the V2 loader,
    the 20 development worlds (identity metadata only; no confirmatory outcome is read);
  * held-outness -- ``manifest_seed_overlap`` over EVERY manifest seed against the maximum
    training band [base_seed, base_seed + iterations x max attempts);
  * the original (comparator) -- every file the PR #79 evidence pinned still has its recorded
    SHA-256 and size; its checkpoints at 24 / 49 / 74 / 99 are hashed and pinned here; its
    measured SHA; its prefix seeds all fall inside this run's band;
  * the effective configuration -- the planned argv resolved by the trainer's OWN
    ``resolve_train_config`` differs from the original's ``train_config`` exactly in the
    declared keys;
  * the fresh output path does not exist.

Writes ``preflight.json`` and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_digest(root: Path) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            lines.append("%s %s\n" % (p.relative_to(root).as_posix(), _sha(p)))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


EXPECTED_CONFIG_DIFF = {"n_iterations", "output_dir", "actor_step_diagnostics",
                        "actor_step_vector_iterations"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured-sha", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-id", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--original-dir", required=True)
    ap.add_argument("--original-pins", required=True, help="PR #79 artifact_sha256.txt")
    ap.add_argument("--original-measured-sha", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wt = Path.cwd().resolve()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    argv = [a.replace("<resolved output dir>", args.output_dir)
            for a in plan["execution"]["argv_after_python"]]
    checks, report = [], {"record": "preflight", "record_version": 1,
                          "development_only": True, "confirmatory_profile_untouched": True}

    def check(name, ok, detail=None):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # --- source identity -----------------------------------------------------
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=str(wt)).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True, cwd=str(wt)).stdout.strip()
    check("head_is_measured_sha", head == args.measured_sha, head)
    check("worktree_clean", dirty == "", dirty or None)
    import match_aou
    from match_aou.rl.training import graph_ppo, graph_train
    paths = {m.__name__: str(Path(m.__file__).resolve())
             for m in (match_aou, graph_ppo, graph_train)}
    check("modules_import_from_worktree",
          all(p.lower().startswith(str(wt).lower()) for p in paths.values()), paths)
    import blade
    bpath = Path(blade.__file__).resolve().parent
    wt_blade = wt / "src/match_aou/integrations/panopticon-main/gym/blade"
    d_import, d_wt = _tree_digest(bpath), _tree_digest(wt_blade)
    check("blade_engine_identical_to_measured_tree", d_import == d_wt,
          {"imported_from": str(bpath), "imported_tree_digest": d_import,
           "measured_tree_digest": d_wt})
    check("step_diagnostics_present",
          graph_train._ACTOR_STEP_DIAGNOSTICS_FILENAME == "train_actor_step_diagnostics.jsonl"
          and hasattr(graph_ppo, "ActorStepReport"), None)
    report["code"] = {"measured_code_sha": args.measured_sha, "head": head,
                      "working_tree_clean": dirty == "", "worktree": str(wt),
                      "module_paths": paths, "blade_path": str(bpath),
                      "blade_tree_digest": d_import}

    # --- effective configuration --------------------------------------------
    parser = graph_train._build_arg_parser()
    parsed = parser.parse_args(argv[2:])       # drop "-m match_aou.rl.training.graph_train"
    cfg, config_source = graph_train.resolve_train_config(
        parsed, explicit=graph_train._explicit_cli_dests(argv[2:]))
    cfg.validate()
    resolved = json.loads(json.dumps(asdict(cfg), default=list))
    orig = Path(args.original_dir)
    orig_rc = json.loads((orig / "run_config.json").read_text(encoding="utf-8"))
    orig_tc = orig_rc["train_config"]
    diff = sorted(k for k in set(resolved) | set(orig_tc)
                  if resolved.get(k, "<absent>") != orig_tc.get(k, "<absent>"))
    check("config_diff_is_exactly_the_declared_keys", set(diff) == EXPECTED_CONFIG_DIFF,
          {k: {"new": resolved.get(k, "<absent>"), "original": orig_tc.get(k, "<absent>")}
           for k in diff})
    check("config_source_cli_defaults", config_source["resolved_from"] == "cli_defaults",
          config_source)
    for key, want in (("n_iterations", 100), ("actor_step_diagnostics", True),
                      ("actor_gradient_diagnostics", False), ("visual_artifacts", False),
                      ("early_stopping", False), ("training_mode", "actor_only"),
                      ("base_seed", 3000000), ("episode_design", "generalized_v2"),
                      ("match_aou_backend", "p1_milp_v1")):
        check("config_%s" % key, resolved.get(key) == want, resolved.get(key))
    check("config_vector_iterations", list(resolved["actor_step_vector_iterations"])
          == [0, 24, 49, 74, 99], resolved["actor_step_vector_iterations"])
    report["effective_config"] = {"argv_after_python": argv, "train_config": resolved,
                                  "config_source": config_source, "diff_vs_original": diff}

    # --- benchmark identity and held-outness --------------------------------
    from match_aou.rl.training.graph_generalized import (
        load_v2_benchmark_manifest, manifest_seed_overlap)
    mf = Path(args.manifest)
    fsha = _sha(mf)
    check("manifest_file_sha256", fsha == args.manifest_sha256, fsha)
    manifest = load_v2_benchmark_manifest(mf)
    check("manifest_id", manifest.manifest_id == args.manifest_id, manifest.manifest_id)
    dev = manifest.profile_worlds("development")
    dev_rows = [{"world_ordinal": int(getattr(w, "world_ordinal", -1)),
                 "seed": int(w.seed), "key": str(getattr(w, "key", ""))} for w in dev]
    check("development_profile_worlds", len(dev_rows) == 20, len(dev_rows))
    lo = int(cfg.base_seed)
    hi = lo + int(cfg.n_iterations) * int(cfg.generalized_max_attempts_per_iteration)
    overlap = manifest_seed_overlap(manifest, start=lo, stop=hi)
    all_seeds = sorted(int(w.seed) for w in manifest.worlds)
    check("held_out_full_manifest", not overlap,
          {"train_band_half_open": [lo, hi], "manifest_n_worlds": len(all_seeds),
           "manifest_seed_min": all_seeds[0], "manifest_seed_max": all_seeds[-1],
           "overlap": list(overlap)})
    report["benchmark"] = {"manifest_path": str(mf), "manifest_file_sha256": fsha,
                           "manifest_id": manifest.manifest_id, "profile": "development",
                           "development_worlds": dev_rows, "regenerated_or_mutated": False}

    # --- the original run: pinned identity ----------------------------------
    pins, pin_rows = {}, []
    for line in Path(args.original_pins).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sha, size, group, path = line.split("  ", 3)
        if group in ("run", "launcher"):
            pins[path] = (sha, int(size))
    for path, (sha, size) in sorted(pins.items()):
        p = Path(path)
        got = (_sha(p), p.stat().st_size) if p.exists() else (None, None)
        pin_rows.append({"path": path, "pinned_sha256": sha, "sha256": got[0],
                         "bytes": got[1], "match": got == (sha, size)})
    check("original_files_match_pr79_pins", pin_rows and all(r["match"] for r in pin_rows),
          [r for r in pin_rows if not r["match"]] or len(pin_rows))
    ckpts = []
    for it in (24, 49, 74, 99):
        p = orig / "checkpoints" / ("ckpt_iter%04d.pt" % it)
        ckpts.append({"path": str(p), "exists": p.exists(),
                      "sha256": _sha(p) if p.exists() else None,
                      "bytes": p.stat().st_size if p.exists() else None})
    check("original_prefix_checkpoints_present", all(c["exists"] for c in ckpts), None)
    check("original_measured_sha",
          orig_rc["provenance"]["git"]["commit"] == args.original_measured_sha,
          orig_rc["provenance"]["git"]["commit"])
    train = [json.loads(x) for x in (orig / "train_records.jsonl").read_text(
        encoding="utf-8").splitlines() if x]
    seeds = []
    with open(orig / "episode_outcomes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("phase") == "train" and r.get("iteration") is not None \
                    and r["iteration"] <= 99:
                seeds.append(int(r["seed"]))
    attempted_prefix = sum(int(r["n_attempted"]) for r in train if r["iteration"] <= 99)
    check("original_prefix_seeds_inside_band",
          seeds and lo <= min(seeds) and max(seeds) < hi,
          {"min": min(seeds), "max": max(seeds), "n_successful": len(seeds),
           "n_attempted": attempted_prefix})
    report["original"] = {"run_dir": str(orig), "measured_code_sha": args.original_measured_sha,
                          "pinned_files": pin_rows, "prefix_checkpoints": ckpts,
                          "run_config_sha256": _sha(orig / "run_config.json"),
                          "prefix_successful_training_episodes": len(seeds),
                          "prefix_training_attempts": attempted_prefix}

    # --- output ----------------------------------------------------------------
    outp = Path(args.output_dir)
    check("output_dir_absent", not outp.exists() and
          not Path(str(outp) + "__launcher").exists(), str(outp))

    report["checks"] = checks
    report["pass"] = all(c["pass"] for c in checks)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for c in checks:
        print("%s  %s" % ("PASS" if c["pass"] else "FAIL", c["check"]))
    print("OVERALL", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
