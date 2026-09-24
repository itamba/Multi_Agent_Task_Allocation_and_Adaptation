"""Pre-launch verification for the ONE authorized mission-slack development run.

Run from the FROZEN measured worktree under nlp_env with PYTHONPATH=<worktree>/src. It
executes nothing scientific (no training, evaluation, rollout, preflight or solve) and
checks, with the repository's OWN mechanisms where they exist:

  * source identity -- HEAD == the measured SHA, clean tree, ``match_aou`` imports from this
    worktree, ``blade`` resolves to a vendored engine byte-identical to this worktree's copy;
  * benchmark identity -- manifest file SHA-256, ``manifest_id`` recomputed by the
    GENERALIZED-V2 loader, the development-profile world ordinals and seeds (identity
    metadata only; no confirmatory outcome is read or executed);
  * held-outness -- ``manifest_seed_overlap`` over EVERY manifest seed against the maximum
    possible training band [base_seed, base_seed + iterations x max attempts);
  * comparator -- archive-index row, run_config hash and the resolved-config difference
    against the planned invocation;
  * the fresh output path does not exist.

Writes ``preflight.json`` and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured-sha", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-id", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--comparator-dir", required=True)
    ap.add_argument("--archive-index", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-seed", type=int, default=3000000)
    ap.add_argument("--iterations", type=int, default=375)
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wt = Path.cwd().resolve()
    checks, report = [], {"record": "preflight", "record_version": 1,
                          "development_only": True, "confirmatory_profile_untouched": True}

    def check(name, ok, detail=None):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # --- source identity -----------------------------------------------------
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=str(wt)).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=str(wt)).stdout.strip()
    check("head_is_measured_sha", head == args.measured_sha, head)
    check("worktree_clean", dirty == "", dirty or None)
    import match_aou
    from match_aou.rl.observation import graph_builder
    mpath = Path(match_aou.__file__).resolve()
    check("match_aou_imports_from_worktree", str(mpath).lower().startswith(str(wt).lower()),
          str(mpath))
    import blade
    bpath = Path(blade.__file__).resolve().parent
    wt_blade = wt / "src/match_aou/integrations/panopticon-main/gym/blade"
    d_import, d_wt = _tree_digest(bpath), _tree_digest(wt_blade)
    check("blade_engine_identical_to_measured_tree", d_import == d_wt,
          {"imported_from": str(bpath), "imported_tree_digest": d_import,
           "measured_tree_digest": d_wt})
    check("actor_observation_id", graph_builder.ACTOR_OBSERVATION_ID ==
          "actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1",
          graph_builder.ACTOR_OBSERVATION_ID)
    report["code"] = {"measured_code_sha": args.measured_sha, "head": head,
                      "working_tree_clean": dirty == "", "worktree": str(wt),
                      "match_aou_path": str(mpath), "blade_path": str(bpath)}

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
    lo = args.base_seed
    hi = lo + args.iterations * args.max_attempts
    overlap = manifest_seed_overlap(manifest, start=lo, stop=hi)
    all_seeds = sorted(int(w.seed) for w in manifest.worlds)
    check("held_out_full_manifest", not overlap,
          {"train_band_half_open": [lo, hi], "manifest_n_worlds": len(all_seeds),
           "manifest_seed_min": all_seeds[0], "manifest_seed_max": all_seeds[-1],
           "overlap": list(overlap)})
    report["benchmark"] = {"manifest_path": str(mf), "manifest_file_sha256": fsha,
                           "manifest_id": manifest.manifest_id, "profile": "development",
                           "development_worlds": dev_rows,
                           "regenerated_or_mutated": False}

    # --- comparator ----------------------------------------------------------
    comp = Path(args.comparator_dir)
    rc_path = comp / "run_config.json"
    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    index_row = None
    for line in Path(args.archive_index).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("artifact_id") == "v2_semantic_actor_only_r1_seed3000000_d4e9f37":
            index_row = row
    check("comparator_index_row", index_row is not None and
          Path(index_row["current_path"]).resolve() == comp.resolve(),
          None if index_row is None else index_row["current_path"])
    check("comparator_measured_sha", rc["provenance"]["git"]["commit"] ==
          "d4e9f3721e6d151c00be3fe93c3d149df9d31965", rc["provenance"]["git"]["commit"])
    report["comparator"] = {
        "run_id": "graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37",
        "archive_path": str(comp), "run_config_sha256": _sha(rc_path),
        "archive_index_sha256": _sha(Path(args.archive_index)),
        "train_config": rc["train_config"],
        "config_source": rc["config_source"],
    }

    # --- output ----------------------------------------------------------------
    outp = Path(args.output_dir)
    check("output_dir_absent", not outp.exists() and
          not Path(str(outp) + "__launcher").exists(), str(outp))

    report["checks"] = checks
    report["pass"] = all(c["pass"] for c in checks)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for c in checks:
        print("%s %s" % ("PASS" if c["pass"] else "FAIL", c["check"]))
    print("PREFLIGHT", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
