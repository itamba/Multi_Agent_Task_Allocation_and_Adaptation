# Acting-ego CTDE FD100 development diagnostic — compact evidence package

**Development evidence only; not confirmatory. Cross-version.** This package preserves the
completed 100-update role-only acting-ego CTDE DEVELOPMENT diagnostic for independent review.

## Formal status

- Implementation candidate `68055e39768d5fa601e5960a9f08823b9e65c08f` (PR #75) was
  **GPT-approved for measurement**, and the run was executed under the user-authorized bounded
  plan recorded in `originals/authorized_plan.json`.
- The run **has completed**.
- **Measurement verdict:** `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` (GPT review of
  this evidence package, 2026-09-19). The interpretation is recorded in
  [`docs/history/measurements.md` §12](../../../docs/history/measurements.md#12-generalized-v2-role-only-acting-ego-ctde-development-diagnostic),
  not here.
- **No merge is authorized.** **No confirmatory profile was used.** Preserving this evidence
  **authorizes no new scientific execution**.

## Identity

| | New run (acting-ego) | Historical comparator (FD100) |
|---|---|---|
| Run ID | `graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` | `graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` |
| Original directory (authoritative, external) | `C:\gruns\graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` | `C:\gruns\graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` |
| Measured code SHA | `68055e39768d5fa601e5960a9f08823b9e65c08f` (branch `task/ctde-acting-ego-conditioning`, PR #75 head, unmerged) | `6ed964a1abd09de2130aee3d0d314c8f32165056` (PR #74 approved head, unmerged when measured) |
| Updates | 100 | 150 (compared only through update 100) |

- **Clean provenance:** `run_config.json:/provenance/git` records commit `68055e3…`,
  `dirty = false`, 0 dirty paths; `preflight.json:/code` records the checkout head, origin branch
  head and PR #75 head all equal to it, with an empty `git status --porcelain --untracked-files=all`.
- **Benchmark:** manifest `C:\gra\benchmarks\v2_preflight_seed2000000_ae42cb0\benchmark_manifest.json`,
  `manifest_id ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`, file SHA-256
  `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`; `development` profile only
  (world ordinals 0 and 1, 20 worlds, 60 members per round); held out against the maximum
  training band `[3000000, 3001200)` over the entire manifest, overlap 0.
- **Cross-version, not a one-field difference.** The measured code differs: PR #75 changes the
  critic's input state (the current decision owner's node takes the encoder's existing EGO role).
  The **resolved configuration** matches the historical FD100 run except `n_iterations 150 → 100`
  and `output_dir`; every other differing `run_config.json` path is derived from `n_iterations`
  (seed-band stop / count, attempt budget, quota) or provenance (git commit / branch, the argv
  `--iterations` / `--out` values, collection time). The list is
  `evidence_summary.json:/acting_ego/config_comparison_vs_historical_fd100`.

## Completion and accounting (new run)

100 / 100 productive updates; 800 / 800 successful training episodes over 800 attempts (budget
1200), 0 failed, 0 replacements; 2982 transitions = `train_records` Σ `n_transitions` = credit
rows; `accounting_reconciled = true`. FD100: 0 clean, 800 damaged (421 MILD, 379 SEVERE), 800
events applied, 800 FD wakes. 5 evaluation rounds, 300 / 300 members successful.
`episode_failures.jsonl` is 0 bytes; `native_exit_code.txt` is `0`; the console's closing
`TRAINING SUMMARY` block and its Traceback count (0) are in
`evidence_summary.json:/acting_ego/completion/console`. Wall clock 13:34:18 → 14:09:27 local,
2026-09-19.

## Reproduced metrics (common range through update 100)

Held-out DEVELOPMENT macro `SEVERE − MILD P(SELF_PRESERVATION_ABORT)`; every round has 10 / 10
base cells defined and 20 / 20 matched groups eligible; directional switches out of 20, reverse
switches 0 throughout.

| Update | Acting-ego macro | switches | Historical FD100 macro | switches |
|---|---|---|---|---|
| 0 | −0.000516 | 0/20 | −0.000516 | 0/20 |
| 25 | −0.0000357 | 0/20 | +0.0000907 | 0/20 |
| 50 | +0.0000326 | 0/20 | +0.00208 | 0/20 |
| 75 | +0.00764 | 0/20 | +0.0428 | 0/20 |
| 100 | −0.00109 | 0/20 | −0.0000086 | 0/20 |

Immediate-FD rows of the FD-selected ego (800 per run in iterations 0–99), within-update
`SEVERE − MILD`, median over updates holding both severities (99 updates over 0–99; 50 over 50–99):

| Quantity | Acting-ego 0–99 | 50–99 | Historical 0–99 | 50–99 |
|---|---|---|---|---|
| `value_old` | −0.0205 | −0.0236 | −0.0000279 | −0.0000019 |
| `value_target` | −0.330 | −0.369 | −0.317 | −0.361 |
| `raw_advantage` | −0.319 | −0.357 | −0.316 | −0.360 |
| `td_residual` | −0.00178 | −0.00306 | −0.0000886 | −0.000109 |

`value_old` by 25-update window (median): acting-ego −0.0024 / −0.0549 / −0.0132 / −0.0583;
historical −0.0006 / +0.0001 / −0.00003 / +0.000007.

Nonterminal immediate-FD decisions (not the episode's last decision): acting-ego 706 (94
terminal), historical 699 (101 terminal); no nonterminal row has `raw_advantage = 0`.

| | Acting-ego | Historical |
|---|---|---|
| median `|td_residual| / |raw_advantage|` | 0.0470 | 0.00307 |
| median `|local|` (TD residual) | 0.00680 | 0.000518 |
| median `|future|` (`γλ·A_{t+1}`) | 0.188 | 0.198 |
| mean `|local|` / mean `|future|` | 0.123 | 0.0547 |
| fraction `|future| > |local|` | 0.921 | 0.980 |
| within-update `SEVERE − MILD` local, median (0–99 / 50–99) | −0.00061 / −0.00113 | +0.000077 / −0.0000055 |
| within-update `SEVERE − MILD` future, median (0–99 / 50–99) | −0.311 / −0.366 | −0.319 / −0.346 |

Actor-gradient separation pressure (updates with a defined contrast; 25 / 25 in each window):

| Window | Run | FD + | non-FD + | total + | FD + ∧ total − | FD median | FD mean | non-FD median | total median |
|---|---|---|---|---|---|---|---|---|---|
| 50–74 | acting-ego | 15 | 15 | 15 | 2 | +0.00213 | −0.00545 | +0.00298 | +0.00581 |
| 50–74 | historical | 20 | 21 | 18 | 3 | +0.0195 | +0.00683 | +0.0190 | +0.0533 |
| 75–99 | acting-ego | 18 | 11 | 13 | 6 | +0.00122 | −0.00023 | −0.00123 | +0.00051 |
| 75–99 | historical | 13 | 9 | 12 | 2 | +0.00026 | −0.0128 | −0.00066 | −0.00068 |

Entropy sign flip (actor-loss vs policy-surrogate separation pressure differ in sign): acting-ego
1 / 99 defined updates, historical 0 / 99 over updates 0–99; minimum surrogate-vs-actor-loss
cosine over all rows 0.878 (acting-ego) and 0.988 (historical, 0–99). Maximum gradient
reconstruction relative error 5.9e-7 (acting-ego).

All quantities are descriptive and non-counterfactual; definitions are in
`evidence_summary.json:/definitions`.

## Files

| Path | Content |
|---|---|
| `evidence_summary.json` | formal status, both SHAs, identity / validity / accounting, benchmark and held-out identity, config comparison, console completion block, trajectories, credit and gradient blocks for both runs, `integrity_checks` |
| `artifact_manifest.json` | SHA-256 and bytes of every top-level file of BOTH run directories; tree digests of `checkpoints/`, `plots/`, `scenarios/` (per-file for checkpoints); external manifest identity |
| `originals/` | **byte-identical copies** of the new run's `authorized_plan.json`, `preflight.json`, `launch.cmd`, `launch_record.json`, `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `train_actor_gradient_diagnostics.jsonl`, `episode_failures.jsonl` (empty), `invocation_start_local.txt`, `native_exit_code.txt` |
| `derived/immediate_fd_credit_rows_*.jsonl` | per-row projection of the immediate-FD, FD-selected-ego credit rows (both runs, iterations 0–99), with the episode-terminal flag |
| `derived/actor_gradient_rows_*.jsonl` | per-update projection of the actor-gradient records (both runs, iterations 0–99) |
| `derived/evaluation_trajectory_historical_fd100_u0_100.jsonl` | historical evaluation rounds 0–100 (the new run's are in `originals/eval_records.jsonl`) |
| `scripts/extract_evidence.py` | standard-library extractor and checker that produced every file above |
| `.gitattributes` | `* -text`: every file is stored byte-exact under any `core.autocrlf` |

**Not committed** (identified by SHA-256 and bytes in `artifact_manifest.json`): the new run's
`train_credit_diagnostics.jsonl` (3 977 235 B), `episode_outcomes.jsonl` (26 134 027 B),
`training_console.log` (1 876 443 B), checkpoints, plots and scenarios, and every file of the
historical run (already indexed by `../semantic_ctde_grad_diag_r1/`).

## Reproduce and verify

From the repository root:

```bash
python research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/scripts/extract_evidence.py --check
python research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/scripts/extract_evidence.py --from-derived
```

`--check` reads both ORIGINAL run directories, regenerates every package file in memory,
requires byte identity with the committed files, and runs the integrity checks (measured SHA and
clean tree, preflight = resolved config, launch-record hashes, config differences limited to the
authorized keys, manifest identity / profile / held-out, completion and accounting, FD100 counts,
exit code, and the historical run's bytes equal to the PR #74 index). `--from-derived` needs no
run directory: it recomputes the trajectory, credit and gradient blocks from `derived/` and
`originals/` alone.

**Recorded, not normalized:** `launch_record.json` was written by PowerShell and begins with a
UTF-8 byte-order mark; it is preserved as written and read with `utf-8-sig`.
