# Actor-only credit-to-update diagnostic R1 — evidence package

**Status: EXECUTED / UNREVIEWED.** Development only; one training seed; no verdict. Everything
below is the executing task's reading of its own evidence, for GPT exact-candidate review.

| Item | Value |
|---|---|
| Run id | `graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` |
| Measured code SHA | `644883b89c208255f5808432462be64be5d7589f` (branch `task/actor-credit-update-diagnostic-r1`, draft PR #80, unmerged when measured), run from the clean detached worktree `C:\gcud1src`; `run_config.json:/provenance/git` records `dirty = false` |
| Original run directory (external, authoritative) | `C:\gruns\graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` and `…__launcher` |
| Comparator | `graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` (measured `3bc944119da08af8e25268c9ee83fc63a8d1e533`), first 100 updates and rounds 0 / 25 / 50 / 75 / 100; read only, pinned before launch |
| Plan | `authorized_plan.json` (written and committed before launch) |
| Contract | [artifacts and metrics §5.3](../../../docs/contracts/artifacts_metrics.md#53-actor-only-step-diagnostics) |

## Layout

- `authorized_plan.json` — the bounded plan, including the declared prefix comparison (tolerance
  0) and its stop rule.
- `prelaunch/` — `preflight.json` / `.log` (all checks PASS except the byte-level BLADE tree
  digest, see below), `preflight_blade_blob_identity.json` (PASS), `nlp_env_step_gate.json` /
  `.log` (G1–G3 PASS under nlp_env, torch 2.7.1), `compare_prefix_selfcheck.json` (PASS), and the
  base-env suite summaries (845 passed, 11 skipped — working tree and measured worktree).
- `run_artifacts/` — byte-identical copies: `run_config.json`, `run_summary.json`,
  `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl` (empty),
  `train_credit_diagnostics.jsonl`, the COMPLETE `train_actor_step_diagnostics.jsonl` (100
  records), `train_actor_step_vectors/` (the five epoch-0 vector files, iterations 0 / 24 / 49 /
  74 / 99, with `layout_json`), and `launcher/` (launcher record, exit code, console log, live
  prefix-monitor report and first-divergence record).
- `extracted/` — deterministic outputs of `scripts/extract_evidence.py` (`--check` regenerates
  them byte-identically): `per_update_diagnostic.jsonl` (every update), `per_update_table.md`,
  `primary_diagnostic.json` (declared windows), `per_epoch_summary.json`,
  `secondary_behaviour.json`, `accounting.json`, `vector_spot_checks.json`,
  `primary_diagnostic.png`; plus `prefix_comparison.json` (`scripts/compare_prefix.py`, final) and
  `training_stream_divergence.json` (`scripts/training_stream_divergence.py`, descriptive).
- `post_run_checks/` — two synthetic engineering probes run after the observation in the section
  "Same-seed prefix" (`scripts/realistic_bitwise_probe.py`, `scripts/numeric_sensitivity_probe.py`).
- `artifact_sha256.txt` — every external source artifact (SHA-256, bytes, path);
  `committed_files.txt` — every committed file with its SHA-256 and Git blob id.
- Not committed (external, identified in `artifact_sha256.txt`): `episode_outcomes.jsonl`
  (≈ 30 MB), the four checkpoints, plots and generated scenarios.

## Validity facts

- **Completion:** launcher `termination_reason = exited`, child exit code `0`, walltime 2139 s
  (cap 4 h); 100 / 100 productive updates, 400 optimizer steps; no `CRASH` / `Traceback` in the
  console; checkpoints at 24 / 49 / 74 / 99.
- **Accounting:** 800 / 800 successful training episodes, 0 failures (no replacement needed);
  5 evaluation rounds, 300 / 300 episodes successful; `accounting_reconciled = true`. Every round:
  20 / 20 groups metric-eligible, 10 / 10 base cells defined.
- **Configuration:** the resolved `train_config` differs from the original's exactly in
  `n_iterations` (375 → 100), `output_dir`, `actor_step_diagnostics` and
  `actor_step_vector_iterations` (pre-launch check with the trainer's own resolver).
- **Identity:** manifest `ef17a68a…46ea8`, file SHA-256 `dd72afc9…a103`, development profile, held
  out against `[3000000, 3001200)`; every original file pinned by the PR #79 package matched its
  SHA-256 and size before launch.
- **BLADE engine:** the byte-level tree digest of the imported copy (main checkout) and of the
  measured worktree differed in all 25 files, **only by line endings** (`core.autocrlf`); both
  copies are Git-blob-identical to the measured tree (`preflight_blade_blob_identity.json`) — the
  same condition the PR #79 pre-launch recorded. Nothing under the engine changed between base and
  measured SHA.
- **Diagnostic stream:** exactly one record per productive update, four epochs each; maximum
  telescoping residual 0.0, maximum epoch-boundary mismatch 0.0, maximum group-sum relative
  residual 3.6e-7, `total_loss` vs real backward gradient relative residual 0.0; the five vector
  files recompute the recorded norms, pressures and first-order predictions to ≤ 1.4e-16
  relative. Iteration 74's batch had no MILD row, so its vector file has no `contrast_grad`
  (reason recorded).

## Same-seed prefix (instrumentation consistency, not replication)

Declared rule: tolerance 0; the first divergence, in the new run's stream order, is classified;
only a `policy_side` first divergence stops the run.

- **Declared first divergence:** `input_side` — pre-update evaluation episode 14 (seed
  `2000448`), an ordinary wake at tick 1804 versus 1805: the pre-declared BLADE run-to-run
  nondeterminism class. 12 / 60 pre-update episodes differ; the pre-update round's evaluation
  record is identical. **The stop rule did not fire and the run continued.**
- **Training stream (descriptive, `training_stream_divergence.json`):** every seed and cell of
  all 800 training episodes matches. Iteration 0 is fully identical (8 / 8 episodes, 17 / 17
  credit rows). **From iteration 1 every training wake differs at float32-ulp level in its actor
  outputs from identical recorded inputs** (first: seed `3000008`, `source_scores[0][0]`
  `0.05741109326481819` vs `0.05741110071539879`, |Δ| 7.5e-9) — i.e. update 0 produced
  ulp-different parameters from identical recorded inputs. Differences grow chaotically (≤ 1.5e-8
  at iteration 1–2, ~2e-6 by 12, ~0.09 by 97); the first input-side training divergence is at
  iteration 13. Checkpoint maximum parameter differences: 5.1e-6 (24), 1.9e-5 (49), 2.7e-3 (74),
  0.045 (99).
- **This is recorded as an unresolved discrepancy, not a claim of reproduction.** Engineering
  evidence: under nlp_env on realistic graph sizes, the base updater (compiled from `bcb1746`),
  the OFF and the ON updater are bit-identical in one process and identical across two processes
  (`post_run_checks/realistic_bitwise_probe_process{1,2}.json`); the update's float result does
  NOT change with heap-allocation / alignment shifts but DOES change with the intra-op thread
  count (1 / 2 / 3 / 4 threads give four bit patterns, `numeric_sensitivity_probe.json`). Neither
  run records its effective thread configuration, so a different effective parallel partition in
  either process can be neither confirmed nor excluded. Had the declared stream order put
  training before evaluation, this ulp-level policy-side difference would have been the first
  divergence; the rule was not adapted after seeing it.
- Evaluation rounds 25–100 therefore compare two drifting trajectories (below).

## Primary diagnostic (every update; declared windows)

`C_B` = mean P(ABORT | SEVERE immediate-FD rows) − mean P(ABORT | MILD rows) of the batch;
pressure = −dot(h, g) of a raw loss-gradient component (first-order change of `C_B` per unit raw
descent step); prediction = dot(h, Δθ) of the actual Adam step. 83 / 100 updates hold both
severities (undefined: 1, 7, 11, 13, 26, 32, 40, 55, 61, 74, 77, 81, 83, 84, 90, 93, 94).

| Window | defined | median full ΔC_B | mean \|ΔC_B\| | ΔC_B + / − | epoch-0 FD pressure + | FD+ & non-FD− | FD+ → total− | median cos(Δθ, h) | raw-total vs Adam-prediction sign agree | prediction vs actual sign agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–24 | 21 | `+3.8e-4` | `1.6e-3` | 11 / 10 | 13 | 3 | 1 | `+0.010` | 52 / 84 | 79 / 83 |
| 25–49 | 22 | `−2.5e-5` | `4.3e-4` | 10 / 11 (1 negligible) | 11 | 2 | 0 | `−0.018` | 56 / 88 | 84 / 84 |
| 50–74 | 22 | `−1.8e-4` | `6.5e-4` | 9 / 13 | 11 | 9 | 6 | `−0.020` | 62 / 88 | 86 / 88 |
| 75–99 | 18 | `+1.8e-3` | `9.7e-3` | 9 / 9 | 14 | 1 | 0 | `+0.058` | 50 / 72 | 52 / 72 |
| all | 83 | `−4.6e-5` | `2.8e-3` | 39 / 43 (1) | 49 | 15 | 7 | `+0.008` | 220 / 332 | 301 / 327 |

Further facts (`primary_diagnostic.json`, `per_epoch_summary.json`):

- The batch contrast itself stays near zero: `C_B` before an update has median `+1.6e-3`
  (q10 `−0.024`, q90 `+0.018`); mean P(ABORT) on SEVERE and MILD FD rows is `0.188` / `0.188`.
- Epoch-0 FD pressure: median `+2.2e-3`, mean `−1.3e-3`; SEVERE rows positive 48 / 83, MILD
  rows 42 / 83. Only in 75–99 is it coherent (14 / 18; SEVERE rows 17 / 18).
- Non-FD pressure is positive in 55 / 83 (median `+2.9e-3`), i.e. mostly aligned, not opposing;
  opposing (FD+ with non-FD−) 15 / 83, concentrated in 50–74 (9 / 22).
- The entropy component is two orders of magnitude smaller (mean |pressure| `3.4e-4` vs FD
  `4.5e-2`) and flips the sign of the total pressure in 1 / 83 updates.
- The actual Adam step is nearly orthogonal to `h`: cos(Δθ, h) median `+0.008` (q10 `−0.20`,
  q90 `+0.22`); its cosine with the raw descent direction −g_total has median `+0.13`. 205 / 332
  epochs are gradient-norm clipped; PPO ratio clipping binds on FD rows in < 1 % of rows.
- First-order predictions match the observed epoch changes closely in 0–74 (summed linearization
  residual median ≤ 4e-7); in 75–99 the summed predictions (mean `+0.013`) overstate the actual
  change (mean `+0.00085`) — residual mean `−0.012`, sign agreement 52 / 72.

## Secondary behaviour (five matched development rounds)

| Updates | Macro SEVERE − MILD P(ABORT) (new / original) | Directional switches (new / original) | Reverse | mean P(ABORT) MILD / SEVERE (new) |
|---:|---:|---:|---:|---:|
| 0 | `+2.62e-4` / `+2.62e-4` (identical round) | 3 / 3 | 0 / 0 | 0.5000 / 0.5003 |
| 25 | `+3.05e-6` / `+3.04e-6` | 0 / 0 | 0 / 0 | 0.1725 / 0.1725 |
| 50 | `−6.73e-7` / `−6.91e-7` | 0 / 0 | 0 / 0 | 0.1372 / 0.1372 |
| 75 | `+1.54e-4` / `+1.10e-4` | 0 / 0 | 0 / 0 | 0.0831 / 0.0833 |
| 100 (final) | `+6.92e-3` / `+5.04e-7` | 0 / 0 | 0 / 0 | 0.1484 / 0.1553 |

Every round: 20 / 20 groups, 10 / 10 base cells; evaluation reward mean equal to the original's
in every round. The final-round difference reflects the drifting trajectories, not a claimed
effect of the observer.

## Reading (development only; hypotheses, not proofs)

**Where the intended separation fails to become policy change — simplest reading:** the policy
barely discriminates the two severities on the FD rows it trains on (`C_B` ≈ 0), the raw FD
signal toward separation is weak and sign-inconsistent for most of the prefix, and the actual
Adam displacement is almost orthogonal to the contrast gradient, so each update moves `C_B` by
~1e-3 in either direction and nothing accumulates. Counteraction by non-FD transitions or by
entropy is not the main factor (visible only in 50–74). When the FD signal becomes coherent
(75–99), the first-order gain of the actual steps is largely cancelled by non-linearity.

Alternatives not excluded: the Adam direction is set by the whole batch's per-parameter
normalization and by clipping, so near-orthogonality may reflect the network parametrization
rather than a defect; `C_B` is a within-batch training-population contrast, not the matched-world
endpoint; one seed, 100 updates, development worlds only.

Non-claims: no causal attribution, no claim of a defective critic, reward or optimizer; negative
advantages and cross-episode ABORT-vs-PLAN means are not action values; no trajectory
reproduction of the original is claimed; no confirmatory evidence exists.
