# Actor-only credit-to-update diagnostic R1 — evidence package

**Status: EXECUTED; GPT exact-head review of `5f5e22ae5f3c108af0a24e18936a87fb5c31dd7b` was
CHANGES_REQUESTED (2026-09-25); review fixes F1–F3 added; awaiting exact-head re-review.**
Development only; one training seed; no verdict. The executing task's reading is below. Three
statuses are kept separate and none of them is an "all gates passed" result:

| Status | State |
|---|---|
| Engineering non-interference gates (synthetic data) | **passed as reported** by this task (tests A1–A11, nlp_env gate G1–G3, comparator self-check); the review re-ran none of them |
| Execution and accounting | **completed**: exit 0, 100 / 100 updates, 800 / 800 training and 300 / 300 evaluation episodes, reconciled |
| Consistency with the original run's trajectory | **UNRESOLVED** — see "Same-seed prefix" and the protocol deviation D1 |

| Item | Value |
|---|---|
| Run id | `graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` |
| Measured code SHA | `644883b89c208255f5808432462be64be5d7589f` (branch `task/actor-credit-update-diagnostic-r1`, draft PR #80, unmerged when measured), run from the clean detached worktree `C:\gcud1src`; `run_config.json:/provenance/git` records `dirty = false` |
| Original run directory (external, authoritative) | `C:\gruns\graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` and `…__launcher` |
| Comparator | `graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` (measured `3bc944119da08af8e25268c9ee83fc63a8d1e533`), first 100 updates and rounds 0 / 25 / 50 / 75 / 100; read only, pinned before launch |
| Plan | `authorized_plan.json` (written and committed before launch; kept unchanged as the historical plan) |
| Contract | [artifacts and metrics §5.3](../../../docs/contracts/artifacts_metrics.md#53-actor-only-step-diagnostics) |

## Protocol deviations (recorded at review; not retroactively authorized)

**D1 — the executed stop rule was narrower than the dispatch requirement.**

- Dispatch packet `CC_Actor_Credit_Update_Diagnostic_R1`, verbatim: "Preserve exact equality where
  obtained and report the first divergence otherwise. Do not silently claim historical trajectory
  reproduction. A material unexplained discrepancy stops the run; do not adapt tolerances after
  seeing it. Declare any numerical comparison tolerances before launch."
- Executed, precommitted rule (`authorized_plan.json:/prefix_comparison/stop_rule`, verbatim): "a
  policy_side first divergence is a MATERIAL UNEXPLAINED DISCREPANCY (different actor outputs from
  identical inputs = different parameters, which the observational diagnostic must never cause):
  the launcher kills the run and preserves partial evidence. An input_side first divergence is the
  pre-declared known class (BLADE run-to-run execution nondeterminism, measurements section 16.3):
  it is reported and the run continues; later records are compared only descriptively".
- Effect: the executed comparator returns `stop` only from the FIRST divergence of one combined
  stream (evaluation records ordered before training). The pre-update evaluation tick mismatch
  became first, so every later comparison — including the independent training stream, which
  diverged after an apparently matching update-0 batch — could no longer stop the run.
  Precommitting that narrower rule made it transparent; **it does not establish compliance with
  the dispatch requirement.** The run did not stop and is not described as having stopped.

**D2 — offline checkpoint loading.** The dispatch said "Checkpoints every 25 updates and final;
save only. No historical checkpoint loading, warm start or resume", and the plan "save only …;
no loading, warm start or resume". `scripts/compare_prefix.py:compare_checkpoints`, run once for
the final report, deserialized the new run's and the original's checkpoints 24 / 49 / 74 / 99 with
`torch.load` for an offline tensor comparison. It was not inference, replay, warm start or resume,
but it was checkpoint loading, which neither the dispatch nor the plan authorized. It is not
repeated; the review fixes load no checkpoint.

**Historical scripts are evidence of what ran, not validated reusable guards.**
`scripts/compare_prefix.py` and `scripts/launch_with_watchdog.py` are kept byte-unchanged as the
record of the executed comparator and launcher. They are NOT a validated consistency guard: a
future run must monitor training / update consistency independently of evaluation, so that a
benign evaluation discrepancy cannot waive a later unexplained learning discrepancy, and a monitor
failure must not silently grant continued execution (this launcher recorded monitor exceptions
and continued; 0 occurred in this run).

## Layout

- `authorized_plan.json` — the bounded plan (historical, unchanged).
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
  `primary_diagnostic.png`; `prefix_comparison.json` (`scripts/compare_prefix.py`, final),
  `training_stream_divergence.json` (`scripts/training_stream_divergence.py`, descriptive) and
  `review_derived_summaries.json` (`scripts/review_derived_summaries.py`, **review-derived
  arithmetic, not a predeclared endpoint**; `--check` rebuilds it byte-identically).
- `prefix_source_extract/` (review fix F2) — the decisive comparison records of BOTH runs as
  original JSONL line bytes, with `index.json` (source paths, pinned SHA-256 checked before and
  after, line numbers, keys, per-line hashes) and `comparison_report.json`;
  `scripts/extract_prefix_source.py verify <package>` re-derives the report byte-identically from
  the committed extract alone (no Windows originals, no model execution).
- `post_run_checks/` — two synthetic engineering probes run after the observation described below
  (`scripts/realistic_bitwise_probe.py`, `scripts/numeric_sensitivity_probe.py`).
- `artifact_sha256.txt` — every external source artifact (SHA-256, bytes, path);
  `committed_files.txt` — every committed package file with its SHA-256, byte size and Git blob id,
  and whether the blob is the file's exact bytes (`raw`) or its autocrlf-filtered form
  (`filtered`). **Two pre-launch entries (`prelaunch/compare_prefix_selfcheck.json`,
  `prelaunch/pytest_full_base_env_working_tree_summary.txt`) are `filtered`: they were committed in
  the measured commit through the repository's line-ending filter, so their listed local SHA-256
  and byte size (CRLF working copy) differ from the committed blob's; this ledger is therefore not
  54 independently verifiable raw SHA-256 hashes.** Commit `46b85ee` stored line-ending-normalized
  copies of 31 evidence files; `5f5e22a` restored their exact bytes (history kept).
- Not committed (external, identified in `artifact_sha256.txt`): `episode_outcomes.jsonl`
  (≈ 30 MB), the four checkpoints, plots and generated scenarios.

## Validity facts

- **Completion:** launcher `termination_reason = exited`, child exit code `0`, walltime 2139 s
  (cap 4 h); 100 / 100 productive updates, 400 optimizer steps; no `CRASH` / `Traceback` in the
  console; checkpoints saved at 24 / 49 / 74 / 99.
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
  residual 3.6e-7, `total_loss` vs real backward gradient relative residual 0.0 (independently
  recomputed by the review from the committed scalar stream). The five vector files recompute
  the recorded norms, pressures and first-order predictions to ≤ 1.4e-16 relative — **CC-reported;
  the review could not read the NPZ contents.** Iteration 74's batch had no MILD row, so its vector
  file has no `contrast_grad` (reason recorded).

## Same-seed prefix (instrumentation consistency, not replication) — UNRESOLVED

- **Declared first divergence:** `input_side` — the pre-update evaluation world with seed
  `2000448`: all three of its members differ by one tick (and `time_norm`) at wake 2 — CLEAN
  (attempt 42) 1804 vs 1805, also 4085 vs 4084 at wake 4; MILD (43) 1806 vs 1805; SEVERE (44)
  1805 vs 1806; 12 / 60 pre-update episodes differ; the
  pre-update round's evaluation record is identical. Under the executed rule (D1) the run
  continued.
- **Training stream (descriptive).** All seeds and cells of the 800 training episodes match.
  Iteration 0 matches after normalization: 8 / 8 episodes and 17 / 17 credit rows; its train
  record differs in five epoch-mean fields (`policy_loss`, `total_loss`, `mean_ratio`,
  `approx_kl`, `grad_norm`; largest |Δ| 6.0e-8). **The first detected training output difference
  occurs at iteration 1, seed `3000008`, wake 0, tick 686, `source_scores[0][0]`
  (`0.05741109326481819` vs `0.05741110071539879`, magnitude `7.450580596923828e-9`), while the
  compared recorded input summaries match.** That episode differs in 45 output leaves across all
  three of its wakes (maximum |Δ| over all numeric leaves 5.96e-8) with no compared input field
  differing. **The time and cause of parameter divergence are not established by this
  comparison:** the outcome records carry selected actor-input summaries, not a serialized
  observation (no complete task-feature matrix, agent-feature matrix or edge tensor), so equal
  summaries do not prove equal full inputs, and a float32 output difference does not by itself
  distinguish different parameters from different forward numerical execution. Checkpoints show
  later parameter differences (maximum 5.1e-6 at 24, 1.9e-5 at 49, 2.7e-3 at 74, 0.045 at 99),
  not their first occurrence.
- **Unit of the per-iteration counts:** `training_stream_divergence.json` classifies each
  differing EPISODE at its first differing wake, and its magnitude column is
  `max_first_field_abs_diff` — the largest first-differing-field difference in the iteration, not
  a maximum over all outputs, and not evidence that every wake differs. From iteration 1 every
  training episode differs; `max_first_field_abs_diff` is ≤ 1.5e-8 at iterations 1–2, ~2e-6 by 12
  and ~0.09 by 97; the first input-side training episode is at iteration 13.
- **Inspectable source (F2):** `prefix_source_extract/` holds, from both pinned outcome files,
  the three seed-2000448 pre-update members, the eight iteration-0 training episodes and the
  seed-3000008 iteration-1 episode, plus both iteration-0 train records and the 17 + 17 credit
  rows, with the compared field list, the normalization and every differing leaf's exact values.
  Every difference that normalization removes in these records is a UUID identity or a wall-clock
  timing field. The extract confirms the revised statements above and revealed no further
  difference.
- **Engineering probes (synthetic; they do not resolve the discrepancy):** under nlp_env on
  realistic graph sizes, the base updater (compiled from `bcb1746`), the OFF and the ON updater are
  bit-identical in one process and across two processes; the update's float result does not change
  with heap-allocation shifts but does change with the intra-op thread count. Neither run recorded
  its effective thread configuration. No cause is claimed, including instrumentation.

## Primary diagnostic (every update; declared windows)

`C_B` = mean P(ABORT | SEVERE immediate-FD rows) − mean P(ABORT | MILD rows) of the batch;
pressure = −dot(h, g) of a raw loss-gradient component (first-order change of `C_B` per unit raw
descent step); prediction = dot(h, Δθ) of the actual Adam step. 83 / 100 updates hold both
severities (undefined: 1, 7, 11, 13, 26, 32, 40, 55, 61, 74, 77, 81, 83, 84, 90, 93, 94). Each
update's `C_B` is measured on a DIFFERENT training batch, so the rows are not one fixed-policy
trajectory.

| Window | defined | median full ΔC_B | mean \|ΔC_B\| | ΔC_B + / − | epoch-0 FD pressure + | FD+ & non-FD− | FD+ → total− | median cos(Δθ, h) | raw-total vs Adam-prediction sign agree | prediction vs actual sign agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–24 | 21 | `+3.8e-4` | `1.6e-3` | 11 / 10 | 13 | 3 | 1 | `+0.010` | 52 / 84 | 79 / 83 |
| 25–49 | 22 | `−2.5e-5` | `4.3e-4` | 10 / 11 (1 negligible) | 11 | 2 | 0 | `−0.018` | 56 / 88 | 84 / 84 |
| 50–74 | 22 | `−1.8e-4` | `6.5e-4` | 9 / 13 | 11 | 9 | 6 | `−0.020` | 62 / 88 | 86 / 88 |
| 75–99 | 18 | `+1.8e-3` | `9.7e-3` | 9 / 9 | 14 | 1 | 0 | `+0.058` | 50 / 72 | 52 / 72 |
| all | 83 | `−4.6e-5` | `2.8e-3` | 39 / 43 (1) | 49 | 15 | 7 | `+0.008` | 220 / 332 | 301 / 327 |

Supplementary **review-derived** arithmetic (`extracted/review_derived_summaries.json`; not
predeclared endpoints; negligible = |x| < 1e-6):

| Window | median \|cos(Δθ, h)\| | q90 \|cos(Δθ, h)\| | median \|Σ epoch residuals\| per update | median Σ \|epoch residual\| per update | median \|epoch residual\| | median \|epoch actual ΔC_B\| | mean \|full ΔC_B\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–24 | `0.0327` | `0.0768` | `1.2874e-4` | `1.287e-4` | `1.91e-5` | `2.38e-4` | `0.001557` |
| 25–49 | `0.1419` | `0.2321` | `7.0048e-7` | `9.93e-7` | `2.47e-7` | `7.50e-5` | `0.000434` |
| 50–74 | `0.1070` | `0.2410` | `1.9536e-6` | `1.98e-6` | `4.66e-7` | `1.16e-4` | `0.000652` |
| 75–99 | `0.1355` | `0.3604` | `1.4742e-2` | `2.375e-2` | `2.74e-3` | `3.34e-3` | `0.009669` |
| all | `0.0810280` | `0.2445881` | `8.47e-6` | `8.47e-6` | `2.14e-6` | `1.59e-4` | `0.002779` |

Epoch-0 FD versus non-FD pressure signs (83 defined updates): FD positive / non-FD positive 34;
FD positive / non-FD negative 15; FD negative / non-FD positive 21; FD negative / non-FD negative
12; FD negligible / non-FD negative 1. Among the 82 non-negligible pairs, 36 have opposing signs
and 46 the same sign.

Further facts (`primary_diagnostic.json`, `per_epoch_summary.json`):

- The batch contrast stays small: `C_B` before an update has median `+1.6e-3` (q10 `−0.024`,
  q90 `+0.018`); mean P(ABORT) on SEVERE and MILD FD rows is `0.188` / `0.188`.
- Epoch-0 FD pressure: median `+2.2e-3`, mean `−1.3e-3`; SEVERE rows positive 48 / 83, MILD
  rows 42 / 83; positive in 14 / 18 updates of 75–99 (SEVERE rows 17 / 18).
- Non-FD pressure is positive toward the chosen contrast in 55 / 83 updates. The 15 / 83 figure
  is specifically non-FD opposition to a POSITIVE FD pressure; opposing signs of either direction
  occur in 36 / 82 non-negligible pairs.
- The entropy component is two orders of magnitude smaller (mean |pressure| `3.4e-4` vs FD
  `4.5e-2`) and flips the sign of the total pressure in 1 / 83 updates.
- Alignment of the actual Adam displacement with the contrast gradient is small in magnitude:
  median |cos(Δθ, h)| `0.081`, q90 `0.245` (signed median `+0.008` mixes both directions); its
  signed cosine with the raw descent direction −g_total has median `+0.13`. 205 / 332 defined
  epochs are gradient-norm clipped; PPO ratio clipping binds on FD rows in < 1 % of rows.
- Linear approximation: per update, the median |Σ epoch linearization residuals| is ≤ 2e-6 in
  25–74 and `1.3e-4` in 0–24; in 75–99 it is `1.47e-2` against a median |epoch actual ΔC_B| of
  `3.3e-3` — a large linearization residual whose mechanism these data do not identify.

## Secondary behaviour (five matched development rounds)

| Updates | Macro SEVERE − MILD P(ABORT) (new / original) | Directional switches (new / original) | Reverse | mean P(ABORT) MILD / SEVERE (new) |
|---:|---:|---:|---:|---:|
| 0 | `+2.62e-4` / `+2.62e-4` (identical round) | 3 / 3 | 0 / 0 | 0.5000 / 0.5003 |
| 25 | `+3.05e-6` / `+3.04e-6` | 0 / 0 | 0 / 0 | 0.1725 / 0.1725 |
| 50 | `−6.73e-7` / `−6.91e-7` | 0 / 0 | 0 / 0 | 0.1372 / 0.1372 |
| 75 | `+1.54e-4` / `+1.10e-4` | 0 / 0 | 0 / 0 | 0.0831 / 0.0833 |
| 100 (final) | `+6.92e-3` / `+5.04e-7` | 0 / 0 | 0 / 0 | 0.1484 / 0.1553 |

Every round: 20 / 20 groups, 10 / 10 base cells; evaluation reward mean equal to the original's
in every round. The post-update rounds have no directional (or reverse) action switch; the final
measured probability contrast is `+0.00691564`. The pre-update round had 3 directional switches,
and a nonzero probability separation is distinct from a change in selected actions. The
final-round difference from the original reflects the drifting trajectories; no effect of the
observer is claimed.

## Reading (development only; descriptive)

- Local FD pressure toward the SEVERE-minus-MILD contrast is inconsistent in sign across many
  batches (positive in 49 / 83) and only consistently positive in 75–99.
- Actual updates do not consistently increase the batch contrast (39 increases, 43 decreases,
  1 negligible); the actual displacement's alignment with the contrast gradient is small in
  magnitude (median |cos| 0.081) in this parameterization.
- Opposing raw non-FD projection is not universal (non-FD positive in 55 / 83; opposing signs in
  36 / 82 non-negligible pairs). This does not rule out whole-batch effects through Adam's moments
  and coordinate scaling.
- The late steps (75–99) carry a large linearization residual.
- The fixed-world evaluations show no post-update directional switches.

These data do not isolate credit assignment, architecture, optimizer or observation sufficiency as
a cause, and do not prove a defective optimizer or locate the causal bottleneck. This run is
`actor_only`: no learned critic supplies its advantages.

Non-claims: no causal attribution; negative advantages and cross-episode ABORT-vs-PLAN means are
not action values; `C_B` is a training-batch contrast on unmatched batches, not the matched-world
endpoint; no trajectory reproduction of the original is claimed; one seed, development worlds
only; no confirmatory evidence exists.
