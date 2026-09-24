# GENERALIZED-V2 actor mission-fuel-slack development run R1 — evidence package

> **Status: EXECUTED / UNREVIEWED.** Development evidence for GPT review. No verdict is assigned
> here; passing checks are not approval. Development profile only; the confirmatory profile was
> not selected, inspected or executed.

| Item | Value |
|---|---|
| Run | `graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` |
| Measured code SHA | `3bc944119da08af8e25268c9ee83fc63a8d1e533` — clean detached worktree `C:\gms1src`; `match_aou` imported from it; the imported BLADE engine is Git-blob-identical to the measured tree (`prelaunch/preflight_blade_blob_identity.json`) |
| Base / branch / PR | `3d05a6ca76a0cbaf5c182d35da8677e7c06f6268` / `task/actor-mission-fuel-slack-dev-r1` / draft PR #79 |
| Intervention | one actor input, `mission_fuel_slack_norm` on the ego agent row (actor observation `actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1`); agent width 1 → 2; nothing else |
| Training | `actor_only`, `generalized_v2`, `p1_milp_v1`, `semantic_k_plus_2_logmeanexp_v1`, base seed 3000000, 375 updates × 8 successful episodes, max 12 attempts / update, no early stopping |
| Benchmark | manifest `ef17a68a…46ea8` (file SHA-256 `dd72afc9…a103`), development profile, 20 groups × CLEAN/MILD/SEVERE, held out from `[3000000, 3004500)` |
| Comparator | semantic actor-only R1 `graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37` at `d4e9f3721e6d151c00be3fe93c3d149df9d31965` (reviewed, measurements §10), read from its archive `C:\gra\runs\development\v2_semantic_actor_only_r1_seed3000000_d4e9f37`, **not rerun** |
| Execution | LOCAL Windows, `nlp_env`; launcher `scripts/launch_with_watchdog.py` (24 h cap); started 2026-09-24 16:53:00 +03:00, ended 18:49:54, walltime 7015 s, exit code 0, termination `exited` |

## Layout

| Path | Content |
|---|---|
| `authorized_plan.json` | the exact bounded plan, committed before launch |
| `prelaunch/` | preflight (manifest identity, held-outness, comparator, source identity), BLADE blob identity, legacy-builder equivalence on the measured tree, test / self-test logs |
| `engineering_checks/legacy_equivalence_check.py` | PO5 check against the base-SHA builder loaded from Git |
| `scripts/` | launcher, pre-launch verification, deterministic extractor, plot |
| `run_artifacts/` | byte-identical copies of `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl` and the launcher record / exit code / start time |
| `artifact_sha256.txt`, `source_manifest.json` | SHA-256 and size of every source, in or out of Git (episode outcomes, credit stream, final checkpoint, console log, manifest, archive index, comparator files) |
| `review_precheck.json` | provenance, plan-vs-invocation, config diff vs comparator, completion, accounting, failures, console classification — **not a verdict** |
| `extracted/primary_endpoint.json` | final-round endpoint with denominators, the trainer's own final-round selection, comparator final round, exploratory maxima |
| `extracted/behaviour_summary.json`, `comparator_behaviour.json` | all 16 rounds: per-cell, per-group, MILD / SEVERE P(ABORT), selected-ABORT counts, switches; cross-checked against each run's trainer `v2_behaviour` |
| `extracted/trajectory_comparison.{json,md,png}` | the readable baseline-versus-new trajectory |
| `extracted/eval_immediate_fd_wakes.jsonl` | all 640 evaluation immediate-FD wakes, with the actor input and audit summary |
| `extracted/eval_post_fd_boundary_wakes.jsonl` | all evaluation post-FD-boundary wakes (separate population) |
| `extracted/eval_wake_feature_audits.jsonl` | every evaluation wake's full pre-action mission-slack audit (all wake kinds) |
| `extracted/train_immediate_fd_feature_rows.jsonl` | every training immediate-FD wake of the certified ego (feature, selection, audit summary) |
| `extracted/feature_summary.json` | feature distributions by phase × wake kind, non-finite counts, severity breakdown (reporting only), and the audit re-check result |
| `extracted/fd_selected_ego_credit_rows.jsonl`, `credit_summary.json` | FD-selected-ego immediate-FD credit rows joined to their wake feature; coverage; 25-update training windows |
| `extracted/secondary_outcomes.json` | reward, utility, deaths, RTB by round and member cell, both runs |
| `extracted/training_wake_counts_by_iteration.json` | training wake counts by kind per iteration |
| `extracted/final_round_fd_source_scores.jsonl`, `final_round_fd_source_score_ranges.json` | (review fix F3) the 40 final-round immediate-FD wakes of each run with unrounded per-node scores and legality, and their exact ranges; produced by `scripts/extract_final_round_source_scores.py` from the hash-pinned original outcomes |

Large originals stay outside Git, unchanged, identified in `artifact_sha256.txt`.

## Reproducing (no training, evaluation or replay)

```bash
python research_evidence/generalized_v2/actor_mission_slack_dev_r1/scripts/extract_evidence.py --out <dir> --copy-run-artifacts --verify-against research_evidence/generalized_v2/actor_mission_slack_dev_r1/artifact_sha256.txt
```

The F3 extract is regenerated with
`python research_evidence/generalized_v2/actor_mission_slack_dev_r1/scripts/extract_final_round_source_scores.py --out research_evidence/generalized_v2/actor_mission_slack_dev_r1`
(it verifies both original outcomes against `artifact_sha256.txt` first).

For the main extractor, copy `authorized_plan.json` and `prelaunch/` into `<dir>` first. Every file under `extracted/`,
`run_artifacts/` and `review_precheck.json` then reproduces byte-identically (verified); the two
hash ledgers differ only by the absolute paths of those two inputs. Standard library only;
`plot_trajectory.py` needs matplotlib.

## Validity facts (pre-check, not a verdict)

- Completion: 375 / 375 updates, final checkpoint `ckpt_iter0374.pt`, exit code 0, 7015 s.
- Training accounting: 3000 successful of 3008 attempted; **8 failed and replaced**, all
  `FuelDamageError` `no_fd_eligible_ego` at `train` / `setup` — **the same eight seeds as the
  comparator**; `accounting_reconciled = true`. The console's 16 `Traceback` lines are exactly
  these eight chained failure prints; no `CRASH`, no integrity abort.
- Evaluation: 16 rounds, 960 / 960 episodes successful; final round `post_update`, 375 updates,
  round 15; 10 / 10 base cells defined; 20 / 20 groups complete and metric-eligible.
- Configuration: `cli_defaults`; the invocation matches the plan; `train_config` differs from the
  comparator only in `output_dir` (and the later-added `actor_gradient_diagnostics = false` key).
- Schemas: episode outcome v5, wake diagnostics v3, credit v1; one actor observation id throughout.
- Feature audit: the executing task's extractor re-checked all 12 750 recorded wakes (value =
  float32 of the audit; legs, fuel convention, slack formula, exclusions, levels and
  nearest-neighbour order) with 0 failures and 0 non-finite values — a preserved report over the
  external originals. The GPT review independently rechecked the 3 506 COMMITTED evaluation-wake
  audits (a different scope) without discrepancy.

## Results (descriptive; development only)

**Primary endpoint (final round, 375 updates):** ten-cell macro `P(ABORT|SEVERE) − P(ABORT|MILD)`
= **−7.45e−10** over 10 / 10 cells and 20 / 20 groups; **0 / 20 directional, 0 / 20 reverse
switches**; selected ABORT 0 / 20 MILD and 0 / 20 SEVERE. Comparator final: +0.000888, 0 / 20.

**Trajectory:** the new run never shows severity-conditioned selection after initialization —
round 0 has 3 / 20 directional switches at near-uniform probabilities (macro +0.00026), and in
every post-update round the absolute macro is at most about 1.10e−4 (maximum 0.000110 at update
75) with 0 switches. The comparator's transient (updates 75–150,
macro up to +0.650, up to 20 / 20 switches) **did not occur** in this run. Table:
`extracted/trajectory_comparison.md`.

**Observed immediate-FD outputs (not a test of feature dependence).** The input separated the
severities in sign on every recorded immediate-FD wake of the certified ego: post-update evaluation
300 / 300 MILD positive and 300 / 300 SEVERE negative (15 rounds re-measuring the same 20 frozen
worlds — 20 distinct values per severity, not 300 independent worlds), pre-update 20 / 20 each,
and — labelled as TRAINING, a stochastic actor on a sampled population — 774 / 774 MILD positive
and 752 / 752 SEVERE negative. The measured immediate-FD outputs nevertheless showed essentially
no severity-conditioned separation: from update 100 the within-round P(ABORT) range across the 40
recorded immediate-FD wakes was below 1e−6 (at most 8.2e−7; exactly 0 at updates 250 and 275).
The comparator's range was 0.0025 at its pre-update round and at least 9.2e−3 in every post-update
round. This is an observed-population result for one training seed: it is **not** proof that the
policy has zero dependence on the feature, and it is not a causal explanation.

**Final-round per-node source scores (inspectable, review fix F3).**
`extracted/final_round_fd_source_scores.jsonl` preserves the 40 final-round immediate-FD wake
records of each run with unrounded per-node scores and legality, and
`extracted/final_round_fd_source_score_ranges.json` gives exact `max − min` ranges (no tolerance;
selection rule and pinned source hashes inside). Node classes come from legality only:
**ego-assigned** (ABORT cell legal) versus **not ego-assigned** (peer-assigned or unassigned — the
record does not distinguish them). New run: every column's range over all 200 nodes of the 40 wakes
is ≤ 9.6e−7 (ego-assigned and not-ego-assigned alike), and P(ABORT) spans 7.5e−9. Comparator: ranges
of 1.0e−3 to 4.6e−3 over its 62 ego-assigned nodes and 0.11 to 0.39 over its 138 not-ego-assigned
nodes, P(ABORT) spanning 0.020. This describes recorded outputs only; no cause is established.

**Training:** immediate-FD ABORT under the stochastic actor was 93 / 774 MILD and 105 / 752 SEVERE
(comparator per measurements §10.6: 64 / 774 and 202 / 752). Credit remains episode / chain-level
(`gamma = 1`, terminal reward); severity differences in advantage are associations, not action
values.

**Secondary (final round):** reward, utility and deaths are identical to the comparator's final
round (both select PLAN at every FD wake): SEVERE episodes average reward −0.650 with an airframe
lost in every one.

## Review status and corrections

GPT exact-head review of `ea60ce38e9f1b91fb3cb736bb4dba01f88f92523` (2026-09-24):
**CHANGES_REQUESTED** — narrow evidence / documentation corrections; no blocking implementation
defect reported, and the review independently reproduced the 16 endpoint values and the committed
audits. Corrections are ADDITIVE commits on the same PR; the original evidence commit
`50c13ee9839a822b12e42fbfed76c8ed55e1f901` is unchanged and is not re-described as containing
them:

- **F1** — the four `prelaunch/nlp_*.log` files that `source_manifest.json` declared committed had
  been excluded by the repository's `*.log` ignore rule; they are now committed byte-for-byte (the
  pre-launch files; nothing rerun), with blob SHA-256 equal to the manifest's.
- **F2** — the scientific wording above was narrowed (observed-population statements, temporal
  qualifiers, repeated-measures counts, the ≈ 1.10e−4 bound).
- **F3** — the per-node score remark is now backed by the committed deterministic extract
  `scripts/extract_final_round_source_scores.py` → `extracted/final_round_fd_source_scores.jsonl`
  and `extracted/final_round_fd_source_score_ranges.json`.

## Limits and non-claims

One run, one training seed, 20 frozen development worlds re-measured every round (repeated
measures). Cross-version comparison: a matched seed number does not give identical initial
weights or RNG trajectories once the input dimension changed. No claim that the feature caused the
near-constant outputs, is useless, or harmed learning in general; no robustness claim; no causal
attribution of the absent transient or of the near-constant outputs; no confirmatory evidence.
