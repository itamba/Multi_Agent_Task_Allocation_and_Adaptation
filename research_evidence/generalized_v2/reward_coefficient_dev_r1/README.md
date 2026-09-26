# REWARD-01 aircraft-loss coefficient development comparison R1 — evidence package

> **Status: EXECUTED / UNREVIEWED.** Development evidence for GPT exact-candidate review. No
> verdict is assigned here; passing checks are not approval. Development profile only; the
> confirmatory profile was not selected, inspected or executed. One training seed pair.

| Item | Value |
|---|---|
| Question | does doubling `aircraft_penalty_coeff` 2.25 → 4.5, everything else fixed, improve acquisition and retention of conditional MILD / SEVERE behaviour and the observed mission / survival trade-off? |
| Measured code SHA | `2b570194dea3612f3796999fc589b73d7082ae31` for **both** arms — clean detached worktree `C:\grc1src`; `match_aou` imported from it; imported BLADE engine Git-blob-identical to the measured tree (`prelaunch/preflight_blade_blob_identity.json`) |
| Base / branch / PR | `79afdd4364d6f4bab777ce52f311c314da1f23be` / `task/reward-coefficient-dev-r1` / draft PR #82 |
| Arm A (control) | `aircraft_penalty_coeff = 2.25`, preset `configs/arm_a_c2p25.json`, run `C:\gruns\reward_c225_r1_s3000000_2b57019` |
| Arm B (intervention) | `aircraft_penalty_coeff = 4.5`, preset `configs/arm_b_c4p5.json`, run `C:\gruns\reward_c450_r1_l2_s3000000_2b57019` (second launch, amendment 1) |
| Common | `actor_only`, `generalized_v2`, `p1_milp_v1`, actor observation `actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1`, `semantic_k_plus_2_logmeanexp_v1`, base seed 3000000, 375 × 8 successful episodes, ≤ 12 attempts / update, FD `seeded_variable` p 0.5 / mild 0.5 / leg 0.3 / margin 1.1, PPO lr 3e-4 / clip 0.2 / entropy 0.01 / 4 epochs / γ 1 / grad-norm 0.5, no early stopping, eval every 25 → 16 rounds; every other field from the §17 `train_config` |
| Benchmark | manifest `ef17a68a…46ea8` (file SHA-256 `dd72afc9…a103`), development profile, 20 triads × CLEAN / MILD / SEVERE = 60 members per round; all 120 manifest seeds outside `[3000000, 3004500)` |
| Execution | LOCAL Windows, `nlp_env`, CPU (torch 2.7.1+cpu, 4 intra-op / 4 inter-op threads, no thread variable set; identical env probes for both arms) |

## Configuration: the single factor

`prelaunch/preflight.json` resolves both presets through the trainer's own `--config` path:
`config_source.resolved_from = config_file`, no CLI override, and the two resolved
`train_config`s differ **only** in `aircraft_penalty_coeff` (2.25 / 4.5) and `output_dir`. Each
differs from the §17 `train_config` only in `output_dir` and the two actor-step diagnostic keys
added after §17 (both off). The run-recorded `run_config.json` of each arm equals that resolution
(arm B's with its amended `--out`). The reward formula, reference and defaults are unchanged; no
production code was touched.

Pre-launch proof obligations (all PASS, `prelaunch/`): each preset's `reward_config()` carries its
coefficient; `compute_episode_reward` has one call site, in `_run_one_episode`, with
`cfg.reward_config()`; on the REAL `compute_episode_reward` with synthetic fixed event-conditioned
references, doubling `c` doubles the penalty exactly, leaves `q` unchanged and leaves zero-loss
outcomes unchanged, and `R(c) = q − c·p`; manifest identity and held-outness; source identity;
32 focused existing tests pass.

## Execution record and the deviation

| | Arm A | Arm B launch 1 | Arm B launch 2 |
|---|---|---|---|
| Start → end (+03:00) | 18:43:37 → 21:04:08 | 21:04:35 → 21:06:35 | 21:41:58 → 23:10:12 |
| Walltime / exit | 8431 s / 0 `exited` | 120 s / 1 **`pre_update_mismatch_stop`** | 5294 s / 0 `exited` |
| Updates | 375 / 375 | 3 (killed) | 375 / 375 |

Combined scientific walltime 13 845 s (3.85 h), within the 8-hour cap.

**Deviation — arm B was launched twice (user-authorized).** The first launch was stopped by the
pre-update identity rule this task wrote (`authorized_plan.json:/pre_update_mismatch_rule`),
which also stopped on wake `ego_id` and `tick`. The read-only audit (`stop_record/`) found **no**
world, selected-action or probability difference (max 1.79e−7); the mismatches were per-episode
generated ego ids in the A5 / A6 worlds (they differ even between arm A's own rounds of one
world) and 1–2-tick wake timing. The user then authorized one new arm-B launch at the same
measured SHA and configuration with a corrected rule (`authorized_plan_amendment_1.json`,
`scripts/compare_pre_update.py` v2, `prelaunch/amendment1_check.json` 12 / 12 PASS), committed
before the launch. Launch 1 is preserved and hash-referenced and is not used below.

Launch 2's live pre-update check **passed**: 60 / 60 members, 0 frozen-world mismatches, identical
wake kind / selected action / leaf at all 161 wakes, max probability difference 1.19e−7; record
only: 24 ego-id and 11 tick differences, 6 members differing only in `ticks`, penalty exactly
2 × arm A's in all 60 (`review_precheck.json:/pre_update_identity`).

## Validity facts (the extractor's pre-check; not a verdict)

- Both arms: measured SHA, clean, `match_aou` from `C:\grc1src`; argv and `train_config` equal the
  pre-launch resolution; `accounting_reconciled = true`; 3000 successful of 3008 training
  attempts; **8 failures, all `FuelDamageError` `no_fd_eligible_ego` at `train` / `setup`, the
  same 8 seeds in both arms** (3001255, 3001265, 3001741, 3001807, 3001879, 3002177, 3002296,
  3002868); 960 / 960 evaluation episodes; 16 rounds, the final one semantically
  `post_update` / 375; 10 / 10 base cells and 20 / 20 groups metric-eligible in every round;
  no non-finite optimisation value; console: 0 `CRASH`, 16 `Traceback` = exactly two per
  accounted failure; exit 0 under the 4-hour cap.
- Every reproduced evaluation quantity matches each trainer's per-round `v2_behaviour`; every
  evaluation reward is reproduced from its own recorded terms (`q`, `p`, `c`) before rescoring.
- The two arms' run-config blocks other than the coefficient, and their environment probes, are
  identical.

## Results (descriptive; development only; repeated measures of 20 frozen worlds)

**Primary endpoint (final round, 375 updates, 10 / 10 cells, 20 / 20 groups in both arms, the same
eligible groups):**

| | macro `P(A|SEV) − P(A|MILD)` | mean `P(A|MILD)` | mean `P(A|SEV)` | dir / rev / both-ABORT / both-non-ABORT |
|---|---:|---:|---:|---|
| Arm A (c = 2.25) | **+3.24e−8** | 0.0315 | 0.0315 | 0 / 0 / 0 / 20 of 20 |
| Arm B (c = 4.5) | **+1.69e−3** | 0.2294 | 0.2311 | 0 / 0 / 0 / 20 of 20 |
| **B − A** | **+1.69e−3** | | | |

Neither arm ends with any severity-conditioned selected action; both select PLAN at every final
immediate-FD wake, and their final-round outcomes are **identical in all 60 members** (utility,
deaths, rewards under either coefficient).

**Trajectory (`extracted/trajectory_comparison.md`, `.png`):**

- **Arm A** moved toward PLAN: mean P(ABORT) fell from 0.50 to 0.03 (not monotonically — 0.08 at
  update 75, 0.30 at 100), with the MILD and SEVERE means differing by at most 1.12e−4 in any
  post-update round; the absolute macro never exceeded 1.12e−4 (update 75) and no post-update
  round had a directional switch.
- **Arm B raised P(ABORT) in BOTH severities together**: from 0.19 (update 25) to 0.57 (175) and
  0.67–0.69 (300–350), always with MILD ≈ SEVERE (absolute macro ≤ 6.3e−4 before the final
  round). The **both-ABORT** group count rose to 9 (150), 17 (175), 18 (250–275) and **20 / 20 at
  updates 300–350**, and then **collapsed** at 375 (P(ABORT) 0.23, both-non-ABORT 20 / 20).
  No directional switch occurred in any post-update round of either arm.
- The training stream agrees: immediate-FD ABORT of the FD-selected ego was 351 / 774 MILD and
  357 / 752 SEVERE in arm B against 97 / 774 and 108 / 752 in arm A (stochastic actor, sampled
  population; not held-out evidence).

**Mission / survival trade-off (evaluation outcomes; `extracted/outcome_summary_by_round.json`,
paired contrasts `extracted/paired_world_contrast*.json*`).** Arm A loses the certified ego in
every SEVERE member from update 25 on (20 airframes per round, all in SEVERE) with mean utility
512. At arm B's both-ABORT rounds (300–350) arm B loses **0** airframes but mean utility falls to
421: MILD 368 vs arm A's 532 per 20 members (utility lower in 19 / 20 paired worlds, no death to
save), SEVERE 336 vs 444 (20 / 20 airframes saved). RTB issuance is reported separately and is not
survival (arm A's PLAN-selecting severe members still receive the executor's later RTB in 18 / 20).

**Algebraic rescoring (same saved trajectories, each with its own recorded reference; no
replay).** At arm B's both-ABORT round 12 (300 updates), paired B − A over the 20 frozen worlds:

| member cell | Δ utility | Δ deaths | Δ R at c = 2.25 | Δ R at c = 4.5 |
|---|---:|---:|---:|---:|
| MILD | −164.0 mean (19 worse, 1 equal) | 0 | −0.379 | −0.379 |
| SEVERE | −108.0 mean (18 worse, 2 equal) | −1.0 (20 / 20) | **+0.263** (20 / 20 better) | +0.844 |
| all 60 | −90.7 | −0.333 | −0.039 | +0.155 |

On these fixed trajectories ABORT-in-SEVERE scored higher than PLAN-in-SEVERE **already at
c = 2.25**, and ABORT-in-MILD lower at either coefficient. This compares two different learned
policies on the same frozen worlds, not a within-state counterfactual; the reference terms were
identical in 60 / 60 pairs at that round (in 3 pairs over the whole run one continuation
checkpoint tick differs by one tick). Final-round rescored means are identical in both arms:
−0.2209 at c = 2.25 and −0.4146 at c = 4.5.

**Credit (training; `extracted/training_summary.json`, `fd_selected_ego_credit_rows.jsonl`).**
Transitions by wake kind: A ordinary 6232 / immediate-FD 1526 / post-FD boundary 1478; B 5934 /
1526 / 932 (fewer boundary wakes after more ABORT latches). Raw and normalized advantages are kept
separate by severity, selected action and 25-update window; they pool different episodes and
updates and are not local action values.

## Interpretation limits and non-claims

- One training seed pair, 20 frozen development worlds re-measured every round: no across-seed
  robustness estimate and no confirmatory evidence.
- The primary endpoint is effectively zero in both arms; B − A = +1.69e−3 with 0 switches is not a
  conditional response.
- The larger coefficient changed behaviour — it induced **unconditional** ABORT (both severities)
  for a long stretch — but did not produce MILD / SEVERE conditioning, and the ABORT regime was
  **not retained** at update 375. ABORT in both severities does not establish the desired
  conditional response; its survival gain in SEVERE came with an unneeded utility loss in MILD.
- Peaks and plateaus (updates 300–350) are exploratory, not endpoints.
- A null result rejects neither all values of `c` nor all reward redesigns; nothing here
  identifies a crossover coefficient or a cause.
- Same seed and source did not give bit-identical simulator trajectories (ego ids, 1–2 ticks);
  the observed identity and divergence are recorded, not investigated further.

## Layout

| Path | Content |
|---|---|
| `authorized_plan.json`, `authorized_plan_amendment_1.json` | the plan (committed before launch) and the user-authorized amendment (committed before arm B launch 2) |
| `configs/` | the two presets |
| `prelaunch/` | preflight, BLADE blob identity, focused test summary, amendment-1 check |
| `stop_record/` | arm-B launch-1 stop: audit, launcher records, hashes of all original run files at the stop |
| `scripts/` | launcher, env probe, pre-update comparator, pre-launch verifiers, stop audit, deterministic extractor, plot |
| `run_artifacts/arm_a/`, `run_artifacts/arm_b/` | byte-identical copies of `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl` and the launcher records |
| `review_precheck.json` | per-arm validity pre-check, cross-arm identity, pre-update identity — not a verdict |
| `extracted/` | primary endpoint, all rounds, acquisition / retention, per-wake and per-episode rows with rescoring terms, paired world contrasts, outcome summaries, training / credit summary, trajectory table and figure |
| `artifact_sha256.txt`, `source_manifest.json` | SHA-256 of every source, in or out of Git (episode outcomes, credit streams, final checkpoints, console logs, manifest) |

Large originals (outcome and credit streams, checkpoints, console logs) stay outside Git,
unchanged, identified by SHA-256.

## Reproducing (no training, evaluation or replay)

Copy `authorized_plan.json`, `configs/` and `prelaunch/` into `<dir>`, then

```bash
python research_evidence/generalized_v2/reward_coefficient_dev_r1/scripts/extract_evidence.py --out <dir> --copy-run-artifacts --verify-against research_evidence/generalized_v2/reward_coefficient_dev_r1/artifact_sha256.txt
```

Every file under `extracted/` (except the figure), `run_artifacts/` and `review_precheck.json`
reproduces byte-identically (verified); the hash ledgers differ only by the absolute paths of the
copied inputs. The figure is `python .../scripts/plot_comparison.py <dir>` (matplotlib).
