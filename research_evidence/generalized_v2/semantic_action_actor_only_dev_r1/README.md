# GENERALIZED-V2 semantic-action actor-only development run R1 — evidence package

> **Status: DEVELOPMENT evidence, preserved for independent GPT review. No scientific verdict has
> been assigned.** Nothing here updates the project's conclusions, contracts or handoff.

| Item | Value |
|---|---|
| Run | `graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37` |
| Measured code SHA | `d4e9f3721e6d151c00be3fe93c3d149df9d31965` (clean tree, branch `main`, recorded by the run) |
| Intervention measured | action representation `semantic_k_plus_2_logmeanexp_v1` (replaces the historical node-indexed joint `k x 3`) |
| Credit instrumentation | `train_credit_diagnostics.jsonl` — **observational**: it copies the credit each update already computed and feeds nothing back |
| Training | `actor_only`, `generalized_v2`, `p1_milp_v1`, base seed 3000000, 375 updates x 8 successful episodes |
| Benchmark | frozen manifest `ef17a68a…46ea8` (file SHA-256 `dd72afc9…a103`), **development** profile only |
| Confirmatory profile | **untouched** — not selected, inspected or executed |
| Comparator | historical actor-only R1 at `ae42cb01677f94868b2873008d87be677e31f0c8` — the comparison is **cross-version** (different code SHA and action representation; same frozen development population and training seed stream) |

**No new execution occurred during evidence preservation.** The extractor is standard-library
Python only: it imports nothing from `match_aou`, starts no subprocess, and runs no training,
evaluation, replay, checkpoint inference or benchmark preflight. It reads the completed run and
the archived comparator, and fails if any source byte changes during extraction.

## Layout

| Path | Content |
|---|---|
| `artifact_sha256.txt` | SHA-256 and size of **every** source artifact — in Git or external (run, comparator, manifest, archive index) |
| `source_manifest.json` | the same artifacts with absolute path, bytes, SHA-256, review role and disposition (`copied` / `external_only`) |
| `review_precheck.json` | validity pre-check: provenance, plan vs resolved config, manifest, completion, accounting, failures, final-round identity, endpoint coverage, schema versions, anomalies. **Not a verdict.** |
| `run_artifacts/` | byte-identical copies of `authorized_plan.json`, `preflight.json`, `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl` |
| `extracted/eval_immediate_fd_wakes.jsonl` | all 640 evaluation immediate-FD wakes (16 rounds x 20 groups x MILD/SEVERE) with round, group, base cell, member, episode identity, semantic P(ABORT), selected action/node/leaf and the actor-input diagnostics of the source record |
| `extracted/behaviour_summary.json` | per-round MILD / SEVERE P(ABORT), SEVERE − MILD pooled and ten-cell macro, switch counts with denominators, per-cell and per-group values, final round |
| `extracted/comparator_r1_behaviour.json` | the same per-round quantities **recomputed from the archived comparator artifacts** |
| `extracted/collapse_timeline.json` | every evaluation round (no selection) beside the comparator, plus 25-iteration training FD-wake windows |
| `extracted/fd_selected_ego_credit_rows.jsonl` | **all** 1526 training credit rows with `wake_kind = immediate_fuel_damage` and `is_fd_selected_ego = true`, verbatim |
| `extracted/credit_summary.json` | counts, descriptive association, matched-within-batch comparison, non-counterfactual ABORT-vs-not comparison, 25-iteration windows, structural credit verification |
| `scripts/extract_evidence.py` | the deterministic extractor that produced everything above |

Large artifacts stay outside Git and are identified by hash only: `episode_outcomes.jsonl`
(86 MB), `train_credit_diagnostics.jsonl` (11.9 MB), `checkpoints/ckpt_iter0374.pt`,
`training_console.log` (`*.log` is git-ignored), `launch.cmd`, `invocation_start_local.txt`
and `native_exit_code.txt`.

## Reproducing

```bash
python research_evidence/generalized_v2/semantic_action_actor_only_dev_r1/scripts/extract_evidence.py --out <dir> --copy-run-artifacts --verify-against research_evidence/generalized_v2/semantic_action_actor_only_dev_r1/artifact_sha256.txt
```

The output is byte-identical to this package when the sources match `artifact_sha256.txt`. Any
hash, schema, provenance, accounting or arithmetic mismatch exits with code 2 and
`EVIDENCE CHECK FAILED`. The extractor cross-checks every reproduced evaluation quantity — the
per-round macro, pooled mean, per-cell means, per-group P(ABORT) and selected actions, and the
switch counts — against the trainer's own per-round `v2_behaviour` in `eval_records.jsonl` and
against `run_summary.json`, for both this run and the comparator. It also checks the comparator's
key hashes against the archive index `C:\gra\metadata\ARTIFACT_INDEX.jsonl`.

## Reproduced quantities (descriptive; see the JSON files for full denominators)

- **Final round** (`post_update`, 375 updates, round 15): ten-cell macro SEVERE − MILD P(ABORT)
  **+0.000888** over **10/10** base cells, 20/20 groups metric-eligible, 0 directional and 0
  reverse switches. Comparator final macro: +0.000000209.
- **Trajectory over all 16 rounds.** Rounds 3–6 (updates 75–150) show macro +0.296, +0.611,
  +0.650, +0.576 with 4, 19, 20, 19 directional switches out of 20. Rounds 7–15 lie between +0.0004
  and +0.0061 with 0 switches. The comparator's largest per-round value is +0.00688 (round 7,
  updates 175), with 0 switches in every round, derived from its archived artifacts.
- **Credit rows** (FD-selected ego, immediate-FD wake): 1526 = 774 MILD + 752 SEVERE. Every
  update's row count equals `train_records.n_transitions` (375/375 updates, 9166 rows).

**Repeated measures.** Every evaluation round re-measures the same 20 frozen development worlds.
Rounds are not independent samples.

## Structural credit fact and interpretation limitation

Actor-only credit in the measured code (`src/match_aou/rl/training/graph_ppo.py`):

- `_chain_returns` — with `gamma == 1.0` every transition of an ego chain receives the episode's
  terminal reward `R`;
- `compute_returns_and_advantages` — baseline = mean `episode_reward` over the batch's
  **episodes** (`baseline = float(np.mean([rec.episode_reward for rec in records]))`);
  `raw_adv = returns - baseline`; normalization
  `advantages = (raw_adv - adv_mean) / (adv_std + cfg.adv_norm_eps)` over the batch's
  **transitions**.

**Verified against this run's persisted credit rows**, not only from code:

- `gamma = 1.0` in `run_config.json` and on every row;
- per row: `return == episode_reward`, `raw_advantage == return − baseline`, and the normalized
  advantage recomputes from the batch moments. The batch moments recompute from the batch's rows,
  and the baseline equals `train_records.baseline`;
- **6689 ego chains checked, 0 with varying raw advantage** (1222 have more than one transition);
- **2999 episodes checked, 0 with varying raw advantage** (2415 have more than one transition);
- the reward is terminal-only. In all 2999 episodes the stored transition rewards sum to `R`. In
  the 1114 episodes with `R ≠ 0`, all of `R` sits on exactly one transition, at the episode's
  latest tick and last in its ego chain.

**Limitation.** Actor-only with `gamma = 1` and a terminal-only reward gives **episode- and
chain-level credit, not local causal credit for the immediate-FD action**. The immediate-FD
transition's advantage is its episode's outcome relative to the batch. MILD versus SEVERE credit
differences therefore partly restate that SEVERE episodes score lower. **ABORT-vs-not advantage
gaps are observational associations between different episodes and worlds, not action-value
estimates.** `credit_summary.json` labels each comparison accordingly:

- `descriptive_association`;
- `matched_within_batch` — same update batch, still different episodes;
- `abort_vs_not_non_counterfactual`.

## Known anomalies (recorded, not repaired)

- **`native_exit_code.txt` is empty (0 bytes).** The launcher line `echo %RC%> file` is parsed by
  cmd as a handle redirect for a one-digit code. Completion is established from
  `run_summary.json`, `train_records.jsonl` and the final checkpoint.
- `v2_behaviour.metric` still reads `severe_minus_mild_aggregate_abort_mass`. The same block
  defines P(ABORT) as the semantic ABORT leaf and sets
  `aggregate_mass_is_not_selected_action_probability: false`. Only the label is legacy.
- The manifest was consumed at its archived path `C:\gra\benchmarks\…`. The comparator recorded
  the pre-archival path; identity is verified by SHA-256 and `manifest_id`.
- There are 8 expected training setup failures (`FuelDamageError: no_fd_eligible_ego`), with
  seeds identical to the comparator's.
