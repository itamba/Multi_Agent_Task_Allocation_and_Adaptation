# GENERALIZED-V2 semantic-action CTDE development run R1 — evidence package

> **Status: DEVELOPMENT evidence, preserved for independent GPT review. GPT has not yet assigned
> this run's validity or scientific verdict.** Nothing here updates the project's conclusions,
> contracts, history or handoff. **Not for merge.**

| Item | Value |
|---|---|
| Run | `graph_rl_v2_semantic_action_ctde_dev_r1_seed3000000_8056266` |
| Measured code SHA | `8056266cff89f677911462b29970346bed0a57c1` (clean tree, branch `main`, recorded by the run) |
| Scientific question | Now that the semantic `k + 2` representation lets the actor learn MILD/SEVERE separation temporarily, does the existing centralized-training critic + GAE path let that separation be **retained** through the fixed 375-update budget? |
| Primary comparator | semantic-action actor-only development R1 `graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37`, measured SHA `d4e9f3721e6d151c00be3fe93c3d149df9d31965`, evidence PR #71 @ `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`. `d4e9f37 → 8056266` changes documentation only |
| Only material training difference | `training_mode`: `actor_only` → `ctde` (centralized critic + GAE credit). CTDE frozen: `critic_lr = 0.0003`, `value_coeff = 0.5`, `gae_lambda = 0.95` |
| Action representation | `semantic_k_plus_2_logmeanexp_v1` in **both** arms |
| Frozen design | `generalized_v2`, `p1_milp_v1`, base seed 3000000, 375 updates × 8 successful episodes, ≤ 12 attempts per iteration, `seeded_variable` FD 0.5 / 0.5, eval / checkpoint every 25, early stopping off; PPO `lr 0.0003`, `clip 0.2`, `entropy 0.01`, `gamma 1.0`, `epochs 4`, `max_grad_norm 0.5`, `adv_norm_eps 1e-8` |
| Benchmark | frozen manifest `ef17a68a…46ea8` (file SHA-256 `dd72afc9…a103`), **development** profile, ordinals `[0, 1]` |
| Confirmatory profile | **untouched** — not selected, inspected or executed |
| Secondary context only | historical CTDE R1 (`ae42cb0`, node-indexed representation; evidence commit `b2bbe7a`) — **not** the primary comparator |

**No new scientific execution occurred during preservation.** The extractor is standard-library
Python only: no `match_aou`, no torch, no subprocess, no training, evaluation, replay,
checkpoint inference or benchmark preflight. It fails if any source byte changes.

## Endpoint and reading rules

- **Primary endpoint:** final round (`updates_completed = 375`), ten-base-cell equal-weight macro
  `SEVERE − MILD P(SELF_PRESERVATION_ABORT)` at the certified ego's immediate-FD wake. No success
  threshold is defined.
- **Trajectory quantities are secondary / descriptive.** All 16 rounds re-measure the **same 20
  development worlds**; they are repeated measures, not independent samples. No best checkpoint
  is selected. The two arms are compared side by side and never pooled.
- **Credit comparisons are not counterfactual action values.** Same-batch comparisons share the
  critic and normalization state but still compare different episodes; ABORT-vs-not gaps are
  observational associations.
- **Recorded limitation — pre-update run-to-run timing nondeterminism.** At update 0 the actor
  weights are proven identical (preflight). The aggregate pre-update endpoint, all 40
  immediate-FD wakes (tick, P(ABORT), selected action) and all selected actions and rewards are
  identical to actor-only. 18 / 60 episodes still differ after normalization: ordinary-wake ticks
  by at most 1, episode length by at most 2 ticks (the launch-time note said "±1"), ~1e-7
  probability drift, UUIDs in free text. The same class appears between historical actor-only
  and CTDE R1 at one SHA. **Per-episode trajectories across arms are not bit-identical.**

## Layout

| Path | Content |
|---|---|
| `artifact_sha256.txt`, `source_manifest.json` | path, bytes, SHA-256, role, disposition and measured SHA of all 47 source artifacts (this run, actor-only run, PR #71 blobs, historical CTDE R1, manifest, archive index) |
| `review_precheck.json` | machine-readable validity facts (provenance, plan vs config, manifest, completion, exit code, accounting, failures, seeds vs actor-only, final round, coverage, schemas, pre-update comparability, anomalies). **Not a verdict** |
| `package_checks.json` | determinism, hash, import-audit and negative-test results |
| `run_artifacts/` | byte-identical copies: `authorized_plan.json`, `preflight.json`, `pre_update_comparability_observation.json`, `launch_record.json`, `run_config.json`, `run_summary.json`, `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl`, `native_exit_code.txt`, `invocation_start_local.txt` |
| `extracted/eval_immediate_fd_wakes.jsonl` | all 640 evaluation immediate-FD wakes (16 rounds × 20 groups × MILD/SEVERE) |
| `extracted/behaviour_summary.json` | CTDE per-round MILD/SEVERE P(ABORT), per-group and per-cell deltas, macro, switches with denominators, final round, retention descriptors |
| `extracted/actor_only_comparator.json` | actor-only 16-round trajectory **reproduced from its run artifacts** and cross-checked against PR #71; provenance; PR #71 training windows and credit structure |
| `extracted/ctde_vs_actor_only.json` | primary endpoint side by side, all 16 rounds side by side, descriptive retention quantities for both arms |
| `extracted/fd_selected_ego_credit_rows.jsonl` | **all** 1526 CTDE credit rows with `wake_kind = immediate_fuel_damage` and `is_fd_selected_ego = true`, verbatim, plus a verified `x_join` block (P(ABORT) at the wake, argmax, decisions after the FD wake) |
| `extracted/credit_summary.json` | severity summaries, same-update-batch comparison, non-counterfactual ABORT-vs-not gaps, 15 × 25-iteration windows, structural GAE verification |
| `extracted/behaviour_credit_timeline.json` | the 15 windows aligned with evaluation rounds, training P(ABORT), credit gaps and critic diagnostics, beside actor-only |
| `extracted/historical_ctde_r1_context.json` | historical CTDE R1 config identity and per-round persisted values (secondary context) |
| `scripts/extract_evidence.py`, `scripts/package_checks.py` | the extractor and the check harness |

Large artifacts stay outside Git and are identified by hash only: `episode_outcomes.jsonl`
(86 MB), `train_credit_diagnostics.jsonl` (11.9 MB), `checkpoints/ckpt_iter0374.pt`,
`training_console.log` (`*.log` is git-ignored) and `launch.cmd`. `.gitattributes` disables
end-of-line conversion so copied files stay byte-identical.

## Reproducing

Materialize PR #71's package as raw blobs (no end-of-line conversion), then:

```bash
python research_evidence/generalized_v2/semantic_action_ctde_dev_r1/scripts/extract_evidence.py --comparator-evidence-dir <PR71 blobs> --out <dir> --copy-run-artifacts --verify-against research_evidence/generalized_v2/semantic_action_ctde_dev_r1/artifact_sha256.txt
```

The output is byte-identical to this package when the sources match. Any mismatch exits with
code 2 and `EVIDENCE CHECK FAILED`.

## Reproduced quantities (descriptive; full denominators in the JSON files)

**Validity facts.** Native exit code `0`; 375 / 375 productive updates; 3000 successful of 3008
attempted (≤ 4500); 8 failures, all `setup` / `FuelDamageError` (`no_fd_eligible_ego`), with
failed seeds identical to actor-only. The attempted seed stream (3000000–3003007) is identical to
actor-only in iteration, attempt ordinal and status for all 3008 seeds. All 16 tracebacks in the
console are those 8 chained failures. 16 rounds, 960 / 960 evaluation episodes. Final round
`post_update` @ 375; 10 / 10 base cells; 20 / 20 groups eligible. 9058 credit rows cover every
transition of every update. Schemas uniform (outcome v4, wake v2, credit v1); representation
uniform; credit `training_mode` uniformly `ctde`.

**Primary endpoint (final round).** CTDE macro `+1.46e-7` (all ten cells between `+1.08e-7` and
`+2.14e-7`), 0 directional / 0 reverse switches. Actor-only: `+0.000888`, 0 / 0.

**Trajectory (CTDE vs actor-only macro, directional switches).**

| update | CTDE MILD / SEVERE P(ABORT) | CTDE macro, sw | actor-only MILD / SEVERE | actor-only macro, sw |
|---|---|---|---|---|
| 0 | 0.4989 / 0.4984 | −5.2e-4, 0 | 0.4989 / 0.4984 | −5.2e-4, 0 |
| 25 | 0.1754 / 0.1754 | −6.4e-6, 0 | 0.1800 / 0.1800 | ~0, 0 |
| 50 | 0.1184 / 0.1184 | +3.6e-6, 0 | 0.1302 / 0.1302 | ~0, 0 |
| 75 | 0.3521 / 0.3521 | +4.6e-5, 0 | 0.1281 / 0.4236 | +0.296, 4 |
| 100 | 0.2043 / 0.2044 | +4.2e-5, 0 | 0.0696 / 0.6808 | +0.611, 19 |
| 125 | 0.1999 / 0.1999 | +5.9e-7, 0 | 0.0665 / 0.7168 | +0.650, 20 |
| 150 | 0.1498 / 0.1498 | +8.4e-7, 0 | 0.0503 / 0.6268 | +0.577, 19 |
| 175 | 0.1741 / 0.1741 | −3.0e-5, 0 | 0.0352 / 0.0381 | +0.003, 0 |
| 200–350 | 0.063–0.340, MILD = SEVERE to 4 d.p. | \|macro\| ≤ 4.0e-5, 0 | 0.029–0.095 | +0.0004 to +0.0061, 0 |
| 375 | 0.0436 / 0.0436 | +1.5e-7, 0 | 0.0391 / 0.0400 | +8.9e-4, 0 |

Descriptively, CTDE shows **no directional switch in any round** and a maximum macro of `4.6e-5`
(update 75). The transient actor-only separation at updates 75–150 has no CTDE counterpart. No
reverse switches occur in either arm.

**Training-side FD wakes.** CTDE MILD and SEVERE mean P(ABORT) at the certified ego's
immediate-FD wake differ by ≤ 0.0253 in every 25-iteration window. Actor-only diverged in windows
50–174 (for example 0.059 vs 0.672 in iterations 125–149).

**Credit (1526 FD-selected immediate-FD rows: 774 MILD, 752 SEVERE).**

- Raw advantage mean MILD `+0.078`, SEVERE `−0.333`. Normalized mean MILD `+0.177`, SEVERE `−1.023`.
- `value_old` mean `−0.1915` in **both** severities; value target MILD `−0.114`, SEVERE `−0.525`.
- TD residual mean MILD `−0.008`, SEVERE `−0.033` (medians −1.2e-6 and −1.8e-6).
- Episode reward mean MILD `−0.099`, SEVERE `−0.564`. ABORT selected MILD 19.1 %, SEVERE 20.1 %.
- Same update batch (309 batches with both severities): SEVERE − MILD raw advantage mean `−0.408`
  (287 negative / 22 positive); value target `−0.408` (289 / 20); `value_old` `+0.0001`
  (194 / 115); TD residual `−0.026` (166 / 143); P(ABORT) `+0.0004` (145 / 164).
- ABORT minus not-ABORT (non-counterfactual): MILD raw `−0.285`, TD `−0.084`, reward `−0.340`;
  SEVERE raw `+0.215`, TD `−0.065`, reward `+0.249`.

**Structural credit verification.** All 9058 rows reconstruct exactly (max error 0.0) under
per-episode GAE over the global decision order: `delta = r + V_next − V_old`, `A = delta + 0.95·A_next`,
`target = A + V_old`, plus the batch normalization. The reward is terminal-only: 1192 episodes
carry it on their final decision.

| | CTDE | actor-only (PR #71) |
|---|---|---|
| ego chains checked | 6689 | 6689 |
| chains with > 1 transition | 1192 | 1222 |
| chains with varying raw advantage | **1192** | **0** |
| episodes checked | 2999 | 2999 |
| episodes with > 1 transition | 2417 | 2415 |
| episodes with varying raw advantage | **2417** | **0** |

`value_old`, TD residual and value target also vary within those chains and episodes. That
establishes finer state- and time-dependent credit, **not** its causal correctness.

**Timeline observation (descriptive).** In every window, SEVERE − MILD same-batch `value_old`
differences stay between −0.0038 and +0.0055, and the credit gap between severities is carried by the value
target / episode outcome. That same-batch raw-advantage gap (−0.300 to −0.613 per window; pooled −0.297 to −0.579) is of similar size to actor-only's
pooled gap (−0.273 to −0.563), while CTDE's behaviour shows no severity separation in any window.

## Known anomalies (recorded, not repaired)

- `v2_behaviour.metric` keeps the legacy label `severe_minus_mild_aggregate_abort_mass`; the value
  is the semantic ABORT leaf (same as PR #71).
- The pre-update timing nondeterminism above; the launch-time description "±1 tick" is corrected
  to "wake ticks ≤ 1, episode length ≤ 2".
- 8 expected certified-FD eligibility setup failures, identical in seed to actor-only.
