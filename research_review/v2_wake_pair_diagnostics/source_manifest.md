# GENERALIZED-V2 immediate-FD wake-pair diagnostics — source manifest

**Temporary, review-only package.** This branch and PR exist so the GPT orchestrator can compare
matched MILD vs SEVERE immediate-fuel-damage wakes directly. They are expected to be closed and
deleted after review and are **not** a permanent evidence archive. Nothing here is a validity,
behaviour or comparison verdict.

**No scientific execution and no policy recomputation occurred.** No training, evaluation,
benchmark preflight, replay, checkpoint loading or inference, policy / logit recomputation, BLADE,
solver, resume, repair or reconstruction was run. `extract_wake_pairs.py` uses the Python standard
library only (no `match_aou` import) and parses existing JSON / JSONL. Every per-wake value is
copied from the recorded `wake_decisions` entry; the only derived values are the arithmetic
comparisons in `matched_mild_severe_pairs.jsonl` and the descriptive counts / mean / min / max in
`extraction_summary.json`.

**Original artifacts were unchanged.** The extractor hashes and stats (SHA-256, bytes,
`st_mtime_ns`) every source file before and after reading; all 20 are identical
(`extraction_summary.json:/sources_unchanged_by_extraction = true`, per-file
`unchanged_after_extraction`). An independent `sha256sum` + `stat` pass before and after the run
also matched byte-for-byte.

## 1. Sources

Measured code SHA of all five runs, read from each `run_config.json:/provenance/git/commit` with
`dirty = false`: `ae42cb01677f94868b2873008d87be677e31f0c8`. All five record benchmark
`manifest_id ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`, profile
`development`. Evidence-commit SHAs below are ledger locations, not measurement identities.

| Variant | Local path | `training_mode` | Hash ledger |
|---|---|---|---|
| `actor_only_r1` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_actor_only_dev_r1_seed3000000_ae42cb0` | `actor_only` | PR #61 `artifact_sha256.txt` @ `1375a881637a9a32721a1630f598adc571422a47` |
| `ctde_r1` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_r1_seed3000000_ae42cb0` | `ctde` | PR #62 `artifact_sha256.txt` @ `b2bbe7a6235c3b9255106826cfb268af7e73f72d` |
| `smallbatch` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_smallbatch_seed3000000_ae42cb0` | `ctde` | PR #64 `artifact_sha256.txt` @ `90516d51beeddacded2b89a321d14291e411f2b0` |
| `largebatch` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_largebatch_seed3000000_ae42cb0` | `ctde` | PR #64 (same) |
| `fd80` | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_fd80_seed3000000_ae42cb0` | `ctde` | PR #64 (same) |

`run_id` in the outputs is the run directory basename (the run configs carry no separate run-id
field).

SHA-256 of the files read — each equals the value in its ledger (for PR #61 / #62
`episode_outcomes.jsonl` is the recorded pre-sharding original):

| Variant | `episode_outcomes.jsonl` | `eval_records.jsonl` | `run_config.json` | `run_summary.json` |
|---|---|---|---|---|
| `actor_only_r1` | `b1bcc0647c45932e1ad6c939d0c740100291dc60a6a83371efb6485d8ebb9da3` | `2987bf78539eddfbf1acc054ce12717d737bfc965e09c970af38aed61960d03c` | `8e104fc012eb1b69543b2d9145ceaf40d3bbe722d74404389b3028f7c2935801` | `db97794b3fec4875afb870ea84d8d2fa2338e497891fbafc68def12649faebc9` |
| `ctde_r1` | `fbb18848eef09f187f9335eaed1aa97d686b1ae4494484e06d333afc27d6327b` | `76951246c1a7be7942ab4216421840a6add2dd499f238c276a0bd5728690f1b0` | `beac06d463639b25ebe4e00b42b7f1a370ea9fe2500c2cfb4d65137e11030b27` | `714fe0cc10189380cce553af8eacd898e28a745bc610842008b013f4b4cfebaf` |
| `smallbatch` | `b763d588fac327dc68cd4ce1c982aabb3307e19bc62cba55cfe020ff6e59d052` | `98dc5967ea9768f7da7a52af9dd8e098f4cc7b4e5518755ec5b8931c8acef9a2` | `676e384b0c9984da62fae3b7a097afa8b18e40eceadfe72c745e0edf2ddc3077` | `71c907cf46af1a18d7d5de3810b8e54a053027bc27a20ae2d8a2ccc868b9161d` |
| `largebatch` | `0307c03f7a37f8de5ee7be53390c7a631eba3c8323e82c0bd34584179ee68d68` | `997c7b30739aff35ecef277aa147d1b06b2ce2479cc2e90eab5494ea12cc0435` | `f36baadccbc07bfcf7d636be942a3f24fd33533c4251e488a6b1998878259149` | `09bdda4342f4d255d354ac915055b86d93e236e1566dae8988082e737724827e` |
| `fd80` | `865bca5c730cccd6420033bd51ee4da87e81ec5443fa3b3b6a20f7a079892d5b` | `57e7b85bec8b193c56241ca0e71dfe4313c224677d07ea70d069ade648150c03` | `8323148ba2c74bee417c6ba89bce18103986bc294fe802fdebbfdabefd3b6a6b` | `87b07700296c86031589c3c6b0b61897454472ba8544e2d3fb06f9db083c359c` |

Byte sizes and `st_mtime_ns` are recorded per file in `extraction_summary.json:/sources`. The
external benchmark manifest file was **not** read.

## 2. Extraction rule

1. Rounds: every record of `eval_records.jsonl` (1 `pre_update` + 15 `post_update` per run). Round
   identity is `(evaluation_stage, updates_completed, eval_round_ordinal, benchmark_manifest_id)`
   ([artifacts_metrics §5](../../docs/contracts/artifacts_metrics.md#5-per-wake-fd-policy-diagnostics)).
   Group keys per round come from the round's own
   `v2_benchmark_groups.benchmark_profile_identity.group_keys`.
2. Members: outcome records with `phase ∈ {pre_update, post_update}`, indexed by round identity,
   `benchmark_group_key` and `benchmark_v2.member_cell` (cross-checked against `cell` and
   `severity`). Training rows are skipped. CLEAN members are counted but produce no wake rows.
3. Wake rows (`immediate_fd_wakes.jsonl`): for every MILD and SEVERE member, each
   `wake_decisions` entry with `wake_kind == "immediate_fuel_damage"`. The complete recorded
   entry is kept under `wake_decision`; selected fields are lifted beside it with episode context
   under `episode`. Zero or several such wakes would be emitted as found and stated as a pair
   reason.
4. Pairs (`matched_mild_severe_pairs.jsonl`): one record per run × round × profile group key.
   Formed only when exactly one MILD and one SEVERE member exist and each has exactly one
   immediate-FD wake; otherwise `pair_formed = false` with explicit `reasons`. Deltas are
   SEVERE − MILD and `None` whenever either side is not a number.
5. Summary (`extraction_summary.json`): per run and round, counts and descriptive statistics
   only, plus a cross-check of extracted selected-action counts against the round's recorded
   `eval_fd_meta_action_counts_{mild,severe}`.

Field naming, from the writer (`src/match_aou/rl/training/graph_tick_loop.py`, the per-wake
record builder): the `dist_to_ego_norm` column is stored as **`task_distance_norm`** (task-feature
column 1), and the name is kept as recorded. The raw per-wake schema has no
`selected_joint_cell_abort_fraction`. That is a run-level rate in
`graph_train._wake_diag_digest` equal to the share of wakes whose
`selected_meta_action_name == "SELF_PRESERVATION_ABORT"`, so each row carries the per-wake
indicator `selected_joint_cell_is_abort` derived from that one recorded field.
`aggregate_p_{plan,engage,abort}` are copied from
`aggregate_probability_per_meta_action` and are total column mass, not the probability of the
selected action.

Elementwise `reachable_by_ego` / `task_distance_norm` comparisons are reported whenever the two
vectors have equal length. The records do not independently certify that task-node order is
identical across the two members, so an elementwise difference is a difference of recorded
vectors by position only.

## 3. Counts

| | Expected | Actual |
|---|---|---|
| Evaluation rounds | 5 × 16 = 80 | 80 |
| Groups per round | 20 (development: world ordinals 0..1 × 10 base cells) | 20 in every round |
| Outcome members per round | 20 CLEAN / 20 MILD / 20 SEVERE | 20 / 20 / 20 in every round |
| Immediate-FD wake rows | 5 × 16 × 20 × 2 = 3200 | 3200 (640 per run) |
| Pair records | 5 × 16 × 20 = 1600 | 1600, all formed; 0 not formed |

Every MILD and SEVERE member carries exactly one immediate-FD wake. Extracted selected-action
counts equal the recorded `eval_fd_meta_action_counts_*` in all 80 rounds for both severities.

Rounds tagged in `extraction_summary.json` `landmarks` (identified, not interpreted):
`smallbatch` updates 200 and 250 (ordinals 4 and 5); `largebatch` update 30 (ordinal 3); final
`post_update` round per run — `actor_only_r1` 375, `ctde_r1` 375, `smallbatch` 750,
`largebatch` 150, `fd80` 375 (ordinal 15 each).

## 4. Integrity checks and schema observations

- Source SHA-256 equals the PR #61 / #62 / #64 ledgers for all 20 files; unchanged after extraction.
- Measured SHA, `dirty = false`, `training_mode`, manifest id and profile as expected in every
  run config, eval round and wake row. `benchmark_v2.reconstructed_identity_verified = true` on all
  3200 rows.
- No incomplete round identity, duplicate round identity, duplicate pair identity, stray group
  key, duplicate member, missing member, or member-condition field disagreement
  (`extraction_summary.json:/anomalies` is empty).
- No `None` in any lifted wake-row field; every lifted diagnostic is present in each recorded
  `wake_decision`. `joint_entropy_normalized` is defined on every row.
- Across all 3200 rows `selected_meta_action_name == joint_argmax_meta_action_name`.
- Within every pair the two members share `seed` and base cell, and both vectors have equal length
  in all 1600 pairs.
- **Observation:** in 7 of 1600 pairs the MILD and SEVERE wake `tick` differ by one tick, all in
  group `A5-D0-w001`: `smallbatch` update 550 (913 / 912), `largebatch` update 10 (912 / 913),
  `fd80` updates 100, 200, 275, 300, 350 (913 / 912) (MILD / SEVERE). Recorded as found
  (`same_wake_tick = false`).
- `fraction_task_distance_clipped` is written as `0.0` by the writer when a wake has no task nodes;
  no such wake occurs here (`n_task_nodes` ranges 2..8 across the 3200 rows).

## 5. Package files

| File | Rows | SHA-256 at generation |
|---|---|---|
| `extract_wake_pairs.py` | — | `66f93860244cdd026d1694f14189214d3471059ac31cc9bb2c56aba7a6acfa43` |
| `immediate_fd_wakes.jsonl` | 3200 | `f8282642169fa8d6cb04e8f9e7a83cb57b2ff6ba8c8023cf57843b43e545d9b7` |
| `matched_mild_severe_pairs.jsonl` | 1600 | `6fc4e7958f9d90515c73a28779253fa46b7ace6b495af128a08e89a567c39262` |
| `extraction_summary.json` | — | `02bf0a8005907d5d75e39803beca72545f5d5914cb690a6ac069613d6e4d0794` |

Regenerate with `python extract_wake_pairs.py` from this directory (any Python 3 with the
standard library; the source run directories must be present at the paths above).
