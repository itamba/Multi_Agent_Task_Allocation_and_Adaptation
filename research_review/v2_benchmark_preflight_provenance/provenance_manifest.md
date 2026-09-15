# GENERALIZED-V2 benchmark preflight — provenance manifest

> **TEMPORARY / REVIEW-ONLY — DO NOT MERGE.** Review transport for the already-existing V2
> benchmark preflight at
> `C:\Users\Itama\PycharmProjects\graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0`.
> Nothing was executed to build this package: no preflight, manifest generation, training,
> evaluation, replay, checkpoint load, BLADE or solver. Only file reads, JSON parsing and
> SHA-256 hashing. Hashes and sizes: [`artifact_sha256.txt`](artifact_sha256.txt),
> [`scenario_sha256.txt`](scenario_sha256.txt).

## 1. Manifest identity

| Fact | Value | Basis |
|---|---|---|
| File SHA-256 / bytes | `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103` / 204244 | recomputed from source bytes |
| `manifest_id` | `ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8` | stored; **recomputed and equal** (§6) |
| `schema` / `schema_version` / `design` | `generalized_v2_benchmark_manifest` / `1` / `generalized_v2` | manifest |
| `label` | `generalized_v2_scientific_benchmark_v1` | manifest |
| `required_match_aou_backend` | `p1_milp_v1` | manifest |
| World groups / members | `n_worlds = 120`, `n_members = 360`, `group_size = 3` (`clean`, `mild`, `severe`) | manifest; 120 `worlds` entries counted |
| Base cells | 10 (`A2..A6` × `D0/D2`), 12 worlds each, ordinals `0..11` | counted |
| `profiles` | `development: [0, 1]`, `confirmatory: [2..11]` | manifest |
| Profile membership | development 20 groups / 60 members; confirmatory 100 groups / 300 members | counted from world ordinals and per-world `profile` |
| Seeds | 120 unique, in `[2000000, 2000640)`; `seed_list_sha256 = 6d65318a…e88518` | recomputed; equals report |

## 2. Producer code provenance

Verbatim from `benchmark_preflight_report.json:/provenance/git`:

| Field | Value |
|---|---|
| `available` | `true` |
| `commit` | `ae42cb01677f94868b2873008d87be677e31f0c8` |
| `branch` | `main` |
| `dirty` | `false` |
| `dirty_path_count` | `0` |
| `repo_root` | `C:\Users\Itama\PycharmProjects\Multi_Agent_Task_Allocation_and_Adaptation` |
| `reason` | `null` |

**Classification: `VERIFIED_EXACT_FROM_PRODUCER_RECORDED_PROVENANCE`** — the full producer
code SHA is read from the report the preflight itself wrote, not derived from the directory
suffix `ae42cb0` and not from the manifest `notes` string (which also names the SHA, as
operator-supplied text).

**Limitation.** This is provenance emitted by the executing project code. It is **not an
external attestation**: its truth rests on that code having run unmodified and on the report
bytes being unaltered since. `ae42cb0…` exists in the repository and is an ancestor of
`main` `fcbedc2…`.

## 3. Historical clean state

The report records `dirty = false` and `dirty_path_count = 0` at preflight time. This is taken
from the artifact only; **nothing about historical cleanliness is inferred from today's
checkout.** Per the code (§7), `git status --porcelain` counts tracked changes and untracked
non-ignored files; git-ignored files are outside that verdict.

## 4. Invocation provenance

**Known from surviving artifacts**

- Effective request (`report:/request`): `worlds_per_cell = 12`, `benchmark_base_seed = 2000000`,
  `max_candidates_per_cell = 64`, `n_base_cells = 10`, candidate span `[2000000, 2000640)`.
- Effective settings: report `policy = deterministic_per_cell_window_fail_closed_v2`;
  `episode_design` = `generalized_v2`, `bounded_backoff_v1`, `certified_both_severities_v1`,
  `completion_boundary_v1`, `event_conditioned_continuation_v1`; `match_aou_backend =
  p1_milp_v1`; `fuel_damage` seeded-variable (probability 0.5, mild 0.5, leg-progress 0.3,
  RTB margin 1.1); `geometry` min target distance 200 km, min known separation 100 km,
  stretch ratio 0.5, no SAMs, randomized red airbases.
- Completion: `status = complete`, `complete = true`, `manifest_written = true`,
  `failure = null`; attempted 120, accepted 120, rejected 0; console prints the same.
  Accepted seeds equal the manifest seeds and the 120 scenario file names.
- Timestamps: `report:/generated_utc = 2026-09-13T12:26:07.718521+00:00`;
  `report:/seconds = 8.463965100003406` (harness-internal, not wall clock);
  `invocation_start_local.txt`: `START_UTC=Sun 09/13/2026 15:25:53.69`,
  `END=Sun 09/13/2026 15:26:08.26`.
- Native exit code: `native_exit_code.txt` = `0` (bytes `0 \r\n`).
- Environment clues: console warnings from
  `C:\Users\Itama\anaconda3\envs\nlp_env\Lib\site-packages\gymnasium\...` (LOCAL Windows,
  `nlp_env`); Windows `repo_root`.

**Unknown**

- the exact original argv;
- whether `--config` was explicitly supplied (backend and geometry could come from a preset
  or from flags/defaults; the report does not record a config path);
- the exact Python version (no artifact records it);
- how `--label` / `--notes` were supplied.

*Code-derived inference, not a fact:* if the entry point was `main()` at `ae42cb0` (the banner
format matches its `print`), `episode_design` can only have come from `--episode-design`
(default V1), so argv would have included `--episode-design generalized_v2`. The actual
entry point is not recorded.

## 5. Authorization and prior review

- **Authorization provenance: `NOT PRESERVED / NOT PROVEN`.** The manifest `notes` string
  ("First authorized scientific GENERALIZED-V2 benchmark population; …") is operator-supplied
  descriptive text passed through to the manifest; it is not an authorization record.
- **Prior review provenance: `NOT PRESERVED / NOT PROVEN`.** This preflight is not labelled as
  previously reviewed.

## 6. Static integrity checks (parse/hash only, no project import)

| Check | Result |
|---|---|
| Manifest file SHA-256 recomputed | `dd72afc9…a103` — matches |
| `manifest_id` recomputed with the producer rule (§7) over the manifest minus `manifest_id` | `ef17a68a…46ea8` — matches stored id |
| File bytes equal canonical JSON of the full record | true |
| Report `/manifest` block: `manifest_id`, `file_sha256`, `seed_list_sha256`, 120 / 360 | all match |
| Five copied files vs source bytes, and committed blob contents vs source SHA-256 | identical |
| Scenario ledger: 120 files, each exactly once, 2048776 bytes | yes |
| Source SHA-256 / size / mtime of all 125 source files before vs after packaging | unchanged |

## 7. Producer-code cross-check at `ae42cb01677f94868b2873008d87be677e31f0c8` (read only)

- `_REPO_ROOT` — `graph_train.py`: `_REPO_ROOT = Path(__file__).resolve().parents[4]`, i.e.
  the checkout containing the imported module file, not the working directory;
  `graph_benchmark_preflight.py` imports it from `.graph_train`.
- `_git_provenance(repo_root)` — `graph_train.py`: runs `git rev-parse HEAD`, then
  `git status --porcelain` (required; `dirty = bool(changed)`, `dirty_path_count =
  len(changed)`, then `available = True`), then optional `git rev-parse --abbrev-ref HEAD`.
- V2 preflight — `_run_v2_benchmark_preflight`: `git_info = dict(provenance if provenance is
  not None else _git_provenance(_REPO_ROOT))`; refuses when not `available`; warns on dirty;
  `_build_v2_report` writes `"provenance": {"git": git_info}`. (`provenance` is a keyword
  override not exposed by the CLI `main()`.)
- Canonical identity — `graph_generalized.py`: `_canonical_json` = `json.dumps(payload,
  sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)`;
  `manifest_identity(payload)` = SHA-256 of its UTF-8 bytes; `build_v2_benchmark_manifest`
  hashes `draft.payload()` (no `manifest_id`); `write_benchmark_manifest` writes
  `_canonical_json(manifest.to_record())`; `v2_manifest_from_record` recomputes over
  `{k: v for k, v in dict(record).items() if k != "manifest_id"}`.

## 8. Technical conformance

The recorded effective settings are descriptively consistent with the implemented V2
benchmark contract ([`training_benchmarks.md` §9](../../docs/contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation)):
V2 schema v1; ten `A × D` base cells; 12 worlds per cell; development `0..1` / confirmatory
`2..11`; policy `deterministic_per_cell_window_fail_closed_v2`; `p1_milp_v1`; per-cell
64-seed half-open windows. **Technical conformance does not establish research authorization
or scientific validity.**

## 9. Known anomalies and caveats (not reasons to rewrite source artifacts)

- **V1 policy label in the console banner.** The console prints
  `policy=deterministic_per_cell_window_v1` while the report records
  `deterministic_per_cell_window_fail_closed_v2`. At `ae42cb0`, `main()` prints the V1
  constant `PREFLIGHT_POLICY` regardless of design; `_build_v2_report` writes
  `PREFLIGHT_V2_POLICY`. The report is the authoritative record.
- **`START_UTC` holds a local time.** `START_UTC=… 15:25:53.69` and `END=… 15:26:08.26`, while
  `generated_utc = 12:26:07.72Z` falls inside that interval only at a +03:00 offset. The label
  is wrong; the value is local time.
- **Exact argv absent** (§4).
- `native_exit_code.txt` and `invocation_start_local.txt` carry CRLF and trailing spaces;
  they are preserved as found.

## 10. Expected closure state for review

Producer SHA gap: closed by producer-recorded provenance. Clean-state gap: closed likewise.
Manifest identity: independently reviewable. Invocation: partially recovered. Authorization
gap: remains. Historical-review gap: remains.
