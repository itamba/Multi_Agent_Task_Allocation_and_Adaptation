# Graph RL project handoff — current snapshot

> **Status: current state only — not a contract and not a history.** This snapshot was written
> on 2026-09-14 on task branch `docs/project-guidance-restructure` against `main` =
> `ae42cb01677f94868b2873008d87be677e31f0c8`. **GitHub is authoritative for live branch, PR and
> ownership state**: resolve live `main` and open PRs first
> ([`cc_review.md` §8](docs/workflows/cc_review.md#8-receiving-a-hand-off)). Contracts live in
> [`docs/contracts/`](docs/contracts/); everything before this snapshot lives in
> [`docs/history/`](docs/history/). Update this file in the same PR as any change to current
> state, replacing stale lines rather than stacking supersession notes.

## 1. Phase

- **Active research line: GENERALIZED-V2** — the two-stage route-relative population (PR #57)
  and its frozen ten-cell benchmark and evaluation construct (PR #59), both merged into `main`.
  GENERALIZED-V1 remains a valid, preserved design.
- **Two GENERALIZED-V2 development-profile R1 arms (actor-only and CTDE) were executed at
  measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`**, and their run packages are
  preserved in the open evidence PRs #61 and #62 (§4). Both consumed an **external** benchmark
  manifest that is neither committed nor inspected, and whose producing code SHA is unverified
  (§4). The research decision that authorized the runs, and any conclusion drawn from them, is
  not recorded in the repository (§7).
- **Closed:** Phase A (fixed cell, FD-BASELINE-v1); the FD-VARIABLE-SEVERITY-v1 actor-only
  baseline; the Phase-B CTDE implementation; GENERALIZED-V1 Tasks 1–5, early stopping and the
  per-wake diagnostics; the deterministic-P1 backend and the certified-FD physical-state repair;
  the GENERALIZED-V1 R1 measurement.

## 2. Active owner and task

| Item | State |
|---|---|
| Writable repository task | documentation restructure on branch `docs/project-guidance-restructure`, authorized by the user's documentation packet; one draft PR; **UNREVIEWED**; no merge authorized |
| Reviewer | GPT orchestrator (read-only; exact-candidate review) |
| Open evidence PRs | #61 and #62 — draft, evidence-only, **untouched by this task**; the user deferred decisions about them |
| Implementation candidates | none open |
| Scientific runs in progress | none recorded |

## 3. Candidates and PRs

| PR | Branch | Head | Status |
|---|---|---|---|
| documentation restructure | `docs/project-guidance-restructure` | resolve on GitHub | draft, unreviewed |
| #61 | `evidence/generalized-v2-actor-only-dev-r1` | `1375a881637a9a32721a1630f598adc571422a47` | draft; body states `READY_FOR_REVIEW / UNREVIEWED`; no GitHub review or comment record |
| #62 | `evidence/generalized-v2-ctde-dev-r1` | `b2bbe7a6235c3b9255106826cfb268af7e73f72d` | draft; body states `READY_FOR_REVIEW / UNREVIEWED`; no GitHub review or comment record |

Both evidence heads are single commits whose parent is `ae42cb01677f94868b2873008d87be677e31f0c8`
and whose trees add only `research_evidence/generalized_v2/…`.

## 4. Runs and evidence — current references

| Measurement | Measured code SHA | Verdict, with its provenance | Record |
|---|---|---|---|
| Phase-A long baseline (`fixed_cell_v1`, FD-BASELINE-v1, actor-only) | `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` | `APPROVE — VALID MEASUREMENT` (independent review) | [measurements](docs/history/measurements.md) |
| FD-VARIABLE-SEVERITY-v1 actor-only baseline | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | `APPROVE — VALID MEASUREMENT`; primary finding negative | [measurements](docs/history/measurements.md) |
| GENERALIZED-V1 R1 (actor-only, legacy backend) | `4af6c5aa5dd28072692bfda63282964b55010aae` | `APPROVE — VALID MEASUREMENT`; primary FD finding negative; comparator `manifest_id 0e15f007ef176bf977f8b93bb91289f48c16f25ee9eee282ffd1a89477f6fc0d` | [measurements](docs/history/measurements.md) |
| Fresh deterministic-P1 arm (GENERALIZED-V1 population, `p1_milp_v1`) | `ae1941035991df4719df212c4b5dd07db89aee4a` | accepted as valid; negative primary MILD-vs-SEVERE result; **not a clean causal comparator to R1**; nothing beyond the SHA is recorded | [measurements §6](docs/history/measurements.md#6-the-p1-arms) |
| Earlier P1 arm | not recorded | **`ABORTED / DO NOT RESUME`** — not a measurement | [measurements §6](docs/history/measurements.md#6-the-p1-arms) |
| GENERALIZED-V2 benchmark manifest (an external artifact, **not** part of either evidence package) | producer code SHA **unverified** — no preflight evidence is in the repository; the consuming runs' SHA does not establish it | `manifest_id ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8` (in both consuming run configs); file SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103` at the external path `C:/Users/Itama/PycharmProjects/graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0/benchmark_manifest.json` (as recorded in PR #62's `artifact_sha256.txt`); not committed; not inspected by GPT or in this restructure; no accessible review record | [measurements §7](docs/history/measurements.md#7-generalized-v2-development-r1-arms) |
| GENERALIZED-V2 development R1 — actor-only | `ae42cb01677f94868b2873008d87be677e31f0c8` | **no accessible verdict record** in PR #61 or its `artifact_sha256.txt` (this does not establish that no review occurred) | evidence PR #61 |
| GENERALIZED-V2 development R1 — CTDE | `ae42cb01677f94868b2873008d87be677e31f0c8` | PR #62's body and `artifact_sha256.txt` record a **prior GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT`**; no GitHub review record exists and this repository has not independently verified it | evidence PR #62 |

**What the two preserved V2 arms' artifacts state** (restated, not re-approved): both run
configs record `episode_design = generalized_v2`, `match_aou_backend = p1_milp_v1`,
`benchmark_profile = development`, the manifest id above, `n_iterations = 375`,
`episodes_per_iteration = 8`, `generalized_max_attempts_per_iteration = 12`,
`base_seed = 3000000`, evaluation and checkpoints every 25 iterations, `early_stopping = false`,
`fuel_damage_mode = seeded_variable`, and a clean provenance at the measured SHA. Both summaries
report `updates_completed = 375`; training 3008 attempted / 3000 successful / 8 failed (all
`setup` `FuelDamageError`); evaluation 960 / 960 / 0 over 16 rounds;
`accounting_reconciled = true`; 3960 outcome records. Their `run_summary.json` carries the known
wrong `generalized.cardinality_sampler` label
([artifacts_metrics §6.1](docs/contracts/artifacts_metrics.md#61-known-summary-label-defect-run_summaryjsongeneralizedcardinality_sampler)).
**None of this is a validity, behaviour or comparison verdict.**

**Standing interpretation rules for these references:** R1 and the fresh P1 arm are distinct
repository and population measurements with no causal solver-quality inference; reviewed
measurements are reused as recorded by default and are executed again only under an authorized
plan that names it; the aborted P1 arm stays `DO NOT RESUME`; the old fixed-cell CTDE measurement
stays out of scope unless the user explicitly asks
([`experiments.md` §4.4, §6](docs/workflows/experiments.md#44-comparator-discipline)).

## 5. Concrete unresolved next actions

1. **GPT exact-candidate review** of the documentation restructure PR. No merge is authorized.
2. **Decide the evidence PRs #61 and #62** (review, merge or retain). The user deferred this.
3. **Recover the missing research decision** on the V2 development arms before any further V2
   scientific action: whether the actor-only arm was reviewed, what comparison conclusion (if
   any) was drawn, and whether use of the confirmatory profile is authorized or has occurred.
4. *Optional, not authorized:* a separate code task to correct the V2
   `generalized.cardinality_sampler` summary label in `graph_train._generalized_summary`,
   without touching archived artifacts.
5. *Open engineering caveat, not authorized:* a playback-export failure still routes as an
   ordinary `run`-stage episode failure
   ([measurements](docs/history/measurements.md#2-measurement-records)).

## 6. Blocked or unauthorized now

- merging this PR, PR #61 or PR #62 without explicit authorization;
- **without an authorized bounded plan that names it**
  ([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)):
  any further V2 benchmark preflight or change to the manifest the preserved arms used; further
  V2 training or evaluation, development or confirmatory; re-running either preserved arm; any
  R1 or P1 rerun, repair or extension; a new control or CTDE arm; the five full cluster runs;
  retuning;
- resuming, repairing or extending the aborted P1 arm (`DO NOT RESUME`);
- early stopping under `generalized_v2` (refused by code) and V2 cardinality above `A = 6`
  (`A = 8` / `A = 10` are engineering evidence only);
- changes to locked layers, BLADE, the solvers, PPO, CTDE, the reward or the observation /
  action contracts;
- `p(destroy) < 1` (deferred) and checkpoint resume (out of scope);
- reviewing or comparing the old fixed-cell CTDE measurement unless the user asks;
- moving or deleting protected refs, evidence refs or preserved run directories.

## 7. Protected refs, preserved evidence and known gaps

**Registry:** [`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup).

**Known gaps in this snapshot:**

- The final research decision on the V2 development arms is unavailable (§5 item 3).
- No verdict record is accessible for the V2 actor-only development arm (which does not
  establish that it was not reviewed); the CTDE verdict is attributed to PR #62's own text only.
- No authorization, producer provenance or review record for the V2 benchmark preflight is
  accessible; the external manifest was not inspected, and whether confirmatory data was used is
  not inferred.
- The R1 run directory, the fresh P1 arm's artifacts and the old fixed-cell CTDE measurement's
  identity are not recorded in the repository.
