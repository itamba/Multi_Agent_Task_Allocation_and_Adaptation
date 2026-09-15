# Graph RL project handoff — current snapshot

> **Status: current state only — not a contract and not a history.** This snapshot was refreshed
> on 2026-09-16, when the approved GENERALIZED-V2 action-representation and credit-instrumentation
> implementation task opened from live `main` `63247404d88f714c6268383321ac25d766406055`.
> **GitHub is authoritative for live branch, PR and ownership state**: resolve live `main` and
> open PRs first ([`cc_review.md` §8](docs/workflows/cc_review.md#8-receiving-a-hand-off)).
> Contracts live in [`docs/contracts/`](docs/contracts/); everything before this snapshot lives in
> [`docs/history/`](docs/history/). Update this file in the same PR as any change to current
> state, replacing stale lines rather than stacking supersession notes.

## 1. Phase

- **The GENERALIZED-V2 development research chapter is closed.** Its population (PR #57), its
  frozen ten-cell benchmark and evaluation construct (PR #59), the development closure record
  (PR #66) and the benchmark-preflight provenance closure (PR #68) are merged. GENERALIZED-V1
  remains a valid, preserved design.
- **GENERALIZED-V2 development diagnosis is closed.** Completed at measured code SHA
  `ae42cb01677f94868b2873008d87be677e31f0c8`: actor-only R1, CTDE R1 and three CTDE development
  diagnostic arms (`smallbatch`, `largebatch`, `fd80`), plus a read-only matched MILD/SEVERE wake
  analysis over all five (§4, [`decisions.md` §1](docs/history/decisions.md#1-decision-log),
  2026-09-15).
- **The V2 benchmark-preflight provenance evidence is reviewed and durable**
  ([measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review)).
- **Closure and cleanup are complete** (2026-09-15): the local artifacts are organized and indexed
  under `C:\gra\`; the temporary evidence and review PRs are closed and their branches removed; the
  retired merged branches are removed; the four Graph-RL execution / source worktrees targeted by
  the cleanup (`C:/g1src`, `C:/p1src`, `C:/Users/Itama/ct1s`,
  `C:/Users/Itama/PycharmProjects/fd_variable_severity_v1_bf1e045f_snapshot`) are removed; the
  protected refs remain
  ([`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup)).
- **`C:/Users/Itama/PycharmProjects/flat-baseline` is intentionally retained** as a safety
  exception, not an incomplete Graph-RL cleanup: it carries protected branch `flat-final`, and its
  ignored files hold about 5.2 GB of flat-RL training outputs that are not archived under
  `C:\gra\` and whose deletion or move was not authorized
  ([`environments_cleanup.md` §4.6](docs/workflows/environments_cleanup.md#46-local-worktrees)).
- **No scientific run is in progress or authorized. The confirmatory profile has not been used.**
- **The GENERALIZED-V2 action-representation design is decided (2026-09-16) and its
  implementation is in flight:** the semantic `k + 2` action representation
  `semantic_k_plus_2_logmeanexp_v1` plus observational per-transition credit instrumentation
  (`train_credit_diagnostics.jsonl`), as one Grade-A code + contract task
  ([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-16). It authorizes
  no training or evaluation.
- **Closed:** Phase A (fixed cell, FD-BASELINE-v1); the FD-VARIABLE-SEVERITY-v1 actor-only
  baseline; the Phase-B CTDE implementation; GENERALIZED-V1 Tasks 1–5, early stopping and the
  per-wake diagnostics; the deterministic-P1 backend and the certified-FD physical-state repair;
  the GENERALIZED-V1 R1 measurement; the documentation restructure (PR #63); the GENERALIZED-V2
  development closure (PR #66); the V2 benchmark-preflight provenance review (PR #68); the local
  artifact archive and Git / worktree cleanup (2026-09-15).

## 2. Active owner and task

| Item | State |
|---|---|
| Writable repository task | **sole owner:** the V2 semantic-action + credit-instrumentation implementation (branch `task/v2-semantic-action-credit-instrumentation`, draft PR #70 to `main`) until it is reviewed and integrated |
| Reviewer | GPT orchestrator (read-only; exact-candidate review) |
| Evidence and review PRs | **none open** — #61, #62, #64, #65 and #67 are closed without merge and their branches deleted |
| Implementation candidates | the draft PR of the branch above — unreviewed; no merge authorized |
| Scientific runs in progress | none; none authorized |

## 3. Candidates and PRs

The only open PR is the implementation task's draft PR (branch
`task/v2-semantic-action-credit-instrumentation`). Resolve live PR state and its exact head on
GitHub.

| PR | Branch | Final state (2026-09-15) |
|---|---|---|
| #59, #60, #63, #66, #68 | task and documentation branches | merged; branches deleted after ancestry verification |
| #61, #62, #64 | `evidence/…` | closed without merge; branches deleted; conclusions in [measurements §7–§8](docs/history/measurements.md#7-generalized-v2-development-r1-arms); sources archived |
| #65 | `review/v2-wake-pair-diagnostics` | closed without merge; branch deleted; conclusions in [measurements §8.6](docs/history/measurements.md#86-matched-immediate-fd-wake-analysis) |
| #67 | `review/v2-benchmark-preflight-provenance` | closed without merge; branch deleted; conclusions in [measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) |

Heads, gates and archive paths:
[`environments_cleanup.md` §4.3–§4.5](docs/workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs).

## 4. Runs and evidence — current references

**Local archive:** machine-readable index `C:\gra\metadata\ARTIFACT_INDEX.jsonl` (authoritative)
and human-readable projection `C:\gra\metadata\ARTIFACT_INDEX.md`; identities in
[measurements §9](docs/history/measurements.md#9-local-artifact-archive-closure).

| Measurement | Measured code SHA | Verdict, with its provenance | Record |
|---|---|---|---|
| Phase-A long baseline (`fixed_cell_v1`, FD-BASELINE-v1, actor-only) | `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` | `APPROVE — VALID MEASUREMENT` (independent review) | [measurements](docs/history/measurements.md) |
| FD-VARIABLE-SEVERITY-v1 actor-only baseline | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | `APPROVE — VALID MEASUREMENT`; primary finding negative | [measurements](docs/history/measurements.md) |
| GENERALIZED-V1 R1 (actor-only, legacy backend) | `4af6c5aa5dd28072692bfda63282964b55010aae` | `APPROVE — VALID MEASUREMENT`; primary FD finding negative; comparator `manifest_id 0e15f007ef176bf977f8b93bb91289f48c16f25ee9eee282ffd1a89477f6fc0d` | [measurements](docs/history/measurements.md) |
| Fresh deterministic-P1 arm (GENERALIZED-V1 population, `p1_milp_v1`) | `ae1941035991df4719df212c4b5dd07db89aee4a` | accepted as valid; negative primary MILD-vs-SEVERE result; **not a clean causal comparator to R1** | [measurements §6](docs/history/measurements.md#6-the-p1-arms) |
| Earlier P1 arm | `8f0d250cd9f96e6b8bce635065701dc47a5ee87e` — recovered from the archived arm's own artifacts | **`ABORTED / DO NOT RESUME`** — not a measurement | [measurements §6.1](docs/history/measurements.md#61-the-aborted-arm), [§9.3](docs/history/measurements.md#93-recovered-local-identities) |
| GENERALIZED-V2 benchmark manifest (external, not committed; archived) | producer-recorded exact code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`, producer-recorded clean — not an external attestation | `manifest_id ef17a68a…46ea8`, file SHA-256 `dd72afc9…a103`, consumed by all five V2 runs below; manifest identity and producer provenance reviewed (PR #67 @ `7f56338c…`, now closed); **historical authorization not proven**; exact argv unknown | [measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) |
| GENERALIZED-V2 development R1 — actor-only and CTDE | `ae42cb01677f94868b2873008d87be677e31f0c8` | CTDE: prior GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT` as recorded in PR #62; actor-only: no standalone verdict string recorded. Both are inputs to the closed development interpretation | [measurements §7](docs/history/measurements.md#7-generalized-v2-development-r1-arms), [§8](docs/history/measurements.md#8-generalized-v2-development-closure) |
| GENERALIZED-V2 CTDE diagnostics — `smallbatch`, `largebatch`, `fd80` | `ae42cb01677f94868b2873008d87be677e31f0c8` | **development diagnostic runs, not confirmatory**; accounting `PASS` | [measurements §8](docs/history/measurements.md#8-generalized-v2-development-closure) |
| GENERALIZED-V2 matched immediate-FD wake analysis (PR #65) | — (read-only extraction) | extraction integrity `APPROVE`; not a measurement | [measurements §8.6](docs/history/measurements.md#86-matched-immediate-fd-wake-analysis) |

**Current research interpretation (development only; numbers in
[measurements §8](docs/history/measurements.md#8-generalized-v2-development-closure)):**

- **No stable severity-conditioned selected-action response was learned.** All five final
  primary endpoints are effectively zero, and 0 of 1600 matched wake pairs differ in selected
  action. These are repeated measures over the same frozen worlds.
- **Actor-visible severity information is demonstrably present:** a large post-damage `fuel_norm`
  difference and consistent `reachable_by_ego` `1→0` flips.
- **Simple batch-size and FD-exposure explanations are strongly weakened.**
- **The measured action representation had a demonstrated joint-cell versus semantic-aggregate
  mismatch.** This is a direct structural finding whose causal role is unresolved; the approved
  semantic representation removes the alias geometry, but no run has measured it.
- **Route-relative representation quality and immediate-FD credit remain open.** Distance is
  near-totally clipped at these wakes, reachability is still the round-trip placeholder (both
  deliberately unchanged by the in-flight task), and no per-transition advantage was recorded
  for any preserved run.
- **CTDE did not establish a benefit for the target behaviour.**

**Standing interpretation rules:** R1 and the fresh P1 arm are distinct repository and
population measurements with no causal solver-quality inference; reviewed measurements are reused
as recorded by default; the aborted P1 arm stays `DO NOT RESUME`; the old fixed-cell CTDE
measurement stays out of scope unless the user explicitly asks
([`experiments.md` §4.3–§4.4](docs/workflows/experiments.md#43-interpretation-rules)).

## 5. Concrete unresolved next actions

**Next action: GPT exact-candidate review** of the implementation draft PR's full head SHA
(`GPT_GITHUB`; review fixes are append-only commits on the same branch and PR). **Not training.**
After an approved and authorized merge, a next development run would still need its own explicit
authorized bounded plan ([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan));
none exists.

**Recorded, not scheduled and not authorized:**

- a separate code task to correct the V2 `generalized.cardinality_sampler` summary label in
  `graph_train._generalized_summary`, without touching archived artifacts;
- an open engineering caveat: a playback-export failure still routes as an ordinary `run`-stage
  episode failure ([measurements](docs/history/measurements.md#2-measurement-records));
- `flat-baseline` is intentionally retained as protected historical state and is **not a blocker**
  for the next Graph-RL research task; any future archival or removal decision for its ignored
  flat-RL outputs requires its own explicit task
  ([`environments_cleanup.md` §4.6](docs/workflows/environments_cleanup.md#46-local-worktrees)).

## 6. Blocked or unauthorized now

- modifying, deleting, pruning, regenerating or rewriting anything under `C:\gra\`, or relocating
  it except under
  [`environments_cleanup.md` §4.4](docs/workflows/environments_cleanup.md#44-preserved-run-directories-and-external-artifacts);
  moving or deleting protected refs;
- any change outside the in-flight task's approved scope — in particular reachability, distance
  normalization / clipping or other observation features, the reward, FD physics, the V2
  population / benchmark / profiles, PPO hyperparameters, batch size, training budget, FD
  exposure, CTDE architecture / features, early stopping, BLADE and the solvers; checkpoint
  migration, warm-start conversion or resume;
- **without an authorized bounded plan that names it**
  ([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)):
  any V2 training, evaluation or benchmark preflight; re-running or extending any preserved run;
  further batch, exposure or training-length tuning of the same knobs; any R1 or P1 rerun,
  repair or extension; the five full cluster runs;
- **any use of the confirmatory profile.** A future confirmatory plan must also name manifest
  `ef17a68a…46ea8`, decide explicitly whether to adopt it, preserve the development /
  confirmatory separation and acknowledge the historical authorization-record gap
  ([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-15);
- resuming, repairing or extending the aborted P1 arm (`DO NOT RESUME`);
- early stopping under `generalized_v2` (refused by code) and V2 cardinality above `A = 6`
  (`A = 8` / `A = 10` are engineering evidence only);
- `p(destroy) < 1` (deferred) and checkpoint resume (out of scope);
- reviewing, reclassifying or comparing the old fixed-cell CTDE measurement or the unclassified
  `ct1` artifact unless the user asks.

## 7. Protected refs, preserved evidence and known gaps

**Registry:** [`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup).
Protected refs: `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final` and tag `pre-cleanup`.
Remaining local worktrees: the main checkout and `C:/Users/Itama/PycharmProjects/flat-baseline`
(branch `flat-final`). Preserved artifacts: the `C:\gra\` archive and its index.

**Known gaps in this snapshot:**

- No per-immediate-FD-transition advantage diagnostic exists for any preserved V2 run; the
  instrumentation that would record one is unmerged code, and every preserved run and checkpoint
  uses the historical node-indexed action representation.
- The PR #65 reachability comparison is positional; task-node identity across paired members is
  not independently certified.
- The V2 benchmark preflight's exact original argv and its historical research-authorization
  record are absent; its historical prior review is not proven
  ([measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review)).
- The `ct1` artifact's identity remains unclassified (`UNKNOWN / possible old fixed-cell CTDE
  arm`); the old fixed-cell CTDE measurement's identity is still not recorded
  ([measurements §9.3](docs/history/measurements.md#93-recovered-local-identities)).
- The derived PR #65 extraction package (wake rows and matched pairs) was not copied into the
  local archive; the five runs it read are archived.
- The archive index's `evidence_ref_status` strings are archive-time text and still describe the
  closed PRs as open; the index is deliberately not rewritten.
