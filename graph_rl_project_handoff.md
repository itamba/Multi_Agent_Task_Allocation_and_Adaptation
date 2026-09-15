# Graph RL project handoff — current snapshot

> **Status: current state only — not a contract and not a history.** This snapshot was refreshed
> on 2026-09-15; at that refresh live `main` was `fcbedc2e2fff58aefdd4d2f1f766c39311a27587`.
> **GitHub is authoritative for live branch, PR and ownership state**: resolve live `main` and
> open PRs first ([`cc_review.md` §8](docs/workflows/cc_review.md#8-receiving-a-hand-off)).
> Contracts live in [`docs/contracts/`](docs/contracts/); everything before this snapshot lives in
> [`docs/history/`](docs/history/). Update this file in the same PR as any change to current
> state, replacing stale lines rather than stacking supersession notes.

## 1. Phase

- **GENERALIZED-V2 development diagnosis is closed.** Its population (PR #57), its frozen ten-cell
  benchmark and evaluation construct (PR #59) and the development closure record (PR #66) are
  merged. GENERALIZED-V1 remains a valid, preserved design.
- **Completed at measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`:** actor-only R1,
  CTDE R1 and three CTDE development diagnostic arms (`smallbatch`, `largebatch`, `fd80`), plus a
  read-only matched MILD/SEVERE wake analysis over all five. The development interpretation is
  closed (§4, [`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-15).
- **No scientific run is in progress. The confirmatory profile has not been used.**
- **Next research work belongs to a fresh orchestrator: GENERALIZED-V2 action-representation
  research design.** This handoff authorizes no implementation and no training.
- **Closed:** Phase A (fixed cell, FD-BASELINE-v1); the FD-VARIABLE-SEVERITY-v1 actor-only
  baseline; the Phase-B CTDE implementation; GENERALIZED-V1 Tasks 1–5, early stopping and the
  per-wake diagnostics; the deterministic-P1 backend and the certified-FD physical-state repair;
  the GENERALIZED-V1 R1 measurement; the documentation restructure (PR #63); the GENERALIZED-V2
  development closure (PR #66); the V2 benchmark-preflight provenance review (§4).

## 2. Active owner and task

| Item | State |
|---|---|
| Writable repository task | **none.** The next writable work is either the closure and cleanup work of §5 or a task the user starts with the next orchestrator; resolve any open writable task on GitHub |
| Reviewer | GPT orchestrator (read-only; exact-candidate review) |
| Evidence and review PRs | #61, #62, #64, #65, #67 — temporary, read-only, pending cleanup (§3) |
| Implementation candidates | none open |
| Scientific runs in progress | none |

## 3. Candidates and PRs

| PR | Branch | Head (verified 2026-09-15) | Status |
|---|---|---|---|
| #66 | `docs/generalized-v2-development-closure` | `157dd9fdbb92fe2835ef2268f485e58b92523a4d` | **merged** (merge commit `fcbedc2e2fff58aefdd4d2f1f766c39311a27587`) |
| #61 | `evidence/generalized-v2-actor-only-dev-r1` | `1375a881637a9a32721a1630f598adc571422a47` | draft; actor-only R1 evidence |
| #62 | `evidence/generalized-v2-ctde-dev-r1` | `b2bbe7a6235c3b9255106826cfb268af7e73f72d` | draft; CTDE R1 evidence |
| #64 | `evidence/ctde-overnight-diagnostics` | `90516d51beeddacded2b89a321d14291e411f2b0` | draft; diagnostic-arm evidence; the preservation candidate was approved |
| #65 | `review/v2-wake-pair-diagnostics` | `d565174e4ecc25eb60a4dd021e1a20025f55f07f` | draft; **temporary** matched-wake review package; extraction integrity `APPROVE` |
| #67 | `review/v2-benchmark-preflight-provenance` | `7f56338cde6aacfa59a52399b2378b98a62ea3aa` | draft; **temporary** provenance-review transport; `APPROVE — provenance package correctness / evidence review` at that exact head |

The approvals of #64, #65 and #67 are recorded from user-transferred packets; GitHub holds no
review or comment record on them. Under the artifact lifecycle decision
([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-15), **none of #61, #62,
#64, #65 or #67 is planned for merge into `main`.** They stay open until explicitly authorized
cleanup removes them; their review conclusions are durably recorded in
[`measurements.md`](docs/history/measurements.md).

## 4. Runs and evidence — current references

| Measurement | Measured code SHA | Verdict, with its provenance | Record |
|---|---|---|---|
| Phase-A long baseline (`fixed_cell_v1`, FD-BASELINE-v1, actor-only) | `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` | `APPROVE — VALID MEASUREMENT` (independent review) | [measurements](docs/history/measurements.md) |
| FD-VARIABLE-SEVERITY-v1 actor-only baseline | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | `APPROVE — VALID MEASUREMENT`; primary finding negative | [measurements](docs/history/measurements.md) |
| GENERALIZED-V1 R1 (actor-only, legacy backend) | `4af6c5aa5dd28072692bfda63282964b55010aae` | `APPROVE — VALID MEASUREMENT`; primary FD finding negative; comparator `manifest_id 0e15f007ef176bf977f8b93bb91289f48c16f25ee9eee282ffd1a89477f6fc0d` | [measurements](docs/history/measurements.md) |
| Fresh deterministic-P1 arm (GENERALIZED-V1 population, `p1_milp_v1`) | `ae1941035991df4719df212c4b5dd07db89aee4a` | accepted as valid; negative primary MILD-vs-SEVERE result; **not a clean causal comparator to R1** | [measurements §6](docs/history/measurements.md#6-the-p1-arms) |
| Earlier P1 arm | not recorded | **`ABORTED / DO NOT RESUME`** — not a measurement | [measurements §6](docs/history/measurements.md#6-the-p1-arms) |
| GENERALIZED-V2 benchmark manifest (external, not committed) | producer-recorded exact code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`, producer-recorded clean — not an external attestation | `manifest_id ef17a68a…46ea8`, file SHA-256 `dd72afc9…a103`, consumed by all five V2 runs below; manifest identity and producer provenance reviewed via temporary PR #67 @ `7f56338c…`; **historical authorization not proven**; exact argv unknown (invocation partial) | [measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) |
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
- **The action representation has a demonstrated joint-cell versus semantic-aggregate mismatch.**
  This is a direct structural finding whose causal role is unresolved.
- **Route-relative representation quality and immediate-FD credit remain open.** Distance is
  near-totally clipped at these wakes, reachability is still the round-trip placeholder, and no
  per-transition advantage was recorded.
- **CTDE did not establish a benefit for the target behaviour.**

**Standing interpretation rules:** R1 and the fresh P1 arm are distinct repository and
population measurements with no causal solver-quality inference; reviewed measurements are reused
as recorded by default; the aborted P1 arm stays `DO NOT RESUME`; the old fixed-cell CTDE
measurement stays out of scope unless the user explicitly asks
([`experiments.md` §4.3–§4.4](docs/workflows/experiments.md#43-interpretation-rules)).

## 5. Concrete unresolved next actions

**Closure work remaining before the next research orchestrator** — each step only as explicitly
authorized cleanup:

1. organize the preserved local artifacts under the approved archive plan and create the local
   artifact index;
2. verify byte and key identities after organization;
3. perform explicit, safe Git / PR / worktree cleanup of #61, #62, #64, #65 and #67 and their
   branches;
4. reconcile the protected-ref and cleanup registry
   ([`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup))
   and the final live state.

**Next research task (belongs to the next orchestrator): GENERALIZED-V2 action-representation
research design** — analysis and design only, including whether to add per-transition advantage
instrumentation before the next development run
([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-15). **No implementation
and no training is authorized.**

**Recorded, not scheduled and not authorized:**

- a separate code task to correct the V2 `generalized.cardinality_sampler` summary label in
  `graph_train._generalized_summary`, without touching archived artifacts;
- an open engineering caveat: a playback-export failure still routes as an ordinary `run`-stage
  episode failure ([measurements](docs/history/measurements.md#2-measurement-records)).

## 6. Blocked or unauthorized now

- merging #61, #62, #64, #65 or #67 (not planned);
- closing or deleting #61, #62, #64, #65 or #67 or their branches, moving or deleting protected
  refs, evidence refs or preserved run directories — only in explicitly authorized cleanup;
- implementing any action-representation, observation, reachability, credit or instrumentation
  change before the design task concludes and a change is authorized; changes to locked layers,
  BLADE, the solvers, PPO, CTDE, the reward or the observation / action contracts;
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
- reviewing or comparing the old fixed-cell CTDE measurement unless the user asks.

## 7. Protected refs, preserved evidence and known gaps

**Registry:** [`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup).
Evidence and review refs stay protected until authorized cleanup removes them.

**Known gaps in this snapshot:**

- The V2 benchmark preflight's exact original argv and its historical research-authorization
  record are absent; its historical prior review is not proven
  ([measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review)).
- No per-immediate-FD-transition advantage diagnostic exists for any V2 run.
- The PR #65 reachability comparison is positional; task-node identity across paired members is
  not independently certified.
- Artifact identities known only locally are not yet durably indexed until the archive cleanup
  (§5) finishes — among them the GENERALIZED-V1 R1 run directory, the fresh P1 arm's artifacts
  and the old fixed-cell CTDE measurement's identity, none of which is recorded in the
  repository.
