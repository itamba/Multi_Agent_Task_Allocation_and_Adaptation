# Graph RL project handoff — current snapshot

> **Status: current state only — not a contract and not a history.** This snapshot was refreshed
> on 2026-09-16, when the semantic-action actor-only development R1 verdict documentation task
> opened from live `main` `d4e9f3721e6d151c00be3fe93c3d149df9d31965`.
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
- **GENERALIZED-V2 development diagnosis under the historical action representation is closed.**
  Completed at measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`: actor-only R1, CTDE
  R1 and three CTDE development diagnostic arms (`smallbatch`, `largebatch`, `fd80`), plus a
  read-only matched MILD/SEVERE wake analysis over all five (§4,
  [`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-15).
- **The semantic `k + 2` action representation `semantic_k_plus_2_logmeanexp_v1` and the
  observational per-transition credit instrumentation (`train_credit_diagnostics.jsonl`) are
  integrated** (PR #70, merged as `d4e9f3721e6d151c00be3fe93c3d149df9d31965`).
- **The semantic-action actor-only development R1 is complete and reviewed:**
  `APPROVE — VALID DEVELOPMENT MEASUREMENT` at measured code SHA
  `d4e9f3721e6d151c00be3fe93c3d149df9d31965`
  ([measurements §10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1)).
  Its evidence is draft PR #71 at `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`, **read-only and not
  for merge**.
- **The next research direction is approved but not yet planned:** a semantic-action CTDE
  development arm ([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-16). It
  authorizes no execution until a bounded run plan is frozen.
- **The V2 benchmark-preflight provenance evidence is reviewed and durable**
  ([measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review)).
- **Closure and cleanup of the historical V2 chapter are complete** (2026-09-15): the local
  artifacts are organized and indexed under `C:\gra\`; the temporary evidence and review PRs of
  that chapter are closed and their branches removed; the retired merged branches are removed; the
  four Graph-RL execution / source worktrees targeted by the cleanup (`C:/g1src`, `C:/p1src`,
  `C:/Users/Itama/ct1s`, `C:/Users/Itama/PycharmProjects/fd_variable_severity_v1_bf1e045f_snapshot`)
  are removed; the protected refs remain
  ([`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup)).
- **`C:/Users/Itama/PycharmProjects/flat-baseline` is intentionally retained** as a safety
  exception, not an incomplete Graph-RL cleanup: it carries protected branch `flat-final`, and its
  ignored files hold about 5.2 GB of flat-RL training outputs that are not archived under
  `C:\gra\` and whose deletion or move was not authorized
  ([`environments_cleanup.md` §4.6](docs/workflows/environments_cleanup.md#46-local-worktrees)).
- **No scientific run is in progress or authorized. The confirmatory profile has not been used.**
- **Closed:** Phase A (fixed cell, FD-BASELINE-v1); the FD-VARIABLE-SEVERITY-v1 actor-only
  baseline; the Phase-B CTDE implementation; GENERALIZED-V1 Tasks 1–5, early stopping and the
  per-wake diagnostics; the deterministic-P1 backend and the certified-FD physical-state repair;
  the GENERALIZED-V1 R1 measurement; the documentation restructure (PR #63); the GENERALIZED-V2
  development closure (PR #66); the V2 benchmark-preflight provenance review (PR #68); the local
  artifact archive and Git / worktree cleanup (2026-09-15); the semantic-action + credit
  implementation (PR #70); the semantic-action actor-only development R1 review (2026-09-16).

## 2. Active owner and task

| Item | State |
|---|---|
| Writable repository task | **sole owner:** the semantic-action development R1 verdict documentation task (branch `docs/v2-semantic-action-dev-r1-verdict`, one draft PR to `main`) until it is reviewed and integrated |
| Reviewer | GPT orchestrator (read-only; exact-candidate review) |
| Evidence PRs | **#71** (`evidence/generalized-v2-semantic-action-actor-only-dev-r1`), draft, **read-only evidence at `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`, not for merge**; no other evidence or review PR is open |
| Candidates | the documentation task's draft PR — unreviewed; no merge authorized |
| Scientific runs in progress | none; none authorized |

## 3. Candidates and PRs

Open PRs: the documentation task's draft PR and evidence PR #71. Resolve live PR state and exact
heads on GitHub.

| PR | Branch | State |
|---|---|---|
| #71 | `evidence/generalized-v2-semantic-action-actor-only-dev-r1` | open draft; exact reviewed candidate `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`; **read-only, not for merge**; conclusions in [measurements §10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1); its lifecycle (close, archive) needs its own explicit authorization |
| #70 | `task/v2-semantic-action-credit-instrumentation` | merged as `d4e9f3721e6d151c00be3fe93c3d149df9d31965` |
| #59, #60, #63, #66, #68 | task and documentation branches | merged; branches deleted after ancestry verification (2026-09-15) |
| #61, #62, #64 | `evidence/…` | closed without merge; branches deleted; conclusions in [measurements §7–§8](docs/history/measurements.md#7-generalized-v2-development-r1-arms); sources archived |
| #65 | `review/v2-wake-pair-diagnostics` | closed without merge; branch deleted; conclusions in [measurements §8.6](docs/history/measurements.md#86-matched-immediate-fd-wake-analysis) |
| #67 | `review/v2-benchmark-preflight-provenance` | closed without merge; branch deleted; conclusions in [measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) |

Heads, gates and archive paths of the 2026-09-15 cleanup:
[`environments_cleanup.md` §4.3–§4.5](docs/workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs).

## 4. Runs and evidence — current references

**Local archive:** machine-readable index `C:\gra\metadata\ARTIFACT_INDEX.jsonl` (authoritative)
and human-readable projection `C:\gra\metadata\ARTIFACT_INDEX.md`; identities in
[measurements §9](docs/history/measurements.md#9-local-artifact-archive-closure). The
semantic-action development R1 run directory is **not yet indexed** there; its original location
and key hashes are in [measurements §10.2](docs/history/measurements.md#102-identity-and-evidence-provenance).

| Measurement | Measured code SHA | Verdict, with its provenance | Record |
|---|---|---|---|
| Phase-A long baseline (`fixed_cell_v1`, FD-BASELINE-v1, actor-only) | `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` | `APPROVE — VALID MEASUREMENT` (independent review) | [measurements](docs/history/measurements.md) |
| FD-VARIABLE-SEVERITY-v1 actor-only baseline | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | `APPROVE — VALID MEASUREMENT`; primary finding negative | [measurements](docs/history/measurements.md) |
| GENERALIZED-V1 R1 (actor-only, legacy backend) | `4af6c5aa5dd28072692bfda63282964b55010aae` | `APPROVE — VALID MEASUREMENT`; primary FD finding negative; comparator `manifest_id 0e15f007ef176bf977f8b93bb91289f48c16f25ee9eee282ffd1a89477f6fc0d` | [measurements](docs/history/measurements.md) |
| Fresh deterministic-P1 arm (GENERALIZED-V1 population, `p1_milp_v1`) | `ae1941035991df4719df212c4b5dd07db89aee4a` | accepted as valid; negative primary MILD-vs-SEVERE result; **not a clean causal comparator to R1** | [measurements §6](docs/history/measurements.md#6-the-p1-arms) |
| Earlier P1 arm | `8f0d250cd9f96e6b8bce635065701dc47a5ee87e` — recovered from the archived arm's own artifacts | **`ABORTED / DO NOT RESUME`** — not a measurement | [measurements §6.1](docs/history/measurements.md#61-the-aborted-arm), [§9.3](docs/history/measurements.md#93-recovered-local-identities) |
| GENERALIZED-V2 benchmark manifest (external, not committed; archived) | producer-recorded exact code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`, producer-recorded clean — not an external attestation | `manifest_id ef17a68a…46ea8`, file SHA-256 `dd72afc9…a103`, consumed by all six V2 runs below; manifest identity and producer provenance reviewed (PR #67 @ `7f56338c…`, now closed); **historical authorization not proven**; exact argv unknown | [measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) |
| GENERALIZED-V2 development R1 — actor-only and CTDE (historical node-indexed action representation) | `ae42cb01677f94868b2873008d87be677e31f0c8` | CTDE: prior GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT` as recorded in PR #62; actor-only: no standalone verdict string recorded. Both are inputs to the closed development interpretation | [measurements §7](docs/history/measurements.md#7-generalized-v2-development-r1-arms), [§8](docs/history/measurements.md#8-generalized-v2-development-closure) |
| GENERALIZED-V2 CTDE diagnostics — `smallbatch`, `largebatch`, `fd80` | `ae42cb01677f94868b2873008d87be677e31f0c8` | **development diagnostic runs, not confirmatory**; accounting `PASS` | [measurements §8](docs/history/measurements.md#8-generalized-v2-development-closure) |
| GENERALIZED-V2 matched immediate-FD wake analysis (PR #65) | — (read-only extraction) | extraction integrity `APPROVE`; not a measurement | [measurements §8.6](docs/history/measurements.md#86-matched-immediate-fd-wake-analysis) |
| **GENERALIZED-V2 semantic-action actor-only development R1** (`semantic_k_plus_2_logmeanexp_v1`) | `d4e9f3721e6d151c00be3fe93c3d149df9d31965` | **`APPROVE — VALID DEVELOPMENT MEASUREMENT`** (GPT, 2026-09-16) on evidence PR #71 @ `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`; development only | [measurements §10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1) |

**Current research interpretation (development only; numbers in
[measurements §8](docs/history/measurements.md#8-generalized-v2-development-closure) and
[§10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1)):**

- **Under the historical action representation no stable severity-conditioned selected-action
  response was learned** (five runs, all final endpoints effectively zero; 0 of 1600 matched wake
  pairs differ in selected action), although actor-visible severity information is demonstrably
  present and simple batch-size and FD-exposure explanations are strongly weakened. CTDE did not
  establish a benefit there.
- **Under the semantic representation the actor learned strong severity-conditioned behaviour
  temporarily:** near-universal deterministic separation at updates 75–150 (up to 20 / 20
  directional switches, macro up to `+0.650`), which the historical representation never showed.
  This is cross-version development evidence that the old action geometry was a **material
  bottleneck / contributor** — not proof that it was the only cause.
- **The separation was not retained:** it collapsed by update 175, and the final primary endpoint
  is effectively zero (`+0.000888`, 0 / 20 switches). **No stable final severity-conditioned
  behaviour is established**, and the representation change did not solve the objective.
- **Actor-only credit is episode / chain-level, not local:** with `gamma = 1` and a terminal-only
  reward, raw advantage is identical across every episode and ego chain, so the immediate-FD
  advantage restates the episode outcome; ABORT-vs-not gaps are non-counterfactual.
- **The remaining problem is primarily retention / optimization / credit stability.** The next
  focused question is whether a richer state-dependent training critic / GAE path can preserve the
  learned separation under the semantic representation. **No claim that CTDE will succeed.**
- **Route-relative representation quality remains open:** distance is near-totally clipped at
  these wakes and reachability is still the round-trip placeholder; this measurement did not
  address either.

**Standing interpretation rules:** R1 and the fresh P1 arm are distinct repository and
population measurements with no causal solver-quality inference; the semantic-action R1 versus
historical actor-only R1 comparison is cross-version, not a contemporaneous control; reviewed
measurements are reused as recorded by default; the aborted P1 arm stays `DO NOT RESUME`; the old
fixed-cell CTDE measurement stays out of scope unless the user explicitly asks
([`experiments.md` §4.3–§4.4](docs/workflows/experiments.md#43-interpretation-rules)).

## 5. Concrete unresolved next actions

**Now: GPT exact-candidate review** of the verdict documentation task's draft PR full head SHA
(`GPT_GITHUB`; review fixes are append-only commits on the same branch and PR). No merge is
authorized.

**Next unresolved action after this documentation candidate is integrated: freeze the bounded
semantic-action CTDE development-run plan** under the approved 2026-09-16 direction
([`decisions.md` §1](docs/history/decisions.md#1-decision-log)). The plan must satisfy
[`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan).
The CTDE run is **neither authorized nor launched**; no hyperparameter sweep is authorized; the
arm must not retune observations, reward, FD exposure, batch size or the action representation;
the confirmatory profile stays untouched.

**Recorded, not scheduled and not authorized:**

- open research gaps not addressed by the semantic-action measurement: the `reachable_by_ego`
  round-trip placeholder and route-relative distance representation / clipping
  ([policy and CTDE §6](docs/contracts/policy_ctde.md#6-known-limitations-and-open-items));
- local causal credit at the immediate-FD decision: actor-only instrumentation cannot provide it
  (measurements §10.7); any change to reward or credit structure would need its own decision;
- evidence PR #71's lifecycle (closing it, archiving the run directory under `C:\gra\` and
  indexing it) requires its own explicit authorization;
- a separate code task to correct the V2 `generalized.cardinality_sampler` summary label in
  `graph_train._generalized_summary`, without touching archived artifacts;
- the launcher `cmd` exit-code redirect defect (`echo %RC%> file` writes an empty file for a
  one-digit code) should be fixed before a launcher is reused; it is not a repository code defect;
- an open engineering caveat: a playback-export failure still routes as an ordinary `run`-stage
  episode failure ([measurements](docs/history/measurements.md#2-measurement-records));
- `flat-baseline` is intentionally retained as protected historical state and is **not a blocker**
  for the next Graph-RL research task; any future archival or removal decision for its ignored
  flat-RL outputs requires its own explicit task
  ([`environments_cleanup.md` §4.6](docs/workflows/environments_cleanup.md#46-local-worktrees)).

## 6. Blocked or unauthorized now

- modifying, merging or closing evidence PR #71, or modifying, deleting or relocating the
  semantic-action R1 run directory, without explicit authorization;
- modifying, deleting, pruning, regenerating or rewriting anything under `C:\gra\`, or relocating
  it except under
  [`environments_cleanup.md` §4.4](docs/workflows/environments_cleanup.md#44-preserved-run-directories-and-external-artifacts);
  moving or deleting protected refs;
- any change outside the documentation task's scope — in particular source code, tests, contracts,
  reachability, distance normalization / clipping or other observation features, the reward, FD
  physics, the V2 population / benchmark / profiles, PPO hyperparameters, batch size, training
  budget, FD exposure, the action representation, CTDE architecture / features, early stopping,
  BLADE and the solvers; checkpoint migration, warm-start conversion or resume;
- **without an authorized bounded plan that names it**
  ([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)):
  any V2 training, evaluation or benchmark preflight — **including the semantic-action CTDE
  development arm**; re-running or extending any preserved run, including the semantic-action
  actor-only R1; any hyperparameter sweep; further batch, exposure or training-length tuning;
  any R1 or P1 rerun, repair or extension; the five full cluster runs;
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
(branch `flat-final`). Preserved artifacts: the `C:\gra\` archive and its index; the
semantic-action R1 run directory at its original location
(`C:\Users\Itama\PycharmProjects\graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37`)
with its evidence package on PR #71.

**Known gaps in this snapshot:**

- Per-transition credit is recorded only for the semantic-action actor-only R1; no preserved run
  under the historical representation has it, and under actor-only it is episode / chain-level,
  not local FD-action credit (measurements §10.7).
- The semantic-action R1 run directory is not yet in the `C:\gra\` archive index.
- The semantic-action R1's `native_exit_code.txt` is empty (launcher defect); completion rests on
  the run's own summary, train records and final checkpoint.
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
