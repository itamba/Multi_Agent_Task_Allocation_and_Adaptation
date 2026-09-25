# Graph RL project handoff — current snapshot

> **Status: current state only — not a contract and not a history.** This snapshot describes the
> repository after the authorized recent-research archival and Git cleanup was completed and its
> closure record was integrated: PR #77 is merged as
> `7273be2ab563cc70de82651ec62bfa719f4f2758`, and its branch was deleted after merge
> verification. **The intended durable branch state is `main` plus the three protected historical
> branches only** (§7); a transient documentation-maintenance branch and PR may exist briefly
> while this file itself is corrected, and leaves nothing behind once integrated. **Since
> 2026-09-25 the active task branch `task/actor-credit-update-diagnostic-r1` and its draft PR also
> exist** (§2); the earlier mission-fuel-slack task is reviewed and integrated (PR #79).
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
  Its evidence PR #71 (`0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`) was closed without merge on
  2026-09-19, after the run was archived; the verdict documentation (PR #72) is merged as
  `8056266cff89f677911462b29970346bed0a57c1`.
- **The semantic-action CTDE development R1 is complete and reviewed (retrospectively, 2026-09-19):**
  `APPROVE — VALID DEVELOPMENT MEASUREMENT`, run
  `graph_rl_v2_semantic_action_ctde_dev_r1_seed3000000_8056266` at measured code SHA
  `8056266cff89f677911462b29970346bed0a57c1`, symmetric central state
  ([measurements §16](docs/history/measurements.md#16-generalized-v2-semantic-action-ctde-development-r1--retrospective-review-closure)).
  No comparable semantic actor-only transient was observed at any of its 16 evaluation rounds; no
  CTDE benefit is established. Its evidence PR #73 (`ad9b545034670a7c7a9d8ff98012d56c0be07f46`)
  was closed without merge on 2026-09-19, after the run was archived; the record is merged with
  PR #76.
- **The CTDE actor-gradient diagnostic instrumentation is integrated** (PR #74, approved
  implementation head `6ed964a1abd09de2130aee3d0d314c8f32165056`, merged as
  `adc213670ce4844a7cf60943ecf50150318e40b1`): an opt-in,
  observational epoch-0 decomposition (`train_actor_gradient_diagnostics.jsonl`,
  [artifacts and metrics §5.2](docs/contracts/artifacts_metrics.md#52-ctde-actor-gradient-diagnostics))
  with a local first-order SEVERE-minus-MILD ABORT separation pressure per gradient component.
- **Two 150-update CTDE actor-gradient development diagnostics are complete and reviewed** at
  measured SHA `6ed964a1…`: the `p = 0.5` diagnostic baseline and the FD100 intervention
  (`fuel_damage_probability 0.5 → 1.0`), both
  `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` (GPT, 2026-09-17)
  ([measurements §11](docs/history/measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics)).
  Their compact Git index is `research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/`
  (merged with PR #74); the original run directories are authoritative and are archived under
  `C:\gra\diagnostics\` (2026-09-19).
- **CTDE role-only acting-ego critic conditioning is integrated** (PR #75, final candidate
  reviewed at `b3a2350f437afee1c66d4fd5a5efa7cb7a353787`, merged as
  `ed33b7e24a652fa00b13c708517012f8b3302496`;
  [`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-19): each central
  decision capture names its acting ego, whose live node takes the encoder's existing EGO role, so
  the critic values `V(global_state, acting_ego)` through the unchanged mean-pool readout, with
  default `gae_lambda` `0.95`
  ([policy and CTDE §4](docs/contracts/policy_ctde.md#4-phase-b-ctde)). The source equals the
  implementation GPT-approved and measured at `68055e39768d5fa601e5960a9f08823b9e65c08f`. CTDE
  measurements before §12 of the measurement history used the earlier symmetric central state.
- **The acting-ego / critic-locality development diagnostics are complete and reviewed**, all
  `APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT` (GPT, 2026-09-19), evidence packages on
  `main` (merged with PR #75):
  - role-only (`…_acting_ego_ctde_fd100_r1_seed3000000_68055e3`,
    [measurements §12](docs/history/measurements.md#12-generalized-v2-role-only-acting-ego-ctde-development-diagnostic)):
    **improved critic / credit locality, not held-out acquisition or retention**;
  - explicit `[mean pool ; acting-ego embedding]` readout, measured at
    `1a1e0c953c54e9d3f46c871158d5ab6bdd881f24`
    ([§13](docs/history/measurements.md#13-generalized-v2-explicit-acting-ego-readout-ctde-development-diagnostic-and-owner-transition-audit)):
    **no locality or behavioural improvement over role-only; the mean-pool dilution hypothesis is
    not supported**. The readout is **historical / superseded and absent from the final source
    tree**; the §13 package also preserves the read-only owner-transition audit;
  - role-only `gae_lambda = 1.0`, measured at `68055e3…`
    ([§14](docs/history/measurements.md#14-generalized-v2-role-only-acting-ego-ctde-gae_lambda--10-development-diagnostic)):
    exact telescoping removed intermediate (including cross-owner) bootstrap from the actor
    advantage, yet held-out separation stayed effectively zero — a **reviewed negative
    development measurement, not a code or default-configuration change**.
- **The V2 benchmark-preflight provenance evidence is reviewed and durable**
  ([measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review)).
- **Closure and cleanup of the historical V2 chapter are complete** (2026-09-15): the local
  artifacts are organized and indexed under `C:\gra\`; the temporary evidence and review PRs of
  that chapter are closed and their branches removed; the retired merged branches are removed; the
  four Graph-RL execution / source worktrees targeted by the cleanup (`C:/g1src`, `C:/p1src`,
  `C:/Users/Itama/ct1s`, `C:/Users/Itama/PycharmProjects/fd_variable_severity_v1_bf1e045f_snapshot`)
  are removed; the protected refs remain
  ([`environments_cleanup.md` §4](docs/workflows/environments_cleanup.md#4-authorized-cleanup)).
- **The recent-research archival and Git cleanup is complete** (2026-09-19, explicitly authorized):
  - **Archive.** The two semantic-action R1 run directories and the five CTDE development
    diagnostic run directories were archived into `C:\gra\` by identity-verified same-volume
    rename. The archive index now has 35 rows
    ([measurements §9.5](docs/history/measurements.md#95-archival-extension-of-2026-09-19)).
  - **Evidence PRs.** #71 and #73 were closed without merge at their exact heads, and their
    branches were deleted.
  - **Merged branches.** The PR #74 and PR #76 branches were deleted.
  - **Closure record integrated.** The cleanup record was merged as PR #77
    (`7273be2ab563cc70de82651ec62bfa719f4f2758`, from reviewed head
    `840d00e3e558f771cf6eb90e439ffacafa27a2ff`), and its branch
    `docs/recent-archive-cleanup-closure` was deleted on 2026-09-20 after its merge,
    exact tip and `main`-reachability were verified. **No research, evidence or task ref
    awaits cleanup.**
  - **Local scratch.** The authorized main-checkout scratch was removed; the protected refs are
    unchanged
    ([`environments_cleanup.md` §4.3–§4.7](docs/workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs)).
  - **Prior observation.** The λ = 1 source worktree `C:/grolelambda1` was already found absent
    and unregistered; that is an observation, not an action of this cleanup.
- **`C:/Users/Itama/PycharmProjects/flat-baseline` is intentionally retained** as a safety
  exception, not an incomplete Graph-RL cleanup: it carries protected branch `flat-final`, and its
  ignored files hold about 5.2 GB of flat-RL training outputs that are not archived under
  `C:\gra\` and whose deletion or move was not authorized
  ([`environments_cleanup.md` §4.6](docs/workflows/environments_cleanup.md#46-local-worktrees)).
- **The actor mission-fuel-slack task is reviewed and integrated** (2026-09-25): one actor
  input, `mission_fuel_slack_norm` on the ego agent row (actor observation
  `actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1`,
  [policy and CTDE §1](docs/contracts/policy_ctde.md#1-graph-observation-stage-3)), and one
  375-update actor-only development run at measured SHA
  `3bc944119da08af8e25268c9ee83fc63a8d1e533`. After a first exact-head review of `ea60ce3…`
  (CHANGES_REQUESTED, 2026-09-24) and additive fixes, the implementation was **APPROVED at exact
  head `26f8bff97b1cbbca64d18626625d2321545080af`** and the run **reviewed as valid negative
  development evidence**; PR #79 was merged as `bcb1746fbc677b3109f73b36699ac3b3780c32a4`
  (tree equal to the approved head's) and its remote branch is deleted
  ([measurements §17](docs/history/measurements.md#17-generalized-v2-actor-mission-fuel-slack-development-r1--reviewed)).
  The measured source worktree `C:/gms1src` and the original run directory are retained.
- **ACTIVE (2026-09-25): the actor-only credit-to-update diagnostic R1** — owned by CC on branch
  `task/actor-credit-update-diagnostic-r1` (one draft PR; resolve its number and exact head on
  GitHub). It adds opt-in observational every-epoch instrumentation of the actor-only
  `PPOUpdater` (`train_actor_step_diagnostics.jsonl`,
  [artifacts and metrics §5.3](docs/contracts/artifacts_metrics.md#53-actor-only-step-diagnostics))
  and executes exactly ONE fresh 100-update actor-only `generalized_v2` development diagnostic,
  compared with the first 100 updates and rounds 0 / 25 / 50 / 75 / 100 of the mission-slack run.
  Plan: `research_evidence/generalized_v2/actor_credit_update_diag_r1/authorized_plan.json`.
  **Task-specific workflow exception** (user-requested;
  [`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-25): no intermediate GPT
  review stop for THIS task only. **Implementation and run are UNREVIEWED**; PR #79's approval is
  not inherited; no merge is authorized. **The run has EXECUTED (2026-09-25)** at measured SHA
  `644883b89c208255f5808432462be64be5d7589f`. GPT's exact-head review of `5f5e22a…` was
  **CHANGES_REQUESTED** (2026-09-25; protocol deviations D1 stop-rule narrowing and D2 offline
  checkpoint loading recorded, prefix consistency UNRESOLVED); the fixes are additive commits on
  PR #80, which awaits one exact-head re-review
  ([measurements §18](docs/history/measurements.md#18-generalized-v2-actor-only-credit-to-update-diagnostic-r1--executed-unreviewed)).
- Earlier single-run authorizations are spent: both actor-gradient diagnostics (further tuning
  stopped, [`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-17) and the
  three acting-ego diagnostics (role-only, explicit readout, role-only λ = 1). **The confirmatory
  profile has not been used.**
- **Closed:** Phase A (fixed cell, FD-BASELINE-v1); the FD-VARIABLE-SEVERITY-v1 actor-only
  baseline; the Phase-B CTDE implementation; GENERALIZED-V1 Tasks 1–5, early stopping and the
  per-wake diagnostics; the deterministic-P1 backend and the certified-FD physical-state repair;
  the GENERALIZED-V1 R1 measurement; the documentation restructure (PR #63); the GENERALIZED-V2
  development closure (PR #66); the V2 benchmark-preflight provenance review (PR #68); the local
  artifact archive and Git / worktree cleanup (2026-09-15); the semantic-action + credit
  implementation (PR #70); the semantic-action actor-only development R1 review (2026-09-16) and
  its verdict documentation (PR #72); the `p = 0.5` and FD100 CTDE actor-gradient development
  diagnostics and their review (2026-09-17); the acting-ego / critic-locality development
  diagnostics — role-only, explicit readout, owner-transition audit, λ = 1 — and their review
  (2026-09-19; [measurements §15](docs/history/measurements.md#15-acting-ego--critic-locality-development-investigation--closing-interpretation));
  the integration of PR #75 (2026-09-19); the retrospective review of the semantic-action CTDE
  development R1 (2026-09-19; [measurements §16](docs/history/measurements.md#16-generalized-v2-semantic-action-ctde-development-r1--retrospective-review-closure))
  and its record alignment (PR #76, merged as `67cd12a438452aadf61f312e68484f26dfa1e7c7`); the
  recent-research archival and Git cleanup (2026-09-19) and its closure record (PR #77, merged as
  `7273be2ab563cc70de82651ec62bfa719f4f2758`, branch deleted 2026-09-20); the actor
  mission-fuel-slack implementation and run, their review and integration (PR #79, merged as
  `bcb1746fbc677b3109f73b36699ac3b3780c32a4`, 2026-09-25).

## 2. Active owner and task

| Item | State |
|---|---|
| Writable repository task | **`task/actor-credit-update-diagnostic-r1`** (CC; one draft PR — resolve on GitHub): the actor-only step instrumentation, its tests and contracts, ONE authorized 100-update development diagnostic with its evidence, and the PR #79 status closure. The only writable task |
| Reviewer | GPT orchestrator (read-only; exact-candidate review) |
| Evidence PRs | none open. #71 and #73 were closed without merge on 2026-09-19, at `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670` and `ad9b545034670a7c7a9d8ff98012d56c0be07f46`; their branches are deleted |
| Candidates | **draft PR #80 (`task/actor-credit-update-diagnostic-r1`) is the sole active candidate** — implementation plus evidence; review CHANGES_REQUESTED on 2026-09-25, fixes added, awaiting exact-head re-review. No merge is authorized. **Resolve GitHub for the live head** |
| Scientific runs in progress | none. The ONE authorized credit-to-update diagnostic `graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` completed on 2026-09-25 (exit code 0) and is EXECUTED / UNREVIEWED; its original directory is `C:\gruns\graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` (not archived); this authorization is spent. The mission-slack run `graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` completed on 2026-09-24 and is reviewed; its original directory `C:\gruns\graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` is not archived (any move needs its own authorization) and is read, never modified, by the active task |

## 3. Candidates and PRs

**One research candidate is active: the draft PR of `task/actor-credit-update-diagnostic-r1`**
(implementation + evidence, unreviewed, not for merge until GPT review and explicit user
authorization). The PRs below are historical states, not open work. **Resolve GitHub for live
exact heads.**

| PR | Branch | State |
|---|---|---|
| #79 | `task/actor-mission-fuel-slack-dev-r1` | merged as `bcb1746fbc677b3109f73b36699ac3b3780c32a4` (2026-09-24) from the approved head `26f8bff97b1cbbca64d18626625d2321545080af` (its second parent; tree `d3a0306e57b8390952a68c521c02a418252ae613`); first review of `ea60ce3…` CHANGES_REQUESTED; remote branch deleted, no local branch (verified 2026-09-25) |
| #77 | `docs/recent-archive-cleanup-closure` | merged as `7273be2ab563cc70de82651ec62bfa719f4f2758` (2026-09-19) from reviewed head `840d00e3e558f771cf6eb90e439ffacafa27a2ff` (its second parent); branch deleted after merge verification (2026-09-20) |
| #76 | `docs/recent-research-cleanup-closure` | merged as `67cd12a438452aadf61f312e68484f26dfa1e7c7` (2026-09-19) from reviewed head `c414ec12a708338690fbf6e1317d93f804b2b22f`; branch deleted after ancestry verification (2026-09-19) |
| #75 | `task/ctde-acting-ego-conditioning` | merged as `ed33b7e24a652fa00b13c708517012f8b3302496` (2026-09-19) from reviewed final head `b3a2350f437afee1c66d4fd5a5efa7cb7a353787`; branch absent; final code = role-only acting-ego critic conditioning, mean-pool readout, `gae_lambda = 0.95`; the explicit readout (measured at `1a1e0c9…`) survives only in history and evidence; the role-only, explicit-readout and λ = 1 evidence packages (measurements §12–§14) are on `main` |
| #74 | `task/v2-ctde-gradient-pressure-diagnostics` | merged as `adc213670ce4844a7cf60943ecf50150318e40b1`; implementation approved at `6ed964a1abd09de2130aee3d0d314c8f32165056`, the measured SHA of both actor-gradient diagnostics; branch (tip `3e29a57dac54361c1a71f43f9487a6860fcae7c3`) deleted after ancestry verification (2026-09-19) |
| #73 | `evidence/generalized-v2-semantic-action-ctde-dev-r1` | **closed without merge** (2026-09-19) at `ad9b545034670a7c7a9d8ff98012d56c0be07f46`; branch deleted; GPT verdict `APPROVE — VALID DEVELOPMENT MEASUREMENT` (retrospective, 2026-09-19), conclusions in [measurements §16](docs/history/measurements.md#16-generalized-v2-semantic-action-ctde-development-r1--retrospective-review-closure); source archived |
| #72 | `docs/v2-semantic-action-dev-r1-verdict` | merged as `8056266cff89f677911462b29970346bed0a57c1` |
| #71 | `evidence/generalized-v2-semantic-action-actor-only-dev-r1` | **closed without merge** (2026-09-19) at `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670`; branch deleted; conclusions in [measurements §10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1); source archived |
| #70 | `task/v2-semantic-action-credit-instrumentation` | merged as `d4e9f3721e6d151c00be3fe93c3d149df9d31965` |
| #59, #60, #63, #66, #68 | task and documentation branches | merged; branches deleted after ancestry verification (2026-09-15) |
| #61, #62, #64 | `evidence/…` | closed without merge; branches deleted; conclusions in [measurements §7–§8](docs/history/measurements.md#7-generalized-v2-development-r1-arms); sources archived |
| #65 | `review/v2-wake-pair-diagnostics` | closed without merge; branch deleted; conclusions in [measurements §8.6](docs/history/measurements.md#86-matched-immediate-fd-wake-analysis) |
| #67 | `review/v2-benchmark-preflight-provenance` | closed without merge; branch deleted; conclusions in [measurements §8.11](docs/history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) |

Heads, gates and archive paths of the 2026-09-15 and 2026-09-19 cleanups (no ref currently
awaits cleanup): [`environments_cleanup.md` §4.3–§4.5](docs/workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs).

## 4. Runs and evidence — current references

**Local archive:** machine-readable index `C:\gra\metadata\ARTIFACT_INDEX.jsonl` (authoritative)
and human-readable projection `C:\gra\metadata\ARTIFACT_INDEX.md`; identities in
[measurements §9](docs/history/measurements.md#9-local-artifact-archive-closure). It holds
**35 items**, including, since 2026-09-19, both semantic-action R1 run directories (under
`runs\development\`) and the two actor-gradient and three acting-ego diagnostic run directories
(under `diagnostics\`); their original → archive paths and identities are in
[measurements §9.5](docs/history/measurements.md#95-archival-extension-of-2026-09-19). The
original locations recorded in measurements §10.2–§16.2 are historical and no longer exist.

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
| **GENERALIZED-V2 semantic-action actor-only development R1** (`semantic_k_plus_2_logmeanexp_v1`) | `d4e9f3721e6d151c00be3fe93c3d149df9d31965` | **`APPROVE — VALID DEVELOPMENT MEASUREMENT`** (GPT, 2026-09-16) on evidence PR #71 @ `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670` (closed without merge); development only | [measurements §10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1) |
| **GENERALIZED-V2 semantic-action CTDE development R1** (`semantic_k_plus_2_logmeanexp_v1`, symmetric central state) | `8056266cff89f677911462b29970346bed0a57c1` | **`APPROVE — VALID DEVELOPMENT MEASUREMENT`** (GPT, retrospective, 2026-09-19) on evidence PR #73 @ `ad9b545034670a7c7a9d8ff98012d56c0be07f46` (closed without merge); development only | [measurements §16](docs/history/measurements.md#16-generalized-v2-semantic-action-ctde-development-r1--retrospective-review-closure) |
| **GENERALIZED-V2 semantic-action CTDE actor-gradient diagnostic — `p = 0.5`** (150 updates) | `6ed964a1abd09de2130aee3d0d314c8f32165056` (PR #74 approved head, unmerged when measured) | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** (GPT, 2026-09-17); development diagnostic only | [measurements §11](docs/history/measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics) |
| **GENERALIZED-V2 semantic-action CTDE actor-gradient diagnostic — FD100 intervention** (150 updates) | `6ed964a1abd09de2130aee3d0d314c8f32165056` (PR #74 approved head, unmerged when measured) | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** (GPT, 2026-09-17); development diagnostic only | [measurements §11](docs/history/measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics) |
| **Role-only acting-ego CTDE diagnostic** (FD100, 100 updates) | `68055e39768d5fa601e5960a9f08823b9e65c08f` | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** (GPT, 2026-09-19); evidence `research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/` (merged with PR #75) | [measurements §12](docs/history/measurements.md#12-generalized-v2-role-only-acting-ego-ctde-development-diagnostic) |
| **Explicit acting-ego readout CTDE diagnostic** (FD100, 100 updates; implementation retired) | `1a1e0c953c54e9d3f46c871158d5ab6bdd881f24` | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** (GPT, 2026-09-19); evidence `research_evidence/generalized_v2/explicit_ego_readout_ctde_fd100_r1/` (merged with PR #75), reviewed at `557e072b94884bef37ece82a064ad877a5a2f636`; includes the owner-transition audit | [measurements §13](docs/history/measurements.md#13-generalized-v2-explicit-acting-ego-readout-ctde-development-diagnostic-and-owner-transition-audit) |
| **Role-only `gae_lambda = 1.0` CTDE diagnostic** (FD100, 100 updates; negative, not the final configuration) | `68055e39768d5fa601e5960a9f08823b9e65c08f` | **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** (GPT, 2026-09-19); evidence `research_evidence/generalized_v2/role_only_ctde_lambda100_fd100_r1/` (merged with PR #75), reviewed at `1076208b68ee9abd76f159535c5f38fe98970ce5` | [measurements §14](docs/history/measurements.md#14-generalized-v2-role-only-acting-ego-ctde-gae_lambda--10-development-diagnostic) |
| **GENERALIZED-V2 actor mission-fuel-slack development R1** (`actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1`, actor-only, 375 updates) | `3bc944119da08af8e25268c9ee83fc63a8d1e533` (task branch, unmerged when measured) | **reviewed as valid negative development evidence** (GPT, 2026-09-25; implementation APPROVED at `26f8bff…`); evidence `research_evidence/generalized_v2/actor_mission_slack_dev_r1/` (merged with PR #79) | [measurements §17](docs/history/measurements.md#17-generalized-v2-actor-mission-fuel-slack-development-r1--reviewed) |
| **GENERALIZED-V2 actor-only credit-to-update diagnostic R1** (actor-only, mission-slack observation, 100 updates, `train_actor_step_diagnostics.jsonl`) | `644883b89c208255f5808432462be64be5d7589f` (task branch, unmerged when measured) | **EXECUTED; review CHANGES_REQUESTED, fixes added, awaiting re-review** — no verdict; original-trajectory consistency UNRESOLVED; evidence `research_evidence/generalized_v2/actor_credit_update_diag_r1/` on draft PR #80 | [measurements §18](docs/history/measurements.md#18-generalized-v2-actor-only-credit-to-update-diagnostic-r1--executed-unreviewed) |

**Current research interpretation (development only; numbers in
[measurements §8](docs/history/measurements.md#8-generalized-v2-development-closure),
[§10](docs/history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1),
[§11](docs/history/measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics) and
[§16](docs/history/measurements.md#16-generalized-v2-semantic-action-ctde-development-r1--retrospective-review-closure)):**

- **Under the historical action representation no stable severity-conditioned selected-action
  response was learned** (five runs, all final endpoints effectively zero; 0 of 1600 matched wake
  pairs differ in selected action), although actor-visible severity information is demonstrably
  present and simple batch-size and FD-exposure explanations are strongly weakened. CTDE did not
  establish a benefit there.
- **Under the semantic representation the actor learned strong severity-conditioned behaviour
  temporarily:** strong separation emerged by update 75 (macro `+0.296`, 4 / 20 directional
  switches), then became near-universal across the frozen development worlds at updates 100–150
  (19 / 20, 20 / 20 and 19 / 20 directional switches; macro up to `+0.650`), before collapsing by
  update 175 — behaviour the historical representation never showed.
  This is cross-version development evidence that the old action geometry was a **material
  bottleneck / contributor** — not proof that it was the only cause.
- **The separation was not retained:** it collapsed by update 175, and the final primary endpoint
  is effectively zero (`+0.000888`, 0 / 20 switches). **No stable final severity-conditioned
  behaviour is established**, and the representation change did not solve the objective.
- **Actor-only credit is episode / chain-level, not local:** with `gamma = 1` and a terminal-only
  reward, raw advantage is identical across every episode and ego chain, so the immediate-FD
  advantage restates the episode outcome; ABORT-vs-not gaps are non-counterfactual.
- **The remaining problem is primarily retention / optimization / credit stability.** No claim
  that CTDE will succeed.
- **Semantic CTDE R1 (symmetric central state, 375 updates; reviewed retrospectively):** 0
  directional switches in every one of 16 rounds (maximum macro ≈ `4.57e-5`, final ≈ `+1.46e-7`)
  — it **did not reproduce the strong actor-only transient in the measured evaluation trajectory**;
  the evidence does not show acquisition of that transient, rather than its loss. Its GAE credit
  varied within every multi-transition chain and episode, unlike actor-only, but `value_old` stayed
  nearly severity-insensitive and no severity-conditioned selected-action behaviour was observed. **No
  CTDE benefit is established**; not proof that CTDE is generally worse.
- **CTDE actor-gradient diagnostics (150 updates each, same measured SHA):**
  - at `p = 0.5`, no meaningful severity-conditioned behaviour appears; FD separation pressure is
    positive in only 56 / 120 defined updates and is itself mostly negative in the 75–99 window
    (14 / 18), so **simple non-FD cancellation is not supported as the dominant explanation**;
  - FD100 materially changes the learning signal (149 / 150 updates hold both severities; FD
    pressure positive 98 / 149; SEVERE ABORT − PLAN normalized advantage `+0.166` → `+0.392`) and
    held-out probability-level separation reaches `+0.0428` at update 75 (0 / 20 switches) — **FD
    exposure is a material contributor to acquisition** — but it collapses by update 100: **FD100
    does not solve retention**, and it is much weaker than the semantic actor-only transient;
  - more exposure does not make `value_old` severity-sensitive at the immediate-FD decision
    (within-update median `SEVERE − MILD` `-0.00056` / `-0.00007` against targets ≈ `-0.35`), and
    the median local TD residual stays tiny relative to GAE advantage: **critic / credit locality
    remains unresolved**;
  - non-FD interference is visible after FD100 acquisition (23 / 149 positive FD pressures turned
    negative in total) but is **not established as the sole or primary cause of collapse**; the
    entropy term flips the sign of separation pressure in 0 / 120 and 0 / 149 defined updates.
  - No causal attribution is made; further tuning is stopped.
- **Acting-ego / critic-locality diagnostics
  ([measurements §12–§15](docs/history/measurements.md#15-acting-ego--critic-locality-development-investigation--closing-interpretation)):**
  the symmetric critic was under-conditioned for the decision owner, and role-only conditioning is
  retained as the semantically correct decision state; but better critic conditioning / locality
  did **not** produce monotonic behavioural improvement — role-only improved locality with weaker
  acquisition than the symmetric critic, the explicit readout added nothing, and λ = 1 removed
  intermediate (including cross-owner) bootstrap from the actor advantage exactly with no
  acquisition. **Critic conditioning, mean-pool dilution and cross-owner GAE bootstrapping are not
  supported as a sufficient or primary explanation** of the remaining failure. Cross-owner
  bootstrapping is a real, descriptive structural feature of the global decision sequence, not an
  established cause of collapse.
- **Route-relative representation quality remains open:** distance is near-totally clipped at
  these wakes and reachability is still the round-trip placeholder; this measurement did not
  address either.
- **Actor mission-fuel-slack input (reviewed valid negative development evidence,
  [measurements §17](docs/history/measurements.md#17-generalized-v2-actor-mission-fuel-slack-development-r1--reviewed)):**
  the input separated MILD from SEVERE in sign at every recorded immediate-FD wake, yet the final
  endpoint is effectively zero, the semantic actor-only transient did not recur, and the measured
  immediate-FD outputs showed essentially no severity-conditioned separation — an
  observed-population result, not proof of zero feature dependence and not a causal attribution.

**Standing interpretation rules:** R1 and the fresh P1 arm are distinct repository and
population measurements with no causal solver-quality inference; the semantic-action R1 versus
historical actor-only R1 comparison is cross-version, not a contemporaneous control; reviewed
measurements are reused as recorded by default; the aborted P1 arm stays `DO NOT RESUME`; the old
fixed-cell CTDE measurement stays out of scope unless the user explicitly asks
([`experiments.md` §4.3–§4.4](docs/workflows/experiments.md#43-interpretation-rules)).

## 5. Concrete unresolved next actions

**Now: the active credit-to-update diagnostic task (§1, §2)** — implementation, engineering
gates, the ONE authorized 100-update development diagnostic, its evidence and ONE exact-head GPT
review of the final candidate. Its question: in actor-only training with the mission-slack input,
do immediate-FD transitions push toward a larger SEVERE-minus-MILD ABORT contrast, do other
transitions counteract that, and how does the real clipped-gradient Adam step change the contrast
on the same observations. **The run has executed; the first exact-head review was
CHANGES_REQUESTED and its fixes are added; the candidate awaits ONE exact-head re-review.** The
executing task's corrected reading (descriptive, not a verdict): local FD pressure is inconsistent
across many batches, actual updates do not consistently increase the batch contrast, late steps
carry a large linearization error, and the fixed-world evaluations show no post-update switches;
no cause is isolated. The same-seed prefix consistency is UNRESOLVED: the first detected training
output difference is at iteration 1 (magnitude 7.45e-9) with matching recorded input summaries,
and when or why parameters diverged is not established
([measurements §18](docs/history/measurements.md#18-generalized-v2-actor-only-credit-to-update-diagnostic-r1--executed-unreviewed)).
Nothing else is scheduled or authorized; any further step (review verdict, merge, a further arm
or run) needs its own decision
([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)).

**Research area the active task belongs to:** actor-side optimization /
stability, gradient-to-policy mapping and retention mechanics
([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-19;
[measurements §15](docs/history/measurements.md#15-acting-ego--critic-locality-development-investigation--closing-interpretation)).
**No actor-side mechanism has been identified.** Of the open questions of
[measurements §11.9](docs/history/measurements.md#119-unresolved-hypotheses-and-next-research-action)
— an investigation list, not a finding that any item is defective — question 2 (**critic
conditioning**) has been taken up by §12–§14 and is not supported as a sufficient or primary
explanation; these remain open:

1. **actor private-observation identifiability** — does the acting ego's immediate-FD private
   graph carry an informative, non-saturated MILD-vs-SEVERE signal (fuel normalization,
   distance / reachability clipping, other actor-visible features)? *The active task tests one
   intervention on this question (explicit mission fuel slack); reachability and distance
   clipping stay unchanged.*
2. **PPO mechanics** — clipping, normalized advantages, repeated epochs, Adam / gradient clipping as
   mechanisms that could acquire and then erase separation; *the active task measures the actual
   every-epoch Adam displacement and its first-order effect on the batch contrast*;
3. **gradient interaction** — why non-FD components sometimes oppose a healthy FD component after
   acquisition; *the active task decomposes the actor-only raw loss gradient by the same four
   groups at every epoch*.

No read-only audit of these is open; this list authorizes no run or code change.

**Known low-priority maintenance:** none open from PR #79 — the `graph_encoder` module
self-test's stale `isinstance(node, int)` assertion was corrected with it.

**Recorded, not scheduled and not authorized:**

- open research gaps not addressed by the semantic-action measurement: the `reachable_by_ego`
  round-trip placeholder and route-relative distance representation / clipping
  ([policy and CTDE §6](docs/contracts/policy_ctde.md#6-known-limitations-and-open-items));
- local causal credit at the immediate-FD decision: actor-only instrumentation cannot provide it
  (measurements §10.7); any change to reward or credit structure would need its own decision;
- undecided local items left untouched by the 2026-09-19 cleanup, each needing its own decision:
  `src/match_aou/rl.zip`, `legacy/run_capture.log`, `src/match_aou/rl/observation/rollouts/` and
  the now-empty directory `C:\gruns\`
  ([`environments_cleanup.md` §4.7](docs/workflows/environments_cleanup.md#47-cleanup-already-performed));
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

The active credit-to-update diagnostic task (§2) is authorized ONLY for what its plan names: the
opt-in observational actor-only step instrumentation, its tests and contracts, ONE 100-update
development diagnostic with its evidence, and the PR #79 status closure. Everything below stays
blocked for it too — in particular any second run, restart, resume, seed replication, extension,
tuning, reward / credit redesign, observation change, confirmatory evaluation, merge or cleanup.

- reopening closed evidence PRs #71 or #73, or deleting the undecided local items of §5, without
  explicit authorization;
- modifying, deleting, pruning, regenerating or rewriting anything under `C:\gra\` — including the
  seven run directories archived on 2026-09-19 — or relocating
  it except under
  [`environments_cleanup.md` §4.4](docs/workflows/environments_cleanup.md#44-preserved-run-directories-and-external-artifacts);
  moving or deleting protected refs;
- any code, test or configuration change, archive move, ref deletion or other cleanup in a
  documentation-maintenance task; any implementation
  not separately authorized — including re-introducing the explicit critic readout or changing
  the default `gae_lambda` — and in particular training semantics,
  action-conditioned gradient subgroups, actor-only gradient instrumentation beyond the active
  task's observational step diagnostic, reachability,
  distance normalization / clipping or other observation features, the reward or reward shaping,
  FD events or physics, the V2 population / benchmark / profiles, PPO hyperparameters (learning
  rate, lambda, entropy, clipping), batch size, stratified loss weighting, oversampling / replay,
  training budget, FD exposure, the action representation, CTDE architecture / critic features,
  early stopping, BLADE and the solvers; checkpoint migration, warm-start conversion or resume;
- **without an authorized bounded plan that names it**
  ([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)):
  any V2 training, evaluation or benchmark preflight — **including any further run using the
  actor-gradient diagnostic** (both diagnostic authorizations are spent); re-running or extending
  any preserved run, including both semantic-action R1 arms, both actor-gradient diagnostic runs
  and the three acting-ego diagnostic runs; any
  hyperparameter sweep; further batch, exposure (including beyond-FD100 oversampling / replay) or
  training-length tuning;
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
Remaining local worktrees: the main checkout, `C:/Users/Itama/PycharmProjects/flat-baseline`
(branch `flat-final`) and the mission-slack measured source `C:/gms1src` (detached at
`3bc944119da08af8e25268c9ee83fc63a8d1e533`, retained; its removal is not authorized) and the
active task's measured source `C:/gcud1src` (detached at
`644883b89c208255f5808432462be64be5d7589f`, retained; no cleanup authorized); `C:/grolelambda1` is absent and
unregistered
([`environments_cleanup.md` §4.6](docs/workflows/environments_cleanup.md#46-local-worktrees)).
Refs awaiting cleanup: none. **The intended durable remote branches are `main`,
`phase-a-baseline`, `pre-ctde-actor-only` and `flat-final`, with the protected tag
`pre-cleanup`** — no evidence, task or documentation cleanup branch is pending. A transient
documentation-maintenance branch may exist while its own PR is open and is deleted on integration
([§4.3, §4.5](docs/workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs)).
Preserved artifacts: the `C:\gra\` archive and its 35-row index. It includes both semantic-action
R1 run directories (their evidence packages survive through GitHub's `refs/pull/71/head` and
`refs/pull/73/head` after the PRs were closed; their conclusions are in measurements §10 and §16),
and the two actor-gradient and three acting-ego diagnostic run directories (authoritative). Those
five runs are indexed compactly by `semantic_ctde_grad_diag_r1/` (merged with PR #74) and by
`acting_ego_ctde_fd100_r1/`, `explicit_ego_readout_ctde_fd100_r1/` and
`role_only_ctde_lambda100_fd100_r1/` (merged with PR #75), all under
`research_evidence/generalized_v2/`. Archive paths:
[measurements §9.5](docs/history/measurements.md#95-archival-extension-of-2026-09-19).

**Known gaps in this snapshot:**

- Per-transition credit is recorded for the two semantic-action R1 arms, the two actor-gradient
  diagnostics and the three acting-ego diagnostics only; no preserved run under the historical
  representation has it, and under actor-only it is episode / chain-level, not local FD-action
  credit (measurements §10.7). `train_actor_gradient_diagnostics.jsonl` exists only for those five
  CTDE diagnostics, and decomposes PPO epoch 0 only. No preserved run before the active task
  carries actor-only gradients or actual parameter displacements
  (`train_actor_step_diagnostics.jsonl`).
- The diagnostics' record streams and checkpoints are not in Git; they are identified by SHA-256
  in the evidence manifests and held only in the `C:\gra\` archive.
- The two semantic-action R1 credit streams were renamed after their evidence commits
  (`train_credit_diagnostics_actor_only.jsonl`, `train_credit_diagnostics_CTDE.jsonl`, both
  recorded as `train_credit_diagnostics.jsonl`). The bytes are identical; the files were archived
  as found, and who renamed them is not recorded
  ([measurements §9.5](docs/history/measurements.md#95-archival-extension-of-2026-09-19)).
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
  closed PRs (#61, #62, #64, #67, and since 2026-09-19 #71 and #73) as open; the index is
  deliberately not rewritten.
