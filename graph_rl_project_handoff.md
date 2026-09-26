# Graph RL project handoff — current snapshot

> **Status: current state only — not a contract and not a history.** This snapshot describes the
> repository after the authorized recent-research archival and Git cleanup was completed and its
> closure record was integrated: PR #77 is merged as
> `7273be2ab563cc70de82651ec62bfa719f4f2758`, and its branch was deleted after merge
> verification. **The intended durable branch state is `main` plus the three protected historical
> branches only** (§7); a transient documentation-maintenance branch and PR may exist briefly
> while this file itself is corrected, and leaves nothing behind once integrated. The
> mission-fuel-slack task (PR #79) and the actor-only credit-to-update diagnostic (PR #80) are
> reviewed and integrated; **no research task is active**. §5.1 records the selected flow / credit
> research questions of 2026-09-26 — questions, not diagnosed causes or authorized work.
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
- **The actor-only credit-to-update diagnostic R1 is reviewed and integrated** (2026-09-25): opt-in
  observational every-epoch instrumentation of the actor-only `PPOUpdater`
  (`train_actor_step_diagnostics.jsonl`,
  [artifacts and metrics §5.3](docs/contracts/artifacts_metrics.md#53-actor-only-step-diagnostics))
  and ONE 100-update actor-only `generalized_v2` development diagnostic at measured SHA
  `644883b89c208255f5808432462be64be5d7589f`. After a first exact-head review of `5f5e22a…`
  (CHANGES_REQUESTED) and additive fixes, GPT **APPROVED exact head
  `eb6401167390513f4dc56a946ea76cc3b314607d`** and the user authorized the merge; PR #80 was merged
  as `2d5719370d17594297111350007dd0c8bedb54e1` and its remote branch is deleted
  ([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-25). **The approval is
  limited:** it covers the instrumentation and the preservation of **qualified descriptive
  development evidence** only. Consistency with the original run's trajectory stays
  **UNRESOLVED**, protocol deviations **D1** (stop-rule narrowing) and **D2** (offline checkpoint
  loading) stay recorded and are not retroactively authorized, and no causal mechanism is
  attributed; merging did not remove these limitations
  ([measurements §18](docs/history/measurements.md#18-generalized-v2-actor-only-credit-to-update-diagnostic-r1--executed-unreviewed),
  whose own status line predates the approval). The measured source worktree `C:/gcud1src` and
  the original run directory are retained; no local cleanup is recorded for this task.
- Earlier single-run authorizations are spent: both actor-gradient diagnostics (further tuning
  stopped, [`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-17) and the
  three acting-ego diagnostics (role-only, explicit readout, role-only λ = 1), the mission-slack
  run and the credit-to-update diagnostic. **The confirmatory profile has not been used.**
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
  `bcb1746fbc677b3109f73b36699ac3b3780c32a4`, 2026-09-25); the actor-only credit-to-update
  diagnostic, its qualified review and integration (PR #80, merged as
  `2d5719370d17594297111350007dd0c8bedb54e1`, 2026-09-25); the flow / credit / batch / PPO /
  evaluation research walkthrough (2026-09-26; its selected questions are §5.1, none started).

## 2. Active owner and task

| Item | State |
|---|---|
| Writable repository task | **no research task.** The only writable task is documentation-only: branch `docs/flow-credit-research-handoff` (CC; one draft PR — resolve on GitHub) records the §5.1 research questions and the PR #80 status closure in this file and [`decisions.md` §1](docs/history/decisions.md#1-decision-log); once integrated, none. The §5.1 questions are unassigned and unstarted |
| Reviewer | GPT orchestrator (read-only; exact-candidate review) |
| Evidence PRs | none open. #71 and #73 were closed without merge on 2026-09-19, at `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670` and `ad9b545034670a7c7a9d8ff98012d56c0be07f46`; their branches are deleted |
| Candidates | no research candidate. The documentation draft PR of the row above is the only candidate; no merge is authorized. **Resolve GitHub for the live head** |
| Scientific runs in progress | none. The credit-to-update diagnostic `graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` completed on 2026-09-25 (exit code 0) and is reviewed as qualified descriptive development evidence (§1); its original directory `C:\gruns\graph_rl_v2_actor_credit_update_diag_r1_seed3000000_644883b` is not archived; its authorization is spent. The mission-slack run `graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` completed on 2026-09-24 and is reviewed; its original directory `C:\gruns\graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441` is not archived. Moving either needs its own authorization |

## 3. Candidates and PRs

**No research candidate is active.** The only candidate is the documentation draft PR of §2. The
PRs below are historical states, not open work. **Resolve GitHub for live exact heads.**

| PR | Branch | State |
|---|---|---|
| #80 | `task/actor-credit-update-diagnostic-r1` | merged as `2d5719370d17594297111350007dd0c8bedb54e1` (2026-09-25) from the approved head `eb6401167390513f4dc56a946ea76cc3b314607d` (its second parent); first review of `5f5e22a…` CHANGES_REQUESTED; the approval is limited to the instrumentation and qualified descriptive development evidence (§1); remote branch deleted, no local branch (verified 2026-09-26) |
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
| **GENERALIZED-V2 actor-only credit-to-update diagnostic R1** (actor-only, mission-slack observation, 100 updates, `train_actor_step_diagnostics.jsonl`) | `644883b89c208255f5808432462be64be5d7589f` (task branch, unmerged when measured) | **APPROVED at exact head `eb64011…`** (GPT, 2026-09-25) as instrumentation plus **qualified descriptive development evidence** — original-trajectory consistency UNRESOLVED; deviations D1 / D2 recorded, not retroactively authorized; no causal attribution; evidence `research_evidence/generalized_v2/actor_credit_update_diag_r1/` (merged with PR #80) | [measurements §18](docs/history/measurements.md#18-generalized-v2-actor-only-credit-to-update-diagnostic-r1--executed-unreviewed) |

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
- **Actor-only credit-to-update diagnostic (qualified descriptive development evidence,
  [measurements §18](docs/history/measurements.md#18-generalized-v2-actor-only-credit-to-update-diagnostic-r1--executed-unreviewed)):**
  local FD pressure is inconsistent across many batches, actual updates do not consistently
  increase the batch contrast, late steps carry a large linearization error, and the fixed-world
  evaluations show no post-update switches. It isolates no cause — not the reward, `c`, critic,
  architecture or optimizer — and its same-seed prefix consistency is UNRESOLVED (first detected
  training-output difference at iteration 1, magnitude `7.45e-9`, with matching recorded input
  summaries).

**Standing interpretation rules:** R1 and the fresh P1 arm are distinct repository and
population measurements with no causal solver-quality inference; the semantic-action R1 versus
historical actor-only R1 comparison is cross-version, not a contemporaneous control; reviewed
measurements are reused as recorded by default; the aborted P1 arm stays `DO NOT RESUME`; the old
fixed-cell CTDE measurement stays out of scope unless the user explicitly asks
([`experiments.md` §4.3–§4.4](docs/workflows/experiments.md#43-interpretation-rules)).

## 5. Concrete unresolved next actions

**Now: nothing is scheduled or authorized.** No research task is active (§2). The §5.1 questions
are recorded for future read-only research chats; each is unassigned and unstarted, and any
further step — a research chat that changes the repository, an implementation or a run — needs
its own decision
([`experiments.md` §2](docs/workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)).
`gamma = 1` remains the user's design requirement.

**Earlier research area and its open list.** Since 2026-09-19 the area has been actor-side
optimization / stability, gradient-to-policy mapping and retention mechanics
([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-19;
[measurements §15](docs/history/measurements.md#15-acting-ego--critic-locality-development-investigation--closing-interpretation));
the mission-slack run (§17) and the credit-to-update diagnostic (§18) were its two measurements.
**No actor-side mechanism has been identified.** Of the open questions of
[measurements §11.9](docs/history/measurements.md#119-unresolved-hypotheses-and-next-research-action)
— an investigation list, not a finding that any item is defective — question 2 (**critic
conditioning**) has been taken up by §12–§14 and is not supported as a sufficient or primary
explanation; these remain open, now linked to §5.1:

1. **actor private-observation identifiability** — does the acting ego's immediate-FD private
   graph carry an informative, non-saturated MILD-vs-SEVERE signal (fuel normalization,
   distance / reachability clipping, other actor-visible features)? *The mission-slack run tested
   one intervention (reviewed negative development evidence); reachability and distance clipping
   are unchanged. Related representation questions: HEAD-01, OBS-01.*
2. **PPO mechanics** — clipping, normalized advantages, repeated epochs, Adam / gradient clipping as
   mechanisms that could acquire and then erase separation; *§18 measured the actual every-epoch
   Adam displacement and its first-order effect on the batch contrast (qualified). Its reading
   continues under BASELINE-01; the PPO / clipping walkthrough selected no standalone change.*
3. **gradient interaction** — why non-FD components sometimes oppose a healthy FD component after
   acquisition; *§18 decomposed the actor-only raw loss gradient by four groups at every epoch;
   reading under BASELINE-01.*

No read-only audit of these is open; this list authorizes no run or code change.

### 5.1 Flow / credit research questions (recorded 2026-09-26)

Selected in the user's flow / credit / batch / PPO / evaluation walkthrough, closed 2026-09-26
([`decisions.md` §1](docs/history/decisions.md#1-decision-log), 2026-09-26). **Research questions
— not diagnosed defects, approved fixes or active ownership.** Each future chat starts read-only.
No reward coefficient, architecture, estimator or experiment is selected. **Keep the representation
of a decision separate from the learning signal assigned to it:** better ego context or readout
cannot be assumed to repair credit, and different scalar credit cannot supply information the
actor lacks.

**Navigation order (not an experiment schedule):** REWARD-01, CREDIT-01, BASELINE-01 and
CRITIC-01 form the credit / critic research track. HEAD-01 and OBS-01 are deferred architecture
candidates; HEAD-01 is the stronger of the two, but it is not a diagnosed cause and not a
mandate to change the action head first.

**Reuse, do not repeat** ([measurements §12–§16](docs/history/measurements.md#15-acting-ego--critic-locality-development-investigation--closing-interpretation)):
role-only acting-ego conditioning improved critic locality without the desired behaviour; the
explicit ego readout added nothing and is retired; at `gamma = lambda = 1` the total CTDE
advantage telescoped to `R − V_t` and no separation emerged
([§14.4](docs/history/measurements.md#144-structural-telescoping-identity)). §18 isolates no cause
(qualified, §1). These results constrain simple explanations, not every redesign. One negative
development result is not a universal impossibility claim. Confirm the exact run and evidence
identity, including `run_config`, before reusing a measurement.

#### REWARD-01 — Does the scored objective express the intended mission / survival trade-off?

- **Status:** selected. A high-priority read-only formulation and calibration question. A larger
  `c` is a candidate, not a diagnosed fix.
- **Known** (`graph_reward._event_conditioned_breakdown`): `D = |U_ref| + eps`
  (`regret_epsilon = 1e-5`); `R = (U_prefix + U_post − U_ref − c · U_aircraft · n_lost) / D`;
  `U_ref = U_prefix + U_cont_ref`. `U_aircraft` is the maximum task utility over the prefix plus
  the continuation-ALLOCATED tasks (0 if empty; `graph_episode_setup._reference_aircraft_utility`).
  It is a target-utility valuation, not a measured aircraft value. The trainer default is
  `TrainConfig.aircraft_penalty_coeff = 2.25`, and `RewardConfig`'s standalone default is `0.0`.
  A run's `run_config` is authoritative. At `U_aircraft = 80`, `c = 2.25` prices one loss at 180
  utility units before normalization. `U_post` scores only continuation-allocated reference tasks;
  other confirmed targets are accounting-only (`unscored_completed_target_ids`). `U_prefix`
  cancels from the regret numerator (leaving `U_post − U_cont_ref`) but stays in `D`.
- **Question:** which objective is intended? The candidates are absolute utility minus an
  airframe cost; reference-relative utility (the current form, dimensionally coherent); or relative
  mission success minus the fleet-loss fraction `c_fleet · n_lost / N_initial`, with `N_initial`
  fixed and independent of losses. Coefficients are not interchangeable without calibration: with
  `U_ref = K·u` and `U_aircraft = u` the current loss term is ≈ `c · n_lost / K`, and dividing
  both sides by `u` adds no information. Is the reward-bearing target scope intended? Keep
  physical kills, confirmed completion and scoring distinct. The reference is frozen before the FD
  response, but its prefix, resources and continuation agents can depend on earlier actions, so
  removing it is not one global action-independent shift. A larger `c` changes the trade-off, not
  causal information or decision specificity. It may favour ABORT in both MILD and SEVERE. After
  standardization, a larger raw penalty need not produce a proportionally larger update.
- **First read-only deliverable:** state the intended objective in plain language, map its terms
  to exact-run reward and accounting records, and assess whether the loss cost is actually weak.
  For the same fixed reference, `E[R_PLAN − R_ABORT] = (ΔU − c · U_aircraft · ΔL) / D`. When
  `ΔL > 0` and `U_aircraft > 0`, the crossover is `c* = ΔU / (U_aircraft · ΔL)`. Ask whether
  evidence can identify a range that favours PLAN in recoverable MILD states and ABORT in SEVERE
  states; sampled-action means across different worlds cannot. Algebraic rescoring
  `R_i(c) = q_i − c · p_i` of complete saved batches can describe sensitivity on fixed trajectories
  only, not retrained behaviour. No sweep, replay or rollout. Examine empty or near-zero reference
  cases only where records contain them.
- **Anchors:** [reward and solvers §1–§2](docs/contracts/reward_solvers.md#2-event-conditioned-continuation-reference);
  `graph_episode_setup._reference_universe`, `_reference_aircraft_utility`,
  `build_continuation_reference`; `graph_reward.realized_utility`, `_event_conditioned_breakdown`;
  `graph_train.TrainConfig` and its `reward_config`.

#### CREDIT-01 — What should one sparse, multi-owner decision's learning signal estimate?

- **Status:** selected. Not a diagnosed ordering bug, and no replacement algorithm is chosen.
- **Known:** one `Transition` is one policy wake, not one tick. Execution continues between wakes
  and after the last one, and that physical tail enters `R`. After ABORT the `rtb_issued` latch
  removes the ego from Phase 1, so it makes no further decisions, but its physical return or loss
  still counts. No-comms, long gaps and `gamma = 1` do not make decisions independent, and a
  private observation is not automatically Markov. actor_only groups by ego
  (`EpisodeRecord.from_trajectory`), but at `gamma = 1` `_chain_returns` gives every transition
  the episode `R`. Reordering therefore cannot change its scalar credit
  ([measurements §10.7](docs/history/measurements.md#107-structural-credit-limitation)). CTDE's
  `_gae_pass` runs per episode over the global decision order. The next sample may belong to
  another ego or to the same tick. `gamma` and `lambda` advance per decision, not per elapsed
  tick, and `lambda = 0.95` still weights later TD residuals at `gamma = 1`. `lambda` is an
  estimator parameter, not a preference for earlier actions.
- **Question:** should a decision's signal estimate its association with team return, its
  improvement over the expected continuation at that decision, or its incremental value against a
  meaningful alternative? First state the sequential team objective: decision state, action,
  consequence interval, reward and terminal boundary. Only then choose per-ego or global credit.
- **CREDIT-02 (linked subquestion):** compare full-return plus conditional-baseline estimates, the
  current GAE, temporal redistribution and counterfactual team contribution for asynchronous,
  partially observed cooperative execution. Separate temporal from inter-agent credit, prediction
  error from estimator bias and variance, and objective preservation from numerical conditioning.
  The ENGAGE(A) → completion(A) idea is a motivation, not a selected rule:
  - an `(ego, target)` confirmation does not prove that ego's decision caused the kill;
  - ABORT can have value with no positive target event;
  - target-only rewards can change the team objective;
  - preserving the return is necessary for a redistribution, but not proof of correct credit.

  **Implementation fact:** actor_only consumes `EpisodeRecord.episode_reward` through
  `_chain_returns`, not the per-transition reward fields, so editing only `Transition.reward`
  leaves its signal unchanged. CTDE does read `Transition.reward` (`episode_rewards_sequence`; the
  terminal `R` sits on the last transition). A proposal must specify both the reward assignment and
  the estimator that consumes it. Literature starting points (no winner is inferred from
  success elsewhere): [COMA](https://arxiv.org/abs/1705.08926) (use the corrected v3),
  [Difference Rewards Policy Gradients](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1475.pdf),
  [RUDDER](https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html)
  ([author project](https://github.com/ml-jku/rudder)), and
  [baseline background](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).
- **RECORD-01 (conditional subcheck):** `EpisodeResult` and `graph_train._EpisodeOutcome` keep
  `ended` (`done` / `terminated` / `truncated`, where the tick cap gives `truncated`) and `ticks`.
  `EpisodeRecord` and `CTDEEpisodeRecord` do not carry the end reason, and `_gae_pass` uses
  `V_next = 0` at every episode's last decision. First check the exact runs' outcome end reasons
  and the intended horizon. A finite task horizon can legitimately use a zero bootstrap. If the
  relevant runs do not truncate unfinished tasks, close this explanation for them. Do not require
  `next_obs` / `done` fields, or samples for deterministic executor ticks, merely because they are
  conventional elsewhere. Preserve the alignment of pre-action graph, action and old log-prob, and
  keep same-tick sequence order.
- **First read-only deliverable:** trace one small existing episode across owners, wake kinds and
  the physical tail. Define successor and terminal meaning. Compare the exact actor_only and CTDE
  formulas with the candidate credit targets, stating information needs, bias and variance, and
  team-objective implications. Reuse existing identifiers, ticks and diagnostics. No replay, new
  instrumentation, CTDE repartitioning or replacement algorithm.
- **Reuse:** the cross-owner value discontinuities of
  [measurements §13.7](docs/history/measurements.md#137-owner-transition-audit-read-only-descriptive).
  Also the `lambda = 1` diagnostic of §14 (`68055e3…`, 100 updates, FD100): evidence against
  intermediate cross-owner bootstrap as a sufficient explanation. It did not test a per-ego CTDE
  redesign and must not be rerun.
- **Anchors:** [policy and CTDE §4](docs/contracts/policy_ctde.md#4-phase-b-ctde),
  [runtime §5](docs/contracts/runtime.md#5-resync-stage-6-and-the-two-phase-tick-loop);
  `graph_tick_loop.Transition`, `EpisodeResult`, `_wake_decision`, `run_episode`;
  `graph_ppo.EpisodeRecord.from_trajectory`, `_chain_returns`, `compute_returns_and_advantages`,
  `CTDEEpisodeRecord.from_episode`, `_gae_pass`, `compute_ctde_advantages`;
  `graph_train._run_one_episode`, `_EpisodeOutcome`; `graph_reward.compute_episode_reward`.

#### BASELINE-01 — Conditional baseline, decision weighting and the actual update direction

- **Status:** selected. It also carries the credit-to-update question left by the PPO / clipping
  walkthrough. Actor-only's coarse baseline is an intentional simple estimator, not automatically a
  bug. CTDE already uses `V(global_state, acting_ego)`, so "add a critic" is not a new proposal.
- **Known** (`compute_returns_and_advantages`, `gamma = 1`):
  - `b` = the mean episode `R` over all records, zero-wake records included;
  - `raw_i = R_episode(i) − b`;
  - `A_i = (raw_i − mean_T(raw)) / (std_T(raw) + eps) = (R_episode(i) − mean_T(R)) / (std_T(R) + eps)`,
    using the population std over transitions.

  Any shared `b` therefore cancels. Episodes with more wakes contribute more samples to these
  moments and to the transition-mean surrogate, but not necessarily proportionally more net
  parameter influence. Assess event frequency and policy-induced stopping before mandating
  per-episode weighting.
- **SEVERE both-negative interpretation (belongs here, not in a sign fix).**
  [Measurements §10.6](docs/history/measurements.md#106-credit-findings) records 752 SEVERE
  immediate-FD rows: 202 ABORT and 550 NON-ABORT. Group means derived from its counts, means and
  gaps (not new runs) are ≈ `−0.176053` / `−0.380680` raw and ≈ `−1.017052` / `−1.193543`
  normalized. They pool different worlds and updates, so they are neither paired action values
  nor a gradient direction. For one observation with exactly two legal actions, pushing PLAN
  down pushes ABORT up, so both groups being negative does not imply that both probabilities
  fall. Other states, or additional legal ENGAGE actions, need a separate reading. An exact
  conditional advantage is policy-mean-zero at each state; a sampled estimate or global batch
  centering need not be. The ideal invariance to an action-independent baseline does not carry
  over to finite-batch, normalized, clipped PPO epochs with Adam.
- **Question:** would a baseline conditioned on the relevant pre-action state or history reduce
  between-world difficulty variance and give a useful local comparison? It must never condition
  on the sampled action. Does the available within-SEVERE signal actually move `P(ABORT)` the
  desired way once sample counts, probabilities, state-dependent gradients, non-FD samples and
  PPO / Adam are combined? A better `V` predictor need not estimate counterfactual contribution.
- **First read-only deliverable:** assess conditioning first. Then trace existing credit through
  sample counts and probabilities, local gradients, non-FD contributions and the actual parameter
  and probability change. Keep raw versus normalized advantages, group means, local pressure and
  actual steps distinct. Reuse the existing per-update diagnostics. Do not, by default, force ABORT
  advantages positive, normalize by chosen action or split MILD / SEVERE batches.
- **Reuse:** measurements §10.6–§10.7, §11 and §12–§16, and the qualified §18.
- **Anchors:** `graph_ppo.compute_returns_and_advantages`, `clipped_surrogate`, `PPOUpdater`,
  `compute_ctde_advantages`, `CTDEUpdater`;
  [artifacts and metrics §5](docs/contracts/artifacts_metrics.md#5-per-wake-fd-policy-diagnostics).

#### CRITIC-01 — Central-graph and acting-ego walkthrough / review

- **Status:** selected. The user requested a deep, intuitive read-only review. It is not a critic
  intervention, so it is consistent with the 2026-09-19 direction that excluded another critic,
  readout or `lambda` intervention.
- **Known:** current `main` already marks the acting agent's node EGO and every other live agent
  node PEER (`CentralGraphObservation.ego_index` feeding the encoder's role embedding). It
  mean-pools all node embeddings (`GraphEncoder.pool`) and applies `ValueHead`; there is no
  explicit ego concatenation. The critic has its own encoder instance with the central widths and
  receives the central `edge_attr`. The retired `[mean pool ; acting-node embedding]` readout
  took the acting node's embedding from that same critic encoder, not from a separately trained
  ego encoder ([measurements §13](docs/history/measurements.md#13-generalized-v2-explicit-acting-ego-readout-ctde-development-diagnostic-and-owner-transition-audit)).
- **First read-only deliverable:**
  1. Map node populations and liveness, node and edge features, normalization, topology, capture
     timing, roles, message passing, pooling and the value head.
  2. Explain what the critic can know about the scored continuation, what may be absent
     (executor, history or reference context; a privileged snapshot is not automatically Markov)
     and what is merely a different encoding.
  3. Explain why the decision owner could change the expected continuation.
  4. Only then compare role-only pooling, the tested explicit readout and any justified
     alternative, naming the missing capability each would address.

  Keep representation, prediction quality, advantage quality and behaviour separate. No automatic
  feature addition, concat / attention / readout change or duplicate experiment. Training-only
  central information must never reach actor inputs, masks or runtime knowledge.
- **Anchors:** [policy and CTDE §4](docs/contracts/policy_ctde.md#4-phase-b-ctde);
  `central_graph_builder.CentralGraphObservation`, `build_central_graph_observation`,
  `CentralStateRecorder.capture`; `graph_encoder.GraphEncoder._node_inputs`, `pool`;
  `graph_ppo.CentralCritic`, `ValueHead`; measurements §12–§16.

#### HEAD-01 — How should the actor's global PLAN / ABORT scores be read from its private graph?

- **Status:** deferred architecture candidate, the stronger of the two. No defect or cause is
  established.
- **Known:** `ActionHead` emits k × 3 SOURCE scores. `graph_action._semantic_dist` already forms
  the semantic k + 2 leaves of `semantic_k_plus_2_logmeanexp_v1` (PR #70; do not propose that
  migration again):
  - PLAN is the logmeanexp of all k PLAN scores;
  - ABORT is the logmeanexp of the ABORT scores on ego-assigned task nodes only;
  - there is one ENGAGE(task_i) leaf per task.

  Legality comes from `build_action_mask`. ENGAGE is legal iff the task is unassigned AND sensed
  AND capable AND reachable, which is not the same as newly discovered this tick. ABORT is legal
  iff the ego has at least one assignment, and PLAN is always legal. The legal count is therefore
  `1 + I(ABORT legal) + #legal ENGAGE`. Masked leaves carry zero mass, and PLAN / ABORT are stored
  with `node_v = None`.
- **Question and first read-only deliverable:** what information and gradient paths does the
  current aggregation keep or lose? PLAN includes task components disconnected from the ego;
  ABORT uses only the ego-assigned subset. Would a direct ego / mission readout for PLAN and
  ABORT, keeping target-specific ENGAGE, be a better decision representation? Masks settle
  legality, not readout design. Gathering unchanged legal logits into a smaller vector preserves
  the distribution, whereas deleting task nodes changes the representation. Use actor-private
  information only; the critic readout experiments do not test this actor question.
- **Anchors:** [policy and CTDE §2](docs/contracts/policy_ctde.md#2-encoder-action-head-and-selection-stage-4)
  and [§6](docs/contracts/policy_ctde.md#6-known-limitations-and-open-items);
  `graph_action.ActionHead`, `build_action_mask`, `_semantic_dist`, `sample_action`,
  `evaluate_action`; `graph_encoder.GraphEncoder.forward`. Test bodies in
  `tests/test_graph_semantic_action_credit.py`:
  `test_s2_plan_is_exact_logmeanexp_and_duplication_gives_no_bonus`,
  `test_s3_abort_is_logmeanexp_over_abort_legal_nodes_only`,
  `test_s8_gradients_route_only_through_legal_source_scores`.

#### OBS-01 — Does an unassigned target need direct ego / remaining-mission context?

- **Status:** deferred architecture candidate, behind the credit track. The structural restriction
  is verified; its behavioural importance is unproven, and no current ENGAGE failure is
  established.
- **Known:** `_wake_decision` passes `precedence_relations=[]`, and the builder constructs only
  ASSIGNMENT edges (SPATIAL is reserved and unbuilt). An unassigned target therefore has only its
  encoder self-loop. It keeps its own ego-relative distance, capability, reachability and sensing
  columns, plus time. But with the other features and topology fixed, the ego row's `fuel_norm`
  and `mission_fuel_slack_norm` cannot message into its embedding
  (`test_po6_slack_reaches_the_assigned_task_scores_only_through_the_graph`). Its final ENGAGE
  probability can still depend on ego context through the competing global PLAN / ABORT leaves,
  and fuel can affect its reachability column and legality.
- **Question:** when judging whether a legal opportunity is worth adding, would an ego-private
  sensed / context relation supply useful mission context? Such a relation must stay semantically
  distinct from ASSIGNMENT: sensing a target is never an engagement already chosen. It may use only
  the acting ego's allowed knowledge — no peer sensing, live peer plans, privileged positions,
  hidden inventory or future outcomes. Strong opportunistic-ENGAGE learning is user-reported
  context unless tied to an exact run, metric and denominator. Frequent engagement alone would not
  test the rejection of costly opportunities.
- **OBS-02 (linked subquestion):** distance, capability, reachability and sensing are ego–target
  relations. Storing them on task nodes is still well-defined in a one-ego graph, and relocating
  unchanged values adds no information. In the encoder, `edge_attr` enters only the additive
  attention bias, not node or value payloads, and the actor passes none today. Moving features
  onto edges is therefore not automatically equivalent. A sensing-only edge would leave known
  out-of-range tasks without that relation. Separate added connectivity from relocation, and never
  copy the critic's pairwise graph into the actor.
- **First read-only deliverable:** trace the actor-private context available to an unassigned
  target and compare designs without peer-state leaks, preserving mask semantics and feature
  access. No graph refactor is selected.
- **Anchors:** [policy and CTDE §1](docs/contracts/policy_ctde.md#1-graph-observation-stage-3)
  and [§6](docs/contracts/policy_ctde.md#6-known-limitations-and-open-items);
  `graph_builder.build_graph_observation`; `graph_tick_loop._wake_decision`;
  `graph_encoder.GraphEncoder._node_inputs`, `_build_adj_bias`, `_GraphAttentionLayer.forward`;
  `tests/test_graph_mission_fuel_slack.py`.

#### Closed dispositions (not open repair tasks)

- **REWARD-00 — realized prefix.** `U_prefix` is realized CONFIRMED all-steps task utility over
  the retained t = 0 tasks. It is taken from a copy of `executor.done` at the post-damage,
  pre-response checkpoint (`build_continuation_reference`). It is frozen there: not a solver
  prediction and not an end-of-episode recomputation. The solver supplies `U_cont_ref`. No repair.
- **PEER-01 — zero peer rows.** Peer rows carry no physical state. That makes neither their
  embeddings zero nor a statement that the peer has zero fuel: the shared agent projection's
  learned (zero-initialized) bias, the PEER role, time and assignment neighbours all contribute.
  No independent repair. A new peer feature needs an explicit allowed-knowledge argument, such as
  an explicitly shared initial fact, kept separate from runtime estimates and hidden live state.
- **SCALE-01 — reward sign and units.** On an unchanged actor_only batch at `gamma = 1`, a common
  return shift cancels and a positive global rescale preserves normalized advantages up to eps and
  numerics. A positive reward is therefore not in itself a fix. Per-step bonuses, per-world
  scaling, `c` changes and target-universe changes are different operations. CTDE value fitting
  need not be scale-invariant. `log π ≤ 0` is normal; keep logits, log-probs and losses distinct
  from advantages. PPO's ratio compares the same stored action's probabilities, not old and new
  episode outcomes.
- **The k × 3 action misconception** is covered by HEAD-01's known facts.
- **PPO / evaluation walkthrough — interpretation guard only.** No standalone repair and no
  clipping, epoch, Adam, entropy or benchmark change is selected. The guard:
  - probability separation, selected-action switches and mission utility / losses are different
    quantities;
  - the primary V2 endpoint is the paired SEVERE-minus-MILD semantic `P(ABORT)` at the certified
    ego's immediate-FD wake, summarized per base cell and then macro-averaged equally over the ten
    cells, with eligibility and denominators reported
    ([training and benchmarks §9](docs/contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation));
    development is 20 triads / 60 members, not 20 episodes;
  - deterministic argmax removes action-sampling randomness, not all simulator nondeterminism;
  - a 0.5 threshold applies only when exactly two actions are legal;
  - repeated development evaluation is not confirmatory, and the profile split stays;
  - a changed reward definition requires reading raw outcomes alongside reward values.

  This guard authorizes no new evaluation, seeds, metrics or threshold changes.

**Known low-priority maintenance:** none open from PR #79 — the `graph_encoder` module
self-test's stale `isinstance(node, int)` assertion was corrected with it.

**Recorded, not scheduled and not authorized:**

- open research gaps not addressed by the semantic-action measurement: the `reachable_by_ego`
  round-trip placeholder and route-relative distance representation / clipping
  ([policy and CTDE §6](docs/contracts/policy_ctde.md#6-known-limitations-and-open-items));
- local causal credit at the immediate-FD decision: actor-only instrumentation cannot provide it
  (measurements §10.7). This is now framed as CREDIT-01 / CREDIT-02 (§5.1); any change to reward
  or credit structure would need its own decision;
- undecided local items left untouched by the 2026-09-19 cleanup, each needing its own decision:
  `src/match_aou/rl.zip`, `legacy/run_capture.log`, `src/match_aou/rl/observation/rollouts/` and
  the directory `C:\gruns\` (empty at that cleanup; it now holds the two unarchived run directories
  of §2)
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

No research task is active. The documentation task of §2 is authorized ONLY to record the §5.1
questions and the PR #80 status closure in this file and the decision log. **Recording a §5.1
question authorizes nothing**, neither for that task nor for a future research chat: no code,
test, configuration, preset, reward, observation, action, critic or optimizer change; no training,
evaluation, preflight, replay, checkpoint loading, tuning, cleanup or merge. The credit-to-update
diagnostic's authorization is spent: no second run, restart, resume, seed replication or extension.

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
  action-conditioned gradient subgroups, actor-only gradient instrumentation beyond the integrated
  observational step diagnostic, any implementation arising from a §5.1 question, reachability,
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
credit-to-update diagnostic's measured source `C:/gcud1src` (detached at
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
  CTDE diagnostics, and decomposes PPO epoch 0 only. Only the credit-to-update diagnostic
  (measurements §18) carries actor-only gradients and actual parameter displacements
  (`train_actor_step_diagnostics.jsonl`).
- Measurements §18's own heading, status line and registry row still record the pre-approval
  state ("executed, unreviewed", awaiting re-review). The approval and merge are recorded here
  (§1) and in [`decisions.md` §1](docs/history/decisions.md#1-decision-log) (2026-09-25); §18 is
  not rewritten by the documentation task that added this note.
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
