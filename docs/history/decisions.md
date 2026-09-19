# Decisions and superseded procedures

> **Historical record — not instructions.** A dated authorization, "next action" or ownership
> statement below was true on its own date only and authorizes nothing now. Blocks were moved
> **verbatim** from base `ae42cb01677f94868b2873008d87be677e31f0c8` (the former handoff §5, §3l,
> §3p.4, §8 and §9.2, and parts of the former `CLAUDE.md` §1 and §8). Inside moved text a bare
> `§N` means that former document's section
> ([compatibility index](../../CLAUDE.md#8-compatibility-index-for-older-references)). Current
> state: [handoff](../../graph_rl_project_handoff.md). Active procedures:
> [`docs/workflows/`](../workflows/).

## 1. Decision log

Append one dated entry per material change; never rewrite an entry — supersede it with a new one.

**A material change is one that alters what the project is doing, who owns it, what is
authorized, or what a measurement means.** Routine progress inside an already-recorded task
does not belong here. Append; never rewrite an entry — supersede it with a new one and say
so.

| Date | Decision | Consequence |
|---|---|---|
| 2026-08-18 | **PHASE A CLOSED** on the approved long-baseline rerun at measured code SHA `737b4bf…` (§3h) | The first scientifically valid measurement of the fuel-damage cell; **not to be re-run, repaired or re-tuned**; its non-claims are binding |
| 2026-08-20 | **FD-VARIABLE-SEVERITY-v1 CODE merged** (PR #27, §3i) | An ADDITIONAL actor-only stress design beside FD-BASELINE-v1; legacy modes preserved byte-for-byte; `p(destroy) < 1` explicitly NOT implemented |
| 2026-08-22 | **Serial ordering SUPERSEDED to PARALLEL**: the variable-severity measurement pinned to an immutable detached snapshot while Phase-B CTDE design/implementation proceeded beside it (§4) | The measurement stayed an ACTOR-ONLY measurement OF ITS PINNED SHA; later CTDE work is outside the measured tree |
| 2026-08-23 | **Variable-severity baseline CLOSED — `APPROVE — VALID MEASUREMENT`, PRIMARY FINDING NEGATIVE**, at measured code SHA `bf1e045f…` (§3j) | A valid negative result, not a defect; **not to be re-run, repaired or re-tuned**; it establishes nothing about centralized training |
| 2026-08-23 | **PHASE-B CTDE IMPLEMENTATION merged** (PR #30, §3k) and **documented** (PR #32); `pre-ctde-actor-only` preserved; CTDE integration gate CLOSED on both halves | The implementation is a locked `CLAUDE.md` §5 contract; **no CTDE scientific comparison was run by it and no benefit was claimed** |
| 2026-08-23 | **Repository CLOSED / IDLE** for transfer (PR #33) | True when written; **SUPERSEDED on 2026-08-25** |
| *(undated here)* | **An OLD-CONTRACT CTDE measurement was EXECUTED** on the old fixed cell (§1) | **OUT OF SCOPE for the generalized redesign; not to be reviewed, re-analysed or compared unless the user EXPLICITLY asks.** No identity, SHA, denominator or verdict is recorded here — this task did not inspect it, and inventing any would be false provenance. Reconciling `CLAUDE.md` §8's pre-existing "not run" wording is a step-6 documentation duty (§3l.8) |
| 2026-08-25 | **GENERALIZED TRAINING / BENCHMARK REDESIGN becomes the ACTIVE research/design phase**, owned by the GPT orchestrator; its APPROVED DESIGN recorded in §3l and **marked NOT YET IMPLEMENTED**; this handoff-bootstrap task is the only writable repository task while its candidate is in flight | The repository is **no longer CLOSED / IDLE**; the old fixed-cell Task-9 framing is **no longer the live next action**; **no implementation candidate is active and no new scientific measurement is running or authorized**; the single next action is §3l.8 step 1, gated on GPT exact-SHA review of this record plus user-authorized continuation (§8) |
| 2026-08-25 | **GENERALIZED-V1 TASK 1 IMPLEMENTED, REVIEWED AND MERGED** — generalized construction cardinality, deterministic bounded B2 backoff and truthful requested-vs-realized accounting (§3l.1, §3l.2); candidate `5b55ca348309b4241d2087c2f60327bc842ea6fa`, integration `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9`, PR #35, Grade A under `GPT_GITHUB` | §3l.8 step 1 is COMPLETE. The historical `exact_v1` path is PRESERVED and remains the DEFAULT, so the approved Phase-A and variable-severity measurements are untouched and are still measurements OF IT; `bounded_backoff_v1` is an OPT-IN ADDITION beside it that **neither harness selects**. `EpisodeContext.construction_audit` records requested-vs-realized from the RAW world snapshots and reaches no `GraphObservation`. **No scientific run occurred and no generalized measurement exists** |
| 2026-08-25 | **GENERALIZED-V1 TASK 2 IMPLEMENTED, REVIEWED AND MERGED** — certified FD eligibility and post-FD completion-boundary adaptation (§3l.3, §3l.4); final candidate `185d39f00335a0bb5e9130cc773da94c914f17f5`, integration `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b`, PR #36, Grade A under `GPT_GITHUB`. *(Historical process evidence: initial reviewed candidate `2f9231d989acf30561ecf10e74cf0c5491771836` → REQUEST FIXES → append-only child `185d39f0…` → APPROVE.)* | §3l.8 step 2 is COMPLETE. Both FD policy seams are OPT-IN with the LEGACY defaults preserved and **selected by neither harness**, so the approved measurements are untouched. A NEW routing exists: `FuelDamageIntegrityError` is an INSTRUMENT abort — for a live certificate contradiction AND for a certified damaged episode that ends without the event firing — while setup ineligibility (`no_fd_eligible_ego`) stays ordinary accounted attrition inside `skip_and_account_v1`. `p(destroy)` stays `1.0` and no new `MetaAction` exists. **No scientific run occurred and no generalized measurement exists** |
| 2026-08-25 | **INTERMEDIATE DOCUMENTATION CHECKPOINT OPENED** because the live handoff had become STALE after two implementation merges — it still claimed nothing generalized was implemented and that Task 1 was the next action — and because `CLAUDE.md` carried no contract for behaviour that is now merged | `CLAUDE.md` gains the Task-1 and Task-2 §5 contracts, their §6 routing and their §7 locks, and its stale actor-only-vs-CTDE "never executed" wording is reconciled CONSERVATIVELY (existence acknowledged, OUT OF SCOPE, **no identity / SHA / denominator / verdict / result recorded and no CTDE benefit claimed**). This handoff records steps 1–2 as COMPLETE, §3l.5–§3l.7 as NOT IMPLEMENTED, and **Task 3 as the next unresolved task that this record does NOT authorize** (§8). It also SUPERSEDES the earlier "`CLAUDE.md` locks only at step 6" rule: locks are written PER COMPLETED TASK, and a FINAL documentation pass is still required after the later steps land. **Two files only; no code, test, config, preset or workflow change; no run; no ref moved; no historical measurement reinterpreted** |
| 2026-08-25 | **GENERALIZED-V1 TASK 3 IMPLEMENTED, REVIEWED AND MERGED** — the event-conditioned MATCH-AOU continuation reference and reward checkpoint (§3l.5); reviewed candidate `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integration `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, PR #38, verdict **APPROVE**, Grade A under `GPT_GITHUB` | §3l.8 step 3 is COMPLETE, so **§3l.5 is now IMPLEMENTED**. The historical `static_t0_v1` reference is PRESERVED and remains the DEFAULT, so the approved Phase-A and variable-severity measurements are untouched and are still measurements OF IT; `event_conditioned_continuation_v1` is an OPT-IN ADDITION beside it that **neither harness selects**. The checkpoint is privileged READ-ONLY measurement that reaches no acting path, advances no simulation time and never adds a THIRD reference solve. The review APPROVED ONE compatibility deviation, `damaged_event_unrealized_t0`, scoped to the LEGACY Task-2 contract and **not** generalized damaged semantics. `p(destroy)` stays `1.0`, no new `MetaAction` exists, and terminal credit placement is unchanged. **No scientific run occurred and no generalized measurement exists** |
| 2026-08-25 | **TASK-3 DOCUMENTATION / LOCK RECORD OPENED**, because reviewed and integrated behaviour existed for §3l.5 while the live documents still said it was NOT IMPLEMENTED and that Task 3 was the next unresolved task | `CLAUDE.md` gains the Task-3 §5 contract, its §4 pipeline placement, its three §6 routing rows and its §7 lock, and its stale “no continuation reference / no reward change” and “`graph_reward` remains FROZEN” wording is corrected in place. This handoff records Tasks 1–3 as COMPLETE, §3l.5 as IMPLEMENTED, §3l.6–§3l.7 as NOT IMPLEMENTED, and **Task 4 as the SINGLE next unresolved task that no documentation record authorizes** (§8). It is written in stable POST-INTEGRATION form: no present-tense claim that any documentation branch or PR is the active writable task. **Two files only; no code, test, config, preset or workflow change; no run; no ref moved; no historical measurement reinterpreted** |
| 2026-08-26 | **GENERALIZED-V1 TASK 4 IMPLEMENTED, REVIEWED AND MERGED** — the episode-design selector, the generalized training cardinality sampler, the frozen stratified benchmark MANIFEST MECHANISM and run-level persistence / aggregate metrics (§3l.6, §3l.7); FINAL approved candidate `db79013897a6e5669f50d53b6e30229b16aea28d`, integration `b4daa8c1a8c870061b26cceb01d4ed34169594e7`, PR #40, verdict **APPROVE**, Grade A under `GPT_GITHUB`. *(Historical process evidence: original reviewed candidate `eef1795f6bb3f0cbc4c163ba489cf5e790df4c41` → review corrections → append-only child `db790138…` → APPROVE, covering manifest integrity, real held-outness and honest generalized construction provenance.)* | §3l.8 step 4 is COMPLETE, so **§3l.6–§3l.7 are now IMPLEMENTED** and **§3l.8 steps 1–4 are ALL COMPLETE**. The FOUR low-level OPT-IN policy seams are now resolved TOGETHER — and only together — by the ONE `episode_design` selector, whose DEFAULT `fixed_cell_v1` preserves the historical bundle the approved measurements were taken on and leaves a default run byte-invariant at the call boundary; `training_mode` stays ORTHOGONAL. Every Task-1/2/3 per-episode structure is now PERSISTED and AGGREGATED with explicit denominators, requested-vs-realized is REPORTED with **no acceptance threshold and no verdict**, and `ReferenceIntegrityError` now routes by stable REASON SLUG (unanswered solve ⇒ accounted attrition; every other reason ⇒ measurement-integrity ABORT). **Task 4 delivered the benchmark MECHANISM ONLY: no FINAL SCIENTIFIC worlds-per-cell scale was SELECTED and no FINAL SCIENTIFIC benchmark population was committed, preserved as the comparator, scheduled or authorized (no benchmark manifest is committed or tracked in the repository; transient manifests built by tests are neither).** `p(destroy)` stays `1.0` and no new `MetaAction` exists. **No scientific run occurred, no generalized measurement exists, and no actor-only-vs-CTDE generalized result exists** |
| 2026-08-30 *(status SUPERSEDED on 2026-08-31 by the integration row at the end of this table; every fact below was accurate ON ITS OWN DATE)* | **GENERALIZED-V1 TASK 5 IMPLEMENTED AND APPROVED AS A STACKED, STILL-UNMERGED TWO-PR STACK** — **PR #42**, branch `task/generalized-v1-task5-summary-phase-fix`, approved head `312f58650b61a85eb72d0554d60715afee862a5c` (the `train_by_*` summary-population correction), and **PR #43**, branch `task/generalized-v1-task5-success-quota-preflight`, FINAL approved head `4af6c5aa5dd28072692bfda63282964b55010aae` (the successful-episode training quota, the bounded attempt budget, the maximum-possible seed band and the deterministic benchmark preflight), Grade A under `GPT_GITHUB`. *(Historical process evidence: PR #43's original implementation candidate `734f1e786593b6ffb94f1f8d7283b1f2fc79d257` → ONE requested review fix → append-only DIRECT CHILD `4af6c5aa…` → APPROVE; no amend, rebase, squash, force-push or history rewrite.)* | §3l.8 step 5 is IMPLEMENTED and APPROVED but **NOT MERGED**: live `main` is still `09eab0673153bd443185ec94530ccf0b042be465`, so **no integration SHA exists for either PR and none may be invented**, and both PRs are **FROZEN / READ-ONLY**. The historical `scheduled_attempts_v1` fixed-cell attempt behaviour and every fixed-cell seed band and check are PRESERVED, so the approved Phase-A and variable-severity measurements are untouched. Held-outness is now checked against the **MAXIMUM POSSIBLE attempt band**, because a failed replacement attempt still spends a seed. Benchmark population SELECTION happens ONCE, before the freeze, and **post-freeze evaluation still performs NO substitution**; a failed preflight creates NO manifest and leaves a durable candidate audit that **is not a benchmark population**. **`p(destroy)` stays `1.0`, no new `MetaAction` exists, and NO generalized measurement RESULT exists** |
| 2026-08-30 | **TASK 5A and TASK 5B reviewed `APPROVE — VALID ENGINEERING VALIDATION`** (§3m.3) | **ENGINEERING EVIDENCE, NEVER A SCIENTIFIC MEASUREMENT — and what makes that true is their DESIGNATED PURPOSE, not an absence of mechanics.** Task 5B really did carry an explicit training seed band `[720000, 720072)`, an explicit benchmark candidate band, production held-out verification, a TRANSIENT frozen manifest and 18 worlds / 54 members for its one evaluation round — all of it existing solely to validate system behaviour, attrition and runtime, and explicitly NOT designated as the scientific comparator or as a policy-performance measurement. Task 5A's repeated **A2-LOW `pre_event_popup_risk`** failure on a TRANSIENT one-world-per-cell benchmark is what exposed the need for eligibility selection BEFORE the freeze; solver runtime dominated, and repeated pre/post values on the SAME world are **not** independent observations. Task 5B, at measured code SHA `4af6c5aa…`, validated the mechanics (24/24 training successes, 18/18 first candidates accepted, 0 observed hidden shortfalls, one transient 54/54 benchmark round with 18/18 complete groups, BONMIN dominating runtime, large `A4-high` variance, and a legitimate ~998 s solve terminating `optimal` — which is why **no short solver timeout was adopted**). **Bounded samples: NO attrition-rate population claim, NO learning claim, NO actor-vs-CTDE claim, and the Task-5B transient benchmark is NOT the R1 comparator** |
| 2026-08-30 | **THE FIRST FULL GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN AUTHORIZED AND DISPATCHED** with the frozen plan in §3m.4 — `actor_only`, 375 iterations × 8 SUCCESSFUL episodes = 3000 successful training episodes, `generalized_max_attempts_per_iteration = 12`, training base seed 740000 with maximum-possible band `[740000, 744500)`, `worlds_per_cell = 3`, R1 benchmark base seed 840000, `max_candidates_per_cell = 12`, evaluation and checkpoint every 25 iterations, no early stopping, no solver timeout, no CTDE arm | **THE R1 BENCHMARK SCALE IS THEREBY SELECTED AND AUTHORIZED (`worlds_per_cell = 3`) AND ITS CONSTRUCTION AUTHORIZED AND DISPATCHED (candidate base seed `840000`, `max_candidates_per_cell = 12`), which SUPERSEDES every earlier "no FINAL SCIENTIFIC scale has been SELECTED / no benchmark population is scheduled or authorized" statement as CURRENT state — those remain accurate only as the historical records they are. **STATE: `AUTHORIZED / DISPATCHED — RESULT PENDING`.** **No concrete R1 manifest has yet been independently reviewed or approved as the comparator, and none is committed or tracked in the repository.** It is UNREVIEWED and has produced no verdict; **nothing about its reward, convergence, attrition, benchmark outcome or validity may be stated or inferred**, and this record deliberately does not claim it is `RUNNING`. It requires **independent GPT artifact review** before any `APPROVE — VALID MEASUREMENT`. The plan is recorded so the eventual artifacts can be checked against what was authorized. **The external long-run task is RUN-ONLY and owns NO repository writes.** No reviewed early-stopping mechanism exists and checkpoint RESUME stays out of scope; cluster readiness is DEFERRED without blocking the local R1 |
| 2026-08-30 *(status SUPERSEDED on 2026-08-31: that checkpoint became PR #44 and IS now merged — see the integration row at the end of this table; every fact below was accurate ON ITS OWN DATE)* | **POST-TASK-5 DOCUMENTATION / LOCK CHECKPOINT OPENED**, because reviewed and approved behaviour existed for §3l.8 step 5 while the live documents still said Task 5 was NOT STARTED and NOT AUTHORIZED, that no writable implementation task was active, and that no generalized run was scheduled or authorized | `CLAUDE.md` gains the Task-5 §5 contract (summary population, the two attempt policies and the bounded budget, the maximum-possible seed band, the deterministic preflight, the complete-manifest rule with immutable post-freeze evaluation, and the failed-preflight durable audit), its four §6 routing rows, and its §7 entries for **PR #42** and **PR #43 recorded as APPROVED and NOT YET INTEGRATED — with no invented merge SHA** — plus the Task-5A / Task-5B engineering-validation entry under a BINDING label; and its stale global "no generalized measurement exists / is running / is scheduled" wording is scoped in place. This handoff records the Task-5 stack, the ownership split (**the long run is RUN-ONLY; this docs candidate is the SOLE writable repository task**), the dispatched R1 as **RESULT PENDING**, the early-stopping / resume / cluster state, and the intended integration sequence in a new §3m — **recorded, not performed: no merge is authorized.** **Two files only; no source, test, config, preset, benchmark manifest or run artifact committed; no run; no ref moved; no historical measurement reinterpreted; no result claimed for R1** |
| 2026-08-31 | **THE WHOLE GENERALIZED-V1 TASK-5 STACK INTEGRATED — A THREE-PR SEQUENCE, ALL MERGED.** **PR #42** `312f58650b61a85eb72d0554d60715afee862a5c` → merge `5dfcd8b632be8dca3c1730018bbf35337d07f077`; **PR #43** `4af6c5aa5dd28072692bfda63282964b55010aae` → merge `b3c2e01f130afe854b09384cd6e1e196de714795`; **PR #44** (the Task-5 documentation lock, append-only child of `61eaa3fe1bdeb7aef3cfb7c10c4d8964caf2ed0e`) `88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` → merge `9b9e9b85a70c8a0019c72ada92ceec3401725795`. PR #43 and PR #44 were each RETARGETED to `main` and **EXACT-BASE RE-REVIEWED** with their heads unchanged | The §3m.5 nine-step sequence was PERFORMED. Every integration is a **normal merge commit** preserving its reviewed candidate as an ancestor / merge parent, with the integrated tree verified equal to the reviewed tree; **no rebase, no squash, no cherry-pick, no force-push and no history rewrite** occurred, so all three candidates remain reachable through normal merge history. **§3l.8 steps 1–5 are now ALL INTEGRATED.** The retired `task/*` Task-5 branches are cleanup-eligible, NOT writable, and were NOT deleted; `flat-final`, `phase-a-baseline`, `pre-ctde-actor-only` and the `pre-cleanup` tag are UNTOUCHED. **R1 remains `AUTHORIZED / DISPATCHED — RESULT PENDING`** — neither `RUNNING`, nor `COMPLETED`, nor `VALID` — and **NO generalized measurement RESULT exists** |
| 2026-08-31 | **POST-INTEGRATION CLOSURE RECORD OPENED**, because the live documents still described the Task-5 stack as approved-but-unmerged, named the merged doc-lock branch as the sole writable task, and carried an integration sequence marked "RECORDED, NOT PERFORMED" | Both documents are moved from that pre-integration state to the exact integrated state: `CLAUDE.md` records the PR #42 / PR #43 integration SHAs in its §5 header and §7 entries, adds the §7 entry for **PR #44**, and updates §8 to say Tasks 1–5 are ALL integrated; this handoff updates its title, live state, §1, §3l.8, §3m.1, §3m.2, §3m.5 (now **PERFORMED**), §4, §8, §9 and this table. **The temporary closure PR is distinguished from the post-merge state: this candidate is the SOLE writable repository task while its own draft PR is open, and on its integration NO writable repository task remains — while GENERALIZED-V1 stays an ACTIVE phase because R1 is pending.** **Two files only; no source, test, config, preset, benchmark manifest or run artifact touched; no run; no ref moved or deleted; no historical measurement reinterpreted; no result claimed or inferred for R1; this record's own integrating merge SHA is deliberately NOT named** |
| 2026-08-26 | **TASK-4 DOCUMENTATION / LOCK RECORD OPENED**, because reviewed and integrated behaviour existed for §3l.6–§3l.7 while the live documents still said Task 4 was the next unresolved task, NOT started and NOT authorized, that neither harness selected any generalized policy, and that the sampler, the manifest and run-level persistence were unimplemented | `CLAUDE.md` gains the Task-4 §5 contract, its §4 selector placement, its five §6 routing rows and its §7 lock; its stale "NEITHER HARNESS EXPOSES" wording in the Task-1/2/3 blocks is corrected in place with the superseded text preserved; and the `ReferenceIntegrityError` routing block is rewritten to record the Task-4 decision that was deliberately deferred to it. This handoff records Tasks 1–4 as COMPLETE, §3l.1–§3l.7 as IMPLEMENTED, **GENERALIZED-V1 TASK 4 as CLOSED with no writable implementation task and no candidate under review**, and **Task 5 (bounded runtime / solver validation) as the SINGLE next unresolved step that no documentation record authorizes** (§8). It records explicitly that **no FINAL SCIENTIFIC benchmark scale has been SELECTED and no FINAL SCIENTIFIC benchmark population or manifest has been committed, preserved as the comparator, scheduled or authorized**, that **no generalized measurement exists, is running, is scheduled or is authorized**, and that the future actor-only and CTDE arms must use the SAME eventual frozen manifest while the approved historical baselines are REUSED and never rerun. Written in stable POST-INTEGRATION form: no present-tense claim that any documentation branch or PR is the active writable task. **Two files only; no code, test, config, preset, benchmark manifest or workflow change; no run; no ref moved; no historical measurement reinterpreted** |
| 2026-08-31 | **BGU CLUSTER EXECUTION ENVIRONMENT VALIDATED AND RECORDED; CLUSTER ENVIRONMENT / RUNTIME READINESS MOVES FROM DEFERRED TO VALIDATED / READY** against exact `main` SHA `926aba66fcaf2b99fc58685eb202888d8deeaf5f`, because cluster access now exists and the environment was independently validated there against that exact SHA with a clean working tree (§3m.6) | A NEW `environment.cluster.yml` records the DIRECT validated surface (`graph_rl_cluster`, conda-forge only + `nodefaults`, Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1, **`pytorch-cpu` 2.13.0**, Pyomo 6.10.1, `coin-or-bonmin` 1.8.9, Gymnasium 0.29.1, Shapely 2.0.6, Haversine 2.9.0) and is deliberately **NOT a transitive lockfile** and deliberately excludes stable-baselines3 / TensorBoard / CUDA / pytest / plotting libraries; vendored BLADE stays a SEPARATE editable install from `src/match_aou/integrations/panopticon-main/gym`. `requirements.txt` stays the broad PYTHON surface but stops CONTRADICTING BLADE — `shapely` is pinned `==2.0.6` to match BLADE's own `install_requires`, the gymnasium floor is kept broad with BLADE's `==0.29.1` extra recorded, and the solver note is corrected to name BONMIN via conda-forge `coin-or-bonmin` instead of suggesting Ipopt/GLPK as substitutes. `CLAUDE.md` §1 now carries TWO execution contexts without contradiction — the PRESERVED local Windows `nlp_env` contract with its existing caveats intact, and the cluster contract with the **LOAD-BEARING `PYTHONNOUSERSITE=1`** isolation rule and the CPU-only PyTorch state — and §2's frozen-BLADE wording now covers both install locations WITHOUT weakening the frozen contract. The handoff gains §3m.6 as VOLATILE operations: the observed `course` QoS limits (`MaxWall 1-00:00:00`, `MaxTRESPU cpu=66 / gres/gpu=1 / mem=64G`, `MaxMemPerCPU 4096 MB`, `DefCpuPerGPU=6`, `MaxNodes=1`), the `sinteractive` conclusion (**no wrapper change needed**; `DefCpuPerGPU` explains the 6-CPU allocation; launch it OUTSIDE the repository), and the note that **no compute-node performance benchmark is a closure gate** because existing engineering evidence already identifies solver/runtime dominance. **THE ENVIRONMENT SMOKE IS ENGINEERING / RUNTIME VALIDATION, NEVER A MEASUREMENT** — the long `graph_train` selftest was EXTERNALLY TERMINATED and **must not be recorded as a full PASS**, and expected fixed-cell attrition and synthetic test tracebacks must not be read as environment failures. **READINESS IS NOT AUTHORIZATION:** no scientific `sbatch` / job-array launcher exists, is designed or is authorized; no partition / queue / walltime decision is made; **no five-run matrix is defined**; **no benchmark manifest is constructed, frozen, committed or approved**; **no CTDE generalized run is authorized**; `p(destroy)` remains `1.0`; and **R1 is UNTOUCHED — still the LOCAL run it was dispatched as, still `AUTHORIZED / DISPATCHED — RESULT PENDING` and UNREVIEWED, with no result stated or inferable.** Historical "cluster readiness is DEFERRED" statements are SUPERSEDED as CURRENT state only and PRESERVED as the records they were. **Four files; ZERO source, test, BLADE, solver, config, preset, benchmark-manifest or run-artifact changes; no run; no ref moved; no historical measurement reinterpreted** |
| 2026-08-31 | **THE CLUSTER ENVIRONMENT REPRODUCIBILITY LOCK IS INTEGRATED, AND THIS POST-MERGE CLOSURE RECORD IS OPENED**, because merging PR #46 made the handoff's own present-tense ownership and base state stale the moment it landed: the document still said the reproducibility-lock candidate was the sole writable task "while its DRAFT PR is open" and still gave the live base as `926aba66…` | **PR #46** — branch `task/cluster-env-repro-lock`, reviewed candidate `cbc227450067d96c630eed208e22b3a5a20efc1b`, GPT verdict **APPROVE**, user-authorized merge — integrated by **NORMAL merge commit `e9f9f4f93412c8c6c3dd8ba81a7e784dc52cc68b`** (`2026-08-31 16:40:13 +0300`), whose ordered parents are `926aba66…` then `cbc22745…`, so the reviewed candidate is preserved as its SECOND PARENT and remains reachable; **the integrated tree was verified IDENTICAL to the reviewed tree** (all four files, and the whole tree). **No rebase, no squash, no cherry-pick, no force-push, no amend and no history rewrite.** The merged branch was then safely deleted only after its tip was proven equal to `cbc22745…`, reachable from integrated `main`, and carrying zero commits outside it. This closure record moves the volatile state to its POST-MERGE condition: **live `main` is `e9f9f4f93412c8c6c3dd8ba81a7e784dc52cc68b`**, PR #46 is MERGED and no longer writable, and **THIS post-merge closure candidate (`task/cluster-env-post-merge-closure`) is the SOLE WRITABLE REPOSITORY TASK while its draft PR is open, with NO writable repository task remaining after its integration until a future task is explicitly opened.** **The `926aba66…` VALIDATION SHA is deliberately NOT rewritten** — the cluster environment really was validated there, and recording that validation later does not move where it was taken. Preserved unchanged: cluster environment / runtime readiness **VALIDATED / READY** and **NOT scientific authorization**; GENERALIZED-V1 ACTIVE because R1 is pending; Tasks 1–5 integrated; Task 5A / 5B ENGINEERING EVIDENCE ONLY; R1 **`AUTHORIZED / DISPATCHED — RESULT PENDING`** and UNREVIEWED with no result stated or inferable; **no CTDE generalized run authorized**; **no five-run scientific matrix defined**; **no frozen scientific benchmark manifest approved or committed**; **no scientific `sbatch` / job-array launcher exists**; `p(destroy)` remains `1.0` with `p(destroy) < 1` DEFERRED. The five open items for the next fresh orchestration thread are unchanged and remain UNDECIDED. **One file; ZERO source, test, BLADE, solver, config, preset, benchmark-manifest, launcher and run-artifact changes; no run; no ref moved beyond deleting the merged PR-#46 branch; no historical measurement reinterpreted** |
| 2026-09-01 | **OPT-IN TRAINING-REWARD EARLY STOPPING (`training_reward_plateau_v1`) IS IMPLEMENTED, REVIEWED `APPROVE` (Grade A, `GPT_GITHUB`) AND INTEGRATED — PR #48, reviewed candidate `bdfd80d546e9d5779e4d52b522d5db6d8eb610e9` → merge `0b9a1d63f257a8ed9555f81a1d2bf10e30168e66`, from base `6f98b4becb39556081389b0e5b48b2dbb7675a5d`**, a normal merge preserving the reviewed candidate as its SECOND PARENT with the integrated tree `411126d1d9641356673efbf47510c335b4cf0f9b` IDENTICAL to the reviewed tree, no rebase / squash / cherry-pick / force-push / history rewrite, and a SINGLE candidate commit with no review-fix chain. THREE files only: `graph_train.py`, `test_graph_train.py`, `test_graph_ctde.py` — no config, no preset, no benchmark manifest, no documentation file | **THIS SUPERSEDES, AS CURRENT STATE ONLY, EVERY "NO REVIEWED EARLY-STOPPING MECHANISM EXISTS" STATEMENT** in this document and in `CLAUDE.md` — each remains accurate as the record it was, through PR #47. The mechanism is **OFF BY DEFAULT**, approved for `generalized_v1` ONLY with the fixed-cell path REFUSED, and decides from the persisted `train_reward_mean` and **nothing else** — no benchmark or held-out reward, no success / feasibility rate, no PPO or CTDE-critic diagnostic, no checkpoint state, no final-comparator result — with the isolation MECHANICAL (a two-keyword pure monitor, and the ordering record → check → attach → flush → break BEFORE that boundary's periodic evaluation and checkpoint). The approved rule is **100 / 25 / 3 / 0.01** in COMPLETED-ITERATION counts, so at the intended 8 successful episodes per iteration monitoring begins after **800 successful episodes** and the **EARLIEST POSSIBLE stop is 175 completed iterations = 1400 successful episodes — the EARLIEST, never a promised or expected stopping point**, and the 1400 figure is the campaign interpretation at 8 episodes per iteration only. `training_mode` is read nowhere, so **actor-only and CTDE stop by the identical rule**, and comparison semantics are `same maximum budget + same frozen stopping rule + same training-population contract` — **NOT `same actual number of completed iterations`**. **IT IS CODE, NOT A MEASUREMENT: no scientific run has used it**, no reward / convergence / runtime-saving / sample-efficiency / performance claim is made or supported, and firing it would record only that the configured plateau rule fired — never a convergence or optimality claim. **THE DISPATCHED ACTOR-ONLY R1 IS UNTOUCHED** and remains on its ORIGINAL FIXED-BUDGET contract with early stopping `none`, still `AUTHORIZED / DISPATCHED — RESULT PENDING` and UNREVIEWED, with nothing about its outcome stated or inferable. **Checkpoint RESUME remains OUT OF SCOPE and `graph_train` remains SAVE-only**; the PLANNED `max_training_attempts` still governs every held-out claim and never shrinks because a run stopped early; **no repository preset enables the policy** and **no benchmark manifest is committed or tracked**; and **no CTDE generalized run is authorized, scheduled or running** |
| 2026-09-01 | **THE EARLY-STOPPING DOCUMENTATION / LOCK RECORD IS OPENED**, because merging PR #48 made both documents' present-tense claims stale the moment it landed: `CLAUDE.md` §8 and this handoff (§1, §3m.4, §7, §8) still said **"no reviewed early-stopping mechanism exists"**, and the handoff still named the PR-#47 post-merge closure candidate as the sole writable task while giving the live base as `e9f9f4f9…` | `CLAUDE.md` gains the GENERALIZED-V1 early-stopping **§5 contract** (the closed policy set and the ONE `early_stopping_enabled` predicate; the PRESERVED fixed-budget default and its `generalized_v1`-only approval; the forbidden-input list and why the isolation is MECHANICAL; the approved state machine in COMPLETED-ITERATION counts with 175 / 1400 as the EARLIEST possible stop; actor-only / CTDE parity; `EarlyStoppingIntegrityError` on a missing `train_reward_mean` inside a monitored window; the load-bearing `train` ordering and single finalization at the ACTUAL final iteration; planned-vs-actual budget semantics with `max_training_attempts` unmoved; SAVE-only checkpoints with resume still deferred; the observability carried by the EXISTING artifacts; and the configuration surface with its `validate()` refusals), **two §6 routing rows** ("Change WHEN a GENERALIZED-V1 run stops training" and "Read why/how a run stopped"), a **§7 lock entry** for PR #48 with its exact SHAs, identical-tree proof, three-file scope and single-commit provenance, and a **§8** status bullet. This handoff gains a 2026-09-01 live-state paragraph, an updated §1 / §3m.2 / §3m.4 / §4 / §7 / §8, and a new **§3m.7**. **Two files only; no source, test, config, preset, benchmark manifest or run artifact changed; no run; no ref moved; no historical measurement reinterpreted; no result claimed for R1; and no merge, implementation, benchmark population, campaign or run authorized** |
| 2026-09-02 | **THE EARLY-STOPPING POST-MERGE CLOSURE RECORD IS OPENED**, because merging PR #49 made this handoff's present-tense ownership and base state stale the moment it landed: it still named the early-stopping DOCUMENTATION / LOCK candidate as the sole writable task "while its DRAFT PR is open" and still gave live `main` as `0b9a1d63…`. **PR #49** — branch `task/generalized-v1-early-stopping-doc-lock`, reviewed candidate `77c26dde1396acc7793d50fbcac840474601bf88` — was integrated by **NORMAL merge commit `f74c288175a1f8228407806bf5c8056beff75239`** (`2026-09-02 13:26:52 Asia/Jerusalem`), ordered parents `0b9a1d63f257a8ed9555f81a1d2bf10e30168e66` then `77c26dde…`, reviewed candidate and integration sharing the IDENTICAL tree `1b944749fdf52ef3d2175e4437428df4ffc0b656`, with no rebase, squash, cherry-pick, force-push or history rewrite | **Live `main` is `f74c288175a1f8228407806bf5c8056beff75239`; PR #48 and PR #49 are BOTH MERGED; both early-stopping candidates and their branches are RETIRED, READ-ONLY historical provenance and NEITHER IS WRITABLE**; the mechanism is **BUILT / REVIEWED / APPROVED / INTEGRATED / DOCUMENTED**, still **OFF BY DEFAULT**, on the approved rule **`100` / `25` / `3` / `0.01`**, whose **175 completed iterations = 1400 successful episodes at 8 per iteration is the EARLIEST POSSIBLE stop, never an expected or guaranteed one**. **R1 is UNTOUCHED and governed by its ORIGINAL FIXED-BUDGET contract with NO early stopping, and remains `AUTHORIZED / DISPATCHED — RESULT PENDING` and UNREVIEWED**; **no generalized CTDE run is authorized**; **checkpoint RESUME remains out of scope**; and **no scientific measurement result was produced by PR #48, PR #49 or this closure task**. **This candidate is the SOLE WRITABLE REPOSITORY TASK while its own draft PR is open, and on its integration NO writable repository task remains — the three early-stopping task branches becoming retired references pending bounded ref-only cleanup (a SEPARATE later operation, not authorized here), with NO NEW TASK IMPLICITLY AUTHORIZED** — while GENERALIZED-V1 stays an ACTIVE phase because R1 is pending. The next scientific action is UNCHANGED: **INDEPENDENT GPT ARTIFACT REVIEW of the actor-only R1 when its artifacts exist**, with no rerun, repair, resume, extension, new actor-only arm, CTDE arm, benchmark replacement, retuning or five-run execution matrix authorized. **ONE FILE ONLY (`graph_rl_project_handoff.md`); `CLAUDE.md` untouched; no technical contract altered or reinterpreted; no source, test, config, preset, benchmark manifest or run artifact touched; no run; no ref moved or deleted; this record's own integrating merge SHA is deliberately NOT named** |
| 2026-09-02 | **THE FINAL EARLY-STOPPING HANDOFF-STABILIZATION RECORD IS OPENED**, because merging PR #50 made this handoff's own present-tense ownership and base state stale the moment it landed — it still named the early-stopping POST-MERGE CLOSURE candidate as the sole writable task "while its DRAFT PR is open" and still gave live `main` as `f74c288175a1f8228407806bf5c8056beff75239`. **PR #50** — branch `task/generalized-v1-early-stopping-post-merge-closure`, reviewed candidate `a7d6dea5375a809e8b59aaee19f763f5769499ea` — was integrated by **NORMAL merge commit `e9cbd80244926680d90c81d9440753b89e22efdc`** (`2026-09-02 16:40:45 Asia/Jerusalem`), ordered parents `f74c288175a1f8228407806bf5c8056beff75239` then `a7d6dea5…`, reviewed candidate and integration sharing the IDENTICAL tree `88f3ce73c42f0c0680e1d62411816606b2b36dda`, with no rebase, squash, cherry-pick, force-push or history rewrite. **THE DESIGN CONSTRAINT OF THIS RECORD IS THAT IT MUST REMAIN TRUE AFTER ITS OWN INTEGRATION**, so it deliberately records `e9cbd802…` as the PR-#50 integration and this record's AUTHORING BASE and NEVER as a durable "live `main`", and it names no candidate or merge SHA of its own. **THE 2026-09-02 PR-#49 ENTRY IMMEDIATELY ABOVE STATES IN ITS CONSEQUENCE COLUMN THAT “Live `main` is `f74c288175a1f8228407806bf5c8056beff75239`”; that clause is SUPERSEDED as CURRENT state by this entry** — under §9.2 an earlier entry is never rewritten, only superseded, and `f74c2881…` is now a HISTORICAL integration and PR #50’s first parent, NOT live `main` | **PR #48, PR #49 AND PR #50 ARE ALL MERGED**, so the early-stopping IMPLEMENTATION, DOCUMENTATION / LOCK and POST-MERGE CLOSURE are COMPLETE and `training_reward_plateau_v1` is **BUILT / REVIEWED / APPROVED / INTEGRATED / DOCUMENTED / CLOSED**, still **OFF BY DEFAULT**, on the approved rule **`100` / `25` / `3` / `0.01`**, whose **175 completed iterations = 1400 successful episodes at 8 per iteration is the EARLIEST POSSIBLE stop, never an expected or guaranteed one**. **This record cannot embed its own future integration SHA under the `CLAUDE.md` §7 hash convention, so EVERY RECEIVING ORCHESTRATOR MUST RESOLVE THE EXACT LIVE `main` FROM GITHUB BEFORE ACTING** (§9.1). **While this final-stabilization PR is OPEN its branch is the sole writable repository task; ONCE IT IS INTEGRATED NO WRITABLE REPOSITORY TASK REMAINS**, all FOUR early-stopping branches becoming retired, cleanup-only references pending a bounded ref-only cleanup that is repository HYGIENE, is a SEPARATE later operation, is NOT authorized here and does NOT displace the scientific next action — the fourth branch cleanup-eligible only from its own integration and NOT before — with **NO NEW TASK IMPLICITLY AUTHORIZED**. **R1 is UNTOUCHED, fixed-budget, with NO early stopping, and remains `AUTHORIZED / DISPATCHED — RESULT PENDING` and UNREVIEWED**; **no generalized CTDE run is authorized**; **checkpoint RESUME remains out of scope**; and **no scientific measurement result was produced by PR #48, PR #49, PR #50 or this record**. The ONE current scientific next action is UNCHANGED: **INDEPENDENT GPT ARTIFACT REVIEW of the actor-only R1 once its artifacts exist**, with no rerun, repair, resume, extension, new actor-only arm, CTDE arm, benchmark replacement, retuning or five-run execution matrix authorized, and the five future campaign items remain FUTURE, OPEN, UNDECIDED, UNAUTHORIZED and **NOT NEXT**. **ONE FILE ONLY (`graph_rl_project_handoff.md`); `CLAUDE.md` untouched; no technical contract altered or reinterpreted; no source, test, config, preset, benchmark manifest or run artifact touched; no run; no ref moved, deleted or repurposed — the preserved scientific / reference refs `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final` and `pre-cleanup` are UNTOUCHED** |
| 2026-09-05 | **THE FIRST FULL GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN IS EXECUTED, INDEPENDENTLY REVIEWED AND `APPROVE — VALID MEASUREMENT`** at measured code SHA `4af6c5aa5dd28072692bfda63282964b55010aae` (§3n.1; `CLAUDE.md` §7 owns the authoritative record): 375/375 iterations and PPO updates, 3000 successful training episodes from 3045 attempts with 45 ordinary accounted `setup` failures ALL deterministically replaced, **ZERO integrity aborts**, 16 evaluation rounds, 864/864 benchmark members successful, 18/18 COMPLETE matched groups in EVERY round, and `accounting_reconciled = true`; a FIXED-BUDGET actor-only run with NO early stopping and NO CTDE arm, against the frozen comparator `manifest_id 0e15f007…fc0d` | **The FIRST scientifically valid GENERALIZED-V1 measurement, and its PRIMARY FD FINDING IS NEGATIVE** — no severity-conditioned mild-vs-severe learning; a GLOBAL move from ABORT toward PLAN across checkpoints while matched mild and severe worlds were treated almost identically. **A VALID NEGATIVE RESULT, NOT A VALIDITY DEFECT**, and **NOT** grounds to re-tune, re-seed, repair, resume, extend or re-run — **NO RERUN, REPAIR, RESUME, EXTENSION OR RETUNING IS AUTHORIZED.** It is **ONE** measurement: **NOT** a five-run population result and **NOT** an actor-only-vs-CTDE comparison, and **no CTDE benefit or deficit is established or may be pre-claimed.** **This SUPERSEDES, as CURRENT state only, every "R1 is `AUTHORIZED / DISPATCHED — RESULT PENDING`" and "NO GENERALIZED SCIENTIFIC MEASUREMENT RESULT EXISTS" statement**, each of which stays accurate as the record it was |
| 2026-09-05 | **THE DIAGNOSTIC REPLAY IS RECORDED AS ENGINEERING / ANALYSIS EVIDENCE** (§3n.2), bundle `SHA-256 812ff43322e134e9a7ca31720007393ff1220ba50c35955b2a724b30d4d5d792`: REPLAY EQUIVALENT TO R1 on 108/108 actions, event ticks and ego ids; `fuel_norm` materially different in ALL 54 matched pairs; `reachable_by_ego` flipped in ALL 54; selected meta-action changed in **0/54**; mean absolute matched aggregate P(ABORT) delta `0.0001177037203753436`; joint-vs-aggregate argmax disagreement 54/108; mean task-distance clipping 98.15 %; normalized joint entropy still HIGH | **ENGINEERING / ANALYSIS EVIDENCE, NEVER A SECOND MEASUREMENT** — it schedules no population, defines no comparator and produces no verdict, and **no reward, learning or performance claim may be drawn from it.** **ACTION ALIASING AND WEAK ROUTE-RELATIVE OBSERVATION CONTEXT ARE SUSPECTS, NOT CAUSALLY PROVEN EXPLANATIONS**: it narrows where to look and authorizes no representation change, observation-feature change, normalizer change, new `MetaAction` or retuning |
| 2026-09-05 | **THE DURABLE PER-WAKE FD POLICY DIAGNOSTICS LAYER IS IMPLEMENTED, REVIEWED `APPROVE` (Grade A, `GPT_GITHUB`) AND INTEGRATED — PR #52**, approved candidate `81a148f80317499d8897db44bd713976962db832` → merge `28eb8dad2643fc79d516b47ec95119a395e76257`, ordered parents `44530abb1cc3f99d01ac867c6621047ac9343661` then `81a148f8…`, integrated tree `86c3b04d104d38c6d6fc5c1e2bdda3bb5c1ab9b7` IDENTICAL to the reviewed candidate's; a cumulative FOUR-COMMIT append-only review chain over SEVEN files (§3n.3; contract in `CLAUDE.md` §5, routed §6, locked §7) | Future runs record per-wake actor diagnostics AT THE DECISION (episode-outcome schema v3 + wake-diagnostics schema v1), so the questions the R1 replay had to answer offline are answerable from durable artifacts — RAW per-wake records PERSISTED in `episode_outcomes.jsonl`, DERIVED summaries in `run_summary.json` and DERIVED plotting input for the figures. **REPORTING-ONLY: reporting consumers read it to persist and summarize it, but no acting, mask, belief, command, PPO/CTDE input, advantage, reward, optimizer, early-stopping, evaluation-scheduling or checkpoint-control path reads it back**; probabilities come from the actor's OWN shared `_masked_dist`; **no RNG draw, no gradient, no control path**; the three wake kinds are DISJOINT and tagged at the TRIGGER; train / pre_update / post_update stay SEPARATE populations; legacy v2 artifacts stay truthful; `fd_policy_sensitivity.png` is OPTIONAL and evaluation-only and `_PLOT_FILENAMES` still names exactly the three REQUIRED figures. **PR #52 produced NO scientific measurement and did NOT modify R1, its artifacts or its verdict** — R1's artifacts are schema v2 and carry no `wake_decisions` |
| 2026-09-05 | **THIS R1-REVIEW + FD MEASUREMENT-HARDENING DOCUMENTATION LOCK IS OPENED**, because merging PR #52 and completing the R1 review made both documents' present-tense claims stale: they still said R1 was `AUTHORIZED / DISPATCHED — RESULT PENDING`, that NO generalized scientific measurement result existed, that no concrete R1 manifest had been reviewed as the comparator, and that the ONE next action was independent GPT artifact review of R1 — and they carried no contract, routing or lock for the merged PR-#52 layer | Documentation only, exactly two files, **no source, test, config, preset, manifest, artifact or workflow change, and no training, benchmark generation, replay, resume, repair, BONMIN run, CTDE work or scientific execution.** **THE NEXT ACTION CHANGES**: R1's review is DISCHARGED, and the ONE next thread is DESIGN / RESEARCH on **global-action representation, route-relative observation context and bounded cluster validation** — which must be EXPLICITLY opened and authorized and which this record neither opens nor schedules. **Once this record is integrated NO writable repository task remains and NO new task becomes implicitly authorized**; **five full cluster runs, a CTDE arm, resume / repair and ANY R1 rerun remain UNAUTHORIZED**; `p(destroy)` stays `1.0`; and `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final` and `pre-cleanup` remain PROTECTED and NEVER cleanup-eligible |
| 2026-09-06 | **THE MATCH-AOU DETERMINISTIC-`p=1` SOLVER AND ITS EXPLICIT BACKEND SEAM ARE IMPLEMENTED, REVIEWED (Grade A, `GPT_GITHUB`) AND INTEGRATED — PR #54**, approved candidate `8f0d250cd9f96e6b8bce635065701dc47a5ee87e` → merge `9979910a0537e829f1d18483011e4d0fab42c257`, ordered parents `fd0d668d5031adef1f3b6af612e584f9ab56454b` then `8f0d250c…`, integrated tree `9507dc0bc16aeeabf5616171e10f5a28480063ec` IDENTICAL to the reviewed candidate's; the approved ISOLATED-SOLVER ancestor is `1462163277322a3ef29eec28c782766edb8ea73b` (§3o.1; contract in `CLAUDE.md` §5, routed §6, locked §7) | WHICH MATCH-AOU objective a run solves becomes an EXPLICIT, INDEPENDENT selector over exactly two ids — `legacy_minlp_v1` (the historical DEFAULT, frozen MINLP through BONMIN) and `p1_milp_v1` (deterministic `p = 1` MILP through SciPy/HiGHS, no EPSILON). **No `auto`, no fallback in either direction, one backend per episode, unknown id RAISES**; the P1 solver is LAZY-loaded and NOT re-exported through `match_aou.solvers`; `MatchAouBackendError` ABORTS rather than becoming attrition; the preflight uses the SAME backend as the later run and there is **no manifest schema change** — reconstructed frozen identity stays the enforcement boundary; valuation is objective-coherent while **the reward formula, `U_prefix`, `U_post`, realized utility, the aircraft penalty, `eps_regret`, terminal credit placement and the no-clamping policy are UNCHANGED**. **P1 IS NOT A TRANSPARENT SPEED/PERFORMANCE SWAP**: it removes the legacy EPSILON stacking incentive, so it can change `A_init`, the hidden geometry, feasibility and POPULATION IDENTITY. **NO SOLVER EQUIVALENCE AND NO ONE-CONFIG-FIELD EXPERIMENTAL EQUIVALENCE IS CLAIMED, NO SCIENTIFIC MEASUREMENT WAS PRODUCED, AND NO P1 PERFORMANCE OR BENEFIT MAY BE PRE-CLAIMED** |
| 2026-09-06 | **THE CERTIFIED-FD PHYSICAL-STATE INTEGRITY REPAIR IS IMPLEMENTED, REVIEWED (Grade A, `GPT_GITHUB`) AND INTEGRATED — PR #55**, first candidate `930987c7bdc19596383a4c4b825f064817812375` → **REQUEST FIXES** → FINAL approved candidate `d36e1338aaac0d55dd081b788a3e8bbcaa310b53` → merge `edf9e840a30a4a4c3b2ef6daa319661c1d6f3cb8`, ordered parents `9979910a…` then `d36e1338…`, integrated tree `0e3c0ff8bc41e5d1d96af9ec3d61a4b5cea59afa` IDENTICAL to the reviewed candidate's; **the requested fix concerned the P1 HISTORICAL-SURFACE TEST, not FD production semantics** (§3o.2; contract in `CLAUDE.md` §5, routed §6, locked §7) | Setup-time certification stays TICK-AWARE and byte-unchanged, while LIVE validation binds **only** the ego's PHYSICAL state — position against the certificate's existing `position_tolerance_km` and pre-damage fuel against its existing `fuel_tolerance`, **NEITHER widened and NEITHER made dynamic** — and **the ABSOLUTE OUTER TICK becomes DIAGNOSTIC ONLY**, because pre-existing frozen-BLADE live-list mutation can skip an airborne ego's whole update. Every delta is computed before any verdict and all three are reported; a genuine physical contradiction still raises `FuelDamageIntegrityError` BEFORE the fuel mutation; world acceptance, certificate construction, the terminal certified-damaged-event-never-realized abort and ordinary `NO_FD_ELIGIBLE_EGO` attrition are unchanged. **BLADE IS UNCHANGED — this is NOT a physics fix.** The P1 surface proof is now pinned between TWO HISTORICAL COMMITS rather than against `HEAD`, so it preserves the PR-#54 scope without prohibiting future evolution. **ENGINEERING EVIDENCE ONLY (reported 659 passed / 11 skipped / 0 failed, plus a bounded seed-740322 reconstruction / replay); NO scientific P1 run was launched or resumed** |
| 2026-09-06 | **ONE ATTEMPTED FULL P1 ARM IS RECORDED AS `ABORTED / DO NOT RESUME`** — aborted during training by `FuelDamageIntegrityError` (§3o.4) | **IT IS NOT A COMPLETED SCIENTIFIC MEASUREMENT**, carries no verdict, and **MUST NOT be resumed, repaired, continued or extended and then silently treated as one**; no reward, learning, attrition or comparison number from it may be reported. **ITS ROOT CAUSE IS CLOSED**: two skipped engine updates before the certified event left its PHYSICAL state correct (position ~7e-11 km, fuel ~6e-9 lbs from the certificate) while its OUTER TICK was late (certified 914, observed 916) — **an instrument premise, not a world fault**. A future fresh P1 full run is a **NEW measurement under the repaired instrument** and is **NOT launched by this record** |
| 2026-09-06 | **THIS P1-BACKEND + CERTIFIED-FD POST-INTEGRATION DOCUMENTATION LOCK IS OPENED**, because merging PR #54 and PR #55 made both documents' present-tense claims stale: they carried no contract, routing or lock for the MATCH-AOU backend, they still described the ABSOLUTE OUTER TICK as binding at live certified-FD validation, and they recorded no aborted P1 arm | Documentation only, exactly two files, **no source, test, config, preset, manifest, artifact, ref or workflow change, and no training, benchmark generation, preflight, replay, resume, repair, BONMIN run, CTDE work or scientific execution.** **THE NEXT ACTION CHANGES**: the immediate step is repository cleanup / handoff, and the next SCIENTIFIC thread is a **FRESH P1 FULL-ARM orchestration in a NEW chat under the repaired instrument**, which must be EXPLICITLY opened and authorized and which **this record does not launch**. **Once this record is integrated NO writable repository task, NO active candidate and NO active scientific run remains, and NO new task becomes implicitly authorized**; **R1 is UNTOUCHED, is not rerun, and no P1-vs-R1 conclusion exists**; resume / repair of the aborted P1 arm, ANY R1 rerun, five full cluster runs and a CTDE arm remain UNAUTHORIZED; `p(destroy)` stays `1.0`; and `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final` and `pre-cleanup` keep their EXISTING roles, remain PROTECTED and are NEVER cleanup-eligible |
| 2026-09-12 | **GENERALIZED-V2 — THE TWO-STAGE ROUTE-RELATIVE POPULATION — IS IMPLEMENTED, REVIEWED (Grade A, `GPT_GITHUB`) AND INTEGRATED — PR #57**, reviewed final candidate `a27a3b140248e95096db38c8d5f717cc0098da4d` → merge `f98b293ededf7f67fbbd8f742e797e08109c254b`, ordered parents `ae1941035991df4719df212c4b5dd07db89aee4a` then `a27a3b14…`, integrated tree `32bc0460cfa1fa2a445dc7b0081a8f085f2c7814` IDENTICAL to the reviewed candidate's; a FOUR-COMMIT append-only review chain (`3a6653a` → `aa9677e` → `7a7f41e` → `a27a3b14…`) over TEN files, with no config, preset, benchmark manifest or documentation file touched (§3p.1; contract in `CLAUDE.md` §5, routed §6, locked §7) | A THIRD episode-design bundle beside `fixed_cell_v1` and `generalized_v1`, resolving the **IDENTICAL four low-level policy ids** as V1 and changing **only the POPULATION**: `A ~ U{2,3,4,5,6}` and `K \| A ~ U{A, A+2}` BEFORE the known-only solve, then `H_requested ~ U{1..R}` AFTER it against the REALIZED routed-ego count `R >= 1`, on TWO disjoint SHA-256 seed domains of its own. **`H` IS NOT KNOWN BEFORE THE ALLOCATION** and the request is never rewritten; `H <= R` because the LOCKED `bounded_backoff_v1` places at most ONE hidden target per ego route, which is **NOT a promise that `H_realized == H_requested`**, and **multi-hidden-per-route stays out of scope, unimplemented and unapproved**. The MATCH-AOU backend selector stays EXPLICIT with no `auto` and no fallback but is **NOT fully orthogonal to `episode_design`**: `generalized_v2` REQUIRES `p1_milp_v1` and REFUSES `legacy_minlp_v1` before execution. `EpisodeDesign.generalized` now means V1 **or** V2, so a V1-only check must say `generalized_v1_design`. The successful-episode quota and its REQUIRED bounded budget apply to BOTH generalized designs. Failure provenance is STAGE-AWARE and WRITE-ONCE through the caller-owned `RouteRelativePopulationRecorder`, carried VERBATIM and **NEVER re-derived**, with historical records growing no key and the outcome schema staying at **version 3**. **NO observation / action / mask / reward / PPO / GAE / CTDE / trigger / executor / solver / BLADE semantics changed; `p(destroy)` stays `1.0`; no new `MetaAction`.** **IT PRODUCED NO SCIENTIFIC MEASUREMENT, NO V2 BENCHMARK AND NO V2 POLICY-PERFORMANCE RESULT, and the Grade-A implementation grade must NOT be projected onto any future V2 measurement** |
| 2026-09-12 | **GENERALIZED-V2 DEFINES NO EVALUATION CONSTRUCT, DELIBERATELY, AND ITS BENCHMARK / EVALUATION DESIGN BECOMES THE SINGLE NEXT UNRESOLVED RESEARCH TASK** (§3p.4) | Four independent refusals, **none of which substitutes another population**: `validate()` REFUSES a `benchmark_manifest` under V2 and REFUSES evaluation outright (the fixed held-out band carries no stratum); `evaluate()` and `evaluate_benchmark()` RAISE; and the deterministic preflight now tests `cfg.design.generalized_v1_design` rather than `cfg.generalized`. The 18-stratum benchmark, its preflight and the approved `training_reward_plateau_v1` stopping rule all stay **`generalized_v1`-only**. **NO V2 evaluation population, stratification, LOW/HIGH interpretation, matched clean/mild/severe construction, worlds-per-cell scale, seed band or manifest identity exists, is defined or may be inferred.** The next chat / orchestrator **MUST BEGIN READ-ONLY** and decide that design BEFORE any scientific V2 comparison run; **naming the task is not authorization to execute anything** |
| 2026-09-12 | **A FRESH DETERMINISTIC-P1 FULL ARM WAS SUBSEQUENTLY COMPLETED AND INDEPENDENTLY ACCEPTED AS A VALID MEASUREMENT, WITH A NEGATIVE PRIMARY MILD-vs-SEVERE RESULT** (§3p.2), **at measured code SHA `ae1941035991df4719df212c4b5dd07db89aee4a`**, recorded here as the MINIMAL provenance for the V2 design decision and **not** as archival run reporting — **beyond that measured SHA no denominators, artifacts, metrics, bundle hashes, run directories or timing tables are recorded, and none may be invented** | **THIS SUPERSEDES, AS CURRENT STATE ONLY, the 2026-09-06 statements that no approved P1 measurement exists and that a fresh P1 full-arm orchestration is the next scientific thread** — each remains accurate as the record it was. It is a **SEPARATE run** from the earlier attempted arm, which remains **`ABORTED / DO NOT RESUME`** and **NOT a measurement**. The P1 solver made **reference solving computationally negligible**; a later **READ-ONLY construction audit** established, **in the audited current V1 domain ONLY**, that hidden shortfall was structurally explained by **ROUTE-COUNT CAPACITY (`no_route`)** rather than by hidden-placement geometry — an **observation over an audited domain, never a universal mathematical guarantee** — and engineering scaling showed P1 stayed cheap beyond `A = 4` **without those larger cells thereby being scientific GENERALIZED-V1 measurements**. **Supported V2 training cardinality stops at `A <= 6`; `A = 8` / `A = 10` are ENGINEERING SCALING EVIDENCE ONLY.** **R1 AND THE FRESH P1 ARM ARE DISTINCT REPOSITORY / POPULATION MEASUREMENTS AND ARE NOT A CLEAN CAUSAL SOLVER-QUALITY COMPARISON** — **R1 at `4af6c5aa5dd28072692bfda63282964b55010aae`, the fresh P1 arm at `ae1941035991df4719df212c4b5dd07db89aee4a`**, differing in the deterministic `p1_milp_v1` allocation semantics versus R1's legacy objective and in everything else merged between those two repository states, with allocation semantics FEEDING route-relative construction so the frozen worlds / populations are **not identical**: no solver equivalence, no one-config-field experimental equivalence, and **no causal reward-difference or solver-quality inference is authorized**. **The earlier ABORTED arm is a DIFFERENT, INVALID, `DO NOT RESUME` attempt and must not be conflated with this measured SHA.** **R1 at `4af6c5aa…` is UNTOUCHED, remains `APPROVE — VALID MEASUREMENT`, and is NOT rerun** |
| 2026-09-12 | **THIS GENERALIZED-V2 POST-MERGE DOCUMENTATION / LOCK IS OPENED**, because merging PR #57 made both documents' present-tense claims stale the moment it landed: they carried no contract, routing or lock for GENERALIZED-V2, they still gave `EPISODE_DESIGNS` as a TWO-element tuple, they still described the MATCH-AOU backend as fully orthogonal to `episode_design` with every design able to run under either backend, they still described the generalized attempt quota and budget as `generalized_v1`-only, and they still named a fresh P1 full-arm orchestration as the next scientific thread under old writable ownership | Documentation only, **exactly two files** (`CLAUDE.md`, `graph_rl_project_handoff.md`); **no source, test, config, preset, manifest, artifact, ref or workflow change, and no training, benchmark generation, preflight, replay, resume, repair, solver run, BLADE run, CTDE work or scientific execution.** `CLAUDE.md` gains the **GENERALIZED-V2 §5 contract**, corrected §4 / §5 statements for the three-design selector, the design-constrained backend, the both-designs attempt quota and the V1-only preflight and stopping rule, **four §6 routing rows**, a **§7 lock entry** for PR #57 and a **§8 phase-state correction**. This handoff gains a 2026-09-12 live-state block, a new **§3p**, and updated §1 / §4 / §8 / §9.2. **THE NEXT ACTION CHANGES**: the single next unresolved research task is **GENERALIZED-V2 EVALUATION / BENCHMARK DESIGN**, which the receiving orchestrator must open READ-ONLY and which **this record neither takes, pre-authorizes nor schedules**. **Once this record is integrated NO writable repository task, NO active candidate and NO active scientific run remains, and NO new task becomes implicitly authorized**; any V2 scientific run, ANY R1 rerun, resume / repair of the aborted P1 arm, a new control arm, five full cluster runs and a CTDE arm all remain UNAUTHORIZED; `p(destroy)` stays `1.0`; and `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final` and `pre-cleanup` keep their EXISTING roles, remain PROTECTED and are NEVER cleanup-eligible. **This record's own integrating merge SHA is deliberately NOT named** (`CLAUDE.md` §7 hash convention), so every receiving orchestrator resolves live `main` from GitHub (§9.1) |
| 2026-09-13 | **THE GENERALIZED-V2 FROZEN TEN-CELL BENCHMARK AND EVALUATION CONSTRUCT IS IMPLEMENTED, REVIEWED (Grade A, `GPT_GITHUB`) AND INTEGRATED — PR #59**: original base `fe7b449c94281bb12fdadc12be89ee36f447c79a`; first candidate `735e1fd89c230590b75fcaf506a7f470c92b2dc9` → **REQUEST FIXES**; append-only final approved candidate `786e8218a00954f7a7f20fe1dfca93ec71a400d4`; NORMAL merge `ea8778d5010fcfccec357c57c2861606ecb58bbc` preserving it as SECOND PARENT with the IDENTICAL tree `727a65f072ae46d72f61287b5c43dbfda9adf72d`; NINE files, with two AUTHORIZED scope extensions (`graph_episode_setup.py` typed `R == 0` classification; a classification-only change to the locked `graph_hidden_placement.py`) (§3q.1; contract in `CLAUDE.md` §5, routed §6, locked §7) | A SEPARATE V2 schema over exactly TEN exogenous `(A, K−A)` base cells, **no LOW/HIGH**, and `R` / `H_requested` / `H/R` / `H_realized` as reporting descriptors only; **12 frozen world groups per cell** (120 groups, 360 members if fully evaluated) as matched CLEAN / MILD / SEVERE triads, split into disjoint and exhaustive `development` (0..1 → 20 / 60) and `confirmatory` (2..11 → 100 / 300) profiles; a strong UUID-free frozen identity whose mismatch ABORTS as `BenchmarkIdentityError`; a loader verifying canonical bytes AND producible population state; a **FAIL-CLOSED** preflight replacing ONLY a strict generator placement refusal, `RouteRelativeNoRoutesError`, `BoundedBackoffExhaustedError` and `NO_FD_ELIGIBLE_EGO`, accepting short realizations and blind to `R`, `H`, reward and behaviour; matched identity-verified evaluation with no runtime substitution; whole-manifest held-outness against the maximum training-attempt band; V2 early stopping still refused; and the primary **SEVERE − MILD aggregate `SELF_PRESERVATION_ABORT` mass at the certified ego's immediate-FD wake**, paired per group, per `(A, D)` cell, equal-weight macro over ten cells, kept apart from the selected-action switch rates, with reward secondary and no inference procedure implemented. **The PR #57 population contract is unchanged.** **PR #59 produced NO scientific benchmark manifest, NO real seed selection, NO training run and NO scientific V2 measurement, and its Grade-A implementation approval must NOT be projected onto any future measurement** |
| 2026-09-13 | **THIS GENERALIZED-V2 BENCHMARK POST-MERGE DOCUMENTATION / LOCK IS OPENED**, because merging PR #59 made both documents' present-tense claims stale: they still said GENERALIZED-V2 defines no evaluation construct, that there is no benchmark to hold the V2 training band out from, that the benchmark preflight is `generalized_v1`-only, and that V2 evaluation / benchmark DESIGN is the single next unresolved research task | Documentation only, **exactly two files** (`CLAUDE.md`, `graph_rl_project_handoff.md`); **no source, test, config, preset, manifest, seed selection, preflight, training, evaluation or other scientific execution, and no ref change.** `CLAUDE.md` gains the **GENERALIZED-V2 BENCHMARK §5 contract**, corrected §5 statements on the dispatch predicates, the held-out band and the V1 preflight path, **four new §6 routing rows** replacing the "what does GENERALIZED-V2 evaluate? NOTHING" row, a **§7 lock entry** for PR #59 and a **§8 phase-state correction**; this handoff gains a 2026-09-13 live-state block, a new **§3q**, and updated §1 / §3p / §4 / §8 / §9.2. **THE CURRENT PHASE IS: GENERALIZED-V2 population + benchmark/evaluation mechanism implemented, reviewed, approved and integrated; scientific benchmark population not yet created.** **THE NEXT UNRESOLVED SCIENTIFIC ACTION is the actual V2 benchmark preflight / frozen scientific manifest creation under the implemented construct, and it is NOT authorized** — it needs its own explicit research decision covering the real seed namespace, the preflight invocation and preservation of the frozen manifest. While this record's draft PR is open it is the SOLE WRITABLE task with nothing else authorized concurrently; **once integrated NO writable repository task, NO active candidate and NO active scientific run remains, and NO new task becomes implicitly authorized.** R1 is untouched and NOT rerun; the fresh P1 arm is historical context, not a clean causal comparator; `A = 8` / `A = 10` stay engineering-only. **This record's own commit and merge SHAs are deliberately NOT named** (`CLAUDE.md` §7 hash convention), so every receiving orchestrator resolves live `main` from GitHub (§9.1) |
| *(undated here; executed at measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`)* | **A GENERALIZED-V2 benchmark preflight and two development-profile R1 arms (actor-only and CTDE) were EXECUTED**, as established by the evidence refs of PR #61 (`1375a881637a9a32721a1630f598adc571422a47`) and PR #62 (`b2bbe7a6235c3b9255106826cfb268af7e73f72d`); the explicit research authorization for them is **not recorded in the repository** | Recorded in the 2026-09-14 restructure from artifacts only ([`measurements.md` §7](measurements.md#7-generalized-v2-development-r1-arms)). It supersedes, as current state only, the 2026-09-13 statements that no V2 manifest, V2 training run or V2 evaluation result exists. Verdict provenance: CTDE — a prior GPT verdict recorded in PR #62's body and `artifact_sha256.txt`, not independently verified; actor-only — no verdict recorded in accessible artifacts. No comparison conclusion is recorded |
| 2026-09-14 | **DOCUMENTATION RESTRUCTURE OPENED** on branch `docs/project-guidance-restructure` under a user-approved packet: `CLAUDE.md` becomes a short mandatory entry point; technical contracts, workflows and history move to `docs/`; the handoff becomes a compact current snapshot; the live transport is `GPT_GITHUB` only, several focused commits are allowed, Grade C no longer exempts a change from exact-candidate review, and code and its documentation change in the same PR | Documentation only — no source, test, config, preset, evidence, frozen-engine change and no run. The candidate is unreviewed until GPT approves its exact SHA; no merge is authorized. Every moved block and every procedural supersession is listed in [`documentation_migration.md`](../documentation_migration.md) |
| 2026-09-15 | **GENERALIZED-V2 DEVELOPMENT DOCUMENTATION CLOSURE OPENED** on branch `docs/generalized-v2-development-closure`, from live `main` `b28df02326026e99d9fc6977ad878af6149bcc41`. The documentation restructure (PR #63) is merged there | Documentation only: no code, test, config, contract, workflow, artifact or evidence change, and no run. It records the development findings in [`measurements.md` §8](measurements.md#8-generalized-v2-development-closure) and the decisions below. PRs #61, #62, #64 and #65 are read-only inputs. **Supersedes as current state only** the 2026-09-14 row's "no merge is authorized" and the undated V2 row's "no comparison conclusion is recorded" |
| 2026-09-15 | **GENERALIZED-V2 DEVELOPMENT INTERPRETATION CLOSED** on actor-only R1, CTDE R1, the three CTDE diagnostic arms (`smallbatch`, `largebatch`, `fd80`) and the full matched-wake analysis (PR #65 @ `d565174e…`) | The current **development** interpretation: the actor receives strong severity-dependent information; simple lack of FD exposure and simple batch-size mismatch are strongly weakened as explanations; CTDE alone did not produce the target separation; **repeated extra training or tuning of the same basic knobs is not the next research action**; **no confirmatory evaluation is authorized or appropriate yet**. No hypothesis is claimed mathematically impossible. Evidence: [`measurements.md` §8](measurements.md#8-generalized-v2-development-closure) |
| 2026-09-15 | **HYPOTHESIS STATUS RECORDED** | **Strongly weakened:** insufficient FD sample exposure; an ordinary PPO batch too small or too large as the main explanation; "just train longer"; absence of any actor-visible severity signal. **Direct structural finding, causal role unresolved:** node-indexed joint-cell selection can disagree materially with aggregate semantic meta-action preference, because `PLAN` and `ABORT` semantics are duplicated over several cells (§8.7). **Still plausible or unresolved:** route-relative observation quality (absolute fuel is available and binary reachability is strongly severity-sensitive, but absolute task distance is heavily clipped, and reachability is still the documented conservative round-trip placeholder rather than remaining-route slack); credit assignment (CTDE's critic / GAE was not sufficient, and per-immediate-FD advantage separation was never recorded, so it remains unmeasured) |
| 2026-09-15 | **NEXT RESEARCH TASK: GENERALIZED-V2 ACTION-REPRESENTATION RESEARCH DESIGN** — analysis and design only | The design question: can the actor keep node-local identity where it is semantically required (`OPPORTUNISTIC_ENGAGEMENT`) while giving effect-global actions such as `PLAN` and `ABORT` a single semantic selection identity, or otherwise remove the demonstrated joint-cell / aggregate mismatch, without weakening the no-communication contract? The design must consider the action mask, sampling, deterministic selection, stored PPO action identity, log-prob and re-scoring, checkpoints and compatibility, the transition schema, and diagnostics and the benchmark endpoint. It also decides whether to add per-transition advantage instrumentation before the next scientific development run. **Authorizes no code change, no contract change and no training or evaluation run** |
| 2026-09-15 | **CONFIRMATORY PROFILE REMAINS UNTOUCHED**; the project stays in development-profile iteration | The confirmatory profile is not used until all three hold: an intervention is selected and development-validated; the benchmark-provenance gap relevant to scientific interpretation is closed ([`measurements.md` §8.10](measurements.md#810-benchmark-provenance-boundary-and-non-claims)); and an explicit bounded confirmatory plan is authorized |
| 2026-09-15 | **ARTIFACT / GIT LIFECYCLE DECISION** (the user's process decision) | Full run artifacts are to be retained in an organized local artifact and run structure; establishing that organized local structure is part of the later explicit cleanup task. Raw run packages are not meant to become permanent large Git branches. Git review branches are temporary transport for the minimum evidence GPT needs. Once review conclusions are durably recorded in repository documentation **and cleanup is explicitly authorized**, the temporary and evidence PRs and branches (currently #61, #62, #64, #65) are closed and deleted safely, so they are not planned for merge into `main`. Future review normally uses one temporary review branch or task at a time. **This records the intended lifecycle only: it performs no deletion and authorizes none.** The protected-ref registry ([`environments_cleanup.md` §4](../workflows/environments_cleanup.md#4-authorized-cleanup)) changes only in the later cleanup task that actually removes them |
| 2026-09-15 | **V2 BENCHMARK-PREFLIGHT PROVENANCE RECOVERY REVIEWED** — the preserved preflight artifact is reviewed for provenance at temporary review PR #67, exact candidate `7f56338cde6aacfa59a52399b2378b98a62ea3aa`, verdict `APPROVE — provenance package correctness / evidence review` ([`measurements.md` §8.11](measurements.md#811-generalized-v2-benchmark-preflight-provenance-review)) | Supersedes the earlier current-state claim that the producer SHA is unknown ([`measurements.md` §8.10](measurements.md#810-benchmark-provenance-boundary-and-non-claims)). The producer-recorded exact code SHA is `ae42cb01677f94868b2873008d87be677e31f0c8` and the producer-recorded checkout was clean (`dirty = false`, `dirty_path_count = 0`) — provenance emitted by the executing project code, not an external attestation. Manifest identity and static integrity are reviewed. The exact original argv remains unknown. **Historical research authorization and historical prior review remain not preserved / not proven; the preflight is not retrospectively authorized.** The review is not a scientific-validity verdict. PR #67 is temporary transport, not planned for merge |
| 2026-09-15 | **FUTURE CONFIRMATORY USE OF MANIFEST `ef17a68a…46ea8`** — refines the benchmark-provenance condition of the "CONFIRMATORY PROFILE REMAINS UNTOUCHED" entry above | The missing historical authorization record is **not repaired by retrospective wording**. Before any future confirmatory-profile execution, the explicit bounded scientific plan ([`experiments.md` §2](../workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)) must: name this exact manifest identity (`manifest_id ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8`, file SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103`); decide explicitly whether this already-frozen manifest is adopted for that confirmatory evaluation; preserve the development / confirmatory separation (development ordinals 0–1, confirmatory ordinals 2–11); and acknowledge the historical authorization-record gap. Regeneration is not required merely because that record is missing, unless a future research-design decision determines it is necessary. The other conditions of the entry above are unchanged. **Authorizes no confirmatory execution now** |
| 2026-09-15 | **LOCAL ARTIFACT ARCHIVE COMPLETED** — 28 artifacts organized under `C:\gra\` by authorized same-volume renames, indexed in `C:\gra\metadata\ARTIFACT_INDEX.jsonl` (SHA-256 `de96d9ba4c04e10c075549d37a1b445a6e513d133ef55591a1849bd0b0b80552`) with move ledger `C:\gra\metadata\ARCHIVE_MOVE_LEDGER.jsonl` (SHA-256 `15287e8fa7b1051f48cd2b4d1d629f61d687c567d0c4858c5248569d8b6f9eb7`) ([`measurements.md` §9](measurements.md#9-local-artifact-archive-closure)) | Implements the local-structure half of the 2026-09-15 lifecycle decision above. Original bytes preserved; **no scientific artifact deleted**, including invalid, aborted, engineering, unclassified and review artifacts; historical paths embedded in the artifacts are **intentionally not rewritten**. The cleanup procedure now permits authorized, indexed, identity-verified same-volume archival relocation while keeping artifact bytes and provenance protected ([`environments_cleanup.md` §4.4](../workflows/environments_cleanup.md#44-preserved-run-directories-and-external-artifacts)). Recovered local identities (first post-B3 probe match, aborted P1 `8f0d250…`, Task 5A `09eab06…`) change no verdict; `ct1` stays unclassified |
| 2026-09-15 | **TEMPORARY GIT EVIDENCE LIFECYCLE COMPLETED** under explicit user authorization ([`environments_cleanup.md` §4.3–§4.7](../workflows/environments_cleanup.md#43-temporary-evidence-and-review-refs)) | Evidence and review PRs **#61, #62, #64, #65 and #67 closed without merge** at their verified heads, and their branches deleted remotely and locally, after their sources were re-verified in the archive and their conclusions confirmed on `main`. Merged branches of **#59, #60, #63, #66 and #68** deleted after ancestry verification. Worktrees `C:/g1src`, `C:/p1src`, `C:/Users/Itama/ct1s` and `C:/Users/Itama/PycharmProjects/fd_variable_severity_v1_bf1e045f_snapshot` removed; `C:/Users/Itama/PycharmProjects/flat-baseline` **retained** because its ignored flat-RL outputs are not archived. Protected refs `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final` and `pre-cleanup` unchanged. Conclusions and evidence identities survive through the permanent repository history, the `C:\gra\` archive and its index, and the protected refs where applicable. Supersedes as current state the lifecycle entry's "currently #61, #62, #64, #65" open-PR list |
| 2026-09-16 | **APPROVED GENERALIZED-V2 ACTION-REPRESENTATION INTERVENTION** (user decision, concluding the 2026-09-15 design task): PLAN_COMPLIANCE and SELF_PRESERVATION_ABORT each get ONE semantic selection identity; OPPORTUNISTIC_ENGAGEMENT stays task-local; the policy acts on ONE `k + 2`-leaf categorical derived from the EXISTING `k × 3` ActionHead scores by count-normalized `logmeanexp` collapse (PLAN over all `k` nodes, ABORT over the abort-legal nodes), identified as `semantic_k_plus_2_logmeanexp_v1` | **Not** a new actor head, **not** a new actor input and **not** `logsumexp` (count normalization is part of the intervention). Stored identity is semantic (`node_v = None` for PLAN / ABORT); PPO / CTDE entropy is the semantic-leaf entropy; the V2 primary endpoint keeps its concept, now reading the one semantic ABORT leaf. Semantic compatibility with historical checkpoints is intentionally broken, with no migration or resume. Total ENGAGE mass still depends on the distinct ENGAGE actions; hierarchical factorization remains a possible future intervention, not authorized. Contract: [`policy_ctde.md` §2](../contracts/policy_ctde.md#2-encoder-action-head-and-selection-stage-4) |
| 2026-09-16 | **APPROVED PER-TRANSITION CREDIT INSTRUMENTATION** before the next development run: a training-only, append-only, versioned `train_credit_diagnostics.jsonl` describing the credit the actual actor-only / CTDE update already computed (returns and baseline; `V_old`, TD residual, GAE advantage and target; normalized advantages), for every trained transition, joined trainer-side to measurement tags | Strictly observational: no extra forward, GAE pass, RNG draw or gradient, and no control path reads it; severity / condition tags never reach an actor, critic, PPO / GAE, mask, reward or optimizer input; a persistence failure stops the run. **Explicit non-scope of both decisions:** reachability, distance normalization / clipping and route-relative observation features; reward; FD physics; the V2 population, frozen benchmark and profile partition; PPO hyperparameters, batch size, training budget and FD exposure; CTDE architecture / features; BLADE; the MATCH-AOU solvers; early stopping; the confirmatory profile. Contract: [`artifacts_metrics.md` §5.1](../contracts/artifacts_metrics.md#51-training-credit-diagnostics) |
| 2026-09-16 | **IMPLEMENTATION OF BOTH 2026-09-16 DECISIONS OPENED** as one Grade-A code + contract task on branch `task/v2-semantic-action-credit-instrumentation` from `63247404d88f714c6268383321ac25d766406055`, transport `GPT_GITHUB` | Supersedes, **as current state and for this task only**, the 2026-09-15 design row's "authorizes no code change" and the handoff's blocked action-representation / credit implementation item. Engineering evidence only (unit and synthetic trainer tests); **no training, evaluation, benchmark preflight, replay or other scientific execution is authorized or run, and the confirmatory profile stays untouched.** The candidate is unreviewed until GPT approves its exact SHA; no merge is authorized |
| 2026-09-16 | **SEMANTIC-ACTION + CREDIT-INSTRUMENTATION IMPLEMENTATION INTEGRATED (PR #70, merge `d4e9f3721e6d151c00be3fe93c3d149df9d31965`), AND ONE SEMANTIC-ACTION ACTOR-ONLY DEVELOPMENT RUN EXECUTED** at that SHA — `graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37`, under its recorded `authorized_plan.json` (SHA-256 `629603a717b9933f7f956c08ef1a802556f6c0076c77bb8bee99ede05e71a467`; the artifact records its source as a user-approved bounded plan transferred via the GPT orchestrator packet): one `actor_only` arm, development profile only, no CTDE arm, no hyperparameter sweep | Supersedes, as current state, the implementation-opening row's "no merge is authorized" and "no training … is authorized or run" for that task only. Evidence preserved at draft PR #71 (not for merge). **The confirmatory profile stays untouched** |
| 2026-09-16 | **SEMANTIC-ACTION ACTOR-ONLY DEVELOPMENT R1 REVIEWED — `APPROVE — VALID DEVELOPMENT MEASUREMENT`** (GPT verdict, as transferred in the user-approved documentation packet), measured code SHA `d4e9f3721e6d151c00be3fe93c3d149df9d31965`, exact evidence candidate `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670` ([`measurements.md` §10](measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1)) | **Final primary endpoint effectively zero:** ten-cell macro `+0.000888290349394083` over 10 / 10 base cells, 0 / 20 directional and 0 / 20 reverse switches — no stable final severity-conditioned behaviour. **Transient separation at updates 75–150 is a real development finding** (macro `+0.296` → `+0.611` → `+0.650` → `+0.576`; 4, 19, 20, 19 of 20 directional switches), which historical actor-only R1 at `ae42cb0…` never showed; **it collapses by update 175** (`+0.0029`, 0 / 20). Cross-version development comparison only — not a contemporaneous control, not confirmatory. The historical action geometry is read as a material bottleneck / contributor, not the proven sole cause; the representation change did not solve the objective. DEVELOPMENT ONLY |
| 2026-09-16 | **ACTOR-ONLY CREDIT INTERPRETATION RECORDED** — an empirical structural result from the persisted credit rows: with `gamma = 1` and a terminal-only episode reward, raw advantage is identical throughout every episode (2999 checked, 0 varying) and every ego chain (6689 checked, 0 varying) ([`measurements.md` §10.7](measurements.md#107-structural-credit-limitation)) | The current actor-only advantage instrumentation does **not** provide local causal credit at the immediate-FD action: that transition's advantage is effectively the episode outcome relative to the batch baseline, so MILD / SEVERE credit differences partly restate episode outcomes. Observed ABORT-vs-not advantage gaps are **descriptive and non-counterfactual**, never action-value estimates |
| 2026-09-16 | **NEXT RESEARCH DIRECTION APPROVED (user decision): DESIGN A SEMANTIC-ACTION CTDE DEVELOPMENT ARM** — to ask whether the centralized training critic + GAE can retain severity-conditioned behaviour now that the action-space bottleneck has been removed | Approves the **research direction only**: **no execution is authorized until a bounded run plan is frozen** ([`experiments.md` §2](../workflows/experiments.md#2-execution-authority--the-authorized-bounded-plan)). No hyperparameter sweep is authorized. The arm must **not** retune observations, reward, FD exposure, batch size or the action representation. No claim that CTDE will succeed. **The confirmatory profile stays untouched.** Reachability and distance representation remain open and are not addressed by this direction |
| 2026-09-16 | **SEMANTIC-ACTION DEVELOPMENT R1 VERDICT DOCUMENTATION RECORD OPENED** on branch `docs/v2-semantic-action-dev-r1-verdict` from `d4e9f3721e6d151c00be3fe93c3d149df9d31965`, transport `GPT_GITHUB` | Documentation only (`measurements.md`, this log, the handoff); no source, test, contract or evidence change; no scientific execution. It is the sole writable repository task while open; PR #71 stays read-only evidence and is not merged |
| 2026-09-17 | **CTDE ACTOR-GRADIENT DIAGNOSTICS INSTRUMENTATION TASK OPENED** (user-approved packet) on branch `task/v2-ctde-gradient-pressure-diagnostics` from `8056266cff89f677911462b29970346bed0a57c1`, transport `GPT_GITHUB`: an opt-in, OFF-by-default, CTDE-only, observational epoch-0 decomposition of the PPO policy-surrogate gradient into `immediate_fd_mild`, `immediate_fd_severe`, `post_fd` and `ordinary`, each group weighted by its real batch mass, plus (GPT review fix request on the same PR) each component's local first-order pressure on the SEVERE-minus-MILD semantic ABORT contrast, to ask whether immediate-FD decisions carry actor-gradient pressure that the rest of the batch cancels | Instrumentation only: no change to training semantics; severity and FD-ego identity stay trainer-side, and only opaque integer group ids reach the updater. **Explicit non-scope:** action-conditioned subgroups, actor-only instrumentation, critic features, reward, FD events, training distribution, batch weighting / oversampling, learning rate, lambda, entropy, batch size, early stopping, BLADE, the solvers and the follow-up experiment. **No scientific execution is authorized or run**; evidence PRs #71 and #73 stay read-only. Contract: [`artifacts_metrics.md` §5.2](../contracts/artifacts_metrics.md#52-ctde-actor-gradient-diagnostics) |
| 2026-09-17 | **ACTOR-GRADIENT DIAGNOSTICS IMPLEMENTATION CANDIDATE APPROVED** by GPT exact-candidate review at head `6ed964a1abd09de2130aee3d0d314c8f32165056` (PR #74, branch `task/v2-ctde-gradient-pressure-diagnostics`) | Approval of the **exact candidate** only; **unmerged** at approval and while both diagnostic runs below were measured. Later documentation commits on the same PR are reviewed as a new exact head; the merge is performed by the GPT orchestrator under the user's existing authorization, only on an approved exact head |
| 2026-09-17 | **`p = 0.5` CTDE ACTOR-GRADIENT DEVELOPMENT DIAGNOSTIC AUTHORIZED, EXECUTED AND REVIEWED** — `graph_rl_v2_semantic_ctde_grad_diag_r1_seed3000000_6ed964a` at measured SHA `6ed964a1…`, under its recorded `authorized_plan.json` (SHA-256 `d157f37345dc6486256896d25bc52b149e00bb39ae3afbfd68f7d8d4a9f11330`; source recorded as a user-approved bounded DEVELOPMENT plan transferred via the GPT orchestrator packet): one CTDE arm, 150 updates × 8, frozen semantic CTDE configuration with the diagnostic on. GPT verdict **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** ([`measurements.md` §11](measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics)) | **Historical authorization scope:** that one run only — no actor-only arm, comparator rerun, hyperparameter sweep, resume / warm start, benchmark preflight, evidence branch or confirmatory access; it is **spent** and authorizes nothing now. Finding: no meaningful severity-conditioned behaviour; FD separation pressure positive in 56 / 120 defined updates; **simple non-FD cancellation is not supported as the dominant explanation** (19 / 120 FD-positive-with-non-FD-negative; 14 / 18 FD-negative in the 75–99 window) |
| 2026-09-17 | **FD100 INTERVENTION DIAGNOSTIC AUTHORIZED, EXECUTED AND REVIEWED** — `graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a` at measured SHA `6ed964a1…`, under its recorded `authorized_plan.json` (SHA-256 `99571b95b09bc93021234d4675915720dbe9ec12b692ee3f94b816ffa3320cf0`; source recorded as a user-approved bounded DEVELOPMENT intervention arm transferred via the GPT orchestrator packet): the `p = 0.5` diagnostic with the single intervention `fuel_damage_probability 0.5 → 1.0` and a fresh output directory. GPT verdict **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** ([`measurements.md` §11](measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics)) | **Historical authorization scope:** that one run only — no manual stratification, oversampling, loss reweighting or forced severity balance, no other PPO / CTDE / observation / reward / FD-mechanics change, and **no further retuning**; it is **spent** and authorizes nothing now. Finding (development only): FD exposure is a **material contributor to acquisition** (FD pressure positive 98 / 149; held-out macro `+0.0428` at update 75, 0 / 20 switches) but **not to retention** (collapse by update 100); `value_old` stays severity-insensitive; non-FD interference is visible after acquisition but not established as the sole or primary cause; no causal attribution |
| 2026-09-17 | **FURTHER TUNING STOPPED; NEXT ACTION IS A READ-ONLY MECHANISM AUDIT** (user-approved documentation packet, on the GPT review of both diagnostics) | **No additional tuning run is approved**; in particular no approval is inferred for greater-than-100%-equivalent oversampling / replay, stratified loss weighting, batch-size, lambda, learning-rate, entropy or clipping changes, reward shaping, critic architecture or observation changes. Next: a **fresh read-only mechanism audit in a new orchestrator chat** before any implementation or scientific execution, prioritizing actor private-observation identifiability of MILD vs SEVERE, critic conditioning of `value_old`, PPO mechanics (clipping, normalized advantages, repeated epochs, Adam / gradient clipping) and FD / non-FD gradient interaction ([`measurements.md` §11.9](measurements.md#119-unresolved-hypotheses-and-next-research-action)). An investigation list, not a finding that any item is defective; **no run or code change is authorized** |
| 2026-09-17 | **DIAGNOSTIC VERDICT DOCUMENTATION ADDED TO PR #74** (user-approved packet; same branch, append-only commits after reviewed head `6ed964a1…`) | Documentation and a compact Git evidence index (`research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/`: README, SHA-256 manifest, machine-readable summary, standard-library extractor) only; no source, test, configuration or contract change; no scientific execution; both original run directories untouched and authoritative |
| 2026-09-19 | **CTDE ACTING-EGO CRITIC CONDITIONING TASK OPENED** (user-approved packet) on branch `task/ctde-acting-ego-conditioning` from live `main` `adc213670ce4844a7cf60943ecf50150318e40b1` (PR #74 merged), transport `GPT_GITHUB`: the smallest decision-conditioned central-state intervention — every decision capture names its acting ego, whose live node takes the encoder's EXISTING EGO role (`ego_index = k + row`) while every other live agent stays PEER, so the critic values `V(global_state, acting_ego)`. Motivated by the reviewed FD100 diagnostic ([measurements §11](measurements.md#11-generalized-v2-semantic-action-ctde-actor-gradient-development-diagnostics)): `value_old` barely separates severities at the immediate-FD decision while its targets do, and the earlier central state had no distinguished acting agent | Changes the locked CTDE central-state contract ([policy and CTDE §4](../contracts/policy_ctde.md#4-phase-b-ctde)) and nothing else: no new feature or identity embedding; encoder, mean `pool()`, `CentralCritic` / `ValueHead`, critic optimizer, PPO, GAE, reward, action representation and the actor's observation and selection are unchanged; a decision capture without a live acting ego fails closed. Every earlier CTDE measurement describes the earlier symmetric critic. The packet also names ONE bounded development diagnostic run, executable only after GPT approves the exact implementation candidate; opening the task executes nothing |
| 2026-09-19 | **ROLE-ONLY ACTING-EGO DEVELOPMENT DIAGNOSTIC AUTHORIZED, EXECUTED AND REVIEWED** — `graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3` at measured SHA `68055e39768d5fa601e5960a9f08823b9e65c08f` (the GPT-approved PR #75 implementation head), under its recorded `authorized_plan.json` (SHA-256 `ea6752304682a4300b3a5ffd50819a566d6da239fbf7de20e68b9356175f7a19`): the FD100 diagnostic configuration with `n_iterations 150 → 100`, cross-version against `6ed964a1…`. GPT verdict **`APPROVE — VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT`** on the compact evidence package ([`measurements.md` §12](measurements.md#12-generalized-v2-role-only-acting-ego-ctde-development-diagnostic)) | **Historical authorization scope:** that one run only; it is **spent**. Finding (development only): role-only acting-ego conditioning **materially improved critic / credit localization** (within-update `SEVERE − MILD value_old` `-0.02049` versus `-0.0000279`; nonterminal median `|td_residual| / |raw_advantage|` `0.047006` versus `0.0030719`) — the missing acting-ego context was a real critic-conditioning defect — but **did not improve held-out acquisition or retention** (macro update 75 `+0.007641` versus `+0.042798`, update 100 `-0.001091`; 0 / 20 switches throughout). Severity credit stays mostly future-dominated; no sole-cause claim; no confirmatory claim; no training-seed variance estimate |
| 2026-09-19 | **NEXT RESEARCH DECISION: EXPLICIT ACTING-EGO CRITIC READOUT** (user-approved packet, on the review above) | The next minimal mechanism test keeps the role-only central state and changes only the critic readout to `V = ValueHead([global mean pool ; acting-ego post-message-passing embedding])` — no new physical or privileged information. Implemented on the same branch / PR #75 as a new exact candidate; **no scientific run is authorized** until GPT approves that candidate and a separate bounded plan is authorized |

### 1.1 Handoff revisions before the decision log began

The former handoff was written on 2026-08-11 and revised on 2026-08-14 (the final-cell probe
harness closure), 2026-08-15 (the first executed short probe and the three research-validity
defects it exposed), 2026-08-16 (the closure of Defects A, B and C and the corrected-cell short
probe rerun) and 2026-08-17 (the first long baseline, the roster / world-truth defect and its
merged correction). Each of those events has its full record in
[`implementation.md`](implementation.md) or [`measurements.md`](measurements.md).

## 2. Closed decisions

From the former handoff §5.

- Offline construction only: solve → place → patch → reload.
- Route prediction is required and supports `num_agents < n_known`.
- One sensing/arrival/attack/kill-confirmation radius: `DETECTION_KM = 50`.
- `round_trip_cost` and the current p=1 `graph_reward` FORMULA remain frozen.
- B1 reference cell: 3 agents, 3 known, 3 hidden; strict 200 km launch-point distance,
  100 km known-target separation and 0.5 stretch ratio. A reference cell, not a law.
- B2: one placement per non-empty ego route, explicit `random.Random`, id-free geometric
  fingerprints and one-way placement-layer imports.
- B3: explicit construction-path selection; env-2 is the runtime source of truth;
  ordered agent IDs survive reload; exact cardinality; airbase-only cell.
- B4: complete provenance precondition, `skip_and_account_v1`, fixed held-out band,
  explicit denominators, six run artifacts, true pre-update evaluation and disjoint
  all-failed / zero-wake / productive states.
- PR #7: per-episode `OK` blocks, direct unique-target-id counts, no false successful
  zeros from a degraded roster, and disjoint per-round eval artifact namespaces. **Its
  routing of a structural roster fault to an accounted `setup` failure is SUPERSEDED** by
  PR #24 (next-but-one entry).
- PR #24 (roster / world-truth integrity): an ALLOCATION is never a world inventory; the
  world comes from the raw pre-solve `known_target_ids` / `executed_target_ids` snapshots;
  beliefs are a subset constraint, not a denominator; the scheduled cell is verified before
  anything is paid for; and a roster/world-integrity fault is a `MeasurementIntegrityError`
  that ABORTS the run instead of entering `skip_and_account_v1` or any scientific
  denominator.
- PR #8 (FD-BASELINE-v1): fuel damage is the ONE selected difficulty factor; deterministic
  private RNG domain; matched forced-clean / forced-damaged evaluation pairs on the same
  held-out seed; the strict window validated twice (planned, then live before mutation);
  RTB measured from real emitted command history, never from `GraphPlanExecutor.rtb_issued`;
  explicit `aircraft_penalty_coeff = 2.25` with the reward formula unchanged.
- PR #10 (FINAL-CELL-VISUAL-ARTIFACTS): artifact capture is opt-in and OFF by default; it
  selects every scheduled attempt rather than a per-seed subset; the executed t=0 snapshot
  comes from env-2 before the controller and the run; recording is armed only through the
  existing setup/tick-loop contract; artifact failures are infrastructure and stay outside
  the scientific ledger.
- The legacy split surface remains retained, not retired.
- **PHASE A IS CLOSED (§3h).** The approved rerun
  `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` at measured code SHA
  `737b4bf` is the cell's ONE valid baseline; the long baseline is not re-run, resumed,
  repaired or re-tuned; the earlier three runs are history only; and the Phase-A conclusion
  carries its explicit non-claims (no global optimality, no monotonic convergence, no
  generalization beyond this fixed cell and held-out seed set, no CTDE benefit). **It is a
  baseline of the LEGACY FD-BASELINE-v1 design**, and the branch `phase-a-baseline`
  (`4f0068847b017795717c5f0e331f647bcfc30547`) preserving its code state is IMMUTABLE.
- PR #27 (FD-VARIABLE-SEVERITY-v1): the legacy modes are preserved byte-for-byte in
  behaviour; severity is drawn from its OWN `fuel_damage_severity_v1` domain so the legacy
  condition/ego stream cannot shift; MILD leaves continuation feasible and SEVERE does not,
  both measured and validated at the LIVE event state with no clamp, retry, downgrade or
  conversion to clean; evaluation is a matched clean/mild/severe TRIAD whose deltas are over
  COMPLETE triads only; the actor receives NO severity label; successful attempts get one
  durable `episode_outcomes.jsonl` record and the severity-response summary is derived from
  it; and a scheduled-vs-executed CELL mismatch is a `MeasurementIntegrityError` that ABORTS
  rather than entering any scientific denominator. **Target destruction stays deterministic
  at `probability = 1`.**
- **THE FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE IS MEASURED, VALID AND NEGATIVE
  (§3j).** At measured code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, on a detached
  clean snapshot: 664 scheduled / 586 successful / 78 accounted `setup` failures,
  `accounting_reconciled = true`, 7/8 complete triads in all 11 rounds, zero infrastructure
  or data-integrity faults, 50/50 productive PPO updates — and **NO severity-conditioned
  FD-wake meta-action separation** between MILD and SEVERE, at `pre_update` or at the final
  `post_update`. **That is a VALID NEGATIVE SCIENTIFIC RESULT**: the actor is not broken,
  training did not fail, `probability = 1` and every locked contract held, and the result
  is not grounds for retuning, re-seeding or re-running. The severity factor is nevertheless
  PHYSICALLY real (RTB yield, deaths and coverage diverge sharply). The `MAX_PATH` precursor
  is `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT` and excluded. **Its numbers are evidence
  about the VARIABLE-SEVERITY cell only, never about the Phase-A legacy cell, and they
  establish NO CTDE benefit.**
- **THE RESEARCH ORDER WAS DELIBERATELY CHANGED TWICE, MOST RECENTLY TO PARALLEL ON
  2026-08-22 (§4), AND THE PARALLEL PHASE HAS NOW DELIVERED ITS MEASUREMENT.** The
  ADDITIONAL actor-only FD-VARIABLE-SEVERITY-v1 baseline was first ordered BEFORE Phase-B
  CTDE; **that serial rule is SUPERSEDED.** The measurement ran on an immutable detached
  snapshot while **PHASE-B CTDE DESIGN AND IMPLEMENTATION PROCEEDED CONCURRENTLY**, and it
  is now closed and reviewed VALID — so **the CTDE INTEGRATION gate's measurement-validity
  half is SATISFIED, while its remaining half — preserving a NEW immutable actor-only
  pre-CTDE reference — is not.** Phase A stays closed and its reference baseline stays
  immutable; **PHASE B REMAINS CENTRALIZED-CRITIC / CTDE**, its technical requirements remain
  OPEN as REVIEW CRITERIA rather than as a locked design, and decentralized
  no-communication EXECUTION is preserved exactly. *(The design/recon-before-implementation
  ENTRY PATH was a real requirement and is now a TRAVERSED step — an implementation
  candidate exists in PR #30, owned by the CTDE orchestrator, neither reviewed nor approved
  by this record; §4 Task 8 owns the live statement.)*
- **PR #35 (GENERALIZED-V1 Task 1 — cardinality + bounded B2 backoff + accounting):** the
  hidden-CARDINALITY policy is EXPLICIT and never inferred; the historical `exact_v1` path
  is the DEFAULT and is preserved down to its RNG draw order and stream position, because
  the approved measurements were taken on it; `bounded_backoff_v1` is an ADDITION beside it
  that enforces `A ∈ {2,3,4}`, `K == A` RAW known targets and `1 ≤ H_requested ≤ A`, walks
  STABLE AGENT ORDINALS (never uuid text) with per-candidate RNG substreams derived before
  any attempt, reuses the SAME approved single-route B2 geometry through one shared
  leg-selection site, and accepts any `H_realized ≥ 1`; realizing fewer is RECORDED, never
  repaired, and realizing none is a refusal; the seed, world, agent population and requested
  count are never silently altered; requested-vs-realized counts come from the RAW world
  snapshots and are verified, not trusted; and nothing from the policy reaches
  `GraphObservation`. `CLAUDE.md` §5 / §6 / §7 is authoritative.
- **PR #36 (GENERALIZED-V1 Task 2 — certified FD eligibility + post-FD boundary wakes):**
  the LEGACY eligibility and single-wake policies remain the DEFAULTS and the measured
  historical path; certified eligibility gets its OWN versioned RNG domain
  (`fuel_damage_eligibility_v1`) so the legacy condition/ego streams do not move by a single
  draw; candidates are stable scheduled ORDINALS; the walk runs for EVERY condition, CLEAN
  included, so a matched clean/mild/severe group shares one accepted-world support and one
  certified ego; a candidate must support BOTH bands on the SAME ego at a TICK-AWARE event
  state whose pre-event route prefix no legal trigger can disturb, with NO ≥2-assignment
  requirement; SETUP ineligibility (`no_fd_eligible_ego`) is ORDINARY ACCOUNTED ATTRITION,
  while a certified/live contradiction — or a certified damaged episode that ENDS without
  the event firing, checked once at the `run_episode` episode-exit seam before the recording
  export — is `FuelDamageIntegrityError`, an INSTRUMENT fault that aborts the run and never
  enters `skip_and_account_v1`; a certified CLEAN episode may legitimately finish with
  `fired == False`, and LEGACY non-fire semantics are unchanged; only the ACTUALLY damaged
  ego enters post-FD adaptation, its boundaries are ego-LOCAL proximity-gated confirmations
  through ONE shared executor reconciliation site, its belief edit touches only its own
  slice, the reconciliation precedes the CTDE capture so samples stay 1:1, a terminal
  completion produces no wake, simultaneous triggers coalesce into one decision, and NO new
  `MetaAction` exists. `CLAUDE.md` §5 / §6 / §7 is authoritative.

## 3. GENERALIZED-V1 approved design

From the former handoff §3l — the design record GENERALIZED-V1 Tasks 1–5 were built from. Its
status banners describe the implementation state on 2026-08-31; the implemented contracts are in
[`docs/contracts/`](../contracts/) and are decisive.

**READ THIS BANNER BEFORE ANY LINE BELOW IT.** This section is the APPROVED DESIGN. As of
this record:

- **§3l.1–§3l.7 are IMPLEMENTED, REVIEWED and MERGED, and §3l.8 STEPS 1 THROUGH 5 ARE ALL
  COMPLETE, REVIEWED AND INTEGRATED** — Task 1 (candidate `5b55ca34…`, integration
  `9b305e4e…`, PR #35), Task 2
  (final candidate `185d39f0…`, integration `ca0dc406…`, PR #36), Task 3 (candidate
  `24a8b1ee…`, integration `df3abf2f…`, PR #38, **APPROVE**), Task 4 (final candidate
  `db790138…`, integration `b4daa8c1…`, PR #40, **APPROVE**) and **Task 5** (PR #42
  `312f5865…` → `5dfcd8b6…`; PR #43 `4af6c5aa…` → `b3c2e01f…`; documentation lock PR #44
  `88352b2f…` → `9b9e9b85…` — §3m.1, §3m.5). *(SUPERSEDED: this bullet previously read
  "to the extent §3l.8 steps 1, 2, 3 and 4 represent". Accurate at that checkpoint.)*
  **Their technical contracts
  are `CLAUDE.md` §4 / §5 / §6 / §7, which is authoritative for them; the prose below is the
  DESIGN they were built from, not a second contract.** The **FOUR low-level policy seams**
  landed as OPT-IN additions whose DEFAULTS are the historical behaviour, and since Task 4
  both harnesses resolve them TOGETHER — and only together — through the ONE
  `episode_design` selector (`fixed_cell_v1` DEFAULT vs `generalized_v1`), which carries
  EXACTLY those four ids. The generalized harness ADDITIONALLY uses the generalized
  cardinality sampler and requires the SEPARATE `fuel_damage_mode` field to be
  `seeded_variable`; **neither is a fifth policy id on `EpisodeDesign`.**
- **WHAT TASK 4 DID AND DID NOT DELIVER.** Task 4 delivered the frozen benchmark's SCHEMA,
  BUILDER, canonical serialization, content hash, verifying LOADER, CONSUMER and identity
  checks — **the MECHANISM.** **It did NOT select a worlds-per-cell SCALE and committed no
  benchmark POPULATION**, and the builder deliberately refuses to invent a world count.
  **Task 5 then added the PRODUCTION selection caller** (`run_benchmark_preflight`), and the
  **R1 scale IS now selected and its construction IS authorized and dispatched** (§3m.4) —
  while **no concrete R1 manifest has yet been independently reviewed or approved as the
  comparator and no benchmark manifest is committed or tracked in the repository.**
  *(SUPERSEDED, and corrected here: this bullet previously read "WHAT IS STILL NOT DECIDED"
  and asserted that no benchmark population had been scheduled or authorized. Accurate
  through Task 4.)* *(That
  negative is scoped deliberately: transient manifests built by tests and engineering
  validation are legitimate, are neither committed nor a scientific population, and
  repository state cannot establish a global negative over local scratch files.)*
  **§3l.8 step 5 — bounded runtime / solver validation — is IMPLEMENTED, REVIEWED, APPROVED
  and INTEGRATED (PR #42, PR #43 and the documentation lock PR #44; §3m.1, §3m.5), and the
  bounded validation itself was performed and reviewed as Task 5A / Task 5B —
  `APPROVE — VALID ENGINEERING VALIDATION`, engineering evidence only (§3m.3).**
  *(SUPERSEDED: this line previously read "is NOT started and is NOT authorized by this
  record". Accurate at that checkpoint.)*
- **NO GENERALIZED SCIENTIFIC MEASUREMENT RESULT EXISTS, and no generalized result may be
  pre-claimed.** Bounded engineering smokes
  taken during implementation validation are not measurements, and neither are Task 5A /
  Task 5B (§3m.3). **A first generalized ACTOR-ONLY R1 long run IS authorized and dispatched
  with its RESULT PENDING and UNREVIEWED** (§3m.4) — dispatch is not a result, and it is
  neither `RUNNING`, nor `COMPLETED`, nor `VALID`. **No CTDE
  generalized campaign is authorized, none is running or scheduled, and no
  actor-only-vs-CTDE generalized result exists.** *(SUPERSEDED, and corrected here: this
  bullet previously read "NO generalized scenario campaign, no generalized episode schedule,
  no frozen benchmark population and NO generalized measurement of any kind exists, is
  running, is scheduled or is authorized" and "No final actor-only … generalized campaign is
  authorized". Accurate through Task 4; the actor-only R1 and its benchmark construction have
  since been authorized and dispatched.)*

*(SUPERSEDED, preserved as history: this banner previously read "NONE OF IT IS
IMPLEMENTED", then that §3l.5 was NOT IMPLEMENTED, then that §3l.6–§3l.7 were NOT
IMPLEMENTED and that neither harness selected any seam. Each was accurate when written and
no longer is.)*

**THE DESIGN PROSE IS STILL NOT A CONTRACT AND NOT A LOCK.** `CLAUDE.md` is authoritative
for every implemented behaviour; where this section and `CLAUDE.md` differ, `CLAUDE.md` and
the code are decisive. **Any future departure from a contract `CLAUDE.md` §5 locks is a
Grade-A change to a locked layer and must travel the normal recon → prompt → review → lock
discipline routed through `CLAUDE.md` §6** — never a fix folded into another task, and never
authorized by its presence in this plan. *(The departures §3l.1–§3l.7 named — B2 exact
cardinality, the fixed 3/3/3 cell, the FD eligibility and failure policy, the t=0 reference
solve on damaged episodes, and the fixed held-out eval band — travelled exactly that
discipline in PR #35, PR #36, PR #38 and PR #40, and landed as ADDITIONS beside the
historical paths rather than as replacements of them: the historical `exact_v1`,
`legacy_selected_ego_v1`, `single_wake_v1`, `static_t0_v1` and fixed-cell eval-band
behaviours are preserved and remain the defaults.)* **`CLAUDE.md` locks are written PER
COMPLETED TASK, for behaviour that has been implemented, reviewed and integrated** — which
supersedes this section's earlier "only at step 6" rule.

**OWNERSHIP AND AUTHORIZATION.** The **GPT orchestrator owns this work.** **This record
authorizes no further implementation, no benchmark population, no training run, no BLADE
scenario, no BONMIN solve and no measurement of any kind** — in particular it does **NOT**
authorize Task 5 (§8).

**WHY THE REDESIGN EXISTS.** Both approved measurements were taken on ONE fixed cell (3
agents, 3 known + 3 hidden) with a single structurally-severe fuel-damage event and a t=0
reference solve. That cell established learnability and ego-local adaptation (§3h) and then
a valid NEGATIVE severity-separation finding (§3j). What it cannot support is any claim
about **generalization across team size and mission load**, nor a reference against which a
post-event decision is scored on the terms the ego actually faces at the moment of the
event. GENERALIZED-V1 is the approved response: vary the population and the hidden load,
guarantee the damage event is constructible in every accepted world, let the damaged ego
decide more than once, and score the damaged half against an **event-conditioned
continuation reference** instead of against a plan the world has already invalidated.

### 3l.1 Approved generalized cardinality

> **STATUS — IMPLEMENTED / REVIEWED / MERGED as part of §3l.8 step 1 (PR #35, `5b55ca34…` → `9b305e4e…`).** The authoritative technical contract is `CLAUDE.md` §5 / §6, locked in §7; the bullets below are the DESIGN it was built from and are preserved as such. It landed as an OPT-IN policy seam whose DEFAULT is the historical behaviour, so it changes nothing about any existing or historical run. *(SUPERSEDED: this line also read "**no harness selects it**". That was accurate before PR #40; since GENERALIZED-V1 Task 4 both harnesses select it — but ONLY as part of the whole approved bundle, ONLY when `episode_design = generalized_v1` is named explicitly, and NEVER by default.)* **No generalized measurement exists.**

- **`A` (agents) in `{2, 3, 4}`.**
- **`K` = `A` known targets** — the known load tracks the team size.
- **`H_requested` in `{1, …, A}`.**
- **Total requested targets are therefore `A + 1` through `2A`.**
- **Generalized B2 placement uses DETERMINISTIC BOUNDED BACKOFF.** A scenario remains VALID
  with any **`H_realized >= 1`**; realizing fewer hidden targets than requested is a
  legitimate, recorded outcome of the generalized path, not a failure.
- **NEVER silently alter the agent population, the seed, the severity or the requested
  count** to make a world succeed. Silent substitution is what makes a denominator
  unreadable.
- **RECORD both the REQUESTED and the REALIZED cardinalities, and the backoff reasons.** A
  world that realized fewer hidden targets must say so in its own artifacts, so
  requested-vs-realized can be inspected as a distribution rather than inferred.

### 3l.2 Approved B2 / reproducibility direction

> **STATUS — IMPLEMENTED / REVIEWED / MERGED as part of §3l.8 step 1 (PR #35, `5b55ca34…` → `9b305e4e…`).** The authoritative technical contract is `CLAUDE.md` §5 / §6, locked in §7; the bullets below are the DESIGN it was built from and are preserved as such. It landed as an OPT-IN policy seam whose DEFAULT is the historical behaviour, so it changes nothing about any existing or historical run. *(SUPERSEDED: this line also read "**no harness selects it**". That was accurate before PR #40; since GENERALIZED-V1 Task 4 both harnesses select it — but ONLY as part of the whole approved bundle, ONLY when `episode_design = generalized_v1` is named explicitly, and NEVER by default.)* **No generalized measurement exists.**

- **PRESERVE the historical EXACT-CARDINALITY path/version.** The generalized backoff path
  is an ADDITION beside it, never a rewrite of it — the approved Phase-A and
  variable-severity measurements were taken on the exact-cardinality behaviour, and moving
  it would invalidate them rather than extend them.
- **The generalized path MAY realize fewer hidden targets than requested** (§3l.1).
- **Deterministic candidate ordering must derive from STABLE AGENT ORDINALS, not UUID
  lexical order.** Generated ids are not seed-derived (`CLAUDE.md` §8), so ordering by them
  would make placement irreproducible across runs of the same seed.
- **Per-candidate RNG behaviour stays reproducible and ACCOUNTABLE** — an episode's hidden
  geometry must remain a pure function of its seed, and the stream position must be
  explainable rather than incidental.
- **NO new hidden-count feature and NO severity feature may enter `GraphObservation`.** The
  acting path still reads only the ego's own sensing and its own fuel; `CLAUDE.md` §3 is not
  up for renegotiation, and a count of what is hidden is exactly the kind of privileged
  quantity an ego cannot sense.

### 3l.3 Approved fuel-damage (FD) redesign

> **STATUS — IMPLEMENTED / REVIEWED / MERGED as part of §3l.8 step 2 (PR #36, `185d39f0…` → `ca0dc406…`).** The authoritative technical contract is `CLAUDE.md` §5 / §6, locked in §7; the bullets below are the DESIGN it was built from and are preserved as such. It landed as an OPT-IN policy seam whose DEFAULT is the historical behaviour, so it changes nothing about any existing or historical run. *(SUPERSEDED: this line also read "**no harness selects it**". That was accurate before PR #40; since GENERALIZED-V1 Task 4 both harnesses select it — but ONLY as part of the whole approved bundle, ONLY when `episode_design = generalized_v1` is named explicitly, and NEVER by default.)* **No generalized measurement exists.**

> **ONE ITEM BELOW IS NOT PART OF THAT IMPLEMENTATION AND STAYS DEFERRED:**
> `p(destroy) < 1`. Target destruction remains DETERMINISTIC at `probability = 1`.

- **`p(destroy)` REMAINS `1.0`. `p < 1` is DEFERRED** and is not part of this redesign.
- **The training mixture REMAINS 50 % CLEAN / 25 % MILD / 25 % SEVERE.**
- **EVERY ACCEPTED GENERATED WORLD MUST BE FD-CAPABLE — even when the sampled condition is
  CLEAN.** FD capability becomes a property of world ACCEPTANCE rather than something
  discovered later, which is what makes a matched clean/mild/severe group constructible in
  the same world by design.
- **The event point stays at the FIXED 30 % first-leg location initially** — one variable at
  a time; moving it is a separate later question.
- **The damaged ego is selected by DETERMINISTIC BOUNDED ELIGIBILITY FALLBACK**, and
  **eligibility must support BOTH MILD and SEVERE on the SAME selected ego** — otherwise the
  matched group is not one world with one factor varied.
- **NO severity downgrade, NO clean conversion, NO seed replacement, NO invisible
  resampling, and NO unbounded retry.** Each of those silently moves the population every
  per-cell statistic is reported over.
- **LIVE validation REMAINS defensive — and its MEANING changes.** Under the generalized
  design a world is CERTIFIED FD-capable at acceptance, so **a certified candidate that then
  FAILS live validation is a MEASUREMENT-INTEGRITY DEFECT, not normal attrition.** It is not
  ordinary accounted episode failure and must not be booked as one.
- **RECORD the candidates considered, the rejection reasons, the selected ego, the event and
  window quantities, and the severity accounting.**

### 3l.4 Approved repeated post-FD decision semantics

> **STATUS — IMPLEMENTED / REVIEWED / MERGED as part of §3l.8 step 2 (PR #36, `185d39f0…` → `ca0dc406…`).** The authoritative technical contract is `CLAUDE.md` §5 / §6, locked in §7; the bullets below are the DESIGN it was built from and are preserved as such. It landed as an OPT-IN policy seam whose DEFAULT is the historical behaviour, so it changes nothing about any existing or historical run. *(SUPERSEDED: this line also read "**no harness selects it**". That was accurate before PR #40; since GENERALIZED-V1 Task 4 both harnesses select it — but ONLY as part of the whole approved bundle, ONLY when `episode_design = generalized_v1` is named explicitly, and NEVER by default.)* **No generalized measurement exists.**

- **ONLY the ego that ACTUALLY RECEIVES the FD event enters persistent post-FD adaptation
  state.** No peer enters it.
- **The IMMEDIATE FD wake REMAINS** exactly as today.
- **ADDITIONAL wakes occur ONLY for that damaged ego**, and only **after ego-local CONFIRMED
  COMPLETION of its current assignment, BEFORE it commits to the next remaining
  assignment.**
- **NON-DAMAGED egos retain existing executor behaviour and receive NO new wake.**
- **A target killed by a PEER counts as a completion boundary ONLY when the damaged ego
  ITSELF locally reaches / confirms that its assigned target is gone.** That is the
  no-communication rule restated for this seam: the boundary is an ego-local confirmation,
  never a peer's outcome learned some other way.
- **If the target is STILL ALIVE after an attack, NO completion wake occurs.**
- **Simultaneous triggers COALESCE into ONE actor wake.**
- **The ACTION SET IS UNCHANGED: `PLAN_COMPLIANCE` and ego-global
  `SELF_PRESERVATION_ABORT`.** **Do NOT add a trim-tail action or any other new action.**
- **NO peer behaviour change, and NO communication channel of any kind may be introduced.**

### 3l.5 Approved reward / reference architecture — **IMPLEMENTED (§3l.8 step 3, PR #38)**

> **STATUS — IMPLEMENTED, REVIEWED AND MERGED.** Reviewed candidate `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integrated `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, **PR #38**, verdict **APPROVE**, Grade A under `GPT_GITHUB`. **`CLAUDE.md` §5 is now AUTHORITATIVE for this layer's contract** ("GENERALIZED-V1 EVENT-CONDITIONED MATCH-AOU CONTINUATION REFERENCE + REWARD CHECKPOINT"), with the pipeline placement in §4, the routing in §6 and the lock in §7. The bullets below are the APPROVED DESIGN this task implemented and are retained as the design record; where design prose and `CLAUDE.md` differ, **`CLAUDE.md` and the code are decisive**. *(SUPERSEDED: this section previously carried a "NOT IMPLEMENTED / this is the next unresolved task / not authorized" banner. That was accurate when written and is history.)*

> **THREE THINGS THE IMPLEMENTATION SETTLED THAT THE DESIGN BELOW DOES NOT STATE.** (1) The policy is **OPT-IN and selected explicitly** by `setup_episode(..., reference_policy=...)`; the historical `static_t0_v1` remains the DEFAULT. *(SUPERSEDED: this clause also read "**neither harness selects the new one**, so every training run and rollout still uses the historical reference". That was accurate before PR #40; since GENERALIZED-V1 Task 4 both harnesses can select it through `episode_design = generalized_v1`, and a run that does NOT name it still uses the historical reference.)* (2) The design's "TWO BONMIN solves per accepted episode" is implemented as **"never a THIRD reference solve"** — the opt-in policy OCCUPIES the existing second reference-solve slot — and it is **AT MOST two**, because a degenerate reference (no open task, or no continuation-capable ego) legitimately performs no solver call and records `solver_invoked=False`. (3) A DAMAGED-scheduled episode whose event never fired receives a full t=0 reference under its OWN kind, `damaged_event_unrealized_t0` — the one review-APPROVED compatibility deviation, existing solely to preserve the already-locked Task-2 LEGACY contract, unreachable under `certified_both_severities_v1`, and **NOT part of the intended generalized damaged semantics** (`CLAUDE.md` §5).

- **CLEAN episodes KEEP the existing second full MATCH-AOU reference solve at t=0.**
- **DAMAGED episodes DO NOT perform that full t=0 reference solve.** Instead, **immediately
  after the actual FD fuel mutation and BEFORE the actor's FD decision**, the live state is
  taken and the episode's SECOND BONMIN call is performed as an **event-conditioned
  MATCH-AOU CONTINUATION reference.**
- **The utility already realized before the event is preserved as `U_prefix`:**
  - **`U_ref = U_prefix + U_cont_reference`**
  - **`U_achieved = U_prefix + <scored post-FD achieved utility>`**
- **Post-FD scored utility remains RESTRICTED to continuation-reference ALLOCATED tasks**,
  preserving the current selected-task reward convention rather than inventing a new one.
- **Kills outside the reference are ACCOUNTING-ONLY and are NOT reward-bearing.**
- **Do NOT silently clamp reward.**
- **THIS IS EXPLICITLY A MATCH-AOU CONTINUATION REFERENCE, NOT A CLAIMED TRUE PHYSICAL
  ROUTE-OPTIMAL ORACLE.** The frozen solver retains its own independent round-trip movement
  model, so the reference means "what MATCH-AOU would allocate from here", never "what a
  physically optimal route would achieve".
- **The design TARGET remains TWO BONMIN solves per accepted episode** — on a damaged
  episode the continuation solve REPLACES the t=0 reference solve rather than being added to
  it.

### 3l.6 Approved evaluation design

> **STATUS — IMPLEMENTED, REVIEWED AND MERGED as §3l.8 step 4 (PR #40, `db790138…` → `b4daa8c1…`, verdict APPROVE).** **`CLAUDE.md` §5 is now AUTHORITATIVE for this layer's contract** ("GENERALIZED-V1 EPISODE-DESIGN SELECTOR, TRAINING CARDINALITY SAMPLER, FROZEN STRATIFIED BENCHMARK MANIFEST AND RUN-LEVEL PERSISTENCE"), with the routing in §6 and the lock in §7. The bullets below are the APPROVED DESIGN this task implemented and are retained as the design record; where design prose and `CLAUDE.md` differ, **`CLAUDE.md` and the code are decisive**. *(SUPERSEDED: this section previously carried a "NOT IMPLEMENTED / nothing below exists in the repository / this is the SINGLE NEXT unresolved implementation task / not authorized" banner. That was accurate when written and is history.)*

> **WHAT WAS IMPLEMENTED, AND WHAT WAS DELIBERATELY LEFT OPEN.** Task 4 delivered the benchmark **MECHANISM** — the 18-stratum schema, the builder, the canonical serialization, the content hash (`manifest_id`), the verifying loader, the consumer (`graph_train.evaluate_benchmark`) and the id-free matched-world identity checks — plus the `episode_design` selector, the training cardinality sampler and run-level persistence. **It did NOT choose the final scientific worlds-per-cell SCALE and did NOT generate, commit or freeze a concrete benchmark POPULATION**: `build_benchmark_manifest` REFUSES to invent a world count, and no manifest file exists in the repository. **Choosing the scale is §3l.8 step 5's business, and it is NOT started and NOT authorized** (§8).

- **A FIXED STRATIFIED BENCHMARK with three dimensions:**
  - `A` in `{2, 3, 4}`;
  - hidden requested load **LOW = 1** and **HIGH = `A`**;
  - condition in `{CLEAN, MILD, SEVERE}`;
  - ⇒ **18 REQUESTED STRATA.**
- **MATCHED CLEAN / MILD / SEVERE worlds preserve the SAME generated world, the SAME hidden
  geometry, the SAME initial allocation, the SAME eligible damaged ego and the SAME event
  point** — **only the damage condition / severity differs.** That is what makes the
  within-seed comparison a within-world comparison.
- **RECORD realized hidden counts. Do NOT automatically discard a HIGH stratum world merely
  because bounded B2 backoff realized fewer hidden targets than requested.**
- **BEFORE any scientific measurement, INSPECT the requested-vs-realized distributions and
  REJECT or REDESIGN the benchmark if the HIGH load systematically degenerates.** A HIGH
  stratum that collapses into the LOW one is not a stratum, and discovering that after a
  measurement would waste the measurement.
- **FUTURE actor-only and CTDE generalized measurements MUST use the EXACT SAME frozen
  benchmark / world manifests.** That shared frozen manifest is what makes the two arms
  comparable.
- **HISTORICAL FIXED-CELL MEASUREMENTS ARE NOT THE GENERALIZED BENCHMARK.** §3h, §3j and any
  old-contract CTDE run measured a different cell under a different contract; none of them
  is this benchmark, its comparator, or an expectation for it.

### 3l.7 Approved diagnostics / metrics direction

> **STATUS — IMPLEMENTED, REVIEWED AND MERGED as §3l.8 step 4 (PR #40, `db790138…` → `b4daa8c1…`, verdict APPROVE).** **`CLAUDE.md` §5 is now AUTHORITATIVE**, with the routing in §6 and the lock in §7. The bullets below are the APPROVED DESIGN this task implemented and are retained as the design record; where design prose and `CLAUDE.md` differ, **`CLAUDE.md` and the code are decisive**. *(SUPERSEDED: this section previously carried a "NOT IMPLEMENTED / nothing below exists in the repository / not authorized" banner, and a "PARTIAL EXCEPTION" note saying that steps 1–3 PRODUCED the per-episode diagnostic structures while **NOTHING PERSISTED OR AGGREGATED THEM** — no `run_config.json` block, no `episode_outcomes.jsonl` field, no `run_summary.json` key, no plot. Each was accurate when written and is history: every one of those structures is now persisted per episode and aggregated per run.)*

> **HOW IT LANDED.** No new artifact file was added. `episode_outcomes.jsonl` (schema version 2) and `episode_failures.jsonl` grew fields carrying `EpisodeContext.construction_audit`, `FdEligibilityAudit` / `FdEventCertificate`, `FuelDamageController.post_fd_outcome` and `EpisodeResult.reference` / `EpisodeReference` WHOLE, beside the requested-vs-realized cardinality, the scored-vs-unscored completion accounting, the aircraft-loss and real-RTB-command diagnostics, and the benchmark stratum / matched-group / world-identity keys. `run_config.json` gained an `episode_design` block, an honest generalized `construction` block and a manifest-aware `provenance` seed-source block. `run_summary.json:/generalized` is DERIVED from the canonical jsonl streams (ONE metric path), every denominator explicit. `measurement_health.png` gained a FOURTH panel showing requested-vs-realized hidden load as a DISTRIBUTION. **Requested-vs-realized is REPORTED for inspection and NO acceptance threshold is applied and NO verdict is returned** — that judgement stays a human / GPT scientific review decision (`CLAUDE.md` §5).

Per accepted episode, the intended observable set is:

- reward, reference utility and achieved utility;
- targets completed, and **scored** targets completed;
- aircraft losses, and damaged-ego survival / RTB;
- **requested and realized hidden cardinality, with the B2 backoff reasons**;
- **FD eligibility candidates, rejection reasons, the event, and pre/post fuel**;
- **post-FD wake count and the selected meta-actions**;
- continuation-solver runtime and allocation count;
- kills outside the reference;
- setup rejection reason.

### 3l.8 The bounded implementation sequence — **STEPS 1–5 ALL COMPLETE, REVIEWED AND INTEGRATED**

The approved ORDER, with each step its own separately scoped, separately reviewed task that
begins only after the previous one is reviewed and integrated:

1. **generalized cardinality + B2 bounded backoff + accounting** (§3l.1, §3l.2) —
   **COMPLETE / REVIEWED / INTEGRATED.** Candidate `5b55ca348309b4241d2087c2f60327bc842ea6fa`,
   integration `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9`, PR #35.
2. **FD eligibility-by-construction + persistent post-FD adaptation + repeated wakes**
   (§3l.3, §3l.4) — **COMPLETE / REVIEWED / INTEGRATED.** Final candidate
   `185d39f00335a0bb5e9130cc773da94c914f17f5`, integration
   `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b`, PR #36; append-only fix chain from the
   REQUEST-FIXES candidate `2f9231d989acf30561ecf10e74cf0c5491771836`.
3. **event-conditioned continuation reference + reward checkpoint / accounting** (§3l.5) —
   **COMPLETE / REVIEWED / INTEGRATED.** Reviewed candidate
   `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integration
   `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, PR #38, verdict **APPROVE**.
   *(SUPERSEDED: this entry previously read "THE NEXT UNRESOLVED IMPLEMENTATION TASK. NOT
   STARTED, AND NOT AUTHORIZED".)*
4. **training sampler + frozen stratified evaluation manifest + metrics** (§3l.6, §3l.7) —
   **COMPLETE / REVIEWED / INTEGRATED.** FINAL approved candidate
   `db79013897a6e5669f50d53b6e30229b16aea28d`, integration
   `b4daa8c1a8c870061b26cceb01d4ed34169594e7`, PR #40, verdict **APPROVE**; append-only fix
   chain from the original reviewed candidate
   `eef1795f6bb3f0cbc4c163ba489cf5e790df4c41`. It delivered the ONE `episode_design`
   harness selector for the whole approved policy bundle, the generalized training
   cardinality sampler on its own rng domain, the frozen stratified benchmark MANIFEST
   MECHANISM (schema, builder, canonical serialization, content hash, verifying loader,
   consumer and identity checks), the reason-based `ReferenceIntegrityError` routing, and
   run-level PERSISTENCE and AGGREGATE METRICS for every per-episode diagnostic structure
   steps 1–3 produce (`construction_audit`, `FdEligibilityAudit` / `FdEventCertificate`,
   `post_fd_outcome`, `EpisodeReference`). **It did NOT choose the final scientific
   worlds-per-cell SCALE and did NOT generate, commit or freeze a concrete benchmark
   POPULATION.** *(SUPERSEDED: this entry previously read "THE SINGLE NEXT UNRESOLVED
   IMPLEMENTATION TASK. NOT STARTED, AND NOT AUTHORIZED BY THIS RECORD".)*
5. **bounded runtime / solver validation BEFORE deciding the final scientific run scale** —
   **GENERALIZED-V1 TASK 5: IMPLEMENTED, REVIEWED, APPROVED AND INTEGRATED.**
   **PR #42**, approved head `312f58650b61a85eb72d0554d60715afee862a5c` (integrated
   `5dfcd8b632be8dca3c1730018bbf35337d07f077`), and
   **PR #43**, FINAL approved head `4af6c5aa5dd28072692bfda63282964b55010aae` (append-only fix
   chain from `734f1e786593b6ffb94f1f8d7283b1f2fc79d257`; integrated
   `b3c2e01f130afe854b09384cd6e1e196de714795`), locked by the documentation **PR #44**
   (`88352b2fc03174e8095d3c7e8a1ef58b60e58e0b`, integrated
   `9b9e9b85a70c8a0019c72ada92ceec3401725795`) — **all three MERGED** (§3m.1, §3m.5).
   *(SUPERSEDED: this entry previously read "both FROZEN / READ-ONLY, neither merged, so no
   integration SHA exists for either". Accurate at that checkpoint.)* It delivered the `train_by_*`
   summary-population correction, the successful-episode training quota with its REQUIRED
   bounded attempt budget and maximum-possible seed band, and the deterministic benchmark
   preflight that SELECTS a complete population once before the freeze and leaves a durable
   audit when it cannot. The bounded validation itself was performed and reviewed as **Task 5A
   and Task 5B — `APPROVE — VALID ENGINEERING VALIDATION`, ENGINEERING EVIDENCE ONLY**
   (§3m.3). **It still did NOT select a FINAL SCIENTIFIC worlds-per-cell scale and did NOT
   commit or freeze a benchmark POPULATION into the repository.** *(SUPERSEDED: this entry
   previously read "THE SINGLE NEXT UNRESOLVED STEP. NOT STARTED, AND NOT AUTHORIZED BY THIS
   RECORD".)*
6. **documentation and `CLAUDE.md` locks for behaviour that already exists** — **DONE FOR
   EVERY IMPLEMENTED STEP (1 THROUGH 5), and its POST-INTEGRATION CLOSURE for the Task-5
   sequence is the pass IN FLIGHT AS PR #45.** Each implemented step carries its own
   technical lock, Task 5's being PR #44 (§3m.1); PR #45 is the separate post-integration
   state reconciliation that follows the §3m.5 merges. **While PR #45 is open that closure
   pass is IN FLIGHT; on its integration the Task-5 post-integration documentation closure
   is COMPLETE and no writable repository task remains.** *(SUPERSEDED: this entry previously
   read "PARTIALLY DONE and CONTINUING", which was accurate while steps still lacked locks
   and before this closure pass existed.)*

**HOW STEP 6 REALLY WORKS, corrected by practice.** This section previously said step 6 was
deliberately LAST — one documentation pass after every implementation step. It is instead
written PER COMPLETED TASK: `CLAUDE.md` §4 / §5 / §6 / §7 records for steps 1, 2, 3, 4 and 5
are written by their own documentation records, each after that step's behaviour was
implemented, reviewed and integrated. **The principle is unchanged and is what matters:** no
`CLAUDE.md` contract is ever written for a design, only for behaviour that already exists.
*(SUPERSEDED: this paragraph previously added "which is exactly why §3l.6–§3l.7 still have
none", and previously said records existed for "steps 1, 2, 3 and 4". They have one now —
the Task-4 §5 contract, its §6 routing and its §7 lock — and step 5 has its own through
PR #44.)*
**THE CLOSING PASS IS A DISTINCT, SCOPED ARTIFACT, NOT AN OPEN-ENDED OBLIGATION.** A
per-task lock records a step's CONTRACT; a post-integration closure pass reconciles the
documents' CURRENT STATE once that step's merges have actually happened. For the Task-5
sequence that closure pass is **PR #45** — IN FLIGHT while its draft PR is open, and
**COMPLETE on its integration**, at which point no writable repository task remains.
**A FINAL documentation pass would be REQUIRED again only for FUTURE, NEWLY IMPLEMENTED
work**, and that generic forward-looking rule does **NOT** mean the CURRENT Task-5
post-integration closure is outstanding. *(SUPERSEDED, and corrected here: this paragraph
previously ended "A FINAL documentation pass is still REQUIRED after any later
implementation step lands, and no per-task record discharges it", with no closure pass named
— which left the Task-5 closure reading as permanently unfinished.)*

### 3l.9 What the redesign explicitly does NOT touch

- **The approved historical measurements stand, unmodified and un-reinterpreted.** The
  Phase-A long baseline (§3h) and the FD-VARIABLE-SEVERITY-v1 actor-only baseline at
  measured code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` (§3j,
  **`APPROVE — VALID MEASUREMENT`**, primary finding NEGATIVE) are **not to be re-run,
  repaired, resumed, re-tuned or replaced**, and every preserved run tree stays preserved
  (§6).
- **`p(destroy) < 1` remains DEFERRED** (§3l.3, §6), as do SAMs / hostile fire and dense
  per-wake reward. None may be bundled into the generalized redesign.
- **No communication channel, no peer feature and no privileged label may reach the acting
  path** (§3l.2, §3l.4; `CLAUDE.md` §3).
- **The reference roles of `main`, `phase-a-baseline`, `pre-ctde-actor-only` and
  `flat-final` are unchanged, and no ref is moved or deleted by this record** (§1, §6).
- **A previously executed OLD-CONTRACT CTDE measurement is OUT OF SCOPE** and must not be
  reviewed or compared unless the user explicitly asks (§1).

## 4. Research ordering, CTDE comparison specification and difficulty selection

From the former `CLAUDE.md` §8.

- **RESEARCH ORDERING — the 2026-08-22 PARALLEL arrangement, NOW FULLY TRAVERSED.**
  The variable-severity MEASUREMENT and Phase-B CTDE were run in parallel by explicit
  user/orchestrator decision — **not** an accidental Phase-A reopening, **not** a
  correction of anything, and **not** a change to any technical CTDE contract. **All four
  items are now COMPLETE**: the measurement is EXECUTED, independently reviewed and VALID,
  and the CTDE integration gate is SATISFIED AND CLOSED. This bullet is therefore the
  arrangement's HISTORICAL RECORD; the live research state is the CTDE bullet below. The
  approved order was:
  1. **PRESERVE the original Phase-A reference baseline.** It is CLOSED, VALID and
     IMMUTABLE — measured code SHA `737b4bf` on the FD-BASELINE-v1 design, run
     `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` (§7). The branch
     `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) preserves the code
     state and must not move. **Nothing below redefines, reopens, re-runs, extends or
     supersedes it.**
  2. **IMPLEMENT (done) and MEASURE (DONE — EXECUTED, REVIEWED, `APPROVE — VALID
     MEASUREMENT`) the ADDITIONAL actor-only FD-VARIABLE-SEVERITY-v1 baseline — ON A
     PINNED, IMMUTABLE, DETACHED SNAPSHOT.** The code is merged and locked (`eecc9b5`,
     integrated `177e969`, PR #27 — §5, §7). The measurement was LOCKED to exact SHA
     `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
     `dd881478b8e2e521054d09bc865437f1308be1a2`, executed from a DETACHED, clean snapshot
     worktree that carried no task branch and was READ-ONLY with respect to the shared
     repository. **§7 now owns the authoritative record**: the valid replacement run, the
     excluded `MAX_PATH` precursor, every denominator, and the NEGATIVE primary finding
     (no severity-conditioned FD-wake meta-action separation). It is, and remains, an
     **ACTOR-ONLY measurement OF THAT PINNED SHA**: repository work landing after
     `bf1e045f…` — Phase-B CTDE included — is simply not in the measured tree, so it can
     neither be attributed to that run nor contaminate it. **Nothing beyond §7's record
     may be claimed for it**, and its negative finding is a valid result, not a defect.
  3. **PHASE-B CTDE DESIGN AND IMPLEMENTATION PROCEEDED CONCURRENTLY** in a separate
     writable task branch / worktree, ungated on the measurement's completion. **DONE:**
     approved candidate `a6f3aa9`, integrated `8390d85`, PR #30 (§5, §7).
  4. **THE CTDE INTEGRATION GATE — SATISFIED AND CLOSED ON BOTH HALVES.** The
     measurement-validity half was met by `APPROVE — VALID MEASUREMENT` at measured code
     SHA `bf1e045f` (§7) — by a NEGATIVE result, which satisfies the gate exactly as a
     positive one would, because the gate is about VALIDITY, never about a favourable
     outcome. The second half — a NEW immutable actor-only pre-CTDE reference preserved
     from the then-current actor-only state — was met by **`pre-ctde-actor-only =
     d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the FIRST parent of the CTDE integration,
     which must not move. The existing branch `phase-a-baseline`
     (`4f0068847b017795717c5f0e331f647bcfc30547`) remains historical provenance for the
     ORIGINAL valid Phase-A reference, was NEVER moved and was NOT repurposed as that new
     reference.
  **OWNERSHIP — HISTORICAL.** While the two ran in parallel the CTDE GPT orchestrator was
  the SOLE WRITABLE repository owner and the FD measurement orchestrator was READ-ONLY on
  its detached snapshot. The user's ONE-TIME writable exception for the FD closure record
  ENDED when that record was integrated, and writable repository ownership RETURNED to the
  CTDE GPT orchestrator, which has since integrated PR #30. Every orchestrator resolves
  live branch and PR state from GitHub itself.
  **THIS SUPERSEDED THE SERIAL ORDER THIS BULLET ITSELF PREVIOUSLY STATED** — that CTDE
  design could begin only after the variable-severity measurement was executed and
  independently reviewed. That serial rule is HISTORY as of 2026-08-22 and must not be
  restated as live; the measurement has since completed and been reviewed VALID anyway. **It also still supersedes the two ORIGINAL ordering claims** that
  Phase-B CTDE was immediately next and that a stochastic/partial fuel-degradation variant
  was deferred until AFTER Phase B. FD-VARIABLE-SEVERITY-v1 is that variant's approved
  form, it is an ADDITIONAL actor-only stress baseline rather than a replacement for the
  Phase-A reference, and its PURPOSE is unchanged: the actor-only response to a
  survivable-vs-unsurvivable loss is measured independently of centralized training — which
  a pinned, detached snapshot preserves exactly, and which is precisely why the two may now
  run at the same time. **`p(destroy) < 1`, SAMs and dense reward are UNAFFECTED and remain
  separate, still-deferred future research changes** (see the difficulty-selection bullet
  below); none of them is part of FD-VARIABLE-SEVERITY-v1 and none may be bundled into its
  baseline.
- **Centralized critic / value head (CTDE) — PHASE B. IMPLEMENTATION CLOSED: REVIEWED AND
  MERGED. THE SCIENTIFIC COMPARISON IS THE ONLY PART STILL OPEN.** (Phase A is closed by
  the valid baseline, §7; the variable-severity factor is merged and its actor-only
  baseline is EXECUTED and independently reviewed `APPROVE — VALID MEASUREMENT` at measured
  code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` — with a NEGATIVE primary finding,
  §7.)
  **THE GATE IS SATISFIED AND CLOSED, on both halves.** The measurement-validity half was
  satisfied by that variable-severity verdict — satisfied by a NEGATIVE result, which
  counts exactly as a positive one would, because the gate tests VALIDITY and never
  favourability. The remaining half, preservation of a NEW immutable actor-only pre-CTDE
  reference, was satisfied by **`pre-ctde-actor-only =
  d437084c5fb1a22c21596a48c58e03f7e15a0115`** (tree
  `d7cc2dcb1b161180e272afc9600175f022c5b5d0`), the FIRST parent of the CTDE integration —
  so it is provably the actor-only state CTDE was merged onto, and it must not move.
  `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) is the SEPARATE ORIGINAL
  Phase-A reference, was never repurposed for this, and likewise must not move.
  **THE IMPLEMENTATION IS DONE AND IS NO LONGER AN OPEN DESIGN QUESTION.** Approved
  candidate `a6f3aa9`, integrated `8390d85`, PR #30 (§7), and locked as a §5 contract by
  this record. Phase B now has TWO SELECTABLE TRAINING MODES — `actor_only` (the DEFAULT
  and the preserved reference path) and `ctde` — chosen by `TrainConfig.training_mode`. The
  size-agnostic value estimator off `GraphEncoder.pool()` EXISTS (`ValueHead` on
  `CentralCritic`'s own encoder instance); the privileged critic inputs and their
  exclusions are ENUMERATED in §5; actor/critic separation, the training-only boundary,
  capture timing, GAE/value semantics, checkpoint distinction and actor-only byte-invariance
  are all IMPLEMENTED AND PROVEN in `tests/test_graph_ctde.py`. **Do not restate any of
  these as open requirements, do not re-enter a design/recon step, and do not rebuild what
  is merged.** Changing any of them is a Grade-A change to a locked layer, routed through §6.
  **WHAT WAS STILL OPEN WHEN THIS BULLET WAS WRITTEN: the FIRST CONTROLLED ACTOR-ONLY vs
  CTDE COMPARISON.** **SCOPING CORRECTION, and it is deliberately minimal.** An
  OLD-FIXED-CELL-CONTRACT CTDE measurement has since been EXECUTED. Its EXISTENCE is
  acknowledged here and nothing else is: **no identity, no measured code SHA, no run
  directory, no denominator, no verdict and no result are recorded, and none may be
  inferred from this document** — none of that was inspected when this record was written,
  and inventing any of it would be false provenance. It is **OUT OF SCOPE for the
  GENERALIZED-V1 phase and for this record**, and it **must NOT be reviewed, re-read,
  re-analysed, or compared against any actor-only baseline unless the user EXPLICITLY asks.**
  **It establishes NO CTDE benefit here**, and it is **NOT** a generalized comparator: it was
  taken under the OLD FIXED-CELL contract, which is a different cell from the generalized
  benchmark. The remainder of this bullet is the PRESERVED specification of what a controlled
  comparison would have to satisfy; it neither authorizes one nor claims one was performed to
  it. Engineering tests, module `_selftest`s, a passing suite and a merged
  implementation measure NOTHING scientific. **No CTDE benefit may be pre-claimed** — not
  from the approved Phase-A result, which explicitly does not establish one; not from the
  executed variable-severity baseline, which measured no CTDE anything and whose negative
  severity finding is **NOT** evidence that centralized training would change it; and not
  from PR #30's implementation evidence. A CTDE claim requires its own executed,
  independently reviewed comparison. The next scientific task is preparing and executing
  that comparison, and it must:
  - **TAKE ITS ACTOR-ONLY ARM FROM THE ALREADY-APPROVED PHASE-A BASELINE, WHICH IS NOT
    RE-RUN.** That arm is already measured — measured code SHA `737b4bf`, run
    `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, `APPROVE — VALID
    MEASUREMENT` (§7) — and it is PRESERVED and NOT to be re-run, resumed, repaired,
    extended or re-tuned. **Nothing in this record authorizes a fresh actor-only run**; a
    newly executed actor-only CONTROL arm is a SEPARATE research-design decision requiring
    explicit user authorization. What the task schedules is the **CTDE arm**;
  - **MATCH THE LOCKED ORIGINAL PHASE-A SCIENTIFIC CELL** — 3 agents, 3 known + 3 hidden,
    200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams = false`,
    `probability = 1`, frozen solver and BLADE, unchanged `graph_reward` formula with
    `aircraft_penalty_coeff = 2.25` — **and that baseline's training / evaluation schedule,
    seed policy, held-out band and evaluation construct** as the authoritative Phase-A
    record establishes them, judged under the SAME validity gate, VALIDITY BEFORE
    PERFORMANCE;
  - **name the EXPERIMENTAL FACTOR correctly: actor-only training vs centralized-critic
    training.** Provenance MUST acknowledge that the historical Phase-A measurement and the
    future CTDE measurement carry **DISTINCT measured code SHAs**, and **must NOT claim the
    two arms' literal repository or configuration artifacts differ only by one
    `training_mode` field** — they do not, and asserting it would be false provenance;
  - **NOT bundle `p(destroy) < 1`, SAMs, dense reward, a solver change, a reward-formula
    change, or any new difficulty factor.** Those remain separate, still-deferred research
    changes (the difficulty-selection bullet below), and bundling one would make the
    comparison uninterpretable.
  A run showing no CTDE improvement, or no productive update, is a valid NEGATIVE
  observation — not a technical failure and not grounds to re-tune or re-run. **No CTDE
  preset exists in the repository**, and creating one belongs to that comparison task.
- **Baseline difficulty selection — CLOSED for the current cell by FD-BASELINE-v1
  (`a8669f4`).** Exactly ONE factor was selected, implemented and locked: `fuel_damage`
  (§5, §7). The following were considered and **NOT selected**; each remains a DEFERRED,
  SEPARATE research change and none may be enabled implicitly or bundled into a probe:
  - **`probability < 1`** — still out. It reopens the reward operand scale, because
    expected oracle utility and realized achieved utility diverge below p=1; the cell
    stays at `probability = 1` and `graph_reward` stays frozen.
  - **Enemy targets that shoot back / SAMs** — still out. `include_sams=False`, and the
    construction path refuses a world whose enemy units are not all airbases; BLADE
    weapon lethality is unchanged.
  - **Dense / per-wake reward** — still out. It was never a consequence of selecting a
    difficulty factor, and is not one now.
  Reopening any of them is a new research-design decision with its own semantics,
  observability, proof obligations and bounded implementation/lock task. The Phase-A
  baseline they were deferred behind is now MEASURED (§7), and Phase-B CTDE is now
  IMPLEMENTED and MERGED, so the ordering constraint that remains is the phase boundary:
  they come AFTER the first controlled actor-only vs CTDE comparison, and must never be
  bundled into it — a difficulty change inside that contrast would make it uninterpretable.
  **`probability < 1` in particular is UNCHANGED by FD-VARIABLE-SEVERITY-v1 and is still
  out.** That factor merged a mild/severe split of the FUEL-DAMAGE EVENT; target
  destruction stays deterministic at `probability = 1`, and nothing in PR #27 implemented
  stochastic target destruction. It remains a separate future Grade-A research task.
  **What DID change is the ordering, not this list**: exactly ONE additional difficulty
  design — FD-VARIABLE-SEVERITY-v1, the approved form of the stochastic/partial
  fuel-degradation variant — was selected, implemented and locked after FD-BASELINE-v1
  (`eecc9b5`, §5, §7), and its actor-only baseline — run CONCURRENTLY with Phase-B CTDE
  design and implementation, on a pinned immutable snapshot — is now EXECUTED and reviewed
  `APPROVE — VALID MEASUREMENT` (§7). Every entry in the list above stays deferred behind
  the phase boundary exactly as stated.

## 5. GENERALIZED-V2 evaluation questions, as open at PR #57

From the former handoff §3p.4. PR #59 answered the design questions; this is preserved as the
record it was.

*(SUPERSEDED as CURRENT state by §3q. The questions below were OPEN when this record was
written and are now ANSWERED by the merged PR #59 construct: the evaluation population is the
ten exogenous `(A, K−A)` base cells with `A ∈ {2,...,6}` — so `A = 5` / `A = 6` DO participate
— `R` and hidden load are NOT strata, there is NO LOW/HIGH interpretation, the matched
structure is the CLEAN / MILD / SEVERE triad, the size is 12 frozen world groups per cell
split into `development` / `confirmatory`, held-outness is checked over the whole manifest
against the maximum training-attempt band, and `A = 8` / `A = 10` remain engineering-only.
What is NOT answered by code is the real benchmark seed namespace and the preflight
invocation — the next, still-unauthorized scientific action (§3q.4) — and comparator
discipline / any fresh control arm, which remain unauthorized. The list below is preserved as
the record it was.)*

**AT PR #57 THIS WAS THE SINGLE NEXT UNRESOLVED RESEARCH TASK, AND NEITHER PR #57 NOR THAT
RECORD TOOK ANY PART OF IT.** The next chat / orchestrator must begin **READ-ONLY** and decide the
V2 evaluation / benchmark design **BEFORE any scientific V2 comparison run**. The open
questions belong THERE, and include at minimum:

- the evaluation POPULATION definition;
- whether and how to stratify `A`, `K`, `R` and hidden load;
- the LOW/HIGH interpretation, if any (the V1 buckets are defined against `A` and do not
  transfer);
- the matched clean / mild / severe evaluation structure;
- benchmark SIZE / worlds-per-cell;
- seed bands and held-out identity;
- whether `A = 5` / `A = 6` participate in the first scientific benchmark;
- whether `A = 8` / `A = 10` remain engineering-only;
- comparator discipline, and whether ANY fresh control arm is authorized.

*(At PR #57, and no longer current:)* **NO V2 EVALUATION POPULATION, STRATIFICATION,
LOW/HIGH INTERPRETATION, MATCHED-GROUP CONSTRUCTION, WORLDS-PER-CELL SCALE, SEED BAND OR
MANIFEST IDENTITY EXISTED, WAS DEFINED OR COULD BE INFERRED.** **NAMING THE TASK WAS NOT
AUTHORIZATION TO EXECUTE ANYTHING.**

## 6. Future campaign items recorded before R1 was reviewed

From the former handoff §8. The statements that R1 was pending are superseded; the five campaign
items themselves remain open, undecided and unauthorized.

**FUTURE CAMPAIGN ITEMS REMAIN OPEN, UNDECIDED AND NOT AUTHORIZED — AND THEY ARE NOT THE
NEXT ACTION.** **THE ONE CURRENT NEXT ACTION IS AND STAYS INDEPENDENT GPT ARTIFACT REVIEW OF
THE DISPATCHED ACTOR-ONLY R1 ONCE ITS ARTIFACTS EXIST**, stated at the top of this section;
**nothing in the list below displaces it, precedes it, runs in parallel with it, or becomes
the next action on this record's integration.** The five items are recorded ONLY so that a
future thread — **one that must be EXPLICITLY opened and authorized, and which this record
neither opens nor schedules** — starts from what is already known rather than from guesswork:

1. the **exact five-run scientific design**;
2. **benchmark-manifest identity, freeze and review**;
3. the **Slurm resource / 24 h-walltime launch decision**;
4. **scientific `sbatch` / job-array design**;
5. **launch and monitoring.**

**ALL FIVE ARE OPEN, UNDECIDED AND UNAUTHORIZED. NOTHING ABOUT THEM IS PRE-DECIDED,
PRE-DESIGNED, PRE-SCHEDULED OR PRE-AUTHORIZED BY THIS RECORD, AND LISTING THEM IS NOT
AUTHORIZING THEM, NOR IS IT OPENING A THREAD TO OWN THEM.** In particular: **the five-run
matrix is NOT defined here**; **no benchmark manifest is constructed, frozen, committed or approved as a
comparator**; **no partition, queue, CPU/memory or walltime choice is made**; **no `sbatch`
script or job array exists or is designed**; and **no launch is scheduled.** The observed
Slurm `course` limits in §3m.6 are VOLATILE INPUTS to item 3, never its answer. **A CTDE
generalized run remains unauthorized**, **`p(destroy)` remains `1.0` with `p(destroy) < 1`
DEFERRED**, and **R1 stays `AUTHORIZED / DISPATCHED — RESULT PENDING` and UNREVIEWED** —
independent GPT artifact review of R1 remains the open scientific thread and is NOT displaced
by this list.
*(SUPERSEDED, and corrected here: this passage previously opened "THE NEXT ACTION AFTER THIS
DOCUMENTATION TASK IS INTEGRATED IS TO OPEN A FRESH ORCHESTRATION THREAD, AND THAT THREAD
OWNS FIVE ITEMS". That named a next action which contradicted this section's own — and this
document's — single next-action contract, and it read as scheduling a thread no one had
authorized. The five items themselves, and every non-claim attached to them, are unchanged.)*

## 7. Superseded workflow procedure

The workflow half of the former `CLAUDE.md` §1, superseded on 2026-09-14 by
[`cc_review.md`](../workflows/cc_review.md) under the user-approved documentation packet: the
`CLAUDE_MOUNTED_MAIN` transport, the one-commit-per-task rule, the universal long status block
and Grade C's exemption from review no longer apply. Historical grade labels and review records
keep their original meaning.

- **User speaks Hebrew.** Code, comments, and docs stay in English.
- **Packet-driven scope.** One closed task at a time. Follow the packet's declared scope,
  grade, and 1–3 proof obligations. Explain material implementation choices before making
  them; stop only for a blocking ambiguity, a red-line conflict, or a material deviation
  from the packet — not after every file.
- **Grade = the trust policy, declared in the packet.** It states how much review the change
  earns, not how hard it is. **C** — hygiene, wording, docs, unreachable fallbacks: trusted,
  no review, no lock ceremony. **B** — "the pipeline runs or it does not": one test on the
  main path, no branch coverage, no goldens; the orchestrator reads the changed files from
  the repo. **A** — a research claim is at stake (no-communication isolation, route-prediction
  and placement fidelity, reproducibility of the geometry, source-of-truth / append-only):
  1–3 proof obligations declared up front, and the orchestrator must approve the exact
  full SHA of the reviewed commit; when the change touches cross-ego isolation or a §5
  locked layer it earns line-by-line review — GPT reads the exact GitHub `base...candidate`
  comparison; under `CLAUDE_MOUNTED_MAIN` the hunks come from CC (see Grade-A routing default
  below). Grade A is set by consequence, not by difficulty — a wrong A is a silent false
  result, not a crash.
- **Candidate commits are required, and transport is MODE-DEPENDENT.** Implement + run the
  required tests, then create the commit the orchestrator will review. Transport is never
  approval: a commit stays `READY_FOR_REVIEW / UNREVIEWED` until the orchestrator approves
  its exact full SHA — and that includes a commit already pushed to `main`. Exactly one of
  the two modes below applies, and **the packet or the user must state which one**; never
  silently infer that task branches or PRs are accessible. Shared by both modes: start from
  the packet's verified full base SHA, stage exactly the declared files, and keep one
  focused commit per task. If the user explicitly marks a task `local-only`, do not push;
  report that the orchestrator cannot independently review it until that restriction is
  lifted.
- **No premature docs.** Don't spawn `README`/`SUMMARY`/per-file docs unasked. One consolidated doc per stable component.
- **Minimal files.** Prefer extending a module over spawning `foo_utils.py` + `foo_config.py`.
- **Transport mode `GPT_GITHUB`** — the GPT orchestrator inspects GitHub directly (branches,
  PRs, files, exact SHAs):
  - start the task branch from the packet's verified full base SHA;
  - create and push a candidate commit on that task branch;
  - open or update its draft PR;
  - the GPT orchestrator reviews the exact `base...candidate` state before merge;
  - do not push directly to `main`, merge, rebase, or force-push unless the user explicitly
    changes the task's authorization.
- **Transport mode `CLAUDE_MOUNTED_MAIN`** — the Claude orchestrator's shared repository view
  is a synchronized mounted snapshot of `main`, exposed as a search interface: not a live
  connector, not a filesystem, not `git`. It retrieves and quotes file content, but it cannot
  list files, count occurrences, prove that something is ABSENT, produce a diff, read history,
  or select an arbitrary SHA, and it can LAG the true `main` head — a stale read is sync lag,
  not regression. Task branches and PRs must **not** be assumed accessible:
  - before editing, verify a clean checkout and exact equality between local `HEAD`,
    `origin/main`, and the packet's full base SHA;
  - stage only the declared files, create one focused commit, and push it directly to `main`
    with a normal **non-force** push;
  - the pushed commit stays `UNREVIEWED` until the Claude orchestrator approves that exact
    full SHA. For Grade A work, do not claim a lock or begin dependent work until that
    post-push exact-SHA approval occurs;
  - if review finds a problem, correct it with a NEW follow-up commit on `main` — never
    amend, rewrite history, reset published commits, or force-push;
  - if `main` has advanced, the push is rejected, or the mounted checkout cannot prove its
    state, stop and report `BLOCKED` (do not pull, merge, rebase, reset, stash, or delete).
- **Grade-A routing default.** `GPT_GITHUB` is the only mode that gates `main` behind a
  reviewable branch, so Grade-A work is routed to the GPT orchestrator by default whenever it
  is available. Grade A under `CLAUDE_MOUNTED_MAIN` is a declared exception — the packet must
  say so explicitly — and carries two consequences. First, the candidate is reviewed only
  AFTER it is pushed, so `main` knowingly carries an `UNREVIEWED` commit until its exact full
  SHA is approved: no lock, and no dependent work, before that approval. Second, CC MUST
  supply focused changed hunks plus targeted test evidence; this is mandatory rather than a
  fallback, because a mounted snapshot shows current state and can never show
  `base...candidate`.
- **Fix chain:** review corrections stay in the same named CC session and produce a NEW
  commit and a new SHA to review — never a rewrite of the reviewed one. In `GPT_GITHUB` they
  land on the same task branch; merge only the unchanged approved head, prefer a merge
  strategy that preserves the reviewed commit, and if integration rewrites it, verify the
  resulting tree before recording its SHA as a lock. In `CLAUDE_MOUNTED_MAIN` they land as
  follow-up commits on `main`, and the lock is the last approved pushed SHA.
- **Status block (mode-aware):** every task ends with: state (`READY_FOR_REVIEW /
  UNREVIEWED` or `BLOCKED`); transport mode; full base SHA; full review SHA; grade; files
  changed; tests / checks run; proof-obligation evidence; deviations judged against the
  packet; `NEW FACTS LEARNED` anchored by **file + symbol or exact string** (mandatory, use
  `NONE` when empty); `CLAUDE.md` deltas needed; and final working-tree state. Task branch
  and draft PR number / URL are required **only** in `GPT_GITHUB` mode; in
  `CLAUDE_MOUNTED_MAIN`, report `main` plus the verified pushed `origin/main` SHA.
- **Output discipline:** never paste whole files, full transcripts, or a large full diff into
  chat. GPT inspects the exact GitHub `base...candidate` diff itself; under
  `CLAUDE_MOUNTED_MAIN` CC supplies focused changed hunks and targeted test evidence on
  request — required for Grade A. Otherwise return targeted run output and direct answers
  only, and put a genuinely long report in the repo.
