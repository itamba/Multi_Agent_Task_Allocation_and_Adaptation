# Multi-Agent Graph RL — GENERALIZED-V1: TASKS 1–5 ALL IMPLEMENTED, REVIEWED, APPROVED AND INTEGRATED (PR #42 + PR #43 + PR #44, ALL MERGED); TASK 5A / 5B ENGINEERING VALIDATION APPROVED; THE FIRST GENERALIZED ACTOR-ONLY R1 LONG RUN IS AUTHORIZED / DISPATCHED WITH ITS RESULT PENDING AND UNREVIEWED; THIS POST-INTEGRATION CLOSURE CANDIDATE IS THE SOLE WRITABLE REPOSITORY TASK WHILE ITS PR IS OPEN, AND ON ITS INTEGRATION NO WRITABLE REPOSITORY TASK REMAINS WHILE GENERALIZED-V1 STAYS AN ACTIVE PHASE / Phase-A + Variable-Severity Baselines CLOSED and VALID / PHASE-B CTDE MERGED AND DOCUMENTED — Handoff

**Supersedes all earlier handoffs.**

Written 2026-08-11; updated 2026-08-14 for the final-cell PROBE HARNESS closure,
2026-08-15 to record the FIRST EXECUTED bounded short probe and the three
research-validity defects it exposed (§3d), 2026-08-16 to record the CLOSURE of ALL
THREE of those defects — **Defect A, ego-global `SELF_PRESERVATION_ABORT`, merged through
PR #17**, **Defect B, the attack-confirmation wait derived from the salvo about to fly,
merged through PR #19**, and **Defect C, physical RTB completion, merged through
PR #21** — 2026-08-16 again to record the **CORRECTED-CELL SHORT-PROBE RERUN** (§3e),
2026-08-17 to record the FIRST EXECUTED LONG BASELINE, the FOURTH and SEPARATE
roster/world-truth defect it exposed, and that defect's merged correction (PR #24)
(§3f, §3g), and **2026-08-18 to record the APPROVED Phase-A LONG-BASELINE RERUN, to CLOSE
PHASE A** (§3h), and **2026-08-20 to record the MERGED FD-VARIABLE-SEVERITY-v1 research
factor (PR #27) and the DELIBERATE research-order change that puts an ADDITIONAL
actor-only variable-severity baseline BEFORE Phase-B CTDE** (§3i, §4), and **2026-08-22 to
record the user's explicit SUPERSESSION of that serial order: the variable-severity
MEASUREMENT is now pinned to an immutable detached snapshot and Phase-B CTDE DESIGN AND
IMPLEMENTATION proceed CONCURRENTLY beside it, with CTDE INTEGRATION into `main` still
gated** (§4, §8), and **2026-08-23 to record that the FD-VARIABLE-SEVERITY-v1 actor-only
baseline is now EXECUTED, independently reviewed `APPROVE — VALID MEASUREMENT` at measured
code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` with a NEGATIVE primary behavioural
finding, to CLOSE Task 7, and to record that the CTDE integration gate's
measurement-validity half is SATISFIED while its pre-CTDE-reference half is not** (§3j,
§4, §8), and **2026-08-23 again to record that PHASE-B CTDE IS NOW IMPLEMENTED,
REVIEWED AND MERGED — approved candidate `a6f3aa9d62931994f416b2241fec4cfac3b018ec`,
integrated `8390d85c2072e9cbe984ce5f2731cef3a9b14985`, PR #30 — that the immutable
pre-CTDE actor-only reference `pre-ctde-actor-only =
d437084c5fb1a22c21596a48c58e03f7e15a0115` exists, that the CTDE INTEGRATION GATE is
therefore CLOSED / SATISFIED on both halves, and that NO CTDE scientific comparison has
been run** (§1, §3k, §4, §8), and **2026-08-23 a THIRD time to CLOSE the volatile
repository/task state before transfer to a fresh chat: PR #32 / `task/phase-b-ctde-doc-lock`
is MERGED and is NO LONGER the current writable task, this chat/repository CLOSURE record is
the sole writable candidate only while it is in flight, and upon its integration the
repository is CLOSED / IDLE with no writable task and no open scientific run** (§1, §4, §7,
§8) — *(that CLOSED / IDLE state was true when written and is now SUPERSEDED by the entry
below)* — and **2026-08-25 to record that the repository is NO LONGER CLOSED / IDLE: the
GENERALIZED TRAINING / BENCHMARK REDESIGN is the ACTIVE research and design phase, owned by
the GPT orchestrator, with its APPROVED DESIGN recorded in a dedicated section and MARKED
NOT YET IMPLEMENTED, with no implementation candidate active and no new scientific
measurement running or authorized** (§1, §3l, §4, §6, §7, §8, §9) — *(that
NOT-YET-IMPLEMENTED framing was true when written and is SUPERSEDED by the entry below)* —
and **2026-08-25 again, in this INTERMEDIATE DOCUMENTATION CHECKPOINT, to record that
GENERALIZED-V1 IMPLEMENTATION TASKS 1 AND 2 ARE IMPLEMENTED, REVIEWED AND MERGED** —
Task 1 (generalized cardinality + deterministic bounded B2 backoff + requested-vs-realized
accounting), candidate `5b55ca348309b4241d2087c2f60327bc842ea6fa`, integrated
`9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9`, PR #35; and Task 2 (certified FD eligibility +
post-FD completion-boundary adaptation), final candidate
`185d39f00335a0bb5e9130cc773da94c914f17f5`, integrated
`ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b`, PR #36 — **while NO generalized scientific
measurement exists, is running or is authorized** (§1, §3l, §4, §8, §9) — and
**2026-08-25 a THIRD time to record that GENERALIZED-V1 IMPLEMENTATION TASK 3 IS ALSO
IMPLEMENTED, REVIEWED AND MERGED**: the event-conditioned MATCH-AOU continuation reference
and reward checkpoint (§3l.5), reviewed candidate
`24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integrated
`df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, **PR #38**, verdict **APPROVE** — so **§3l.5 is
now IMPLEMENTED** — *(that record then named Task 4 as the single next unresolved
implementation task, NOT started and NOT authorized; that framing was accurate when written
and is SUPERSEDED by the entry below)* — and **2026-08-26 to record that GENERALIZED-V1
IMPLEMENTATION TASK 4 IS ALSO IMPLEMENTED, REVIEWED AND MERGED**: the episode-design
selector, the generalized training cardinality sampler, the frozen stratified benchmark
MANIFEST MECHANISM and run-level persistence / aggregate metrics (§3l.6–§3l.7), FINAL
approved candidate `db79013897a6e5669f50d53b6e30229b16aea28d`, integrated
`b4daa8c1a8c870061b26cceb01d4ed34169594e7`, **PR #40**, verdict **APPROVE** — so
**§3l.6–§3l.7 are now IMPLEMENTED, §3l.8 steps 1–4 are ALL COMPLETE, GENERALIZED-V1 TASK 4
is CLOSED, no Task-4 implementation branch or PR is writable or active, no implementation
candidate remains under review**, the **SINGLE NEXT UNRESOLVED STEP is GENERALIZED-V1
TASK 5 — bounded runtime / solver validation BEFORE the final scientific run scale is
selected — which is NOT started and is NOT authorized by this record**, **no FINAL
SCIENTIFIC benchmark worlds-per-cell scale has been SELECTED and no FINAL SCIENTIFIC
benchmark population or manifest has been committed, preserved as the comparator, scheduled
or authorized**, and **still NO generalized
scientific measurement exists, is running, is scheduled or is authorized** (§1, §3l, §4, §7,
§8, §9) — *(that entry's "Task 5 NOT STARTED / no generalized run scheduled or authorized"
framing, AND its "no FINAL SCIENTIFIC worlds-per-cell scale has been SELECTED / no benchmark
population is scheduled or authorized" framing, were both accurate when written and are
SUPERSEDED by the entry below)* — and **2026-08-30, in THIS post-Task-5 documentation /
lock checkpoint, to record that GENERALIZED-V1 TASK 5 IS IMPLEMENTED AND APPROVED AS A STACKED,
STILL-UNMERGED TWO-PR STACK**: **PR #42**, branch `task/generalized-v1-task5-summary-phase-fix`,
approved head `312f58650b61a85eb72d0554d60715afee862a5c` (the `train_by_*` summary-population
correction), and **PR #43**, branch `task/generalized-v1-task5-success-quota-preflight`, FINAL
approved head `4af6c5aa5dd28072692bfda63282964b55010aae` (the successful-episode training quota,
the bounded attempt budget, the maximum-possible seed band and the deterministic benchmark
preflight) — **both APPROVED, both FROZEN / READ-ONLY, and NEITHER MERGED**, with THIS
documentation candidate the **SOLE WRITABLE REPOSITORY TASK**; to record **Task 5A and Task 5B as
`APPROVE — VALID ENGINEERING VALIDATION`, which is ENGINEERING EVIDENCE and NEVER a scientific
measurement**; and to record that **the FIRST full GENERALIZED-V1 ACTOR-ONLY R1 long run is
AUTHORIZED / DISPATCHED with its RESULT PENDING and UNREVIEWED — no reward, convergence,
attrition, benchmark or validity claim about it exists or may be inferred** (§1, §3m, §4, §7,
§8, §9) — *(that record's "STILL-UNMERGED TWO-PR STACK / SOLE WRITABLE DOCUMENTATION
CANDIDATE / no merge is authorized" framing was accurate when written and is SUPERSEDED by
the entry below)* — and **2026-08-31, in THIS POST-INTEGRATION CLOSURE record, to record that
THE WHOLE GENERALIZED-V1 TASK-5 STACK IS NOW INTEGRATED**: **PR #42**
(`312f58650b61a85eb72d0554d60715afee862a5c` → merge
`5dfcd8b632be8dca3c1730018bbf35337d07f077`), **PR #43**
(`4af6c5aa5dd28072692bfda63282964b55010aae` → merge
`b3c2e01f130afe854b09384cd6e1e196de714795`) and the Task-5 DOCUMENTATION LOCK **PR #44**
(`88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` → merge
`9b9e9b85a70c8a0019c72ada92ceec3401725795`) — a THREE-PR integration sequence, every step a
NORMAL MERGE COMMIT preserving its reviewed candidate as an ancestor / merge parent, with
**no rebase, no squash, no cherry-pick, no force-push and no history rewrite** — so
**GENERALIZED-V1 TASKS 1 THROUGH 5 ARE ALL IMPLEMENTED, REVIEWED, APPROVED AND INTEGRATED**,
while **the actor-only R1 long run remains `AUTHORIZED / DISPATCHED — RESULT PENDING` and
UNREVIEWED, and NO generalized scientific measurement RESULT exists** (§1, §3m, §4, §7, §8,
§9).

**THE LIVE STATE (2026-08-31), STATED FIRST BECAUSE IT SUPERSEDES EVERY LIVE-STATE PARAGRAPH
BELOW IT, INCLUDING THE 2026-08-30 AND 2026-08-26 ONES.** The ACTIVE phase is still
**GENERALIZED-V1**, and **§3l.8 STEP 5 IS NO LONGER "APPROVED BUT UNMERGED": THE ENTIRE
TASK-5 STACK IS INTEGRATED.** Live `main` at this record's base is
`9b9e9b85a70c8a0019c72ada92ceec3401725795` (`2026-08-31 00:39:29 +0300`, the PR-#44
documentation merge), and the exact integration chain was **PR #42** `312f5865…` → merge
`5dfcd8b6…`, **PR #43** `4af6c5aa…` → merge `b3c2e01f…`, **PR #44** `88352b2f…` → merge
`9b9e9b85…`. **THIS POST-INTEGRATION CLOSURE CANDIDATE IS THE SOLE WRITABLE REPOSITORY TASK
WHILE ITS OWN DRAFT PR IS OPEN; ON ITS INTEGRATION THE REPOSITORY RETURNS TO A CLEAN
CHECKPOINT WITH NO WRITABLE REPOSITORY TASK, NO OPEN TASK-5 PR AND NO ACTIVE TASK-5
CANDIDATE.** **GENERALIZED-V1 NEVERTHELESS REMAINS AN ACTIVE PROJECT PHASE, because the
actor-only R1 long run is still pending** — the repository being idle is not the phase being
closed. **Task 5A and Task 5B are `APPROVE — VALID ENGINEERING VALIDATION` — ENGINEERING
EVIDENCE ONLY, never a measurement, and no reward, learning, attrition-rate or actor-vs-CTDE
claim may be drawn from either.** **THE FIRST FULL GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN IS
`AUTHORIZED / DISPATCHED — RESULT PENDING`**: it is UNREVIEWED, it has produced no verdict,
and **nothing about its reward, convergence, attrition, benchmark outcome or scientific
validity may be stated or inferred** — no elapsed time implies otherwise, and it is neither
`RUNNING`, nor `COMPLETED`, nor `VALID`. Its frozen plan is §3m.4 and it requires
**independent GPT artifact review** before any `APPROVE — VALID MEASUREMENT` verdict. The
external long-run task is **RUN-ONLY and owns NO repository writes**. **The R1 benchmark
SCALE remains SELECTED and AUTHORIZED (`worlds_per_cell = 3`) and its CONSTRUCTION remains
AUTHORIZED and DISPATCHED; no concrete R1 manifest has been independently reviewed or
approved as the comparator, and none is committed or tracked in the repository.** **No CTDE
generalized run exists, is scheduled or is authorized**, and **no actor-only-vs-CTDE
generalized result exists.** **The NEXT action is NOT another implementation task: it is
INDEPENDENT GPT ARTIFACT REVIEW of the actor-only R1 when its artifacts exist** (§8) — until
then no rerun, no repair, no resume, no extension, no CTDE arm, no benchmark replacement and
no retuning, each of which would need its own research decision and authorization.

**THE PRECEDING LIVE STATE (2026-08-30), PRESERVED AS THE RECORD IT WAS AND SUPERSEDED WHERE
THE PARAGRAPH ABOVE SAYS SO.** The ACTIVE phase is still **GENERALIZED-V1**, and
**§3l.8 STEP 5 IS NO LONGER "NOT STARTED": TASK 5 IS IMPLEMENTED AND APPROVED, AS A STACKED,
STILL-UNMERGED TWO-PR STACK.** Live `main` at this record's checkpoint is
`09eab0673153bd443185ec94530ccf0b042be465`; **PR #42** (`312f58650b61a85eb72d0554d60715afee862a5c`)
and **PR #43** (`4af6c5aa5dd28072692bfda63282964b55010aae`) are **APPROVED, FROZEN / READ-ONLY and
NOT MERGED**, and **this documentation candidate — `task/generalized-v1-task5-doc-lock`, branched
from exact `4af6c5aa…` — is the SOLE WRITABLE REPOSITORY TASK and is itself NOT MERGED.** The
stack is `main` → PR #42 → PR #43 → docs PR; **no merge is authorized by this record** and the
intended later integration sequence is §3m.5. **Task 5A and Task 5B are `APPROVE — VALID
ENGINEERING VALIDATION` — ENGINEERING EVIDENCE ONLY, never a measurement, and no reward,
learning, attrition-rate or actor-vs-CTDE claim may be drawn from either.** **THE FIRST FULL
GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN IS AUTHORIZED / DISPATCHED AND ITS RESULT IS PENDING**: it
is UNREVIEWED, it has produced no verdict, and **nothing about its reward, convergence, attrition,
benchmark outcome or scientific validity may be stated or inferred** — its frozen plan is recorded
in §3m.4 so the eventual artifacts can be checked against what was authorized, and it requires
**independent GPT artifact review** before any `APPROVE — VALID MEASUREMENT` verdict. The external
long-run task is **RUN-ONLY and owns NO repository writes**. **No CTDE generalized run exists, is
scheduled or is authorized**; **no reviewed early-stopping mechanism exists** and checkpoint
RESUME stays out of scope (`graph_train` is SAVE-only); **cluster readiness is DEFERRED** because
cluster access is not available, which does NOT block the local R1; and **`p(destroy)` remains
`1.0`** with `p(destroy) < 1` DEFERRED (§3m). *(SUPERSEDED by this paragraph: the 2026-08-26
record's statements that Task 5 is the single next unresolved step and is NOT STARTED, that no
writable implementation task is active, and that no generalized run is scheduled or authorized.
Each was accurate when written, AS IS every "no scale selected / no benchmark authorized"
statement in the preserved paragraphs below — those are SUPERSEDED as current state by
§3m.4 and by §1's four-fact block, and are retained only as the record they were. **THE
CURRENT STATE IS FOUR DISTINCT FACTS:** **the R1 benchmark SCALE is SELECTED and AUTHORIZED (`worlds_per_cell = 3`) and its
CONSTRUCTION is AUTHORIZED and DISPATCHED (candidate base seed `840000`,
`max_candidates_per_cell = 12`); no concrete R1 manifest has yet been independently reviewed
or approved as the comparator, and none is committed or tracked in the repository; and no
GENERALIZED scientific measurement RESULT exists**, **and no generalized result — including
any actor-only-vs-CTDE comparison — may be pre-claimed.**)*

**THE PRECEDING LIVE STATE (2026-08-26), PRESERVED AS THE RECORD IT WAS AND SUPERSEDED ONLY WHERE
THE PARAGRAPH ABOVE SAYS SO.** **The repository is NOT closed and NOT idle.** The ACTIVE phase is
**GENERALIZED-V1**, and its bounded implementation sequence is now COMPLETE THROUGH STEP 4.
**§3l.8 STEPS 1, 2, 3 AND 4 ARE ALL COMPLETE, REVIEWED AND INTEGRATED** — Task 1 (candidate
`5b55ca34…`, integration `9b305e4e…`, PR #35), Task 2 (final candidate `185d39f0…`,
integration `ca0dc406…`, PR #36), Task 3 (candidate `24a8b1ee…`, integration `df3abf2f…`,
PR #38, verdict **APPROVE**) and Task 4 (final candidate `db790138…`, integration
`b4daa8c1…`, PR #40, verdict **APPROVE**) — so **§3l.1–§3l.7 are now implemented TO THE
EXTENT those four tasks represent**, and their technical contracts are recorded in
`CLAUDE.md` §4 / §5 / §6 / §7. The **FOUR low-level policy seams** stay OPT-IN with their
historical defaults and are resolved TOGETHER — and only together — by the ONE
`episode_design` selector (`fixed_cell_v1` DEFAULT vs `generalized_v1`); the generalized
harness ADDITIONALLY uses the generalized cardinality sampler and requires the SEPARATE
`fuel_damage_mode` field to be `seeded_variable`, neither of which is a policy id on
`EpisodeDesign`. Every Task-1/2/3 per-episode diagnostic structure is now PERSISTED and
AGGREGATED. **GENERALIZED-V1 TASK 4 IS CLOSED: its implementation branch and PR are no
longer writable or active, and NO implementation candidate remains under review.**
*(HISTORICAL as of 2026-08-30 — every clause in this sentence is SUPERSEDED as CURRENT
state by the four-fact block in the live-state paragraph above and by §3m.4; it is retained
as the 2026-08-26 record it was.)* **NO GENERALIZED SCIENTIFIC MEASUREMENT EXISTS, IS
RUNNING, IS SCHEDULED OR IS AUTHORIZED,
no FINAL SCIENTIFIC benchmark worlds-per-cell SCALE has been SELECTED, no FINAL SCIENTIFIC
benchmark population or manifest has been committed, preserved as the comparator, scheduled
or authorized, no final actor-only or CTDE generalized campaign is authorized, and no
generalized result of any kind — including any actor-only-vs-CTDE comparison — may be
pre-claimed.** When those two arms are eventually run
they MUST use the SAME eventual frozen manifest. `p(destroy)` remains `1.0` and
`p(destroy) < 1` remains DEFERRED; the solver and BLADE remain FROZEN; the approved Phase-A
(§3h) and FD-VARIABLE-SEVERITY-v1 (§3j) measurements remain preserved and are REUSED as what
they are — neither re-run, repaired, resumed, re-tuned nor reinterpreted.
**THE SINGLE NEXT UNRESOLVED STEP IS GENERALIZED-V1 TASK 5 — bounded runtime / solver
validation BEFORE the final scientific run scale is decided (§3l.8 step 5). IT IS NOT
STARTED AND IS NOT AUTHORIZED BY THIS RECORD** (§8). *(SUPERSEDED, and preserved only as
history: the 2026-08-25 handoff-bootstrap record's statements that "EVERY LINE OF THAT
DESIGN IS NOT YET IMPLEMENTED", that "no implementation candidate is active", that "nothing
generalized is implemented", and that step 1 was the live next action; the Task-1/2
checkpoint's statements that §3l.5 is NOT IMPLEMENTED and that Task 3 is the next unresolved
task; and the Task-3 record's statements that §3l.6–§3l.7 remain NOT IMPLEMENTED, that
"neither harness selects any generalized policy", that nothing persists or aggregates the
diagnostic structures, and that Task 4 is the next unresolved task and is NOT started and
NOT authorized. Each was accurate when written and is no longer true. `CLAUDE.md` is
likewise no longer untouched by the generalized phase: it now carries the Task-1, Task-2,
Task-3 and Task-4 contracts and locks.)*
Everything below this paragraph is the PRESERVED record of the closed phases that precede
the redesign, and it remains accurate about those phases.

**THE STATE OF THE CLOSED PHASES, STATED PLAINLY.** **PHASE A IS CLOSED.** The authorized
long-baseline rerun —
`training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, measured code SHA
`737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`, ONE invocation, native exit 0,
`cli_overrides = []` — was executed once, independently reviewed, and approved
**`APPROVE — VALID MEASUREMENT`**. It is **the FIRST scientifically valid measurement of the
fuel-damage cell** (§3h; `CLAUDE.md` §7 owns the authoritative record). The three-defect
CODE correction and the FOURTH roster/world-truth correction (PR #24) all remain merged, and
they are what made a sound measurement possible. **The long baseline is NOT to be re-run,
resumed, repaired or re-tuned.**

**AND THE ADDITIONAL VARIABLE-SEVERITY BASELINE IS NOW MEASURED — VALID, AND ITS PRIMARY
FINDING IS NEGATIVE.** The factor's CODE is CLOSED / APPROVED / MERGED — approved candidate
`eecc9b5d91bce4a98a070a29307cc12af0d4c4a3`, integrated `177e969446ef6c01c729484f2ea9969c94a27330`,
PR #27 (§3i). **Its ACTOR-ONLY BASELINE has since been EXECUTED ONCE from the immutable
DETACHED snapshot at exact SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
`dd881478b8e2e521054d09bc865437f1308be1a2`, independently reviewed, and approved
`APPROVE — VALID MEASUREMENT`** (§3j; `CLAUDE.md` §7 owns the authoritative record and every
denominator). **THE PRIMARY BEHAVIOURAL FINDING IS NEGATIVE**: the deterministic held-out
actor showed NO severity-conditioned FD-wake meta-action separation between a survivable
MILD loss and an unsurvivable SEVERE one, before or after training — **a valid negative
scientific result, not a defect and not grounds for retuning.** An earlier attempt at the
SAME contract is `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT`: a Windows `MAX_PATH`
playback-export failure systematically removed the whole `post_update` SEVERE arm, which is
an infrastructure failure rather than a result. Both trees are preserved. **The research
order that put this measurement in PARALLEL with Phase-B CTDE (2026-08-22) did its job: the
measurement stayed an ACTOR-ONLY measurement OF ITS PINNED SHA**, so later CTDE work was
never in the measured tree and can neither be attributed to that run nor contaminate it.
**AND PHASE-B CTDE IS NOW IMPLEMENTED, REVIEWED AND MERGED.** Approved candidate
`a6f3aa9d62931994f416b2241fec4cfac3b018ec`, integrated
`8390d85c2072e9cbe984ce5f2731cef3a9b14985`, PR #30 (§3k). **The CTDE INTEGRATION GATE IS
CLOSED / SATISFIED on BOTH halves**: the measurement-validity half by the variable-severity
`APPROVE — VALID MEASUREMENT` verdict — a negative result satisfies it exactly as a
positive one would — and the reference half by **`pre-ctde-actor-only =
d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the FIRST parent of the CTDE integration and
therefore provably the actor-only state CTDE was merged onto. It must not move, and
`phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) remains the SEPARATE
ORIGINAL Phase-A reference that was never repurposed for it. **PR #30 RAN NO SCIENTIFIC
COMPARISON, and NO CTDE BENEFIT IS CLAIMED ANYWHERE IN THIS DOCUMENT** — a merged
implementation and a passing suite measure nothing scientific (`CLAUDE.md` §5 owns the CTDE
contract, §7 the lock). *(An OLD-CONTRACT CTDE measurement has since been executed; it is
OUT OF SCOPE for the generalized redesign and is NOT reviewed, compared or claimed here —
§1. `CLAUDE.md`'s stale "not run" wording HAS BEEN RECONCILED, conservatively, by THIS
documentation checkpoint: it now acknowledges only that such a measurement exists and is out
of scope, and records no identity, measured SHA, denominator, verdict or result for it and
no CTDE benefit.)* None of this is a reopening of Phase A and none of it changes any
technical CTDE contract: the Phase-A reference baseline stays
CLOSED, VALID and IMMUTABLE, and the branch `phase-a-baseline`
(`4f0068847b017795717c5f0e331f647bcfc30547`) preserving its code state must not move.
`p(destroy) < 1` remains a SEPARATE, later research task and was NOT implemented by PR #27.
B1–B4, the first real post-B3 instrumented probe, the B4 observability follow-up (PR #7),
**FD-BASELINE-v1** (PR #8), **FINAL-CELL-VISUAL-ARTIFACTS** (PR #10), the repository
code-hygiene cleanup (PR #11), the documentation hygiene (PR #12) and now the
**FINAL-CELL PROBE HARNESS** (PR #14) are all CLOSED. None of them changes research
behaviour: the solver, BLADE, reward, fuel-damage semantics, PPO math, seed schedules,
scenario construction and matched-pair evaluation are exactly as their own locks left them.

Baseline **difficulty selection is finished**, the **inspection surface is in place**, and
the **operator harness a run is driven from — preset, run layout and figures — is merged**.
FOUR runs of the merged cell now exist, and **the fourth is the valid one.** The FIRST short
probe (`training_output_20260815_173029`, from clean `main` at
`238062d7d284334432d9c39d7543fb0bbf39ea7c`) passed every mechanical harness and accounting
check **and** exposed three research-validity defects (§3d). They were corrected in sequence
— A, then B, then C, as separate reviewed tasks: **all three are CLOSED and MERGED** (PR #17,
PR #19 and PR #21). The SAME bounded probe shape was rerun ONCE from the corrected `main`
(`training_output_20260816_162130`, exact code SHA
`900ff0b24898eccfa2e35d2db05c4e0229c64ce3`) and reviewed `VALID` (§3e) — **and that verdict
has since been SUPERSEDED** (§3f). The FIRST LONG BASELINE was then executed
(`training_output_long_baseline_100x8_seed0`, exact code SHA
`c30b6982ba605d60976cc303256da4b5528b0e63`) and independently reviewed as **engineering
`REQUEST FIXES`, scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED`** (§3f): a roster
that read ALLOCATED-ONLY solver output as world inventory destroyed 143 of its 800 training
attempts while the run reported itself reconciled. That FOURTH, SEPARATE defect was then
**CLOSED / APPROVED / MERGED** (§3g, PR #24). **The RERUN on the same scientific contract
into a new directory then PASSED the validity gate and is the Phase-A baseline** (§3h). The
first three runs remain preserved and are HISTORY ONLY — their numbers are never a scientific
baseline, and validity is still judged BEFORE performance.

**AND THE REPOSITORY WAS THEN CLOSED FOR HANDOFF — HISTORICAL (2026-08-23), SUPERSEDED BY
THE LIVE STATE ABOVE.** The Phase-B CTDE
documentation/lock task is CLOSED / APPROVED / MERGED — approved candidate
`c607f3fabcbd58f6f10cfde6bcc34068f09e4121`, verdict **`APPROVE`**, integrated by normal
merge commit `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96` (PR #32), ordered parents
`8390d85c2072e9cbe984ce5f2731cef3a9b14985` then `c607f3f…` — so BOTH the Phase-B CTDE
IMPLEMENTATION and its DOCUMENTATION are closed. **This closure record changes NO technical
contract**: it exists only to retire the now-stale claim that PR #32 /
`task/phase-b-ctde-doc-lock` is still the current writable task. After it is integrated
there was, at that moment, **no writable task, no open code candidate, no open PR and no
scientific run in progress.** **THAT IDLE STATE HAS SINCE ENDED**: the read-only walkthrough
it anticipated took place, and the project then entered the GENERALIZED TRAINING / BENCHMARK
REDESIGN — the live phase recorded in the LIVE STATE paragraph above and specified in §3l.
**The old fixed-cell Task-9 framing is NO LONGER the live next action** (§4, §8); what it
says about the Phase-A comparator, the non-claims and the prohibitions remains binding if
that task is ever resumed. **NO CTDE benefit is claimed anywhere in this document, and NO
actor-only rerun is authorized.**

This handoff is volatile and deliberately thin. Technical contracts live in `CLAUDE.md`;
code and tests remain decisive. Where a fact is already in `CLAUDE.md` this document
cross-references it rather than duplicating it.

## 0. How this workspace reads the repository

- Repository access is capability-aware, not shared:
  - GPT resolves repository facts through GitHub first — branches, PRs, files and exact
    SHAs. GitHub is a search interface, not a filesystem.
  - Claude resolves facts through a synchronized mounted snapshot of `main` plus Git
    evidence CC reports. It can lag and cannot prove a historical diff on its own.
  - Either orchestrator asks one focused question only when its available access cannot
    establish a blocking fact.
- Tie every repository claim to an explicit full SHA. Read code, tests, `CLAUDE.md` and
  this handoff at that same SHA. Project Sources, memory, chat summaries and pasted
  narratives are not evidence of current repository state.
- Cite code by file + symbol or exact string, never by line number.
- `CLAUDE.md` §1 owns grading, transport, candidate review, fixes and status reporting.

## 1. Current state

- **LIVE PHASE (2026-08-31) — GENERALIZED-V1: TASKS 1 THROUGH 5 ARE ALL IMPLEMENTED,
  REVIEWED, APPROVED AND INTEGRATED (PR #42 → `5dfcd8b6…`, PR #43 → `b3c2e01f…`, PR #44 →
  `9b9e9b85…`). THIS POST-INTEGRATION CLOSURE CANDIDATE IS THE SOLE WRITABLE REPOSITORY TASK
  WHILE ITS OWN DRAFT PR IS OPEN, AND ON ITS INTEGRATION NO WRITABLE REPOSITORY TASK REMAINS.
  THE FIRST GENERALIZED ACTOR-ONLY R1 LONG RUN IS AUTHORIZED / DISPATCHED WITH
  ITS RESULT PENDING AND UNREVIEWED. NO GENERALIZED MEASUREMENT RESULT EXISTS.** The
  repository is **no longer CLOSED / IDLE** and **no longer DESIGN-ONLY**, and
  **GENERALIZED-V1 REMAINS AN ACTIVE PROJECT PHASE even once no writable task remains**,
  because R1 is still pending. The approved design
  is in **§3l**; §3l.8 steps 1 through 5 are ALL MERGED, and the Task-5 stack, its
  engineering validation, the dispatched R1 plan and the PERFORMED integration sequence are
  in **§3m**. *(SUPERSEDED, and corrected here: this bullet previously read "TASKS 1–4 ARE
  MERGED, AND TASK 5 IS IMPLEMENTED AND APPROVED AS A STACKED, STILL-UNMERGED TWO-PR STACK …
  THIS DOCUMENTATION CANDIDATE IS THE SOLE WRITABLE REPOSITORY TASK AND IS NOT MERGED", and
  pointed at an INTENDED integration sequence. Accurate at that checkpoint; not now.)*
  *(SUPERSEDED: this bullet previously read
  "TASK 4 IS CLOSED. NO WRITABLE IMPLEMENTATION TASK IS ACTIVE. THE SINGLE NEXT UNRESOLVED
  STEP IS TASK 5 … NOT STARTED AND … NOT AUTHORIZED. NO GENERALIZED MEASUREMENT EXISTS."
  Every clause was accurate when written.)*
  - **GENERALIZED-V1 TASK 1 — IMPLEMENTED / REVIEWED / MERGED.** Generalized construction
    cardinality, deterministic bounded B2 backoff, and truthful requested-vs-realized
    accounting (§3l.1, §3l.2). Approved candidate
    `5b55ca348309b4241d2087c2f60327bc842ea6fa`, integrated
    `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9`, **PR #35**. Grade A under `GPT_GITHUB`.
  - **GENERALIZED-V1 TASK 2 — IMPLEMENTED / REVIEWED / MERGED.** Certified FD eligibility
    and post-FD completion-boundary adaptation (§3l.3, §3l.4). FINAL approved candidate
    `185d39f00335a0bb5e9130cc773da94c914f17f5`, integrated
    `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b`, **PR #36**. Grade A under `GPT_GITHUB`.
    **HISTORICAL PROCESS EVIDENCE ONLY:** the initial reviewed candidate
    `2f9231d989acf30561ecf10e74cf0c5491771836` received **REQUEST FIXES**, and the
    correction landed as the APPEND-ONLY CHILD COMMIT `185d39f0…` — never amended, rebased,
    squashed or force-pushed.
  - **GENERALIZED-V1 TASK 3 — IMPLEMENTED / REVIEWED / MERGED.** The event-conditioned
    MATCH-AOU continuation reference and reward checkpoint (§3l.5). Reviewed candidate
    `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integrated
    `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, **PR #38**, verdict **APPROVE**. Grade A
    under `GPT_GITHUB`, implementation mode BUILD. Candidate and integration share the
    IDENTICAL tree `187aed9105eca5db799f4508374dc86811001b9d`; the candidate is the merge's
    SECOND PARENT. **ONE OPT-IN reward-reference policy
    (`event_conditioned_continuation_v1`) beside the historical `static_t0_v1`, which stays
    the DEFAULT and is untouched.** The GPT review examined and APPROVED one implementation
    deviation — the reference kind `damaged_event_unrealized_t0` — as a COMPATIBILITY
    RESOLUTION preserving the already-locked Task-2 LEGACY contract; it is **NOT** the
    intended generalized damaged semantics and is unreachable under
    `certified_both_severities_v1` (`CLAUDE.md` §5).
  - **GENERALIZED-V1 TASK 4 — IMPLEMENTED / REVIEWED / MERGED / CLOSED.** The episode-design
    selector, the generalized training cardinality sampler, the frozen stratified benchmark
    MANIFEST MECHANISM, and run-level persistence / aggregate metrics (§3l.6, §3l.7). FINAL
    approved candidate `db79013897a6e5669f50d53b6e30229b16aea28d`, integrated
    `b4daa8c1a8c870061b26cceb01d4ed34169594e7`, **PR #40**, verdict **APPROVE**. Grade A
    under `GPT_GITHUB`, implementation mode BUILD. Candidate and integration share the
    IDENTICAL tree `f7cfd5cb2a551bddd5bfecf78fdcc83e2dcedef7`; the candidate is the merge's
    SECOND PARENT (ordered parents `f4e8d3b8ddc61525fe0cde6b61ca4d611ebd2eed`, then
    `db790138…`). **HISTORICAL PROCESS EVIDENCE ONLY:** the original reviewed candidate
    `eef1795f6bb3f0cbc4c163ba489cf5e790df4c41` received review corrections, and they landed
    as the APPEND-ONLY CHILD COMMIT `db790138…` — never amended, rebased, squashed or
    force-pushed — covering manifest integrity, real held-outness and honest generalized
    construction provenance. **Its authoritative technical contract is `CLAUDE.md` §5, with
    its §4 placement, §6 routing and §7 lock.** **The Task-4 implementation branch and PR are
    no longer writable or active, and NO Task-4 candidate remains under review.**
  - **GENERALIZED-V1 TASK 5 — IMPLEMENTED, REVIEWED, APPROVED AND INTEGRATED, AS A
    THREE-PR SEQUENCE (§3m.1, §3m.5).** **PR #42**, branch
    `task/generalized-v1-task5-summary-phase-fix`, approved
    head `312f58650b61a85eb72d0554d60715afee862a5c` (base `09eab067…`): the `train_by_*`
    summary-population correction — **integrated by merge
    `5dfcd8b632be8dca3c1730018bbf35337d07f077`**. **PR #43**, branch
    `task/generalized-v1-task5-success-quota-preflight`, FINAL approved head
    `4af6c5aa5dd28072692bfda63282964b55010aae` (PR base ORIGINALLY the PR-#42 branch, later
    RETARGETED to `main`): the
    successful-episode training quota with deterministic replacement, the REQUIRED bounded
    attempt budget, the MAXIMUM POSSIBLE training seed band, and the deterministic benchmark
    preflight with its complete-manifest rule and durable failed-preflight audit —
    **integrated by merge `b3c2e01f130afe854b09384cd6e1e196de714795`**. **PR #44**, branch
    `task/generalized-v1-task5-doc-lock`, FINAL approved documentation head
    `88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` (append-only child of
    `61eaa3fe1bdeb7aef3cfb7c10c4d8964caf2ed0e`, likewise RETARGETED to `main`): the Task-5
    documentation lock over `CLAUDE.md` and this handoff — **integrated by merge
    `9b9e9b85a70c8a0019c72ada92ceec3401725795`**. Grade A
    under `GPT_GITHUB`. **ALL THREE ARE APPROVED AND ALL THREE ARE MERGED**, each by a NORMAL
    MERGE COMMIT preserving its reviewed candidate as an ancestor / merge parent, with the
    integrated tree verified equal to the reviewed tree in every case. **PR #43 and PR #44
    were each EXACT-BASE RE-REVIEWED after retargeting**, because changing a PR's base
    invalidates a base-relative verdict even though the candidate SHA is unchanged; neither
    head moved. **HISTORICAL PROCESS EVIDENCE ONLY:** PR
    #43's original implementation candidate was
    `734f1e786593b6ffb94f1f8d7283b1f2fc79d257`; GPT requested ONE append-only review fix; the
    final candidate `4af6c5aa…` is its DIRECT CHILD, with no amend, rebase, squash,
    force-push or history rewrite — and PR #44's `88352b2f…` is likewise the append-only
    DIRECT CHILD of `61eaa3fe…`. **Their authoritative technical contract is `CLAUDE.md`
    §5, routed in §6 and locked in §7.** *(SUPERSEDED, and corrected here: this bullet
    previously described a STILL-UNMERGED TWO-PR stack with "no integration SHA … and none
    may be invented", and named the then-open doc-lock branch as the SOLE WRITABLE REPOSITORY
    TASK. Accurate at that checkpoint; the integration SHAs now exist and are recorded
    above.)*
  - **TASK 5A AND TASK 5B — `APPROVE — VALID ENGINEERING VALIDATION`, AND NEITHER IS A
    SCIENTIFIC MEASUREMENT (§3m.3; `CLAUDE.md` §7).** **What makes them not measurements is
    their DESIGNATED PURPOSE, not an absence of mechanics**: Task 5B really did have an
    explicit training seed band `[720000, 720072)`, an explicit benchmark candidate band,
    production held-out verification, a TRANSIENT frozen manifest and 18 worlds / 54 members
    for its one evaluation round — all of it existing solely to validate system behaviour,
    attrition and runtime, and explicitly NOT designated as the scientific comparator or as a
    policy-performance measurement. **No reward or learning claim, no generalized-performance
    claim, no actor-vs-CTDE claim**, bounded sample sizes are an explicit limitation, and
    **the Task-5B transient manifest is NOT the R1 comparator and must never be promoted into
    R1.** *(SUPERSEDED, and corrected here: this bullet previously said they carry "no
    scientific contract, no seed schedule, no held-out band, no frozen comparator and no
    population denominator" — too broad, and factually wrong for Task 5B. The
    engineering-only label is UNCHANGED.)*
  - **THE FIRST GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN IS `AUTHORIZED / DISPATCHED — RESULT
    PENDING` (§3m.4).** It is UNREVIEWED and has produced no verdict. **Nothing about its
    reward, convergence, attrition, benchmark outcome or scientific validity may be stated or
    inferred**, and this record deliberately does not claim it is `RUNNING`. Its frozen plan
    is recorded in §3m.4 so the eventual artifacts can be checked against what was
    authorized, and it requires **independent GPT artifact review** before any
    `APPROVE — VALID MEASUREMENT` verdict. **The external long-run task is RUN-ONLY and owns
    NO repository writes.** **No CTDE generalized run exists, is scheduled or is authorized.**
  - **WHAT IS THEREFORE IMPLEMENTED.** **§3l.1–§3l.7 are implemented to the extent §3l.8
    steps 1–4 represent.** The **FOUR low-level policy seams** — `hidden_policy`,
    `eligibility_policy`, `post_fd_wake_policy`, `reference_policy` — are OPT-IN with
    historical defaults and are resolved TOGETHER, and only together, by
    `TrainConfig.episode_design` / `RolloutConfig.episode_design` ∈ (`fixed_cell_v1`
    DEFAULT, `generalized_v1`). **`graph_generalized.EpisodeDesign` carries exactly those
    four and no others.** The generalized harness ADDITIONALLY uses the generalized
    cardinality sampler, and `validate()` requires the SEPARATE `fuel_damage_mode` field to
    be `seeded_variable` — **neither is a fifth policy id on `EpisodeDesign`**. Every
    Task-1/2/3 per-episode diagnostic structure is now
    PERSISTED into the existing `episode_outcomes.jsonl` / `episode_failures.jsonl` and
    AGGREGATED into `run_summary.json:/generalized`, with a fourth
    `measurement_health.png` panel showing requested-vs-realized hidden load. *(SUPERSEDED:
    the Task-3 record stated that §3l.6–§3l.7 were NOT IMPLEMENTED, that neither harness
    exposed any generalized policy, and that nothing persisted or aggregated the diagnostic
    structures. Each was accurate when written and is not now.)*
  - **THE SCALE / AUTHORIZATION / RESULT STATE — FOUR DIFFERENT FACTS, AND THEY MUST NOT BE
    COLLAPSED INTO ONE NEGATIVE.**
    1. **THE R1 SCIENTIFIC BENCHMARK SCALE IS SELECTED AND AUTHORIZED: `worlds_per_cell = 3`**
       (§3m.4). The scale was the decision the bounded runtime / solver validation existed to
       inform, and it has been taken.
    2. **THE R1 BENCHMARK CONSTRUCTION IS AUTHORIZED AND DISPATCHED** — candidate base seed
       `840000`, `max_candidates_per_cell = 12`, a NEW R1 benchmark to be built by the
       deterministic preflight BEFORE training (§3m.4).
    3. **NO CONCRETE R1 MANIFEST HAS BEEN INDEPENDENTLY REVIEWED OR APPROVED AS THE
       COMPARATOR.** R1 is `RESULT PENDING`; **do not claim an R1 manifest exists** unless
       execution evidence later establishes it. **No benchmark manifest is committed or
       tracked in the repository**, and `configs/graph_train/final_cell_probe.json` remains
       the ONLY repository preset and is `fixed_cell_v1`, so **no repository preset selects
       `generalized_v1`.** *(That repository negative is scoped deliberately: transient
       manifests built in memory or in a temporary directory by tests and engineering
       validation are legitimate, are neither committed nor a scientific comparator, and
       repository state cannot establish a global negative over local scratch files.)*
    4. **NO GENERALIZED SCIENTIFIC MEASUREMENT RESULT EXISTS** — no reward, convergence or
       validity result, and **no actor-only-vs-CTDE generalized result.**
    *(SUPERSEDED, and corrected here: this bullet previously read "No FINAL SCIENTIFIC
    benchmark worlds-per-cell SCALE has been SELECTED, and no FINAL SCIENTIFIC benchmark
    POPULATION or manifest has been committed, preserved as the comparator, scheduled or
    authorized", and closed with "Choosing the scale is Task 5's business and comes after
    bounded runtime / solver validation". All of that was accurate through Task 4; the scale
    and the build have since been decided and authorized, and facts 3 and 4 are what
    survives.)* **HISTORICALLY: Task 4 delivered the manifest MECHANISM only** — schema,
    builder, canonical serialization, content hash, verifying loader, consumer and identity
    checks — with the builder REFUSING to invent a world count and no production caller at
    all; **Task 5 added that production caller,
    `graph_benchmark_preflight.run_benchmark_preflight`, which creates a manifest only after
    every base-cell quota has filled and creates NONE when a preflight fails.** *(All four
    facts are claims about SCIENTIFIC artifacts ONLY. The GENERALIZED-V1 Task-1/2/3/4
    TECHNICAL contracts DO exist and are integrated, and the Task-5 contracts DO exist as
    reviewed, APPROVED and INTEGRATED work as well; all are AUTHORITATIVE in `CLAUDE.md`
    §4 / §5 / §6 / §7.)*
  - **NO generalized scientific measurement RESULT exists, and no generalized result of any
    kind may be pre-claimed.** A first generalized **ACTOR-ONLY** R1 long run IS authorized
    and dispatched, and its result is **PENDING and UNREVIEWED** (§3m.4) — dispatch is not a
    result, and nothing about its reward, convergence, attrition, benchmark outcome or
    validity may be stated or inferred until independent GPT artifact review. **No CTDE
    generalized campaign is authorized, none is running or scheduled, and no
    actor-only-vs-CTDE generalized result exists.** When those two arms are eventually run
    they **MUST use the SAME eventual frozen manifest** — the shared `manifest_id` is what
    makes them comparable. Bounded real-BLADE / BONMIN smokes taken during implementation
    validation, and the Task-5A / Task-5B validations, are ENGINEERING evidence only: no
    scientific contract, no seed schedule, no held-out band, no denominator (§3m.3).
    *(SUPERSEDED: this bullet previously read "NO generalized scientific measurement exists,
    is running, is scheduled or is authorized" and "No final actor-only or CTDE generalized
    campaign is authorized". Accurate when written; the actor-only R1 has since been
    authorized and dispatched.)*
  - **`p(destroy)` REMAINS `1.0`; the solver / BONMIN and the vendored BLADE engine remain
    FROZEN**; the action set is unchanged and no new `MetaAction` exists; terminal-on-last
    reward credit placement, PPO, GAE, the encoder and the actor/critic boundary are
    unchanged under BOTH reference policies; and `training_mode` (`actor_only` / `ctde`)
    remains an ORTHOGONAL training-algorithm selector that alters no episode-design
    contract. **`p(destroy) < 1` remains DEFERRED.**
  - **THE HISTORICAL APPROVED BASELINES ARE REUSED, NOT RERUN.** The Phase-A long baseline
    (§3h, `737b4bf`) and the FD-VARIABLE-SEVERITY-v1 actor-only baseline (§3j, `bf1e045f`)
    are preserved, valid, and measurements of the `fixed_cell_v1` bundle; they are **not to
    be re-run, repaired, resumed or re-tuned**, and they are **NOT** generalized baselines,
    comparators or expectations.
  - **WHAT IS NEXT.** GENERALIZED-V1 Task 5 is IMPLEMENTED and APPROVED (§3m.1), and the R1
    long run is DISPATCHED with its RESULT PENDING (§3m.4), so the live next actions are the
    INTEGRATION of the approved stack under the exact-base re-review discipline of §3m.5 and
    the INDEPENDENT GPT ARTIFACT REVIEW of R1's eventual artifacts. **No documentation record
    authorizes a merge, a further implementation, a benchmark population, a campaign or a
    run** (§8). *(SUPERSEDED: this bullet previously read "THE SINGLE NEXT UNRESOLVED STEP IS
    GENERALIZED-V1 TASK 5 … It is NOT started, and this record does NOT authorize it." That
    was accurate when written.)*
  - **EARLY STOPPING, RESUME AND CLUSTER — CURRENT CAMPAIGN STATE (§3m.4).** **No reviewed
    early-stopping mechanism exists**; R1 uses a fixed 3000-success budget; **checkpoint
    RESUME is out of scope and `graph_train` is SAVE-only**; any future early stopping is a
    SEPARATE research / design decision and **must not select against the same final benchmark
    without an explicit validation / test design**, and none of it is implemented or
    authorized here. **Cluster campaign readiness is DEFERRED** because cluster access is not
    available — which does **not** block the local actor-only R1 — and no scheduler, queue or
    runbook detail exists or may be invented.
  - **`CLAUDE.md` IS AUTHORITATIVE FOR EVERY TECHNICAL CONTRACT, AND IT NOW RECORDS TASKS 1
    THROUGH 4.** Their §5 contracts, §4 placements, §6 routing and §7 lock entries are
    written, because reviewed and integrated behaviour exists for them. **Locks are written
    PER COMPLETED TASK, never for a design**, which supersedes this document's earlier "only
    at step 6" rule; a FINAL documentation pass is still required after any later
    implementation step lands.

- **A PREVIOUSLY EXECUTED OLD-CONTRACT CTDE MEASUREMENT IS OUT OF SCOPE FOR THIS PHASE.** A
  CTDE measurement was executed under the **OLD FIXED-CELL contract**. It is **outside the
  GENERALIZED-V1 redesign** and **must NOT be reviewed, re-read, re-analysed, compared
  against any actor-only baseline, or treated as evidence for or against centralized
  training — unless the user EXPLICITLY asks for it.** This record deliberately states **no
  run identity, no measured code SHA, no denominator and no validity verdict** for it: this
  documentation task did not inspect that run, and inventing any of those would be false
  provenance. **No CTDE benefit is claimed or implied anywhere in this document.** Note for
  a receiving orchestrator: `CLAUDE.md` was written before that run existed and described the
  actor-only vs CTDE comparison as un-run. **That text has since been reconciled
  CONSERVATIVELY by this checkpoint** — `CLAUDE.md` §2, §5 and §8 now acknowledge only that
  such an old-contract measurement EXISTS and is OUT OF SCOPE, and deliberately record no
  identity, measured SHA, denominator, verdict or result for it, and claim no CTDE benefit.
  **That is the whole of the reconciliation**: the run was still not inspected, reviewed or
  compared, and doing so remains out of scope unless the user EXPLICITLY asks. Historical fixed-cell measurements — that CTDE run included — are
  **NOT the generalized benchmark** and are not its comparator (§3l).
- **BASE of THIS documentation record:** `b4daa8c1a8c870061b26cceb01d4ed34169594e7`,
  committed `2026-08-26 11:30:07 +0300` — the `main` head produced by the **GENERALIZED-V1
  Task-4 integration (PR #40)**, whose ordered parents are
  `f4e8d3b8ddc61525fe0cde6b61ca4d611ebd2eed` then
  `db79013897a6e5669f50d53b6e30229b16aea28d`. **That is the exact base this Task-4
  DOCUMENTATION / LOCK record was DERIVED ON**, and it is a statement about this record's
  derivation only — **not** a claim about live `main`, which this record's own integration
  necessarily advances past. *(HISTORICAL derivation provenance: the Task-3 record was
  derived on `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, the PR-#38 integration; the
  Task-1/2 checkpoint on `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b`, the PR-#36
  integration.)* Neither this documentation commit nor its future merge can name its own
  SHA, and inventing either would be a false provenance claim.
  **Every receiving orchestrator therefore resolves the live full `main` SHA from GitHub and
  re-reads `CLAUDE.md` and this handoff at THAT SHA — GitHub is authoritative for live
  branch and PR state, never this document** (§9).
- *(HISTORICAL BASE of the handoff-bootstrap record.)* `76abdc480e80a84f1503208730d4525cd5e89b69`,
  committed `2026-08-23 15:34:35 +0300`, tree
  `237325d2c2a41950eab103a8b08c9442e5c9fa97` — the `main` head produced by the **chat /
  repository CLOSURE merge (PR #33, branch `task/ctde-chat-closure-handoff`)**, ordered
  parents `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96` then
  `802696b4adf702ef78aa459d470d3f24cb76cc49`. **That is the exact base this
  GENERALIZED-V1 HANDOFF BOOTSTRAP candidate was DERIVED ON**, and it is a statement about
  this record's derivation only — **not** a claim about live `main`, which this record's own
  integration necessarily advances past. Neither this documentation commit nor its future
  merge can name its own SHA, and inventing either would be a false provenance claim.
  **Every receiving orchestrator therefore resolves the live full `main` SHA from GitHub and
  re-reads `CLAUDE.md` and this handoff at THAT SHA — GitHub is authoritative for live
  branch and PR state, never this document** (§9).
- *(HISTORICAL BASE of the PREVIOUS record.)* `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96`,
  committed `2026-08-23 13:32:11 Asia/Jerusalem`, tree
  `8a0b7a0aa9e65ebf01fce99c9b27ee25886ba7a6` — the `main` head produced by the **Phase-B
  CTDE documentation/lock merge (PR #32)**, and the base the CHAT / REPOSITORY CLOSURE
  candidate was derived on. That candidate has since been INTEGRATED as the base named in
  the bullet above (PR #33), so this SHA is now historical provenance only — **not** a claim
  about live `main`. Neither this
  documentation commit nor its future merge can name its own SHA, and inventing either
  would be a false provenance claim. **Every receiving orchestrator therefore resolves the
  live full `main` SHA from GitHub and re-reads both documents at that SHA — GitHub is
  authoritative for live branch and PR state, never this document.** *(The PR #30 CTDE CODE
  integration `8390d85c2072e9cbe984ce5f2731cef3a9b14985`, tree
  `9686c107b8864f00a7d4403d70faf42ab561d2fb`, was the base of the PREVIOUS documentation
  record; it remains recorded as that code integration in the next bullet and in §3k.)*
- **PHASE-B CTDE CODE — CLOSED / REVIEWED / MERGED, AND ALREADY LIVE ON `main`.** Approved
  candidate `a6f3aa9d62931994f416b2241fec4cfac3b018ec`
  (`2026-08-22 21:01:46 Asia/Jerusalem`), integrated by merge commit
  **`8390d85c2072e9cbe984ce5f2731cef3a9b14985`** (PR #30, merged by a normal merge with the
  approved candidate preserved as the SECOND PARENT). Ordered merge parents:
  `d437084c5fb1a22c21596a48c58e03f7e15a0115`, then `a6f3aa9…`; integration tree
  `9686c107b8864f00a7d4403d70faf42ab561d2fb`. Grade A under `GPT_GITHUB`; append-only fix
  chain `d70d07f829a44e6f19100c338d4dde89f4f47bf6` (initial reviewed candidate) → `a6f3aa9…`
  (APPROVED), never amended, rebased or force-pushed. **SIX files**
  (`src/match_aou/rl/observation/central_graph_builder.py`,
  `src/match_aou/rl/training/graph_ppo.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_ctde.py`,
  `tests/test_graph_ppo.py`) — **no documentation file was part of that code integration**,
  which is exactly what THIS record closes. `CLAUDE.md` §5 owns the technical contract, §6
  the routing, §7 the lock and its evidence, §8 the un-run comparison; §3k summarizes it
  here. **NO SCIENTIFIC MEASUREMENT was executed for PR #30 — no baseline, no probe, no
  scientific rollout, and no actor-only vs CTDE comparison.** Bounded ENGINEERING smokes
  against real BLADE + BONMIN *did* run during BUILD validation (§3k); they are wiring
  evidence only and their outcomes are never scientific evidence.
- **PHASE-B CTDE DOCUMENTATION / LOCK — CLOSED / APPROVED / MERGED.** Approved candidate
  `c607f3fabcbd58f6f10cfde6bcc34068f09e4121`, verdict **`APPROVE`**, integrated by a normal
  merge commit **`7b6c07586811374f3f35e26ed33e1fcf4a9f2e96`** (PR #32, branch
  `task/phase-b-ctde-doc-lock`), ordered merge parents
  `8390d85c2072e9cbe984ce5f2731cef3a9b14985`, then `c607f3f…`. It recorded the PR #30 CTDE
  technical contract in `CLAUDE.md` §5, its routing in §6 and its lock in §7. **It changed
  no code, test, config or preset, executed no run, and pre-claimed no CTDE result.** With
  it merged the CTDE IMPLEMENTATION and its DOCUMENTATION are both CLOSED, and
  `task/phase-b-ctde-doc-lock` is a RETIRED, cleanup-eligible branch — **not** an active
  task.
- **`pre-ctde-actor-only` = `d437084c5fb1a22c21596a48c58e03f7e15a0115`** (tree
  `d7cc2dcb1b161180e272afc9600175f022c5b5d0`) is the IMMUTABLE reference preserving the
  IMMEDIATE PRE-CTDE actor-only state. It is the CTDE integration's FIRST parent, so "the
  actor-only state CTDE was merged onto" is a git fact rather than a claim. Preserving it
  was the CTDE integration gate's remaining prerequisite; **the gate is now CLOSED /
  SATISFIED on both halves**, and this ref **must not move**. It is DISTINCT from
  **`phase-a-baseline` = `4f0068847b017795717c5f0e331f647bcfc30547`**, the ORIGINAL Phase-A
  reference, which was never moved or repurposed for it and likewise must not move.
- **THE VARIABLE-SEVERITY MEASURED CODE SHA IS A DIFFERENT, DURABLE THING:**
  `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
  `dd881478b8e2e521054d09bc865437f1308be1a2` (committed
  `2026-08-20 14:50:24 Asia/Jerusalem`, the `main` head from the variable-severity
  documentation merge, PR #28) is the exact code SHA the approved FD-VARIABLE-SEVERITY-v1
  actor-only baseline was measured at, from a DETACHED clean snapshot (§3j). It is a
  durable MEASUREMENT identity and must never be read as a live head or as this record's
  base.
- **THE PHASE-A MEASURED CODE SHA IS A DIFFERENT, DURABLE THING:**
  `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` (`2026-08-17 19:25:42 Asia/Jerusalem`, the
  `main` head from the roster-integrity documentation merge, PR #25) is the exact code SHA
  the approved Phase-A long baseline was measured at (§3h). It is a durable MEASUREMENT
  identity and must never be read as a live head or as this record's base. The code state
  it names is preserved on the branch **`phase-a-baseline` =
  `4f0068847b017795717c5f0e331f647bcfc30547`**, which **must not move**. *(The PR #24
  roster/world-truth CODE integration commit `f37ea1c8559405d5de24a9c2dd9e740227acaeeb`
  was the base of an EARLIER documentation record; it remains recorded as that code
  integration in the next bullet and in §3g.)*
- **FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE — EXECUTED / INDEPENDENTLY REVIEWED /
  `APPROVE — VALID MEASUREMENT`; PRIMARY FINDING NEGATIVE.** Run root
  `C:\Users\Itama\f7r2` at exact measured code SHA
  `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, ONE invocation through `--config` from its
  own measurement contract, native exit code 0,
  `config_source.resolved_from = config_file`, `cli_overrides = []`, from a clean DETACHED
  snapshot. **664 scheduled / 586 successful / 78 accounted `setup` episode failures**,
  `accounting_reconciled = true`, **7/8 complete clean+mild+severe triads in all 11
  evaluation rounds**, ZERO infrastructure or data-integrity faults, 50/50 productive PPO
  updates. **THE PRIMARY BEHAVIOURAL RESULT IS NEGATIVE: no severity-conditioned FD-wake
  meta-action separation** — MILD and SEVERE both `PLAN_COMPLIANCE 7/7` at `pre_update` and
  at the final `post_update`, and identical distributions across all ten `post_update`
  rounds — even though the physical outcomes diverge sharply, so the severity factor itself
  is real. An earlier attempt at the SAME contract is **`INCONCLUSIVE/BLOCKED — INVALID
  MEASUREMENT`** (a Windows `MAX_PATH` playback-export failure removed the entire
  `post_update` SEVERE arm) and is EXCLUDED from every scientific reading. **It changed no
  tracked file** — a run of merged code, not a candidate. `CLAUDE.md` §7 owns the
  authoritative record, every denominator, the evidence hashes and the explicit non-claims;
  §3j summarizes it here. **BOTH run trees are PRESERVED** and must not be modified, moved,
  repackaged, deleted or regenerated.
- **FD-VARIABLE-SEVERITY-v1 CODE — CLOSED / APPROVED / MERGED.** Approved candidate
  `eecc9b5d91bce4a98a070a29307cc12af0d4c4a3`, integrated by
  `177e969446ef6c01c729484f2ea9969c94a27330` (`2026-08-20 12:15:28 Asia/Jerusalem`,
  PR #27). Ordered merge parents: `4f0068847b017795717c5f0e331f647bcfc30547`, then
  `eecc9b5…`. Candidate and integration share the IDENTICAL tree
  `37ebd8c56266fdd862cc7244c5f22a6ac95e438c`, and the candidate→integration comparison
  contains ZERO changed files. Grade A under `GPT_GITHUB`; append-only fix chain
  `73752d872a8cd17f703790ef41bee46a734170bb` (REQUEST FIXES) → `eecc9b5…` (APPROVED);
  FIVE files (`src/match_aou/rl/training/graph_fuel_damage.py`,
  `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_rollout.py`, `tests/test_graph_fuel_damage.py`,
  `tests/test_graph_train.py`). `CLAUDE.md` §5 owns the contract, §6 the routing, §7 the
  lock. **NO scientific baseline, training run, probe or rollout was executed for PR #27** —
  the measurement came later and separately (previous bullet, §3j). §3i summarizes the code
  lock.
- **ROSTER / WORLD-TRUTH INTEGRITY — CLOSED / APPROVED / MERGED.** Approved candidate
  `36365f210e8a659a641a7713f612c7e0ec1d4665` (`2026-08-17T14:01:10+03:00`), reviewed
  `APPROVE`, integrated by `f37ea1c8559405d5de24a9c2dd9e740227acaeeb`
  (`2026-08-17T15:48:30+03:00`, PR #24). Candidate and integration share the IDENTICAL tree
  `f801538080f2ad282766d32346580189fa949f0c`, so the integrated tree is exactly the reviewed
  tree. Grade A under `GPT_GITHUB`, FIVE files
  (`src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_setup_seam.py`,
  `tests/test_graph_train.py`, `tests/test_graph_fuel_damage.py`). **The approved
  semantics:** `solve_and_normalize()` stays ALLOCATED-ONLY and `belief_tasks` /
  `oracle_tasks` stay ALLOCATIONS — **an allocation is not a world inventory**. The world
  now comes from two RAW pre-solve snapshots, `EpisodeContext.known_target_ids` and
  `executed_target_ids`; beliefs are a SUBSET constraint on the known world rather than its
  denominator; hidden ids are executed minus known in executed-world order;
  `_require_scheduled_cell` checks the approved 3-known / 3-hidden / 6-total cell before
  fuel planning and before execution; and a roster/world-integrity fault is a
  `MeasurementIntegrityError` that **ABORTS** train and eval as infrastructure /
  data-integrity — never `EpisodeAttemptError`, `episode_failures.jsonl`,
  `skip_and_account_v1`, a condition tally or any scientific denominator. After
  `run_episode`, playback synchronization and confirmed-id validation precede the reward; an
  `incomplete` manifest truthfully lists real playback already written; and a manifest
  cannot become `complete` when expected and observed world counts disagree. Reward, PPO,
  the oracle allocation, fuel damage, B2, the seeds, the schedules, the tick loop, the
  executor, the generator and vendored BLADE are UNCHANGED. `CLAUDE.md` §5 owns the
  contract, §6 the routing, §7 the lock. **No training run, probe, rollout, seed sweep or
  baseline rerun occurred during the correction.** §3g summarizes it.
- **FIRST LONG BASELINE — EXECUTED / INDEPENDENTLY REVIEWED / SCIENTIFICALLY
  INCONCLUSIVE.** Run `training_output_long_baseline_100x8_seed0` at exact code SHA
  `c30b6982ba605d60976cc303256da4b5528b0e63`, one invocation, `cli_overrides = []`, native
  exit code 0. Engineering verdict **`REQUEST FIXES`**; scientific verdict **`INCONCLUSIVE —
  ROSTER/DATA INTEGRITY FAILED`**. §3f records its exact contract, timing, accounting,
  failure breakdown and evidence hashes; `CLAUDE.md` §7 owns the authoritative record and §8
  the gate. **It changed no tracked file** — a run of merged code, not a candidate. Its
  reward, paired-delta, survival, fuel-damage-yield and PPO-performance outputs are **raw
  historical outputs only and are not scientific evidence.** It is PRESERVED and must not be
  modified, moved, repackaged, deleted or regenerated.
- **CORRECTED-CELL SHORT-PROBE RERUN — EXECUTED / REVIEWED / VERDICT SUPERSEDED.**
  Run `training_output_20260816_162130`, from a clean checkout at exact code SHA
  `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, ONE invocation of the reviewed preset with
  no typed override. Originally reviewed **`VALID MEASUREMENT / CORRECTED SHORT-PROBE
  PASS`**; **that verdict is SUPERSEDED** by **`INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY
  REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR`** (§3f). Its own ledger accounts clean
  train seed 4 as a `setup` `EpisodeRosterError`, and PR #24 establishes that such a fault
  must ABORT rather than shrink a denominator — so **its reward and performance numbers are
  no longer scientific evidence, and it did not pass or release the long-baseline validity
  gate.** RETAINED unchanged: its run identity, invocation, provenance, mechanical
  accounting, artifact completeness, playback witnesses, and the OPERATIONAL WITNESSING of
  all three defect corrections. §3e records it and marks the supersession; `CLAUDE.md` §7
  owns the authoritative record and the evidence hashes. **It changed no tracked file.**
- **DEFECT C (RTB ISSUANCE is not physical RTB COMPLETION) — CLOSED / APPROVED /
  MERGED.** Approved candidate `ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b`, integrated by
  merge commit `0de9f21eb9e8904f06f836f4ecd010bc46c788b6` (PR #21). The candidate was
  merged with a MERGE COMMIT and preserved as its SECOND PARENT (integration parents, in
  order: `6e97940733d2c7cf8c4ffc7033180c65f644ae17` then `ea62e4e…`); candidate and
  integration share the IDENTICAL tree `6d05cc5ea9af0f6bdcd4a2d6865767bcbe525ebe`, and
  the candidate→integration comparison contains ZERO changed files. Implementation fixed
  base `6e97940733d2c7cf8c4ffc7033180c65f644ae17`. Grade A under `GPT_GITHUB`, mode
  BUILD, exactly SIX files (`src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`, `tools/graph_executor_smoke.py`,
  `tests/test_graph_setup_seam.py`, `tests/test_graph_fuel_damage.py`). The first
  candidate `5a0809df1a490df6ff266343788655d32fcefd81` was corrected by the NEW CHILD
  COMMIT `ea62e4e…` on the same branch and PR, with no amend, rebase, force-push or
  history rewrite. **The approved semantics:** `is_done(observation)` requires the LIVE
  post-step observation and has no observation-free default; assignment completion still
  comes from executor semantic state (`plans`, ego-private task resolution and the
  proximity-confirmed `done` set), while the PHYSICAL half comes from `_physical_state`
  (airborne / landed / removed) and `_note_dead` reconciles a newly observed death
  idempotently into `executor.dead` — for every ego, before the global verdict — so a
  death on the ride home reaches `EpisodeResult.n_dead` and the UNCHANGED reward formula
  charges the airframe really lost. `rtb_issued` keeps its ONE job as the single-issue
  BLADE-toggle guard and is never survival, landing or completion; an ego whose latch is
  set leaves Phase 1 entirely while peers continue normally and Phase 2 still runs every
  tick. The `add_return_to_base=False` contract is preserved and BLADE is unchanged.
  `CLAUDE.md` §4, §5 (Execution, Stage 1, and the tick loop), §6 and §7 own the
  authoritative contract, routing and lock. **This closes DEFECT C ONLY.** No probe,
  training run, rollout or baseline was run for it.
- **DEFECT B (the attack-confirmation wait derived from the salvo about to fly) —
  CLOSED / APPROVED / MERGED.** Approved candidate
  `39a16f2e5e1a3302d545c11b072e037e9702dffe`, integrated by merge commit
  `60a82d17398e9d14be1c2684cc72fafd020e0d9b` (PR #19). The candidate was merged with a
  MERGE COMMIT and preserved as its SECOND PARENT; candidate and integration share the
  IDENTICAL tree `ee86f0782ac50ee8bd0ee2fe634393a9cfc53a66`, and the
  candidate→integration comparison contains ZERO changed files. Implementation fixed base
  `cefda78b18ea2daeda5014bab9a75a0945ef8e37`. Grade A under `GPT_GITHUB`, mode SURGICAL,
  exactly TWO files (`src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `tests/test_graph_setup_seam.py`). The first candidate
  `45a0352312ae308df76a506a8e2e9907a9531a43` had its RUNTIME IMPLEMENTATION ACCEPTED and
  received REQUEST-FIXES on three DOCUMENTATION-accuracy findings; the correction landed as
  a NEW CHILD COMMIT on the same branch and PR, with no amend, rebase, force-push or history
  rewrite, and its runtime-relevant executor token stream is unchanged. `CLAUDE.md` §5
  (Execution, Stage 1), §6 and §7 own the authoritative contract, routing and lock.
  **This closes DEFECT B ONLY.** No probe, training run, rollout or baseline was run for
  it.
- **DEFECT A (ego-global `SELF_PRESERVATION_ABORT`) — CLOSED / APPROVED / MERGED.**
  Approved candidate `d56fda636ab5ec1a5cce6076f07acac5556d10cb`, integrated by merge
  commit `f094e0b32e5e67b79757edbfe4e73c1fe01b0a87` (PR #17). The candidate was merged
  with a MERGE COMMIT and preserved as its SECOND PARENT; candidate and integration share
  the IDENTICAL tree `70e5af2446f0a1b0674eb10819c9451753260560`, and the
  candidate→integration comparison contains ZERO changed files. Grade A under
  `GPT_GITHUB`, mode SURGICAL, exactly five files. The first candidate
  `c306455085de408c7bf383135c27e600ff3f1428` received REQUEST-FIXES for three
  documentation inaccuracies; the correction landed as a NEW CHILD COMMIT on the same
  branch and PR, with no amend, rebase, force-push or history rewrite. `CLAUDE.md` §5
  (Stage 4 SELECTION, Stage 5 EFFECT) and §7 own the authoritative contract and lock.
  **That closed DEFECT A ONLY.** No probe, training run, rollout or baseline was run for
  it.
- **B1 — CLOSED / MERGED / LOCKED.**
  `d6758ac1899621b2ceebcb63afb5e8577184cd91`, merged by
  `bd087c3c18b96f1fe847b4987c73f394a43249c1` (PR #2).
- **B2 — CLOSED / MERGED / LOCKED.**
  `e22aee359e06591bdb179ef06a566db90f83a558`, merged by
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3).
- **B3 — CLOSED / MERGED / LOCKED.**
  `dd14ab418c71e3bd615f1198d0c612502642d29b`, merged by
  `14224531db9deb700f6e397203177eb8c701c6cc` (PR #4).
- **B4 preparation — CLOSED / MERGED / LOCKED.**
  `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c`, merged by
  `ba936606deada050ed9298600ee9041fc330af6c` (PR #6).
- **First real post-B3 instrumented probe — CLOSED / REVIEWED MEASUREMENT of the
  PRE-FD cell.** Ran from the clean exact code SHA
  `a3f0838616990987bcb8a51665fa75d84edf5952`. §2 records what it measured and what it
  does NOT establish.
- **B4 per-episode observability follow-up — CLOSED / MERGED / LOCKED.**
  `211e12e49b676637362d42effdb80988dd0e55eb`, merged by
  `ffb95a6ee90df45b2d89802b321dcadcbc272821` (PR #7). Candidate
  `24241690572a7a5264e24348db5e9412b41bc47a` received REQUEST-FIXES; the approved
  correction was a new commit, never rewritten.
- **FD-BASELINE-v1 — CLOSED / MERGED / LOCKED, and it remains so.** Approved candidate
  `a8669f450708c2508753c49ab16fd1028b29607d`, integrated by
  `1cecb0ac99f839d47ffeea12c8871aec77e66640` (PR #8); the merged tree was independently
  verified identical to the approved candidate tree. The FIRST candidate
  `1cf53fcee3ee05b3466c8391cbc6bb04420a0985` received REQUEST-FIXES; the correction landed
  as a NEW CHILD COMMIT on the same branch and PR, with no amend, rebase, force-push or
  history rewrite. §3 summarizes the factor; `CLAUDE.md` §5 and §7 own the authoritative
  contract and lock. Nothing in the visual-artifact work below changed it.
- **FINAL-CELL-VISUAL-ARTIFACTS — CLOSED / MERGED / REVIEWED.** Approved candidate
  `24d1835f31d2e6aac04b418308a8753c392ac951`, integrated by
  `771f2107211fb3f984b64482b799613260e19aca` (PR #10); the merged tree was verified
  byte-identical to the approved candidate tree (`git diff --quiet 24d1835 771f210`).
  Grade A under `GPT_GITHUB`, mode SURGICAL, exactly two files
  (`src/match_aou/rl/training/graph_train.py`, `tests/test_graph_train.py`). §3b
  summarizes it; `CLAUDE.md` §5, §6 and §7 own the contract, routing and lock.
- **Repository code-hygiene cleanup — CLOSED / APPROVED / MERGED.** Approved candidate
  `2a3f89cf2d027581308493a98767ae658107d6d1`, integrated by
  `6e2757dd30100f429d492f4d23fd8b5f57cf4fac` (PR #11). Grade A under `GPT_GITHUB`, mode
  SURGICAL. The retired `blade_executor_minimal.py` is **removed from current `main`**, and
  its one live helper `nearest_neighbor_order` now lives in
  `src/match_aou/utils/scheduling_utils.py`, imported by BOTH `GraphPlanExecutor._eligible`
  and `graph_hidden_placement.predict_route` — so online execution and offline route
  prediction still share exactly ONE implementation. Verified no behaviour change: placement
  geometry and executor eligibility were byte-identical to the pre-merge base over seed and
  world sweeps, full suite 216 passed / 4 skipped, import purity 12/12. No training,
  rollout, BONMIN solve or scientific probe was run. `CLAUDE.md` §7 owns the lock.

- **FINAL-CELL PROBE HARNESS — CLOSED / MERGED / APPROVED.** Approved candidate
  `61e539ed62fcf1e3fe25a83d213cae06f5afa98e`, integrated by
  `a5f389a2af328640e19db51d3277a33167c08f25` (PR #14); GitHub verification established
  that candidate…merge has ZERO changed files, so the merged tree is exactly the reviewed
  tree. Grade A under `GPT_GITHUB`, mode SURGICAL — **the grade was corrected from B to A
  during review** because `graph_train.py` belongs to the §5 locked trainer contract, with
  no implementation redo required. Four files: `src/match_aou/rl/training/graph_train.py`,
  `tests/test_graph_train.py`, `README.md`, and the new
  `configs/graph_train/final_cell_probe.json`. TWO REQUEST-FIXES rounds landed as new child
  commits on the same branch and PR (`4238e0e` → `de51883` → `61e539e`) with no amend,
  rebase, force-push or history rewrite. §3c summarizes it; `CLAUDE.md` §5, §6 and §7 own
  the contract, routing and lock.

- **No scientific probe or training run had been performed on the merged final cell at the
  time of the locks above.** Every test behind every one of these locks is solver-free and
  drives the pipeline through stubbed engine seams. The locks certify implementation; they
  say nothing about how the cell behaves. **The harness lock in particular measured
  nothing**: it certifies that a probe can be configured, run and read, not that the cell
  learns anything. **THAT SCIENTIFIC MEASUREMENT GAP IS STILL FULLY OPEN.** The executed
  runs that followed — the first short probe (§3d), the corrected rerun (§3e) and the long
  baseline (§3f) — expanded OPERATIONAL, DIAGNOSTIC and PLAYBACK evidence substantially:
  they proved the harness runs and accounts for itself, exposed four real defects, and
  witnessed three of those defects' corrections in real playback. **None of them closed any
  part of the scientific PERFORMANCE-measurement gap**, because none of them is a valid
  measurement of the cell (§3e, §3f, `CLAUDE.md` §8).
- **FIRST BOUNDED SHORT PROBE OF THE FINAL FUEL-DAMAGE CELL — EXECUTED / REVIEWED /
  SCIENTIFICALLY INCONCLUSIVE.** Run `training_output_20260815_173029`, executed from a
  clean checkout at exact code SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`. It confirms
  HARNESS AND ACCOUNTING OPERABILITY and nothing more: the process exited normally,
  `run_summary.json` reported `accounting_reconciled=true`, training accounting was 8
  attempted / 6 successful / 2 `setup` failures, evaluation was 16/16 successful with 4/4
  complete matched pairs in BOTH the `pre_update` and the `post_update` round, and two
  productive PPO updates completed. It ALSO exposed three research-validity defects in
  merged, previously locked behaviour — abort semantics, premature attack re-fire, and
  episode termination on RTB ISSUANCE rather than RTB COMPLETION. **Its post-update reward
  improvement is therefore NOT scientific evidence about the fuel-damage cell**. §3d records
  the run state, the three defects and the decided direction. **All three defects are now
  CORRECTED and MERGED** (see the Defect-A, Defect-B and Defect-C bullets at the top of
  this section and `CLAUDE.md` §7). That does NOT rehabilitate this run: it was executed at
  `238062d…`, before any of the three corrections existed, so its numbers remain historical
  evidence about the OLD behaviour and are not evidence about the corrected code. The
  SEPARATE corrected-cell rerun is recorded in §3e; it was **ORIGINALLY judged VALID**, and
  **that former verdict is SUPERSEDED** by `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY
  REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR` (§3e, §3f). **NO run of the fuel-damage
  cell has passed the validity gate** — neither this probe, nor that rerun, nor the long
  baseline.
- **Repository documentation hygiene — CLOSED / APPROVED / MERGED.** Approved candidate
  `52064c2d306df7c8447d159df20e6e189a59bf85`, integrated by
  `5f78904e3af1e2e47386c9b0e01ddbaa273724f5` (PR #12); the approved candidate tree was
  verified identical to the integration tree. Grade C, documents and dead data only —
  `README.md` replaced, `docs/BLADE_API_DOCUMENTATION.md` audited against the vendored
  fork, four obsolete scenario JSONs and two dead utility symbols removed. It changed no
  research behaviour. The FIRST candidate `6302847bdc8b5e40313763b4b167af85dd0a462e`
  received REQUEST-FIXES; the correction landed as a new child commit, never rewritten.
  `CLAUDE.md` §7 owns the lock.
- **All PRIOR repository-hygiene work is CLOSED, and this document is the phase's final
  closure record.** Both hygiene tasks are merged (PR #11 code, PR #12 documentation) and
  their branches `task/repo-code-hygiene` and `task/repo-doc-hygiene` have been **deleted**
  locally and remotely by safe deletion, each verified an ancestor of `main` first; every
  reviewed candidate tip stays reachable on GitHub through `refs/pull/<n>/head`. The earlier
  implementation branch `task/final-cell-visual-artifacts` was likewise verified and
  deleted. `flat-final` (`4d44c3454a5561a6cb9d7aed593d59a40068d6d7`) and the `pre-cleanup`
  tag (peeling to `561b7cb7f2d873e584a8c0dabe71df8050f1b4ed`) are untouched.
  No further repository-hygiene task follows it. **Live branch and PR state — including
  whatever delivered this record — must be resolved from GitHub, never from this document**,
  and the next orchestrator performs fresh exact-SHA initialization: **resolve the current
  full `main` SHA from the repository** rather than reusing any SHA named here.
- **VOLATILE STATE — WHAT IS DONE, WHAT IS WRITABLE, AND WHO OWNS IT (updated 2026-08-25).**
    *(The 2026-08-22 entry recorded the variable-severity measurement as RUNNING and NOT
    REVIEWED, and the 2026-08-23 entry recorded the repository as CLOSED / IDLE after the
    closure record merged; both were true when written and are superseded by the state
    below.)*
  - **FD-VARIABLE-SEVERITY-v1 MEASUREMENT: CLOSED — EXECUTED, INDEPENDENTLY REVIEWED,
    `APPROVE — VALID MEASUREMENT`, PRIMARY FINDING NEGATIVE** (§3j). Measured on the
    immutable DETACHED snapshot at exact SHA
    `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
    `dd881478b8e2e521054d09bc865437f1308be1a2`, clean working tree. Its `MAX_PATH`
    precursor is `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT` and excluded. `CLAUDE.md` §7
    owns the authoritative record; nothing beyond it may be claimed.
  - **PHASE-B CTDE: IMPLEMENTATION IS MERGED.** **PR #30** (branch
    `task/phase-b-ctde-build`) was REVIEWED and APPROVED on head
    `a6f3aa9d62931994f416b2241fec4cfac3b018ec` and INTEGRATED as
    `8390d85c2072e9cbe984ce5f2731cef3a9b14985` (§3k). Its technical contract is now
    AUTHORITATIVE in `CLAUDE.md` §5, routed in §6 and locked in §7 — *(historical, already
    traversed: CTDE was required to ENTER through DESIGN / RECON before implementation, and
    an earlier revision of this document observed PR #30 read-only as an unmerged draft)*.
    **What remains OPEN is the SCIENTIFIC COMPARISON, not the implementation** (§4,
    Task 9), and **no CTDE benefit is claimed**.
  - **ACTIVE PHASE: GENERALIZED-V1 — §3l.8 STEPS 1, 2, 3, 4 AND 5 ARE ALL MERGED (PR #35,
    PR #36, PR #38, PR #40, and the Task-5 sequence PR #42 → `5dfcd8b6…`, PR #43 →
    `b3c2e01f…`, PR #44 → `9b9e9b85…`), so §3l.1–§3l.7 are IMPLEMENTED and the Task-5 stack
    is INTEGRATED (§3m.1, §3m.5)** (§3l). *(SUPERSEDED: this bullet previously read "STEP 5
    IS IMPLEMENTED AND APPROVED BUT NOT MERGED (PR #42 + PR #43, both FROZEN)".)* The GPT
    orchestrator owns the work. The FOUR
    low-level policy seams are OPT-IN with historical defaults and are resolved TOGETHER,
    and only together, by the ONE `episode_design` selector whose DEFAULT is
    `fixed_cell_v1`; the generalized cardinality sampler and the SEPARATE `fuel_damage_mode`
    field sit beside that resolution and are NOT policy ids on `EpisodeDesign`.
    **GENERALIZED-V1 TASK 4 IS CLOSED — no writable implementation branch, no active PR, no
    candidate under review.** The R1 benchmark **SCALE IS SELECTED AND AUTHORIZED**
    (`worlds_per_cell = 3`) and its **CONSTRUCTION IS AUTHORIZED AND DISPATCHED** (candidate
    base seed `840000`, `max_candidates_per_cell = 12`); **no concrete R1 manifest has yet
    been independently reviewed or approved as the comparator and none is committed or
    tracked in the repository**; and **no generalized measurement RESULT exists.** A first
    generalized ACTOR-ONLY R1 long run **is** authorized and dispatched, with its result
    PENDING and unreviewed (§3m.4; the four facts in full are in the block above).
    *(SUPERSEDED: this bullet previously read "STEPS 1, 2 AND 3 ARE MERGED … §3l.6–§3l.7 ARE
    NOT IMPLEMENTED … neither harness selects them", then "STEP 5 … IS NOT STARTED", then
    "no generalized measurement exists, is running, is scheduled or is authorized", and then
    "No FINAL SCIENTIFIC benchmark scale has been SELECTED and no FINAL SCIENTIFIC benchmark
    population or manifest has been … scheduled or authorized". Each was accurate when
    written; not now.)*
  - **WRITABLE STATE — STATED IN STABLE POST-INTEGRATION FORM, so this bullet does not go
    stale when a documentation record merges.**
    - **NO documentation record is a standing writable task.** Each one is the sole writable
      candidate only while its own draft PR is open; on integration that branch and PR become
      HISTORY like every row in the parenthetical below, and the repository carries **NO
      writable repository task and NO active candidate** until the next authorized task
      opens one.
    - **A CLEAN CHECKPOINT INSIDE AN ACTIVE PHASE IS NOT A CLOSED / IDLE REPOSITORY:**
      **GENERALIZED-V1 remains the ACTIVE project phase**, §3l.8 **Tasks 1, 2, 3, 4 and 5 are
      ALL MERGED (§3m.1, §3m.5)** — and the phase stays ACTIVE because **R1 is pending**, not
      because any repository task is open. *(SUPERSEDED: this line previously read "Tasks 1, 2
      and 3 are
      MERGED, and Task 4 (§3l.6–§3l.7) is the NEXT UNRESOLVED IMPLEMENTATION TASK", then
      "TASK 4 IS CLOSED, and Task 5 … is the NEXT UNRESOLVED STEP", then "Task 5 is
      IMPLEMENTED AND APPROVED as a STACKED, STILL-UNMERGED TWO-PR STACK". Each was accurate
      when written; not now.)*
    - **THE TASK-5 CODE CANDIDATES ARE MERGED, AND THEIR BRANCHES ARE RETIRED.** **PR #42**
      (`312f58650b61a85eb72d0554d60715afee862a5c` → `5dfcd8b632be8dca3c1730018bbf35337d07f077`)
      and **PR #43**
      (`4af6c5aa5dd28072692bfda63282964b55010aae` → `b3c2e01f130afe854b09384cd6e1e196de714795`),
      with the documentation lock **PR #44**
      (`88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` → `9b9e9b85a70c8a0019c72ada92ceec3401725795`),
      are APPROVED and MERGED: **do not
      modify, push to, rebase or force-push any of their retired branches, which are
      cleanup-eligible rather than writable.** **NO MERGE IS AUTHORIZED BY ANY
      DOCUMENTATION RECORD**; the §3m.5 integration was performed after GPT exact-SHA review
      plus explicit user authorization, with
      PR #43 and PR #44 each **EXACT-BASE RE-REVIEWED after retargeting**. *(SUPERSEDED: this
      bullet previously read "TWO CODE CANDIDATES EXIST AND BOTH ARE FROZEN / READ-ONLY …
      APPROVED and NOT merged".)*
      *(SUPERSEDED: these lines previously read "TASK 5 IS NOT STARTED AND NOT AUTHORIZED …
      nothing is implemented" and "NO IMPLEMENTATION CANDIDATE IS ACTIVE — the last CODE
      candidate was PR #40 and it is merged". Both were accurate when written.)*
    - **THE EXTERNAL GENERALIZED-V1 LONG-RUN TASK IS RUN-ONLY AND OWNS NO REPOSITORY
      WRITES.** It may execute and produce artifacts outside the repository; it may not
      create, edit, push or merge any branch, and it commits no run artifact, config, preset
      or benchmark manifest. **This documentation candidate is the ONE current writable
      repository task, and no other branch may be edited concurrently** (§3m.2).
    *(Historical: `task/generalized-v1-cardinality-b2` merged as PR #35,
    `task/generalized-v1-fd-adaptation` as PR #36, `task/generalized-v1-task12-doc-lock` as
    PR #37, `task/generalized-v1-task3-continuation-reference` as PR #38,
    `task/generalized-v1-task3-doc-lock` as PR #39 and
    `task/generalized-v1-task4-harness-benchmark` as PR #40;
    `task/generalized-v1-handoff-bootstrap`
    was the writable task under an EARLIER record and merged as PR #34; before it,
    `task/ctde-chat-closure-handoff` (PR #33) and `task/phase-b-ctde-doc-lock` (PR #32);
    earlier still, the FD closure orchestrator held a ONE-TIME writable exception scoped to
    the variable-severity record alone, on branch
    `task/fd-variable-severity-valid-doc-lock`, and that exception ENDED when the record was
    integrated and writable ownership returned to the CTDE orchestrator.)*
  - **REFERENCES THAT MUST REMAIN — FOUR preserved branches plus one tag, with DISTINCT
    roles that are NEVER interchangeable.** **`main`**; **`phase-a-baseline` =
    `4f0068847b017795717c5f0e331f647bcfc30547`**, the IMMUTABLE ORIGINAL Phase-A scientific
    reference and the historical comparator for the first CTDE scientific comparison (§3h,
    §4 Task 9); **`pre-ctde-actor-only` = `d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the
    LATER immediate actor-only state onto which CTDE was merged (the CTDE integration's
    FIRST parent); and **`flat-final` = `4d44c3454a5561a6cb9d7aed593d59a40068d6d7`**,
    preserving the retired flat-RL path. The annotated tag **`pre-cleanup`** (peeling to
    `561b7cb7f2d873e584a8c0dabe71df8050f1b4ed`) is preserved alongside them. **Neither
    `phase-a-baseline` nor `pre-ctde-actor-only` supersedes the other and neither may be
    substituted for the other. NONE of these refs is cleanup-eligible, and none may move.**
  - **CLEANUP-ELIGIBLE TASK BRANCHES — RETIRED, NOT ACTIVE.** Each was independently
    verified to be a strict ancestor of `main` at
    `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96`, carrying no unintegrated commit:
    `task/ctde-parallel-order-doc-lock` (`1aa8eef865351959d61e229de86040620cb2cb50`),
    `task/fd-variable-severity-valid-doc-lock`
    (`92941587b1ad225573af763e50b129a552861b18`), `task/phase-b-ctde-build`
    (`a6f3aa9d62931994f416b2241fec4cfac3b018ec`) and `task/phase-b-ctde-doc-lock`
    (`c607f3fabcbd58f6f10cfde6bcc34068f09e4121`). They are ELIGIBLE for the immediately
    following BOUNDED branch-cleanup task and were deliberately NOT deleted by this record.
    `task/ctde-chat-closure-handoff` (`802696b4adf702ef78aa459d470d3f24cb76cc49`) joined
    them once PR #33 merged. `task/generalized-v1-handoff-bootstrap` joined them
    once PR #34 merged, and `task/generalized-v1-cardinality-b2` /
    `task/generalized-v1-fd-adaptation` joined them once PR #35 / PR #36 merged, and
    `task/generalized-v1-task12-doc-lock` /
    `task/generalized-v1-task3-continuation-reference` joined them once PR #37 / PR #38
    merged. Any documentation branch becomes cleanup-eligible only AFTER its own candidate is
    reviewed and integrated. **No documentation record deletes a branch or moves a ref**
    (§6).
  - **CTDE INTEGRATION GATE — CLOSED / SATISFIED ON BOTH HALVES.** The requirement that the
    variable-severity measurement COMPLETE and receive an **INDEPENDENT VALIDITY VERDICT**
    is **MET** (§3j) — met by a NEGATIVE result, which satisfies the gate exactly as a
    positive one would, because the gate tests VALIDITY, not favourability. The second
    requirement — a NEW immutable actor-only pre-CTDE reference preserved from the
    then-current actor-only state — is **MET** by **`pre-ctde-actor-only` =
    `d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the CTDE integration's FIRST parent,
    which **must not move**. `phase-a-baseline` remains historical provenance for the
    ORIGINAL Phase-A reference and was never moved or repurposed for it. **PR #30 itself ran
    NO scientific comparison** (§4, Task 9). An OLD-CONTRACT CTDE measurement has since been
    executed and is **OUT OF SCOPE for the generalized redesign** — not reviewed, compared
    or claimed here (§1) — and the fixed-cell Task-9 framing is **no longer the live next
    action** (§4, §8).

  The integrating merge's SHA for THIS record is deliberately NOT named: it does not exist
  while this is written, and inventing it would be a false provenance claim. **Live branch
  and PR state is resolved from GitHub, never from this document.**
- **POST-MERGE REPOSITORY CLEANUP — DONE.** *(Historical: this was a GATE on the long-baseline
  rerun, and it was satisfied before that run. It is recorded here as completed, not as
  outstanding work.)* All THREE obsolete task branches were safely retired
  — verified against the live repository at `737b4bf`, where only `main` and the preserved
  `flat-final` remain and no open PR exists:
  - `task/roster-world-truth-fix` (the merged PR #24 code candidate);
  - `task/long-baseline-execution` (the branch the inconclusive long baseline recorded);
  - `task/roster-world-truth-doc-lock` (that documentation branch, once merged as PR #25).

  **Each deletion happened ONLY after that branch's tip was verified reachable from the
  integrated `main` history or from its applicable merged PR** — safe deletion, never a
  force delete of unreachable work; every reviewed candidate tip also stays reachable on
  GitHub through `refs/pull/<n>/head`. **EXPLICITLY PRESERVED and NOT part of that cleanup:**
  branch `flat-final` (`4d44c3454a5561a6cb9d7aed593d59a40068d6d7`), the annotated tag
  `pre-cleanup` (peeling to `561b7cb7f2d873e584a8c0dabe71df8050f1b4ed`), **every preserved
  scientific artifact** (`training_output_20260815_173029`,
  `training_output_20260816_162130`, `training_output_long_baseline_100x8_seed0`, and now
  the approved `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`), and all
  GitHub PR refs and history. **The branch `phase-a-baseline`
  (`4f0068847b017795717c5f0e331f647bcfc30547`), which preserves the Phase-A reference code
  state, is likewise PRESERVED and must not move.** **The same discipline now applies
  to the FOUR cleanup-eligible `task/*` branches enumerated in the VOLATILE STATE
  bullet above, and to `task/ctde-chat-closure-handoff` once THIS record is merged** —
  retirement is the GPT
  orchestrator's action, only after each tip is verified reachable from integrated `main`,
  and it is NOT part of this documentation task. *(Historical: the branches
  `task/variable-fd-severity-baseline` and `task/variable-fd-severity-doc-lock`, named by an
  earlier revision of this bullet, are no longer present on the remote.)*

## 2. Historical probe — evidence about the EASY PRE-FD cell only

Measured code: exact clean SHA `a3f0838616990987bcb8a51665fa75d84edf5952`. Shape: two
iterations × four scheduled train attempts, seeds `[0,8)`, with the same fixed four
held-out seeds `[1000000,1000004)` before training and after two completed updates.

- Provenance complete (exact SHA, `dirty=false`, Windows / `nlp_env`, vendored BLADE,
  BONMIN); the run completed normally in 79.21 s.
- `pre_update`, `updates_completed=0`: reward mean `-0.4999997395829586`, **4/4**.
- Training **7/8** successful, all with wakes; **24 transitions**; two productive
  iterations and two PPO updates. The single failure was train seed 2 at `setup` — B2
  produced two placements for three requested hidden targets — attempted once, recorded
  once, never retried or replaced.
- `post_update`, `updates_completed=2`: reward mean `5.000007394910353e-7`, **4/4**.
- `run_summary.json:accounting_reconciled=true`; all six durable artifacts existed.
- Evidence SHA-256 for the six artifacts is recorded in `CLAUDE.md` §7 under `a3f0838`.

**That cell contained NO difficulty factor.** It is preserved as historical evidence that
the pipeline collects data and updates, and that the easy cell had headroom the loop could
close. It is **not** a baseline, and it is **not** evidence about the fuel-damage cell —
those numbers must never be reused as the new cell's expected behaviour, and the probe in
§4 must not be judged against them.

## 3. What PR #8 closed — FD-BASELINE-v1

Authoritative contract: `CLAUDE.md` §5 ("FD-BASELINE-v1 — the difficulty factor") and §4
for the tick placement. Lock, fix chain and verification: `CLAUDE.md` §7 under `a8669f4`.
Summarized here only far enough to hand over.

**The selected factor and its research role.** The pre-FD cell was learned in a two-update
probe: the static plan already sat close to the oracle, and the only adaptation on offer
was engaging a pop-up the ego happened to fly past. FD-BASELINE-v1 adds exactly ONE
attributable source of adaptation difficulty — a seeded, ego-local, one-shot fuel-damage
event partway along an ego's first planned leg, sized so that flying home stays feasible
while completing the route and returning does not. That turns `SELF_PRESERVATION_ABORT`
from a never-correct action into a live alternative, and the explicit
`RewardConfig(aircraft_penalty_coeff=2.25)` is what gives losing the airframe a cost.

**The scenario cell is otherwise unchanged**: 3 agents, 3 known + 3 route-relative hidden
airbase targets, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams=False`,
`probability = 1`, unchanged BLADE weapon lethality, frozen solver, unchanged PPO, and the
`graph_reward` formula untouched.

**Deferred, NOT bundled.** `probability < 1`, enemy fire / SAMs and dense (per-wake) reward
were considered and **not selected**. Each remains a separate research change with its own
semantics, observability and proof obligations, and none may be enabled implicitly — see
`CLAUDE.md` §8. In particular `p` remains 1, so the reward operand scale is not reopened.

**Verification at the approved head** (`CLAUDE.md` §7 has the full statement): full suite
192 passed / 4 skipped, 35 fuel-damage tests, 73 graph-train tests, import purity 12/12,
`graph_trigger` selftest green, `git diff --check` clean.

## 3b. What PR #10 closed — FINAL-CELL-VISUAL-ARTIFACTS

Authoritative contract: `CLAUDE.md` §5 ("Visual artifacts — the opt-in inspection
surface"); routing in §6; lock and verification in §7 under `24d1835`. Summarized here
only far enough to hand over.

**What it is for.** A finished probe is otherwise a directory of numbers. With
`TrainConfig.visual_artifacts` / `--visual-artifacts` enabled, every scheduled
`pre_update` / `train` / `post_update` attempt keeps one collision-free bundle under
`<run_dir>/visual_artifacts/`: the byte-identical generated known-only scenario, the
authoritative executed t=0 scenario taken from `ctx.game.export_scenario()` on the env-2
game before the fuel-damage controller exists, the BLADE playback produced through the
existing `setup_episode(recording_export_path=...)` + `run_episode` contract, and an
`artifact_manifest.json` stating phase, iteration, `updates_completed`, ordinals, exact
seed, scheduled condition and exact scenario tag. A whole bundle is `complete`; a bundle
left behind by a failed attempt is clearly marked `incomplete`.

**What it is not.** It is observation, not measurement. Nothing captured is read back into
the pipeline. **OFF by default, and OFF is byte-unchanged** — no directory, no identity, no
copy, no `Game.export_scenario` call, and neither the `recording_export_path` nor the
`artifacts` keyword is passed at all. No seed, scenario tag, scenario name, RNG draw,
policy inference, PPO input, solver input, reward, fuel-damage semantic, failure taxonomy
or BLADE behaviour changed.

**Failure routing.** An artifact filesystem / serialization failure is INFRASTRUCTURE:
`_VisualArtifactError` is re-raised ahead of the broad episode handlers, aborts the run
loudly, and never enters `skip_and_account_v1` or `episode_failures.jsonl`, so it cannot
shrink a scientific denominator by masquerading as an episode failure.

**Verification at the approved head** (`CLAUDE.md` §7 has the full statement):
`tests/test_graph_train.py` 89 passed in BOTH environments, setup-seam + fuel-damage
regressions 66 passed / 4 skipped, import purity 12/12, full suite 208 passed / 4 skipped,
`git diff --check` clean, plus three mutation checks confirming the load-bearing tests
falsify.

**No live BLADE/BONMIN probe, training run, rollout, artifact-generating smoke or
scientific baseline was performed** for either PR #8 or PR #10.

## 3c. What PR #14 closed — the FINAL-CELL PROBE HARNESS

Operator-facing only. `CLAUDE.md` §5 ("Experiment harness") owns the contract; this is the
short form.

- **`--config <path>`** loads a JSON preset of `TrainConfig` FIELDS (nested PPO knobs under
  `"ppo"`). Precedence: dataclass / CLI defaults < preset < EXPLICITLY typed CLI flags.
  `TrainConfig` stays the configuration authority; an unrecognized key raises.
- **`configs/graph_train/final_cell_probe.json`** is the repository's ONLY preset and is
  the bounded short probe: 2 scheduled training iterations × 4 scheduled attempts, base
  seed 0, `eval_every` 2, four fixed held-out seeds from 1_000_000 → one `pre_update` and
  one `post_update` matched round, the final 3-agent / 3-known / 3-hidden cell,
  FD-BASELINE-v1, visual artifacts ENABLED. **Two scheduled iterations do NOT imply two
  productive PPO updates** — `updates_completed` may be 0, 1 or 2, since a zero-wake
  iteration runs no epochs.
- **`run_config.json:/config_source`** is always a structured object with exactly three
  truthful `resolved_from` kinds: `config_file`, `cli_defaults`, `direct_config`.
- **Figures moved to `<run_dir>/plots/`** and the single four-panel `training_plot.png` is
  retired: `training_performance.png`, `policy_diagnostics.png`, `measurement_health.png`.
  Training reward is separated from held-out evaluation; clean and damaged held-out means
  are shown separately and are each over THAT condition's own successful episodes; the
  within-seed comparison is the matched-pair delta over COMPLETE pairs; measurement health
  preserves the episode, pair, wake and per-condition denominators; and the shared x
  quantity is PPO updates completed before the measurement.
- **Run organization is unchanged in kind**: everything still lives under ONE run root —
  records, `scenarios/`, `checkpoints/`, optional `visual_artifacts/`, and now `plots/`.
- **It ran no probe.** No BONMIN solve, BLADE episode, training run, rollout or selftest
  was executed for this PR; every test is solver-free, and the figures were rendered from
  SYNTHETIC records.

## 3d. The executed bounded short probe — operability CONFIRMED, scientific reading BLOCKED

Run identifier `training_output_20260815_173029`, executed from a clean checkout at exact
code SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`, in the merged preset's shape: 2
scheduled training iterations × 4 scheduled attempts, plus the fixed held-out matched
`pre_update` / `post_update` rounds.

**What it establishes — harness and accounting OPERABILITY only.**

- the process exited normally;
- `run_summary.json` reported `accounting_reconciled=true`;
- training accounting: **8 attempted, 6 successful, 2 `setup` failures**;
- evaluation accounting: **16/16 successful**, with **4/4 complete matched pairs in the
  `pre_update` round and 4/4 in the `post_update` round**;
- **two productive PPO updates** completed.

These facts say the instrument runs and accounts for itself. They do **not** authorize the
long baseline, because the same run exposed the three research-validity defects below.

**Defect A — `SELF_PRESERVATION_ABORT` WAS node-indexed, not an ego-global abort.**
**STATUS: CLOSED / APPROVED / MERGED through PR #17** — approved candidate
`d56fda636ab5ec1a5cce6076f07acac5556d10cb`, integrated by
`f094e0b32e5e67b79757edbfe4e73c1fe01b0a87`, identical tree
`70e5af2446f0a1b0674eb10819c9451753260560` and a zero-file candidate→integration
comparison. The observations below are preserved as HISTORICAL EVIDENCE about the probe's
own SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`; they describe behaviour that no longer
exists on current `main`.

- At this SHA `graph_effect.apply_meta_action` removes only the assignment(s) whose
  `task_idx == node_v`, so SPA aborts ONE task rather than the ego's mission.
- Probe playback showed a fuel-damaged KC-135 selecting SPA while its existing BLADE route
  continued and further assignments remained.
- **The decided behaviour — now IMPLEMENTED and MERGED:** an **ego-global mission
  abort**. Selecting `SELF_PRESERVATION_ABORT` on ANY legal cell clears ALL of that ego's
  remaining assignments, so the executor reaches its empty-plan RTB path. It was **NOT
  implemented at the SHA above**; it IS implemented on current `main` (`f094e0b`).
- The existing `k × 3` action-head structure was **not** redesigned: the action surface,
  `NUM_META_ACTIONS`, the mask rules and the sampled/stored/PPO-re-scored `(node, meta)`
  cell are all unchanged, and only the EFFECT of the abort cell became ego-global. The
  merged implementation is proven end to end — pure effect layer, the real
  `graph_tick_loop._wake_decision` chain, and a solver-free REAL-BLADE test in which the
  stale route is replaced by the ride home (`CLAUDE.md` §7).
- Execution-seam fact that narrows the diagnosis: `graph_tick_loop._wake_decision` already
  resyncs the edited ego plan before Phase 2, and an actually EMPTY plan should make
  `GraphPlanExecutor` emit `aircraft_return_to_base`, whose BLADE handling replaces the
  stale route with the home-base route. The observed stale route is therefore currently
  explained by SPA not emptying the plan — **not** by evidence of a missing resync call.

**Defect B — premature re-fire exhausts weapons.**
**STATUS: CLOSED / APPROVED / MERGED through PR #19** — approved candidate
`39a16f2e5e1a3302d545c11b072e037e9702dffe`, integrated by
`60a82d17398e9d14be1c2684cc72fafd020e0d9b`, identical tree
`ee86f0782ac50ee8bd0ee2fe634393a9cfc53a66` and a zero-file candidate→integration
comparison. The observations below are preserved as HISTORICAL EVIDENCE about the probe's
own SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`; they describe behaviour that no longer
exists on current `main`.

- In the `post_update` damaged eval seed `1000003`, B-2 Spirit #698 engaged its
  route-relative hidden targets successfully but reached the final known target
  `Floridistan AFB #4067` with **zero onboard weapons**, and then remained over it until
  fuel exhaustion.
- **Artifact RECONSTRUCTION of the sequence — read as a reconstruction, not as a
  controlled measurement:** at approximately t=5140 the final 2 AIM-120 launched at Hidden
  Airbase #003; at approximately t=5240 2 AIM-9 launched at Hidden Airbase #001 from about
  47.2 km; at approximately t=5300, before that slower AIM-9 salvo resolved, the fixed
  60-tick confirmation cooldown expired and a redundant second salvo consumed the final
  2 AGM-65 — and the AIM-9 salvo killed the target in that same engine update, leaving the
  B-2 with no weapons for the final known target. The distances and tick indices here are
  inferred from the run's artifacts; **the merged fix's own real-BLADE proof is a separate,
  controlled construction and neither of its two arms is a rerun of this episode.**
- Code anchors: `GraphPlanExecutor.kill_confirm_ticks`,
  `GraphPlanExecutor._command_for_ego`, `Game.handle_aircraft_attack`,
  `weaponEngagement.launch_weapon`.
- **The decided direction — now IMPLEMENTED and MERGED:** not raising the constant blindly,
  but DERIVING a conservative confirmation wait from the ACTUAL auto-selected live weapon
  and the CURRENT engagement distance, with the configured `kill_confirm_ticks` kept as its
  FLOOR and FALLBACK. Current lethality, the two-argument attack command and FROZEN BLADE
  behaviour are preserved, and the probabilistic-miss / weapons-exhaustion redesign stayed
  OUT of scope. `CLAUDE.md` §5 (Execution, Stage 1) owns the contract; §7 owns the lock.
- **What the merged real-BLADE proof measured**, both engagements inside the single
  `DETECTION_KM = 50` envelope and at the production default `kill_confirm_ticks = 60`:
  at **~47.2 km** — the distance reconstructed from this probe's artifacts — the
  conservative bound is 62 and the derived wait 63, so the flat 60 was ALREADY below the
  bound, and the control arm escaped a redundant salvo by exactly ONE tick (real
  confirmation on call 60). At **~49.0 km** — a CONTROL ARM in the same envelope, far
  enough out that the one-tick escape is gone (bound 64, derived wait 65, real confirmation
  on call 62 against a re-fire on call 61) — the flat-60 arm DOES exhibit the premature
  re-fire and loses the reserve, while the derived wait fires exactly once and keeps it.
  The 49.0 km arm demonstrates the SAME MECHANISM inside the same envelope; it is **not**
  a rerun of the probe world above. **Neither arm is a scientific probe result.**

**Defect C — RTB ISSUANCE is not physical RTB COMPLETION.**
**STATUS: CLOSED / APPROVED / MERGED through PR #21** — approved candidate
`ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b`, integrated by
`0de9f21eb9e8904f06f836f4ecd010bc46c788b6`, identical tree
`6d05cc5ea9af0f6bdcd4a2d6865767bcbe525ebe` and a zero-file candidate→integration
comparison. The observations below are preserved as HISTORICAL EVIDENCE about the probe's
own SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`; they describe behaviour that no longer
exists on current `main`.

- At this SHA `GraphPlanExecutor.is_done()` treated the `rtb_issued` lifecycle latch as
  RTB-resolved, and `run_episode` stopped when `executor.is_done()` became true — so an
  episode could end immediately after an RTB command while the aircraft was still
  airborne.
- In the `post_update` damaged eval seed `1000000`, the damaged KC-135 completed its work
  and an RTB command was eventually issued while it no longer had enough fuel to physically
  reach home; the episode nevertheless ended before the resulting fuel exhaustion / death
  could occur, recorded `dead=0`, and contributed reward 0.
- **That observation is INFERRED FROM THIS RUN'S ARTIFACTS. It is NOT one of the merged
  fix's proof tests**, which are a separate, controlled real-BLADE construction and are
  not a rerun of this episode.
- **The decided direction — now IMPLEMENTED and MERGED:** separate **"RTB command
  issued"** from **"RTB physically resolved"**. `is_done(observation)` now requires the
  LIVE post-step observation; assignment completion still comes from executor semantic
  state, the physical half from `_physical_state` (airborne / landed / removed), and
  `_note_dead` reconciles a newly observed death into `executor.dead` for every ego before
  the verdict, so a burn-out on the ride home reaches `EpisodeResult.n_dead` and the
  UNCHANGED reward formula charges it. The single-issue RTB toggle protection is
  preserved, an ego that has committed to return leaves Phase 1 while peers continue, and
  BLADE stays FROZEN. `CLAUDE.md` §4 and §5 own the contract; §6 the routing; §7 the lock.
- **What the merged proof measured**, against the real engine: an ego ordered home from
  120 km out with fuel to spare keeps ticking past the order and the episode ends only
  once BLADE has put it back in an airbase inventory (`dead=0`); given HALF the fuel the
  engine itself says the trip needs, the same construction removes it mid-return, counts
  it dead and the unchanged reward path charges the airframe. The death is pinned by a
  DIRECT CAUSAL WITNESS on `Game.remove_aircraft`: exactly ONE recorded removal of that
  ego, at `current_fuel <= 0`, with no replacement airframe in any inventory. **Neither
  arm is a scientific probe result.**

**Scientific interpretation.**

- The first probe is USEFUL and did its job: it successfully exposed these defects.
- Its post-update reward improvement **must NOT** be treated as scientific evidence for the
  fuel-damage cell, because episode termination (Defect C) and abort semantics (Defect A)
  can distort the measured airframe penalty — precisely the quantity FD-BASELINE-v1 exists
  to make real. **Closing all three defects does NOT rehabilitate this run:** it was
  executed at `238062d…`, before any of the corrections existed, so its numbers remain
  historical evidence about the OLD behaviour and are not evidence about current `main`.
- All three fixes were reviewed and merged, their documentation/lock duty closed, and the
  SAME bounded short-probe shape was then rerun ONCE from the corrected `main`. **That
  rerun is recorded in §3e**; closing the three defects never constituted a measurement by
  itself.
- **The gate language written in this section is SUPERSEDED.** At the time it was written
  the long baseline was blocked on that rerun, and the rerun was then read as releasing it.
  Neither statement survives: the rerun's verdict is SUPERSEDED (§3e), a first long baseline
  was subsequently RUN and is `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED` (§3f), and the
  authorized rerun that followed the PR #24 correction is the one that PASSED the validity
  gate (§3h). Nothing in this section is outstanding work.

## 3e. The corrected-cell short-probe RERUN — EXECUTED / REVIEWED / VERDICT SUPERSEDED

> **READ THIS FIRST.** This section's ORIGINAL verdict was `VALID MEASUREMENT / CORRECTED
> SHORT-PROBE PASS`. **It is SUPERSEDED** by `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY
> REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR` (§3f, §3g; `CLAUDE.md` §7). Every FACT
> below — identity, invocation, provenance, accounting, artifact completeness, playback
> witnesses — is PRESERVED and unchanged, and the operational witnessing of Defects A, B and
> C stands. What no longer holds is the SCIENTIFIC reading: **the reward numbers, the
> per-condition means, the paired deltas, the death counts and the PPO-productivity figures
> below are raw historical outputs and NOT scientific evidence about the fuel-damage cell**,
> and this run did **not** pass or release the long-baseline validity gate.

Run identifier `training_output_20260816_162130`, executed from a clean checkout at exact
code SHA `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` (`2026-08-16T15:26:55+03:00`), in the
merged preset's shape. `CLAUDE.md` §7 owns the authoritative measurement record, including
the six evidence SHA-256 hashes; §8 owns the gate. Summarized here far enough to hand over.

**Exact provenance and configuration.** ONE invocation, native exit code **0**:
`conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json`.
Tracked checkout clean before and after; **no code, configuration, preset or
research-semantic change accompanied the run**. Preset blob
`3c85e5bdc780600fe1ee528b3e35fc71591fe4b7`, `config_source.resolved_from = config_file`,
`cli_overrides = []` — the reviewed preset ran with NO typed override, so there is no
deviation to report. Provenance complete: `git.available=true`, exact SHA, `branch=main`,
`dirty=false`, Windows / `nlp_env`, vendored BLADE, BONMIN available and probed `ok`.
`difficulty.factor = fuel_damage_baseline_v1`, `aircraft_penalty_coeff = 2.25`,
`reward.formula_changed = false`. **Elapsed time is TWO DISTINCT QUANTITIES, never
merged:** the harness's own `run_summary.json:run_seconds = 204.50847799982876`, and the
externally measured invocation wall clock of **223.117 s** (the preserved probe runner's
`timing.txt`: `PROBE_START_UTC = 2026-08-16T13:21:24.1839202Z` → `PROBE_END_UTC =
2026-08-16T13:25:07.3008671Z`). The harness figure excludes process start-up, `conda run`
dispatch, imports and teardown, so it is NOT the wall clock.

**Verdict AS ORIGINALLY REVIEWED: `VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS`** —
judged by the §4 validity gate and not by whether reward improved: complete clean
provenance; `accounting_reconciled = true`; no INFRASTRUCTURE failure; **4/4 complete
matched pairs in BOTH rounds**; complete PPO-update and artifact evidence. **THAT VERDICT IS
SUPERSEDED AND NO LONGER HOLDS** — see the box at the top of this section and §3f. The
reason is the `EpisodeRosterError` in the failure list below: at review time the documents
described it as an accounted `setup` failure, so it presented as ordinary episode attrition;
PR #24 establishes it was a MEASUREMENT/DATA-INTEGRITY fault that must ABORT the run, which
means one of the 24 scheduled attempts left this population through an instrument defect and
the denominator cannot be read as sound. The earlier review was not wrong about what it
inspected — the ROUTING it judged against was.

**Accounting — `skip_and_account_v1`, every denominator explicit.** **24 scheduled
attempts, 22 successful, 2 failed, both at `setup`.** Clean **11 attempted / 10 successful
/ 1 failed**; damaged **13 attempted / 12 successful / 1 failed**. Training 8 / 6 / 2 with
all six successes producing wakes; evaluation **16/16 successful**. No retry, no
substitution, no seed-band shift. The two failures:

- train seed 2, **damaged**, `setup`, `RuntimeError` — B2 produced only two placements for
  `n_hidden=3` (the static solve left one ego without a non-empty route);
- train seed 4, **clean**, `setup`, `EpisodeRosterError` — one t=0 known target was absent
  from the executed world.

**Fuel-damage event yield — mind the denominator.** The event fired and woke its selected
ego in **12/12 successfully completed damaged episodes**, which is **12/13 SCHEDULED
damaged attempts**, because the damaged seed-2 attempt failed at `setup` before an event
could exist. It must never be stated as 12/12 scheduled damaged attempts.

**Matched held-out evaluation** — same fixed band, each seed run `forced_clean` and
`forced_damaged`, 8 attempted / 8 successful per round:

| Round | Pairs | Clean mean | Damaged mean | Paired delta | Eval deaths | Unique targets |
|---|---|---|---|---|---|---|
| `pre_update` (`updates_completed=0`) | 4/4 | `-0.4999997395829586` /4 | `-0.8749999192707323` /4 | `-0.37500017968777366` over 4/4 | 4 | 3.00 |
| `post_update` (`updates_completed=2`) | 4/4 | `-0.12499955989518505` /4 | `-0.4583330529509838` /4 | `-0.33333349305579874` over 4/4 | 0 | 4.25 |

Held-out OVERALL mean moved `-0.6874998294268455` → `-0.2916663064230844`, **each over 8/8
completed eval episodes**. Post-update damaged **real RTB command yield 4/4**. Training
produced **26 transitions** across 6 successful episodes, **two productive iterations** and
`updates_completed = 2`. The deterministic post-update eval meta-action mix was
`PLAN_COMPLIANCE 0 / OPPORTUNISTIC_ENGAGEMENT 18 / SELF_PRESERVATION_ABORT 13` (the
`pre_update` round was `52 / 0 / 0`). **All of these are SHORT-PROBE OBSERVATIONS, not
estimates of converged policy performance**, and the two per-condition means are each over
their own successful subset — the only within-seed claim is the paired delta.

**Artifact completeness**, reported alongside the scientific denominators and never in
place of one: **24 bundles and 24 manifests; 22 `complete`** (matching the 22 successful
attempts) **and 2 `incomplete`** (matching the two `setup` failures). Every complete bundle
holds its known-only scenario, executed t=0 scenario and BLADE playback; executed-world
cardinality reconciles at **3 known / 3 hidden / 6 total** on every one; **neither
incomplete bundle fabricated a playback**. The complete scientific artifact is preserved.

**Playback witnesses and corroborating run evidence.** All of the following concerns the
attempt preserved as
`visual_artifacts/post_update_r001_e003_m1_seed1000003_damaged_tag901007`, and **two
evidence sources are involved which are deliberately not merged.** The BLADE **playback
JSON** proves PHYSICAL state only — position, fuel, `rtb`, route, weapon inventory, airbase
membership — is sampled every ten ticks (offsets below are from the recording's first
frame), and records **no per-wake meta-action label**; the jsonl records persist per-round
and per-iteration meta-action AGGREGATES only. The **preserved console transcript** of the
per-episode `OK` blocks (`probe_console.log`, SHA-256
`97bf45d56a3b224ef0ebe5a362bb7415b73e88520d192c891e347cb2412f31c4`, re-verified read-only
before citation) is the only artifact recording a SELECTED ACTION LABEL, and it does so for
the fuel-damage wake specifically.

**Defect A — KC-135R Stratotanker #76** (ego `0a14f756-13f2-4c78-8aa8-446da245aee5`, the id
the playback binds to that name). *Console transcript, this exact attempt:* `[eval
stage=post_update ep=3 damaged seed=1000003] OK` records `fired=True tick=269
progress=0.300`, `fuel_before=203494.4 fuel_after=70026.7 factor=0.3441` and `fd_wake=True
fd_meta=SELF_PRESERVATION_ABORT rtb_command=True` — **the action label is an attribution
from that record, not something visible in playback.** *Playback, independently:* fuel falls
from `203578.18` at the T+260 sample to `70017.43` at T+270 (sampled values, hence not
identical to the console's exact event-time pair), `rtb` flips `False → True` on that frame,
the route is replaced by a route to base at once, the aircraft lands ≈ T+540, and it never
resumes the abandoned assignment queue. The two sources agree, and the physical signature is
**consistent with the merged EGO-GLOBAL abort semantics**.

**Defect B — B-2 Spirit #698** (playback evidence alone; no action label claimed). It fires
**one salvo per target** — Floridistan AFB #1794 ≈ T+2320 (AIM-120 `4 → 2`) and Hidden
Airbase #003 ≈ T+5140 (AIM-120 `2 → 0`) — each BLADE two-argument attack launching two
physical AIM-120 missiles, with no redundant second salvo and no flat-timeout re-fire loop.
Salvo quantity, lethality and ammunition management are unchanged.

**Defect C — the same B-2** (playback, corroborated by the transcript's terminal line). It
enters RTB ≈ T+5240, the episode keeps ticking while it physically flies home, it lands
≈ T+7705, and only then does the episode finish; the console block for the same attempt
independently reports `ended=done ticks=7705 dead=0`. That is physical completion, not
command issuance.

**Deferred research hypothesis — NOT a defect and NOT a proven action attribution.** Hidden
Airbase #001 lies close to Hidden Airbase #003; the sampled playback puts the B-2 at
**50.07 km** from Hidden #001 at T+5230 while not yet in RTB, against a 50 km detection
threshold, with RTB visible on the next sampled frame at T+5240. Playback is sampled every
ten ticks and **nothing preserved records an ORGANIC wake's selected meta-action** — the
jsonl records carry aggregates only, and the console transcript labels the fuel-damage wake
alone, which is a different ego in a different episode phase — so it is **plausible but not
proven** that the B-2 crossed the threshold between samples and selected
`SELF_PRESERVATION_ABORT`. Treat possible over-conservatism as a future research hypothesis
about policy calibration, relevant to a later variable-FD-severity experiment. **It opens no
defect, changes no reward, retunes no policy, and it is NOT what superseded this probe's
verdict** — the roster/data-integrity fault is (§3f). It neither blocks nor shapes the fresh
long baseline.

## 3f. The FIRST executed LONG BASELINE — INCONCLUSIVE: ROSTER/DATA INTEGRITY FAILED

Run identifier `training_output_long_baseline_100x8_seed0`, executed at exact code SHA
`c30b6982ba605d60976cc303256da4b5528b0e63` (`2026-08-16T21:47:25+03:00`, the PR #23 merge),
recorded Git branch `task/long-baseline-execution`, `dirty=false`, `dirty_path_count=0`.
`CLAUDE.md` §7 owns the authoritative measurement record and the evidence hashes; §8 owns
the gate. Summarized here far enough to hand over. **It changed no tracked file** — a run of
merged code, not a candidate.

**Exact provenance and configuration.** ONE invocation, native exit code **0**:

```text
PYTHONPATH=src conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config training_output_long_baseline_100x8_seed0/long_baseline_contract.json
```

`config_source.resolved_from = config_file`, `cli_overrides = []` — **no typed override, no
ad-hoc knob**; `difficulty.factor = fuel_damage_baseline_v1`,
`aircraft_penalty_coeff = 2.25`, `reward.formula_changed = false`; Windows / `nlp_env`
(CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`. **Elapsed time is TWO
DISTINCT QUANTITIES, never merged:** the harness's own
`run_summary.json:run_seconds = 7764.3988857`, and the externally measured invocation wall
clock of **7778.704310178757 s** (`timing.txt`:
`PROBE_START_UTC = 2026-08-16T19:18:32.509409Z` →
`PROBE_END_UTC = 2026-08-16T21:28:11.191698Z`). The harness figure excludes process
start-up, `conda run` dispatch, imports and teardown, so it is NOT the wall clock.

**The scientific contract** (from the preserved `long_baseline_contract.json`, a MEASUREMENT
contract and deliberately not a repository preset — **this is the contract the rerun in §4
repeats unchanged**): 100 scheduled training iterations × 8 training attempts, train seeds
`[0, 800)`; evaluation every 5 iterations over 8 FIXED held-out seeds `[1000000, 1000008)`,
each seed evaluated `forced_clean` AND `forced_damaged`; **21 evaluation rounds** including
the initial `pre_update`; 3 agents / 3 known / 3 hidden; FD-BASELINE-v1 unchanged; visual
artifacts enabled for every scheduled attempt; 10 checkpoints.

**Verdict: engineering `REQUEST FIXES`; scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY
FAILED`.**

**Mechanical accounting — historical fact, and NOT validity.** **1,136 scheduled attempts,
860 successful, 276 failed.** Training 800 / 566 / 234; evaluation 336 / 294 / 42;
`accounting_reconciled = true`; **100 productive iterations and 100 PPO updates.** Every
count reconciles — **and that is the point**: a run can be perfectly self-consistent about a
population an instrument defect silently shrank, so these numbers must never be offered as
evidence that the measurement was sound.

**Failures by error type** (all 276 booked at pipeline stage `setup`; by condition, clean 123
/ damaged 153):

| Error | Count | Where |
|---|---:|---|
| `EpisodeRosterError` | 143 | ALL training, over **83 distinct iterations** — 75 clean, 68 damaged |
| B2 exact-cardinality `RuntimeError` | 101 | 59 train, 42 eval |
| `FuelDamageError` | 32 | every one a DAMAGED training attempt |

The 143 roster failures have two shapes: **126 PRE-run**, claiming a t=0 known target was
absent from the executed world (125 naming one target, 1 naming two), and **17 POST-run**,
raised after a real episode and a real playback because a CONFIRMED target id fell outside
the incorrectly shortened roster.

**What the independent artifact review established.** Every one of the 126 pre-run roster
failures had a FULL SIX-TARGET authoritative `executed_t0_scenario.json`; the 17 post-run
failures left real playback files their `incomplete` manifests did not list; and **11
`complete` manifests reported observed `3 known / 2 hidden / 5 total` while their own
authoritative executed-t0 scenarios held `3 + 3 = 6`**. **Root cause: allocated-only solver
output was being read as world inventory** — closed by §3g.

**What must NOT be reported from this run.** Its reward improvement, per-condition means,
paired deltas, survival, fuel-damage yield and PPO performance are **NOT scientific
evidence.** They remain in the preserved records and may be referred to only as raw
historical outputs of an inconclusive run; they are deliberately not tabulated here so they
cannot be lifted out of context as a baseline.

**What is NOT a defect here.** The **101 B2 exact-cardinality and 32 fuel-window failures
are NOT corrected by PR #24** and are not faults. They are EXPECTED SCIENTIFIC OUTCOMES
under the current contract (`skip_and_account_v1`: attempt once, record once, report the
smaller successful population next to its denominator) and **must not be relaxed, retried,
retuned or reclassified.** Only the ROSTER fault changed category, because only it was an
instrument defect.

**Preservation.** The run directory is preserved and must not be modified, moved, copied,
repackaged, deleted or regenerated. `CLAUDE.md` §7 records the eight verified artifact
SHA-256 hashes plus the two review-package hashes and the review ZIP's own hash.

## 3g. What PR #24 closed — roster / world-truth integrity

Authoritative contract: `CLAUDE.md` §5 (the Stage-0 "WORLD INVENTORY IS NOT ORACLE
ALLOCATION" block and the "Roster / world-truth integrity" block); routing in §6; lock and
verification in §7. Summarized here only far enough to hand over.

**The defect — a FOURTH, SEPARATE one.** It is **not** a regression in Defects A, B or C;
their corrections remain merged and operationally witnessed. `_episode_target_roster`
answered "which targets does this episode contain?" from `ctx.beliefs` (known) and
`ctx.oracle_tasks` (executed). Both are **ALLOCATIONS**: `solve_and_normalize` returns an
allocated-only task list by contract, so every target the solver did not select was absent
from them while still sitting in the world the executor flew through, sensed, attacked and
confirmed. The roster under-counted its own world, then FAILED the episode for the
discrepancy it had itself introduced — **as an accounted `setup` failure**, which is how the
long baseline lost 143 training attempts to a measurement defect while reporting itself
healthy and reconciled.

**The merged correction.** `solve_and_normalize()` stays allocated-only and the oracle
denominator is untouched — what changed is where "the world" comes from:
`EpisodeContext.known_target_ids` and `executed_target_ids`, two RAW snapshots taken BEFORE
their solves, verified non-empty with known ⊆ executed, and required (not defaulted) at
`_finish_context`. Beliefs became a SUBSET constraint on the known world instead of its
denominator; hidden ids are executed minus known in executed-world order;
`_require_scheduled_cell` checks the approved 3 / 3 / 6 cell before fuel planning and before
execution; and roster/world-integrity faults are `MeasurementIntegrityError` (with
`EpisodeRosterError` as its subclass) which **ABORT** the run as infrastructure /
data integrity — never entering `EpisodeAttemptError`, `episode_failures.jsonl`,
`skip_and_account_v1`, a condition tally or a scientific denominator. **That deliberately
reverses PR #7's routing**, because a data-integrity fault is a property of the INSTRUMENT,
not of the episode. After `run_episode` the order is contractual: synchronize the playback
(`_AttemptArtifacts.sync_recordings`, discovery only — nothing created or fabricated),
validate the confirmed ids against the executed-world snapshot, and only then compute a
reward; `finalize` refuses to mark a bundle `complete` when expected and observed world
counts disagree, writing the observed counts and leaving the status `incomplete`.

**What did NOT change.** Reward, PPO, the oracle allocation, fuel-damage semantics, B2
placement, the seed formulas, the evaluation schedule, the tick loop, the executor, the
generator and vendored BLADE.

**Verification at the approved head** (`CLAUDE.md` §7 has the full statement): focused
base-environment suite 207 passed / 4 skipped; full suite 272 passed / 4 skipped; standalone
`nlp_env` `tests/test_graph_train.py` 119 passed, `tests/test_graph_fuel_damage.py` 41
passed, `tests/test_graph_setup_seam.py` 39 passed / 0 skipped including the real-BLADE and
BONMIN tiers; `git diff --check` clean. **No training run, probe, rollout, seed sweep or
baseline rerun occurred during the correction.**

## 3h. The Phase-A LONG BASELINE (RERUN) — EXECUTED / REVIEWED / `APPROVE — VALID MEASUREMENT`

**This is the FIRST scientifically valid measurement of the fuel-damage cell, and it CLOSES
PHASE A.** `CLAUDE.md` §7 owns the authoritative record, the full denominators and the
evidence hashes; this section is the volatile summary.

Run directory `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, measured
code SHA `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` (clean checkout, complete provenance,
`dirty=false`). ONE invocation, native exit code **0**, `config_source.resolved_from =
config_file`, **`cli_overrides = []`**. Elapsed time is two distinct quantities and is never
merged: harness `run_seconds = 8493.042731400012`, external wall clock
**8509.632915974 s**.

**Contract fidelity.** The preserved `long_baseline_contract.json` of the invalid first long
baseline is the authority. The rerun contract was cloned from it with **exactly ONE field
changed — `output_dir`**: 27 keys, identical key sets AND order, one differing key, every
other value identical. Train seeds `[0, 800)`, held-out seeds `[1000000, 1000008)`, the
100 × 8 schedule, `eval_every = 5`, the matched-pair design, the PPO settings, the
3-agent / 3-known / 3-hidden cell and every FD-BASELINE-v1 parameter are UNCHANGED.

**Validity gate — judged BEFORE performance, and PASSED on all five clauses.** Complete Git
provenance on a clean checkout; `accounting_reconciled = true`; **ZERO** infrastructure or
data-integrity faults (no `EpisodeRosterError`, no `MeasurementIntegrityError`, no
`_VisualArtifactError`, and no crash outside the episode taxonomy); at least one COMPLETE
matched pair in `pre_update`; at least one in the final `post_update`.

**Accounting.** 1,136 scheduled attempts, **993 successful**, **143 accounted scientific
episode failures**. Train 800 = 699 + 101; eval 336 = 294 + 42 over 21 rounds. Artifact
bundles: **993 `complete`**, **143 `incomplete`**, and **all 993 complete bundles reconcile
expected against observed 3 known / 3 hidden / 6 executed targets**.

**Failures — both families are EXPECTED, not defects.** Exactly **101 B2 exact-cardinality
`RuntimeError`** and exactly **42 `FuelDamageError`**, all at `setup`. Independent review
established the exact set relations: the 101 B2 failures are **the same scheduled attempts
as in the invalid old run**; `EpisodeRosterError` went **143 → 0**; and the **10 additional
`FuelDamageError` attempts are exactly attempts that were `EpisodeRosterError` in the old
run**. Held-out seed **`1000005`** fails B2 for BOTH members in ALL 21 rounds, so
matched-pair yield is a STRUCTURAL **7/8 every round**.

**Learning.** **100/100 productive PPO updates**, 2,566 transitions. Matched paired reward
delta **−0.375000 over 7/8 pairs** (`pre_update`) → **−0.071429 over 7/8 pairs** (final
`post_update`). Evaluation deaths **7 → 0**. Final clean reward **0 on all 7** feasible
held-out worlds; final damaged reward **0 on 5 of 7**, with the residual concentrated in
seed **1000004** (`−0.333333`, 4/6 targets, 0 deaths) and seed **1000007** (`−0.166667`,
5/6 targets, 0 deaths) — both **preserved the aircraft** at the cost of target coverage. In
**all 7** completed damaged held-out worlds the deterministic fuel-damage decision moved
from `PLAN_COMPLIANCE` before training to `SELF_PRESERVATION_ABORT` after training, and
selected playback witnesses independently confirm a real PHYSICAL behavioural change
including survival and RTB under the final policy.

**Fuel-exposure caveat, against the right denominator.** Successful damaged TRAINING
episodes 324; damage events actually fired **323**. The one non-firing successful damaged
training episode is **seed 424** (iteration 53): its selected ego returned before reaching
the 0.30 leg-progress trigger. **No defect is inferred** — a plan existed and every LIVE
quantity is `n/a`, so the artifacts record that it did not fire, not why. **Evaluation
exposure is complete: 147 / 147.**

**Independent review.** Read-only package `long_baseline_rerun_737b4bf_gpt_review.zip`
(SHA-256 `f2582c0ca7f460a5f51bd515aeb0506f0476e8e06e4039312b7371858a08b932`) carried both
runs' core evidence, all 1,136 original manifests, selected raw playback bundles and derived
audits. Verdict: **`APPROVE — VALID MEASUREMENT`**.

**Preservation.** This run directory and the invalid
`training_output_long_baseline_100x8_seed0` are both preserved and must not be modified,
moved, repackaged, deleted or regenerated. The old run stays explicitly INCONCLUSIVE, and
the superseded short-probe verdicts stay historical — none of them becomes evidence.

**The Phase-A conclusion, and its explicit NON-CLAIMS.** The result establishes **end-to-end
learnability and meaningful ego-local runtime adaptation in the LOCKED Phase-A reference
cell**. It does **NOT** establish global optimality, **NOT** monotonic convergence, **NOT**
generalization beyond this fixed cell and this held-out seed set, and **NOT** any benefit
from centralized training. Those are subsequent research questions — the first of which is
§4.

## 3i. What PR #27 closed — FD-VARIABLE-SEVERITY-v1 (CODE ONLY)

**CODE CLOSED / APPROVED / MERGED. THIS SECTION IS THE CODE LOCK ONLY — the measurement
came later and separately, and §3j owns it.** Approved candidate
`eecc9b5d91bce4a98a070a29307cc12af0d4c4a3`, integrated
`177e969446ef6c01c729484f2ea9969c94a27330` (PR #27), identical tree
`37ebd8c56266fdd862cc7244c5f22a6ac95e438c`, zero changed files candidate→integration.
Grade A under `GPT_GITHUB`. `CLAUDE.md` §5 owns the full contract, §6 the routing and §7
the lock; this section is the volatile summary only.

**WHAT IT IS.** An ADDITIONAL actor-only difficulty design layered on FD-BASELINE-v1. Under
the legacy factor every damaged episode is structurally SEVERE, so "damaged" and
"continuing is infeasible" are the same fact and an actor can learn `fuel damage ⇒ abort`
without reading its own fuel gauge. The extension splits the damaged half into a MILD case
where continuing remains genuinely feasible and a SEVERE case where it does not.

**WHAT IT PRESERVES — and this is the load-bearing half.** The LEGACY modes (`off`,
`seeded_mixture`, `forced_clean`, `forced_damaged`) are UNCHANGED: same seeds, same
conditions, same selected egos, same planned-midpoint target. Severity is drawn from its
OWN deterministic domain `fuel_damage_severity_v1`, kept separate from the legacy
`fuel_damage_v1` domain precisely so it cannot shift the condition/ego stream — drawing it
in the v1 stream would change WHICH EGO every damaged episode selects and would invalidate
the approved Phase-A measurement instead of extending it.

**THE TWO LIVE BANDS**, measured at the event tick against the live window and live fuel
(`F_rtb` = RTB floor, `F_cont` = continue requirement, both carrying the 1.10 reserve):

- **MILD** — `F_rtb < F_cont < F_after < F_before`: a real loss, safe RTB feasible,
  continuation and eventual return STILL feasible.
- **SEVERE** — `F_rtb ≤ F_after < F_cont ≤ F_before`: a real loss, safe RTB feasible,
  continuation infeasible. This is exactly the legacy interval.

The post-damage value is the band's MIDPOINT derived from the LIVE state. An invalid band
raises BEFORE the mutation; **nothing is clamped, retried, downgraded to the other
severity, given a replacement ego, or converted to a clean episode.**

**TRAINING DISTRIBUTION.** `fuel_damage_mode = seeded_variable` with
`P(damaged) = 0.50` and `P(mild | damaged) = 0.50` ⇒ the approved flat
**0.50 clean / 0.25 mild / 0.25 severe**.

**EVALUATION.** A matched **CLEAN / MILD / SEVERE TRIAD** per held-out seed — same seed,
same generated world, same hidden geometry, the same deterministically selected ego for
both damaged members, distinct artifact tags. **A complete triad requires all three
members to succeed**, and the three within-seed deltas (`mild − clean`, `severe − clean`,
`severe − mild`) are over COMPLETE triads only. The **PRIMARY behavioural evidence is the
severity-conditioned FD-wake meta-action response**, reported per cell with its own FD-wake
denominators — **not** reward, and **"mild must choose `PLAN_COMPLIANCE`" is NOT a
correctness rule**; opportunistic engagement under a survivable loss can be rational.
Successful attempts get one durable record each in **`episode_outcomes.jsonl`** (failures
stay in `episode_failures.jsonl`; the two streams are disjoint), and the summary's
severity-response table is DERIVED from that file rather than from a parallel in-memory
aggregate.

**NO-COMMS AND SCOPE ARE UNCHANGED.** The actor receives NO severity label — no severity
feature reaches `GraphObservation`, and only its own real `fuel_norm` changes. Target
destruction stays deterministic at `probability = 1`; weapon lethality, `graph_reward`, the
frozen solver, PPO and BLADE are untouched. **`p(destroy) < 1` was NOT implemented here**
and remains a separate future Grade-A research task.

**THE REVIEW FIX (Grade-A fix chain, append-only).** The first candidate
`73752d872a8cd17f703790ef41bee46a734170bb` received **REQUEST FIXES**: `_ConditionTally.
success` checked only that the EXECUTED cell was a LEGAL cell, so a scheduled MILD attempt
that executed as SEVERE passed the membership test and booked the ATTEMPT in one cell's
denominator and the REWARD in another — corrupting BOTH scientific denominators at once and
letting a triad delta be taken between members the schedule never paired. The approved child
`eecc9b5…` makes the scheduled cell a REQUIRED keyword and requires **scheduled == executed
EQUALITY before any successful accounting**; both production call sites pass their scheduled
cell and the guard runs FIRST, so a mismatched episode reaches neither the tally counters and
rewards, nor a matched-group member reward or delta, nor `episode_outcomes.jsonl`, nor the
PPO buffer. A mismatch is a `MeasurementIntegrityError` INFRASTRUCTURE abort — never an
accounted scientific episode failure. **It has NOT been observed in the real simulator**: the
regression test injects the divergence through a stub, because normal production does not
currently generate it.

**REVIEWED EVIDENCE at the approved head.** `tests/test_graph_fuel_damage.py` **60 passed**;
`tests/test_graph_train.py` **119 passed**; both standalone `nlp_env` runners **60 / 119
passed**; full suite **291 passed, 4 skipped**; `git diff --check` clean.
`graph_train --selftest` — TEST 1 passed, TEST 2 passed, **TEST 3 failed IDENTICALLY TO THE
BASE** on the already-known B2 seed-2 exact-cardinality case: a pre-existing expected outcome
of the current contract, **not a PR #27 regression**.

**NO scientific baseline, long training run, probe, rollout or artifact-generating smoke was
executed for PR #27** — that remains true of the CODE lock. Nothing in THIS section is
evidence about the variable-severity cell; **the baseline was measured afterwards and is
recorded in §3j** (`CLAUDE.md` §7 owns the authoritative record).

## 3j. The FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE — EXECUTED / REVIEWED / `APPROVE — VALID MEASUREMENT`, WITH A NEGATIVE PRIMARY FINDING

**VALID, AND THE HEADLINE RESULT IS NEGATIVE.** `CLAUDE.md` §7 owns the authoritative
record — full identity, provenance, every denominator, the physical outcomes, the reward
table, the artifact audit, the evidence hashes and the explicit non-claims. This section is
the volatile summary only and deliberately does not duplicate it.

**IDENTITY.** Run root `C:\Users\Itama\f7r2` (contract `c.json`, output `r`, console
`console.log`, timing `timing.json`), at exact measured code SHA
`bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
`dd881478b8e2e521054d09bc865437f1308be1a2`, executed from a clean DETACHED snapshot that
carried no task branch. ONE invocation through `--config`, native exit code **0**,
`config_source.resolved_from = config_file`, **`cli_overrides = []`**, complete provenance,
`dirty = false`. **It changed no tracked file** — a run of merged code, not a candidate.
Two elapsed quantities, never merged: harness `run_seconds = 5998.791282300022`, external
wall clock `6021.3954213 s`.

**THE EXCLUDED PRECURSOR — `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT`.** An earlier
attempt at the SAME contract
(`…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640\training_output_fd_variable_severity_v1_50x8_seed0`)
hit a Windows `MAX_PATH` playback-export failure — a 267-character recording path against
the 260-character limit — producing 70 `run`-stage `FileNotFoundError`s that were **ALL
`post_update` SEVERE members** (10 rounds × the 7 feasible held-out seeds). That
systematically removed the entire post-training severe arm, which is precisely the arm the
experiment exists to measure. **It is an infrastructure failure, NOT a negative scientific
result**, and it is excluded from every scientific reading. The replacement contract is the
precursor's own contract with **`output_dir` as the ONLY differing field** (25 keys,
identical key set and order). Both trees are PRESERVED.

**ACCOUNTING.** **664 scheduled / 586 successful / 78 failed**,
`accounting_reconciled = true`. Training 400 / 355 / 45; evaluation 264 / 231 / 33 across
11 rounds. Per cell — training `clean 202/190/12`, `mild 92/76/16`, `severe 106/89/17`;
evaluation `clean 88/77/11`, `mild 88/77/11`, `severe 88/77/11`. Every failure is at stage
`setup`: 58 B2 exact-cardinality `RuntimeError` and 20 `FuelDamageError`. All 33 evaluation
failures are held-out seed `1000005` (11 rounds × 3 members) — the structural B2 world that
caps triad yield, reported and never repaired. **ZERO `FileNotFoundError`, ZERO
`MeasurementIntegrityError` / `EpisodeRosterError`, ZERO `_VisualArtifactError`, and zero
crash outside the episode taxonomy.** Artifacts: 664 bundles / 664 manifests, 586
`complete` / 78 `incomplete`, 586 playbacks, all complete bundles reconciling 3 known /
3 hidden / 6 executed; maximum actual path 139 characters.

**VALIDITY, JUDGED BEFORE PERFORMANCE — PASSED on all four clauses** (§4): complete clean
provenance; `accounting_reconciled = true`; no infrastructure or data-integrity failure;
and **at least one COMPLETE matched group in BOTH the `pre_update` and the `post_update`
round — in fact 7/8 complete triads in every one of the 11 rounds.**

**THE PRIMARY BEHAVIOURAL RESULT — NEGATIVE.** Rates are over **FD WAKES**, never episodes.
At `pre_update` MILD and SEVERE each produced 7 wakes and chose `PLAN_COMPLIANCE 7/7`; at
the FINAL `post_update` (`updates_completed = 50`) MILD and SEVERE again each produced 7
wakes and chose `PLAN_COMPLIANCE 7/7`. Pooled across all ten `post_update` rounds the two
distributions are IDENTICAL: 70 wakes each, `PLAN_COMPLIANCE 63 = 0.900`,
`SELF_PRESERVATION_ABORT 7 = 0.100`, engage 0. **The deterministic held-out actor did not
differentiate a survivable MILD loss from an unsurvivable SEVERE one.** Training successes
(stochastic policy, context only): MILD 76 wakes `60 / 16`, SEVERE 89 wakes `66 / 23`.

**DENOMINATOR CAVEAT — the statistical unit is the final round's 7 complete triads.** The
70 `post_update` observations per severity REUSE those same seven feasible seeds across ten
checkpoints; they describe the learning trajectory and are **NOT 70 independent held-out
worlds.**

**THE SEVERITY FACTOR IS PHYSICALLY REAL — the null is behavioural, not physical.** Over
successful `post_update` evaluation outcomes: CLEAN 0 RTB / 0 deaths / 6.000 mean unique
coverage; MILD 70 RTB / 0 deaths / 5.957; SEVERE 43 RTB / **63 deaths** / 5.700. At the
final round every one of the seven feasible SEVERE worlds loses an airframe, while all
seven clean and all seven mild worlds finish 6/6 with no losses.

**PPO WAS PRODUCTIVE.** 50/50 scheduled iterations productive, 0 zero-wake, 0 all-failed,
`updates_completed = 50`, 1,405 transitions, training reward `-0.51547582 → -0.07738026`.
So the correct reading is exact: **training worked and improved overall performance, and it
still did not produce the targeted severity differentiation.**

**HOW TO READ IT — and how NOT to.** This is a **VALID NEGATIVE SCIENTIFIC RESULT**. It
does **NOT** say the actor is broken, that training or PPO failed, that the actor ignores
fuel, that MILD "should" have chosen `PLAN_COMPLIANCE` (that was never a correctness rule),
that the finding generalizes beyond this fixed cell and held-out seed band, or that
centralized training would change it. **No CTDE benefit is measured or implied here**, and
the result is not grounds to re-run, re-tune or re-seed.

**ENGINEERING CAVEAT — history, not a change.** The precursor proved that a BLADE
playback-export failure currently surfaces as an ordinary `run`-stage `EpisodeAttemptError`
and so enters the SCIENTIFIC failure ledger (`graph_tick_loop.run_episode`'s
`ctx.game.export_recording()` → `graph_train._run_one_episode`'s
`raise EpisodeAttemptError("run", exc) from exc`), unlike an artifact fault routed through
`_VisualArtifactError`, which aborts as infrastructure. The valid run had ZERO such
failures. Rerouting it would be its own separately reviewed task; **no code was changed
here.**

## 3k. What PR #30 closed — PHASE-B CTDE (CODE ONLY; NO COMPARISON RUN)

Volatile summary. **`CLAUDE.md` §5 owns the technical contract, §6 the routing and §7 the
lock and its evidence** — this section does not duplicate them.

- **IDENTITY.** Approved candidate `a6f3aa9d62931994f416b2241fec4cfac3b018ec`
  (`2026-08-22 21:01:46 Asia/Jerusalem`), integrated by merge commit
  `8390d85c2072e9cbe984ce5f2731cef3a9b14985` (PR #30). Ordered parents
  `d437084c5fb1a22c21596a48c58e03f7e15a0115`, then `a6f3aa9…`; integration tree
  `9686c107b8864f00a7d4403d70faf42ab561d2fb`. **Grade A under `GPT_GITHUB`, mode BUILD**
  (the SURGICAL mode belongs to the separate documentation-lock task, not to the
  implementation). Append-only fix chain: initial reviewed candidate
  `d70d07f829a44e6f19100c338d4dde89f4f47bf6` → APPROVED `a6f3aa9…`, on one branch and one
  PR, with no amend, rebase, squash or force-push.
- **SIX FILES**, the complete `d437084…...8390d85…` comparison:
  `src/match_aou/rl/observation/central_graph_builder.py` (new),
  `src/match_aou/rl/training/graph_ppo.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_ctde.py` (new),
  `tests/test_graph_ppo.py`. **No documentation file was part of the code integration.** No
  vendored BLADE, solver, `graph_reward`, `graph_fuel_damage`, encoder, action-space,
  episode-setup, generator, scenario, config or preset file was touched, and the Phase-A
  cell, the seed schedules, the evaluation design and every preserved run artifact are
  unchanged.
- **THE SHAPE, IN FIVE LINES** (the contract is `CLAUDE.md` §5): TWO training modes chosen
  by `TrainConfig.training_mode` — `actor_only` (the DEFAULT and the preserved reference
  path, which constructs no critic, no central observation, no value loss and no CTDE
  advantage) and `ctde`; EXECUTION stays decentralized in BOTH modes, because the actor
  still reads only its own private `GraphObservation` and `evaluate` never builds a critic;
  the critic is a SEPARATE `GraphEncoder` instance plus a `ValueHead` on its own optimizer,
  with disjoint parameters, two separate backwards and detached advantages; the central
  graph is the LIVE world with no distinguished ego, and it is deliberately denied the
  oracle, the reward, the seed, the severity label, the known/hidden split and any future
  outcome; and credit is GAE over the episode's GLOBAL ordered decision sequence, captured
  one central state per decision immediately BEFORE that decision.
- **THE THREE REVIEW FIXES:** reject `value_coeff == 0` under `training_mode='ctde'`;
  persist the critic's four diagnostics on a CTDE training record (absent, not null, on the
  `actor_only` path); and correct inaccurate `ValueHead` initialization prose — **prose
  only, the initialization is unchanged.**
- **CC-REPORTED ENGINEERING EVIDENCE — IMPLEMENTATION VALIDATION, NOT SCIENTIFIC EVIDENCE,
  IN TWO SEPARATELY LABELLED PARTS.** *(i) TESTS, solver-free:* full suite 334 passed / 4
  skipped; `tests/test_graph_ctde.py` 43 passed; `tests/test_graph_train.py` 119 passed;
  `tests/test_graph_ppo.py` 18 passed; the standalone `nlp_env` CTDE runner 43 passed;
  `git diff --check` clean. *(ii) BOUNDED ENGINEERING SMOKES, which DID run against REAL
  BLADE and REAL BONMIN:* during the BUILD candidate's validation, TWO bounded `nlp_env`
  smokes ran BOTH training modes end-to-end — 2/2 episodes, one PPO update,
  `accounting_reconciled = true`, no `CRASH`/`Traceback` — writing only to the scratchpad,
  never into the repository. They proved the wiring executes and surfaced the
  `baseline`-vs-critic-value defect the contract now pins. The later append-only review-fix
  validation needed no new run of them.
- **NO SCIENTIFIC MEASUREMENT OCCURRED FOR PR #30** — no baseline, no probe, no scientific
  rollout, and **no actor-only vs CTDE comparison. NO CTDE benefit is established or may be
  pre-claimed** — not from the Phase-A result, not from the variable-severity baseline
  (whose NEGATIVE severity finding is **NOT** evidence that centralized training would
  change it), not from a passing test suite, and **not from the two engineering smokes,
  whose rewards and outcomes must never be promoted into scientific evidence** — a bounded
  smoke has no scientific contract, no seed schedule, no held-out band and no denominator.
  *(Scope note: this bullet is about PR #30. An OLD-CONTRACT CTDE measurement was executed
  LATER; it is OUT OF SCOPE for the generalized redesign and is not reviewed, compared or
  claimed anywhere in this document — §1.)*

## 3l. GENERALIZED-V1 — the ACTIVE phase: APPROVED DESIGN, **§3l.1–§3l.7 ALL IMPLEMENTED (STEPS 1–4, MERGED); STEP 5 IMPLEMENTED AND APPROVED BUT NOT YET MERGED (§3m)**

**READ THIS BANNER BEFORE ANY LINE BELOW IT.** This section is the APPROVED DESIGN. As of
this record:

- **§3l.1–§3l.7 are IMPLEMENTED, REVIEWED and MERGED, to the extent §3l.8 steps 1, 2, 3 and
  4 represent** — Task 1 (candidate `5b55ca34…`, integration `9b305e4e…`, PR #35), Task 2
  (final candidate `185d39f0…`, integration `ca0dc406…`, PR #36), Task 3 (candidate
  `24a8b1ee…`, integration `df3abf2f…`, PR #38, **APPROVE**) and Task 4 (final candidate
  `db790138…`, integration `b4daa8c1…`, PR #40, **APPROVE**). **Their technical contracts
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
  **§3l.8 step 5 — bounded runtime / solver validation before that scale is selected — is
  NOT started and is NOT authorized by this record.**
- **NO generalized scenario campaign, no generalized episode schedule, no frozen benchmark
  population and NO generalized measurement of any kind exists, is running, is scheduled or
  is authorized — and no generalized result may be pre-claimed.** Bounded engineering smokes
  taken during implementation validation are not measurements. **No final actor-only or CTDE
  generalized campaign is authorized, and no actor-only-vs-CTDE generalized result exists.**

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

### 3l.8 The bounded implementation sequence — **STEPS 1–4 COMPLETE AND MERGED; STEP 5 IMPLEMENTED AND APPROVED BUT NOT YET MERGED**

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
6. **documentation and `CLAUDE.md` locks for behaviour that already exists** — **PARTIALLY
   DONE and CONTINUING.**

**HOW STEP 6 REALLY WORKS, corrected by practice.** This section previously said step 6 was
deliberately LAST — one documentation pass after every implementation step. It is instead
written PER COMPLETED TASK: `CLAUDE.md` §4 / §5 / §6 / §7 records for steps 1, 2, 3 and 4
are written by their own documentation records, each after that step's behaviour was
implemented, reviewed and integrated. **The principle is unchanged and is what matters:** no
`CLAUDE.md` contract is ever written for a design, only for behaviour that already exists.
*(SUPERSEDED: this paragraph previously added "which is exactly why §3l.6–§3l.7 still have
none". They have one now — the Task-4 §5 contract, its §6 routing and its §7 lock.)* **A
FINAL documentation pass is still REQUIRED after any later implementation step lands**, and
no per-task record discharges it.

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

## 3m. GENERALIZED-V1 TASK 5 — the INTEGRATED STACK, the ENGINEERING VALIDATION, and the DISPATCHED R1 long run

**This section is VOLATILE STATE. The technical contracts it points at live in `CLAUDE.md`
§5 (the GENERALIZED-V1 TASK 5 block), routed in §6 and locked in §7.**

### 3m.1 The implementation stack — APPROVED AND INTEGRATED

Task 5's implementation is **TWO separately reviewed implementation candidates plus their
documentation lock, ALL THREE APPROVED and ALL THREE now INSIDE `main`.** Live `main` at this
record's base is `9b9e9b85a70c8a0019c72ada92ceec3401725795` (`2026-08-31 00:39:29 +0300`, the
PR-#44 documentation merge). *(SUPERSEDED: this subsection previously read "APPROVED, FROZEN,
NOT YET MERGED", described TWO candidates "still OUTSIDE `main`" with live `main` at
`09eab067…`, and named the doc-lock branch as the SOLE WRITABLE REPOSITORY TASK. All of that
was accurate at that checkpoint and is not now.)*

- **PR #42 — `task/generalized-v1-task5-summary-phase-fix`, head
  `312f58650b61a85eb72d0554d60715afee862a5c`** (`2026-08-29 21:42:32 +0300`), base
  `09eab067…`. The `train_by_*` summary-population correction. **APPROVED and MERGED** —
  integration `5dfcd8b632be8dca3c1730018bbf35337d07f077` (`2026-08-31 00:06:47 +0300`), a
  NORMAL merge commit whose SECOND PARENT is the reviewed candidate, integrated tree verified
  equal to the reviewed tree.
- **PR #43 — `task/generalized-v1-task5-success-quota-preflight`, FINAL approved head
  `4af6c5aa5dd28072692bfda63282964b55010aae`** (`2026-08-30 18:02:14 +0300`), PR base
  ORIGINALLY the PR-#42 branch and RETARGETED to `main` after PR #42 merged. The
  successful-episode training quota, the bounded attempt budget, the
  maximum-possible seed band, and the deterministic benchmark preflight. **APPROVED and
  MERGED** — integration `b3c2e01f130afe854b09384cd6e1e196de714795`
  (`2026-08-31 00:13:23 +0300`), same normal-merge and tree-equality properties. **The
  retarget changed the BASE, never the CANDIDATE**: the head remained `4af6c5aa…` and was
  EXACT-BASE RE-REVIEWED against `main` before merging, with a byte-identical effective
  reviewed delta.
  **HISTORICAL PROCESS EVIDENCE ONLY:** the original implementation candidate was
  `734f1e786593b6ffb94f1f8d7283b1f2fc79d257`; GPT requested ONE append-only review fix; the
  final candidate `4af6c5aa…` is its DIRECT CHILD. **No amend, rebase, squash, force-push or
  history rewrite occurred.**
- **PR #44 — THE DOCUMENTATION LOCK, `task/generalized-v1-task5-doc-lock`**, FINAL approved
  head `88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` (`2026-08-30 23:51:07 +0300`), the
  append-only DIRECT CHILD of `61eaa3fe1bdeb7aef3cfb7c10c4d8964caf2ed0e`; branched from exact
  `4af6c5aa…` and RETARGETED to `main` after PR #43 merged, then EXACT-BASE RE-REVIEWED with
  its head unchanged. Scope was exactly `CLAUDE.md` and `graph_rl_project_handoff.md`.
  **APPROVED and MERGED** — integration `9b9e9b85a70c8a0019c72ada92ceec3401725795`
  (`2026-08-31 00:39:29 +0300`), same normal-merge and tree-equality properties.

**THE STACK WAS `main` → PR #42 → PR #43 → docs PR, AND IT HAS BEEN FULLY INTEGRATED IN THAT
ORDER.** The performed sequence, with its exact merge SHAs, is §3m.5. **THIS
POST-INTEGRATION CLOSURE CANDIDATE IS NOW THE SOLE WRITABLE REPOSITORY TASK while its own
draft PR is open; on its integration no writable repository task remains, and no Task-5 PR or
candidate is open or active.**

### 3m.2 Writable ownership — one writable task, and the long run owns none of it

- **The external GENERALIZED-V1 long-run task is RUN-ONLY and owns NO repository writes.**
  It may execute and it may produce run artifacts outside the repository; it may not create,
  edit, push or merge any branch, and no run artifact, config, preset or benchmark manifest
  it produces is committed by it.
- **This post-integration closure task is the ONE current writable repository task, and only
  while its own draft PR is open; on its integration NO writable repository task remains.** No
  other branch may be edited concurrently. *(SUPERSEDED: this bullet previously read "This
  documentation task is the ONE current writable repository task … and PR #42 and PR #43 are
  frozen". Those two PRs are now MERGED, so there is nothing left to freeze — their retired
  branches are cleanup-eligible, not writable.)*
- Ownership of the work remains with the **GPT orchestrator**; every orchestrator resolves
  live branch and PR state from GitHub, never from this document.

### 3m.3 Engineering validation — Task 5A and Task 5B

**THE LABEL IS BINDING: ENGINEERING VALIDATION — NOT SCIENTIFIC MEASUREMENT.** Both were
independently reviewed **`APPROVE — VALID ENGINEERING VALIDATION`**.

**WHAT MAKES THEM NOT MEASUREMENTS IS THEIR DESIGNATED PURPOSE, NOT AN ABSENCE OF
MECHANICS — AND THE DISTINCTION MATTERS BECAUSE TASK 5B REALLY HAD THEM.** Task 5B carried
an explicit training seed band `[720000, 720072)`, an explicit benchmark candidate band,
the PRODUCTION held-out verification, a TRANSIENT frozen manifest, and 18 worlds / 54
members for its ONE evaluation round. **Those mechanics existed solely to validate system
behaviour, attrition and runtime, and were explicitly NOT designated as the scientific
comparator or as a policy-performance measurement.** Both runs were AUTHORIZED, EXECUTED
and REVIEWED as engineering validation, and the label rests on that designation.
**Therefore, bindingly: no reward or learning claim, no generalized-performance claim, no
actor-vs-CTDE claim, and NO promotion of Task 5B's transient manifest into R1.**
*(SUPERSEDED, and corrected here: this section previously said neither run had a scientific
contract, a seed schedule, a held-out band, a frozen comparator or a population
denominator. That is too broad and is factually wrong for Task 5B; the engineering-only
label is UNCHANGED.)* `CLAUDE.md` §7 carries the same record; the detail is not duplicated
here.

- **TASK 5A.** A bounded end-to-end generalized rehearsal against a **TRANSIENT
  one-world-per-cell** benchmark. Its load-bearing finding is the one that changed the
  design: a repeated **A2-LOW `pre_event_popup_risk`** failure exposed the need for
  **eligibility selection BEFORE the freeze** — which is what `graph_benchmark_preflight`
  now provides. **Solver runtime dominated execution.** **Repeated pre/post values measured
  on the SAME world are NOT independent runtime observations.** No reward and no policy
  behaviour from it is promoted anywhere.
- **TASK 5B**, at measured code SHA `4af6c5aa5dd28072692bfda63282964b55010aae`. The
  validated engineering facts, recorded only because execution planning depends on them:
  generalized training **24/24 successful, 0 ordinary failures**; benchmark preflight
  **18/18 first candidates accepted, 0 rejected**; **0 observed requested→realized hidden
  shortfalls** in those bounded samples; one **TRANSIENT** benchmark round **54/54 members
  successful, 18/18 complete matched groups**; the **real BONMIN reference solver dominated
  runtime**; **`A4-high` showed very large runtime variance**; and **one legitimate training
  solve of roughly 998 seconds terminated `optimal`**, which is exactly why **no short solver
  timeout was adopted**.
  **THE SAMPLE-SIZE LIMITATION IS BINDING.** These are bounded samples. **No attrition-rate
  population claim** may be made from them — "0 rejections in 18 first candidates" and "0
  shortfalls" describe those samples and are not an estimate of a full campaign's rate,
  which is precisely why `generalized_max_attempts_per_iteration` and
  `max_candidates_per_cell` are REQUIRED operator inputs with no defaults. **The Task-5B
  transient benchmark is NOT the R1 comparator.**

### 3m.4 The first GENERALIZED-V1 ACTOR-ONLY long run (R1) — AUTHORIZED / DISPATCHED — RESULT PENDING

**STATE: `AUTHORIZED / DISPATCHED — RESULT PENDING`.** This record does **NOT** state that
the run is currently executing, that it completed, that it is scientifically valid, or what
it produced. **No reward, convergence, attrition, benchmark, denominator or validity claim
about R1 exists or may be inferred from any document.** This documentation task holds no
independently checkable execution evidence for a live process, so `RUNNING` is deliberately
not claimed.

**THE FROZEN PLAN, recorded so the eventual artifacts can be checked against what was
authorized:**

| Knob | Value |
|---|---|
| `training_mode` | `actor_only` |
| iterations | 375 |
| SUCCESSFUL episodes per iteration | 8 |
| total successful training episodes | 3000 |
| `generalized_max_attempts_per_iteration` | 12 |
| training base seed | 740000 |
| MAXIMUM POSSIBLE training seed band | `[740000, 744500)` |
| `worlds_per_cell` | 3 |
| R1 benchmark base seed | 840000 |
| `max_candidates_per_cell` | 12 |
| evaluation | every 25 iterations |
| checkpoint | every 25 iterations |
| early stopping | none |
| solver timeout | none adopted |
| CTDE arm | **not run** |

The seed band is the **MAXIMUM POSSIBLE ATTEMPT BAND** — `375 × 12 = 4500` — because a
failed replacement attempt still spends a seed (`CLAUDE.md` §5). **No CTDE run exists, is
scheduled or is authorized**, and **no actor-only-vs-CTDE generalized result exists.**

**FOUR DIFFERENT FACTS, AND THEY MUST NOT BE COLLAPSED INTO ONE NEGATIVE.**

1. **THE R1 SCIENTIFIC BENCHMARK SCALE IS SELECTED AND AUTHORIZED: `worlds_per_cell = 3`.**
   This is the decision the bounded runtime / solver validation existed to inform, and it
   has been taken. **Statements elsewhere that "no FINAL SCIENTIFIC worlds-per-cell scale
   has been SELECTED" are HISTORICAL** — accurate through Task 4 — and are superseded here.
2. **THE R1 BENCHMARK CONSTRUCTION IS AUTHORIZED AND DISPATCHED:** candidate base seed
   `840000`, `max_candidates_per_cell = 12`, a **NEW R1 benchmark to be constructed by the
   deterministic preflight BEFORE training.**
3. **NO CONCRETE R1 MANIFEST HAS YET BEEN INDEPENDENTLY REVIEWED OR APPROVED AS THE
   COMPARATOR.** R1 is `RESULT PENDING`. **Do not claim an R1 manifest exists** unless
   execution evidence later establishes it, and **no benchmark manifest is committed or
   tracked in the repository.** The Task-5B transient manifest is **NOT** it (§3m.3).
4. **NO GENERALIZED SCIENTIFIC MEASUREMENT RESULT EXISTS** — no reward, convergence or
   validity result, and no actor-only-vs-CTDE result.

**THE REVIEW REQUIREMENT IS UNCHANGED AND IS NOT DISCHARGED BY DISPATCH.** The eventual run
requires **independent GPT artifact review** before any **`APPROVE — VALID MEASUREMENT`**
verdict, judged under the same gate every measurement is: **VALIDITY BEFORE PERFORMANCE**, a
mean never read without its denominator, an all-failed batch `null` and never `0.0`,
within-world claims only from COMPLETE matched groups, and FD-wake rates over FD WAKES. **A
null or negative generalized result is a valid observation, not a technical failure and not
grounds to re-tune, re-seed or re-run.** The requested-vs-realized hidden-cardinality
inspection (§3l.6) is a **human / GPT scientific-review decision** — the code REPORTS the
distribution and applies no threshold.

**EARLY STOPPING — CURRENT CAMPAIGN STATE.** **No reviewed early-stopping mechanism
exists.** R1 uses a **fixed 3000-success budget**. **Checkpoint RESUME is out of scope and
`graph_train` is SAVE-only**, as its own documented contract says. Any future early stopping
is a **SEPARATE research / design decision**, and it **must not select against the same
final benchmark without an explicit validation / test design** — stopping on the comparator
would make the comparator part of the training signal. **It is not implemented here and is
not authorized here.**

**CLUSTER READINESS IS DEFERRED**, because cluster access is not currently available. **This
does NOT block the local actor-only R1.** No scheduler, queue, partition or runbook detail
exists, and none may be invented.

**`p(destroy)` REMAINS `1.0`; `p(destroy) < 1` REMAINS DEFERRED** (§3l.3, §6).

### 3m.5 The integration sequence — PERFORMED

**THE NINE-STEP SEQUENCE BELOW WAS PERFORMED, EXACTLY AS PLANNED, ON 2026-08-30/31.**
*(SUPERSEDED: this subsection was previously titled "Intended integration sequence —
RECORDED, NOT PERFORMED" and opened "NO MERGE IS AUTHORIZED BY THIS DOCUMENTATION RECORD".
That was accurate while the stack was unmerged; the steps themselves are unchanged, and what
follows is the same list recorded as history.)* The stack was `main` → PR #42 → PR #43 →
docs PR, and the executed sequence was:

1. merged approved **PR #42** into `main` — merge
   `5dfcd8b632be8dca3c1730018bbf35337d07f077`;
2. refreshed the exact live `main` SHA from GitHub;
3. retargeted **PR #43** from the PR-#42 branch to `main`;
4. **exact-base re-reviewed PR #43 before merging** — changing a PR's base invalidates a
   base-relative verdict even though the candidate SHA is unchanged;
5. merged the **unchanged approved** PR #43 head `4af6c5aa…` — merge
   `b3c2e01f130afe854b09384cd6e1e196de714795`;
6. refreshed the exact live `main` SHA again;
7. retargeted the **docs PR** (PR #44) to `main`;
8. **exact-base re-reviewed the docs candidate**;
9. merged the **unchanged** docs candidate `88352b2f…` — merge
   `9b9e9b85a70c8a0019c72ada92ceec3401725795`, which is the live `main` this closure record
   is based on.

**No rebase, no squash, no cherry-pick, no force-push and no history rewrite occurred at any
step.** Every integration is a normal merge commit that preserves its reviewed candidate as
an ancestor / merge parent, and in all three cases the integrated tree was verified equal to
the reviewed tree, so **all three candidate commits remain reachable through normal merge
history.**

## 4. Current work — GENERALIZED-V1 (STEPS 1–5 ALL MERGED; THE POST-INTEGRATION CLOSURE CANDIDATE IS THE SOLE WRITABLE TASK WHILE ITS PR IS OPEN; THE ACTOR-ONLY R1 IS DISPATCHED WITH ITS RESULT PENDING); Phase-A and Variable-Severity baselines CLOSED; PHASE-B CTDE MERGED

Start with fresh exact-SHA initialization against the current `main` (§9). **A documentation
record neither authorizes nor runs anything: it RECORDS state only.** No documentation record
authorizes CC to implement Task 5 or any later generalized step, to freeze a benchmark
population, to run training, to generate a scenario, to call BONMIN, to re-run any completed
baseline, or to review a previously executed old-contract CTDE measurement.

**THE ACTIVE PHASE IS GENERALIZED-V1. ITS IMPLEMENTATION SEQUENCE IS NOW MERGED THROUGH
STEP 5 (§3m.1, §3m.5).**
**PR #42** (`312f58650b61a85eb72d0554d60715afee862a5c` → `5dfcd8b632be8dca3c1730018bbf35337d07f077`),
**PR #43** (`4af6c5aa5dd28072692bfda63282964b55010aae` → `b3c2e01f130afe854b09384cd6e1e196de714795`)
and the documentation lock **PR #44**
(`88352b2fc03174e8095d3c7e8a1ef58b60e58e0b` → `9b9e9b85a70c8a0019c72ada92ceec3401725795`)
are ALL APPROVED and ALL MERGED;
**this post-integration closure candidate is the SOLE WRITABLE REPOSITORY TASK while its own
draft PR is open, and on its integration no writable repository task remains** — while
**GENERALIZED-V1 itself stays an ACTIVE phase, because R1 is still pending.**
*(SUPERSEDED: this passage previously read "MERGED THROUGH STEP 4, AND STEP 5 IS IMPLEMENTED
AND APPROVED AS A STACKED, STILL-UNMERGED TWO-PR STACK … both APPROVED, FROZEN / READ-ONLY
and NOT merged … no merge is authorized here". Accurate at that checkpoint.)* **Task 5A and
Task 5B are
`APPROVE — VALID ENGINEERING VALIDATION` — engineering evidence, never a measurement**
(§3m.3). **The FIRST GENERALIZED-V1 ACTOR-ONLY R1 long run is `AUTHORIZED / DISPATCHED —
RESULT PENDING`, unreviewed, with no verdict and nothing about its outcome stated or
inferable** (§3m.4). **The external long-run task is RUN-ONLY and owns NO repository writes.**
*(SUPERSEDED by the three paragraphs above: this section's "TASK 4 IS CLOSED … NO
implementation candidate remains under review … THE SINGLE NEXT UNRESOLVED STEP IS
GENERALIZED-V1 TASK 5 … NOT started and NOT authorized" framing, its claim that no
generalized run is scheduled or authorized, AND its claim that no FINAL SCIENTIFIC
worlds-per-cell scale has been SELECTED and no benchmark population has been scheduled or
authorized — the R1 scale IS selected and its construction IS authorized and dispatched
(§3m.4). Each was accurate when written. The rest of this section is preserved and remains
accurate.)*

**THE PRECEDING RECORD (2026-08-26), PRESERVED.** The repository is **NOT closed and NOT idle**. **§3l.8 STEPS 1, 2, 3 AND 4
ARE ALL COMPLETE, REVIEWED AND INTEGRATED** — Task 1 (`5b55ca34…` → `9b305e4e…`, PR #35),
Task 2 (`185d39f0…` → `ca0dc406…`, PR #36), Task 3 (`24a8b1ee…` → `df3abf2f…`, PR #38,
**APPROVE**) and Task 4 (`db790138…` → `b4daa8c1…`, PR #40, **APPROVE**) — so §3l.1–§3l.7 are
implemented to the extent those four tasks represent, and their contracts are recorded in
`CLAUDE.md` §4 / §5 / §6 / §7. The FOUR low-level policy seams are OPT-IN with historical
defaults and are resolved TOGETHER, and only together, by the ONE `episode_design` selector,
which carries exactly those four ids; the generalized harness ADDITIONALLY uses the
generalized cardinality sampler and requires the SEPARATE `fuel_damage_mode` field to be
`seeded_variable`, neither of which is a policy id on `EpisodeDesign`. **GENERALIZED-V1
TASK 4 IS CLOSED: its implementation branch and PR are no longer writable or active, and NO
implementation candidate remains under review.** *(HISTORICAL as of 2026-08-30 — the
measurement, scale and authorization clauses that follow are SUPERSEDED as CURRENT state by
§3m.4 and by §1's four-fact block.)* **NO generalized scientific measurement
exists, is running, is scheduled or is authorized**, **no FINAL SCIENTIFIC benchmark
worlds-per-cell scale has been SELECTED and no FINAL SCIENTIFIC benchmark population or
manifest has been committed, preserved as the comparator, scheduled or authorized**, and
**no final actor-only or CTDE
generalized campaign is authorized.** The **GPT orchestrator owns the work**. **THE SINGLE
NEXT UNRESOLVED STEP IS GENERALIZED-V1 TASK 5** — bounded runtime / solver validation BEFORE
the final scientific run scale is decided (§3l.8 step 5) — and **it is NOT started and NOT
authorized** (§8). *(SUPERSEDED: the handoff-bootstrap record's "DESIGN ONLY / every line NOT
YET IMPLEMENTED / step 1 is next" framing, the Task-1/2 checkpoint's "§3l.5 is NOT IMPLEMENTED
/ Task 3 is next" framing, and the Task-3 record's "§3l.6–§3l.7 are NOT IMPLEMENTED / neither
harness selects any of them / Task 4 is next and unauthorized" framing were each accurate when
written and are history now.)*

**WHAT THAT CHANGES ABOUT THE PREVIOUS RECORD.** The 2026-08-23 closure record's
`REPOSITORY CLOSED / IDLE` state is **HISTORICAL and SUPERSEDED**: it was accurate when
written, the read-only walkthrough it anticipated took place, and the project then entered
the redesign. **Task 9 below — the first controlled actor-only vs CTDE comparison on the OLD
FIXED CELL — is NO LONGER the live next action** (see its own heading). Its constraints,
non-claims and prohibitions are preserved unchanged and remain binding if it is ever
resumed, and **nothing here claims any CTDE result, benefit or validity.**


**THE ORDERING, STATED ONCE — PARALLEL SINCE 2026-08-22, AND NOW FULLY TRAVERSED.** By
explicit user/orchestrator decision the approved arrangement was: **(1)** preserve the
original Phase-A reference baseline; **(2)** implement (DONE, PR #27) and MEASURE (**DONE —
EXECUTED, INDEPENDENTLY REVIEWED, `APPROVE — VALID MEASUREMENT`, primary finding
NEGATIVE**, §3j) the additional actor-only FD-VARIABLE-SEVERITY-v1 baseline, **measured on
the immutable detached snapshot at exact SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`,
tree `dd881478b8e2e521054d09bc865437f1308be1a2`**; **(3)** **PHASE-B CTDE DESIGN AND
IMPLEMENTATION PROCEED CONCURRENTLY** in a separate writable task branch / worktree, since
that snapshot was isolated — **DONE, merged as PR #30** (§3k); and **(4)** **CTDE
INTEGRATION into `main` gated** — **that gate is now CLOSED / SATISFIED on both halves**,
the measurement-validity half by item 2's verdict and the reference half by
`pre-ctde-actor-only` = `d437084c5fb1a22c21596a48c58e03f7e15a0115`. **All four items are
complete**, so this paragraph is now the arrangement's HISTORICAL record. **THE LIVE STATE
IS TASK 10 / GENERALIZED-V1: §3l.8 steps 1–5 are ALL MERGED (PR #42 → `5dfcd8b6…`,
PR #43 → `b3c2e01f…`, PR #44 → `9b9e9b85…`), this post-integration closure candidate is the
SOLE WRITABLE REPOSITORY TASK while its own draft PR is open, and the actor-only R1 long run
is AUTHORIZED / DISPATCHED with its RESULT PENDING** (§3m, §4 Task 10, §8). *(SUPERSEDED:
this clause previously pointed at Tasks 8 and 9 as the live state, then at "steps 1–3
MERGED, TASK 4 next", then at "TASK 4 is CLOSED, and TASK 5 … is NOT STARTED and NOT
AUTHORIZED", and most recently at "step 5 is IMPLEMENTED AND APPROVED as the STILL-UNMERGED
PR #42 + PR #43 stack"; those records are historical and are unchanged.)*

**This supersedes the serial "(3) only then proceed to Phase-B CTDE design" rule this
section previously stated** — that rule is HISTORY and must not be restated as live. It also
still supersedes the ORIGINAL claim that Phase-B CTDE was immediately next and that a
stochastic/partial fuel-degradation variant was deferred until after Phase B —
FD-VARIABLE-SEVERITY-v1 is that variant's approved form. **It is NOT a reopening of Phase A,
and it changes NO technical CTDE contract.** The Phase-A baseline stays CLOSED, VALID and
IMMUTABLE (§3h), and `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) must
not move — it is historical provenance for the ORIGINAL Phase-A reference and is never
repurposed as the future pre-CTDE reference.

**OWNERSHIP — THE GPT ORCHESTRATOR HOLDS IT, INCLUDING FOR THIS RECORD.** Through the
parallel phase the CTDE GPT orchestrator was the SOLE WRITABLE repository owner and the FD
measurement orchestrator was READ-ONLY on its detached snapshot. *(Historical: the user
granted the FD orchestrator a ONE-TIME writable exception scoped to the variable-severity
closure record alone; that exception ENDED when the record was integrated, and the FD
orchestrator reverted to READ-ONLY with no writable branch or PR.)* Writable ownership then
returned to the GPT orchestrator, which integrated PR #30, the CTDE documentation/lock
PR #32 and the chat/repository closure PR #33 — **all three are CLOSED** — and **it now owns
the GENERALIZED work (§3l, Task 10)** — it integrated the handoff bootstrap (PR #34), then
GENERALIZED-V1 Task 1 (PR #35), Task 2 (PR #36), their documentation checkpoint (PR #37),
Task 3 (PR #38), its documentation/lock record (PR #39) and Task 4 (PR #40). **NO
implementation candidate is active** — the Task-4 branch and PR are CLOSED and no longer
writable — and **no documentation record is a standing writable task**: each is the sole
writable candidate only while its own draft PR is open.
**The CTDE integration gate's repository-side prerequisite is DISCHARGED**:
`pre-ctde-actor-only` = `d437084c5fb1a22c21596a48c58e03f7e15a0115` exists and must not
move.

**Tasks 0–5 are ALL DONE.** Defect A (`d56fda6`, PR #17), Defect B (`39a16f2`, PR #19) and
Defect C (`ea62e4e`, PR #21) closed the three-defect CODE correction. The corrected-cell
short-probe rerun ran and its verdict was later SUPERSEDED (§3e). Defect 4, roster /
world-truth integrity, closed (`36365f2`, integrated `f37ea1c`, PR #24 — §3g). **Task 5, ONE
FRESH LONG BASELINE FROM SCRATCH, has been EXECUTED, independently reviewed and APPROVED
(§3h)** — it is the Phase-A baseline. **Nothing in Tasks 0–5 is outstanding, and the Phase-A
long baseline is NOT to be re-run, resumed, repaired, extended or re-tuned.**

**Task 6 — FD-VARIABLE-SEVERITY-v1 CODE. DONE (§3i, PR #27).** Merged and locked; the
measurement is Task 7 below.

**Task 7 — THE BOUNDED 50 × 8 ACTOR-ONLY VARIABLE-SEVERITY BASELINE. CLOSED — EXECUTED,
INDEPENDENTLY REVIEWED, `APPROVE — VALID MEASUREMENT`, PRIMARY FINDING NEGATIVE (§3j).** It
was a MEASUREMENT task, not an implementation task, and it changed no code, test, config or
preset.

- **CLOSURE.** Measured at exact code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
  `dd881478b8e2e521054d09bc865437f1308be1a2`, from a DETACHED clean snapshot worktree that
  was READ-ONLY with respect to the shared repository and carried no task branch. **It is
  and remains an ACTOR-ONLY measurement OF THAT SHA**: repository work landing afterwards —
  Phase-B CTDE included — is not in the measured tree and can neither be attributed to that
  run nor contaminate it. §3j is the volatile summary; `CLAUDE.md` §7 owns the authoritative
  record, every denominator, the evidence hashes and the explicit non-claims.
- **THE RESULT, IN ONE LINE.** Validity PASSED on all four clauses (7/8 complete triads in
  every one of the 11 rounds, `accounting_reconciled = true`, zero infrastructure or
  data-integrity faults, complete clean provenance) — and the deterministic held-out actor
  showed **NO severity-conditioned FD-wake meta-action separation**, at `pre_update` or at
  the final `post_update`. **A VALID NEGATIVE RESULT.** An earlier attempt at the same
  contract is `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT` (Windows `MAX_PATH`
  playback-export failure removed the whole `post_update` SEVERE arm) and is excluded.
- **IT IS NOT TO BE RE-RUN, resumed, repaired, extended or re-tuned.** A valid measurement
  exists; a negative finding is a result, not a reason to run it again. Both run trees are
  PRESERVED.
- **The approved run shape, which the executed run followed exactly** (kept as the CONTRACT
  of record; §3j and `CLAUDE.md` §7 hold what it produced):
  - **50 scheduled training iterations × 8 scheduled training attempts = 400 scheduled
    training attempts**, `base_seed = 0`;
  - **evaluation every 5 iterations, INCLUDING the initial `pre_update` round ⇒ 11
    evaluation rounds**;
  - **8 fixed held-out seeds in the EXISTING eval band**, each evaluated as a matched
    **clean / mild / severe TRIAD** ⇒ **11 × 8 × 3 = 264 scheduled evaluation attempts**;
  - **664 scheduled attempts in total; NO early stopping.**
- Training runs `fuel_damage_mode = seeded_variable` at the approved
  **0.50 clean / 0.25 mild / 0.25 severe** distribution. Everything else is the LOCKED cell:
  3 agents, 3 known + 3 hidden, 200 km / 100 km geometry, `DETECTION_KM = 50`,
  `include_sams = false`, `probability = 1`, frozen solver and BLADE, unchanged
  `graph_reward` formula with `aircraft_penalty_coeff = 2.25`, unchanged PPO.
- The run task chose a FRESH, NON-OVERWRITING output directory and captured its own
  provenance; §3j and `CLAUDE.md` §7 record which.
- **ONE invocation, driven through `--config` from its own measurement contract file** (the
  long-baseline pattern, deliberately NOT a repository preset), with `cli_overrides = []`.
- **The PRIMARY behavioural evidence is the severity-conditioned FD-WAKE meta-action
  response** — mild and severe meta-action counts, rates, and explicit wake denominators —
  **not reward**; reported alongside clean / mild / severe reward means, the three matched
  deltas (`mild − clean`, `severe − clean`, `severe − mild`) over COMPLETE TRIADS ONLY,
  attempted / successful / failed counts, and RTB / death / target-coverage outcomes. **The
  ten `post_update` rounds reuse the same seven feasible held-out seeds**, so 70
  observations per severity are a trajectory across checkpoints, not 70 independent worlds.
- **`p(destroy) < 1`, SAMs and dense reward were NOT part of this baseline** and must not be
  bundled into any successor.

**Task 8 — PHASE-B CENTRALIZED-CRITIC / CTDE IMPLEMENTATION. CLOSED — REVIEWED, APPROVED
AND MERGED (§3k, PR #30).** Approved candidate `a6f3aa9d62931994f416b2241fec4cfac3b018ec`,
integrated `8390d85c2072e9cbe984ce5f2731cef3a9b14985`, ordered parents
`d437084c5fb1a22c21596a48c58e03f7e15a0115` then `a6f3aa9…`, integration tree
`9686c107b8864f00a7d4403d70faf42ab561d2fb`.

**THE INTEGRATION GATE IS CLOSED / SATISFIED ON BOTH HALVES.** The measurement-validity
half was met by Task 7's independent verdict (`APPROVE — VALID MEASUREMENT`, §3j) —
satisfied by a NEGATIVE result, which counts exactly as a positive one would, because the
gate tests VALIDITY and never favourability. The remaining half — a NEW immutable
actor-only pre-CTDE reference preserved from the then-current actor-only state — was met by
**`pre-ctde-actor-only` = `d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the CTDE
integration's FIRST parent. **That ref must not move**, and **`phase-a-baseline` =
`4f0068847b017795717c5f0e331f647bcfc30547`** remains the SEPARATE original Phase-A
reference, never moved and never repurposed for it.

**WHAT IS NOW SETTLED, AND MUST NOT BE RE-OPENED AS A DESIGN QUESTION.** The five review
criteria this section used to list as requirements are IMPLEMENTED, REVIEWED and LOCKED as
a `CLAUDE.md` §5 contract: the size-agnostic value estimator off `GraphEncoder.pool()`; the
preserved decentralized no-communication EXECUTION in both modes; the EXPLICIT enumeration
of the training-only privileged inputs AND their exclusions; actor/critic separation with
its proof obligations; and the actor-only preservation and checkpoint distinction. **Do not
send CTDE back to a design/recon step, and do not rebuild what is merged.** Changing any of
it is a Grade-A change to a locked layer, routed through `CLAUDE.md` §6.

**Task 9 — THE FIRST CONTROLLED ACTOR-ONLY vs CTDE SCIENTIFIC COMPARISON ON THE OLD FIXED
CELL. NO LONGER THE LIVE NEXT ACTION — SUPERSEDED IN ORDERING BY THE GENERALIZED REDESIGN
(Task 10, §3l).** **TWO things changed since this task was written, and they are separate.**
(i) The live phase is now the GENERALIZED redesign, so this fixed-cell comparison is NOT
what the repository does next. (ii) **A CTDE measurement HAS since been executed under the
OLD FIXED-CELL contract; it is OUT OF SCOPE for the generalized redesign and must NOT be
reviewed, re-analysed or compared unless the user EXPLICITLY asks** (§1) — this record
states no identity, no measured SHA, no denominator and no verdict for it, because it did
not inspect it. **Everything below is PRESERVED UNCHANGED and remains binding if this task
is ever resumed**, and none of it is authorized by this record. **No CTDE
benefit may be pre-claimed** — not from the Phase-A result, not from the variable-severity
baseline (whose NEGATIVE severity finding is **NOT** evidence that centralized training
would change it), and **not from CTDE implementation work, engineering tests, bounded
engineering smokes or a passing test suite, none of which measure anything scientific.**

**THE COMPARATOR IS THE EXISTING APPROVED PHASE-A ACTOR-ONLY BASELINE — IT IS NOT RE-RUN.**
The actor-only arm is already measured: the approved Phase-A long baseline at measured code
SHA `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`, run
`training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, `APPROVE — VALID
MEASUREMENT` (§3h; `CLAUDE.md` §7 owns the authoritative record). It is PRESERVED and
**NOT authorized to be re-run, resumed, repaired, extended or re-tuned** — so **this
documentation task does NOT authorize a fresh actor-only run**, and the work Task 9 actually
schedules is the **CTDE arm**, measured and then compared against that existing record. If a
future orchestrator wants a newly executed actor-only CONTROL arm, that is a **separate
research-design decision requiring explicit user authorization**, never an implication of
this record.

Its binding constraints:
- the CTDE run **MATCHES THE LOCKED ORIGINAL PHASE-A SCIENTIFIC CELL** — 3 agents, 3 known
  + 3 hidden, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams = false`,
  `probability = 1`, frozen solver and BLADE, unchanged `graph_reward` formula with
  `aircraft_penalty_coeff = 2.25` — judged against the approved Phase-A baseline (§3h)
  under the SAME validity gate below, **VALIDITY BEFORE PERFORMANCE**;
- it **also matches that approved baseline'''s training / evaluation SCHEDULE, seed policy,
  held-out band and evaluation construct**, exactly as the authoritative Phase-A record
  (`CLAUDE.md` §7) establishes them — that is what makes the two records comparable;
- **THE EXPERIMENTAL FACTOR UNDER TEST IS ACTOR-ONLY TRAINING vs CENTRALIZED-CRITIC
  TRAINING.** State it that way, and **do NOT claim the two arms''' repository or
  configuration artifacts differ only by one `training_mode` field**: the approved Phase-A
  measurement and any future CTDE measurement have **DISTINCT measured code SHAs**
  (`737b4bf…` vs whatever the CTDE run is pinned to), and honest provenance must say so
  rather than assert a literal same-SHA / same-config equality that does not hold;
- it **MUST NOT bundle `p(destroy) < 1`, SAMs, dense reward, a solver change, a
  reward-formula change, or any new difficulty factor** (§6). Bundling one would make the
  comparison uninterpretable;
- a run showing no CTDE improvement, or no productive update, is a **valid NEGATIVE
  observation** — not a technical failure, and not grounds to re-tune or re-run;
- **no CTDE preset exists in the repository**, and creating one belongs to this task.

**What makes a run VALID — carried forward unchanged, and it now has a passing precedent.**
A run counts as a valid measurement when ALL of:

- Git provenance is COMPLETE and the checkout was clean;
- `run_summary.json:accounting_reconciled` is true;
- no INFRASTRUCTURE or DATA-INTEGRITY failure occurred. A `_VisualArtifactError`, a
  `MeasurementIntegrityError` / `EpisodeRosterError` — including the scheduled-vs-executed
  CELL mismatch PR #27 added (§3i) — or any crash outside the
  `generation` / `setup` / `run` / `reward` episode taxonomy ABORTS the run and is not a
  scientific result;
- **at least one COMPLETED matched group exists in BOTH the `pre_update` and the
  `post_update` round.** A group counts only when EVERY member completed — two members for
  a legacy pair, **all three for a variable-severity triad.**

The Phase-A rerun (§3h) satisfied all four, and so did the variable-severity baseline
(§3j); the two runs before the Phase-A rerun did not, on the data-integrity clause (§3e,
§3f), and the variable-severity precursor did not either, on the same clause.

**A negative result is still a valid result — and §3j is now the worked example.** No
improvement, no severity-conditioned behavioural difference, or zero productive PPO updates,
is a valid NEGATIVE SCIENTIFIC OBSERVATION — not a technical failure, and not grounds to
re-run, re-tune or re-seed. The variable-severity baseline measured exactly that: productive
training and no severity-conditioned separation.

**Interpretation rules survive unchanged:** a held-out mean is never read without its
denominator; an all-failed batch reports `null`, never `0.0`; an empty successful-group
population is `null` too; the per-condition / per-cell means are each over their own
successful subset, so the within-seed claims are the matched deltas over COMPLETE groups
alone (`CLAUDE.md` §5); and FD-wake meta-action rates are reported over FD WAKES, never over
episodes. **Do not reuse §2's, §3e's or §3f's numbers as any expectation** — §2 measured a
different, easier cell, and §3e and §3f are both scientifically INCONCLUSIVE, as is the
variable-severity `MAX_PATH` precursor (§3j). **TWO valid scientific baselines now exist and
they measure DIFFERENT cells: §3h is the LEGACY FD-BASELINE-v1 baseline, and §3j is the
FD-VARIABLE-SEVERITY-v1 baseline. Neither is an expectation for the other, and neither is an
expectation for any CTDE comparison.**

**Task 10 — GENERALIZED-V1. IMPLEMENTATION STEPS 1–5 ALL MERGED; THE POST-INTEGRATION
CLOSURE CANDIDATE IS THE SOLE WRITABLE REPOSITORY TASK WHILE ITS PR IS OPEN, AND ON ITS
INTEGRATION NO WRITABLE REPOSITORY TASK REMAINS; THE ACTOR-ONLY R1 LONG RUN IS AUTHORIZED /
DISPATCHED WITH ITS RESULT PENDING (§3m). THIS IS THE LIVE PHASE — ACTIVE BECAUSE R1 IS
PENDING, NOT BECAUSE A REPOSITORY TASK IS OPEN.** *(SUPERSEDED: this heading previously read
"TASK 4 CLOSED; NO WRITABLE IMPLEMENTATION TASK; STEP 5 IS THE NEXT UNRESOLVED STEP AND IS NOT
STARTED AND NOT AUTHORIZED", and then "STEP 5 IMPLEMENTED AND APPROVED AS A STACKED,
STILL-UNMERGED TWO-PR STACK; THE DOCUMENTATION CANDIDATE IS THE SOLE WRITABLE REPOSITORY
TASK".)* Its APPROVED DESIGN is recorded in **§3l** and is not
repeated here: generalized cardinality with bounded B2 backoff (§3l.1–§3l.2), FD capability
by construction with deterministic bounded ego eligibility (§3l.3), repeated post-FD decision
points for the damaged ego alone (§3l.4), the event-conditioned MATCH-AOU continuation
reference (§3l.5), the fixed stratified 18-stratum benchmark (§3l.6), and the diagnostics
that make all of it readable (§3l.7).

- **STATUS: STEPS 1, 2, 3 AND 4 ALL IMPLEMENTED, REVIEWED AND MERGED, so §3l.1–§3l.7 are
  implemented to the extent those four tasks represent.** Task 1 — candidate
  `5b55ca348309b4241d2087c2f60327bc842ea6fa`, integration
  `9b305e4ee427dd27fac6beee8fc4f7a8a763f7f9`, PR #35. Task 2 — final candidate
  `185d39f00335a0bb5e9130cc773da94c914f17f5`, integration
  `ca0dc406ad11eb18e11e87e7f9ddf2e4e457f64b`, PR #36, reached through the append-only fix
  chain from the REQUEST-FIXES candidate `2f9231d989acf30561ecf10e74cf0c5491771836`.
  Task 3 — candidate `24a8b1ee42b1d32731fa7f5cef09fcfab50bb33e`, integration
  `df3abf2f2eb3ac9c02bc4bd3d8320e095075bd25`, PR #38, verdict **APPROVE**, with ONE
  review-APPROVED compatibility deviation (`damaged_event_unrealized_t0`) recorded
  explicitly and scoped to the LEGACY Task-2 contract. Task 4 — FINAL approved candidate
  `db79013897a6e5669f50d53b6e30229b16aea28d`, integration
  `b4daa8c1a8c870061b26cceb01d4ed34169594e7`, PR #40, verdict **APPROVE**, reached through
  the append-only fix chain from the original reviewed candidate
  `eef1795f6bb3f0cbc4c163ba489cf5e790df4c41` (manifest integrity, real held-outness, honest
  generalized construction provenance).
  Task 5 — **PR #42** (`312f58650b61a85eb72d0554d60715afee862a5c`, integration
  `5dfcd8b632be8dca3c1730018bbf35337d07f077`) and **PR #43** (FINAL
  approved head `4af6c5aa5dd28072692bfda63282964b55010aae`, append-only child of
  `734f1e786593b6ffb94f1f8d7283b1f2fc79d257`, integration
  `b3c2e01f130afe854b09384cd6e1e196de714795`), locked by the documentation **PR #44**
  (`88352b2fc03174e8095d3c7e8a1ef58b60e58e0b`, append-only child of `61eaa3fe…`, integration
  `9b9e9b85a70c8a0019c72ada92ceec3401725795`) — **ALL THREE APPROVED AND ALL THREE MERGED**
  (§3m.1, §3m.5). *(SUPERSEDED: this line previously read "BOTH APPROVED, BOTH FROZEN /
  READ-ONLY and NEITHER MERGED, so no integration SHA exists for either".)*
  **NO generalized measurement RESULT exists.** A first generalized **ACTOR-ONLY** R1 long
  run IS authorized and dispatched with its result **PENDING and UNREVIEWED** (§3m.4); **no
  CTDE generalized run exists, is scheduled or is authorized.** The bounded real-BLADE /
  BONMIN smokes taken during each task's validation, and the Task-5A / Task-5B validations
  (§3m.3), are ENGINEERING evidence only and carry no scientific contract, seed schedule,
  held-out band or denominator. *(SUPERSEDED: this bullet previously read "No generalized
  world CAMPAIGN has been generated and NO generalized measurement exists, is running, is
  scheduled or is authorized". Accurate when written.)*
- **THE FOUR LOW-LEVEL POLICY SEAMS ARE OPT-IN, AND SINCE TASK 4 BOTH HARNESSES RESOLVE
  THEM TOGETHER — AND ONLY TOGETHER.** `TrainConfig.episode_design` /
  `RolloutConfig.episode_design` ∈ (`fixed_cell_v1` DEFAULT, `generalized_v1`) resolve the
  whole approved bundle through ONE site; `graph_generalized.EpisodeDesign` carries EXACTLY
  `hidden_policy`, `eligibility_policy`, `post_fd_wake_policy` and `reference_policy` — four
  ids, not five — and there is deliberately NO standalone field for any of them on either
  config, so a run can never resolve half a bundle. **Two generalized-path behaviours sit
  BESIDE that resolution and are NOT policy ids:** the generalized cardinality sampler
  (harness / population behaviour, consulted because `cfg.generalized` is true) and
  `fuel_damage_mode`, which stays a SEPARATE config field that `validate()` merely REQUIRES
  to be `seeded_variable` under `generalized_v1`. A run that does not name `generalized_v1`
  still builds the
  historical `exact_v1` + `legacy_selected_ego_v1` + `single_wake_v1` world and scores it
  against `static_t0_v1`, and its calls into `_run_one_episode` / `setup_episode` are
  byte-invariant. Every Task-1/2/3 per-episode diagnostic structure is now persisted and
  aggregated. *(SUPERSEDED: this bullet previously read "ALL THREE MERGED FAMILIES ARE
  OPT-IN, AND NEITHER HARNESS SELECTS THEM … Nothing persists or aggregates the new
  diagnostic structures. Wiring them up is step 4." That was accurate before PR #40.)*
- **WHAT IS STILL NOT DECIDED.** **No FINAL SCIENTIFIC benchmark worlds-per-cell SCALE has
  been SELECTED**, and **no FINAL SCIENTIFIC benchmark POPULATION or manifest has been
  committed, preserved as the comparator, scheduled or authorized.** No benchmark manifest
  is committed or tracked in the repository — *a negative scoped deliberately: transient
  manifests built by tests and engineering validation are legitimate, are neither committed
  nor a reviewed comparator, and repository state cannot establish a global negative over
  local scratch files.* Task 4 delivered the manifest MECHANISM only, and its builder
  REFUSES to invent a world count; **Task 5 added the production selection caller, and the
  R1 scale IS selected and its construction IS authorized and dispatched** (§3m.4).
  `configs/graph_train/final_cell_probe.json` remains the ONLY repository preset and it is
  `fixed_cell_v1`, so no repository preset selects `generalized_v1`, and **no FINAL
  SCIENTIFIC benchmark/run preset has been committed to the repository.** *(A claim about SCIENTIFIC
  artifacts ONLY — the GENERALIZED-V1 Task-1/2/3/4 TECHNICAL contracts DO exist, are locked,
  and are authoritative in `CLAUDE.md` §4 / §5 / §6 / §7.)*
- **OWNERSHIP:** the **GPT orchestrator** owns the work. **No documentation record is a
  standing writable task** — each is the sole writable candidate only while its own draft PR
  is open — and **no implementation candidate is active**: the Task-4 branch and PR are
  CLOSED and no longer writable, and no Task-4 candidate remains under review.
- **THE SEQUENCE IS §3l.8: STEPS 1–5 ALL COMPLETE, REVIEWED AND INTEGRATED.** Each was a
  separate, separately
  scoped, separately reviewed bounded task, and each started only after the previous one was
  reviewed and integrated. *(SUPERSEDED: this bullet previously read "STEPS 1–4 COMPLETE,
  STEP 5 NEXT".)* **`CLAUDE.md` locks are written PER COMPLETED TASK, for behaviour
  that already exists — never for a design** — and a FINAL documentation pass is still
  required after any later step lands.
- **STEP 5 IS IMPLEMENTED, APPROVED AND INTEGRATED (§3m.1), AND THE INTEGRATION HAS BEEN
  PERFORMED.** It ran under the exact-base re-review
  discipline of §3m.5 — merged PR #42, refreshed live `main`, RETARGETED PR #43 and
  **exact-base re-reviewed it before merging** (changing a PR's base invalidates a
  base-relative verdict even though the candidate SHA is unchanged), merged the unchanged
  head, refreshed `main`, retargeted and re-reviewed the docs candidate, merged it — with
  **no rebase, squash, cherry-pick, force-push or history rewrite** at any step. The
  dispatched R1 still requires **independent GPT artifact review** before any
  `APPROVE — VALID MEASUREMENT` verdict (§3m.4, §8). *(SUPERSEDED: this bullet previously
  gated step 3, then step 4, then step 5; all have since been implemented and reviewed —
  steps 3 and 4 integrated as PR #38 and PR #40, and step 5 as PR #42, PR #43 and PR #44. Its
  "NO MERGE IS AUTHORIZED BY THIS DOCUMENTATION RECORD" line was accurate while the stack was
  unmerged; the merges were separately authorized and have since been performed.)*
- **BINDING CONSTRAINTS CARRIED INTO EVERY STEP.** Several approved directions in §3l
  deliberately depart from contracts `CLAUDE.md` §5 previously locked — B2 exact
  cardinality, the fixed 3/3/3 cell, the FD eligibility and failure policy, the
  damaged-episode t=0 reference solve, and the fixed held-out eval band — and **all five have
  now been addressed by Tasks 1–4, each as its own reviewed OPT-IN seam beside the preserved
  historical default.** **Any further such departure is a Grade-A change to a locked layer,
  routed through `CLAUDE.md` §6**, never folded into another task. In addition: the
  historical exact-cardinality path is PRESERVED, no severity, cardinality, stratum or
  hidden-count feature may reach `GraphObservation` or `CentralGraphObservation`, no new
  meta-action is added, no peer behaviour changes, no communication channel is introduced,
  `p(destroy)` stays `1.0`, and `p(destroy) < 1`, SAMs and dense reward stay OUT (§6).
- **THE COMPARATOR PROBLEM, STATED ONCE.** The generalized benchmark is a **NEW** benchmark.
  **Historical fixed-cell measurements are NOT it** — §3h, §3j and any old-contract CTDE run
  measured a different cell under a different contract, so none of them is a generalized
  baseline, a generalized comparator or a generalized expectation. **The approved historical
  baselines are REUSED as what they are and are never re-run, repaired, resumed or
  re-tuned.** Future actor-only and CTDE generalized measurements must run against the
  **exact same eventual frozen benchmark / world manifest** (§3l.6); that shared
  `manifest_id` is what makes them comparable to each other. **No final actor-only or CTDE
  generalized campaign is authorized, and no actor-only-vs-CTDE generalized result exists.**
- **VALIDITY BEFORE PERFORMANCE, unchanged.** The validity gate stated under Task 9 above
  applies to any generalized measurement too, and the interpretation rules survive unchanged
  — a mean is never read without its denominator, an all-failed batch is `null` and never
  `0.0`, within-world claims come only from COMPLETE matched groups, and FD-wake rates are
  reported over FD WAKES. **A null or negative generalized result is a valid observation,
  not a technical failure and not grounds to re-tune, re-seed or re-run.**
- **ONE ADDITIONAL PRE-MEASUREMENT CHECK IS PART OF THE DESIGN, NOT AN OPTION.** Before any
  scientific generalized measurement, the requested-vs-realized hidden-cardinality
  distributions must be INSPECTED, and the benchmark REJECTED or REDESIGNED if the HIGH load
  systematically degenerates (§3l.6). **Task 4 REPORTS that distribution** — per episode, in
  `run_summary.json:/generalized`, and in the fourth `measurement_health.png` panel — **and
  deliberately applies NO acceptance threshold and returns NO verdict.** The judgement is a
  human / GPT scientific review decision, never a computed one.

## 5. Closed decisions

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

## 6. Out of scope for the current work

The additional actor-only FD-VARIABLE-SEVERITY-v1 baseline (§4, Task 7) is **CLOSED,
VALID and NEGATIVE** (§3j), **PHASE-B CTDE IMPLEMENTATION is CLOSED, REVIEWED and MERGED**
(§4, Task 8; §3k), and **the current work is the GENERALIZED TRAINING / BENCHMARK REDESIGN
(§4, Task 10; §3l), whose §3l.8 STEPS 1, 2, 3, 4 AND 5 ARE ALL MERGED — so §3l.1–§3l.7 are
IMPLEMENTED AND THE TASK-5 STACK IS INTEGRATED TOO (PR #42 → `5dfcd8b6…`, PR #43 →
`b3c2e01f…`, PR #44 → `9b9e9b85…`; §3m.1, §3m.5) — the R1 BENCHMARK SCALE IS SELECTED AND
AUTHORIZED (`worlds_per_cell = 3`)
AND ITS CONSTRUCTION IS AUTHORIZED AND DISPATCHED, and NO CONCRETE R1 MANIFEST HAS YET BEEN
INDEPENDENTLY REVIEWED OR APPROVED AS THE COMPARATOR — none is committed or tracked in the
repository — and NO GENERALIZED SCIENTIFIC MEASUREMENT RESULT EXISTS (§1, §3m.4).**
*(SUPERSEDED: this preamble previously said step 5 was NOT STARTED, then that no FINAL
SCIENTIFIC BENCHMARK SCALE HAD BEEN SELECTED and no population had been SCHEDULED OR
AUTHORIZED, and most recently that step 5 was "IMPLEMENTED AND APPROVED AS A STACKED,
STILL-UNMERGED TWO-PR STACK". Each was accurate when written.)*
Out of scope:

- **IMPLEMENTING ANYTHING FURTHER, OR MERGING ANYTHING, FROM A DOCUMENTATION RECORD.**
  **No merge is authorized by this document.** The Task-5 stack was integrated under §3m.5
  after GPT exact-SHA review and explicit user-authorized continuation, with PR #43 and PR #44
  each EXACT-BASE RE-REVIEWED after retargeting — that is HISTORY, and it is not a standing
  authorization to merge anything else, this closure candidate included. **No concrete FINAL
  benchmark population, no
  FURTHER scale decision, no generalized preset, no early-stopping mechanism, no cluster
  runbook and no further scientific schedule may be produced on the strength of this
  document** — the R1 scale and build were authorized SEPARATELY and explicitly (§3m.4), not
  by any documentation record. *(Steps 1–5 were each separately authorized, their code is
  DONE and INTEGRATED, and the Task-5 integration sequence has been performed; that is
  history, not a standing authorization for anything further.)* *(SUPERSEDED: this bullet
  previously read "§3l.8 step 5 remains a PLAN and this document does not authorize it" and
  "no bounded runtime/solver validation campaign … may be produced". Step 5 has since been
  implemented and approved, and the bounded validation was performed and reviewed as Task 5A
  and Task 5B — ENGINEERING VALIDATION ONLY, §3m.3.)*
- **CLAIMING, IMPLYING OR ASSUMING THAT ANY GENERALIZED BEHAVIOUR HAS BEEN MEASURED, THAT A
  FINAL BENCHMARK POPULATION EXISTS, OR THAT A FINAL SCALE WAS SELECTED.** Steps 1–5 are
  implemented CODE with ENGINEERING evidence only: **no generalized measurement RESULT
  exists, no generalized result may be pre-claimed, no FINAL benchmark manifest has been
  committed or preserved as the comparator, and no actor-only-vs-CTDE generalized result
  exists** (§3l, §3m). **The dispatched actor-only R1 is `RESULT PENDING` and UNREVIEWED —
  dispatch is not a result, and nothing about its outcome may be stated or inferred** (§3m.4).
  **The Task-5B transient benchmark is NOT the R1 comparator**, and Task 5A / Task 5B are
  ENGINEERING VALIDATION, never measurement (§3m.3);
- **WRITING `CLAUDE.md` RECORDS FOR AN UNIMPLEMENTED GENERALIZED DESIGN.** `CLAUDE.md` now
  records steps 1, 2, 3, 4 and 5 as INTEGRATED behaviour, each with its exact candidate AND
  its exact merge SHA, per §7's hash convention. It must record nothing for behaviour that
  has not been reviewed; *(SUPERSEDED: this bullet previously said step 5 was recorded as
  "REVIEWED AND APPROVED CANDIDATES that are explicitly NOT YET INTEGRATED — with no invented
  merge SHA". Accurate at that checkpoint; the merge SHAs now exist and are recorded.)*
- **CREATING A SECOND HANDOFF OR A SEPARATE DESIGN DOCUMENT.** There is ONE handoff — this
  file — and the generalized design lives in §3l of it;
- **REVIEWING, RE-ANALYSING OR COMPARING THE PREVIOUSLY EXECUTED OLD-CONTRACT CTDE
  MEASUREMENT** — out of scope for the redesign unless the user EXPLICITLY asks (§1, §4
  Task 9);
- **TREATING ANY HISTORICAL FIXED-CELL MEASUREMENT AS THE GENERALIZED BENCHMARK, ITS
  COMPARATOR OR AN EXPECTATION FOR IT** (§3l.6);
- **RUNNING ANYTHING** from a documentation task — no training, no rollout, no BLADE
  scenario, no BONMIN solve, no probe, no artifact generation;
- **DELETING, MOVING OR CREATING ANY BRANCH OR TAG from this record** beyond the named task
  branch it pushes (§1);

- **any training run driven from THIS documentation task** — it authorizes none;
- **CHANGING ANY CODE, TEST, CONFIG OR PRESET IN RESPONSE TO TASK 7.** It was a
  MEASUREMENT of what PR #27 merged, and its NEGATIVE finding is a result, not a bug
  report. Retuning the severity bands, the mixture, the RTB margin, the leg-progress
  threshold, the reward coefficient, the seeds, the schedule or the harness is out of
  scope, and so is "fixing" anything the run revealed — including the playback-export
  routing caveat (§3j), which is a future engineering task of its own;
- **RE-RUNNING, resuming, repairing, extending or re-tuning the variable-severity
  baseline.** A valid measurement exists (§3j); a negative finding does not make it less
  valid;
- **CHANGING THE MERGED CTDE LAYER.** Phase-B CTDE is IMPLEMENTED, REVIEWED and MERGED
  (§3k, §4 Task 8), and its contract is locked in `CLAUDE.md` §5. Out of scope here:
  altering the central privileged inputs, the actor/critic boundary, the GAE/value
  semantics, the capture timing, checkpoint compatibility or actor-only preservation —
  each is a Grade-A change to a locked layer, routed through `CLAUDE.md` §6, never a fix
  folded into another task. Also out of scope: **treating any CTDE implementation work,
  engineering test or passing suite as scientific evidence**, **claiming any CTDE benefit**
  before the Task-9 comparison is executed and independently reviewed, and **modifying
  either preserved variable-severity measurement tree in any way**;
- **MOVING, DELETING OR REPURPOSING EITHER IMMUTABLE REFERENCE.** `pre-ctde-actor-only`
  (`d437084c5fb1a22c21596a48c58e03f7e15a0115`) discharged the CTDE gate's repository-side
  prerequisite and `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) is the
  SEPARATE original Phase-A reference. Both are immutable, neither substitutes for the
  other, and `flat-final`, the `pre-cleanup` tag, PR #30's history and every preserved
  scientific artifact stay preserved too;
- **weakening decentralized no-communication EXECUTION in any way.** No severity label may
  reach the acting path: peer nodes stay featureless, an ego still acts only on its own
  sensing and its own fuel, and `CLAUDE.md` §3 is not up for renegotiation;
- **re-running, resuming, repairing, extending or re-tuning the Phase-A long baseline.** A
  valid measurement exists (§3h); re-running it would not make it more valid. Its scientific
  contract — seeds, the 100 × 8 schedule, evaluation cadence, matched-pair design, PPO
  settings, cell geometry and FD-BASELINE-v1 parameters — is frozen as the reference
  baseline. **The variable-severity baseline is an ADDITIONAL run, not a replacement,
  a rerun or an extension of it;**
- **re-interpreting the approved Phase-A result, or claiming more than it establishes.** The
  non-claims in §3h are binding: no global optimality, no monotonic convergence, no
  generalization beyond this fixed cell and held-out seed set, no CTDE benefit. **It is a
  baseline of the LEGACY FD-BASELINE-v1 design and is not an expectation for the
  variable-severity cell;**
- **claiming MORE than the variable-severity result establishes.** §3j and `CLAUDE.md` §7
  are the whole of it: the negative finding does NOT mean the actor is broken, that training
  or PPO failed, that the actor ignores fuel, that MILD "should" have chosen
  `PLAN_COMPLIANCE`, that 70 post-update observations per severity are 70 independent
  worlds, that it generalizes beyond this fixed cell and held-out seed band, or that
  centralized training would change it;
- selecting or enabling a FURTHER difficulty factor — **`probability < 1` / stochastic
  target destruction, hostile fire / SAMs, and dense/per-wake reward all remain SEPARATE,
  still-deferred research changes**, none of them implemented by PR #27 and none of them
  bundleable into Task 7 or Task 8;
- **reopening Defects A, B or C, the roster/world-truth defect, or acting on the §3e
  over-safety hypothesis** — all four defects are closed, approved and merged, the first
  three are operationally witnessed, and the hypothesis is a future research question about
  policy calibration, not a defect, a reward change or a retune;
- **relaxing, retrying, retuning or reclassifying the B2 exact-cardinality or fuel-window
  failures** — they are not faults but expected scientific outcomes under
  `skip_and_account_v1` (§3f, §3h). This includes held-out seed `1000005`, whose structural
  B2 failure caps matched-group yield at 7/8 in the Phase-A pairs and in every
  variable-severity triad round alike: it is a property of that world, and it is reported,
  never repaired;
- reworking the merged FD-BASELINE-v1 mechanism, the merged visual-artifact surface or
  the merged probe harness (preset, `--config` precedence, `config_source`, run layout,
  the three figures), or their reviewed research decisions — a run RUNS what is merged.
  The §3d validity correction and the §3g roster correction were the ONLY authorized
  exceptions, each scoped to its own defect, and all are CLOSED; none ever was a licence to
  retune the cell, the reward, the seeds or the harness;
- extending artifact capture (per-seed filters, new artifact kinds, artifact-derived
  metrics) — a run uses what is merged;
- **modifying, repackaging, moving, copying, deleting or regenerating any preserved
  scientific artifact**, in particular `training_output_20260815_173029`,
  `training_output_20260816_162130`, `training_output_long_baseline_100x8_seed0`, the
  approved `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, and **BOTH
  variable-severity trees — the VALID run root `C:\Users\Itama\f7r2` and the INVALID
  `MAX_PATH` precursor
  `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640`** — an inconclusive or
  invalid run is still evidence, and preserved artifacts are what made the roster defect
  provable, the Phase-A result reviewable and the `MAX_PATH` failure diagnosable;
- **deleting ANY branch or tag from THIS CC task** — it retires nothing, and in particular
  it does not touch `task/phase-b-ctde-build` or PR #30, which belong to the CTDE
  orchestrator. Retiring any merged branch is a separate action owned by the GPT
  orchestrator, performed only after each tip is verified reachable from integrated `main`;
- **moving or deleting `phase-a-baseline`
  (`4f0068847b017795717c5f0e331f647bcfc30547`)** — it preserves the Phase-A reference code
  state and is IMMUTABLE — **or `pre-ctde-actor-only`
  (`d437084c5fb1a22c21596a48c58e03f7e15a0115`)**, which preserves the immediate pre-CTDE
  actor-only state and is equally IMMUTABLE. They are DISTINCT references and neither
  substitutes for the other;
- **deleting or moving `flat-final` or the `pre-cleanup` tag** — they are permanently
  preserved;
- checkpoint loading/resume;
- low-known-cell solver timeout unless the chosen cell needs `known <= 2`;
- ETA/peer-dropout, reachability-model and legacy-split retirement work;
- further repository/documentation hygiene — the README rewrite and the BLADE fork
  documentation audit were closed by PR #11's follow-up hygiene task, and Phase B neither
  needs nor may reopen them.

**THE FORMER "FUTURE DESIGN TOPIC" IS NOW REALIZED — and its ordering was deliberately
inverted.** Earlier handoffs recorded, as an unauthorized and unspecified future topic, a
SEPARATE stochastic/partial fuel-degradation difficulty in which RTB stays feasible but
mission continuation is not deterministically forced to be impossible — a softer, less
binary version of the FD-BASELINE-v1 window — to be considered **after** Phase B. **That
topic is FD-VARIABLE-SEVERITY-v1: it is specified, implemented, reviewed and MERGED
(§3i, PR #27), and by explicit decision its baseline was ordered BEFORE Phase B rather than
after — an ordering SUPERSEDED again on 2026-08-22 to PARALLEL, and its pinned measurement
is now EXECUTED and reviewed `APPROVE — VALID MEASUREMENT` with a NEGATIVE primary finding
(§3j), so the CTDE integration gate's measurement-validity half is SATISFIED and only the
NEW immutable actor-only pre-CTDE reference remains** (§4). FD-BASELINE-v1 itself is
unchanged, and the Phase-A reference baseline measured on it is untouched. What remains genuinely future and unauthorized is the rest of
the deferred list — `p(destroy) < 1`, hostile fire / SAMs and dense/per-wake reward — each
still its own research-design decision with its own semantics, observability, proof
obligations and bounded implementation/lock task.

## 7. Documentation duties

| Trigger | Duty |
|---|---|
| B1–B4 preparation lands — **DONE** | Contracts and locks recorded in `CLAUDE.md` |
| First real post-B3 probe completes — **DONE** | Exact code SHA, denominators, yield, failure stage, transitions and pre/post held-out measurements recorded |
| PR #7 observability follow-up lands — **DONE** | Unique-target semantics, per-episode output, eval artifact preservation and fix-chain lock recorded |
| Selected baseline-difficulty factors land — **DONE for FD-BASELINE-v1** | Contract in `CLAUDE.md` §5, tick placement in §4, routing in §6, lock + fix chain in §7, selection closure and deferrals in §8 — recorded without pre-claiming any result |
| Repository code-hygiene + documentation alignment lands — **DONE for PR #11 and its follow-up** | Helper ownership, retired-executor removal and the lock recorded in `CLAUDE.md` §5–§7; README replaced, BLADE fork documentation audited, obsolete scenarios and dead utility symbols removed |
| Visual-artifact support lands — **DONE for PR #10** | Contract in `CLAUDE.md` §5, routing in §6, lock in §7, and the §8 note that the bounded probe MAY enable it — recorded without pre-claiming any result |
| Probe harness lands — **DONE for PR #14** | Preset, `--config` precedence, three-kind `config_source` and the `plots/` figures recorded as contracts in `CLAUDE.md` §5, routed in §6, locked in §7 with the two-round fix chain; the retired four-panel `training_plot.png` removed from the contracts — recorded without pre-claiming any result |
| FIRST final-cell short probe completes — **DONE for `training_output_20260815_173029`** | Run identity, exact code SHA, accounting and denominators, and the three research-validity defects it exposed recorded in §3d — as findings only, with no scientific claim about the cell and no long-baseline authorization |
| §3d validity correction lands — **DONE: ALL THREE DEFECTS RECORDED and their locks integrated (PR #18, PR #20, PR #22)** | Record each corrected contract in `CLAUDE.md` §5–§7 with its own lock and fix chain, ONE DEFECT AT A TIME and only once that defect is merged, never in advance — a DOCUMENTATION-RECORDING rule, not a constraint on how the fixes are broken into tasks (`CLAUDE.md` §8 owns the sequential-defect policy, which is historical workflow context). **Defect A — DONE:** the ego-global `SELF_PRESERVATION_ABORT` selection/effect contracts are in `CLAUDE.md` §5 (Stages 4 and 5), the lock, fix chain and evidence in §7, and the defect state in §8. **Defect B — DONE:** the derived attack-confirmation wait contract is in `CLAUDE.md` §5 (Execution, Stage 1), routed in §6, with the lock, append-only fix chain and evidence in §7 and the defect state in §8. **Defect C — DONE:** the physical-completion contract is in `CLAUDE.md` §4 (the terminal loop) and §5 (Execution, Stage 1, and the tick loop), routed in §6, with the lock, append-only fix chain and evidence in §7 and the defect state in §8. The three-defect correction is COMPLETE: every lock is integrated into `main`, and the corrected-cell rerun in the row below has since witnessed all three operationally |
| Probe RERUN completes on the corrected cell — **DONE for `training_output_20260816_162130`; VERDICT LATER SUPERSEDED** | Exact config, provenance, denominators, clean/damaged and matched-pair populations, failures by stage, event/wake/RTB/death outcomes, reward headroom, update evidence, artifact completeness and playback witnesses recorded in §3e, with the authoritative measurement record and evidence hashes in `CLAUDE.md` §7 and the gate in §8. **Its original `VALID` verdict is SUPERSEDED by `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR`; the historical facts are retained and the supersession is stated wherever the verdict appears** |
| FIRST LONG BASELINE completes — **DONE for `training_output_long_baseline_100x8_seed0`; INCONCLUSIVE** | Exact contract and resolved configuration, complete provenance, the two elapsed quantities, every denominator, the failure breakdown by error type and by condition, the artifact findings and the evidence hashes recorded in §3f, with the authoritative measurement record in `CLAUDE.md` §7 and the gate in §8. **Verdict: engineering `REQUEST FIXES`, scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED`; its reward, paired-delta, survival, fuel-damage-yield and PPO-performance outputs are raw historical outputs and NOT scientific evidence** |
| Roster / world-truth correction lands — **DONE for PR #24** | Allocation-is-not-inventory and the two raw pre-solve snapshots recorded as contracts in `CLAUDE.md` §5, routed in §6, locked in §7 with the identical-tree evidence; the superseded PR-#7 routing corrected at its own site; the two affected measurements' verdicts revised in §7; and §8 updated so it no longer claims a passed gate or an unrun long baseline. §3g summarizes it here — recorded without pre-claiming any result |
| FRESH LONG BASELINE completes — **DONE for `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`; `APPROVE — VALID MEASUREMENT`** | Resolved configuration (the SAME contract as the preserved `long_baseline_contract.json`, differing ONLY in `output_dir`), complete provenance, the two elapsed quantities, every denominator, clean/damaged and matched-pair populations, failures by stage and error type, event/wake/RTB/death outcomes, reward headroom, productive-update yield and artifact completeness recorded in §3h, with the authoritative record, the Phase-A conclusion, its explicit NON-CLAIMS and the evidence hashes in `CLAUDE.md` §7 and the phase state in §8. **Validity was judged BEFORE performance and PASSED on all five clauses. This CLOSES PHASE A** |
| FD-VARIABLE-SEVERITY-v1 CODE lands — **DONE for PR #27** | Legacy preservation, the independent severity RNG domain, the two live mild/severe bands, the live-midpoint target policy, the matched clean/mild/severe TRIAD evaluation, the durable `episode_outcomes.jsonl` stream and the scheduled-vs-executed CELL measurement-integrity abort recorded as contracts in `CLAUDE.md` §5, routed in §6, locked in §7 with the append-only fix chain and the identical-tree / zero-diff integration proof; §2's locked-layer wording widened to both designs; §8's research ordering corrected to put the additional actor-only baseline BEFORE Phase-B CTDE. §3i summarizes it here — **recorded without pre-claiming any result, and `p(destroy) < 1` explicitly NOT implemented**. **HISTORICAL NOTE: that SERIAL ordering was itself SUPERSEDED to PARALLEL on 2026-08-22 — see the row below** |
| CTDE PARALLEL-ORDER / OWNERSHIP SUPERSESSION recorded — **DONE for 2026-08-22** | Record, in `CLAUDE.md` §8 and here (§1, §4, §5, §6, §8), that the serial "measure first, then CTDE" order is SUPERSEDED; that the FD-VARIABLE-SEVERITY-v1 measurement is PINNED to the immutable detached snapshot `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` / tree `dd881478b8e2e521054d09bc865437f1308be1a2` and remains an ACTOR-ONLY measurement OF THAT SHA; that Phase-B CTDE DESIGN AND IMPLEMENTATION may proceed CONCURRENTLY in a separate writable branch; that CTDE INTEGRATION into `main` stays gated on that measurement's INDEPENDENT VALIDITY VERDICT, preceded by a NEW immutable actor-only pre-CTDE reference chosen later; and that the CTDE GPT orchestrator is the SOLE WRITABLE repository owner while the FD orchestrator stays READ-ONLY. **Sequencing and ownership only — NO technical CTDE contract was defined, and no result was pre-claimed.** **HISTORICAL: that measurement has since COMPLETED and received its independent validity verdict, and writable ownership was temporarily and exceptionally lent to the FD orchestrator for the closure record — see the two rows below** |
| THE ADDITIONAL ACTOR-ONLY VARIABLE-SEVERITY BASELINE completes — **DONE at measured code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`; `APPROVE — VALID MEASUREMENT`; PRIMARY FINDING NEGATIVE** | Run identity, exact measured code SHA, resolved configuration and its one invocation, complete provenance, both elapsed quantities, EVERY denominator (attempted / successful / failed, per cell and overall), the failure breakdown by stage and error type, the clean / mild / severe reward means, the three matched deltas over COMPLETE TRIADS ONLY, the severity-conditioned FD-wake meta-action counts and rates WITH their wake denominators, RTB / death / target-coverage outcomes, artifact completeness and the evidence hashes recorded in `CLAUDE.md` §7, with the phase state in §8 and the volatile summary in §3j. The EXCLUDED `MAX_PATH` precursor is recorded as `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT` with its own hashes, and the playback-export routing caveat is recorded as engineering history rather than fixed. **Validity was judged BEFORE performance and PASSED on all four clauses; the NEGATIVE severity response is a valid observation, not a defect** |
| OWNERSHIP RETURNS to the CTDE GPT orchestrator — **EFFECTIVE UPON INTEGRATION of the record above (2026-08-23)** | Record, here (§1, §4, §8) and in `CLAUDE.md` §8, that the user granted the FD measurement orchestrator a ONE-TIME writable exception scoped to the variable-severity closure record ALONE; that PR #30 / `task/phase-b-ctde-build` stayed untouched throughout; that **UPON INTEGRATION of that record into `main`, sole writable repository ownership RETURNS to the CTDE GPT orchestrator** and the FD orchestrator reverts to READ-ONLY with no writable branch or PR; and that the CTDE orchestrator's immediate repository-side prerequisite before any CTDE integration is to preserve the **NEW immutable actor-only pre-CTDE reference** in its own separately reviewed task. **Ownership and sequencing only — no technical CTDE contract is defined, no pre-CTDE reference is chosen or created, `phase-a-baseline` does not move, and the integrating merge SHA is deliberately not named** |
| PHASE-B CTDE IMPLEMENTATION lands — **DONE for PR #30** | The two training modes and `TrainConfig.training_mode` as the ONLY selector, actor-only preservation (keyword omission, the POISON test and its CONTROL), the training-only critic and its structural separation from the actor, the central graph's liveness / no-ego symmetry / features / edges and the ENUMERATED privileged-input EXCLUSIONS, the same-tick capture seam and its 1:1 alignment, GAE over the GLOBAL decision sequence with a zero terminal next value, the preserved meaning of `baseline` / `train_reward_mean`, the persisted critic diagnostics and the checkpoint distinction recorded as a contract in `CLAUDE.md` §5, routed in §6, and locked in §7 with the append-only fix chain, the exact six changed files, the two DISTINCT immutable references (`pre-ctde-actor-only` vs `phase-a-baseline`) and the CC-reported engineering evidence LABELLED as implementation validation. §2's locked-layer wording widened to include the CTDE layer; §4's pipeline diagram, §5's encoder note and §8's gate updated so nothing still says the critic is open or that there is no value head. §3k summarizes it here — **recorded WITHOUT pre-claiming any result: no scientific run occurred for PR #30 and no actor-only vs CTDE comparison has been executed** |
| PHASE-B CTDE DOCUMENTATION / LOCK lands — **DONE for PR #32** | Approved candidate `c607f3fabcbd58f6f10cfde6bcc34068f09e4121`, verdict `APPROVE`, integrated by normal merge `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96` with ordered parents `8390d85c2072e9cbe984ce5f2731cef3a9b14985` then `c607f3f…`, recorded in §1 with the CTDE technical contract, routing and lock living in `CLAUDE.md` §5–§7 and summarized here in §3k. **No code, test, config or preset changed; no run occurred; no CTDE result was pre-claimed** |
| CHAT / REPOSITORY CLOSURE recorded — **DONE (2026-08-23)** | Record, here (§1, §4, §7, §8), that PR #32 / `task/phase-b-ctde-doc-lock` is CLOSED / MERGED and is no longer the current writable task; that `main` = `7b6c07586811374f3f35e26ed33e1fcf4a9f2e96` / tree `8a0b7a0aa9e65ebf01fce99c9b27ee25886ba7a6` is this record's base and its historical closure provenance; that this closure branch is the SOLE writable candidate only while in flight and that repository state is CLOSED / IDLE upon its integration; the FOUR preserved reference roles (`main`, `phase-a-baseline`, `pre-ctde-actor-only`, `flat-final`, plus the `pre-cleanup` tag) and their NON-interchangeability; the four `task/*` branches as RETIRED and cleanup-eligible rather than active; that the next USER activity is a READ-ONLY interactive walkthrough; and that the next repository/scientific task remains Task 9, NOT started. **Volatile state ONLY — `CLAUDE.md` is untouched, no technical contract changes, no CTDE benefit is claimed, no actor-only rerun is authorized, `p(destroy) < 1` / SAMs / dense reward / solver and reward changes / new difficulty factors stay outside Task 9, and this record's own integrating merge SHA is deliberately NOT named** |
| The FIRST actor-only vs CTDE COMPARISON on the OLD FIXED CELL completes — **NO LONGER THE LIVE NEXT ACTION (§4, Task 9); an OLD-CONTRACT CTDE run WAS executed and is OUT OF SCOPE / NOT REVIEWED HERE (§1)** | Record run identity, exact measured code SHA, the resolved configuration, complete provenance, every denominator, the held-out results with their own denominators, artifact completeness and the evidence hashes in `CLAUDE.md` §7, with the phase state in §8. **The comparator is the ALREADY-APPROVED original Phase-A actor-only baseline (`737b4bf…`, §3h), which is NOT re-run** — so record the CTDE arm's own measured SHA and state EXPLICITLY that the two records carry DISTINCT measured code SHAs; never claim the arms differ only by one `training_mode` field. The factor under test is actor-only training vs centralized-critic training, with the CTDE run matching the approved baseline's locked cell, schedule, seed policy, held-out band and evaluation construct. A newly executed actor-only CONTROL arm would be a separate research-design decision needing explicit user authorization. **Validity is judged BEFORE performance; a null or negative CTDE result is a valid observation, not a technical failure** |
| GENERALIZED-V1 HANDOFF BOOTSTRAP recorded — **DONE (2026-08-25)** | Record, here (§1, §3l, §4, §6, §7, §8, §9), that the repository is NO LONGER CLOSED / IDLE; that the GENERALIZED TRAINING / BENCHMARK REDESIGN is the ACTIVE research/design phase owned by the GPT orchestrator; that NO implementation candidate is active, NOTHING generalized is implemented, and NO new scientific measurement is running or authorized; that this handoff-bootstrap task is the only writable repository task while its candidate is in flight; the APPROVED DESIGN in a dedicated §3l section — generalized cardinality with bounded B2 backoff, B2/reproducibility direction, the FD redesign, repeated post-FD decision semantics, the event-conditioned continuation reference, the 18-stratum benchmark, the diagnostics and the six-step planned sequence — **every part of it MARKED NOT YET IMPLEMENTED**; that a previously executed OLD-CONTRACT CTDE measurement is OUT OF SCOPE and not to be reviewed or compared unless the user explicitly asks; and the dated decision log plus the transfer/initialization protocol in §9. **Volatile state and PLAN ONLY — `CLAUDE.md` is untouched, no technical contract is defined, changed or locked, no code/test/config/preset changes, no run occurs, no generalized behaviour is claimed to exist, the approved historical measurements are not reinterpreted, and no ref is moved or deleted** |
| GENERALIZED-V1 Task 1 lands (§3l.8 step 1) — **DONE for PR #35** | The two explicit hidden-CARDINALITY policies, the PRESERVED `exact_v1` default (geometry, ego order, draw order and rng stream position pinned), the generalized cell `A ∈ {2,3,4}` / `K == A` raw known / `1 ≤ H_requested ≤ A`, the deterministic ordinal-driven bounded walk with per-candidate substreams derived up front, the shared single-route geometry site, `H_realized ≥ 1` acceptance with short realization RECORDED and never repaired, the verified requested-vs-realized `ConstructionAudit` taken from the RAW world snapshots, and the fact that nothing reaches `GraphObservation` — recorded as a contract in `CLAUDE.md` §5, routed in §6, locked in §7 with the exact four changed files, both proof-obligation sets and the CC-reported engineering evidence LABELLED as implementation validation. **Recorded WITHOUT pre-claiming any result: no scientific run occurred and no generalized measurement exists.** Also record that NEITHER harness exposes the policy *(TRUE when PR #35 landed and SUPERSEDED by PR #40, which added the `episode_design` selector)* |
| GENERALIZED-V1 Task 2 lands (§3l.8 step 2) — **DONE for PR #36** | The two opt-in FD policy seams with LEGACY defaults preserved, the separate `fuel_damage_eligibility_v1` RNG domain, ordinal-driven certified eligibility running for EVERY condition including CLEAN, the tick-aware certificate with its one-tick tolerance bracket and pre-event route-stability certification, `no_fd_eligible_ego` as ORDINARY ACCOUNTED ATTRITION versus `FuelDamageIntegrityError` as an INSTRUMENT abort (live contradiction AND the terminal never-fired check at the single `run_episode` exit seam, before the recording export), the certified-CLEAN non-fire allowance, unchanged LEGACY non-fire semantics, the ONE shared executor confirmation site, damaged-ego-only post-FD adaptation with local reconciliation before the CTDE capture, the append-only `POST_FD_COMPLETION` trigger with NO new `MetaAction`, and the SEPARATE boundary diagnostics — recorded as a contract in `CLAUDE.md` §5, routed in §6 (including the measurement-integrity routing row), locked in §7 with the exact nine changed files and the append-only REQUEST-FIXES → approved-child chain. **The bounded real-BLADE / BONMIN smoke is LABELLED engineering validation, never a measurement; no generalized result is pre-claimed.** Also record that NEITHER harness exposes either policy and that nothing persists the new diagnostics *(both TRUE when PR #36 landed and SUPERSEDED by PR #40, which added the `episode_design` selector and full run-level persistence)* |
| GENERALIZED-V1 Task 3 lands (§3l.8 step 3) — **DONE for PR #38** | The ONE OPT-IN reward-reference policy beside the PRESERVED `static_t0_v1` default, `EpisodeContext.reference_policy` as the single stored source with `uses_event_conditioned_reference` as the canonical runtime predicate and an unknown id REFUSED before any BLADE object exists, the deferred second solve and its three sites (CLEAN t=0 before the first tick; the DAMAGED continuation reference at the top of the firing tick immediately after the real `current_fuel` mutation and before boundary / trigger / `central.capture` / decision / `env.step`; the legacy never-fired t=0 build at the episode-exit seam before the recording export), the checkpoint as READ-ONLY privileged measurement that advances no simulation time and reaches no acting path, the continuation universe as the retained RAW t=0 world minus the realized prefix and NEVER a private belief, continuation agents rebuilt from the LIVE post-event world with dead / RTB-committed / non-airborne egos EXCLUDED by recorded reason, the `U_ref = U_prefix + U_cont_ref` / `U_achieved = U_prefix + U_post` arithmetic with `U_prefix` FROZEN at the checkpoint, `U_post` restricted to continuation-allocated tasks, out-of-reference kills ACCOUNTING-ONLY, `U_aircraft` from the reward-bearing reference universe, NO clamping, `u_oracle is None`, and terminal-on-last credit placement UNCHANGED, the audited solve seam distinguishing an unanswered solve from an answered zero with the historical public triple byte-unchanged, `ReferenceIntegrityError` and its CURRENT ordinary-episode-failure routing, and the typed `EpisodeReference` on `EpisodeResult.reference` — recorded as a contract in `CLAUDE.md` §5, placed in §4, routed in §6 and locked in §7 with the exact five changed files and the identical-tree integration proof. **The review-APPROVED `damaged_event_unrealized_t0` compatibility deviation is recorded EXPLICITLY and scoped as a LEGACY preservation, never as generalized damaged semantics.** The at-most-two-BONMIN / never-a-third statement is recorded in its accurate form, including the degenerate `solver_invoked=False` skip. **The bounded real-BLADE / BONMIN smoke is LABELLED engineering validation, never a measurement; no generalized result is pre-claimed.** Also record that NEITHER harness exposes the policy and that nothing persists `EpisodeReference` *(both TRUE when PR #38 landed and SUPERSEDED by PR #40, which added the `episode_design` selector and full run-level persistence)* |
| GENERALIZED-V1 Task 4 lands (§3l.8 step 4) — **DONE for PR #40** | The ONE `episode_design` selector resolving the COMPLETE approved bundle through a single site with `fixed_cell_v1` as the PRESERVED DEFAULT and NO per-policy harness field, `training_mode` as an ORTHOGONAL and unchanged training-algorithm selector, the generalized TRAINING cardinality sampler (`A ~ U{2,3,4}`, `K == A`, `H_requested | A ~ U{1..A}`) on its OWN SHA-256 rng domain disjoint from the three fuel-damage domains / the placement rng / global `random` / torch, `H_requested` NEVER rewritten and short realization RECORDED rather than retried or replaced, the 18-stratum matched CLEAN/MILD/SEVERE benchmark MECHANISM with its canonical content-addressed manifest whose loader authenticates the EXACT STORED payload AND independently requires it to equal the canonical payload (so a tampered manifest AND a self-consistently rehashed noncanonical forgery are both REFUSED), id-free world identity with no generated-uuid equality anywhere, matched-group and frozen-preflight identity VERIFIED with `BenchmarkIdentityError` routed as a measurement-integrity ABORT, deltas over COMPLETE three-member groups ONLY with no retry or substitution, the manifest's ACTUAL seeds checked against the training band at LOAD time before the run directory or any scientific compute, provenance naming the manifest as the real eval seed source with the configured band retained and marked UNUSED, UNCHANGED historical fixed-cell eval-band semantics, the reason-based `ReferenceIntegrityError` routing (an unanswered solve = accounted attrition; every other reason = measurement-integrity ABORT, read from the SLUG and never the message) with the reward arithmetic itself unchanged, run-level PERSISTENCE of the construction audit / FD eligibility audit and certificate / post-FD adaptation / reference decomposition and continuation-solver audit / scored-vs-unscored completion / aircraft-loss and selected-ego RTB / benchmark stratum and world identity, and generalized aggregates with EXPLICIT denominators — recorded as a contract in `CLAUDE.md` §5, placed in §4, routed in §6 and locked in §7 with the exact eight changed files, the append-only fix chain and the identical-tree integration proof. **Record that the generalized `run_config.json` cardinality is DYNAMIC and that the old 3/3/3 count fields are explicitly marked UNUSED, with realized per-episode cardinality living in `episode_outcomes.jsonl`, while fixed-cell construction and provenance remain historical and byte-unchanged.** **Record that requested-vs-realized HIGH-load behaviour is REPORTED for inspection and that NO automatic acceptance threshold is invented — the judgement is a human / GPT scientific review decision.** **Record that Task 4 delivered the manifest SCHEMA / BUILDER / LOADER / CONSUMER / FREEZE MECHANISM and did NOT choose the final scientific world count and did NOT commit or generate a final benchmark population — that is step 5's business — and that NO generalized measurement and NO actor-only-vs-CTDE generalized result exists or may be pre-claimed** |
| GENERALIZED-V1 Task 5 lands (§3l.8 step 5) — **DONE, AND NOW INTEGRATED: PR #42 → `5dfcd8b632be8dca3c1730018bbf35337d07f077`, PR #43 → `b3c2e01f130afe854b09384cd6e1e196de714795`** *(this row's original "BOTH APPROVED AND FROZEN, NEITHER MERGED" status, and its closing "NO integration SHA exists for either PR and none may be invented", were accurate when written and are SUPERSEDED by those merge SHAs)* | The `train_by_*` summary buckets counting TRAINING attempts only (a persisted-summary correction that changes no episode behaviour and leaves the canonical streams byte-unchanged, with training FAILURES still represented and every other generalized block keeping its own population); the successful-episode training QUOTA under `generalized_v1` with `episodes_per_iteration` counting SUCCESSFUL episodes, the REQUIRED and never-defaulted `generalized_max_attempts_per_iteration`, ONE run-wide monotone attempt ordinal whose seed is `train_attempt_seed`, an ordinary failure recorded once / spending its seed / never retried / never entering the PPO-CTDE buffer / replaced by the next deterministic attempt, `TrainingQuotaError` on exhaustion with NO partial update, the PRESERVED `scheduled_attempts_v1` fixed-cell behaviour, and the unchanged actor-vs-CTDE execution semantics; held-outness checked against the MAXIMUM POSSIBLE attempt band `[base_seed, base_seed + n_iterations * generalized_max_attempts_per_iteration)` because a failed replacement still spends a seed; and the deterministic benchmark PREFLIGHT — explicit `worlds_per_cell` / `benchmark_base_seed` / `max_candidates_per_cell` with NO scientific-scale default, six INDEPENDENT deterministic base-cell candidate windows, first valid candidates accepted in ascending order with each rejected seed spent exactly once, ordinary world-construction / certified-FD-ineligibility rejections replaceable BEFORE the freeze, accepted bounded-backoff worlds NOT rejected solely for `hidden_realized < hidden_requested` (short realization is AUDIT DATA, never an automatic failure), integrity faults PROPAGATING rather than becoming population-selection attrition, NO policy built and NO episode run so no reward or learned behaviour can influence acceptance, the COMPLETE-manifest rule, IMMUTABLE post-freeze evaluation with NO substitution in `evaluate_benchmark`, and the FAILED-preflight durable audit (`status = failed_incomplete`, `complete = false`, `manifest_written = false`, `manifest = null`, the failure block naming the exhausted cell / window / requested-accepted-missing counts, attempted seeds and rejection tallies SURVIVING, completed cells preserved, unattempted cells NAMED, the report written BEFORE the raise when an output directory exists, `BenchmarkPreflightError.report` / `.report_path` when there is none, and a pre-existing manifest NAMED via `stale_manifest_path` rather than deleted or adopted) — recorded as a contract in `CLAUDE.md` §5, routed in §6 and locked in §7 with the exact changed files and PR #43's append-only review-fix chain. **Record Task 5A and Task 5B as `APPROVE — VALID ENGINEERING VALIDATION` under a BINDING label — engineering evidence, never a measurement — with their bounded sample sizes stated as an explicit limitation and NO attrition-rate population claim, NO learning claim and NO actor-vs-CTDE claim.** **Record the dispatched actor-only R1 as `AUTHORIZED / DISPATCHED — RESULT PENDING` with its frozen plan and NOTHING about its outcome; record that it needs independent GPT artifact review before any `APPROVE — VALID MEASUREMENT`; and record that NO integration SHA exists for either PR and none may be invented** |
| The GENERALIZED-V1 TASK-5 STACK is INTEGRATED (§3m.5) — **DONE (2026-08-30/31)** *(this row previously read "NOT DONE; NO MERGE IS AUTHORIZED BY ANY DOCUMENTATION RECORD"; the merges were separately authorized and have since been performed)* | The nine-step sequence was PERFORMED exactly as planned: merged PR #42 (→ `5dfcd8b632be8dca3c1730018bbf35337d07f077`), refreshed the exact live `main`, RETARGETED PR #43 to `main` and **exact-base re-reviewed it before merging** — changing a PR's base invalidates a base-relative verdict even though the candidate SHA is unchanged — then merged the unchanged approved head (→ `b3c2e01f130afe854b09384cd6e1e196de714795`), refreshed `main`, retargeted and re-reviewed the documentation candidate, and merged it unchanged (→ `9b9e9b85a70c8a0019c72ada92ceec3401725795`, PR #44). **No rebase, no squash, no cherry-pick, no force-push, no history rewrite** occurred; every integration is a normal merge commit preserving its reviewed commit as an ancestor / merge parent, with the integrated tree verified equal to the reviewed tree. Each integration SHA is recorded here, per §7's hash convention, by the first record able to name it |
| The DISPATCHED GENERALIZED ACTOR-ONLY R1 completes — **DISPATCHED; RESULT PENDING; UNREVIEWED** | Record run identity, exact measured code SHA, resolved configuration, complete provenance, every denominator per stratum, realized-cardinality accounting, the within-world matched deltas over COMPLETE matched groups only, FD-wake meta-action responses over FD-WAKE denominators, artifact completeness and the evidence hashes in `CLAUDE.md` §7, with the phase state in §8. **Nothing may be recorded before independent GPT artifact review, and no reward, convergence or validity claim may be pre-stated. VALIDITY IS JUDGED BEFORE PERFORMANCE; a null or negative result is a valid observation, not a technical failure and not grounds to re-tune, re-seed or re-run.** The requested-vs-realized hidden-cardinality inspection stays a HUMAN / GPT decision — the code reports the distribution and applies no threshold |
| A future EARLY-STOPPING mechanism is proposed — **NOT IMPLEMENTED, NOT REVIEWED, NOT AUTHORIZED** | It is a SEPARATE research / design decision with its own proof obligations. **It must not select against the same final benchmark without an explicit validation / test design** — stopping on the comparator makes the comparator part of the training signal. Checkpoint RESUME remains out of scope; `graph_train` is SAVE-only. R1 uses a fixed 3000-success budget |
| CLUSTER campaign readiness — **DEFERRED; cluster access is not available** | Not a blocker for the local actor-only R1. **Invent no scheduler, queue, partition or runbook detail**; record it only once real access and a reviewed design exist |
| *(HISTORICAL, superseded by the Task-5 row above)* The REMAINING GENERALIZED step lands (§3l.8 step 5) — **was NOT STARTED and NOT AUTHORIZED when written** | A separately scoped, separately reviewed bounded task, beginning only after the previous is reviewed and integrated. Record the reviewed contract, its routing and its lock in `CLAUDE.md` §5–§7 **after reviewed behaviour exists** — never in advance and never for a design. *(SUPERSEDED: this row previously said the records wait for step 6 alone, and later covered steps 4–5; practice is one documentation pass per completed task, step 4 is DONE, and a FINAL pass is still required after any later step lands.)* Departures from currently locked contracts (B2 exact cardinality, the fixed 3/3/3 cell, FD eligibility and failure policy, the damaged-episode t=0 reference solve, the fixed held-out eval band — **all five already addressed by Tasks 1–4 as their own reviewed OPT-IN seams**) are **Grade-A changes routed through `CLAUDE.md` §6**, each with its own proof obligations, and the historical paths are PRESERVED beside the generalized ones. **Requested-vs-realized cardinality, backoff reasons, FD eligibility candidates and rejections, post-FD wake counts and continuation-solver accounting are already first-class observables (§3l.7) — record any change to them without pre-claiming any result** |
| The FROZEN GENERALIZED BENCHMARK POPULATION is built — **NOT STARTED; the MECHANISM exists since PR #40, the POPULATION and its SCALE do NOT** | Record the 18 requested strata (`A` ∈ {2,3,4} × hidden load LOW=1 / HIGH=`A` × CLEAN/MILD/SEVERE), the matched-world construction (same world, hidden geometry, initial allocation, eligible damaged ego and event point; only the damage condition differs), the world manifests and their identities, and the REQUESTED-vs-REALIZED hidden-count distributions. **Inspect those distributions and REJECT or REDESIGN the benchmark if the HIGH load systematically degenerates — BEFORE any scientific measurement.** Future actor-only and CTDE generalized measurements must use the EXACT SAME frozen manifests; **historical fixed-cell measurements are NOT this benchmark and are not its comparator**. **Task 4 delivered the schema, builder, canonical serialization, content hash, verifying loader, consumer and identity checks; it deliberately did NOT choose the worlds-per-cell scale and did NOT generate, commit or freeze a population, and its builder REFUSES to invent a world count. Choosing the scale comes AFTER §3l.8 step 5's bounded runtime / solver validation.** |
| The FIRST GENERALIZED scientific measurement completes — **NOT RUN, NOT AUTHORIZED** | Record run identity, exact measured code SHA, resolved configuration, complete provenance, every denominator per stratum, realized-cardinality accounting, the within-world matched deltas over COMPLETE matched groups only, FD-wake meta-action responses over FD-WAKE denominators, artifact completeness and the evidence hashes in `CLAUDE.md` §7, with the phase state in §8. **Validity is judged BEFORE performance; a null or negative generalized result is a valid observation, not a technical failure** |

## 8. Next action

**THE LIVE STATE IS ONE OPEN THREAD, AND IT IS NOT AN IMPLEMENTATION TASK: INDEPENDENT GPT
ARTIFACT REVIEW OF THE DISPATCHED ACTOR-ONLY R1 LONG RUN, ONCE ITS ARTIFACTS EXIST**
(§3m.4). *(SUPERSEDED: this section previously read "TWO OPEN THREADS", the first being
INTEGRATION of the approved, frozen, still-unmerged Task-5 stack under §3m.5. That
integration has since been PERFORMED — PR #42 → `5dfcd8b6…`, PR #43 → `b3c2e01f…`,
PR #44 → `9b9e9b85…` — so only thread (B) remains.)*
The live phase is
GENERALIZED-V1 (§3l, §3m, §4 Task 10); **steps 1, 2, 3, 4 and 5 are ALL COMPLETE, REVIEWED
and INTEGRATED** (PR #35, `5b55ca34…` → `9b305e4e…`; PR #36, `185d39f0…` → `ca0dc406…`;
PR #38, `24a8b1ee…` → `df3abf2f…`, **APPROVE**; PR #40, `db790138…` → `b4daa8c1…`,
**APPROVE**; PR #42, `312f5865…` → `5dfcd8b6…`; PR #43, `4af6c5aa…` → `b3c2e01f…`; PR #44,
`88352b2f…` → `9b9e9b85…`) — live `main` at this record's base is
`9b9e9b85a70c8a0019c72ada92ceec3401725795`. **This post-integration closure candidate is the
SOLE WRITABLE REPOSITORY TASK while its own draft PR is open, and on its integration the
repository returns to a clean checkpoint with NO writable repository task, NO open Task-5 PR
and NO active Task-5 candidate** — while **GENERALIZED-V1 remains an ACTIVE phase, because
R1 is pending.** The
external long-run task is **RUN-ONLY and owns NO repository writes**, and the GPT orchestrator
owns the work.

**NO MERGE IS AUTHORIZED BY THIS RECORD.** **UNTIL R1's ARTIFACTS ARE INDEPENDENTLY REVIEWED
THERE IS NO RERUN, NO REPAIR, NO RESUME, NO EXTENSION, NO CTDE ARM, NO BENCHMARK REPLACEMENT
AND NO RETUNING** — each would be a separate research decision requiring its own explicit
authorization. **R1 remains `AUTHORIZED / DISPATCHED — RESULT PENDING`: it is neither
`RUNNING`, nor `COMPLETED`, nor `VALID`, and elapsed time implies nothing about it.**

**THE DISPATCHED R1 IS `AUTHORIZED / DISPATCHED — RESULT PENDING`: unreviewed, with no verdict,
and NOTHING about its reward, convergence, attrition, benchmark outcome or scientific validity
may be stated or inferred.** It requires independent GPT artifact review before any
**`APPROVE — VALID MEASUREMENT`** verdict, judged under the unchanged gate — **VALIDITY BEFORE
PERFORMANCE** — and **a null or negative generalized result is a valid observation, not a
technical failure and not grounds to re-tune, re-seed or re-run.** **Task 5A and Task 5B are
ENGINEERING VALIDATION ONLY and establish nothing scientific** (§3m.3). **No CTDE generalized
run exists, is scheduled or is authorized**, **no reviewed early-stopping mechanism exists**
(R1 uses a fixed 3000-success budget; checkpoint RESUME is out of scope and `graph_train` is
SAVE-only), and **cluster readiness is DEFERRED** for want of access — which does not block
the local R1 (§3m.4).

No other task is next: not the old fixed-cell Task-9 CTDE comparison (superseded in ordering,
§4), not a review of the previously executed old-contract CTDE measurement (out of scope
unless the user explicitly asks, §1), and not a re-run of any approved historical baseline
(prohibited, §6 — the approved baselines are REUSED as what they are, never rerun, repaired,
resumed or re-tuned).
**THE SCALE / AUTHORIZATION / RESULT STATE IS FOUR DISTINCT FACTS:** **the R1 benchmark SCALE is SELECTED and AUTHORIZED (`worlds_per_cell = 3`) and its
CONSTRUCTION is AUTHORIZED and DISPATCHED (candidate base seed `840000`,
`max_candidates_per_cell = 12`); no concrete R1 manifest has yet been independently reviewed
or approved as the comparator, and none is committed or tracked in the repository; and no
GENERALIZED scientific measurement RESULT exists** — no reward,
convergence or validity result, and **no actor-only-vs-CTDE generalized result.** (No
benchmark manifest is committed or tracked in the repository; transient manifests built by
tests and engineering validation are neither committed nor a comparator, and nothing here
claims a global negative over local scratch files.) **No CTDE generalized campaign is
authorized, none is running or scheduled, and no generalized result — including any
actor-only-vs-CTDE comparison — may be pre-claimed.** *(SUPERSEDED, and corrected here: this
paragraph previously asserted that no FINAL SCIENTIFIC worlds-per-cell scale had been
SELECTED and that no benchmark population had been scheduled or authorized. Accurate through
Task 4; not now.)* When those two arms are eventually run they **MUST use the SAME eventual
frozen manifest.** Steps 1–5 delivered implemented CODE with engineering evidence only, and
`p(destroy) < 1` remains DEFERRED.
*(SUPERSEDED: this section previously named step 1, then step 3, then step 4, then step 5 as
the single next action, and stated that Task 5 was NOT started and NOT authorized and that no
generalized run was scheduled or authorized. Each was accurate when written; steps 1–4 have
since been implemented, reviewed and integrated, step 5 is implemented and approved but not
merged, and the actor-only R1 has been authorized and dispatched with its result pending.
What is UNCHANGED: no concrete R1 manifest has yet been independently reviewed or approved as
the comparator, none is committed or tracked in the repository, and no generalized measurement
RESULT exists. What has CHANGED is that the R1 scale IS selected and its construction IS
authorized and dispatched.)*

**THE REST OF THIS SECTION IS THE PRESERVED RECORD OF THE CLOSED PHASES**, accurate about
them and unchanged in meaning.

Implementation for the Phase-A baseline cell is COMPLETE and locked, its inspection surface
is merged, repository hygiene is CLOSED (PR #11 code, PR #12 documentation), the **probe
harness is CLOSED** (PR #14), and the **FD-VARIABLE-SEVERITY-v1 research factor is CLOSED /
APPROVED / MERGED** (PR #27, §3i). **FOUR defects were found, corrected, approved and
merged** — **Defect A, ego-global `SELF_PRESERVATION_ABORT`** (approved `d56fda6`,
integrated `f094e0b`, PR #17), **Defect B, the attack-confirmation wait derived from the
salvo about to fly** (approved `39a16f2`, integrated `60a82d1`, PR #19), **Defect C,
physical RTB completion** (approved `ea62e4e`, integrated `0de9f21`, PR #21) and **the
FOURTH, SEPARATE roster / world-truth defect — allocation read as world inventory**
(approved `36365f2`, integrated `f37ea1c`, PR #24, §3g). Those corrections were integrated
as `main` = `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` (PR #25), which is **the exact code
SHA the Phase-A baseline was measured at** — a durable MEASUREMENT identity, **not** a claim
about live `main`. **The variable-severity baseline has its OWN durable measured SHA**,
`bf1e045f90f74361e4ee944f7bd683a3ea72d04b` / tree
`dd881478b8e2e521054d09bc865437f1308be1a2`, the variable-severity documentation merge
(PR #28) — also not a live head (§1, §3j). **This record's BASE is a THIRD SHA:**
`76abdc480e80a84f1503208730d4525cd5e89b69`, tree
`237325d2c2a41950eab103a8b08c9442e5c9fa97` — the **chat / repository CLOSURE merge
(PR #33)**, which is an already-existing DOCUMENTATION INTEGRATION and not a measurement
identity. *(The PREVIOUS record's base was `8390d85c2072e9cbe984ce5f2731cef3a9b14985`, the
Phase-B CTDE code merge, PR #30; that is historical provenance now.)* **Resolve the live
full `main` SHA from GitHub** (§9).

**THE MEASUREMENT STATE, STATED PLAINLY. PHASE A IS CLOSED, AND ITS BASELINE IS
IMMUTABLE.** The authorized long-baseline rerun —
`training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` at exact code SHA
`737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`, ONE invocation, native exit 0,
`cli_overrides = []` — was independently reviewed **`APPROVE — VALID MEASUREMENT`** and is
**the FIRST scientifically valid measurement of the fuel-damage cell** (§3h). It passed the
validity gate on all clauses BEFORE any performance was read. **It is the ONLY run whose
reward, paired delta, survival, fuel-damage yield and PPO performance are scientific
evidence about this cell**, it is a measurement of the **LEGACY FD-BASELINE-v1** design, and
the branch **`phase-a-baseline` = `4f0068847b017795717c5f0e331f647bcfc30547`** preserves its
code state and must not move. The three earlier runs remain preserved and are HISTORY ONLY:
the first short probe (§3d), the corrected short-probe rerun (§3e, verdict SUPERSEDED) and
the first long baseline (§3f, `INCONCLUSIVE`). Their mechanical accounting reconciled in
every case, which is precisely the lesson: **a self-consistent ledger over a population an
instrument defect shrank is not a measurement.**

**What Phase A establishes — and what it does not.** It establishes end-to-end learnability
and meaningful ego-local runtime adaptation in the locked reference cell: a matched paired
penalty of `−0.375000 → −0.071429` over a structural 7/8 pairs across 100 productive PPO
updates, evaluation deaths `7 → 0`, and a deterministic shift from `PLAN_COMPLIANCE` to
`SELF_PRESERVATION_ABORT` in all seven completed damaged held-out worlds, corroborated by
physical playback. It establishes **NO global optimality, NO monotonic convergence, NO
generalization beyond this fixed cell and this held-out seed set, and NO benefit from
centralized training.** Those non-claims are binding on every downstream document, and
**none of these numbers is an expectation for the variable-severity cell.**

**THE PARALLEL PHASE HAS DELIVERED ITS MEASUREMENT: THE ADDITIONAL ACTOR-ONLY
VARIABLE-SEVERITY BASELINE IS CLOSED, VALID AND NEGATIVE.** The FD-VARIABLE-SEVERITY-v1
CODE is merged and locked (approved `eecc9b5…`, integrated `177e969…`, identical tree
`37ebd8c…`, zero changed files candidate→integration, PR #27), and its actor-only baseline
has now been **EXECUTED ONCE from the immutable detached snapshot at exact SHA
`bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, independently reviewed, and approved
`APPROVE — VALID MEASUREMENT`** (§3j; `CLAUDE.md` §7 owns the authoritative record). **THE
PRIMARY BEHAVIOURAL FINDING IS NEGATIVE**: 664 scheduled / 586 successful / 78 accounted
`setup` episode failures, `accounting_reconciled = true`, 7/8 complete matched triads in all
11 rounds, zero infrastructure or data-integrity faults and 50/50 productive PPO updates —
and yet the deterministic held-out actor chose `PLAN_COMPLIANCE` in all 7 completed MILD and
all 7 completed SEVERE worlds at `pre_update` AND at the final `post_update`, with identical
distributions across all ten `post_update` rounds. **That is a valid negative scientific
result**, and the severity factor is nevertheless physically real (RTB yield, deaths and
coverage diverge sharply). An earlier attempt at the same contract is
`INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT` — a Windows `MAX_PATH` playback-export failure
removed the entire `post_update` SEVERE arm — and is excluded from every scientific reading;
both trees are PRESERVED. The parallel arrangement was a DELIBERATE user/orchestrator
decision made on **2026-08-22**, superseding TWO earlier ones (the original "Phase-B CTDE is
next", then the serial "measure first, then design CTDE"), and it did its job: **the pinned
snapshot kept the measurement isolated from concurrent CTDE work, so it is an ACTOR-ONLY
measurement OF THAT SHA and nothing later can be attributed to it.** **CONSEQUENCE FOR
CTDE: the INTEGRATION gate's measurement-validity half is now SATISFIED — by a negative
result, which counts exactly as a positive one would — and its remaining half is not.** It
is not a reopening of Phase A, and it changes no technical CTDE contract. `p(destroy) < 1`
was NOT implemented by PR #27 and remains a separate, later research task.

**PHASE-B CTDE IS IMPLEMENTED, REVIEWED AND MERGED — AND THE INTEGRATION GATE IS CLOSED.**
Approved candidate `a6f3aa9d62931994f416b2241fec4cfac3b018ec`, integrated
`8390d85c2072e9cbe984ce5f2731cef3a9b14985`, ordered parents
`d437084c5fb1a22c21596a48c58e03f7e15a0115` then `a6f3aa9…`, integration tree
`9686c107b8864f00a7d4403d70faf42ab561d2fb`, PR #30 (§3k; `CLAUDE.md` §5 owns the contract,
§6 the routing, §7 the lock). **Both halves of the gate are SATISFIED**: the
measurement-validity half by the variable-severity `APPROVE — VALID MEASUREMENT` verdict —
a NEGATIVE result satisfies it exactly as a positive one would — and the reference half by
**`pre-ctde-actor-only` = `d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the CTDE
integration's FIRST parent and therefore provably the actor-only state CTDE was merged
onto. **That ref must not move**, and **`phase-a-baseline` =
`4f0068847b017795717c5f0e331f647bcfc30547`** remains the SEPARATE original Phase-A
reference, never moved and never repurposed for it.

**WHAT PHASE B NOW HAS, AND WHAT IT STILL DOES NOT.** It has TWO selectable training modes
— `actor_only` (the DEFAULT and the preserved reference path, which builds no critic, no
central observation, no value loss and no CTDE advantage) and `ctde` — chosen by
`TrainConfig.training_mode` alone. EXECUTION stays decentralized in BOTH modes: the actor
reads only its own private `GraphObservation`, and `evaluate` never constructs a critic.
**What it does NOT have is a scientific result. NO ACTOR-ONLY vs CTDE COMPARISON HAS BEEN
RUN, and NO CTDE BENEFIT IS ESTABLISHED OR MAY BE PRE-CLAIMED** — not from the approved
Phase-A result, which explicitly establishes none; not from the executed variable-severity
baseline, whose NEGATIVE severity finding is **NOT** evidence that centralized training
would change it; and not from PR #30's implementation work, engineering tests or passing
suite, none of which measure anything scientific.

**THE FIRST CONTROLLED ACTOR-ONLY vs CTDE COMPARISON ON THE OLD FIXED CELL — PRESERVED,
BUT NO LONGER THE NEXT ACTION** (§4, Task 9; superseded in ORDERING by the generalized
redesign, and note that an OLD-CONTRACT CTDE measurement has since been executed and is OUT
OF SCOPE unless the user explicitly asks, §1). **Its comparator is the ALREADY-APPROVED
original Phase-A actor-only baseline**
(measured code SHA `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`, §3h), which is PRESERVED and
**NOT authorized to be re-run** — so what Task 9 schedules is the **CTDE arm**, matching
that baseline's **LOCKED cell** (3 agents, 3 known + 3 hidden, 200 km / 100 km geometry,
`DETECTION_KM = 50`, `include_sams = false`, `probability = 1`, frozen solver and BLADE,
unchanged `graph_reward` formula with `aircraft_penalty_coeff = 2.25`) **and its training /
evaluation schedule, seed policy, held-out band and evaluation construct**, judged under
the SAME validity gate above, VALIDITY BEFORE PERFORMANCE. **The factor under test is
actor-only training vs centralized-critic training.** Provenance must state that the two
records carry **DISTINCT measured code SHAs**, and must NOT claim the arms' literal
repository or configuration artifacts differ only by one `training_mode` field. **This
document authorizes no fresh actor-only run**; a newly executed actor-only CONTROL arm is a
separate research-design decision requiring explicit user authorization. The comparison
**must NOT bundle `p(destroy) < 1`, SAMs, dense reward, a solver change, a reward-formula
change, or any new difficulty factor** (§6) — bundling one would make it uninterpretable. A
run showing no CTDE improvement is a **valid NEGATIVE observation**, not a technical
failure. **No CTDE preset exists in the repository**, and creating one belongs to that task.

**OWNERSHIP.** The **GPT orchestrator owns the GENERALIZED work** and holds sole writable
repository ownership. **No documentation record is a standing writable task** — each is the
sole writable candidate only while its own draft PR is open, and on integration that branch
and PR become history — so on THIS closure candidate's integration NO writable repository
task remains. **NO implementation candidate is active**, and **the next thing that
may happen is INDEPENDENT GPT ARTIFACT REVIEW of the dispatched actor-only R1 once its
artifacts exist — which no documentation record authorizes.** *(SUPERSEDED: this bullet
previously named "step 4 of §3l.8" as the next thing; steps 4 and 5 have both since been
implemented, reviewed and integrated.)* *(Historical:
`task/generalized-v1-cardinality-b2` closed as PR #35, `task/generalized-v1-fd-adaptation`
as PR #36, `task/generalized-v1-task12-doc-lock` as PR #37 and
`task/generalized-v1-task3-continuation-reference` as PR #38;
`task/generalized-v1-handoff-bootstrap` was the writable task under an EARLIER
record and closed as PR #34; before it, `task/ctde-chat-closure-handoff` closed as PR #33 and
`task/phase-b-ctde-doc-lock` as PR #32; earlier still, the FD measurement orchestrator's
ONE-TIME writable exception, scoped to the variable-severity closure record alone, ENDED when
that record was integrated. The `REPOSITORY CLOSED / IDLE` state that record announced is
SUPERSEDED — the repository is active again, and now in implementation.)*
This record's own integrating merge SHA is deliberately NOT named — it does not exist while
this is written, and inventing it would be a false provenance claim. **GitHub remains
authoritative for live branch and PR state — resolve it there, never from this document.**
Retiring any merged branch is a separate action owned by the GPT orchestrator, done only
once each tip is verified reachable from integrated `main`; the FOUR cleanup-eligible
`task/*` branches enumerated in §1 — plus `task/ctde-chat-closure-handoff` since PR #33
merged — are ELIGIBLE for that bounded follow-up task and were NOT deleted here, while
`pre-ctde-actor-only`, `phase-a-baseline`, `flat-final`, the `pre-cleanup` tag, PR #30's,
PR #32's and PR #33's history and every preserved scientific artifact stay PRESERVED and are
NEVER cleanup-eligible.

**NEITHER the Phase-A long baseline NOR the variable-severity baseline is to be re-run,
resumed, repaired, extended or re-tuned.** A valid measurement exists for each; re-running
would not make either more valid, and a NEGATIVE finding is a result rather than a reason to
try again. The Phase-A scientific contract is frozen as the reference baseline, and the
variable-severity run is an ADDITIONAL measurement of a DIFFERENT cell beside it, never a
replacement for it.

Resolve live branch and PR state from GitHub; this document does not track it.

**THE NEXT REPOSITORY ACTION, RESTATED ONCE AND ONLY ONCE: INDEPENDENT GPT ARTIFACT REVIEW
OF THE DISPATCHED ACTOR-ONLY R1 ONCE ITS ARTIFACTS EXIST — ONLY AFTER GPT EXACT-SHA REVIEW OF
THE CURRENT DOCUMENTATION STATE PLUS USER-AUTHORIZED CONTINUATION. IT IS NOT AN
IMPLEMENTATION TASK.** *(SUPERSEDED: this paragraph previously listed TWO next actions, the
first being INTEGRATION of the still-unmerged Task-5 stack under §3m.5. That integration has
been PERFORMED.)*
Nothing else is next, and **no merge is authorized by any documentation record.**
**§3l.8 steps 1–5 are ALL IMPLEMENTED, REVIEWED and INTEGRATED (PR #35, PR #36, PR #38,
PR #40, PR #42, PR #43 and the documentation lock PR #44)**; **the R1 benchmark SCALE IS selected and its CONSTRUCTION IS authorized and
dispatched, while no concrete R1 manifest has yet been independently reviewed or approved as
the comparator and none is committed or tracked in the repository**; **no generalized
measurement RESULT exists** and the dispatched actor-only R1 is **RESULT PENDING and
UNREVIEWED**; `p(destroy) < 1`, SAMs, dense reward, solver or
reward-formula changes, early stopping and any further difficulty factor remain OUTSIDE it
(§6, §3m.4); and the approved historical measurements are REUSED as what they are — neither
re-run nor reinterpreted. *(SUPERSEDED: this paragraph previously named GENERALIZED
IMPLEMENTATION TASK 1 as the next action and described the §3l design as approved but NOT
IMPLEMENTED, and later named TASK 5 as the next action while stating that no generalized
measurement was running, scheduled or authorized. Each was accurate when written; not now.)*

**This document authorizes neither an implementation nor a training run.**

## 9. Transfer protocol and decision log

### 9.1 How a receiving orchestrator initializes — MANDATORY, IN THIS ORDER

**No SHA in this document is a claim about live `main`.** Every SHA here is either a
historical integration, a durable MEASUREMENT identity, or this record's own derivation
base. Before acting on anything, a receiving orchestrator (GPT or Claude) **MUST**:

1. **RESOLVE THE LIVE FULL `main` SHA FROM GITHUB.** GitHub is authoritative for live branch
   and PR state; this document is not, and neither is a chat summary, a Project Source, a
   memory entry or a pasted narrative.
2. **RE-READ `CLAUDE.md` AND THIS HANDOFF AT THAT SAME EXACT SHA.** Reading one document at
   one SHA and the other at another is how a stale contract gets applied to current code.
   `CLAUDE.md` owns every technical contract; this handoff owns volatile state and the
   active plan; **code and tests remain decisive.**
3. **INSPECT ACTIVE REPOSITORY STATE BEFORE ACTING** — open PRs, open candidates, live task
   branches, and **who currently holds writable ownership.** Never infer from this document
   that a branch, PR or ownership assignment still holds; §1 and §4 record what was true
   when written.
4. **ONLY THEN ACT** — and act within the declared transport mode, grade and scope discipline
   of `CLAUDE.md` §1.

**Two standing rules that do not expire:** a documentation record never authorizes an
implementation or a run, and **validity is judged before performance** — a valid negative
result is a result, while a run whose denominator an instrument defect shrank is not a
result at all.

### 9.2 Decision log — append one dated entry per MATERIAL change

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
