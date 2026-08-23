# Multi-Agent Graph RL — Phase-A Closure / VALID VARIABLE-SEVERITY BASELINE (NEGATIVE FINDING) / PHASE-B CTDE MERGED — Handoff

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
been run** (§1, §3k, §4, §8).

**THE STATE, STATED PLAINLY.** **PHASE A IS CLOSED.** The authorized long-baseline rerun —
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
ORIGINAL Phase-A reference that was never repurposed for it. **NO CTDE SCIENTIFIC
COMPARISON HAS BEEN RUN, and no CTDE benefit is claimed** — a merged implementation and a
passing suite measure nothing scientific (`CLAUDE.md` §5 owns the CTDE contract, §7 the
lock, §8 the un-run comparison). None of this is a reopening of Phase A
and none of it changes any technical CTDE contract: the Phase-A reference baseline stays
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

- **BASE of THIS documentation record:** `8390d85c2072e9cbe984ce5f2731cef3a9b14985`,
  committed `2026-08-23 02:43:17 Asia/Jerusalem`, tree
  `9686c107b8864f00a7d4403d70faf42ab561d2fb` — the `main` head produced by the **Phase-B
  CTDE code merge (PR #30)**. **That is the exact base this PHASE-B CTDE
  documentation/lock candidate was DERIVED ON**, and it is a statement about this record's
  derivation only — **not** a claim about live `main`,
  which this record's own integration necessarily advances past it. Neither this
  documentation commit nor its future merge can name its own SHA, and inventing either
  would be a false provenance claim. **Every receiving orchestrator therefore resolves the
  live full `main` SHA from GitHub and re-reads both documents at that SHA — GitHub is
  authoritative for live branch and PR state, never this document.**
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
  here. **NO training run, baseline, probe, rollout, BONMIN solve or BLADE episode was
  executed for PR #30**, and **no actor-only vs CTDE comparison has been run.**
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
- **VOLATILE STATE — WHAT IS DONE, WHAT IS WRITABLE, AND WHO OWNS IT (2026-08-23).**
    *(The 2026-08-22 entry recorded the measurement as RUNNING and NOT REVIEWED; that was
    true when written and is now superseded by the state below.)*
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
  - **CURRENT WRITABLE TASK:** this PHASE-B CTDE documentation/lock task — branch
    `task/phase-b-ctde-doc-lock` and its draft PR — held by the CTDE GPT orchestrator,
    which holds sole writable repository ownership. **No CODE candidate is open**: PR #30
    was the last one and it is merged (§3k). *(Historical: the FD closure orchestrator held
    a ONE-TIME writable exception scoped to the variable-severity record alone, on branch
    `task/fd-variable-severity-valid-doc-lock`; that exception ENDED when the record was
    integrated and writable ownership returned to the CTDE orchestrator.)*
  - **CTDE INTEGRATION GATE — CLOSED / SATISFIED ON BOTH HALVES.** The requirement that the
    variable-severity measurement COMPLETE and receive an **INDEPENDENT VALIDITY VERDICT**
    is **MET** (§3j) — met by a NEGATIVE result, which satisfies the gate exactly as a
    positive one would, because the gate tests VALIDITY, not favourability. The second
    requirement — a NEW immutable actor-only pre-CTDE reference preserved from the
    then-current actor-only state — is **MET** by **`pre-ctde-actor-only` =
    `d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the CTDE integration's FIRST parent,
    which **must not move**. `phase-a-baseline` remains historical provenance for the
    ORIGINAL Phase-A reference and was never moved or repurposed for it. **What is still
    OPEN is the SCIENTIFIC COMPARISON** (§4, Task 9), which has not been run.

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
  state, is likewise PRESERVED and must not move.** **The same discipline applies to
  `task/variable-fd-severity-baseline` (the merged PR #27 implementation branch) and to
  `task/variable-fd-severity-doc-lock` once THIS record is merged** — retirement is the GPT
  orchestrator's action, only after each tip is verified reachable from integrated `main`,
  and it is NOT part of this documentation task.

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
  `9686c107b8864f00a7d4403d70faf42ab561d2fb`. Grade A under `GPT_GITHUB`, mode SURGICAL.
  Append-only fix chain: initial reviewed candidate
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
- **CC-REPORTED ENGINEERING EVIDENCE — IMPLEMENTATION VALIDATION, NOT SCIENTIFIC
  EVIDENCE.** Full solver-free suite 334 passed / 4 skipped; `tests/test_graph_ctde.py` 43
  passed; `tests/test_graph_train.py` 119 passed; `tests/test_graph_ppo.py` 18 passed; the
  standalone `nlp_env` CTDE runner 43 passed; `git diff --check` clean.
- **NO SCIENTIFIC RUN OCCURRED FOR PR #30** — no training run, baseline, probe, rollout,
  BONMIN solve or BLADE episode. **NO actor-only vs CTDE comparison has been executed and
  NO CTDE benefit is established or may be pre-claimed** — not from the Phase-A result, not
  from the variable-severity baseline (whose NEGATIVE severity finding is **NOT** evidence
  that centralized training would change it), and not from a passing test suite.

## 4. Current work — VARIABLE-SEVERITY BASELINE CLOSED (VALID, NEGATIVE); PHASE-B CTDE MERGED; THE CTDE COMPARISON NOT RUN

Start with fresh exact-SHA initialization against the current `main`. **THIS documentation
task neither authorizes nor runs anything: it RECORDS state only, and it does NOT authorize
CC to run training, to re-run the completed baseline described below, or to write a
critic.** The CTDE authorization recorded here is a statement about SEQUENCING and
OWNERSHIP — the work itself is a SEPARATE task owned by the CTDE GPT orchestrator, and no
technical CTDE contract is defined or locked by this record.

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
complete**, so this paragraph is now the arrangement's HISTORICAL record; the live state is
Task 8 and Task 9 below.

**This supersedes the serial "(3) only then proceed to Phase-B CTDE design" rule this
section previously stated** — that rule is HISTORY and must not be restated as live. It also
still supersedes the ORIGINAL claim that Phase-B CTDE was immediately next and that a
stochastic/partial fuel-degradation variant was deferred until after Phase B —
FD-VARIABLE-SEVERITY-v1 is that variant's approved form. **It is NOT a reopening of Phase A,
and it changes NO technical CTDE contract.** The Phase-A baseline stays CLOSED, VALID and
IMMUTABLE (§3h), and `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) must
not move — it is historical provenance for the ORIGINAL Phase-A reference and is never
repurposed as the future pre-CTDE reference.

**OWNERSHIP — THE CTDE GPT ORCHESTRATOR HOLDS IT, INCLUDING FOR THIS RECORD.** Through the
parallel phase the CTDE GPT orchestrator was the SOLE WRITABLE repository owner and the FD
measurement orchestrator was READ-ONLY on its detached snapshot. *(Historical: the user
granted the FD orchestrator a ONE-TIME writable exception scoped to the variable-severity
closure record alone; that exception ENDED when the record was integrated, and the FD
orchestrator reverted to READ-ONLY with no writable branch or PR.)* Writable ownership then
returned to the CTDE GPT orchestrator, which integrated PR #30, and **it owns THIS
documentation/lock task as well** — branch `task/phase-b-ctde-doc-lock` and its draft PR
are the only writable candidate while this record is in flight. **The CTDE integration
gate's repository-side prerequisite is DISCHARGED**: `pre-ctde-actor-only` =
`d437084c5fb1a22c21596a48c58e03f7e15a0115` exists and must not move.

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

**Task 9 — THE FIRST CONTROLLED ACTOR-ONLY vs CTDE SCIENTIFIC COMPARISON. OPEN. NOT RUN.**
This is the next unresolved task and the only genuinely open part of Phase B. **No CTDE
benefit may be pre-claimed** — not from the Phase-A result, not from the variable-severity
baseline (whose NEGATIVE severity finding is **NOT** evidence that centralized training
would change it), and **not from CTDE implementation work, engineering tests or a passing
test suite, none of which measure anything scientific.** Its binding constraints:

- it runs **ON THE LOCKED ORIGINAL PHASE-A SCIENTIFIC CELL** — 3 agents, 3 known + 3
  hidden, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams = false`,
  `probability = 1`, frozen solver and BLADE, unchanged `graph_reward` formula with
  `aircraft_penalty_coeff = 2.25` — judged against the approved Phase-A baseline (§3h)
  under the SAME validity gate below, **VALIDITY BEFORE PERFORMANCE**;
- it is a CONTROLLED contrast in which **`training_mode` is the ONLY differing factor**, so
  the two arms share the seed schedule, the held-out band, the difficulty design and the
  evaluation construct;
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

## 6. Out of scope for the current work

The additional actor-only FD-VARIABLE-SEVERITY-v1 baseline (§4, Task 7) is **CLOSED,
VALID and NEGATIVE** (§3j), and **PHASE-B CTDE IMPLEMENTATION is CLOSED, REVIEWED and
MERGED** (§4, Task 8; §3k), so the current work is **the first controlled actor-only vs
CTDE scientific comparison** (§4, Task 9), which has NOT been run. Out of scope:

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
| The FIRST actor-only vs CTDE COMPARISON completes — **NOT RUN** | Record run identity, exact measured code SHA, the resolved configuration of BOTH arms and the single differing factor (`training_mode`), complete provenance, every denominator per arm, the held-out results with their own denominators, the within-arm and between-arm claims each stated against what they are averaged over, artifact completeness and the evidence hashes in `CLAUDE.md` §7, with the phase state in §8. **Validity is judged BEFORE performance; a null or negative CTDE result is a valid observation, not a technical failure** |

## 8. Next action

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
`8390d85c2072e9cbe984ce5f2731cef3a9b14985`, tree
`9686c107b8864f00a7d4403d70faf42ab561d2fb` — the **Phase-B CTDE code merge (PR #30)**,
which is an already-existing CODE INTEGRATION and not a measurement identity. Resolve the
live full `main` SHA from GitHub.

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

**THE NEXT UNRESOLVED TASK IS THE FIRST CONTROLLED ACTOR-ONLY vs CTDE COMPARISON** (§4,
Task 9). It runs **ON THE LOCKED ORIGINAL PHASE-A SCIENTIFIC CELL** — 3 agents, 3 known +
3 hidden, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams = false`,
`probability = 1`, frozen solver and BLADE, unchanged `graph_reward` formula with
`aircraft_penalty_coeff = 2.25` — judged against the approved Phase-A baseline (§3h) under
the SAME validity gate above, VALIDITY BEFORE PERFORMANCE. It must be a CONTROLLED contrast
in which **`training_mode` is the ONLY differing factor**, and it **must NOT bundle
`p(destroy) < 1`, SAMs, dense reward, a solver change, a reward-formula change, or any new
difficulty factor** (§6) — bundling one would make the comparison uninterpretable. A run
showing no CTDE improvement is a **valid NEGATIVE observation**, not a technical failure.
**No CTDE preset exists in the repository**, and creating one belongs to that task.

**OWNERSHIP.** The CTDE GPT orchestrator holds sole writable repository ownership,
including for THIS documentation/lock task — branch `task/phase-b-ctde-doc-lock` and its
draft PR are the only writable candidate while this record is in flight. *(Historical: the
FD measurement orchestrator's ONE-TIME writable exception, scoped to the variable-severity
closure record alone, ENDED when that record was integrated.)* This record's own integrating
merge SHA is deliberately NOT named — it does not exist while this is written, and inventing
it would be a false provenance claim. **GitHub remains authoritative for live branch and PR
state — resolve it there, never from this document.** Retiring any merged branch is a
separate action owned by the GPT orchestrator, done only once each tip is verified reachable
from integrated `main`; `pre-ctde-actor-only`, `phase-a-baseline`, `flat-final`, the
`pre-cleanup` tag, PR #30's history and every preserved scientific artifact stay PRESERVED.

**NEITHER the Phase-A long baseline NOR the variable-severity baseline is to be re-run,
resumed, repaired, extended or re-tuned.** A valid measurement exists for each; re-running
would not make either more valid, and a NEGATIVE finding is a result rather than a reason to
try again. The Phase-A scientific contract is frozen as the reference baseline, and the
variable-severity run is an ADDITIONAL measurement of a DIFFERENT cell beside it, never a
replacement for it.

Resolve live branch and PR state from GitHub; this document does not track it.

**This document authorizes neither an implementation nor a training run.**
