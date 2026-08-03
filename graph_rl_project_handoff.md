# Multi-Agent Graph RL — FD-BASELINE-v1 Closure / Final-Cell Probe Handoff

**Supersedes all earlier handoffs.**

Written 2026-08-03. B1–B4, the first real post-B3 instrumented probe, the B4 observability
follow-up (PR #7) and **FD-BASELINE-v1** (PR #8) are all CLOSED. The commit that lands this
handoff changes documents only; it does not change code, tests, configuration, dependencies
or workflows.

Baseline **difficulty selection is finished**: exactly one factor was selected, implemented,
reviewed and merged. What has NOT happened is any measurement of the resulting cell. The
next task is a bounded short scientific probe of the merged fuel-damage configuration,
before any long baseline.

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
- **FD-BASELINE-v1 — CLOSED / MERGED / LOCKED.** Approved candidate
  `a8669f450708c2508753c49ab16fd1028b29607d`, integrated by
  `1cecb0ac99f839d47ffeea12c8871aec77e66640` (PR #8); the merged tree was independently
  verified identical to the approved candidate tree. The FIRST candidate
  `1cf53fcee3ee05b3466c8391cbc6bb04420a0985` received REQUEST-FIXES; the correction landed
  as a NEW CHILD COMMIT on the same branch and PR, with no amend, rebase, force-push or
  history rewrite. §3 summarizes the factor; `CLAUDE.md` §5 and §7 own the authoritative
  contract and lock.
- **No active task, candidate or PR after this documentation PR merges; ownership is
  RELEASED.** Immediately before this documentation change, `main` was at
  `1cecb0ac99f839d47ffeea12c8871aec77e66640` with PR #8 merged and no open code PR. This
  docs-only commit advances `main`, so **the receiving orchestrator must resolve the new
  full `main` SHA** rather than reuse any SHA named above as a base. The code branch
  `task/fuel-damage-baseline-v1` and this documentation branch are both retained.

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
those numbers must never be reused as the new cell's expected behaviour.

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

**No live BLADE/BONMIN probe, training run, rollout or scientific baseline was performed.**
Every test is solver-free and drives the pipeline through stubbed engine seams. The lock
certifies the implementation; it says nothing about how the cell behaves.

## 4. Next task — a bounded SHORT SCIENTIFIC PROBE of the merged fuel-damage cell

Start with fresh exact-SHA initialization against the new `main`. This task must be
designed and separately authorized; **this documentation task neither authorizes nor runs
it.**

1. Design the probe: exact iterations × episodes, exact train and held-out seed bands, and
   the pass/fail reading rules — decided BEFORE it runs, so no result is chosen after the
   fact.
2. Run it once, from a clean checkout with complete Git provenance, on the merged cell as
   configured (no ad-hoc knob changes, no second difficulty factor).
3. Report, all with explicit denominators:
   - complete provenance and the exact resolved configuration;
   - scheduled **clean vs damaged** attempt populations and their successes/failures;
   - **matched-pair yield** and the paired reward delta over pairs whose BOTH members
     completed, next to its pair denominator;
   - **failures by pipeline stage** (`generation` / `setup` / `run` / `reward`), including
     any `setup` planning refusal and any `run`-stage live-window refusal;
   - per-episode **event / wake / real RTB command / death** outcomes;
   - reward headroom, and whether the PPO updates were productive.
4. Only if that probe passes may a long baseline be proposed. Interpretation rules survive
   unchanged: a held-out mean is never read without its denominator; an all-failed batch
   reports `null`, never `0.0`; and an empty successful-pair population is `null` too.

**Do not pre-claim any probe result**, and do not reuse the §2 numbers as its expectation.

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
- PR #7: per-episode `OK` blocks, direct unique-target-id counts, accounted structural
  roster failures and disjoint per-round eval artifact namespaces.
- PR #8 (FD-BASELINE-v1): fuel damage is the ONE selected difficulty factor; deterministic
  private RNG domain; matched forced-clean / forced-damaged evaluation pairs on the same
  held-out seed; the strict window validated twice (planned, then live before mutation);
  RTB measured from real emitted command history, never from `GraphPlanExecutor.rtb_issued`;
  explicit `aircraft_penalty_coeff = 2.25` with the reward formula unchanged.
- The legacy split surface remains retained, not retired.

## 6. Out of scope for the next task

- a long training run before the final-cell probe passes;
- selecting or enabling a SECOND difficulty factor (`probability < 1`, hostile fire/SAMs,
  dense reward) — each is its own research change;
- reworking the merged FD-BASELINE-v1 mechanism or its reviewed research decisions;
- checkpoint loading/resume;
- centralized critic / CTDE;
- low-known-cell solver timeout unless the chosen cell needs `known ≤ 2`;
- ETA/peer-dropout, reachability-model and legacy-split retirement work;
- README rewrite.

## 7. Documentation duties

| Trigger | Duty |
|---|---|
| B1–B4 preparation lands — **DONE** | Contracts and locks recorded in `CLAUDE.md` |
| First real post-B3 probe completes — **DONE** | Exact code SHA, denominators, yield, failure stage, transitions and pre/post held-out measurements recorded |
| PR #7 observability follow-up lands — **DONE** | Unique-target semantics, per-episode output, eval artifact preservation and fix-chain lock recorded |
| Selected baseline-difficulty factors land — **DONE for FD-BASELINE-v1** | Contract in `CLAUDE.md` §5, tick placement in §4, routing in §6, lock + fix chain in §7, selection closure and deferrals in §8 — recorded without pre-claiming any result |
| Final-cell short probe completes — **NEXT MEASUREMENT TRIGGER** | Record exact config, provenance, denominators, clean/damaged and matched-pair populations, failures by stage, event/wake/RTB/death outcomes, reward headroom and update evidence before authorizing a long baseline |

## 8. Next action

Implementation for the final Phase-A baseline cell is COMPLETE and locked. Ownership is
released after this documentation push; the next orchestrator performs fresh exact-SHA
initialization against the new `main` and designs the bounded short probe in §4.

**This document authorizes neither an implementation nor a training run.**
