# Multi-Agent Graph RL — Probe + B4 Observability Closure / Baseline-Design Handoff

**Supersedes all earlier handoffs.**

Written 2026-08-02. B1–B4 preparation, the first real post-B3 instrumented probe, and
the B4 observability follow-up are CLOSED. The commit that lands this handoff changes
documents only; it does not change code, tests, configuration, dependencies or workflows.

The short probe proved that the reference cell has learning headroom and that the PPO
loop can collect data and update. It is **not a baseline**. The next task is to choose the
difficulty factors for the actual baseline cell, implement and lock only the selected
factors, then run a fresh short probe on that final configuration before any long run.

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
- **B4 preparation — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c`, integrated by
  `ba936606deada050ed9298600ee9041fc330af6c` (PR #6). `CLAUDE.md` §5 owns
  the trainer/run-auditability contract and §7 owns the fix-chain lock.
- **First real post-B3 instrumented probe — CLOSED / REVIEWED MEASUREMENT.** It ran from
  the clean exact code SHA `a3f0838616990987bcb8a51665fa75d84edf5952` with no tracked
  delta. §2 records the measured evidence. It is a short probe, not a baseline.
- **B4 per-episode observability follow-up — CLOSED / MERGED / LOCKED.** Reviewed code
  SHA `211e12e49b676637362d42effdb80988dd0e55eb`, integrated by merge commit
  `ffb95a6ee90df45b2d89802b321dcadcbc272821` (PR #7). The earlier candidate
  `24241690572a7a5264e24348db5e9412b41bc47a` received REQUEST-FIXES; the approved
  correction was a new commit, never rewritten.
- **No active task, candidate or PR; ownership is RELEASED after this documentation
  push.** Immediately before the docs update, GitHub showed `main` at
  `ffb95a6ee90df45b2d89802b321dcadcbc272821`, no open PR and no `task/*` branch.
  This docs-only commit advances `main`, so the next orchestrator must resolve its new
  full SHA rather than reuse any SHA named above as a base.

## 2. First real post-B3 probe — measured evidence

Measured code: exact clean SHA
`a3f0838616990987bcb8a51665fa75d84edf5952`. Exact shape: two iterations × four
scheduled train attempts, seeds `[0,8)`, with the same fixed four held-out seeds
`[1000000,1000004)` before training and after two completed updates. The cell remained
3 agents, 3 known + 3 hidden airbase targets, strict 200 km launch-point distance,
100 km known-target separation, 0.5 route-stretch ratio, unified 50 km detection radius,
`include_sams=False`, randomized red-airbase positions and unchanged PPO defaults.

- Provenance was complete: exact SHA, `dirty=false`, Windows / `nlp_env`, vendored BLADE
  and BONMIN all recorded. The run completed normally in 79.21 s.
- `pre_update`, `updates_completed=0`: reward mean
  `-0.4999997395829586`, **4/4 successful**, all four with wakes.
- Training: **7/8 successful**; all seven successes had wakes; iteration 0 was 3/4,
  iteration 1 was 4/4; **24 transitions** total; two productive iterations and two PPO
  updates.
- The only failure was train seed 2 at `setup`. B2 produced two placements for three
  requested hidden targets because the static solution left one ego without a non-empty
  route. It was attempted once, recorded once and neither retried nor replaced.
- `post_update`, `updates_completed=2`: reward mean
  `5.000007394910353e-7`, **4/4 successful**, numerical zero.
- `run_summary.json:accounting_reconciled=true`; all six expected durable artifacts
  existed.

Interpretation: the easy reference cell had genuine headroom, yielded enough wakes and
transitions for two updates, and the fixed held-out band reached numerical zero. This is
useful implementation/research-direction evidence, not a baseline estimate.

The original `kills_mean` / `eval_kills_mean` values are not unique-target counts:
`GraphPlanExecutor.done` stores `(ego_id,target_id)` confirmations. They did not affect
reward, PPO, wakes, failures or the learning-curve result; reward already deduplicated by
target id. PR #7 replaced their semantics for future runs (§3).

Evidence SHA-256:

- `run_config.json`:
  `36ec89cdb93f89c0b6e40163491159bf2045235b86b2fad47fe03f2f86141237`
- `train_records.jsonl`:
  `af4ec1851425fbcd0330651c05e384d0e44dad67f8aa1f56080543d8247ad82d`
- `eval_records.jsonl`:
  `2c972efaf85d465ab4f2ffce164ba19ac2a6c189db1e2faf83de6b0d201a7439`
- `episode_failures.jsonl`:
  `32d51d2d2ec017491f2fbe6bf133e103361752ced66ba39aac51e9b35b03a08e`
- `run_summary.json`:
  `d2e24714eecdf48bd5f1478ba1c119f405bef5d82067840776daa26dd4270c80`
- `training_plot.png`:
  `c6dec3ac99c5bd35fe627f77b2e97f432cb33235ce07f7efed8f0c05d7a9521b`

## 3. What PR #7 closed

Full contract and lock: `CLAUDE.md` §5 and §7, reviewed at
`211e12e49b676637362d42effdb80988dd0e55eb` and integrated at
`ffb95a6ee90df45b2d89802b321dcadcbc272821`.

- Every successful train, pre-update eval and post-update eval attempt prints one
  immediate labelled `OK` block with phase, indices, exact seed, reward, wakes, ending,
  ticks, dead count, elapsed time and target roster by BLADE name.
- Authoritative trainer counts are unique over `target_id`:
  `targets_confirmed_unique_mean` / `eval_targets_confirmed_unique_mean`.
  `kills_mean` / `eval_kills_mean` remain compatibility aliases to those corrected
  values. Names are display-only.
- A structural roster defect is an accounted `setup` failure, never a successful false
  zero. Only name lookup may degrade to `<unnamed target>`, without changing ids,
  denominators or counts.
- Every eval round owns a disjoint scenario-file tag namespace while reusing the same
  fixed held-out seeds, so pre- and post-update scenario JSONs coexist.
- No policy, reward, PPO, executor, tick-loop, scenario-content or seed behavior changed.

Verification at the approved head: `tests/test_graph_train.py` 73 passed; import purity
12 passed; full suite 157 passed, 4 skipped; standalone `nlp_env` runner all 73 passed;
`git diff --check` clean. The authorized smoke was implementation validation, not a
scientific run.

## 4. Next task — baseline difficulty design and preparation

Start with fresh exact-SHA initialization and keep this as a research-design decision
before implementation.

1. Decide which factors belong in the final Phase-A baseline cell:
   `fuel_damage`, `probability < 1`, enemy targets that shoot back, and any additional
   candidates.
2. For each selected factor, close its semantics, interaction with BLADE/solver/reward,
   observability, reproducibility, failure policy and proof obligations. Do not enable a
   bundle implicitly.
3. Implement and lock the selected factors through separate bounded tasks. Preserve the
   no-communication, route-placement, exact-cardinality and provenance contracts.
4. Run a new short instrumented probe on the final selected configuration using PR #7's
   per-episode output and unique-target metrics.
5. Start a long baseline only if that probe shows complete provenance, explicit
   denominators, acceptable failure/data yield, organic wakes, reward headroom and
   productive PPO updates.

Do not rerun the old 2×4 probe merely to obtain prettier logs; PR #7 did not change the
behavior it measured, and the easy cell is about to be reconsidered.

## 5. Closed decisions

- Offline construction only: solve → place → patch → reload.
- Route prediction is required and supports `num_agents < n_known`.
- One sensing/arrival/attack/kill-confirmation radius: `DETECTION_KM = 50`.
- `round_trip_cost` and the current p=1 `graph_reward` remain frozen.
- B1 reference cell: 3 agents, 3 known, 3 hidden; strict 200 km launch-point distance,
  100 km known-target separation and 0.5 stretch ratio. It is a reference cell, not a law.
- B2: one placement per non-empty ego route, explicit `random.Random`, id-free geometric
  fingerprints and one-way placement-layer imports.
- B3: explicit construction-path selection; env-2 is the runtime source of truth;
  ordered agent IDs survive reload; exact cardinality; airbase-only cell.
- B4: complete provenance precondition, `skip_and_account_v1`, fixed held-out band,
  explicit denominators, six run artifacts, true pre-update evaluation and disjoint
  all-failed / zero-wake / productive states.
- PR #7: per-episode `OK` blocks, direct unique-target-id counts, accounted structural
  roster failures and disjoint per-round eval artifact namespaces.
- The legacy split surface remains retained, not retired.

## 6. Out of scope for the next design task

- a long training run before the final-cell probe passes;
- checkpoint loading/resume;
- centralized critic / CTDE;
- dense reward unless explicitly selected as a separate research change;
- low-known-cell solver timeout unless the chosen cell needs `known ≤ 2`;
- ETA/peer-dropout, reachability-model and legacy-split retirement work;
- README rewrite.

## 7. Documentation duties

| Trigger | Duty |
|---|---|
| B1–B4 preparation lands — **DONE** | Contracts and locks recorded in `CLAUDE.md` |
| First real post-B3 probe completes — **DONE** | Exact code SHA, denominators, yield, failure stage, transitions and pre/post held-out measurements recorded |
| PR #7 observability follow-up lands — **DONE** | Unique-target semantics, per-episode output, eval artifact preservation and fix-chain lock recorded |
| Selected baseline-difficulty factors land — **NEXT TRIGGER** | Record each reviewed contract and lock without pre-claiming results |
| Final-cell short probe completes — **NEXT MEASUREMENT TRIGGER** | Record exact config, provenance, denominators, data yield, wakes, reward headroom and update evidence before authorizing a long baseline |

## 8. Next action

B1–B4 preparation, the first post-B3 probe and the B4 observability follow-up are
closed. Ownership is released after this documentation push. The next orchestrator
performs fresh exact-SHA initialization and begins the baseline difficulty-design
decision in §4.

**This document authorizes neither an implementation nor a training run.**
