# Multi-Agent Graph RL — B4 Closure / First-Run Handoff

**Supersedes all earlier handoffs.**

Written 2026-07-29. **B1, B2, B3 and B4 preparation are all CLOSED / MERGED / LOCKED.** The
offline scenario-construction phase is complete AND the trainer is now auditable. The commit
that lands this document changes documents only — no code, no test delta.

**No real post-B3 training run has ever been performed.** B4 built the instrument, not the
measurement. The next task is a short, separately authorized instrumented probe (§4).

This handoff is volatile and deliberately thin. Technical contracts live in `CLAUDE.md`;
code and tests remain decisive. Where a fact is already in `CLAUDE.md` this document
cross-references it instead of restating it — a duplicated fact is a fact that will drift.

## 0. How this workspace reads the repository

- Repository access is **capability-aware**, not shared. The two orchestrators see different
  things and must resolve facts through what each actually has:
  - **GPT** resolves repository facts through GitHub first — branches, PRs, files, and exact
    SHAs. That access is a **search** interface, not a filesystem: it cannot list files,
    count occurrences, prove that something is ABSENT, or run `git`.
  - **Claude** resolves facts through a synchronized mounted snapshot of `main` plus the Git
    evidence CC reports. It is the weaker of the two views: like GPT's it is a **search**
    interface — it cannot list files, count occurrences, prove that something is ABSENT, or
    run `git` — and in addition it cannot produce a diff and can lag the true `main` head.
    Task branches and PRs must not be assumed accessible.
  - Either orchestrator asks the user ONE focused question only when its own available
    access cannot establish the required fact.
- Every repository claim must be tied to an **explicit full SHA**. At that SHA, code and
  tests are authoritative, followed by `CLAUDE.md` and this handoff read at the SAME SHA.
  Project Sources, memory, chat summaries, and pasted reports are **not** evidence of current
  repository state.
- Cite code by **file + symbol or exact string**, never by line number.
- Documents can lag code, and lag each other.
- `CLAUDE.md` §1 defines the grade scale as a **trust policy**, and the Grade-A routing
  default between the two orchestrators. This handoff declares only a grade per task.

## 1. Current state

- **B1 — CLOSED / MERGED / LOCKED.** `d6758ac1899621b2ceebcb63afb5e8577184cd91`, merged by
  `bd087c3c18b96f1fe847b4987c73f394a43249c1` (PR #2).
- **B2 — CLOSED / MERGED / LOCKED.** `e22aee359e06591bdb179ef06a566db90f83a558`, merged by
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3).
- **B3 — CLOSED / MERGED / LOCKED.** `dd14ab418c71e3bd615f1198d0c612502642d29b`, merged by
  `14224531db9deb700f6e397203177eb8c701c6cc` (PR #4).
- **B4 preparation — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c`, integrated into `main` by merge commit
  `ba936606deada050ed9298600ee9041fc330af6c` (**PR #6**, merged). The merged tree is
  byte-identical to the approved one. Grade A: the first candidate
  (`dc2142627dc40886667170fc2121fe50336329cd`) was REQUEST-FIXES and the fix chain landed as
  a NEW commit — the reviewed commit was never amended or force-pushed. `CLAUDE.md` §5
  carries the contract and §7 records the lock; both are the authority, so §2 stays short.
- **B4 is PREPARATION, not a baseline.** It changed two files
  (`rl/training/graph_train.py`, `tests/test_graph_train.py`) and no pipeline behaviour.
- **No real post-B3 training run or held-out sweep has occurred.** The seed-0 rollout
  recorded at the B3 lock is ONE reference episode at one seed — a feasibility measurement,
  **not a baseline**. The pre-B3 measurements (0 wakes at `reward = +0.0000`) remain
  **invalid as learning evidence**: they measured a world with nothing to discover.
  No iteration-0 or final held-out number exists to cite.
- **No active implementation task, candidate, or PR.** Ownership is RELEASED after this
  documentation push. As before, the reviewed B4 SHA is the last reviewed **code baseline**,
  NOT the base SHA for the next task — this docs-only commit advances `main`, so the next
  task must resolve the current full `main` SHA immediately before dispatch and declare THAT
  as its base.
- The task branches `task/b1-generator-configuration`,
  `task/b2-route-relative-hidden-placement`, `task/b3-setup-seam`,
  `task/b3-documentation-lock` and `task/b4-training-run-instrumentation` have all been
  deleted remotely after verification. `main`, `flat-final` and the `pre-cleanup` tag are
  untouched.

## 2. What B4 delivered

Full contract in `CLAUDE.md` §5 ("Trainer + run auditability") and the §7 `1b48145` lock
entry. At summary level:

- **`skip_and_account_v1`** — every scheduled train/eval seed is attempted at most once; a
  failure is never retried, replaced, or allowed to shift a seed band; failures never enter a
  PPO buffer or a reward aggregate; each is recorded exactly once; attempts, successes,
  failures and denominators stay explicit, and reward statistics describe the successful /
  exact-cardinality-feasible subset.
- **Provenance is a precondition** — collected before the run creates any artifact;
  `available` requires the full SHA *and* the clean/dirty verdict; incomplete provenance
  writes an inspectable attempted `run_config.json` and then refuses before policy,
  generator, episode or optimizer work; a known-dirty tree warns and may run.
- **Artifacts** — `run_config.json` (versioned `provenance`), `train_records.jsonl`,
  `eval_records.jsonl`, append-only immediately-flushed `episode_failures.jsonl`, derived
  `run_summary.json`, and ONE four-panel `training_plot.png` drawn from the jsonl alone.
- **Evaluation timing** — a deterministic held-out `pre_update` round runs after the initial
  policy is built and before the first training episode, buffer insert and optimizer step,
  recorded at `updates_completed = 0`; later rounds carry their real completed-update count.
- **Classification** — `all_failed`, `zero_wake` and `productive` are disjoint; an all-failed
  batch or round reports a missing reward (`null`), never `0.0`; a successful zero-wake
  episode is a real successful episode.
- **Unchanged**: PPO behaviour and hyperparameters, seed formulas, the fixed held-out band,
  reward, checkpoint payload, solver, construction, geometry, exact cardinality, and every
  B1–B3 locked interface.

**Evidence at the lock.** `tests/test_graph_train.py` 55 passed; base suite 139 passed,
4 skipped; import purity 12/12; standalone `nlp_env` runner all 55 passed;
`git diff --check` clean. **No `graph_train --selftest`, no BONMIN training probe and no real
training run were executed.**

## 3. Facts the first run depends on

`CLAUDE.md` §3 (no-communication invariants, the single `DETECTION_KM = 50` radius, launch
point == the BLUE airbase), §2 (frozen BLADE engine, frozen solver) and §5 (layer contracts)
remain in force and are NOT restated here. Three items the next task must hold onto:

- **Exact cardinality is settled, the distribution policy is not.** `skip_and_account_v1` is
  how the current cell behaves; the general `n_hidden != usable ego routes` distribution
  policy B2 named remains a separate, still-open design task. The pre-B4 12-seed measurement
  is not a run-time rate — the probe measures the actual yield. `CLAUDE.md` §8.
- **480 vs 479.99968** — raw utility versus the frozen-`EPSILON` reward-side operand. Both
  correct for their own operand; do not "fix" either or compare one against the other.
- **Solver stall and timeout** — solves with `known ≤ 2` can stall for roughly 15 minutes
  against a typical 45 s. The locked cell is clear of it and both configs WARN, but a solve
  timeout is still required before any low-known or n-randomized configuration enters a long
  run. It does NOT exist yet.

## 4. Next task — a short instrumented probe (separately authorized)

**Not a long run.** The loop has never run to completion. The next orchestrator:

1. performs fresh exact-SHA initialization — resolve the current `main` HEAD (this
   documentation commit, not the B4 reviewed or merge SHA) and declare it as the base SHA,
   and select the transport mode per `CLAUDE.md` §1;
2. **confirms complete provenance and the resolved config first** — read `run_config.json`'s
   `provenance` block and construction cell before trusting any number that follows;
3. **obtains the true pre-update held-out measurement** — the `evaluation_stage =
   "pre_update"` / `updates_completed = 0` record. This is the origin of the learning curve
   and the only honest "untrained policy" number;
4. **inspects failure denominators and data yield** — `episode_failures.jsonl`,
   `run_summary.json`'s attempted/successful/failed counts, and the plot's fourth panel. A
   held-out mean is never read without its denominator;
5. **watches the first two iterations** before a long run is proposed. Scale: two bonmin
   solves per episode at roughly 45 s each.

Read saturation carefully: near-ceiling held-out performance **at `updates_completed = 0`**
means the cell lacks learning headroom and must be revisited before compute is spent, whereas
saturation **after training** is expected and intended in this phase.

Do not invent the final training command, configuration, or an expected result in a planning
document. Difficulty returns later through `p < 1`, `fuel_damage`, and targets that shoot
back — not by weakening this phase's scenario.

## 5. Closed decisions

Carried forward unchanged; `CLAUDE.md` is the authority for each.

- Offline construction only; patch-and-reload, never regenerate after solving.
- Route prediction is required and supports `num_agents < n_known`.
- One sensing/arrival/attack/kill-confirmation radius: `DETECTION_KM = 50`.
- `round_trip_cost` and `graph_reward` remain unchanged and frozen. The flat `R ≈ −1/3` was
  scenario degeneracy, never a reward bug.
- **B1 reference cell (`d6758ac`):** fleet 3, `n_known = 3`, `n_hidden = 3` — a cell, not a
  law. **B1 geometry:** `min_target_distance_km = 200`, `min_known_separation_km = 100`, as a
  TRUE great-circle floor under `strict_geometry`. `RolloutConfig` follows `TrainConfig`
  field-for-field.
- **B2 (`e22aee3`):** the placement geometry, eligibility rules, one placement per non-empty
  ego route, an explicit `random.Random`, id-free geometric fingerprints, and the one-way
  import direction (the placement layer never imports `graph_episode_setup`).
- **B3 (`dd14ab4`):** the `(n_hidden, placement_rng)` pair selects construction mode
  explicitly and is never inferred; env-2 is the sole runtime source of truth; agent IDs must
  survive reload; cardinality is exact; the cell is airbase-only.
- **B4 (`1b48145`):** `skip_and_account_v1`; provenance as a precondition; the six run
  artifacts; pre-update evaluation at `updates_completed = 0`; all-failed / zero-wake /
  productive disjoint; a missing aggregate is `null`, never `0.0`.
- **The legacy split surface is RETAINED, not retired** (`split_tasks`, `partial_ratio`,
  `derived_split`, `split_preview`, `num_red_airbases`, Layer 1). The construction path
  simply does not consult it.

## 6. Out of scope

Do not mix the first probe with:

- checkpoint loading / resume (still deferred);
- the solver timeout for low-known cells (needed, but its own task);
- centralized critic / CTDE, dense reward, `p < 1`, ETA/peer-dropout work, or fuel damage;
- changes to `round_trip_cost`, the solver, reward, geometry, cardinality, or the
  reachability mask;
- retiring `split_tasks`, `derived_split`, `split_preview`, or Layer 1;
- the README rewrite.

## 7. Documentation duties, with their trigger

| Trigger | Duty |
|---|---|
| B1 lands — **DONE** | update `CLAUDE.md` §6's "change the training scenario cell" row |
| B1's parameterization closed — **DONE** | retire the `min_target_distance_km` item in `CLAUDE.md` §8 |
| B2 lands — **DONE** | add the `CLAUDE.md` §6 placement row and the §7 `e22aee3` lock |
| B3 lands — **DONE** | rewrite `CLAUDE.md` §4's pipeline diagram for the two paths, document the §5 setup contract, add the §7 `dd14ab4` lock, mark the `split_meta["outcome"]` §8 item legacy-path-only, and replace the pre-B3 zero-headroom item |
| B4 preparation lands — **DONE** | add the `CLAUDE.md` §5 trainer/run-auditability contract, extend the §6 training row, add the §7 `1b48145` lock, close the §8 B4-preparation item, resolve the §8 exact-cardinality item to `skip_and_account_v1`, and record the complete-provenance requirement — all completed in this documentation commit |
| **The first real run / probe completes — NEXT** | record the measured `pre_update` (`updates_completed = 0`) held-out number and its denominator, the observed data yield and failure stages, and whatever the exact-cardinality policy turned out to be in practice |

## 8. Next action

B1, B2, B3 and B4 preparation are closed; ownership is released after this documentation
push. The next orchestrator performs fresh exact-SHA initialization and dispatches the short
instrumented probe described in §4.

**No training run is authorized by this document.**
