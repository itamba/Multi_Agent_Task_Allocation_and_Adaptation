# Multi-Agent Graph RL — Scenario-Construction Handoff

**Supersedes all earlier handoffs.**

Written 2026-07-28. **B1 and B2 are both CLOSED / MERGED / LOCKED.** B1 reviewed code SHA:
`d6758ac1899621b2ceebcb63afb5e8577184cd91`, merged by `bd087c3c18b96f1fe847b4987c73f394a43249c1`
(PR #2). B2 reviewed code SHA: `e22aee359e06591bdb179ef06a566db90f83a558`, merged by
`8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3). **B3 — the setup seam — is the next
task.** The commit that lands this document changes documents only — no code, no test delta.

This handoff describes the next phase only. Technical contracts and frozen layers live in
`CLAUDE.md`; code and tests remain decisive. Where a fact is already in `CLAUDE.md` this
document cross-references it instead of restating it — a duplicated fact is a fact that will
drift.

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
- Cite code by **file + symbol or exact string**, never by line number. This is the same
  anchoring rule surgical OLD→NEW edits already require, so recon and dispatch share one
  convention.
- Documents can lag code, and lag each other. Observed at the `384845b` baseline: the code
  carried all three scenario-construction preconditions while `CLAUDE.md` still showed the
  pre-migration §1 and a `PENDING` §7 entry.
- `CLAUDE.md` §1 now defines the grade scale as a **trust policy** — C trusted with no
  review, B the orchestrator reads the changed files, A the orchestrator approves the exact
  full reviewed SHA, with line-by-line review when cross-ego isolation or a §5 locked layer
  is touched: GPT reads the exact GitHub `base...candidate` comparison, and Claude receives
  focused changed hunks or targeted evidence from CC. No full diff is pasted into chat. This
  handoff declares only a grade per task; the definition, and the Grade-A routing default
  between the two orchestrators, belong there.

## 1. Current state

- **B1 — CLOSED / MERGED / LOCKED.** Reviewed code SHA:
  `d6758ac1899621b2ceebcb63afb5e8577184cd91`, integrated into `main` by merge commit
  `bd087c3c18b96f1fe847b4987c73f394a43249c1` (PR #2, merged). `CLAUDE.md` §7 records the
  lock under `d6758ac`. That reviewed SHA is the last reviewed **code baseline** — it is
  **not** the base SHA for the next task. Docs-only commits may advance `main` further, so
  every implementation task must resolve the current full `main` SHA immediately before
  dispatch and declare THAT as its base. No active implementation task, candidate, or PR.
  The old task branch `task/b1-generator-configuration` may still exist remotely; it is
  not active. Ownership is released to the next orchestrator after this documentation
  push.
- Verified tests for the reviewed B1 commit: suite 84 (incl. the P7–P12 construction
  preconditions and the `graph_train` / `graph_rollout` construction tests), import
  purity 12/12, module selftests, and the bonmin selftest under `nlp_env`.
- **B2 — CLOSED / MERGED / LOCKED.** Reviewed code SHA:
  `e22aee359e06591bdb179ef06a566db90f83a558`, integrated into `main` by merge commit
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3, merged). `CLAUDE.md` §7 records the
  lock under `e22aee3`. Verified on the integrated merge: **18** focused B2 tests, import
  purity **12/12**, full suite **102**, `git diff --check` clean, plus all 18 B2 tests and
  all 12 import-purity entry modules green through the `nlp_env` `__main__` runners. No
  bonmin or live BLADE run was required. The task branch
  `task/b2-route-relative-hidden-placement` may still exist remotely; it is not active.
- No active implementation task, candidate, or PR. Ownership is released to the next
  orchestrator after this documentation push. As with B1, the reviewed B2 SHA is the last
  reviewed **code baseline**, NOT the base SHA for the next task — docs-only commits
  advance `main` further, so B3 must resolve the current full `main` SHA immediately
  before dispatch and declare THAT as its base.
- Phase-A trainer exists, but **no full training run has ever been performed.** Neither B1
  nor B2 closing unblocks one — B3 (setup seam) must still land first, because until it
  does nothing consumes the B2 placement layer and the world still contains no hidden
  targets (§4, and `CLAUDE.md` §8 "Pre-B3 zero-headroom"). B4 remains blocked.
- `CLAUDE.md` §7's workflow + handoff migration entry carries its recorded SHA `a5a4137`; the
  hash-convention duty for that entry is discharged.

## 2. Phase goal

Construct scenarios in which a hidden target appears **mid-route, before the planned
engagement**. The research event is that one ego privately senses an opportunity and must
choose whether to continue its assignment, engage the pop-up, or abort. There is no
communication and no live injection scheduler.

The required pipeline is:

1. Generate a world containing only known targets.
2. Build environment 1, reset it, and extract the agents and known tasks.
3. Solve the known set to produce `A_init` and `belief_tasks`. `split_tasks` is **not
   called** — there is nothing to split.
4. Predict each ego's actually flown route from that solution.
5. Place hidden targets relative to valid route segments.
6. Patch the existing scenario JSON; do not regenerate it.
7. Build environment 2, reset it, and re-extract environment-bound agents and all tasks.
8. Solve the full set for the oracle used by the reward denominator.
9. Build independent beliefs from `(belief_tasks, A_init)` and run the existing executor.

BLADE therefore runs the full world, while each ego initially knows only the known-target
plan. A hidden target may enter an ego's belief only through that ego's own sensing and the
existing trigger path.

## 3. Design facts that the implementation depends on

### Route prediction

**IMPLEMENTED in B2** (`rl/training/graph_hidden_placement.py` → `predict_route`): the
facts below are no longer requirements to satisfy but the contract the merged code already
meets. `predict_route` imports the frozen helper and calls it separately per level,
chaining the returned end location, seeded from the shared launch point; an executor-queue
equivalence test pins it against `BladeExecutorMinimal`. It is a PURE function — nothing in
`setup_episode` calls it yet (that is B3).

- `nearest_neighbor_order` lives in
  `src/match_aou/utils/blade_utils/blade_executor_minimal.py`; it is pure and deterministic
  and returns a full greedy tour plus the end location. Starting at `start_location` it
  repeatedly selects the closest remaining located assignment, tie-breaking by
  `(task_idx, step_idx)`, and advances the current position to that target. Unlocated
  assignments go last; with `start_location=None` it degrades to `(task_idx, step_idx)`
  order without advancing.
- The executor (`blade_graph_executor.py`) imports that same helper and seeds it from the
  ego's LIVE observed position, under `nn_ordering=True` by default. Offline prediction must
  call the same function, seeded from the launch point — not a reimplementation.
- A shortcut that assumes one target per ego is invalid: with several assigned targets the
  real route is a polyline such as `base → A → B → C`, and `base → B` is never flown.
- Leg 1 is exact. Later legs may change only under near-ties, because live replanning
  restarts from a physical position up to `DETECTION_KM` short of the previous target, while
  prediction advances all the way to it.
- Later legs therefore require a safe nearest-neighbor margin. If the margin fails, reject
  the geometry or fall back to leg 1.
- Escape hatch if prediction proves fragile: the executor accepts `nn_ordering=False`, which
  makes the order exactly `(task_idx, step_idx)` — perfectly predictable, no margin needed —
  at the cost of longer, less realistic flight paths. It changes the ORDER only, never WHICH
  steps execute. Not the default, and all measured evidence so far used `True`.

### Segment geometry

**IMPLEMENTED in B2** as the closed numbers in §5. The guaranteed portion `G = L - D`, the
`(1 - s/L)·D` later-leg origin uncertainty, and the "favour the far half" bias are all
enforced by `place_hidden_targets` / `validate_placement`, which fail loudly rather than
clamp. The scenario/data-seam consequences below remain B3's to satisfy.

- All egos launch from the BLUE airbase (`CLAUDE.md` §3), so routes form a star from one
  origin. Hidden targets near that origin may be sensed by several egos almost
  simultaneously — privately, so not a no-comms violation, but it destroys the asymmetry the
  phase is about. Placement should favor the far half of a leg.
- The guaranteed-flown part of a leg ends `DETECTION_KM` before its target: inside that
  radius the ego attacks and issues no new movement, and the stale route drags it inward
  until the kill confirms. Placement and validation must use that guaranteed portion, not
  the full ideal segment.
- A corollary for legs 2+: the effective ORIGIN of the next leg is the previous target's
  **vicinity**, not the previous target exactly. Treat the origin end of a later leg as
  uncertain by up to `DETECTION_KM` and do not place a target that depends on it.
- Geometry determines discovery timing. Do not add a scheduler.
- Keep the unified `DETECTION_KM = 50 km` contract (`CLAUDE.md` §3, and §8 records that
  radius expansion was CANCELLED, not deferred — do not reopen it here).

Reference aircraft values, derived from `CLASS_RANGE_TIERS` and the base template's knots
(× 1.852). Verified against both sources:

| Aircraft | Effective one-way range | Reference speed |
|---|---:|---:|
| F-16 | 400 km | ≈ 2413 km/h |
| F-35A | 900 km | ≈ 1932 km/h |
| B-2 | 1500 km | ≈ 1046 km/h |
| KC-135R | 2100 km | ≈ 854 km/h |

`MAX_SIM_TICKS = 14400` seconds.

### What P6 guarantees, and what it does not

`tests/test_scenario_construction_preconditions.py` P6 pins the four red-airbase coordinates
for a fixed seed, making one claim falsifiable: **the generator's** placement of KNOWN
targets is a function of the base coordinates and the rng stream only, never of an
aircraft's own position.

B2 is a different layer. Hidden-target placement is deliberately a function of the SOLVED
routes, and therefore of the fleet's composition. That is not a P6 violation and P6 must not
be read as forbidding it — reading it that way leads straight back to the generator-side
shortcut this phase rejected. If P6 ever fails, stop and report the two coordinate lists;
never edit the test or the generator to make it pass.

### Scenario and data seam

- `solution` maps agent ids to positional task assignments. `Task` and `Step` are portable
  pure data; `Agent` objects are bound to the environment and must be re-extracted from
  environment 2.
- The generator writes a JSON file and returns its **path**, while `setup_episode` takes the
  JSON **content** as a string. That is where the patch step belongs: between them the
  scenario is just a dict.
- The base template is minified. Preserve its representation and use the generator's
  existing exact-replacement / deep-copy idioms; do not round-trip the whole template
  through a formatting rewrite.
- Append hidden red-airbase targets to the existing scenario so known-target positions and
  indices remain stable.
- Reload the patched world and derive oracle tasks from it. Do not trust an independently
  constructed task list.
- Patch once; never regenerate after solving `A_init`, because regeneration changes ids and
  fleet state.

### Reproducibility

- New placement code must accept an explicit `random.Random`. **Satisfied in B2:**
  `place_hidden_targets` requires one and raises on anything else; the RNG draw order is
  fixed and documented (leg choice → fraction → offset).
- Generated UUIDs are not seed-derived (`CLAUDE.md` §8). Compare reproducibility by a
  geometric fingerprint such as `(latitude, longitude, utility)`, never by target id.
  **Satisfied in B2:** `geometric_fingerprint` returns coordinates only, and egos are
  iterated in sorted id order so the solution dict's insertion order cannot change the
  result. B3 must keep this discipline when it patches targets into the scenario JSON.

### Constraints already measured

- **The `95c09dd` measurements are PRE-launch-point-fix and are not guaranteed to
  reproduce.** `CLAUDE.md` §7 records that the fix made `round_trip_cost` a symmetric
  out-and-back instead of a launch→target→base triangle, so a given seed MAY now yield a
  different allocation. That puts `std(R) = 0.1443`, `adv_std_raw = 0.1197` and the clean
  11/12 split outcome on the wrong side of the fix. `U_oracle = 480` is unaffected (it is a
  sum of utilities, not an allocation), so the cell stays comparable — but any baseline
  expectation for B4 must be re-measured, not inherited.
- **RESOLVED by B1 (`d6758ac`).** The pre-B1 50 km floor (== the sensing radius,
  measured from the launch point) put the fixed P6 fixture's easy targets only 58.8 km
  and 63.2 km out, and Layer 1 clustering pulled the same fixture's known pairs to 13.7
  km and 28.9 km apart — both destroyed the mid-route pop-up semantics this phase
  depends on. The strict B1 construction path now enforces a TRUE great-circle
  `min_target_distance_km=200` km floor and `min_known_separation_km=100` km known-target
  separation (`build_variation_config`, `VariationConfig.strict_geometry=True`), and
  disables Layer 1 entirely on that path (`ensure_discovery_chain=False`). Legacy
  non-strict generator callers are unaffected. Recorded in `CLAUDE.md` §7/§8.
- **RESOLVED by B1.** `RolloutConfig` no longer diverges from `TrainConfig`: it mirrors
  the same reference-cell fields field-for-field and validates them the same way, as
  `run_rollout`'s first statement, before any directory, policy, generator, or BLADE
  import. Diagnostic rollouts and training runs build the same default world.
- **Measured pre-B3 (expected, not a regression).** A live default-cell rollout episode
  completed with 0 wakes and `reward=+0.0000`: one agent per known target, no hidden
  target to discover, so the static known-only plan already achieves the oracle. **B4
  must NOT use this known-only result as its learning baseline.**
- Solver runs with `known ≤ 2` can stall for roughly 15 minutes against a typical 45 s
  (`CLAUDE.md` §8). Do not use such training configurations without a timeout.
- With the discovery-chain pass disabled, its seven statistics keys are absent; access them
  with `.get`, never `[...]`.

## 4. Work sequence

### B1 — generator and configuration — **CLOSED / MERGED / LOCKED**

Reviewed code SHA `d6758ac1899621b2ceebcb63afb5e8577184cd91`, integrated into `main` by
merge commit `bd087c3c18b96f1fe847b4987c73f394a43249c1` (PR #2, merged).

Delivered: explicit `num_agents` / `n_known` / `n_hidden` on `TrainConfig` (mirrored
field-for-field on `RolloutConfig`); a configurable `min_target_distance_km` (200 km) and
`min_known_separation_km` (100 km), both enforced as a TRUE great-circle floor via
`VariationConfig.strict_geometry` / `min_target_separation_km`; known-only emission with
Layer 1's discovery-chain pass disabled on the construction path only
(`ensure_discovery_chain=False`); propagation through `TrainConfig`, `RolloutConfig`, and
CLI without hard-coded duplicate defaults; `RolloutConfig.validate()` aligned with
`TrainConfig.validate()`'s construction checks. `derived_split`, `split_preview`,
`split_tasks`, and their tests are untouched and green — the construction path simply
stops consulting them; retiring them is still a separate, later phase.

Deliberately NOT delivered: hidden-target placement. B1 emits ZERO hidden targets; that
remains B2/B3 work.

Evidence: suite 64 → 84, import purity 12/12, module selftests and the bonmin selftest
green under `nlp_env`.

**B2 is now the next phase (below).**

### B2 — route-relative hidden-target placement — CLOSED / MERGED / LOCKED

Reviewed code SHA `e22aee359e06591bdb179ef06a566db90f83a558`, merged by
`8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3). `CLAUDE.md` §7 records the lock under
`e22aee3` and §6 points at the module; that entry is the technical authority, so this one
stays short.

Delivered: `rl/training/graph_hidden_placement.py` — a PURE layer
(`PlacementParameters`, `HiddenPlacement`, `predict_route`, `place_hidden_targets`,
`validate_placement`, `geometric_fingerprint`, `HiddenPlacementError`) with no BLADE,
gym/gymnasium, torch, solver, `setup_episode`, file-I/O or global-RNG dependency.
`(solution, belief_tasks, launch_point, parameters, rng) → hidden coordinates + audit
metadata`, **one placement per non-empty ego route**, egos iterated in sorted id order.
`detection_km` arrives through parameters — it is deliberately NOT imported from
`graph_episode_setup`.

Closed geometry (identical to §5): `f ~ Uniform[0.60, 0.85]` over `G = L - D`;
`guard_km = 10`; leg-1 cap `D - guard`; later-leg origin uncertainty `(1 - s/L)·D` and cap
`D - guard - origin_uncertainty`; strict later-leg `gap > 2·D` with equality rejected and
one-remaining-candidate passing trivially; uniform choice among eligible later legs with
valid leg 1 as fallback; loud failure plus independent geometric re-validation of every
returned placement.

Two review fixes are part of the locked behaviour. **F1:** assignment fields require
genuine `numbers.Integral` values — `bool` and coercible non-integral values (fractional
floats, integral-valued floats, numeric strings) all raise, because `int(...)` had silently
accepted `(0.9, 0, 0)` AS `(0, 0, 0)` and thereby changed the predicted route. **F2:**
`validate_placement` checks the recorded required tie margin on EVERY later leg before
branching, closing a hole where a `single_candidate` record could skip the requirement.

Evidence on the integrated merge: **18** focused B2 tests, import purity **12/12**, full
suite **102**, `git diff --check` clean, and all 18 B2 tests + all 12 import-purity entry
modules green through the `nlp_env` `__main__` runners.

**Nothing consumes this layer yet.** It is not connected to scenario patch/reload or to
`setup_episode` — that is B3, the next task.

### B3 — setup seam — NEXT TASK

Integrate solve → place → patch → reload into `setup_episode`, re-extract environment-2
agents, build the full oracle, and preserve the current `graph_rollout` caller. It consumes
the locked B2 API; it does not reimplement placement geometry.

Grade A because this touches a locked layer. Prove that a hidden world target enters only
the belief of the ego that privately sensed it, and add one end-to-end episode smoke test.

Note a structural side effect worth recording when it lands: `CLAUDE.md` §8 flags that
`setup_episode` does not guard `split_meta["outcome"]`, so an undiscoverable hidden target
can pass silently. On the construction path `split_tasks` is not called at all and discovery
is guaranteed by geometry, so that hazard disappears by construction rather than by a guard.
It still applies to the legacy path.

### B4 — training run

Only after B1–B3 are merged and verified. First add a small provenance manifest containing
the code SHA, command/config, seeds, environment, and solver versions. Note the scale: two
bonmin solves per episode at roughly 45 s each, so a few hundred episodes is 10–15 hours of
wall clock. Watch the first two iterations before committing to the full run — the loop has
never run to completion.

Read saturation carefully, because the two directions look alike and mean opposite things.
Near-ceiling held-out performance **at iteration 0** means the cell has no learning headroom
and must be revisited before spending compute (`CLAUDE.md` §8). Reward saturating near the
ceiling **after training** is expected and intended in this phase: with guaranteed discovery
and no resistance, egos should collect nearly everything. Difficulty returns later through
`p < 1`, `fuel_damage`, and targets that shoot back — not by weakening this phase's scenario.

## 5. B2 placement decisions — CLOSED

All four questions that were open before B2 are closed, implemented, and locked in
`e22aee3`. Nothing here awaits user confirmation:

1. **Fraction range** — `f ~ Uniform[0.60, 0.85]` sampled over the GUARANTEED portion
   `G = L - D`, so the projection sits at `s = f·G` from the predicted leg origin and is
   biased away from the common star origin.
2. **Offset guard** — `guard_km = 10`. Leg 1's origin is the exact launch point, so its cap
   is `D - guard` (40 km at `D = 50`). A later leg's origin is the previous target's
   VICINITY, so it budgets residual origin uncertainty `(1 - s/L)·D` and caps the signed
   offset at `D - guard - origin_uncertainty`; its whole approved fraction interval must
   also project beyond the uncertain origin vicinity, so eligibility never depends on the
   fraction actually drawn. The offset is sampled symmetrically — both sides of the route
   occur.
3. **Tie margin** — legs 2+ require the STRICT condition `gap > 2 × DETECTION_KM`
   (100 km at `D = 50`); **equality is rejected**. One remaining located assignment passes
   trivially, because no competitor can reverse the ordering. Leg 1 needs no margin at all:
   prediction and execution start from the same launch point.
4. **Eligible legs** — choose UNIFORMLY among all eligible later legs using the supplied
   rng; fall back to a valid leg 1 when none qualifies; raise when leg 1 is invalid too. A
   failed margin is never weakened and the coordinate is never moved onto an unstable leg.

Also closed: **exactly one placement per non-empty ego route** is the current B2 contract.
A general `n_hidden != number of usable ego routes` distribution policy is a SEPARATE future
design task — it is not solved, and B3 must not assume it is.

## 6. Closed decisions

- Offline construction only; no live world mutation and no injection scheduler.
- Patch-and-reload, never regenerate after solving.
- Route prediction is required and must support `num_agents < n_known`.
- One sensing/arrival/attack/kill-confirmation radius: `DETECTION_KM = 50`.
- Acceptance for this phase is structural correctness and feasibility, not final
  learnability.
- No full training run before B1–B3 close.
- `round_trip_cost` and `graph_reward` remain unchanged and frozen. The flat `R ≈ −1/3` was
  scenario degeneracy, never a reward bug.
- Random fleet class mix is acceptable; use relative placement parameters.
- **B1 reference cell (`d6758ac`):** fleet 3, `n_known = 3`, `n_hidden = 3` planned
  (comparable with `U_oracle = 480`) — a cell, not a law; a later phase varies the counts
  per episode.
- **B1 construction geometry (`d6758ac`):** `min_target_distance_km = 200` km and
  `min_known_separation_km = 100` km, both enforced as a TRUE great-circle floor under
  `strict_geometry`.
- **Known-only emission during B1 (`d6758ac`):** exactly `n_known` targets generated,
  `n_hidden` PLANNED only, Layer 1's discovery-chain relocation disabled on this path.
- **`RolloutConfig` follows `TrainConfig`'s construction surface (`d6758ac`):** the same
  reference-cell fields, field-for-field, validated the same way — no shared import
  (structurally aligned, compared by an anti-drift test).
- **B2 placement geometry and eligibility (`e22aee3`):** the four decisions enumerated in
  §5 — fraction range, guard and offset caps, the strict `gap > 2·D` tie margin, and
  uniform-among-eligible-later-legs with leg-1 fallback — plus one placement per non-empty
  ego route, an explicit `random.Random`, and id-free geometric fingerprints.
- **B2 stays pure (`e22aee3`):** the placement layer takes `detection_km` through
  `PlacementParameters` and never imports `graph_episode_setup`, so it carries no BLADE,
  gym/gymnasium, torch, solver, setup, file-I/O or global-RNG dependency. B3 connects it;
  B3 must not push those dependencies back into it.

## 7. Out of scope

Do not mix this phase with:

- retiring `split_tasks`, `derived_split`, `split_preview`, or Layer 1;
- legacy import cleanup or deletion;
- changes to `round_trip_cost`, the solver, reward, or reachability mask;
- centralized critic, dense reward, `p < 1`, ETA/peer-dropout work, or fuel damage;
- the README rewrite.

## 8. Documentation duties, with their trigger

Each of these is correct today and becomes wrong at a specific moment. Do them then, not
before, and not later.

| Trigger | Duty |
|---|---|
| B1 lands — **DONE** | update `CLAUDE.md` §6's "change the training scenario cell" row — completed in the B1 documentation lock commit |
| B1's parameterization closed — **DONE** | retire the `min_target_distance_km` item in `CLAUDE.md` §8 — completed in the B1 documentation lock commit |
| B2 lands — **DONE** | add the `CLAUDE.md` §6 placement row and the §7 `e22aee3` lock, and refresh §8's "Pre-B3 zero-headroom" item — completed in the B2 documentation lock commit |
| B3 lands | rewrite `CLAUDE.md` §4's pipeline diagram — it still shows `split_tasks (partial ⊊ full)`, which is correct until then; and mark the `split_meta["outcome"]` §8 item as legacy-path-only |

## 9. Next action

B1 and B2 are closed; **B3 — the setup seam — is next.** No B2 decision remains open (§5),
so there is nothing to confirm with the user before dispatch. The next orchestrator:

1. performs fresh exact-SHA initialization — resolve the current `main` HEAD (this
   documentation commit, not the B2 reviewed or merge SHA) and declare it as the base SHA,
   and select the transport mode (`GPT_GITHUB` or `CLAUDE_MOUNTED_MAIN`) per `CLAUDE.md`
   §1; the procedure lives there, do not duplicate it here;
2. conducts focused B3 recon — `setup_episode`, `graph_rollout`'s caller contract, the
   scenario-JSON patch surface, and the locked B2 API it will consume;
3. designs and dispatches the Grade-A setup-seam task (Grade A: it touches a §5 locked
   layer);
4. integrates solve → place → patch → reload;
5. re-extracts environment-2 agents and tasks after the reload;
6. builds the full oracle used by the reward denominator;
7. preserves the independent per-ego beliefs and the existing `graph_rollout` caller;
8. proves private sensing isolation — a hidden world target enters ONLY the belief of the
   ego that privately sensed it — plus one end-to-end episode smoke test.

B4 (training run) stays blocked until B3 closes and is verified.
