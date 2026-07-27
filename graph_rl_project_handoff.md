# Multi-Agent Graph RL — Scenario-Construction Handoff

**Supersedes all earlier handoffs.**

Written 2026-07-28. Baseline code SHA: `384845b19805a29920d26e495b88451ca2a5b900`
(the commit that lands this document changes documents only — no code, no test delta).

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
  - **Claude** uses the mounted `main` state and the Git evidence CC reports first. Task
    branches and PRs must not be assumed accessible.
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
  handoff declares only a grade per task; the definition belongs there.

## 1. Current state

- Last reviewed code SHA: `384845b19805a29920d26e495b88451ca2a5b900`. That is the last
  reviewed **code baseline** — it is **not** the base SHA for the next task. Docs-only commits
  have advanced `main` since, so every implementation task must resolve the current full
  `main` SHA immediately before dispatch and declare THAT as its base. Active task branch /
  PR: none.
- Verified tests reported for that baseline: suite 64, import purity 12/12, module
  selftests, and the bonmin selftest under `nlp_env`.
- Phase-A trainer exists, but **no full training run has ever been performed.** This is
  deliberate: close scenario construction first.
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

- New placement code must accept an explicit `random.Random`.
- Generated UUIDs are not seed-derived (`CLAUDE.md` §8). Compare reproducibility by a
  geometric fingerprint such as `(latitude, longitude, utility)`, never by target id.

### Constraints already measured

- **The `95c09dd` measurements are PRE-launch-point-fix and are not guaranteed to
  reproduce.** `CLAUDE.md` §7 records that the fix made `round_trip_cost` a symmetric
  out-and-back instead of a launch→target→base triangle, so a given seed MAY now yield a
  different allocation. That puts `std(R) = 0.1443`, `adv_std_raw = 0.1197` and the clean
  11/12 split outcome on the wrong side of the fix. `U_oracle = 480` is unaffected (it is a
  sum of utilities, not an allocation), so the cell stays comparable — but any baseline
  expectation for B4 must be re-measured, not inherited.
- The current easy-zone floor equals the sensing radius. In the fixed P6 fixture, easy
  targets appeared only 58.8 km and 63.2 km from launch.
- Layer 1 clustering pulled the fixture's target pairs to 13.7 km and 28.9 km separation,
  which destroys route diversity. Known-only generation must disable that pass and add a
  minimum pairwise separation. Both consequences are also recorded in `CLAUDE.md` §8.
- Solver runs with `known ≤ 2` can stall for roughly 15 minutes against a typical 45 s
  (`CLAUDE.md` §8). Do not use such training configurations without a timeout.
- `graph_rollout.py` still carries `(3,3)` + `PARTIAL_RATIO` defaults that differ from
  `TrainConfig`, and it is the SECOND caller of `setup_episode`; preserve its calling
  contract when changing `setup_episode`. Whether the harness should follow the trainer is
  an open decision (§5.4), not a silent fix.
- With the discovery-chain pass disabled, its seven statistics keys are absent; access them
  with `.get`, never `[...]`.

## 4. Work sequence

### B1 — generator and configuration

Implement:

- explicit `n_known` and `n_hidden`;
- configurable fleet size with `num_agents ≤ n_known`;
- a higher minimum target distance;
- minimum pairwise separation between known targets;
- known-only emission with the discovery-chain pass disabled;
- propagation through `TrainConfig` and CLI without hard-coded duplicate defaults.

**Scope boundary — read before dispatching.** This is not "add two integers". `derived_split`
is woven through `graph_train` in roughly eight places: the function itself, `split_preview`,
`validate()`'s hazard warnings, the startup echo, `run_config.json`, and several docstrings.
B1 **adds** `n_known` / `n_hidden` and stops *using* the derived split on the construction
path; it does **not** delete `derived_split`, `split_preview`, or the test that enforces
`derived_split ≡ split_tasks`, all of which must stay green (§7). Retiring them is a separate
phase.

Grade B, except for any change that alters a research invariant. Require one main-path test
and one test of the separation constraint.

### B2 — route-relative hidden-target placement

Create a pure placement layer with no BLADE or torch dependency:

`(solution, belief_tasks, launch_point, parameters, rng) → hidden coordinates + metadata`

It must:

- derive every eligible route polyline through `nearest_neighbor_order`;
- enforce the later-leg tie margin;
- place at a relative fraction along the guaranteed portion of a leg, biased away from the
  common origin;
- use a perpendicular offset no larger than `DETECTION_KM - guard`;
- validate the constructed scenario before simulation and fail loudly.

Keep every parameter relative, never absolute: route lengths vary widely because the fleet's
range mix varies per episode.

Grade A. Proof obligations:

1. Every placement lies within the sensing bound and its closest approach lies inside the
   guaranteed portion of its reference leg.
2. Synthetic multi-target solutions prove legs 2 and 3 plus tie-margin rejection/fallback.
   With a 1:1 first cell (§5.1) this path is never exercised at runtime, so the unit tests
   are the only proof it works.
3. Identical seeds produce identical geometric fingerprints.

### B3 — setup seam

Integrate solve → place → patch → reload into `setup_episode`, re-extract environment-2
agents, build the full oracle, and preserve the current `graph_rollout` caller.

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

## 5. Decisions to close before implementation

1. Confirm the first reference cell. Recommendation: fleet 3, `n_known = 3`, `n_hidden = 3`
   for comparability with `U_oracle = 480`; prove multi-target routes synthetically in B2
   rather than changing the first run cell. Whatever is chosen is **a cell, not a law** —
   never hard-code the counts, because a later phase varies them per episode.
2. Choose numerical construction parameters:
   - `min_target_distance_km`;
   - minimum known-target separation;
   - placement fraction range along a leg;
   - perpendicular-offset guard;
   - nearest-neighbor tie margin (order of `2 × DETECTION_KM` was the suggested starting
     magnitude, not a decision).
3. Confirm eligible legs. Recommendation: all legs that pass the margin, with leg 1 as the
   fallback.
4. Decide whether `graph_rollout`'s `RolloutConfig` follows `TrainConfig` once B1 changes the
   configuration surface. Today they diverge, which makes diagnostic rollouts and training
   runs non-comparable by default (`CLAUDE.md` §8). Deciding "not yet" is fine; deciding by
   accident is not.

Do not dispatch B1 until the B1-relevant numerical values are closed. B2-only placement
parameters may be closed after B1 if its interfaces do not hard-code them.

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
| B1 lands | update `CLAUDE.md` §6's "change the training scenario cell" row — it still names `num_red_airbases` / `partial_ratio` / `derived_split` |
| B1's parameterization closes | retire the `min_target_distance_km` item in `CLAUDE.md` §8 |
| B3 lands | rewrite `CLAUDE.md` §4's pipeline diagram — it still shows `split_tasks (partial ⊊ full)`, which is correct until then; and mark the `split_meta["outcome"]` §8 item as legacy-path-only |

## 9. Next action

Close the B1 numerical decisions in §5.2 using the evidence in §3, then dispatch B1 as one
bounded task: resolve the fresh `main` HEAD and declare it as the base SHA, select the
declared orchestrator transport mode (`GPT_GITHUB` or `CLAUDE_MOUNTED_MAIN`), and use the
corresponding transport defined in `CLAUDE.md` §1. The procedure lives there — do not
duplicate it here. A status block is required in either mode.
