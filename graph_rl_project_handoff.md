# Multi-Agent Graph RL — Scenario-Construction Handoff

**Supersedes all earlier handoffs.**

Written 2026-07-27. Remote baseline verified:
`main @ 384845b19805a29920d26e495b88451ca2a5b900`.

This handoff describes the next phase only. Technical contracts and frozen layers live
in `CLAUDE.md` at the same SHA; code and tests remain decisive.

## 1. Current state

- Last reviewed code SHA: `384845b19805a29920d26e495b88451ca2a5b900`.
- Active code task / branch / PR: none.
- Verified tests reported for that baseline: suite 64, import purity 12/12, module
  selftests, and bonmin selftest under `nlp_env`.
- Phase-A trainer exists, but no full training run has been performed. This is deliberate:
  close scenario construction first.
- Workflow-document migration is complete only when this handoff and the
  candidate-commit/Git-transport version of `CLAUDE.md` are present together on `main`.
  Project Sources are bootstrap copies only.

## 2. Phase goal

Construct scenarios in which a hidden target appears **mid-route, before the planned
engagement**. The research event is that one ego privately senses an opportunity and must
choose whether to continue its assignment, engage the pop-up, or abort. There is no
communication and no live injection scheduler.

The required pipeline is:

1. Generate a world containing only known targets.
2. Build environment 1, reset it, and extract the agents and known tasks.
3. Solve the known set to produce `A_init` and `belief_tasks`.
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

- `nearest_neighbor_order` is pure and deterministic. Starting at the launch point, it
  repeatedly selects the closest remaining assignment, tie-breaking by
  `(task_idx, step_idx)`.
- Offline prediction must call the same function used by the executor. A shortcut that
  assumes one target per ego is invalid: with several assigned targets, the real route is a
  polyline such as `base → A → B → C`; `base → B` is not a flown segment.
- Leg 1 is exact. Later legs may change only under near-ties because live replanning begins
  up to `DETECTION_KM` short of the previous target.
- Later legs therefore require a safe nearest-neighbor margin. If the margin fails, reject
  the geometry or fall back to leg 1.

### Segment geometry

- All egos launch from the BLUE airbase. Hidden targets near the common origin may be sensed
  by several egos almost simultaneously, so placement should favor the far half of a leg.
- The guaranteed-flown part of a leg ends `DETECTION_KM` before its target. Placement and
  validation must use that guaranteed portion, not the full ideal segment.
- Geometry determines discovery timing. Do not add a scheduler.
- Keep the unified `DETECTION_KM = 50 km` contract.

Reference aircraft values:

| Aircraft | Effective one-way range | Reference speed |
|---|---:|---:|
| F-16 | 400 km | ≈ 2413 km/h |
| F-35A | 900 km | ≈ 1932 km/h |
| B-2 | 1500 km | ≈ 1046 km/h |
| KC-135R | 2100 km | ≈ 854 km/h |

`MAX_SIM_TICKS = 14400` seconds.

### Scenario and data seam

- `solution` maps agent ids to positional task assignments. `Task` and `Step` are portable
  pure data; `Agent` objects are bound to the environment and must be re-extracted from
  environment 2.
- The base template is minified. Preserve its representation and use the generator's
  existing exact-replacement/deep-copy idioms; do not round-trip the whole template through
  a formatting rewrite.
- Append hidden red-airbase targets to the existing scenario so known-target positions and
  indices remain stable.
- Reload the patched world and derive oracle tasks from it. Do not trust an independently
  constructed task list.
- Patch once; never regenerate after solving `A_init`, because regeneration changes ids and
  fleet state.

### Reproducibility

- New placement code must accept an explicit `random.Random`.
- Generated UUIDs are not seed-derived. Compare reproducibility by a geometric fingerprint
  such as `(latitude, longitude, utility)`, not by target id.

### Constraints already measured

- The current easy-zone floor equals the sensing radius. In the fixed P6 fixture, easy
  targets appeared only 58.8 km and 63.2 km from launch.
- Layer 1 clustering pulled known-target pairs to 13.7 km and 28.9 km separation, which
  destroys route diversity. Known-only generation must disable that pass and add a minimum
  pairwise separation.
- Solver runs with `known ≤ 2` can stall for roughly 15 minutes. Do not use such training
  configurations without a timeout.
- `graph_rollout.py` still has defaults that differ from `TrainConfig`; preserve its calling
  contract when changing `setup_episode`.
- With the discovery-chain pass disabled, its statistics keys are absent; access them with
  `.get`.

## 4. Work sequence

### B1 — generator and configuration

Implement:

- explicit `n_known` and `n_hidden`;
- configurable fleet size with `num_agents ≤ n_known`;
- a higher minimum target distance;
- minimum pairwise separation between known targets;
- known-only emission with the discovery-chain pass disabled;
- propagation through `TrainConfig` and CLI without hard-coded duplicate defaults.

Grade B except for any change that alters a research invariant. Require one main-path test
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

Grade A. Proof obligations:

1. Every placement lies within the sensing bound and its closest approach lies inside the
   guaranteed portion of its reference leg.
2. Synthetic multi-target solutions prove legs 2 and 3 plus tie-margin rejection/fallback.
3. Identical seeds produce identical geometric fingerprints.

### B3 — setup seam

Integrate solve → place → patch → reload into `setup_episode`, re-extract environment-2
agents, build the full oracle, and preserve the current `graph_rollout` caller.

Grade A because this touches a locked layer. Prove that a hidden world target enters only
the belief of the ego that privately sensed it, and add one end-to-end episode smoke test.

### B4 — training run

Only after B1–B3 are merged and verified. First add a small provenance manifest containing
the code SHA, command/config, seeds, environment, and solver versions. Watch the first two
iterations before committing to the full 10–15 hour run. If held-out evaluation is already
near the oracle ceiling at iteration 0, revisit the scenario cell before spending compute.

## 5. Decisions to close before implementation

1. Confirm the first reference cell. Recommendation: fleet 3, `n_known = 3`,
   `n_hidden = 3` for comparability; prove multi-target routes synthetically in B2 rather
   than changing the first run cell.
2. Choose numerical construction parameters:
   - `min_target_distance_km`;
   - minimum known-target separation;
   - placement fraction range along a leg;
   - perpendicular-offset guard;
   - nearest-neighbor tie margin.
3. Confirm eligible legs. Recommendation: all legs that pass the margin, with leg 1 as the
   fallback.

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
- `round_trip_cost` and `graph_reward` remain unchanged and frozen.
- Random fleet class mix is acceptable; use relative placement parameters.

## 7. Out of scope

Do not mix this phase with:

- retiring `split_tasks`, `derived_split`, `split_preview`, or Layer 1;
- legacy import cleanup or deletion;
- changes to `round_trip_cost`, the solver, reward, or reachability mask;
- centralized critic, dense reward, `p < 1`, ETA/peer-dropout work, or fuel damage;
- the README rewrite.

## 8. Next action

After the workflow-document migration is merged, start a fresh orchestrator from the new
`main` SHA. Its first substantive task is to close the B1 numerical decisions using the
evidence above, then dispatch B1 as one bounded task.
