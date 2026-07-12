# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## 1. Communication & Workflow (read first)

- **User speaks Hebrew.** Code, comments, and docs stay in English.
- **Work step-by-step.** One file / one change at a time. Explain the reasoning *before* editing, then wait for feedback.
- **No premature documentation.** Don't create `README.md`, `INDEX.md`, `SUMMARY.md`, or per-file docs without being asked. One consolidated doc per major component, only after it's stable.
- **Minimal files.** Prefer extending an existing module over spawning `foo_utils.py` + `foo_config.py` + `foo_helpers.py`.
- **Macro constants for feature toggles.** New toggleable features mirror the pattern at the top of `train_full.py` (`VARY_SCENARIOS`, `FUEL_DAMAGE_ENABLED`, `VALIDATE_EVERY`) — a module-level constant *and* a CLI flag.
- **Git workflow:** per-task commits, `git add .` (not per-file staging), single `git push` at session end. Some sessions are explicitly local-only (no git operations) — respect that constraint when stated in the task prompt.
- **Environment:** Windows + PyCharm terminal + `nlp_env` conda env, Python 3.10+. Avoid POSIX-only shell idioms (`rm -rf`, `&&` chains) — use Python or PowerShell equivalents.
- **🛑 ALWAYS run solver commands via `conda run -n nlp_env ...`.** Any command that invokes the MINLP solver (`train_full.py`, or anything that calls `bonmin`) **must** run inside `nlp_env`, e.g. `conda run -n nlp_env python train_full.py ...`. The `anaconda3` **base** env lacks `bonmin`, and `train_full.py` then **fails SILENTLY**: the episode aborts (`!CRASH ep0001`, `Total episodes: 0`) but the process still **exits 0** — there is no visible error and no nonzero exit code. **Never trust the exit code alone — always verify `Total episodes` > 0** (and no `CRASH`/`Traceback` in the log) before claiming a run succeeded.
- **Run all commands from the repo root.**

---

## 2. Critical: What NOT to touch without explicit discussion

### 🛑 BLADE engine (vendored Panopticon fork)

`Game.py`, `Scenario.py`, `Side.py`, `blade.py`, `weaponEngagement.py`, `Airbase.py`, `Aircraft.py`, `Facility.py`, `Weapon.py`, `Ship.py`, `ReferencePoint.py`, `PlaybackRecorder.py` — **frozen copy**. Do not refactor, reformat, or "improve" them. If the API we use changes, discuss the upgrade path explicitly first.

**Known deliberate workarounds (treat as load-bearing):**
- `CHARACTER_LIMIT` override at the top of `train_full.py` (500MB recordings).
- `game.current_scenario.name` set manually before `game.start_recording()` (e.g. `f"ep{N:03d}_rl"`) — otherwise recordings are named `"New Scenario"`.

**Additive `Game.py` edits for the Phase-2 graph executor (backward-compatible):**
- `Game.handle_aircraft_attack` now accepts an optional 2-arg form (`handle_aircraft_attack(aircraft_id, target_id)`): `weapon_id` defaults to the aircraft's highest-engagement-range weapon (`get_weapon_with_highest_engagement_range()`), `weapon_quantity` defaults to `2`. Existing 4-arg callers (`plan_editor.py`, `blade_executor_minimal.py`) are byte-identical in behavior.
- `Game.launch_aircraft_from_airbase` now accepts an optional `aircraft_id` for targeted launch (find by `str(ac.id)`, `remove()` then append; id absent from inventory → `return None`, never launch the wrong aircraft). Omitting it preserves the existing FIFO `pop(0)` behavior exactly.
- **Why:** the graph executor is the **sole** BLADE translation layer and calls the 2-arg attack and targeted launch; defaulting `weapon_quantity = 2` keeps the flat path's "one ATTACK step ⇒ target destroyed" invariant. Dispatch is parser-free (`Game.handle_action` runs `exec(f"self.{action}")`), so optional params are sufficient — no parser change.
- **Runtime engine (env fix, DONE):** the vendored engine under `src/match_aou/integrations/panopticon-main/gym` is now **editable-installed** into `nlp_env` (`pip install -e …/panopticon-main/gym`), replacing a stale non-editable `blade` install that predated these edits. So `import blade` always resolves to the edited vendored engine — one engine for the executor and the eventual `train_full` wiring; no `sys.path` injection needed.
- *Forward note (not an action):* the episode-start "launch all aircraft" workaround in `train_full.py` becomes removable once the executor uses targeted launch.

### 🛑 MATCH-AOU solver internals

`match_aou_MINLP_solver.py` is in its **original, advisor-approved form**. The advisor's directive is: **address allocation pathologies through scenario design, not solver constraints.** Three solver-level workarounds were tried and explicitly rolled back:

1. A hard `single_agent_per_step` constraint — worked, but too rigid.
2. A fuel penalty in the objective (`λ · Σ travel_cost · x[i,j,k]`) — worked, but solver-level.
3. Earlier attempts to force `probability = 1.0` *as a solver patch* — out of scope.

**Do not re-add any of these to the solver.** The current approach uses `probability = 1.0` at the **task-construction level** (in `scenario_factory.generate_all_enemy_tasks`, not in the solver) combined with class-based fuel tiers and easy/stretch zones. This makes the redundancy term `(1 - p + ε)^m` evaluate to ≈ 0 marginal utility for redundant agents, so the solver naturally produces 1-to-1 allocations without solver-side modifications. See sections 4 and 7.

The one **advisor-approved** change to the movement-budget constraint is the per-target **round-trip** charge (`round_trip_cost`: out + back, `risk_factor = 0`) — a fidelity correction to an existing constraint, **distinct from** the three rolled-back workarounds above (it is not `single_agent_per_step`, not an objective fuel penalty, not a probability patch). See section 4.

### 🛑 Observation-layer fuel damage

`fuel_damage.py` intentionally modifies **only the observation vector** (`obs.vector[0]` and `obs.self_state.fuel_norm`). It does **not** touch BLADE physics. This asymmetry is the learning signal: the oracle has full fuel in its plan, the RL sees reduced fuel and must learn to RTB. Do not "fix" by patching actual BLADE fuel values.

### ⚠️ Flat observation path is the frozen baseline (Phase-2 graph side-by-side)

The flat observation path (`observation_builder.py` + `observation_types.py` + `config.py`'s `ObservationConfig`) is the **active baseline** and is **frozen** while the Phase-2 graph path is being built. The graph-side layers (`graph_builder.py`, `graph_action.py`, `graph_effect.py`, `blade_graph_executor.py`, and the `graph_encoder.py` encoder) are implemented side-by-side and are **not wired into `train_full.py` yet**. Don't modify the flat path to accommodate graph work — keep the two paths separate until the graph training path is deliberately wired. See sections 3 and 8.

### 🛑 BladeExecutorMinimal invariants

`blade_executor_minimal.py` enforces three load-bearing rules:
- At most **one action per tick** (BLADE constraint).
- **Level gating** by `level_order` (precedence).
- **FIFO launch validation** against the airbase inventory.

Don't bypass any without a corresponding BLADE-side change. *(Intra-level command **order** is a greedy nearest-neighbor optimization — see §4 — and may change freely; it does not affect these three invariants or which steps execute.)*

**Domain/BLADE boundary:** domain objects (`Step`/`Task`) contain **no** BLADE command strings. `BladeExecutorMinimal` is the **sole** BLADE translation layer — it builds every command (move / launch / attack / RTB) from semantic `Step` data (`step_kind`, `target_id`, `location`) plus the solver's assignment. The attack string is built directly in the executor (no template, no placeholders); this is behavior-preserving and verified byte-identical.

### ⚠️ Phase-2 graph executor — BUILT & REVIEWED (`blade_graph_executor.py`)

`GraphPlanExecutor` (in `blade_graph_executor.py`) is **built and reviewed** but **not yet wired into `train_full.py`**. It consumes the updated `solution` from `graph_effect.apply_meta_action` and is the **sole** BLADE translation layer (move / launch / attack / RTB) for the graph path; `graph_effect` stays BLADE-free. It is **separate from** the frozen `blade_executor_minimal.py`, which remains the active flat/validation executor.

**Execution model — done-on-CONFIRMED-KILL** (replaces the earlier done-on-emit): an ego marks a target done **only** after it confirms `get_target → None` **within its own sensor range** (proximity-gated; preserves no-communication — it never learns a far target was killed by a peer). It fires **once, then waits** (loiters in range) for the engine to resolve the kill, with a per-`(ego, target)` re-fire throttle (`kill_confirm_ticks`); re-fire only matters once task `probability < 1.0` is introduced (a launch can miss). Levels are **per-agent** and read `done` directly, and `resync` swaps a single ego's plan slice without touching `done`, so a mid-episode OE/abort edit never redoes completed work and never gates on peers.

**Crash handling:** a `dead` set lets `is_done()` skip crashed egos (accept the lost task utility rather than hang); because levels are per-agent, a crash never affects peers' timing.

**RTB:** single-issue latched (BLADE's `aircraft_return_to_base` is a toggle). `SELF_PRESERVATION_ABORT` reaches the executor as an empty ego plan → RTB. *Latent invariant:* the latch is safe only while the BLUE doctrine `AIRCRAFT_RTB_WHEN_OUT_OF_RANGE` is **off** (it is, in `strike_training_4v5.json`) — otherwise the engine toggles `rtb` on bingo fuel behind the latch's back. See the inline comment.

Smoke test: `tools/graph_executor_smoke.py` (3/3 targets destroyed, launch + RTB, `is_done`). Keep the executor as the sole BLADE translation layer; BLADE engine files remain frozen — any local BLADE change requires explicit discussion.

### ⚠️ Phase-2 trigger layer — BUILT & REVIEWED (`action/graph_trigger.py`)

`decide_triggers` is the **WHEN gate**: the RL is NOT run on a periodic scan — each ego flies the static plan A_init "blind" via the executor until an EVENT wakes the policy. Two event kinds, both from the ego's OWN sensing (`sensed_target_ids`):
- **POP-UP** — a sensed enemy id absent from `belief_tasks` → append a pop-up `Task` (`make_attack_task`) to the **append-only** `belief_tasks` (`task_idx` is positional and indexes `solution` tuples, so tasks are never removed/reordered). It is **not** added to `belief_solution` — that is the policy's job via `graph_effect` `OPPORTUNISTIC_ENGAGEMENT`.
- **PEER-OVERDUE** — a sensed target A_init assigned to a peer, past its ETA → remove that peer's tuple(s) from **this ego's** `belief_solution` copy, making the target unassigned+sensed (pop-up-like) for the policy. This is the "remove the peer edge" half of the retired Cooperative-Recovery meta-action; the policy + `graph_effect` do the "add ego edge" half.

**Per-ego private belief (no-communication).** Each ego reasons over a **private** copy of `(tasks, solution)`, == A_init at t=0, edited ONLY from its own sensing. `decide_triggers` (like `graph_effect.apply_meta_action`) is **pure**: new copies, no global, never reads a peer's runtime state; peer-overdue removes only from the acting ego's belief. The executor commands each agent from its own belief-slice via per-ego `resync(ego_id=...)`, so ego A's pop-up/takeover is invisible to ego B. The orchestrator that holds the N private beliefs — the only place that would bridge egos — **is not built yet**.

**ETA is a placeholder.** `never_overdue` (+inf) keeps PEER-OVERDUE structurally present but **DORMANT**; the real ETA (haversine distance / cruise speed / task ordering / levels) is a deliberate later effort. Only the injected `eta(peer_id, task_idx)` seam exists now.

`_selftest` (`env PYTHONPATH=src python -m match_aou.rl.action.graph_trigger`) proves the no-comms / append-only red lines: pop-up is append-only, peer-overdue removes only the overdue peer's tuple (a co-resident ego entry is untouched), the dormant default ETA fires nothing, inputs are never mutated, and a separately-held ego-B belief stays byte-identical after ego A's edit. **Not wired into `train_full.py`** (no orchestrator).

### 🛑 Legacy DQN code

Anything under `legacy/` (at repo root) is archived DQN-era code. Don't import from it, don't resurrect it. Active entry point: `train_full.py`. Active executor: `blade_executor_minimal.py`.

**Inactive modules (deletion candidates, not the active path):** `match_aou_parser.py` and `blade_plan_utils.py` have no live importers (`blade_plan_utils` is no longer exported from `blade_utils/__init__.py`). They are excluded from the WI-3 grep-zero acceptance for this reason — they are not part of the active path.

### ⚠️ Old checkpoints

Checkpoints built with a different `MAX_AGENTS` or under a different scenario-generation regime (e.g. pre-refactor with `probability=0.6`) may be incompatible with current setups. Discard or move to an `archive_*` subfolder under `training_output/`; do not silently mix with current-run outputs.

---

## 3. Key Files Map

"I want to change X" → go to Y.

### Entry points & orchestration

| I want to… | Go to |
|---|---|
| Change training loop, episode lifecycle, CLI args | `train_full.py` |
| Change the default scenario | `strike_training_4v5.json` (base template) |
| Change logging format, progress prints, action-parse regex | `_log_blade_action` / `_log_progress` in `train_full.py` |

### RL layer (`src/match_aou/rl/`)

Subpackages: `observation/`, `action/`, `agent/`, `training/`. `plan_editor.py` and `shared_utils.py` sit at the `rl/` root.

| I want to… | Go to |
|---|---|
| Change what the agent sees (flat feature vector — active baseline) | `observation/observation_builder.py` + `observation_types.py` + `self_features.py` + `target_extraction.py` + `plan_parsing.py` + `plan_context.py` |
| Change the Phase-2 GRAPH observation (nodes / edges / features) | `observation/graph_builder.py` → `build_graph_observation` (`GraphObservation` with `task_features[k,6]` + `agent_features[a,1]` = `[fuel_norm]` (ego real / peers `0.0`); `GraphObservationConfig`; `EdgeType`) |
| Change the Phase-2 GRAPH action / meta-action mask (decision core) | `action/graph_action.py` → `MetaAction`, `build_action_mask` (k×3 additive `{0,-inf}` mask), `ActionHead` (shared per-node MLP), `sample_action` |
| Change the Phase-2 GRAPH effect (meta-action → plan edit) | `action/graph_effect.py` → `apply_meta_action` (pure plan editor; BLADE-free) |
| Change the Phase-2 GRAPH trigger (WHEN to wake the policy) | `action/graph_trigger.py` → `decide_triggers` (pure; POP-UP appends a pop-up Task to append-only `belief_tasks`; PEER-OVERDUE removes an overdue peer's tuple from the ego's `belief_solution` copy; `TriggerKind`; injected `+inf` `never_overdue` ETA stub) |
| Change the Phase-2 GRAPH encoder (graph → per-task embeddings) | `agent/graph_encoder.py` → `GraphEncoder` (edge-masked symmetrized multi-head attention; per-relation + self-loop bias; TASK/EGO/PEER role embedding; injected `time_norm`; `forward(obs, edge_attr=None) -> [k, embed_dim]` per-task non-pooled; `pool()` critic hook) |
| Change the Phase-2 GRAPH executor / plan re-sync | `utils/blade_utils/blade_graph_executor.py` (`GraphPlanExecutor`); consumes the updated `solution` from `graph_effect`, preserves already-finished work via the `done` set, supports front-inserted levels, and is the sole BLADE translation layer for the graph path |
| Change obs dim / top_k / normalization | `observation/config.py` (`ObservationConfig`) |
| Compute travel time / fuel needed | `observation/observation_utils.py` — fuel helpers only (`target_id` is an explicit `Step` field, read directly; no action-string parsing) |
| Add / remove / change actions | `action/action_config.py` (`ActionType`), `action/action_utils.py`, `action/action_validation.py` |
| Translate a **flat-baseline** action token into a BLADE command | `plan_editor.py` (flat path only; not a reference for graph effect/executor work) |
| Change the network | `agent/network.py` (`ActorCriticNetwork`) |
| Change PPO hyperparameters | `training/ppo_trainer.py` (`PPOConfig`) — or override from `train_full.py` |
| Change the reward function | `training/reward.py` |
| Change GAE / buffer behavior | `training/rollout_buffer.py` |
| Change fuel damage events | `training/fuel_damage.py` |
| Change the oracle signal (what RL imitates) | `training/oracle.py` |
| Change per-episode setup (partial/full split, solve, auto-launch) | `training/episode_initializer.py` |

> **Phase-2 graph observation:** `observation/graph_builder.py` is the Phase-2 **graph** representation (task/agent nodes + typed edges — `GraphObservation` with `task_features[k,6]`, `agent_features[a,1]` (`[0]` fuel_norm — REAL for the ego, `0.0` for every peer), COO `edge_index`/`edge_type` over the `EdgeType` IntEnum, plus `time_norm`; configured via `GraphObservationConfig`). **No-communication is enforced structurally:** peers are featureless (peer fuel is unsensable — a real comms leak — and peer position/observation are dropped too), so the only runtime info in the graph is the ego's own sensors. The static plan A_init enters ONLY via `ASSIGNMENT` edges + the featureless peer nodes that anchor them. The agent-node set is `ego ∪ assigned same-side peers` (the currently-sensed-peer union was removed), so every peer node has ≥1 `ASSIGNMENT` edge by construction and `ASSIGNMENT` edges still cover the **complete static allocation** ("no `ASSIGNMENT` edge ⇒ genuine pop-up"). The ego's runtime **sensing** now lives in the `sensed` task-feature column (`task_features[:,5]`, ego-only, binary, recomputed from the ego's current position every build), **not** in a `SPATIAL` edge — `SPATIAL` construction was removed and `EdgeType.SPATIAL` is now **reserved/deferred** (kept in the IntEnum to avoid renumbering `ASSIGNMENT`/`PRECEDENCE`). `ASSIGNMENT` is the only constructed relation (plus deferred `PRECEDENCE`). Node-typing / `is_ego` is **deferred to the encoder** (`agent/graph_encoder.py`, now built — a learned TASK/EGO/PEER role embedding) — the builder stays a pure projection of `(world, solution)`. It is built **side-by-side** with the flat builder above: the flat builder remains the **active baseline**, and the graph path is **not yet wired into `train_full.py`**. It ships with an in-file `_selftest()` (`env PYTHONPATH=src python -m match_aou.rl.observation.graph_builder` under `nlp_env`). The Phase-2 action layer that consumes it is `action/graph_action.py` (see the table row above and section 8). See sections 2 and 8. **Why a `sensed` column at all (rationale now realized in the trigger layer):** because `tasks` is **append-only** (a pop-up is appended, never removed — see `graph_trigger`) and the policy selects over the WHOLE graph, `unassigned` does **not** imply `currently-sensed` — a stale pop-up that has since left range stays in `tasks`, unassigned. So `sensed` in the mask enforces **"act only on what you sense now"** = the no-communication constraint. It gates **only** `OPPORTUNISTIC_ENGAGEMENT` (which also requires `unassigned`); `PLAN_COMPLIANCE` and `SELF_PRESERVATION_ABORT` do not depend on it. It is **not** there to distinguish a pop-up from a solver-unassigned target (moot — solver-unassigned targets never enter `tasks`).

### BLADE integration (`src/match_aou/utils/blade_utils/`)

The boundary between scenario content and MATCH-AOU object construction is **strict**:

- `scenario_generator.py` owns scenario *content* — units, positions, fleet composition, fuel tiers, target placement zones, and the generation-time discovery-chain check (Layer 1).
- `scenario_factory.py` owns BLADE-observation → MATCH-AOU object conversion. Stateless. Single source of truth for `Agent` and `Task` construction. `train_full.py` imports `generate_all_enemy_tasks` directly — **no local duplicates of task/agent construction allowed in `train_full.py`**.

| I want to… | Go to |
|---|---|
| Create Agents from a BLADE scenario | `scenario_factory.py` → `create_agents_from_scenario` |
| Generate Tasks from enemy units | `scenario_factory.py` → `generate_all_enemy_tasks` (default `probability=1.0`) |
| Enumerate enemy targets / build a Task template (shared, zero-duplication) | `scenario_factory.py` → `iter_enemy_targets` (facilities→airbases→ships order, so `task_idx` is stable) + `make_attack_task` (utility by unit class: Facility 100 / Airbase 80 / Ship 95, default 80) — consumed by BOTH `generate_all_enemy_tasks` and the trigger's pop-up path |
| Expose the ego's OWN current sensing (the trigger's eyes) | `blade_graph_executor.py` → `GraphPlanExecutor.sensed_target_ids(observation, ego_id) -> {id: unit}` (world-scan via `iter_enemy_targets`; reuses the confirmed-kill distance+liveness predicate) |
| Execute a MATCH-AOU plan in BLADE | `blade_executor_minimal.py` → `BladeExecutorMinimal.next_action` |
| Change intra-level step ordering (nearest-neighbor) | `blade_executor_minimal.py` → `_build_nn_queue` / `nearest_neighbor_order` (`nn_ordering` flag) |
| Schedule/resolve plan actions | `blade_plan_utils.py` |
| Sync MATCH-AOU agents with BLADE observation | `observation_utils.py` → `update_agents_from_observation` |
| Change per-episode variation (counts, zones, fleet mix) | `scenario_generator.py` (`VariationConfig`, `ScenarioGenerator`) |
| Change fuel-based reachability rules | `ReachabilityCalculator` in `scenario_generator.py` |
| Change per-class fuel tiers | `CLASS_RANGE_TIERS` constant at the top of `scenario_generator.py` |
| Change generation-time discovery-chain logic (Layer 1) | `_ensure_discovery_chain`, `_compute_zone_bounds`, `_connect_zone_targets` in `scenario_generator.py` |
| Change split-time discovery-chain logic (Layer 2) | `split_tasks` in `train_full.py` |
| Add a new aircraft/facility class to the pool | Add JSON to `extra_template_paths` — pools auto-extract |

### MATCH-AOU solver

| I want to… | Go to |
|---|---|
| Change the objective / constraints | `match_aou_MINLP_solver.py` (with extreme caution — see section 2) |
| Change post-solve scheduling / level ordering | `scheduling_utils.py`, `topology_utils.py` |
| Change domain objects | `agent.py`, `task.py`, `step.py` (carries the semantic `StepKind` enum — a DOMAIN concept, *not* a BLADE string), `location.py`, `capability.py` |

> **Reference `.md` files in the repo** (`BLADE_API_DOCUMENTATION.md`, `MATCH_AOU_API.md`, `INTEGRATION_GUIDE.md`, `RL_MODULE_DOCUMENTATION.md`) are hand-written and may lag the code. **Prefer reading the code.**

---

## 4. Gotchas & Conventions

- **Target IDs** are an explicit semantic field on `Step` (`step.target_id`), set by `scenario_factory` to `str(unit.id)` and read directly by the executor and the observation layer. No regex, no action-string parsing — the old silent-break risk on BLADE's action format is gone.
- **Landed aircraft** are moved by BLADE from `scenario.aircraft` → `airbase.aircraft`. "All agents returned to base" is checked by this transition, not a flag.
- **Fresh UUIDs per episode.** `ScenarioGenerator` assigns new UUIDs to cloned units — never assume a unit ID persists across episodes.
- **`effort` and `Quantity` have no effect on solver allocation.** `effort` is not consumed anywhere in the MINLP. `Quantity` is only checked as a capability-name match. Multi-agent-per-target was a pure objective-function artifact under `p < 1.0` — fixed by setting `probability = 1.0` at task construction (see below).
- **`EPSILON = 1e-6` in the MINLP objective is load-bearing.** The objective contains `(1 − p + ε)^m`; without `ε`, `0^0` at `p = 1.0`, `m = 0` crashes Pyomo. Don't drop `EPSILON`, even when current scenarios use `p = 1.0`.
- **Task `probability` defaults to `1.0`** in `generate_all_enemy_tasks` — intentional anti-stacking mechanism. Marginal utility of additional agents on a task is `≈ ε^m`, effectively zero, so the solver produces 1-to-1 allocations naturally. If a future experiment needs `p < 1.0`, the single-agent-per-target behavior must come from elsewhere (currently zone-based reachability provides it via fuel asymmetry).
- **BLUE strike-weapon lethality is `1.0` (data, not code).** In `strike_training_4v5.json` every BLUE aircraft weapon (AIM-120 / AIM-9 / AGM-65) is set to `lethality = 1.0`, so a launched salvo is a guaranteed kill (eliminating the ~12% miss from BLADE's stochastic kill — `weaponEngagement.py`: `random_float(0, 1) ≤ weapon.lethality`). Under the graph executor's **done-on-confirmed-kill** model this means the confirm-guard always confirms within a few ticks (loiter ≈ weapon flight time) and the ego fires exactly **once** — which is also why the smoke's "every assigned target destroyed" holds. (Done-on-emit was the earlier shortcut, valid only at lethality 1.0; it was replaced by done-on-confirmed-kill so the executor stays correct once BLUE `probability < 1.0` is introduced.) RED SAM lethality is left **stochastic and unchanged** (`0.9 / 0.85 / 0.7`) on purpose: probabilistic loss of peers en route is exactly what drives the need for **peer-loss recovery** — now handled by the trigger layer's peer-overdue -> pop-up path (there is no CR meta-action) — so forcing RED to `1.0` would distort the threat model. `scenario_generator` clones weapon stats verbatim (it only reassigns ids / red side tags), so the scenario JSON is the **single source** of lethality. A stochastic-strike enrichment (BLUE `< 1.0`) is a deliberate future option, not enabled now.
- **Movement budget charges an explicit per-target round-trip; `risk_factor = 0`.** The constraint adds `round_trip_cost(agent, step_loc) · x[i,j,k]` (outbound `start → step` + return `step → return_location`, or `start` if unset) and caps it at `budget × (1 − risk_factor)`. `round_trip_cost` lives in `match_aou_MINLP_solver.py` and is the **single source of truth**: the validation audit (`run_validation_episode`) imports the same helper for both reachability (`round_trip_cost ≤ budget`) and per-agent `used`, so the constraint and the audit compute the **same number for a given `(agent, target)` by construction**. `train_full.solve_match_aou()` passes `risk_factor = 0.0` and the audit's `RISK = 0.0` (so `cap = budget`); the solver's `risk_factor` parameter stays in place for a possible Phase-2 `σ > 0`. The audit's one-way `target_cost` is retained only for the informational `cheapest=` display — never for reach or `used`.
- **Budget uses a star topology.** The movement-budget constraint charges each assigned target as an explicit **round-trip** (out + back, via `round_trip_cost`), summed **independently per target** — a star. (Earlier doc/code described this leg as one-way, which was inaccurate; the return leg is now charged explicitly.) No inter-target routing: conservative for multi-target chaining, exact for the current one-target-per-agent regime. Known conservative limitation, not a bug.
- **Intra-level step order is greedy nearest-neighbor (WI-4).** `BladeExecutorMinimal` builds each agent's queue by sorting on `level_order` first (topological order *between* levels is preserved), then ordering the steps *within* each level greedily by nearest target location (`nearest_neighbor_order`, haversine via `Location.distance_to`, tie-break `(task_idx, step_idx)`), chaining the end position of one level into the start of the next (the lowest level starts at the agent's location). Controlled by the `nn_ordering` constructor flag (**default `True`**); `nn_ordering=False` reproduces the legacy `(level_order, task_idx, step_idx)` sort byte-for-byte. This is a flight-path/fuel optimization only — it **does not change which steps execute**: the set of issued `(task, step)` pairs, `completed_task_steps`, and `last_attack_target_id` are unchanged; only the *order* of same-agent, same-level commands differs (move/attack targets may be reissued in a different order, and an arrival-threshold-gated move may move to a different target).
- **`top_k` is stuck at 1–3** because `ActionType` is a hand-enumerated enum. Extending beyond 3 requires edits in `action_config.py`, `action_utils.py`, `action_validation.py` — see docstring in `action_config.py`.
- **Reward is MATCH-AOU-utility-based only** — no fuel bonuses, no coverage bonuses (advisor direction). Hybrid: per-step (weight 0.3) + episode-end utility ratio (weight 0.7, scaled 5.0). See `reward.py`.
- **Critic is zero-padded to `MAX_AGENTS`.** Episodes run with variable agent counts (typically 2–3), but the critic input is always `obs_dim × MAX_AGENTS=5`. Missing slots are zeroed.
- **Discovery chain is two-layer.** Layer 1 (gen-time, in `scenario_generator._ensure_discovery_chain`) ensures per-zone connectivity at scenario generation: every target sees ≥1 radar neighbor *within its own zone*, preserving easy/stretch separation. Layer 2 (split-time, in `train_full.split_tasks`) runs zone-aware rejection sampling on the partial/full mask: every hidden target must have a known radar neighbor; isolated targets are pinned to known. Cap at 20 resamples, then warning fallback. **Both layers are necessary** — Layer 1 alone allowed connected pairs to both end up masked together; Layer 2 without Layer 1 has no anchors to resample around.

---

## 5. Dev Commands

> **🛑 Run everything below inside `nlp_env`.** Every `python train_full.py ...` command (and anything that calls the solver) must run as `conda run -n nlp_env python train_full.py ...`. The `anaconda3` base env has no `bonmin`, so the run **fails silently** — episode aborts (`Total episodes: 0`) while the process still **exits 0**. Don't trust the exit code; confirm `Total episodes` > 0 and no `CRASH`/`Traceback`. The bare `python ...` forms shown below assume `nlp_env` is already the active env.

**Install (one-time):**
```bash
pip install -r requirements.txt
# Also requires bonmin (a Pyomo-compatible MINLP solver) on PATH.
```

**Smoke test (single episode — fastest end-to-end regression check):**
```bash
python train_full.py --episodes 1 --validate-every 0 --record-every 0
```

**Unit tests under `tests/`** — `test_action_space.py`, `test_observation.py`, `RL_test_end_to_end.py`. Not wired into CI. Run individually with `python tests/<file>.py` (or `pytest tests/` if/when a pytest config is added).

**Phase-2 graph self-tests** (side-by-side graph path; run from repo root under `nlp_env`):
```bash
python -m match_aou.rl.observation.graph_builder
python -m match_aou.rl.action.graph_action
python -m match_aou.rl.action.graph_effect
python -m match_aou.rl.action.graph_trigger                 # trigger layer (no BLADE/torch)
python -m match_aou.utils.blade_utils.scenario_factory      # shared enemy-enum / Task template
python -m match_aou.utils.blade_utils.blade_graph_executor  # sensed_target_ids sensing exposure
```

**Standard baseline run (uses canonical defaults — see toggle table below):**
```bash
python train_full.py --episodes 1000
```

**Short verification run (e.g. after a refactor — frequent validation, dense recording):**
```bash
python train_full.py --episodes 50 --validate-every 10 --record-every 5 --save-freq 25
```

**Inspect a generated scenario:**
```bash
python -m match_aou.utils.blade_utils.scenario_generator strike_training_4v5.json
```

### Output layout (per training run)

```
training_output/
├── logs/                            # training log files
├── models/                          # checkpoints — preserved across runs
│   └── archive_baseline_*/          # archived historical baselines
├── scenarios/episode_{N:03d}.json   # regenerated each run
└── recordings/                       # populated per --record-every cadence
    ├── ep{N:03d}_rl.jsonl
    └── ep{N:03d}_validation.jsonl
```

`scenarios/` is regenerated each run; `recordings/` is populated only on episodes that match `--record-every`. `models/` accumulates across runs — clean it manually between major regimes (e.g. moving to a new reward function or a different `MAX_AGENTS`) and archive valuable historical baselines under `archive_*/` subfolders.

### Feature toggles (macros at top of `train_full.py`) and CLI defaults

| Constant / flag | Default | Effect |
|---|---|---|
| `VARY_SCENARIOS` | `True` | Per-episode scenario variation |
| `VARY_BASE` | `False` | Randomize blue base position |
| `INCLUDE_SAMS` | `False` | When `False`, targets are RED airbases only |
| `FUEL_DAMAGE_ENABLED` | `False` | Mid-episode fuel damage surprises (off until baseline behavior is learned) |
| `VALIDATE_EVERY` | `100` | Oracle-only validation cadence (`0` disables) |
| `DISCOVERY_SCAN_INTERVAL` | `50` | Ticks between discovery scans |
| `MAX_AGENTS` | `5` | Critic padding size (changing invalidates checkpoints) |
| `MAX_SIM_TICKS` | `14400` | Per-episode tick cap |
| `PARTIAL_RATIO` | `2/3` | Fraction of tasks in the partial set |
| `--episodes` | `1000` | Total episodes per run |
| `--save-freq` | `100` | Checkpoint cadence |
| `--validate-every` | `100` | Validation episode cadence |
| `--record-every` | `50` | BLADE recording cadence (0 = never) |
| `--max-target-dist` | `2500.0` | Max target distance from base (km) |
| `--stretch-ratio` | `0.5` | Fraction of targets in the stretch zone |
| `--min-red-airbases` | `3` | Minimum RED airbases (raised from 2 to avoid budget-pinning edge cases under `PARTIAL_RATIO=2/3`) |

---

## 6. Project Overview

MSc thesis: **Multi-Agent Task Allocation and Adaptation** (Ben-Gurion University, advisor Dr. Shahaf Shperberg).

Three layers that work together:

1. **MATCH-AOU** — MINLP optimization (Pyomo + bonmin) that generates optimal task allocations for heterogeneous agents operating *without* communication. Serves as the **oracle**.
2. **BLADE** — vendored Panopticon fork (physics-based military simulation).
3. **MAPPO RL layer** — CTDE (Centralized Training, Decentralized Execution). Learns to **adapt** the oracle plan in real time when surprise events occur mid-mission (target discovery, fuel damage).

**Research setup:** agents start with a MATCH-AOU plan based on **partial** information (~2/3 of targets). During execution they discover new targets and may be damaged. The RL learns to deviate from the oracle plan when appropriate — using a fully-informed oracle as the imitation target. Scenario: blue-side aircraft destroy red-side targets (RED airbases; SAMs optional).

**Key design principle:** allocation pathologies (stacking, fuel exhaustion, undiscoverable targets) are addressed through **scenario design**, not solver constraints. The solver stays minimal and approved; scenario-side mechanisms (fuel tiers, easy/stretch zones, two-layer discovery chain, `probability=1.0` task construction) shape what the solver sees so it produces well-behaved plans naturally.

---

## 7. Architecture

`train_full.py` is the orchestrator. Each episode: `ScenarioGenerator` produces a scenario (with class-based fuel tiers and zone-aware target placement) → `split_tasks` partitions into partial/full sets with discovery-chain validation → MATCH-AOU solves both sets → `BladeExecutorMinimal` plays the oracle plan tick-by-tick in BLADE → the MAPPO RL layer is consulted only on **event triggers** (new target discovery, fuel damage) and may override the executor's action.

**Event-driven RL loop** (inside `train_episode`):

```
for tick in range(max_ticks):
    executor_action = executor.next_action(obs, tick)    # always
    newly_damaged  = fuel_dmg.check_and_activate(tick)   # every tick
    is_scan_tick   = tick % DISCOVERY_SCAN_INTERVAL == 0 # every 50 ticks

    if is_scan_tick or newly_damaged:
        build obs → detect discoveries → if trigger:
            actor.sample(obs) → compute oracle action → reward → buffer.store()

    env.step(rl_override_action or executor_action)

# end of episode:
buffer.compute_gae(); trainer.update()  # PPO, K=4 epochs, clip=0.2, γ=0.99, λ=0.95
```

**RL decisions are purely event-driven.** No periodic NOOPs pollute the rollout buffer — decisions fire only on (a) new target discovery, or (b) fuel damage.

**Shapes at a glance:** Observation — 30 features (6 self + 18 for top-3 targets + 6 plan context), normalized to `[0,1]`. Action — 5 discrete (`NOOP`, `INSERT_ATTACK_0/1/2`, `FORCE_RTB`). Actor: `30 → 128 → 64 → 5`. Critic: `30 × MAX_AGENTS → 128 → 64 → 1` with zero-padding.

**Validation audit (in `run_validation_episode`)** silently collects per-target reachability, oracle plan, and per-attack records during the episode, then prints a compact diagnostic block: per-target line (`reach=[…] plan=[…] hit=Y/N cheapest=…`), per-agent line (`budget=B cap=B*(1-RISK) used=U/cap plan=[…]`, with `RISK=0` so `cap=B` the full budget, and `used` summed as per-target round-trips via the shared `round_trip_cost` helper), headline `Hit:` summary, and `ANOMALY:` / `Oracle plan incomplete` warnings. Lets you distinguish solver bugs from solver capacity limitations in seconds.

---

## 8. Open TODOs (priority only)

**Phase 1 — static allocation: complete.** The solver's round-trip fuel/budget accounting with `risk_factor = 0` (WI-1), the `StepKind` + `target_id` domain refactor with the executor as the sole BLADE translation layer (WI-3), and greedy nearest-neighbor intra-level execution ordering (WI-4) are all merged and verified. The remaining items concern Phase-2 RL adaptation.

**Phase 2 — graph + Transformer RL: in progress.** The RL layer is being rebuilt from the MATCH-AOU paper as a **graph + Transformer** architecture, replacing the flat observation vector. STATUS: three graph-side decision layers are done, all **side-by-side** with the frozen flat path and **not yet wired into `train_full.py`**:

- `observation/graph_builder.py` (observation layer) — `GraphObservation` + `GraphObservationConfig`. Task nodes are stable `task_idx` nodes; agent nodes are `ego ∪ assigned same-side peers` (the currently-sensed-peer union was removed). `agent_features` is `[a, 1]` = `[fuel_norm]`: REAL for the ego, `0.0` for every peer — peers are **featureless** structural anchors. This enforces no-communication structurally: peer fuel is unsensable (a real comms leak) and peer position/observation are dropped too, so the only runtime info in the graph is the ego's own sensors; the static plan A_init enters ONLY via `ASSIGNMENT` edges + the featureless peer nodes that anchor them. Every peer node has ≥1 `ASSIGNMENT` edge by construction, and `ASSIGNMENT` edges still cover the **complete static allocation**, so **"no `ASSIGNMENT` edge ⇒ genuine pop-up"** is reliable. The ego's sensing now lives in the `sensed` column `task_features[:,5]` (ego-only, binary, recomputed each build from the ego's current position), **not** in a `SPATIAL` edge. `SPATIAL` construction was removed; `EdgeType.SPATIAL` is **reserved/deferred** (kept in the IntEnum so `ASSIGNMENT`/`PRECEDENCE` keep their values). `build_action_mask` now reads `sensed` from the column (`task_features[:,5] >= 0.5`) instead of ego-origin SPATIAL edges — **mask topology unchanged (behavior-preserving)**, effect layer untouched. Node-typing / `is_ego` is **deferred to the (not-yet-built) encoder**; `task_features[k,6]` (new `[5] sensed`), ASSIGNMENT/PRECEDENCE construction unchanged. The builder is stateless: graph = projection of `(world, solution)`.

- `action/graph_action.py` (decision core) — the node-wise **k×3** meta-action mechanism. `MetaAction` is locked to `PLAN_COMPLIANCE`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT`; the paper's "Local Queue Optimization" is dropped, and Cooperative Recovery is removed (4->3) — peer-failure recovery is handled upstream by the trigger layer (a peer-overdue sensed target becomes a pop-up the policy may OPPORTUNISTIC_ENGAGEMENT), so a CR column would be dead. `build_action_mask` is pure-numpy additive `{0, -inf}` and reads `capable`/`reachable` from task-feature columns only — it never recomputes reachability. The mask encodes hard structural/physical constraints only: peer-failure inference is learned from graph/time, not hard-coded. `ActionHead` is a shared per-node MLP and `sample_action` uses a joint masked categorical over flattened `k*3` with masked-safe entropy.

- `action/graph_effect.py` (effect layer) — `apply_meta_action` is the pure-function semantic twin of `build_action_mask`. It maps ⟨meta_action m, node v⟩ to a plan edit on the authoritative `solution` dict, **not** to graph-edge mutation and **not** to a BLADE command. `PLAN_COMPLIANCE` returns an equal-but-not-same copy; `OPPORTUNISTIC_ENGAGEMENT` appends `(task_idx=v, attack_step_idx, level=min(existing ego levels)-1 or 0)`; `SELF_PRESERVATION_ABORT` removes the ego's tuple(s) for task `v`. The function is BLADE-free, torch-free, never mutates input, normalizes solution keys to `str`, and makes OE idempotent by `task_idx`.

**Trigger layer: BUILT (with `_selftest`), not wired.** `action/graph_trigger.py` (`decide_triggers`) is the **WHEN gate** upstream of the observation: a pure function over ONE ego's private belief that wakes the policy on **POP-UP** (a sensed enemy absent from `belief_tasks` → append a pop-up Task to the append-only `belief_tasks`) or **PEER-OVERDUE** (a sensed peer-assigned target past its ETA → drop the peer's tuple from the ego's `belief_solution` copy, making it pop-up-like). It does the "remove peer edge" half of the retired CR meta-action; the policy + `graph_effect` do the "add ego edge" half. Its sole sensor input is `GraphPlanExecutor.sensed_target_ids(observation, ego_id) -> {id: unit}` (the ego's own in-range live enemies, world-scan via the shared `scenario_factory.iter_enemy_targets`, reusing the confirmed-kill distance+liveness predicate). New shared helpers `iter_enemy_targets` + `make_attack_task` (single enemy-enumeration + Task template, utility by unit class Facility 100 / Airbase 80 / Ship 95) are consumed by BOTH `generate_all_enemy_tasks` and the trigger's pop-up path (zero duplication; `generate_all_enemy_tasks` behavior unchanged, verified via `graph_builder` + bonmin). ETA is an injected `+inf` placeholder (`never_overdue`) so PEER-OVERDUE is structurally present but **DORMANT** — the real ETA (haversine / cruise speed / task ordering / levels) is a later effort. Side-by-side with the frozen flat path and **not wired into `train_full.py`** (no orchestrator); the meta-action / mask contract stays **k×3** (CR removed in B1). `_selftest`: `env PYTHONPATH=src python -m match_aou.rl.action.graph_trigger`.

**Executor layer: BUILT & REVIEWED, not wired.** `blade_graph_executor.py` (`GraphPlanExecutor`) consumes the updated `solution` and translates it to BLADE move/launch/attack/RTB — the sole BLADE translation layer for the graph path, side-by-side with the frozen `blade_executor_minimal.py` and **not yet wired into `train_full.py`**. Execution is **done-on-confirmed-kill** (proximity-gated confirm-guard, no-comms), fire-once-then-wait with a per-`(ego, target)` re-fire throttle (`kill_confirm_ticks`), per-agent levels, **per-ego private task lists** (`self.tasks: Dict[str, List[Task]]`, fanned out to all agents at init; `_resolve_step(ego_id, …)` resolves against the acting ego's OWN list; `resync(ego_id=…, tasks=…)` updates only that ego's slice) so a pop-up sensed by ego A never enters ego B's task-view — enforced by keying and proven by the `_selftest` no-comms checks ISO-1..3 (append-only isolation, out-of-range non-resolution, same-index-collision distinctness), `resync`-based mid-episode plan edits that preserve completed work, single-issue RTB latch, and a `dead`-set so `is_done()` doesn't hang on crashes. See section 2. **Carry-forward:** calibrate `kill_confirm_ticks` once `probability < 1.0` lands (default 60 never bites at lethality 1.0; smoke measured 36 ticks for AIM-120 from 50 km — consider deriving it at runtime from weapon flight time).

**Encoder layer: BUILT (with `_selftest`), not wired.** `agent/graph_encoder.py` (`GraphEncoder`) turns a `GraphObservation` into **per-task-node** embeddings `[k, embed_dim]` that the actor's `ActionHead(embed_dim=encoder.embed_dim)` consumes directly — side-by-side with the frozen flat `agent/network.py` and **not yet wired into `train_full.py`**. It guarantees the three mandatory properties: permutation-invariance over nodes, size-agnosticism (native to any `(k, a, E)`, no `MAX_AGENTS` padding), and a **per-task** (NOT pooled) output. Locked interface: `forward(obs, edge_attr=None) -> Tensor[k, embed_dim]`, single-graph (no batch dim); defaults `model_dim=64, embed_dim=64, num_heads=4, num_layers=2`, with parametric `agent_feat_dim=1 / edge_attr_dim=1` so a future builder column drops in without reopening the encoder. `task_feat_dim` now DEFAULTS to `TASK_FEATURE_DIM` (a single source of truth defined in `graph_builder`, currently 6: utility, dist, capable, reachable, probability, sensed), which the encoder imports — so the builder's `task_features[k, 6]` and the encoder's input projection can never desync again; a future column updates the one constant and both follow. Internals (our engineering choice, not prescribed): type-specific input projections; a learned TASK/EGO/PEER **role embedding** (node-typing done HERE — the builder defers it — with a reserved MISSION 4th role); injected `time_norm`; hand-rolled multi-head attention RESTRICTED to the graph edges (no PyG/DGL — only torch/numpy) over an **augmented edge set** (forward + reversed/SYMMETRIZED + per-node SELF_LOOP), with a learnable per-relation `type_bias[7, num_heads]` materialized as a dense `[N,N,num_heads]` matrix and non-edges filled with a LARGE FINITE NEGATIVE (self-loops guarantee ≥1 incoming edge ⇒ no empty-softmax NaN); pre-LN attention + position-wise FFN per layer. Output slices the task rows and applies `Linear(model_dim -> embed_dim)`. A `pool()` method (mean over all node embeddings → `[embed_dim]`) is the HOOK for the future centralized critic; **no value head is built now**. `_selftest` (`env PYTHONPATH=src python -m match_aou.rl.agent.graph_encoder` under `nlp_env`) asserts shape/finiteness end-to-end into `ActionHead`+`sample_action`, finite grads on every param (incl. the `edge_attr` path), the `a=2`/`a=1` sizes, the isolated-task / zero-edge attention-stress cases, and task- and peer-permutation invariance with the ego held fixed.

**`edge_attr` deferred feature (encoder is ready for it).** `GraphEncoder.forward` accepts an optional `edge_attr` (`[E, edge_attr_dim]` aligned with `obs.edge_index`, projected `Linear(edge_attr_dim -> num_heads)` and added per-head to a relation's fwd+rev score; self-loops carry none). It is `None` today — the builder emits no `edge_attr` field yet. **Reserved use:** a normalized **expected-execution-time on ASSIGNMENT edges** driving **Cooperative-Recovery timing** inference. Whether that value is a static expected-exec-time or a dynamic relative-slack is a deferred **builder** decision; the encoder is agnostic.

**Detection/attack range — UNIFIED.** Sensing = attack = arrival is a **single radius** (50 km). The executor's `arrival_threshold_km` and the builder's `GraphObservationConfig.detection_range_km` (now default `50.0`) are the SAME physical radius and **MUST be kept equal** — a future orchestrator will pass one value to both. No orchestrator exists yet, so today they are two equal literals with a mutual comment; collapse to a single source when the orchestrator is built. There is **no** separate "detection > attack" radar range in this model (radar-commit-beyond-weapons-range is a deliberate future enrichment, not the baseline). Do not silently switch either to BLADE `aircraft.range`.

**OPEN (Phase-2): `reachable_by_ego` model.** `graph_builder._reachable_by_ego` currently uses a full **round-trip from the current position** — a conservative placeholder that is wrong at runtime (a pop-up near base may look reachable even if diverting starves already-assigned tasks). The intended model is **marginal**: detour cost of inserting the target into the existing route checked against remaining fuel slack. `build_action_mask` reads `reachable` straight from the task-feature column, so the swap is isolated to `graph_builder`.

**OPEN (Phase-2): `assigned_to_peer`.** Currently derived from `ASSIGNMENT` edges and consumed by `build_action_mask`; not exposed as a task-feature column. May be added later pending advisor input.

**Still open after the encoder:**
- variable-size PPO buffer and an `evaluate_action` path for stored graph actions during PPO update epochs (the encoder is single-graph / no batch dim — batching is this buffer's concern);
- training wiring into `train_full.py` (wire the graph builder → encoder → `ActionHead`/`sample_action` → `graph_effect` → `blade_graph_executor` chain);
- reward review/rework under graph meta-actions and the current `probability=1.0` task regime;
- re-enable `FUEL_DAMAGE_ENABLED = True` only after a clean discovery-only baseline;
- archive old baselines/checkpoints that came from incompatible regimes (e.g. `probability=0.6`) and produce a fresh canonical baseline.

**OPEN (Phase-2, cross-layer — pending advisor; being scoped in a SEPARATE chat, do NOT implement here): deterministic peer-dropout trigger.** Move "peer overdue ⇒ remove its ASSIGNMENT edge before graph build" OUT of the learned policy into a deterministic pre-build trigger. This collapses Cooperative-Recovery into Opportunistic-Engagement (once the peer's ASSIGNMENT edge is gone the target reads as a pop-up), so it needs a deadline-calibration param plus a `was_assigned_to_peer` task feature to preserve recovered-vs-popup semantics. Touches the builder (edge removal + new feature) and the mask/effect interpretation — hence cross-layer.

**Carry-forward open items (unchanged):** the `reachable_by_ego` **marginal-detour** model (still a round-trip placeholder in `graph_builder`), the `assigned_to_peer` task-feature column (still edge-derived, not a column), and `kill_confirm_ticks` calibration once `probability < 1.0` lands.
