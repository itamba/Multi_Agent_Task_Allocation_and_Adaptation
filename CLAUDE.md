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

### 🛑 MATCH-AOU solver internals

`match_aou_MINLP_solver.py` is in its **original, advisor-approved form**. The advisor's directive is: **address allocation pathologies through scenario design, not solver constraints.** Three solver-level workarounds were tried and explicitly rolled back:

1. A hard `single_agent_per_step` constraint — worked, but too rigid.
2. A fuel penalty in the objective (`λ · Σ travel_cost · x[i,j,k]`) — worked, but solver-level.
3. Earlier attempts to force `probability = 1.0` *as a solver patch* — out of scope.

**Do not re-add any of these to the solver.** The current approach uses `probability = 1.0` at the **task-construction level** (in `scenario_factory.generate_all_enemy_tasks`, not in the solver) combined with class-based fuel tiers and easy/stretch zones. This makes the redundancy term `(1 - p + ε)^m` evaluate to ≈ 0 marginal utility for redundant agents, so the solver naturally produces 1-to-1 allocations without solver-side modifications. See sections 4 and 7.

The one **advisor-approved** change to the movement-budget constraint is the per-target **round-trip** charge (`round_trip_cost`: out + back, `risk_factor = 0`) — a fidelity correction to an existing constraint, **distinct from** the three rolled-back workarounds above (it is not `single_agent_per_step`, not an objective fuel penalty, not a probability patch). See section 4.

### 🛑 Observation-layer fuel damage

`fuel_damage.py` intentionally modifies **only the observation vector** (`obs.vector[0]` and `obs.self_state.fuel_norm`). It does **not** touch BLADE physics. This asymmetry is the learning signal: the oracle has full fuel in its plan, the RL sees reduced fuel and must learn to RTB. Do not "fix" by patching actual BLADE fuel values.

### ⚠️ Flat observation path is the frozen baseline (Phase-2 graph in progress)

The flat observation path (`observation_builder.py` + `observation_types.py` + `config.py`'s `ObservationConfig`) is the **active baseline** and is **frozen** while the Phase-2 graph path (`graph_builder.py`) is under construction. Don't modify the flat path to accommodate the graph work — the two live **side-by-side** until the graph path is wired into `train_full.py`. See sections 3 and 8.

### 🛑 BladeExecutorMinimal invariants

`blade_executor_minimal.py` enforces three load-bearing rules:
- At most **one action per tick** (BLADE constraint).
- **Level gating** by `level_order` (precedence).
- **FIFO launch validation** against the airbase inventory.

Don't bypass any without a corresponding BLADE-side change. *(Intra-level command **order** is a greedy nearest-neighbor optimization — see §4 — and may change freely; it does not affect these three invariants or which steps execute.)*

**Domain/BLADE boundary:** domain objects (`Step`/`Task`) contain **no** BLADE command strings. `BladeExecutorMinimal` is the **sole** BLADE translation layer — it builds every command (move / launch / attack / RTB) from semantic `Step` data (`step_kind`, `target_id`, `location`) plus the solver's assignment. The attack string is built directly in the executor (no template, no placeholders); this is behavior-preserving and verified byte-identical.

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
| Change the Phase-2 GRAPH observation (nodes / edges / features) | `observation/graph_builder.py` → `build_graph_observation` (`GraphObservation` with `task_features[k,5]` + `agent_features[a,3]` incl. `is_observed`; `GraphObservationConfig`; `EdgeType`) |
| Change the Phase-2 GRAPH action / meta-action mask (decision core) | `action/graph_action.py` → `MetaAction`, `build_action_mask` (k×4 additive `{0,-inf}` mask), `ActionHead` (shared per-node MLP), `sample_action` |
| Change the Phase-2 GRAPH effect (meta-action → plan edit) | `action/graph_effect.py` → `apply_meta_action` (pure plan editor; BLADE-free) |
| Change obs dim / top_k / normalization | `observation/config.py` (`ObservationConfig`) |
| Compute travel time / fuel needed | `observation/observation_utils.py` — fuel helpers only (`target_id` is an explicit `Step` field, read directly; no action-string parsing) |
| Add / remove / change actions | `action/action_config.py` (`ActionType`), `action/action_utils.py`, `action/action_validation.py` |
| Translate an action token into a BLADE command | `plan_editor.py` |
| Change the network | `agent/network.py` (`ActorCriticNetwork`) |
| Change PPO hyperparameters | `training/ppo_trainer.py` (`PPOConfig`) — or override from `train_full.py` |
| Change the reward function | `training/reward.py` |
| Change GAE / buffer behavior | `training/rollout_buffer.py` |
| Change fuel damage events | `training/fuel_damage.py` |
| Change the oracle signal (what RL imitates) | `training/oracle.py` |
| Change per-episode setup (partial/full split, solve, auto-launch) | `training/episode_initializer.py` |

> **Phase-2 graph observation:** `observation/graph_builder.py` is the Phase-2 **graph** representation (task/agent nodes + typed edges — `GraphObservation` with `task_features[k,5]`, `agent_features[a,3]` (`[0]` fuel_norm, `[1]` dist_to_ego, `[2]` `is_observed`), COO `edge_index`/`edge_type` over the `EdgeType` IntEnum, plus `time_norm`; configured via `GraphObservationConfig`). Agent nodes + `ASSIGNMENT` edges cover the **complete static allocation** (not just sensed peers); unsensed assigned peers carry `is_observed = 0` with fuel/dist masked to `0.0`. It is built **side-by-side** with the flat builder above: the flat builder remains the **active baseline**, and the graph path is **not yet wired into `train_full.py`**. It ships with an in-file `_selftest()` (`env PYTHONPATH=src python -m match_aou.rl.observation.graph_builder` under `nlp_env`). The Phase-2 action layer that consumes it is `action/graph_action.py` (see the table row above and section 8). See sections 2 and 8.

### BLADE integration (`src/match_aou/utils/blade_utils/`)

The boundary between scenario content and MATCH-AOU object construction is **strict**:

- `scenario_generator.py` owns scenario *content* — units, positions, fleet composition, fuel tiers, target placement zones, and the generation-time discovery-chain check (Layer 1).
- `scenario_factory.py` owns BLADE-observation → MATCH-AOU object conversion. Stateless. Single source of truth for `Agent` and `Task` construction. `train_full.py` imports `generate_all_enemy_tasks` directly — **no local duplicates of task/agent construction allowed in `train_full.py`**.

| I want to… | Go to |
|---|---|
| Create Agents from a BLADE scenario | `scenario_factory.py` → `create_agents_from_scenario` |
| Generate Tasks from enemy units | `scenario_factory.py` → `generate_all_enemy_tasks` (default `probability=1.0`) |
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

**Phase 1 — static allocation: complete.** The solver's round-trip fuel/budget accounting with `risk_factor = 0` (WI-1), the `StepKind` + `target_id` domain refactor with `BladeExecutorMinimal` as the sole BLADE translation layer (WI-3), and greedy nearest-neighbor intra-level execution ordering (WI-4) are all merged and verified. The items below concern RL training/adaptation and remain open.

**Phase 2 — graph + Transformer RL: in progress.** The RL layer is being rebuilt from the MATCH-AOU paper as a **graph + Transformer** architecture, replacing the flat observation vector. STATUS: three layers done, all **side-by-side** with the frozen flat path and **not yet wired into `train_full.py`**:
- `observation/graph_builder.py` (observation layer) — `GraphObservation` + `GraphObservationConfig`. `agent_features` is `[a, 3]` (added `is_observed`; unsensed assigned peers carry `is_observed = 0` with fuel/dist masked to `0.0`), and agent nodes + `ASSIGNMENT` edges now cover the **complete static allocation** (not just sensed peers), so **"no `ASSIGNMENT` edge ⇒ genuine pop-up"** is reliable.
- `action/graph_action.py` (decision core) — the node-wise **k×4** meta-action head: `MetaAction` (4-action set — `PLAN_COMPLIANCE`, `COOPERATIVE_RECOVERY`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT`; the paper's "Local Queue Optimization" deliberately dropped), `build_action_mask` (pure-numpy additive `{0, -inf}` k×4 mask from edges + task-feature columns — never recomputes reachability), `ActionHead` (shared per-node MLP; weights shared by construction → zero-shot over task count), and `sample_action` (joint masked softmax over the flattened k×4, version-independent masked-safe entropy). In-file `_selftest()` passes under `nlp_env`, including a regression node for the unsensed-peer-target case.
- `action/graph_effect.py` (effect layer) — `apply_meta_action`: the pure-function semantic twin of `build_action_mask`. Maps ⟨meta_action m, node v⟩ to a plan edit on the `solution` dict (the single source of truth), **NOT** to graph edges and **NOT** to a BLADE command. `PLAN_COMPLIANCE` = no-op; `COOPERATIVE_RECOVERY` / `OPPORTUNISTIC_ENGAGEMENT` both append `(v, attack_step_idx, min_ego_level − 1)` — the negative level pushes the new target to the **FRONT** of the executor's `level_order` queue while the ego's original assignments stay in `solution` and resume afterward; `SELF_PRESERVATION_ABORT` drops the ego's tuple(s) to `v` (key retained, possibly empty). Returns a new dict (never mutates input); BLADE-free, torch-free; in-file `_selftest()` passes under `nlp_env`.

  **Next (NOT yet built):** the graph **encoder** (edge-aware Transformer → node embeddings; required for coordination), the **rebuilt executor** that consumes the updated `solution` and translates the plan edit into BLADE move/attack/RTB (the effect layer deliberately stops at the plan and does not build BLADE commands), a **variable-size PPO buffer** + `evaluate_action` path, **training wiring** into `train_full.py`, and the **reward rework** (utility-based).

  **Executor forward note:** when the executor is rebuilt, it must recompute `current_level = min(...)` after a plan edit and be able to drop to a level *below* its current one (the `min − 1` front-insertion). Today's executor only advances levels upward and sets `current_level` once at construction.

- **OPEN (Phase-2): `reachable_by_ego` model.** `graph_builder._reachable_by_ego` currently uses a full **round-trip from the current position** — a conservative placeholder that is wrong at runtime (a pop-up near base is always "reachable" even if diverting would starve already-assigned tasks). The intended model is **marginal**: the detour cost of inserting the target into the existing route, checked against the ego's remaining fuel slack. `build_action_mask` reads `reachable` straight from the task-feature column (never recomputes), so swapping this model is isolated to `graph_builder`.
- **OPEN (Phase-2): `assigned_to_peer`.** Derived from `ASSIGNMENT` edges and consumed by `build_action_mask`; still not exposed as a task-feature column. May be added later pending advisor input.

- **Re-enable `FUEL_DAMAGE_ENABLED = True`** once a clean baseline of discovery-only behavior is established. Until then, RL learns one type of surprise at a time.
- **Fresh baseline under canonical configuration.** The existing 1000-episode baseline (92.9% trigger rate, 30%→40% discovery accuracy) was performed under `probability=0.6` and the older single-graph discovery chain. Numbers will not be directly comparable to runs under the current configuration; archive the old baseline checkpoints under `training_output/models/archive_baseline_p06/` and produce a fresh baseline under the new canonical setup.
- **Reward function review** (advisor follow-up). Verify that the per-step (weight 0.3) + episode-end utility-ratio (weight 0.7, scaled 5.0) hybrid behaves well under `probability=1.0` task utilities, which differ in absolute magnitude from the earlier `p=0.6` regime.
