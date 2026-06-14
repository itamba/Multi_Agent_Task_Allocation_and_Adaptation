# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## 1. Communication & Workflow (read first)

- **User speaks Hebrew.** Code, comments, and docs stay in English.
- **Work step-by-step.** One file / one change at a time. Explain the reasoning *before* editing, then wait for feedback.
- **No premature documentation.** Don't create `README.md`, `INDEX.md`, `SUMMARY.md`, or per-file docs without being asked. One consolidated doc per major component, only after it's stable.
- **Minimal files.** Prefer extending an existing module over spawning `foo_utils.py` + `foo_config.py` + `foo_helpers.py`.
- **Macro constants for feature toggles.** New toggleable features mirror the pattern at the top of `train_full.py` (`VARY_SCENARIOS`, `FUEL_DAMAGE_ENABLED`, `VALIDATE_EVERY`) — a module-level constant *and* a CLI flag.
- **Git workflow:** per-task commits, `git add .` (not per-file staging), single `git push` at session end.
- **Environment:** Windows + PyCharm terminal + `nlp_env` conda env, Python 3.10+. Avoid POSIX-only shell idioms (`rm -rf`, `&&` chains) — use Python or PowerShell equivalents.
- **Run all commands from the repo root.**

---

## 2. Critical: What NOT to touch without explicit discussion

### 🛑 BLADE engine (vendored Panopticon fork)

`Game.py`, `Scenario.py`, `Side.py`, `blade.py`, `weaponEngagement.py`, `Airbase.py`, `Aircraft.py`, `Facility.py`, `Weapon.py`, `Ship.py`, `ReferencePoint.py`, `PlaybackRecorder.py` — **frozen copy**. Do not refactor, reformat, or "improve" them. If the API we use changes, discuss the upgrade path explicitly first.

**Known deliberate workarounds (treat as load-bearing):**
- `CHARACTER_LIMIT` override at the top of `train_full.py` (500MB recordings).
- `game.current_scenario.name` set manually before `game.start_recording()` (e.g. `f"ep{N:03d}_rl"`) — otherwise recordings are named `"New Scenario"`.

### 🛑 MATCH-AOU solver internals

`match_aou_MINLP_solver.py` is in its **original, advisor-approved form**. Three modifications were tried and **explicitly rolled back**:

1. Forcing `probability = 1.0` to eliminate the `(1 − p)^m` redundancy term — insufficient; the `ε` safety term still rewarded redundant assignment.
2. A hard `single_agent_per_step` constraint — worked, but too rigid for future `p < 1.0` scenarios.
3. A fuel penalty in the objective (`λ · Σ travel_cost · x[i,j,k]`) — worked, but advisor chose not to modify the solver.

**Do not re-add any of these.** The redundancy pathology at `p < 1.0` is a property of the objective `1 − (1 − p + ε)^m`. It must be addressed through `ScenarioGenerator` zone design (fuel-based reachability asymmetry), not solver constraints.

### 🛑 Observation-layer fuel damage

`fuel_damage.py` intentionally modifies **only the observation vector** (`obs.vector[0]` and `obs.self_state.fuel_norm`). It does **not** touch BLADE physics. This asymmetry is the learning signal: the oracle has full fuel in its plan, the RL sees reduced fuel and must learn to RTB. Do not "fix" by patching actual BLADE fuel values.

### 🛑 BladeExecutorMinimal invariants

`blade_executor_minimal.py` enforces three load-bearing rules:
- At most **one action per tick** (BLADE constraint).
- **Level gating** by `level_order` (precedence).
- **FIFO launch validation** against the airbase inventory.

Don't bypass any without a corresponding BLADE-side change.

### 🛑 Legacy DQN code

Anything under `legacy/` (at repo root) is archived DQN-era code — `legacy/train_full_dqn.py`, `legacy/dqn_training/`, `legacy/blade_executor.py`. Don't import from it, don't resurrect it. Active entry point: `train_full.py`. Active executor: `blade_executor_minimal.py`.

### ⚠️ Old checkpoints

Checkpoints built with a different `MAX_AGENTS` are **incompatible** with the current `MAX_AGENTS=5` critic. Discard rather than attempt to load.

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
| Change what the agent sees (feature vector) | `observation/observation_builder.py` + `observation_types.py` + `self_features.py` + `target_extraction.py` + `plan_parsing.py` + `plan_context.py` |
| Change obs dim / top_k / normalization | `observation/config.py` (`ObservationConfig`) |
| Parse an action string / extract target IDs / fuel helpers | `observation/observation_utils.py` (`extract_target_id_from_action`, `is_attack_action`, `calculate_fuel_needed`) |
| Add / remove / change actions | `action/action_config.py` (`ActionType`), `action/action_utils.py`, `action/action_validation.py` |
| Translate an action token into a BLADE command | `plan_editor.py` |
| Change the network | `agent/network.py` (`ActorCriticNetwork`) |
| Change PPO hyperparameters | `training/ppo_trainer.py` (`PPOConfig`) — or override from `train_full.py` |
| Change the reward function | `training/reward.py` |
| Change GAE / buffer behavior | `training/rollout_buffer.py` |
| Change fuel damage events | `training/fuel_damage.py` (`FuelDamageConfig`, `FuelDamageManager`) |
| Change the oracle signal (what RL imitates) | `training/oracle.py` |
| Change per-episode setup (partial/full split, solve, auto-launch) | `training/episode_initializer.py` (`EpisodeInitializer`) |

### BLADE integration (`src/match_aou/utils/blade_utils/`)

| I want to… | Go to |
|---|---|
| Create Agents from a BLADE scenario | `scenario_factory.py` → `create_agents_from_scenario` |
| Generate Tasks from enemy units | `scenario_factory.py` → `generate_all_enemy_tasks` |
| Execute a MATCH-AOU plan in BLADE | `blade_executor_minimal.py` → `BladeExecutorMinimal.next_action` |
| Schedule/resolve plan actions | `blade_plan_utils.py` |
| Sync MATCH-AOU agents with BLADE observation (position/fuel/weapons) | `observation_utils.py` → `update_agents_from_observation` |
| Change per-episode variation (counts, zones, fleet mix) | `scenario_generator.py` (`VariationConfig`, `ScenarioGenerator`) |
| Change fuel-based reachability rules | `ReachabilityCalculator` in `scenario_generator.py` |
| Add a new aircraft/facility class to the pool | Add JSON to `extra_template_paths` — pools auto-extract |

### MATCH-AOU solver

| I want to… | Go to |
|---|---|
| Change the objective / constraints | `match_aou_MINLP_solver.py` |
| Change post-solve scheduling / level ordering | `scheduling_utils.py`, `topology_utils.py` |
| Change domain objects | `agent.py`, `task.py`, `step.py`, `step_type.py`, `location.py`, `capability.py` |

> **Reference `.md` files in the repo** (`BLADE_API_DOCUMENTATION.md`, `MATCH_AOU_API.md`, `INTEGRATION_GUIDE.md`, `RL_MODULE_DOCUMENTATION.md`) are hand-written and may lag the code. **Prefer reading the code.**

---

## 4. Gotchas & Conventions

- **Target IDs** are extracted from BLADE action strings via regex (`extract_target_id_from_action` in `rl/observation/observation_utils.py`). A change to BLADE's action format breaks the regex silently.
- **Landed aircraft** are moved by BLADE from `scenario.aircraft` → `airbase.aircraft`. "All agents returned to base" is checked by this transition, not a flag.
- **Fresh UUIDs per episode.** `ScenarioGenerator` assigns new UUIDs to cloned units — never assume a unit ID persists across episodes.
- **`effort` and `Quantity` have no effect on solver allocation.** `effort` is not consumed anywhere in the MINLP. `Quantity` is only checked as a capability-name match. Multi-agent-per-target is a pure objective-function artifact — don't try to fix redundancy by tweaking these fields.
- **`EPSILON = 1e-6` in the MINLP objective is load-bearing.** The objective contains `(1 − p + ε)^m`; without `ε`, `0^0` at `p = 1.0`, `m = 0` crashes Pyomo. Don't drop `EPSILON`, even when current scenarios use `p = 1.0`.
- **Task `probability` defaults to `1.0`** in `generate_all_enemy_tasks` — intentional, sidesteps redundancy pathology. `p < 1.0` scenarios depend on zone-based distribution landing first.
- **Budget uses a star topology.** The movement-budget constraint measures each step as an independent round-trip from the agent's start location — it does not account for chaining targets. Known conservative limitation, not a bug.
- **`top_k` is stuck at 1–3** because `ActionType` is a hand-enumerated enum. Extending beyond 3 requires edits in `action_config.py`, `action_utils.py`, `action_validation.py` — see docstring in `action_config.py`.
- **Reward is MATCH-AOU-utility-based only** — no fuel bonuses, no coverage bonuses (advisor direction). Hybrid: per-step (weight 0.3) + episode-end utility ratio (weight 0.7, scaled 5.0). See `reward.py`.
- **Critic is zero-padded to `MAX_AGENTS`.** Episodes run with variable agent counts (typically 2–3), but the critic input is always `obs_dim × MAX_AGENTS=5`. Missing slots are zeroed.

---

## 5. Dev Commands

**Install (one-time):**
```bash
pip install -r requirements.txt
# Also requires bonmin (a Pyomo-compatible MINLP solver) on PATH.
```

**Smoke test (single episode — fastest end-to-end regression check):**
```bash
python train_full.py --scenario data/scenarios/strike_training_4v5.json --episodes 1
```

**Unit tests under `tests/`** — `test_action_space.py`, `test_observation.py`, `RL_test_end_to_end.py`. Not wired into CI. Run individually with `python tests/<file>.py` (or `pytest tests/` if/when a pytest config is added).

**Typical training run (varied scenarios, no SAMs, fuel damage on):**
```bash
python train_full.py --scenario data/scenarios/strike_training_4v5.json --episodes 500
```

**Heterogeneous fleet + stretch zones:**
```bash
python train_full.py \
    --scenario data/scenarios/strike_training_4v5.json \
    --vary-scenarios --vary-base --base-shift-km 150 \
    --min-aircraft 2 --max-aircraft 3 \
    --min-facilities 2 --max-facilities 4 \
    --max-target-dist 500 --stretch-ratio 0.3 \
    --episodes 500
```

**Oracle-only validation episodes:**
```bash
python train_full.py --validate-every 10
# Produces ep{N:03d}_validation.jsonl alongside RL recordings.
```

**Inspect a generated scenario:**
```bash
python -m match_aou.utils.blade_utils.scenario_generator data/scenarios/strike_training_4v5.json
```

### Output layout (per training run)

```
training_output/
├── logs/{training.log, episode_{N:03d}.log}
├── models/checkpoint_ep{N}.pt       # preserved across runs
├── scenarios/episode_{N:03d}.json   # WIPED each run
└── recordings/                       # WIPED each run
    ├── ep{N:03d}_rl.jsonl
    └── ep{N:03d}_validation.jsonl
```

### Feature toggles (macros at top of `train_full.py`)

| Constant | Default | Effect |
|---|---|---|
| `VARY_SCENARIOS` | `True` | Per-episode scenario variation |
| `VARY_BASE` | `False` | Randomize blue base position |
| `INCLUDE_SAMS` | `False` | When `False`, targets are RED airbases only |
| `FUEL_DAMAGE_ENABLED` | `True` | Mid-episode fuel damage surprises |
| `VALIDATE_EVERY` | `10` | Oracle-only validation cadence (`0` disables) |
| `DISCOVERY_SCAN_INTERVAL` | `50` | Ticks between discovery scans |
| `MAX_AGENTS` | `5` | Critic padding size (changing invalidates checkpoints) |
| `MAX_SIM_TICKS` | `14400` | Per-episode tick cap |
| `PARTIAL_RATIO` | `2/3` | Fraction of tasks in the partial set |

---

## 6. Project Overview

MSc thesis: **Multi-Agent Task Allocation and Adaptation** (Ben-Gurion University, advisor Dr. Shahaf Shperberg).

Three layers that work together:

1. **MATCH-AOU** — MINLP optimization (Pyomo + bonmin) that generates optimal task allocations for heterogeneous agents operating *without* communication. Serves as the **oracle**.
2. **BLADE** — vendored Panopticon fork (physics-based military simulation).
3. **MAPPO RL layer** — CTDE (Centralized Training, Decentralized Execution). Learns to **adapt** the oracle plan in real time when surprise events occur mid-mission (target discovery, fuel damage).

**Research setup:** agents start with a MATCH-AOU plan based on **partial** information (~2/3 of targets). During execution they discover new targets and may be damaged. The RL learns to deviate from the oracle plan when appropriate — using a fully-informed oracle as the imitation target. Scenario: blue-side aircraft destroy red-side targets (RED airbases; SAMs optional).

---

## 7. Architecture

`train_full.py` is the orchestrator. Each episode: `ScenarioGenerator` produces a scenario → MATCH-AOU solves both the partial and full task sets → `BladeExecutorMinimal` plays the oracle plan tick-by-tick in BLADE → the MAPPO RL layer is consulted only on **event triggers** (new target discovery, fuel damage) and may override the executor's action.

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

---

## 8. Open TODOs (priority only)

- **Zone-based target distribution in `ScenarioGenerator`** (advisor-directed). Drive natural task distribution across heterogeneous fleets via fuel differentiation between agent types. `stretch_target_ratio` is the scaffolding; the follow-up is making zones actually drive the distribution so the solver's redundancy issue becomes self-limiting.
- **After zone-based distribution lands:** transition to `probability < 1.0` scenarios. The single-agent-per-target workaround can be formally removed at that point.
