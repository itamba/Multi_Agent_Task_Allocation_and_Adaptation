# Multi-Agent Task Allocation and Adaptation

MSc research software for **runtime adaptation of a static multi-agent task allocation**,
under a hard **no-communication** constraint, executed in a physics-based military
simulation.

A MATCH-AOU MINLP solver produces one optimal allocation offline. At runtime each agent
flies that plan alone: it senses only through its own sensors, keeps its own private
belief about the plan, and — when its own observations warrant it — a Graph-RL policy
edits that belief. Agents never exchange information, directly or indirectly.

---

## 1. Research objective

Given a fleet of heterogeneous strike agents and a set of targets, MATCH-AOU computes a
static allocation `A_init`. That plan is optimal only for the world known when it was
solved. During execution the world changes — targets appear that were not in the plan, and
an agent can suffer damage that invalidates its remaining route.

The research question is how an agent should **adapt a static allocation at runtime using
only its own information**. Concretely:

- the allocation is produced by optimization, not learned;
- adaptation is learned, and is **event-triggered** — the policy is consulted when the
  agent's own sensing (or an exogenous event) says something changed, not on a fixed
  decision interval;
- adaptation is **decentralized and communication-free**: nothing an agent learns may
  originate from a peer's sensors, position, fuel or decisions;
- a full-information **oracle** solution is computed per episode purely to normalize the
  training reward — it is a centralized *training* signal and is never visible to a policy
  at execution time.

Execution runs in **BLADE**, a vendored fork of the Panopticon simulation engine
(aircraft dynamics, fuel burn, weapon engagement, kill resolution).

---

## 2. Architecture

```
                    offline                                runtime
   ┌──────────────────────────────────┐   ┌──────────────────────────────────────┐
   │ scenario generator               │   │ per tick:                            │
   │   └─ known-only world            │   │                                      │
   │ MATCH-AOU solve  ──> A_init      │   │  Phase 1 (per ego, one snapshot):    │
   │ hidden-target placement          │   │    own sensing ──> trigger?          │
   │   (route-relative, guaranteed    │   │      └─ wake ──> graph observation   │
   │    to be flown past)             │   │              ──> Graph Transformer   │
   │ scenario patch + reload          │   │              ──> masked meta-action  │
   │ MATCH-AOU solve  ──> oracle      │   │              ──> edit OWN belief     │
   │   (all targets; training only)   │   │              ──> executor resync     │
   └──────────────────────────────────┘   │                                      │
                                          │  Phase 2 (once):                     │
                                          │    GraphPlanExecutor.next_actions()  │
                                          │    ──> env.step(commands)  [BLADE]   │
                                          └──────────────────────────────────────┘
                                                          │
                                          terminal, oracle-normalized reward
                                                          │
                                                    PPO update
```

**Private beliefs.** The episode mints *N* independent `Belief(tasks, solution)` objects,
one per agent. All start byte-equal to the normalized `A_init`, but they are fully
independent copies. Editing one agent's belief can never touch another's.

**The graph is a projection, not state.** `solution` is the single source of truth. The
graph observation is rebuilt from `(world, solution)` on every wake and is never mutated.
Every decision is an edit to a belief's `solution`; the graph re-derives on the next build.

**Structural no-communication.** Peer nodes in the graph are *featureless* — a peer's fuel,
position and observations are deliberately dropped, because reading them would be a
communication channel. `A_init` enters only through `ASSIGNMENT` edges and the featureless
peer nodes anchoring them. The only runtime sensing in the graph is the ego's own `sensed`
column, recomputed from the ego's own position.

**Policy.** A Graph Transformer encoder (edge-masked multi-head attention over typed
relations, implemented directly in PyTorch — no PyG/DGL) produces per-task-node
embeddings; an action head emits three meta-actions per node — `PLAN_COMPLIANCE`,
`OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT` — under a hard legality mask.

**Execution.** `GraphPlanExecutor` is the sole translation layer from a plan to BLADE
commands (move / launch / attack / return-to-base). It keeps per-agent private task lists
and marks a target done only on a **confirmed kill within the agent's own sensor range**.

---

## 3. Core invariants

These are load-bearing; the authoritative statements live in `CLAUDE.md` §3.

| Invariant | Meaning |
|---|---|
| **No communication** | An agent acts only on its own sensors and its own belief. It never learns what a peer sensed, killed or decided. |
| **`solution` is the source of truth** | The graph is a stateless projection rebuilt each trigger, never mutated. |
| **Tasks are append-only** | Within an episode a pop-up task is appended, never removed — positional `task_idx` indexes into `solution` tuples and must stay valid. |
| **Peer runtime state is not exposed** | Peer graph rows carry no features. |
| **One radius** | Sensing = attack = arrival = kill-confirmation = discovery = `DETECTION_KM` (50 km) in the current cell. BLADE's per-aircraft `aircraft.range` is deliberately *not* used for discovery. |
| **Event-triggered** | The policy wakes on a pop-up, a peer-overdue gate, or a fuel-damage event — never on a periodic timer. |
| **Actor-only PPO** | Phase A has **no centralized critic**. `GraphEncoder.pool()` exists as the seam where a CTDE critic would attach; it is not implemented on `main`. |

---

## 4. Current experiment cell

The primary scenario template is `data/scenarios/strike_training_4v5.json`. Every episode
is a seeded variation generated from it.

The current reference cell (defaults in `TrainConfig` / `RolloutConfig`):

- **3 agents**, all launching from the same BLUE airbase;
- **3 known targets** — present at `t=0` and solved into `A_init`;
- **3 hidden targets** — placed *route-relative*, on a leg the assigned agent is
  geometrically guaranteed to fly past within its sensing radius, so discovery comes from
  geometry rather than from a connectivity heuristic;
- enemy targets are airbases only (`include_sams=False`); they do not shoot back;
- engagement probability is 1.0;
- geometry floors of 200 km (launch base to any target) and 100 km (between known targets);
- one difficulty factor, **FD-BASELINE-v1**: a deterministic, seeded, ego-local
  fuel-damage event that puts exactly one agent into a strict decision window where flying
  home is still feasible but completing its route is not. Training draws clean/damaged
  episodes from a seeded mixture; evaluation runs matched clean/damaged pairs on the same
  seed.

**On results.** The pipeline, the difficulty factor and the inspection tooling are
implemented, reviewed and locked. **No long baseline has been run on this cell**, and no
result is claimed for it. An earlier short probe measured a strictly easier, pre-fuel-damage
configuration; those numbers are historical and are explicitly *not* a baseline for the
current cell. The next planned step is a single bounded short probe — see
`graph_rl_project_handoff.md`.

---

## 5. Repository layout

```
Multi_Agent_Task_Allocation_and_Adaptation/
├── CLAUDE.md                        # authoritative technical & research contracts
├── graph_rl_project_handoff.md      # volatile: current phase, next task
├── requirements.txt
├── configs/
│   └── graph_train/
│       └── final_cell_probe.json    # the bounded short final-cell probe preset
├── data/
│   └── scenarios/
│       └── strike_training_4v5.json # the one active scenario template
├── docs/
│   └── BLADE_API_DOCUMENTATION.md   # API reference for THIS vendored fork
├── src/match_aou/
│   ├── models/                      # Agent, Task, Step, StepKind, Location, Capability
│   ├── solvers/                     # MATCH-AOU MINLP solver (frozen)
│   ├── utils/
│   │   ├── scheduling_utils.py      # post-solve filter/level + nearest_neighbor_order
│   │   ├── topology_utils.py        # topological levels from precedence
│   │   └── blade_utils/
│   │       ├── blade_graph_executor.py  # GraphPlanExecutor — sole BLADE translation layer
│   │       ├── scenario_factory.py      # scenario -> Agents / Tasks
│   │       └── scenario_generator.py    # seeded scenario variations
│   ├── rl/
│   │   ├── observation/graph_builder.py # (world, solution) -> GraphObservation
│   │   ├── agent/graph_encoder.py       # Graph Transformer encoder (+ pool() critic seam)
│   │   ├── action/
│   │   │   ├── graph_action.py          # action head, legality mask, sampling
│   │   │   ├── graph_effect.py          # apply a meta-action to a solution (pure)
│   │   │   └── graph_trigger.py         # WHEN the policy wakes (pure)
│   │   ├── training/
│   │   │   ├── belief.py                # per-ego private Belief
│   │   │   ├── graph_episode_setup.py   # episode construction: solve -> place -> patch -> reload
│   │   │   ├── graph_hidden_placement.py# route-relative hidden-target geometry (pure)
│   │   │   ├── graph_tick_loop.py       # the two-phase tick
│   │   │   ├── graph_fuel_damage.py     # FD-BASELINE-v1 difficulty factor (pure)
│   │   │   ├── graph_reward.py          # terminal oracle-normalized regret
│   │   │   ├── graph_ppo.py             # PPO core (actor-only)
│   │   │   ├── graph_train.py           # training entry point
│   │   │   └── graph_rollout.py         # diagnostic rollout entry point
│   │   └── shared_utils.py              # small shared numeric helpers
│   └── integrations/
│       └── panopticon-main/gym/blade/   # vendored BLADE engine (frozen)
├── tests/                           # solver-free unit/integration tests
└── tools/
    └── graph_executor_smoke.py      # end-to-end executor smoke (needs BLADE + BONMIN)
```

The pure layers (`graph_trigger`, `graph_effect`, `graph_hidden_placement`,
`graph_fuel_damage`) import no simulator, no solver and no PyTorch, which is what makes
them hand-testable. `tests/test_import_purity.py` enforces that boundary.

---

## 6. Environment and installation

The maintained environment is **Windows + PyCharm with a conda environment named
`nlp_env`**, Python 3.10+. Commands below assume the repository root as the working
directory.

**1. Python dependencies** (`numpy`, `scipy`, `torch`, `pyomo`, `gymnasium`, `shapely`,
`haversine`):

```bash
pip install -r requirements.txt
```

**2. The vendored BLADE engine** is editable-installed, so `import blade` resolves to the
fork in this repository rather than to any other copy:

```bash
pip install -e src/match_aou/integrations/panopticon-main/gym
```

**3. BONMIN** is required for real MATCH-AOU solves. It is provided by the `nlp_env`
environment. Anything that *solves* — training, rollouts, the executor smoke — must run
under `nlp_env`:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --help
```

`--no-capture-output` avoids a Windows console re-encoding failure on Unicode output.

**4. `match_aou` itself is not installed as a package.** `src/` must be on `PYTHONPATH`.
In PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

`tools/graph_executor_smoke.py` and the test files insert `src/` on `sys.path` themselves,
so they need no `PYTHONPATH`.

> A base conda environment also resolves `blade` and `gymnasium` (same vendored fork), which
> is why the solver-free test suite can run outside `nlp_env`. It does **not** have BONMIN,
> and a missing solver there fails quietly — never judge a solve by its exit code alone.

---

## 7. Entry points

| Command | What it does |
|---|---|
| `python -m match_aou.rl.training.graph_train` | **Training.** Runs PPO; updates weights; writes a run directory. |
| `python -m match_aou.rl.training.graph_rollout` | **Diagnostics only.** Drives the full pipeline and reports per-episode statistics. **No learning, no weight update.** |
| `python tools/graph_executor_smoke.py` | **Executor smoke.** One solved scenario end-to-end in BLADE, asserting launch → strike → RTB. |

### Training

`--iterations` is required for a real training run:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --iterations 20 --episodes 8 --seed 0
```

#### Running from a JSON preset

A run's shape can be declared in a JSON file instead of a command line, which is what
makes a bounded experiment reproducible from the repository rather than from a shell
history. The repository owns one preset: the **bounded short final-cell probe**.

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json
```

That preset is 2 iterations x 4 training episodes, plus one `pre_update` and one
`post_update` held-out round of 4 matched pairs each, on the final cell (3 agents,
3 known + 3 hidden targets, no SAMs) with FD-BASELINE-v1 and `--visual-artifacts` on.
It is a **short probe, not a baseline**; a long baseline is separately authorized and is
deliberately not presettable.

From PyCharm, the same run is a *Module name* run configuration —
module `match_aou.rl.training.graph_train`, parameters
`--config configs/graph_train/final_cell_probe.json`, working directory the repository
root, interpreter `nlp_env`, and `PYTHONPATH` including `src`.

A preset names `TrainConfig` **field** names (nested PPO knobs under `"ppo"`); keys
beginning with `_` are comments, and an unknown key is refused rather than ignored.
Resolution is **dataclass defaults < preset < explicitly typed flags**, so a preset can
be adjusted for one run without editing it:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json --seed 7
```

The resolved configuration and the preset it came from are both recorded in
`run_config.json` (`train_config` and `config_source`, the latter listing which fields
the preset supplied and which a flag overrode). `config_source` is **always a structured
object**, never `null`, and its `resolved_from` names one of three truthful provenances:

| `resolved_from` | What produced the config |
|---|---|
| `config_file` | a command line naming a JSON preset (`path` says which) |
| `cli_defaults` | a command line with no `--config` |
| `direct_config` | a `TrainConfig` built in Python and passed straight to `train()` — no command line, no preset |

So "no preset was used" is a stated fact rather than a missing key, and a run driven from
a script or notebook is never recorded as though a command line had resolved it.

Selected options (`--help` is authoritative):

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | — | JSON preset of `TrainConfig` fields; explicit flags override it |
| `--iterations` | — | PPO iterations; required to train (may come from `--config`) |
| `--episodes` | 8 | training episodes per iteration |
| `--seed` | 0 | base seed; pins initial weights and anchors the episode seed schedule |
| `--out` | `training_output_<timestamp>` | run directory |
| `--eval-every` / `--eval-episodes` | 5 / 8 | held-out evaluation cadence and size |
| `--eval-base-seed` | 1000000 | start of the held-out seed band (must not overlap training seeds) |
| `--num-agents`, `--n-known`, `--n-hidden` | 3, 3, 3 | the scenario cell |
| `--fuel-damage-mode`, `--fuel-damage-probability` | mixture, 0.5 | difficulty-factor scheduling |
| `--visual-artifacts` | off | opt-in per-attempt inspection bundles |
| `--plot RUN_DIR` | — | re-draw an existing run directory's figures into `<RUN_DIR>/plots/` and exit (no training) |

Training refuses to start unless Git provenance is complete — both the full commit SHA and
a clean/dirty verdict must be determinable, so a run is always attributable to exact code.
A dirty tree warns loudly but runs.

### Diagnostic rollout

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_rollout --episodes 20 --seed 0
```

Options: `--episodes`, `--seed`, `--out` (default `rollouts`), `--deterministic`,
`--record-first` (record episode 0 with the BLADE playback recorder).

### Executor smoke

```bash
conda run -n nlp_env --no-capture-output python tools/graph_executor_smoke.py
```

### Tests

The suite is solver-free and runs under a plain `pytest`:

```bash
python -m pytest -q
```

Test files carrying a `__main__` runner can also be executed directly under `nlp_env`,
which is how they are checked in the project environment (`pytest` is not installed there):

```bash
conda run -n nlp_env --no-capture-output python tests/test_graph_hidden_placement.py
```

---

## 8. Training outputs

A run directory is the record of the run. `graph_train` writes:

| File | Contents |
|---|---|
| `run_config.json` | the fully resolved configuration, including nested PPO settings and a Git `provenance` block |
| `train_records.jsonl` | one record per training iteration |
| `eval_records.jsonl` | one record per held-out evaluation round |
| `episode_failures.jsonl` | append-only, flushed immediately: every failed episode attempt with its pipeline stage, exact seed and traceback |
| `run_summary.json` | derived from the jsonl files, with an accounting reconciliation flag |
| `plots/` | three figures drawn from the jsonl files alone (below) |
| `scenarios/` | the generated scenario JSON for each attempt |
| `checkpoints/` | saved encoder + head + optimizer state |

#### Plots

Figures are derived artifacts, so they live under `<run_dir>/plots/` rather than among
the records, and each one carries a single claim:

| Figure | What it shows |
|---|---|
| `training_performance.png` | training reward; held-out evaluation **split into clean and damaged**; the matched-pair delta `R_damaged - R_clean` |
| `policy_diagnostics.png` | meta-action mix and policy entropy over the training decisions |
| `measurement_health.png` | the denominators — training and eval success fractions, wake coverage, the **pair** success fraction, and **per-condition held-out completion** |

The two condition curves in `training_performance.png` are each a mean over **that
condition's own successful episodes**, so if one condition fails more held-out seeds than
the other they are not averages over the same seeds and the gap between them is not a
within-seed effect. The panel and its legend say so, and the per-condition
attempted/successful counts in `measurement_health.png` are what make the asymmetry
inspectable. The **matched-pair delta is the within-seed comparison** — it uses only
pairs whose both members completed, so it stays valid when the two populations differ.

All three share one x-axis: **PPO updates completed before the measurement**. Training
points sit at `updates_completed_before` (the updates the policy that generated those
episodes had received) and evaluation points at `updates_completed`, so the untrained
policy's first batch and its `pre_update` held-out measurement share an origin. Reward is
oracle-normalized regret where `0` is the optimum, so a batch or round with no successful
episode is **dropped** from a curve rather than drawn at 0 — `measurement_health.png` is
where that gap becomes visible, and the two figures are meant to be read together.

Any run directory can be re-plotted later without retraining:

```bash
python -m match_aou.rl.training.graph_train --plot training_output_20260101_120000
```

matplotlib is optional: if it is missing, plotting prints a notice and the run still
completes — the jsonl files are the record.

Every scheduled seed is attempted **at most once**. A failure is recorded and never
retried, replaced or substituted, so each reported statistic describes the successful
subset and is published next to its denominator. An all-failed batch reports its reward as
`null`, never `0.0` — the reward is oracle-normalized regret, where `0` is the optimum.

`--visual-artifacts` is off by default and is **observation, not measurement** — nothing it
captures is read back into the pipeline. When enabled, each scheduled attempt gets one
bundle under `<run_dir>/visual_artifacts/` containing the generator's known-only scenario,
the authoritative executed `t=0` scenario, the BLADE playback recording, and a manifest
stating the attempt's identity and whether the bundle is complete.

---

## 9. Documentation map

| Document | Role |
|---|---|
| `README.md` | public orientation — this file |
| `CLAUDE.md` | **authoritative** technical and research contracts: invariants, locked layer interfaces, build history, open questions |
| `graph_rl_project_handoff.md` | volatile: current phase state and the next task |
| `docs/BLADE_API_DOCUMENTATION.md` | API reference for the vendored BLADE fork *as it exists in this repository* |

Where this README and `CLAUDE.md` disagree, `CLAUDE.md` wins; where `CLAUDE.md` and the
code disagree, the code wins.

---

## 10. Historical code

An earlier flat (non-graph) RL path — a MAPPO/CTDE design over a fixed-width observation
vector — was retired and deleted from `main`. It is preserved in full on the `flat-final`
branch and the `pre-cleanup` tag, and nothing in `src/` or `tools/` references it.

---

## Academic context

Part of MSc research at Ben-Gurion University of the Negev, Department of Software and
Information Systems Engineering.
