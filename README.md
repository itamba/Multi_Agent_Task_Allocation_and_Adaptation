# Multi-Agent Task Allocation and Adaptation

MSc research software for **runtime adaptation of a static multi-agent task allocation**,
under a hard **no-communication** constraint, executed in a physics-based military
simulation.

A MATCH-AOU optimization solver produces one allocation offline. At runtime each agent flies
that plan alone: it senses only through its own sensors, keeps its own private belief about
the plan, and — when its own observations warrant it — a Graph-RL policy edits that belief.
Agents never exchange information, directly or indirectly.

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
- a **reference** solution is computed per episode purely to normalize the training reward —
  a full-information t=0 oracle under the historical reference policy, or an event-conditioned
  MATCH-AOU continuation reference under the generalized designs. It is a centralized
  *training* signal and is never visible to a policy at execution time.

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
   │ reference solve (training only)  │   │              ──> edit OWN belief     │
   │   (t=0 oracle, or a continuation │   │              ──> executor resync     │
   │    reference at the FD event)    │   │                                      │
   └──────────────────────────────────┘   │  Phase 2 (once):                     │
                                          │    GraphPlanExecutor.next_actions()  │
                                          │    ──> env.step(commands)  [BLADE]   │
                                          └──────────────────────────────────────┘
                                                          │
                                          terminal, reference-normalized reward
                                                          │
                                          PPO update (actor-only, or CTDE critic)
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
Training is **actor-only by default**; `training_mode = "ctde"` adds a centralized critic
during training only. Evaluation and inference are actor-only in both modes.

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
| **One radius** | Sensing = attack = arrival = kill-confirmation = discovery = `DETECTION_KM` (50 km). BLADE's per-aircraft `aircraft.range` is deliberately *not* used for discovery. |
| **Event-triggered** | The policy wakes on a pop-up, a peer-overdue gate, a fuel-damage event or a post-damage completion boundary — never on a periodic timer. |
| **Decentralized execution** | The acting path reads only the ego's private observation. A CTDE critic, when selected, exists during training only and reads privileged state that never reaches the actor. |

---

## 4. Episode designs, solvers and research state

Every episode is a seeded variation of `data/scenarios/strike_training_4v5.json`. **Which
population an episode comes from is selected explicitly** by `--episode-design`:

| Design | Population | Status |
|---|---|---|
| `fixed_cell_v1` (**code default**) | the historical cell: 3 agents, 3 known + 3 route-relative hidden airbase targets, 200 km / 100 km geometry, one fuel-damage factor | preserved; every approved fixed-cell measurement was taken on it. **It is not the current research population.** |
| `generalized_v1` | `A ∈ {2,3,4}`, `K = A`, hidden load sampled per episode; certified fuel damage with mild / severe severities; event-conditioned continuation reference; 18-stratum frozen benchmark | preserved, valid design |
| `generalized_v2` | same mechanisms; two-stage route-relative population (`A ∈ {2..6}`, `K ∈ {A, A+2}` before the known-only solve, hidden load against the realized routed-ego count after it); ten-cell frozen benchmark with `development` and `confirmatory` profiles | **current research line**; requires `p1_milp_v1` |

**Allocation backends** are selected explicitly by `--match-aou-backend`, with no fallback:
`legacy_minlp_v1` (the **default**, the frozen MINLP through BONMIN) and `p1_milp_v1` (a
deterministic `p = 1` MILP through SciPy/HiGHS). The two objectives are **not** equivalent:
changing the backend can change the allocation, the hidden geometry and therefore the
population.

Targets are enemy airbases only (`include_sams = False`), and target destruction is
deterministic (`probability = 1`).

**Results.** Measurements exist and are recorded, with their verdicts, denominators and
explicit non-claims, in [`docs/history/measurements.md`](docs/history/measurements.md); the
current phase, open evidence and next actions are in
[`graph_rl_project_handoff.md`](graph_rl_project_handoff.md). *(Earlier versions of this README
said no long baseline and no CTDE implementation existed; both statements are historical.)*

---

## 5. Repository layout

```
Multi_Agent_Task_Allocation_and_Adaptation/
├── CLAUDE.md                        # mandatory entry point: invariants, boundaries, reading table
├── graph_rl_project_handoff.md      # current state: phase, owner, evidence, next actions
├── requirements.txt                 # Python dependency surface (not a lock)
├── environment.cluster.yml          # validated BGU cluster environment (conda-forge)
├── configs/
│   └── graph_train/
│       └── final_cell_probe.json    # the one repository preset: a fixed-cell short probe
├── data/
│   └── scenarios/
│       └── strike_training_4v5.json # the one active scenario template
├── docs/
│   ├── contracts/                   # normative technical contracts per layer
│   ├── workflows/                   # review, experiment and environment procedures
│   ├── history/                     # implementation, measurement and decision history
│   ├── documentation_migration.md   # map of the 2026-09 documentation restructure
│   └── BLADE_API_DOCUMENTATION.md   # API reference for THIS vendored fork
├── src/match_aou/
│   ├── models/                      # Agent, Task, Step, StepKind, Location, Capability
│   ├── solvers/
│   │   ├── match_aou_MINLP_solver.py    # legacy MINLP objective (frozen)
│   │   ├── match_aou_p1_milp_solver.py  # deterministic p = 1 MILP objective
│   │   └── match_aou_backend.py         # explicit backend selection
│   ├── utils/
│   │   ├── scheduling_utils.py      # post-solve filter/level + nearest_neighbor_order
│   │   ├── topology_utils.py        # topological levels from precedence
│   │   └── blade_utils/
│   │       ├── blade_graph_executor.py  # GraphPlanExecutor — sole BLADE translation layer
│   │       ├── scenario_factory.py      # scenario -> Agents / Tasks
│   │       └── scenario_generator.py    # seeded scenario variations
│   ├── rl/
│   │   ├── observation/
│   │   │   ├── graph_builder.py         # (world, solution) -> GraphObservation
│   │   │   └── central_graph_builder.py # training-only CTDE central state
│   │   ├── agent/graph_encoder.py       # Graph Transformer encoder
│   │   ├── action/
│   │   │   ├── graph_action.py          # action head, legality mask, sampling
│   │   │   ├── graph_effect.py          # apply a meta-action to a solution (pure)
│   │   │   └── graph_trigger.py         # WHEN the policy wakes (pure)
│   │   ├── training/
│   │   │   ├── belief.py                    # per-ego private Belief
│   │   │   ├── graph_episode_setup.py       # construction: solve -> place -> patch -> reload
│   │   │   ├── graph_hidden_placement.py    # route-relative hidden-target geometry (pure)
│   │   │   ├── graph_tick_loop.py           # the two-phase tick
│   │   │   ├── graph_fuel_damage.py         # fuel-damage designs and certification (pure)
│   │   │   ├── graph_reward.py              # terminal reference-normalized reward
│   │   │   ├── graph_ppo.py                 # PPO core: actor-only and CTDE
│   │   │   ├── graph_generalized.py         # episode designs, samplers, benchmark manifests
│   │   │   ├── graph_benchmark_preflight.py # deterministic benchmark population selection
│   │   │   ├── graph_train.py               # training entry point
│   │   │   └── graph_rollout.py             # diagnostic rollout entry point
│   │   └── shared_utils.py              # small shared numeric helpers
│   └── integrations/
│       └── panopticon-main/gym/blade/   # vendored BLADE engine (frozen)
├── tests/                           # unit and integration tests
└── tools/
    ├── graph_executor_smoke.py      # end-to-end executor smoke (needs BLADE + BONMIN)
    └── benchmark_match_aou_p1_milp.py # engineering comparison of the two backends
```

The pure layers (`graph_trigger`, `graph_effect`, `graph_hidden_placement`,
`graph_fuel_damage`) import no simulator, no solver and no PyTorch, which is what makes
them hand-testable. `tests/test_import_purity.py` enforces that boundary.

---

## 6. Environment and installation

Two execution contexts are maintained: **local Windows + PyCharm with a conda environment
named `nlp_env`**, and the **BGU Slurm cluster with `graph_rl_cluster`** (see
[`environment.cluster.yml`](environment.cluster.yml)). The full rules are in
[`docs/workflows/environments_cleanup.md`](docs/workflows/environments_cleanup.md). Commands
below assume the repository root as the working directory and the local context.

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

**3. Solvers.** The legacy backend needs **BONMIN**, which ships inside `nlp_env` locally and
comes from conda-forge on the cluster. The P1 backend needs SciPy's `milp` (HiGHS). Anything
that *solves* — training, rollouts, preflights, the executor smoke — runs under the solver
environment:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --help
```

`--no-capture-output` avoids a Windows console re-encoding failure on Unicode output. On the
cluster every validation or scientific command also sets `PYTHONNOUSERSITE=1`.

**4. `match_aou` itself is not installed as a package.** `src/` must be on `PYTHONPATH`.
In PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

`tools/graph_executor_smoke.py` and the test files insert `src/` on `sys.path` themselves,
so they need no `PYTHONPATH`.

> A base conda environment also resolves `blade` and `gymnasium` (same vendored fork), which
> is why the solver-free tests can run outside `nlp_env`. It does **not** have BONMIN, and a
> missing solver fails quietly — never judge a solve by its exit code alone.

---

## 7. Entry points

| Command | What it does |
|---|---|
| `python -m match_aou.rl.training.graph_train` | **Training.** Runs PPO; updates weights; writes a run directory. |
| `python -m match_aou.rl.training.graph_rollout` | **Diagnostics only.** Drives the full pipeline and reports per-episode statistics. **No learning, no weight update.** |
| `python -m match_aou.rl.training.graph_benchmark_preflight` | **Benchmark population selection.** Builds a frozen benchmark manifest before training. |
| `python tools/graph_executor_smoke.py` | **Executor smoke.** One solved scenario end-to-end in BLADE, asserting launch → strike → RTB. |

**Running any of these for a scientific purpose needs explicit authorization** — see
[`docs/workflows/experiments.md`](docs/workflows/experiments.md).

### Training

`--iterations` is required for a real training run:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --iterations 20 --episodes 8 --seed 0
```

#### Running from a JSON preset

A run's shape can be declared in a JSON file instead of a command line. The repository owns
one preset, a **fixed-cell** bounded short probe (`fixed_cell_v1`, 2 iterations × 4 training
episodes, one `pre_update` and one `post_update` held-out round of 4 matched pairs, visual
artifacts on). It is a short probe, not a baseline, and not the current research population:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json
```

From PyCharm, the same run is a *Module name* run configuration —
module `match_aou.rl.training.graph_train`, parameters
`--config configs/graph_train/final_cell_probe.json`, working directory the repository
root, interpreter `nlp_env`, and `PYTHONPATH` including `src`.

A preset names `TrainConfig` **field** names (nested PPO knobs under `"ppo"`, CTDE knobs under
`"ctde"`); keys beginning with `_` are comments, and an unknown key is refused rather than
ignored. Resolution is **dataclass defaults < preset < explicitly typed flags**:

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json --seed 7
```

The resolved configuration and its source are recorded in `run_config.json` (`train_config`
and `config_source`). `config_source` is **always a structured object**, never `null`, and its
`resolved_from` names one of three provenances:

| `resolved_from` | What produced the config |
|---|---|
| `config_file` | a command line naming a JSON preset (`path` says which) |
| `cli_defaults` | a command line with no `--config` |
| `direct_config` | a `TrainConfig` built in Python and passed straight to `train()` |

Selected options (`--help` is authoritative):

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | — | JSON preset of `TrainConfig` fields; explicit flags override it |
| `--iterations` | — | PPO iterations (the maximum budget under early stopping); required to train |
| `--episodes` | 8 | episodes per iteration — attempts on the fixed cell, **successful** episodes on the generalized designs |
| `--seed` | 0 | base seed; pins initial weights and anchors the episode seed schedule |
| `--out` | `training_output_<timestamp>` | run directory |
| `--episode-design` | `fixed_cell_v1` | `fixed_cell_v1`, `generalized_v1` or `generalized_v2` |
| `--match-aou-backend` | `legacy_minlp_v1` | `legacy_minlp_v1` or `p1_milp_v1`; `generalized_v2` requires `p1_milp_v1` |
| `--training-mode` | `actor_only` | `actor_only` or `ctde` (critic during training only) |
| `--generalized-max-attempts-per-iteration` | — | required attempt budget for the generalized designs |
| `--benchmark-manifest` / `--benchmark-profile` | — | frozen benchmark for generalized evaluation; the profile (`development` / `confirmatory`) is required for `generalized_v2` |
| `--early-stopping` | off | training-reward plateau stop; approved for `generalized_v1` only |
| `--eval-every` / `--eval-episodes` | 5 / 8 | held-out evaluation cadence and size (fixed-cell band) |
| `--eval-base-seed` | 1000000 | start of the fixed-cell held-out seed band |
| `--num-agents`, `--n-known`, `--n-hidden` | 3, 3, 3 | the fixed cell (ignored by the generalized designs) |
| `--fuel-damage-mode`, `--fuel-damage-probability` | `seeded_mixture`, 0.5 | fuel-damage scheduling; the generalized designs require `seeded_variable` |
| `--visual-artifacts` | off | opt-in per-attempt inspection bundles |
| `--plot RUN_DIR` | — | re-draw an existing run directory's figures into `<RUN_DIR>/plots/` and exit |

Training refuses to start unless Git provenance is complete — both the full commit SHA and
a clean/dirty verdict must be determinable, so a run is always attributable to exact code.
A dirty tree warns loudly but runs.

### Diagnostic rollout

```bash
conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_rollout --episodes 20 --seed 0
```

Options: `--episodes`, `--seed`, `--out` (default `rollouts`), `--deterministic`,
`--record-first`, `--episode-design`, `--fuel-damage-mode`, `--match-aou-backend`.

### Executor smoke

```bash
conda run -n nlp_env --no-capture-output python tools/graph_executor_smoke.py
```

### Tests

The solver-free tests run under a plain `pytest` from the base environment:

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
| `run_config.json` | the fully resolved configuration, its source, and a Git `provenance` block |
| `train_records.jsonl` | one record per training iteration |
| `eval_records.jsonl` | one record per held-out evaluation round |
| `episode_outcomes.jsonl` | one durable record per successful attempt, including per-wake actor diagnostics |
| `episode_failures.jsonl` | append-only: every failed episode attempt with its pipeline stage, exact seed and traceback |
| `run_summary.json` | derived from the jsonl files, with an accounting reconciliation flag |
| `plots/` | figures drawn from the jsonl files alone (below) |
| `scenarios/` | the generated scenario JSON for each attempt |
| `checkpoints/` | saved model and optimizer state (save-only; there is no resume) |

#### Plots

| Figure | What it shows |
|---|---|
| `training_performance.png` | training reward; held-out evaluation **per condition**; the matched-group deltas |
| `policy_diagnostics.png` | meta-action mix and raw policy entropy over the training decisions |
| `measurement_health.png` | the denominators — success fractions, wake coverage, matched-group completion, and requested-vs-realized hidden load on generalized runs |
| `fd_policy_sensitivity.png` | optional: held-out immediate fuel-damage-wake policy diagnostics, when recorded |

Per-condition curves are each a mean over **that condition's own successful episodes**, so the
gap between them is not a within-seed effect; the **matched-group deltas are the within-world
comparison**, over complete groups only. All figures share one x-axis: **PPO updates completed
before the measurement**. Reward is reference-normalized regret where `0` is the reference, so
a batch or round with no successful episode is **dropped** from a curve rather than drawn at 0,
and an all-failed batch reports `null`, never `0.0`.

Any run directory can be re-plotted later without retraining:

```bash
python -m match_aou.rl.training.graph_train --plot training_output_20260101_120000
```

matplotlib is optional: if it is missing, plotting prints a notice and the run still
completes — the jsonl files are the record.

**Failure accounting.** On the fixed cell every scheduled seed is attempted **at most once**; a
failure is recorded and never retried or replaced. On the generalized designs each iteration
fills a quota of successful episodes from a bounded, deterministic sequence of attempts; a
failed attempt spends its seed and is replaced by the next one. Measurement-integrity faults
abort the run instead of entering either ledger.

`--visual-artifacts` is off by default and is **observation, not measurement** — nothing it
captures is read back into the pipeline. When enabled, each scheduled attempt gets one bundle
under `<run_dir>/visual_artifacts/` containing the known-only scenario, the executed `t=0`
scenario, the BLADE playback recording, and a manifest.

---

## 9. Documentation map

| Document | Role |
|---|---|
| `README.md` | stable orientation — this file |
| `CLAUDE.md` | mandatory entry point: invariants, frozen layers, permission boundaries, task-triggered reading |
| `graph_rl_project_handoff.md` | current state: phase, active task, evidence, next actions |
| `docs/contracts/` | authoritative technical contracts per layer |
| `docs/workflows/` | review, experiment and environment procedures |
| `docs/history/` | implementation, measurement and decision history |
| `docs/BLADE_API_DOCUMENTATION.md` | API reference for the vendored BLADE fork *as it exists in this repository* |

Where this README and the contracts disagree, the contracts win; where a contract and the code
disagree, the code decides what happens and the disagreement must be investigated.

---

## 10. Historical code

An earlier flat (non-graph) RL path — a MAPPO/CTDE design over a fixed-width observation
vector — was retired and deleted from `main`. It is preserved in full on the `flat-final`
branch and the `pre-cleanup` tag, and nothing in `src/` or `tools/` references it.

---

## Academic context

Part of MSc research at Ben-Gurion University of the Negev, Department of Software and
Information Systems Engineering.
