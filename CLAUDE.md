# CLAUDE.md

Guidance for Claude Code in this repository. This is the **Multi-Agent GRAPH RL** project
(MATCH-AOU Phase-2): a no-communication multi-agent policy that adapts a static task
allocation at runtime over a graph representation. The old flat RL path is **retired** — deleted
from `main` in Step 3 of the cleanup, preserved on `flat-final` (`4d44c34`) + tag `pre-cleanup` —
**this document describes the graph model only.**

---

## 1. Communication & workflow (read first)

- **User speaks Hebrew.** Code, comments, and docs stay in English.
- **Step-by-step.** One file / one change at a time. Explain the reasoning *before* editing, then wait for feedback.
- **STOP before committing.** Implement + run tests, then stop and report (diff + test output) for line-by-line review. Locking (CLAUDE.md update + commit) is a **separate** step, only after approval.
- **No premature docs.** Don't spawn `README`/`SUMMARY`/per-file docs unasked. One consolidated doc per stable component.
- **Minimal files.** Prefer extending a module over spawning `foo_utils.py` + `foo_config.py`.
- **Git:** per-task commits, explicit staging of exactly the touched files, local-only unless told otherwise (no push). Respect "local-only" when stated.
- **Environment:** Windows + PyCharm terminal + `nlp_env` conda env, Python 3.10+. Avoid POSIX-only idioms; use Python/PowerShell equivalents. Run from repo root.
- **🛑 Solver/bonmin commands MUST run under `nlp_env`** (`conda run -n nlp_env ...`; add `--no-capture-output` to avoid Windows cp1255 re-encode crashes on Unicode prints). The base env lacks `bonmin` and fails **silently** (exits 0). **Never trust the exit code alone** — verify no `CRASH`/`Traceback` and that the run actually solved before claiming success.

---

## 2. Do NOT touch without explicit discussion

### 🛑 BLADE engine (vendored Panopticon fork) — FROZEN
`Game.py`, `Scenario.py`, `Side.py`, `blade.py`, `weaponEngagement.py`, `Airbase.py`, `Aircraft.py`, `Facility.py`, `Weapon.py`, `Ship.py`, `ReferencePoint.py`, `PlaybackRecorder.py` — do not refactor/reformat/"improve". If the API changes, discuss the upgrade path first. The engine is editable-installed into `nlp_env` (`pip install -e …/panopticon-main/gym`), so `import blade` resolves to the edited vendored engine.

**Load-bearing additive `Game.py` edits (graph executor depends on these):**
- `Game.handle_aircraft_attack(aircraft_id, target_id)` 2-arg form: `weapon_id` → highest-engagement-range weapon; `weapon_quantity` → 2 (keeps "one ATTACK step ⇒ target destroyed"). 4-arg callers unchanged.
- `Game.launch_aircraft_from_airbase(base, aircraft_id=None)`: targeted launch by `str(ac.id)` (absent → `return None`, never launch the wrong one). Omitting `aircraft_id` preserves FIFO `pop(0)`.
- `game.current_scenario.name` must be set before `start_recording()` or recordings are named "New Scenario".

### 🛑 MATCH-AOU solver — FROZEN (advisor-approved form)
`match_aou_MINLP_solver.py`. Advisor directive: **address allocation pathologies through scenario design, not solver constraints.** Do NOT re-add: a `single_agent_per_step` constraint, an objective fuel penalty, or a probability patch (all tried and rolled back). The only approved change is the per-target **round-trip** movement charge (`round_trip_cost`, `risk_factor=0`). Objective: `Σ_j y[j]·u_j·Π_k[1 − (1 − p_jk + EPSILON)^(Σ_i x[i,j,k])]`, `EPSILON = 1e-6`. `y[j]==1 ⇔ every step of task j has ≥1 agent ⇔ task appears in ≥1 assignment tuple` (the y/x linking constraints guarantee this — relied on by normalization and reward).

### 🛑 The BUILT graph layers are stable & reviewed
All nine graph layers below are BUILT, REVIEWED, and LOCKED (see §7 commits). Their **interfaces are contracts** — change them only through the same recon→prompt→review→lock discipline, and never in a way that weakens the no-communication guarantee (§3).

---

## 3. Architecture — the load-bearing invariants

Everything derives from **NO-COMMUNICATION**: at runtime an ego acts only on its **own** sensors; it never learns anything a peer sensed or did.

- **Per-ego PRIVATE belief.** Each ego owns a `Belief(tasks, solution)` — its private view. All N beliefs start byte-equal to the normalized static plan **A_init** at t=0, but are **mutually independent** (`deepcopy` tasks + `_copy_solution` solution). Editing ego A's belief never touches ego B's. The orchestrator owns the N beliefs; each consumer gets its slice.
- **`solution` is the source of truth; the graph is a STATELESS projection** rebuilt from `(world, solution)` every trigger, never mutated. Every "edit" is an edit to a belief's `solution`; the graph re-derives on the next build.
- **`tasks` are APPEND-ONLY** within an episode (positional `task_idx` indexes `solution` tuples). A pop-up is appended, never removed.
- **RL is EVENT-TRIGGERED**, not periodic. An ego flies A_init "blind" via the executor until an EVENT (from its OWN sensing) wakes the policy.
- **done-on-CONFIRMED-KILL.** An ego marks a target done only after it confirms the kill **within its own sensor range** (never learns a peer killed a far target).
- **Decision/effect/trigger layers are PURE** (no BLADE, no torch) and hand-testable.
- **Structural no-comms in the graph:** peer nodes are **featureless** (peer fuel/position/observation are dropped — sensing them would be a comms leak). A_init enters ONLY via `ASSIGNMENT` edges + the featureless peer nodes that anchor them. Runtime sensing is the ego's own `sensed` task-feature column, recomputed from the ego's position each build. "No `ASSIGNMENT` edge ⇒ genuine pop-up."

**Detection/attack range — ONE radius.** Sensing = attack = arrival = discovery = **`DETECTION_KM` (50 km)**, threaded from `graph_episode_setup` into the executor's `arrival_threshold_km`, the builder's `GraphObservationConfig.detection_range_km`, `split_tasks`' discovery adjacency, and the generator's connectivity (`VariationConfig.detection_km`). **Never** use BLADE `aircraft.range` for discovery (it varies per aircraft; we set the radius ourselves). No separate "detection > attack" radar range in the baseline.

---

## 4. The pipeline (end-to-end, all BUILT)

```
scenario_generator (clustered targets, per-zone discovery connectivity at DETECTION_KM)
  → setup_episode: env.reset → extract (agents, tasks) → split_tasks (partial ⊊ full)
                   → solve_and_normalize ×2 (partial→A_init/belief_tasks; full→oracle)
                   → N independent Beliefs → one GraphPlanExecutor
                   → EpisodeContext
  → run_episode(policy, ctx): per tick, TWO PHASES —
       Phase 1 (per ego, one obs snapshot, NO env.step):
         sensed_target_ids → decide_triggers → (on wake) _wake_decision:
           build_graph_observation → GraphEncoder → ActionHead
           → build_action_mask → sample_action → apply_meta_action → executor.resync
       Phase 2: env.step(executor.next_actions(obs))   # ONE step for the whole tick
     until is_done / terminated / truncated → EpisodeResult(trajectory)
  → compute_episode_reward(ctx, result): fills Transition.reward (terminal)
  → PPO buffer + evaluate_action + outer training loop  # BUILT (graph_ppo, graph_train)
  → [OPEN] centralized critic (CTDE)  # Phase B
```

A diagnostic rollout harness (`rl/training/graph_rollout.py`) wraps this exact
skeleton (generate -> setup -> run -> reward) for N episodes with a random-weight
policy — the outer-loop seam the PPO task will inherit. Validated: 20/20 episodes,
organic wakes (75% of episodes), rewards in [-1, ~0].

---

## 5. The layers (BUILT & REVIEWED — locked interfaces)

**Episode-setup (Stage 0) — `rl/training/graph_episode_setup.py` + `rl/training/belief.py`.**
`setup_episode(scenario_json, ...) -> EpisodeContext`. Wires env (`gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=…)`, `obs,info = env.reset()`, blue side by `side.name=="BLUE"`) → extract (`create_agents_from_scenario` picks blue; `generate_all_enemy_tasks`) → `split_tasks` → `solve_and_normalize` twice → N `Belief`s → one `GraphPlanExecutor`. `EpisodeContext` carries `env, game, agents, agent_ids, beliefs, executor, a_init, oracle_solution, oracle_tasks, split_meta, observation` (the reset seed the loop reads first). `Belief.independent(tasks, solution)` mints an independent per-ego copy. `solve_and_normalize(agents, tasks) -> (solution, belief_tasks, unselected)` = `MatchAou(...).solve("bonmin")` → `post_solve_filter_and_level(...)` (allocated-only filter + `task_idx` remap + `level`); **never returns the raw pre-filter list**. `split_tasks(all_tasks, partial_ratio, *, detection_km, max_attempts) -> (partial, full, meta)` = discovery-chain rejection sampler: builds task adjacency at `detection_km`, pins isolated targets to known, resamples until every hidden target has a KNOWN neighbour within `detection_km` (so it's discoverable at runtime), `partial ⊊ full`. Graph-native; imports NOTHING from the flat path. Independence + allocated-only proven in `_selftest`. `EpisodeContext.record: bool = False` — recording is ARMED iff a `recording_export_path` was given; setup never starts the recorder (the tick-loop drives it).

**Execution (Stage 1) — `utils/blade_utils/blade_graph_executor.py`.**
`GraphPlanExecutor` is the **sole** BLADE translation layer (move/launch/attack/RTB). `__init__(*, tasks, solution, agents, arrival_threshold_km=DETECTION_KM, add_return_to_base=True, nn_ordering=True, kill_confirm_ticks=60)`. **Per-ego private state:** `self.tasks: Dict[ego_id, List[Task]]` (fanned out at init; diverges only via `resync`), `self.plans` per-ego; `_resolve_step(ego_id, assignment)` is the sole reader of `self.tasks`. Key methods: `next_actions(obs) -> List[str]` (one command/ego/tick), `resync(new_solution, *, ego_id, tasks=None)` (swaps one ego's slice, **never resets `done`**), `is_done()` (skips `dead` egos, requires RTB latched), `sensed_target_ids(obs, ego_id) -> {id: unit}` (world-scan within `arrival_threshold_km`; the trigger's eyes). done-on-confirmed-kill, per-`(ego,target)` re-fire throttle, single-issue RTB latch (safe only while doctrine `AIRCRAFT_RTB_WHEN_OUT_OF_RANGE` is off — it is in `strike_training_4v5.json`), `dead` set for crashes. No-comms isolation proven in `_selftest` (ISO-1..3: a pop-up appended to ego A never enters ego B's task-view; same-index pop-ups resolve per-ego).

**Trigger (Stage 2) — `rl/action/graph_trigger.py`.**
`decide_triggers(belief_tasks, belief_solution, sensed_targets, eta=never_overdue, *, ego_id, clock) -> (new_tasks, new_solution, wake, events)`. PURE (no BLADE/torch), copy-on-write (never mutates inputs). The WHEN gate: **POP-UP** (ego senses an unassigned target → appends a pop-up Task to append-only `belief_tasks`) and **PEER-OVERDUE** (ego senses a peer's target AND its ETA passed → removes that peer tuple from the ego's `belief_solution` copy, so it reads as a pop-up — deterministic *gating*, the policy still chooses). ETA is dormant (`never_overdue` = +inf) for now.

**Build (Stage 3) — `rl/observation/graph_builder.py`.**
`build_graph_observation(scenario, agent_id, current_plan=None, current_time=0, tasks=None, solution=None, precedence_relations=None, config=None) -> GraphObservation`. Stateless projection of `(world, solution)`. `task_features[k, TASK_FEATURE_DIM]` (=6: utility, dist-to-ego, capable, reachable, probability, **sensed**; `TASK_FEATURE_DIM` is the single source of truth the encoder imports), `agent_features[a,1]` (fuel_norm: REAL for ego, `0.0` for peers), COO `edge_index`/`edge_type` over the `EdgeType` IntEnum, `time_norm`. **`ASSIGNMENT` is the only constructed relation** (`SPATIAL` reserved/unused — sensing moved to the `sensed` column; `PRECEDENCE` deferred). Agent set = `ego ∪ assigned same-side peers`. Requires the ego **airborne** (raises otherwise — always satisfied since build only follows a wake, which requires sensing, which requires airborne).

**Encode + decide (Stage 4) — `rl/agent/graph_encoder.py` + `rl/action/graph_action.py`.**
`GraphEncoder.forward(obs, edge_attr=None) -> Tensor[k, embed_dim]` — per-task-node embeddings (NOT pooled), single-graph (no batch dim). Defaults `model_dim=64, embed_dim=64, num_heads=4, num_layers=2, task_feat_dim=TASK_FEATURE_DIM`. Edge-masked symmetrized multi-head attention (torch/numpy only, no PyG/DGL) over `forward + reversed + SELF_LOOP` edges with a learned per-relation `type_bias`; learned TASK/EGO/PEER role embedding (node-typing done HERE, reserved MISSION 4th role); injected `time_norm`; self-loops guarantee no empty-softmax NaN. `pool()` = mean over nodes → the **hook for the future centralized critic** (no value head yet). `edge_attr` accepted but `None` today (reserved for expected-exec-time on ASSIGNMENT edges). — `ActionHead(embed_dim, hidden_dim=64, num_meta_actions=3).forward([k,embed]) -> [k,3]`. `build_action_mask(obs, ...) -> [k,3]` (hard physical/structural legality; `OPPORTUNISTIC_ENGAGEMENT` gated by `unassigned` AND `sensed`). `sample_action(logits, mask, deterministic=False) -> (meta:int, node_v:int, log_prob, entropy)`. `evaluate_action(logits, mask, meta, node_v) -> (log_prob, entropy)` re-scores a stored decision through the SAME private `_masked_dist` construction site (grad-mode caller-controlled; masked / out-of-bounds cells fail loud). **Meta-actions (3):** `PLAN_COMPLIANCE`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT` (Cooperative-Recovery removed — handled upstream by the peer-overdue trigger).

**Effect (Stage 5) — `rl/action/graph_effect.py`.**
`apply_meta_action(solution, obs, ego_id, meta_action, node_v, tasks) -> new_solution`. PURE (BLADE-free, torch-free), copy-on-write (`_copy_solution`, never mutates input). engage = add ego→task assignment; abort = drop the ego's assignments; comply = no-op. Does NOT touch the graph — the edge appears on the next rebuild.

**Resync (Stage 6)** — `GraphPlanExecutor.resync` (above): swaps the ego's plan slice without resetting `done`.

**Reward (Stage 7) — `rl/training/graph_reward.py`.**
`compute_episode_reward(ctx, result, cfg=RewardConfig()) -> EpisodeReward`. **Terminal, utility-based** (v1): `R = (U_achieved − c·U_aircraft·n_lost − U_oracle)/(|U_oracle| + eps_regret)`, placed on the last wake's `Transition` (others `0.0`; empty trajectory ⇒ nothing attached). `U_oracle = plan_value(ctx.oracle_solution, ctx.oracle_tasks)` — **bit-faithful to `MatchAou._add_objective`** (reuses the solver `EPSILON`; the `y[j]` factor is provably redundant given the y/x constraints; proven under bonmin in `_selftest` T1). `U_achieved = realized_utility(ctx.oracle_tasks, ctx.executor.done)` — full utility IFF all a task's targets are confirmed-killed, **deduped over ego**. `c = aircraft_penalty_coeff` default **0.0**; `n_lost = len(ctx.executor.dead)`; `eps_regret=1e-5` is a division guard (distinct from solver EPSILON). **No-comms:** a centralized/privileged TRAINING signal — MAY read global state, but MUTATES ONLY `Transition.reward` (proven byte-unchanged on real objects in T7). **KNOWN v1 assumption `probability=1.0`** (expected `U_oracle` vs realized `U_achieved` coincide only at p=1; `R∈[-1,~0]`; revisit at p<1).

**The two-phase tick (Stages 2–6) — `rl/training/graph_tick_loop.py`.**
`run_episode(policy, ctx, cfg=None, *, deterministic=False, max_ticks=None) -> EpisodeResult`. Strict two phases per tick: **Phase 1** runs every ego's `sensed → decide_triggers → (on wake) _wake_decision` against the SAME `obs` snapshot with **no** `env.step`; **Phase 2** issues ONE `env.step(executor.next_actions(obs))`. Because BLADE advances only after all egos decided on the identical snapshot, Phase-1 ego order cannot affect the outcome (structural no-comms; proven in `_selftest`: `env.step` count == tick count). `_wake_decision` is the per-wake chain (Stage 3→6) under `torch.no_grad`, editing ONLY the acting ego's belief. `Policy` (`build_policy()`) bundles encoder+head, built ONCE, lives across episodes. Seam for reward/PPO: `EpisodeResult.trajectory: List[Transition]`. The loop does NOT own the agent lifecycle (executor owns `dead`/`done`/`rtb`/`is_done`). **Recording:** armed by setup (`ctx.record`), driven here — start + forced t=0 frame before the loop, throttled `record_step` after each Phase-2 step (before the exit checks), forced terminal frame + `export_recording` after the loop (all exit paths). A pure READ of engine state; default off is a no-op — observational purity proven in `_selftest` TEST 1b (identical `(ended, ticks, n_wakes)` with recording on/off). Artifact: `{export_path}/{scenario_name} Recording {start} - {end}.jsonl`.

---

## 6. File map — "I want to…"

| … | Go to |
|---|---|
| Change episode setup / split / solve+normalize / Belief | `rl/training/graph_episode_setup.py`, `rl/training/belief.py` |
| Change the tick-loop / policy bundle / rollout | `rl/training/graph_tick_loop.py` |
| Run a diagnostic rollout (no training) | `rl/training/graph_rollout.py` (`RolloutConfig`, `run_rollout`) |
| Run PPO training / plot a run | `rl/training/graph_train.py` (`TrainConfig`, `train`, `plot_training`) |
| Change the reward | `rl/training/graph_reward.py` (`compute_episode_reward`/`plan_value`/`realized_utility`/`RewardConfig`) |
| Change WHEN the policy wakes | `rl/action/graph_trigger.py` (`decide_triggers`, `TriggerKind`, `never_overdue`) |
| Change the graph representation | `rl/observation/graph_builder.py` (`GraphObservation`, `GraphObservationConfig`, `EdgeType`, `TASK_FEATURE_DIM`) |
| Change the encoder | `rl/agent/graph_encoder.py` (`GraphEncoder`, `pool()` critic hook) |
| Change actions / mask / sampling | `rl/action/graph_action.py` (`MetaAction`, `ActionHead`, `build_action_mask`, `sample_action`) |
| Change how a decision edits the plan | `rl/action/graph_effect.py` (`apply_meta_action`) |
| Change BLADE execution / plan re-sync | `utils/blade_utils/blade_graph_executor.py` (`GraphPlanExecutor`) |
| Expose the ego's own sensing | `blade_graph_executor.py` → `sensed_target_ids` |
| Create Agents / Tasks from a scenario | `scenario_factory.py` → `create_agents_from_scenario` / `generate_all_enemy_tasks` (`probability=1.0`) / `iter_enemy_targets` + `make_attack_task` (utility: Facility 100 / Airbase 80 / Ship 95) |
| Change scenario content / zones / fleet / fuel tiers | `scenario_generator.py` (`VariationConfig`, `ScenarioGenerator`, `CLASS_RANGE_TIERS`) |
| Change generation-time discovery connectivity (Layer 1, at `DETECTION_KM`) | `scenario_generator.py` → `_ensure_discovery_chain` / `_compute_zone_bounds` / `_connect_zone_targets` |
| Change split-time discovery masking (Layer 2, at `DETECTION_KM`) | `rl/training/graph_episode_setup.py` → `split_tasks` |
| Change the solver objective/constraints | `match_aou_MINLP_solver.py` (extreme caution — §2) |
| Change post-solve scheduling / levels | `scheduling_utils.py`, `topology_utils.py` |
| Change domain objects | `agent.py`, `task.py`, `step.py` (`StepKind`), `location.py`, `capability.py` |

> Shared domain infra (`scenario_generator`, `scenario_factory`, `scheduling_utils`, `topology_utils`, solver, `models`, `blade_utils`) is used by the graph path and is NOT old-model. Hand-written `.md` API docs may lag — prefer the code.

---

## 7. Build history (graph RL orchestrator)

> **Hash convention:** a commit cannot cite its own SHA, so each lock's hash is recorded in the NEXT commit that touches this file — never in the lock itself. The amend route (commit → fill the hash → `--amend`) is **DEPRECATED**: amending shifts HEAD to a new SHA, leaving the recorded hash pointing at the dangling pre-amend commit (that is what produced the stale `f831e69`, fixed here to `95c3189`).

- `777bd85` — executor per-ego private task lists (no-comms ISO-1..3).
- `9a211ab` — encoder `task_feat_dim` derives from builder `TASK_FEATURE_DIM` (fix 5/6 desync).
- `1ba583c` — Stage-0 episode setup + `Belief` (allocated-only baseline, independent per-ego beliefs).
- `b4a29ba` — two-phase tick-loop + `Policy`/`Transition` seam.
- `87cb17b` — terminal utility-regret reward.
- `ae97d7d` — real discovery-chain split on unified `DETECTION_KM` (generator + split + executor).
- `95c3189` — full-pipeline rollout harness (pure consumer; 20-ep validation).
- `f680710` — BLADE playback recording wired (armed in setup via `EpisodeContext.record`, driven in `run_episode`; purity-proven, TEST 1b).
- `561b7cb` — docs: §7 hash-convention inline + recording-lock SHA fill.
- `814734e` — graph_builder: inline `_compute_fuel_norm` (sever the flat observation seam; Step 1 of the flat-path cleanup).
- `d9b8c17` — strip the five flat `__init__` re-export vectors + lock `tests/test_import_purity.py` (Step 2 of the flat-path cleanup).
- `ab54ac3` — delete the retired flat path: 38 paths removed (Step 3 of the flat-path cleanup).
- `7f324fd` — doc hygiene + workspace pruning: **Step 4, the FINAL lock of the cleanup phase.** ~20 stale docstring/comment sites reworded across the nine graph modules + `blade_executor_minimal` + `graph_executor_smoke` (present-tense references to the deleted flat path → truthful today or past-tense provenance); README's two stale `train_full.py` sites fixed; `.gitignore` gains `generated_scenarios/`; workspace pruned to two worktrees (main + `../flat-baseline`) with the three stale `claude/*` branches deleted. Comment-only in code — zero code lines changed; all six layer selftests + 12/12 import-purity green.
- `b96d29f` — **final doc sweep** (post-cleanup coda). Four flat-era docs deleted: `LOGS_GUIDE.md`, `RUN_SUMMARY.md`, `docs/INTEGRATION_GUIDE.md`, `docs/MATCH_AOU_API.md` — the first three document deleted code; the fourth documents the live solver but through a dead API (`StepType`, removed in `5eeaf3c`) in every example. All four preserved on `flat-final` (`4d44c34`) + tag `pre-cleanup`. README's Documentation section lost its three dead links (the two deleted `docs/` files + `RL_MODULE_DOCUMENTATION.md`, orphaned back in `ab54ac3`), leaving only the live `BLADE_API_DOCUMENTATION.md`. Untracked mid-cleanup snapshot `src/match_aou.zip` deleted. `training_output*/` added to `.git/info/exclude` — a local-only, never-tracked safety net shared by both worktrees, deliberately broader than `.gitignore`'s `training_output_*/`. Docs-only; 12/12 import-purity green.
- `be3729d` — per-episode RNG reseed in `graph_rollout` (PPO-phase step 1).
  Every episode now reseeds global `random` + torch with `base_seed+i` at the top
  of its iteration (the generator already uses its own `random.Random(seed)`),
  making episode i a pure function of its seed given the policy weights (still
  pinned ONCE before the loop by `torch.manual_seed(base_seed)`). Records gain
  `known_target_ids` — the t=0 known-split identity, snapshotted from
  `ctx.beliefs` BEFORE `run_episode` (after wakes the N beliefs legitimately
  diverge per ego; that divergence is the no-comms guarantee). Proven by a
  throwaway two-part check: (T1) same-config double run ⇒ field-identical
  records; (T2) episode-in-isolation reproduces the split of the same seed
  inside a longer run, with scenario content per seed differing ONLY in the two
  unseeded episode-tag fields (`/currentScenario/id` uuid4 + `/name` episode
  index — a finding to remember: scenario ids are NOT seed-derived; unit ids
  ARE template-stable). Regression: 15/15 pytest incl. 12/12 import purity.
- `830bd32` — `evaluate_action` + shared `_masked_dist` construction site in
  `graph_action` (PPO-phase step 2). `sample_action` and `evaluate_action` now
  build the joint masked distribution through ONE private helper, so rollout and
  PPO-update distributions are identical BY CONSTRUCTION (drift there is a silent
  research bug). `evaluate_action(logits, mask, meta, node_v) -> (log_prob,
  entropy)`: grad mode is caller-controlled (no no_grad inside); masked or
  out-of-bounds stored cells raise ValueError (fail loud — a masked stored action
  means mask reconstruction diverged). Proven: bit-identical pre/post-refactor
  baseline incl. a 50-draw RNG sweep; bitwise sample/evaluate agreement; epoch-0
  ratio exactly 1.0 on a real policy; finite grads on every exercised
  encoder+head param (edge_attr_proj legitimately unexercised, exact-name
  whitelisted); masked/out-of-bounds guards. Regression: 20/20 pytest incl.
  12/12 import purity; tick-loop selftest green end-to-end.
- `628e45f` — `graph_ppo`: the PPO core, Phase A actor-only (PPO-phase step 3).
  EpisodeRecord (per-ego chains — the Phase-B GAE seam contract) + PPOBuffer +
  compute_returns_and_advantages (THE REPLACEABLE COMPONENT: return == episode R at
  the dormant gamma=1.0; baseline = mean R over EPISODES incl. zero-wake;
  advantages normalized with eps guard) + clipped_surrogate + PPOUpdater (one Adam
  over encoder+head, per-transition re-encode -> rebuilt mask -> evaluate_action ->
  clip, entropy bonus, one backward/epoch, grad-norm clip; empty batch = clean
  no-op; NO value loss — PHASE-B SEAM comments mark where the critic joins).
  Proven in _selftest + tests/test_graph_ppo.py (18 tests): epoch-0 ratio == 1 and
  loss == -mean(A_norm); learning direction (positive-advantage action rises);
  clip branches hand-checked + clip_fraction > 0 live; per-ego grouping order;
  degenerate batches (all-same-R, empty, all-zero-wake) NaN-free; finite grads
  (edge_attr_proj exact-name exempt); import purity green.
- `PENDING` — `graph_train`: the outer PPO Trainer, Phase A (PPO-phase step 4 — the
  LAST piece of Phase A). New leaf module `rl/training/graph_train.py`, purely
  additive: no locked layer touched, and deliberately NOT in the import-purity
  ENTRY_MODULES (it imports BLADE, like graph_rollout). Wraps the locked pipeline into
  a real run: per iteration it collects `episodes_per_iteration` stochastic episodes
  through the rollout skeleton (one generator, per-episode reseed, env.close in
  finally) into a fresh PPOBuffer, runs ONE PPOUpdater.update built ONCE for the run,
  clears, and appends a scalar record. Owns the seeding schedule: train seed =
  base_seed + (iteration*eps + j); eval on a FIXED DISJOINT band (eval_base_seed + e),
  enforced by TrainConfig.validate (overlap raises). Deterministic eval every N iters
  on the held-out band (no buffer / no update). Save-only checkpoints
  (encoder+head+optimizer+PPOConfig); resume DEFERRED. 3-panel plot (learning curve vs
  R=0 oracle ceiling / meta-action mix / entropy) from the jsonl, drawn in a CHILD
  process with KMP_DUPLICATE_LIB_OK — torch+matplotlib abort together on this
  Windows/OpenMP stack, so the flag is confined to a numerics-free child and training
  never depends on matplotlib. Proven: tests/test_graph_train.py (8 tests — checkpoint
  round-trip incl. optimizer state, seed-schedule + band disjointness, plot-from-jsonl;
  suite 38 -> 46, import purity 12/12) and _selftest under nlp_env (short real run;
  EVAL PURITY — train records byte-identical eval-on vs eval-off; honest zero-wake via
  max_ticks=5). FINDING (see §8): the Trainer is correct, but every episode returns
  R ~ -1/3, so a baseline run as-is will NOT learn — a reward/scenario issue, not a
  Trainer bug.

---

## 8. OPEN (not built)

- **Phase-A baseline run — BLOCKED on a flat-reward finding (the Trainer itself is BUILT — §7, `graph_train.py`).** The full PPO loop is done (generate → setup → run → reward → `EpisodeRecord` → `PPOUpdater.update`, with logging / eval / checkpointing). BUT a real short run shows **every episode returns R ~ -1/3 to ~12 dp, across all seeds, train and eval** → `adv_std_raw ~ 0` → advantages ~ 0 → the only gradient is the entropy bonus. So a baseline run as-is will NOT learn (the reward does not discriminate between episodes). Episodes end `done` with `kills_mean ~ 4.25` — systematic under-achievement (~2/3 of oracle utility every scenario), NOT the 2:1-stacking truncation path below. Mechanism TBD in `graph_reward` + scenario design; resolve BEFORE spending compute on the reported baseline. `adv_std_raw` sits in every train record precisely to keep this visible. (Note: this also gates Phase B — a critic predicting a scenario-constant R yields ~0 advantage too.)
- **Centralized critic / value head (CTDE):** size-agnostic value estimator off `GraphEncoder.pool()`; needs a dedicated CTDE design (training on all-agent info while keeping execution no-comms). **A new planning chat.**
- **Reward densification + p<1:** per-wake/dense regret (vs today's terminal scalar) and the operand-scale rework for `probability<1.0` (expected-oracle vs realized-achieved diverge below p=1).
- **Solver 2:1 stacking (scenario-design fix, NOT solver constraints):** the anti-div-by-zero `EPSILON` nudges utility enough to assign 2 agents even at `probability=1.0`; a redundant agent chasing an already-killed target never proximity-confirms, so episodes end via `truncated`. The learned policy should recover this via `SELF_PRESERVATION_ABORT`→RTB once trained; the root fix is `EPSILON`/scenario-side.
- **Peer-dropout as a deterministic pre-build trigger** (advisor-pending, separate chat): move "peer overdue ⇒ drop its ASSIGNMENT edge" out of the policy; needs a deadline param + a `was_assigned_to_peer` feature to keep recovered-vs-popup semantics.
- **`reachable_by_ego` marginal-detour model:** `graph_builder._reachable_by_ego` is a conservative round-trip placeholder; intended model is marginal detour-cost vs remaining fuel slack (isolated to the builder; the mask reads the column).
- **`assigned_to_peer` as a task-feature column** (currently edge-derived), **real ETA** (enables PEER-OVERDUE; currently `never_overdue`), **`kill_confirm_ticks` calibration** once p<1 lands, **re-enable fuel-damage** after a clean baseline.
> **Flat-path cleanup phase: CLOSED.** All four steps are locked (§7: `814734e`, `d9b8c17`, `ab54ac3`, `7f324fd`), plus a final doc sweep as a coda. The 38 deleted paths are preserved on `flat-final` (`4d44c34`) + tag `pre-cleanup`. Nothing in `src/` or `tools/` references the flat path. `LOGS_GUIDE.md`, `RUN_SUMMARY.md`, `docs/MATCH_AOU_API.md`, and `docs/INTEGRATION_GUIDE.md` were **deleted in the final sweep** (superseding the earlier decision to keep the first two as run-log records — the run logs live on the preserved refs, and `train_full` prose in `main` was more confusing than useful). **Kept, flagged for future passes:** `README.md` (minimally truthful — dead links pruned, but its MAPPO/flat architecture prose still awaits its own rewrite task) and `docs/BLADE_API_DOCUMENTATION.md` (documents the frozen vendored engine; unaudited against the current fork).
