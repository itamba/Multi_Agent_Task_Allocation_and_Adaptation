# CLAUDE.md

Repository guidance for Claude Code. This is the **Multi-Agent GRAPH RL** project
(MATCH-AOU Phase-2): a no-communication multi-agent policy that adapts a static task
allocation at runtime over a graph representation. The old flat RL path is **retired** — deleted
from `main` in Step 3 of the cleanup, preserved on TWO DISTINCT refs — branch `flat-final`
(`4d44c34`) and the annotated tag `pre-cleanup` (commit `561b7cb`, the last commit before the
cleanup began) — **this document describes the graph model only.**

---

## 1. Communication & workflow (read first)

- **User speaks Hebrew.** Code, comments, and docs stay in English.
- **Packet-driven scope.** One closed task at a time. Follow the packet's declared scope,
  grade, and 1–3 proof obligations. Explain material implementation choices before making
  them; stop only for a blocking ambiguity, a red-line conflict, or a material deviation
  from the packet — not after every file.
- **Grade = the trust policy, declared in the packet.** It states how much review the change
  earns, not how hard it is. **C** — hygiene, wording, docs, unreachable fallbacks: trusted,
  no review, no lock ceremony. **B** — "the pipeline runs or it does not": one test on the
  main path, no branch coverage, no goldens; the orchestrator reads the changed files from
  the repo. **A** — a research claim is at stake (no-communication isolation, route-prediction
  and placement fidelity, reproducibility of the geometry, source-of-truth / append-only):
  1–3 proof obligations declared up front, and the orchestrator must approve the exact
  full SHA of the reviewed commit; when the change touches cross-ego isolation or a §5
  locked layer it earns line-by-line review — GPT reads the exact GitHub `base...candidate`
  comparison; under `CLAUDE_MOUNTED_MAIN` the hunks come from CC (see Grade-A routing default
  below). Grade A is set by consequence, not by difficulty — a wrong A is a silent false
  result, not a crash.
- **Candidate commits are required, and transport is MODE-DEPENDENT.** Implement + run the
  required tests, then create the commit the orchestrator will review. Transport is never
  approval: a commit stays `READY_FOR_REVIEW / UNREVIEWED` until the orchestrator approves
  its exact full SHA — and that includes a commit already pushed to `main`. Exactly one of
  the two modes below applies, and **the packet or the user must state which one**; never
  silently infer that task branches or PRs are accessible. Shared by both modes: start from
  the packet's verified full base SHA, stage exactly the declared files, and keep one
  focused commit per task. If the user explicitly marks a task `local-only`, do not push;
  report that the orchestrator cannot independently review it until that restriction is
  lifted.
- **No premature docs.** Don't spawn `README`/`SUMMARY`/per-file docs unasked. One consolidated doc per stable component.
- **Minimal files.** Prefer extending a module over spawning `foo_utils.py` + `foo_config.py`.
- **Transport mode `GPT_GITHUB`** — the GPT orchestrator inspects GitHub directly (branches,
  PRs, files, exact SHAs):
  - start the task branch from the packet's verified full base SHA;
  - create and push a candidate commit on that task branch;
  - open or update its draft PR;
  - the GPT orchestrator reviews the exact `base...candidate` state before merge;
  - do not push directly to `main`, merge, rebase, or force-push unless the user explicitly
    changes the task's authorization.
- **Transport mode `CLAUDE_MOUNTED_MAIN`** — the Claude orchestrator's shared repository view
  is a synchronized mounted snapshot of `main`, exposed as a search interface: not a live
  connector, not a filesystem, not `git`. It retrieves and quotes file content, but it cannot
  list files, count occurrences, prove that something is ABSENT, produce a diff, read history,
  or select an arbitrary SHA, and it can LAG the true `main` head — a stale read is sync lag,
  not regression. Task branches and PRs must **not** be assumed accessible:
  - before editing, verify a clean checkout and exact equality between local `HEAD`,
    `origin/main`, and the packet's full base SHA;
  - stage only the declared files, create one focused commit, and push it directly to `main`
    with a normal **non-force** push;
  - the pushed commit stays `UNREVIEWED` until the Claude orchestrator approves that exact
    full SHA. For Grade A work, do not claim a lock or begin dependent work until that
    post-push exact-SHA approval occurs;
  - if review finds a problem, correct it with a NEW follow-up commit on `main` — never
    amend, rewrite history, reset published commits, or force-push;
  - if `main` has advanced, the push is rejected, or the mounted checkout cannot prove its
    state, stop and report `BLOCKED` (do not pull, merge, rebase, reset, stash, or delete).
- **Grade-A routing default.** `GPT_GITHUB` is the only mode that gates `main` behind a
  reviewable branch, so Grade-A work is routed to the GPT orchestrator by default whenever it
  is available. Grade A under `CLAUDE_MOUNTED_MAIN` is a declared exception — the packet must
  say so explicitly — and carries two consequences. First, the candidate is reviewed only
  AFTER it is pushed, so `main` knowingly carries an `UNREVIEWED` commit until its exact full
  SHA is approved: no lock, and no dependent work, before that approval. Second, CC MUST
  supply focused changed hunks plus targeted test evidence; this is mandatory rather than a
  fallback, because a mounted snapshot shows current state and can never show
  `base...candidate`.
- **Fix chain:** review corrections stay in the same named CC session and produce a NEW
  commit and a new SHA to review — never a rewrite of the reviewed one. In `GPT_GITHUB` they
  land on the same task branch; merge only the unchanged approved head, prefer a merge
  strategy that preserves the reviewed commit, and if integration rewrites it, verify the
  resulting tree before recording its SHA as a lock. In `CLAUDE_MOUNTED_MAIN` they land as
  follow-up commits on `main`, and the lock is the last approved pushed SHA.
- **Status block (mode-aware):** every task ends with: state (`READY_FOR_REVIEW /
  UNREVIEWED` or `BLOCKED`); transport mode; full base SHA; full review SHA; grade; files
  changed; tests / checks run; proof-obligation evidence; deviations judged against the
  packet; `NEW FACTS LEARNED` anchored by **file + symbol or exact string** (mandatory, use
  `NONE` when empty); `CLAUDE.md` deltas needed; and final working-tree state. Task branch
  and draft PR number / URL are required **only** in `GPT_GITHUB` mode; in
  `CLAUDE_MOUNTED_MAIN`, report `main` plus the verified pushed `origin/main` SHA.
- **Output discipline:** never paste whole files, full transcripts, or a large full diff into
  chat. GPT inspects the exact GitHub `base...candidate` diff itself; under
  `CLAUDE_MOUNTED_MAIN` CC supplies focused changed hunks and targeted test evidence on
  request — required for Grade A. Otherwise return targeted run output and direct answers
  only, and put a genuinely long report in the repo.
- **Environment:** Windows + PyCharm terminal + `nlp_env` conda env, Python 3.10+. Avoid POSIX-only idioms; use Python/PowerShell equivalents. Run from repo root.
- **`pytest` is NOT installed in `nlp_env`** — it lives in the base env, so `python -m pytest` under `nlp_env` fails with `No module named pytest`. Solver-free suites can be run with the base-env `pytest`; test files carrying a `__main__` runner (e.g. `tests/test_graph_train.py`) should ALSO be run directly under `nlp_env`.
- **`blade` and `gymnasium` DO resolve in the base env** (measured at `dd14ab4`): `import blade` returns the SAME vendored fork the editable install points at (`src/match_aou/integrations/panopticon-main/gym/blade/__init__.py`), and `gymnasium` imports cleanly. This CLOSES the former "is the base env the same fork?" question — a base-env test may build a `Game`, `gymnasium.make("blade/BLADE-v0", …)` and `env.reset()`, which is what lets `tests/test_graph_setup_seam.py`'s BLADE tier run under plain `pytest`. **The missing base-env dependency is BONMIN, not BLADE**: `shutil.which("bonmin")` is `None` in the base env and `…/envs/nlp_env/Library/bin/bonmin.EXE` under `nlp_env`. Nothing here relaxes the solver rule below — anything that SOLVES still runs under `nlp_env`.
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
Every graph layer in §5 — the nine pipeline stages plus the trainer contract and the
FD-BASELINE-v1 difficulty factor — is BUILT, REVIEWED, and LOCKED (see §7 commits). Their **interfaces are contracts** — change them only through the same recon→prompt→review→lock discipline, and never in a way that weakens the no-communication guarantee (§3).

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

**Launch point == the BLUE airbase.** Each aircraft record carries its airbase's own
coordinates, and `launch_aircraft_from_airbase` never repositions — so every ego goes
airborne OVER its base. Two consequences the geometry depends on: all egos share ONE
origin (route geometry is a star from that point, so a target placed near the origin
is sensed by ALL egos, privately but simultaneously — placement must favour the far
half of a route to preserve the asymmetry no-comms is about), and
`Agent.location == Agent.return_location`, making the solver's `round_trip_cost` a
symmetric out-and-back. Guarded by `tests/test_scenario_construction_preconditions.py`
P1/P2 and by `_adjust_aircraft_count`'s base-anchored fallback.

---

## 4. The pipeline (end-to-end, all BUILT)

`setup_episode` has TWO explicit paths, selected by whether the
`(n_hidden, placement_rng)` PAIR was supplied — never inferred from `partial_ratio`.
Both end at the same `EpisodeContext` and both solve TWICE, independently.

```
LEGACY SPLIT PATH  (both omitted — unchanged, still the default signature)
scenario_generator (clustered targets, per-zone discovery connectivity at DETECTION_KM)
  → setup_episode: env.reset → extract (agents, tasks) → split_tasks (partial ⊊ full)
                   → solve_and_normalize ×2 (partial→A_init/belief_tasks; full→oracle)
                   → N independent Beliefs → one GraphPlanExecutor → EpisodeContext

CONSTRUCTION PATH  (both supplied — what training and rollout use)
scenario_generator (KNOWN-ONLY world: n_known targets, Layer 1 OFF, geometry STRICT)
  → setup_episode: env-1.reset → extract (agents, known tasks)
                   → solve_and_normalize (known → A_init/belief_tasks)
                   → place_hidden_targets (LOCKED B2, one per non-empty ego route)
                   → patch the scenario JSON (append n_hidden enemy airbases)
                   → CLOSE env-1
                   → env-2.reset on the patched JSON → RE-EXTRACT agents + all tasks
                   → solve_and_normalize (ALL env-2 targets → oracle)
                   → N independent Beliefs + one GraphPlanExecutor, built from
                     ENV-2 OBJECTS ONLY → EpisodeContext
                   # split_tasks is NOT called; discovery is guaranteed by geometry
```

From the `EpisodeContext` onward BOTH paths are identical:

```
EpisodeContext
  → run_episode(policy, ctx, fuel_damage=None): per tick, TWO PHASES —
       TOP OF TICK (optional, FD-BASELINE-v1): fuel_damage.maybe_apply(obs, tick)
         # ONE physical mutation of the selected ego's live current_fuel, at most once
         # per episode, BEFORE any ego is processed. Returns that ego's id on the firing
         # tick and None otherwise.
       Phase 1 (per ego, one obs snapshot, NO env.step):
         sensed_target_ids → decide_triggers(..., fuel_damage=<ego is the selected one>)
           → (on wake) _wake_decision:
             build_graph_observation → GraphEncoder → ActionHead
             → build_action_mask → sample_action → apply_meta_action → executor.resync
       Phase 2: commands = executor.next_actions(obs)
                fuel_damage.note_commands(commands)   # READ-ONLY measurement
                env.step(commands)                    # ONE step for the whole tick
     until is_done / terminated / truncated → EpisodeResult(trajectory)
  → compute_episode_reward(ctx, result, cfg.reward_config()): fills Transition.reward
  → PPO buffer + evaluate_action + outer training loop  # BUILT (graph_ppo, graph_train)
  → [OPEN] centralized critic (CTDE)  # Phase B
```

**The exogenous-event seam (FD-BASELINE-v1) sits at the TOP of a tick, never inside
Phase 1.** `run_episode`'s `fuel_damage` parameter is optional and defaults to `None`, so
a loop without it is byte-unchanged. When supplied, the controller is consulted ONCE per
tick before the per-ego loop begins, and four properties follow from that placement:

- the physical mutation of one live BLADE aircraft's `current_fuel` happens before any
  ego senses, so **every ego — the damaged one included — observes the SAME post-event
  world snapshot** and Phase-1 ego ITERATION ORDER still cannot affect the outcome;
- **only the selected ego receives the ego-local `FUEL_DAMAGE` wake** (`decide_triggers`
  is called with `fuel_damage=True` for that ego alone, and `False` for every peer);
- the trigger **edits neither `belief_tasks` nor `belief_solution`** — the changed
  quantity is the ego's own live fuel, which the builder reads off the aircraft, so the
  event only sets `wake`;
- Phase-2 emitted commands are observed **only for measurement** (`note_commands` is a
  read-only scan that records whether the selected ego's actual
  `aircraft_return_to_base` command was issued). Nothing in the loop's control flow reads
  it back.

Both `rl/training/graph_train.py` (`_run_one_episode`) and the diagnostic harness
`rl/training/graph_rollout.py` (`run_rollout`) drive the CONSTRUCTION path: they
generate the known-only world, then call `setup_episode(..., n_hidden=cfg.n_hidden,
placement_rng=random.Random(seed))`. The placement rng is explicit and per-episode —
it never rides on module-global `random` — so an episode's hidden geometry is a pure
function of its seed. Neither harness passes `partial_ratio` any more; the legacy
split surface is retained and tested but is not on this path.

Reference evidence at the B3 lock (`dd14ab4`, ONE seed-0 episode through
`graph_rollout --episodes 1 --seed 0`): 3 agents, 3 known + 3 hidden = 6 targets,
`ended=done`, 4 organic wakes, reward `-0.3333`. That remains one reference episode,
not a baseline sweep.

The first real post-B3 instrumented probe later ran against exact code SHA
`a3f0838616990987bcb8a51665fa75d84edf5952`: two iterations × four scheduled training
episodes, with four fixed held-out episodes before and after training. It measured
`pre_update = -0.4999997395829586` (4/4), 7/8 successful training attempts, one accounted
`setup` failure at seed 2, 24 wake-transitions, two PPO updates, and
`post_update = 5.000007394910353e-7` (4/4; numerical zero). This establishes headroom and
a functioning learning loop, but it is a SHORT PROBE, not a baseline.

---

## 5. The layers (BUILT & REVIEWED — locked interfaces)

**Episode-setup (Stage 0) — `rl/training/graph_episode_setup.py` + `rl/training/belief.py`.**
`setup_episode(scenario_json, ..., n_hidden=None, placement_rng=None) -> EpisodeContext`.
Wires env via `_build_env` (`gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=…)`, `obs,info = env.reset()`, blue side by `side.name=="BLUE"`) → `_extract_world` (`create_agents_from_scenario` picks blue; `generate_all_enemy_tasks`) → `solve_and_normalize` twice → `_finish_context` (N `Belief`s + one `GraphPlanExecutor`). `EpisodeContext` carries `env, game, agents, agent_ids, beliefs, executor, a_init, oracle_solution, oracle_tasks, split_meta, observation` (the reset seed the loop reads first), `record`, and `placements`. `Belief.independent(tasks, solution)` mints an independent per-ego copy. `solve_and_normalize(agents, tasks) -> (solution, belief_tasks, unselected)` = `MatchAou(...).solve("bonmin")` → `post_solve_filter_and_level(...)` (allocated-only filter + `task_idx` remap + `level`); **never returns the raw pre-filter list**. Graph-native; imports NOTHING from the flat path. Independence + allocated-only proven in `_selftest`. `EpisodeContext.record: bool = False` — recording is ARMED iff a `recording_export_path` was given; setup never starts the recorder (the tick-loop drives it), and only the RETURNED env is ever armed.

**PATH SELECTION (`_resolve_construction_mode`, runs BEFORE any BLADE object exists).**
`n_hidden` and `placement_rng` are a PAIR. Both omitted → the LEGACY split path
(`_setup_episode_legacy`), behaviourally unchanged. Both supplied → CONSTRUCTION mode
(`_setup_episode_construction`). Exactly one supplied → `ValueError`. `n_hidden` must be a
genuine non-negative `numbers.Integral` (`bool` rejected, mirroring B2's `_as_assignment`);
`placement_rng` must be an explicit `random.Random`, never module-global randomness. The
mode is NEVER inferred from `partial_ratio`. `n_hidden=0` is a legal construction probe: it
places nothing, patches nothing, and still does not call `split_tasks`.

**LEGACY PATH — `split_tasks(all_tasks, partial_ratio, *, detection_km, max_attempts) -> (partial, full, meta)`** = discovery-chain rejection sampler: builds task adjacency at `detection_km`, pins isolated targets to known, resamples until every hidden target has a KNOWN neighbour within `detection_km` (so it's discoverable at runtime), `partial ⊊ full`. Retained, tested, and reachable; the construction path simply never calls it.

**CONSTRUCTION PATH — solve → place → patch → reload.** Env-1 is TEMPORARY: reset, extract,
`_require_airbase_only_targets` (every enemy unit must be a BLADE `Airbase`; a SAM facility
or ship raises — mixed target semantics are a separate design task, and `TrainConfig` /
`RolloutConfig` `validate()` reject `include_sams=True` up front), `_shared_launch_point`
(verifies ONE origin for every ego AND `Agent.location == Agent.return_location`),
solve the known set → `a_init` + `belief_tasks`, then the LOCKED B2
`place_hidden_targets(a_init, belief_tasks, launch_point, PlacementParameters(detection_km),
placement_rng)`. **Cardinality is exact:** `len(placements) == n_hidden` or `RuntimeError` —
never truncated, padded, duplicated, or redistributed across routes (B2's contract is one
placement per non-empty ego route, so a solve that leaves an ego idle FAILS the episode).
`build_patched_scenario` then patches ONCE — deep-copy a safe enemy-airbase prototype
(`_select_hidden_prototype`: enemy-side, schema-complete, EMPTY aircraft inventory, single
unambiguous `sideId`), fresh uuid4, deterministic unique name, the placement coordinates,
empty `aircraft`, appended at the END of `currentScenario.airbases` so no known target
moves. Env-1 is closed in a `finally` on EVERY success and failure path.

**`_build_env` OWNS CLEANUP UNTIL IT RETURNS** (review fix, part of the lock): the window
between `gymnasium.make` and its `return` is reachable by no caller guard — the callers'
`finally`/`except` blocks are keyed on the value it hands back — so any `BaseException`
there (a failing `env.reset()`, or the side-selection loop after it) closes the environment
exactly once via `_close_quietly` and re-raises the ORIGINAL exception unchanged. Both
environments are built through this one helper, so the guard covers both windows.

**ENV-2 IS THE SOLE RUNTIME SOURCE OF TRUTH.** It is reset on the patched JSON, agents and
tasks are RE-EXTRACTED from it, `_require_agent_ids_preserved` requires the ORDERED agent-id
list to be identical across the reload (else A_init's keys no longer address the runtime
egos), every known target must still be present, and the world must hold exactly
`known + n_hidden` targets. `_rematerialize_known_tasks(world_tasks, known_target_ids)`
re-looks-up the belief tasks as ENV-2 objects **by target id, in A_init's exact positional
order**, so `task_idx` stays valid. The oracle is a SEPARATE solve over ALL env-2 targets
(so every hidden target is in it). If anything after env-2's reset fails, env-2 is closed
before the error propagates, and only env-2 is ever returned.
**NO env-1 `Agent` or `Task` object may enter the returned context** — only pure data
crosses (the normalized `a_init` assignments and the ordered known-target id strings).

**`EpisodeContext.placements: Tuple[HiddenPlacement, ...]`** is the id-free placement audit
(`()` on the legacy path and on an `n_hidden=0` probe). Construction `split_meta` is
TRUTHFUL — it never claims `split_tasks` ran: `outcome`/`mode` are `"construction"`, and it
carries `known`, `hidden`, `partial`, `full` (WORLD TARGETS EMITTED, keeping the legacy key
names the training/rollout records read), plus `n_hidden_requested`, `allocated_known`, and
`geometric_fingerprint` — coordinates only, because generated uuids are not seed-derived
(§8). Reproducibility is judged by that fingerprint, never by id.

**Execution (Stage 1) — `utils/blade_utils/blade_graph_executor.py`.**
`GraphPlanExecutor` is the **sole** BLADE translation layer (move/launch/attack/RTB). Its intra-level travel ordering comes from the SHARED pure helper `nearest_neighbor_order`, imported from `utils/scheduling_utils.py` — the SAME function `graph_hidden_placement.predict_route` calls, which is what keeps online execution and offline route prediction from drifting apart (`2a3f89c`). `__init__(*, tasks, solution, agents, arrival_threshold_km=DETECTION_KM, add_return_to_base=True, nn_ordering=True, kill_confirm_ticks=60)`. **Per-ego private state:** `self.tasks: Dict[ego_id, List[Task]]` (fanned out at init; diverges only via `resync`), `self.plans` per-ego; `_resolve_step(ego_id, assignment)` is the sole reader of `self.tasks`. Key methods: `next_actions(obs) -> List[str]` (one command/ego/tick), `resync(new_solution, *, ego_id, tasks=None)` (swaps one ego's slice, **never resets `done`**), `is_done()` (skips `dead` egos, requires RTB latched), `sensed_target_ids(obs, ego_id) -> {id: unit}` (world-scan within `arrival_threshold_km`; the trigger's eyes). done-on-confirmed-kill, per-`(ego,target)` re-fire throttle, single-issue RTB latch (safe only while doctrine `AIRCRAFT_RTB_WHEN_OUT_OF_RANGE` is off — it is in `strike_training_4v5.json`), `dead` set for crashes. No-comms isolation proven in `_selftest` (ISO-1..3: a pop-up appended to ego A never enters ego B's task-view; same-index pop-ups resolve per-ego).

**Trigger (Stage 2) — `rl/action/graph_trigger.py`.**
`decide_triggers(belief_tasks, belief_solution, sensed_targets, eta=never_overdue, *, ego_id, clock, fuel_damage=False) -> (new_tasks, new_solution, wake, events)`. PURE (no BLADE/torch), copy-on-write (never mutates inputs). The WHEN gate over THREE `TriggerKind` members: **POP-UP** (ego senses an unassigned target → appends a pop-up Task to append-only `belief_tasks`), **PEER-OVERDUE** (ego senses a peer's target AND its ETA passed → removes that peer tuple from the ego's `belief_solution` copy, so it reads as a pop-up — deterministic *gating*, the policy still chooses), and **FUEL_DAMAGE** (FD-BASELINE-v1). ETA is dormant (`never_overdue` = +inf) for now. `FUEL_DAMAGE` is EXOGENOUS — it cannot be detected from sensing, so the orchestrator passes `fuel_damage=True` for AT MOST ONE ego per tick; the flag defaults to `False`, so every pre-FD caller is byte-unchanged. It **edits NEITHER `belief_tasks` NOR `belief_solution`** (the changed quantity is the ego's own live fuel, which the builder reads off the aircraft) and only sets `wake`, appending a `(FUEL_DAMAGE, NO_TASK_INDEX)` event — `NO_TASK_INDEX = -1` is a sentinel, deliberately not `0`, because `0` is a valid task index. A tick carrying both a fuel-damage event and a pop-up still produces exactly ONE wake.

**Build (Stage 3) — `rl/observation/graph_builder.py`.**
`build_graph_observation(scenario, agent_id, current_plan=None, current_time=0, tasks=None, solution=None, precedence_relations=None, config=None) -> GraphObservation`. Stateless projection of `(world, solution)`. `task_features[k, TASK_FEATURE_DIM]` (=6: utility, dist-to-ego, capable, reachable, probability, **sensed**; `TASK_FEATURE_DIM` is the single source of truth the encoder imports), `agent_features[a,1]` (fuel_norm: REAL for ego, `0.0` for peers), COO `edge_index`/`edge_type` over the `EdgeType` IntEnum, `time_norm`. **`ASSIGNMENT` is the only constructed relation** (`SPATIAL` reserved/unused — sensing moved to the `sensed` column; `PRECEDENCE` deferred). Agent set = `ego ∪ assigned same-side peers`. Requires the ego **airborne** (raises otherwise — always satisfied since build only follows a wake, which requires sensing, which requires airborne).

**Encode + decide (Stage 4) — `rl/agent/graph_encoder.py` + `rl/action/graph_action.py`.**
`GraphEncoder.forward(obs, edge_attr=None) -> Tensor[k, embed_dim]` — per-task-node embeddings (NOT pooled), single-graph (no batch dim). Defaults `model_dim=64, embed_dim=64, num_heads=4, num_layers=2, task_feat_dim=TASK_FEATURE_DIM`. Edge-masked symmetrized multi-head attention (torch/numpy only, no PyG/DGL) over `forward + reversed + SELF_LOOP` edges with a learned per-relation `type_bias`; learned TASK/EGO/PEER role embedding (node-typing done HERE, reserved MISSION 4th role); injected `time_norm`; self-loops guarantee no empty-softmax NaN. `pool()` = mean over nodes → the **hook for the future centralized critic** (no value head yet). `edge_attr` accepted but `None` today (reserved for expected-exec-time on ASSIGNMENT edges). — `ActionHead(embed_dim, hidden_dim=64, num_meta_actions=3).forward([k,embed]) -> [k,3]`. `build_action_mask(obs, ...) -> [k,3]` (hard physical/structural legality; `OPPORTUNISTIC_ENGAGEMENT` gated by `unassigned` AND `sensed`). `sample_action(logits, mask, deterministic=False) -> (meta:int, node_v:int, log_prob, entropy)`. `evaluate_action(logits, mask, meta, node_v) -> (log_prob, entropy)` re-scores a stored decision through the SAME private `_masked_dist` construction site (grad-mode caller-controlled; masked / out-of-bounds cells fail loud). **Meta-actions (3):** `PLAN_COMPLIANCE`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT` (Cooperative-Recovery removed — handled upstream by the peer-overdue trigger).

**Effect (Stage 5) — `rl/action/graph_effect.py`.**
`apply_meta_action(solution, obs, ego_id, meta_action, node_v, tasks) -> new_solution`. PURE (BLADE-free, torch-free), copy-on-write (`_copy_solution`, never mutates input). engage = add ego→task assignment; abort = drop the ego's assignments; comply = no-op. Does NOT touch the graph — the edge appears on the next rebuild.

**Resync (Stage 6)** — `GraphPlanExecutor.resync` (above): swaps the ego's plan slice without resetting `done`.

**Reward (Stage 7) — `rl/training/graph_reward.py`.**
`compute_episode_reward(ctx, result, cfg=RewardConfig()) -> EpisodeReward`. **Terminal, utility-based** (v1): `R = (U_achieved − c·U_aircraft·n_lost − U_oracle)/(|U_oracle| + eps_regret)`, placed on the last wake's `Transition` (others `0.0`; empty trajectory ⇒ nothing attached). `U_oracle = plan_value(ctx.oracle_solution, ctx.oracle_tasks)` — **bit-faithful to `MatchAou._add_objective`** (reuses the solver `EPSILON`; the `y[j]` factor is provably redundant given the y/x constraints; proven under bonmin in `_selftest` T1). `U_achieved = realized_utility(ctx.oracle_tasks, ctx.executor.done)` — full utility IFF all a task's targets are confirmed-killed, **deduped over ego**. `c = aircraft_penalty_coeff` — this module's own default is **0.0**, but BOTH harnesses now pass an explicit `RewardConfig(aircraft_penalty_coeff=2.25)` (FD-BASELINE-v1, below); the FORMULA is unchanged. `n_lost = len(ctx.executor.dead)`; `eps_regret=1e-5` is a division guard (distinct from solver EPSILON). **No-comms:** a centralized/privileged TRAINING signal — MAY read global state, but MUTATES ONLY `Transition.reward` (proven byte-unchanged on real objects in T7). **KNOWN v1 assumption `probability=1.0`** (expected `U_oracle` vs realized `U_achieved` coincide only at p=1; `R∈[-1,~0]`; revisit at p<1).

**The two-phase tick (Stages 2–6) — `rl/training/graph_tick_loop.py`.**
`run_episode(policy, ctx, cfg=None, *, deterministic=False, max_ticks=None, fuel_damage=None) -> EpisodeResult`. Strict two phases per tick: **Phase 1** runs every ego's `sensed → decide_triggers → (on wake) _wake_decision` against the SAME `obs` snapshot with **no** `env.step`; **Phase 2** issues ONE `env.step(executor.next_actions(obs))`. The optional `fuel_damage` controller (FD-BASELINE-v1) is consulted at the TOP of a tick, before Phase 1, and its Phase-2 `note_commands` call is a read-only measurement — see §4 and the FD contract below; `None` (the default) leaves the loop byte-unchanged. Because BLADE advances only after all egos decided on the identical snapshot, Phase-1 ego order cannot affect the outcome (structural no-comms; proven in `_selftest`: `env.step` count == tick count). `_wake_decision` is the per-wake chain (Stage 3→6) under `torch.no_grad`, editing ONLY the acting ego's belief. `Policy` (`build_policy()`) bundles encoder+head, built ONCE, lives across episodes. Seam for reward/PPO: `EpisodeResult.trajectory: List[Transition]`. The loop does NOT own the agent lifecycle (executor owns `dead`/`done`/`rtb`/`is_done`). **Recording:** armed by setup (`ctx.record`), driven here — start + forced t=0 frame before the loop, throttled `record_step` after each Phase-2 step (before the exit checks), forced terminal frame + `export_recording` after the loop (all exit paths). A pure READ of engine state; default off is a no-op — observational purity proven in `_selftest` TEST 1b (identical `(ended, ticks, n_wakes)` with recording on/off). Artifact: `{export_path}/{scenario_name} Recording {start} - {end}.jsonl`.

**Trainer + run auditability (B4) — `rl/training/graph_train.py`.**
The outer PPO loop's *research-validity* contract. It changed NO pipeline layer: PPO
objectives/hyperparameters/checkpoint payload, reward and oracle normalization, the
solver, construction/geometry/exact cardinality, the seed formulas and the fixed
held-out band are all exactly as B1–B3 (§5, §7) left them.

- **Exact-cardinality policy = `skip_and_account_v1`.** Every scheduled train/eval seed
  is attempted **at most once**; a failure is never retried, never replaced by another
  seed, and never shifts a band. Failures never enter a PPO buffer or a reward
  aggregate, and each is recorded exactly once. Attempts, successes, failures and
  denominators stay explicit, so every reward statistic describes the SUCCESSFUL /
  exact-cardinality-feasible subset — and says so (`aggregates_over`).
- **Git provenance is a training PRECONDITION.** `collect_provenance` runs before the
  run creates ANY artifact (not merely before the engine/policy/solver) — `output_dir`
  may sit inside the repo and its own untracked files would otherwise read as dirty
  source state. `_git_provenance` sets `available=True` only when BOTH the full commit
  SHA and the clean/dirty verdict were determined (a SHA alone does not say what ran).
  Incomplete provenance writes the attempted `run_config.json`, then `train` REFUSES
  before policy, generator, episode or optimizer work. A KNOWN-dirty tree warns loudly
  and may run.
- **Run artifacts** (a run directory is the record): `run_config.json` carrying the
  versioned `provenance` block, `train_records.jsonl`, `eval_records.jsonl`, the
  append-only immediately-flushed `episode_failures.jsonl` (phase, eval stage, updates
  completed, iteration, attempt ordinal, episode index / eval tag, exact seed, pipeline
  stage `generation|setup|run|reward`, original exception + traceback), the derived
  `run_summary.json` (`build_run_summary` reads the jsonl — ONE metric path, with
  `accounting_reconciled` cross-checking record counts against the ledger), and ONE
  four-panel `training_plot.png` (`plot_training`, jsonl-only, torch-free child).
- **Evaluation timing.** A deterministic held-out `pre_update` round runs after the
  initial policy is built and BEFORE the first training episode, buffer insert and
  optimizer step, recorded with `updates_completed = 0` and `iteration = null`. Later
  rounds carry their REAL completed-update count, so none can be read as "iteration 0".
- **Classification (`_iteration_outcome`) — three DISJOINT states:** `all_failed`
  (nothing completed; measured nothing), `zero_wake` (episodes completed, no ego woke)
  and `productive`. Both of the first two end at `n_epochs_run == 0`, so the classifier
  judges episode counts, not the updater. An all-failed batch or eval round reports its
  reward as `null`, **never `0.0`** — the reward is oracle-normalized regret, so 0 is
  the OPTIMUM. A successful zero-wake episode is a real successful episode.
- `TrainConfig` gains no scenario semantics here; `evaluate` gained `stage` /
  `updates_completed` / `failures_path`. `updates_completed` counts only updates that
  actually ran epochs, and is the learning-curve x-axis.
- **Per-episode observability and confirmation semantics (PR #7).** Every successful
  train / `pre_update` / `post_update` attempt prints one immediate, labelled `OK` block
  with its phase, indices, exact seed, reward, wakes, ending, ticks, dead count, elapsed
  time and the known/hidden target roster by BLADE name. `GraphPlanExecutor.done` remains
  a set of `(ego_id, target_id)` CONFIRMATIONS and may exceed the world target count;
  trainer target metrics instead count unique `target_id` values directly through
  `_unique_confirmed_target_ids(ctx.executor.done)`. The authoritative aggregates are
  `targets_confirmed_unique_mean` / `eval_targets_confirmed_unique_mean`;
  `kills_mean` / `eval_kills_mean` are compatibility aliases fed from the same corrected
  number. Names are presentation only. A name lookup may degrade to `<unnamed target>`
  without changing an id or count, while malformed/inconsistent roster structure raises
  `EpisodeRosterError`, becomes an accounted `setup` failure, and never contributes a
  false successful zero. Reward and PPO semantics are unchanged; the reward already
  deduplicated by target id.
- **Per-round eval scenario preservation (PR #7).** `eval_episode_tag` gives every eval
  round a deterministic, disjoint file-tag namespace. Tags affect artifact names only:
  every round still evaluates the same fixed held-out seed band. `TrainConfig.validate`
  rejects tag ranges that could collide, so pre- and post-update scenario JSONs coexist.
- **Visual artifacts — the opt-in inspection surface (PR #10).** `TrainConfig.
  visual_artifacts` / `--visual-artifacts`, **OFF at both surfaces by default**. It is
  OBSERVATION, not measurement: nothing captured is ever read back into the pipeline.
  When enabled it selects EVERY scheduled `pre_update` / `train` / `post_update`
  attempt — there is deliberately no per-seed filter, which would be a second
  artifact-selection language beside the seed schedule — and stores one collision-free
  bundle per attempt under `<run_dir>/visual_artifacts/`. Each bundle holds:
  - `known_only_scenario.json` — the generator's known-only world copied **byte for
    byte** (never regenerated, normalized, reserialized or rebuilt from tasks); the
    original under `<run_dir>/scenarios` is untouched;
  - `executed_t0_scenario.json` — the AUTHORITATIVE executed world, serialized from
    `ctx.game.export_scenario()` on the **env-2** game, called EXACTLY ONCE and BEFORE
    `build_fuel_damage_controller`, `run_episode`, the top-of-tick fuel mutation, any
    policy decision and any `env.step`. Env-1, `build_patched_scenario` output, the
    placement audit, the beliefs and the oracle tasks are derived views and none of them
    substitutes for it;
  - the BLADE playback `.jsonl`, produced ONLY through the existing locked contract —
    armed by `setup_episode(recording_export_path=<attempt dir>)`, started / stepped /
    exported by `run_episode`. No recorder internal is called and no scenario name is
    mutated; the per-attempt directory is what keeps recordings apart. All chunks are
    listed if the recorder ever splits one;
  - `artifact_manifest.json` — the attempt's identity stated EXPLICITLY (phase,
    iteration, `updates_completed`, eval round / episode / pair-member ordinals, attempt
    ordinal, training episode index, exact seed, scheduled condition, exact
    `episode_tag`), plus target-count expectations vs observations. `status` is
    `incomplete` until the bundle is whole and `complete` only once the three files
    exist; a bundle may be read as full only when it is `complete`.

  **Failure routing.** An artifact filesystem / serialization failure is INFRASTRUCTURE:
  it raises `_VisualArtifactError`, which the train and eval attempt handlers re-raise
  AHEAD of their broad `except Exception`. It therefore aborts the run loudly, is never
  written as a `generation` / `setup` / `run` / `reward` failure, never enters
  `skip_and_account_v1`, and cannot shrink a scientific denominator by masquerading as
  an episode failure. A NORMAL episode failure is unaffected — it stays in the existing
  stage taxonomy and leaves a clearly marked `incomplete` bundle holding whichever
  pre-failure artifacts were valid; no recording is ever fabricated, because the tick
  loop deliberately exports none when the loop raised. A directory collision raises
  rather than overwriting or merging two attempts.

  **OFF-path invariance.** With `visual_artifacts=False` no `visual_artifacts/` directory
  is created, no identity is constructed, no scenario is copied, `Game.export_scenario`
  is not called, and NEITHER keyword is passed at all — `_recording_kwargs` omits
  `recording_export_path` from the `setup_episode` call and `_artifact_kwargs` omits
  `artifacts` from the `_run_one_episode` call, so both are the pre-feature calls.
  Neither path adds an RNG object or draw (bundle names derive only from already-resolved
  schedule metadata), and seeds, scenario tags, scenario names, policy inference, PPO
  inputs, the solver, the reward, fuel-damage semantics, the failure taxonomy and BLADE
  are all unchanged. The resolved flag is recorded in `run_config.json` through the
  existing `asdict(cfg)` path and echoed in the startup header.

**FD-BASELINE-v1 — the difficulty factor — `rl/training/graph_fuel_damage.py`**
(consumed by `graph_tick_loop.run_episode`, `graph_train` and `graph_rollout`).

THE **ONE** SELECTED DIFFICULTY FACTOR of the current final Phase-A baseline cell. The
scenario is otherwise UNCHANGED: 3 agents, 3 known + 3 route-relative hidden airbase
targets, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams=False`,
`probability = 1`, unchanged BLADE weapon lethality, frozen solver, unchanged PPO. No
second factor is bundled in (§8).

- **Deterministic private RNG domain.** `derive_fuel_damage_seed` is
  `SHA-256("fuel_damage_v1:<episode_seed>")`, so the clean/damaged draw and the ego
  selection depend on the episode seed ALONE — not on `hash()` (per-process salted), not
  on global `random`, and not on the placement rng (whose stream position depends on how
  many placements were rejected). TRAINING uses `fuel_damage_mode = seeded_mixture` at
  `P(damaged) = 0.5`; the mixture bit is drawn first in EVERY mode, including the forced
  ones, so the stream position matches and a forced-damaged episode selects the same ego
  the mixture would have.
- **Matched-pair EVALUATION.** Each held-out seed is attempted TWICE per round, once
  `forced_clean` and once `forced_damaged`, on the SAME `eval_seed` — hence the same
  generated world, the same `A_init` and the same hidden geometry — with DISTINCT
  artifact tags (`eval_member_tag`, slot `e*2 + m`) so both worlds coexist as files.
  `TrainConfig.validate` sizes the tag namespace for the doubling.
- **The event.** A damaged episode selects ONE ego with a non-empty initial route
  (sorted id order, so the draw never depends on dict insertion order) and plans a
  ONE-SHOT event at ~30 % of that ego's FIRST planned leg. Route prediction REUSES the
  frozen `graph_hidden_placement.predict_route`, so the window can never be measured
  against a route the executor does not fly.
- **The strict decision window.** The post-damage target lies in the half-open interval
  `[margin·fuel(direct RTB), margin·fuel(rest of route + return))` at `margin = 1.10`
  (the engine's own reserve): flying straight home stays feasible, completing the
  remaining route and then returning does not. The chosen value is the interval MIDPOINT.
  All fuel arithmetic is BLADE's own, transcribed in `fuel_for_distance_km` from
  `Game.get_fuel_needed_to_return_to_base` (km → nm → hours at the aircraft's KNOTS
  speed → lbs/hr); `speed` / `fuel_rate` are read off the LIVE aircraft, never off
  `Agent` (`scenario_factory` substitutes a 250 kt planning speed for a grounded unit).
- **THE WINDOW IS VALIDATED TWICE — planned, then live.** `plan_fuel_damage` validates it
  before the run at the PROJECTED event point, and `FuelDamageController.maybe_apply`
  RE-MEASURES it through the same `measure_window` site from the aircraft's ACTUAL
  position and validates against its ACTUAL fuel immediately before mutating. The
  projection is optimistic by construction: it charges fuel for distance FLOWN, while
  `Game.update_all_aircraft_position` burns `fuel_rate / 3600` on EVERY tick including
  route-less ones (the launch tick is exactly that).
- **Failure policy.** A failed LIVE strict-window check raises BEFORE the mutation, so a
  refused event leaves the engine untouched, and the attempt is accounted as a `run`-stage
  failure. A planning failure (no eligible ego, no valid window) raises at `setup` and is
  **never silently downgraded to a clean episode** — that would move the population every
  per-condition statistic is reported over. Both land in `skip_and_account_v1`: recorded
  once, no retry, no substitution, no band shift. A `forced_clean` member computes no
  window at all, so the two members of a pair fail independently or not at all.
- **Locality (no-communication).** The real `current_fuel` mutation happens at the TOP of
  a tick, BEFORE Phase 1, so every ego reasons from the same post-event snapshot and ego
  iteration order stays irrelevant. Only the selected ego wakes. `FUEL_DAMAGE` carries no
  peer state, and peer graph rows remain FEATURELESS (`agent_features[peer, 0] = 0.0`),
  so the damaged value is unreachable from any peer's graph. The damaged ego's own graph
  at that same wake necessarily carries the post-damage `fuel_norm`, because
  `_compute_fuel_norm` reads the live object this layer already mutated.
- **RTB is COMMAND HISTORY.** `FuelDamageOutcome.rtb_command_issued` is True only if
  `run_episode` really emitted `aircraft_return_to_base('<selected ego>')` in a Phase-2
  command list, observed by `FuelDamageController.note_commands`. It is NEVER derived
  from `GraphPlanExecutor.rtb_issued`: that is a lifecycle LATCH which `_command_for_ego`
  also sets True for a DEAD ego — precisely because no command was, or could be, emitted —
  so reading it would report an ego that flew its plan into the ground as both an RTB and
  a death. `rtb_command_for` is a documented mirror of the executor's one emission site,
  kept out of its import closure to preserve this layer's purity, with the equivalence
  test-enforced against a real `GraphPlanExecutor`.
- **Reward.** `RewardConfig(aircraft_penalty_coeff=2.25)` is passed EXPLICITLY by both
  harnesses (`TrainConfig.reward_config()` / `RolloutConfig.reward_config()`), because
  `graph_reward`'s own default is `0.0` and losing an airframe would otherwise be free.
  **The `graph_reward` formula itself is UNCHANGED** — only the coefficient it already
  accepted, and the resolved value is recorded in `run_config.json:/difficulty`.
  CONSEQUENCE FOR READING A REWARD: with `c > 0` the penalty term is real, so `R` is no
  longer confined to `~[-1, 0]` — an episode that loses an airframe can score below `-1`.
  The §5 Stage-7 range note describes the `c = 0.0` case.
- **Observability.** Records and the per-episode `OK` block distinguish clean from damaged
  episodes and PLANNED from LIVE bounds (`FuelDamagePlan.rtb_fuel_floor` vs
  `FuelDamageOutcome.live_rtb_fuel_floor` — kept under separate names, printed side by
  side, never merged). They report whether the event fired and when, observed progress,
  fuel before/after and the damage factor, whether `FUEL_DAMAGE` caused a wake and which
  meta-action it produced, the real RTB command, deaths, condition-specific attempt counts
  and reward means, and the matched-pair reward delta over pairs whose BOTH members
  completed. **An empty successful-pair population is `null`, never numerical zero** — 0
  is the oracle optimum and would read as "the event changed nothing".
- **Purity.** The layer imports no BLADE, gymnasium, torch or solver, does no file I/O and
  holds no module-global randomness; live engine objects are touched only through
  duck-typed attributes. That is what makes the whole factor hand-testable and keeps it
  safe inside `graph_tick_loop`'s import-purity closure.

---

## 6. File map — "I want to…"

| … | Go to |
|---|---|
| Change episode setup / solve+normalize / Belief | `rl/training/graph_episode_setup.py`, `rl/training/belief.py` |
| Change the CONSTRUCTION seam (solve → place → patch → reload) | `rl/training/graph_episode_setup.py` → `_setup_episode_construction`, plus its helpers `_resolve_construction_mode`, `_shared_launch_point`, `_require_airbase_only_targets`, `_select_hidden_prototype`, `build_patched_scenario`, `_require_agent_ids_preserved`, `_rematerialize_known_tasks`, `_build_env` / `_extract_world` / `_close_quietly` / `_finish_context` |
| Change the LEGACY split path (retained, not deleted) | `rl/training/graph_episode_setup.py` → `_setup_episode_legacy`, `split_tasks` |
| Change the tick-loop / policy bundle / rollout | `rl/training/graph_tick_loop.py` |
| Run a diagnostic rollout (no training) | `rl/training/graph_rollout.py` (`RolloutConfig`, `run_rollout`) |
| Run PPO training / plot a run | `rl/training/graph_train.py` (`TrainConfig`, `train`, `plot_training`). A run writes `run_config.json` (+ `provenance`), `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl`, `run_summary.json` and one 4-panel `training_plot.png`. **`train` refuses to start unless Git provenance is COMPLETE** (full SHA + clean/dirty verdict) — see the §5 trainer contract; `collect_provenance` / `_git_provenance` / `_iteration_outcome` / `build_run_summary` / `eval_episode_tag` / `_format_episode_block` / `_unique_confirmed_target_ids` / `_episode_target_roster` |
| Change the training scenario cell (target counts) | `rl/training/graph_train.py` (`TrainConfig.num_agents` / `n_known` / `n_hidden` / `min_target_distance_km` / `min_known_separation_km`, `build_variation_config`); mirrored field-for-field on `rl/training/graph_rollout.py` (`RolloutConfig`). The generator writes `n_known`; setup patches in `n_hidden`, so **emitted targets are `n_known + n_hidden`** (`TrainConfig.n_targets_emitted`). Legacy `num_red_airbases` / `partial_ratio` / `derived_split` / `split_preview` survive and are still tested but are NOT consulted by the construction path (B1, `d6758ac`). |
| Place hidden targets along a predicted ego route (PURE geometry — no BLADE / torch / solver / setup import) | `rl/training/graph_hidden_placement.py` (`PlacementParameters`, `HiddenPlacement`, `predict_route`, `place_hidden_targets`, `validate_placement`, `geometric_fingerprint`). CONSUMED by construction-mode `setup_episode` (B3, `dd14ab4`); the import direction is one-way — this layer must never import `graph_episode_setup`. `predict_route` imports `nearest_neighbor_order` from `utils/scheduling_utils.py`, NOT from any executor module. |
| Change the SHARED intra-level nearest-neighbor ordering (route prediction + execution at once) | `utils/scheduling_utils.py` (`nearest_neighbor_order`). ONE implementation with TWO consumers — `blade_graph_executor.GraphPlanExecutor._eligible` and `graph_hidden_placement.predict_route`. Changing it changes BOTH; that shared identity is the route-fidelity invariant (`2a3f89c`). Pinned by `tests/test_graph_executor_nn_ordering.py`. |
| Change the FD-BASELINE-v1 MECHANISM (rng domain, window, event, live re-validation, RTB measurement) | `rl/training/graph_fuel_damage.py` (`FuelDamageMode`, `FuelDamageParameters`, `FuelDamagePlan`, `FuelDamageOutcome`, `FuelDamageController.maybe_apply` / `live_bounds` / `note_commands` / `note_wake`, `measure_window`, `plan_fuel_damage`, `build_fuel_damage_plan` / `build_fuel_damage_controller`, `derive_fuel_damage_seed`, `resolve_condition`, `fuel_for_distance_km`, `rtb_command_for`). PURE — no BLADE / gym / torch / solver import; must never import `graph_episode_setup`. Injected into the tick via `run_episode(..., fuel_damage=...)`. |
| Change the FD training MIXTURE / matched EVALUATION / FD reporting | `rl/training/graph_train.py` (`TrainConfig.fuel_damage_mode` / `fuel_damage_probability` / `fuel_damage_leg_progress` / `fuel_damage_rtb_margin` / `aircraft_penalty_coeff`, `fuel_damage_parameters()`, `reward_config()`, `_run_one_episode(..., fuel_damage_mode=...)`, `evaluate` matched pairs, `eval_member_tag`, `_ConditionTally`, `_fuel_damage_lines`, `build_run_summary`). `RewardConfig(aircraft_penalty_coeff=2.25)` is passed explicitly here; `graph_reward` stays frozen. |
| Keep the DIAGNOSTIC harness at configuration parity with training | `rl/training/graph_rollout.py` (`RolloutConfig` mirrors the FD knobs field-for-field + `fuel_damage_parameters()` / `reward_config()`; `run_rollout` builds the controller and passes the same explicit `RewardConfig`). Rollouts run the seeded MIXTURE only — matched pairs are an evaluation construct and live in `graph_train.evaluate`. |
| Capture per-attempt VISUAL ARTIFACTS (known-only scenario + executed t=0 scenario + BLADE playback + manifest) | `rl/training/graph_train.py` (`TrainConfig.visual_artifacts` and the `--visual-artifacts` flag, `_AttemptIdentity`, `_AttemptArtifacts` with `open` / `capture_known_only_scenario` / `capture_executed_t0_scenario` / `finalize` / `to_manifest`, `_VisualArtifactError`, `_recording_kwargs`, `_artifact_kwargs`; consumed by `_run_one_episode(..., artifacts=...)` and wired from `train` / `evaluate(..., artifacts_root=...)`). OFF by default and OFF is byte-unchanged — see the §5 trainer contract. `graph_tick_loop`, `graph_episode_setup`, `PlaybackRecorder.py` and `Game.py` are NOT touched; recording is armed only through `setup_episode(recording_export_path=...)`. |
| Change the reward | `rl/training/graph_reward.py` (`compute_episode_reward`/`plan_value`/`realized_utility`/`RewardConfig`) |
| Change WHEN the policy wakes | `rl/action/graph_trigger.py` (`decide_triggers`, `TriggerKind`, `never_overdue`) |
| Change the graph representation | `rl/observation/graph_builder.py` (`GraphObservation`, `GraphObservationConfig`, `EdgeType`, `TASK_FEATURE_DIM`) |
| Change the encoder | `rl/agent/graph_encoder.py` (`GraphEncoder`, `pool()` critic hook) |
| Change actions / mask / sampling | `rl/action/graph_action.py` (`MetaAction`, `ActionHead`, `build_action_mask`, `sample_action`) |
| Change how a decision edits the plan | `rl/action/graph_effect.py` (`apply_meta_action`) |
| Change BLADE execution / plan re-sync | `utils/blade_utils/blade_graph_executor.py` (`GraphPlanExecutor`) |
| Expose the ego's own sensing | `blade_graph_executor.py` → `sensed_target_ids` |
| Create Agents / Tasks from a scenario | `scenario_factory.py` → `create_agents_from_scenario` / `generate_all_enemy_tasks` (`probability=1.0`) / `iter_enemy_targets` + `make_attack_task` (utility: Facility 100 / Airbase 80 / Ship 95) |
| Change scenario content / zones / fleet / fuel tiers | `scenario_generator.py` (`VariationConfig` incl. `strict_geometry` (raise instead of silently weakening requested geometry) and `min_target_separation_km` (pairwise known-target floor, default 0.0 = off); `ScenarioGenerator`, `CLASS_RANGE_TIERS`) |
| Change generation-time discovery connectivity (Layer 1, at `DETECTION_KM`) | `scenario_generator.py` → `_ensure_discovery_chain` / `_compute_zone_bounds` / `_connect_zone_targets`; switch the whole pass OFF with `VariationConfig.ensure_discovery_chain=False` |
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
- `b96d29f` — **final doc sweep** (post-cleanup coda). Four flat-era docs deleted: `LOGS_GUIDE.md`, `RUN_SUMMARY.md`, `docs/INTEGRATION_GUIDE.md`, `docs/MATCH_AOU_API.md` — the first three document deleted code; the fourth documents the live solver but through a dead API (`StepType`, removed in `5eeaf3c`) in every example. All four preserved on `flat-final` (`4d44c34`) + the annotated tag `pre-cleanup` (peel it: `pre-cleanup^{commit}` → `561b7cb`). README's Documentation section lost its three dead links (the two deleted `docs/` files + `RL_MODULE_DOCUMENTATION.md`, orphaned back in `ab54ac3`), leaving only the live `BLADE_API_DOCUMENTATION.md`. Untracked mid-cleanup snapshot `src/match_aou.zip` deleted. `training_output*/` added to `.git/info/exclude` — a local-only, never-tracked safety net shared by both worktrees, deliberately broader than `.gitignore`'s `training_output_*/`. Docs-only; 12/12 import-purity green.
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
- `21e4d14` — `graph_train`: the outer PPO Trainer, Phase A (PPO-phase step 4 — the
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
- `95c09dd` — **Phase-A baseline scenario cell + config visibility** (`graph_train`
  only; no locked layer touched). Defaults retargeted from the measured-degenerate
  `(3,3)` to the SELECTED cell: `num_red_airbases=(6,6)`, `partial_ratio=0.5` →
  **known 3 / hidden 3**, `U_oracle=480`. Why this cell: 6 targets > the 4-agent fleet
  (which comes from the base template `strike_training_4v5.json`, NOT a config knob)
  removes the forced 2:1 stacking that pinned every episode at R = −1/3; `known ≥ 3`
  keeps bonmin out of its B&B symmetry stall (~15 min/episode at `known ≤ 2`);
  measured `std(R) = 0.1443`, split outcome clean 11/12, zero dominating-set breaks.
  New in the module: **`derived_split`** — a MIRROR of `split_tasks`'
  `max(1, int(n·partial_ratio))` and the single arithmetic site behind the startup
  echo, `validate()`'s hazard warnings, and `run_config.json`; its equivalence to the
  locked authority is test-enforced over an n × ratio grid that INCLUDES the degenerate
  `n < 2` branch, asserted against real `split_tasks` returns rather than only its
  `meta`. `TrainConfig.split_preview` previews both ends of a range. CLI gains
  `--num-red-airbases` (`N` or `LO,HI`), `--partial-ratio`, `--stretch-target-ratio`,
  every default read OFF `TrainConfig` (drift-guarded by test); `include_sams` and any
  radius are deliberately NOT exposed. `validate()` now WARNS — never raises — on
  `known < 3` and `hidden == 0`, judged at the range's LOW end (both quantities are
  non-decreasing in n, so that is the worst case). Every run writes
  `run_dir/run_config.json` (full resolved config incl. nested `PPOConfig`, the derived
  split, the base-scenario name) and the startup header echoes `known/hidden` — the
  standing defence against the truncation trap: `int()` truncates, so at n=6
  `1.0/3.0` gives known 2 while the decimal `0.333` gives known 1. Never
  auto-corrected: the config you type is the config you get, and the header tells you
  what it is. Proven: suite 46 → 59, import purity 12/12, `--selftest` green
  end-to-end under `nlp_env`. LIVE EVIDENCE the §8 blocker is lifted:
  `adv_std_raw = 0.1197` (was ~0), per-episode R spanning −0.5 … −0.1667 (exact
  multiples of 80/480), three distinct iteration means, and `OPPORTUNISTIC_ENGAGEMENT`
  firing 7× in 12 episodes — matching the instrument's measurement for this cell
  exactly.
- `384845b` — **Scenario-construction preconditions** (step 1 of 3 of the offline
  scenario-construction phase; no locked layer touched). Three fixes the inverted
  build order depends on. (1) **LAUNCH POINT.** The base template parked the four
  BLUE aircraft at `(32.35416…, 34.81240…)` while their own airbase sits at
  `(32.85416…, 35.31240…)` — 72.7 km away — and `Game.launch_aircraft_from_airbase`
  only moves the object between lists without repositioning it, so every episode
  put the fleet airborne 72.7 km from its base. The four aircraft records now carry
  the airbase's coordinates. The template JSON is **MINIFIED** (one line, no
  newlines): edit it by exact string replacement and NEVER round-trip it through
  `json.dump`. Intended consequence: `Agent.location == Agent.return_location`, so
  the solver's `round_trip_cost` is now a symmetric out-and-back instead of a
  launch→target→base triangle, and a given seed MAY now yield a different
  allocation. (2) **The source of that skew** — `_adjust_aircraft_count`'s
  empty-inventory branch placed a new aircraft at `base − 0.5°/0.5°`; it now anchors
  to the base, making the defect unreproducible. (3) **`VariationConfig.
  ensure_discovery_chain: bool = True`** gates Layer 1's CALL SITE (body untouched).
  Default = today's behaviour; `False` skips the relocation pass, which the
  construction path requires: with only the KNOWN targets generated, Layer 1 would
  cluster them into ≤`DETECTION_KM` pairs and collapse the route diversity that
  hidden-target placement is measured against. When `False`, the seven stats keys
  Layer 1 stamps (`easy_relocated`/`_total`/`_isolated`, the three `stretch_*`,
  `min_radar_km`) are ABSENT from `last_generation_stats` — read them with `.get`,
  never `[...]` (`graph_episode_setup._selftest_generator` indexes `min_radar_km`
  directly and is safe only because it runs with the chain ON). Verified: no seeded
  rng consumer runs after Step 5.25, so gating the call does not shift the rng
  stream. Proven: `tests/test_scenario_construction_preconditions.py`, suite 59 →
  64, import purity 12/12, both module selftests + the bonmin selftest green under
  `nlp_env`. The load-bearing test is **P6** — the four RED-airbase coordinates for
  a fixed seed are byte-identical before and after, which makes the phase's
  foundational claim falsifiable: target placement is a function of the BASE
  coordinates and the rng stream ONLY, never of the aircraft's own position. **P5**
  proves the switch is a true skip by monkeypatching `_ensure_discovery_chain` to
  raise (`generate()` has no try/except, so the raise cannot be swallowed).
- `a5a4137` — **workflow + handoff migration** (documents only; no code touched, no test
  delta). §1 moves from STOP-before-commit / local-only to Git transport: an explicit base
  SHA, a reviewable candidate commit, and a mandatory status block. Adds the **grade = trust
  policy** definition that §1 previously referenced without defining. Fills the two SHAs the
  hash convention deferred (`95c09dd`, `384845b`; the entries above carried `PENDING` until
  this commit). The continuing handoff (`graph_rl_project_handoff.md`) lands in the SAME
  commit: it declares each task's grade, so the two documents are only coherent together.
  **Corrected in place (docs-only, no separate history entry):** that entry asserted that
  both orchestrators read this repository through equivalent direct Git connectors, and that
  a task branch + draft PR is the one universal transport. Both claims are false. Access is
  **capability-aware**, and §1 now carries two transport modes over ONE shared `CLAUDE.md`
  and ONE shared handoff — no per-orchestrator forks of either document. `GPT_GITHUB`: the
  GPT orchestrator resolves branches, PRs, files, and exact SHAs through GitHub and reviews
  the exact `base...candidate` comparison. `CLAUDE_MOUNTED_MAIN`: the Claude orchestrator's
  shared view is CC's synchronized mounted checkout of `main`, so the reviewable artifact is
  an exact post-push `main` SHA plus focused hunks and targeted test evidence requested from
  CC — task branches and PRs are not assumed reachable. The packet or the user declares the
  mode; it is never inferred.
- `f319095` — **shared-document workflow correction** (documents only; no code, no test
  delta). §1's `CLAUDE_MOUNTED_MAIN` description and handoff §0 now state that side's real
  capability: a synchronized mounted snapshot of `main` as a **search** interface that cannot
  diff, cannot prove absence, and can lag. Adds the **Grade-A routing default** — Grade A goes
  to `GPT_GITHUB` when available because it is the only mode that gates `main` behind a
  branch; Grade A under `CLAUDE_MOUNTED_MAIN` is a declared exception with mandatory hunks +
  targeted evidence from CC, and no lock or dependent work before exact-SHA approval. The
  grade-definition and output-discipline sites became pointers to that one bullet rather than
  second copies.
- `d6758ac` — **B1: offline scenario-construction configuration — CLOSED / MERGED /
  LOCKED** (known-only cell, step 1 of 3 of the offline scenario-construction phase;
  integrated into `main` by merge commit `bd087c3`, PR #2). States the reference cell
  outright instead of deriving known/hidden from a ratio: `TrainConfig` gains
  `num_agents=3` (`<= n_known`), `n_known=3`, `n_hidden=3` (PLANNED for B2/B3 — B1 emits
  ZERO hidden targets), `min_target_distance_km=200.0`, `min_known_separation_km=100.0`.
  `build_variation_config` is the ONE site turning a `TrainConfig` into the generator's
  `VariationConfig`: exactly `n_known` targets, `ensure_discovery_chain=False` (Layer 1
  disabled ONLY on this construction path — it would cluster known targets and flatten
  route diversity), `strict_geometry=True` (the generator raises rather than silently
  weakening the requested geometry). The review fix re-measures the ring-sampled
  candidate with the real `_haversine_km` — the ring's flat-earth degree-conversion had
  let true sub-floor targets through a 300-seed sweep — so `strict_geometry` now enforces
  a TRUE great-circle `min_target_distance_km` / `min_target_separation_km` floor;
  STRICT-only, so every legacy non-strict caller (incl. `P6`'s pinned fixture) is
  byte-unchanged (P9c, P11). `RolloutConfig` (`graph_rollout.py`) mirrors the same
  reference-cell fields field-for-field and now validates them as `run_rollout`'s FIRST
  statement, before any directory, policy, generator, or BLADE import — closing the
  `TrainConfig`/`RolloutConfig` divergence recorded in §8. Does NOT build hidden-target
  placement (B2/B3 work). Proven: suite 64 → 84, import purity 12/12, module selftests
  and the bonmin selftest green under `nlp_env`.
- `e22aee3` — **B2: route-relative hidden-target placement — CLOSED / MERGED /
  LOCKED** (step 2 of 3 of the offline scenario-construction phase). Reviewed code SHA
  `e22aee359e06591bdb179ef06a566db90f83a558`, integrated into `main` by merge commit
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3). New leaf module
  `rl/training/graph_hidden_placement.py` + `tests/test_graph_hidden_placement.py`; no
  existing file touched, so no locked layer moved. **PURE**: no BLADE, gym/gymnasium,
  torch, solver, `setup_episode`, file I/O, or module-global randomness — `rng` is an
  explicit required `random.Random`, and `detection_km` arrives through
  `PlacementParameters` rather than being imported from `graph_episode_setup` (importing
  it would drag the layer into the setup/solver/executor closure). Contract:
  `place_hidden_targets(solution, belief_tasks, launch_point, parameters, rng) ->
  Tuple[HiddenPlacement, ...]`, egos iterated in SORTED id order so the result never
  depends on the solution dict's insertion order; reproducibility is judged by
  `geometric_fingerprint` (coordinates only — **no UUIDs**, per §8's "Added enemy airbases
  are not seed-stable by id"). **Route prediction reuses the SHARED
  `nearest_neighbor_order`** (imported, never reimplemented — originally from the minimal
  executor, relocated to `utils/scheduling_utils.py` by `2a3f89c` with the body unchanged):
  ascending `level_order`, the
  helper called separately inside each level, its returned end location chained into the
  next, first level seeded from the shared launch point — so prediction cannot drift from
  execution. **Cardinality: exactly ONE placement per non-empty ego route**; a general
  `n_hidden != usable ego routes` distribution is a separate future design task and is NOT
  solved here. **Geometry:** only `G = L - D` of a leg is guaranteed flown (inside `D` of
  the target the ego attacks and issues no new movement); the perpendicular PROJECTION sits
  at `s = f·G` with `f ~ Uniform[0.60, 0.85]`; sensing guard `10 km`; leg-1 max |offset| =
  `D - guard` (40 km at `D = 50`); a later leg budgets residual origin uncertainty
  `(1 - s/L)·D` and caps |offset| at `D - guard - origin_uncertainty`, and its whole
  approved fraction interval must project beyond the uncertain origin vicinity. **Later
  legs require the STRICT nearest-neighbor condition `gap > 2·D`** (equality rejected; one
  remaining candidate passes trivially). Selection: uniform among eligible later legs,
  else fall back to a valid leg 1, else raise. Everything fails LOUDLY —
  `HiddenPlacementError`, no silent clamping and no weakened margin — and every returned
  placement is re-measured by an INDEPENDENT bearing-based cross-track/along-track path
  before it is returned. Two review fixes are part of the locked behaviour: **F1** —
  `_as_assignment` never coerces; fields must be genuine `numbers.Integral` values (a numpy
  integer still works, normalized to `int`), `bool` is rejected despite subclassing `int`,
  and fractional floats, integral-VALUED floats and numeric strings all raise (`int(...)`
  had silently accepted `(0.9, 0, 0)` AS `(0, 0, 0)`, quietly changing the predicted
  route). **F2** — `validate_placement` checks the recorded `tie_margin_required_km` for
  EVERY `leg_index > 1` BEFORE branching, so the `single_candidate` path can no longer skip
  the requirement; missing, non-finite and incorrect values all raise. Verified on the
  integrated merge: 18 focused B2 tests, 12/12 import purity, full suite 100 → **102**,
  `git diff --check` clean, plus all 18 B2 tests and all 12 import-purity entry modules
  green through the `nlp_env` `__main__` runners. No bonmin or live BLADE run is involved.
  Consumed by construction-mode `setup_episode` from B3 (`dd14ab4`) onward.
- `dd14ab4` — **B3: the setup seam — CLOSED / MERGED / LOCKED** (step 3 of 3 of the
  offline scenario-construction phase; this closes the phase). Reviewed code SHA
  `dd14ab418c71e3bd615f1198d0c612502642d29b`, integrated into `main` by merge commit
  `14224531db9deb700f6e397203177eb8c701c6cc` (PR #4); the merged tree is byte-identical
  to the approved one (`git diff --quiet dd14ab4 1422453`). `setup_episode` gains TWO
  EXPLICIT PATHS chosen by the `(n_hidden, placement_rng)` PAIR — both omitted keeps the
  unchanged legacy `split_tasks` path, both supplied runs CONSTRUCTION, exactly one
  raises before any BLADE object exists, and the mode is never inferred from
  `partial_ratio`. Construction is **solve → place → patch → reload**: a known-only env-1
  is solved for `A_init`, the LOCKED B2 layer places one hidden target per non-empty ego
  route, `build_patched_scenario` appends exactly that many enemy airbases to
  `currentScenario.airbases` (deep-copied prototype, fresh uuid4, deterministic name,
  empty inventory, known targets and their positions untouched), env-1 is closed, and
  env-2 is reloaded on the patched JSON. **Env-2 is the sole runtime source of truth**:
  agents and tasks are re-extracted from it, agent IDs must survive the reload as an
  ORDERED list, known belief tasks are re-materialized from env-2 BY TARGET ID in A_init's
  positional order (so `task_idx` stays valid), the oracle is an independent solve over
  ALL env-2 targets, and no env-1 `Agent`/`Task` object reaches the returned context.
  Guards fail LOUDLY and never repair: airbase-only cell (`TrainConfig` /
  `RolloutConfig` `validate()` also reject `include_sams=True`), shared launch point plus
  `Agent.location == Agent.return_location`, exact `len(placements) == n_hidden` (no
  truncation, padding, duplication or redistribution), unsafe/ambiguous prototype, name/id
  collision, agent-id drift, known-target loss, world cardinality. `n_hidden=0` is a legal
  probe that never calls `split_tasks`. `EpisodeContext.placements` is the id-free
  placement audit and construction `split_meta` reports truthful
  `known/hidden/partial/full` plus a coordinates-only `geometric_fingerprint`;
  `graph_train` / `graph_rollout` now pass `n_hidden` + a fresh `random.Random(seed)` and
  report REAL emitted counts (`n_targets_emitted == n_known + n_hidden`). **Private-sensing
  isolation is proven through the integrated setup/tick seam** (not by re-testing
  `Belief.independent`): a setup-constructed hidden world target reported as sensed only by
  ego A enters ONLY ego A's belief and executor slice through the unmodified `run_episode`
  Phase-1 chain, while every peer belief and slice stays byte-unchanged and the target was
  in NO belief at t=0. One review fix is part of the lock: **`_build_env` now owns cleanup
  until it returns successfully** — a failure in `env.reset()` (or the side selection after
  it) sits in a window no caller guard can reach, so it closes the environment exactly once
  and re-raises the original exception unchanged; the regression test drives `_build_env`
  directly and was verified to FAIL against the pre-fix body. Verified: base suite
  **118 passed, 4 skipped** (102 → 121 collected), 12/12 import purity, `git diff --check`
  clean; **19/19** `tests/test_graph_setup_seam.py` checks under `nlp_env`, plus the
  placement, train, import-purity and legacy `graph_episode_setup` runners. Live seed-0
  reference rollout: 3 agents, **3 known + 3 hidden = 6 full targets**,
  `U_oracle = 479.99968` (the frozen solver EPSILON form of a raw 480), **4 organic wakes**,
  `ended=done`, reward `-0.3333`, both bonmin solves successful, no `CRASH`/`Traceback`.
- `1b48145` — **B4: auditable training-run instrumentation — CLOSED / MERGED / LOCKED.**
  Reviewed code SHA `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c`, integrated into `main` by
  merge commit `ba936606deada050ed9298600ee9041fc330af6c` (PR #6); the merged tree is
  byte-identical to the approved one (`git diff --quiet 1b48145 ba93660`). **PREPARATION,
  NOT A MEASURED BASELINE — no training run was performed.** Two files only
  (`rl/training/graph_train.py`, `tests/test_graph_train.py`); the full contract is in §5
  ("Trainer + run auditability"). Grade A under `GPT_GITHUB`: the first candidate
  (`dc2142627dc40886667170fc2121fe50336329cd`) was REQUEST-FIXES, and the fix chain landed
  as a NEW commit on the same branch — the reviewed commit was never amended, rebased or
  force-pushed, and `1b48145` is the approved head. **Review fixes, both now part of the
  lock:** (F1) an all-failed iteration was being counted as ZERO-WAKE because both states
  end at `n_epochs_run == 0` — `_iteration_outcome` now classifies `all_failed` /
  `zero_wake` / `productive` as three disjoint states from episode counts, the summary
  carries all three counters, and the console prints at most one flag instead of both;
  (F2) provenance was collected AFTER the run directory existed, so a run's own untracked
  artifacts under an in-repo `output_dir` could register as dirty source state — collection
  now precedes every artifact, `available=True` requires the SHA **and** the clean/dirty
  verdict (a failed `git status` no longer leaves `available=True` with `dirty=None`; the
  recovered SHA is still reported with an explicit reason), and `train` refuses incomplete
  provenance after writing the attempted `run_config.json`. A third requested fix
  (a duplicate `seed = train_seed(...)`) was NOT APPLIED: it is not present at the reviewed
  SHA — the assignment occurs once, and a whole-file consecutive-duplicate scan found only
  the intentional nested `try:` in `_run_one_episode` (the outer owns the env-closing
  `finally`, the inner attributes the `setup` stage). **Incidental measured fix:**
  `bonmin -v` emits byte `0x81`, and `subprocess` `text=True` decodes on a reader THREAD —
  the `UnicodeDecodeError` killed that thread, printed a traceback on every run and returned
  an EMPTY stdout with rc 0, so the probe would have recorded `ok` with no output; probes
  now capture bytes and decode leniently (`_probe_command`). `_stats` was removed as
  orphaned by this change (`graph_rollout` keeps its own separate copy). Verified:
  `tests/test_graph_train.py` **55 passed** (34 → 55), base suite **139 passed, 4 skipped**
  (118 → 139), 12/12 import purity, `git diff --check` clean, and all **55** green through
  the standalone `__main__` runner under `nlp_env`. **No `graph_train --selftest`, no live
  BONMIN training probe and no real training run were executed** — the tests are
  solver-free, driving `train` through stubbed episode/generator/update seams and an
  injected Git verdict.

- `a3f0838` — **First real post-B3 instrumented probe — CLOSED / REVIEWED MEASUREMENT.**
  The Grade-A measurement is attributable to exact clean code SHA
  `a3f0838616990987bcb8a51665fa75d84edf5952` on the measurement-only branch
  `task/b4-first-instrumented-probe`; no tracked file changed and no PR or candidate
  commit existed. The exact cell was two iterations × four scheduled train episodes,
  seeds `[0,8)`, plus the same fixed held-out seeds `[1000000,1000004)` before the first
  update and after two completed updates. Provenance was complete
  (`git.available=true`, exact SHA, `dirty=false`, Windows / `nlp_env`, vendored BLADE,
  BONMIN available), the process completed normally in 79.21 s, all six expected
  artifacts existed, and `run_summary.json:accounting_reconciled=true`. Measured:
  `pre_update=-0.4999997395829586` over **4/4**; training **7/8** successful with every
  successful episode producing wakes, **24 transitions**, two productive iterations and
  two PPO updates; one attempt failed exactly once — train seed 2 at `setup`, because B2
  produced two placements for three requested hidden targets after the static solve left
  one ego without a non-empty route; final `post_update=5.000007394910353e-7` over
  **4/4** (numerical zero). This proves learning headroom, usable data yield and a working
  update/eval loop. It is a SHORT PROBE, not a baseline. The original
  `kills_mean` / `eval_kills_mean` fields counted `(ego_id,target_id)` confirmations and
  are invalid as unique-target counts; reward, wake, failure, transition and PPO evidence
  remain valid because reward already deduplicated by target id and the count was not a
  PPO input. Evidence SHA-256:
  `run_config.json=36ec89cdb93f89c0b6e40163491159bf2045235b86b2fad47fe03f2f86141237`,
  `train_records.jsonl=af4ec1851425fbcd0330651c05e384d0e44dad67f8aa1f56080543d8247ad82d`,
  `eval_records.jsonl=2c972efaf85d465ab4f2ffce164ba19ac2a6c189db1e2faf83de6b0d201a7439`,
  `episode_failures.jsonl=32d51d2d2ec017491f2fbe6bf133e103361752ced66ba39aac51e9b35b03a08e`,
  `run_summary.json=d2e24714eecdf48bd5f1478ba1c119f405bef5d82067840776daa26dd4270c80`,
  `training_plot.png=c6dec3ac99c5bd35fe627f77b2e97f432cb33235ce07f7efed8f0c05d7a9521b`.
- `211e12e` — **B4 follow-up: per-episode observability, unique-target semantics and
  per-round eval artifacts — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `211e12e49b676637362d42effdb80988dd0e55eb`, integrated by merge commit
  `ffb95a6ee90df45b2d89802b321dcadcbc272821` (PR #7). Exactly two files changed:
  `src/match_aou/rl/training/graph_train.py` and `tests/test_graph_train.py`; policy,
  reward, PPO, executor, tick-loop, scenario content and seed semantics are unchanged.
  Every successful attempt now prints one immediate `OK` block; unique-target aggregates
  are derived directly from unique target ids and never from display names; structural
  roster defects become accounted `setup` failures instead of false successful zeros;
  and each eval round keeps a disjoint scenario-tag namespace while reusing the same
  held-out seeds. Grade A under `GPT_GITHUB`: candidate
  `24241690572a7a5264e24348db5e9412b41bc47a` received REQUEST-FIXES because a degraded
  roster could silently report a false `0/0` and its docstring claimed the helper never
  raised. The correction landed as a NEW commit, never amend/rebase/force-push, and
  `211e12e` was reviewed and approved. Verified at the approved head:
  `tests/test_graph_train.py` **73 passed**, import purity **12 passed**, full suite
  **157 passed, 4 skipped**, standalone `nlp_env` runner all **73 passed**, and
  `git diff --check` clean. One authorized smoke produced three `OK` blocks, the seed-0
  reference `R=-0.3333` / 4 wakes / `targets_confirmed_unique=4/6`, and coexisting
  pre/post eval scenario files; that smoke validates implementation only and is not a
  scientific result.
- `a8669f4` — **FD-BASELINE-v1: the deterministic, ego-local fuel-damage difficulty —
  CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `a8669f450708c2508753c49ab16fd1028b29607d`, integrated by merge commit
  `1cecb0ac99f839d47ffeea12c8871aec77e66640` (PR #8); the merged tree is byte-identical
  to the approved one (`git diff --quiet a8669f4 1cecb0a`). Grade A under `GPT_GITHUB`.
  The full technical contract is in §5 ("FD-BASELINE-v1 — the difficulty factor") and the
  tick placement in §4; this entry records the LOCK, not the mechanism.
  **Reviewed scope: SEVEN cumulative files** — `rl/training/graph_fuel_damage.py` and
  `tests/test_graph_fuel_damage.py` (both new), plus `rl/action/graph_trigger.py`,
  `rl/training/graph_tick_loop.py`, `rl/training/graph_train.py`,
  `rl/training/graph_rollout.py` and `tests/test_graph_train.py`. No BLADE, solver,
  `graph_reward` formula, PPO, encoder, action-space, feature-width, detection-radius,
  hidden-placement, cardinality or executor-fuel-policy change.
  **Fix chain.** The FIRST candidate `1cf53fcee3ee05b3466c8391cbc6bb04420a0985` received
  REQUEST-FIXES on two measurement-honesty defects; the correction landed as a NEW CHILD
  COMMIT on the same branch and PR — never amend, rebase, force-push or history rewrite —
  and that fix commit touched FIVE of the seven files (`graph_fuel_damage.py`,
  `graph_tick_loop.py`, `graph_train.py` and the two test files). The two defects and
  their closure:
  (F1) the per-episode RTB output was derived from `GraphPlanExecutor.rtb_issued`, a
  LIFECYCLE LATCH that `_command_for_ego` also sets True for a DEAD ego — precisely
  because no command was emitted — so an ego that flew its plan into the ground counted
  as an RTB *and* a death. It is now taken from ACTUAL COMMAND HISTORY: `run_episode`
  hands each Phase-2 command list to `FuelDamageController.note_commands`, which latches
  only on a real `aircraft_return_to_base('<ego>')`.
  (F2) the preflight projection charges fuel for distance FLOWN while the engine burns
  `fuel_rate / 3600` every tick including route-less ones, so live fuel at the event is
  always below `projected_fuel_at_event` and the only guard was
  `fuel_before > post_damage_fuel`. The strict window is now RE-MEASURED from the live
  position through the same `measure_window` site and re-validated against live fuel
  BEFORE the mutation; a failure raises before anything is touched and is accounted as a
  `run`-stage failure, with planned and live bounds recorded under separate names.
  **Verified at the approved head:** full suite **192 passed, 4 skipped**,
  `tests/test_graph_fuel_damage.py` **35 passed**, `tests/test_graph_train.py`
  **73 passed**, import purity **12/12**, the `graph_trigger` selftest green, and
  `git diff --check` clean.
  **NO live BLADE/BONMIN probe, training run, rollout or scientific baseline was
  performed** — every test is solver-free and drives the pipeline through stubbed engine
  seams. Nothing in this lock is evidence about the cell's behaviour; §8 owns the gate.
- `24d1835` — **FINAL-CELL-VISUAL-ARTIFACTS: opt-in per-attempt inspection bundles —
  CLOSED / MERGED / REVIEWED.** Reviewed code SHA
  `24d1835f31d2e6aac04b418308a8753c392ac951`, integrated by merge commit
  `771f2107211fb3f984b64482b799613260e19aca` (PR #10); the merged tree is byte-identical
  to the approved one (`git diff --quiet 24d1835 771f210`). Grade A under `GPT_GITHUB`,
  implementation mode SURGICAL. The full technical contract is in §5 ("Visual artifacts —
  the opt-in inspection surface") and the routing in §6; this entry records the LOCK, not
  the mechanism.
  **Reviewed scope: EXACTLY TWO files** — `src/match_aou/rl/training/graph_train.py` and
  `tests/test_graph_train.py`. No generator, setup, tick-loop, fuel-damage, PPO, reward,
  solver, executor or vendored-BLADE file was touched, and no scenario semantics, seed
  formula, scenario-tag formula, scenario name, RNG draw, policy inference, PPO input,
  failure taxonomy or checkpoint/plot behaviour changed. `visual_artifacts` defaults to
  `False`, so a run that does not opt in is byte-unchanged.
  **Verified at the approved head:** `tests/test_graph_train.py` **89 passed** (73 → 89)
  under the base-env `pytest` AND all **89 passed** through the standalone `__main__`
  runner under `nlp_env`; `tests/test_graph_setup_seam.py` + `tests/test_graph_fuel_damage.py`
  **66 passed, 4 skipped**; import purity **12 passed**; full suite **208 passed,
  4 skipped** (192 → 208); `git diff --check` clean. Three mutation checks confirmed the
  load-bearing tests falsify (OFF path passing `recording_export_path=None`; the executed
  t=0 export moved after `build_fuel_damage_controller`; the `_VisualArtifactError`
  re-raise disabled) — each was caught and then reverted.
  **NEW FACT that shaped the implementation:** `tests/test_graph_fuel_damage.py` carries
  its OWN `_run_stub_training` with an independent `fake_run_one_episode` stub, so a new
  keyword passed UNCONDITIONALLY to `_run_one_episode` would break that unauthorized file.
  `_artifact_kwargs` therefore omits the keyword entirely on the OFF path — which is also
  the stronger invariance claim.
  **NO live BLADE/BONMIN probe, training run, rollout, artifact-generating smoke or
  scientific baseline was performed** — every test is solver-free and drives the pipeline
  through stubbed engine seams. Nothing in this lock is evidence about the cell's
  behaviour; §8 still owns the gate.

- `2a3f89c` — **Repository code hygiene: the retired minimal executor removed —
  CLOSED / MERGED / APPROVED.** Reviewed code SHA
  `2a3f89cf2d027581308493a98767ae658107d6d1`, integrated by
  `6e2757dd30100f429d492f4d23fd8b5f57cf4fac` (PR #11). Grade A under `GPT_GITHUB`,
  implementation mode SURGICAL — Grade A because it touched the locked
  `GraphPlanExecutor` and B2 route-prediction imports, even though the intended runtime
  behaviour was unchanged.
  `blade_executor_minimal.py` had stopped being an executor long ago and survived only
  because two live consumers imported its pure helper. **`nearest_neighbor_order` moved to
  `src/match_aou/utils/scheduling_utils.py`** — the environment-agnostic scheduling layer —
  with the body LINE-FOR-LINE unchanged, the only difference being the local type alias
  `Assignment` → that module's pre-existing, identical `Assignment3`. Both consumers
  (`GraphPlanExecutor._eligible`, `graph_hidden_placement.predict_route`) now import that
  ONE implementation, which IS the route-fidelity invariant. The retired file was deleted
  with **no shim and no re-export** (it remains on `flat-final`); stale prose was corrected
  in `blade_graph_executor.py`, `graph_hidden_placement.py` and `models/step.py` (the last
  still named the retired class as "the sole translation layer").
  Tests: `tests/test_executor_nn_ordering.py` → `tests/test_graph_executor_nn_ordering.py`,
  rewritten against current code (3 → 11 tests: the pure helper's order, exact tie-break,
  unlocated-last, no-anchor and end-location chaining, plus `GraphPlanExecutor` legacy-vs-NN
  ordering, live-position seeding, current-minimum-level gating and per-ego isolation).
  `tests/test_graph_hidden_placement.py` PO2 no longer uses the retired executor as its
  oracle: it consumes the CURRENT executor's own `_eligible` level by level from a live
  position, so the check stays independent of `predict_route` instead of becoming a
  tautological re-call of the shared helper.
  **Reviewed proof.** (i) HELPER SEMANTIC IDENTITY — empty normalized line diff against the
  base body. (ii) EXECUTOR / B2 FIDELITY — measured against the base SHA on both source
  trees: placement geometry byte-identical over 60 seeds × every `HiddenPlacement` field,
  the `geometric_fingerprint`, six predicted-route orderings and the post-placement RNG
  STREAM POSITION (so no draw was added, removed or reordered); executor eligibility
  byte-identical over 120 randomized worlds × both ordering modes × airborne / grounded /
  post-kill, plus emitted commands. Four mutation checks confirmed the load-bearing tests
  falsify — breaking `predict_route`'s chaining ALONE fails PO2, and so does breaking
  `_eligible`'s live-position seeding ALONE. (iii) NO DEPENDENCY LEAK — no Python or test
  import of `blade_executor_minimal` remains, `nearest_neighbor_order` has exactly one
  definition, and `graph_hidden_placement` purity IMPROVED (it no longer pulls in the
  `blade_utils` package at all).
  Verified: `tests/test_graph_executor_nn_ordering.py` **11 passed**,
  `tests/test_graph_hidden_placement.py` **18 passed**, import purity **12 passed**, full
  suite **216 passed, 4 skipped** (208 → 216), both `nlp_env` `__main__` runners green, and
  `git diff --check` clean. **NO training run, rollout, BONMIN solve, BLADE smoke or
  scientific probe was performed** — nothing here is evidence about the cell's behaviour;
  §8 still owns the gate.

---

## 8. OPEN (not built)

- **THE NEXT GATE — a fresh SHORT INSTRUMENTED PROBE on the FINAL fuel-damage cell.**
  Difficulty selection is CLOSED (next item) and FD-BASELINE-v1 is merged and locked
  (`a8669f4`, §7), so the open question is no longer *what* to build but *how the built
  cell behaves* — and NOTHING has measured that. **No live BLADE/BONMIN episode, training
  run, rollout or probe has been executed against the fuel-damage cell**; the lock rests
  entirely on solver-free tests through stubbed engine seams. The next task is therefore
  a bounded, separately authorized short probe of the merged cell, which must report:
  complete provenance; explicit denominators everywhere; the scheduled clean vs damaged
  populations; matched-pair yield and the paired reward delta with its pair denominator;
  failures by pipeline stage; how often the event actually fired, woke the selected ego,
  produced a real RTB command, or ended in a death; reward headroom; and whether the PPO
  updates were productive. **A long baseline stays BLOCKED until that probe passes**, and
  no result may be pre-claimed for it. A held-out mean is never read without its
  denominator; `graph_reward` remains FROZEN unless a separately reviewed p<1 design
  requires an explicit reward-contract change.
  That probe MAY run with `--visual-artifacts` (§5, `24d1835`), which preserves each
  successful attempt's known-only scenario, executed t=0 scenario and BLADE playback for
  inspection. It is an observation surface only: enabling it neither authorizes the probe
  nor changes anything the probe measures, and artifact completeness is reported ALONGSIDE
  the scientific denominators, never in place of one.
  *Historical, and about the EASY PRE-FD CELL only:* the clean-code probe at
  `a3f0838616990987bcb8a51665fa75d84edf5952` measured pre-update headroom
  (`-0.4999997395829586`, 4/4), train yield 7/8 with one accounted seed-2 `setup` failure,
  24 transitions, two productive PPO updates and a final held-out numerical zero
  (`5.000007394910353e-7`, 4/4). That cell had no difficulty factor; **those numbers are
  not evidence about the fuel-damage cell** and must not be reused as its baseline.
- **Complete Git provenance is REQUIRED for a real training run (`1b48145`).** `train`
  raises before policy, generator, episode or optimizer work unless BOTH the full commit SHA
  and the clean/dirty verdict were determined, so a run cannot be launched from a checkout
  where `git` is unavailable, times out, or cannot read the index. A dirty tree is a
  hazard, not a blocker: it WARNS and runs. Consequence for tooling: anything driving
  `train` outside a working checkout must inject the verdict (the tests patch
  `_git_provenance`) rather than expect it to be optional.
- **Centralized critic / value head (CTDE):** size-agnostic value estimator off `GraphEncoder.pool()`; needs a dedicated CTDE design (training on all-agent info while keeping execution no-comms). **A new planning chat.**
- **Baseline difficulty selection — CLOSED for the current cell by FD-BASELINE-v1
  (`a8669f4`).** Exactly ONE factor was selected, implemented and locked: `fuel_damage`
  (§5, §7). The following were considered and **NOT selected**; each remains a DEFERRED,
  SEPARATE research change and none may be enabled implicitly or bundled into a probe:
  - **`probability < 1`** — still out. It reopens the reward operand scale, because
    expected oracle utility and realized achieved utility diverge below p=1; the cell
    stays at `probability = 1` and `graph_reward` stays frozen.
  - **Enemy targets that shoot back / SAMs** — still out. `include_sams=False`, and the
    construction path refuses a world whose enemy units are not all airbases; BLADE
    weapon lethality is unchanged.
  - **Dense / per-wake reward** — still out. It was never a consequence of selecting a
    difficulty factor, and is not one now.
  Reopening any of them is a new research-design decision with its own semantics,
  observability, proof obligations and bounded implementation/lock task — and, per the
  gate above, it comes AFTER the final-cell probe, not instead of it.
- **Solver 2:1 stacking (scenario-design fix, NOT solver constraints):** the anti-div-by-zero `EPSILON` nudges utility enough to assign 2 agents even at `probability=1.0`; a redundant agent chasing an already-killed target never proximity-confirms, so episodes end via `truncated`. The learned policy should recover this via `SELF_PRESERVATION_ABORT`→RTB once trained; the root fix is `EPSILON`/scenario-side.
- **Peer-dropout as a deterministic pre-build trigger** (advisor-pending, separate chat): move "peer overdue ⇒ drop its ASSIGNMENT edge" out of the policy; needs a deadline param + a `was_assigned_to_peer` feature to keep recovered-vs-popup semantics.
- **`reachable_by_ego` marginal-detour model:** `graph_builder._reachable_by_ego` is a conservative round-trip placeholder; intended model is marginal detour-cost vs remaining fuel slack (isolated to the builder; the mask reads the column).
- **`assigned_to_peer` as a task-feature column** (currently edge-derived), **real ETA** (enables PEER-OVERDUE; currently `never_overdue`), **`kill_confirm_ticks` calibration** if p<1 lands.
- **`setup_episode` does not guard `split_meta["outcome"]` — LEGACY-PATH-ONLY since B3
  (`dd14ab4`).** `split_tasks` can return `warn-fallback` or `exhaust` — meaning a hidden
  target has NO known neighbour within `DETECTION_KM` and is therefore undiscoverable at
  runtime — and the LEGACY path proceeds SILENTLY. Measured: breaks appear only where KNOWN
  is small relative to hidden (known 1 → 12/12 broke; known 2 with 6 hidden → 7/12; known 2
  with 4 hidden → clean), so the driver is the CONTROL RATIO, not target density. **The
  construction path is immune by construction**: it never calls `split_tasks`, and
  discoverability comes from the locked B2 geometry (the hidden target is placed on a leg
  the ego is guaranteed to fly within `DETECTION_KM` of) rather than from an adjacency
  chain. Since training and rollout both use the construction path, this is now a hazard of
  the retained legacy surface only. Options if the legacy path is ever driven again:
  reject-and-reseed the episode, raise, or tie config to a control-ratio floor plus a
  guard. Touches a §5 locked file → full recon→prompt→review→lock cycle.
- **Exact-cardinality construction failures — RESOLVED as `skip_and_account_v1` by B4
  (`1b48145`).** B2's locked contract is ONE placement per non-empty ego route, and B3
  requires `len(placements) == n_hidden` exactly, so when bonmin leaves an ego unassigned
  there are fewer routes than `n_hidden` and `setup_episode` raises. Measured on the default
  cell over seeds 0–11: **10/12 gave 3 usable ego routes; seeds 2 and 8 gave only 2**; seed
  0 (the reference) is clean. **The decision is to ACCEPT the loss and account for it**: the
  seed is attempted once, its failure is recorded once in `episode_failures.jsonl` with the
  pipeline stage, and the batch simply carries a smaller successful population that every
  statistic reports next to its denominator (§5). The rejected alternatives stay rejected —
  no reseeding past a failure, no retry, no band shift, and above all no weakening of the
  cardinality check, the B2 geometry, or the loud failure. The general
  `n_hidden != usable ego routes` distribution policy B2 named remains a SEPARATE, still-open
  design task; `skip_and_account_v1` is how the current cell behaves until it exists. The
  first real probe measured the actual scheduled yield as **7/8** train attempts:
  seed 2 failed once at `setup` because only two non-empty ego routes existed for three
  requested hidden targets; it was recorded once and never retried or replaced. The
  earlier 10/12 construction sample remains context, not the run-time rate of a future
  baseline configuration.
- **Added enemy airbases are not seed-stable by id.** `ScenarioGenerator` mints a FRESH
  uuid for every red airbase it ADDS on each `generate()`, even at a fixed seed
  (geometry and utility identical, id different). The base template holds 3 red
  airbases, so at the locked `(6,6)` HALF the targets are minted per run. Consequence:
  `graph_rollout`'s `known_target_ids` and any id-keyed cross-run comparison are
  unreliable for added targets — compare by geometry fingerprint `(lat, lon, utility)`.
  Scenario `/currentScenario/id` and `/name` are likewise unseeded; template unit ids
  ARE stable.
- **bonmin needs a solve timeout at `known ≤ 2`.** With ≤2 known tasks against the
  4-agent fleet, branch-and-bound hits a symmetry stall — one measured episode took
  ~15 min against ~45 s typical. The locked cell is clear of it and
  `TrainConfig.validate` now WARNS, but a timeout is required before any low-known or
  n-randomized config enters training.
- **Single-radius invariant (§3) — CLOSED.** Sensing-radius expansion was cancelled.
  Keep the unified `DETECTION_KM = 50` contract for sensing, arrival, attack,
  kill-confirmation, generator connectivity, and split adjacency. Do not reopen this as
  part of scenario construction.
- **`RolloutConfig`/`TrainConfig` construction-default divergence — CLOSED by B1
  (`d6758ac`).** `RolloutConfig` now mirrors `TrainConfig`'s reference-cell fields
  field-for-field (`num_agents`, `n_known`, `n_hidden`, `min_target_distance_km`,
  `min_known_separation_km`, `include_sams`, `randomize_red_airbase_positions`,
  `stretch_target_ratio`) and validates them the same way, as `run_rollout`'s FIRST
  statement. Diagnostic rollouts and training runs now build the same default world.
- **`min_target_distance_km` — RESOLVED for the construction path by B1 (`d6758ac`).**
  The pre-B1 50 km floor (== `DETECTION_KM`, measured from the launch point) put the P6
  fixture's easy targets only **58.8 km** / **63.2 km** out — discoverable seconds after
  wheels-up — while Layer 1 pulled the same fixture's known pairs to **13.7 km** /
  **28.9 km** apart, both destroying the mid-route pop-up semantics this phase depends
  on. The strict B1 construction path (`build_variation_config`,
  `VariationConfig.strict_geometry=True`) now enforces a TRUE great-circle
  `min_target_distance_km=200` km floor and a `min_known_separation_km=100` km
  known-target separation, and disables Layer 1 entirely on that path
  (`ensure_discovery_chain=False`). Legacy non-strict generator callers are unaffected —
  `strict_geometry` defaults to `False` and `min_target_separation_km` defaults to `0.0`
  (off), so every pre-B1 caller's placement and rng stream stay byte-identical (P9c, P11;
  `P6` unchanged).
- **Post-B3 headroom exists — CLOSED first by the `dd14ab4` reference and then measured
  by the `a3f0838` probe.** The default cell emits `n_known + n_hidden` = 3 + 3 = 6
  targets. The B3 seed-0 reference completed `ended=done` with 4 organic wakes,
  `u_achieved=320`, `U_oracle=479.99968` and reward `-0.3333`. The later fixed-band
  probe measured the untrained deterministic policy at
  `pre_update=-0.4999997395829586` over 4/4 and the same band after two PPO updates at
  `post_update=5.000007394910353e-7` over 4/4. Thus the cell had real headroom and the
  loop could close it in a short run. Neither result is a baseline: one is a single
  rollout and the other is a 2×4 diagnostic probe on the easy reference cell.
  **Both predate FD-BASELINE-v1** (`a8669f4`): the target counts are unchanged, but that
  cell carried NO difficulty factor and no death penalty, so neither number describes the
  fuel-damage cell's headroom. Only the pending final-cell probe (the gate above) can.
  *Historical, pre-B3 only:* the cell emitted no hidden targets and measured 0 wakes at
  `reward=+0.0000` because nothing existed to discover; that result is invalid as
  learning evidence. The authoritative target-count fields for future runs are the
  PR-#7 unique-target aggregates, not the probe's old confirmation-count aliases.
- **Raw utility 480 vs reward-side `U_oracle = 479.99968` — keep the distinction.** The
  six airbase targets sum to exactly `6 × 80 = 480` raw utility, but `graph_reward.plan_value`
  is bit-faithful to `MatchAou._add_objective` and carries the frozen anti-div-by-zero
  `EPSILON = 1e-6`: a 1-agent task contributes `80·(1 − 1e-6)` and a 2-agent task
  `80·(1 − 1e-12)`, giving `479.99968` for the measured seed-0 allocation. Both numbers are
  correct for their own operand; do not "fix" either, and do not compare one to the other.
- **`match_aou.*` inherits `pyomo` from the ROOT package (verified, not a B2 regression).**
  `src/match_aou/__init__.py` contains `from .solvers import MatchAou`, so importing ANY
  `match_aou.*` module eagerly pulls in the solver and therefore `pyomo` — including all
  twelve `tests/test_import_purity.py` `ENTRY_MODULES`. `test_import_purity.py` only denies
  flat-only modules (`DENY_MODULES`), so it has never surfaced this. B2 did NOT introduce
  the dependency: its own purity check
  (`tests/test_graph_hidden_placement.py::test_module_has_no_blade_torch_or_solver_dependency`)
  bans `blade` / `gymnasium` / `gym` / `torch` outright — all absent — and treats pyomo as
  inherited root-package behaviour, proving that exemption with a control that imports
  plain `match_aou` and asserts pyomo is already present, so the test fails if the root
  package is ever made lazy. Recorded as a precise fact, NOT as authorization to refactor
  the root package.
> **Repository hygiene / documentation alignment: CLOSED** (Grade C, follows `2a3f89c`;
> its own SHA is recorded by the next commit that touches this file, per the §7 hash
> convention). Verified before each removal with exhaustive `git grep`.
> **`README.md` fully replaced** from current repository truth: every path in its layout
> tree exists, every command was run locally before being documented, and the stale-term
> scan (`MAPPO`, `CTDE`, `ActorCriticNetwork`, `30 features`, `centralized critic`,
> `strike_training_2v3`, `plan_editor.py`, `decision-interval`) leaves only the
> §3 invariant stating there is NO centralized critic and one clearly historical sentence.
> **`docs/BLADE_API_DOCUMENTATION.md` audited against the vendored fork** and corrected:
> `blade/__init__.py` exports ONLY the gym registration, so `from blade import Game` binds
> the MODULE and fails later as `'module' object is not callable` — the module-path import
> is now documented; `Scenario.load_from_file` / `Scenario.from_json` DO NOT EXIST and were
> removed in favour of the real `Game(current_scenario=Scenario())` + `game.load_scenario(
> json_string)` path; the gym class is `BLADE` taking `game=`, not `BladeEnv` taking
> `scenario_file=`; `add_strike_mission` is really `create_strike_mission`;
> `Scenario.is_hostile`'s second parameter is NAMED `target_id` but is not a unit id — it
> delegates to `Relationships.is_hostile`, which tests membership in
> `hostiles[side_id]`, and in this fork's scenarios that maps side id → hostile SIDE ids
> (every engine call site passes another unit's `.side_id`); passing a unit id returns
> `False` SILENTLY, so resolve `target.side_id` first; `get_next_coordinates` takes
> origin/destination/speed, not bearing/distance; detection and weapon engagement ranges are
> NAUTICAL MILES while `get_distance_between_two_points` returns KILOMETRES; `DoctrineType`
> has no `ATTACK_HOSTILE`; and the dead `blade_executor_minimal` / `execute_plan`
> integration section was replaced by the real `GraphPlanExecutor` one. The two additive
> fork APIs and the "set `current_scenario.name` BEFORE `start_recording()`" rule are
> documented as contracts. Every claim was machine-checked against the fork.
> **Scenario set reduced to the one active template** (user decision): `close_scenario.json`,
> `far_scenario.json`, `match-aou_demo_2agents.json` and `strike_training_2v3.json` deleted;
> only `data/scenarios/strike_training_4v5.json` remains tracked, and it was NOT modified.
> No code or test referenced any of the four — the sole references were in the old README.
> **Dead utility symbols removed:** `rl/shared_utils.py` loses `nm_to_km` and
> `normalize_value` (zero references across all tracked files; only `haversine_distance` and
> `clip_to_01` are live, both consumed by `graph_builder`), and its module docstring now
> describes only what it provides. `requirements.txt` lost its stale `MAPPO` comment;
> dependency membership and versions are unchanged. No runtime behaviour changed.

> **Flat-path cleanup phase: CLOSED.** All four steps are locked (§7: `814734e`, `d9b8c17`, `ab54ac3`, `7f324fd`), plus a final doc sweep as a coda. The 38 deleted paths are preserved on TWO DISTINCT refs — branch `flat-final` (`4d44c34`) and the annotated tag `pre-cleanup` (commit `561b7cb`; `git rev-parse pre-cleanup` returns the TAG OBJECT `cce4e1e`, so peel it with `pre-cleanup^{commit}`). Nothing in `src/` or `tools/` references the flat path. `LOGS_GUIDE.md`, `RUN_SUMMARY.md`, `docs/MATCH_AOU_API.md`, and `docs/INTEGRATION_GUIDE.md` were **deleted in the final sweep** (superseding the earlier decision to keep the first two as run-log records — the run logs live on the preserved refs, and `train_full` prose in `main` was more confusing than useful). **Both remaining documentation debts are now CLOSED** by the repository-hygiene task that follows `2a3f89c` (§7): `README.md` was REPLACED outright — written from current repository truth, with the MAPPO/CTDE/flat-observation prose gone except one explicitly historical sentence — and `docs/BLADE_API_DOCUMENTATION.md` was AUDITED against the vendored fork and rewritten where it was wrong.
