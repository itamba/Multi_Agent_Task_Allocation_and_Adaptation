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
Every graph layer in §5 — the nine pipeline stages, the trainer contract, the
difficulty-factor layer in BOTH its designs (the LEGACY FD-BASELINE-v1 and its
FD-VARIABLE-SEVERITY-v1 mild/severe extension), and the **Phase-B CTDE training layer** —
is BUILT, REVIEWED, and LOCKED (see §7 commits). Their **interfaces are contracts** —
change them only through the same recon→prompt→review→lock discipline, and never in a way
that weakens the no-communication guarantee (§3).

**The Phase-B CTDE training layer is BUILT / REVIEWED / MERGED** (approved candidate
`a6f3aa9`, integrated `8390d85`, PR #30 — §7), and this documentation task is what makes
it a LOCKED contract like the layers beside it. Two things must be read together and never
separated:

- **`actor_only` REMAINS THE DEFAULT AND THE PRESERVED REFERENCE PATH.** A run that does
  not select `ctde` constructs no critic, no central observation, no value loss and no
  CTDE advantage — the Phase-A path is not emulated, it is simply the one that runs.
  Preserving it is load-bearing: the approved Phase-A baseline (`737b4bf`, §7) was
  measured on it.
- **NOTHING SCIENTIFIC IS CLAIMED FOR CTDE.** No actor-only vs CTDE comparison has been
  executed. Engineering tests, a passing suite and a merged implementation measure
  nothing, and **no CTDE benefit over actor-only is established** (§8 owns the gate).

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
         # SKIPPED for a dead ego, and for one whose rtb_issued latch is already set
         # (committed to return: no sensing, trigger, wake, belief edit or transition).
         sensed_target_ids → decide_triggers(..., fuel_damage=<ego is the selected one>)
           → (on wake) _wake_decision:
             build_graph_observation → GraphEncoder → ActionHead
             → build_action_mask → sample_action → apply_meta_action → executor.resync
       Phase 2: commands = executor.next_actions(obs)
                fuel_damage.note_commands(commands)   # READ-ONLY measurement
                obs, _reward, terminated, truncated, _info = env.step(commands)
                executor.is_done(obs)                 # PHYSICAL completion, POST-step
     until is_done / terminated / truncated → EpisodeResult(trajectory)
  → compute_episode_reward(ctx, result, cfg.reward_config()): fills Transition.reward
  → PPO buffer + evaluate_action + outer training loop  # BUILT (graph_ppo, graph_train)
  → [BUILT, OPT-IN] Phase-B CTDE: a TRAINING-ONLY centralized critic
       # `TrainConfig.training_mode = 'ctde'` adds, per actor decision, a
       # CentralStateRecorder capture immediately BEFORE `_wake_decision`, then
       # CentralCritic + GAE + CTDEUpdater. `actor_only` (the DEFAULT) builds
       # none of it and the loop above is byte-unchanged. EXECUTION is
       # decentralized in BOTH modes: the actor still reads only its own
       # private GraphObservation. See §5, and §8 for the un-run comparison.
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

**EPISODE COMPLETION IS PHYSICAL, AND IT IS JUDGED AFTER THE STEP (locked by Defect C,
`ea62e4e`).** The tick's completion check is `executor.is_done(obs)` where `obs` is the
observation `env.step` JUST RETURNED — the world the step produced, not the snapshot the
egos decided on. Three consequences:

- **A non-dead ego counts as physically resolved only once BLADE has actually put it
  back into an airbase inventory.** Issuing `aircraft_return_to_base` is an ORDER, not
  an outcome, so the episode keeps ticking while the aircraft really flies home.
- **A death during the return is reconciled BEFORE `EpisodeResult` is built**, so it
  reaches `EpisodeResult.n_dead` and therefore the terminal reward's `n_lost` — an ego
  that runs its tank dry on the way home is charged as the airframe it really lost.
- **`terminated` and `truncated` behaviour is UNCHANGED**: they are still checked after
  the completion check, in that order, with the same meanings.

**An ego that has committed to return LEAVES PHASE 1.** Once its `rtb_issued` latch is
set it is skipped for the entire Phase-1 chain — no sensing, no `decide_triggers`, no
wake, no policy inference, no belief edit, no `Transition` — so the extra ticks the ride
home costs cannot manufacture fresh decisions out of a mission that is already over.
**Phase 2 still runs every tick for every ego** (that is what lets BLADE land it or
exhaust its fuel), PEERS ARE UNTOUCHED and continue normally, and the one-snapshot
two-phase structure — hence the structural no-communication property — is exactly as
before.

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
Wires env via `_build_env` (`gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=…)`, `obs,info = env.reset()`, blue side by `side.name=="BLUE"`) → `_extract_world` (`create_agents_from_scenario` picks blue; `generate_all_enemy_tasks`) → `solve_and_normalize` twice → `_finish_context` (N `Belief`s + one `GraphPlanExecutor`). `EpisodeContext` carries `env, game, agents, agent_ids, beliefs, executor, a_init, oracle_solution, oracle_tasks, split_meta, observation` (the reset seed the loop reads first), `record`, `placements`, and the two RAW pre-solve world snapshots `known_target_ids` / `executed_target_ids` (the roster-integrity contract below). `Belief.independent(tasks, solution)` mints an independent per-ego copy. `solve_and_normalize(agents, tasks) -> (solution, belief_tasks, unselected)` = `MatchAou(...).solve("bonmin")` → `post_solve_filter_and_level(...)` (allocated-only filter + `task_idx` remap + `level`); **never returns the raw pre-filter list**. Graph-native; imports NOTHING from the flat path. Independence + allocated-only proven in `_selftest`. `EpisodeContext.record: bool = False` — recording is ARMED iff a `recording_export_path` was given; setup never starts the recorder (the tick-loop drives it), and only the RETURNED env is ever armed.

**WORLD INVENTORY IS NOT ORACLE ALLOCATION (locked by the roster-integrity fix,
`36365f2`).** `solve_and_normalize` returns an **ALLOCATED-ONLY** task list by contract —
for BOTH solves. So `belief_tasks` is "the known targets the solver assigned" and
`oracle_tasks` is "the targets the ORACLE assigned"; **neither is an inventory of what
exists.** A target the solver left unselected is absent from both and is nevertheless
physically in the world, sensible, attackable and confirmable. Reading either one as a
world inventory is the defect this contract closes, and it is what made the long baseline
scientifically inconclusive (§7, §8).

`EpisodeContext` therefore carries TWO IMMUTABLE RAW SNAPSHOTS, both taken by
`_world_target_ids` **BEFORE** their solve ever runs, both deduplicated by target id with
first occurrence winning, and both raising on a task that names no target (silently
dropping one would shorten the inventory):

- **`known_target_ids`** — every raw KNOWN-world target id, captured before the known
  solve filtered it. The t=0 known-world inventory.
- **`executed_target_ids`** — every raw target id in the AUTHORITATIVE returned
  environment, in the world's own order, captured before the oracle solve filtered it.
  The t=0 EXECUTED-world inventory: known half plus hidden half.

Both are set on BOTH paths (the legacy path snapshots `split_tasks`' `partial` / `full`),
and `_finish_context` takes them as REQUIRED keywords rather than defaulted ones — so a
future third path cannot reach a context silently carrying an empty world inventory, the
one shape in which allocated-only data gets read as world truth again. It VERIFIES rather
than trusts: the executed snapshot must be non-empty and the known half must be a SUBSET
of it, else `RuntimeError`. **Anything asking "which targets does this episode contain?"
reads these two fields.** `oracle_tasks` / `oracle_solution` are UNCHANGED and remain
exactly right for the reward's oracle denominator — that is a question about ALLOCATION,
and it was always correct. These ids are a RUNTIME snapshot, never a cross-run
reproducibility key: generated target uuids are not seed-derived (§8), so cross-run
comparison is still `geometric_fingerprint(ctx.placements)`.

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
`GraphPlanExecutor` is the **sole** BLADE translation layer (move/launch/attack/RTB). Its intra-level travel ordering comes from the SHARED pure helper `nearest_neighbor_order`, imported from `utils/scheduling_utils.py` — the SAME function `graph_hidden_placement.predict_route` calls, which is what keeps online execution and offline route prediction from drifting apart (`2a3f89c`). `__init__(*, tasks, solution, agents, arrival_threshold_km=DETECTION_KM, add_return_to_base=True, nn_ordering=True, kill_confirm_ticks=60)`. **Per-ego private state:** `self.tasks: Dict[ego_id, List[Task]]` (fanned out at init; diverges only via `resync`), `self.plans` per-ego; `_resolve_step(ego_id, assignment)` is the sole reader of `self.tasks`. Key methods: `next_actions(obs) -> List[str]` (one command/ego/tick), `resync(new_solution, *, ego_id, tasks=None)` (swaps one ego's slice, **never resets `done`**), `is_done(observation)` (**the live observation is REQUIRED**, no default — physical completion; see the Defect-C contract below), `sensed_target_ids(obs, ego_id) -> {id: unit}` (world-scan within `arrival_threshold_km`; the trigger's eyes). done-on-confirmed-kill, per-`(ego,target)` re-fire throttle, single-issue RTB latch (safe only while doctrine `AIRCRAFT_RTB_WHEN_OUT_OF_RANGE` is off — it is in `strike_training_4v5.json`), `dead` set for crashes. No-comms isolation proven in `_selftest` (ISO-1..3: a pop-up appended to ego A never enters ego B's task-view; same-index pop-ups resolve per-ego).

**THE ATTACK-CONFIRMATION WAIT IS DERIVED PER SALVO (locked by Defect B, `39a16f2`).**
`kill_confirm_ticks=60` REMAINS a constructor parameter, but it is now the configured
**MINIMUM and FALLBACK** — not the universal wait armed for every salvo. At ATTACK-ISSUE
time, `_confirmation_wait_ticks(scenario, ego_id, distance_km)` asks the ACTING LIVE
AIRCRAFT's own `get_weapon_with_highest_engagement_range()` — the SAME selector BLADE's
two-argument attack path uses, so the wait is measured against the weapon the engine will
really launch — and pairs it with the CURRENT engagement distance `_command_for_ego`
already computed for this tick. `_salvo_travel_ticks` turns that pair into a conservative
full-distance BOUND:

```text
travel_bound =
    ceil(distance_km
         × KILOMETERS_TO_NAUTICAL_MILES
         ÷ abs(speed_knots)
         × 3600)
confirmation_wait =
    max(kill_confirm_ticks, travel_bound + 1)
```

- **`KILOMETERS_TO_NAUTICAL_MILES = 0.539957` is TRANSCRIBED, not imported**, to preserve
  this module's BLADE-free import closure (it is an import-purity `ENTRY_MODULE`), exactly
  as `graph_fuel_damage` transcribes its own engine constant. The transcription is compared
  against the ACTUAL FROZEN-ENGINE constant (`blade.utils.constants`) in the BLADE test
  tier, which is what would catch drift.
- **The bound is DELIBERATELY NOT an exact reconstruction of BLADE's discrete
  launch / update / endgame schedule**, and must never be read as the number of engine ticks
  a salvo actually takes. Real engagements resolve EARLIER than the bound; the wait only has
  to be long enough.
- **A FINITE NEGATIVE speed is NOT a fallback case** — it is normalized with `abs`,
  matching frozen BLADE's own `platform_speed if platform_speed >= 0 else -platform_speed`,
  and yields the same bound as its magnitude.
- **Fallback to the configured value** covers only: no weapon selected (empty rack, or a
  duck-typed aircraft exposing no selector), and weapon-speed data that is missing,
  non-numeric, non-finite or zero AFTER that normalization — plus an unusable (non-finite
  or negative) engagement distance.
- **The existing confirmed-kill guard still runs FIRST**: when the target is confirmed gone
  it clears the cooldown and advances immediately, so a longer wait throttles RE-FIRE only
  and never delays plan advancement. **Cooldown identity remains per `(ego_id, target_id)`.**
- **Frozen behaviour is untouched:** the emitted two-argument
  `handle_aircraft_attack(ego_id, target_id)` command, its weapon quantity, weapon lethality
  and every vendored BLADE file are unchanged.
- **NO-COMMS:** the derivation reads ONLY the acting ego's own live aircraft and its own
  engagement distance. No peer aircraft, peer inventory, peer belief or peer assignment can
  move this ego's wait.
- **Still out of scope:** general ammunition management and any probabilistic-miss policy.
  An ego with an empty rack still emits its attack and the engine simply launches nothing,
  exactly as before.
- **Defect C was NOT addressed by this fix.** At THIS lock `is_done()` still treated the
  `rtb_issued` latch as RTB-resolved. It was closed separately and afterwards by
  `ea62e4e` — the contract is the next block.

The accepted real-BLADE evidence, both engagements inside the single `DETECTION_KM = 50`
attack envelope, at the production default `kill_confirm_ticks = 60`:

| Engagement | Bound | Derived wait | Real confirmation | Flat-60 result |
|---|---:|---:|---:|---|
| ~47.2 km | 62 | 63 | call 60 | no redundant fire |
| ~49.0 km | 64 | 65 | call 62 | redundant attack on call 61 |

At ~47.2 km — the distance reconstructed from the first short probe's artifacts — the flat
constant was ALREADY below the salvo's bound, and the control arm escaped a redundant salvo
by exactly ONE tick. The ~49.0 km row is a CONTROL demonstrating the SAME premature-refire
mechanism inside the SAME envelope, where that one-tick escape is gone; it is **not** an
exact rerun of the original probe world, and neither row is a scientific probe result.

**COMPLETION IS PHYSICAL, NOT ISSUANCE (locked by Defect C, `ea62e4e`).**
`GraphPlanExecutor.is_done(observation)` takes the LIVE observation as a REQUIRED
argument and has NO observation-free default — a defaulted one would silently restore
the retired command-issuance notion of completion. The verdict has two halves, from two
different sources, and only the second changed:

- **ASSIGNMENTS — still EXECUTOR SEMANTIC STATE, exactly as before:** the ego's `plans`
  slice, the steps `_resolve_step` resolves against its OWN ego-private `tasks`, and the
  proximity-confirmed `done` set. **Nothing here reads the observation.** A dead ego's
  remaining assignments stay terminally unsatisfiable and an unresolvable index stays
  implicitly satisfied, both unchanged.
- **PHYSICAL LIFECYCLE — read from the observation, through the ONE classification site
  `_physical_state(ego_id, observation)`,** which returns exactly one of three states
  because the FROZEN engine keeps every aircraft it still has in exactly one home:
  present in `scenario.aircraft` ⇒ **airborne**; absent from the air but present in some
  `airbase.aircraft` inventory ⇒ **landed** (`Game.land_aicraft` appends there, then
  removes it from the air); absent from BOTH ⇒ **removed/dead** (`Game.remove_aircraft`,
  which `update_all_aircraft_position` calls at `current_fuel <= 0`). It reads only that
  ego's own entries and mutates nothing.
- **`_note_dead(ego_id)` reconciles a newly observed death idempotently** into
  `executor.dead` (and latches `rtb_issued` so the RTB branch stays a no-op for an ego
  that can no longer be commanded). The SAME classifier is used by `_command_for_ego`
  on the PRE-step world and by `is_done` on the POST-step world, so a death is
  reconciled from both sides of `env.step`.
- **Death reconciliation covers EVERY ego before the global verdict.** `is_done` runs a
  total first pass over the (sorted) egos and only then decides, so an early "not done"
  cannot hide a peer's death and the `dead` set the caller reads afterwards is complete
  for this tick.
- **`rtb_issued` remains ONLY the single-issue BLADE-toggle guard**
  (`aircraft_return_to_base` is a BLADE toggle; issuing it twice cancels the RTB). It is **not** survival, **not**
  landing and **not** terminal completion, and it is set for dead and landed egos too.
  Measuring a REAL return still means reading the EMITTED COMMAND history —
  `graph_fuel_damage.FuelDamageController.note_commands` — never this flag.
- **With `add_return_to_base=True`, a live ego whose work is complete or empty is NOT
  terminal while airborne.** With `add_return_to_base=False` the physical check is
  skipped entirely, preserving the existing no-return-required contract for callers that
  opted out.
- **No BLADE engine behaviour changed.** The classification reads what the frozen engine
  already exposes; the vendored files are byte-unchanged (§2).

**Trigger (Stage 2) — `rl/action/graph_trigger.py`.**
`decide_triggers(belief_tasks, belief_solution, sensed_targets, eta=never_overdue, *, ego_id, clock, fuel_damage=False) -> (new_tasks, new_solution, wake, events)`. PURE (no BLADE/torch), copy-on-write (never mutates inputs). The WHEN gate over THREE `TriggerKind` members: **POP-UP** (ego senses an unassigned target → appends a pop-up Task to append-only `belief_tasks`), **PEER-OVERDUE** (ego senses a peer's target AND its ETA passed → removes that peer tuple from the ego's `belief_solution` copy, so it reads as a pop-up — deterministic *gating*, the policy still chooses), and **FUEL_DAMAGE** (FD-BASELINE-v1). ETA is dormant (`never_overdue` = +inf) for now. `FUEL_DAMAGE` is EXOGENOUS — it cannot be detected from sensing, so the orchestrator passes `fuel_damage=True` for AT MOST ONE ego per tick; the flag defaults to `False`, so every pre-FD caller is byte-unchanged. It **edits NEITHER `belief_tasks` NOR `belief_solution`** (the changed quantity is the ego's own live fuel, which the builder reads off the aircraft) and only sets `wake`, appending a `(FUEL_DAMAGE, NO_TASK_INDEX)` event — `NO_TASK_INDEX = -1` is a sentinel, deliberately not `0`, because `0` is a valid task index. A tick carrying both a fuel-damage event and a pop-up still produces exactly ONE wake.

**Build (Stage 3) — `rl/observation/graph_builder.py`.**
`build_graph_observation(scenario, agent_id, current_plan=None, current_time=0, tasks=None, solution=None, precedence_relations=None, config=None) -> GraphObservation`. Stateless projection of `(world, solution)`. `task_features[k, TASK_FEATURE_DIM]` (=6: utility, dist-to-ego, capable, reachable, probability, **sensed**; `TASK_FEATURE_DIM` is the single source of truth the encoder imports), `agent_features[a,1]` (fuel_norm: REAL for ego, `0.0` for peers), COO `edge_index`/`edge_type` over the `EdgeType` IntEnum, `time_norm`. **`ASSIGNMENT` is the only constructed relation** (`SPATIAL` reserved/unused — sensing moved to the `sensed` column; `PRECEDENCE` deferred). Agent set = `ego ∪ assigned same-side peers`. Requires the ego **airborne** (raises otherwise — always satisfied since build only follows a wake, which requires sensing, which requires airborne).

**Encode + decide (Stage 4) — `rl/agent/graph_encoder.py` + `rl/action/graph_action.py`.**
`GraphEncoder.forward(obs, edge_attr=None) -> Tensor[k, embed_dim]` — per-task-node embeddings (NOT pooled), single-graph (no batch dim). Defaults `model_dim=64, embed_dim=64, num_heads=4, num_layers=2, task_feat_dim=TASK_FEATURE_DIM`. Edge-masked symmetrized multi-head attention (torch/numpy only, no PyG/DGL) over `forward + reversed + SELF_LOOP` edges with a learned per-relation `type_bias`; learned TASK/EGO/PEER role embedding (node-typing done HERE, reserved MISSION 4th role); injected `time_norm`; self-loops guarantee no empty-softmax NaN. `pool()` = mean over nodes → the size-agnostic **critic hook**, now CONSUMED by the Phase-B `CentralCritic` (its own SECOND `GraphEncoder` instance + `ValueHead`; the ACTOR's encoder and head are unchanged and carry no value head — see the CTDE contract below). `edge_attr` accepted but `None` today (reserved for expected-exec-time on ASSIGNMENT edges). — `ActionHead(embed_dim, hidden_dim=64, num_meta_actions=3).forward([k,embed]) -> [k,3]`. `build_action_mask(obs, ...) -> [k,3]` (hard physical/structural legality; `OPPORTUNISTIC_ENGAGEMENT` gated by `unassigned` AND `sensed`). `sample_action(logits, mask, deterministic=False) -> (meta:int, node_v:int, log_prob, entropy)`. `evaluate_action(logits, mask, meta, node_v) -> (log_prob, entropy)` re-scores a stored decision through the SAME private `_masked_dist` construction site (grad-mode caller-controlled; masked / out-of-bounds cells fail loud). **Meta-actions (3):** `PLAN_COMPLIANCE`, `OPPORTUNISTIC_ENGAGEMENT`, `SELF_PRESERVATION_ABORT` (Cooperative-Recovery removed — handled upstream by the peer-overdue trigger).
**SELECTION CONTRACT (locked by Defect A, `d56fda6`).** The action surface REMAINS `k × 3`, and EVERY meta-action retains NODE-INDEXED SELECTION IDENTITY: the selected `(node_v, meta_action)` cell is what `sample_action` samples, what `Transition` stores, and what `evaluate_action` re-scores under PPO, with `node_v` still bounds-checked to `[0, k)`. **Selection identity is NOT effect scope**, and the three members differ on the second: `PLAN_COMPLIANCE` performs NO plan edit; `OPPORTUNISTIC_ENGAGEMENT` has a NODE-LOCAL effect (it assigns the ego to THAT task node); `SELF_PRESERVATION_ABORT` has an EGO-GLOBAL effect (Stage 5). `build_action_mask` governs SELECTION only — its per-column legality rules, `NUM_META_ACTIONS`, the logit/mask shape and the sampling/evaluation action identities are all UNCHANGED by Defect A.

**Effect (Stage 5) — `rl/action/graph_effect.py`.**
`apply_meta_action(solution, obs, ego_id, meta_action, node_v, tasks) -> new_solution`. PURE (BLADE-free, torch-free), copy-on-write (`_copy_solution`, never mutates input). comply = no-op; engage = add an ego→task assignment AT THE SELECTED NODE. **ABORT IS EGO-GLOBAL (locked by Defect A, `d56fda6`):** selecting `SELF_PRESERVATION_ABORT` on ANY legal cell clears **ALL** of the acting ego's REMAINING assignments, and **the selected node does NOT scope the effect** — every legal abort cell of a given ego therefore produces the identical empty slice. Only `solution[str(ego_id)]` is written: **peer assignment slices, peer beliefs and every task list stay untouched**, `tasks` remains append-only, and `GraphPlanExecutor.done` is not reset. An ego with no key already has an empty mission, so the dict SHAPE is preserved as found (no key is invented). The layer stays PURE and **issues no BLADE command of any kind**: `graph_tick_loop._wake_decision` resyncs ONLY the acting ego's executor slice, and the resulting EMPTY PLAN reaches `GraphPlanExecutor.next_actions` in **Phase 2 of the SAME tick** — the wake, this plan edit and the resync all happen in Phase 1, before any `env.step` — where the PRE-EXISTING empty-plan branch emits the single latched `aircraft_return_to_base`. Nothing new was built for RTB. Does NOT touch the graph — the edge appears on the next rebuild.

**Resync (Stage 6)** — `GraphPlanExecutor.resync` (above): swaps the ego's plan slice without resetting `done`.

**Reward (Stage 7) — `rl/training/graph_reward.py`.**
`compute_episode_reward(ctx, result, cfg=RewardConfig()) -> EpisodeReward`. **Terminal, utility-based** (v1): `R = (U_achieved − c·U_aircraft·n_lost − U_oracle)/(|U_oracle| + eps_regret)`, placed on the last wake's `Transition` (others `0.0`; empty trajectory ⇒ nothing attached). `U_oracle = plan_value(ctx.oracle_solution, ctx.oracle_tasks)` — **bit-faithful to `MatchAou._add_objective`** (reuses the solver `EPSILON`; the `y[j]` factor is provably redundant given the y/x constraints; proven under bonmin in `_selftest` T1). `U_achieved = realized_utility(ctx.oracle_tasks, ctx.executor.done)` — full utility IFF all a task's targets are confirmed-killed, **deduped over ego**. `c = aircraft_penalty_coeff` — this module's own default is **0.0**, but BOTH harnesses now pass an explicit `RewardConfig(aircraft_penalty_coeff=2.25)` (FD-BASELINE-v1, below); the FORMULA is unchanged. `n_lost = len(ctx.executor.dead)`; `eps_regret=1e-5` is a division guard (distinct from solver EPSILON). **No-comms:** a centralized/privileged TRAINING signal — MAY read global state, but MUTATES ONLY `Transition.reward` (proven byte-unchanged on real objects in T7). **KNOWN v1 assumption `probability=1.0`** (expected `U_oracle` vs realized `U_achieved` coincide only at p=1; `R∈[-1,~0]`; revisit at p<1).

**The two-phase tick (Stages 2–6) — `rl/training/graph_tick_loop.py`.**
`run_episode(policy, ctx, cfg=None, *, deterministic=False, max_ticks=None, fuel_damage=None) -> EpisodeResult`. Strict two phases per tick: **Phase 1** runs every ego's `sensed → decide_triggers → (on wake) _wake_decision` against the SAME `obs` snapshot with **no** `env.step`; **Phase 2** issues ONE `env.step(executor.next_actions(obs))`, and the tick's completion verdict is `executor.is_done(<the POST-STEP obs that step just returned>)` — completion is a PHYSICAL fact about the world the step produced (Defect C, `ea62e4e`), so an episode keeps ticking while an ordered-home aircraft actually flies home, and a death on that return is reconciled into `executor.dead` by the same call, BEFORE the loop returns, hence into `EpisodeResult.n_dead`. An ego whose `rtb_issued` latch is set is SKIPPED for the whole of Phase 1 from then on — no sensing, trigger, wake, policy inference, belief edit or `Transition` — while Phase 2 still runs for it every tick and peers continue normally. The optional `fuel_damage` controller (FD-BASELINE-v1) is consulted at the TOP of a tick, before Phase 1, and its Phase-2 `note_commands` call is a read-only measurement — see §4 and the FD contract below; `None` (the default) leaves the loop byte-unchanged. Because BLADE advances only after all egos decided on the identical snapshot, Phase-1 ego order cannot affect the outcome (structural no-comms; proven in `_selftest`: `env.step` count == tick count). `_wake_decision` is the per-wake chain (Stage 3→6) under `torch.no_grad`, editing ONLY the acting ego's belief. `Policy` (`build_policy()`) bundles encoder+head, built ONCE, lives across episodes. Seam for reward/PPO: `EpisodeResult.trajectory: List[Transition]`. The loop does NOT own the agent lifecycle (executor owns `dead`/`done`/`rtb`/`is_done`); it only hands `is_done` the post-step observation and READS the answer. **The reward seam is unchanged:** `graph_reward`'s formula still reads `n_lost = len(ctx.executor.dead)` — what changed is that the set is now truthful at episode end. **Recording:** armed by setup (`ctx.record`), driven here — start + forced t=0 frame before the loop, throttled `record_step` after each Phase-2 step (before the exit checks), forced terminal frame + `export_recording` after the loop (all exit paths). A pure READ of engine state; default off is a no-op — observational purity proven in `_selftest` TEST 1b (identical `(ended, ticks, n_wakes)` with recording on/off). Artifact: `{export_path}/{scenario_name} Recording {start} - {end}.jsonl`.

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
  `accounting_reconciled` cross-checking record counts against the ledger), and the
  THREE figures under `plots/` (`plot_training`, jsonl-only, torch-free child — see the
  harness contract below). A run root holds records, `scenarios/`, `checkpoints/`,
  optional `visual_artifacts/` and `plots/` as separate things.
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
  `EpisodeRosterError` and never contributes a false successful zero. Reward and PPO
  semantics are unchanged; the reward already deduplicated by target id.
  **PR #7's ROUTING of that error is SUPERSEDED and must not be restated:** it was an
  accounted `setup` failure then; since `36365f2` (§7) `EpisodeRosterError` is a
  `MeasurementIntegrityError` and ABORTS the run as INFRASTRUCTURE — see the
  roster/world-truth integrity contract below.
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
    `incomplete` until the bundle is whole, and — since `36365f2` — `complete` only once
    the three files exist AND the observed world cardinality RECONCILES with the scheduled
    cell; a bundle may be read as full only when it is `complete`. See the
    roster/world-truth integrity contract below for the two corrections this claim
    depends on (`sync_recordings`, and `finalize`'s reconciliation).

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

**Roster / world-truth integrity (PR #24) — `rl/training/graph_train.py` +
`rl/training/graph_episode_setup.py`.**
The *measurement-integrity* half of the trainer contract, and the correction that a real
long baseline forced. It changed NO pipeline layer: the reward formula, PPO,
oracle allocation, fuel-damage semantics, B2 placement, seed formulas, the evaluation
schedule, the tick loop, the executor, the generator and vendored BLADE are all exactly as
their own locks left them (§7).

- **The roster's WORLD comes from the two raw pre-solve snapshots, never from an
  allocation.** `_episode_target_roster(ctx)` reads `ctx.known_target_ids` (KNOWN) and
  `ctx.executed_target_ids` (EXECUTED) through the validating accessor
  `_world_snapshot_ids`, and derives HIDDEN by SUBTRACTION — executed minus known, in
  EXECUTED-WORLD ORDER (`ctx.placements` is deliberately id-free, so it cannot supply
  hidden ids). **`ctx.oracle_tasks` IS NOT READ THERE AT ALL** and must not be
  reintroduced. The BELIEFS are still checked, but only in the role they can play: they
  are allocated-only too, so they are a **SUBSET** constraint on the known world, never
  its denominator — a belief naming a target the known snapshot does not hold still
  raises, because the egos would have been planned against something the world does not
  contain. The t=0 belief-agreement check is unchanged.
- **`MeasurementIntegrityError` — INFRASTRUCTURE, never a scientific outcome.**
  `EpisodeRosterError` is now a subclass of it (the NAME is retained so an audit trail
  keeps reading; what changed is where it GOES, not what it means). It is a sibling of
  `_VisualArtifactError` and routed identically: it names no pipeline stage because it did
  not happen in one, and it **ABORTS the run**. It is NEVER wrapped in
  `EpisodeAttemptError`, never appended to `episode_failures.jsonl`, never counted against
  a condition tally, and never enters `skip_and_account_v1` — so it can no longer shrink a
  scientific denominator while reading as ordinary episode attrition. Both attempt handlers
  re-raise it AHEAD of their broad `except Exception` (`except (_VisualArtifactError,
  MeasurementIntegrityError)`), and an UNEXPECTED exception raised inside the roster code is
  normalized into the same loud path with its cause preserved. **This is a deliberate
  reversal of PR #7's routing**, and the reason is that a data-integrity fault is a property
  of the INSTRUMENT, not of the episode: every episode it touches is suspect, and the ones
  it does not touch cannot be assumed unaffected.
- **`_require_scheduled_cell(roster, cfg)` — the roster must describe the cell the
  schedule asked for.** `n_known` known, `n_hidden` hidden, `n_targets_emitted` executed,
  or `EpisodeRosterError`. `setup_episode`'s construction path already enforces that
  cardinality loudly on its own side, so a roster that disagrees is not a scenario that
  came out differently — it is this module measuring the world wrongly. Checked BEFORE the
  fuel-damage plan and BEFORE `run_episode`, so nothing is paid for and no partial
  measurement exists.
- **ORDER AFTER `run_episode` IS CONTRACTUAL: synchronize the playback, validate the
  world, and only then compute a reward.** `_AttemptArtifacts.sync_recordings()` runs
  immediately after `run_episode` returns and DISCOVERS the playback chunks the completed
  run really wrote into the still-`incomplete` manifest — nothing is created, renamed or
  fabricated; a completed run with no playback file is itself a `_VisualArtifactError`,
  because the tick-loop contract exports one on every completed run and none when the loop
  raised. Only then is the confirmed-id set reconciled against the executed-world snapshot,
  and only a world that validated is allowed to produce a reward and a successful outcome.
  The long baseline ran it the other way round, which is how 17 episodes exported a real
  recording that no manifest listed.
- **A manifest cannot claim `complete` against its own files.** `_AttemptArtifacts.finalize`
  requires all three artifacts AND reconciles expected vs observed target counts; on a
  mismatch it still WRITES the observed counts, leaves the status `incomplete`, and raises
  `_VisualArtifactError`. `complete` is a CLAIM, so a manifest certifying a world its own
  `executed_t0_scenario.json` contradicts is worse than no manifest.
- **The authoritative count is unchanged**, and is still
  `len(_unique_confirmed_target_ids(executor.done))` and nothing else — never derived from
  how many ids the roster managed to NAME. A name that will not resolve still degrades to
  `<unnamed target>` and changes no id and no count.

**Experiment harness: JSON presets, run layout and the three figures (PR #14) —
`rl/training/graph_train.py` + `configs/graph_train/final_cell_probe.json`.**
The OPERATOR-facing surface. It changed no pipeline layer, no scenario semantics, no seed
schedule, no PPO/reward/fuel-damage behaviour and no evaluation record field; what it
changes is how a run is CONFIGURED and how its results are PRESENTED.

- **JSON presets — `--config <path>`.** Stdlib `json` only, no new dependency. A preset
  names `TrainConfig` FIELDS (nested PPO knobs under `"ppo"`), never CLI flag spellings,
  so `TrainConfig` remains the one configuration authority and there is no second naming
  scheme to drift from it. Keys beginning with `_` are comments; an UNRECOGNIZED key
  RAISES rather than being ignored, because a knob silently left at its default produces
  a run whose file says one thing and whose behaviour is another. Resolution is
  **dataclass/CLI defaults < preset < EXPLICITLY typed CLI flags**. "Explicit" is
  measured by re-parsing argv through a throwaway parser whose defaults are
  `argparse.SUPPRESS` (`_explicit_cli_dests`) — a parsed namespace cannot tell an absent
  flag from one passed its own default, and inferring it from the VALUE would let every
  differing default silently override the preset. Both parse passes consume ONE vector
  (`_effective_argv`): `argparse` reads `None` as `sys.argv[1:]`, and `main()` is normally
  called with no argument, so reading `None` as `[]` anywhere would make every typed flag
  look un-typed. Symbols: `load_config_file`, `resolve_train_config`, `_effective_argv`,
  `_explicit_cli_dests`, `_CLI_FIELD_BY_DEST` / `_CLI_PPO_FIELD_BY_DEST` (the ONE
  dest→field mapping), `_CONFIG_TUPLE_FIELDS`.
- **The repository short-probe preset — `configs/graph_train/final_cell_probe.json`.**
  The bounded short probe, and deliberately the ONLY preset the repository owns: 2
  scheduled training iterations × 4 scheduled attempts, `base_seed = 0`, `eval_every = 2`,
  4 fixed held-out seeds from `1_000_000`, giving one `pre_update` and one `post_update`
  matched round; the final 3-agent / 3-known / 3-hidden cell with its 200 km / 100 km
  geometry and `include_sams = false`; FD-BASELINE-v1 unchanged; `visual_artifacts = true`.
  Every field it sets that also has a dataclass default AGREES with that default, so the
  preset RESTATES the approved cell instead of retuning it (test-enforced). **TWO
  SCHEDULED ITERATIONS DO NOT IMPLY TWO PRODUCTIVE PPO UPDATES:** `updates_completed`
  advances only when the updater actually runs epochs, so a successful zero-wake iteration
  leaves it unchanged and the value may be 0, 1 or 2. Productive-update yield is one of
  the things the probe MEASURES; nothing may assume it. No long-baseline preset exists.
- **`run_config.json:/config_source` — ALWAYS a structured object, never `null`.** One
  schema, one construction site (`config_source_record`, whose `resolved_from` is a
  REQUIRED argument, never inferred), and exactly THREE truthful kinds
  (`_CONFIG_SOURCE_KINDS`): `config_file` (a command line naming a preset; `path` /
  `absolute_path` say which), `cli_defaults` (a command line with no `--config`), and
  `direct_config` (a `TrainConfig` built in Python and handed straight to `train()` — what
  `_selftest` and any importing script do). The record also carries `config_fields` (what
  the preset supplied) and `cli_overrides` (what an explicit flag took back off it), and
  it is validated for internal consistency: only `config_file` may carry a path, and it
  must carry one. The three kinds exist because a provenance field that is WRONG in a
  believable way — a direct call recorded as `cli_defaults` — is worse than one that is
  absent.
- **Figures: `<run_dir>/plots/`, three files, one claim each.** The legacy single
  four-panel `training_plot.png` is RETIRED and is no longer written. `plot_training` and
  `plot_training_subprocess` now return the LIST of figures written (empty when there is
  nothing to plot or matplotlib is missing; matplotlib stays optional and never fails a
  run). `run_summary.json` carries `plots_dir` + `plot_paths`, and the legacy `plot_path`
  key survives as a documented ALIAS of the performance figure.
  - `training_performance.png` — training reward; held-out evaluation as TWO SEPARATE
    per-condition series; the matched-pair delta.
  - `policy_diagnostics.png` — meta-action mix and policy entropy over the TRAINING
    decisions only.
  - `measurement_health.png` — the denominators, titled as health and explicitly NOT
    performance: train `success_fraction` and `wake_fraction_of_successful`, eval episode
    `success_fraction`, eval `pair_success_fraction`, the absolute counts, and
    PER-CONDITION held-out completion (`eval_n_<condition>_attempted` / `_successful`).
- **The two presentation invariants.** (1) **Condition means vs the paired delta.**
  `eval_reward_mean_clean` and `eval_reward_mean_damaged` are each a mean over THAT
  condition's own SUCCESSFUL episodes, so when one condition fails more held-out seeds
  the two curves are not averages over the same completed seeds and their gap is NOT a
  within-seed effect — the panel title, both legend entries and the per-condition
  denominators say so. The ONLY within-seed comparison is `eval_paired_reward_delta`, over
  pairs whose BOTH members completed. Pooling the two conditions into one held-out curve —
  what the retired dashboard drew — averages across the very factor the cell was built to
  study, and is drawn only as an explicitly labelled fallback for pre-FD records carrying
  no per-condition means. (2) **The honest x-axis.** All three figures share ONE quantity,
  stamped on each: PPO updates completed BEFORE the measurement. Training points sit at
  `updates_completed_before` (the updates the policy that GENERATED those episodes had
  received) and eval points at `updates_completed`, so the untrained policy's first batch
  and its `pre_update` round share an origin. A batch or round with NO successful episode
  is DROPPED from a curve rather than drawn at 0 — the reward is oracle-normalized regret,
  so 0 is the OPTIMUM and plotting a total data loss there would invert its meaning; the
  gap is accounted for in `measurement_health.png`.

**FD-BASELINE-v1 — the LEGACY difficulty factor, and the PRESERVED Phase-A semantics —
`rl/training/graph_fuel_damage.py`** (consumed by `graph_tick_loop.run_episode`,
`graph_train` and `graph_rollout`).

THE **ONE** SELECTED DIFFICULTY FACTOR of the Phase-A reference baseline cell, and the
design the approved Phase-A long-baseline measurement (`737b4bf`, §7) was taken on. The
scenario is otherwise UNCHANGED: 3 agents, 3 known + 3 route-relative hidden airbase
targets, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams=False`,
`probability = 1`, unchanged BLADE weapon lethality, frozen solver, unchanged PPO. No
second factor is bundled in (§8).

**THIS CONTRACT IS UNCHANGED BY FD-VARIABLE-SEVERITY-v1** (the block that follows). The
legacy modes — `off`, `seeded_mixture`, `forced_clean`, `forced_damaged`
(`FuelDamageMode.LEGACY`) — keep the same seeds, the same conditions, the same selected
egos, the same PLANNED-midpoint target (`TARGET_POLICY_PLANNED_MIDPOINT`) and the same
live check order. That preservation is load-bearing rather than tidy: an approved
measurement exists on these modes, and a factor that quietly moved them would invalidate
that baseline instead of extending it.

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

**FD-VARIABLE-SEVERITY-v1 — the mild/severe extension of the SAME event —
`rl/training/graph_fuel_damage.py` + `rl/training/graph_train.py` +
`rl/training/graph_rollout.py`.**

MERGED AND LOCKED (`eecc9b5`, §7), and **MEASURED ONCE — the actor-only baseline at
measured code SHA `bf1e045f` is EXECUTED, independently reviewed and
`APPROVE — VALID MEASUREMENT`, and its PRIMARY behavioural finding is NEGATIVE: the
deterministic held-out actor showed NO severity-conditioned FD-wake meta-action
separation** (§7 owns the authoritative record and every denominator; §8 owns the phase
state). Nothing beyond that record may be claimed for this design. It is an ADDITIONAL
actor-only stress design layered on the legacy factor, not a replacement for it and not a
reopening of the closed Phase-A reference.

- **WHY.** Under FD-BASELINE-v1 every damaged episode is structurally SEVERE, so
  "damaged" and "continuing is infeasible" are the SAME fact and a trained actor can
  learn the shortcut `fuel damage ⇒ abort` without ever reading its own fuel gauge. The
  variable design splits the damaged half into a band where continuing REMAINS feasible
  and a band where it does not, which is what makes the response a real decision that has
  to be read off the ego's own live fuel.
- **The modes.** `FuelDamageMode.VARIABLE` = `seeded_variable` (TRAINING),
  `forced_mild` and `forced_severe` (the two damaged EVALUATION members).
  `forced_clean` is DELIBERATELY NOT in that tuple — a clean member has no severity under
  either design, and listing it would make "is this a variable-severity run?"
  unanswerable from one evaluation member's mode. `FuelDamageParameters.variable_severity`
  is the ONE predicate behind that question; a RUN's design is keyed off its TRAINING
  mode (`TrainConfig.variable_severity` ⇔ `fuel_damage_mode == seeded_variable`).
- **The scheduled distribution.** `seeded_variable` draws the clean/damaged bit with
  EXACTLY the `seeded_mixture` draw — same domain, same order, same `probability` — and a
  damaged episode is THEN assigned a severity. With `fuel_damage_probability = 0.50` and
  `fuel_damage_mild_probability = P(mild | damaged) = 0.50` that is the approved flat
  **0.50 clean / 0.25 mild / 0.25 severe**. It is stated as TWO independent knobs so that
  "how often is anything damaged" stays the knob it has always been, and
  `_scheduled_cell_probabilities` records the PRODUCT in `run_config.json` so a mis-set
  conditional is visible in the artifact rather than only in the results.
- **SEVERITY HAS ITS OWN RNG DOMAIN, AND THE SEPARATION IS LOAD-BEARING.** The legacy
  condition/ego draws stay in `fuel_damage_v1` (`derive_fuel_damage_seed`); severity comes
  from `fuel_damage_severity_v1` (`derive_fuel_damage_severity_seed`), same SHA-256
  construction, separate stream. Taking the mild/severe bit from the v1 stream would
  insert a draw BETWEEN the mixture bit and the ego selection and change WHICH EGO every
  damaged episode picks — silently invalidating the approved FD-BASELINE-v1 measurement
  instead of extending it. With two domains the decisions are orthogonal: severity cannot
  move the ego and the ego cannot move severity. `resolve_severity` returns `None` under
  every legacy mode — "this episode carries no severity LABEL", which is a different
  statement from "this episode was mild", so a legacy record is never re-read as a
  variable one — and the forced severity modes still TAKE the draw and discard it, so a
  forced member's stream position matches its seeded counterpart's.
- **THE TWO LIVE BANDS** (`severity_band`, the ONE arithmetic site for both severities AND
  for the legacy design), measured against the same `measure_window` output — `F_rtb` =
  `rtb_fuel_floor`, `F_cont` = `continue_fuel_requirement`, both already carrying the
  `margin = 1.10` reserve:
  - **MILD** — the OPEN interval `(F_cont, F_before)`: `F_rtb < F_cont < F_after <
    F_before`. A real LOSS (strictly below the pre-damage fuel), safe RTB feasible, and
    completing the remaining route and THEN returning still genuinely feasible.
  - **SEVERE** — the half-open interval `[F_rtb, F_cont)`: `F_rtb ≤ F_after < F_cont ≤
    F_before`. A real loss, safe RTB feasible, continuation infeasible. **This is exactly
    the legacy interval**, which is why "severe reproduces the legacy physics" is
    checkable rather than merely asserted. What still differs is WHERE the interval is
    measured, below.
- **TARGET POLICY — the one behavioural difference from legacy.** The legacy design keeps
  `TARGET_POLICY_PLANNED_MIDPOINT`: it applies the PLANNED value and validates it live.
  The variable design uses `TARGET_POLICY_LIVE_SEVERITY_MIDPOINT` — the post-damage fuel
  is DERIVED at the event tick as the midpoint of the severity's band measured from the
  LIVE window and the LIVE fuel. Mild and severe are statements about the fuel the ego
  really holds where it really is, and a value fixed before the run could only be CHECKED
  against that, never guaranteed to land in the right band of it. The midpoint is the
  point furthest from both ends, so neither bound is decided by floating-point noise.
  `FuelDamageController.maybe_apply` keeps the two designs' live checks in separate
  helpers (`_live_legacy_target` / `_live_variable_target`) precisely so the legacy CHECK
  ORDER — which decides what an already-measured `run`-stage failure reports — cannot be
  disturbed.
- **FAILURE POLICY IS UNCHANGED AND STILL LOUD.** `_require_valid_band` checks four facts
  in order (non-degenerate interval; the midpoint really inside it, honouring the
  inclusivity that distinguishes mild from severe; a real loss; safe RTB still
  affordable), each with its own message and a `planned` / `live` label. It raises BEFORE
  the mutation, so a refused event leaves the engine untouched. **NOTHING is clamped,
  weakened, re-planned, retried, downgraded to the other severity, given a replacement
  ego, or converted to a clean episode** — a silent downgrade would move the population
  every per-condition statistic is reported over. The attempt lands in
  `skip_and_account_v1` exactly as before.
- **NO-COMMUNICATION AND OBSERVABILITY ARE UNCHANGED.** The actor is never told which
  case it is in: **no severity label reaches `GraphObservation`**, no severity feature and
  no new node/edge/column exist, and the only thing that changes in the ego's input is its
  OWN real `fuel_norm` — which is exactly what the decision has to be read off. Peer graph
  rows stay featureless, so no peer fuel leaks. The layer's PURITY is unchanged (no BLADE,
  gymnasium, torch, solver, file I/O or module-global randomness).
- **WHAT IS EXPLICITLY NOT IN THIS DESIGN.** Target destruction stays DETERMINISTIC at
  `probability = 1`; BLADE weapon lethality, the frozen solver, `graph_reward`'s formula,
  PPO, the encoder, the action space, `DETECTION_KM`, B2 placement, the seed schedules and
  the vendored engine are all unchanged. **`p(destroy) < 1` is a SEPARATE future Grade-A
  research task and was NOT implemented here** (§8).

**FD-VARIABLE-SEVERITY-v1 measurement surface — matched TRIADS and the durable outcome
stream — `rl/training/graph_train.py`.**

- **The matched CLEAN / MILD / SEVERE TRIAD** (`_EVAL_TRIAD_MEMBERS`, in attempt order,
  each member a `(cell, mode)` pair). All three members use the SAME held-out seed —
  hence the same generated world, the same solved `A_init`, the same hidden geometry, and
  for the two damaged members the SAME deterministically selected ego — and differ ONLY in
  the fuel-damage event. That is what lets "did the actor respond DIFFERENTLY to a
  survivable loss than to an unsurvivable one?" be asked WITHIN one world rather than
  across worlds. The clean member reuses the existing `forced_clean` mode rather than
  needing a new one. Members carry DISTINCT artifact tags (`eval_member_tag`, slot
  `e·group_size + m`), and `TrainConfig.validate` sizes the tag namespace against the
  group size the run will really use. **A legacy run keeps its clean/damaged PAIR and
  evaluation NEVER silently becomes a triad** — only a `seeded_variable` run evaluates
  triads; `TrainConfig.eval_group_kind` reports `pair` or `triad` so a reader never has to
  count members.
- **A CELL IS A REPORTING LABEL, NOT A NEW CONDITION.** `_ConditionTally` stores per CELL
  — `clean` / `damaged` for a legacy run, `clean` / `mild` / `severe` for a variable one —
  and the clean/damaged keys are DERIVED by pooling (`cell_condition`). For a legacy run
  the cells ARE the conditions, the pooling is the identity, and every emitted key keeps
  exactly the value it had.
- **PRIMARY BEHAVIOURAL EVIDENCE: the severity-conditioned FD-WAKE META-ACTION RESPONSE**,
  not reward. It is tracked per cell with its OWN denominator — **FD WAKES**, which is at
  most the cell's successful-episode count and CAN be smaller (an event can fire without
  the policy ever being woken by it), so it is stored and reported separately rather than
  inferred. `_ConditionTally.success` counts EVERY successful episode of the cell but
  increments `fd_wakes[cell]` only on `wake_occurred`, so `fd_wakes[cell] <=
  successful(cell)` — with EQUALITY when every successful episode in that cell did produce
  an FD wake. The point of the separate denominator is that the two CAN diverge, never
  that they must.
  Rates are `None`, never `0.0`, on an empty wake population. **"Mild must always choose
  `PLAN_COMPLIANCE`" is NOT encoded anywhere as a correctness rule and must not be** —
  opportunistic engagement under a survivable loss can be rational; what is measured is
  whether the response DIFFERS, not whether it matches a prescribed label.
- **THE THREE WITHIN-SEED DELTAS** (`_EVAL_TRIAD_DELTAS`): `mild − clean`,
  `severe − clean`, `severe − mild`. **Every one is averaged over COMPLETE matched groups
  ONLY** — a triad needs ALL THREE members to succeed, a group with a failed member
  contributes to NONE of the deltas, is never repaired from its surviving members (a
  clean+mild pair inside a failed triad yields no mild−clean delta either), and is still
  visible in the attempt counts. The per-cell reward MEANS remain each over THAT cell's
  own successful subset, so — exactly as for the legacy pair (§5, the two presentation
  invariants) — the only within-seed claims are the deltas.
- **`episode_outcomes.jsonl` — ONE durable record per SUCCESSFUL attempt.** The
  per-iteration and per-round records are AGGREGATES, and an aggregate cannot be
  un-averaged: "how did the actor respond to MILD, episode by episode, and in which
  worlds?" is a per-episode question. Each record states its own identity, event and
  outcome once, appended and FLUSHED immediately, so a run killed mid-batch still accounts
  for every completed attempt. **It never duplicates the ledger** — failed attempts stay
  in `episode_failures.jsonl` and appear here NOT AT ALL, so the two files are disjoint by
  construction. **Missing is `null`, never `0`** (a clean episode has no fuel reading, an
  unfired event has no tick, an absent wake has no meta-action; a zero would read as a
  measurement). **`run_summary.json:/severity_response` is DERIVED FROM THIS FILE**
  (`_severity_response_from_outcomes`, with `severity_response_source` naming it), not
  from a separate in-memory aggregate — one metric path, so the summary cannot describe a
  run its own artifacts do not.
- **Everything the legacy reporting surface carried is preserved**: attempted /
  successful / failed counts per cell, per-cell reward means, RTB command yield (still
  from `FuelDamageOutcome.rtb_command_issued`, i.e. real Phase-2 COMMAND HISTORY, never
  the executor's `rtb_issued` latch), deaths and target coverage. The legacy
  damaged−clean delta key is `null` under a triad run, whose three named deltas are the
  complete statement of it.

**PHASE-B CTDE — the TRAINING-ONLY centralized critic —
`rl/observation/central_graph_builder.py` + `rl/training/graph_ppo.py` (its §7 block)
+ `rl/training/graph_tick_loop.py` + `rl/training/graph_train.py`.**

BUILT, REVIEWED and MERGED (`a6f3aa9`, integrated `8390d85`, PR #30 — §7). **NO
actor-only vs CTDE scientific comparison has been executed, and no CTDE benefit is
established** (§8 owns that gate). What follows is the IMPLEMENTED contract, derived from
the integrated code, not a design proposal.

- **TWO TRAINING MODES, SELECTED BY `TrainConfig.training_mode` AND BY NOTHING ELSE.**
  `TRAINING_MODES` = (`actor_only`, `ctde`); `actor_only` is the DEFAULT. The ONE predicate
  behind every branch is `TrainConfig.ctde_enabled`, which reads `training_mode` and
  nothing else. **`value_coeff` IS NOT A MODE SELECTOR**: under `training_mode='ctde'`,
  `validate()` REJECTS `value_coeff <= 0` outright, because a run so configured would build
  central observations and take its advantages from a critic it never trains — neither
  reference algorithm, and recorded as CTDE either way. `validate()` also bounds
  `gae_lambda` to `[0, 1]` and requires `critic_lr > 0`; on an `actor_only` run the unused
  CTDE block may hold any value and is not validated.
- **`actor_only` IS PRESERVED, NOT EMULATED.** It constructs NO critic, NO
  `CentralStateRecorder`, NO `CTDEBuffer`, NO `CTDEUpdater` and NO `CTDEEpisodeRecord`, and
  it computes no central observation, value loss or CTDE advantage. The keyword-omission
  helpers `_ctde_kwargs` / `_central_kwargs` return `{}` rather than a `None`-valued
  keyword, so `_run_one_episode` and `run_episode` are called with EXACTLY their pre-CTDE
  arguments — the stronger invariance claim, and the same pattern `_artifact_kwargs`
  already used. `graph_ppo`'s actor-only half (`EpisodeRecord` / `PPOBuffer` /
  `compute_returns_and_advantages` / `PPOUpdater`) is BYTE-UNCHANGED. This is proven by a
  POISON test: every central-CTDE construction site is replaced by a raiser and an
  `actor_only` run still completes, with a companion CONTROL that flips the mode and shows
  the poison really fires.
- **DECENTRALIZED EXECUTION IS UNCHANGED IN BOTH MODES.** The runtime actor path is still
  `private ego GraphObservation → GraphEncoder → ActionHead → mask → sample`. No central
  state, no peer privileged state, no critic value and no critic parameter reaches action
  selection or `evaluate_action`: `CTDEUpdater._forward_logits` re-encodes the stored
  PRIVATE `tr.gobs` and nothing else, and the advantage crossing from critic to actor is a
  plain python float. `evaluate` takes NO critic argument and constructs neither a critic
  nor a recorder — held-out evaluation is actor-only in both modes — and a CTDE-trained
  actor runs with the critic object absent. §3 is not weakened by centralized TRAINING.
- **ARCHITECTURE — ACTOR AND CRITIC SHARE NOTHING.** `CentralCritic` owns its OWN
  `GraphEncoder` INSTANCE (the same class, constructed with the CENTRAL feature widths —
  all three were already constructor parameters, so the encoder itself was NOT changed) plus
  its own `ValueHead`, and `CTDEUpdater` builds a SECOND Adam over the critic's parameters
  alone. The two parameter sets are DISJOINT — no sharing, tying or copying — and the actor
  loss and the value loss are backpropagated in TWO SEPARATE `backward()` calls, each with
  its own grad-norm clip and its own `optimizer.step()`. `ValueHead` is a
  `Linear → Tanh → Linear` MLP over the pooled `[embed_dim]` summary, orthogonally
  initialized (hidden at the default `sqrt(2)` gain, OUTPUT at `std=1.0`, the conventional
  value-head gain), so the untrained critic is an arbitrary small-magnitude function of the
  state and **NOT zero everywhere** — nothing relies on it being zero, because a uniform
  offset cancels in the batch-mean subtraction of `compute_ctde_advantages`.
- **THE CENTRAL GRAPH IS THE LIVE WORLD, AND PRESENCE IS LIVENESS.**
  `build_central_graph_observation(scenario, *, agent_ids, executor, current_time, config)`
  is STATELESS, like the actor builder, and returns a `CentralGraphObservation` —
  a DISTINCT type, not a `GraphObservation` and not a subclass of one, carrying NO
  `agent_id` field, so a central state can never be mistaken for an actor state.
  - Task nodes are one per LIVE enemy target, enumerated through the SAME
    `generate_all_enemy_tasks` current-world extraction episode setup uses — so the
    inventory is the RAW LIVE WORLD and an unallocated target is present. **`oracle_tasks`
    is NOT read**, which is the roster-integrity contract (Stage 0) honoured rather than
    repeated. A destroyed target simply has no node; there is no dead/alive flag.
  - Agent nodes are one per originally-scheduled same-side agent that is physically alive,
    in the caller's scheduled order. `live_aircraft` collapses the executor's own three-way
    classification: airborne (in `scenario.aircraft`) or landed (in some `airbase.aircraft`
    inventory) is LIVE; absent from both is dead and loses its node. **RTB ISSUANCE AND
    LANDING ARE NOT DEATH** — an ego ordered home keeps its node even though Phase 1 stops
    processing it.
  - **THERE IS NO DISTINGUISHED EGO.** `ego_index` is `NO_EGO_INDEX` (`-1`), and the shared
    encoder marks a node EGO only for `0 <= ego_index < N`, so every agent node keeps the
    same role and the graph is SYMMETRIC over live agents. No encoder change was needed and
    none was made.
  - **FEATURES, exactly as implemented.** `task_features[k, 2]` = `[utility_norm,
    probability]` (`CENTRAL_TASK_FEATURE_DIM`). `agent_features[a, 1]` = `[fuel_norm]`
    (`CENTRAL_AGENT_FEATURE_DIM`) — **REAL for EVERY live agent**, which is the exact
    asymmetry the actor graph must not have (peer fuel is unsensable under
    no-communication; the point of a centralized critic is that TRAINING may read it).
    `time_norm` is the actor's own `current_time / max_sim_ticks`, clipped, from the SAME
    `GraphObservationConfig` the actor builder is using on that episode.
  - **EDGES: the COMPLETE live-agent → live-target bipartite relation**, one
    `EdgeType.SPATIAL` edge each (`CENTRAL_EDGE_TYPE`; SPATIAL is RESERVED / unused in the
    actor graph, so borrowing the code changes nothing the actor builds), with
    `edge_attr[E, 5]` = `[distance_norm, capable, reachable, sensed, assigned]`
    (`CENTRAL_EDGE_ATTR_DIM`). `reachable` IMPORTS the actor's own
    `graph_builder._reachable_by_ego` round-trip model rather than reimplementing one;
    `sensed` is privileged ALL-AGENT sensing at the ONE unified `DETECTION_KM`; and
    **`assigned` is CURRENT executor plan membership** — `plan_target_ids` resolves
    `executor.plans[agent]` against `executor.tasks[agent]` with `_resolve_step`'s own
    bounds semantics (a documented MIRROR kept out of that module's import closure, its
    equivalence TEST-ENFORCED against a real `GraphPlanExecutor`), **never from
    `oracle_solution`, never from a private belief, and never from t=0 `A_init` after
    runtime adaptation**. It is plan MEMBERSHIP, not eligibility: no `done` filter and no
    level gating.
  - **PRIVILEGED MEANS "ALL AGENTS, RIGHT NOW" — IT DOES NOT MEAN "THE ANSWER".** The
    critic is deliberately NOT given `oracle_solution` / `oracle_tasks` / `U_oracle` / any
    reward component, the episode seed, the scheduled fuel-damage severity or condition
    label, the known-vs-hidden split, future RNG, or any future outcome. **Do not add a
    feature this list does not name.**
  - **SIZE IS VARIABLE, WITH ONE FLOOR.** There is no padding and no fixed cardinality:
    the encoder is size-agnostic and its self-loops keep an empty edge set safe. `k` (live
    targets) MAY legitimately be **0** — every target destroyed is a normal late-episode
    state. The live-agent count is likewise variable, but **at an ACTUAL DECISION CAPTURE it
    is at least 1**: a decision requires an airborne ego, so that ego always has a node.
    `CentralCritic.forward` does carry an all-empty `n_nodes == 0` guard returning a finite
    zero, but that branch is DEFENSIVE — it makes the output finite by construction rather
    than by an argument about the caller, and it is **not a reachable normal decision
    state**.
- **MULTI-AGENT TEMPORAL SEMANTICS — ONE CENTRAL STATE PER ACTUAL DECISION.**
  `run_episode(..., central=CentralStateRecorder())` calls `capture` INSIDE the `if wake`
  branch and IMMEDIATELY BEFORE `_wake_decision`, and nowhere else — so sample `i` is the
  global state the team was in when decision `i` was made, BEFORE that decision changed
  anything, and `recorder.samples` is aligned 1:1 and index-for-index with
  `EpisodeResult.trajectory`. `CTDEEpisodeRecord` VALIDATES that alignment on construction,
  so a drifted capture seam fails LOUD rather than mispairing a value with a decision.
  **WAKE ORDERING IS STILL SEQUENTIAL, NOT A JOINT SAME-TICK ACTION.** With two egos waking
  on one tick the order is `capture(A) → act(A)+resync(A) → capture(B) → act(B)+resync(B)
  → env.step`: no `env.step` between them, so B's PHYSICAL world equals A's, while B's
  central `assigned` feature legitimately reflects A's already-applied resync. That is
  CAUSAL, not a leak — the critic is centralized by design, and B still DECIDES from its
  own private observation alone. The `central` parameter defaults to `None`, which leaves
  the loop byte-identical to its pre-CTDE behaviour.
- **CREDIT PATH: GAE OVER THE GLOBAL DECISION SEQUENCE.** `compute_ctde_advantages` is the
  CTDE REPLACEMENT for `compute_returns_and_advantages`; it does not call it, and it never
  runs on an `actor_only` run. `V_old` is evaluated ONCE for every sample under
  `torch.no_grad` BEFORE epoch 0 and stays fixed for the whole update, so the regression
  target cannot chase the network fitting it. `compute_gae` runs PER EPISODE over the
  episode's SINGLE ordered decision sequence — **deliberately NOT regrouped per ego**, which
  is what `EpisodeRecord` does for the Phase-A per-ego credit structure — with
  `delta_t = r_t + gamma*V_next[t] - V_old[t]`, `A_t = delta_t + gamma*gae_lambda*A_{t+1}`,
  `target_t = A_t + V_old[t]`, and **`V_next` of the LAST decision is ZERO** (the episode
  genuinely ends there). Per-decision rewards are READ off the transitions
  (`episode_rewards_sequence`), so the credit math consumes exactly what the unchanged
  terminal reward layer produced. Advantages are normalized across ALL decision samples of
  the batch under the same `adv_norm_eps` guard the actor-only path uses, and the actor
  consumes them DETACHED. The critic takes an MSE value loss scaled by `value_coeff`; there
  is no value clipping in v1. `gamma` comes from `PPOConfig` — deliberately NOT duplicated
  on `CTDEConfig`, so a run has ONE discount factor. **Current defaults, from the code:**
  `critic_lr = 3e-4`, `value_coeff = 0.5`, `gae_lambda = 0.95`.
- **ZERO-WAKE EPISODES.** A zero-wake episode contributes NO actor sample, NO critic sample
  and NO baseline mass to a CTDE update. It remains a **valid scientific episode outcome**,
  never a failure, and keeps its existing reward-diagnostic accounting — a batch with zero
  decisions is the same clean no-op `PPOUpdater` documents, reported with
  `n_epochs_run == 0`.
- **`baseline` KEEPS ITS ACTOR-ONLY MEANING IN BOTH MODES, AND THIS IS LOAD-BEARING.**
  `CTDEUpdater.update` reports `baseline` as the batch's mean EPISODE REWARD (zero-wake
  episodes included) — NOT the critic's mean value, even though the CTDE baseline really is
  the critic. `graph_train` records that key as an iteration's `train_reward_mean`, so
  putting a value estimate there would make one recorded field mean a reward under
  `actor_only` and a value under `ctde`, and the two modes' learning curves would stop being
  comparable while still looking as though they were. The critic's own estimate is reported
  SEPARATELY. A CTDE training record additionally persists the CRITIC's four diagnostics —
  `value_loss`, `value_mean`, `value_target_mean`, `critic_grad_norm` — copied straight out
  of the dict `CTDEUpdater.update` returned and NEVER recomputed; they are added ONLY when
  `ctde_enabled`, so an `actor_only` record is byte-unchanged with those keys ABSENT rather
  than null (a nullable key would invite reading "no critic" as "a critic that scored 0").
  `run_config.json` carries a `training` block: `mode`, `ctde_enabled`, and the resolved
  `ctde` config or `null`.
- **CHECKPOINTS.** `save_checkpoint(policy, updater, iteration, ckpt_dir, critic=None)`.
  **THE ACTOR-ONLY PAYLOAD IS UNCHANGED** — with `critic is None` (every `actor_only` run)
  it holds EXACTLY the five keys it always held (`iteration` / `encoder` / `head` /
  `optimizer` / `ppo_config`), nothing renamed and nothing added, not even a mode label, so
  a Phase-A checkpoint stays readable by anything that could read one. A CTDE run saves
  strictly MORE: the same five keys (`encoder` / `head` / `optimizer` are the ACTOR's) plus
  `training_mode`, `critic_encoder`, `value_head`, `critic_optimizer` and `ctde_config`.
  There is deliberately NO second "actor export" file — the actor portion of the one payload
  already suffices for later inference, precisely because the actor's keys did not move.
  **There is NO loader and NO resume**, in either mode; restoring a run remains a separate
  deferred task, and no export functionality beyond the above exists.
- **PRESETS.** A preset may set `training_mode` and a nested `"ctde"` block (the sibling of
  `"ppo"`), read only by a `ctde` run. The CTDE block has NO CLI flags of its own — it is
  deliberately a preset-only layer, so there is no second naming scheme to drift from
  `CTDEConfig`. **No CTDE preset exists in the repository**, and adding one belongs to the
  comparison task (§8), not here.
- **SCIENTIFIC NON-CLAIMS, BINDING.** The proof tests, the module `_selftest`s and a passing
  suite are ENGINEERING evidence and measure nothing scientific. **No actor-only vs CTDE
  comparison has been run, and no CTDE benefit — in reward, survival, sample efficiency,
  behavioural separation or anything else — is established or may be pre-claimed.** A CTDE
  claim requires its own executed, independently reviewed comparison under the same validity
  gate (§8).

**SCHEDULED CELL vs EXECUTED CELL — a measurement-integrity abort
(`_ConditionTally.success`, the approved review fix `eecc9b5`).**
`success(out, *, expected_cell)` takes the SCHEDULE's cell as a **REQUIRED keyword** and
requires `executed_cell == expected_cell` — **equality, not membership**. Membership alone
cannot see the fault: under FD-VARIABLE-SEVERITY-v1 a scheduled `mild` that executed as
`severe` names a cell the run legitimately reports, so a membership test ACCEPTS it and
books the ATTEMPT in one cell's denominator and the REWARD in another. **That corrupts
BOTH denominators at once** — the scheduled cell reads as a failure that never happened,
the executed cell as a success that was never scheduled — and a triad's within-seed delta
would be taken between two members the schedule never paired. The keyword is required
deliberately: an optional one would let a future call site skip the check by omission.
Three disjoint faults are named separately (the SCHEDULE names an unreported cell; the
EXECUTION reports an unreported cell; both reportable but DISAGREEING), all are
`MeasurementIntegrityError`, and every check runs BEFORE any state is mutated, so a
rejected episode leaves the tally byte-unchanged. **BOTH production call sites pass their
scheduled cell and the guard runs FIRST**, so a mismatched episode reaches NEITHER the
per-cell counters and rewards, NOR a matched-group member reward or delta, NOR
`episode_outcomes.jsonl`, NOR the PPO buffer. It ABORTS the run as INFRASTRUCTURE exactly
as a roster fault does — never an accounted scientific episode failure. **This has NOT
been observed in the real simulator**: the regression test INJECTS the divergence through
a stub, because normal production does not currently generate it.

---

## 6. File map — "I want to…"

| … | Go to |
|---|---|
| Change episode setup / solve+normalize / Belief | `rl/training/graph_episode_setup.py`, `rl/training/belief.py` |
| Ask "which targets does this episode CONTAIN?" (world inventory, NOT allocation) | `rl/training/graph_episode_setup.py` (`EpisodeContext.known_target_ids` / `executed_target_ids`, `_world_target_ids`, and `_finish_context`'s required-keyword non-empty + subset verification). Both are RAW snapshots taken BEFORE their solve. **Never** answer it from `oracle_tasks`, `belief_tasks` or the beliefs — `solve_and_normalize` is allocated-only by contract, so those omit every unselected target. See the §5 roster-integrity contract |
| Change the CONSTRUCTION seam (solve → place → patch → reload) | `rl/training/graph_episode_setup.py` → `_setup_episode_construction`, plus its helpers `_resolve_construction_mode`, `_shared_launch_point`, `_require_airbase_only_targets`, `_select_hidden_prototype`, `build_patched_scenario`, `_require_agent_ids_preserved`, `_rematerialize_known_tasks`, `_build_env` / `_extract_world` / `_close_quietly` / `_finish_context` |
| Change the LEGACY split path (retained, not deleted) | `rl/training/graph_episode_setup.py` → `_setup_episode_legacy`, `split_tasks` |
| Change the tick-loop / policy bundle / rollout | `rl/training/graph_tick_loop.py` |
| Run a diagnostic rollout (no training) | `rl/training/graph_rollout.py` (`RolloutConfig`, `run_rollout`) |
| Run PPO training / plot a run | `rl/training/graph_train.py` (`TrainConfig`, `train`, `plot_training`). A run writes `run_config.json` (+ `provenance` + `config_source`), `train_records.jsonl`, `eval_records.jsonl`, `episode_failures.jsonl`, `run_summary.json`, `scenarios/`, `checkpoints/` and the three figures under `plots/`. **`train` refuses to start unless Git provenance is COMPLETE** (full SHA + clean/dirty verdict) — see the §5 trainer contract; `collect_provenance` / `_git_provenance` / `_iteration_outcome` / `build_run_summary` / `eval_episode_tag` / `_format_episode_block` / `_unique_confirmed_target_ids` / `_episode_target_roster` |
| Change how a run FAILS on a measurement/data-integrity fault (as opposed to an episode fault) | `rl/training/graph_train.py` (`MeasurementIntegrityError`, its subclass `EpisodeRosterError`, `_world_snapshot_ids`, `_episode_target_roster`, `_require_scheduled_cell`, and the `except (_VisualArtifactError, MeasurementIntegrityError)` re-raises in the train and eval attempt handlers) — plus `_ConditionTally.attempt` and `_ConditionTally.success(out, *, expected_cell)`, whose scheduled-vs-executed CELL equality check is routed identically (§5). It ABORTS the run and is NEVER written to `episode_failures.jsonl`, counted against a condition, folded into a matched group, appended to `episode_outcomes.jsonl`, added to the PPO buffer, or entered into `skip_and_account_v1` — that routing is the §5 roster-integrity contract and deliberately reverses PR #7's |
| Configure a run from a FILE, or add a preset | `configs/graph_train/final_cell_probe.json` (the ONLY repository preset: the bounded short probe) + `rl/training/graph_train.py` (`--config`, `load_config_file`, `resolve_train_config`, `_effective_argv`, `_explicit_cli_dests`, `_CLI_FIELD_BY_DEST` / `_CLI_PPO_FIELD_BY_DEST`). Presets name `TrainConfig` FIELDS; precedence is defaults < preset < explicitly typed flags. See the §5 harness contract |
| Change what a run RECORDS about where its config came from | `rl/training/graph_train.py` (`config_source_record`, `_CONFIG_SOURCE_KINDS` = `config_file` / `cli_defaults` / `direct_config`, `write_run_config`). Always a structured object, never `null`; `resolved_from` is required, never inferred |
| Change a FIGURE (or add one) | `rl/training/graph_train.py` (`plot_training`, `_plots_dir`, `_plot_training_performance`, `_plot_policy_diagnostics`, `_plot_measurement_health`, `_PLOT_FILENAMES`, `_PLOT_X_LABEL` / `_PLOT_X_SEMANTICS`, `_xy`, `plot_training_subprocess`). Figures go to `<run_dir>/plots/`; the two presentation invariants in §5 (condition means vs complete-pair delta, and the honest x-axis) are contractual |
| Change the training scenario cell (target counts) | `rl/training/graph_train.py` (`TrainConfig.num_agents` / `n_known` / `n_hidden` / `min_target_distance_km` / `min_known_separation_km`, `build_variation_config`); mirrored field-for-field on `rl/training/graph_rollout.py` (`RolloutConfig`). The generator writes `n_known`; setup patches in `n_hidden`, so **emitted targets are `n_known + n_hidden`** (`TrainConfig.n_targets_emitted`). Legacy `num_red_airbases` / `partial_ratio` / `derived_split` / `split_preview` survive and are still tested but are NOT consulted by the construction path (B1, `d6758ac`). |
| Place hidden targets along a predicted ego route (PURE geometry — no BLADE / torch / solver / setup import) | `rl/training/graph_hidden_placement.py` (`PlacementParameters`, `HiddenPlacement`, `predict_route`, `place_hidden_targets`, `validate_placement`, `geometric_fingerprint`). CONSUMED by construction-mode `setup_episode` (B3, `dd14ab4`); the import direction is one-way — this layer must never import `graph_episode_setup`. `predict_route` imports `nearest_neighbor_order` from `utils/scheduling_utils.py`, NOT from any executor module. |
| Change the SHARED intra-level nearest-neighbor ordering (route prediction + execution at once) | `utils/scheduling_utils.py` (`nearest_neighbor_order`). ONE implementation with TWO consumers — `blade_graph_executor.GraphPlanExecutor._eligible` and `graph_hidden_placement.predict_route`. Changing it changes BOTH; that shared identity is the route-fidelity invariant (`2a3f89c`). Pinned by `tests/test_graph_executor_nn_ordering.py`. |
| Change the LEGACY FD-BASELINE-v1 MECHANISM (rng domain, window, event, live re-validation, RTB measurement) — the PRESERVED Phase-A semantics | `rl/training/graph_fuel_damage.py` (`FuelDamageMode`, `FuelDamageParameters`, `FuelDamagePlan`, `FuelDamageOutcome`, `FuelDamageController.maybe_apply` / `live_bounds` / `note_commands` / `note_wake`, `measure_window`, `plan_fuel_damage`, `build_fuel_damage_plan` / `build_fuel_damage_controller`, `derive_fuel_damage_seed`, `resolve_condition`, `fuel_for_distance_km`, `rtb_command_for`). PURE — no BLADE / gym / torch / solver import; must never import `graph_episode_setup`. Injected into the tick via `run_episode(..., fuel_damage=...)`. **The approved Phase-A measurement lives on these modes — do not move them; the mild/severe extension has its own row below.** |
| Change the FD-VARIABLE-SEVERITY-v1 MECHANISM (severity draw, the two live bands, the live-midpoint target) | `rl/training/graph_fuel_damage.py` (`FuelDamageMode.VARIABLE` = `seeded_variable` / `forced_mild` / `forced_severe`, `SEVERITY_MILD` / `SEVERITY_SEVERE` / `SEVERITIES`, `FUEL_DAMAGE_SEVERITY_RNG_DOMAIN`, `derive_fuel_damage_severity_seed`, `resolve_severity`, `FuelDamageParameters.mild_probability` / `variable_severity` / `target_policy`, `TARGET_POLICY_LIVE_SEVERITY_MIDPOINT`, `severity_band` / `_SeverityBand` / `_require_valid_band`, and `FuelDamageController._live_variable_target` beside the untouched `_live_legacy_target`). The severity domain is SEPARATE from `fuel_damage_v1` on purpose (§5) — merging them would move the ego every damaged episode selects and invalidate the approved Phase-A baseline. Same PURITY rules as the row above. |
| Change the FD training MIXTURE / matched EVALUATION / FD reporting | `rl/training/graph_train.py` (`TrainConfig.fuel_damage_mode` / `fuel_damage_probability` / `fuel_damage_mild_probability` / `fuel_damage_leg_progress` / `fuel_damage_rtb_margin` / `aircraft_penalty_coeff`, `fuel_damage_parameters()`, `reward_config()`, `_run_one_episode(..., fuel_damage_mode=...)`, `evaluate` matched groups, `eval_member_tag`, `_ConditionTally`, `_fuel_damage_lines`, `build_run_summary`). `RewardConfig(aircraft_penalty_coeff=2.25)` is passed explicitly here; `graph_reward` stays frozen. |
| Change the matched CLEAN/MILD/SEVERE TRIAD evaluation, or a within-seed DELTA | `rl/training/graph_train.py` (`_EVAL_TRIAD_MEMBERS`, `_EVAL_TRIAD_DELTAS` beside the unchanged `_EVAL_PAIR_MEMBERS` / `_EVAL_PAIR_DELTAS`, `_EVAL_GROUP_KIND_PAIR` / `_EVAL_GROUP_KIND_TRIAD`, `TrainConfig.variable_severity` / `eval_group_members` / `eval_group_size` / `eval_group_kind` / `eval_group_deltas` / `reported_cells`, `_scheduled_cell_probabilities`, `_difficulty_factor_name`, and `evaluate`'s complete-group test). A legacy run keeps its PAIR; only a `seeded_variable` run evaluates triads. **Every delta is over COMPLETE groups only** — see §5. |
| Read what an episode ACTUALLY did, per successful attempt (not an aggregate) | `rl/training/graph_train.py` (`_EPISODE_OUTCOMES_FILENAME` = `episode_outcomes.jsonl`, `_episode_outcome_record`, `_append_episode_outcome_record`, `_severity_response_from_outcomes` and the `severity_response` / `severity_response_source` / `episode_outcomes_recorded` keys of `run_summary.json`). SUCCESSFUL attempts only — failures stay in `episode_failures.jsonl` and the two streams are disjoint by construction. The severity-response table is DERIVED from this file, never from a parallel in-memory aggregate. |
| Keep the DIAGNOSTIC harness at configuration parity with training | `rl/training/graph_rollout.py` (`RolloutConfig` mirrors the FD knobs field-for-field + `fuel_damage_parameters()` / `reward_config()`; `run_rollout` builds the controller and passes the same explicit `RewardConfig`; `fuel_damage_mild_probability` mirrors the training knob and `seeded_variable` is selectable here too). Rollouts run a SEEDED design only — `seeded_mixture` or `seeded_variable` — because matched pairs and triads are an evaluation construct and live in `graph_train.evaluate`. |
| Capture per-attempt VISUAL ARTIFACTS (known-only scenario + executed t=0 scenario + BLADE playback + manifest) | `rl/training/graph_train.py` (`TrainConfig.visual_artifacts` and the `--visual-artifacts` flag, `_AttemptIdentity`, `_AttemptArtifacts` with `open` / `capture_known_only_scenario` / `capture_executed_t0_scenario` / `sync_recordings` / `finalize` (which reconciles expected vs observed world counts before it will say `complete`) / `to_manifest`, `_VisualArtifactError`, `_recording_kwargs`, `_artifact_kwargs`; consumed by `_run_one_episode(..., artifacts=...)` and wired from `train` / `evaluate(..., artifacts_root=...)`). OFF by default and OFF is byte-unchanged — see the §5 trainer contract. `graph_tick_loop`, `graph_episode_setup`, `PlaybackRecorder.py` and `Game.py` are NOT touched; recording is armed only through `setup_episode(recording_export_path=...)`. |
| SELECT a training mode — ordinary scientific USE of the already-built CTDE layer | `rl/training/graph_train.py` (`TrainConfig.training_mode` ∈ `TRAINING_MODES` = `actor_only` / `ctde`, `TrainConfig.ctde_enabled`, the nested `ctde` preset block over `CTDEConfig`). Choosing a mode, writing a preset that sets it, or running a comparison is **CONFIGURATION and MEASUREMENT, not a contract change** — it needs no layer review. The DEFAULT is `actor_only`, and it is the path the approved Phase-A baseline was measured on. `value_coeff` is NOT a mode selector: `ctde` REJECTS `value_coeff <= 0` (§5) |
| Change the CENTRAL GRAPH the critic sees (privileged inputs, liveness, features, edges, exclusions) | `rl/observation/central_graph_builder.py` (`CentralGraphObservation`, `build_central_graph_observation`, `CentralStateRecorder`, `live_aircraft`, `plan_target_ids`, `NO_EGO_INDEX`, `CENTRAL_TASK_FEATURE_DIM` / `CENTRAL_AGENT_FEATURE_DIM` / `CENTRAL_EDGE_ATTR_DIM` / `CENTRAL_EDGE_TYPE`). **RESEARCH-VALIDITY / GRADE A**: what the critic may read is the no-communication boundary itself. Adding any input the §5 exclusion list names — `oracle_solution` / `oracle_tasks` / `U_oracle` / a reward component / the seed / a scheduled FD severity or condition label / the known-vs-hidden split / future RNG or outcome — is a new research decision, never a fix. PURE: no torch, no BLADE/gym import; it must never import `graph_episode_setup` |
| Change the ACTOR / CRITIC BOUNDARY, or CTDE value / GAE semantics | `rl/training/graph_ppo.py` (`CTDEConfig`, `ValueHead`, `CentralCritic`, `build_central_critic`, `CTDEEpisodeRecord`, `CTDEBuffer`, `compute_gae`, `compute_ctde_advantages`, `CTDEUpdater`, `episode_rewards_sequence`) beside the UNTOUCHED actor-only `EpisodeRecord` / `PPOBuffer` / `compute_returns_and_advantages` / `PPOUpdater`. **RESEARCH-VALIDITY / GRADE A**: disjoint parameter sets, two separate backwards, detached advantages, GAE over the GLOBAL decision sequence with a zero terminal next value, and fixed pre-epoch `V_old` are all contract (§5). Proofs live in `tests/test_graph_ctde.py` and `tests/test_graph_ppo.py` |
| Change WHEN the central state is CAPTURED | `rl/training/graph_tick_loop.py` — `run_episode`'s `central` parameter and the `capture(...)` call inside the `if wake` branch, IMMEDIATELY BEFORE `_wake_decision`. **RESEARCH-VALIDITY / GRADE A**: the capture point IS the 1:1 alignment `CTDEEpisodeRecord` validates, and moving it silently repairs the mispairing into a wrong value-to-decision match. `central=None` (the default) leaves the loop byte-unchanged |
| Change ACTOR-ONLY PRESERVATION or CHECKPOINT compatibility | `rl/training/graph_train.py` (`_ctde_kwargs` / `_central_kwargs` — keyword OMISSION, never a `None` keyword; `save_checkpoint(..., critic=None)`'s exactly-five-key actor-only payload and the CTDE additions; the `ctde_enabled`-gated critic diagnostics on a training record; the `training` block of `run_config.json`). **RESEARCH-VALIDITY / GRADE A**: `actor_only` byte-invariance is what keeps the approved Phase-A baseline comparable, and it is pinned by the POISON test + its CONTROL in `tests/test_graph_ctde.py` |
| Change the reward | `rl/training/graph_reward.py` (`compute_episode_reward`/`plan_value`/`realized_utility`/`RewardConfig`) |
| Change WHEN the policy wakes | `rl/action/graph_trigger.py` (`decide_triggers`, `TriggerKind`, `never_overdue`) |
| Change the graph representation | `rl/observation/graph_builder.py` (`GraphObservation`, `GraphObservationConfig`, `EdgeType`, `TASK_FEATURE_DIM`) |
| Change the encoder | `rl/agent/graph_encoder.py` (`GraphEncoder`, `pool()` critic hook). ONE class with TWO instantiations — the ACTOR's, and the Phase-B `CentralCritic`'s separate instance at the CENTRAL feature widths. Changing it changes BOTH; the actor head carries no value head, and the critic's `ValueHead` lives in `graph_ppo`. |
| Change actions / mask / sampling | `rl/action/graph_action.py` (`MetaAction`, `ActionHead`, `build_action_mask`, `sample_action`) |
| Change how a decision edits the plan | `rl/action/graph_effect.py` (`apply_meta_action`) |
| Change BLADE execution / plan re-sync | `utils/blade_utils/blade_graph_executor.py` (`GraphPlanExecutor`) |
| Change the ATTACK-CONFIRMATION WAIT (how long an ego holds before re-firing) | `utils/blade_utils/blade_graph_executor.py` — `_salvo_travel_ticks` (the conservative travel BOUND + the transcribed `KILOMETERS_TO_NAUTICAL_MILES`), `_confirmation_wait_ticks` (live-weapon selection + the `max(kill_confirm_ticks, bound + 1)` floor) and the ATTACK BRANCH of `_command_for_ego` that arms the per-`(ego_id, target_id)` cooldown. Proofs — pure tiers and the real-BLADE tier (incl. the engine-constant comparison) — live in `tests/test_graph_setup_seam.py`. **The vendored engine stays FROZEN** (§2): the wait is derived from what BLADE already exposes, never by editing it. |
| Change RTB / EPISODE-COMPLETION semantics (when an episode is allowed to end) | `utils/blade_utils/blade_graph_executor.py` — `is_done(observation)` (the two-half verdict), `_physical_state` (the ONE airborne / landed / removed classification site) and `_note_dead` (idempotent death reconciliation) — plus the returning-ego Phase-1 guard in `rl/training/graph_tick_loop.py` (`run_episode`, the `rtb_issued` skip) and the post-step `is_done(obs)` call site. Proofs: the pure lifecycle tier and the real-BLADE `P7` ride-home tier in `tests/test_graph_setup_seam.py`, and the returning-ego `POC-1..4` tier in `tests/test_graph_fuel_damage.py`. **The vendored BLADE engine stays FROZEN** (§2): completion is decided from states the engine already exposes, never by editing it. |
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
  *(That is the Phase-A actor-only state this commit built, and it is still exactly what
  `PPOUpdater` does. The critic joined LATER, as a SEPARATE `CTDEUpdater` beside it —
  see the Phase-B CTDE lock at the end of §7.)*
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
  roster defects were routed to accounted `setup` failures instead of false successful
  zeros — **that ROUTING is SUPERSEDED: since `36365f2` such a fault is a
  `MeasurementIntegrityError` and ABORTS the run**, and the false-successful-zero fix this
  PR made is unaffected; and each eval round keeps a disjoint scenario-tag namespace while
  reusing the same held-out seeds. Grade A under `GPT_GITHUB`: candidate
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

- `52064c2` — **Repository / documentation hygiene — CLOSED / MERGED / APPROVED.**
  Approved candidate `52064c2d306df7c8447d159df20e6e189a59bf85`, integrated by
  `5f78904e3af1e2e47386c9b0e01ddbaa273724f5` (PR #12); the approved candidate tree was
  verified identical to the integration tree. Grade C under `GPT_GITHUB`, implementation
  mode SURGICAL. The FIRST candidate `6302847bdc8b5e40313763b4b167af85dd0a462e` received
  REQUEST-FIXES on two documentation-correctness findings — a backwards
  `Scenario.is_hostile` claim and stale volatile handoff state — and the correction landed
  as a NEW CHILD COMMIT on the same branch and PR, never amended, rebased or force-pushed.
  Scope: `README.md` replaced, `docs/BLADE_API_DOCUMENTATION.md` audited against the
  vendored fork, four obsolete scenario JSONs deleted, two dead utility symbols removed,
  the stale `requirements.txt` comment corrected, and `CLAUDE.md` / the handoff aligned —
  the §8 note below states the detail and is NOT repeated here. Verified: import purity
  **12 passed**, full suite **216 passed, 4 skipped** (unchanged from the base, since no
  runtime code changed), `git diff --check` clean. **NO scientific run, BONMIN solve,
  rollout, probe or artifact generation was performed.** After the merge the obsolete
  branches `task/repo-code-hygiene` and `task/repo-doc-hygiene` were deleted (safe
  deletion only, both already ancestors of `main`); `flat-final` and `pre-cleanup` were
  untouched, and every reviewed candidate tip remains reachable on GitHub through
  `refs/pull/<n>/head`.

- `61e539e` — **FINAL-CELL PROBE HARNESS: JSON presets, run layout and three semantic
  figures — CLOSED / MERGED / APPROVED.** Reviewed code SHA
  `61e539ed62fcf1e3fe25a83d213cae06f5afa98e`, integrated by merge commit
  `a5f389a2af328640e19db51d3277a33167c08f25` (PR #14); the merged tree is byte-identical
  to the approved one (`git diff` between the two reports zero changed files). Grade A
  under `GPT_GITHUB`, implementation mode SURGICAL — **the grade was corrected from B to A
  during review**, because `graph_train.py` is part of the §5 locked trainer contract; no
  implementation redo was required, since the strongest reasoning model, three proof
  obligations, the exact-SHA branch workflow and the broad test set were already in place.
  The full technical contract is in §5 ("Experiment harness") and the routing in §6; this
  entry records the LOCK, not the mechanism.
  **Reviewed scope: FOUR files** — `src/match_aou/rl/training/graph_train.py`,
  `tests/test_graph_train.py`, `README.md`, and the new
  `configs/graph_train/final_cell_probe.json`. No solver, BLADE, reward, fuel-damage,
  tick-loop, PPO, episode-setup, scenario-construction, seed-schedule, matched-pair
  evaluation or visual-artifact semantics were touched, and no evaluation RECORD FIELD was
  added, removed or redefined — the figures read fields that already existed.
  **Fix chain: TWO REQUEST-FIXES rounds, each landing as a NEW CHILD COMMIT on the same
  branch and PR** — never amended, rebased or force-pushed. Candidate
  `4238e0ee79faf3c1bde414fa041d410e44c07b38` → `de51883f20f28aadb4e6a9fa2a6f679a9eaded2f`
  → the approved `61e539e`. The five findings and their closure:
  (F1) `_explicit_cli_dests` read `argv=None` as an EMPTY command line while `argparse`
  reads it as `sys.argv[1:]`; since `main()` is normally called with no argument, every
  flag an operator really typed looked un-typed and a preset could silently override it.
  Both passes now consume one `_effective_argv` vector.
  (F2) `config_source` had two contradictory contracts (structured for CLI-only runs in
  code, `null` in the docs). Settled on ALWAYS-structured, one schema, one helper.
  (F3) the two held-out condition means are each over their own successful subset, so
  their gap is not a within-seed comparison; `measurement_health.png` gained PER-CONDITION
  completion counts and the performance panel's title and legends now say what each mean
  is over. Evaluation semantics and the matched-pair computation were NOT changed.
  (F4) the `config_source` fallback INFERRED `cli_defaults` from the absence of a path,
  which mislabelled every direct `train(cfg)` call — `_selftest` included — as
  CLI-resolved; `resolved_from` became a required argument and `direct_config` a third
  truthful kind.
  (F5) the preset's own prose promised a post-update round "after both updates", which the
  schedule cannot guarantee; it now says both SCHEDULED ITERATIONS and states that
  `updates_completed` may be 0, 1 or 2. PROSE ONLY — the schedule fields are byte-unchanged
  and test-pinned.
  **Verified at the approved head:** `tests/test_graph_train.py` **108 passed** (89 → 108),
  import purity **12 passed**, full suite **235 passed, 4 skipped** (216 → 235), all 108
  green through the standalone `__main__` runner under `nlp_env`, and `git diff --check`
  clean. Five mutation checks confirmed the load-bearing tests falsify (the `argv=None`
  reading; the unguarded `config_source`; the blanked per-condition series; the
  `cli_defaults` fallback; the restored "after both updates" prose) — each was caught and
  then reverted. Figures were rendered from SYNTHETIC records through the real `--plot`
  CLI and inspected.
  **NO BONMIN solve, BLADE episode, training run, rollout, selftest, probe or baseline was
  executed** — every test is solver-free and drives the pipeline through stubbed engine
  seams. **This lock certifies the HARNESS, not the cell**: no reward improvement and no
  fuel-damage behaviour has been measured on it. §8 still owns the gate.

- `d56fda6` — **DEFECT A: ego-global `SELF_PRESERVATION_ABORT` — CLOSED / MERGED /
  APPROVED.** Approved candidate SHA `d56fda636ab5ec1a5cce6076f07acac5556d10cb`,
  integrated by merge commit `f094e0b32e5e67b79757edbfe4e73c1fe01b0a87` (PR #17). The
  candidate was merged with a MERGE COMMIT and preserved as its SECOND PARENT; candidate
  and integration share the identical tree `70e5af2446f0a1b0674eb10819c9451753260560`, and
  the candidate→integration comparison contains ZERO changed files. Grade A under
  `GPT_GITHUB`, implementation mode SURGICAL. The technical contract is in §5 (Stage 4
  SELECTION, Stage 5 EFFECT); this entry records the LOCK, not the mechanism.
  **THE DEFECT.** The first executed bounded short probe (`training_output_20260815_173029`,
  from `238062d7d284334432d9c39d7543fb0bbf39ea7c`) showed `apply_meta_action` removing only
  the assignments whose `task_idx == node_v`, so SPA aborted ONE TASK rather than the ego's
  MISSION — playback showed a fuel-damaged KC-135 selecting SPA while its BLADE route
  continued and further assignments remained. The approved behaviour is an EGO-GLOBAL
  mission abort, reaching the ALREADY-EXISTING wake → resync → empty-plan → single-latched-RTB
  path. **The `k × 3` action head was NOT redesigned.**
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR.** The first candidate
  `c306455085de408c7bf383135c27e600ff3f1428` received REQUEST-FIXES for THREE
  documentation inaccuracies — a comment claiming the RTB is issued "on the next tick"
  (it is issued on the next `GraphPlanExecutor.next_actions` call, which is Phase 2 of the
  SAME tick), a stale `graph_fuel_damage` docstring still saying SPA would "drop the
  assignment", and a `MetaAction` docstring wrongly grouping `PLAN_COMPLIANCE` with
  `OPPORTUNISTIC_ENGAGEMENT` as acting on the selected node. The correction landed as a NEW
  CHILD COMMIT `d56fda6` — never amend, rebase, squash, force-push or history rewrite —
  and its non-docstring/non-comment token stream was verified identical to `c306455`.
  **CUMULATIVE SCOPE: EXACTLY FIVE FILES** — `src/match_aou/rl/action/graph_effect.py`
  (the sole runtime change), `src/match_aou/rl/action/graph_action.py` and
  `src/match_aou/rl/training/graph_fuel_damage.py` (both DOCUMENTATION-ONLY: their
  docstring text necessarily CHANGED, so their complete token streams are NOT identical —
  what was verified identical to the base is their NON-DOCSTRING / NON-COMMENT,
  runtime-relevant token stream), plus `tests/test_graph_fuel_damage.py` and
  `tests/test_graph_setup_seam.py`. No BLADE, executor, tick-loop, PPO, encoder, reward,
  solver, generator, scenario, seed-schedule, fuel-damage-mechanism, trainer, rollout,
  preset or artifact file was touched.
  **PROOF OBLIGATIONS.** PO1 — ego-global effect and private isolation: a multi-assignment
  actor across BOTH of its legal abort cells yields the identical empty slice, every peer
  slice is value-unchanged, the input `solution` and `tasks` are unmutated, out-of-range
  `node_v` still raises, and the real builder + real `build_action_mask` confirm the `k × 3`
  shape with abort legal on exactly the ego's own assigned nodes. PO2 — the REAL
  `graph_tick_loop._wake_decision` chain (real builder, mask, `sample_action`,
  `apply_meta_action` and `GraphPlanExecutor.resync`; only encoder/head stubbed to force a
  deterministic cell): before Phase 2 the actor's belief slice and executor plan are both
  empty while every peer is unchanged, then exactly ONE `aircraft_return_to_base`, no stale
  move/attack for that ego, and no second RTB toggle. PO3 — a solver-free REAL-BLADE tier:
  a real launched aircraft flying a real executor-issued mission route has `rtb` set, the
  stale waypoint removed and a route ending at its ACTUAL home base, with no second
  executor RTB; `Game.py` is byte-unchanged.
  **VERIFIED at the approved head:** full base suite **238 passed, 4 skipped** (235 → 238);
  focused base pytest (fuel damage, setup seam, action evaluate, import purity) **70 passed,
  4 skipped**; `tests/test_graph_fuel_damage.py` standalone `nlp_env` runner **37 passed**;
  `tests/test_graph_setup_seam.py` standalone `nlp_env` runner **20 passed, 0 skipped**,
  including the real-BLADE + BONMIN solver tier with no `CRASH`/`Traceback`; both
  action-layer selftests green under `nlp_env`; `git diff --check` clean. Falsifiability was
  demonstrated: with the old node-filtered body temporarily restored all four regressions
  fail and the `graph_effect` selftest fails at case (3); the mutation was reverted
  byte-identically and is not in the history.
  **NO scientific probe, training run, rollout or baseline was executed.** **This closes
  DEFECT A ONLY — as of THIS lock, Defects B and C both remained OPEN. Both have since
  been closed — B by `39a16f2` and C by `ea62e4e`, below; §8 owns the current state.**

- `39a16f2` — **DEFECT B: the attack-confirmation wait DERIVED from the salvo about to fly
  — CLOSED / APPROVED / MERGED.** Approved candidate SHA
  `39a16f2e5e1a3302d545c11b072e037e9702dffe`, integrated by merge commit
  `60a82d17398e9d14be1c2684cc72fafd020e0d9b` (PR #19). The candidate was merged with a
  MERGE COMMIT and preserved as its SECOND PARENT; candidate and integration share the
  IDENTICAL tree `ee86f0782ac50ee8bd0ee2fe634393a9cfc53a66` (verified locally), and the
  candidate→integration comparison contains ZERO changed files. Implementation fixed base
  `cefda78b18ea2daeda5014bab9a75a0945ef8e37`. Grade A under `GPT_GITHUB`, implementation
  mode SURGICAL. The technical contract is in §5 (Execution, Stage 1) and the routing in
  §6; this entry records the LOCK, not the mechanism.
  **THE DEFECT.** `GraphPlanExecutor` armed a FLAT `kill_confirm_ticks` for every salvo
  (default 60, and no caller passed it), so a slower auto-selected weapon could still be
  airborne when the wait expired — the executor then issued a redundant salvo that burned
  the ego's last weapons, measured in the first short probe's `post_update` damaged eval
  seed `1000003` (§8). The approved behaviour DERIVES the wait per salvo from the live
  auto-selected weapon and the current engagement distance, with the configured value kept
  as its FLOOR and FALLBACK. **The default was not merely raised, and frozen BLADE was not
  touched.**
  **CUMULATIVE SCOPE: EXACTLY TWO FILES** —
  `src/match_aou/utils/blade_utils/blade_graph_executor.py` and
  `tests/test_graph_setup_seam.py`. No vendored BLADE, solver, reward, PPO, encoder,
  action-space, tick-loop, trainer, rollout, fuel-damage, scenario, preset or artifact file
  was touched.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR.**
  (1) First candidate `45a0352312ae308df76a506a8e2e9907a9531a43` — the RUNTIME
  IMPLEMENTATION was ACCEPTED; GPT requested corrections because the transcribed
  `KILOMETERS_TO_NAUTICAL_MILES` was only compared against ANOTHER LITERAL rather than the
  engine's own constant, because the continuous-time bound was described inaccurately as
  exact engine ticks, and because the fallback prose contradicted the accepted
  negative-speed `abs` normalization.
  (2) The correction landed as the NEW CHILD COMMIT `39a16f2` — never amend, rebase,
  squash, force-push or history rewrite. Its RUNTIME-RELEVANT executor token stream is
  UNCHANGED from the first candidate; what it added is the real BLADE constant comparison,
  and what it corrected are the bound / timing / fallback claims.
  **PROOF OBLIGATIONS.** PO1 — the derivation really consults the LIVE BLADE selector
  `get_weapon_with_highest_engagement_range()`, the formula, the `max(…, bound + 1)` floor
  and every fallback branch behave as specified, and no peer aircraft, inventory, belief or
  assignment can move the acting ego's wait. PO2 — against the REAL engine: the
  redundant-salvo mechanism is exhibited by the flat-60 control arm and PREVENTED by the
  derived wait, the weapon reserve survives, and the confirmed-kill guard still advances the
  plan on the call the kill becomes visible (earlier than the bound), issuing RTB
  immediately. PO3 — the two-argument attack command, the per-`(ego_id, target_id)` cooldown
  identity, no-comms isolation and every frozen layer are preserved.
  **VERIFIED at the approved head:** full base suite **246 passed, 4 skipped**; focused
  suite **78 passed, 4 skipped**; standalone `tests/test_graph_setup_seam.py` under
  `nlp_env` **28 passed, 0 skipped** — the real-BLADE tier RAN, the engine-constant
  comparison RAN and the BONMIN tier RAN, with no `CRASH` and no `Traceback`; the executor
  selftest green; `git diff --check` clean.
  **NO scientific probe, training run, rollout or baseline was executed.** **This closes
  DEFECT B ONLY — as of THIS lock Defect C remained OPEN. It has since been closed by
  `ea62e4e` below; §8 owns the current state.**

- **DEFECT C: physical RTB completion — CLOSED / APPROVED / MERGED.** Approved candidate
  SHA `ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b`, integrated by merge commit
  `0de9f21eb9e8904f06f836f4ecd010bc46c788b6` (PR #21). The candidate was merged with a
  MERGE COMMIT and preserved as its SECOND PARENT (integration parents, in order:
  `6e97940733d2c7cf8c4ffc7033180c65f644ae17` then `ea62e4e…`); candidate and integration
  share the IDENTICAL tree `6d05cc5ea9af0f6bdcd4a2d6865767bcbe525ebe` (verified locally),
  and the candidate→integration comparison contains ZERO changed files. Implementation
  fixed base `6e97940733d2c7cf8c4ffc7033180c65f644ae17`. Grade A under `GPT_GITHUB`,
  implementation mode BUILD. The technical contract is in §4 (the pipeline's terminal
  loop) and §5 (Execution Stage 1, “COMPLETION IS PHYSICAL, NOT ISSUANCE”, and the
  tick-loop entry); the routing is in §6. This entry records the LOCK, not the mechanism.
  **THE DEFECT.** `GraphPlanExecutor.is_done()` read the `rtb_issued` lifecycle LATCH as
  RTB-resolved and `run_episode` stopped as soon as it became true, so an episode could
  end while the aircraft was still airborne — measured in the first short probe's
  `post_update` damaged eval seed `1000000`, which recorded `dead=0` and reward 0 for an
  ego that could not physically reach home. The approved behaviour separates “RTB command
  ISSUED” from “RTB physically RESOLVED” while PRESERVING the single-issue toggle guard.
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend,
  rebase, squash, force-push or history rewrite. First candidate
  `5a0809df1a490df6ff266343788655d32fcefd81` (parent `6e97940…`) carried the runtime
  correction; the review correction landed as the NEW CHILD COMMIT `ea62e4e…` (parent
  `5a0809d…`), which names `is_done`'s two distinct sources explicitly and proves the
  burn-out branch directly.
  **CUMULATIVE SCOPE: EXACTLY SIX FILES** —
  `src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`, `tools/graph_executor_smoke.py`,
  `tests/test_graph_setup_seam.py` and `tests/test_graph_fuel_damage.py`. No vendored
  BLADE, solver, reward, PPO, encoder, action-space, trainer, rollout, fuel-damage
  mechanism, scenario, preset or artifact file was touched.
  **PROOF OBLIGATIONS.** PO1 — completion is decided from the LIVE post-step observation:
  issuance is not landing, a non-dead ego must be in an airbase inventory, an ego absent
  from both the air and every inventory is reconciled into `dead`, the reconciliation pass
  is total before any verdict, and the `add_return_to_base=False` contract is preserved.
  PO2 — against the REAL engine, the ride home reaches the terminal result: with fuel to
  spare the episode continues past the order and ends only on landing; with half the fuel
  the engine itself says the trip needs, the ego is removed mid-return, counted dead and
  charged by the unchanged reward path. PO3 — a returning ego is frozen out of Phase 1
  (no sensing, trigger, wake, inference, belief edit or transition; its belief is
  byte-frozen from the moment it commits) while peers continue normally, and no peer can
  decide whether the returning ego is home or lost.
  **ACCEPTED EVIDENCE.** Full base suite **257 passed, 4 skipped**; focused suite
  **89 passed, 4 skipped**; standalone `tests/test_graph_setup_seam.py` under `nlp_env`
  **35 passed, 0 skipped** — the real-BLADE tier RAN and the BONMIN tier RAN, with no
  `CRASH` and no `Traceback`; standalone `tests/test_graph_fuel_damage.py` **41 passed,
  0 failed**; the executor and tick-loop selftests green; `tools/graph_executor_smoke.py`
  reported **`SMOKE PASS`** with **3/3 egos physically returned to airbases**;
  `git diff --check` clean. The real-BLADE tier supplies BOTH lifecycle outcomes
  directly — a sufficient-fuel ego that lands, and an insufficient-fuel ego that dies —
  and the death is pinned by a DIRECT CAUSAL WITNESS on `Game.remove_aircraft`: exactly
  ONE recorded removal of that ego, at `current_fuel <= 0`, with no replacement airframe
  in any inventory (the landing branch would show positive fuel and an inventory entry, and
  a weapon kill bypasses `remove_aircraft` entirely, so it could leave no record at all).
  The returning-ego freeze and peer continuation are shown on the real `run_episode`.
  **NO scientific probe, training run, rollout or baseline was executed** — nothing in
  this lock is a corrected-cell measurement. **This closes DEFECT C ONLY**; with Defects
  A and B already closed, the three-defect CODE correction is complete, but that is an
  implementation fact, not a scientific result (§8 owns the gate).

- **CORRECTED-CELL BOUNDED SHORT PROBE — EXECUTED / INDEPENDENTLY REVIEWED / VERDICT
  SUPERSEDED: SCIENTIFICALLY INCONCLUSIVE.** **READ THE SUPERSEDING VERDICT BELOW BEFORE
  ANY NUMBER IN THIS ENTRY.** Everything recorded here about run IDENTITY, invocation,
  provenance, accounting, artifact completeness and playback is PRESERVED HISTORICAL
  EVIDENCE and unchanged; what changed is the SCIENTIFIC verdict, which the later
  roster-integrity review invalidated (the superseding-verdict paragraph at the end of this
  entry, and the roster fix's own lock further below). The measurement is attributable to
  exact clean code SHA
  `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` (committed `2026-08-16T15:26:55+03:00`), the
  `main` head produced by the Defect-C documentation merge (PR #22). Per §7's hash
  convention this entry is keyed to the MEASURED CODE SHA; the documentation commit that
  creates it, and the merge that integrates it, cannot name their own SHAs and are
  deliberately not invented here. **No tracked file changed for the measurement** — it is
  a run of merged code, not a candidate.
  **RUN IDENTITY.** Run directory `training_output_20260816_162130`. Exactly ONE
  invocation, native exit code **0**:
  `conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json`.
  The tracked checkout was clean before and after. Preset blob
  `3c85e5bdc780600fe1ee528b3e35fc71591fe4b7`,
  `config_source.resolved_from = config_file`, `cli_overrides = []` — so the probe ran the
  reviewed preset with NO typed override and NO ad-hoc knob. Provenance complete:
  `git.available=true`, exact SHA, `branch=main`, `dirty=false`, `dirty_path_count=0`,
  Windows / `nlp_env` (CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`,
  `difficulty.factor = fuel_damage_baseline_v1` with `aircraft_penalty_coeff = 2.25` and
  `reward.formula_changed = false`. **Elapsed time is TWO DISTINCT QUANTITIES and they are
  never merged:** the harness's own `run_summary.json:run_seconds = 204.50847799982876`,
  and the externally measured invocation wall clock of **223.117 s** (the preserved probe
  runner's `timing.txt`: `PROBE_START_UTC = 2026-08-16T13:21:24.1839202Z` →
  `PROBE_END_UTC = 2026-08-16T13:25:07.3008671Z`). The harness figure excludes process
  start-up, `conda run` dispatch, imports and interpreter teardown, so it is NOT the wall
  clock and must not be labelled as one. **No code, configuration, preset or
  research-semantic change accompanied the run.**
  **VALIDITY VERDICT AS ORIGINALLY REVIEWED — `VALID MEASUREMENT / CORRECTED SHORT-PROBE
  PASS`. THAT VERDICT IS SUPERSEDED AND NO LONGER HOLDS** (see the superseding-verdict
  paragraph at the end of this entry); it is quoted here only so the review history stays
  legible. As originally judged, against the pre-declared validity gate and NOT against
  whether reward improved: exact clean Git
  provenance is complete; `run_summary.json:accounting_reconciled = true`; no
  INFRASTRUCTURE failure occurred (no `_VisualArtifactError`, and both recorded failures
  sit inside the `generation`/`setup`/`run`/`reward` episode taxonomy); BOTH evaluation
  rounds carry **4/4 complete matched pairs**; and PPO-update and artifact evidence are
  complete. At the time of that review this was read as closing the research-validity
  gate the FIRST probe exposed; **that reading is SUPERSEDED — the gate was not closed by
  this run** (end of this entry). What DOES survive unaffected: Defects A, B and C remain
  CLOSED / APPROVED / MERGED and are OPERATIONALLY WITNESSED in real playback from this run
  rather than in proof tests alone — the roster defect concerns which targets the
  MEASUREMENT counted, not whether those three corrections work.
  **ACCOUNTING — every denominator explicit, `skip_and_account_v1` unchanged.**
  **24 scheduled attempts** (8 train + 16 eval), **22 successful**, **2 failed, both at
  `setup`**. By condition: **clean 11 attempted / 10 successful / 1 failed**; **damaged 13
  attempted / 12 successful / 1 failed**. Training 8 attempted / 6 successful / 2 failed,
  all 6 successes producing wakes; evaluation **16/16 successful, 0 failed**.
  `accounting_reconciled = true`. **No retry, no substitution, no seed-band shift** — each
  failed seed was attempted once and recorded once:
  - train seed 2, **damaged**, `setup`, `RuntimeError` — B2 produced only two placements
    for `n_hidden=3`, because the static solve left one of the three egos without a
    non-empty route (the known exact-cardinality behaviour, §8);
  - train seed 4, **clean**, `setup`, `EpisodeRosterError` — one t=0 known target was
    absent from the executed world, so the roster would not have covered what runs.

  **FD FIRING RATE — state it against the right denominator.** The fuel-damage event fired
  and woke its selected ego in **12/12 successfully completed damaged episodes** (4 train +
  8 eval), which is **12/13 SCHEDULED damaged attempts**: the damaged seed-2 attempt failed
  during `setup`, before an event could exist. It must never be described as 12/12
  scheduled damaged attempts.
  **MATCHED HELD-OUT EVALUATION — the only within-seed claim is the paired delta.** Both
  rounds ran the same fixed band `[1000000,1000004)`, each seed twice (`forced_clean` +
  `forced_damaged`), 8 attempted / 8 successful per round.
  - `pre_update` (`updates_completed = 0`): **4/4 complete pairs**; clean mean
    `-0.4999997395829586` over 4; damaged mean `-0.8749999192707323` over 4; paired delta
    `-0.37500017968777366` over 4/4 pairs; **eval deaths 4**; unique targets confirmed
    mean `3.00`; meta-action mix `PLAN_COMPLIANCE 52 / OPPORTUNISTIC_ENGAGEMENT 0 /
    SELF_PRESERVATION_ABORT 0`.
  - `post_update` (`updates_completed = 2`): **4/4 complete pairs**; clean mean
    `-0.12499955989518505` over 4; damaged mean `-0.4583330529509838` over 4; paired delta
    `-0.33333349305579874` over 4/4 pairs; **eval deaths 0**; unique targets confirmed
    mean `4.25`; damaged **real RTB command yield 4/4**; deterministic meta-action mix
    `PLAN_COMPLIANCE 0 / OPPORTUNISTIC_ENGAGEMENT 18 / SELF_PRESERVATION_ABORT 13`.

  Held-out OVERALL mean moved `-0.6874998294268455` → `-0.2916663064230844`, **each over
  8/8 completed eval episodes**. Training: 6 successful episodes, **26 transitions**, **two
  productive iterations**, `updates_completed = 2`, train reward mean
  `-0.6874997530379319`, and `ended_counts` all `done` in both eval rounds. **These are
  SHORT-PROBE OBSERVATIONS, not estimates of converged policy performance**, and the two
  per-condition means are each over their own successful subset (§5).
  **ARTIFACT COMPLETENESS — reported ALONGSIDE the scientific denominators, never in place
  of one.** **24 visual-artifact bundles and 24 manifests**; **22 `complete`**, exactly
  matching the 22 successful attempts, and **2 `incomplete`**, exactly matching the two
  `setup` failures. Every `complete` bundle holds its known-only scenario, its executed
  t=0 scenario and its BLADE playback. Expected vs observed executed-world cardinality
  reconciles on every complete bundle at **3 known / 3 hidden / 6 total**. **Neither
  incomplete bundle fabricated a playback** (neither carries a recording at all). The
  complete scientific artifact remains preserved.
  **PLAYBACK WITNESSES AND CORROBORATING RUN EVIDENCE.** All of the following concerns the
  attempt preserved as
  `visual_artifacts/post_update_r001_e003_m1_seed1000003_damaged_tag901007`. **TWO evidence
  sources are involved and they are deliberately NOT merged.** (i) The BLADE **playback
  JSON** directly proves PHYSICAL state — position, fuel, `rtb`, route, weapon inventory,
  airbase membership. It is sampled every ten ticks (offsets below are stated from the
  recording's first frame) and it **does NOT record any per-wake meta-action label**;
  neither do `train_records.jsonl` / `eval_records.jsonl`, which persist per-round and
  per-iteration meta-action AGGREGATES only. (ii) The **preserved console transcript** of
  the run's per-episode `OK` blocks (`probe_console.log`, SHA-256
  `97bf45d56a3b224ef0ebe5a362bb7415b73e88520d192c891e347cb2412f31c4`, re-verified read-only
  before being cited) is the ONLY artifact that records a SELECTED ACTION LABEL, and it does
  so for the fuel-damage wake specifically.
  - **Defect A — KC-135R Stratotanker #76** (ego `0a14f756-13f2-4c78-8aa8-446da245aee5`, the
    id the playback binds to that name).
    *From the console transcript, for this exact attempt* — `[eval stage=post_update ep=3
    damaged seed=1000003] OK` records `fired=True tick=269 progress=0.300`,
    `fuel_before=203494.4 fuel_after=70026.7 factor=0.3441`, and
    `fd_wake=True fd_meta=SELF_PRESERVATION_ABORT rtb_command=True`. **The action label is
    an ATTRIBUTION FROM THAT RECORD, not something visible in playback.**
    *From the playback, independently* — the PHYSICAL signature at the same event: fuel
    falls from `203578.18` at the T+260 sample to `70017.43` at T+270 (sampled values, and
    therefore not identical to the console's exact event-time pair above), `rtb` flips
    `False → True` on that same frame, the route is replaced by a route to base at once, the
    aircraft lands at ≈ T+540, and it never resumes the abandoned assignment queue.
    The two sources agree, and the physical signature is **consistent with the merged
    EGO-GLOBAL abort semantics** rather than with removal of only the current assignment.
  - **Defect B — B-2 Spirit #698** (playback evidence alone; no action label is claimed).
    **ONE salvo per target**: one against Floridistan AFB #1794 at ≈ T+2320 (AIM-120
    `4 → 2`) and one against Hidden Airbase #003 at ≈ T+5140 (AIM-120 `2 → 0`). Each BLADE
    two-argument attack launches **two physical AIM-120 missiles**. No redundant second
    salvo against either target, and no repeated flat-timeout attack loop. Defect B changed
    NEITHER BLADE salvo quantity, NOR lethality, NOR general ammunition management — only
    the wait before a re-fire.
  - **Defect C — B-2 Spirit #698** (playback evidence, corroborated by the transcript's
    terminal line). It enters RTB at ≈ T+5240, **the episode keeps ticking while it
    physically flies home**, it lands at ≈ T+7705, and only then does the episode finish;
    the console block for the same attempt independently reports `ended=done ticks=7705
    dead=0`. That is physical completion, not RTB-command issuance.

  **DEFERRED RESEARCH HYPOTHESIS — NOT a defect, and NOT a proven action attribution.**
  Hidden Airbase #001 lies close to Hidden Airbase #003. In the sampled playback the B-2 is
  **50.07 km** from Hidden #001 at T+5230 while NOT yet in RTB — against a `DETECTION_KM`
  threshold of 50 km — and the next sampled frame, T+5240, shows RTB. Because playback is
  sampled every ten ticks and NOTHING preserved records an ORGANIC wake's selected
  meta-action — the jsonl records carry per-round / per-iteration AGGREGATES only, and the
  console transcript labels the FUEL-DAMAGE wake alone, which is a different ego in a
  different episode phase — it is PLAUSIBLE BUT NOT PROVEN that the B-2 crossed the threshold
  between samples and selected `SELF_PRESERVATION_ABORT` on the resulting wake. Treat
  possible over-conservatism as a FUTURE RESEARCH HYPOTHESIS about policy calibration —
  relevant to a later variable-FD-severity experiment. **Do not open a new defect, change
  the reward, retune the policy, or let this hypothesis invalidate or block this probe.**
  **EVIDENCE SHA-256** (verified read-only against the preserved run directory before being
  recorded here):
  `run_config.json=700f18fb54e485a12e0ab96a9b128353550c16c9f240e549bb40b01a303fbd22`,
  `train_records.jsonl=8581b4c50ad622ba2312c48434444cc57f560dfcd81c2495a03adf224666b16e`,
  `eval_records.jsonl=baa29c7281cf1dfbed9d928a83d60ab1e3c4826770de5ebcdf55ca34f12a68f2`,
  `episode_failures.jsonl=20c022a14971afb2776e774a268dbf0e6e6c0221fd2ac5e24dbe77e6c2f29784`,
  `run_summary.json=3038b754c82fb2dcb56d97632af4a24faa27dde68d2499701c228e3a208751fa`,
  `checkpoints/ckpt_iter0001.pt=605a05fde0084050fb66821e8da234bacacf1039a13f9bd0bd446876b9c2ba71`.
  **SUPERSEDING VERDICT (recorded with the roster-integrity lock below) — `INCONCLUSIVE —
  LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE SCIENTIFIC DENOMINATOR`.** This run's
  own ledger records clean train seed 4 as an ACCOUNTED `setup` `EpisodeRosterError`. The
  approved roster-integrity correction (`36365f2`, below) establishes what that error
  actually was: a MEASUREMENT/DATA-INTEGRITY fault, which must ABORT the run — not an
  episode outcome that may quietly shrink a scientific denominator. So one of this probe's
  24 scheduled attempts was removed from the population by an instrument defect while the
  run reported itself reconciled, and a denominator produced that way cannot be read as
  sound. **CONSEQUENCES, stated exactly.** (i) The reward numbers, per-condition means,
  paired deltas, death counts, fuel-damage yield and PPO-productivity figures above are NO
  LONGER SCIENTIFIC EVIDENCE about the fuel-damage cell; they remain identifiable as raw
  historical outputs of this run and nothing more. (ii) The claim that this run PASSED or
  permanently released the long-baseline validity gate is WITHDRAWN. (iii) Everything
  factual is RETAINED and unchanged — the run identity, the one invocation and its exit
  code, the preset blob and `cli_overrides = []`, the complete provenance, the two elapsed
  quantities, the mechanical accounting, the 24/24 artifact bundles, the evidence hashes,
  and the three playback witnesses. (iv) The earlier review was NOT wrong about what it
  inspected; it was made against documentation that then described `EpisodeRosterError` as
  an accounted `setup` failure, so the fault presented itself as ordinary episode attrition.
  The verdict changed because that ROUTING was later found to be the defect. (v) This is
  **NOT** a fourth defect in Defects A, B or C — their corrections remain merged and
  witnessed; it is a SEPARATE roster/source-of-truth defect, closed by `36365f2`.

- **FIRST LONG BASELINE — EXECUTED / INDEPENDENTLY REVIEWED / `INCONCLUSIVE —
  ROSTER/DATA INTEGRITY FAILED`.** The engineering verdict was `REQUEST FIXES`. The run is
  attributable to exact code SHA `c30b6982ba605d60976cc303256da4b5528b0e63`
  (`2026-08-16T21:47:25+03:00`, the PR #23 merge), recorded Git branch
  `task/long-baseline-execution`, `dirty=false`, `dirty_path_count=0`, Windows /
  `nlp_env` (CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`. Per §7's
  hash convention this entry is keyed to the MEASURED CODE SHA. **No tracked file changed
  for the measurement** — it is a run of merged code, not a candidate.
  **RUN IDENTITY.** Run directory `training_output_long_baseline_100x8_seed0`. Exactly ONE
  invocation, native exit code **0**:
  `PYTHONPATH=src conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config training_output_long_baseline_100x8_seed0/long_baseline_contract.json`.
  `config_source.resolved_from = config_file`, `cli_overrides = []` — **no typed override
  and no ad-hoc knob**; `difficulty.factor = fuel_damage_baseline_v1`,
  `aircraft_penalty_coeff = 2.25`, `reward.formula_changed = false`. **Elapsed time is TWO
  DISTINCT QUANTITIES and they are never merged:** the harness's own
  `run_summary.json:run_seconds = 7764.3988857`, and the externally measured invocation
  wall clock of **7778.704310178757 s** (`timing.txt`:
  `PROBE_START_UTC = 2026-08-16T19:18:32.509409Z` →
  `PROBE_END_UTC = 2026-08-16T21:28:11.191698Z`). The harness figure excludes process
  start-up, `conda run` dispatch, imports and teardown, so it is NOT the wall clock.
  **THE SCIENTIFIC CONTRACT** (from the preserved `long_baseline_contract.json`, which is a
  MEASUREMENT contract and deliberately **not** a repository preset): 100 scheduled
  training iterations × 8 training attempts, train seeds `[0, 800)`; evaluation every 5
  iterations over 8 FIXED held-out seeds `[1000000, 1000008)`, each seed evaluated as
  `forced_clean` AND `forced_damaged`; **21 evaluation rounds** including the initial
  `pre_update`; the final 3-agent / 3-known / 3-hidden cell with its 200 km / 100 km
  geometry and `include_sams = false`; FD-BASELINE-v1 unchanged (`seeded_mixture`,
  `P(damaged) = 0.5`, leg progress `0.3`, RTB margin `1.10`); visual artifacts enabled for
  every scheduled attempt; `checkpoint_every = 10` → **10 checkpoints**.
  **MECHANICAL ACCOUNTING — historical fact, and NOT validity.** **1,136 scheduled
  attempts, 860 successful, 276 failed.** Training 800 attempted / 566 successful / 234
  failed; evaluation 336 attempted / 294 successful / 42 failed;
  `accounting_reconciled = true`; **100 productive iterations and 100 PPO updates**
  (`updates_completed = 100`). Every one of those counts reconciles, and **that is exactly
  the problem**: a run can be perfectly self-consistent about a population an instrument
  defect silently shrank, so these counts must never be offered as evidence that the
  measurement was sound.
  **FAILURE BREAKDOWN** (`failures_by_pipeline_stage` = `{"setup": 276}`;
  `failures_by_error_type` = `{"RuntimeError": 101, "EpisodeRosterError": 143,
  "FuelDamageError": 32}`; `failures_by_condition` = `{"clean": 123, "damaged": 153}`):
  - **143 `EpisodeRosterError`, ALL in training, spread over 83 distinct iterations** — 75
    clean and 68 damaged. Two shapes: **126 PRE-run** roster failures claiming a t=0 known
    target was absent from the executed world (125 naming one target, 1 naming two), and
    **17 POST-run** failures raised after a real episode and a real playback because a
    CONFIRMED target id fell outside the incorrectly shortened roster.
  - **101 B2 exact-cardinality `RuntimeError`** — 59 train, 42 eval.
  - **32 `FuelDamageError`**, every one a DAMAGED TRAINING attempt.
  **INDEPENDENT ARTIFACT REVIEW — what falsified the run.** Every one of the 126 pre-run
  roster failures had a FULL SIX-TARGET authoritative `executed_t0_scenario.json`; the 17
  post-run failures left real playback files their `incomplete` manifests did not list; and
  **11 `complete` manifests reported observed `3 known / 2 hidden / 5 total` while their own
  authoritative executed-t0 scenarios held `3 + 3 = 6`** (re-verified here across all 1,136
  preserved manifests: 860 `complete`, 276 `incomplete`, and exactly 11 of the complete ones
  carrying that `5`-total observation). **ROOT CAUSE: allocated-only solver output was being
  read as world inventory** — closed by `36365f2` below.
  **VERDICT: engineering `REQUEST FIXES`; scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY
  FAILED`.** **Do NOT report this run's reward improvement, per-condition means, paired
  deltas, survival, fuel-damage yield or PPO performance as scientific evidence.** Those
  values are present in the preserved records and may be referred to ONLY as raw historical
  outputs of an inconclusive run; they are deliberately not tabulated here, precisely so
  they cannot be lifted out of context as a baseline. **The 101 B2 and 32 fuel-window
  failures are NOT corrected by `36365f2`** — they remain EXPECTED scientific outcomes under
  the current contract (§8) and must not be relaxed, retried, retuned or reclassified.
  **PRESERVATION.** The run directory is preserved and must not be modified, moved, copied,
  repackaged, deleted or regenerated. **EVIDENCE SHA-256** (verified read-only against the
  preserved run directory before being recorded here):
  `long_baseline_contract.json=18d0dede02b8b89cfff8867aefdd68901d995f1664dcbba7342e26a9bbed02ac`,
  `run_config.json=fae72de5f7c10ec5c9264330510d9ab9fac8af34c5270f1912a0f4d36b9526e2`,
  `run_summary.json=6ea6842ed981219c7dd45fb9cbc63587a7e4c26d57a7602c7f6676aaa31d2848`,
  `train_records.jsonl=d9d94f9a18448565a31a45c2ec950d1882e7b130e3cd16a181acc1074b4aa96c`,
  `eval_records.jsonl=e378da7ff7c3cb21d63ca016fa5d0911fe6209a87d757467b89a64fc043b7edb`,
  `episode_failures.jsonl=dcf7871e16102a5b9d090a16276603d47611763fdfbb125c584978edbd3cac32`,
  `timing.txt=828d5c0d390c2e082ca4d22c652c8d36b51b23c5140c29139651897f8b8fa10a`,
  `long_baseline_console.log=00d35661b6246091d1e199944c90571233606f76c8d571287215d9b218b60233`.
  The review package additionally carried
  `review_metadata/package_manifest.json=a5b7acd5d607958e54b81e8ba2f354155f18c19a9803432bbbee1f0e9fbfb2c4`
  and
  `review_metadata/playback_audit.jsonl=4c5bbb1c9cd629c20dde761137f8b2fd9e83ff2b5b4a398410b2abf7c7242137`,
  inside a review ZIP whose own SHA-256 is
  `b22aecec7b1c99d3689ce6ad34d8c467473bb7a50d0d3bd25a3a5f4c370440af`; those two live in the
  review package rather than in the preserved run directory, so they are recorded from the
  review record rather than re-derived here.

- **ROSTER / WORLD-TRUTH INTEGRITY: executed-world inventory separated from oracle
  allocation — CLOSED / APPROVED / MERGED.** Approved candidate SHA
  `36365f210e8a659a641a7713f612c7e0ec1d4665` (`2026-08-17T14:01:10+03:00`), reviewed
  `APPROVE`, integrated by `f37ea1c8559405d5de24a9c2dd9e740227acaeeb`
  (`2026-08-17T15:48:30+03:00`, PR #24). **Candidate and integration share the IDENTICAL
  tree `f801538080f2ad282766d32346580189fa949f0c`, so the integrated tree is exactly the
  reviewed tree.** Grade A under `GPT_GITHUB`. The technical contract is in §5
  ("Roster / world-truth integrity" and the Stage-0 "WORLD INVENTORY IS NOT ORACLE
  ALLOCATION" block); the routing is in §6. This entry records the LOCK, not the mechanism.
  **THE DEFECT — a FOURTH, SEPARATE one, not a regression in Defects A, B or C.** The
  trainer answered "which targets does this episode contain?" from `ctx.beliefs` (known) and
  `ctx.oracle_tasks` (executed). Both are ALLOCATIONS: `solve_and_normalize` returns an
  allocated-only task list by contract, so every target the solver did not select was
  missing from them while still sitting in the world the executor flew through, sensed,
  attacked and confirmed. The roster therefore under-counted its own world and then FAILED
  the episode for the discrepancy it had itself introduced — **as an accounted `setup`
  failure**, which is why the long baseline above lost 143 of 800 training attempts to a
  measurement defect across 83 iterations while reporting itself healthy and reconciled.
  **THE APPROVED SEMANTICS.** `solve_and_normalize()` REMAINS allocated-only, and
  `belief_tasks` / `oracle_tasks` REMAIN allocations rather than world inventories —
  nothing about the oracle denominator changed. `known_target_ids` snapshots all raw
  known-world target ids before solver filtering; `executed_target_ids` snapshots all raw
  AUTHORITATIVE-world target ids before solver filtering; belief ids must agree across egos
  at t=0 and be a SUBSET of the known snapshot; hidden ids are executed MINUS known in
  executed-world order. The approved 3-known / 3-hidden / 6-total cell is checked
  (`_require_scheduled_cell`) BEFORE fuel-damage planning and before execution.
  Roster/world-integrity faults ABORT train and eval as
  INFRASTRUCTURE / DATA-INTEGRITY failures (`MeasurementIntegrityError`, with
  `EpisodeRosterError` as its subclass): they do NOT enter `EpisodeAttemptError`,
  `episode_failures.jsonl`, `skip_and_account_v1`, condition failure tallies, or any
  scientific denominator. After `run_episode`, playback synchronization
  (`_AttemptArtifacts.sync_recordings`) and confirmed-id validation happen BEFORE the
  reward; an `incomplete` manifest truthfully lists real playback that was already written;
  and a manifest cannot become `complete` when expected and observed world counts disagree.
  **Reward, PPO, oracle allocation, fuel damage, B2, seeds, schedules, the tick loop, the
  executor, the generator and vendored BLADE were UNCHANGED.**
  **REVIEWED SCOPE: FIVE files** — `src/match_aou/rl/training/graph_episode_setup.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_setup_seam.py`,
  `tests/test_graph_train.py`, `tests/test_graph_fuel_damage.py`.
  **ACCEPTED IMPLEMENTATION EVIDENCE:** focused base-environment suite **207 passed, 4
  skipped**; full suite **272 passed, 4 skipped**; standalone `tests/test_graph_train.py`
  under `nlp_env` **119 passed**; standalone `tests/test_graph_fuel_damage.py` **41
  passed**; standalone `tests/test_graph_setup_seam.py` **39 passed, 0 skipped**, including
  the real-BLADE and BONMIN tiers; `git diff --check` clean.
  **NO training run, probe, rollout, seed sweep or baseline rerun occurred during the
  correction.** **CONSEQUENCE FOR THE TWO AFFECTED MEASUREMENTS:** the long baseline above
  is `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED`, and the corrected-cell short probe's
  `VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS` verdict is SUPERSEDED by
  `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE SCIENTIFIC
  DENOMINATOR`. This lock certifies the CODE correction; it is **not** a measurement of the
  cell, and no result may be pre-claimed for the rerun §8 then authorized. **That rerun has
  since been EXECUTED, independently reviewed and APPROVED — see the VALID long-baseline
  entry below, which is the authoritative measurement of this cell.**

- **PHASE-A LONG BASELINE (RERUN) — EXECUTED / INDEPENDENTLY REVIEWED / `APPROVE — VALID
  MEASUREMENT`. THE FIRST SCIENTIFICALLY VALID MEASUREMENT OF THE FUEL-DAMAGE CELL.** The
  measurement is attributable to exact clean code SHA
  `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc` (committed `2026-08-17 19:25:42 Asia/Jerusalem`),
  the `main` head produced by the roster-integrity documentation merge (PR #25). Per §7's
  hash convention this entry is keyed to the MEASURED CODE SHA; the documentation commit that
  creates it, and the merge that integrates it, cannot name their own SHAs and are
  deliberately not invented here. **No tracked file changed for the measurement** — it is a
  run of already-reviewed merged code, not a candidate.
  **RUN IDENTITY.** Run directory
  `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`. Exactly ONE invocation,
  native exit code **0**, `PYTHONPATH=src`, from the repository root:
  `conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf/long_baseline_contract.json`.
  `config_source.resolved_from = config_file` and **`cli_overrides = []`** — the reviewed
  measurement contract with NO typed override and NO ad-hoc knob. Provenance complete:
  `git.available=true`, exact SHA, `branch` resolved, `dirty=false`, `dirty_path_count=0`,
  Windows / `nlp_env` (CPython 3.12.3), torch 2.7.1+cpu, gymnasium 1.0.0, vendored BLADE,
  BONMIN available; `difficulty.factor = fuel_damage_baseline_v1` with
  `aircraft_penalty_coeff = 2.25` and `reward.formula_changed = false`. **Elapsed time is TWO
  DISTINCT QUANTITIES and they are never merged:** the harness's own
  `run_summary.json:run_seconds = 8493.042731400012`, and the externally measured invocation
  wall clock of **8509.632915974 s** (`timing.txt`:
  `PROBE_START_UTC = 2026-08-17T22:32:34.530300100Z` →
  `PROBE_END_UTC = 2026-08-18T00:54:24.166136300Z`). The harness figure excludes process
  start-up, `conda run` dispatch, imports and teardown, so it is NOT the wall clock.
  **CONTRACT FIDELITY.** The scientific contract is the preserved authoritative
  `long_baseline_contract.json` of the invalid first long baseline (SHA-256
  `18d0dede…02ac`), cloned with **exactly ONE field changed — `output_dir`**. Verified before
  execution: 27 keys, identical key SETS and identical key ORDER, exactly one differing key,
  every other value identical. So the train seeds `[0, 800)`, the held-out seeds
  `[1000000, 1000008)`, the 100 × 8 schedule, `eval_every = 5`, the matched
  forced-clean / forced-damaged pair design, the PPO settings, the 3-agent / 3-known /
  3-hidden cell with its 200 km / 100 km geometry, `include_sams = false` and every
  FD-BASELINE-v1 parameter are UNCHANGED. **A directory name is not a scientific parameter.**
  **ACCOUNTING — every denominator explicit, `skip_and_account_v1` unchanged.**
  **1,136 scheduled attempts, 993 successful, 143 failed**, `accounting_reconciled = true`.
  Training 800 attempted / 699 successful / 101 failed, every success wake-bearing;
  evaluation 336 attempted / 294 successful / 42 failed across **21 rounds**. **ZERO
  `EpisodeRosterError`, ZERO `MeasurementIntegrityError`, ZERO `_VisualArtifactError`, and
  zero crash outside the `generation`/`setup`/`run`/`reward` episode taxonomy** — every one
  of the 143 failures is an accounted episode failure, each recorded exactly once, with no
  retry, no substitution and no band shift.
  **FAILURES — both families are EXPECTED SCIENTIFIC OUTCOMES, not defects.** Exactly **101
  B2 exact-cardinality `RuntimeError`** and exactly **42 `FuelDamageError`**, all at stage
  `setup`. The independent review established two exact set relations: the 101 B2 failures
  are **exactly the same scheduled attempts as in the invalid old long run**, and
  `EpisodeRosterError` went **143 (old) → 0 (new)**, with the **10 additional
  `FuelDamageError` attempts relative to the old run being exactly attempts that were
  `EpisodeRosterError` there**. Held-out seed **`1000005`** fails B2 for BOTH matched members
  in ALL 21 evaluation rounds, so the matched-pair yield is a STRUCTURAL **7/8 in every
  round** — a property of that seed's world, not stochastic attrition.
  **LEARNING — reported only after the validity gate passed.** 100 scheduled iterations gave
  **100/100 productive PPO updates** (`updates_completed = 100`) over **2,566 transitions**,
  with 0 zero-wake and 0 all-failed iterations. The matched held-out paired reward delta
  improved from **−0.375000 over 7/8 pairs** at `pre_update` to **−0.071429 over 7/8 pairs**
  at the final `post_update`, and evaluation aircraft deaths fell from **7 to 0**. Final
  clean reward reached **0 on all 7** exact-cardinality-feasible held-out worlds; final
  damaged reward reached **0 on 5 of those 7**, with the residual damaged cost concentrated
  in exactly two worlds — seed **1000004** (`−0.333333`, 4/6 targets, 0 deaths) and seed
  **1000007** (`−0.166667`, 5/6 targets, 0 deaths), i.e. both PRESERVED THE AIRCRAFT at the
  cost of incomplete target coverage. In all **7** completed damaged held-out worlds the
  deterministic fuel-damage decision changed from `PLAN_COMPLIANCE` before training to
  `SELF_PRESERVATION_ABORT` after training, and selected playback witnesses independently
  confirm this as a REAL PHYSICAL behavioural change including survival and RTB under the
  final policy. **The two per-condition means are each over their own successful subset**, so
  the within-seed claim is the matched-pair delta alone (§5).
  **FUEL-EXPOSURE CAVEAT — state it against the right denominator.** Successful damaged
  TRAINING episodes number **324** while damage events actually fired **323** times; the one
  non-firing successful damaged training episode is **seed 424** (iteration 53), whose
  selected ego returned before reaching the 0.30 leg-progress trigger. **No defect is
  inferred** — a fuel-damage PLAN existed and every LIVE quantity is recorded as `n/a`, so
  the artifacts record that the event did not fire and do not record why. **Evaluation
  exposure is COMPLETE: 147 / 147** successful damaged eval episodes fired and woke their
  selected ego.
  **ARTIFACT COMPLETENESS — reported ALONGSIDE the scientific denominators, never in place of
  one.** 1,136 bundles and 1,136 manifests: **993 `complete`** (exactly the successful
  attempts) and **143 `incomplete`** (exactly the failed ones). **All 993 complete bundles
  reconcile expected against observed 3 known / 3 hidden / 6 executed targets** — the
  five-target contradiction that falsified the old run does not occur once. No incomplete
  bundle fabricated a playback, and no completed run left an unlisted one.
  **INDEPENDENT REVIEW.** A read-only evidence package
  `long_baseline_rerun_737b4bf_gpt_review.zip` (SHA-256
  `f2582c0ca7f460a5f51bd515aeb0506f0476e8e06e4039312b7371858a08b932`) carried the core
  evidence of both runs, all 1,136 original manifests, selected raw playback bundles and
  derived audits. The GPT verdict is **`APPROVE — VALID MEASUREMENT`**.
  **EVIDENCE SHA-256** (verified read-only against the preserved run directory):
  `long_baseline_contract.json=f5b5984317ea503862fcf76670bac0f4c3f147f39d8daf969ab90009ff438c1f`,
  `run_config.json=eeb4f449ead84b5cf7a72c6248a810169e8eb5f36fa7ba94384cab8a9bd1fb4a`,
  `run_summary.json=ee32e8b7b6735351700d19fc560c840307c19c0af7a446525e09dc586154b71d`,
  `train_records.jsonl=29c2a40bce2267af5aff60d281258e27eadd432db2291f3a2d0f29854a4cc1bd`,
  `eval_records.jsonl=116022680a7d8df97466c7c43faa7b6ff5b3b783403709f5d285bca66ca1995f`,
  `episode_failures.jsonl=313990a1428d8bde71c25db6bbc55b33a731ef942d0dff994e1e06b16a4b6ea1`,
  `timing.txt=27a11d9867f88dc04ec6f5d9d1ff75c1fc51aefbaa5bdc9ddf61fbd3b92b4a9c`,
  `long_baseline_rerun_console.log=be3c97d3106b8d18523d433e60e98dd06a1d98f00051d9752b9b01d25deff9a6`.
  **PRESERVATION.** This run directory is preserved and must not be modified, moved, copied,
  repackaged, deleted or regenerated — and neither may the invalid
  `training_output_long_baseline_100x8_seed0`, which remains preserved and explicitly
  scientifically INCONCLUSIVE.
  **PHASE-A SCIENTIFIC CONCLUSION — VALID BASELINE.** The first scientifically valid
  long-baseline measurement of FD-BASELINE-v1 was obtained from the clean actor-only,
  no-communication graph-RL stack at measured code SHA
  `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`. Across the seven exact-cardinality-feasible
  held-out matched worlds, the paired fuel-damage reward penalty improved from `−0.375000`
  before training to `−0.071429` after 100 productive PPO updates, while evaluation aircraft
  deaths fell from 7 to 0. In all seven completed damaged held-out worlds the deterministic
  policy changed from `PLAN_COMPLIANCE` before training to `SELF_PRESERVATION_ABORT` after
  training. Final clean performance reached reward 0 on all seven feasible worlds; final
  damaged performance reached reward 0 on five, while the remaining two preserved the
  aircraft at the cost of incomplete target coverage. These results establish **end-to-end
  learnability and meaningful ego-local runtime adaptation in the locked Phase-A reference
  cell.** **THE EXPLICIT NON-CLAIMS:** they do **NOT** establish global optimality, **NOT**
  monotonic convergence, **NOT** generalization beyond this fixed cell and this held-out seed
  set, and **NOT** any benefit from centralized training. Those are subsequent research
  questions. **PHASE A IS CLOSED BY THIS ENTRY.**

- `eecc9b5` — **FD-VARIABLE-SEVERITY-v1: the mild/severe fuel-damage research factor with
  matched clean/mild/severe evaluation — CLOSED / APPROVED / MERGED.** Approved candidate
  SHA `eecc9b5d91bce4a98a070a29307cc12af0d4c4a3`, integrated by merge commit
  `177e969446ef6c01c729484f2ea9969c94a27330` (`2026-08-20 12:15:28 Asia/Jerusalem`,
  PR #27). The candidate was merged with a MERGE COMMIT and preserved as its SECOND
  PARENT (ordered parents: `4f0068847b017795717c5f0e331f647bcfc30547`, then
  `eecc9b5…`); candidate and integration share the IDENTICAL tree
  `37ebd8c56266fdd862cc7244c5f22a6ac95e438c` (verified locally), and the
  candidate→integration comparison contains ZERO changed files. Grade A under
  `GPT_GITHUB`. The technical contract is in §5 (the FD-VARIABLE-SEVERITY-v1 mechanism
  block, its measurement-surface block, and the scheduled-vs-executed cell block) and the
  routing in §6. This entry records the LOCK, not the mechanism.
  **THE RESEARCH PROBLEM.** Under the merged FD-BASELINE-v1 design EVERY damaged episode
  is structurally SEVERE, so "damaged" and "continuing is infeasible" are the SAME fact
  and a trained actor can learn the shortcut `fuel damage ⇒ abort` without ever reading
  its own fuel gauge. The approved extension splits the damaged half into a MILD band
  where continuing remains genuinely feasible and a SEVERE band where it does not, so the
  response has to be read off the ego's own live fuel. **The LEGACY modes are UNCHANGED —
  same seeds, same conditions, same selected egos, same planned-midpoint target — because
  an approved measurement exists on them** (`737b4bf`, the entry above), and a factor that
  moved them would invalidate that baseline instead of extending it.
  **APPEND-ONLY FIX CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. First candidate
  `73752d872a8cd17f703790ef41bee46a734170bb` (parent `4f00688…`) received REQUEST-FIXES on
  ONE measurement-integrity defect; the correction landed as the NEW CHILD COMMIT
  `eecc9b5…` (parent `73752d8…`), touching TWO files
  (`src/match_aou/rl/training/graph_train.py`, `tests/test_graph_fuel_damage.py`).
  **THE FINDING AND ITS CLOSURE.** `_ConditionTally.success` checked only that the EXECUTED
  cell was a LEGAL cell of the run. Under the new design that membership test ACCEPTS a
  scheduled `mild` that executed as `severe` — booking the ATTEMPT in one cell's
  denominator and the REWARD in another, corrupting BOTH at once, and letting a triad's
  within-seed delta be taken between two members the schedule never paired. The approved
  fix makes `expected_cell` a REQUIRED keyword and requires scheduled == executed
  EQUALITY before ANY accounting, with both production call sites (training and
  evaluation) passing their scheduled cell and the guard running FIRST — so a mismatched
  episode reaches neither the tally, nor a matched-group member reward or delta, nor
  `episode_outcomes.jsonl`, nor the PPO buffer. A mismatch is a
  `MeasurementIntegrityError` INFRASTRUCTURE abort, never an accounted scientific episode
  failure. **It has NOT been observed in the real simulator**: the regression test INJECTS
  the divergence through a stub, because normal production does not currently generate it.
  **CUMULATIVE REVIEWED SCOPE: EXACTLY FIVE FILES** —
  `src/match_aou/rl/training/graph_fuel_damage.py`,
  `src/match_aou/rl/training/graph_train.py`,
  `src/match_aou/rl/training/graph_rollout.py`, `tests/test_graph_fuel_damage.py` and
  `tests/test_graph_train.py`. No vendored BLADE, solver, `graph_reward`, PPO, encoder,
  action-space, tick-loop, executor, episode-setup, hidden-placement, generator, scenario,
  preset, config or README file was touched. Target destruction remains DETERMINISTIC at
  `probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate
  future Grade-A research task** (§8).
  **ACCEPTED EVIDENCE at the approved head:** `tests/test_graph_fuel_damage.py`
  **60 passed**; `tests/test_graph_train.py` **119 passed**; both standalone `nlp_env`
  `__main__` runners **60 passed / 119 passed**; full suite **291 passed, 4 skipped**;
  `git diff --check` clean. `graph_train --selftest` — TEST 1 passed, TEST 2 passed, and
  **TEST 3 failed IDENTICALLY TO THE BASE** on the already-known B2 seed-2
  exact-cardinality case (§8): a pre-existing expected outcome of the current contract,
  **not a PR #27 regression**.
  **NO scientific baseline, long training run, probe, rollout or artifact-generating smoke
  was executed for PR #27** — that remains true of THIS lock, which certifies CODE only.
  Nothing in this entry is a measurement of the variable-severity cell. **The measurement
  was taken separately and afterwards**, at measured code SHA `bf1e045f`, and it is
  recorded in the next §7 entry; §8 owns the phase state.
- **FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE — EXECUTED / INDEPENDENTLY REVIEWED /
  `APPROVE — VALID MEASUREMENT`. THE FIRST AND ONLY SCIENTIFICALLY VALID MEASUREMENT OF THE
  VARIABLE-SEVERITY CELL, AND ITS PRIMARY BEHAVIOURAL FINDING IS NEGATIVE.** The measurement
  is attributable to exact clean code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
  `dd881478b8e2e521054d09bc865437f1308be1a2` (committed
  `2026-08-20 14:50:24 Asia/Jerusalem`, the `main` head produced by the variable-severity
  documentation merge, PR #28). Per §7's hash convention this entry is keyed to the MEASURED
  CODE SHA; the documentation commit that creates it, and the merge that integrates it,
  cannot name their own SHAs and are deliberately not invented here. **No tracked file
  changed for the measurement** — it is a run of already-reviewed merged code, not a
  candidate. It was executed from a clean DETACHED snapshot at that SHA
  (`provenance.git.repo_root` = `…\fd_variable_severity_v1_bf1e045f_snapshot`,
  `branch = HEAD`, `dirty = false`, `dirty_path_count = 0`), so repository work landing
  after `bf1e045f…` — Phase-B CTDE included — is outside the measured tree and neither
  contaminated the run nor is attributable to it.
  **RUN IDENTITY.** External measurement root `C:\Users\Itama\f7r2`; contract `c.json`;
  output directory `r`; console `console.log`; timing `timing.json`. Exactly ONE
  invocation, native exit code **0**:
  `conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config "C:\Users\Itama\f7r2\c.json"`.
  `config_source.resolved_from = config_file` and **`cli_overrides = []`** — the reviewed
  measurement contract with NO typed override and NO ad-hoc knob. Provenance complete:
  `git.available = true`, exact SHA, clean, Windows 10.0.19045 / `nlp_env`
  (CPython 3.12.3), vendored BLADE, BONMIN available and probed `ok`;
  `difficulty.factor = fuel_damage_variable_severity_v1` with
  `fuel_damage_mode = seeded_variable`, scheduled cell probabilities
  `clean 0.50 / mild 0.25 / severe 0.25`, `target_policy = live_severity_midpoint_v1`,
  `aircraft_penalty_coeff = 2.25` and `reward.formula_changed = false`. **Elapsed time is
  TWO DISTINCT QUANTITIES and they are never merged:** the harness's own
  `run_summary.json:run_seconds = 5998.791282300022`, and the externally measured
  invocation wall clock of **6021.3954213 s** (`timing.json`:
  `start_utc = 2026-08-22T14:36:14.1838098Z` → `end_utc = 2026-08-22T16:16:35.5973421Z`).
  The harness figure excludes process start-up, `conda run` dispatch, imports and teardown,
  so it is NOT the wall clock.
  **CONTRACT FIDELITY — a REPLACEMENT, not a redesign.** The scientific contract is the
  invalid precursor's own contract cloned with **exactly ONE field changed —
  `output_dir`**. Verified read-only before this record: 25 keys, identical key SETS and
  identical key ORDER, exactly one differing key, every other value identical. So the
  §8-approved run shape is unchanged: **50 scheduled training iterations × 8 scheduled
  training attempts = 400**, `base_seed = 0`; evaluation every 5 iterations INCLUDING the
  initial `pre_update` ⇒ **11 evaluation rounds**; **8 fixed held-out seeds** from
  `1_000_000`, each a matched **clean / mild / severe TRIAD** ⇒ **11 × 8 × 3 = 264**
  scheduled evaluation attempts; **664 scheduled attempts in total; NO early stopping**;
  the locked cell of 3 agents / 3 known / 3 hidden with its 200 km / 100 km geometry,
  `DETECTION_KM = 50`, `include_sams = false`, target-destruction `probability = 1`, frozen
  solver and BLADE, unchanged `graph_reward` formula and unchanged actor-only PPO;
  `visual_artifacts = true`. **A directory name is not a scientific parameter.**
  **THE EXCLUDED PRECURSOR — `INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT`.** An earlier
  attempt at the SAME contract, preserved at
  `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640\training_output_fd_variable_severity_v1_50x8_seed0`,
  is **preserved historical / ENGINEERING evidence ONLY and is EXCLUDED from every
  scientific reading.** Its ledger carries the same 78 `setup` failures (58 B2
  `RuntimeError` + 20 `FuelDamageError`) **plus 70 additional `run`-stage
  `FileNotFoundError`s, and all 70 are `post_update` SEVERE evaluation members** — 10
  post_update rounds × the 7 exact-cardinality-feasible held-out seeds, i.e. **the entire
  post-training severe arm, which is precisely the arm the experiment exists to measure.**
  The cause is a Windows `MAX_PATH` playback-export failure: the BLADE recording path was
  **267 characters** against the 260-character limit. **This is NOT a negative scientific
  result** — it is an infrastructure failure that systematically deleted one experimental
  cell, and a population destroyed that way yields no result rather than a null one. The
  precursor is preserved and must not be modified, moved, copied, repackaged, deleted or
  regenerated. Its evidence SHA-256 (re-verified read-only before this record):
  `fd_variable_severity_v1_contract.json=77e5992994235abd0962547b549dbfb17889cb51f745993f4c8f2a89a2824326`,
  `invocation_timing.json=cf531cd2f0574674d66d21cb16e12d963aeb269fac9d9ae55cdaebf1a47c26b6`,
  `run_config.json=de20663b689f0a00f483c30ac92e6b240c5df2873d5d3f0aa9145b550f71ea1e`,
  `episode_outcomes.jsonl=bd0f1c4009cec2ea596db372f3d0e76c41c8e0a2352aa3775b108fcad2e544c6`,
  `episode_failures.jsonl=430728726b90383a47e1f8f62b997ed82a12ea0f23c205e88b8a7a360a536e0b`,
  `run_summary.json=79b30564da6f8157de536f5678dbd67851516b0baa20539034929b4f7b6f0e85`.
  **ACCOUNTING — every denominator explicit, `skip_and_account_v1` unchanged.**
  **664 scheduled attempts, 586 successful, 78 failed**, `accounting_reconciled = true`,
  and `586 + 78 = 664`. Training **400 attempted / 355 successful / 45 failed**, every
  success wake-bearing; evaluation **264 attempted / 231 successful / 33 failed** across
  **11 rounds**. Per CELL — training `clean 202 / 190 / 12`, `mild 92 / 76 / 16`,
  `severe 106 / 89 / 17`; evaluation `clean 88 / 77 / 11`, `mild 88 / 77 / 11`,
  `severe 88 / 77 / 11`. **Every one of the 78 failures is at stage `setup`**: **58 B2
  exact-cardinality `RuntimeError`** and **20 `FuelDamageError`** (no valid strict fuel
  band/window). All **33 evaluation failures are held-out seed `1000005`** — 11 rounds × 3
  triad members — the same structural B2 world the Phase-A baseline also lost, reported and
  never repaired. **ZERO `FileNotFoundError`, ZERO `MeasurementIntegrityError`, ZERO
  `EpisodeRosterError`, ZERO `_VisualArtifactError`, and zero crash outside the
  `generation`/`setup`/`run`/`reward` episode taxonomy.** Outcome and failure identities
  are unique and DISJOINT (586 records in `episode_outcomes.jsonl`, 78 in
  `episode_failures.jsonl`, zero overlap), and no scheduled-vs-executed CELL mismatch abort
  occurred, so every successful attempt was booked in the cell the schedule asked for.
  **MATCHED TRIADS — a structural 7/8 in EVERY round.** All 11 evaluation rounds report
  **7/8 complete clean+mild+severe triads**, including `pre_update`
  (`updates_completed = 0`) and the final `post_update` (`updates_completed = 50`). Seed
  `1000005`'s B2 failure is the ceiling; it is a property of that world, not stochastic
  attrition.
  **PRIMARY BEHAVIOURAL RESULT — NO SEVERITY-CONDITIONED META-ACTION SEPARATION. THIS IS
  THE FINDING, AND IT IS NEGATIVE.** Rates are over **FD WAKES**, never over episodes.
  - **`pre_update`** — MILD: 7 wakes, `PLAN_COMPLIANCE 7/7`, abort 0, engage 0.
    SEVERE: 7 wakes, `PLAN_COMPLIANCE 7/7`, abort 0, engage 0.
  - **FINAL `post_update` (`updates_completed = 50`)** — MILD: 7 wakes,
    `PLAN_COMPLIANCE 7/7`. SEVERE: 7 wakes, `PLAN_COMPLIANCE 7/7`.
  - **ALL TEN `post_update` ROUNDS COMBINED** — MILD: 70 wakes,
    `PLAN_COMPLIANCE 63 = 0.900`, `SELF_PRESERVATION_ABORT 7 = 0.100`, engage 0.
    SEVERE: 70 wakes, `PLAN_COMPLIANCE 63 = 0.900`, `SELF_PRESERVATION_ABORT 7 = 0.100`,
    engage 0. **The two distributions are IDENTICAL.**
  - **TRAINING successes** (stochastic policy, context only) — MILD: 76 wakes,
    `PLAN_COMPLIANCE 60`, `SELF_PRESERVATION_ABORT 16`. SEVERE: 89 wakes,
    `PLAN_COMPLIANCE 66`, `SELF_PRESERVATION_ABORT 23`.

  The deterministic held-out actor did NOT differentiate a survivable MILD fuel loss from
  an unsurvivable SEVERE one in its FD-wake action choice, before OR after training. **This
  is a VALID NEGATIVE SCIENTIFIC RESULT.** It does **NOT** mean the actor is broken, that
  training failed, that PPO did not learn, that the actor never uses fuel at all, or that
  the result generalizes beyond this fixed cell and this held-out seed band. And **"mild
  must choose `PLAN_COMPLIANCE`" is NOT a correctness rule** (§5) — what was measured is
  whether the response DIFFERS, not whether it matched a prescribed label.
  **DENOMINATOR / INDEPENDENCE CAVEAT — load-bearing.** The clean statistical unit for the
  FINAL held-out policy is the final round's **7 complete matched triads**. The 70
  `post_update` observations per severity REUSE those same seven feasible held-out seeds
  across ten checkpoints; they describe the learning TRAJECTORY across checkpoints, and
  **they are NOT 70 independent held-out worlds and must never be used to inflate sample
  size.**
  **PHYSICAL OUTCOMES — THE SEVERITY FACTOR IS REAL.** The absence of behavioural
  separation is NOT because the two cases are physically equivalent. Over the successful
  `post_update` evaluation outcomes: **CLEAN** 70 episodes, 0 RTB commands, 0 deaths, mean
  unique target coverage **6.000 / 6**; **MILD** 70 episodes, **70** RTB commands, 0
  deaths, **5.957 / 6**; **SEVERE** 70 episodes, **43** RTB commands, **63** deaths,
  **5.700 / 6**. At the FINAL round all seven feasible clean worlds and all seven feasible
  mild worlds reach 6/6 with 0 deaths, while **every one of the seven feasible severe
  worlds loses one airframe** — five at reward ≈ `−0.375` with 6/6 coverage, and seeds
  **1000004** and **1000007** at ≈ `−0.541666` with 5/6 coverage. RTB yield is real Phase-2
  COMMAND HISTORY (`FuelDamageOutcome.rtb_command_issued`), never the executor's
  `rtb_issued` latch (§5).
  **REWARD AND THE THREE WITHIN-SEED DELTAS — over COMPLETE TRIADS ONLY.** Per-cell means
  are each over THAT cell's own successful subset, so subtracting two of them is not a
  matched effect; the only within-seed claims are the deltas below, each over `n = 7`
  complete triads.
  - `pre_update` (n = 7): clean `-0.49999970`, mild `-0.49999970`, severe `-0.87499991`;
    `mild − clean = 0.0`, `severe − clean = -0.37500021`, `severe − mild = -0.37500021`.
  - final `post_update` (n = 7): clean `5.714293e-07`, mild `5.714293e-07`, severe
    `-0.42261872`; `mild − clean = 0.0`, `severe − clean = -0.42261929`,
    `severe − mild = -0.42261929`.

  **PPO PRODUCTIVITY — training WAS productive.** **50 / 50 scheduled training iterations
  productive**, 0 zero-wake, 0 all-failed, `n_epochs_run = 4` in every iteration,
  `updates_completed = 50`, **1,405 transitions**. Training reward improved from
  `-0.51547582` to `-0.07738026`. **The correct reading is therefore precise:** actor-only
  PPO training was productive and improved overall performance, and it nevertheless did NOT
  produce the targeted held-out MILD-vs-SEVERE behavioural differentiation. **No claim that
  CTDE would fix this is made or supported here** — no CTDE benefit is measured by this run.
  **ARTIFACT COMPLETENESS — reported ALONGSIDE the scientific denominators, never in place
  of one.** **664 bundles and 664 manifests: 586 `complete`** (exactly the successful
  attempts) and **78 `incomplete`** (exactly the failed ones), **586 playbacks**, 2,520
  files, ≈ **4,430.6 MB**. **All 586 complete bundles reconcile expected against observed
  3 known / 3 hidden / 6 executed targets.** No incomplete bundle fabricated a playback,
  and no completed run left an unlisted one. **No path-related artifact failure occurred:
  the maximum actual artifact/playback path is 139 characters**, because this run
  deliberately used the short external root after the precursor proved the `MAX_PATH`
  hazard.
  **ENGINEERING CAVEAT — HISTORICAL FACT, NOT A CODE CHANGE AND NOT FIXED HERE.** The
  precursor proved that a BLADE playback-export failure can currently surface as an
  ORDINARY `run`-stage `EpisodeAttemptError` and therefore enter the SCIENTIFIC failure
  ledger: `graph_tick_loop.run_episode`'s `ctx.game.export_recording()` propagates into
  `graph_train._run_one_episode`, which does `raise EpisodeAttemptError("run", exc) from exc`.
  An artifact/serialization fault raised through `_VisualArtifactError` is INFRASTRUCTURE
  and aborts the run (§5); this one is not routed that way and was accounted as ordinary
  episode attrition. **The valid run had ZERO such failures**, so nothing about this
  measurement depends on it. It is recorded as measurement-infrastructure history and a
  future engineering caveat; **no code was changed for it in this record**, and changing
  that routing would be its own separately reviewed task.
  **INDEPENDENT REVIEW.** The GPT orchestrator independently reviewed the replacement
  measurement and issued **`APPROVE — VALID MEASUREMENT`**. The precursor's verdict is
  **`INCONCLUSIVE/BLOCKED — INVALID MEASUREMENT`**.
  **EVIDENCE SHA-256** (re-verified read-only against the preserved measurement root before
  being recorded here):
  `c.json=a3961058cc36b2e1b83199e87d0799d3d8042e4cd1966930328f40c451e0fa02`,
  `timing.json=cc4cab1dd718b80712b8d3e79ed2ecb453ac1a9eedbfa711277594cf6f7d96e0`,
  `console.log=6618acb98b0b77439ed2e494c0705df4cb4033f37ac767e9b7044084978435fd`,
  `r/run_config.json=3e43e54b7f3b0685e4770e3c75e6a57a523147460c2785d31c7f470b62202375`,
  `r/train_records.jsonl=a79b54e9c15c169872968a6b63f1699f4c9c7502d6cf61054f311317d1bea806`,
  `r/eval_records.jsonl=4a0d867872901bc778673ed574487abca600c0afdd3c70f64d17f3909c951aa5`,
  `r/episode_outcomes.jsonl=e47a4c35ece4349f870b961d42fe3a29b44cd4f064f793e675c022a7cb240239`,
  `r/episode_failures.jsonl=88d7869881b137696a2a4e8e921f8a2537db651165ec8183bb6990be3c5305e3`,
  `r/run_summary.json=f0ced3cd612b22f1160ff7bda38a2884df98b7fb559ffef44af737d0e1cf4d5e`,
  `r/plots/training_performance.png=86b5b689cbdd98a9cb271746539a0f90cbe52d25f4353ca6619d557e5191fd71`,
  `r/plots/policy_diagnostics.png=a9b24a239320fb17b8d9aad139d63a5fb1a317031c8557a6744bb2a10ef943ef`,
  `r/plots/measurement_health.png=76c6af6a56619701d09ed52fe02ae4fc0fc12f80fbbf36157a2214947ab8bbb1`.
  `run_summary.json:/severity_response` is DERIVED from `episode_outcomes.jsonl`
  (`severity_response_source`), which is the ONE metric path (§5).
  **PRESERVATION.** BOTH measurement trees — the VALID run root `C:\Users\Itama\f7r2` and
  the INVALID precursor
  `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640` — are preserved and must
  not be modified, moved, copied, repackaged, deleted or regenerated, and neither may any
  earlier preserved run.
  **SCIENTIFIC CONCLUSION.** A valid actor-only baseline of FD-VARIABLE-SEVERITY-v1 was
  obtained from the clean, no-communication graph-RL stack at measured code SHA
  `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`. The severity construction is PHYSICALLY REAL
  in this cell — mild and severe diverge sharply in RTB yield, airframe survival and target
  coverage — and actor-only PPO training was productive across 50/50 updates, improving
  training reward from `-0.51547582` to `-0.07738026`. **Nevertheless the deterministic
  held-out actor showed NO severity-conditioned FD-wake meta-action separation**: at
  `pre_update` and at the final `post_update` it chose `PLAN_COMPLIANCE` in all 7 completed
  MILD and all 7 completed SEVERE matched worlds, and across all ten `post_update`
  checkpoints the two per-severity distributions are identical. **THE EXPLICIT
  NON-CLAIMS:** this does **NOT** establish that the actor is broken, that training or PPO
  failed, that the actor ignores fuel entirely, that MILD "should" have chosen
  `PLAN_COMPLIANCE`, that 70 post-update observations per severity are 70 independent
  held-out worlds, that the finding generalizes beyond this fixed cell and this held-out
  seed band, or that centralized training would change it. Those are subsequent research
  questions, and **this is a valid negative result — not a defect, and not grounds for
  retuning, re-seeding or re-running.**

- `a6f3aa9` — **PHASE-B CTDE: the centralized critic during TRAINING only, with
  `actor_only` preserved as the default — CLOSED / APPROVED / MERGED.** Approved candidate
  SHA `a6f3aa9d62931994f416b2241fec4cfac3b018ec` (`2026-08-22 21:01:46 Asia/Jerusalem`),
  integrated by merge commit `8390d85c2072e9cbe984ce5f2731cef3a9b14985` (PR #30). The
  candidate was merged with a normal MERGE COMMIT and preserved as its SECOND PARENT
  (ordered parents: `d437084c5fb1a22c21596a48c58e03f7e15a0115`, then `a6f3aa9…`), and the
  integration tree is `9686c107b8864f00a7d4403d70faf42ab561d2fb`. **Grade A under
  `GPT_GITHUB`, implementation mode BUILD** — it created a new layer and a new module, and
  the SURGICAL mode belongs to the SEPARATE documentation-lock task that recorded it, never
  to the implementation itself. The technical contract is in §5 ("PHASE-B CTDE — the
  TRAINING-ONLY centralized critic"), the pipeline placement in §4 and the routing in §6.
  This entry records the LOCK, not the mechanism.
  **THE TWO IMMUTABLE REFERENCES, AND THEY ARE DISTINCT.**
  `pre-ctde-actor-only = d437084c5fb1a22c21596a48c58e03f7e15a0115` (tree
  `d7cc2dcb1b161180e272afc9600175f022c5b5d0`) is the NEW immutable reference preserving the
  IMMEDIATE PRE-CTDE actor-only state — it is the integration's FIRST parent, so "the
  actor-only state CTDE was merged onto" is a git fact rather than a claim. Preserving it
  was the CTDE integration gate's remaining prerequisite (§8), and it must not move.
  `phase-a-baseline = 4f0068847b017795717c5f0e331f647bcfc30547` is the SEPARATE, ORIGINAL
  Phase-A reference, is NOT repurposed as the pre-CTDE reference, and likewise must not
  move. Neither is the FD-VARIABLE-SEVERITY-v1 measured code SHA
  `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, which is a durable MEASUREMENT identity and
  never a code reference (its `APPROVE — VALID MEASUREMENT` record, with its NEGATIVE
  primary finding, is above and is UNCHANGED by this lock).
  **APPEND-ONLY REVIEW CHAIN, two commits on one branch and one PR** — never amend, rebase,
  squash, force-push or history rewrite. Initial reviewed candidate
  `d70d07f829a44e6f19100c338d4dde89f4f47bf6` (`2026-08-22 20:02:21 Asia/Jerusalem`) carried
  the implementation; the review correction landed as the NEW CHILD COMMIT `a6f3aa9…`,
  which is the APPROVED head. The three findings and their closure:
  (F1) `training_mode='ctde'` accepted `value_coeff == 0`. Such a run would build central
  observations and take its advantages from a critic it never trains, leaving the baseline
  a frozen random function — neither the `actor_only` reference algorithm nor the approved
  CTDE one, and recorded and read as CTDE either way. `TrainConfig.validate` now REQUIRES
  `> 0`, refused before any compute; the default is unchanged at `0.5`, `actor_only`
  validation is untouched, and `value_coeff` is still NOT a mode selector.
  (F2) CTDE training records did not persist the critic's diagnostics. `value_loss`,
  `value_mean`, `value_target_mean` and `critic_grad_norm` are now copied straight out of
  the dict `CTDEUpdater.update` already returned — never recomputed — and added ONLY when
  `ctde_enabled`, so an `actor_only` record is byte-unchanged with those keys ABSENT rather
  than null. No actor-side metric changed meaning.
  (F3) `ValueHead`'s docstring claimed a small init giving an initial value function "~0
  everywhere" while the code passes `std=1.0`. **PROSE ONLY — THE INITIALIZATION IS
  UNCHANGED**: `graph_ppo.py`'s runtime token stream (comments and docstrings stripped) is
  IDENTICAL to the initial candidate's.
  **REVIEWED SCOPE: EXACTLY SIX FILES**, verified as the complete
  `d437084…...8390d85…` comparison —
  `src/match_aou/rl/observation/central_graph_builder.py` (new),
  `src/match_aou/rl/training/graph_ppo.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_train.py`, `tests/test_graph_ctde.py` (new) and
  `tests/test_graph_ppo.py`. **NO documentation file was part of the code integration** —
  that is what this documentation task closes. No vendored BLADE, solver, `graph_reward`,
  `graph_fuel_damage`, encoder, action-space, episode-setup, hidden-placement, generator,
  scenario, config or preset file was touched, and the Phase-A cell, the seed schedules,
  the evaluation design, the failure taxonomy and every preserved run artifact are
  unchanged.
  **PROOF SURFACE** (`tests/test_graph_ctde.py`, and the import/dependency contract in
  `tests/test_graph_ppo.py`): actor-only preservation under a POISON that raises at every
  central-CTDE construction site, with a CONTROL proving the poison fires under `ctde`;
  disjoint actor/critic parameter sets; each backward leaving the other side's gradients
  `None`; the actor advantage a detached scalar; privileged central features unable to move
  an actor logit; a central state rejected as an actor observation; the `NO_EGO_INDEX`
  role symmetry; evaluation constructing neither critic nor recorder; a CTDE-trained actor
  running with the critic absent; hand-computed GAE, the zero terminal next value, the
  per-episode boundary and value targets fixed across epochs; `baseline` proven to be the
  mean episode REWARD and not the critic value; zero-wake handling; loud failure on
  misaligned central samples; variable graph sizes finite without padding; the exactly-five
  actor-only checkpoint keys beside the CTDE payload; and the persisted critic diagnostics
  with their absence on the `actor_only` path.
  **CC-REPORTED ENGINEERING EVIDENCE — IMPLEMENTATION VALIDATION, NOT SCIENTIFIC
  EVIDENCE. It has TWO parts, and they are labelled separately because they are different
  kinds of evidence.**
  (i) **TESTS — solver-free, stubbed engine seams.** At the approved head: full solver-free
  suite **334 passed, 4 skipped**; `tests/test_graph_ctde.py` **43 passed**;
  `tests/test_graph_train.py` **119 passed**; `tests/test_graph_ppo.py` **18 passed**; the
  standalone `nlp_env` CTDE `__main__` runner **43 passed**; `git diff --check` clean. Four
  mutation checks confirmed the fix-commit regressions falsify (the permissive
  `value_coeff` bound, dropped persistence, recomputed-instead-of-copied values, and CTDE
  keys leaking onto the `actor_only` path), each reverted.
  (ii) **BOUNDED ENGINEERING SMOKES — REAL BLADE AND REAL BONMIN, and they DID happen.**
  During the BUILD candidate's validation, **TWO bounded smokes under `nlp_env` ran BOTH
  training modes end-to-end against the real engine and the real solver**: 2/2 episodes,
  one PPO update, `accounting_reconciled = true`, no `CRASH` and no `Traceback`, writing
  only to the scratchpad and never into the repository. They are ENGINEERING evidence that
  the wiring executes, and they are what surfaced the `baseline`-vs-critic-value defect the
  contract now pins (§5). **Their rewards and episode outcomes are NOT scientific evidence
  and must never be promoted into any**, and the later append-only review-fix validation
  needed no new run of them.
  **NO SCIENTIFIC MEASUREMENT OF ANY KIND WAS EXECUTED FOR PR #30** — no baseline, no
  probe, no scientific rollout, and above all **no actor-only vs CTDE comparison. NO CTDE
  benefit is established or may be pre-claimed.** Two bounded engineering smokes are not a
  measurement: they have no scientific contract, no seed schedule, no held-out band and no
  denominator. This lock certifies the IMPLEMENTATION; §8 owns the gate and the next
  scientific task.

---

## 8. OPEN (not built)

- **PHASE A IS CLOSED. A SCIENTIFICALLY VALID LONG-BASELINE MEASUREMENT OF THE FUEL-DAMAGE
  CELL EXISTS (measured code SHA `737b4bf`, §7). THE ADDITIONAL ACTOR-ONLY
  FD-VARIABLE-SEVERITY-v1 BASELINE IS NOW ALSO EXECUTED, INDEPENDENTLY REVIEWED AND
  `APPROVE — VALID MEASUREMENT` (measured code SHA `bf1e045f`, §7) — WITH A NEGATIVE
  PRIMARY FINDING. AND PHASE-B CTDE IS NOW IMPLEMENTED, REVIEWED AND MERGED
  (`a6f3aa9` / `8390d85`, PR #30, §5 and §7) — SO THE NEXT SCIENTIFIC TASK IS THE FIRST
  CONTROLLED ACTOR-ONLY vs CTDE COMPARISON ON THE LOCKED ORIGINAL PHASE-A CELL, WHICH HAS
  NOT BEEN RUN AND FOR WHICH NO BENEFIT IS CLAIMED.** The variable-severity measurement ran
  on an immutable DETACHED snapshot while CTDE design and implementation proceeded beside
  it in a separate writable task branch. **The earlier serial claim — that CTDE may begin
  ONLY AFTER that measurement — was SUPERSEDED on 2026-08-22, and the CTDE INTEGRATION gate
  is now SATISFIED AND CLOSED ON BOTH HALVES**: the measurement-validity half by that
  `APPROVE — VALID MEASUREMENT` verdict, and the reference half by
  **`pre-ctde-actor-only = d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the FIRST parent of
  the CTDE integration, which must not move (`phase-a-baseline` remains the SEPARATE
  original Phase-A reference and was never repurposed). See the research-ordering bullet
  below for the historical arrangement, and the CTDE bullet below it for the live state.
  Difficulty selection is CLOSED
  (below) and FD-BASELINE-v1 is merged and locked (`a8669f4`, §7), so the open question was
  never *what* to build but *how the built cell behaves* — and for THIS cell that question is
  now ANSWERED by the approved rerun
  `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, whose full record,
  denominators, explicit NON-CLAIMS and evidence hashes are in §7. **FOUR runs of the
  LEGACY FD-BASELINE-v1 cell exist. The first three are NOT valid measurements and survive
  as HISTORY ONLY:**
  - **First short probe — `training_output_20260815_173029`**, from clean `main` at
    `238062d7d284334432d9c39d7543fb0bbf39ea7c`. It established HARNESS AND ACCOUNTING
    OPERABILITY ONLY and exposed three research-validity defects (next bullet), so **its
    reward numbers are NOT scientific evidence about the fuel-damage cell** and remain
    historical evidence about the PRE-CORRECTION behaviour.
  - **Corrected short-probe rerun — `training_output_20260816_162130`**, from a clean
    checkout at exact code SHA `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, one invocation of
    the reviewed preset through `--config`, native exit code 0, `cli_overrides = []`. It was
    originally reviewed as `VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS`; **that verdict
    is SUPERSEDED** by `INCONCLUSIVE — LATER ROSTER/DATA-INTEGRITY REVIEW INVALIDATED THE
    SCIENTIFIC DENOMINATOR` (§7). Its own ledger accounts clean train seed 4 as a `setup`
    `EpisodeRosterError`, and the approved roster-integrity correction (`36365f2`)
    establishes that such a fault is a measurement/data-integrity failure that must ABORT —
    not an episode outcome that may shrink a denominator. **Its reward and performance
    numbers are therefore no longer scientific evidence**, and the claim that it PASSED or
    permanently released the long-baseline validity gate is WITHDRAWN. What it DOES still
    establish is preserved: its run identity, provenance, mechanical accounting, artifact
    completeness, and the OPERATIONAL WITNESSING of all three defect corrections in real
    playback.
  - **First long baseline — `training_output_long_baseline_100x8_seed0`**, from exact code
    SHA `c30b6982ba605d60976cc303256da4b5528b0e63`, one invocation with `cli_overrides = []`
    and native exit code 0. **EXECUTED, independently reviewed, engineering `REQUEST
    FIXES`, scientific `INCONCLUSIVE — ROSTER/DATA INTEGRITY FAILED`** (§7 owns the full
    record, contract, denominators, failure breakdown and evidence hashes). 1,136 scheduled
    attempts, 860 successful, 276 failed, `accounting_reconciled = true` and 100 PPO
    updates — and **143 training attempts were nevertheless destroyed by the roster defect
    across 83 iterations while the run reported itself healthy**, with 11 `complete`
    manifests claiming a five-target world their own authoritative executed-t0 scenarios
    contradicted. **Do not report its reward, paired deltas, survival, fuel-damage yield or
    PPO performance as scientific evidence**; they are raw historical outputs only. It is
    preserved and must not be modified, moved, repackaged, deleted or regenerated.
  - **Phase-A long baseline (RERUN) — `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`**,
    from a clean checkout at exact code SHA `737b4bfdfa083b0b8f59e8e4274b719a34ab78fc`, ONE
    invocation of the preserved measurement contract through `--config`, native exit code 0,
    `cli_overrides = []`. **EXECUTED, independently reviewed, `APPROVE — VALID
    MEASUREMENT`.** **THIS IS THE AUTHORITATIVE MEASUREMENT OF THE CELL** and the only one
    whose reward, paired-delta, survival, event-yield and PPO numbers are scientific
    evidence. §7 owns the full record; the headline is 1,136 scheduled / 993 successful /
    143 accounted episode failures with `accounting_reconciled = true` and ZERO
    infrastructure or data-integrity faults, 100/100 productive PPO updates, a matched
    paired delta of `−0.375000 → −0.071429` over a structural 7/8 pairs, and evaluation
    deaths `7 → 0`.
  **TWO FURTHER runs exist on the VARIABLE-SEVERITY design, and they measure a SEPARATE
  cell — never a rerun, replacement or extension of the four above:**
  - **Invalid variable-severity precursor —
    `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640\training_output_fd_variable_severity_v1_50x8_seed0`**,
    at exact code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`. **`INCONCLUSIVE/BLOCKED —
    INVALID MEASUREMENT`.** A Windows `MAX_PATH` playback-export failure (267-character
    recording path) produced 70 `run`-stage `FileNotFoundError`s that were ALL
    `post_update` SEVERE members — 10 rounds × the 7 feasible held-out seeds — so the
    entire post-training severe arm, the arm the experiment exists to measure, was
    systematically removed. **That is an infrastructure failure, NOT a negative scientific
    result.** Preserved as engineering history only; §7 owns its hashes.
  - **FD-VARIABLE-SEVERITY-v1 actor-only baseline (REPLACEMENT) — run root
    `C:\Users\Itama\f7r2`**, same exact code SHA
    `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, ONE invocation of a contract differing from
    the precursor's in `output_dir` ALONE, native exit code 0, `cli_overrides = []`.
    **EXECUTED, independently reviewed, `APPROVE — VALID MEASUREMENT`.** **THIS IS THE
    AUTHORITATIVE MEASUREMENT OF THE VARIABLE-SEVERITY CELL.** §7 owns the full record; the
    headline is **664 scheduled / 586 successful / 78 accounted `setup` episode failures**
    with `accounting_reconciled = true` and ZERO infrastructure or data-integrity faults,
    **7/8 complete matched triads in all 11 rounds**, 50/50 productive PPO updates — and a
    **NEGATIVE primary finding: NO severity-conditioned FD-wake meta-action separation**,
    with MILD and SEVERE both at `PLAN_COMPLIANCE 7/7` at `pre_update` and at the final
    `post_update`, and identical `63/70` vs `7/70` distributions across all ten
    `post_update` rounds. Physical outcomes nevertheless DIVERGE sharply, so the severity
    factor itself is real. Its numbers are evidence about the VARIABLE-SEVERITY cell only,
    never about the Phase-A legacy cell.
  **What this establishes and what it does not.** The roster/world-truth defect was closed in
  CODE (`36365f2`, integrated `f37ea1c`, PR #24 — §5, §7), the authorized rerun was then
  executed ONCE on the SAME scientific contract into a NEW output directory, and it PASSED
  the validity gate. **Phase A is therefore CLOSED**, and **the long baseline is NOT to be
  re-run, resumed, repaired, extended or re-tuned** — a valid measurement exists, and
  re-running it would not make it more valid. The approved result establishes end-to-end
  learnability and meaningful ego-local runtime adaptation in the LOCKED Phase-A reference
  cell. It establishes **NO global optimality, NO monotonic convergence, NO generalization
  beyond this fixed cell and this held-out seed set, and NO benefit from centralized
  training** (§7 states these non-claims in full, and they must be carried forward verbatim
  in meaning). The interpretation rules survive unchanged: a held-out mean is never read
  without its denominator; an all-failed batch reports `null`, never `0.0`; an empty
  successful-pair population is `null` too; the held-out per-condition means are each over
  their own successful subset, so the within-seed claim is the matched-pair delta over
  COMPLETE pairs alone; and `graph_reward` remains FROZEN unless a separately reviewed p<1
  design requires an explicit reward-contract change. The invalid old long run may be
  compared against **only as ENGINEERING evidence** — never as a scientific baseline.
  **The B2 exact-cardinality and fuel-window failures BOTH long baselines recorded — 101 and
  32 in the invalid first run, 101 and 42 in the approved rerun — are NOT corrected by
  `36365f2`, and they are NOT defects.** They remain EXPECTED SCIENTIFIC
  OUTCOMES under the current contract — `skip_and_account_v1` attempts each seed once,
  records it once and reports the smaller successful population next to its denominator (§5,
  and the exact-cardinality bullet below) — and they must not be relaxed, retried, retuned
  or reclassified. Only the ROSTER fault changed category, because only it was an instrument
  defect.
  **The deferred over-safety observation is a HYPOTHESIS, not a defect and not a semantic
  change.** The corrected rerun's playback shows a B-2 at 50.07 km from a second hidden
  target — against the 50 km `DETECTION_KM` threshold — on the sampled frame before it
  enters RTB, and the preserved artifact does not persist that wake's selected meta-action,
  so the attribution is PLAUSIBLE BUT NOT PROVEN (§7). It is recorded as a future research
  hypothesis about policy calibration, relevant to a later variable-FD-severity experiment.
  **It opens no defect, changes no reward, retunes no policy, and it is not what
  invalidated either measurement** — the roster/data-integrity fault is. It neither blocked
  nor shaped the Phase-A long baseline, and it remains a deferred hypothesis for a later
  variable-FD-severity experiment.
  **The harness every one of these runs used is MERGED and LOCKED** (`61e539e`, §7): driven
  through `--config` — the SHORT PROBES by the repository preset
  `configs/graph_train/final_cell_probe.json`, the TWO LONG BASELINES by their own
  measurement contract, which is deliberately NOT a repository preset — and writing
  `run_config.json` (with `provenance` and a structured `config_source`), the three jsonl
  records, `run_summary.json`, `scenarios/`, `checkpoints/` and the three figures under
  `plots/`. Two reading rules survive unchanged: the PRESET schedules TWO ITERATIONS, which
  does not guarantee two PRODUCTIVE PPO updates (`updates_completed` may be 0, 1 or 2 —
  the corrected rerun measured 2, but that yield is always a MEASUREMENT, never an
  assumption); and the held-out per-condition means are each over their own successful
  subset, so the within-seed claim is the matched-pair delta over COMPLETE pairs alone
  (§5). What counts as a VALID measurement — as opposed to a favourable one — is stated in
  the handoff: a run that produces no reward improvement, or no productive update, is a
  valid NEGATIVE observation, not a technical failure. A run whose DENOMINATOR was corrupted
  by an instrument defect is the opposite case — not a negative result but no result, which
  is exactly why `MeasurementIntegrityError` now aborts instead of being accounted (§5).
  All four runs used `--visual-artifacts` (§5, `24d1835`; the repository preset enables
  it, and both long-baseline contracts set it explicitly). It preserves each attempt's
  known-only scenario, executed t=0 scenario and BLADE playback for inspection, and it is an
  observation surface only: enabling it neither authorizes a run nor changes anything a run
  measures, and artifact completeness is reported ALONGSIDE the scientific denominators,
  never in place of one. **It is also what made the roster defect provable** — the preserved
  authoritative executed-t0 scenarios are what showed the full six-target world behind every
  under-counted roster.
  *Historical, and about the EASY PRE-FD CELL only:* the clean-code probe at
  `a3f0838616990987bcb8a51665fa75d84edf5952` measured pre-update headroom
  (`-0.4999997395829586`, 4/4), train yield 7/8 with one accounted seed-2 `setup` failure,
  24 transitions, two productive PPO updates and a final held-out numerical zero
  (`5.000007394910353e-7`, 4/4). That cell had no difficulty factor; **those numbers are
  not evidence about the fuel-damage cell** and must not be reused as its baseline.
- **The three research-validity defects the first short probe exposed — DEFECTS A, B and C
  are ALL CLOSED: implemented, reviewed, approved and merged.** *Historical workflow
  context, kept because it is how these three were run and is not a new prohibition:* the
  SEQUENTIAL-DEFECT POLICY made the DEFAULT breakdown A, then B, then C — each its own
  separately reviewed, separately locked task — with the probe rerun a task of its own,
  never folded into a defect fix; bundling two defects required FOCUSED RECON proving
  them technically INSEPARABLE plus an explicit GPT / user decision, and was never
  authorized. All three closed under that default sequence.
  - **Defect A — `SELF_PRESERVATION_ABORT` was node-scoped, not an ego-global abort:
    CLOSED / MERGED / APPROVED.** Approved `d56fda6`, integrated by `f094e0b` (PR #17) —
    the lock and its evidence are in §7, the contract in §5 Stages 4 and 5. It changed
    abort SEMANTICS only; the `k × 3` action surface, PPO, reward, fuel-damage mechanism
    and BLADE are untouched.
  - **Defect B — PREMATURE ATTACK RE-FIRE EXHAUSTED WEAPONS: CLOSED / APPROVED /
    MERGED.** Approved `39a16f2`, integrated by `60a82d1` (PR #19) — the lock and its
    evidence are in §7, the contract in §5 (Execution, Stage 1) and the routing in §6.
    *The defect, historically:* `GraphPlanExecutor` armed a FIXED 60-tick confirmation wait
    for every salvo (`kill_confirm_ticks` was constructor-configurable but no caller passed
    it), so a slower salvo still in flight could let that wait expire and a redundant second
    salvo consume the last weapons — measured in the first short probe's `post_update`
    damaged eval seed `1000003`, where a B-2 reached its final known target with ZERO
    onboard weapons and then loitered to fuel exhaustion. *The merged correction:* the wait
    is DERIVED per salvo from the ACTUAL auto-selected live weapon and the CURRENT
    engagement distance, with the configured value kept as its FLOOR and FALLBACK — the
    default was NOT merely raised. Lethality, the two-argument attack command and the
    FROZEN vendored BLADE engine are unchanged, and a probabilistic-miss /
    weapons-exhaustion redesign remains OUT of scope.
  - **Defect C — RTB ISSUANCE is not physical RTB COMPLETION: CLOSED / APPROVED /
    MERGED.** Approved `ea62e4e`, integrated by `0de9f21` (PR #21) — the lock and its
    evidence are in §7, the contract in §4 and §5 (Execution, Stage 1, and the tick loop)
    and the routing in §6.
    *The defect, historically:* `GraphPlanExecutor.is_done()` treated the `rtb_issued`
    lifecycle LATCH as RTB-resolved and `run_episode` stopped when it became true, so an
    episode could end while the aircraft was still airborne — measured in the first short
    probe's `post_update` damaged eval seed `1000000`, which recorded `dead=0` and reward 0
    for an ego that could not physically reach home. *The merged correction:*
    `is_done(observation)` requires the LIVE post-step observation; assignment completion
    still comes from executor semantic state, while the PHYSICAL half comes from
    `_physical_state` (airborne / landed / removed) and `_note_dead` reconciles a death on
    the ride home into `executor.dead` before the verdict, so the unchanged reward formula
    receives the truthful terminal loss. `rtb_issued` keeps its ONE job as the single-issue
    toggle guard, and an ego that has committed to return leaves Phase 1 while peers
    continue. The vendored BLADE engine is unchanged.
  **Gating, as it now stands:** the vendored BLADE engine stays FROZEN unless separately
  authorized (§2). The ONE authorized corrected-cell short-probe rerun **HAS been executed
  AND independently reviewed** — `training_output_20260816_162130` at
  `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` — and all three of these defects are
  OPERATIONALLY WITNESSED in its real playback, not only in proof tests. **That witnessing
  survives intact**; what did NOT survive is that run's scientific verdict, which a LATER
  roster/data-integrity review superseded (§7, and the first §8 bullet above). The distinction
  is exact: Defects A, B and C are about what the SIMULATION did, and the roster defect is
  about which targets the MEASUREMENT counted. **A FIRST long baseline was run and was
  scientifically INCONCLUSIVE for that separate reason; the roster defect was then corrected
  and the authorized rerun PASSED the validity gate**, so the cell now has a valid
  measurement (`737b4bf`, §7) and nothing about these three defects is outstanding. Neither
  short probe's numbers, and none of the invalid first long baseline's numbers, are a
  baseline expectation — the approved rerun is the only scientific baseline.
- **A FOURTH, SEPARATE defect — the ROSTER read an ALLOCATION as a WORLD INVENTORY:
  CLOSED / APPROVED / MERGED.** Approved `36365f2`, integrated by `f37ea1c` (PR #24, tree
  `f8015380`) — the lock and its evidence are in §7, the contract in §5 (the Stage-0
  "WORLD INVENTORY IS NOT ORACLE ALLOCATION" block and the roster-integrity block), the
  routing in §6. **It is NOT a regression in Defects A, B or C** — their corrections remain
  merged, witnessed and untouched. *The defect, historically:* `_episode_target_roster`
  answered "which targets does this episode contain?" from `ctx.beliefs` and
  `ctx.oracle_tasks`, both ALLOCATED-ONLY by `solve_and_normalize`'s contract, so any target
  the solver left unselected was missing from the roster while still in the world the
  executor flew through — and the episode was then FAILED for that self-inflicted
  discrepancy AS AN ACCOUNTED `setup` FAILURE. *The merged correction:* the world comes from
  the two RAW pre-solve snapshots `EpisodeContext.known_target_ids` / `executed_target_ids`;
  the beliefs are a SUBSET constraint, not a denominator; `_require_scheduled_cell` checks
  the scheduled cell before anything is paid for; and a roster/world-integrity fault is a
  `MeasurementIntegrityError` that ABORTS the run as INFRASTRUCTURE instead of shrinking a
  scientific denominator. Reward, PPO, the oracle allocation, fuel damage, B2, the seeds,
  the schedules, the tick loop, the executor, the generator and FROZEN BLADE are all
  unchanged. **Consequence, and it is the reason this bullet exists:** the two measurements
  of the merged cell that PRECEDED this correction are permanently and scientifically
  INCONCLUSIVE (§7). The correction is what made a sound measurement possible: the rerun that
  followed it PASSED the validity gate and is the cell's valid baseline (`737b4bf`, §7).
- **Complete Git provenance is REQUIRED for a real training run (`1b48145`).** `train`
  raises before policy, generator, episode or optimizer work unless BOTH the full commit SHA
  and the clean/dirty verdict were determined, so a run cannot be launched from a checkout
  where `git` is unavailable, times out, or cannot read the index. A dirty tree is a
  hazard, not a blocker: it WARNS and runs. Consequence for tooling: anything driving
  `train` outside a working checkout must inject the verdict (the tests patch
  `_git_provenance`) rather than expect it to be optional.
- **RESEARCH ORDERING — the 2026-08-22 PARALLEL arrangement, NOW FULLY TRAVERSED.**
  The variable-severity MEASUREMENT and Phase-B CTDE were run in parallel by explicit
  user/orchestrator decision — **not** an accidental Phase-A reopening, **not** a
  correction of anything, and **not** a change to any technical CTDE contract. **All four
  items are now COMPLETE**: the measurement is EXECUTED, independently reviewed and VALID,
  and the CTDE integration gate is SATISFIED AND CLOSED. This bullet is therefore the
  arrangement's HISTORICAL RECORD; the live research state is the CTDE bullet below. The
  approved order was:
  1. **PRESERVE the original Phase-A reference baseline.** It is CLOSED, VALID and
     IMMUTABLE — measured code SHA `737b4bf` on the FD-BASELINE-v1 design, run
     `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` (§7). The branch
     `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) preserves the code
     state and must not move. **Nothing below redefines, reopens, re-runs, extends or
     supersedes it.**
  2. **IMPLEMENT (done) and MEASURE (DONE — EXECUTED, REVIEWED, `APPROVE — VALID
     MEASUREMENT`) the ADDITIONAL actor-only FD-VARIABLE-SEVERITY-v1 baseline — ON A
     PINNED, IMMUTABLE, DETACHED SNAPSHOT.** The code is merged and locked (`eecc9b5`,
     integrated `177e969`, PR #27 — §5, §7). The measurement was LOCKED to exact SHA
     `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
     `dd881478b8e2e521054d09bc865437f1308be1a2`, executed from a DETACHED, clean snapshot
     worktree that carried no task branch and was READ-ONLY with respect to the shared
     repository. **§7 now owns the authoritative record**: the valid replacement run, the
     excluded `MAX_PATH` precursor, every denominator, and the NEGATIVE primary finding
     (no severity-conditioned FD-wake meta-action separation). It is, and remains, an
     **ACTOR-ONLY measurement OF THAT PINNED SHA**: repository work landing after
     `bf1e045f…` — Phase-B CTDE included — is simply not in the measured tree, so it can
     neither be attributed to that run nor contaminate it. **Nothing beyond §7's record
     may be claimed for it**, and its negative finding is a valid result, not a defect.
  3. **PHASE-B CTDE DESIGN AND IMPLEMENTATION PROCEEDED CONCURRENTLY** in a separate
     writable task branch / worktree, ungated on the measurement's completion. **DONE:**
     approved candidate `a6f3aa9`, integrated `8390d85`, PR #30 (§5, §7).
  4. **THE CTDE INTEGRATION GATE — SATISFIED AND CLOSED ON BOTH HALVES.** The
     measurement-validity half was met by `APPROVE — VALID MEASUREMENT` at measured code
     SHA `bf1e045f` (§7) — by a NEGATIVE result, which satisfies the gate exactly as a
     positive one would, because the gate is about VALIDITY, never about a favourable
     outcome. The second half — a NEW immutable actor-only pre-CTDE reference preserved
     from the then-current actor-only state — was met by **`pre-ctde-actor-only =
     d437084c5fb1a22c21596a48c58e03f7e15a0115`**, the FIRST parent of the CTDE integration,
     which must not move. The existing branch `phase-a-baseline`
     (`4f0068847b017795717c5f0e331f647bcfc30547`) remains historical provenance for the
     ORIGINAL valid Phase-A reference, was NEVER moved and was NOT repurposed as that new
     reference.
  **OWNERSHIP — HISTORICAL.** While the two ran in parallel the CTDE GPT orchestrator was
  the SOLE WRITABLE repository owner and the FD measurement orchestrator was READ-ONLY on
  its detached snapshot. The user's ONE-TIME writable exception for the FD closure record
  ENDED when that record was integrated, and writable repository ownership RETURNED to the
  CTDE GPT orchestrator, which has since integrated PR #30. Every orchestrator resolves
  live branch and PR state from GitHub itself.
  **THIS SUPERSEDED THE SERIAL ORDER THIS BULLET ITSELF PREVIOUSLY STATED** — that CTDE
  design could begin only after the variable-severity measurement was executed and
  independently reviewed. That serial rule is HISTORY as of 2026-08-22 and must not be
  restated as live; the measurement has since completed and been reviewed VALID anyway. **It also still supersedes the two ORIGINAL ordering claims** that
  Phase-B CTDE was immediately next and that a stochastic/partial fuel-degradation variant
  was deferred until AFTER Phase B. FD-VARIABLE-SEVERITY-v1 is that variant's approved
  form, it is an ADDITIONAL actor-only stress baseline rather than a replacement for the
  Phase-A reference, and its PURPOSE is unchanged: the actor-only response to a
  survivable-vs-unsurvivable loss is measured independently of centralized training — which
  a pinned, detached snapshot preserves exactly, and which is precisely why the two may now
  run at the same time. **`p(destroy) < 1`, SAMs and dense reward are UNAFFECTED and remain
  separate, still-deferred future research changes** (see the difficulty-selection bullet
  below); none of them is part of FD-VARIABLE-SEVERITY-v1 and none may be bundled into its
  baseline.
- **THE AUTHORIZED MEASUREMENT CONTRACT — the bounded actor-only FD-VARIABLE-SEVERITY-v1
  baseline. EXECUTED ON THE PINNED IMMUTABLE SNAPSHOT; INDEPENDENTLY REVIEWED
  `APPROVE — VALID MEASUREMENT`; §7 OWNS THE RESULT.** The measurement was LOCKED to exact
  SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b`, tree
  `dd881478b8e2e521054d09bc865437f1308be1a2`, in a DETACHED, clean snapshot worktree that
  was READ-ONLY with respect to the shared repository. **This bullet remains the run SHAPE
  and nothing more** — the executed run's identity, denominators, evidence hashes and its
  NEGATIVE primary finding live in §7, and no number may be quoted from here. Because the
  snapshot was pinned and detached, later `main` work — Phase-B CTDE included — lies
  outside the measured tree and can neither be attributed to that run nor contaminate it.
  **The measurement is NOT to be re-run, resumed, repaired, extended or re-tuned:** a valid
  measurement exists, and a NEGATIVE finding is a result, not a reason to run it again. The
  approved shape, which the executed run followed exactly:
  - **50 scheduled training iterations × 8 scheduled training attempts = 400 scheduled
    training attempts**, `base_seed = 0`;
  - **evaluation every 5 iterations, INCLUDING the initial `pre_update` round ⇒ 11
    evaluation rounds**;
  - **8 fixed held-out seeds in the EXISTING eval band**, each evaluated as a matched
    **clean / mild / severe TRIAD** ⇒ **11 × 8 × 3 = 264 scheduled evaluation attempts**;
  - **664 scheduled attempts in total; NO early stopping.**
  Training runs `fuel_damage_mode = seeded_variable` at the approved
  **0.50 clean / 0.25 mild / 0.25 severe** distribution (§5). Everything else is the
  LOCKED cell: 3 agents, 3 known + 3 hidden, 200 km / 100 km geometry,
  `DETECTION_KM = 50`, `include_sams = false`, `probability = 1`, frozen solver and BLADE,
  unchanged `graph_reward` formula with `aircraft_penalty_coeff = 2.25`, unchanged PPO.
  The run task chose a FRESH, NON-OVERWRITING output directory and captured its own
  provenance; §7 records which. The interpretation rules carry over unchanged and are not
  optional: the PRIMARY behavioural evidence is the severity-conditioned FD-WAKE
  meta-action response with its own FD-wake denominators; a mean is never read without its
  denominator; the only within-seed claims are the three deltas over COMPLETE triads; an
  empty population is `null`, never `0.0`; and a run that shows no reward improvement, no
  productive update, or **no severity-conditioned behavioural difference — which is what
  this one measured** — is a valid NEGATIVE observation, not a technical failure. **"Mild
  must choose `PLAN_COMPLIANCE`" is NOT a correctness criterion** (§5). One further rule
  the executed run makes concrete: the ten `post_update` rounds REUSE the same seven
  feasible held-out seeds, so 70 observations per severity are a TRAJECTORY across
  checkpoints and **not 70 independent worlds** — the clean statistical unit for the final
  policy is the final round's 7 complete triads (§7).
- **Centralized critic / value head (CTDE) — PHASE B. IMPLEMENTATION CLOSED: REVIEWED AND
  MERGED. THE SCIENTIFIC COMPARISON IS THE ONLY PART STILL OPEN.** (Phase A is closed by
  the valid baseline, §7; the variable-severity factor is merged and its actor-only
  baseline is EXECUTED and independently reviewed `APPROVE — VALID MEASUREMENT` at measured
  code SHA `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` — with a NEGATIVE primary finding,
  §7.)
  **THE GATE IS SATISFIED AND CLOSED, on both halves.** The measurement-validity half was
  satisfied by that variable-severity verdict — satisfied by a NEGATIVE result, which
  counts exactly as a positive one would, because the gate tests VALIDITY and never
  favourability. The remaining half, preservation of a NEW immutable actor-only pre-CTDE
  reference, was satisfied by **`pre-ctde-actor-only =
  d437084c5fb1a22c21596a48c58e03f7e15a0115`** (tree
  `d7cc2dcb1b161180e272afc9600175f022c5b5d0`), the FIRST parent of the CTDE integration —
  so it is provably the actor-only state CTDE was merged onto, and it must not move.
  `phase-a-baseline` (`4f0068847b017795717c5f0e331f647bcfc30547`) is the SEPARATE ORIGINAL
  Phase-A reference, was never repurposed for this, and likewise must not move.
  **THE IMPLEMENTATION IS DONE AND IS NO LONGER AN OPEN DESIGN QUESTION.** Approved
  candidate `a6f3aa9`, integrated `8390d85`, PR #30 (§7), and locked as a §5 contract by
  this record. Phase B now has TWO SELECTABLE TRAINING MODES — `actor_only` (the DEFAULT
  and the preserved reference path) and `ctde` — chosen by `TrainConfig.training_mode`. The
  size-agnostic value estimator off `GraphEncoder.pool()` EXISTS (`ValueHead` on
  `CentralCritic`'s own encoder instance); the privileged critic inputs and their
  exclusions are ENUMERATED in §5; actor/critic separation, the training-only boundary,
  capture timing, GAE/value semantics, checkpoint distinction and actor-only byte-invariance
  are all IMPLEMENTED AND PROVEN in `tests/test_graph_ctde.py`. **Do not restate any of
  these as open requirements, do not re-enter a design/recon step, and do not rebuild what
  is merged.** Changing any of them is a Grade-A change to a locked layer, routed through §6.
  **WHAT IS STILL GENUINELY OPEN: the FIRST CONTROLLED ACTOR-ONLY vs CTDE COMPARISON. IT
  HAS NOT BEEN RUN.** Engineering tests, module `_selftest`s, a passing suite and a merged
  implementation measure NOTHING scientific. **No CTDE benefit may be pre-claimed** — not
  from the approved Phase-A result, which explicitly does not establish one; not from the
  executed variable-severity baseline, which measured no CTDE anything and whose negative
  severity finding is **NOT** evidence that centralized training would change it; and not
  from PR #30's implementation evidence. A CTDE claim requires its own executed,
  independently reviewed comparison. The next scientific task is preparing and executing
  that comparison, and it must:
  - **TAKE ITS ACTOR-ONLY ARM FROM THE ALREADY-APPROVED PHASE-A BASELINE, WHICH IS NOT
    RE-RUN.** That arm is already measured — measured code SHA `737b4bf`, run
    `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf`, `APPROVE — VALID
    MEASUREMENT` (§7) — and it is PRESERVED and NOT to be re-run, resumed, repaired,
    extended or re-tuned. **Nothing in this record authorizes a fresh actor-only run**; a
    newly executed actor-only CONTROL arm is a SEPARATE research-design decision requiring
    explicit user authorization. What the task schedules is the **CTDE arm**;
  - **MATCH THE LOCKED ORIGINAL PHASE-A SCIENTIFIC CELL** — 3 agents, 3 known + 3 hidden,
    200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams = false`,
    `probability = 1`, frozen solver and BLADE, unchanged `graph_reward` formula with
    `aircraft_penalty_coeff = 2.25` — **and that baseline's training / evaluation schedule,
    seed policy, held-out band and evaluation construct** as the authoritative Phase-A
    record establishes them, judged under the SAME validity gate, VALIDITY BEFORE
    PERFORMANCE;
  - **name the EXPERIMENTAL FACTOR correctly: actor-only training vs centralized-critic
    training.** Provenance MUST acknowledge that the historical Phase-A measurement and the
    future CTDE measurement carry **DISTINCT measured code SHAs**, and **must NOT claim the
    two arms' literal repository or configuration artifacts differ only by one
    `training_mode` field** — they do not, and asserting it would be false provenance;
  - **NOT bundle `p(destroy) < 1`, SAMs, dense reward, a solver change, a reward-formula
    change, or any new difficulty factor.** Those remain separate, still-deferred research
    changes (the difficulty-selection bullet below), and bundling one would make the
    comparison uninterpretable.
  A run showing no CTDE improvement, or no productive update, is a valid NEGATIVE
  observation — not a technical failure and not grounds to re-tune or re-run. **No CTDE
  preset exists in the repository**, and creating one belongs to that comparison task.
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
  observability, proof obligations and bounded implementation/lock task. The Phase-A
  baseline they were deferred behind is now MEASURED (§7), and Phase-B CTDE is now
  IMPLEMENTED and MERGED, so the ordering constraint that remains is the phase boundary:
  they come AFTER the first controlled actor-only vs CTDE comparison, and must never be
  bundled into it — a difficulty change inside that contrast would make it uninterpretable.
  **`probability < 1` in particular is UNCHANGED by FD-VARIABLE-SEVERITY-v1 and is still
  out.** That factor merged a mild/severe split of the FUEL-DAMAGE EVENT; target
  destruction stays deterministic at `probability = 1`, and nothing in PR #27 implemented
  stochastic target destruction. It remains a separate future Grade-A research task.
  **What DID change is the ordering, not this list**: exactly ONE additional difficulty
  design — FD-VARIABLE-SEVERITY-v1, the approved form of the stochastic/partial
  fuel-degradation variant — was selected, implemented and locked after FD-BASELINE-v1
  (`eecc9b5`, §5, §7), and its actor-only baseline — run CONCURRENTLY with Phase-B CTDE
  design and implementation, on a pinned immutable snapshot — is now EXECUTED and reviewed
  `APPROVE — VALID MEASUREMENT` (§7). Every entry in the list above stays deferred behind
  the phase boundary exactly as stated.
- **Solver 2:1 stacking (scenario-design fix, NOT solver constraints):** the anti-div-by-zero `EPSILON` nudges utility enough to assign 2 agents even at `probability=1.0`; a redundant agent chasing an already-killed target never proximity-confirms, so episodes end via `truncated`. The learned policy should recover this via `SELF_PRESERVATION_ABORT`→RTB once trained; the root fix is `EPSILON`/scenario-side.
- **Peer-dropout as a deterministic pre-build trigger** (advisor-pending, separate chat): move "peer overdue ⇒ drop its ASSIGNMENT edge" out of the policy; needs a deadline param + a `was_assigned_to_peer` feature to keep recovered-vs-popup semantics.
- **`reachable_by_ego` marginal-detour model:** `graph_builder._reachable_by_ego` is a conservative round-trip placeholder; intended model is marginal detour-cost vs remaining fuel slack (isolated to the builder; the mask reads the column).
- **`assigned_to_peer` as a task-feature column** (currently edge-derived), **real ETA** (enables PEER-OVERDUE; currently `never_overdue`), **`kill_confirm_ticks` FLOOR calibration** if p<1 lands — the per-salvo TRAVEL component is now derived (§5, Defect B), so what is left open is how long to wait past a confirmed MISS before deliberately re-firing.
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
  fuel-damage cell's headroom. That headroom is now measured by the approved Phase-A long
  baseline (`737b4bf`, §7), which is the only valid source for it.
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
> **Repository hygiene / documentation alignment: CLOSED** (Grade C, approved candidate
> `52064c2d306df7c8447d159df20e6e189a59bf85`, integrated by
> `5f78904e3af1e2e47386c9b0e01ddbaa273724f5`, PR #12 — see the §7 entry). Verified before
> each removal with exhaustive `git grep`.
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
