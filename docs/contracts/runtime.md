# Runtime contract — pipeline, episode setup, execution, triggers and the tick loop

> **Read this when** you change or review episode construction or `setup_episode`, world
> inventory versus allocation, the `GraphPlanExecutor` (movement, attack, confirmation wait,
> confirmed-kill reconciliation, RTB and physical completion), trigger logic, the two-phase
> tick loop and its per-tick ordering, playback recording, or anything that depends on those
> orderings. Reading another contract does not substitute for this one when a change touches
> the tick.
>
> **Status: normative, current technical contract** for the code on `main`. Lock history is in
> [`implementation.md`](../history/implementation.md), block provenance in
> [`documentation_migration.md`](../documentation_migration.md), and current run and evidence
> state in the [handoff](../../graph_rl_project_handoff.md). Frozen-engine limits:
> [`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion).
>
> Related contracts: [construction and fuel damage](construction_fuel_damage.md) ·
> [policy and CTDE](policy_ctde.md) · [reward and solvers](reward_solvers.md) ·
> [training and benchmarks](training_benchmarks.md) · [artifacts and metrics](artifacts_metrics.md).

## 1. The end-to-end pipeline

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
       # private GraphObservation. Contract: docs/contracts/policy_ctde.md §4.
```

**FIVE OPT-IN GENERALIZED-V1 SEAMS SIT BESIDE THE PIPELINE ABOVE, AND THE DIAGRAM
DESCRIBES THE DEFAULT.** The CONSTRUCTION path's placement step accepts a second
hidden-CARDINALITY policy (`bounded_backoff_v1`) beside the default `exact_v1`; the
fuel-damage layer accepts a certified eligibility policy and a completion-boundary wake
policy beside its two legacy defaults; when the latter is enabled, `run_episode` runs
one further top-of-tick step (`_post_fd_boundary`, after the event call and before Phase 1)
and one further terminal step (`require_certified_event_realized`, at the episode-exit seam,
before the recording export); and `setup_episode` accepts a second REWARD-REFERENCE policy
(`event_conditioned_continuation_v1`) beside the default `static_t0_v1`, which MOVES the
episode's second MATCH-AOU reference solve out of setup and into `run_episode`. **All five
default OFF, and with the defaults the pipeline above is exactly what runs.** Their contracts:
[hidden cardinality](construction_fuel_damage.md#1-hidden-cardinality-policies),
[certified FD and post-FD boundaries](construction_fuel_damage.md#4-certified-fd-eligibility-live-certificate-check-and-post-fd-boundaries),
[the continuation reference](reward_solvers.md#2-event-conditioned-continuation-reference).

**COUNT THE SEAMS AND THE POLICY IDS SEPARATELY — THEY ARE DIFFERENT QUANTITIES.** The
paragraph above counts FIVE PIPELINE SEAM SITES (its fifth being the two further
`run_episode` steps the completion-boundary policy opens). Those five sites are opened by
**FOUR** low-level POLICY IDS: `hidden_policy`, `eligibility_policy`, `post_fd_wake_policy`
and `reference_policy`. **A seam count is not a policy-id count**, and neither is a count of
what the generalized harness does besides resolving policies.

**SINCE GENERALIZED-V1 TASK 4 (`db79013`, integrated `b4daa8c`, PR #40) THOSE FOUR POLICY
IDS ARE RESOLVED TOGETHER BY ONE HARNESS KNOB, AND NEVER INDIVIDUALLY.**
`TrainConfig.episode_design` / `RolloutConfig.episode_design` ∈
`graph_generalized.EPISODE_DESIGNS` = (`fixed_cell_v1`, `generalized_v1`,
`generalized_v2`), DEFAULTING to `fixed_cell_v1`. `graph_generalized.EpisodeDesign`
carries the design id plus **EXACTLY those four** low-level policy ids and no others, and
`resolve_episode_design` resolves exactly them. `fixed_cell_v1` resolves the four
historical ids, so the diagram above is still exactly what a default run executes;
`generalized_v1` resolves the COMPLETE approved bundle in one word; and `generalized_v2`
resolves the **IDENTICAL FOUR IDS** as `generalized_v1` — it changes the POPULATION an
episode is drawn from, never the episode MECHANISMS. **There is deliberately no
per-policy harness field** — the four are resolved from the one selector and are not
independently settable from a config, a preset or a CLI flag — so a run can never resolve
half a bundle.

**TWO GENERALIZED-PATH BEHAVIOURS SIT BESIDE THAT RESOLUTION AND ARE NOT POLICY IDS ON
`EpisodeDesign`.** (1) **`fuel_damage_mode` REMAINS A SEPARATE `TrainConfig` /
`RolloutConfig` FIELD.** `validate()` REQUIRES it to be `seeded_variable` under
`generalized_v1`, but the selector neither carries it nor sets it, and it keeps its own
independent value on the fixed-cell path. (2) **The generalized TRAINING CARDINALITY SAMPLER
is harness / POPULATION behaviour selected on the generalized path**, not a fifth policy id:
`episode_cardinality` consults it because `cfg.generalized` is true, and `EpisodeDesign`
neither names nor returns it. The selector contract is
[training and benchmarks §5](training_benchmarks.md#5-episode-designs-generalized-v1-sampler-and-18-stratum-benchmark).

**THE REWARD-REFERENCE SEAM (GENERALIZED-V1 Task 3, `24a8b1e`) CHANGES WHERE AND AGAINST
WHAT THE SECOND SOLVE HAPPENS — NEVER THE CREDIT PLACEMENT.** Under the default
`static_t0_v1` the diagram above is exactly what runs: `setup_episode` solved the full t=0
reference, `EpisodeResult.reference` is `None`, and `compute_episode_reward` takes the
unchanged static-oracle branch. Under `event_conditioned_continuation_v1` `run_episode`
owns that second solve at exactly ONE of three places per episode, and the CHECKPOINT
ORDERING on a damaged tick is contractual:

```
  actual `current_fuel` mutation      # FuelDamageController.maybe_apply
    → build_continuation_reference    # THE CHECKPOINT — measurement only, no env.step
    → _post_fd_boundary               # post-FD completion boundary (opt-in)
    → decide_triggers                 # Phase 1 sensing / triggers / wake
    → central.capture                 # CTDE (opt-in), immediately before the decision
    → _wake_decision                  # the actor decision
    → Phase 2 → env.step
```

The other two places are a CLEAN episode's full t=0 reference, taken BEFORE the first tick,
and — for a damaged-scheduled episode whose event never fired, reachable under the LEGACY
FD eligibility policy only — a full t=0 reference at the episode-exit seam, before the
recording export. `compute_episode_reward` then branches on `result.reference`: present ⇒
the event-conditioned arithmetic, absent ⇒ the historical static one, and an episode that
DECLARES the opt-in policy yet arrives without a reference raises rather than falling back
on the static oracle that policy never solved. **Terminal-on-last credit placement, PPO,
GAE, the action set and the trigger layer are IDENTICAL under both policies.**

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

## 2. Episode setup (Stage 0)

**Episode-setup (Stage 0) — `rl/training/graph_episode_setup.py` + `rl/training/belief.py`.**
`setup_episode(scenario_json, ..., n_hidden=None, placement_rng=None) -> EpisodeContext`.
Wires env via `_build_env` (`gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=…)`, `obs,info = env.reset()`, blue side by `side.name=="BLUE"`) → `_extract_world` (`create_agents_from_scenario` picks blue; `generate_all_enemy_tasks`) → `solve_and_normalize` twice → `_finish_context` (N `Belief`s + one `GraphPlanExecutor`). `EpisodeContext` carries `env, game, agents, agent_ids, beliefs, executor, a_init, oracle_solution, oracle_tasks, split_meta, observation` (the reset seed the loop reads first), `record`, `placements`, the two RAW pre-solve world snapshots `known_target_ids` / `executed_target_ids` (the roster-integrity contract below), and the GENERALIZED-V1 pair `reference_policy` / `t0_reference_tasks` (the reward-reference contract below). `Belief.independent(tasks, solution)` mints an independent per-ego copy. `solve_and_normalize(agents, tasks) -> (solution, belief_tasks, unselected)` = `MatchAou(...).solve("bonmin")` → `post_solve_filter_and_level(...)` (allocated-only filter + `task_idx` remap + `level`); **never returns the raw pre-filter list**. Since `24a8b1e` it is a THIN PROJECTION of `solve_and_normalize_audited`, which returns the same triple plus a `SolveAudit`; **the public triple is byte-for-byte unchanged in every branch, the failure one included** — see the audited-solve contract below. Graph-native; imports NOTHING from the flat path. Independence + allocated-only proven in `_selftest`. `EpisodeContext.record: bool = False` — recording is ARMED iff a `recording_export_path` was given; setup never starts the recorder (the tick-loop drives it), and only the RETURNED env is ever armed.

**WORLD INVENTORY IS NOT ORACLE ALLOCATION (locked by the roster-integrity fix,
`36365f2`).** `solve_and_normalize` returns an **ALLOCATED-ONLY** task list by contract —
for BOTH solves. So `belief_tasks` is "the known targets the solver assigned" and
`oracle_tasks` is "the targets the ORACLE assigned"; **neither is an inventory of what
exists.** A target the solver left unselected is absent from both and is nevertheless
physically in the world, sensible, attackable and confirmable. Reading either one as a
world inventory is the defect this contract closes (its history:
[the first long baseline](../history/measurements.md#2-measurement-records)).

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
reproducibility key: generated target uuids are not seed-derived
([training and benchmarks §11](training_benchmarks.md#11-known-limitations-and-open-items)), so cross-run
comparison is still `geometric_fingerprint(ctx.placements)`.

**PATH SELECTION (`_resolve_construction_mode`, runs BEFORE any BLADE object exists).**
`n_hidden` and `placement_rng` are a PAIR. Both omitted → the LEGACY split path
(`_setup_episode_legacy`), behaviourally unchanged. Both supplied → CONSTRUCTION mode
(`_setup_episode_construction`). Exactly one supplied → `ValueError`. `n_hidden` must be a
genuine non-negative `numbers.Integral` (`bool` rejected, mirroring B2's `_as_assignment`);
`placement_rng` must be an explicit `random.Random`, never module-global randomness. The
mode is NEVER inferred from `partial_ratio`. `n_hidden=0` is a legal construction probe: it
places nothing, patches nothing, and still does not call `split_tasks`. A THIRD argument,
`hidden_policy`, selects the hidden-CARDINALITY policy and DEFAULTS to the historical
`exact_v1`; it is validated here too, and selecting `bounded_backoff_v1` without the
construction pair is REFUSED rather than silently ignored (see the GENERALIZED-V1
cardinality block below).

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
`geometric_fingerprint` — coordinates only, because generated uuids are not seed-derived.
Reproducibility is judged by that fingerprint, never by id.

The hidden-cardinality policies selected on this seam (`exact_v1`, `bounded_backoff_v1`) are
contracted in [construction and fuel damage §1](construction_fuel_damage.md#1-hidden-cardinality-policies);
the GENERALIZED-V2 route-relative hidden load is contracted in
[training and benchmarks §8](training_benchmarks.md#8-generalized-v2-population).

## 3. Execution (Stage 1)

**Execution (Stage 1) — `utils/blade_utils/blade_graph_executor.py`.**
`GraphPlanExecutor` is the **sole** BLADE translation layer (move/launch/attack/RTB). Its intra-level travel ordering comes from the SHARED pure helper `nearest_neighbor_order`, imported from `utils/scheduling_utils.py` — the SAME function `graph_hidden_placement.predict_route` calls, which is what keeps online execution and offline route prediction from drifting apart (`2a3f89c`). `__init__(*, tasks, solution, agents, arrival_threshold_km=DETECTION_KM, add_return_to_base=True, nn_ordering=True, kill_confirm_ticks=60)`. **Per-ego private state:** `self.tasks: Dict[ego_id, List[Task]]` (fanned out at init; diverges only via `resync`), `self.plans` per-ego; `_resolve_step(ego_id, assignment)` is the sole reader of `self.tasks`. Key methods: `next_actions(obs) -> List[str]` (one command/ego/tick), `resync(new_solution, *, ego_id, tasks=None)` (swaps one ego's slice, **never resets `done`**), `is_done(observation)` (**the live observation is REQUIRED**, no default — physical completion; see the Defect-C contract below), `sensed_target_ids(obs, ego_id) -> {id: unit}` (world-scan within `arrival_threshold_km`; the trigger's eyes), plus the GENERALIZED-V1 pair `reconcile_confirmed_for_ego(ego_id, scenario) -> Tuple[str, ...]` and `has_open_assignments(ego_id, scenario) -> bool` (see the ONE-CONFIRMATION-SITE contract below). done-on-confirmed-kill, per-`(ego,target)` re-fire throttle, single-issue RTB latch (safe only while doctrine `AIRCRAFT_RTB_WHEN_OUT_OF_RANGE` is off — it is in `strike_training_4v5.json`), `dead` set for crashes. No-comms isolation proven in `_selftest` (ISO-1..3: a pop-up appended to ego A never enters ego B's task-view; same-index pop-ups resolve per-ego).

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
- **Episode completion is a separate contract** (physical completion, below).

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

**CONFIRMED-KILL RECONCILIATION HAS ONE IMPLEMENTATION AND TWO CALLERS
(GENERALIZED-V1, `185d39f`).** The executor's historical confirm-guard was EXTRACTED
VERBATIM out of `_command_for_ego` into the private `_reconcile_confirmed(ego_id, scenario,
live, eligible) -> (newly_confirmed, remaining_eligible, blocked)`, at its historical point
in that method, so there is exactly ONE implementation of it rather than a second copy. Every
element is preserved and none is re-derived: the PROXIMITY GATE (the target must be within
this ego's OWN `arrival_threshold_km` — what stops an ego learning that a FAR target was
killed by a peer), the LIVENESS FACT (`scenario.get_target(target_id) is None`, probed only
for the single target the ego is engaging and only once in range), the MUTATIONS (add
`(ego_id, target_id)` to `done` — still the SOLE done signal, emitting an attack still does
not mark done — and drop that pair's re-fire cooldown), the LOOP (recompute eligibility and
keep going, so several consecutive already-gone heads are confirmed in ONE call), and the
UNEXECUTABLE HEAD (which stops the walk and, through `blocked`, still makes
`_command_for_ego` skip the ego this tick exactly as it always did).

Two PUBLIC methods expose it for the post-FD boundary seam, and neither changes anything for
any other ego or any command timing:

- **`reconcile_confirmed_for_ego(ego_id, scenario)`** returns the target ids newly confirmed
  by THIS call, in confirmation order. It emits no command, touches no peer, and mutates only
  `done` / `attack_cooldown` for THIS ego — the same two mutations Phase 2 has always made. A
  dead ego, or one with no live position (grounded / removed), confirms nothing. **It is
  IDEMPOTENT through the monotone `done` set**, which is why an early call leaves the tick's
  emitted command BYTE-IDENTICAL: Phase 2's own reconciliation on the SAME observation finds
  nothing further and emits precisely the command it would have emitted anyway, on the same
  tick.
- **`has_open_assignments(ego_id, scenario)`** is `bool(_eligible(...))` — "does this ego
  still have a mission", read from the same semantic state the executor already decides
  eligibility and RTB from, rather than from a belief slice that may still list work the ego
  has already confirmed. It recomputes and records nothing, and consults no peer.

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
  already exposes; the vendored files are byte-unchanged
  ([`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion)).

## 4. Triggers (Stage 2)

**Trigger (Stage 2) — `rl/action/graph_trigger.py`.**
`decide_triggers(belief_tasks, belief_solution, sensed_targets, eta=never_overdue, *, ego_id, clock, fuel_damage=False, post_fd_completion=False) -> (new_tasks, new_solution, wake, events)`. PURE (no BLADE/torch), copy-on-write (never mutates inputs). The WHEN gate over FOUR `TriggerKind` members: **POP-UP** (ego senses an unassigned target → appends a pop-up Task to append-only `belief_tasks`), **PEER-OVERDUE** (ego senses a peer's target AND its ETA passed → removes that peer tuple from the ego's `belief_solution` copy, so it reads as a pop-up — deterministic *gating*, the policy still chooses), **FUEL_DAMAGE** (FD-BASELINE-v1), and **POST_FD_COMPLETION** (GENERALIZED-V1, below). ETA is dormant (`never_overdue` = +inf) for now. `FUEL_DAMAGE` is EXOGENOUS — it cannot be detected from sensing, so the orchestrator passes `fuel_damage=True` for AT MOST ONE ego per tick; the flag defaults to `False`, so every pre-FD caller is byte-unchanged. It **edits NEITHER `belief_tasks` NOR `belief_solution`** (the changed quantity is the ego's own live fuel, which the builder reads off the aircraft) and only sets `wake`, appending a `(FUEL_DAMAGE, NO_TASK_INDEX)` event — `NO_TASK_INDEX = -1` is a sentinel, deliberately not `0`, because `0` is a valid task index. A tick carrying both a fuel-damage event and a pop-up still produces exactly ONE wake.
**`POST_FD_COMPLETION = 3` is APPEND-ONLY and behaves the same way** (GENERALIZED-V1,
`185d39f`): every existing member keeps the integer an already-recorded artifact used, the
`post_fd_completion` keyword defaults to `False` so every pre-existing caller is
byte-unchanged (the tick loop OMITS it entirely unless it is really True), it carries
`NO_TASK_INDEX` because it is not an observation ABOUT a target, and it edits NEITHER
`belief_tasks` NOR `belief_solution` — for a DIFFERENT reason from `FUEL_DAMAGE`: the edit
has ALREADY happened, because the orchestrator reconciled the confirmed assignment out of
that ego's own belief and resynced its own executor slice before calling. Both non-task
causes are appended BEFORE the sensing scan, and a tick carrying a boundary AND a pop-up
still produces exactly ONE wake with both events recorded. **NO-COMMS:** it is set for AT
MOST ONE ego per tick — the one that really lost fuel — and only after THAT EGO ITSELF
confirmed, inside its own sensor radius, that its assigned target is gone; a target killed
by a peer while this ego is far away is deliberately NOT a boundary.

## 5. Resync (Stage 6) and the two-phase tick loop

**Resync (Stage 6)** — `GraphPlanExecutor.resync` (above): swaps the ego's plan slice without resetting `done`.

**The two-phase tick (Stages 2–6) — `rl/training/graph_tick_loop.py`.**
`run_episode(policy, ctx, cfg=None, *, deterministic=False, max_ticks=None, fuel_damage=None) -> EpisodeResult`. Strict two phases per tick: **Phase 1** runs every ego's `sensed → decide_triggers → (on wake) _wake_decision` against the SAME `obs` snapshot with **no** `env.step`; **Phase 2** issues ONE `env.step(executor.next_actions(obs))`, and the tick's completion verdict is `executor.is_done(<the POST-STEP obs that step just returned>)` — completion is a PHYSICAL fact about the world the step produced (Defect C, `ea62e4e`), so an episode keeps ticking while an ordered-home aircraft actually flies home, and a death on that return is reconciled into `executor.dead` by the same call, BEFORE the loop returns, hence into `EpisodeResult.n_dead`. An ego whose `rtb_issued` latch is set is SKIPPED for the whole of Phase 1 from then on — no sensing, trigger, wake, policy inference, belief edit or `Transition` — while Phase 2 still runs for it every tick and peers continue normally. The optional `fuel_damage` controller (FD-BASELINE-v1) is consulted at the TOP of a tick, before Phase 1, and its Phase-2 `note_commands` call is a read-only measurement — see §1 above and [FD-BASELINE-v1](construction_fuel_damage.md#2-fd-baseline-v1); `None` (the default) leaves the loop byte-unchanged. Under GENERALIZED-V1 (`185d39f`) that same controller is consulted at TWO more places, both no-ops under the legacy defaults: `_post_fd_boundary` runs at the TOP of the tick immediately AFTER the event call and before Phase 1 (only when `fuel_damage.boundary_wakes_enabled`), and ONE terminal `fuel_damage.require_certified_event_realized(...)` runs at the EPISODE-EXIT seam after the loop and **BEFORE the recording export** — see [certified FD eligibility](construction_fuel_damage.md#4-certified-fd-eligibility-live-certificate-check-and-post-fd-boundaries). Because BLADE advances only after all egos decided on the identical snapshot, Phase-1 ego order cannot affect the outcome (structural no-comms; proven in `_selftest`: `env.step` count == tick count). `_wake_decision` is the per-wake chain (Stage 3→6) under `torch.no_grad`, editing ONLY the acting ego's belief. `Policy` (`build_policy()`) bundles encoder+head, built ONCE, lives across episodes. Seam for reward/PPO: `EpisodeResult.trajectory: List[Transition]`. The loop does NOT own the agent lifecycle (executor owns `dead`/`done`/`rtb`/`is_done`); it only hands `is_done` the post-step observation and READS the answer. **The reward seam is unchanged:** `graph_reward`'s formula still reads `n_lost = len(ctx.executor.dead)` — what changed is that the set is now truthful at episode end. **Recording:** armed by setup (`ctx.record`), driven here — start + forced t=0 frame before the loop, throttled `record_step` after each Phase-2 step (before the exit checks), forced terminal frame + `export_recording` after the loop (all exit paths). A pure READ of engine state; default off is a no-op — observational purity proven in `_selftest` TEST 1b (identical `(ended, ticks, n_wakes)` with recording on/off). Artifact: `{export_path}/{scenario_name} Recording {start} - {end}.jsonl`.

## 6. Code routing

| Task | Files and symbols | Contract |
|---|---|---|
| change episode setup, solve-and-normalize or beliefs | `rl/training/graph_episode_setup.py`: `setup_episode`, `solve_and_normalize`, `_finish_context`; `rl/training/belief.py`: `Belief` | §2 |
| answer "which targets does this episode contain?" (world inventory, never an allocation) | `graph_episode_setup.py`: `EpisodeContext.known_target_ids`, `executed_target_ids`, `_world_target_ids` — never `oracle_tasks`, `belief_tasks` or beliefs | §2 |
| change the construction seam (solve → place → patch → reload) | `graph_episode_setup.py`: `_setup_episode_construction`, `_resolve_construction_mode`, `_shared_launch_point`, `_require_airbase_only_targets`, `_select_hidden_prototype`, `build_patched_scenario`, `_require_agent_ids_preserved`, `_rematerialize_known_tasks`, `_build_env`, `_extract_world`, `_close_quietly` | §2; [construction §1](construction_fuel_damage.md#1-hidden-cardinality-policies) |
| change the retained legacy split path | `graph_episode_setup.py`: `_setup_episode_legacy`, `split_tasks` | §2 |
| change the tick loop, policy bundle or recording | `rl/training/graph_tick_loop.py`: `run_episode`, `_wake_decision`, `Policy`, `build_policy` | §1, §5 |
| change when the policy wakes | `rl/action/graph_trigger.py`: `decide_triggers`, `TriggerKind` (values append-only), `never_overdue`, `NO_TASK_INDEX` | §4 |
| change BLADE execution or plan resync | `utils/blade_utils/blade_graph_executor.py`: `GraphPlanExecutor`, `next_actions`, `resync`, `sensed_target_ids` | §3, §5 |
| change confirmed-kill reconciliation (one implementation, two callers) | `blade_graph_executor.py`: `_reconcile_confirmed`, `reconcile_confirmed_for_ego`, `has_open_assignments`; `tests/test_graph_executor_nn_ordering.py` | §3 |
| change the attack-confirmation wait | `blade_graph_executor.py`: `_salvo_travel_ticks`, `_confirmation_wait_ticks`, the attack branch of `_command_for_ego`; `tests/test_graph_setup_seam.py` | §3 |
| change RTB or episode-completion semantics | `blade_graph_executor.py`: `is_done`, `_physical_state`, `_note_dead`; the `rtb_issued` Phase-1 skip in `graph_tick_loop.run_episode`; `tests/test_graph_setup_seam.py`, `tests/test_graph_fuel_damage.py` | §3, §5 |
| create agents and tasks from a scenario | `utils/blade_utils/scenario_factory.py`: `create_agents_from_scenario`, `generate_all_enemy_tasks`, `iter_enemy_targets`, `make_attack_task` | §2 |
| change scenario content, geometry or discovery connectivity | `utils/blade_utils/scenario_generator.py`: `VariationConfig` (`strict_geometry`, `min_target_separation_km`, `ensure_discovery_chain`), `ScenarioGenerator`, `CLASS_RANGE_TIERS`, `_ensure_discovery_chain` | §2 |
| change post-solve scheduling or levels | `utils/scheduling_utils.py`, `utils/topology_utils.py` | [construction §1](construction_fuel_damage.md#1-hidden-cardinality-policies) (shared `nearest_neighbor_order`) |
| change domain objects | `models/`: `agent.py`, `task.py`, `step.py` (`StepKind`), `location.py`, `capability.py` | — |

Every file above is frozen-engine-safe only as far as it reads what BLADE exposes; the engine
itself stays frozen ([`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion)).

> Shared domain infra (`scenario_generator`, `scenario_factory`, `scheduling_utils`, `topology_utils`, solver, `models`, `blade_utils`) is used by the graph path and is NOT old-model. Hand-written `.md` API docs may lag — prefer the code.

## 7. Known limitations and open items

- **Peer-dropout as a deterministic pre-build trigger** (advisor-pending, separate chat): move "peer overdue ⇒ drop its ASSIGNMENT edge" out of the policy; needs a deadline param + a `was_assigned_to_peer` feature to keep recovered-vs-popup semantics.
- **`assigned_to_peer` as a task-feature column** (currently edge-derived), **real ETA** (enables PEER-OVERDUE; currently `never_overdue`), **`kill_confirm_ticks` FLOOR calibration** if p<1 lands — the per-salvo TRAVEL component is now derived (§3), so what is left open is how long to wait past a confirmed MISS before deliberately re-firing.
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
  guard. It touches a locked layer, so it is a reviewed research-validity change
  ([`cc_review.md` §4](../workflows/cc_review.md#4-risk-and-verification)).
- **Single-radius invariant ([`CLAUDE.md` §3](../../CLAUDE.md#3-architecture--the-load-bearing-invariants)) — CLOSED.** Sensing-radius expansion was cancelled.
  Keep the unified `DETECTION_KM = 50` contract for sensing, arrival, attack,
  kill-confirmation, generator connectivity, and split adjacency. Do not reopen this as
  part of scenario construction.
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
