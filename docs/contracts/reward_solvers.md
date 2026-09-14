# Reward and solver contract — terminal reward, reference policies and MATCH-AOU backends

> **Read this when** you change or review the terminal reward, oracle or reference
> normalization, the event-conditioned continuation checkpoint, reference-integrity routing,
> any MATCH-AOU solve, backend selection (`legacy_minlp_v1` / `p1_milp_v1`), solver
> timeouts, or anything that compares allocations across backends. The frozen legacy solver's
> hard constraints are in [`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion).
>
> **Status: normative technical contract.** Sections 1–3 and 5 were moved **verbatim** from
> `CLAUDE.md` at base `ae42cb01677f94868b2873008d87be677e31f0c8` (Stage 7, the continuation
> reference and the MATCH-AOU backend blocks of former §5, rows of former §6, items of former
> §8). Inside moved text a bare `§N` means that **former** `CLAUDE.md` section — resolve it with
> the [compatibility index](../../CLAUDE.md#8-compatibility-index-for-older-references). A
> statement inside a block that a run or result "does not exist" describes that block's own PR
> scope when it merged; current run and evidence state lives in the
> [handoff](../../graph_rl_project_handoff.md).
>
> Related contracts: [runtime](runtime.md) · [construction and fuel damage](construction_fuel_damage.md) ·
> [training and benchmarks](training_benchmarks.md).

## 1. Terminal reward (Stage 7)

**Reward (Stage 7) — `rl/training/graph_reward.py`.**
`compute_episode_reward(ctx, result, cfg=RewardConfig()) -> EpisodeReward`. **Terminal, utility-based** (v1): `R = (U_achieved − c·U_aircraft·n_lost − U_oracle)/(|U_oracle| + eps_regret)`, placed on the last wake's `Transition` (others `0.0`; empty trajectory ⇒ nothing attached). `U_oracle = plan_value(ctx.oracle_solution, ctx.oracle_tasks)` — **bit-faithful to `MatchAou._add_objective`** (reuses the solver `EPSILON`; the `y[j]` factor is provably redundant given the y/x constraints; proven under bonmin in `_selftest` T1). `U_achieved = realized_utility(ctx.oracle_tasks, ctx.executor.done)` — full utility IFF all a task's targets are confirmed-killed, **deduped over ego**. `c = aircraft_penalty_coeff` — this module's own default is **0.0**, but BOTH harnesses now pass an explicit `RewardConfig(aircraft_penalty_coeff=2.25)` (FD-BASELINE-v1, below); the FORMULA is unchanged. `n_lost = len(ctx.executor.dead)`; `eps_regret=1e-5` is a division guard (distinct from solver EPSILON). **No-comms:** a centralized/privileged TRAINING signal — MAY read global state, but MUTATES ONLY `Transition.reward` (proven byte-unchanged on real objects in T7). **KNOWN v1 assumption `probability=1.0`** (expected `U_oracle` vs realized `U_achieved` coincide only at p=1; `R∈[-1,~0]`; revisit at p<1). **THIS PARAGRAPH DESCRIBES THE DEFAULT `static_t0_v1` REFERENCE POLICY, WHICH IS UNCHANGED AND IS THE PATH EVERY APPROVED MEASUREMENT WAS TAKEN ON** (`737b4bf`, `bf1e045f` — §7). Under the opt-in `event_conditioned_continuation_v1` policy the SAME function normalizes by `U_ref` instead and `EpisodeReward.u_oracle` is `None`; the formula above is untouched and is still what runs whenever `EpisodeResult.reference is None` (`_static_t0_breakdown`, lifted out byte-for-byte). See the GENERALIZED-V1 reward-reference contract below.

## 2. Event-conditioned continuation reference

**GENERALIZED-V1 EVENT-CONDITIONED MATCH-AOU CONTINUATION REFERENCE + REWARD CHECKPOINT —
`rl/training/graph_episode_setup.py` + `rl/training/graph_tick_loop.py` +
`rl/training/graph_reward.py` (`24a8b1e`, integrated `df3abf2`, PR #38 — §7).**

ONE OPT-IN policy seam, a VERSIONED string, DEFAULTING to the merged historical behaviour —
so every existing call site obtains the historical reference automatically.

| knob | DEFAULT (historical, preserved) | GENERALIZED-V1 addition |
|---|---|---|
| `setup_episode(..., reference_policy=...)` | `static_t0_v1` | `event_conditioned_continuation_v1` |

`graph_reward.REFERENCE_POLICIES` is the closed set. **`EpisodeContext.reference_policy` is
the SINGLE STORED SOURCE of the policy**, validated by `setup_episode`'s
`_resolve_reference_policy` BEFORE any BLADE object exists — an unknown id RAISES
`ValueError`, is never coerced to the default, never case-folded into a match and never
ignored, because a run that silently fell back to the historical reference while its record
claimed the opt-in one would be unreadable exactly where it matters.
**`graph_reward.uses_event_conditioned_reference(ctx)` is the CANONICAL RUNTIME PREDICATE**
over that stored value — the one the tick loop and the reward branch on. It is not the only
place the policy is examined: `setup_episode` / `_finish_context` also VALIDATE it and USE it
to decide whether to retain the deferred-reference inputs, and `_t0_reference_or_deferred`
compares it to decide whether setup solves the reference at all. What is contractual is that
there is ONE stored value and ONE runtime predicate over it, not that the string is compared
in exactly one expression. The predicate reads the attribute DUCK-TYPED and resolves an
ABSENT field to `static_t0_v1`, so a context that declares no policy can only ever land on
the PRESERVED path, never on the opt-in one.

**`static_t0_v1` IS THE DEFAULT AND IS THE PRESERVED HISTORICAL CONTRACT.** Setup performs
the full t=0 reference solve exactly where it always did; `oracle_solution` / `oracle_tasks`
are populated exactly as before; `EpisodeContext.t0_reference_tasks` stays `()` and
`EpisodeResult.reference` stays `None`; `compute_episode_reward` takes
`_static_t0_breakdown`, which was LIFTED OUT of the old function body unchanged — the same
reads, the same operand order, the same single `denom`, the same folding of the penalty.
**The approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) measurements were
taken on this path and remain measurements OF IT.** On this path NO `EpisodeReference` object
exists at all, and that ABSENCE is how a reader tells which policy ran — exactly as
`EpisodeContext.construction_audit` being `None` identifies the historical hidden-cardinality
policy.

**`event_conditioned_continuation_v1` IS AN OPT-IN ADDITION BESIDE IT, never a rewrite of
it.** It DEFERS setup's second solve to `run_episode`. Under it `_t0_reference_or_deferred`
performs NO solve and returns the EMPTY pair, so `oracle_solution` / `oracle_tasks` are
DELIBERATELY EMPTY and must never be read as a reference; `_finish_context` instead retains
`EpisodeContext.t0_reference_tasks` — the RAW pre-solve t=0 EXECUTED-WORLD task list, the
same one `executed_target_ids` was snapshotted from — and RAISES if the policy defers a
solve with nothing retained to solve it from. The retention is CONDITIONAL: under the
historical policy the field stays `()`, because keeping a second copy of the world's tasks
alive there would be a silent behaviour difference dressed up as an optimisation. **Ids alone
could not replace it**: an id carries no utility, no probability and no location, and once a
target is destroyed the live BLADE world can no longer supply them either, so scoring
`U_prefix` after the fact needs the t=0 task OBJECTS.

**WHERE THE SECOND SOLVE HAPPENS — EXACTLY ONE OF THREE PLACES PER EPISODE, all owned by
`run_episode`.** The policy MOVES solve #2; **it never ADDS a third reference solve.**

- **CLEAN (or NO fuel-damage controller at all) — the FULL t=0 reference, BEFORE the first
  tick.** `build_t0_reference(ctx, kind=REFERENCE_KIND_CLEAN_T0)` runs before RECORDING
  STARTS — before `run_episode`'s `start_recording()` / forced t=0 frame — and therefore
  before any BLADE state can have advanced. (Recording is ARMED much earlier, by setup, when
  a `recording_export_path` was supplied; ARMING and STARTING are different events, and this
  reference precedes the START. See Stage 0 for the arming contract.) The condition is READ
  OFF the controller's already-resolved `plan.is_damaged` rather than re-derived, so this
  seam never touches a fuel-damage RNG domain. **Its inputs are t=0 inputs whenever it runs**:
  `ctx.agents` are the frozen `_extract_world` snapshots carrying t=0 location and t=0 fuel
  as `budget`, and `ctx.t0_reference_tasks` is the raw t=0 world — neither tracks the live
  engine — so it reproduces the reference setup would have computed under the historical
  policy from bit-identical inputs. `U_prefix` is `0.0` BY CONSTRUCTION here, not measured as
  zero: a t=0 reference allocates over the whole world, so every realized target stays
  scorable in `U_post` and splitting a prefix out of it would double-count.
- **DAMAGED — the CONTINUATION reference, at the TOP of the FIRING tick.**
  `build_continuation_reference(ctx, scenario=obs, tick=tick, damaged_ego_id=...)` runs
  immediately AFTER `FuelDamageController.maybe_apply` performed the real `current_fuel`
  mutation and BEFORE anything reacts to it. **THE ORDERING IS CONTRACTUAL** (§4): the real
  mutation → the continuation reference → the post-FD completion boundary → the triggers →
  the CTDE `central.capture` → the actor decision → Phase 2 / `env.step`. So the reference
  describes the world the actor is ABOUT to decide in, not the world it decided into.
- **DAMAGED WHOSE EVENT NEVER FIRED — the full t=0 reference at the EPISODE-EXIT seam.** See
  the legacy-compatibility block below; it is still solve #2, because the checkpoint above
  never happened.

**THE CHECKPOINT IS PRIVILEGED MEASUREMENT, AND IT IS READ-ONLY WITH RESPECT TO THE
EPISODE.** It writes no belief, no executor plan / `done` / RTB state, no actor
`GraphObservation`, no `CentralGraphObservation`, no policy parameter and no BLADE state; it
touches the engine only by reading it. **No field of it reaches the acting path** — the
no-communication red line (§3) is untouched, in exactly the sense `graph_reward` as a whole
already is a centralized TRAINING signal. Solver WALL-CLOCK time passes while it runs;
**SIMULATION time does not, because the checkpoint issues no `env.step`.**

**WHAT THE CONTINUATION REFERENCE SOLVES.**

- **TASKS — the retained RAW t=0 executed-world universe MINUS the realized prefix.**
  `_reference_universe` splits that universe ONCE, through
  `graph_reward.realized_task_indices` — the SAME all-steps rule `realized_utility` sums
  over — so the two halves PARTITION it exactly and a task counted in `U_prefix` is provably
  absent from the continuation universe. The prefix tasks are taken BY INDEX, never by
  re-matching target ids, because two tasks naming no target would both resolve to `""`.
  **It is the authoritative world inventory, NEVER any ego's private belief**: a belief is
  one ego's partial view, and using it as the global reference universe would make the
  reference depend on who happened to have sensed what. Utility and probability semantics are
  the t=0 ones, unchanged.
- **AGENTS — the continuation-capable ORIGINAL egos, REBUILT FROM THE LIVE POST-EVENT
  WORLD.** `_continuation_agents` runs the SAME `scenario_factory.create_agents_from_scenario`
  conversion setup uses, applied to the post-mutation observation, so an ego's
  `Agent.location` is where it REALLY is and its `Agent.budget` is the fuel it REALLY holds —
  the damaged ego's reduced `current_fuel` included. Reconstructing that mapping locally
  would be a second conversion that could drift from the one the episode was planned with.
  **THE POPULATION IS FILTERED, NEVER INVENTED**: it is drawn from `ctx.agent_ids`, the
  authoritative scheduled ego sequence in its own order, and an ego is dropped with a stable
  recorded `CONTINUATION_EXCLUSION_REASONS` slug when it CANNOT continue — `dead` (the
  executor reconciled its removal), `rtb_committed` (its single-issue RTB latch is set, so
  Phase 1 no longer processes it and reallocating it would be a reference the execution layer
  could not honour), `not_airborne` (the engine does not hold it in `scenario.aircraft`: it
  landed or was removed). The scan is READ-ONLY. The damaged ego is given NO special standing
  — the continuation is a TEAM allocation; `damaged_ego_id` is recorded in the failure
  messages so a refused checkpoint names the event it belongs to.

**IT IS A REFERENCE, NOT AN ORACLE, and this layer never calls it one.** A damaged episode's
reference is CONDITIONED on an event that had already happened when it was solved, so it is
not the fully-informed t=0 optimum. It is also a **MATCH-AOU ALLOCATION reference**: it
states what the frozen solver would ALLOCATE from the post-event state, and is **NOT a claim
that the resulting physical routes are optimal** — the solver retains its own independent
round-trip movement model.

**THE REWARD ARITHMETIC (`_event_conditioned_breakdown`).**

```text
U_ref      = U_prefix + U_cont_ref          # verified on the reference itself
U_achieved = U_prefix + U_post
ratio      = (U_achieved - U_ref)      / (|U_ref| + eps_regret)
penalty    = (c * U_aircraft * n_lost) / (|U_ref| + eps_regret)
R          = ratio - penalty                # NEVER clamped
```

- **`U_prefix` IS FROZEN AT THE CHECKPOINT**, against a COPY of `executor.done` taken at that
  instant, and is taken STRAIGHT OFF the reference at reward time. It is deliberately NOT
  recomputed from the larger end-of-episode `done` set: doing so would credit
  post-checkpoint kills to the prefix AND leave them scorable in `U_post`, counting them
  twice.
- **`U_post` SCORES ONLY `reference.tasks`** — the ALLOCATED-ONLY continuation list. A
  confirmed kill on any other target (one the reference did not allocate, or one the prefix
  already paid for) contributes NO utility and is **ACCOUNTING-ONLY**, reported through
  `unique_completed_targets` / `scored_completed_targets` / `unscored_completed_targets` /
  `unscored_completed_target_ids`.
- **`U_aircraft` COMES FROM THE REWARD-BEARING REFERENCE UNIVERSE** — the prefix tasks plus
  the allocated tasks (`_reference_aircraft_utility`, `0.0` on an empty universe) — so the
  death penalty stays on the same utility scale as the numerator it is subtracted from,
  WITHOUT relabelling the continuation reference as an "oracle" task list.
- **`n_lost`, `c` and `eps_regret` keep their historical meanings**, and the reward is **NOT
  CLAMPED**: with `c > 0` and a real airframe loss it legitimately falls below `-1`, exactly
  as it already can on the static path.
- **`EpisodeReward.u_oracle` IS `None` UNDER THIS POLICY** — there IS no static full-set
  optimum, setup deliberately did not solve one, and `0.0` would fabricate a perfect oracle.
  `EpisodeReward.u_ref` is the denominator source under BOTH policies (on the static path it
  EQUALS `u_oracle`), so a consumer asking "what was this normalized by?" reads that one
  field and is correct either way.
- **WHAT THE ADDED `EpisodeReward` FIELDS HOLD ON THE HISTORICAL PATH — the distinction is
  exact, and it is NOT "everything is `None`".** Two of the added fields are NOT optional and
  carry real historical values: **`reference_policy` is `static_t0_v1`** (the policy that
  really ran, stated rather than implied) and **`u_ref` is `u_oracle`** (`_static_t0_breakdown`
  sets it from the static optimum; the `0.0` on the dataclass is only the field default).
  The **OPTIONAL SCALAR AND COUNT checkpoint fields** are `None` there — `reference_kind`,
  `checkpoint_tick`, `u_prefix`, `u_cont_ref`, `u_post`, `unique_completed_targets`,
  `scored_completed_targets`, `unscored_completed_targets` — **`None`, never `0.0` / `0`**,
  because on a normalized regret scale `0` is the OPTIMUM and on a count it reads as a
  measurement of nothing rather than as an absent measurement. **The ONE EXCEPTION is
  `unscored_completed_target_ids`, which is NOT `Optional` and keeps its TYPED EMPTY-TUPLE
  default `()`** — an empty sequence of ids is not an absent measurement, and a consumer
  iterating it needs no `None` guard on either path.
- **FOR A CLEAN EPISODE UNDER THIS POLICY `U_prefix == 0` and the reference IS the full t=0
  reference, so the arithmetic COLLAPSES to the static formula** — the checkable property
  that the opt-in path does not silently move the clean condition.
- **CREDIT PLACEMENT IS UNCHANGED UNDER BOTH POLICIES.** Terminal-on-last: every transition
  is set to `0.0` and the LAST is overwritten with `R`; an EMPTY trajectory still attaches
  nothing and still returns the breakdown. **The ONLY mutation remains `Transition.reward`.**
  PPO, GAE, the action set, the trigger layer and the actor/critic boundary are untouched.

**AUDITED SOLVE INTEGRITY — AN UNANSWERED SOLVE IS NOT AN ANSWERED ZERO.**
`solve_and_normalize_audited` is now THE ONE MATCH-AOU normalization site, and
`solve_and_normalize` is a thin projection of it whose **public triple is byte-for-byte
unchanged in every branch, the failure branch included**, so every historical caller is
unaffected. The frozen solver (§2) is untouched. `SolveAudit` records what
`MatchAou.solve` already distinguished but the triple could not express:

- **`invoked=False`** — the degenerate short-circuit fired (no tasks, or no agents) and NO
  solver was called. `accepted` is vacuously `True`, `termination_condition` is
  `SOLVE_NOT_ATTEMPTED`, and this is a SKIPPED solve, not a failed one.
- **`invoked=True, accepted=False`** — `raw_solution is None`: the solver did NOT reach
  acceptable optimality. **A FAILED QUESTION.**
- **`invoked=True, accepted=True` with an empty allocation** — the solver terminated
  acceptably and selected nothing. **AN ANSWERED QUESTION whose answer is "allocate
  nothing"**, and a perfectly legitimate reference of value `0`.
- `SOLVE_TERMINATION_UNAVAILABLE` is a RECORD-COMPLETENESS fallback for the audit STRING
  only; it never affects `accepted`, which is decided solely by whether `MatchAou.solve`
  returned a solution.

`_solve_reference` is the single site that CONSUMES the distinction: `invoked and not
accepted` raises `ReferenceIntegrityError`, because turning an unanswered question into an
empty reference would hand the episode a zero denominator and therefore `-0/eps == 0` — the
OPTIMUM — for a solve that never happened. **An accepted `{}` allocation, and a skipped
degenerate solve, are BOTH legitimate zero references and are returned as-is.**

**THE SOLVE BUDGET, STATED PRECISELY.** Task 3 **NEVER ADDS A THIRD REFERENCE SOLVE**: the
opt-in policy OCCUPIES the existing second reference-solve slot rather than creating a new
one, which `_t0_reference_or_deferred` guarantees by skipping setup's reference solve exactly
when `run_episode` will own it. **It is therefore AT MOST two BONMIN invocations per accepted
episode, and never three.** It is deliberately NOT stated as "exactly two": a degenerate
reference — no open task, or no continuation-capable ego — legitimately performs NO solver
call at all and records `solver_invoked=False`, so an accepted episode may cost one. *(The
module docstrings and the candidate commit message phrase this as "exactly two"; the
behaviour is the at-most-two form above, and this contract is the accurate statement.)*

**`ReferenceIntegrityError` — WHAT IT IS, AND HOW IT IS ROUTED (the Task-4 decision, now
TAKEN).** A plain `RuntimeError` subclass, deliberately NOT a `FuelDamageError` /
`EpisodeRosterError` / `MeasurementIntegrityError` sibling, because `graph_reward` must not
import the trainer. It is raised when a reference the episode DEPENDS on cannot be produced
honestly: the reference solve did not reach acceptable optimality; the event-conditioned
policy is in force but NO reference reached the reward (so the deliberately-empty static pair
would have been read as one); nothing was retained to solve from; an unknown reference KIND;
or a reference whose own arithmetic does not reconcile
(`|u_ref - (u_prefix + u_cont_ref)| > 1e-9`, checked in `EpisodeReference.__post_init__` —
VERIFIED, not asserted in prose).

**SINCE GENERALIZED-V1 TASK 4 (`db79013`) THE EXCEPTION CARRIES A STABLE MACHINE-READABLE
REASON, AND THE ROUTING READS THAT SLUG AND NOTHING ELSE — NEVER THE MESSAGE TEXT.**
`reason` is a REQUIRED keyword on `__init__` (an optional one would let a future raise site
skip the classification by omission and fall into whichever routing happened to be the
default — the same class of defect that made a roster fault look like ordinary attrition),
and it is validated against the closed set `REFERENCE_FAULT_REASONS`. The set splits into
exactly TWO routing classes:

- **ORDINARY ACCOUNTED ATTRITION — `REFERENCE_ATTRITION_REASONS`, which today is exactly
  `reference_solve_unacceptable`.** The question was ASKED and the solver did not ANSWER it.
  Nothing was contradicted and no other episode is implicated, so the scheduled attempt is
  spent: `graph_train` wraps it as `EpisodeAttemptError("run", ...)` or
  `EpisodeAttemptError("reward", ...)` depending on which stage raised it, records it ONCE in
  `episode_failures.jsonl` with `reference_fault_reason` set, and `skip_and_account_v1` moves
  on — no retry, no substitution, no seed replacement, and never a conversion into a
  zero-valued reference.
- **MEASUREMENT-INTEGRITY ABORT — every other reason: `reference_missing`,
  `reference_universe_unavailable`, `reference_kind_unknown`,
  `reference_arithmetic_contradiction`.** The INSTRUMENT contradicted itself, so every
  episode the reference layer touched is suspect. `graph_train` re-raises it AHEAD of every
  broad handler — in `_run_one_episode`'s run and reward blocks and in the train, legacy-eval
  and benchmark-eval attempt handlers, spelled `except (_VisualArtifactError,
  MeasurementIntegrityError, FuelDamageIntegrityError, BenchmarkIdentityError,
  ReferenceIntegrityError)` — so it names no pipeline stage, is NEVER written to
  `episode_failures.jsonl`, never counted against a condition or stratum tally, never folded
  into a matched group and never entered into `skip_and_account_v1`. Identical routing, and
  identical reasoning, to the roster and certificate contracts.

**`graph_reward.reference_fault_aborts(exc)` IS THE ONE PREDICATE A HARNESS ROUTES ON**, and
`ReferenceIntegrityError.is_measurement_integrity` is the property behind it, so the
classification lives beside the reasons it classifies instead of being re-derived — possibly
differently — at four handler sites. A non-`ReferenceIntegrityError` returns `False`; this
layer still takes no decision on the trainer's behalf and still imports no trainer.
*(SUPERSEDED, and corrected here: this paragraph previously read "It is NOT routed as an
aborting instrument fault … Whether it should instead ABORT is a Task-4 decision … It is
unreachable in practice today, because no harness selects the policy." That was accurate
before PR #40; the decision has since been taken as above, and both harnesses now select the
policy through `episode_design`.)*

**LEGACY COMPATIBILITY EDGE CASE — `damaged_event_unrealized_t0`, AND IT IS NOT GENERALIZED
DAMAGED SEMANTICS.** A DAMAGED-SCHEDULED episode whose event NEVER FIRED physically ran as a
clean one, so at the episode-exit seam it receives a FULL t=0 reference built from the
RETAINED t=0 inputs, recorded under its OWN kind — because "the event did not fire" and "no
event was scheduled" are different facts, and it must never be read as a scheduled clean
episode. It sits BEFORE the recording export, for the same reason
`require_certified_event_realized` does: `graph_train` synchronizes a completed run's
playback into its manifest only after `run_episode` returns, so exporting first and raising
second would leave a real recording no manifest lists.
**IT EXISTS ONLY TO PRESERVE THE ALREADY-LOCKED TASK-2 LEGACY CONTRACT**, under which a
scheduled damaged episode may legitimately finish without the FD event firing — an approved
measurement contains exactly such an episode (§7: the Phase-A rerun's seed 424).
**UNDER `certified_both_severities_v1` IT IS UNREACHABLE**: the tick loop's terminal
`require_certified_event_realized` rejects certified + damaged + not-fired FIRST, as a
`FuelDamageIntegrityError` instrument abort. **So this fallback is NOT part of the intended
GENERALIZED benchmark's damaged semantics, and must not be generalized into them.** *(This
was the one implementation deviation the GPT review examined and APPROVED, as a
compatibility resolution rather than a semantic extension — recorded here rather than
hidden.)*

**`EpisodeReference` — THE TYPED PER-EPISODE REFERENCE / ACCOUNTING RECORD.** Frozen,
produced ONLY under the opt-in policy, by `build_t0_reference` / `build_continuation_reference`
and by nothing else, and carried out of the episode as `EpisodeResult.reference` — ONE
owner, ONE surface, EXACTLY ONE per episode under that policy and NONE at all under the
historical one. It carries: `policy` and `kind` (`REFERENCE_KINDS` = `clean_t0` /
`damaged_event_checkpoint` / `damaged_event_unrealized_t0`); `checkpoint_tick` (the damaged
checkpoint's tick, and `None` — never `0` — for a t=0 reference, because tick 0 is a real
tick); the utility decomposition `u_prefix` / `u_cont_ref` / `u_ref` / `u_aircraft`; the
allocated `solution` and its ALLOCATED-ONLY `tasks` (deliberately NOT a world inventory — the
same contract `solve_and_normalize` has always had); the reward-bearing
`reference_target_ids`; the prefix identity `prefix_target_ids`; `candidate_task_count` (how
many tasks were OFFERED, larger than `len(tasks)` whenever the solver legitimately left some
unselected); the continuation-agent population `continuation_agent_ids` with its
`excluded_agents` `((ego_id, reason), ...)` pairs — so a dead or RTB-committed ego is
RECORDED rather than inferred from an absence; and the solver audit `solver_invoked` /
`solver_accepted` / `solver_termination` / `solver_seconds`. `solver_accepted` is always
`True` on a RETURNED object — the builder raises instead — and exists to make that guarantee
legible. `is_event_checkpoint` is the kind predicate; `to_record()` is a JSON-ready view of
plain builtins that deliberately omits `tasks` and `solution`, and whose ids are WITHIN-RUN
accounting identifiers, **never a cross-run reproducibility key** (generated uuids are not
seed-derived — §8).

**BOTH HARNESSES NOW SELECT THIS POLICY — THROUGH THE BUNDLE, NEVER ON ITS OWN
(GENERALIZED-V1 Task 4, `db79013`).** `TrainConfig.episode_design` /
`RolloutConfig.episode_design` resolve `reference_policy` from the one selector, and
`graph_train._generalized_setup_kwargs` / `graph_rollout.run_rollout` pass it to
`setup_episode` ONLY on the generalized path — on `fixed_cell_v1` the keyword is OMITTED
entirely, so **a default run still runs `static_t0_v1` with `EpisodeResult.reference` `None`
on every episode**, from the pre-Task-4 call. **Neither config carries a standalone
`reference_policy` field**, so the policy cannot be enabled apart from the bundle.
`EpisodeReference` and the reward decomposition are now PERSISTED per episode and AGGREGATED
per run — see the GENERALIZED-V1 harness / population contract below. *(SUPERSEDED, and
corrected here: this paragraph previously read "NEITHER HARNESS EXPOSES THIS POLICY YET …
Nothing persists or aggregates `EpisodeReference` … That run-level persistence, and the
generalized sampler / evaluation manifest that would consume it, is Task-4 work and is NOT
implemented". That was accurate before PR #40 and is not now.)* **No generalized scientific
measurement exists** (§8).

**WHAT IS EXPLICITLY NOT IN THIS DESIGN.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate future
Grade-A research task**. No dense or per-wake reward shaping, no change to terminal credit
placement, no generalized training sampler, no evaluation manifest, no new metric or plot, no
new `MetaAction`, no peer behaviour change and no communication channel of any kind. BLADE,
the solver, PPO, GAE, the encoder, the action space, the trigger layer, the actor/critic
boundary, `DETECTION_KM`, B2 geometry, the fuel-damage mechanism and the seed schedules are
all unchanged. *(That list is a statement about TASK 3's scope and stays accurate as one.
The generalized training sampler, the evaluation manifest and the new metrics and plot panel
it excludes were implemented AFTERWARDS, as the SEPARATE Task-4 harness / population layer
contracted immediately below — `graph_reward`'s static formula is still unchanged there
too.)*

## 3. MATCH-AOU allocation backends

**THE MATCH-AOU ALLOCATION BACKEND — AN EXPLICIT SELECTOR OVER TWO NON-INTERCHANGEABLE
OBJECTIVES — `solvers/match_aou_backend.py` (NEW) + `solvers/match_aou_p1_milp_solver.py`
(NEW) + `rl/training/graph_episode_setup.py` + `rl/training/graph_reward.py` +
`rl/training/graph_train.py` + `rl/training/graph_rollout.py` +
`rl/training/graph_benchmark_preflight.py` (`8f0d250`, integrated `9979910`, PR #54 — §7).**

WHICH MATCH-AOU objective a run solves is now a first-class, INDEPENDENT, EXPLICIT
selector. It adds no episode mechanism: the bounded-backoff geometry, the FD certification
physics, the post-FD boundary semantics, the continuation-reference ARITHMETIC, the Task-4
selector / sampler / manifest / persistence, the Task-5 quota / budget / preflight, the
opt-in early-stopping rule and the per-wake diagnostics are exactly the contracts above.
What changes is WHICH ALLOCATION is optimal, and therefore what the rest of the pipeline is
built on top of.

| id | what it solves |
|---|---|
| `legacy_minlp_v1` (**DEFAULT**) | the FROZEN general MINLP `match_aou_MINLP_solver.MatchAou` through BONMIN — the historical objective, `EPSILON = 1e-6` and all (§2) |
| `p1_milp_v1` | the deterministic `p = 1` MILP `match_aou_p1_milp_solver.MatchAouP1MILP` through SciPy/HiGHS — exact covered utility, **no `EPSILON` anywhere in it** |

`MATCH_AOU_BACKENDS` is the CLOSED set of exactly those two ids and
`DEFAULT_MATCH_AOU_BACKEND` is `legacy_minlp_v1`; `resolve_match_aou_backend` is the ONE
validation site and `uses_p1_milp` the ONE predicate.

**`legacy_minlp_v1` REMAINS THE HISTORICAL DEFAULT AND IS THE PRESERVED PATH.** A caller
that says nothing gets the frozen `MatchAou` through BONMIN — the objective **every
approved measurement was taken on** (`737b4bf`, `bf1e045f`, and the R1 generalized
measurement at `4af6c5aa…` — §7). The keyword-OMISSION discipline `_artifact_kwargs` /
`_ctde_kwargs` / `_cardinality_kwargs` / `_generalized_setup_kwargs` already use is applied
here too: `graph_train._backend_setup_kwargs` and `graph_episode_setup._backend_kwargs`
return `{}` on the historical backend, so a legacy run makes EXACTLY its pre-integration
calls rather than passing a keyword carrying a value the callee would have chosen itself —
the stronger invariance claim.

**THE SELECTION RULES, ALL OF THEM DELIBERATE.**

- **SELECTION IS EXPLICIT, AND THE DESIGN CONSTRAINS THE VALID VALUE SET WITHOUT EVER
  MAKING THE CHOICE.** The backend is NEVER inferred from `episode_design`, from task
  probabilities, from whether BONMIN or SciPy happens to be importable, or from any other
  state — a run that reached a different allocation objective because of what was installed
  on the machine would be a measurement nobody chose — and `_backend_setup_kwargs` is a
  SEPARATE helper from `_generalized_setup_kwargs` precisely so no coupling is suggested
  beyond the one stated here. It remains ORTHOGONAL to `training_mode`. **What it is NOT is
  fully orthogonal to `episode_design`:** `fixed_cell_v1` and `generalized_v1` each accept
  EITHER approved objective, while **`generalized_v2` REQUIRES `p1_milp_v1` and REFUSES
  `legacy_minlp_v1` before any episode executes** — a REFUSAL of a contradictory request,
  never a selection made on the run's behalf and never a silent override. The reason is
  stated in the GENERALIZED-V2 block below: V2 resolves its hidden load from the number of
  NON-EMPTY ROUTES the known-only allocation produced, and the legacy objective's EPSILON
  stacking incentive changes which allocations are optimal and therefore which egos are
  routed, so two objectives would make one design id mean two population selectors.
  *(SUPERSEDED, and corrected here: this bullet previously read "It is ORTHOGONAL to
  `episode_design` and to `training_mode`: either design may run under either backend."
  That was accurate before PR #57 and is not now.)*
- **THERE IS NO `auto`, NO FALLBACK IN EITHER DIRECTION, AND NO PER-SOLVE SWITCHING.** A
  refused P1 solve is never rescued by the legacy solver, and an unknown id RAISES rather
  than resolving to the default — a run that quietly solved the legacy objective while its
  config said `p1_milp_v1` is a mislabelled measurement, which is worse than a crash.
- **ONE EPISODE STORES AND USES ONE BACKEND, COHERENTLY.**
  `EpisodeContext.match_aou_backend` is the SINGLE STORED SOURCE, `_finish_context` takes it
  as a **REQUIRED keyword** and RESOLVES rather than trusts it, and every DEFERRED reference
  solve reads it back off the context through `graph_reward.episode_match_aou_backend`
  instead of re-deciding. So the known-world `A_init`, the static t=0 reference, the clean
  t=0 reference, the damaged continuation checkpoint and the unrealized-event compatibility
  reference are ALL solved under the same objective. A third construction path that omitted
  the keyword could reach a context claiming the historical backend while its plan had been
  solved under P1 — which is exactly the shape the required keyword refuses.

**A BACKEND / CONFIGURATION FAULT ABORTS — IT IS NEVER ORDINARY EPISODE ATTRITION.**
`MatchAouBackendError` is a `RuntimeError` (deliberately NOT a `ValueError`), a sibling of
the integrity-abort family this project already routes that way, and it lives beside the ids
it classifies because the solver layer must not import the trainer. It is raised for an
unknown id, for the P1 backend selected while its SciPy/HiGHS stack is not importable, and
for an input outside the P1 contract reaching a P1-selected runtime (a multi-step task,
`p != 1`, or precedence) — `graph_episode_setup` re-raises the solver's own
`P1MilpUnsupportedInputError` / `P1MilpBackendUnavailableError` as this one stable type.
Every one of those says the INSTRUMENT is configured against a domain it does not model,
which implicates every episode it touched, so `graph_train`, `graph_rollout` and
`graph_benchmark_preflight` re-raise it AHEAD of every broad handler: it names no pipeline
stage, is NEVER written to `episode_failures.jsonl`, never counted against a condition or
stratum tally, never entered into `skip_and_account_v1`, never replaced by the next training
seed, never turned into a rejected benchmark candidate — **and NEVER answered by silently
solving the other objective.** **A solve that simply did not reach acceptable optimality is
NOT this exception** and keeps the existing solve-failure / `ReferenceIntegrityError`
attrition semantics unchanged.

**IMPORT WEIGHT — LAZY, AND NOT RE-EXPORTED.** `match_aou_backend` holds ids, validation and
`load_p1_milp_solver` and nothing else; it imports neither the P1 module nor SciPy at module
scope, so naming a backend costs a caller nothing. **The P1 solver is NOT re-exported
through `match_aou.solvers.__init__`**, whose surface is still exactly `MatchAou` and
`round_trip_cost`, so importing the package does not reach the MILP stack.

**WHAT `p1_milp_v1` IS SPECIALIZED TO, AND WHAT IT REFUSES.** Deterministic `p = 1`,
**one-step tasks**, **no precedence**. Its variables are `x[i,j]` binary plus a
NON-integral `0 <= y[j] <= 1` pinned from both sides by `x[i,j] <= y[j]` and
`y[j] <= sum_i x[i,j]`; the objective is exactly `sum_j utility_j * y_j`. **CAPABILITY and
the MOVEMENT BUDGET are preserved exactly** — capability by pinning an incapable pair's
upper bound to `0`, the budget by importing `round_trip_cost` from the FROZEN module rather
than re-deriving its geometry, with the legacy missing-location handling reproduced. There
is no step dimension, no nonlinear term, no exponent, no `EPSILON`, no precedence variable
and no Big-M. **`optimal` is the ONLY accepted termination** (the legacy `locallyOptimal` is
a nonlinear-solver concept), and a time/iteration-limited incumbent is deliberately mapped
to a NON-accepted condition rather than reported as an optimal allocation. **An input
outside that contract is REFUSED, never coerced**, and **DEGENERACY IS DELIBERATELY NOT
BROKEN** — no one-agent-per-task bound, no fuel or assignment penalty, no epsilon
tie-breaker and no lexicographic second objective, because inventing one would silently
change allocation semantics the pipeline was measured against.

**IT IS NOT A TRANSPARENT SPEED OR PERFORMANCE REPLACEMENT, AND NO EQUIVALENCE IS CLAIMED.**
At `p = 1` the legacy objective still pays a strictly positive
`utility * (EPSILON - EPSILON^2)` for each REDUNDANT agent on an already-covered task, so
stacking is genuinely optimal FOR THAT OBJECTIVE; the P1 formulation removes that incentive
outright. **Removing it changes WHICH ALLOCATIONS ARE OPTIMAL.** The two objectives can
agree on the optimal COVERED-TASK SET in the exercised domain — that is an OBSERVATION from
engineering comparison, not a guarantee — but they do **NOT** generally share an optimal
ALLOCATION set, and the difference is systematic rather than incidental. **CONSEQUENCE,
STATED PLAINLY: selecting `p1_milp_v1` can change `A_init`, and because route-relative
hidden placement predicts routes FROM `A_init` (§5, B2 / bounded backoff), it can change the
hidden geometry, episode feasibility, the certified FD event and therefore the POPULATION
IDENTITY itself.** That is why it is a reviewed research decision and not a drop-in swap.

**THE BENCHMARK PREFLIGHT USES THE SAME SELECTED BACKEND AS THE LATER RUN.**
`graph_benchmark_preflight` passes `_backend_setup_kwargs(cfg)` into `setup_episode` and
records `match_aou_backend` on its REPORT — deliberately on the report and not inside the
manifest, because **THERE IS NO MANIFEST SCHEMA CHANGE FOR THE BACKEND**: the manifest
already carries each world's id-free frozen identity, and `require_world_matches_manifest`
REFUSES a member whose RECONSTRUCTED geometry disagrees, so **reconstructed frozen identity
remains the enforcement boundary** and a manifest frozen under one backend cannot be
silently reused under the other. The report key is what lets a reader see WHY, not merely
that it happened.

**REWARD AND REFERENCE VALUATION ARE OBJECTIVE-COHERENT — AND THE REWARD FORMULA IS
UNCHANGED.** `graph_reward.plan_value(solution, tasks, *, backend=...)` DEFAULTS to the
historical backend and, on it, is the **unchanged EPSILON arithmetic, operand for operand**;
on `p1_milp_v1` it is exact covered utility (`_p1_plan_value`), which likewise REFUSES a
multi-step task or `p != 1` as a `MatchAouBackendError`. The backend passed is the EPISODE's
own stored one, so scoring a P1 allocation with legacy arithmetic — or the reverse — cannot
happen; that would normalize an episode by a number no solver ever optimized.
**UNCHANGED BY THIS INTEGRATION:** `U_prefix`, `U_post`, `realized_utility`, the aircraft
penalty, the regret epsilon `eps_regret` (still a DIVISION GUARD, distinct from the solver
`EPSILON`), terminal-on-last credit placement, and the fact that the reward is **NEVER
CLAMPED**. PPO, GAE, the encoder, the critic, the action set, the trigger layer, the
fuel-damage mechanism, `DETECTION_KM`, the B2 geometry, the seed formulas and the vendored
BLADE engine are all untouched, the frozen `match_aou_MINLP_solver.py` is untouched, and the
episode-outcome schema stays at version 3.

**CONFIGURATION AND PROVENANCE.** `TrainConfig.match_aou_backend` and
`RolloutConfig.match_aou_backend` (both defaulting to `legacy_minlp_v1`) are settable from a
JSON preset and from `--match-aou-backend`, whose `choices` are the closed id set; both
`validate()` methods resolve the id BEFORE any compute, and `graph_benchmark_preflight`
exposes the same flag. `run_config.json:/provenance/solver` records `match_aou_backend`
beside the unchanged BONMIN probe — the backend is part of a result's identity in exactly
the way the solver executable is. **No repository preset selects `p1_milp_v1`**:
`configs/graph_train/final_cell_probe.json` remains the ONLY repository preset and is
untouched.

**NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no backend id, solver audit field,
termination string or objective value enters `GraphObservation` or
`CentralGraphObservation`. **NO SCIENTIFIC MEASUREMENT WAS PRODUCED BY THIS INTEGRATION, AND
NO P1 PERFORMANCE, BENEFIT, LEARNING OR COMPARISON CLAIM MAY BE PRE-CLAIMED** (§8).

## 4. Code routing

| … | Go to |
|---|---|
| Route a REFERENCE fault — accounted attrition vs measurement-integrity ABORT | `rl/training/graph_reward.py` (`REFERENCE_FAULT_REASONS`, `REFERENCE_ATTRITION_REASONS`, `ReferenceIntegrityError(..., reason=...)` and `.is_measurement_integrity`, `reference_fault_aborts`) + `rl/training/graph_episode_setup.py` (the reason-carrying raise sites in `_solve_reference` / `build_t0_reference` / `build_continuation_reference`) + `rl/training/graph_train.py` (the `reference_fault_aborts` branches in `_run_one_episode`'s run and reward blocks, and the `except (_VisualArtifactError, MeasurementIntegrityError, FuelDamageIntegrityError, BenchmarkIdentityError, ReferenceIntegrityError)` re-raises in the train and both eval attempt handlers). **RESEARCH-VALIDITY / GRADE A**: `reason` is REQUIRED and closed, the routing reads the SLUG and NEVER the message, an unanswered solve is ordinary accounted attrition inside `skip_and_account_v1`, and every other reason ABORTS exactly as a roster or certificate fault does (§5) |
| SELECT the MATCH-AOU ALLOCATION OBJECTIVE — `legacy_minlp_v1` (DEFAULT, frozen MINLP through BONMIN) vs `p1_milp_v1` (deterministic `p = 1` MILP through SciPy/HiGHS) | `solvers/match_aou_backend.py` (`MATCH_AOU_BACKENDS`, `MATCH_AOU_BACKEND_LEGACY_MINLP_V1` / `MATCH_AOU_BACKEND_P1_MILP_V1`, `DEFAULT_MATCH_AOU_BACKEND`, `resolve_match_aou_backend`, `uses_p1_milp`, `load_p1_milp_solver`, `MatchAouBackendError`) + `solvers/match_aou_p1_milp_solver.py` (`MatchAouP1MILP`, `P1MilpResults`, `P1MilpSolverSection`, `P1MilpUnsupportedInputError`, `P1MilpBackendUnavailableError`, `TERMINATION_OPTIMAL`) + `rl/training/graph_episode_setup.py` (`_backend_kwargs`, `solve_and_normalize_audited(..., backend=...)`, `EpisodeContext.match_aou_backend`, `_finish_context`'s REQUIRED `match_aou_backend` keyword) + `rl/training/graph_reward.py` (`plan_value(..., backend=...)`, `_p1_plan_value`, `episode_match_aou_backend`) + `rl/training/graph_train.py` (`TrainConfig.match_aou_backend`, `_backend_setup_kwargs`, `--match-aou-backend`) + `rl/training/graph_rollout.py` and `rl/training/graph_benchmark_preflight.py` (the same field and flag). **RESEARCH-VALIDITY / GRADE A**: selection is EXPLICIT — never inferred from `episode_design`, from task probabilities or from what is installed — there is **no `auto` and no fallback in either direction**, an unknown id RAISES, and ONE episode stores and uses ONE backend for every solve it performs. **THE DESIGN CONSTRAINS THE VALID VALUE SET WITHOUT MAKING THE CHOICE:** `fixed_cell_v1` and `generalized_v1` each accept EITHER approved objective, while **`generalized_v2` REQUIRES `p1_milp_v1` and REFUSES `legacy_minlp_v1` before execution** (`GENERALIZED_V2_REQUIRED_BACKEND`, checked in both `validate()` methods) — a REFUSAL of a contradictory request, never an override. So it is NOT fully orthogonal to `episode_design`; it remains orthogonal to `training_mode`. **`legacy_minlp_v1` is the DEFAULT and is the objective every approved measurement was taken on**, reached through keyword OMISSION so a legacy run makes exactly its pre-integration calls. **P1 is NOT a transparent speed/performance swap**: it removes the legacy EPSILON stacking incentive, so it changes which allocations are optimal and can change `A_init`, the hidden geometry, feasibility and population identity. The frozen `match_aou_MINLP_solver.py` is untouched, the P1 solver is LAZY-loaded and NOT re-exported through `match_aou.solvers`, and **no repository preset selects `p1_milp_v1`** (§5, §8) |
| Route a BACKEND / CONFIGURATION fault — abort, never attrition and never a fallback | `solvers/match_aou_backend.py` (`MatchAouBackendError` — a `RuntimeError`, deliberately NOT a `ValueError`) + `rl/training/graph_episode_setup.py` (the re-raise of `P1MilpUnsupportedInputError` / `P1MilpBackendUnavailableError` as that one stable type) + the `except MatchAouBackendError: raise` guards in `rl/training/graph_train.py` (`_run_one_episode`'s setup, run and reward blocks and the train / legacy-eval / benchmark-eval attempt handlers), `rl/training/graph_rollout.py` and `rl/training/graph_benchmark_preflight.py`. **RESEARCH-VALIDITY / GRADE A**: an unknown id, an unreachable P1 stack, or an input outside the P1 contract (multi-step, `p != 1`, precedence) all say the INSTRUMENT is configured against a domain it does not model, so the run ABORTS — it names no pipeline stage, is NEVER written to `episode_failures.jsonl`, never counted against a condition or stratum tally, never entered into `skip_and_account_v1`, never replaced by the next training seed, never turned into a rejected benchmark candidate, **and NEVER answered by silently solving the other objective**. A solve that merely did not reach acceptable optimality is NOT this exception and keeps the existing attrition semantics (§5) |
| Change the reward | `rl/training/graph_reward.py` (`compute_episode_reward`/`plan_value`/`realized_utility`/`RewardConfig`) |
| SELECT or change the REWARD-REFERENCE POLICY (`static_t0_v1` vs GENERALIZED-V1 `event_conditioned_continuation_v1`) | `rl/training/graph_reward.py` (`REFERENCE_POLICIES`, `REFERENCE_POLICY_STATIC_T0_V1`, `REFERENCE_POLICY_EVENT_CONDITIONED_V1`, `uses_event_conditioned_reference`) + `rl/training/graph_episode_setup.py` (`setup_episode(..., reference_policy=...)`, `_resolve_reference_policy`, `EpisodeContext.reference_policy` / `t0_reference_tasks`, `_t0_reference_or_deferred`). **RESEARCH-VALIDITY / GRADE A**: `static_t0_v1` is the DEFAULT and is the behaviour the approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) measurements were taken on — moving it moves what those measurements mean. `EpisodeContext.reference_policy` is the ONE stored source and `uses_event_conditioned_reference` the canonical runtime predicate; an unknown id RAISES before any BLADE object exists. **SELECTING it is now done through `episode_design`, never through a standalone field** — see the episode-design row below (§5, §8) |
| Change the CONTINUATION CHECKPOINT TIMING or the REFERENCE CONSTRUCTION | `rl/training/graph_tick_loop.py` (`run_episode`'s three reference sites — the pre-first-tick CLEAN t=0 build, the `build_continuation_reference` call at the TOP of the firing tick immediately after `maybe_apply`, and the episode-exit `damaged_event_unrealized_t0` build BEFORE the recording export — plus `EpisodeResult.reference`) + `rl/training/graph_episode_setup.py` (`build_t0_reference`, `build_continuation_reference`, `_continuation_agents`, `_reference_universe`, `_solve_reference`, `_reference_aircraft_utility`, `SolveAudit`, `solve_and_normalize_audited`, `SOLVE_NOT_ATTEMPTED` / `SOLVE_TERMINATION_UNAVAILABLE`). **RESEARCH-VALIDITY / GRADE A**: the ORDERING is the contract — real `current_fuel` mutation → continuation reference → post-FD boundary → trigger → `central.capture` → actor decision → Phase 2 — so the reference describes the world the actor is ABOUT to decide in; the checkpoint is READ-ONLY measurement that issues no `env.step`; the task universe is the retained RAW t=0 world minus the realized prefix and NEVER a private belief; continuation agents are rebuilt from the LIVE post-event world with dead / RTB-committed / non-airborne egos EXCLUDED by recorded reason; and an unanswered solve is REFUSED (`ReferenceIntegrityError`) rather than recorded as an answered zero (§5) |
| Change `U_prefix` / `U_post` / `U_ref` ARITHMETIC or the REWARD-BEARING TARGET SCOPE | `rl/training/graph_reward.py` (`_event_conditioned_breakdown` beside the UNTOUCHED `_static_t0_breakdown`, `EpisodeReference` and its `__post_init__` reconciliation, `REFERENCE_KINDS`, `CONTINUATION_EXCLUSION_REASONS`, `ReferenceIntegrityError`, `realized_task_indices` — the ONE all-steps rule both halves share — `task_target_ids`, and `EpisodeReward`'s `u_ref` / `u_oracle` / `u_prefix` / `u_cont_ref` / `u_post` / `scored_completed_targets` / `unscored_completed_target_ids` fields). **RESEARCH-VALIDITY / GRADE A**: `U_prefix` is FROZEN at the checkpoint and never recomputed from the final `done` set; `U_post` scores ONLY continuation-allocated tasks and a kill outside that set is ACCOUNTING-ONLY; `U_aircraft` comes from the reward-bearing reference universe; the reward is NEVER clamped; `u_oracle` is `None` under the opt-in policy, while on the historical one `reference_policy` is `static_t0_v1`, `u_ref` EQUALS `u_oracle`, the OPTIONAL scalar/count checkpoint fields are `None` — never `0.0` / `0` — and `unscored_completed_target_ids` keeps its typed empty-tuple default `()` rather than `None`; and terminal-on-last credit placement is IDENTICAL under both policies (§5) |
| Change the solver objective/constraints | `match_aou_MINLP_solver.py` (extreme caution — §2). It is the `legacy_minlp_v1` backend and stays FROZEN; the SEPARATE `p1_milp_v1` objective lives in `match_aou_p1_milp_solver.py` and is selected, never substituted — see the backend row above |

## 5. Known limitations and open items

- **Solver 2:1 stacking (scenario-design fix, NOT solver constraints):** the anti-div-by-zero `EPSILON` nudges utility enough to assign 2 agents even at `probability=1.0`; a redundant agent chasing an already-killed target never proximity-confirms, so episodes end via `truncated`. The learned policy should recover this via `SELF_PRESERVATION_ABORT`→RTB once trained; the root fix is `EPSILON`/scenario-side.
- **bonmin needs a solve timeout at `known ≤ 2`.** With ≤2 known tasks against the
  4-agent fleet, branch-and-bound hits a symmetry stall — one measured episode took
  ~15 min against ~45 s typical. The locked cell is clear of it and
  `TrainConfig.validate` now WARNS, but a timeout is required before any low-known or
  n-randomized config enters training.
- **Raw utility 480 vs reward-side `U_oracle = 479.99968` — keep the distinction.** The
  six airbase targets sum to exactly `6 × 80 = 480` raw utility, but `graph_reward.plan_value`
  is bit-faithful to `MatchAou._add_objective` and carries the frozen anti-div-by-zero
  `EPSILON = 1e-6`: a 1-agent task contributes `80·(1 − 1e-6)` and a 2-agent task
  `80·(1 − 1e-12)`, giving `479.99968` for the measured seed-0 allocation. Both numbers are
  correct for their own operand; do not "fix" either, and do not compare one to the other.
