# Construction and fuel-damage contract — hidden cardinality, FD designs, certification and post-FD boundaries

> **Read this when** you change or review hidden-target placement or hidden-cardinality
> policies, either fuel-damage design (FD-BASELINE-v1 or FD-VARIABLE-SEVERITY-v1), certified
> FD eligibility, the live certificate check, post-FD completion-boundary wakes, or how frozen
> BLADE tick behaviour bears on the fuel-damage event.
>
> **Status: normative, current technical contract** for the code on `main`. Lock history is in
> [`implementation.md`](../history/implementation.md), block provenance in
> [`documentation_migration.md`](../documentation_migration.md), and current run and evidence
> state in the [handoff](../../graph_rl_project_handoff.md). The frozen-engine live-list
> behaviour that the live certificate check depends on is stated in
> [`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion).
>
> Related contracts: [runtime](runtime.md) · [reward and solvers](reward_solvers.md) ·
> [training and benchmarks](training_benchmarks.md) · [artifacts and metrics](artifacts_metrics.md).

## 1. Hidden-cardinality policies

**GENERALIZED-V1 HIDDEN CARDINALITY — TWO EXPLICIT POLICIES ON ONE CONSTRUCTION SEAM —
`rl/training/graph_hidden_placement.py` + `rl/training/graph_episode_setup.py`
(`5b55ca3`).**

The policy is SELECTED by `setup_episode(..., hidden_policy=...)` and is NEVER inferred.
`HIDDEN_CARDINALITY_POLICIES = (HIDDEN_POLICY_EXACT_V1, HIDDEN_POLICY_BOUNDED_BACKOFF_V1)`;
an unknown id raises before any BLADE object exists. **ONLY the PLACEMENT STEP differs
between the two** — the patch, the env-2 reload, env-2's authority, the raw world snapshots,
the re-materialized known tasks and the oracle solve are the SAME code for both, because
both patch in exactly as many targets as were REALIZED.

- **`exact_v1` IS THE DEFAULT, AND IT IS THE PRESERVED HISTORICAL CONTRACT.** Everything
  stated above about `place_hidden_targets` still holds: sorted-ego iteration, one placement
  per non-empty ego route, the same loud failure messages, the same
  `len(placements) == n_hidden` check, and `n_hidden=0` still a legal probe. Its geometry,
  chosen legs, sampled fractions, sampled offsets AND the episode rng's post-call STREAM
  POSITION are test-pinned against the pre-generalized implementation, so one added, removed
  or reordered draw fails. **The approved Phase-A (`737b4bf`) and FD-VARIABLE-SEVERITY-v1
  (`bf1e045f`) measurements were taken on this path and remain measurements OF IT.** On this
  path `EpisodeContext.construction_audit` stays `None` and construction `split_meta` grows
  NO new key — that ABSENCE is how a reader tells which policy ran.
- **`bounded_backoff_v1` IS AN OPT-IN ADDITION BESIDE IT, never a rewrite of it.** It
  enforces the approved GENERALIZED CELL first, against the RAW pre-solve world and BEFORE
  any solve (`_require_generalized_cardinality`, so an out-of-cell request costs no bonmin
  call and leaves no partial construction behind): `A` agents in
  `GENERALIZED_AGENT_COUNTS = (2, 3, 4)`, `K == A` **RAW KNOWN** targets (never an
  allocated-only count, which omits every unselected target), and `1 <= H_requested <= A` —
  so a requested world holds `A + 1` … `2A` targets. `n_hidden=0` is REFUSED here; it is an
  `exact_v1` probe only.

**THE WALK (`place_hidden_targets_bounded`), and what it may and may not do.** The candidate
population is EVERY scheduled ego in the AUTHORITATIVE pre-solve agent sequence
(`agent_ordinals`, which construction supplies as env-1's ORDERED agent ids) — **including
egos the allocated-only `A_init` omitted entirely**, which are recorded as `no_route` rather
than being invisible. A candidate's identity is its ORDINAL in that sequence. Then, in
order: (1) `_ordinal_permutation` draws a seed-driven Fisher-Yates permutation of the
ordinals, written out explicitly rather than delegated to `random.shuffle` so the exact draw
sequence is a stated contract; (2) `_candidate_substream_seeds` derives ONE independent
64-bit geometry seed per ordinal in ONE bounded burst, in ascending ordinal order, **BEFORE
any candidate is attempted**; (3) candidates are attempted in the permuted order, each on
its OWN `random.Random` substream; (4) the walk STOPS at `hidden_requested` successes or at
candidate exhaustion; (5) the result is ACCEPTED when `H_realized >= 1`.

- **STEPS 1 AND 2 BOTH PRECEDE EVERY ATTEMPT, AND THAT IS LOAD-BEARING.** Whether candidate
  N succeeded or was rejected cannot shift candidate N+1's fraction and offset draws, and
  the episode rng's END POSITION depends only on the candidate COUNT, never on how many
  attempts the walk took.
- **ORDINALS, NEVER UUID TEXT.** Generated agent and target ids are not seed-derived
  ([runtime §2](runtime.md#2-episode-setup-stage-0)),
  so a permutation keyed on id strings would make an episode's hidden geometry
  irreproducible across runs of the same seed. Two rosters with opposite lexical orders and
  the same ordinal→route mapping produce identical results. Substreams come from the seeded
  rng, never from `hash()` (salted per process).
- **THE GEOMETRY IS THE SAME APPROVED SINGLE-ROUTE B2 GEOMETRY, REUSED AND NOT
  REIMPLEMENTED.** Both policies share ONE leg-selection site, `_select_leg`, extracted
  VERBATIM from `place_hidden_targets`'s per-ego body — including the fact that the leg
  choice is the ego's FIRST draw, before `_construct_placement`'s fraction and offset — and
  both then run the same `_construct_placement` and the same independent
  `validate_placement` re-measurement. The leg rule, the guaranteed portion `G = L - D`, the
  offset budget, the `gap > 2·D` nearest-neighbour margin and the sensing guard are NONE of
  them weakened, re-tuned or bypassed. **AT MOST ONE hidden target per ego route** —
  multiple placements on one route are deliberately out of scope in this version, and a
  repeated ordinal raises.
- **`H_realized < H_requested` IS A LEGITIMATE RECORDED OUTCOME, NEVER A REPAIRED ONE.**
  Nothing is duplicated, padded, truncated, redistributed, retried or relaxed; the agent
  population, the seed, the world and the REQUESTED count are never silently altered to make
  a world succeed — a silently reduced request is what makes a denominator unreadable.
  `H_realized == 0` is a REFUSAL (`HiddenPlacementError`, naming every candidate outcome).

**ACCOUNTING — `EpisodeContext.construction_audit: Optional[ConstructionAudit]`.** A TYPED
record on the returned context, not a console line, because requested-vs-realized is meant
to be read as a DISTRIBUTION across episodes (a HIGH hidden-load stratum that quietly
collapses into the LOW one is not a stratum). It carries `policy`, `agent_count`, and
requested-vs-realized `known` / `hidden` / `total`, plus the embedded `BoundedBackoffAudit`:
`candidate_count`, `candidate_order`, `considered_ordinals` (a PREFIX of the order, because
the walk really is bounded), the per-candidate `BackoffCandidate` records with their stable
machine-readable `BACKOFF_REJECTION_REASONS` slug (`no_route`, `route_unresolvable`,
`no_eligible_leg`, `geometry_rejected`), `selected_ordinals`, and the id-free
`geometric_fingerprint`. Two things about it are contractual:

- **THE COUNTS ARE WORLD COUNTS, NOT ALLOCATION COUNTS.** `known_realized` and
  `total_realized` come from the RAW pre-solve snapshots `known_target_ids` /
  `executed_target_ids`, never from `belief_tasks` / `oracle_tasks` — the Stage-0
  "WORLD INVENTORY IS NOT ORACLE ALLOCATION" contract honoured, not repeated. It is
  VERIFIED rather than trusted: the raw executed world must equal `known + H_realized`, and
  the audit's own realized count must equal the number of placements, else `RuntimeError`.
- **NOTHING HERE REACHES THE ACTING PATH.** No count, no policy id, no candidate ordinal
  and no rejection reason enters `GraphObservation`. A count of what is hidden is exactly
  the privileged quantity an ego cannot sense
  ([`CLAUDE.md` §3](../../CLAUDE.md#3-architecture--the-load-bearing-invariants)). The generalized-only `split_meta` keys
  (`hidden_policy`, `hidden_realized`, `construction_audit`) are added ONLY under this
  policy, so nothing reading a historical record sees a new field.

**BOTH HARNESSES NOW SELECT THIS POLICY — THROUGH THE BUNDLE, NEVER ON ITS OWN
(GENERALIZED-V1 Task 4, `db79013`).** `TrainConfig.episode_design` /
`RolloutConfig.episode_design` resolve `hidden_policy` from the one selector
(`graph_generalized.resolve_episode_design`), and `graph_train._generalized_setup_kwargs` /
`graph_rollout.run_rollout` pass it to `setup_episode` ONLY on the generalized path — on
`fixed_cell_v1` the keyword is OMITTED entirely, so setup resolves its own `exact_v1`
default exactly as it always did and the historical call is byte-unchanged. **There is no
standalone `hidden_policy` field on either config**, so this policy cannot be enabled apart
from the bundle. The requested-vs-realized
audit is PERSISTED and AGGREGATED — see
[artifacts and metrics §4](artifacts_metrics.md#4-generalized-persistence-and-aggregates).
Measurements taken on these designs are listed in the [handoff](../../graph_rl_project_handoff.md).

The GENERALIZED-V2 route-relative hidden load, which drives `bounded_backoff_v1` with a
request resolved after the known-only solve, is contracted in
[training and benchmarks §8](training_benchmarks.md#8-generalized-v2-population).

## 2. FD-BASELINE-v1

**FD-BASELINE-v1 — the LEGACY difficulty factor, and the PRESERVED Phase-A semantics —
`rl/training/graph_fuel_damage.py`** (consumed by `graph_tick_loop.run_episode`,
`graph_train` and `graph_rollout`).

THE **ONE** SELECTED DIFFICULTY FACTOR of the Phase-A reference baseline cell, and the
design the approved Phase-A long-baseline measurement (`737b4bf`) was taken on. The
scenario is otherwise UNCHANGED: 3 agents, 3 known + 3 route-relative hidden airbase
targets, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams=False`,
`probability = 1`, unchanged BLADE weapon lethality, frozen solver, unchanged PPO. No
second factor is bundled in.

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
  The range note in [reward and solvers §1](reward_solvers.md#1-terminal-reward-stage-7) describes
  the `c = 0.0` case.
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

## 3. FD-VARIABLE-SEVERITY-v1

**FD-VARIABLE-SEVERITY-v1 — the mild/severe extension of the SAME event —
`rl/training/graph_fuel_damage.py` + `rl/training/graph_train.py` +
`rl/training/graph_rollout.py`.**

MERGED AND LOCKED (`eecc9b5`), and **MEASURED ONCE — the actor-only baseline at
measured code SHA `bf1e045f` is EXECUTED, independently reviewed and
`APPROVE — VALID MEASUREMENT`, and its PRIMARY behavioural finding is NEGATIVE: the
deterministic held-out actor showed NO severity-conditioned FD-wake meta-action
separation** ([measurement history](../history/measurements.md#2-measurement-records) owns the
record and every denominator). Nothing beyond that record may be claimed for this design. It is an ADDITIONAL
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
  research task and was NOT implemented here**.

The matched clean / mild / severe triad evaluation and the durable per-episode outcome stream
that measure this design are contracted in
[artifacts and metrics §3](artifacts_metrics.md#3-matched-triads-and-the-episode-outcome-stream).

## 4. Certified FD eligibility, live certificate check and post-FD boundaries

**GENERALIZED-V1 CERTIFIED FD ELIGIBILITY + POST-FD COMPLETION-BOUNDARY ADAPTATION —
`rl/training/graph_fuel_damage.py` + `rl/training/graph_tick_loop.py` +
`rl/action/graph_trigger.py` + `utils/blade_utils/blade_graph_executor.py` +
`rl/training/graph_train.py` (`185d39f`).**

TWO OPT-IN policy seams, both carried on `FuelDamageParameters`, both VERSIONED strings, and
both DEFAULTING to the merged legacy behaviour — so every existing construction site,
`TrainConfig.fuel_damage_parameters()` and `RolloutConfig.fuel_damage_parameters()`
included, obtains the legacy default automatically:

| knob | DEFAULT (legacy, preserved) | GENERALIZED-V1 addition |
|---|---|---|
| `eligibility_policy` | `legacy_selected_ego_v1` | `certified_both_severities_v1` |
| `post_fd_wake_policy` | `single_wake_v1` | `completion_boundary_v1` |

`FD_ELIGIBILITY_POLICIES` / `POST_FD_WAKE_POLICIES` are the closed sets `validate()` checks;
`FuelDamageParameters.certified_eligibility` and `.completion_boundary_wakes` are the ONE
predicate behind each question, so neither can be spelled two ways. Both resolved values,
and both derived booleans, are recorded by `to_record()`, so ONE schema reads both designs.

**THE LEGACY PATHS ARE PRESERVED AND ARE STILL THE MEASURED ONES.** Under
`legacy_selected_ego_v1` the `fuel_damage_v1` stream's second draw still picks uniformly
among the SORTED ids of egos with a non-empty initial route, the window is still the
PLANNED distance projection the controller re-validates live, the live CHECK ORDER is
untouched (`_live_legacy_target` / `_live_variable_target` stay in separate helpers), and
**every live failure is still an ordinary `FuelDamageError` accounted by
`skip_and_account_v1`**. Under `single_wake_v1` the damaged ego gets the immediate
`FUEL_DAMAGE` wake and nothing else. The approved FD-BASELINE-v1 (`737b4bf`) and
FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) measurements were taken on these defaults and remain
measurements OF THEM.

**A THIRD PRIVATE RNG DOMAIN, AND THE SEPARATION IS LOAD-BEARING.**
`FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN = "fuel_damage_eligibility_v1"`, via
`derive_fuel_damage_eligibility_seed` (the same SHA-256 construction as the other two),
drives the certified walk's candidate permutation ALONE. Taking that permutation from
`fuel_damage_v1` would insert draws BETWEEN that stream's mixture bit and its ego selection
and change WHICH EGO every legacy damaged episode picks — silently invalidating the approved
measurements instead of extending them. With three domains the decisions are orthogonal:
eligibility cannot move the condition or the legacy ego, and neither can move eligibility.

**CERTIFIED ELIGIBILITY — FD CAPABILITY BECOMES A PROPERTY OF WORLD ACCEPTANCE.**
`_certified_eligibility_walk` is a DETERMINISTIC BOUNDED walk over the candidate population
`ctx.agent_ids` — the AUTHORITATIVE scheduled agent sequence — where a candidate's identity
is its ORDINAL there, never generated id text (generated ids are not seed-derived). Egos the
allocated-only `A_init` omitted are INCLUDED and rejected truthfully as `no_route`, because
"this ego had nothing to fly" is a finding a silently shortened population could not report.
At most `len(agent_ids)` candidates, each attempted at most ONCE, stopping at the FIRST
acceptance; nothing is retried, no seed redrawn, no world replaced, no severity changed and
no episode converted to clean.

- **THE WALK RUNS FOR EVERY CONDITION — CLEAN INCLUDED — and depends on the EPISODE SEED
  ALONE.** That is the whole mechanism: clean, mild and severe members of the same
  world+seed walk the same candidates in the same order and certify the SAME ego, so the
  three share ONE accepted-world support and a matched group is constructible BY DESIGN
  rather than discovered afterwards. A CLEAN plan keeps `FuelDamagePlan.ego_id is None` —
  that field still means "the ego this episode actually damages" — and the counterfactual
  selection lives on `FdEligibilityAudit.selected_ego_id` instead.
- **WHAT A CANDIDATE MUST SATISFY (`certify_fd_candidate`, PURE — every input is a plain
  number, `Location`, string or sequence), checked in this order:** (1) LEG 1 IS REAL —
  positive great-circle length, and the `leg_progress_threshold` is actually crossed while
  still flying it; (2) THE PRE-EVENT PREFIX IS STABLE — at EVERY Phase-1 position strictly
  BEFORE the event tick the ego must neither sense a live world target absent from its own
  t=0 belief inventory (`pre_event_popup_risk`; a POP_UP would wake the actor and move the
  route out from under the certificate, and the test covers ANY such target, not merely the
  construction path's "hidden" half) nor already be inside the unified arrival/detection
  radius of its current first assigned target (`pre_event_assignment_boundary`; Phase 2
  could attack, confirm and advance before the certified state exists); (3) BOTH SEVERITY
  BANDS EXIST AT THE EVENT STATE ON THE SAME EGO — `F_rtb < F_continue < F_before` with each
  of the two intervals wider than ONE TICK OF BURN; (4) THE SAME HOLDS ACROSS THE WHOLE
  TOLERATED BRACKET.
- **PRE-EVENT STABILITY IS CERTIFIED, NEVER ENFORCED AT RUNTIME.** No legitimate actor
  trigger is suppressed and no actor behaviour is changed — the CANDIDATE is rejected
  instead, truthfully. The pop-up test asks exactly the question the runtime sensor asks: it
  imports `scenario_factory.iter_enemy_targets`, the same enumeration
  `GraphPlanExecutor.sensed_target_ids` scans, and it takes `detection_km` from
  `ctx.executor.arrival_threshold_km` rather than inventing a second radius, so the
  certificate cannot disagree with the sensor.
- **THE EVENT PREDICTION IS TICK-AWARE, FROM THE FROZEN ENGINE'S OWN ONE-SECOND MODEL.**
  `engine_leg_distance_km` transcribes `get_next_coordinates` (floor and all),
  `predict_leg_states` walks the leg, and `fuel_before` is `launch - tick · fuel_rate/3600`
  because the engine burns that EVERY airborne tick including route-less ones. **No reserve
  is invented.** The ONE derived allowance is `CERTIFICATE_TICK_TOLERANCE = 1` — the
  engine's own observation quantum, deliberately NOT a free parameter (raising it would
  certify states the engine cannot produce) — and the certificate is validated across that
  whole `bracket_ticks` bracket, not at the nominal tick alone. **THAT QUANTUM IS ALSO WHAT
  DERIVES THE TWO PHYSICAL TOLERANCES the LIVE check binds** — `position_tolerance_km` (one
  tick of travel) and `fuel_tolerance` (one tick of burn) — **and the ABSOLUTE OUTER TICK is
  deliberately NOT one of them at live-validation time**; see the certified-promise contract
  below. `KILOMETERS_TO_NAUTICAL_MILES
  = 0.539957` is TRANSCRIBED here too, and is NOT the reciprocal of
  `NAUTICAL_MILES_TO_METERS/1000`: the engine uses 1852 for the FUEL question and 0.539957
  for the MOVEMENT question, and each transcription is used for the question the engine uses
  it for.
- **A ONE-ASSIGNMENT EGO IS DELIBERATELY ELIGIBLE. There is NO `>= 2`-assignment
  requirement** — imposing one would bias the generalized sample toward solver-stacked
  allocations, which is a research question and not a physical one. Such an ego simply has
  no later completion boundary to reach.
- **RECORDS.** `FdEventCertificate` (plain scalars only, so it round-trips through a jsonl
  record and can be compared field by field between two runs of the same seed) and
  `FdEligibilityAudit` (`policy`, `rng_domain`, `derived_seed`, `candidate_count`,
  `candidate_order`, `considered_ordinals`, the per-candidate `FdEligibilityCandidate`
  records with their stable `FD_ELIGIBILITY_REJECTION_REASONS` slug, `selected_ordinal`,
  `selected_ego_id`, `certificate`), both hung off `FuelDamagePlan`.

**THE TWO FAILURE ROUTINGS, AND THEY ARE OPPOSITES.**

- **SETUP INELIGIBILITY IS ORDINARY ACCOUNTED ATTRITION.** When the bounded walk considers
  every candidate and certifies none it raises a plain `FuelDamageError` carrying the stable
  marker `NO_FD_ELIGIBLE_EGO = "no_fd_eligible_ego"`. **Nothing was certified, so nothing
  was contradicted**: the attempt is wrapped as `EpisodeAttemptError("setup", ...)`, recorded
  once in `episode_failures.jsonl`, and `skip_and_account_v1` moves to the next scheduled
  seed — exactly like a B2 exact-cardinality or fuel-window failure.
- **`FuelDamageIntegrityError` IS AN INSTRUMENT FAULT AND ABORTS THE RUN.** It is a SIBLING
  of `FuelDamageError`, **deliberately NOT a subclass**, so nothing that catches one catches
  the other. It is raised only when a world CERTIFIED FD-capable — proven to support BOTH
  severities at a predicted event state before a single tick was paid for — then contradicts
  its own certificate, which means the certificate does not describe the simulator and makes
  every episode the certifier touched suspect. `graph_train` re-raises it AHEAD of every
  broad handler, in `_run_one_episode`'s setup and run blocks and in both the train and eval
  attempt handlers, now spelled `except (_VisualArtifactError, MeasurementIntegrityError,
  FuelDamageIntegrityError)`. It therefore **names no pipeline stage, is NEVER written to
  `episode_failures.jsonl`, never counted against a condition tally and never entered into
  `skip_and_account_v1`** — the same routing, and the same reason, as the roster-integrity
  contract above. It lives in `graph_fuel_damage` rather than in the trainer because this
  layer must not import `graph_train`; the trainer imports it and routes it.

**THE CERTIFIED PROMISE IS GUARDED FROM BOTH SIDES.**

- **FROM THE INSIDE — `FuelDamageController.maybe_apply`.** On a certified damaged episode
  `_require_certificate_holds` checks the promise against the LIVE aircraft first, and the
  SAME live physics every other episode runs then follows; any `FuelDamageError` it would
  have raised is re-raised as `FuelDamageIntegrityError` instead. Everything else is
  unchanged: the check happens BEFORE the mutation, so a refused event leaves the engine
  untouched, and nothing is clamped, weakened, re-planned, downgraded to the other severity
  or converted to a clean episode. **WHAT THAT LIVE CHECK BINDS WAS CORRECTED BY THE
  CERTIFIED-FD PHYSICAL-STATE INTEGRITY REPAIR (`d36e133`, integrated `edf9e84`, PR #55) —
  the contract immediately below is authoritative.**
- **FROM THE OUTSIDE — `FuelDamageController.require_certified_event_realized(*, scenario,
  ticks)`.** A CERTIFIED DAMAGED episode that ENDS with the event never having fired is the
  same instrument fault: accepting it would admit a world whose certificate did not hold
  into a scientific population as a successful damaged episode. FOUR cells, ONE of which
  raises — certified + damaged + fired → returns; **certified + damaged + NOT fired →
  `FuelDamageIntegrityError`**; **certified + CLEAN → returns**, because nothing was
  scheduled to fire and the certificate is a COUNTERFACTUAL, so `fired == False` is the
  correct outcome; **LEGACY, either condition → returns ALWAYS**, because the legacy policy
  makes no certified promise, a damaged episode whose ego never reaches the threshold is an
  ordinary recorded observation there, and an approved measurement contains exactly such an
  episode ([measurement history](../history/measurements.md#2-measurement-records): the Phase-A
  rerun's seed 424). It is PURE and MUTATES NOTHING — it never
  applies a late event to satisfy itself; `scenario` and `ticks` are DIAGNOSTIC ONLY.
  **CALLED ONCE, AT THE SINGLE `graph_tick_loop.run_episode` EPISODE-EXIT SEAM**, which is
  the one path every scientific consumer goes through, so the predicate is not duplicated
  across the trainer and the diagnostic rollout — **and BEFORE the recording export**,
  because `graph_train` synchronizes a completed run's playback into its manifest only after
  `run_episode` returns, so exporting first and raising second would leave a real recording
  no manifest lists (the exact defect the roster-integrity correction closed). An episode
  that raises exports nothing.

**THE LIVE CERTIFICATE CHECK BINDS THE EGO'S PHYSICAL STATE, AND ONLY THAT
(`FuelDamageController._require_certificate_holds`, the certified-FD physical-state
integrity repair `d36e133`, integrated `edf9e84`, PR #55).**

**SETUP-TIME CERTIFICATION REMAINS TICK-AWARE AND IS BYTE-UNCHANGED.** `event_tick`,
`movement_count`, `bracket_ticks`, `CERTIFICATE_TICK_TOLERANCE == 1` and the fuel and
position tolerance derivations built from that quantum are exactly what they were, and the
certificate is still validated across the whole bracket rather than at the nominal tick
alone. **World acceptance, the bounded eligibility walk, `certify_fd_candidate`, the
pre-event stability tests and certificate CONSTRUCTION are all untouched.**

**WHAT A LIVE CONTRADICTION IS JUDGED ON — EXACTLY TWO QUANTITIES, EACH AGAINST ITS OWN
EXISTING TOLERANCE:**

1. **POSITION** — the live aircraft's great-circle offset from the certificate's
   `event_location`, against the certificate's own **`position_tolerance_km`**;
2. **PRE-DAMAGE FUEL** — `|fuel_before − cert.fuel_before|`, against the certificate's own
   **`fuel_tolerance`**.

**THE ABSOLUTE OUTER TICK IS DIAGNOSTIC ONLY AND CANNOT, BY ITSELF, ABORT A RUN.** It was
binding once, on the premise that an airborne ego receives exactly one engine update per
outer tick — and **that premise, not the certifier, is what was wrong**: frozen BLADE can
skip an airborne ego's entire update when a preceding aircraft leaves
`scenario.aircraft` mid-pass
([`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion)), so an ego whose peers land is physically EARLIER than
the tick count implies while its own state still matches the certificate exactly. **A TICK
MISMATCH ALONE IS THEREFORE NOT A CERTIFICATE CONTRADICTION AND MUST NEVER BE DESCRIBED AS
ONE.** `tick_delta` is still computed on every path — it is precisely what a skipped engine
update looks like from outside — and it is reported, never enforced.

**WHAT WAS DELIBERATELY NOT DONE, AND EACH ABSENCE IS PART OF THE CONTRACT.** **NEITHER
PHYSICAL TOLERANCE WAS WIDENED**, none was scaled, and **no dynamic or state-dependent
tolerance was introduced** — both are still exactly the certificate's own engine quanta
plus their documented float epsilon, so a state one quantum beyond either still aborts at
any tick. **No engine-update counter, no skip counter and no reconstructed tick coordinate
was added**, in this layer or anywhere else: the repair removes a false invariant rather
than modelling around it. **BLADE IS UNCHANGED** — this is not a physics fix and must never
be read as one.

**EVERY DELTA IS COMPUTED BEFORE ANY VERDICT IS TAKEN**, and a failure reports position,
fuel AND tick together, each with its own pass/fail verdict, so one contradicted quantity
can never hide the state of the other and a preserved message is diagnosable without a
replay. **A GENUINE PHYSICAL CONTRADICTION STILL RAISES `FuelDamageIntegrityError`
BEFORE THE FUEL MUTATION**, so a contradicted certificate still leaves the engine
untouched, and it is still an INSTRUMENT ABORT rather than accounted attrition.
**EVERYTHING ELSE IN THE TWO FAILURE ROUTINGS IS UNCHANGED:** the terminal
`require_certified_event_realized` still treats a certified damaged episode whose event
never fired as an integrity ABORT, and ordinary setup ineligibility
(`NO_FD_ELIGIBLE_EGO`) is still ORDINARY ACCOUNTED ATTRITION inside `skip_and_account_v1`.

**THE DIAGNOSED SIGNATURE THIS REPAIR ACCEPTS, recorded because it is the reason the
repair exists:** certified event tick 914 (movement count 913, bracket 913..915), two
preceding peers landed, the selected ego lost two engine visits, the threshold crossing
observed at outer tick 916 — **and at that state its position matched the certified event
position to ~7e-11 km and its pre-damage fuel to ~6e-9 lbs.** The certificate's PHYSICAL
promise held exactly; only the bookkeeping coordinate differed, and the retired
tick-binding check would have destroyed a scientifically sound episode over a quantity the
engine never promised. It is pinned by
`tests/test_graph_fuel_damage.py::test_g1_11b_the_absolute_outer_tick_is_diagnostic_never_binding`
(which first ASSERTS the observed tick really is outside `tick_tolerance` and outside
`bracket_ticks`, so the regression is falsifiable), beside
`test_g1_11c_a_physical_contradiction_aborts_and_reports_all_three_deltas` and
`test_g1_11d_the_certificate_construction_and_world_acceptance_are_unchanged`.

**POST-FD COMPLETION-BOUNDARY ADAPTATION (`completion_boundary_v1`).** The IMMEDIATE
`FUEL_DAMAGE` wake is UNCHANGED. Additionally, **and ONLY for the ego that REALLY LOST
FUEL**, the tick loop's `_post_fd_boundary` runs at the TOP of a later tick — after the
event call, before Phase 1 — and does four things in order: (1) RETIRES the state when the
ego can no longer reach a boundary, recording WHICH via `deactivate_adaptation`
(`POST_FD_DEACTIVATED_RTB = "rtb_committed"` or `POST_FD_DEACTIVATED_DEAD = "ego_dead"`),
because "no further boundary happened" and "the ego went home" are different facts;
(2) RECONCILES that ego's own confirmed completions through the executor's single
confirmation site `reconcile_confirmed_for_ego`; (3) CLEANS the confirmed assignments out of
THAT EGO'S PRIVATE belief solution (`_drop_confirmed_assignments`, copy-on-write, shaped
exactly like `graph_effect`'s plan edits — only `solution[ego_id]` differs, peer slices are
copied through byte-for-byte, no key is invented or removed, and `tasks` is NOT touched
because indices are positional and APPEND-ONLY) and resyncs ONLY its own executor slice;
(4) reports whether a decision is worth asking for.

- **`FuelDamageController.post_fd_ego` IS THE SINGLE SITE THAT ENFORCES "ONLY THE ACTUALLY
  DAMAGED EGO".** It is `None` until the REAL mutation has happened — the arming assignment
  sits AFTER `aircraft.current_fuel = target` in `maybe_apply` — `None` forever on a clean
  episode (a counterfactual certificate arms nothing), and it can never name a peer.
- **A TERMINAL COMPLETION PRODUCES NO WAKE.** `has_open_assignments` decides it: with no
  remaining work there is nothing to decide, and the executor's PRE-EXISTING empty-plan path
  issues the single latched RTB exactly as before. "Boundary without a wake" and "boundary
  that failed to wake" are therefore counted apart, never inferred from a wake count.
- **PEERS AND COMMAND TIMING ARE UNTOUCHED.** No peer is reconciled and no peer belief or
  slice is read or written. Nothing changes for the damaged ego's commands either: `done` is
  monotone, so Phase 2's own reconciliation on the SAME observation finds nothing further and
  emits precisely the command it would have emitted anyway, on the same tick.
- **CTDE ALIGNMENT IS PRESERVED, AND THE ORDER MATTERS.** The reconciliation and resync
  happen BEFORE Phase 1, hence before `central.capture` (which sits inside the `if wake`
  branch immediately before `_wake_decision`), so a boundary sample describes the RECONCILED
  execution state rather than a plan the world already moved past — and the samples stay
  exactly 1:1 with actor transitions, which `CTDEEpisodeRecord` still validates.
- **THE ACTION SET IS UNCHANGED. NO new `MetaAction` exists** and none was added.
- **THE IMMEDIATE-WAKE AND BOUNDARY DIAGNOSTICS ARE SEPARATE, DELIBERATELY.**
  `FuelDamageOutcome.wake_occurred` / `wake_meta_action` keep meaning the IMMEDIATE
  fuel-damage wake — an approved measurement is reported over them — so boundary decisions
  go to `note_boundary` / `note_boundary_wake` and surface on the SEPARATE
  `FuelDamageController.post_fd_outcome` (`PostFdAdaptationOutcome`: `policy`, `ego_id`,
  `armed`, `active`, `deactivation_reason`, and per-`PostFdBoundary` counts
  `boundaries_confirmed`, `boundaries_with_remaining_mission`, `boundaries_terminal`,
  `boundary_wakes`, `boundary_ticks`, `boundary_meta_actions`). `note_boundary_wake` matches
  a decision to its boundary BY TICK and never overwrites a boundary that already carries
  one — one boundary is one decision. `note_boundary` ignores any ego other than the damaged
  one, so a mis-wired caller cannot attribute a peer's completion to this event.

**PURITY IS UNCHANGED**: `graph_fuel_damage` still imports no BLADE, gymnasium, torch or
solver, does no file I/O and holds no module-global randomness (its one new import,
`scenario_factory.iter_enemy_targets`, itself imports only `...models`), and it must never
import `graph_episode_setup`. **NOTHING GENERALIZED REACHES THE ACTING PATH**: no certificate
field, no policy id, no candidate ordinal, no rejection reason and no boundary count enters
`GraphObservation`; peer graph rows stay featureless; the world-truth inputs the certifier
reads are read ONCE, at setup, to decide whether the world is ACCEPTED.

**BOTH HARNESSES NOW SELECT THESE POLICIES — THROUGH THE BUNDLE, NEVER INDIVIDUALLY
(GENERALIZED-V1 Task 4, `db79013`).** `TrainConfig.fuel_damage_parameters()` and
`RolloutConfig.fuel_damage_parameters()` set `eligibility_policy` and `post_fd_wake_policy`
from `self.design`, the ONE resolution of `episode_design`. Under `fixed_cell_v1` the
resolved ids ARE the historical defaults, so the constructed `FuelDamageParameters` is
identical to the pre-Task-4 one; under `generalized_v1` both certified policies are in
force together. **Neither config carries a standalone `eligibility_policy` or
`post_fd_wake_policy` field**, so neither can be enabled apart from the bundle.
`FdEligibilityAudit`, `FdEventCertificate` and `PostFdAdaptationOutcome` are now PERSISTED
per episode and AGGREGATED per run — see
[artifacts and metrics §4](artifacts_metrics.md#4-generalized-persistence-and-aggregates).
 Measurements taken on these designs are listed in the
[handoff](../../graph_rl_project_handoff.md).

**WHAT IS EXPLICITLY NOT IN THIS DESIGN.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate
future Grade-A research task**. The event point stays at the fixed 30 % first-leg location.
No continuation reference, no `U_prefix`, no reward-formula change, no generalized training
sampler, no evaluation manifest, no new metric or plot, no new meta-action, no trim-tail
action, no peer behaviour change and no communication channel of any kind. BLADE, the
solver, `graph_reward`, the encoder, the action space, `DETECTION_KM`, B2 geometry and the
seed schedules are all unchanged. *(That list is a statement about TASK 2's scope and stays
accurate as one. The continuation reference and `U_prefix` it excludes were implemented
AFTERWARDS, as the SEPARATE Task-3 seam contracted in
[reward and solvers §2](reward_solvers.md#2-event-conditioned-continuation-reference) — `graph_reward`'s
static formula is still unchanged there too.)*

## 5. Code routing

| Task | Files and symbols | Contract |
|---|---|---|
| choose or change the hidden-cardinality policy | `rl/training/graph_hidden_placement.py`: `HIDDEN_CARDINALITY_POLICIES`, `place_hidden_targets_bounded`, `_select_leg`, `_ordinal_permutation`, `_candidate_substream_seeds`, `BoundedBackoffAudit`, `BACKOFF_REJECTION_REASONS`; `rl/training/graph_episode_setup.py`: `setup_episode(hidden_policy=...)`, `_require_generalized_cardinality`, `ConstructionAudit` | §1; selected only through `episode_design` ([training and benchmarks §5](training_benchmarks.md#5-episode-designs-generalized-v1-sampler-and-18-stratum-benchmark)) |
| place hidden targets along a predicted route (pure geometry) | `graph_hidden_placement.py`: `PlacementParameters`, `HiddenPlacement`, `predict_route`, `place_hidden_targets`, `validate_placement`, `geometric_fingerprint` | §1 |
| change the shared nearest-neighbour ordering (route prediction and execution at once) | `utils/scheduling_utils.py`: `nearest_neighbor_order`; `tests/test_graph_executor_nn_ordering.py` | §1; [runtime §3](runtime.md#3-execution-stage-1) |
| change the legacy FD-BASELINE-v1 mechanism | `rl/training/graph_fuel_damage.py`: `FuelDamageMode`, `FuelDamageParameters`, `FuelDamagePlan`, `FuelDamageOutcome`, `FuelDamageController`, `measure_window`, `plan_fuel_damage`, `build_fuel_damage_controller`, `derive_fuel_damage_seed`, `fuel_for_distance_km`, `rtb_command_for` | §2 |
| change the FD-VARIABLE-SEVERITY-v1 mechanism | `graph_fuel_damage.py`: `FUEL_DAMAGE_SEVERITY_RNG_DOMAIN`, `derive_fuel_damage_severity_seed`, `resolve_severity`, `severity_band`, `_require_valid_band`, `FuelDamageController._live_variable_target` | §3 |
| choose or change certified FD eligibility or the live certificate check | `graph_fuel_damage.py`: `FD_ELIGIBILITY_POLICIES`, `derive_fuel_damage_eligibility_seed`, `certify_fd_candidate`, `_certified_eligibility_walk`, `FdEventCertificate`, `FdEligibilityAudit`, `NO_FD_ELIGIBLE_EGO`, `CERTIFICATE_TICK_TOLERANCE`, `FuelDamageController._require_certificate_holds` / `require_certified_event_realized`, `FuelDamageIntegrityError`; the episode-exit call in `graph_tick_loop.run_episode` | §4 |
| choose or change post-FD completion-boundary wakes | `graph_fuel_damage.py`: `POST_FD_WAKE_POLICIES`, `FuelDamageController.post_fd_ego` / `note_boundary`, `PostFdAdaptationOutcome`; `rl/training/graph_tick_loop.py`: `_post_fd_boundary`, `_drop_confirmed_assignments`; `rl/action/graph_trigger.py`: `TriggerKind.POST_FD_COMPLETION`; `utils/blade_utils/blade_graph_executor.py`: `reconcile_confirmed_for_ego`, `has_open_assignments` | §4 |

`graph_fuel_damage` and `graph_hidden_placement` must stay pure (no BLADE, gymnasium, torch or
solver import) and must never import `graph_episode_setup`. Every row above is a
research-validity change ([`cc_review.md` §4](../workflows/cc_review.md#4-risk-and-verification)).

## 6. Known limitations and open items

- **Exact-cardinality construction failures — RESOLVED as `skip_and_account_v1` by B4
  (`1b48145`).** B2's locked contract is ONE placement per non-empty ego route, and B3
  requires `len(placements) == n_hidden` exactly, so when bonmin leaves an ego unassigned
  there are fewer routes than `n_hidden` and `setup_episode` raises. Measured on the default
  cell over seeds 0–11: **10/12 gave 3 usable ego routes; seeds 2 and 8 gave only 2**; seed
  0 (the reference) is clean. **The decision is to ACCEPT the loss and account for it**: the
  seed is attempted once, its failure is recorded once in `episode_failures.jsonl` with the
  pipeline stage, and the batch simply carries a smaller successful population that every
  statistic reports next to its denominator
  ([training and benchmarks §1](training_benchmarks.md#1-trainer-and-run-auditability)). The rejected alternatives stay rejected —
  no reseeding past a failure, no retry, no band shift, and above all no weakening of the
  cardinality check, the B2 geometry, or the loud failure. **PARTIALLY ADDRESSED, AND ONLY AS AN OPT-IN.** The general
  `n_hidden != usable ego routes` distribution policy B2 named now has ONE concrete answer:
  the GENERALIZED-V1 `bounded_backoff_v1` cardinality policy (§1 above), which accepts
  any `H_realized >= 1` and records requested-vs-realized instead of refusing the world.
  It is **NOT the default, is reachable only through `episode_design`, and changes NOTHING
  about the default behaviour above**: `exact_v1` still refuses, `skip_and_account_v1` still accounts, and the
  measured seed-2 / seed-8 outcomes are unchanged for every run that exists. Distributing
  SEVERAL hidden targets across ONE ego route remains out of scope and unimplemented. The
  first real probe measured the actual scheduled yield as **7/8** train attempts:
  seed 2 failed once at `setup` because only two non-empty ego routes existed for three
  requested hidden targets; it was recorded once and never retried or replaced. The
  earlier 10/12 construction sample remains context, not the run-time rate of a future
  baseline configuration.
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
