"""graph_fuel_damage.py -- FD-BASELINE-v1: the deterministic, ego-local fuel-damage event.

THE ONE DIFFICULTY FACTOR OF THE FINAL PHASE-A BASELINE CELL. The previous cell
(3 agents, 3 known + 3 route-relative hidden airbase targets) was learned in a two-update
probe: the static plan was already close to the oracle and the only adaptation on offer
was "engage the pop-up you happen to fly past". This module adds exactly ONE attributable
source of adaptation difficulty -- a seeded, ego-local, one-shot loss of fuel -- and
nothing else. No probabilistic attacks, no hostile fire, no SAMs, no new cardinalities, no
dense reward, no PPO change.

WHAT IT CREATES: A REAL DECISION
--------------------------------
At roughly 30% of the way along its first planned leg, ONE ego in a damaged episode loses
fuel. The post-damage quantity is chosen inside a STRICT window that is validated before
the episode runs:

    1.10 * fuel(event point -> home base)  <=  post_damage_fuel
                                           <   1.10 * fuel(rest of the route -> home base)

so flying home is feasible with the engine's own 1.10 reserve while completing the plan
and then returning is not. The ego must therefore choose between PLAN_COMPLIANCE (finish
the route, and on this cell run the tank dry -- ``Game.update_all_aircraft_position``
removes an aircraft at ``current_fuel <= 0``, which the reward charges at
``aircraft_penalty_coeff``) and SELF_PRESERVATION_ABORT (abort the ego's REMAINING
MISSION -- the effect layer clears ALL of that ego's remaining assignments, not only the
selected node's -- after which the executor's existing empty-plan path issues its single
latched RTB, keeping the airframe at the cost of the targets it gives up). Neither branch
is forced anywhere: the policy still decides.

THE WINDOW IS VALIDATED TWICE: PLANNED, THEN LIVE
-------------------------------------------------
The window is built BEFORE the episode from a projection -- where the ego is expected to
be at 30% of leg 1, and how much fuel it is expected to hold there. That projection is
knowingly optimistic: it charges fuel for distance FLOWN, while
``Game.update_all_aircraft_position`` decrements ``fuel_rate / 3600`` on EVERY tick
including ones where the aircraft has no route yet (the launch tick is exactly that). So
live fuel at the event is always somewhat below ``projected_fuel_at_event``.

Rather than trust the projection, :meth:`FuelDamageController.maybe_apply` RE-MEASURES the
window from the aircraft's actual position and validates the mutation against the ego's
actual fuel before touching anything. If the live window does not hold -- the ego is
already below what continuing would cost, or the target would no longer cover the RTB
leg, or it would no longer make continuing infeasible -- it raises BEFORE mutating, and
the attempt is accounted as a ``run``-stage failure. Planned and live bounds are recorded
under separate names (``FuelDamagePlan.rtb_fuel_floor`` vs
``FuelDamageOutcome.live_rtb_fuel_floor``) so a reader always knows which one a number is.

"DID IT RETURN TO BASE" IS COMMAND HISTORY, NOT EXECUTOR STATE
--------------------------------------------------------------
``FuelDamageOutcome.rtb_command_issued`` is True only if ``run_episode`` really emitted
``aircraft_return_to_base('<ego>')`` in a Phase-2 command list, observed through
:meth:`FuelDamageController.note_commands`. It deliberately does NOT read
``GraphPlanExecutor.rtb_issued``: that is a lifecycle LATCH which ``_command_for_ego``
also sets True for a DEAD ego -- precisely because no command was, or could be, emitted --
so reading it would report an ego that flew its plan into the ground as having returned to
base, and count one episode as both an RTB and a death in the aggregate that exists to
show whether the event produced an abort.

THE FUEL MODEL IS BLADE'S, NOT OURS
-----------------------------------
Every quantity here is computed with the engine's own arithmetic, transcribed from
``Game.get_fuel_needed_to_return_to_base`` (see :func:`fuel_for_distance_km`):

    nm    = km * 1000 / NAUTICAL_MILES_TO_METERS      (1852)
    hours = nm / aircraft.speed                       (speed is in KNOTS)
    fuel  = hours * aircraft.fuel_rate                (lbs/hr)

and the reserve multiplier is the engine's own ``* 1.1`` from the
``AIRCRAFT_RTB_WHEN_OUT_OF_RANGE`` doctrine check. There is deliberately NO second,
competing fuel model: a window computed against arithmetic the engine does not use would
be a window the episode does not actually have. ``speed`` and ``fuel_rate`` are read off
the LIVE BLADE Aircraft, never off ``Agent`` -- ``scenario_factory`` substitutes a
250 kt planning speed for a grounded unit, which is right for the solver and wrong here.

The engine burns ``fuel_rate / 3600`` per tick unconditionally (one tick == one second),
so fuel consumed over a flown distance is exactly ``fuel_for_distance_km`` of it. That
identity is what lets a pre-run projection describe a mid-run state.

NO-COMMUNICATION (the research red line this module must not cross)
-------------------------------------------------------------------
The event is EGO-LOCAL and ONE-SHOT, and three properties keep it that way:

  1. IT IS APPLIED BEFORE THE PER-EGO PHASE-1 LOOP, never inside it. The tick loop calls
     :meth:`FuelDamageController.maybe_apply` once at the top of a tick; every ego then
     senses and decides against the SAME post-event snapshot. Mutating BLADE midway
     through the ego loop would make the outcome depend on ego iteration order -- an
     implicit communication channel, and the exact thing the two-phase backbone exists
     to prevent.
  2. ONLY THE SELECTED EGO WAKES. ``maybe_apply`` returns the damaged ego's id, and the
     tick loop passes ``fuel_damage=True`` to ``decide_triggers`` for that ego ALONE.
     Peers' triggers are untouched.
  3. NO PEER CAN OBSERVE IT. ``graph_builder`` gives the ego row a real ``fuel_norm`` and
     every peer row ``0.0`` -- peers are featureless by construction -- so the damaged
     fuel value is unreachable from any peer's graph. The damaged ego's OWN graph, built
     at that same wake from the same mutated aircraft, necessarily carries the
     POST-damage ``fuel_norm``: ``_compute_fuel_norm`` reads the live object this module
     already mutated.

FD-VARIABLE-SEVERITY-v1: THE SAME EVENT, WITH TWO PHYSICAL SEVERITIES
---------------------------------------------------------------------
The design above makes EVERY damaged episode structurally severe, so "damaged" and
"continuing is infeasible" are the same fact and an actor can learn the shortcut
``fuel damage => abort`` without ever reading its fuel gauge. The variable-severity modes
(:attr:`FuelDamageMode.VARIABLE`) split the damaged half into two physically different
cases, measured at the LIVE event state against the same BLADE arithmetic:

    MILD    continue_requirement < post_damage_fuel < fuel_before
            -- a real loss, but completing the route and returning is STILL feasible;
    SEVERE  rtb_floor <= post_damage_fuel < continue_requirement
            -- flying home stays feasible, continuing does not (the legacy interval).

The post-damage value is the MIDPOINT of whichever band applies, derived at the event
tick from the live window and the live fuel rather than fixed before the run: mild and
severe are statements about the fuel the ego really holds where it really is, and a value
chosen from a projection could only be CHECKED against that, never guaranteed to land in
the right band of it. The LEGACY modes are untouched -- same seeds, same conditions, same
selected egos, same planned-midpoint target, same four checks -- because an approved
measurement exists on them.

The policy is never told which case it is in. No severity feature reaches
``GraphObservation``; the only thing that changes in the ego's input is its own real
``fuel_norm``, which is exactly what the decision has to be read off.

DETERMINISM AND THE RNG DOMAINS
-------------------------------
The clean/damaged draw and the ego selection come from a PRIVATE rng domain derived from
the episode seed alone (:func:`derive_fuel_damage_seed`): a SHA-256 of
``"fuel_damage_v1:<seed>"``. Not ``hash()`` (salted per process), not global ``random``
(shared with whatever else the episode consumed), and not the placement rng (whose stream
position depends on how many placements were attempted). Consequently no amount of
earlier RNG consumption -- by the generator, by the solver, by hidden placement, by
torch's action sampling -- can change which episode is damaged or which ego is hit.

The two draws are taken in a FIXED ORDER from one rng: the mixture bit first (drawn even
in forced modes, so the stream position is identical), then the ego. A forced-damaged
episode therefore selects the same ego a seeded-mixture episode of the same seed would --
which is what makes the matched clean/damaged evaluation pair comparable.

SEVERITY HAS ITS OWN DOMAIN, ``"fuel_damage_severity_v1"``
(:func:`derive_fuel_damage_severity_seed`), and that separation is load-bearing rather
than tidy. Taking the mild/severe bit from the v1 stream would insert a draw between the
mixture bit and the ego selection, changing WHICH EGO every damaged episode picks -- which
would silently invalidate the approved FD-BASELINE-v1 measurement instead of extending it.
With two domains the decisions are orthogonal: severity cannot move the ego, the ego
cannot move severity, and the three members of a matched clean/mild/severe TRIAD run the
same world, the same ``A_init``, the same hidden geometry and the same selected ego.

PURITY
------
No BLADE import, no gymnasium, no torch, no solver, no file I/O, no module-global
randomness. Live engine objects are touched only through duck-typed attribute access
(``id`` / ``latitude`` / ``longitude`` / ``speed`` / ``fuel_rate`` / ``current_fuel`` /
``max_fuel``), so this module is hand-testable with plain stubs and is safe in the
import-purity closure of ``graph_tick_loop``. Route prediction REUSES the frozen
``graph_hidden_placement.predict_route`` rather than reimplementing it, exactly as B2
reuses ``nearest_neighbor_order``: a second copy of the route arithmetic would let the
window be computed against a route the executor does not fly.

FAILURE POLICY
--------------
Everything fails LOUDLY. WHICH exception it fails with is the whole question, because the
two classes are routed in opposite directions, and GENERALIZED-V1 added the second one.

UNDER THE LEGACY ELIGIBILITY POLICY (:data:`FD_ELIGIBILITY_LEGACY_V1`, the DEFAULT) the
description below is complete and UNCHANGED -- it is the behaviour the approved
FD-BASELINE-v1 and FD-VARIABLE-SEVERITY-v1 measurements were taken on:

    Everything fails as :class:`FuelDamageError`. In particular, a scheduled DAMAGED
    episode with no valid strict window is NOT quietly downgraded to a clean episode --
    that would silently change the population every measurement is reported over. It
    raises, the trainer wraps it as an ``EpisodeAttemptError("setup", ...)``, and
    ``skip_and_account_v1`` records it exactly once with no retry, no substitution and no
    band shift. A FORCED-CLEAN episode never computes a window at all, so it can never
    fail for this reason. A damaged episode whose selected ego simply never reaches the
    threshold is NOT a failure either: it finishes with ``fired == False``, which is a
    recorded observation about that episode and nothing more.

UNDER :data:`FD_ELIGIBILITY_CERTIFIED_V1` two of those sentences stop being true, and
both changes are deliberate (handoff 3l.3):

  * A FORCED-CLEAN EPISODE DOES DO WORK AT SETUP. Eligibility is a property of WORLD
    ACCEPTANCE, so the clean member runs the SAME complete certification walk as its
    matched mild and severe siblings -- that is what gives the three one accepted-world
    support. It computes no window it will apply and mutates nothing, but it CAN be
    rejected, with the stable reason :data:`NO_FD_ELIGIBLE_EGO`. That rejection is still
    a :class:`FuelDamageError` and still ordinary accounted attrition: nothing had been
    certified, so nothing was contradicted.
  * A CERTIFIED WORLD THAT DOES NOT DELIVER ITS CERTIFIED EVENT IS AN INSTRUMENT FAULT,
    not an episode outcome, and raises :class:`FuelDamageIntegrityError` -- which ABORTS
    the run and is never wrapped, ledgered, tallied or skipped. There are exactly two
    ways to reach it, and they are the two halves of one promise:
      - the LIVE event state contradicts the certificate, or the live physics refuses the
        event the certificate said was constructible
        (:meth:`FuelDamageController.maybe_apply`);
      - the episode ENDS with the certified damaged event never having fired at all
        (:meth:`FuelDamageController.require_certified_event_realized`). Under the legacy
        policy that is an ordinary observation; under the certified one it means a world
        proven capable before a tick was paid for failed to materialize its own
        certificate, so admitting it as a successful damaged episode would put a world
        whose certificate did not hold into a scientific population.
    A CERTIFIED CLEAN episode is of course allowed to finish with ``fired == False``:
    nothing was scheduled to fire, and its certificate is a counterfactual.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...models import Location, StepKind, Task
# The SAME enemy-enumeration `GraphPlanExecutor.sensed_target_ids` scans, imported
# rather than reimplemented so the certified policy's pre-event pop-up test asks
# exactly the question the runtime sensor asks. `scenario_factory` imports only
# `...models`, so this costs the layer none of its BLADE/gym/torch-free purity.
from ...utils.blade_utils.scenario_factory import iter_enemy_targets
from .graph_hidden_placement import predict_route

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level_order)

__all__ = [
    "CONDITION_CLEAN",
    "CONDITION_DAMAGED",
    "CONDITIONS",
    "FUEL_DAMAGE_RNG_DOMAIN",
    "FUEL_DAMAGE_SEVERITY_RNG_DOMAIN",
    "SEVERITIES",
    "SEVERITY_MILD",
    "SEVERITY_SEVERE",
    "TARGET_POLICY_LIVE_SEVERITY_MIDPOINT",
    "TARGET_POLICY_PLANNED_MIDPOINT",
    "FuelDamageController",
    "FuelDamageError",
    "FuelDamageMode",
    "FuelDamageOutcome",
    "FuelDamageParameters",
    "FuelDamagePlan",
    "build_fuel_damage_controller",
    "build_fuel_damage_plan",
    "derive_fuel_damage_seed",
    "derive_fuel_damage_severity_seed",
    "fuel_for_distance_km",
    "measure_window",
    "plan_fuel_damage",
    "resolve_condition",
    "resolve_severity",
    "rtb_command_for",
    "severity_band",
    # --- GENERALIZED-V1 step 2: certified eligibility -------------------------
    "CERTIFICATE_TICK_TOLERANCE",
    "FD_ELIGIBILITY_CERTIFIED_V1",
    "FD_ELIGIBILITY_LEGACY_V1",
    "FD_ELIGIBILITY_POLICIES",
    "FD_ELIGIBILITY_REJECTION_REASONS",
    "FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN",
    "KILOMETERS_TO_NAUTICAL_MILES",
    "NO_FD_ELIGIBLE_EGO",
    "REASON_DEGENERATE_LEG",
    "REASON_EVENT_UNREACHABLE",
    "REASON_INVALID_BAND",
    "REASON_NO_AIRCRAFT",
    "REASON_NO_ROUTE",
    "REASON_PRE_EVENT_ASSIGNMENT_BOUNDARY",
    "REASON_PRE_EVENT_POPUP_RISK",
    "REASON_ROUTE_UNRESOLVABLE",
    "WAYPOINT_SNAP_KM",
    "FdEligibilityAudit",
    "FdEligibilityCandidate",
    "FdEventCertificate",
    "FuelDamageIntegrityError",
    "certify_fd_candidate",
    "derive_fuel_damage_eligibility_seed",
    "eligibility_ordinal_permutation",
    "engine_leg_distance_km",
    "predict_leg_states",
    # --- GENERALIZED-V1 step 2: post-FD completion-boundary adaptation --------
    "POST_FD_DEACTIVATED_DEAD",
    "POST_FD_DEACTIVATED_RTB",
    "POST_FD_WAKE_COMPLETION_BOUNDARY_V1",
    "POST_FD_WAKE_POLICIES",
    "POST_FD_WAKE_SINGLE_V1",
    "PostFdAdaptationOutcome",
    "PostFdBoundary",
]


# =============================================================================
# 0. Constants transcribed from the FROZEN engine (never re-derived)
# =============================================================================

# blade.utils.constants.NAUTICAL_MILES_TO_METERS. Transcribed rather than imported
# because importing it would drag the engine into this module's closure and break the
# purity that makes the layer hand-testable; the value is a physical constant and the
# transcription is asserted against the engine in the setup-seam tier.
NAUTICAL_MILES_TO_METERS = 1852.0

# The engine's own reserve multiplier, from the AIRCRAFT_RTB_WHEN_OUT_OF_RANGE branch of
# `Game.update_all_aircraft_position` (`current_fuel < fuel_needed * 1.1`). The doctrine
# itself is OFF for BLUE in `strike_training_4v5.json` -- so the engine never auto-RTBs
# behind the executor's single-issue latch -- but the MARGIN is the engine's definition
# of "can still get home", and the approved window uses exactly it.
DEFAULT_RTB_SAFETY_MARGIN = 1.10

# The approved event trigger: the first tick at which the selected ego has covered this
# fraction of its FIRST planned leg.
DEFAULT_LEG_PROGRESS_THRESHOLD = 0.30

# The approved training mixture: half the scheduled training episodes are damaged.
DEFAULT_DAMAGE_PROBABILITY = 0.5

# VARIABLE SEVERITY (FD-VARIABLE-SEVERITY-v1): P(mild | damaged). The approved
# distribution is P(clean) = 0.50, P(mild) = 0.25, P(severe) = 0.25, which is exactly
# `DEFAULT_DAMAGE_PROBABILITY = 0.5` for "is it damaged at all" and this value for "and
# if so, how badly". The conditional is a SEPARATE, explicitly recorded knob rather than
# a hard-coded branch, because it is the parameter the whole experiment turns on.
DEFAULT_MILD_PROBABILITY = 0.5

# The private rng domain string (see the module docstring). Bump the suffix if the draw
# ORDER or the derivation ever changes -- the point of a versioned domain is that a run's
# clean/damaged assignment can be reproduced from its seed alone, forever.
FUEL_DAMAGE_RNG_DOMAIN = "fuel_damage_v1"

# A SECOND, INDEPENDENT domain for the mild/severe draw. It is deliberately not another
# value taken from the v1 stream: drawing severity there would shift the position every
# later v1 draw reads from, which would change WHICH EGO a damaged episode selects and so
# make every legacy FD-v1 seed irreproducible. Two domains keep the two decisions
# orthogonal -- a run's clean/damaged assignment and its selected ego are byte-identical
# whether or not severity exists, and the severity of a seed is reproducible on its own.
FUEL_DAMAGE_SEVERITY_RNG_DOMAIN = "fuel_damage_severity_v1"

# The leg the event is scheduled on. 1-based, matching `graph_hidden_placement._Leg.index`.
EVENT_LEG_INDEX = 1


# =============================================================================
# 0b. GENERALIZED-V1 step 2: the two OPT-IN policy seams
# =============================================================================
# Both are EXPLICIT, VERSIONED strings, both default to the historical behaviour, and
# both are carried on `FuelDamageParameters` so an existing construction site obtains
# the legacy default automatically. The approved FD-BASELINE-v1 and
# FD-VARIABLE-SEVERITY-v1 measurements were taken on the legacy defaults; the
# generalized policies are ADDED BESIDE them and never replace them.

# WHICH EGO IS DAMAGED, AND WHAT THAT REQUIRES OF THE WORLD.
#
#   legacy_selected_ego_v1        DEFAULT, and byte-for-byte the merged behaviour: the
#                                 `fuel_damage_v1` stream's second draw picks uniformly
#                                 among the SORTED ids of egos with a non-empty initial
#                                 route, and the window is a distance PROJECTION that
#                                 the controller re-validates live.
#   certified_both_severities_v1  GENERALIZED-V1 (handoff 3l.3): FD capability becomes a
#                                 property of WORLD ACCEPTANCE. A deterministic bounded
#                                 walk over STABLE SCHEDULED ORDINALS certifies ONE ego
#                                 that supports BOTH the mild and the severe band at a
#                                 TICK-AWARE prediction of the event state, and whose
#                                 pre-event route prefix no legal trigger can disturb.
#                                 The walk runs for EVERY condition -- CLEAN included --
#                                 so the three members of a matched group share one
#                                 accepted-world support.
FD_ELIGIBILITY_LEGACY_V1 = "legacy_selected_ego_v1"
FD_ELIGIBILITY_CERTIFIED_V1 = "certified_both_severities_v1"
FD_ELIGIBILITY_POLICIES: Tuple[str, ...] = (
    FD_ELIGIBILITY_LEGACY_V1,
    FD_ELIGIBILITY_CERTIFIED_V1,
)

# HOW MANY TIMES THE DAMAGED EGO DECIDES AFTER THE EVENT.
#
#   single_wake_v1            DEFAULT, and byte-for-byte the merged behaviour: the
#                             immediate FUEL_DAMAGE wake and nothing else.
#   completion_boundary_v1    GENERALIZED-V1 (handoff 3l.4): the immediate wake is
#                             UNCHANGED, and the ACTUALLY DAMAGED ego additionally wakes
#                             at each of its own locally CONFIRMED assignment
#                             completions, BEFORE it commits movement to the next
#                             remaining assignment. No peer is affected and no ego that
#                             was not really damaged ever enters the state.
POST_FD_WAKE_SINGLE_V1 = "single_wake_v1"
POST_FD_WAKE_COMPLETION_BOUNDARY_V1 = "completion_boundary_v1"
POST_FD_WAKE_POLICIES: Tuple[str, ...] = (
    POST_FD_WAKE_SINGLE_V1,
    POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
)

# A THIRD private rng domain, for the certified policy's candidate-ordinal permutation
# ONLY. It is a separate domain for exactly the reason "fuel_damage_severity_v1" is:
# taking the permutation from the "fuel_damage_v1" stream would insert draws between
# that stream's mixture bit and its ego selection, changing WHICH EGO every legacy
# damaged episode picks and so invalidating the approved measurements instead of
# extending them. With three domains the decisions are orthogonal -- eligibility cannot
# move the condition or the legacy ego, and neither can move eligibility.
FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN = "fuel_damage_eligibility_v1"

# blade.utils.constants.KILOMETERS_TO_NAUTICAL_MILES -- the km -> nm unit constant the
# FROZEN engine divides a platform's KNOTS speed into inside
# blade.utils.utils.get_next_coordinates, which is the ONE site that moves an aircraft.
# Transcribed rather than imported, exactly as NAUTICAL_MILES_TO_METERS above and exactly
# as blade_graph_executor transcribes this same constant: importing it would drag the
# engine into this module's closure. The transcription is compared against the ENGINE'S
# OWN value in the BLADE test tier.
#
# NOTE it is NOT the reciprocal of NAUTICAL_MILES_TO_METERS/1000 to full precision
# (1000/1852 = 0.5399568...), and the two are NOT interchangeable: the engine uses 1852
# for the FUEL question (Game.get_fuel_needed_to_return_to_base) and 0.539957 for the
# MOVEMENT question. Each transcription is used for the question the engine uses it for.
KILOMETERS_TO_NAUTICAL_MILES = 0.539957

# The engine's own positional quantum: Game.update_all_aircraft_position snaps onto a
# waypoint and pops it once the aircraft is within this distance, rather than at 0.
WAYPOINT_SNAP_KM = 0.5

# THE ONE DERIVED NUMERICAL ALLOWANCE of the certified policy, and it is a QUANTUM, not
# a guess. The event is observed at DISCRETE tick boundaries: the controller compares
# the ego's progress against the threshold once per tick, so the state it can be caught
# in is quantized to one tick of travel and one tick of burn. The certificate's own
# recursion reproduces the engine's per-tick leg arithmetic exactly, but measures
# distance through Location.distance_to (the haversine package's mean earth radius,
# 6371.0088 km) while the engine measures it with EARTH_RADIUS_KM = 6371 -- a 1.4e-6
# relative difference, i.e. under 0.3 m over a 200 km leg. That is far below one tick of
# travel and can only move the crossing by ONE tick, and only when the predicted progress
# sits within 1.4e-6 of the threshold. So the certificate is issued over a bracket of
# +/- this many ticks and the live check accepts the same bracket. It is deliberately
# NOT a free parameter: raising it would certify states the engine cannot produce.
CERTIFICATE_TICK_TOLERANCE = 1

# Float noise on top of the one-tick quanta above, stated separately so neither is
# mistaken for the other. 1 m of position covers the earth-radius discrepancy over the
# whole theatre with two orders of magnitude to spare; the fuel epsilon is relative to
# the launch quantity because fuel is measured in tens of thousands of lbs.
_CERTIFICATE_POSITION_EPS_KM = 1e-3
_CERTIFICATE_FUEL_RELATIVE_EPS = 1e-9

# STABLE, MACHINE-READABLE eligibility rejection reasons. They are the vocabulary an
# audit is aggregated over, so they are constants rather than message text.
REASON_NO_ROUTE = "no_route"                      # no assignment in the known-only A_init
REASON_ROUTE_UNRESOLVABLE = "route_unresolvable"  # the predicted route has no geometry
REASON_NO_AIRCRAFT = "no_aircraft"                # no engine aircraft / no speed or rate
REASON_DEGENERATE_LEG = "degenerate_leg"          # leg 1 has no positive length
REASON_EVENT_UNREACHABLE = "event_unreachable"    # the threshold is never crossed in flight
REASON_PRE_EVENT_POPUP_RISK = "pre_event_popup_risk"
REASON_PRE_EVENT_ASSIGNMENT_BOUNDARY = "pre_event_assignment_boundary"
REASON_INVALID_BAND = "invalid_band"              # mild and severe are not both supported
FD_ELIGIBILITY_REJECTION_REASONS: Tuple[str, ...] = (
    REASON_NO_ROUTE,
    REASON_ROUTE_UNRESOLVABLE,
    REASON_NO_AIRCRAFT,
    REASON_DEGENERATE_LEG,
    REASON_EVENT_UNREACHABLE,
    REASON_PRE_EVENT_POPUP_RISK,
    REASON_PRE_EVENT_ASSIGNMENT_BOUNDARY,
    REASON_INVALID_BAND,
)

# The stable machine-readable reason a world is REJECTED at setup because the bounded
# walk exhausted every candidate. It is a NORMAL setup rejection -- accounted exactly
# like a B2 exact-cardinality or fuel-window failure -- and deliberately NOT the
# integrity exception, which describes a world that WAS certified and then contradicted
# its own certificate live.
NO_FD_ELIGIBLE_EGO = "no_fd_eligible_ego"

# Why persistent post-FD adaptation stopped. Recorded rather than inferred: "the ego
# never reached another boundary" and "the ego committed to return" are different facts.
POST_FD_DEACTIVATED_RTB = "rtb_committed"
POST_FD_DEACTIVATED_DEAD = "ego_dead"

CONDITION_CLEAN = "clean"
CONDITION_DAMAGED = "damaged"
CONDITIONS = (CONDITION_CLEAN, CONDITION_DAMAGED)

# The two DAMAGED severities of FD-VARIABLE-SEVERITY-v1. A severity is a refinement of
# `CONDITION_DAMAGED`, never a third condition: both mild and severe episodes ARE damaged
# episodes, and every clean/damaged count keeps its existing meaning.
SEVERITY_MILD = "mild"
SEVERITY_SEVERE = "severe"
SEVERITIES = (SEVERITY_MILD, SEVERITY_SEVERE)

# How a plan's post-damage fuel target was chosen -- recorded on the plan so a reader
# never has to infer it from the mode.
#
#   PLANNED_MIDPOINT       -- LEGACY FD-BASELINE-v1, unchanged: the midpoint of the
#                             PLANNED window, chosen before the run and only VALIDATED
#                             against the live state.
#   LIVE_SEVERITY_MIDPOINT -- FD-VARIABLE-SEVERITY-v1: the midpoint of the LIVE feasible
#                             severity band, derived at the event tick from the live
#                             window and the live fuel. The plan still carries a
#                             PROJECTED target under the same field, which is what the
#                             preflight feasibility check is made against; it is not the
#                             value applied.
TARGET_POLICY_PLANNED_MIDPOINT = "planned_midpoint_v1"
TARGET_POLICY_LIVE_SEVERITY_MIDPOINT = "live_severity_midpoint_v1"

# Numerical floor for a quantity that must be strictly positive to divide by or to
# describe a real leg. Well below any physical value here (fuel is in thousands of lbs,
# legs in hundreds of km), so it only ever catches a degenerate input.
_EPS = 1e-9


class FuelDamageError(RuntimeError):
    """The fuel-damage event could not be planned, or could not be applied as planned.

    Raised -- never swallowed, never clamped, never downgraded to a clean episode. At
    plan time the trainer wraps it as an ``EpisodeAttemptError("setup", ...)``; at
    apply time it surfaces through ``run_episode`` and is attributed to ``run``. Either
    way ``skip_and_account_v1`` records the attempt once and moves to the next scheduled
    seed.
    """


class FuelDamageIntegrityError(RuntimeError):
    """A world CERTIFIED FD-capable at setup contradicted its own certificate live.

    GENERALIZED-V1 (handoff 3l.3). Deliberately NOT a subclass of
    :class:`FuelDamageError`, and deliberately not caught by anything that catches one:
    the two describe opposite situations and must never share a routing.

      * :class:`FuelDamageError` is an ordinary SCIENTIFIC outcome -- this world could
        not carry the event, so the attempt is recorded once by ``skip_and_account_v1``
        and the schedule moves on. Under the LEGACY eligibility policy every live
        failure is one of these, and that routing is UNCHANGED.
      * THIS is an INSTRUMENT failure. Under
        :data:`FD_ELIGIBILITY_CERTIFIED_V1` the world was certified capable of BOTH
        severities at a predicted event state before a single tick was paid for, so a
        live event state that contradicts the certificate means the certificate does not
        describe the simulator -- which makes every episode the certifier touched
        suspect, not just this one. It ABORTS the run, exactly as
        ``graph_train.MeasurementIntegrityError`` and ``_VisualArtifactError`` do, and it
        is never written to ``episode_failures.jsonl``, never counted against a condition
        tally and never entered into ``skip_and_account_v1``.

    A candidate that is INELIGIBLE AT SETUP is emphatically not this exception: nothing
    was certified, so nothing was contradicted. That is the normal
    :data:`NO_FD_ELIGIBLE_EGO` setup rejection.

    It lives here rather than in the trainer because this module must not import
    ``graph_train`` (that would be a circular dependency, and would cost this layer the
    purity that makes it hand-testable). The trainer imports it and routes it.
    """


class FuelDamageMode:
    """How an episode's clean/damaged condition -- and its SEVERITY -- is decided.

    A closed set of strings rather than an enum, so a mode round-trips through
    ``run_config.json`` and a jsonl record as itself, with no decoding step between the
    artifact and the reader.

    THE LEGACY FD-BASELINE-v1 MODES. Every damaged episode is structurally SEVERE: safe
    RTB stays feasible and continuing does not. They are unchanged by
    FD-VARIABLE-SEVERITY-v1 -- same seeds, same conditions, same selected egos, same
    planned-midpoint target -- because a merged, measured baseline exists on them and a
    factor that quietly moved it would invalidate that measurement rather than extend it.

      * ``off``            -- the factor is disabled entirely; every episode is clean and
                              no controller is built. This is the pre-FD behaviour.
      * ``seeded_mixture`` -- TRAINING. The condition is a deterministic function of the
                              episode seed (see :func:`resolve_condition`).
      * ``forced_clean``   -- EVALUATION, member A of a matched pair, and also the CLEAN
                              member of a variable-severity matched TRIAD.
      * ``forced_damaged`` -- EVALUATION, member B of a matched pair. Same generator seed
                              and same placement seed as member A, so the two run the
                              SAME world and differ only in the event.

    THE VARIABLE-SEVERITY MODES (FD-VARIABLE-SEVERITY-v1). They exist because a
    uniformly severe event lets a trained actor learn the shortcut "fuel damage =>
    abort": the label is redundant with the physics. Splitting the damaged half into a
    band where continuing REMAINS feasible and a band where it does not makes the
    response a real decision that has to be read off the ego's own live fuel.

      * ``seeded_variable`` -- TRAINING. The clean/damaged draw is EXACTLY the
                               ``seeded_mixture`` draw (same domain, same order, same
                               probability), and a damaged episode is additionally
                               assigned a severity from its own domain
                               (:func:`resolve_severity`).
      * ``forced_mild``     -- EVALUATION, the MILD member of a matched triad.
      * ``forced_severe``   -- EVALUATION, the SEVERE member of a matched triad.

    The policy is never told which of these ran: no severity label reaches
    ``GraphObservation``, and the only thing that changes in the ego's input is its own
    real ``fuel_norm``.
    """

    OFF = "off"
    SEEDED_MIXTURE = "seeded_mixture"
    FORCED_CLEAN = "forced_clean"
    FORCED_DAMAGED = "forced_damaged"

    SEEDED_VARIABLE = "seeded_variable"
    FORCED_MILD = "forced_mild"
    FORCED_SEVERE = "forced_severe"

    # The modes that carry a mild/severe severity. `forced_clean` is deliberately NOT
    # here: it is shared by both designs (a clean member has no severity in either), and
    # listing it would make "is this a variable-severity run?" unanswerable from the mode
    # of one evaluation member.
    VARIABLE = (SEEDED_VARIABLE, FORCED_MILD, FORCED_SEVERE)
    LEGACY = (OFF, SEEDED_MIXTURE, FORCED_CLEAN, FORCED_DAMAGED)

    ALL = LEGACY + VARIABLE

    # mode -> the severity it forces, for the two modes that force one.
    _FORCED_SEVERITY = {FORCED_MILD: SEVERITY_MILD, FORCED_SEVERE: SEVERITY_SEVERE}


@dataclass(frozen=True)
class FuelDamageParameters:
    """The approved FD-BASELINE-v1 knobs, frozen so a shared default cannot be mutated.

    Attributes:
        mode: one of :attr:`FuelDamageMode.ALL`.
        probability: P(damaged) under ``seeded_mixture``. Ignored by the forced modes,
            but still recorded, because it is what a rerun of the same TRAINING config
            would use.
        leg_progress_threshold: fraction of the first planned leg at which the event
            fires. Must be strictly inside ``(0, 1)``: at 0 the ego has not left the
            base (its remaining route equals its whole route and the RTB floor is 0), and
            at 1 it is already at the target.
        rtb_safety_margin: the reserve multiplier applied to BOTH ends of the window. The
            engine's own value is :data:`DEFAULT_RTB_SAFETY_MARGIN`; anything below 1.0
            would describe a reserve smaller than the fuel actually needed.
        mild_probability: P(mild | damaged) under ``seeded_variable``. Ignored by every
            LEGACY mode and by the forced severity modes, but still recorded, because it
            is what a rerun of the same TRAINING config would use. Together with
            ``probability`` it IS the approved 0.50 / 0.25 / 0.25 distribution, stated as
            two independent numbers rather than as one three-way table so that
            "how often is anything damaged" stays the same knob it has always been.
    """

    mode: str = FuelDamageMode.SEEDED_MIXTURE
    probability: float = DEFAULT_DAMAGE_PROBABILITY
    leg_progress_threshold: float = DEFAULT_LEG_PROGRESS_THRESHOLD
    rtb_safety_margin: float = DEFAULT_RTB_SAFETY_MARGIN
    mild_probability: float = DEFAULT_MILD_PROBABILITY
    # --- GENERALIZED-V1 step 2 (both default to the merged behaviour) ---
    eligibility_policy: str = FD_ELIGIBILITY_LEGACY_V1
    post_fd_wake_policy: str = POST_FD_WAKE_SINGLE_V1

    def validate(self) -> None:
        """Refuse a self-inconsistent parameter set before any episode is built."""
        if self.mode not in FuelDamageMode.ALL:
            raise ValueError(
                "fuel-damage mode must be one of %r, got %r"
                % (list(FuelDamageMode.ALL), self.mode)
            )
        if not (0.0 <= float(self.probability) <= 1.0):
            raise ValueError(
                "fuel-damage probability must be in [0, 1], got %r" % (self.probability,)
            )
        if not (0.0 <= float(self.mild_probability) <= 1.0):
            raise ValueError(
                "fuel-damage mild probability (P(mild | damaged)) must be in [0, 1], "
                "got %r" % (self.mild_probability,)
            )
        if not (0.0 < float(self.leg_progress_threshold) < 1.0):
            raise ValueError(
                "fuel-damage leg progress threshold must be in (0, 1) -- 0 is still at "
                "the base and 1 is already at the target -- got %r"
                % (self.leg_progress_threshold,)
            )
        if float(self.rtb_safety_margin) < 1.0:
            raise ValueError(
                "fuel-damage RTB safety margin must be >= 1.0 (it is a RESERVE on top of "
                "the fuel actually needed), got %r" % (self.rtb_safety_margin,)
            )
        if self.eligibility_policy not in FD_ELIGIBILITY_POLICIES:
            raise ValueError(
                "fuel-damage eligibility policy must be one of %r, got %r"
                % (list(FD_ELIGIBILITY_POLICIES), self.eligibility_policy)
            )
        if self.post_fd_wake_policy not in POST_FD_WAKE_POLICIES:
            raise ValueError(
                "post-FD wake policy must be one of %r, got %r"
                % (list(POST_FD_WAKE_POLICIES), self.post_fd_wake_policy)
            )

    @property
    def enabled(self) -> bool:
        """False only for :attr:`FuelDamageMode.OFF` -- the pre-FD behaviour."""
        return self.mode != FuelDamageMode.OFF

    @property
    def variable_severity(self) -> bool:
        """True iff this mode assigns a mild/severe SEVERITY to a damaged episode.

        The ONE predicate behind every "is this the variable-severity design?" decision
        in this module and in its two harnesses, so the answer cannot be spelled two
        ways. ``forced_clean`` is False here even inside a variable-severity run: a clean
        member has no severity, and asking a member's own mode is not how a RUN's design
        is decided (the trainer keys that off its TRAINING mode).
        """
        return self.mode in FuelDamageMode.VARIABLE

    @property
    def target_policy(self) -> str:
        """How a damaged episode's post-damage fuel is chosen under this mode."""
        return (TARGET_POLICY_LIVE_SEVERITY_MIDPOINT if self.variable_severity
                else TARGET_POLICY_PLANNED_MIDPOINT)

    @property
    def certified_eligibility(self) -> bool:
        """True iff FD capability is a WORLD-ACCEPTANCE property (GENERALIZED-V1).

        The ONE predicate behind every "is this the certified policy?" decision, so the
        question cannot be spelled two ways. False is the merged default.
        """
        return self.eligibility_policy == FD_ELIGIBILITY_CERTIFIED_V1

    @property
    def completion_boundary_wakes(self) -> bool:
        """True iff the ACTUALLY damaged ego wakes again at its own completions."""
        return self.post_fd_wake_policy == POST_FD_WAKE_COMPLETION_BOUNDARY_V1

    def to_record(self) -> Dict[str, Any]:
        """The parameter set as plain JSON scalars, for ``run_config.json`` / records."""
        return {
            "mode": str(self.mode),
            "probability": float(self.probability),
            "leg_progress_threshold": float(self.leg_progress_threshold),
            "rtb_safety_margin": float(self.rtb_safety_margin),
            "rng_domain": FUEL_DAMAGE_RNG_DOMAIN,
            "event_leg_index": EVENT_LEG_INDEX,
            # --- FD-VARIABLE-SEVERITY-v1 -------------------------------------------
            # Always present, so one schema reads both designs; `variable_severity`
            # says whether the two severity fields describe this run at all.
            "variable_severity": bool(self.variable_severity),
            "mild_probability": float(self.mild_probability),
            "severity_rng_domain": FUEL_DAMAGE_SEVERITY_RNG_DOMAIN,
            "severities": list(SEVERITIES),
            "target_policy": str(self.target_policy),
            # --- GENERALIZED-V1 step 2 ---
            # Always present, so ONE schema reads both designs; the two booleans beside
            # them are derived, and are recorded so a reader never has to know which
            # string means which behaviour.
            "eligibility_policy": str(self.eligibility_policy),
            "certified_eligibility": bool(self.certified_eligibility),
            "eligibility_rng_domain": FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN,
            "post_fd_wake_policy": str(self.post_fd_wake_policy),
            "completion_boundary_wakes": bool(self.completion_boundary_wakes),
        }


# =============================================================================
# 1. The private RNG domain (deterministic from the episode seed ALONE)
# =============================================================================

def derive_fuel_damage_seed(episode_seed: int) -> int:
    """Derive this module's private rng seed from the episode seed.

    ``SHA-256("fuel_damage_v1:<seed>")``, first 8 bytes big-endian. Three properties are
    load-bearing and none of them is available from the obvious alternatives:

      * INDEPENDENT of every other consumer. Global ``random`` is shared with whatever
        else an episode touches, and the placement rng's stream position depends on how
        many candidate placements were rejected -- either would make "which episode is
        damaged" a function of unrelated pipeline details.
      * STABLE ACROSS PROCESSES AND RELEASES. Python's ``hash()`` is randomized per
        process by PYTHONHASHSEED, so a run could not be reproduced tomorrow.
      * WELL MIXED. Consecutive episode seeds (the training band is literally
        ``base_seed + g``) must not produce a correlated damaged/clean pattern; a
        cryptographic digest gives that for free where ``seed * k + c`` does not.
    """
    payload = ("%s:%d" % (FUEL_DAMAGE_RNG_DOMAIN, int(episode_seed))).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def derive_fuel_damage_severity_seed(episode_seed: int) -> int:
    """Derive the SEVERITY domain's private rng seed from the episode seed.

    ``SHA-256("fuel_damage_severity_v1:<seed>")``, first 8 bytes big-endian -- the same
    construction as :func:`derive_fuel_damage_seed`, over a DIFFERENT domain string.

    WHY A SECOND DOMAIN RATHER THAN A SECOND DRAW. The v1 stream is consumed in a fixed
    order -- mixture bit, then ego -- and that order is what makes a forced-damaged
    evaluation member select the ego its seeded counterpart would. Taking severity from
    that same stream would insert a draw between the two and change every damaged
    episode's selected ego, which would silently invalidate the approved FD-BASELINE-v1
    measurement rather than extend it. With two domains the two decisions are
    independent by construction: severity cannot move the ego, and the ego cannot move
    severity.
    """
    payload = (
        "%s:%d" % (FUEL_DAMAGE_SEVERITY_RNG_DOMAIN, int(episode_seed))
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def derive_fuel_damage_eligibility_seed(episode_seed: int) -> int:
    """Derive the ELIGIBILITY domain's private rng seed from the episode seed.

    ``SHA-256("fuel_damage_eligibility_v1:<seed>")``, first 8 bytes big-endian -- the
    same construction as :func:`derive_fuel_damage_seed` and
    :func:`derive_fuel_damage_severity_seed`, over a THIRD domain string.

    WHY A THIRD DOMAIN RATHER THAN MORE DRAWS FROM AN EXISTING ONE. The v1 stream is
    consumed in a fixed order -- mixture bit, then legacy ego -- and that order is what
    makes a forced-damaged evaluation member select the ego its seeded counterpart
    would. Drawing the certified policy's candidate permutation there would insert draws
    between the two and change every legacy damaged episode's selected ego, silently
    invalidating the approved FD-BASELINE-v1 and FD-VARIABLE-SEVERITY-v1 measurements
    rather than extending them. With three domains the three decisions are independent
    by construction, and -- the property the certified policy is built on -- the
    eligibility walk is a function of the EPISODE SEED ALONE, so CLEAN, MILD and SEVERE
    members of one matched group certify the SAME ego on the SAME world.
    """
    payload = (
        "%s:%d" % (FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN, int(episode_seed))
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _fuel_damage_rng(episode_seed: int) -> random.Random:
    """A fresh, private :class:`random.Random` for one episode seed."""
    return random.Random(derive_fuel_damage_seed(episode_seed))


def _fuel_damage_eligibility_rng(episode_seed: int) -> random.Random:
    """A fresh, private :class:`random.Random` in the ELIGIBILITY domain."""
    return random.Random(derive_fuel_damage_eligibility_seed(episode_seed))


def eligibility_ordinal_permutation(
    count: int, rng: random.Random
) -> Tuple[int, ...]:
    """A seed-driven permutation of ``range(count)`` -- the candidate visit order.

    Fisher-Yates written out explicitly rather than delegated to ``random.shuffle``: the
    permutation is part of a reproducibility contract, so the exact draw sequence
    (``rng.randrange(i + 1)`` for descending ``i``) is stated here instead of inherited
    from a library internal that is free to change.

    ORDINALS, never ids. Generated agent ids are not seed-derived (``CLAUDE.md`` Sec 8),
    so a permutation keyed on their text would make an episode's damaged ego
    irreproducible across runs of the same seed -- the exact defect the generalized B2
    ordering rule (handoff 3l.2) exists to prevent.

    A DELIBERATE MIRROR of ``graph_hidden_placement._ordinal_permutation``: the same
    algorithm, over a DIFFERENT private domain and for a different decision. It is
    written here rather than imported so that a future change to hidden placement's
    candidate ordering cannot silently move which ego a certified episode damages; the
    equivalence of the two implementations is TEST-ENFORCED, exactly as
    :func:`rtb_command_for` mirrors the executor's one emission site.
    """
    ordinals = list(range(int(count)))
    for i in range(len(ordinals) - 1, 0, -1):
        j = rng.randrange(i + 1)
        ordinals[i], ordinals[j] = ordinals[j], ordinals[i]
    return tuple(ordinals)


def _fuel_damage_severity_rng(episode_seed: int) -> random.Random:
    """A fresh, private :class:`random.Random` in the SEVERITY domain."""
    return random.Random(derive_fuel_damage_severity_seed(episode_seed))


def resolve_condition(*, episode_seed: int, params: FuelDamageParameters) -> str:
    """The SCHEDULED condition of an episode -- ``clean`` or ``damaged``.

    PURE, and deliberately independent of the world: the trainer needs the condition
    before an episode is built (to schedule it) and after it has failed (to account for
    it by condition), and neither of those has a context to inspect.

    The mixture draw is taken as the FIRST value of the private rng in every mode,
    including the forced ones. It is discarded there, but taking it keeps the stream
    position identical, so :func:`plan_fuel_damage`'s ego selection is the same draw
    whether the episode was scheduled by the mixture or forced by an evaluation pair.

    ``seeded_variable`` resolves the condition through EXACTLY this path -- the same
    domain, the same single draw, the same ``probability`` -- so a seed's clean/damaged
    assignment is identical under the legacy and the variable design. Only what happens
    to a damaged episode afterwards differs (:func:`resolve_severity`), and the two
    forced severity modes force ``damaged`` the way ``forced_damaged`` does.
    """
    params.validate()
    if params.mode == FuelDamageMode.OFF:
        return CONDITION_CLEAN
    rng = _fuel_damage_rng(episode_seed)
    drawn = rng.random() < float(params.probability)
    if params.mode == FuelDamageMode.FORCED_CLEAN:
        return CONDITION_CLEAN
    if params.mode in (FuelDamageMode.FORCED_DAMAGED, FuelDamageMode.FORCED_MILD,
                       FuelDamageMode.FORCED_SEVERE):
        return CONDITION_DAMAGED
    return CONDITION_DAMAGED if drawn else CONDITION_CLEAN


def resolve_severity(
    *, episode_seed: int, params: FuelDamageParameters
) -> Optional[str]:
    """The SCHEDULED severity of a damaged episode -- ``mild``, ``severe`` or ``None``.

    PURE, world-free, and a function of the episode seed ALONE within its own versioned
    domain, for the same three reasons :func:`derive_fuel_damage_seed` gives: it must be
    unreachable from global ``random`` / torch / the placement stream, stable across
    processes and releases, and well mixed over consecutive seeds.

    ``None`` means "this episode has no severity", which is a different statement from
    "mild" and is returned in exactly two cases: a LEGACY mode (whose damaged episodes
    are all structurally severe but carry no severity LABEL, and must keep reporting
    ``null`` so a legacy record is never re-read as a variable-severity one), and a
    CLEAN episode under any mode.

    Under ``seeded_variable`` the single draw is ``rng.random() < mild_probability``.
    The forced severity modes take that draw too and discard it: it costs nothing, keeps
    the stream position identical across the three members of a matched triad, and means
    a later severity-domain draw (should one ever be added) cannot make a forced member
    diverge from the seeded episode it is supposed to reproduce.
    """
    params.validate()
    if not params.variable_severity:
        return None
    if resolve_condition(episode_seed=episode_seed, params=params) != CONDITION_DAMAGED:
        return None
    drawn = (_fuel_damage_severity_rng(episode_seed).random()
             < float(params.mild_probability))
    forced = FuelDamageMode._FORCED_SEVERITY.get(params.mode)
    if forced is not None:
        return forced
    return SEVERITY_MILD if drawn else SEVERITY_SEVERE


# =============================================================================
# 2. The engine's fuel arithmetic + great-circle helpers (pure)
# =============================================================================

def fuel_for_distance_km(
    distance_km: float, *, speed_knots: float, fuel_rate: float
) -> float:
    """Fuel burned flying ``distance_km``, in BLADE's own arithmetic.

    A transcription of ``Game.get_fuel_needed_to_return_to_base``:
    kilometres -> metres -> nautical miles, divided by the aircraft's KNOTS speed to get
    hours, multiplied by its lbs/hr ``fuel_rate``. Written as one function so every
    quantity in the window -- the RTB floor, the continue requirement and the projected
    fuel at the event -- comes out of a single site that matches the engine.

    Raises:
        FuelDamageError: on a non-positive or non-finite speed / rate / distance, which
            would make the window meaningless rather than merely large.
    """
    d = float(distance_km)
    v = float(speed_knots)
    rate = float(fuel_rate)
    if not math.isfinite(d) or d < 0.0:
        raise FuelDamageError("distance must be finite and >= 0, got %r" % (distance_km,))
    if not math.isfinite(v) or v <= _EPS:
        raise FuelDamageError(
            "aircraft speed must be finite and > 0 knots to derive a flight time, got %r"
            % (speed_knots,)
        )
    if not math.isfinite(rate) or rate <= _EPS:
        raise FuelDamageError(
            "aircraft fuel_rate must be finite and > 0 lbs/hr, got %r" % (fuel_rate,)
        )
    nautical_miles = (d * 1000.0) / NAUTICAL_MILES_TO_METERS
    hours = nautical_miles / v
    return hours * rate


def _unit_vector(loc: Location) -> Tuple[float, float, float]:
    """Unit 3-vector of a lat/lon on the sphere."""
    phi = math.radians(float(loc.latitude))
    lam = math.radians(float(loc.longitude))
    cos_phi = math.cos(phi)
    return (cos_phi * math.cos(lam), cos_phi * math.sin(lam), math.sin(phi))


def interpolate_great_circle(start: Location, end: Location, fraction: float) -> Location:
    """The point ``fraction`` of the way along the great circle ``start -> end``.

    Spherical linear interpolation on unit vectors -- the same surface the haversine
    distances everywhere else in this module are measured on, so the along-track distance
    from ``start`` to the returned point is exactly ``fraction * distance(start, end)``.
    A flat lat/lon lerp would not have that property, and the RTB floor is measured FROM
    this point.

    Degenerate inputs (coincident or antipodal endpoints) return ``start``: there is no
    meaningful interpolation, and the caller's leg-length guard has already rejected the
    only case that matters.
    """
    f = float(fraction)
    a = _unit_vector(start)
    b = _unit_vector(end)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    omega = math.acos(dot)
    sin_omega = math.sin(omega)
    if abs(sin_omega) < _EPS:
        return Location(float(start.latitude), float(start.longitude))
    s0 = math.sin((1.0 - f) * omega) / sin_omega
    s1 = math.sin(f * omega) / sin_omega
    v = tuple(s0 * x + s1 * y for x, y in zip(a, b))
    norm = math.sqrt(sum(c * c for c in v))
    if norm < _EPS:
        return Location(float(start.latitude), float(start.longitude))
    x, y, z = (c / norm for c in v)
    return Location(math.degrees(math.asin(z)), math.degrees(math.atan2(y, x)))


def _polyline_length_km(points: Sequence[Location]) -> float:
    """Total great-circle length of an ordered polyline (0.0 for fewer than 2 points)."""
    return sum(
        points[i].distance_to(points[i + 1]) for i in range(len(points) - 1)
    )


@dataclass(frozen=True)
class _Window:
    """The strict fuel window measured FROM one position, with its two distances.

    ``[rtb_fuel_floor, continue_fuel_requirement)`` is the half-open interval a
    post-damage quantity must sit in: at or above the floor the ego can still fly home
    with the engine's reserve, and strictly below the requirement it can no longer
    complete its route and then return with that same reserve.
    """

    rtb_distance_km: float
    continue_distance_km: float
    rtb_fuel_floor: float
    continue_fuel_requirement: float


def measure_window(
    *,
    position: Location,
    route: Sequence[Location],
    home_base: Location,
    speed_knots: float,
    fuel_rate: float,
    margin: float,
) -> _Window:
    """Measure the strict window from ``position`` -- the ONE arithmetic site.

    Called TWICE per damaged episode with the same route, home base and aircraft
    parameters, and a different position: once at plan time from the PROJECTED event
    point, and once at fire time from the aircraft's LIVE position. Sharing this function
    is the point -- a planned bound and a live bound computed by two similar-looking
    expressions could drift apart, and the whole reason the live re-check exists is that
    the two must be comparable.

    The RTB leg is position -> home base, exactly what
    ``Game.get_fuel_needed_to_return_to_base`` measures. The continue leg is
    position -> every remaining route target in predicted order -> home base, so the
    already-flown part of the route is never charged twice.
    """
    points = list(route)
    if not points:
        raise FuelDamageError("cannot measure a window against an empty route")
    rtb_distance_km = float(position.distance_to(home_base))
    continue_distance_km = float(
        _polyline_length_km([position] + points) + points[-1].distance_to(home_base)
    )
    return _Window(
        rtb_distance_km=rtb_distance_km,
        continue_distance_km=continue_distance_km,
        rtb_fuel_floor=float(margin) * fuel_for_distance_km(
            rtb_distance_km, speed_knots=speed_knots, fuel_rate=fuel_rate
        ),
        continue_fuel_requirement=float(margin) * fuel_for_distance_km(
            continue_distance_km, speed_knots=speed_knots, fuel_rate=fuel_rate
        ),
    )


@dataclass(frozen=True)
class _SeverityBand:
    """The feasible post-damage fuel interval for ONE severity, and its midpoint.

    ``[low, high]`` with explicit inclusivity, because the two bands are genuinely
    different intervals and collapsing that difference is how a mild event would be
    allowed to land exactly on the boundary that makes continuing infeasible:

      * SEVERE (and LEGACY FD-BASELINE-v1, which has the same interval)
        ``[rtb_fuel_floor, continue_fuel_requirement)`` -- flying straight home stays
        feasible with the engine's reserve, completing the route and returning does not.
      * MILD ``(continue_fuel_requirement, fuel_before)`` -- OPEN at both ends. Above the
        continue requirement, so completing the route AND returning is still genuinely
        feasible; strictly below the pre-damage fuel, so the event is a real LOSS rather
        than a relabelled no-op.

    ``target`` is the MIDPOINT, the approved deterministic choice and the point furthest
    from both ends, so neither bound is decided by floating-point noise.
    """

    severity: Optional[str]
    low: float
    high: float
    low_inclusive: bool
    high_inclusive: bool
    target: float

    @property
    def width(self) -> float:
        return float(self.high) - float(self.low)

    def contains(self, value: float) -> bool:
        """Is ``value`` inside this interval, honouring both inclusivity flags?"""
        v = float(value)
        low_ok = (v >= self.low) if self.low_inclusive else (v > self.low)
        high_ok = (v <= self.high) if self.high_inclusive else (v < self.high)
        return low_ok and high_ok

    def describe(self) -> str:
        """``[lo, hi)`` style text for an error message."""
        return "%s%.3f, %.3f%s" % (
            "[" if self.low_inclusive else "(", self.low, self.high,
            "]" if self.high_inclusive else ")",
        )


def severity_band(
    *, window: _Window, fuel_before: float, severity: Optional[str]
) -> _SeverityBand:
    """The feasible band for ``severity``, measured against ``window``.

    THE ONE ARITHMETIC SITE for both severities and for the legacy design, called TWICE
    per damaged episode with the same severity and a different window: once at plan time
    against the PROJECTED window and projected fuel, and once at fire time against the
    LIVE window and the LIVE fuel. Sharing it is the point -- a planned band and a live
    band computed by two similar-looking expressions could drift apart, and the live
    re-measurement exists precisely so the two are comparable.

    ``severity is None`` selects the LEGACY FD-BASELINE-v1 interval, which is the same
    interval as ``severe``. That is not a coincidence to be tidied away: the legacy
    design's every damaged episode IS the severe case, and stating it as one shared
    interval is what makes "severe reproduces the legacy physics" checkable rather than
    merely asserted. What still differs is WHERE the interval is measured -- legacy
    chooses its target from the PLANNED window and only validates it live, while the
    variable design derives the target from the LIVE window (see
    :data:`TARGET_POLICY_LIVE_SEVERITY_MIDPOINT`).

    Raises:
        FuelDamageError: on an unknown severity string. Never falls back to a band.
    """
    if severity is not None and severity not in SEVERITIES:
        raise FuelDamageError(
            "severity must be one of %r or None, got %r" % (list(SEVERITIES), severity)
        )
    if severity == SEVERITY_MILD:
        low = float(window.continue_fuel_requirement)
        high = float(fuel_before)
        low_inclusive = False
        high_inclusive = False
    else:  # SEVERE, and the legacy design's single band
        low = float(window.rtb_fuel_floor)
        high = float(window.continue_fuel_requirement)
        low_inclusive = True
        high_inclusive = False
    return _SeverityBand(
        severity=severity, low=low, high=high,
        low_inclusive=low_inclusive, high_inclusive=high_inclusive,
        target=0.5 * (low + high),
    )


def _require_valid_band(
    band: _SeverityBand,
    *,
    ego_id: str,
    fuel_before: float,
    window: _Window,
    where: str,
) -> float:
    """Validate ``band`` and return its target, or raise. NOTHING is clamped.

    Four facts, checked in this order and each with its own message, because "the event
    could not be applied" is not one failure but four different physical situations:

      1. the interval is NON-DEGENERATE -- a band of zero (or negative) width has no
         interior, so there is no quantity that means what the severity says;
      2. the midpoint really lies inside it, honouring the inclusivity that distinguishes
         mild from severe;
      3. the mutation is a real LOSS (strictly below the pre-damage fuel);
      4. flying straight home stays feasible -- the target is at or above the RTB floor,
         which the reserve contract requires of BOTH severities. For severe it is the
         band's own lower bound; for mild it follows from the window being non-empty, and
         it is checked anyway rather than argued, because it is the one property a
         difficulty factor must never quietly lose.

    ``where`` is ``"planned"`` or ``"live"`` and appears in every message, so a reader of
    an accounted failure knows immediately whether the preflight projection or the real
    event state refused it -- they are different findings and land in different pipeline
    stages.
    """
    label = band.severity or "legacy"
    if not (band.width > _EPS):
        raise FuelDamageError(
            "ego %s: no %s %s fuel band -- the interval %s is empty, so there is no "
            "post-damage quantity that means %r. RTB floor %.3f over %.1f km, continue "
            "requirement %.3f over %.1f km, fuel before %.3f."
            % (ego_id, where, label, band.describe(), label, window.rtb_fuel_floor,
               window.rtb_distance_km, window.continue_fuel_requirement,
               window.continue_distance_km, float(fuel_before))
        )
    if not band.contains(band.target):
        raise FuelDamageError(
            "ego %s: the %s %s post-damage fuel %.6f is not inside its own band %s"
            % (ego_id, where, label, band.target, band.describe())
        )
    if not (band.target < float(fuel_before)):
        raise FuelDamageError(
            "ego %s: the %s %s post-damage fuel %.6f is not below the pre-damage fuel "
            "%.6f, so the event would not be a loss"
            % (ego_id, where, label, band.target, float(fuel_before))
        )
    if band.target < window.rtb_fuel_floor:
        raise FuelDamageError(
            "ego %s: the %s %s post-damage fuel %.6f is below the RTB floor %.3f over "
            "%.1f km; the ego could not fly home with the reserve, so the event would be "
            "a kill rather than a decision"
            % (ego_id, where, label, band.target, window.rtb_fuel_floor,
               window.rtb_distance_km)
        )
    return float(band.target)


# =============================================================================
# 2b. GENERALIZED-V1 step 2: the TICK-AWARE event certificate
# =============================================================================
# The LEGACY design projects the event point by DISTANCE ("30 % of leg 1") and its fuel
# by the distance flown, then relies on a live re-check to catch the difference. That is
# preserved untouched for `legacy_selected_ego_v1`.
#
# The CERTIFIED policy cannot work that way, because it has to promise BEFORE the
# episode runs that BOTH severities are physically constructible at the state the event
# will really be observed in. So it predicts the DISCRETE state instead, out of the
# frozen engine's own one-second semantics:
#
#   tick 0  Phase 2 emits `launch_aircraft_from_airbase`. `Game.step` runs
#           `handle_action` and THEN `update_game_state`, so the aircraft is appended to
#           `scenario.aircraft` with NO route and `update_all_aircraft_position` burns
#           `fuel_rate / 3600` for it without moving it. ONE burn, ZERO movement.
#   tick 1  Phase 2 emits `move_aircraft` (the ego is airborne, the target is beyond the
#           unified radius and the live route is empty). The same update then performs
#           the FIRST movement leg and burns again.
#   tick T  (T >= 1) the observation the tick loop holds at the TOP of the tick -- the
#           one `FuelDamageController.maybe_apply` reads -- is the one tick T-1's
#           `env.step` returned. It therefore shows exactly ``T - 1`` movement legs and
#           exactly ``T`` burns.
#
# Hence: movements ``m`` are observed at tick ``m + 1`` holding
# ``fuel_at_launch - (m + 1) * fuel_rate / 3600``. There is NO invented reserve anywhere
# in that derivation; the only allowance is :data:`CERTIFICATE_TICK_TOLERANCE`, which is
# the engine's own observation quantum (see its comment).


def engine_leg_distance_km(remaining_km: float, *, speed_knots: float) -> float:
    """ONE tick's ground travel toward a waypoint, in the FROZEN ENGINE's arithmetic.

    A transcription of ``blade.utils.utils.get_next_coordinates``, which is the only
    site that moves an aircraft:

        total_time_seconds = max(floor(remaining_km * KM_TO_NM / |speed| * 3600), 1e-4)
        leg                = remaining_km / total_time_seconds

    and, when the whole remaining distance is shorter than that leg, the engine jumps
    straight to the waypoint -- reproduced here by returning ``remaining_km``.

    THE FLOOR IS NOT COSMETIC. It makes the per-tick leg slightly LONGER than the
    nominal ``speed / (KM_TO_NM * 3600)``, and it is recomputed from the CURRENT
    remaining distance every tick, so the aircraft's progress is not a straight line in
    tick count. Reproducing it is the difference between predicting the event tick and
    guessing it.

    Raises:
        FuelDamageError: on a non-finite / non-positive speed, or a non-finite or
            negative remaining distance -- the same loud policy as
            :func:`fuel_for_distance_km`, and for the same reason.
    """
    d = float(remaining_km)
    v = abs(float(speed_knots))  # the engine's own `speed if speed >= 0 else -speed`
    if not math.isfinite(d) or d < 0.0:
        raise FuelDamageError(
            "remaining distance must be finite and >= 0, got %r" % (remaining_km,)
        )
    if not math.isfinite(v) or v <= _EPS:
        raise FuelDamageError(
            "aircraft speed must be finite and non-zero to derive a movement leg, got %r"
            % (speed_knots,)
        )
    total_time_seconds = max(
        math.floor(d * KILOMETERS_TO_NAUTICAL_MILES / v * 3600.0), 0.0001
    )
    leg = d / total_time_seconds
    return d if d < leg else leg


@dataclass(frozen=True)
class _LegState:
    """The ego's own leg-1 state as observed at the TOP of one tick.

    ``progress`` is measured exactly as :meth:`FuelDamageController._progress_of`
    measures it at runtime -- ``(L - remaining) / L`` against the great-circle distance
    to the first planned target -- so the predicted crossing and the observed crossing
    are the same quantity, not two similar ones.
    """

    movements: int
    tick: int
    remaining_km: float
    progress: float
    leg_km: float
    arrived: bool


def predict_leg_states(
    *, leg_length_km: float, speed_knots: float, max_movements: Optional[int] = None
) -> Tuple[_LegState, ...]:
    """The ordered per-tick states of a direct flight down leg 1, engine-exact.

    Index ``m`` of the result is the state after ``m`` movement legs, observed at the
    top of tick ``m + 1`` (see this section's header). Element 0 is therefore the
    airborne-but-not-yet-moving state at the top of tick 1, with ``progress == 0``.

    The walk stops when the engine would snap onto the waypoint
    (``remaining < WAYPOINT_SNAP_KM``, the check ``update_all_aircraft_position`` makes
    BEFORE moving) or at ``max_movements``. Every step moves at least
    ``1 / (KM_TO_NM / speed * 3600)`` km, so the walk is bounded by construction; the
    explicit cap below is a second guard, not the primary one.

    Raises:
        FuelDamageError: on a degenerate leg or an unusable speed.
    """
    L = float(leg_length_km)
    if not math.isfinite(L) or L <= _EPS:
        raise FuelDamageError(
            "cannot predict leg states for a leg of length %r km" % (leg_length_km,)
        )
    v = abs(float(speed_knots))
    if not math.isfinite(v) or v <= _EPS:
        raise FuelDamageError(
            "cannot predict leg states at speed %r kt" % (speed_knots,)
        )
    # Ticks to fly the whole leg at the NOMINAL per-tick distance, which is a lower
    # bound on the real one (the engine's floor only ever lengthens a leg).
    hard_cap = int(math.ceil(L * KILOMETERS_TO_NAUTICAL_MILES / v * 3600.0)) + 2
    cap = hard_cap if max_movements is None else min(int(max_movements), hard_cap)

    states: List[_LegState] = []
    remaining = L
    for m in range(cap + 1):
        arrived = remaining < WAYPOINT_SNAP_KM
        leg = 0.0 if arrived else engine_leg_distance_km(remaining, speed_knots=v)
        states.append(
            _LegState(
                movements=m,
                tick=m + 1,
                remaining_km=remaining,
                progress=(L - remaining) / L,
                leg_km=leg,
                arrived=arrived,
            )
        )
        if arrived:
            break
        remaining = max(0.0, remaining - leg)
    return tuple(states)


@dataclass(frozen=True)
class FdEventCertificate:
    """The PROMISE a certified world makes about its own fuel-damage event.

    Issued at setup, BEFORE a tick is paid for, and checked again against the real event
    state before the mutation (:meth:`FuelDamageController.maybe_apply`). Everything on
    it is a plain scalar so it round-trips through a jsonl record and can be compared
    field by field between two runs of the same seed.

    THE CERTIFICATE COVERS A BRACKET, NOT A POINT. The bands are validated at every tick
    in ``bracket_ticks`` -- the nominal tick and its neighbours within
    :data:`CERTIFICATE_TICK_TOLERANCE` -- because the tick the engine actually observes
    the crossing on is quantized (see that constant). A certificate that held only at
    the nominal tick would be a certificate the engine is free to miss by one tick.
    """

    ego_id: str
    event_tick: int
    movement_count: int
    progress: float
    latitude: float
    longitude: float
    fuel_before: float
    fuel_per_tick: float
    leg_length_km: float
    leg_km_per_tick: float
    route_length: int
    first_target_latitude: float
    first_target_longitude: float
    rtb_distance_km: float
    continue_distance_km: float
    rtb_fuel_floor: float
    continue_fuel_requirement: float
    mild_band_low: float
    mild_band_high: float
    mild_target: float
    severe_band_low: float
    severe_band_high: float
    severe_target: float
    band_margin_fuel: float
    required_band_margin_fuel: float
    tick_tolerance: int
    fuel_tolerance: float
    position_tolerance_km: float
    bracket_ticks: Tuple[int, ...]
    pre_event_ticks_checked: int
    detection_km: float

    @property
    def event_location(self) -> Location:
        """The nominal event point as a :class:`Location`."""
        return Location(float(self.latitude), float(self.longitude))

    def to_record(self) -> Dict[str, Any]:
        """The certificate as plain JSON scalars."""
        return {
            "ego_id": str(self.ego_id),
            "event_tick": int(self.event_tick),
            "movement_count": int(self.movement_count),
            "progress": float(self.progress),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "fuel_before": float(self.fuel_before),
            "fuel_per_tick": float(self.fuel_per_tick),
            "leg_length_km": float(self.leg_length_km),
            "leg_km_per_tick": float(self.leg_km_per_tick),
            "route_length": int(self.route_length),
            "first_target_latitude": float(self.first_target_latitude),
            "first_target_longitude": float(self.first_target_longitude),
            "rtb_distance_km": float(self.rtb_distance_km),
            "continue_distance_km": float(self.continue_distance_km),
            "rtb_fuel_floor": float(self.rtb_fuel_floor),
            "continue_fuel_requirement": float(self.continue_fuel_requirement),
            "mild_band_low": float(self.mild_band_low),
            "mild_band_high": float(self.mild_band_high),
            "mild_target": float(self.mild_target),
            "severe_band_low": float(self.severe_band_low),
            "severe_band_high": float(self.severe_band_high),
            "severe_target": float(self.severe_target),
            "band_margin_fuel": float(self.band_margin_fuel),
            "required_band_margin_fuel": float(self.required_band_margin_fuel),
            "tick_tolerance": int(self.tick_tolerance),
            "fuel_tolerance": float(self.fuel_tolerance),
            "position_tolerance_km": float(self.position_tolerance_km),
            "bracket_ticks": list(int(t) for t in self.bracket_ticks),
            "pre_event_ticks_checked": int(self.pre_event_ticks_checked),
            "detection_km": float(self.detection_km),
        }


@dataclass(frozen=True)
class FdEligibilityCandidate:
    """ONE candidate the certified walk considered, and what became of it.

    ``ordinal`` is the AUTHORITATIVE identity -- the candidate's index in the scheduled
    agent sequence -- and ``ego_id`` is an episode-local convenience beside it. Generated
    ids are not seed-derived (``CLAUDE.md`` Sec 8), so cross-run comparison is by ordinal
    and by physical geometry, NEVER by id text.
    """

    ordinal: int
    ego_id: str
    accepted: bool
    reason: Optional[str] = None
    detail: Optional[str] = None
    route_length: Optional[int] = None
    first_target_id: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "ordinal": int(self.ordinal),
            "ego_id": str(self.ego_id),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "detail": self.detail,
            "route_length": self.route_length,
            "first_target_id": self.first_target_id,
        }


@dataclass(frozen=True)
class FdEligibilityAudit:
    """The complete, JSON-ready record of ONE episode's certified eligibility walk.

    It exists because "which ego was damaged" is not, by itself, an answer: a bounded
    deterministic walk that rejected two candidates and accepted a third has to be able
    to say WHICH candidates, in WHICH order and for WHICH reasons, or its denominator is
    unreadable. Task 4 persists and aggregates it; Task 2 only has to make it truthful.

    ``selected_ego_id`` is the COUNTERFACTUAL selection on a CLEAN episode -- the ego the
    matched mild and severe members of the same world+seed will really damage. It is
    stored here rather than on ``FuelDamagePlan.ego_id``, which keeps its historical
    meaning of "the ego this episode actually damages" and therefore stays ``None`` on a
    clean plan.
    """

    policy: str
    rng_domain: Optional[str] = None
    derived_seed: Optional[int] = None
    candidate_count: int = 0
    candidate_order: Tuple[int, ...] = ()
    considered_ordinals: Tuple[int, ...] = ()
    candidates: Tuple[FdEligibilityCandidate, ...] = ()
    selected_ordinal: Optional[int] = None
    selected_ego_id: Optional[str] = None
    certificate: Optional[FdEventCertificate] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "policy": str(self.policy),
            "rng_domain": self.rng_domain,
            "derived_seed": self.derived_seed,
            "candidate_count": int(self.candidate_count),
            "candidate_order": [int(o) for o in self.candidate_order],
            "considered_ordinals": [int(o) for o in self.considered_ordinals],
            "candidates": [c.to_record() for c in self.candidates],
            "selected_ordinal": self.selected_ordinal,
            "selected_ego_id": self.selected_ego_id,
            "certificate": (
                None if self.certificate is None else self.certificate.to_record()
            ),
        }


def certify_fd_candidate(
    *,
    ego_id: str,
    launch_point: Location,
    route_points: Sequence[Location],
    home_base: Location,
    speed_knots: float,
    fuel_rate: float,
    fuel_at_launch: float,
    params: FuelDamageParameters,
    detection_km: float,
    world_targets: Sequence[Tuple[str, Location]],
    belief_target_ids: Sequence[str],
) -> Tuple[Optional[FdEventCertificate], Optional[Tuple[str, str]]]:
    """Decide whether ONE ego can carry BOTH severities, and certify it if so.

    PURE -- every input is a plain number, a :class:`Location`, a string or a sequence of
    them, so the whole decision is testable without an engine, a solver or a scenario.
    Returns ``(certificate, None)`` on acceptance and ``(None, (reason, detail))`` on
    rejection; exactly one half is ever populated, mirroring
    ``graph_hidden_placement._select_leg``'s shape.

    THE FOUR THINGS A CANDIDATE MUST SATISFY, checked in this order:

      1. LEG 1 IS REAL -- positive great-circle length, and the threshold is actually
         crossed while the ego is still flying it.
      2. THE PRE-EVENT PREFIX IS STABLE (handoff 3l.4's no-communication rule applied to
         the certificate). At EVERY Phase-1 position strictly before the event tick the
         ego must neither sense a live world target absent from its own t=0 belief
         inventory -- which ``decide_triggers`` would classify as a POP_UP and wake the
         actor on, moving the route out from under the certificate -- nor already be
         inside the unified arrival/detection radius of its current first assigned
         target, where Phase 2 could attack, confirm and advance before the certified
         state exists. Neither is repaired by suppressing a trigger or changing actor
         behaviour: the CANDIDATE is rejected instead, truthfully.
      3. BOTH SEVERITY BANDS EXIST AT THE EVENT STATE, on the SAME ego --
         ``F_rtb < F_continue < F_before`` with each of the two intervals wider than ONE
         TICK OF BURN. That margin is the engine's own quantum, not a tuned constant: a
         band narrower than a single tick's fuel could be crossed by the quantization
         the certificate already tolerates.
      4. THE SAME HOLDS ACROSS THE WHOLE TOLERATED BRACKET, so the promise does not
         depend on the engine picking the nominal tick.

    A one-assignment ego is DELIBERATELY eligible. Requiring two or more would bias the
    generalized sample toward solver-stacked allocations, which is a research question
    and not a physical one; such an ego simply has no later completion boundary to reach.

    Args:
        ego_id: the candidate, for messages only -- selection is by ordinal.
        launch_point: the ONE shared launch point (== the ego's home base).
        route_points: the PREDICTED route in flown order (``predict_route``), leg 1's
            endpoint first.
        home_base: where RTB resolves to.
        speed_knots / fuel_rate / fuel_at_launch: read off the LIVE engine aircraft,
            never off ``Agent`` (``scenario_factory`` substitutes a 250 kt planning speed
            for a grounded unit).
        params: the approved knobs; ``leg_progress_threshold`` and ``rtb_safety_margin``
            are the two this reads.
        detection_km: the ONE unified sensing / arrival / attack radius, taken from the
            executor so the certificate cannot disagree with the runtime sensor.
        world_targets: ``(target_id, Location)`` for EVERY live enemy target in the
            world -- current world truth, used for a setup-only certificate and never
            exposed to the actor.
        belief_target_ids: the target ids in the ego's OWN t=0 belief inventory. Anything
            in ``world_targets`` and not in here is a pop-up waiting to happen, whether
            or not it is one of the construction path's "hidden" targets.
    """
    params.validate()
    points = list(route_points)
    if not points:
        return None, (REASON_NO_ROUTE, "ego %s has an empty predicted route" % ego_id)

    threshold = float(params.leg_progress_threshold)
    margin = float(params.rtb_safety_margin)
    known_ids = {str(t) for t in belief_target_ids}

    first_target = points[0]
    leg_length_km = float(launch_point.distance_to(first_target))
    if not math.isfinite(leg_length_km) or leg_length_km <= _EPS:
        return None, (
            REASON_DEGENERATE_LEG,
            "ego %s: leg 1 has length %r km, so no event point exists on it"
            % (ego_id, leg_length_km),
        )

    try:
        states = predict_leg_states(
            leg_length_km=leg_length_km, speed_knots=speed_knots
        )
        # Validates speed and fuel_rate through the ONE engine-arithmetic site before
        # anything is derived from them; the returned value (fuel over zero distance) is
        # zero and is deliberately discarded.
        fuel_for_distance_km(0.0, speed_knots=speed_knots, fuel_rate=fuel_rate)
    except FuelDamageError as exc:
        return None, (REASON_NO_AIRCRAFT, str(exc))
    # The engine burns exactly this EVERY airborne tick, unconditionally and regardless
    # of whether the aircraft moved (`Game.update_all_aircraft_position`).
    fuel_per_tick = float(fuel_rate) / 3600.0

    event_index = next(
        (i for i, st in enumerate(states) if st.progress >= threshold), None
    )
    if event_index is None:
        return None, (
            REASON_EVENT_UNREACHABLE,
            "ego %s: %.0f%% of leg 1 is never observed -- the engine snaps onto the "
            "target at %.3f km before a tick shows that progress"
            % (ego_id, 100.0 * threshold, WAYPOINT_SNAP_KM),
        )

    # ---- (2) the pre-event prefix must be stable -------------------------------
    # Every Phase-1 position STRICTLY BEFORE the event tick. Tick 0 is excluded by
    # construction: the ego is still in its airbase inventory then, and
    # `sensed_target_ids` returns {} for an ego with no live position. The EVENT tick
    # itself is deliberately allowed to carry a simultaneous pop-up -- the fuel mutation
    # happens at the top of the tick, before Phase 1, and the trigger layer coalesces
    # both events into ONE actor decision.
    for st in states[:event_index]:
        here = interpolate_great_circle(launch_point, first_target, st.progress)
        if st.remaining_km <= detection_km:
            return None, (
                REASON_PRE_EVENT_ASSIGNMENT_BOUNDARY,
                "ego %s: at tick %d (progress %.4f) it is already %.2f km from its first "
                "assigned target, inside the %.1f km arrival radius, so Phase 2 could "
                "attack, confirm and advance before the certified event state exists"
                % (ego_id, st.tick, st.progress, st.remaining_km, detection_km),
            )
        for target_id, target_loc in world_targets:
            if str(target_id) in known_ids:
                continue
            distance_km = float(here.distance_to(target_loc))
            if distance_km <= detection_km:
                return None, (
                    REASON_PRE_EVENT_POPUP_RISK,
                    "ego %s: at tick %d (progress %.4f) target %s -- absent from its own "
                    "t=0 belief -- is %.2f km away, inside the %.1f km sensor radius, so "
                    "a POP_UP could wake the actor and move the route before the event"
                    % (ego_id, st.tick, st.progress, target_id, distance_km,
                       detection_km),
                )

    # ---- (3) + (4) both bands, over the whole tolerated bracket ----------------
    bracket_indices = [
        i for i in range(
            event_index - CERTIFICATE_TICK_TOLERANCE,
            event_index + CERTIFICATE_TICK_TOLERANCE + 1,
        )
        if 0 <= i < len(states)
    ]
    nominal: Optional[Dict[str, Any]] = None
    band_margin_fuel = math.inf
    leg_km_per_tick = 0.0
    for i in bracket_indices:
        st = states[i]
        here = interpolate_great_circle(launch_point, first_target, st.progress)
        fuel_before = float(fuel_at_launch) - st.tick * fuel_per_tick
        try:
            window = measure_window(
                position=here, route=points, home_base=home_base,
                speed_knots=speed_knots, fuel_rate=fuel_rate, margin=margin,
            )
        except FuelDamageError as exc:
            return None, (REASON_INVALID_BAND, str(exc))

        severe_width = window.continue_fuel_requirement - window.rtb_fuel_floor
        mild_width = fuel_before - window.continue_fuel_requirement
        band_margin_fuel = min(band_margin_fuel, severe_width, mild_width)
        leg_km_per_tick = max(leg_km_per_tick, st.leg_km)
        if severe_width <= fuel_per_tick:
            return None, (
                REASON_INVALID_BAND,
                "ego %s: at tick %d the SEVERE band is %.6f lbs wide, not more than one "
                "tick of burn (%.6f lbs); the two ends are within the engine's own "
                "quantization of each other. RTB floor %.3f over %.1f km, continue "
                "requirement %.3f over %.1f km."
                % (ego_id, st.tick, severe_width, fuel_per_tick, window.rtb_fuel_floor,
                   window.rtb_distance_km, window.continue_fuel_requirement,
                   window.continue_distance_km),
            )
        if mild_width <= fuel_per_tick:
            return None, (
                REASON_INVALID_BAND,
                "ego %s: at tick %d the MILD band is %.6f lbs wide, not more than one "
                "tick of burn (%.6f lbs); the ego could not lose fuel and still complete "
                "its route. Fuel before %.3f, continue requirement %.3f over %.1f km."
                % (ego_id, st.tick, mild_width, fuel_per_tick, fuel_before,
                   window.continue_fuel_requirement, window.continue_distance_km),
            )

        mild = severity_band(
            window=window, fuel_before=fuel_before, severity=SEVERITY_MILD
        )
        severe = severity_band(
            window=window, fuel_before=fuel_before, severity=SEVERITY_SEVERE
        )
        try:
            _require_valid_band(
                mild, ego_id=str(ego_id), fuel_before=fuel_before, window=window,
                where="certified",
            )
            _require_valid_band(
                severe, ego_id=str(ego_id), fuel_before=fuel_before, window=window,
                where="certified",
            )
        except FuelDamageError as exc:
            return None, (REASON_INVALID_BAND, str(exc))

        if i == event_index:
            nominal = {
                "state": st, "here": here, "fuel_before": fuel_before,
                "window": window, "mild": mild, "severe": severe,
            }

    assert nominal is not None  # event_index is always inside its own bracket
    st = nominal["state"]
    here = nominal["here"]
    window = nominal["window"]
    mild = nominal["mild"]
    severe = nominal["severe"]

    certificate = FdEventCertificate(
        ego_id=str(ego_id),
        event_tick=int(st.tick),
        movement_count=int(st.movements),
        progress=float(st.progress),
        latitude=float(here.latitude),
        longitude=float(here.longitude),
        fuel_before=float(nominal["fuel_before"]),
        fuel_per_tick=float(fuel_per_tick),
        leg_length_km=float(leg_length_km),
        leg_km_per_tick=float(leg_km_per_tick),
        route_length=len(points),
        first_target_latitude=float(first_target.latitude),
        first_target_longitude=float(first_target.longitude),
        rtb_distance_km=float(window.rtb_distance_km),
        continue_distance_km=float(window.continue_distance_km),
        rtb_fuel_floor=float(window.rtb_fuel_floor),
        continue_fuel_requirement=float(window.continue_fuel_requirement),
        mild_band_low=float(mild.low),
        mild_band_high=float(mild.high),
        mild_target=float(mild.target),
        severe_band_low=float(severe.low),
        severe_band_high=float(severe.high),
        severe_target=float(severe.target),
        band_margin_fuel=float(band_margin_fuel),
        required_band_margin_fuel=float(fuel_per_tick),
        tick_tolerance=int(CERTIFICATE_TICK_TOLERANCE),
        # Both tolerances are ONE QUANTUM PER TOLERATED TICK plus a documented float
        # epsilon -- never a round number chosen to make a check pass.
        fuel_tolerance=float(
            CERTIFICATE_TICK_TOLERANCE * fuel_per_tick
            + _CERTIFICATE_FUEL_RELATIVE_EPS * abs(float(fuel_at_launch))
        ),
        position_tolerance_km=float(
            CERTIFICATE_TICK_TOLERANCE * leg_km_per_tick + _CERTIFICATE_POSITION_EPS_KM
        ),
        bracket_ticks=tuple(int(states[i].tick) for i in bracket_indices),
        pre_event_ticks_checked=int(event_index),
        detection_km=float(detection_km),
    )
    return certificate, None


# =============================================================================
# 3. The plan: plain, testable data describing the scheduled event
# =============================================================================

@dataclass(frozen=True)
class FuelDamagePlan:
    """Everything decided about one episode's fuel-damage event BEFORE it runs.

    Frozen and scalar-only (plus the two id/coordinate fields) so it can be asserted on
    field by field, recorded verbatim in a jsonl record, and compared between two runs of
    the same seed. A CLEAN plan carries ``condition == "clean"`` and ``None`` for every
    damaged-only field -- never a zero, which would read as a measured quantity.

    Attributes:
        condition: ``clean`` or ``damaged``.
        mode: the :class:`FuelDamageMode` the condition was resolved under.
        derived_seed: :func:`derive_fuel_damage_seed` of the episode seed -- the private
            rng domain, recorded so the draw is reproducible from the artifact alone.
        eligible_ego_ids: UNDER THE LEGACY POLICY, the egos the selection drew from
            (non-empty initial routes), in the SORTED order the draw used. UNDER
            ``certified_both_severities_v1`` the bounded walk stops at its first
            ACCEPTED candidate, so this carries that one certified ego and the full
            candidate history lives in ``eligibility_audit`` -- which is where a reader
            must look to see who else was considered and why they were rejected.
        ego_id: the selected ego, ``None`` when clean.
        leg_index: which planned leg the event sits on (always :data:`EVENT_LEG_INDEX`
            in v1; recorded rather than implied so a later variant is legible).
        progress_threshold: the fraction of that leg at which the event fires.
        rtb_safety_margin: the reserve multiplier both window ends were built with.
            Recorded on the PLAN because the controller re-measures the window at the
            live state and must use the identical margin to do it.
        leg_length_km: great-circle length of the first planned leg.
        route_points: the PREDICTED route as ``((lat, lon), ...)`` in flown order --
            leg 1's endpoint first. Carried in full, not just leg 1, because the window
            has to be RE-MEASURED against the live aircraft when the event actually fires
            (see :meth:`FuelDamageController.live_bounds`): the planned bounds describe
            the point the ego was projected to be at, and a projection is not a
            measurement.
        home_base_latitude / home_base_longitude: where "return to base" resolves to --
            the point the RTB leg is measured to, planned and live alike.
        event_latitude / event_longitude: the PLANNED event point. The planned window is
            computed here; the point the event really fired at is measured separately
            (see :class:`FuelDamageOutcome`).
        rtb_distance_km: PLANNED event point -> home base, great circle.
        continue_distance_km: PLANNED event point -> every remaining planned target in
            predicted order -> home base.
        rtb_fuel_floor: ``margin * fuel(rtb_distance_km)`` -- the LOW end of the PLANNED
            window. The live counterpart is ``FuelDamageOutcome.live_rtb_fuel_floor``.
        continue_fuel_requirement: ``margin * fuel(continue_distance_km)`` -- the HIGH,
            exclusive end of the PLANNED window.
        projected_fuel_at_event: fuel the ego is PROJECTED to hold when the event fires,
            i.e. its launch fuel minus what the flown part of leg 1 costs. It is an
            underestimate of consumption by construction -- the engine also burns
            ``fuel_rate / 3600`` on ticks where the aircraft does not move (the launch
            tick has no route yet) -- which is exactly why the live re-check exists.
        post_damage_fuel: the PLANNED post-damage fuel. Under the LEGACY design this is
            the value the live aircraft is set to -- the midpoint of the planned strict
            window, re-validated against the live one before it is applied. Under
            FD-VARIABLE-SEVERITY-v1 it is a PROJECTION only: the midpoint of the planned
            SEVERITY band, which the preflight feasibility check is made against, while
            the value really applied is derived at the event tick from the LIVE band.
            ``target_policy`` says which of the two this field is, so a reader never has
            to infer it from the mode.
        severity: ``mild`` / ``severe`` under a variable-severity mode; ``None`` for a
            clean episode AND for every LEGACY damaged episode. ``None`` is a real
            statement -- "this episode carries no severity label" -- and is deliberately
            not spelled ``severe`` for the legacy design even though the legacy band and
            the severe band coincide, so a legacy record can never be re-read as a
            variable-severity one.
        severity_derived_seed: :func:`derive_fuel_damage_severity_seed` of the episode
            seed, recorded so the severity draw is reproducible from the artifact alone.
            ``None`` under a legacy mode, which never consults that domain.
        mild_probability: P(mild | damaged) the severity was drawn with; ``None`` under a
            legacy mode.
        target_policy: :data:`TARGET_POLICY_PLANNED_MIDPOINT` or
            :data:`TARGET_POLICY_LIVE_SEVERITY_MIDPOINT`.
        planned_band_low / planned_band_high: the PLANNED severity band the projected
            target was taken from, recorded next to it so the projection can be checked
            without re-deriving the window.
        speed_knots / fuel_rate / max_fuel / fuel_at_launch: the live aircraft parameters
            the arithmetic used, recorded so a window can be re-derived from the record.
    """

    condition: str
    mode: str
    derived_seed: int
    eligible_ego_ids: Tuple[str, ...]

    ego_id: Optional[str] = None
    leg_index: Optional[int] = None
    progress_threshold: Optional[float] = None
    rtb_safety_margin: Optional[float] = None
    leg_length_km: Optional[float] = None
    route_points: Tuple[Tuple[float, float], ...] = ()
    home_base_latitude: Optional[float] = None
    home_base_longitude: Optional[float] = None
    event_latitude: Optional[float] = None
    event_longitude: Optional[float] = None
    rtb_distance_km: Optional[float] = None
    continue_distance_km: Optional[float] = None
    rtb_fuel_floor: Optional[float] = None
    continue_fuel_requirement: Optional[float] = None
    projected_fuel_at_event: Optional[float] = None
    post_damage_fuel: Optional[float] = None
    speed_knots: Optional[float] = None
    fuel_rate: Optional[float] = None
    max_fuel: Optional[float] = None
    fuel_at_launch: Optional[float] = None

    # --- FD-VARIABLE-SEVERITY-v1 (all None under a LEGACY mode) ---
    severity: Optional[str] = None
    severity_derived_seed: Optional[int] = None
    mild_probability: Optional[float] = None
    target_policy: str = TARGET_POLICY_PLANNED_MIDPOINT
    planned_band_low: Optional[float] = None
    planned_band_high: Optional[float] = None

    # --- GENERALIZED-V1 step 2 ---
    # Both policies are recorded on EVERY plan, clean and damaged alike, so a reader
    # never has to infer which design produced a record. `eligibility_audit` and
    # `certificate` are populated ONLY under `certified_both_severities_v1`; under the
    # legacy default they stay None, which is a real statement ("this plan made no
    # certified promise") rather than an empty one.
    eligibility_policy: str = FD_ELIGIBILITY_LEGACY_V1
    post_fd_wake_policy: str = POST_FD_WAKE_SINGLE_V1
    eligibility_audit: Optional[FdEligibilityAudit] = None
    certificate: Optional[FdEventCertificate] = None

    @property
    def is_damaged(self) -> bool:
        return self.condition == CONDITION_DAMAGED

    @property
    def is_variable_severity(self) -> bool:
        """True iff the applied target is derived from the LIVE band, not the plan."""
        return self.target_policy == TARGET_POLICY_LIVE_SEVERITY_MIDPOINT

    @property
    def is_certified(self) -> bool:
        """True iff this plan carries a GENERALIZED-V1 event certificate.

        The predicate the controller branches on at fire time: a certified plan's live
        validation failures are INSTRUMENT failures
        (:class:`FuelDamageIntegrityError`), a legacy plan's are ordinary accounted
        scientific attrition (:class:`FuelDamageError`). Reads the POLICY rather than
        merely "is a certificate attached", so a clean certified plan -- which has an
        audit and a counterfactual certificate but nothing to fire -- still answers the
        question about its own design.
        """
        return self.eligibility_policy == FD_ELIGIBILITY_CERTIFIED_V1

    @property
    def completion_boundary_wakes(self) -> bool:
        """True iff the ACTUALLY damaged ego wakes again at its own completions."""
        return self.post_fd_wake_policy == POST_FD_WAKE_COMPLETION_BOUNDARY_V1

    @property
    def event_location(self) -> Optional[Location]:
        """The planned event point as a :class:`Location` (``None`` when clean)."""
        if self.event_latitude is None or self.event_longitude is None:
            return None
        return Location(float(self.event_latitude), float(self.event_longitude))

    @property
    def route_locations(self) -> Tuple[Location, ...]:
        """The predicted route as :class:`Location` objects, in flown order."""
        return tuple(Location(float(lat), float(lon)) for lat, lon in self.route_points)

    @property
    def first_target_location(self) -> Optional[Location]:
        """Leg 1's endpoint -- the first predicted target (``None`` when clean)."""
        route = self.route_locations
        return route[0] if route else None

    @property
    def home_base_location(self) -> Optional[Location]:
        """Where RTB resolves to (``None`` when clean)."""
        if self.home_base_latitude is None or self.home_base_longitude is None:
            return None
        return Location(
            float(self.home_base_latitude), float(self.home_base_longitude)
        )

    def to_record(self) -> Dict[str, Any]:
        """The plan as plain JSON scalars (``None`` stays ``null``, never ``0``)."""
        return {
            "condition": str(self.condition),
            "mode": str(self.mode),
            "derived_seed": int(self.derived_seed),
            "eligible_ego_ids": list(self.eligible_ego_ids),
            "ego_id": self.ego_id,
            "leg_index": self.leg_index,
            "progress_threshold": self.progress_threshold,
            "rtb_safety_margin": self.rtb_safety_margin,
            "leg_length_km": self.leg_length_km,
            "route_points": [list(p) for p in self.route_points],
            "home_base_latitude": self.home_base_latitude,
            "home_base_longitude": self.home_base_longitude,
            "event_latitude": self.event_latitude,
            "event_longitude": self.event_longitude,
            "rtb_distance_km": self.rtb_distance_km,
            "continue_distance_km": self.continue_distance_km,
            "rtb_fuel_floor": self.rtb_fuel_floor,
            "continue_fuel_requirement": self.continue_fuel_requirement,
            "projected_fuel_at_event": self.projected_fuel_at_event,
            "post_damage_fuel": self.post_damage_fuel,
            "speed_knots": self.speed_knots,
            "fuel_rate": self.fuel_rate,
            "max_fuel": self.max_fuel,
            "fuel_at_launch": self.fuel_at_launch,
            # --- FD-VARIABLE-SEVERITY-v1 ---
            "severity": self.severity,
            "severity_derived_seed": self.severity_derived_seed,
            "mild_probability": self.mild_probability,
            "target_policy": str(self.target_policy),
            "planned_band_low": self.planned_band_low,
            "planned_band_high": self.planned_band_high,
            # --- GENERALIZED-V1 step 2 ---
            "eligibility_policy": str(self.eligibility_policy),
            "post_fd_wake_policy": str(self.post_fd_wake_policy),
            "eligibility_audit": (
                None if self.eligibility_audit is None
                else self.eligibility_audit.to_record()
            ),
            "certificate": (
                None if self.certificate is None else self.certificate.to_record()
            ),
        }


@dataclass(frozen=True)
class FuelDamageOutcome:
    """What the event actually DID -- measured, not planned.

    Every damaged-only field is ``None`` until the event fires, and stays ``None`` on a
    clean episode and on a damaged episode whose ego never reached the threshold (it
    died, was never launched, or diverged from the predicted route). ``fired`` is the one
    boolean that says which of those happened; a zero would not.

    PLANNED VS LIVE BOUNDS ARE NAMED APART, ON PURPOSE. ``FuelDamagePlan.rtb_fuel_floor``
    and ``continue_fuel_requirement`` are the window computed BEFORE the run, at the
    point the ego was projected to reach. The ``live_*`` fields here are the window
    re-measured AT the event, from the aircraft's actual position and actual fuel, and
    they are the bounds the mutation was really validated against. They differ because a
    projection is not a measurement: the engine burns ``fuel_rate / 3600`` every tick
    including ones where the aircraft has no route yet, so live fuel at the event is
    always somewhat below ``projected_fuel_at_event``.

    ``rtb_command_issued`` is COMMAND HISTORY, not executor state. It is True iff
    ``run_episode`` really emitted ``aircraft_return_to_base('<ego>')`` in a Phase-2
    command list. It deliberately does NOT read ``GraphPlanExecutor.rtb_issued``, which
    is a lifecycle LATCH: that flag is also set True for a DEAD ego precisely because no
    command was (or could be) emitted, so reusing it would report an ego that crashed
    flying its plan as having returned to base.
    """

    condition: str
    ego_id: Optional[str]
    fired: bool
    event_tick: Optional[int] = None
    observed_progress: Optional[float] = None
    observed_latitude: Optional[float] = None
    observed_longitude: Optional[float] = None
    fuel_before: Optional[float] = None
    fuel_after: Optional[float] = None
    damage_factor: Optional[float] = None
    # The window as re-measured AT the event (see the class docstring).
    live_rtb_distance_km: Optional[float] = None
    live_continue_distance_km: Optional[float] = None
    live_rtb_fuel_floor: Optional[float] = None
    live_continue_fuel_requirement: Optional[float] = None
    wake_occurred: bool = False
    wake_meta_action: Optional[int] = None
    rtb_command_issued: Optional[bool] = None
    # --- FD-VARIABLE-SEVERITY-v1 ---
    # `severity` is the plan's, repeated here so the outcome record stands alone; the
    # two live band ends are what the applied target was really taken from under the
    # variable design, and are `None` under the legacy one (whose target came from the
    # plan and was only VALIDATED live).
    severity: Optional[str] = None
    live_band_low: Optional[float] = None
    live_band_high: Optional[float] = None
    max_fuel: Optional[float] = None

    @property
    def continuation_margin(self) -> Optional[float]:
        """``fuel_after - live_continue_fuel_requirement``, or ``None``.

        THE quantity that separates the two severities physically, in the units the
        engine burns: POSITIVE means the ego can still complete its route and return
        with the reserve (mild), NEGATIVE means it cannot (severe). Derived here, at one
        site, rather than by every reader subtracting two recorded fields -- a reader who
        subtracted the PLANNED requirement instead would get a number that looks right
        and describes a state the episode was never in.
        """
        if self.fuel_after is None or self.live_continue_fuel_requirement is None:
            return None
        return float(self.fuel_after) - float(self.live_continue_fuel_requirement)

    @property
    def fuel_after_fraction_of_max(self) -> Optional[float]:
        """``fuel_after / max_fuel`` -- the tank state as the graph's own ``fuel_norm``.

        ``damage_factor`` is ``fuel_after / fuel_before``, which says how much of the
        tank the EVENT took; this says how full the tank IS afterwards, which is the
        quantity the policy actually observes (``graph_builder._compute_fuel_norm``
        normalizes by ``max_fuel``). Both are reported because they answer different
        questions and neither can be derived from the other without ``fuel_before``.
        """
        if self.fuel_after is None or not self.max_fuel:
            return None
        max_fuel = float(self.max_fuel)
        if not math.isfinite(max_fuel) or max_fuel <= _EPS:
            return None
        return float(self.fuel_after) / max_fuel

    def to_record(self) -> Dict[str, Any]:
        """The outcome as plain JSON scalars."""
        return {
            "condition": str(self.condition),
            "ego_id": self.ego_id,
            "fired": bool(self.fired),
            "event_tick": self.event_tick,
            "observed_progress": self.observed_progress,
            "observed_latitude": self.observed_latitude,
            "observed_longitude": self.observed_longitude,
            "fuel_before": self.fuel_before,
            "fuel_after": self.fuel_after,
            "damage_factor": self.damage_factor,
            "live_rtb_distance_km": self.live_rtb_distance_km,
            "live_continue_distance_km": self.live_continue_distance_km,
            "live_rtb_fuel_floor": self.live_rtb_fuel_floor,
            "live_continue_fuel_requirement": self.live_continue_fuel_requirement,
            "wake_occurred": bool(self.wake_occurred),
            "wake_meta_action": self.wake_meta_action,
            "rtb_command_issued": self.rtb_command_issued,
            # --- FD-VARIABLE-SEVERITY-v1 ---
            "severity": self.severity,
            "live_band_low": self.live_band_low,
            "live_band_high": self.live_band_high,
            "max_fuel": self.max_fuel,
            "continuation_margin": self.continuation_margin,
            "fuel_after_fraction_of_max": self.fuel_after_fraction_of_max,
        }


# =============================================================================
# 4. Planning (pure arithmetic -- no BLADE object in sight)
# =============================================================================

def plan_fuel_damage(
    *,
    condition: str,
    mode: str,
    derived_seed: int,
    eligible_ego_ids: Sequence[str],
    ego_id: Optional[str],
    launch_point: Optional[Location],
    home_base: Optional[Location],
    route_points: Optional[Sequence[Location]],
    speed_knots: Optional[float],
    fuel_rate: Optional[float],
    max_fuel: Optional[float],
    fuel_at_launch: Optional[float],
    params: FuelDamageParameters,
    severity: Optional[str] = None,
    severity_derived_seed: Optional[int] = None,
    eligibility_audit: Optional[FdEligibilityAudit] = None,
    certificate: Optional[FdEventCertificate] = None,
) -> FuelDamagePlan:
    """Build (and VALIDATE) the strict fuel window for one episode.

    Pure: every input is a plain number, a :class:`Location` or an id string, so the whole
    window arithmetic is testable without an engine, a solver or a scenario.

    A CLEAN condition returns immediately with a clean plan -- no window is computed, so a
    forced-clean evaluation member can never fail because a window would have been
    unavailable (an approved requirement: the two members of a matched pair must fail
    independently or not at all).

    For a DAMAGED condition the four validated facts are, in order:

      1. the first planned leg has positive length (otherwise there is no route to place
         an event on);
      2. ``continue_fuel_requirement > rtb_fuel_floor`` -- the window is a non-empty OPEN
         interval. It normally is by the triangle inequality (flying the rest of the route
         and then home is at least as far as flying straight home), but a degenerate
         collinear route could collapse it, and a collapsed window has no midpoint to
         choose;
      3. ``projected_fuel_at_event >= continue_fuel_requirement`` -- the ego COULD have
         continued before the damage. Without this the episode would not contain a
         decision at all: the plan was already infeasible and the event changed nothing.
         MILD needs this STRICTLY (its band is bounded above by the pre-damage fuel and
         below by the continue requirement, so an equality would leave no interior), and
         :func:`_require_valid_band`'s width check is what enforces that;
      4. the chosen ``post_damage_fuel`` really lies inside its severity's own band, is
         strictly below the projected fuel, and stays at or above the RTB floor
         (:func:`_require_valid_band`).

    The chosen value is the MIDPOINT of the band -- the approved deterministic choice,
    and the one furthest from both ends, so neither bound is decided by floating-point
    noise.

    ``severity`` selects the band (:func:`severity_band`). ``None`` is the LEGACY
    FD-BASELINE-v1 design and produces byte-identical arithmetic to the pre-severity
    code: the same interval, the same midpoint, the same four checks. Under a
    variable-severity mode the value computed here is a PROJECTION used for the preflight
    feasibility verdict, and the value really applied is re-derived from the LIVE band at
    the event tick -- ``target_policy`` records which of the two this plan carries.

    ``certificate`` (GENERALIZED-V1 step 2) replaces the DISTANCE projection with the
    TICK-AWARE one :func:`certify_fd_candidate` already validated: the event point and
    the pre-damage fuel come from the certificate instead of from
    ``interpolate_great_circle`` at the threshold fraction and
    ``launch_fuel - fuel(fraction * leg)``. Everything after that -- the window, the
    three checks, the band and its midpoint -- is the SAME code on a different pair of
    inputs. ``certificate is None`` (the default) is the legacy projection, byte-for-byte.

    Raises:
        FuelDamageError: if any of the four facts does not hold, or if a required input
            is missing for a damaged condition. Never returns a clean plan instead, and
            never substitutes the other severity's band.
    """
    params.validate()
    if condition not in CONDITIONS:
        raise FuelDamageError("condition must be one of %r, got %r" % (list(CONDITIONS), condition))
    if severity is not None and severity not in SEVERITIES:
        raise FuelDamageError(
            "severity must be one of %r or None, got %r" % (list(SEVERITIES), severity)
        )
    # A severity is a property of a DAMAGED episode. Labelling a clean one would make
    # "which cell is this episode in?" answerable two contradictory ways.
    if condition == CONDITION_CLEAN and severity is not None:
        raise FuelDamageError(
            "a clean episode cannot carry a severity, got %r" % (severity,)
        )
    if params.variable_severity and condition == CONDITION_DAMAGED and severity is None:
        raise FuelDamageError(
            "mode %r is a variable-severity mode, so a damaged episode must carry a "
            "severity; none was supplied" % (params.mode,)
        )

    eligible = tuple(str(e) for e in eligible_ego_ids)
    if condition == CONDITION_CLEAN:
        # `severity_derived_seed` is recorded on a CLEAN plan too, for exactly the reason
        # `derived_seed` already is: it identifies the private domain the episode's draws
        # came from, and an artifact that names it can be reproduced without knowing
        # which branch the episode took.
        return FuelDamagePlan(
            condition=CONDITION_CLEAN,
            mode=str(mode),
            derived_seed=int(derived_seed),
            eligible_ego_ids=eligible,
            severity=None,
            severity_derived_seed=(
                None if severity_derived_seed is None else int(severity_derived_seed)
            ),
            mild_probability=(
                float(params.mild_probability) if params.variable_severity else None
            ),
            target_policy=params.target_policy,
            eligibility_policy=params.eligibility_policy,
            post_fd_wake_policy=params.post_fd_wake_policy,
            # A CLEAN certified plan still carries the walk and the counterfactual
            # certificate: that is what makes "the three members of this matched group
            # share one accepted-world support" checkable from the clean member's own
            # artifact. `ego_id` stays None -- nothing is damaged here.
            eligibility_audit=eligibility_audit,
            certificate=certificate,
        )

    # ---- required inputs for a damaged plan --------------------------------------
    if not ego_id:
        raise FuelDamageError(
            "a damaged episode needs a selected ego, but none was supplied "
            "(eligible=%r)" % (list(eligible),)
        )
    if launch_point is None or home_base is None:
        raise FuelDamageError(
            "ego %s: a damaged episode needs both a launch point and a home base to "
            "measure the window against" % ego_id
        )
    points = list(route_points or [])
    if not points:
        raise FuelDamageError(
            "ego %s: the predicted route is empty, so there is no first leg to place the "
            "event on; only egos with a non-empty initial route are eligible" % ego_id
        )
    if speed_knots is None or fuel_rate is None:
        raise FuelDamageError(
            "ego %s: the window must be measured with the ENGINE's own speed (%r kt) and "
            "fuel_rate (%r lbs/hr); neither may be missing, and neither may be taken "
            "from Agent (scenario_factory substitutes a 250 kt planning speed for a "
            "grounded unit)" % (ego_id, speed_knots, fuel_rate)
        )

    fraction = float(params.leg_progress_threshold)
    margin = float(params.rtb_safety_margin)

    first_target = points[0]
    leg_length_km = float(launch_point.distance_to(first_target))
    if not math.isfinite(leg_length_km) or leg_length_km <= _EPS:
        raise FuelDamageError(
            "ego %s: the first planned leg has length %r km; an event cannot be placed "
            "at %.0f%% of a zero-length leg" % (ego_id, leg_length_km, 100.0 * fraction)
        )

    if certificate is None:
        # LEGACY: the point at `fraction` of leg 1 by DISTANCE.
        event_point = interpolate_great_circle(launch_point, first_target, fraction)
    else:
        if str(certificate.ego_id) != str(ego_id):
            raise FuelDamageError(
                "ego %s: was handed a certificate issued for ego %s; a certificate is a "
                "promise about ONE ego's own flight and cannot be transferred"
                % (ego_id, certificate.ego_id)
            )
        # CERTIFIED: the point the ego is really predicted to be at on the tick the
        # threshold is really observed (see `certify_fd_candidate`).
        event_point = certificate.event_location

    # The PLANNED window, measured from the projected event point through the same one
    # site the live re-check uses. `predict_route` is the frozen structural reproduction
    # of the executor's ordering, reused rather than reimplemented, so the requirement
    # cannot be measured against a route the ego does not fly.
    window = measure_window(
        position=event_point, route=points, home_base=home_base,
        speed_knots=speed_knots, fuel_rate=fuel_rate, margin=margin,
    )
    rtb_distance_km = window.rtb_distance_km
    continue_distance_km = window.continue_distance_km
    rtb_fuel_floor = window.rtb_fuel_floor
    continue_fuel_requirement = window.continue_fuel_requirement

    def _fuel(distance_km: float) -> float:
        return fuel_for_distance_km(
            distance_km, speed_knots=speed_knots, fuel_rate=fuel_rate
        )

    if fuel_at_launch is None:
        raise FuelDamageError("ego %s: launch fuel is unknown" % ego_id)
    launch_fuel = float(fuel_at_launch)
    if not math.isfinite(launch_fuel) or launch_fuel <= 0.0:
        raise FuelDamageError(
            "ego %s: launch fuel must be finite and > 0, got %r" % (ego_id, fuel_at_launch)
        )
    # A PROJECTION, and knowingly an optimistic one: the engine also burns
    # `fuel_rate / 3600` on ticks where the aircraft does not move (the launch tick has
    # no route yet), so live fuel at the event is always somewhat below this. It is the
    # right quantity for the PREFLIGHT premise -- "the plan was feasible before the
    # damage" -- and it is why the controller re-measures everything against the live
    # aircraft before it mutates anything.
    if certificate is None:
        projected_fuel_at_event = launch_fuel - _fuel(fraction * leg_length_km)
    else:
        # The certificate already charged the engine's REAL burn: one tick per airborne
        # tick including the route-less launch tick, which is precisely the optimism the
        # legacy projection has to leave to the live re-check.
        projected_fuel_at_event = float(certificate.fuel_before)

    # (2) the window must be a non-empty OPEN interval.
    if not (continue_fuel_requirement > rtb_fuel_floor + _EPS):
        raise FuelDamageError(
            "ego %s: no strict fuel window -- the continue requirement (%.3f) does not "
            "exceed the RTB floor (%.3f), so there is no quantity that makes returning "
            "feasible and continuing infeasible. Route: %.1f km remaining+return vs "
            "%.1f km direct home."
            % (ego_id, continue_fuel_requirement, rtb_fuel_floor,
               continue_distance_km, rtb_distance_km)
        )

    # (3) the pre-damage fuel must have been sufficient for the continuing requirement.
    if projected_fuel_at_event < continue_fuel_requirement:
        raise FuelDamageError(
            "ego %s: no strict fuel window -- the projected fuel at the event (%.3f) is "
            "already below the continue requirement (%.3f), so the plan was infeasible "
            "before any damage and the event would create no decision."
            % (ego_id, projected_fuel_at_event, continue_fuel_requirement)
        )

    # (4) the severity's own band, and the midpoint it yields. `severity is None` is the
    # LEGACY interval `[rtb_floor, continue_req)`, so this is the unchanged legacy
    # arithmetic; a variable-severity mode selects mild or severe instead.
    band = severity_band(
        window=window, fuel_before=projected_fuel_at_event, severity=severity
    )
    post_damage_fuel = _require_valid_band(
        band, ego_id=str(ego_id), fuel_before=projected_fuel_at_event,
        window=window, where="planned",
    )

    return FuelDamagePlan(
        condition=CONDITION_DAMAGED,
        mode=str(mode),
        derived_seed=int(derived_seed),
        eligible_ego_ids=eligible,
        ego_id=str(ego_id),
        leg_index=EVENT_LEG_INDEX,
        progress_threshold=fraction,
        rtb_safety_margin=margin,
        leg_length_km=leg_length_km,
        # The WHOLE predicted route and the home base, not just leg 1: the controller
        # re-measures the window against the live aircraft before it mutates anything,
        # and it can only do that if the plan carries the geometry to re-measure with.
        route_points=tuple(
            (float(p.latitude), float(p.longitude)) for p in points
        ),
        home_base_latitude=float(home_base.latitude),
        home_base_longitude=float(home_base.longitude),
        event_latitude=float(event_point.latitude),
        event_longitude=float(event_point.longitude),
        rtb_distance_km=rtb_distance_km,
        continue_distance_km=continue_distance_km,
        rtb_fuel_floor=float(rtb_fuel_floor),
        continue_fuel_requirement=float(continue_fuel_requirement),
        projected_fuel_at_event=float(projected_fuel_at_event),
        post_damage_fuel=float(post_damage_fuel),
        speed_knots=float(speed_knots),
        fuel_rate=float(fuel_rate),
        max_fuel=None if max_fuel is None else float(max_fuel),
        fuel_at_launch=launch_fuel,
        severity=severity,
        severity_derived_seed=(
            None if severity_derived_seed is None else int(severity_derived_seed)
        ),
        mild_probability=(
            float(params.mild_probability) if params.variable_severity else None
        ),
        target_policy=params.target_policy,
        planned_band_low=float(band.low),
        planned_band_high=float(band.high),
        eligibility_policy=params.eligibility_policy,
        post_fd_wake_policy=params.post_fd_wake_policy,
        eligibility_audit=eligibility_audit,
        certificate=certificate,
    )


# =============================================================================
# 4b. GENERALIZED-V1 step 2: post-FD adaptation records
# =============================================================================

@dataclass(frozen=True)
class PostFdBoundary:
    """ONE locally CONFIRMED assignment-completion boundary of the damaged ego.

    A boundary exists the moment the DAMAGED EGO ITSELF confirms, inside its OWN
    arrival/sensor radius, that an assigned target is gone -- the same proximity-gated
    confirmation the executor has always used as its sole ``done`` signal. That is why
    a peer's kill far away is not a boundary and an emitted attack is not a boundary:
    neither is something this ego observed.

    ``remaining_mission`` is what separates a DECISION POINT from the end of the
    mission, and the two are counted apart rather than inferred from the wake count: a
    boundary with no remaining assignments deliberately produces NO wake (there is
    nothing left to decide, and the executor's existing empty-plan path issues the
    single latched RTB), so "boundary without a wake" and "boundary that failed to wake"
    must not be the same number.
    """

    tick: int
    confirmed_target_ids: Tuple[str, ...]
    remaining_mission: bool
    wake_occurred: bool = False
    meta_action: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "tick": int(self.tick),
            "confirmed_target_ids": [str(t) for t in self.confirmed_target_ids],
            "remaining_mission": bool(self.remaining_mission),
            "wake_occurred": bool(self.wake_occurred),
            "meta_action": self.meta_action,
        }


@dataclass(frozen=True)
class PostFdAdaptationOutcome:
    """What persistent post-FD adaptation actually DID, as frozen data.

    Deliberately a SEPARATE object from :class:`FuelDamageOutcome` rather than more
    fields on it. ``FuelDamageOutcome.wake_occurred`` / ``wake_meta_action`` mean the
    IMMEDIATE fuel-damage wake and must keep meaning exactly that -- an approved
    measurement is reported over them -- so the later boundary decisions are counted
    here, under their own denominator, and never folded into that pair.

    Task 4 owns run-level persistence and aggregation of these numbers; Task 2 only has
    to make them exist and be truthful.
    """

    policy: str
    ego_id: Optional[str]
    armed: bool
    active: bool
    deactivation_reason: Optional[str]
    boundaries: Tuple[PostFdBoundary, ...]

    @property
    def boundaries_confirmed(self) -> int:
        """Every locally confirmed completion boundary, terminal ones included."""
        return len(self.boundaries)

    @property
    def boundaries_with_remaining_mission(self) -> int:
        """Boundaries that were real decision OPPORTUNITIES (work was still left)."""
        return sum(1 for b in self.boundaries if b.remaining_mission)

    @property
    def boundaries_terminal(self) -> int:
        """Boundaries that ended the mission, and therefore correctly woke nobody."""
        return sum(1 for b in self.boundaries if not b.remaining_mission)

    @property
    def boundary_wakes(self) -> int:
        """Boundaries that really produced an actor decision."""
        return sum(1 for b in self.boundaries if b.wake_occurred)

    @property
    def boundary_ticks(self) -> Tuple[int, ...]:
        return tuple(int(b.tick) for b in self.boundaries)

    @property
    def boundary_meta_actions(self) -> Tuple[int, ...]:
        """The meta-action of every boundary that woke, in boundary order."""
        return tuple(
            int(b.meta_action) for b in self.boundaries
            if b.wake_occurred and b.meta_action is not None
        )

    def to_record(self) -> Dict[str, Any]:
        """The adaptation record as plain JSON scalars (``None`` stays ``null``)."""
        return {
            "policy": str(self.policy),
            "ego_id": self.ego_id,
            "armed": bool(self.armed),
            "active": bool(self.active),
            "deactivation_reason": self.deactivation_reason,
            "boundaries_confirmed": int(self.boundaries_confirmed),
            "boundaries_with_remaining_mission": int(
                self.boundaries_with_remaining_mission
            ),
            "boundaries_terminal": int(self.boundaries_terminal),
            "boundary_wakes": int(self.boundary_wakes),
            "boundary_ticks": [int(t) for t in self.boundary_ticks],
            "boundary_meta_actions": [int(m) for m in self.boundary_meta_actions],
            "boundaries": [b.to_record() for b in self.boundaries],
        }


# =============================================================================
# 5. The runtime controller (the ONLY place that mutates BLADE)
# =============================================================================

def rtb_command_for(ego_id: str) -> str:
    """The exact BLADE command string a return-to-base for ``ego_id`` is emitted as.

    A MIRROR of the ONE site that emits it, ``GraphPlanExecutor._rtb_or_latch``'s
    ``f"aircraft_return_to_base('{ego_id}')"``. It is a second copy of a format string,
    and the equivalence is TEST-ENFORCED against a command list produced by a real
    ``GraphPlanExecutor`` -- the same discipline ``graph_train.derived_split`` uses to
    mirror ``split_tasks``. The alternative, importing the executor here, would drag the
    BLADE translation layer into this module's closure and cost it the purity that makes
    it hand-testable.

    WHY MATCH THE COMMAND AT ALL, rather than reading ``executor.rtb_issued``: that flag
    is a LIFECYCLE LATCH, not command history. ``_command_for_ego`` sets it True for a
    DEAD ego specifically because no command was emitted (and none could be), so an ego
    that crashed while flying its plan would be reported as having returned to base --
    and, on a fuel-damage episode, counted as both an RTB and a death.
    """
    return "aircraft_return_to_base('%s')" % ego_id


def _find_live_aircraft(scenario: Any, ego_id: str) -> Optional[Any]:
    """The ego's AIRBORNE aircraft object, or ``None``.

    Scans ``scenario.aircraft`` only -- deliberately NOT airbase inventories. A grounded
    ego has not started its leg, and a dead one has been removed by the engine; in both
    cases there is nothing to damage and the event simply has not fired yet.
    """
    for aircraft in getattr(scenario, "aircraft", []) or []:
        if str(getattr(aircraft, "id", "")) == str(ego_id):
            return aircraft
    return None


class FuelDamageController:
    """Applies ONE episode's fuel-damage event, at most once, and records what happened.

    Owns no plan arithmetic (that is :func:`plan_fuel_damage`) and no policy logic. Its
    entire job is: watch the selected ego's progress along its first leg at the START of
    each tick, mutate the live aircraft's ``current_fuel`` exactly once when the threshold
    is crossed, and tell the caller which ego (if any) must be woken with
    ``TriggerKind.FUEL_DAMAGE``.

    A CLEAN plan makes every method a no-op, so the tick loop has ONE code path.
    """

    def __init__(self, plan: FuelDamagePlan) -> None:
        self.plan = plan
        # Leg 1's endpoint, taken from the plan: runtime progress is `(L - distance to
        # it) / L`. Holding it as plan DATA rather than as controller state is what makes
        # the firing condition reproducible from a recorded plan alone.
        self._first_target: Optional[Location] = plan.first_target_location
        if plan.is_damaged and self._first_target is None:
            raise FuelDamageError(
                "ego %s: a damaged plan must carry leg 1's endpoint, without which "
                "runtime progress cannot be measured" % plan.ego_id
            )
        self._route: Tuple[Location, ...] = plan.route_locations
        self._home_base: Optional[Location] = plan.home_base_location
        if plan.is_damaged and (not self._route or self._home_base is None):
            raise FuelDamageError(
                "ego %s: a damaged plan must carry its predicted route and home base, "
                "without which the window cannot be re-measured against the live "
                "aircraft before the mutation" % plan.ego_id
            )
        self._fired = False
        self._event_tick: Optional[int] = None
        self._observed_progress: Optional[float] = None
        self._observed_lat: Optional[float] = None
        self._observed_lon: Optional[float] = None
        self._fuel_before: Optional[float] = None
        self._fuel_after: Optional[float] = None
        self._live_window: Optional[_Window] = None
        self._live_band: Optional[_SeverityBand] = None
        self._wake_occurred = False
        self._wake_meta_action: Optional[int] = None
        # COMMAND HISTORY, not the executor's lifecycle latch: None until the episode
        # runs (and forever on a clean episode, which has no selected ego), then True iff
        # `run_episode` really emitted this ego's `aircraft_return_to_base`.
        self._rtb_command_issued: Optional[bool] = (
            False if plan.is_damaged else None
        )
        # GENERALIZED-V1 step 2: persistent post-FD adaptation. ARMED by the real fuel
        # mutation and by nothing else -- a CLEAN counterfactual certificate arms
        # nothing, because no ego was damaged, and no peer can ever enter the state.
        self._post_fd_active = False
        self._post_fd_deactivation_reason: Optional[str] = None
        self._boundaries: List[Dict[str, Any]] = []

    # ---- runtime ---------------------------------------------------------------

    @property
    def fired(self) -> bool:
        """True once the event has been applied. Never resets -- the event is one-shot."""
        return self._fired

    def observed_progress(self, scenario: Any) -> Optional[float]:
        """The selected ego's fraction of the first planned leg covered, or ``None``.

        Measured as ``(L - remaining) / L`` where ``remaining`` is the great-circle
        distance from the ego's LIVE position to the first planned target. That is
        monotone as the ego closes on the target and needs no along-track projection,
        which matters because the ego flies a direct ``move_aircraft`` route to exactly
        that point.

        ``None`` when the ego is not airborne (still in its airbase inventory, or removed
        after running out of fuel) or when the plan is clean.
        """
        if not self.plan.is_damaged or self.plan.leg_length_km is None:
            return None
        aircraft = _find_live_aircraft(scenario, str(self.plan.ego_id))
        if aircraft is None:
            return None
        return self._progress_of(aircraft)

    def _progress_of(self, aircraft: Any) -> Optional[float]:
        leg_km = float(self.plan.leg_length_km or 0.0)
        if leg_km <= _EPS or self._first_target is None:
            return None
        here = Location(
            float(getattr(aircraft, "latitude", 0.0)),
            float(getattr(aircraft, "longitude", 0.0)),
        )
        remaining_km = float(here.distance_to(self._first_target))
        return (leg_km - remaining_km) / leg_km

    def live_bounds(self, aircraft: Any) -> _Window:
        """Re-measure the strict window from the aircraft's LIVE position.

        The plan's bounds were measured at the point the ego was PROJECTED to reach at
        the threshold. The point it actually reaches differs -- tick granularity means it
        crosses the threshold slightly beyond it -- and the fuel it actually holds
        differs more: ``Game.update_all_aircraft_position`` burns ``fuel_rate / 3600``
        EVERY tick including ones with no route, and the launch tick has no route yet, so
        ``projected_fuel_at_event`` is an optimistic estimate by construction. Validating
        the mutation against the planned numbers would therefore be validating it against
        a state the episode is not in.

        Uses the same :func:`measure_window` site as the plan, with the same route, home
        base, speed and fuel_rate, so the two are directly comparable.
        """
        here = Location(
            float(getattr(aircraft, "latitude", 0.0)),
            float(getattr(aircraft, "longitude", 0.0)),
        )
        return measure_window(
            position=here,
            route=self._route,
            home_base=self._home_base,
            speed_knots=float(self.plan.speed_knots or 0.0),
            fuel_rate=float(self.plan.fuel_rate or 0.0),
            margin=float(self.plan.rtb_safety_margin or 0.0),
        )

    def _live_legacy_target(
        self, *, ego_id: str, tick: int, fuel_before: float, live: _Window
    ) -> float:
        """LEGACY FD-BASELINE-v1's live validation. BYTE-FOR-BYTE the merged behaviour.

        Lifted out of :meth:`maybe_apply` unchanged -- the same four checks, in the same
        ORDER, with the same messages. The order is part of the contract, not an
        implementation detail: which check fires first is what an accounted ``run``-stage
        failure reports, and reordering them would silently relabel a class of already
        measured failures.
        """
        target = float(self.plan.post_damage_fuel or 0.0)
        # (1) the mutation must be a LOSS. Implied by (2) and (4) together, but checked
        # first and separately because it is the most direct thing that can be wrong and
        # deserves its own message.
        if not (fuel_before > target):
            raise FuelDamageError(
                "ego %s: at tick %d the live fuel (%.3f) is not above the planned "
                "post-damage fuel (%.3f); applying the event would ADD fuel. The "
                "pre-run projection (%.3f) does not describe this run."
                % (ego_id, int(tick), fuel_before, target,
                   float(self.plan.projected_fuel_at_event or 0.0))
            )
        # (2) the ego must really have been able to continue, AT THE LIVE STATE.
        if fuel_before < live.continue_fuel_requirement:
            raise FuelDamageError(
                "ego %s: at tick %d the live fuel (%.3f) is already below the LIVE "
                "continue requirement (%.3f over %.1f km); the plan was infeasible "
                "before any damage, so the event would create no decision. Planned "
                "requirement was %.3f over %.1f km."
                % (ego_id, int(tick), fuel_before, live.continue_fuel_requirement,
                   live.continue_distance_km,
                   float(self.plan.continue_fuel_requirement or 0.0),
                   float(self.plan.continue_distance_km or 0.0))
            )
        # (3) direct RTB must remain feasible with the margin, AT THE LIVE POSITION.
        if target < live.rtb_fuel_floor:
            raise FuelDamageError(
                "ego %s: at tick %d the planned post-damage fuel (%.3f) is below the "
                "LIVE RTB floor (%.3f over %.1f km); the ego could not fly home with the "
                "%.2f reserve, so the event would be a kill rather than a decision."
                % (ego_id, int(tick), target, live.rtb_fuel_floor,
                   live.rtb_distance_km, float(self.plan.rtb_safety_margin or 0.0))
            )
        # (4) continuation + return must be infeasible, AT THE LIVE STATE.
        if not (target < live.continue_fuel_requirement):
            raise FuelDamageError(
                "ego %s: at tick %d the planned post-damage fuel (%.3f) is not below the "
                "LIVE continue requirement (%.3f over %.1f km); the ego could still "
                "complete its route and return, so the event would create no decision."
                % (ego_id, int(tick), target, live.continue_fuel_requirement,
                   live.continue_distance_km)
            )
        return target

    def _live_variable_target(
        self, *, ego_id: str, tick: int, fuel_before: float, live: _Window,
        live_band: _SeverityBand,
    ) -> float:
        """FD-VARIABLE-SEVERITY-v1's live derivation: the LIVE band's own midpoint.

        The premise is checked FIRST and separately: the ego must really have been able
        to complete its route and return had the event not happened. For SEVERE that is
        the same statement the legacy design makes; for MILD it is additionally what
        makes the band ``(continue_requirement, fuel_before)`` non-empty, so a violation
        has to be reported as the missing premise it is rather than as an empty interval
        the reader would then have to diagnose.

        Everything after it is :func:`_require_valid_band`, which is also what the plan
        was validated with -- one site for both measurements of the same band.
        """
        if fuel_before < live.continue_fuel_requirement:
            raise FuelDamageError(
                "ego %s: at tick %d the live fuel (%.3f) is already below the LIVE "
                "continue requirement (%.3f over %.1f km); the plan was infeasible "
                "before any damage, so a %s event would create no decision. Planned "
                "requirement was %.3f over %.1f km."
                % (ego_id, int(tick), fuel_before, live.continue_fuel_requirement,
                   live.continue_distance_km, str(self.plan.severity),
                   float(self.plan.continue_fuel_requirement or 0.0),
                   float(self.plan.continue_distance_km or 0.0))
            )
        return _require_valid_band(
            live_band, ego_id=ego_id, fuel_before=fuel_before,
            window=live, where="live",
        )

    def _live_target(
        self, *, ego_id: str, tick: int, fuel_before: float, live: _Window,
        live_band: _SeverityBand,
    ) -> float:
        """Dispatch to the design's own live check. Neither branch changed."""
        if self.plan.is_variable_severity:
            return self._live_variable_target(
                ego_id=ego_id, tick=tick, fuel_before=fuel_before,
                live=live, live_band=live_band,
            )
        return self._live_legacy_target(
            ego_id=ego_id, tick=tick, fuel_before=fuel_before, live=live,
        )

    def _require_certificate_holds(
        self, *, aircraft: Any, tick: int, fuel_before: float
    ) -> None:
        """GENERALIZED-V1: the live event state must match the promise, or ABORT.

        Under :data:`FD_ELIGIBILITY_CERTIFIED_V1` the world was certified FD-capable
        BEFORE a tick was paid for: a specific tick, a specific point on leg 1 and a
        specific pre-damage fuel, with both severity bands validated across the tolerated
        bracket. If the run then arrives somewhere else, the certifier does not describe
        this simulator -- which makes every episode it touched suspect, not just this one
        -- so this raises :class:`FuelDamageIntegrityError` and the run stops. It is
        emphatically NOT ordinary attrition and is never accounted as an episode failure.

        THE THREE TOLERANCES ARE QUANTA, NOT SLACK. Each is
        :data:`CERTIFICATE_TICK_TOLERANCE` engine quanta -- one tick of burn, one tick of
        travel -- plus a documented float epsilon; see the certificate's own fields.

        Raises:
            FuelDamageIntegrityError: on any of the three mismatches. Raised BEFORE the
                mutation, so a contradicted certificate leaves the engine untouched.
        """
        cert = self.plan.certificate
        if cert is None:
            raise FuelDamageIntegrityError(
                "ego %s: the plan declares the certified eligibility policy (%s) but "
                "carries no certificate, so there is nothing to validate the live event "
                "against" % (str(self.plan.ego_id), self.plan.eligibility_policy)
            )
        ego_id = str(self.plan.ego_id)

        observed_tick = int(tick)
        if abs(observed_tick - int(cert.event_tick)) > int(cert.tick_tolerance):
            raise FuelDamageIntegrityError(
                "ego %s: CERTIFICATE CONTRADICTED -- the event was certified for tick %d "
                "(+/- %d, the engine's own observation quantum) and was observed at tick "
                "%d. The tick-aware prediction does not describe this run."
                % (ego_id, int(cert.event_tick), int(cert.tick_tolerance), observed_tick)
            )

        here = Location(
            float(getattr(aircraft, "latitude", 0.0)),
            float(getattr(aircraft, "longitude", 0.0)),
        )
        offset_km = float(here.distance_to(cert.event_location))
        if offset_km > float(cert.position_tolerance_km):
            raise FuelDamageIntegrityError(
                "ego %s: CERTIFICATE CONTRADICTED -- the event was certified at "
                "(%.6f, %.6f) and fired %.4f km away, beyond the %.4f km tolerance "
                "(%d tick(s) of travel plus float noise)."
                % (ego_id, float(cert.latitude), float(cert.longitude), offset_km,
                   float(cert.position_tolerance_km), int(cert.tick_tolerance))
            )

        fuel_offset = abs(float(fuel_before) - float(cert.fuel_before))
        if fuel_offset > float(cert.fuel_tolerance):
            raise FuelDamageIntegrityError(
                "ego %s: CERTIFICATE CONTRADICTED -- the pre-damage fuel was certified at "
                "%.6f and measured %.6f, a difference of %.6f beyond the %.6f tolerance "
                "(%d tick(s) of burn at %.6f lbs/tick plus float noise)."
                % (ego_id, float(cert.fuel_before), float(fuel_before), fuel_offset,
                   float(cert.fuel_tolerance), int(cert.tick_tolerance),
                   float(cert.fuel_per_tick))
            )

    # ---- persistent post-FD adaptation (GENERALIZED-V1 step 2) ----------------

    @property
    def boundary_wakes_enabled(self) -> bool:
        """True iff this episode's policy grants the damaged ego later boundary wakes.

        A property of the PLAN, so it is answerable before the event fires; it says
        nothing about whether any ego is damaged yet (see :attr:`post_fd_ego`).
        """
        return self.plan.is_damaged and self.plan.completion_boundary_wakes

    @property
    def post_fd_ego(self) -> Optional[str]:
        """The ego in persistent post-FD adaptation state, or ``None``.

        ``None`` until the REAL fuel mutation has happened, and ``None`` forever on a
        clean episode -- a counterfactual certificate arms nothing. The tick loop reads
        this to decide whose completions to reconcile, so it is the single place the
        "only the ACTUALLY damaged ego" rule is enforced.
        """
        if not self._post_fd_active:
            return None
        return None if self.plan.ego_id is None else str(self.plan.ego_id)

    def deactivate_adaptation(self, reason: str) -> None:
        """Stop granting boundary wakes, recording WHY (idempotent, first reason wins).

        Called when the damaged ego commits to return or dies: from either state it can
        no longer reach a completion boundary, and the executor no longer processes it in
        Phase 1 at all.
        """
        if not self._post_fd_active:
            return
        self._post_fd_active = False
        if self._post_fd_deactivation_reason is None:
            self._post_fd_deactivation_reason = str(reason)

    def note_boundary(
        self, *, ego_id: str, tick: int, confirmed_target_ids: Sequence[str],
        remaining_mission: bool,
    ) -> None:
        """Record ONE locally confirmed completion boundary. Measurement only.

        Takes no engine object and performs no plan edit: the confirmation, the belief
        cleanup and the executor resync all happen in the tick loop, which owns the
        executor. This layer stays BLADE-free and only counts.

        Ignored for any ego other than the damaged one, so a mis-wired caller cannot
        attribute a peer's completion to this event.
        """
        if not self._post_fd_active or self.plan.ego_id is None:
            return
        if str(ego_id) != str(self.plan.ego_id):
            return
        self._boundaries.append({
            "tick": int(tick),
            "confirmed_target_ids": tuple(str(t) for t in confirmed_target_ids),
            "remaining_mission": bool(remaining_mission),
            "wake_occurred": False,
            "meta_action": None,
        })

    def note_boundary_wake(
        self, *, ego_id: str, tick: int, meta_action: int
    ) -> None:
        """Attribute a decision to the boundary that caused it, at ``tick``.

        Matched by TICK rather than by "the most recent boundary": the two are the same
        thing today, and the tick makes the record self-describing if a future variant
        ever separates them. A boundary that already carries a decision is never
        overwritten -- one boundary is one decision.
        """
        if self.plan.ego_id is None or str(ego_id) != str(self.plan.ego_id):
            return
        for record in reversed(self._boundaries):
            if int(record["tick"]) == int(tick) and not record["wake_occurred"]:
                record["wake_occurred"] = True
                record["meta_action"] = int(meta_action)
                return

    @property
    def post_fd_outcome(self) -> PostFdAdaptationOutcome:
        """The post-FD adaptation record, as frozen data that outlives the environment."""
        return PostFdAdaptationOutcome(
            policy=str(self.plan.post_fd_wake_policy),
            ego_id=(
                str(self.plan.ego_id)
                if (self._post_fd_active or self._boundaries
                    or self._post_fd_deactivation_reason is not None)
                else None
            ),
            armed=bool(
                self._post_fd_active or self._boundaries
                or self._post_fd_deactivation_reason is not None
            ),
            active=bool(self._post_fd_active),
            deactivation_reason=self._post_fd_deactivation_reason,
            boundaries=tuple(
                PostFdBoundary(
                    tick=int(r["tick"]),
                    confirmed_target_ids=tuple(r["confirmed_target_ids"]),
                    remaining_mission=bool(r["remaining_mission"]),
                    wake_occurred=bool(r["wake_occurred"]),
                    meta_action=r["meta_action"],
                )
                for r in self._boundaries
            ),
        )

    def maybe_apply(self, scenario: Any, tick: int) -> Optional[str]:
        """Apply the event if this is the first tick at or past the threshold.

        CALL ONCE PER TICK, AT THE START OF PHASE 1, BEFORE ANY EGO IS PROCESSED. That
        ordering is not a convenience: every ego must sense and decide against the same
        post-event snapshot, or the outcome would depend on Phase-1 ego iteration order
        and the no-communication guarantee would be gone.

        THE WINDOW IS RE-VALIDATED AGAINST THE LIVE AIRCRAFT BEFORE ANYTHING IS MUTATED
        (see :meth:`live_bounds` for why the planned bounds are not sufficient).

        WHICH TARGET IS APPLIED DEPENDS ON THE DESIGN, and the plan says which:

          * LEGACY FD-BASELINE-v1 (``target_policy == planned_midpoint_v1``) applies the
            PLANNED value and validates it against the live state. UNCHANGED.
          * FD-VARIABLE-SEVERITY-v1 (``live_severity_midpoint_v1``) DERIVES the value
            here, as the midpoint of the severity's band measured at the live state
            (:func:`severity_band`). A mild event has to leave continuation genuinely
            feasible and a severe one has to leave it genuinely infeasible, and both
            statements are about the fuel the ego really holds at the point it really
            reached -- a target fixed before the run could only be checked against that,
            never guaranteed to sit in the right band of it.

        The two designs' live checks live in :meth:`_live_legacy_target` and
        :meth:`_live_variable_target`, kept apart precisely so the legacy CHECK ORDER --
        which decides what an already-measured ``run``-stage failure reports -- cannot be
        disturbed by the new one. Both require the same physical facts: the ego really
        could have continued before the event, the mutation is a real LOSS, and the
        result still affords flying straight home with the engine's reserve.

        Returns:
            The damaged ego's id on the tick the event fires (the caller must wake exactly
            that ego with ``TriggerKind.FUEL_DAMAGE``), and ``None`` on every other tick --
            including every tick of a clean episode and every tick after the event.

        Raises:
            FuelDamageError: if any fact fails at the live state. Raised BEFORE the
                mutation, so a refused event leaves the engine untouched; the attempt is
                then accounted as a ``run``-stage failure by ``skip_and_account_v1``.
                Nothing is clamped, weakened, re-planned, downgraded to the other
                severity, or converted to a clean episode.
        """
        if self._fired or not self.plan.is_damaged:
            return None
        ego_id = str(self.plan.ego_id)
        aircraft = _find_live_aircraft(scenario, ego_id)
        if aircraft is None:
            return None  # not airborne yet, or already removed by the engine
        progress = self._progress_of(aircraft)
        if progress is None or progress < float(self.plan.progress_threshold or 0.0):
            return None

        fuel_before = float(getattr(aircraft, "current_fuel", 0.0))
        live = self.live_bounds(aircraft)
        live_band = severity_band(
            window=live, fuel_before=fuel_before, severity=self.plan.severity
        )

        if self.plan.is_certified:
            # GENERALIZED-V1: this world PROMISED both severities at a predicted event
            # state. Check the promise itself first, then run the SAME live physics as
            # every other episode -- with every failure re-routed as the instrument fault
            # it is, because a certified world that cannot carry its own event is not a
            # scientific outcome (handoff 3l.3).
            self._require_certificate_holds(
                aircraft=aircraft, tick=tick, fuel_before=fuel_before
            )
            try:
                target = self._live_target(
                    ego_id=ego_id, tick=tick, fuel_before=fuel_before,
                    live=live, live_band=live_band,
                )
            except FuelDamageError as exc:
                raise FuelDamageIntegrityError(
                    "ego %s: CERTIFICATE CONTRADICTED at tick %d -- this world was "
                    "certified to support BOTH severities at the predicted event state, "
                    "and the live physics refused the event anyway: %s"
                    % (ego_id, int(tick), exc)
                ) from exc
        else:
            target = self._live_target(
                ego_id=ego_id, tick=tick, fuel_before=fuel_before,
                live=live, live_band=live_band,
            )

        # THE mutation: the real BLADE aircraft, exactly once, in the real fuel units the
        # engine decrements every tick. Not an observation-only simulation -- the engine
        # must be the one that removes the aircraft if the policy flies it dry.
        aircraft.current_fuel = target

        self._fired = True
        self._event_tick = int(tick)
        self._observed_progress = float(progress)
        self._observed_lat = float(getattr(aircraft, "latitude", 0.0))
        self._observed_lon = float(getattr(aircraft, "longitude", 0.0))
        self._fuel_before = fuel_before
        self._fuel_after = target
        self._live_window = live
        # Recorded only for the design that DERIVED its target from this band; under the
        # legacy design the band was a yardstick, not a source, and reporting it as
        # `live_band_*` would suggest the target came from it.
        self._live_band = live_band if self.plan.is_variable_severity else None
        # ARM persistent post-FD adaptation. This is the ONLY site that does it, and it
        # sits AFTER the real mutation on purpose: the state belongs to the ego that
        # actually lost fuel, never to a scheduled one, never to a clean counterfactual
        # and never to a peer.
        if self.plan.completion_boundary_wakes:
            self._post_fd_active = True
        return ego_id

    def require_certified_event_realized(
        self, *, scenario: Any = None, ticks: Optional[int] = None
    ) -> None:
        """THE TERMINAL HALF of the certified promise: the event must actually have FIRED.

        GENERALIZED-V1 (handoff 3l.3). :meth:`maybe_apply` guards the promise from the
        inside -- if the run REACHES the event and the state contradicts the certificate,
        it aborts. This guards it from the outside: if the run ENDS and the event never
        happened at all, a world that was proven FD-capable before a tick was paid for
        failed to deliver its own certificate, and accepting that episode would admit it
        into a scientific population as a successful DAMAGED episode. It is an INSTRUMENT
        fault, exactly like a live contradiction, and it is routed identically.

        THE FOUR CELLS, and only one of them raises:

          * CERTIFIED + DAMAGED + fired  -- returns; the normal successful path.
          * CERTIFIED + DAMAGED + NOT fired -- raises :class:`FuelDamageIntegrityError`.
          * CERTIFIED + CLEAN -- returns. Nothing was scheduled to fire and the
            certificate is a COUNTERFACTUAL (it names the ego the matched mild and severe
            members will damage), so ``fired == False`` is the correct outcome, never a
            fault.
          * LEGACY (either condition) -- returns, ALWAYS. The legacy policy makes no
            certified promise, a damaged episode whose ego never reaches the threshold is
            an ordinary recorded observation there, and an approved measurement contains
            exactly such an episode. Adding a terminal requirement to it would change the
            behaviour that measurement was taken on rather than extend it.

        PURE, AND IT MUTATES NOTHING -- not the engine, not the plan, not this
        controller's own state. It never applies a late event to satisfy itself: a
        certificate that did not materialize is a fact to report, not a state to repair.
        ``scenario`` and ``ticks`` are DIAGNOSTIC ONLY, both optional, and are read only
        to say how far the ego actually got.

        CALL IT ONCE, AT THE EPISODE-EXIT SEAM, before a non-fire can be accepted as a
        result -- ``graph_tick_loop.run_episode`` does exactly that, which is the single
        path every scientific consumer (the trainer and the diagnostic rollout alike)
        goes through. Do not duplicate the predicate across callers.

        Raises:
            FuelDamageIntegrityError: only in the second cell above.
        """
        if not (self.plan.is_certified and self.plan.is_damaged):
            return
        if self._fired:
            return

        cert = self.plan.certificate
        ego_id = str(self.plan.ego_id)
        progress = None if scenario is None else self.observed_progress(scenario)
        reached = (
            "the ego was not airborne at the end of the episode"
            if progress is None
            else "the ego reached %.4f of its first leg" % progress
        )
        raise FuelDamageIntegrityError(
            "ego %s: CERTIFICATE NOT REALIZED -- this world was certified FD-capable at "
            "setup (event scheduled for tick %s at %.1f%% of a %.1f km first leg), the "
            "episode ended after %s tick(s), and the event never fired. %s. A certified "
            "world that does not deliver its own event is an instrument fault, not a "
            "damaged episode that happened to be uneventful, so it is not accepted as a "
            "scientific result."
            % (ego_id,
               "unknown" if cert is None else str(cert.event_tick),
               100.0 * float(self.plan.progress_threshold or 0.0),
               float(self.plan.leg_length_km or 0.0),
               "an unknown number of" if ticks is None else str(int(ticks)),
               reached)
        )

    def note_commands(self, commands: Sequence[str]) -> None:
        """Observe ONE tick's emitted BLADE command list (Phase 2), read-only.

        THE source of ``rtb_command_issued``. It latches True the first time the selected
        ego's :func:`rtb_command_for` string really appears in a command list, which is
        the only evidence that a return-to-base was ACTUALLY ORDERED.

        Deliberately not derived from ``GraphPlanExecutor.rtb_issued``: that is a
        lifecycle latch which ``_command_for_ego`` also sets True for a DEAD ego -- there
        precisely because no command was emitted -- so an ego that flew its plan into the
        ground would otherwise be counted as an RTB *and* as a death.
        """
        if not self.plan.is_damaged or self._rtb_command_issued:
            return
        wanted = rtb_command_for(str(self.plan.ego_id))
        for command in commands or ():
            if str(command) == wanted:
                self._rtb_command_issued = True
                return

    def note_wake(self, *, ego_id: str, meta_action: int) -> None:
        """Record that the fuel-damage wake produced a decision, and which one.

        Called by the tick loop only for the wake the event itself caused, so a later
        organic wake of the same ego cannot overwrite it (``_wake_occurred`` latches).
        """
        if not self.plan.is_damaged or self._wake_occurred:
            return
        if str(ego_id) != str(self.plan.ego_id):
            return
        self._wake_occurred = True
        self._wake_meta_action = int(meta_action)

    # ---- reporting -------------------------------------------------------------

    @property
    def outcome(self) -> FuelDamageOutcome:
        """What actually happened, as frozen data that outlives the environment."""
        factor: Optional[float] = None
        if self._fuel_before is not None and self._fuel_before > _EPS:
            factor = float(self._fuel_after or 0.0) / float(self._fuel_before)
        return FuelDamageOutcome(
            condition=self.plan.condition,
            ego_id=self.plan.ego_id,
            fired=self._fired,
            event_tick=self._event_tick,
            observed_progress=self._observed_progress,
            observed_latitude=self._observed_lat,
            observed_longitude=self._observed_lon,
            fuel_before=self._fuel_before,
            fuel_after=self._fuel_after,
            damage_factor=factor,
            live_rtb_distance_km=(
                None if self._live_window is None
                else self._live_window.rtb_distance_km),
            live_continue_distance_km=(
                None if self._live_window is None
                else self._live_window.continue_distance_km),
            live_rtb_fuel_floor=(
                None if self._live_window is None
                else self._live_window.rtb_fuel_floor),
            live_continue_fuel_requirement=(
                None if self._live_window is None
                else self._live_window.continue_fuel_requirement),
            wake_occurred=self._wake_occurred,
            wake_meta_action=self._wake_meta_action,
            rtb_command_issued=self._rtb_command_issued,
            severity=self.plan.severity,
            live_band_low=(
                None if self._live_band is None else float(self._live_band.low)),
            live_band_high=(
                None if self._live_band is None else float(self._live_band.high)),
            max_fuel=self.plan.max_fuel,
        )


# =============================================================================
# 6. The EpisodeContext adapter (duck-typed -- still no BLADE import)
# =============================================================================

def _task_target_location(task: Task) -> Optional[Location]:
    """The ATTACK step's location for a task (``steps[0]`` fallback), or ``None``."""
    steps = getattr(task, "steps", None) or []
    step = next(
        (s for s in steps if getattr(s, "step_kind", None) == StepKind.ATTACK),
        steps[0] if steps else None,
    )
    return None if step is None else getattr(step, "location", None)


def _task_target_id(task: Task) -> str:
    """The ATTACK step's ``target_id`` for a task (``steps[0]`` fallback), or ``""``.

    A local mirror of ``graph_trigger._task_target_id`` and ``graph_builder``'s attack-
    step lookup, for the same reason those two are mirrors of each other: importing a
    private symbol from a module this layer only consumes would couple them, and the
    rule ("a task IS its attack step's target") is one line.
    """
    steps = getattr(task, "steps", None) or []
    for step in steps:
        if getattr(step, "step_kind", None) == StepKind.ATTACK:
            return str(getattr(step, "target_id", ""))
    return str(getattr(steps[0], "target_id", "")) if steps else ""


def _predicted_route(
    assignments: Sequence[Assignment],
    belief_tasks: Sequence[Task],
    launch_point: Location,
    *,
    ego_id: str,
) -> Tuple[List[Location], List[str]]:
    """The ego's predicted route, as ordered target locations AND their target ids.

    REUSES the frozen ``graph_hidden_placement.predict_route`` -- the structural
    reproduction of the executor's own intra-level nearest-neighbor ordering -- rather
    than reimplementing it, so a window can never be measured against a route the ego
    does not fly. Extracted so the legacy selection and the certified walk share ONE
    route derivation; the error text is the legacy one, unchanged.
    """
    ordered = predict_route(
        [tuple(a) for a in assignments], list(belief_tasks), launch_point
    )
    points: List[Location] = []
    target_ids: List[str] = []
    for assignment in ordered:
        task = belief_tasks[int(assignment[0])]
        loc = _task_target_location(task)
        if loc is None:
            raise FuelDamageError(
                "ego %s: predicted assignment %r resolves to a step with no location"
                % (ego_id, assignment)
            )
        points.append(loc)
        target_ids.append(_task_target_id(task))
    return points, target_ids


def _predicted_route_points(
    assignments: Sequence[Assignment],
    belief_tasks: Sequence[Task],
    launch_point: Location,
    *,
    ego_id: str,
) -> List[Location]:
    """:func:`_predicted_route` without the ids -- the legacy call site's shape."""
    points, _ids = _predicted_route(
        assignments, belief_tasks, launch_point, ego_id=ego_id
    )
    return points


def _eligible_ego_ids(a_init: Dict[str, Sequence[Assignment]]) -> List[str]:
    """Egos with a NON-EMPTY initial route, in sorted id order.

    Sorted so the selection draw does not depend on the solution dict's insertion order --
    the same rule B2's ``place_hidden_targets`` follows, and for the same reason.
    """
    return sorted(str(k) for k, v in (a_init or {}).items() if v)


def _find_aircraft_anywhere(scenario: Any, ego_id: str) -> Optional[Any]:
    """The ego's aircraft object whether it is airborne or still in its airbase.

    Used at PLAN time (t = 0), when every ego is still in the BLUE base's inventory and
    ``scenario.aircraft`` is therefore empty.
    """
    live = _find_live_aircraft(scenario, ego_id)
    if live is not None:
        return live
    for base in getattr(scenario, "airbases", []) or []:
        for aircraft in getattr(base, "aircraft", []) or []:
            if str(getattr(aircraft, "id", "")) == str(ego_id):
                return aircraft
    return None


def _certified_eligibility_walk(
    ctx: Any, *, episode_seed: int, params: FuelDamageParameters
) -> FdEligibilityAudit:
    """GENERALIZED-V1: the deterministic BOUNDED walk that certifies ONE ego.

    IT RUNS FOR EVERY CONDITION -- CLEAN INCLUDED -- and depends on the EPISODE SEED
    ALONE. That is the whole mechanism behind handoff 3l.3's "every accepted generated
    world must be FD-capable": clean, mild and severe members of the same world+seed walk
    the same candidates in the same order and certify the same ego, so the three share
    one accepted-world support and a matched group is constructible by design rather
    than discovered afterwards. A clean episode simply never applies the mutation.

    THE CANDIDATE POPULATION IS ``ctx.agent_ids`` -- the AUTHORITATIVE scheduled agent
    sequence -- and a candidate's ORDINAL is its index there. Egos the allocated-only
    ``A_init`` omitted entirely are INCLUDED and then rejected truthfully as
    :data:`REASON_NO_ROUTE`, rather than being invisible: "this ego had nothing to fly"
    is a finding, and a population that silently excluded it could not report it.
    Ordering NEVER derives from id text (``CLAUDE.md`` Sec 8: generated ids are not
    seed-derived).

    The walk is BOUNDED: at most ``len(agent_ids)`` candidates, each attempted at most
    once, stopping at the FIRST acceptance. Nothing is retried, no seed is redrawn, no
    world is replaced, no severity is changed and no episode is converted to clean.

    Raises:
        FuelDamageError: when the context cannot supply what the walk needs, and --
            with the stable machine-readable marker :data:`NO_FD_ELIGIBLE_EGO` -- when
            every candidate was considered and none could be certified. That is a
            NORMAL setup rejection, accounted exactly like a B2 cardinality failure, and
            deliberately not the integrity exception.
    """
    agent_ids = [str(a) for a in (getattr(ctx, "agent_ids", None) or [])]
    if not agent_ids:
        raise FuelDamageError(
            "the certified eligibility policy needs the scheduled agent order "
            "(ctx.agent_ids); it is empty, so there are no ordinals to walk"
        )
    if len(set(agent_ids)) != len(agent_ids):
        raise FuelDamageError(
            "ctx.agent_ids holds duplicate ids %r; an ordinal would not address exactly "
            "one ego" % (agent_ids,)
        )

    executor = getattr(ctx, "executor", None)
    detection_km = getattr(executor, "arrival_threshold_km", None)
    if detection_km is None:
        raise FuelDamageError(
            "the certified eligibility policy needs the ONE unified arrival/detection "
            "radius from the executor; ctx.executor.arrival_threshold_km is unavailable, "
            "and inventing a second radius here would let the certificate disagree with "
            "the runtime sensor"
        )
    detection_km = float(detection_km)

    scenario = getattr(getattr(ctx, "game", None), "current_scenario", None)
    if scenario is None:
        raise FuelDamageError(
            "the certified eligibility policy needs the live scenario to read the "
            "engine's own speed / fuel_rate and the world's target inventory"
        )

    a_init = dict(getattr(ctx, "a_init", None) or {})
    beliefs = dict(getattr(ctx, "beliefs", None) or {})
    agents_by_id = {
        str(getattr(a, "id", "")): a for a in (getattr(ctx, "agents", None) or [])
    }

    eligibility_seed = derive_fuel_damage_eligibility_seed(episode_seed)
    order = eligibility_ordinal_permutation(
        len(agent_ids), random.Random(eligibility_seed)
    )

    candidates: List[FdEligibilityCandidate] = []
    considered: List[int] = []
    selected_ordinal: Optional[int] = None
    certificate: Optional[FdEventCertificate] = None

    for ordinal in order:
        considered.append(ordinal)
        ego_id = agent_ids[ordinal]

        raw = a_init.get(ego_id)
        if not raw:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=REASON_NO_ROUTE,
                detail="ego %s has no assigned route in the known-only A_init" % ego_id,
            ))
            continue

        belief = beliefs.get(ego_id)
        belief_tasks = list(getattr(belief, "tasks", None) or [])
        if not belief_tasks:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=REASON_ROUTE_UNRESOLVABLE,
                detail="ego %s has no t=0 belief tasks to predict a route from" % ego_id,
            ))
            continue

        agent = agents_by_id.get(ego_id)
        launch_point = None if agent is None else getattr(agent, "location", None)
        if agent is None or launch_point is None:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=REASON_ROUTE_UNRESOLVABLE,
                detail=("ego %s has no MATCH-AOU Agent with a launch location in the "
                        "episode context" % ego_id),
            ))
            continue
        # `setup_episode` VERIFIES `Agent.location == Agent.return_location` on the
        # construction path; the return location is still preferred because it is what
        # BLADE resolves the home base to.
        home_base = getattr(agent, "return_location", None) or launch_point

        try:
            route_points, route_target_ids = _predicted_route(
                raw, belief_tasks, launch_point, ego_id=ego_id
            )
        except FuelDamageError as exc:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=REASON_ROUTE_UNRESOLVABLE, detail=str(exc),
            ))
            continue
        if not route_points:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=REASON_NO_ROUTE,
                detail="ego %s has an assignment slice that predicts no route" % ego_id,
            ))
            continue
        first_target_id = route_target_ids[0] or None

        aircraft = _find_aircraft_anywhere(scenario, ego_id)
        if aircraft is None:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=REASON_NO_AIRCRAFT,
                detail=("ego %s has no live BLADE aircraft; the certificate must be "
                        "measured against the ENGINE's own speed / fuel_rate, never "
                        "against the solver's planning speed" % ego_id),
                route_length=len(route_points),
                first_target_id=first_target_id,
            ))
            continue

        cert, rejection = certify_fd_candidate(
            ego_id=ego_id,
            launch_point=launch_point,
            route_points=route_points,
            home_base=home_base,
            speed_knots=getattr(aircraft, "speed", None),
            fuel_rate=getattr(aircraft, "fuel_rate", None),
            fuel_at_launch=getattr(aircraft, "current_fuel", None),
            params=params,
            detection_km=detection_km,
            # CURRENT WORLD TRUTH, for a SETUP-ONLY certificate. None of it reaches
            # `GraphObservation`, the actor, the critic or any runtime decision -- it is
            # read once, here, to decide whether this world is ACCEPTED.
            world_targets=list(
                iter_enemy_targets(scenario, getattr(agent, "side_color", None))
            ),
            belief_target_ids=[_task_target_id(t) for t in belief_tasks],
        )
        if rejection is not None:
            candidates.append(FdEligibilityCandidate(
                ordinal=ordinal, ego_id=ego_id, accepted=False,
                reason=rejection[0], detail=rejection[1],
                route_length=len(route_points),
                first_target_id=first_target_id,
            ))
            continue

        selected_ordinal = ordinal
        certificate = cert
        candidates.append(FdEligibilityCandidate(
            ordinal=ordinal, ego_id=ego_id, accepted=True,
            route_length=len(route_points),
            first_target_id=first_target_id,
        ))
        break  # BOUNDED: the walk stops at its first acceptance

    audit = FdEligibilityAudit(
        policy=FD_ELIGIBILITY_CERTIFIED_V1,
        rng_domain=FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN,
        derived_seed=int(eligibility_seed),
        candidate_count=len(agent_ids),
        candidate_order=order,
        considered_ordinals=tuple(considered),
        candidates=tuple(candidates),
        selected_ordinal=selected_ordinal,
        selected_ego_id=(
            None if selected_ordinal is None else agent_ids[selected_ordinal]
        ),
        certificate=certificate,
    )
    if selected_ordinal is None:
        reasons = "; ".join(
            "ordinal %d (%s): %s" % (c.ordinal, c.ego_id, c.reason) for c in candidates
        ) or "no candidate was even considered"
        raise FuelDamageError(
            "%s: no ego could be certified for BOTH severities from %d candidate(s). "
            "Candidate outcomes: %s"
            % (NO_FD_ELIGIBLE_EGO, len(considered), reasons)
        )
    return audit


def _build_certified_plan(
    ctx: Any,
    *,
    episode_seed: int,
    params: FuelDamageParameters,
    condition: str,
    severity: Optional[str],
    derived_seed: int,
    severity_derived_seed: Optional[int],
) -> FuelDamagePlan:
    """Run the certified walk, then build the plan the certificate describes."""
    audit = _certified_eligibility_walk(
        ctx, episode_seed=episode_seed, params=params
    )
    certificate = audit.certificate
    assert certificate is not None  # the walk raises rather than accepting without one
    ego_id = str(audit.selected_ego_id)

    if condition == CONDITION_CLEAN:
        # The walk still ran, and the audit still names the ego the matched mild and
        # severe members will damage -- but `ego_id` stays None, because a clean episode
        # damages nobody and that field means what it always has.
        return plan_fuel_damage(
            condition=CONDITION_CLEAN, mode=params.mode, derived_seed=derived_seed,
            eligible_ego_ids=(ego_id,), ego_id=None, launch_point=None,
            home_base=None, route_points=None, speed_knots=None, fuel_rate=None,
            max_fuel=None, fuel_at_launch=None, params=params,
            severity=None, severity_derived_seed=severity_derived_seed,
            eligibility_audit=audit, certificate=certificate,
        )

    agent = next(
        (a for a in (getattr(ctx, "agents", None) or [])
         if str(getattr(a, "id", "")) == ego_id),
        None,
    )
    if agent is None:  # pragma: no cover - the walk already resolved it
        raise FuelDamageError(
            "ego %s: certified, then not found among the episode's agents" % ego_id
        )
    launch_point = getattr(agent, "location", None)
    home_base = getattr(agent, "return_location", None) or launch_point
    belief = (getattr(ctx, "beliefs", None) or {}).get(ego_id)
    belief_tasks = list(getattr(belief, "tasks", None) or [])
    route_points = _predicted_route_points(
        dict(getattr(ctx, "a_init", None) or {}).get(ego_id, []),
        belief_tasks, launch_point, ego_id=ego_id,
    )
    scenario = getattr(getattr(ctx, "game", None), "current_scenario", None)
    aircraft = None if scenario is None else _find_aircraft_anywhere(scenario, ego_id)
    if aircraft is None:  # pragma: no cover - the walk already resolved it
        raise FuelDamageError(
            "ego %s: certified, then no live BLADE aircraft to damage" % ego_id
        )

    return plan_fuel_damage(
        condition=CONDITION_DAMAGED,
        mode=params.mode,
        derived_seed=derived_seed,
        eligible_ego_ids=(ego_id,),
        ego_id=ego_id,
        launch_point=launch_point,
        home_base=home_base,
        route_points=route_points,
        speed_knots=getattr(aircraft, "speed", None),
        fuel_rate=getattr(aircraft, "fuel_rate", None),
        max_fuel=getattr(aircraft, "max_fuel", None),
        fuel_at_launch=getattr(aircraft, "current_fuel", None),
        params=params,
        severity=severity,
        severity_derived_seed=severity_derived_seed,
        eligibility_audit=audit,
        certificate=certificate,
    )


def build_fuel_damage_plan(
    ctx: Any, *, episode_seed: int, params: FuelDamageParameters
) -> FuelDamagePlan:
    """Resolve one episode's condition, select the ego, and build its window.

    CALL AFTER ``setup_episode`` AND BEFORE ``run_episode``. Both halves matter: the
    predicted route comes from ``ctx.a_init`` against the t=0 belief tasks (which diverge
    per ego afterwards -- that divergence IS the no-communication guarantee), and the
    launch fuel is read off an aircraft that has not burned a tick yet.

    Reads six things from the context, all duck-typed: ``a_init``, ``beliefs``,
    ``agents``, and (through ``game.current_scenario``) the live aircraft. It imports
    nothing from ``graph_episode_setup``, so the dependency runs one way only.

    Returns:
        The :class:`FuelDamagePlan`. It carries leg 1's endpoint, so the controller needs
        nothing from this function beyond the plan itself.

    Raises:
        FuelDamageError: on a damaged episode with no eligible ego, an unresolvable
            aircraft, or no valid strict window (see :func:`plan_fuel_damage`).
    """
    params.validate()
    condition = resolve_condition(episode_seed=episode_seed, params=params)
    severity = resolve_severity(episode_seed=episode_seed, params=params)
    derived_seed = derive_fuel_damage_seed(episode_seed)
    # Recorded only when it was really consulted: a legacy plan carrying a severity seed
    # would claim a draw that never happened.
    severity_derived_seed = (
        derive_fuel_damage_severity_seed(episode_seed)
        if params.variable_severity else None
    )

    if params.certified_eligibility:
        # GENERALIZED-V1: eligibility is a WORLD precondition, so the walk runs for
        # EVERY condition -- clean included -- before the branch below is even reached.
        return _build_certified_plan(
            ctx, episode_seed=episode_seed, params=params, condition=condition,
            severity=severity, derived_seed=derived_seed,
            severity_derived_seed=severity_derived_seed,
        )

    a_init = dict(getattr(ctx, "a_init", None) or {})
    eligible = _eligible_ego_ids(a_init)

    if condition == CONDITION_CLEAN:
        return plan_fuel_damage(
            condition=CONDITION_CLEAN, mode=params.mode, derived_seed=derived_seed,
            eligible_ego_ids=eligible, ego_id=None, launch_point=None,
            home_base=None, route_points=None, speed_knots=None, fuel_rate=None,
            max_fuel=None, fuel_at_launch=None, params=params,
            severity=None, severity_derived_seed=severity_derived_seed,
        )

    if not eligible:
        raise FuelDamageError(
            "a damaged episode was scheduled but NO ego has a non-empty initial route, "
            "so there is nothing to damage; the episode is not silently downgraded to "
            "clean"
        )

    # The mixture bit is consumed first so the stream position matches `resolve_condition`
    # (see the module docstring): forced-damaged and seeded-damaged pick the same ego.
    rng = _fuel_damage_rng(episode_seed)
    rng.random()
    ego_id = str(rng.choice(eligible))

    belief = (getattr(ctx, "beliefs", None) or {}).get(ego_id)
    belief_tasks = list(getattr(belief, "tasks", None) or [])
    if not belief_tasks:
        raise FuelDamageError("ego %s: no t=0 belief tasks to predict a route from" % ego_id)

    agent = next(
        (a for a in (getattr(ctx, "agents", None) or [])
         if str(getattr(a, "id", "")) == ego_id),
        None,
    )
    if agent is None:
        raise FuelDamageError("ego %s: no MATCH-AOU Agent in the episode context" % ego_id)
    launch_point = getattr(agent, "location", None)
    if launch_point is None:
        raise FuelDamageError("ego %s: the agent has no launch location" % ego_id)
    # `setup_episode` VERIFIES `Agent.location == Agent.return_location` on the
    # construction path (`_shared_launch_point`), so these coincide; the return location
    # is still preferred because it is what BLADE resolves the home base to.
    home_base = getattr(agent, "return_location", None) or launch_point

    route_points = _predicted_route_points(
        a_init.get(ego_id, []), belief_tasks, launch_point, ego_id=ego_id
    )
    if not route_points:
        raise FuelDamageError(
            "ego %s: was selected as eligible but its predicted route is empty" % ego_id
        )

    scenario = getattr(getattr(ctx, "game", None), "current_scenario", None)
    aircraft = None if scenario is None else _find_aircraft_anywhere(scenario, ego_id)
    if aircraft is None:
        raise FuelDamageError(
            "ego %s: no live BLADE aircraft in the scenario; the window must be measured "
            "against the engine's own speed / fuel_rate, never against the solver's "
            "planning speed" % ego_id
        )

    return plan_fuel_damage(
        condition=CONDITION_DAMAGED,
        mode=params.mode,
        derived_seed=derived_seed,
        eligible_ego_ids=eligible,
        ego_id=ego_id,
        launch_point=launch_point,
        home_base=home_base,
        route_points=route_points,
        speed_knots=getattr(aircraft, "speed", None),
        fuel_rate=getattr(aircraft, "fuel_rate", None),
        max_fuel=getattr(aircraft, "max_fuel", None),
        fuel_at_launch=getattr(aircraft, "current_fuel", None),
        params=params,
        severity=severity,
        severity_derived_seed=severity_derived_seed,
    )


def build_fuel_damage_controller(
    ctx: Any, *, episode_seed: int, params: FuelDamageParameters
) -> FuelDamageController:
    """:func:`build_fuel_damage_plan` plus a ready-to-run :class:`FuelDamageController`.

    Always returns a controller, including for a clean episode -- a clean controller is a
    no-op on every call, so the tick loop keeps ONE code path and the observability record
    exists for both conditions.
    """
    return FuelDamageController(
        build_fuel_damage_plan(ctx, episode_seed=episode_seed, params=params)
    )
