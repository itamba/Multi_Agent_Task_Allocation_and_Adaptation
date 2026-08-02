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
``aircraft_penalty_coeff``) and SELF_PRESERVATION_ABORT (drop the assignment, let the
executor's existing empty-plan branch issue its single RTB, and keep the airframe at the
cost of the target). Neither branch is forced anywhere: the policy still decides.

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

DETERMINISM AND THE RNG DOMAIN
------------------------------
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
Everything fails LOUDLY as :class:`FuelDamageError`. In particular, a scheduled DAMAGED
episode with no valid strict window is NOT quietly downgraded to a clean episode -- that
would silently change the population every measurement is reported over. It raises, the
trainer wraps it as an ``EpisodeAttemptError("setup", ...)``, and ``skip_and_account_v1``
records it exactly once with no retry, no substitution and no band shift. A FORCED-CLEAN
episode never computes a window at all, so it can never fail for this reason.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...models import Location, StepKind, Task
from .graph_hidden_placement import predict_route

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level_order)

__all__ = [
    "CONDITION_CLEAN",
    "CONDITION_DAMAGED",
    "CONDITIONS",
    "FUEL_DAMAGE_RNG_DOMAIN",
    "FuelDamageController",
    "FuelDamageError",
    "FuelDamageMode",
    "FuelDamageOutcome",
    "FuelDamageParameters",
    "FuelDamagePlan",
    "build_fuel_damage_controller",
    "build_fuel_damage_plan",
    "derive_fuel_damage_seed",
    "fuel_for_distance_km",
    "plan_fuel_damage",
    "resolve_condition",
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

# The private rng domain string (see the module docstring). Bump the suffix if the draw
# ORDER or the derivation ever changes -- the point of a versioned domain is that a run's
# clean/damaged assignment can be reproduced from its seed alone, forever.
FUEL_DAMAGE_RNG_DOMAIN = "fuel_damage_v1"

# The leg the event is scheduled on. 1-based, matching `graph_hidden_placement._Leg.index`.
EVENT_LEG_INDEX = 1

CONDITION_CLEAN = "clean"
CONDITION_DAMAGED = "damaged"
CONDITIONS = (CONDITION_CLEAN, CONDITION_DAMAGED)

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


class FuelDamageMode:
    """How an episode's clean/damaged condition is decided. A closed set of strings.

    Strings rather than an enum so a mode round-trips through ``run_config.json`` and a
    jsonl record as itself, with no decoding step between the artifact and the reader.

      * ``off``            -- the factor is disabled entirely; every episode is clean and
                              no controller is built. This is the pre-FD behaviour.
      * ``seeded_mixture`` -- TRAINING. The condition is a deterministic function of the
                              episode seed (see :func:`resolve_condition`).
      * ``forced_clean``   -- EVALUATION, member A of a matched pair.
      * ``forced_damaged`` -- EVALUATION, member B of a matched pair. Same generator seed
                              and same placement seed as member A, so the two run the
                              SAME world and differ only in the event.
    """

    OFF = "off"
    SEEDED_MIXTURE = "seeded_mixture"
    FORCED_CLEAN = "forced_clean"
    FORCED_DAMAGED = "forced_damaged"

    ALL = (OFF, SEEDED_MIXTURE, FORCED_CLEAN, FORCED_DAMAGED)


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
    """

    mode: str = FuelDamageMode.SEEDED_MIXTURE
    probability: float = DEFAULT_DAMAGE_PROBABILITY
    leg_progress_threshold: float = DEFAULT_LEG_PROGRESS_THRESHOLD
    rtb_safety_margin: float = DEFAULT_RTB_SAFETY_MARGIN

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

    @property
    def enabled(self) -> bool:
        """False only for :attr:`FuelDamageMode.OFF` -- the pre-FD behaviour."""
        return self.mode != FuelDamageMode.OFF

    def to_record(self) -> Dict[str, Any]:
        """The parameter set as plain JSON scalars, for ``run_config.json`` / records."""
        return {
            "mode": str(self.mode),
            "probability": float(self.probability),
            "leg_progress_threshold": float(self.leg_progress_threshold),
            "rtb_safety_margin": float(self.rtb_safety_margin),
            "rng_domain": FUEL_DAMAGE_RNG_DOMAIN,
            "event_leg_index": EVENT_LEG_INDEX,
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


def _fuel_damage_rng(episode_seed: int) -> random.Random:
    """A fresh, private :class:`random.Random` for one episode seed."""
    return random.Random(derive_fuel_damage_seed(episode_seed))


def resolve_condition(*, episode_seed: int, params: FuelDamageParameters) -> str:
    """The SCHEDULED condition of an episode -- ``clean`` or ``damaged``.

    PURE, and deliberately independent of the world: the trainer needs the condition
    before an episode is built (to schedule it) and after it has failed (to account for
    it by condition), and neither of those has a context to inspect.

    The mixture draw is taken as the FIRST value of the private rng in every mode,
    including the forced ones. It is discarded there, but taking it keeps the stream
    position identical, so :func:`plan_fuel_damage`'s ego selection is the same draw
    whether the episode was scheduled by the mixture or forced by an evaluation pair.
    """
    params.validate()
    if params.mode == FuelDamageMode.OFF:
        return CONDITION_CLEAN
    rng = _fuel_damage_rng(episode_seed)
    drawn = rng.random() < float(params.probability)
    if params.mode == FuelDamageMode.FORCED_CLEAN:
        return CONDITION_CLEAN
    if params.mode == FuelDamageMode.FORCED_DAMAGED:
        return CONDITION_DAMAGED
    return CONDITION_DAMAGED if drawn else CONDITION_CLEAN


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
        eligible_ego_ids: the egos the selection drew from (non-empty initial routes),
            in the SORTED order the draw used.
        ego_id: the selected ego, ``None`` when clean.
        leg_index: which planned leg the event sits on (always :data:`EVENT_LEG_INDEX`
            in v1; recorded rather than implied so a later variant is legible).
        progress_threshold: the fraction of that leg at which the event fires.
        leg_length_km: great-circle length of the first planned leg.
        first_target_latitude / first_target_longitude: leg 1's ENDPOINT -- the first
            target of the predicted route. Recorded because runtime progress is measured
            against it (``(L - distance_to_it) / L``), so a plan fully determines the
            firing condition without any hidden controller state.
        event_latitude / event_longitude: the PLANNED event point. The window is computed
            here; the runtime point is measured separately (see :class:`FuelDamageOutcome`).
        rtb_distance_km: event point -> home base, great circle.
        continue_distance_km: event point -> every remaining planned target in predicted
            order -> home base.
        rtb_fuel_floor: ``margin * fuel(rtb_distance_km)`` -- the LOW end of the window.
        continue_fuel_requirement: ``margin * fuel(continue_distance_km)`` -- the HIGH,
            exclusive end.
        projected_fuel_at_event: fuel the ego is projected to hold when the event fires,
            i.e. its launch fuel minus what the flown part of leg 1 costs.
        post_damage_fuel: the deterministic value the live aircraft is set to -- the
            MIDPOINT of the strict window.
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
    leg_length_km: Optional[float] = None
    first_target_latitude: Optional[float] = None
    first_target_longitude: Optional[float] = None
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

    @property
    def is_damaged(self) -> bool:
        return self.condition == CONDITION_DAMAGED

    @property
    def event_location(self) -> Optional[Location]:
        """The planned event point as a :class:`Location` (``None`` when clean)."""
        if self.event_latitude is None or self.event_longitude is None:
            return None
        return Location(float(self.event_latitude), float(self.event_longitude))

    @property
    def first_target_location(self) -> Optional[Location]:
        """Leg 1's endpoint as a :class:`Location` (``None`` when clean)."""
        if self.first_target_latitude is None or self.first_target_longitude is None:
            return None
        return Location(
            float(self.first_target_latitude), float(self.first_target_longitude)
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
            "leg_length_km": self.leg_length_km,
            "first_target_latitude": self.first_target_latitude,
            "first_target_longitude": self.first_target_longitude,
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
        }


@dataclass(frozen=True)
class FuelDamageOutcome:
    """What the event actually DID -- measured, not planned.

    Every damaged-only field is ``None`` until the event fires, and stays ``None`` on a
    clean episode and on a damaged episode whose ego never reached the threshold (it
    died, was never launched, or diverged from the predicted route). ``fired`` is the one
    boolean that says which of those happened; a zero would not.
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
    wake_occurred: bool = False
    wake_meta_action: Optional[int] = None

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
            "wake_occurred": bool(self.wake_occurred),
            "wake_meta_action": self.wake_meta_action,
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
         decision at all: the plan was already infeasible and the event changed nothing;
      4. the chosen ``post_damage_fuel`` really lies inside ``[floor, requirement)`` and
         strictly below the projected fuel, so the mutation is a genuine LOSS.

    The chosen value is the MIDPOINT of the window -- the approved deterministic choice,
    and the one furthest from both ends, so neither bound is decided by floating-point
    noise.

    Raises:
        FuelDamageError: if any of the four facts does not hold, or if a required input
            is missing for a damaged condition. Never returns a clean plan instead.
    """
    params.validate()
    if condition not in CONDITIONS:
        raise FuelDamageError("condition must be one of %r, got %r" % (list(CONDITIONS), condition))

    eligible = tuple(str(e) for e in eligible_ego_ids)
    if condition == CONDITION_CLEAN:
        return FuelDamagePlan(
            condition=CONDITION_CLEAN,
            mode=str(mode),
            derived_seed=int(derived_seed),
            eligible_ego_ids=eligible,
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

    event_point = interpolate_great_circle(launch_point, first_target, fraction)

    # RTB leg: the engine measures fuel-to-home from the aircraft's live position to its
    # home base, so this is exactly that distance at the planned event point.
    rtb_distance_km = float(event_point.distance_to(home_base))

    # Continue leg: the REST of the predicted route (starting from the event point, so
    # the flown part of leg 1 is not charged twice) plus the return home. `predict_route`
    # is the frozen structural reproduction of the executor's ordering, reused rather
    # than reimplemented so the requirement cannot be measured against a route the ego
    # does not fly.
    continue_distance_km = float(
        _polyline_length_km([event_point] + points) + points[-1].distance_to(home_base)
    )

    def _fuel(distance_km: float) -> float:
        return fuel_for_distance_km(
            distance_km, speed_knots=speed_knots, fuel_rate=fuel_rate
        )

    rtb_fuel_floor = margin * _fuel(rtb_distance_km)
    continue_fuel_requirement = margin * _fuel(continue_distance_km)

    if fuel_at_launch is None:
        raise FuelDamageError("ego %s: launch fuel is unknown" % ego_id)
    launch_fuel = float(fuel_at_launch)
    if not math.isfinite(launch_fuel) or launch_fuel <= 0.0:
        raise FuelDamageError(
            "ego %s: launch fuel must be finite and > 0, got %r" % (ego_id, fuel_at_launch)
        )
    projected_fuel_at_event = launch_fuel - _fuel(fraction * leg_length_km)

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

    post_damage_fuel = 0.5 * (rtb_fuel_floor + continue_fuel_requirement)

    # (4) the chosen value really is inside the window and really is a loss.
    if not (rtb_fuel_floor <= post_damage_fuel < continue_fuel_requirement):
        raise FuelDamageError(
            "ego %s: the selected post-damage fuel %.6f is not inside the strict window "
            "[%.6f, %.6f)" % (ego_id, post_damage_fuel, rtb_fuel_floor,
                              continue_fuel_requirement)
        )
    if not (post_damage_fuel < projected_fuel_at_event):
        raise FuelDamageError(
            "ego %s: the selected post-damage fuel %.6f is not below the projected fuel "
            "%.6f, so the event would not be a loss"
            % (ego_id, post_damage_fuel, projected_fuel_at_event)
        )

    return FuelDamagePlan(
        condition=CONDITION_DAMAGED,
        mode=str(mode),
        derived_seed=int(derived_seed),
        eligible_ego_ids=eligible,
        ego_id=str(ego_id),
        leg_index=EVENT_LEG_INDEX,
        progress_threshold=fraction,
        leg_length_km=leg_length_km,
        first_target_latitude=float(first_target.latitude),
        first_target_longitude=float(first_target.longitude),
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
    )


# =============================================================================
# 5. The runtime controller (the ONLY place that mutates BLADE)
# =============================================================================

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
        self._fired = False
        self._event_tick: Optional[int] = None
        self._observed_progress: Optional[float] = None
        self._observed_lat: Optional[float] = None
        self._observed_lon: Optional[float] = None
        self._fuel_before: Optional[float] = None
        self._fuel_after: Optional[float] = None
        self._wake_occurred = False
        self._wake_meta_action: Optional[int] = None

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

    def maybe_apply(self, scenario: Any, tick: int) -> Optional[str]:
        """Apply the event if this is the first tick at or past the threshold.

        CALL ONCE PER TICK, AT THE START OF PHASE 1, BEFORE ANY EGO IS PROCESSED. That
        ordering is not a convenience: every ego must sense and decide against the same
        post-event snapshot, or the outcome would depend on Phase-1 ego iteration order
        and the no-communication guarantee would be gone.

        Returns:
            The damaged ego's id on the tick the event fires (the caller must wake exactly
            that ego with ``TriggerKind.FUEL_DAMAGE``), and ``None`` on every other tick --
            including every tick of a clean episode and every tick after the event.

        Raises:
            FuelDamageError: if the live fuel at the event is not strictly above the
                planned post-damage value. That would make the "damage" an increase, and
                it means the pre-run projection did not describe the run; it fails loudly
                as a ``run``-stage attempt failure rather than silently clamping.
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
        target = float(self.plan.post_damage_fuel or 0.0)
        if not (fuel_before > target):
            raise FuelDamageError(
                "ego %s: at tick %d the live fuel (%.3f) is not above the planned "
                "post-damage fuel (%.3f); applying the event would ADD fuel. The "
                "pre-run projection (%.3f) does not describe this run."
                % (ego_id, int(tick), fuel_before, target,
                   float(self.plan.projected_fuel_at_event or 0.0))
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
        return ego_id

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
            wake_occurred=self._wake_occurred,
            wake_meta_action=self._wake_meta_action,
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
    derived_seed = derive_fuel_damage_seed(episode_seed)
    a_init = dict(getattr(ctx, "a_init", None) or {})
    eligible = _eligible_ego_ids(a_init)

    if condition == CONDITION_CLEAN:
        return plan_fuel_damage(
            condition=CONDITION_CLEAN, mode=params.mode, derived_seed=derived_seed,
            eligible_ego_ids=eligible, ego_id=None, launch_point=None,
            home_base=None, route_points=None, speed_knots=None, fuel_rate=None,
            max_fuel=None, fuel_at_launch=None, params=params,
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

    ordered = predict_route(
        [tuple(a) for a in a_init.get(ego_id, [])], belief_tasks, launch_point
    )
    route_points: List[Location] = []
    for assignment in ordered:
        loc = _task_target_location(belief_tasks[int(assignment[0])])
        if loc is None:
            raise FuelDamageError(
                "ego %s: predicted assignment %r resolves to a step with no location"
                % (ego_id, assignment)
            )
        route_points.append(loc)
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
