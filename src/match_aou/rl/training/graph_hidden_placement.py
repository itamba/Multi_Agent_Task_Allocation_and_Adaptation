"""graph_hidden_placement.py

Route-relative hidden-target placement (offline scenario-construction phase, step 2 of 3).

WHAT THIS LAYER IS
------------------
A PURE geometry layer sitting between a known-only MATCH-AOU solution and the future
setup seam. Given

    (solution, belief_tasks, launch_point, parameters, rng)

it predicts each ego's actually-intended route, picks a stable leg of that route, and
constructs ONE hidden-target coordinate per ego route that the reference ego is
guaranteed to fly past inside its sensing radius BEFORE it reaches the planned target of
that leg.

It is deliberately free of BLADE, gymnasium, torch, the solver, scenario JSON, file I/O
and module-global randomness: the caller supplies an explicit ``random.Random``, and
``detection_km`` arrives through :class:`PlacementParameters` rather than being imported
from ``graph_episode_setup`` (importing it would drag this layer into the
setup/solver/executor dependency closure). Everything it needs from the rest of the
project is the pure domain models plus the frozen executor helper
``nearest_neighbor_order`` -- IMPORTED, never reimplemented, so route prediction cannot
drift from route execution.

THE GEOMETRY (closed decisions)
-------------------------------
Route. For every ego, ascending ``level_order``; inside each level the greedy
nearest-neighbor tour from the existing helper; the helper's returned end location is
chained into the next level. The first level is seeded from the SHARED launch point
(``CLAUDE.md`` Sec 3, "Launch point == the BLUE airbase"). The result is a polyline
``launch -> target 1 -> target 2 -> ...``; a LEG is one adjacent segment of it, so
``launch -> target 2`` is not a leg when target 1 is visited first.

Guaranteed portion. With ``D = detection_km`` and a predicted leg of length ``L``, only
``G = L - D`` is guaranteed to be flown: inside ``D`` of the leg's target the ego attacks
and issues no new movement (``CLAUDE.md`` Sec 3, "Detection/attack range -- ONE radius").
A leg with ``G <= 0`` is geometrically invalid. The perpendicular PROJECTION of the
hidden target -- not merely the hidden target itself -- is placed at ``s = f * G`` from
the predicted leg origin with ``f ~ Uniform[fraction_min, fraction_max]``, biasing it
away from the common star origin.

Perpendicular offset. Leg 1 starts at the exact launch point, so its offset cap is
``D - guard_km`` (40 km at the reference ``D = 50``, ``guard = 10``). Legs 2+ start in the
PREVIOUS target's vicinity rather than at it -- live replanning may begin up to ``D``
short of the predicted origin -- so for predicted progress ``u = s / L`` the residual
origin uncertainty is conservatively budgeted as ``(1 - u) * D`` and the cap shrinks to
``D - guard_km - (1 - u) * D``. A later leg is also required to keep its whole approved
fraction interval beyond the first ``D`` kilometres after the predicted origin, so
eligibility never depends on the sampled fraction.

Nearest-neighbor stability. Leg 1 needs no margin (prediction and execution share the
launch point). Every later leg is only eligible when the nearest-neighbor decision that
created it had a strict margin ``gap = d2 - d1 > 2 * D`` over the second-nearest
remaining assignment of the same level; one remaining candidate passes trivially, because
no competitor can reverse the ordering. The margin protects the EXISTENCE and ORDER of
the predicted leg -- it is not a hidden-coordinate distance check.

Selection. Among the eligible later legs one is chosen uniformly with the supplied rng;
if none is eligible the placement falls back to leg 1; if leg 1 is invalid too,
:class:`HiddenPlacementError` is raised. A failed margin is never weakened, the
nearest-neighbor ordering is never changed, and the coordinate is never moved onto an
unstable leg.

CARDINALITY
-----------
Exactly ONE placement per ego route (three placements for three egos in the B1 reference
cell). A general ``n_hidden != number of usable ego routes`` distribution policy is a
later, explicit design task and is deliberately NOT invented here.

NUMERICS
--------
The construction is exact spherical geometry on unit vectors (no flat-earth degree
conversion). Every returned placement is then re-measured through an INDEPENDENT
bearing-based cross-track / along-track computation seeded by ``Location.distance_to``,
and rejected if the two disagree. There is no silent clamping: a geometry that cannot
satisfy the request raises.

Run the proofs:
    pytest tests/test_graph_hidden_placement.py -q
    python tests/test_graph_hidden_placement.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import groupby
from numbers import Integral
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ...models import Location, Task
# The FROZEN executor's pure ordering helper. Imported, never reproduced or modified:
# offline route prediction and online route execution must be the same function
# (graph_rl_project_handoff.md, "Route prediction").
from ...utils.blade_utils.blade_executor_minimal import nearest_neighbor_order

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level_order)

# Mean Earth radius in km. Matches the `haversine` package that `Location.distance_to`
# prefers; its pure-Python fallback uses 6371.0, a 1.4e-6 relative difference (sub-metre
# over a 300 km leg) that is orders of magnitude below `validation_tolerance_km`.
EARTH_RADIUS_KM = 6371.0088

# Closed decision, not a knob: a later leg is eligible only at a strict
# `gap > 2 * detection_km`.
TIE_MARGIN_DETECTION_MULTIPLE = 2.0

__all__ = [
    "Assignment",
    "EARTH_RADIUS_KM",
    "HiddenPlacement",
    "HiddenPlacementError",
    "PlacementParameters",
    "geometric_fingerprint",
    "place_hidden_targets",
    "predict_route",
    "validate_placement",
]


class HiddenPlacementError(ValueError):
    """A hidden target could not be placed under the requested research geometry.

    Raised for every failure mode of this layer -- malformed input, an unusable route,
    an impossible guaranteed portion, an exhausted offset budget, or a returned
    placement that fails its own independent re-measurement. The layer never degrades
    silently: it either returns geometry that satisfies the request or it raises.
    """


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlacementParameters:
    """Immutable placement geometry request.

    ``detection_km`` is supplied by the CALLER (never imported from the setup layer) and
    is the single radius of ``CLAUDE.md`` Sec 3: sensing == arrival == attack ==
    kill-confirmation.

    ``guard_km`` is the sensing safety margin held back from the offset cap.
    ``fraction_min``/``fraction_max`` bound the sampled position along the GUARANTEED
    portion of the chosen leg. ``tolerance_km`` is the small numeric slack used for
    eligibility comparisons (it never relaxes a strict inequality by a meaningful amount),
    and ``validation_tolerance_km`` bounds the disagreement allowed between the spherical
    construction and its independent re-measurement.
    """

    detection_km: float
    guard_km: float = 10.0
    fraction_min: float = 0.60
    fraction_max: float = 0.85
    tolerance_km: float = 1e-6
    validation_tolerance_km: float = 0.05

    @property
    def tie_margin_km(self) -> float:
        """Strict nearest-neighbor margin a leg 2+ decision must exceed."""
        return TIE_MARGIN_DETECTION_MULTIPLE * float(self.detection_km)

    @property
    def leg1_max_abs_offset_km(self) -> float:
        """Offset cap on leg 1, whose origin (the launch point) is exact."""
        return float(self.detection_km) - float(self.guard_km)

    def validate(self) -> None:
        """Raise :class:`HiddenPlacementError` unless every field is usable."""
        for name in (
            "detection_km",
            "guard_km",
            "fraction_min",
            "fraction_max",
            "tolerance_km",
            "validation_tolerance_km",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise HiddenPlacementError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)):
                raise HiddenPlacementError(f"{name} must be finite, got {value!r}")

        if float(self.detection_km) <= 0.0:
            raise HiddenPlacementError(
                f"detection_km must be positive, got {self.detection_km!r}"
            )
        if float(self.guard_km) < 0.0:
            raise HiddenPlacementError(
                f"guard_km must be non-negative, got {self.guard_km!r}"
            )
        if float(self.guard_km) >= float(self.detection_km):
            raise HiddenPlacementError(
                f"guard_km ({self.guard_km}) must be smaller than detection_km "
                f"({self.detection_km}); otherwise no offset is admissible"
            )
        if not (0.0 < float(self.fraction_min) <= float(self.fraction_max) <= 1.0):
            raise HiddenPlacementError(
                "fraction bounds must satisfy 0 < fraction_min <= fraction_max <= 1, got "
                f"({self.fraction_min}, {self.fraction_max})"
            )
        if float(self.tolerance_km) < 0.0 or float(self.validation_tolerance_km) < 0.0:
            raise HiddenPlacementError("tolerances must be non-negative")


# ---------------------------------------------------------------------------
# Result / metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HiddenPlacement:
    """One constructed hidden-target coordinate plus everything needed to audit it.

    Every field is either an input anchor or a measured quantity, so an auditor can
    rebuild the placement from this record alone (that is exactly what
    :func:`validate_placement` does). Deliberately carries NO target id and no uuid:
    reproducibility is judged by geometric fingerprint (``CLAUDE.md`` Sec 8 -- generated
    ids are not seed-derived).
    """

    ego_id: str
    leg_index: int                      # 1-based index along the predicted polyline
    origin_assignment: Optional[Assignment]   # None on leg 1 (origin is the launch point)
    target_assignment: Assignment
    origin_latitude: float
    origin_longitude: float
    target_latitude: float
    target_longitude: float
    leg_length_km: float                # L
    guaranteed_km: float                # G = L - detection_km
    fraction: float                     # f, sampled in [fraction_min, fraction_max]
    arc_km: float                       # s = f * G, projection offset from leg origin
    offset_km: float                    # signed perpendicular offset (+ = left of travel)
    origin_uncertainty_km: float        # 0 on leg 1; (1 - s/L) * detection_km on legs 2+
    max_abs_offset_km: float            # detection_km - guard_km - origin_uncertainty_km
    min_projection_km: float            # 0 on leg 1; detection_km on legs 2+
    tie_gap_km: Optional[float]         # None on leg 1 or when a single candidate remained
    tie_margin_required_km: Optional[float]   # None on leg 1
    single_candidate: bool              # True when no competitor could reverse the order
    latitude: float
    longitude: float

    @property
    def location(self) -> Location:
        """A FRESH mutable :class:`Location` for the final coordinate.

        Minted per call so the record itself stays immutable (``Location`` is not).
        """
        return Location(self.latitude, self.longitude, 0)

    @property
    def fingerprint(self) -> Tuple[float, float]:
        """Id-free geometric identity of this placement."""
        return (self.latitude, self.longitude)


def geometric_fingerprint(
    placements: Sequence[HiddenPlacement], *, ndigits: int = 9
) -> Tuple[Tuple[float, float], ...]:
    """Id-free reproducibility fingerprint of a placement sequence.

    Coordinates only, in the returned (deterministic, ego-sorted) order. Never includes
    a target id or uuid, because those are not seed-derived (``CLAUDE.md`` Sec 8).
    """
    return tuple(
        (round(float(p.latitude), ndigits), round(float(p.longitude), ndigits))
        for p in placements
    )


# ---------------------------------------------------------------------------
# Spherical geometry (construction)
# ---------------------------------------------------------------------------

def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _to_vector(loc: Location) -> Tuple[float, float, float]:
    lat = math.radians(float(loc.latitude))
    lon = math.radians(float(loc.longitude))
    cos_lat = math.cos(lat)
    return (cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat))


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm <= 0.0 or not math.isfinite(norm):
        raise HiddenPlacementError("degenerate direction vector in placement geometry")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _cross(
    u: Tuple[float, float, float], v: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _to_location(v: Tuple[float, float, float]) -> Location:
    x, y, z = _normalize(v)
    return Location(math.degrees(math.asin(_clamp_unit(z))), math.degrees(math.atan2(y, x)), 0)


def _offset_point(
    origin: Location, target: Location, arc_km: float, offset_km: float
) -> Location:
    """Point at ``arc_km`` along the ``origin -> target`` great circle, then ``offset_km``
    perpendicular to it (positive = LEFT of the direction of travel).

    Exact on the sphere: the perpendicular hop leaves the along-track projection at
    exactly ``arc_km`` and puts the point at exactly ``|offset_km|`` from the leg's great
    circle, because the hop direction is the great circle's own pole.
    """
    a = _to_vector(origin)
    b = _to_vector(target)
    dot = _clamp_unit(sum(ai * bi for ai, bi in zip(a, b)))
    if dot >= 1.0:
        raise HiddenPlacementError("leg origin and target coincide; no direction to fly")
    tangent = _normalize(tuple(b[i] - dot * a[i] for i in range(3)))

    ang = float(arc_km) / EARTH_RADIUS_KM
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    point = tuple(a[i] * cos_a + tangent[i] * sin_a for i in range(3))
    heading = tuple(-a[i] * sin_a + tangent[i] * cos_a for i in range(3))

    normal = _cross(point, heading)  # unit pole of the leg's great circle; + = left
    theta = float(offset_km) / EARTH_RADIUS_KM
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return _to_location(tuple(point[i] * cos_t + normal[i] * sin_t for i in range(3)))


# ---------------------------------------------------------------------------
# Spherical geometry (independent re-measurement -- the validation path)
# ---------------------------------------------------------------------------

def _initial_bearing_rad(a: Location, b: Location) -> float:
    lat1 = math.radians(float(a.latitude))
    lat2 = math.radians(float(b.latitude))
    dlon = math.radians(float(b.longitude) - float(a.longitude))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.atan2(y, x)


def _cross_track_km(origin: Location, target: Location, point: Location) -> float:
    """Signed perpendicular distance from ``point`` to the ``origin -> target`` great
    circle (positive = LEFT of travel, matching :func:`_offset_point`).

    Independent of the construction: it goes through ``Location.distance_to`` (haversine)
    and initial bearings, not through the unit-vector algebra that built the point.
    """
    delta13 = origin.distance_to(point) / EARTH_RADIUS_KM
    bearing13 = _initial_bearing_rad(origin, point)
    bearing12 = _initial_bearing_rad(origin, target)
    return -math.asin(_clamp_unit(math.sin(delta13) * math.sin(bearing13 - bearing12))) * (
        EARTH_RADIUS_KM
    )


def _along_track_km(
    origin: Location, target: Location, point: Location, cross_track_km: float
) -> float:
    """Signed distance from ``origin`` to ``point``'s perpendicular foot on the leg."""
    delta13 = origin.distance_to(point) / EARTH_RADIUS_KM
    delta_xt = abs(float(cross_track_km)) / EARTH_RADIUS_KM
    cos_xt = math.cos(delta_xt)
    if cos_xt == 0.0:
        raise HiddenPlacementError("degenerate cross-track angle in placement validation")
    along = math.acos(_clamp_unit(math.cos(delta13) / cos_xt)) * EARTH_RADIUS_KM
    bearing13 = _initial_bearing_rad(origin, point)
    bearing12 = _initial_bearing_rad(origin, target)
    return along if math.cos(bearing13 - bearing12) >= 0.0 else -along


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _check_location(loc: object, what: str) -> Location:
    lat = getattr(loc, "latitude", None)
    lon = getattr(loc, "longitude", None)
    if lat is None or lon is None:
        raise HiddenPlacementError(f"{what} is not a Location (got {loc!r})")
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        raise HiddenPlacementError(f"{what} has non-numeric coordinates ({lat!r}, {lon!r})")
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        raise HiddenPlacementError(f"{what} has non-finite coordinates ({lat_f}, {lon_f})")
    if not (-90.0 <= lat_f <= 90.0) or not (-180.0 <= lon_f <= 180.0):
        raise HiddenPlacementError(f"{what} is out of range ({lat_f}, {lon_f})")
    return loc  # type: ignore[return-value]


def _as_assignment(raw: object, *, ego_id: str) -> Assignment:
    """Normalize one assignment to a plain int triple, rejecting anything non-integral.

    Deliberately NOT ``int(...)``: coercion would silently rewrite malformed semantic
    input -- ``(0.9, 0, 0)`` would be accepted as task 0 and the predicted route this
    layer measures against would quietly become a different route. Only genuine integral
    values are accepted (``numbers.Integral``, so a numpy integer still works), and they
    are normalized to built-in ``int``. ``bool`` is rejected explicitly despite
    subclassing ``int``: ``True`` is not a task index. Fractional floats, integral-valued
    floats and numeric strings all raise.
    """
    try:
        parts = tuple(raw)  # type: ignore[arg-type]
    except TypeError:
        raise HiddenPlacementError(f"ego {ego_id}: assignment {raw!r} is not a tuple")
    if len(parts) != 3:
        raise HiddenPlacementError(
            f"ego {ego_id}: assignment {raw!r} must be (task_idx, step_idx, level_order)"
        )
    fields: List[int] = []
    for name, value in zip(("task_idx", "step_idx", "level_order"), parts):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise HiddenPlacementError(
                f"ego {ego_id}: assignment {raw!r} has a non-integral {name} "
                f"({value!r} of type {type(value).__name__}); it is never coerced"
            )
        fields.append(int(value))
    return (fields[0], fields[1], fields[2])


def _step_location(belief_tasks: Sequence[Task], assignment: Assignment) -> Location:
    """Resolve an assignment's step location, failing LOUDLY.

    The executor treats an invalid index as "nothing to execute" and silently reorders it
    last; a placement layer must not, because a dropped assignment would silently change
    the predicted route it is measuring against.
    """
    task_idx, step_idx, _level = assignment
    if not (0 <= task_idx < len(belief_tasks)):
        raise HiddenPlacementError(
            f"assignment {assignment} has task_idx {task_idx} out of range "
            f"(0..{len(belief_tasks) - 1})"
        )
    steps = getattr(belief_tasks[task_idx], "steps", None)
    if steps is None:
        raise HiddenPlacementError(f"task {task_idx} has no steps")
    if not (0 <= step_idx < len(steps)):
        raise HiddenPlacementError(
            f"assignment {assignment} has step_idx {step_idx} out of range "
            f"(0..{len(steps) - 1})"
        )
    loc = getattr(steps[step_idx], "location", None)
    if loc is None:
        raise HiddenPlacementError(f"assignment {assignment} resolves to a step with no location")
    return _check_location(loc, f"assignment {assignment} step location")


# ---------------------------------------------------------------------------
# Route prediction
# ---------------------------------------------------------------------------

def predict_route(
    assignments: Sequence[Assignment],
    belief_tasks: Sequence[Task],
    start_location: Location,
) -> Tuple[Assignment, ...]:
    """Predict one ego's flown order, structurally reproducing the executor.

    Ascending ``level_order``; ``nearest_neighbor_order`` called separately inside each
    level; the helper's returned end location chained into the next level; the first level
    seeded from ``start_location``. Every assignment is resolved (and therefore validated)
    up front, so the helper never sees an "unlocated" assignment and can never move one to
    the back of the queue behind this layer's back. ``(task_idx, step_idx)`` tie-breaking
    is the helper's own and is not re-implemented here.
    """
    _check_location(start_location, "start_location")
    resolved = [_step_location(belief_tasks, a) for a in assignments]
    location_by_assignment: Dict[Assignment, Location] = {}
    for assignment, loc in zip(assignments, resolved):
        location_by_assignment[tuple(assignment)] = loc  # type: ignore[index]

    def _location_of(assignment: Assignment) -> Optional[Location]:
        return location_by_assignment[tuple(assignment)]  # type: ignore[index]

    ordered_all: List[Assignment] = []
    current: Optional[Location] = start_location
    by_level = sorted(assignments, key=lambda a: int(a[2]))  # stable, like the executor
    for _level, group in groupby(by_level, key=lambda a: int(a[2])):
        ordered, end = nearest_neighbor_order(
            list(group), location_of=_location_of, start_location=current
        )
        ordered_all.extend(ordered)
        current = end
    return tuple(tuple(a) for a in ordered_all)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Leg geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Leg:
    """One adjacent segment of a predicted route polyline (1-based ``index``)."""

    index: int
    origin: Location
    target: Location
    origin_assignment: Optional[Assignment]
    target_assignment: Assignment
    length_km: float
    tie_gap_km: Optional[float]
    single_candidate: bool


def _tie_state(
    route: Sequence[Assignment],
    position: int,
    location_of: Dict[Assignment, Location],
    current: Location,
) -> Tuple[Optional[float], bool]:
    """Margin of the nearest-neighbor decision that produced ``route[position]``.

    The competitors are exactly the SAME-LEVEL assignments not yet picked at that moment
    (the ones appearing later in the route), because ``nearest_neighbor_order`` runs
    independently per level and level order itself is topological and therefore fixed.
    """
    level = int(route[position][2])
    competitors = [a for a in route[position + 1:] if int(a[2]) == level]
    if not competitors:
        return None, True
    chosen = current.distance_to(location_of[route[position]])
    runner_up = min(current.distance_to(location_of[a]) for a in competitors)
    return runner_up - chosen, False


def _build_legs(
    route: Sequence[Assignment],
    belief_tasks: Sequence[Task],
    launch_point: Location,
) -> List[_Leg]:
    location_of = {a: _step_location(belief_tasks, a) for a in route}
    legs: List[_Leg] = []
    previous_loc: Location = launch_point
    previous_assignment: Optional[Assignment] = None
    for position, assignment in enumerate(route):
        target_loc = location_of[assignment]
        index = position + 1
        if index == 1:
            gap, single = None, False  # leg 1 needs no margin: prediction starts where flight does
        else:
            gap, single = _tie_state(route, position, location_of, previous_loc)
        legs.append(
            _Leg(
                index=index,
                origin=previous_loc,
                target=target_loc,
                origin_assignment=previous_assignment,
                target_assignment=assignment,
                length_km=float(previous_loc.distance_to(target_loc)),
                tie_gap_km=gap,
                single_candidate=single,
            )
        )
        previous_loc = target_loc
        previous_assignment = assignment
    return legs


def _leg_rejection(leg: _Leg, params: PlacementParameters) -> Optional[str]:
    """``None`` when the leg can carry a placement, else the reason it cannot."""
    detection = float(params.detection_km)
    tol = float(params.tolerance_km)

    if not math.isfinite(leg.length_km) or leg.length_km <= 0.0:
        return f"leg length {leg.length_km} is not a positive finite distance"

    guaranteed = leg.length_km - detection
    if guaranteed <= tol:
        return (
            f"guaranteed portion {guaranteed:.6f} km <= 0 "
            f"(length {leg.length_km:.6f} km, detection {detection} km)"
        )

    if leg.index == 1:
        return None  # exact origin: no uncertainty, no margin, cap is detection - guard

    # The whole approved fraction interval must project beyond the uncertain origin
    # vicinity, so eligibility never depends on the fraction actually drawn.
    min_arc = float(params.fraction_min) * guaranteed
    if min_arc <= detection + tol:
        return (
            f"earliest projection {min_arc:.6f} km does not clear the uncertain origin "
            f"vicinity ({detection} km)"
        )

    # Worst case over the interval: origin uncertainty decreases as the fraction grows.
    budget = detection - float(params.guard_km) - (1.0 - min_arc / leg.length_km) * detection
    if budget <= tol:
        return f"offset budget {budget:.6f} km is exhausted by origin uncertainty"

    if leg.single_candidate:
        return None  # no competitor can reverse the ordering
    gap = leg.tie_gap_km
    if gap is None:
        return "missing nearest-neighbor tie gap"
    if not (gap > params.tie_margin_km + tol):
        return (
            f"nearest-neighbor margin {gap:.6f} km does not exceed "
            f"{params.tie_margin_km:.6f} km"
        )
    return None


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _construct_placement(
    ego_id: str, leg: _Leg, params: PlacementParameters, rng: random.Random
) -> HiddenPlacement:
    """Sample and build one placement on an already-eligible leg.

    RNG draw order (fixed, and part of the reproducibility contract): the leg choice is
    drawn by the caller, then the fraction, then the signed offset.
    """
    detection = float(params.detection_km)
    guaranteed = leg.length_km - detection

    fraction = float(params.fraction_min) + rng.random() * (
        float(params.fraction_max) - float(params.fraction_min)
    )
    arc = fraction * guaranteed
    if leg.index == 1:
        uncertainty = 0.0
        min_projection = 0.0
    else:
        uncertainty = (1.0 - arc / leg.length_km) * detection
        min_projection = detection
    max_abs_offset = detection - float(params.guard_km) - uncertainty
    if max_abs_offset <= 0.0:
        raise HiddenPlacementError(
            f"ego {ego_id}: leg {leg.index} offset budget collapsed to {max_abs_offset:.6f} km"
        )

    # Symmetric about the route: both sides are reachable, and 0 is admissible.
    offset = (2.0 * rng.random() - 1.0) * max_abs_offset
    point = _offset_point(leg.origin, leg.target, arc, offset)

    return HiddenPlacement(
        ego_id=ego_id,
        leg_index=leg.index,
        origin_assignment=leg.origin_assignment,
        target_assignment=leg.target_assignment,
        origin_latitude=float(leg.origin.latitude),
        origin_longitude=float(leg.origin.longitude),
        target_latitude=float(leg.target.latitude),
        target_longitude=float(leg.target.longitude),
        leg_length_km=leg.length_km,
        guaranteed_km=guaranteed,
        fraction=fraction,
        arc_km=arc,
        offset_km=offset,
        origin_uncertainty_km=uncertainty,
        max_abs_offset_km=max_abs_offset,
        min_projection_km=min_projection,
        tie_gap_km=leg.tie_gap_km,
        tie_margin_required_km=None if leg.index == 1 else params.tie_margin_km,
        single_candidate=leg.single_candidate,
        latitude=float(point.latitude),
        longitude=float(point.longitude),
    )


def validate_placement(placement: HiddenPlacement, params: PlacementParameters) -> None:
    """Re-derive a placement from its own metadata and raise on any disagreement.

    Pure and independent of how the placement was built: distances come from
    ``Location.distance_to`` and the projection from bearing-based cross-track /
    along-track formulas, not from the unit-vector construction. Checks
      * every coordinate finite and in range;
      * the recorded leg length, guaranteed portion, arc, origin uncertainty and offset
        cap all recompute from the recorded anchors;
      * the closest approach to the reference leg is within the sensing budget
        (``detection_km - guard_km``) and within the leg's own recorded cap;
      * the closest-approach PROJECTION lies inside the guaranteed portion, and beyond the
        uncertain origin region on legs 2+;
      * legs 2+ record the required tie margin (always, single-candidate included) and
        then either a passing strict gap or the single-candidate case; leg 1 records
        neither tie field.
    """
    params.validate()
    detection = float(params.detection_km)
    tol = float(params.validation_tolerance_km)

    if int(placement.leg_index) < 1:
        raise HiddenPlacementError(f"leg_index {placement.leg_index} must be 1-based")

    origin = _check_location(
        Location(placement.origin_latitude, placement.origin_longitude, 0), "placement origin"
    )
    target = _check_location(
        Location(placement.target_latitude, placement.target_longitude, 0), "placement target"
    )
    point = _check_location(
        Location(placement.latitude, placement.longitude, 0), "placement coordinate"
    )
    for name in ("leg_length_km", "guaranteed_km", "fraction", "arc_km", "offset_km",
                 "origin_uncertainty_km", "max_abs_offset_km", "min_projection_km"):
        value = float(getattr(placement, name))
        if not math.isfinite(value):
            raise HiddenPlacementError(f"placement.{name} is not finite ({value})")

    # --- metadata recomputes -------------------------------------------------
    length = float(origin.distance_to(target))
    if abs(length - float(placement.leg_length_km)) > tol:
        raise HiddenPlacementError(
            f"leg_length_km {placement.leg_length_km:.6f} != measured {length:.6f}"
        )
    guaranteed = length - detection
    if guaranteed <= 0.0:
        raise HiddenPlacementError(
            f"leg has no guaranteed portion (length {length:.6f} km, detection {detection} km)"
        )
    if abs(guaranteed - float(placement.guaranteed_km)) > tol:
        raise HiddenPlacementError(
            f"guaranteed_km {placement.guaranteed_km:.6f} != measured {guaranteed:.6f}"
        )
    if not (float(params.fraction_min) - 1e-12 <= float(placement.fraction)
            <= float(params.fraction_max) + 1e-12):
        raise HiddenPlacementError(
            f"fraction {placement.fraction} outside "
            f"[{params.fraction_min}, {params.fraction_max}]"
        )
    arc = float(placement.fraction) * guaranteed
    if abs(arc - float(placement.arc_km)) > tol:
        raise HiddenPlacementError(
            f"arc_km {placement.arc_km:.6f} != fraction * guaranteed {arc:.6f}"
        )

    if int(placement.leg_index) == 1:
        expected_uncertainty = 0.0
        expected_min_projection = 0.0
    else:
        expected_uncertainty = (1.0 - arc / length) * detection
        expected_min_projection = detection
    if abs(expected_uncertainty - float(placement.origin_uncertainty_km)) > tol:
        raise HiddenPlacementError(
            f"origin_uncertainty_km {placement.origin_uncertainty_km:.6f} != "
            f"expected {expected_uncertainty:.6f}"
        )
    if abs(expected_min_projection - float(placement.min_projection_km)) > tol:
        raise HiddenPlacementError(
            f"min_projection_km {placement.min_projection_km:.6f} != "
            f"expected {expected_min_projection:.6f}"
        )
    expected_cap = detection - float(params.guard_km) - expected_uncertainty
    if expected_cap <= 0.0:
        raise HiddenPlacementError(
            f"offset budget {expected_cap:.6f} km is non-positive for this geometry"
        )
    if abs(expected_cap - float(placement.max_abs_offset_km)) > tol:
        raise HiddenPlacementError(
            f"max_abs_offset_km {placement.max_abs_offset_km:.6f} != expected {expected_cap:.6f}"
        )
    if abs(float(placement.offset_km)) > float(placement.max_abs_offset_km) + tol:
        raise HiddenPlacementError(
            f"offset {placement.offset_km:.6f} km exceeds its cap "
            f"{placement.max_abs_offset_km:.6f} km"
        )

    # --- independent geometric re-measurement --------------------------------
    cross = _cross_track_km(origin, target, point)
    along = _along_track_km(origin, target, point, cross)
    if abs(cross - float(placement.offset_km)) > tol:
        raise HiddenPlacementError(
            f"measured closest approach {cross:.6f} km != recorded offset "
            f"{placement.offset_km:.6f} km"
        )
    sensing_budget = detection - float(params.guard_km)
    if abs(cross) > sensing_budget + tol:
        raise HiddenPlacementError(
            f"closest approach {abs(cross):.6f} km exceeds the sensing budget "
            f"{sensing_budget:.6f} km"
        )
    if abs(along - arc) > tol:
        raise HiddenPlacementError(
            f"measured projection {along:.6f} km != recorded arc {arc:.6f} km"
        )
    if along > guaranteed + tol:
        raise HiddenPlacementError(
            f"projection {along:.6f} km falls outside the guaranteed portion "
            f"{guaranteed:.6f} km"
        )
    if along < float(placement.min_projection_km) - tol:
        raise HiddenPlacementError(
            f"projection {along:.6f} km is inside the uncertain origin region "
            f"({placement.min_projection_km:.6f} km)"
        )

    # --- nearest-neighbor stability ------------------------------------------
    if int(placement.leg_index) == 1:
        if placement.tie_gap_km is not None or placement.tie_margin_required_km is not None:
            raise HiddenPlacementError("leg 1 must not record a tie margin")
        return

    # EVERY later leg must record the requirement it was judged against, whichever branch
    # it then takes -- including the single-candidate one. Checked BEFORE the branch, so a
    # record cannot claim "no competitor" and thereby skip the requirement entirely.
    required = placement.tie_margin_required_km
    if required is None:
        raise HiddenPlacementError(
            f"leg {placement.leg_index} records no required tie margin "
            f"(expected {params.tie_margin_km})"
        )
    required = float(required)
    if not math.isfinite(required):
        raise HiddenPlacementError(
            f"tie_margin_required_km {placement.tie_margin_required_km} is not finite"
        )
    if abs(required - params.tie_margin_km) > 1e-9:
        raise HiddenPlacementError(
            f"tie_margin_required_km {placement.tie_margin_required_km} != "
            f"{params.tie_margin_km}"
        )

    if placement.single_candidate:
        if placement.tie_gap_km is not None:
            raise HiddenPlacementError(
                "single-candidate legs have no competitor and must record no tie gap"
            )
        return
    if placement.tie_gap_km is None:
        raise HiddenPlacementError(f"leg {placement.leg_index} records no tie gap")
    gap = float(placement.tie_gap_km)
    if not math.isfinite(gap):
        raise HiddenPlacementError(f"tie_gap_km {placement.tie_gap_km} is not finite")
    if not (gap > params.tie_margin_km):
        raise HiddenPlacementError(
            f"nearest-neighbor margin {gap:.6f} km does not exceed "
            f"the required {params.tie_margin_km:.6f} km"
        )


def place_hidden_targets(
    solution: Mapping[str, Sequence[Assignment]],
    belief_tasks: Sequence[Task],
    launch_point: Location,
    parameters: PlacementParameters,
    rng: random.Random,
) -> Tuple[HiddenPlacement, ...]:
    """Construct exactly one hidden-target coordinate per ego route.

    Egos are iterated in sorted-id order, so the result never depends on ``solution``'s
    insertion order. Per ego: predict the route, prefer a uniformly chosen ELIGIBLE later
    leg, fall back to leg 1, and raise when neither is usable. Every returned placement is
    passed through :func:`validate_placement` before it leaves this function.

    Returns the placements in that deterministic ego order. The current cardinality is one
    placement per ego route -- distributing a different ``n_hidden`` across routes is a
    later, explicit design task.
    """
    parameters.validate()
    _check_location(launch_point, "launch_point")
    if not isinstance(rng, random.Random):
        raise HiddenPlacementError(
            f"rng must be an explicit random.Random, got {type(rng).__name__}"
        )
    if not solution:
        raise HiddenPlacementError("solution is empty: there is no route to place against")

    by_ego: Dict[str, Sequence[Assignment]] = {}
    for key, value in solution.items():
        ego_id = str(key)
        if ego_id in by_ego:
            raise HiddenPlacementError(f"solution has duplicate ego id {ego_id!r} after str()")
        by_ego[ego_id] = value

    placements: List[HiddenPlacement] = []
    for ego_id in sorted(by_ego):
        raw = by_ego[ego_id]
        if raw is None or len(raw) == 0:
            raise HiddenPlacementError(f"ego {ego_id} has no usable assigned route")
        assignments = [_as_assignment(a, ego_id=ego_id) for a in raw]

        route = predict_route(assignments, belief_tasks, launch_point)
        legs = _build_legs(route, belief_tasks, launch_point)
        if not legs:
            raise HiddenPlacementError(f"ego {ego_id} has no usable assigned route")

        eligible_later = [leg for leg in legs[1:] if _leg_rejection(leg, parameters) is None]
        if eligible_later:
            leg = rng.choice(eligible_later)
        else:
            leg1_rejection = _leg_rejection(legs[0], parameters)
            if leg1_rejection is not None:
                raise HiddenPlacementError(
                    f"ego {ego_id}: no later leg is eligible and leg 1 is unusable "
                    f"({leg1_rejection})"
                )
            leg = legs[0]

        placement = _construct_placement(ego_id, leg, parameters, rng)
        validate_placement(placement, parameters)
        placements.append(placement)

    if len(placements) != len(by_ego):
        raise HiddenPlacementError(
            f"constructed {len(placements)} placements for {len(by_ego)} ego routes"
        )
    return tuple(placements)
