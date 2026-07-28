"""
Unit tests for route-relative hidden-target placement (offline scenario-construction
phase, step 2 of 3) -- `src/match_aou/rl/training/graph_hidden_placement.py`.

They discharge the three declared proof obligations:

  PO1  GUARANTEED SENSING GEOMETRY
       Every returned placement lies within the permitted sensing/guard budget of its
       reference leg and its CLOSEST APPROACH projects inside the guaranteed-flown
       portion. Covered on leg 1 over a seed sweep (both signs of the perpendicular
       offset), at the exact fraction/offset boundaries through a scripted rng, on a
       later leg with its reduced (origin-uncertainty-adjusted) budget, and by the loud
       failure of an impossible leg.

  PO2  MULTI-LEG FIDELITY AND FALLBACK
       Explicit multi-target fixtures -- never the 1:1 reference cell -- prove a
       placement on leg 2, a placement on leg 3, STRICT rejection at a 100 km
       nearest-neighbor gap with D = 50 (and acceptance just above it), the
       one-remaining-target trivial pass, fallback to leg 1 when a later leg is
       unstable, and that the predicted route is byte-identical to the queue the frozen
       executor builds through the same `nearest_neighbor_order` (level grouping +
       chained start location included).

  PO3  REPRODUCIBILITY
       Identically seeded `random.Random` instances produce identical geometric
       fingerprints AND identical metadata records, regardless of the solution dict's
       insertion order. Fingerprints are coordinates only -- never target ids or uuids
       (CLAUDE.md Sec 8: generated ids are not seed-derived).

Fixture geometry is built with an independent spherical destination helper (`_dest`) and
every fixture asserts its own premises (which target is nearest, what the gaps are), so a
test that stops proving what it claims fails instead of passing vacuously.

Pure: no bonmin, no BLADE `Game`, no gymnasium env, no torch, no file I/O. The only
heavier import is the frozen `BladeExecutorMinimal` (pure Python) used as the independent
oracle for route prediction.

Run:
    pytest tests/test_graph_hidden_placement.py -q                          (base env)
    conda run -n nlp_env --no-capture-output \\
        python tests/test_graph_hidden_placement.py                         (nlp_env)
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.models import Agent, Location, Step, StepKind, Task  # noqa: E402
from match_aou.rl.training.graph_hidden_placement import (  # noqa: E402
    EARTH_RADIUS_KM,
    HiddenPlacement,
    HiddenPlacementError,
    PlacementParameters,
    geometric_fingerprint,
    place_hidden_targets,
    predict_route,
    validate_placement,
)
from match_aou.utils.blade_utils.blade_executor_minimal import (  # noqa: E402
    BladeExecutorMinimal,
)

DETECTION_KM = 50.0  # the unified radius (CLAUDE.md Sec 3), supplied via parameters
GUARD_KM = 10.0
PARAMS = PlacementParameters(detection_km=DETECTION_KM, guard_km=GUARD_KM)

# The real BLUE airbase of `strike_training_4v5.json` -- every ego launches from here
# (CLAUDE.md Sec 3, "Launch point == the BLUE airbase").
LAUNCH = Location(32.85416, 35.31240, 0)


# ---------------------------------------------------------------------------
# Fixture helpers (independent of the module under test)
# ---------------------------------------------------------------------------

def _dest(origin: Location, bearing_deg: float, distance_km: float) -> Location:
    """Great-circle destination -- the standard direct formula, not the module's code."""
    lat1 = math.radians(origin.latitude)
    lon1 = math.radians(origin.longitude)
    theta = math.radians(bearing_deg)
    delta = distance_km / EARTH_RADIUS_KM
    lat2 = math.asin(
        math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(theta)
    )
    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(lat1),
        math.cos(delta) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return Location(math.degrees(lat2), math.degrees(lon2), 0)


def _task_at(loc, name: str, utility: float = 100.0) -> Task:
    return Task([Step(loc, name, [], 1.0, 1, StepKind.ATTACK)], utility)


def _tasks(*locs) -> list:
    return [_task_at(loc, f"tgt-{i}") for i, loc in enumerate(locs)]


def _plan(*task_indices, level: int = 0) -> list:
    return [(int(t), 0, int(level)) for t in task_indices]


class _ScriptedRandom(random.Random):
    """A real `random.Random` whose `random()` returns queued values.

    Only `random()` is scripted; `choice()` goes through `getrandbits`, so scripting the
    uniform draws never perturbs leg selection.
    """

    def __init__(self, values):
        super().__init__(0)
        self._queue = list(values)

    def random(self):  # noqa: D102
        assert self._queue, "scripted rng exhausted (production code drew more than expected)"
        return self._queue.pop(0)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _raises(exc_type, fn, *args, **kwargs) -> Exception:
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__} from {getattr(fn, '__name__', fn)}")


# ---------------------------------------------------------------------------
# PO1 -- guaranteed sensing geometry
# ---------------------------------------------------------------------------

def test_po1_leg1_sweep_satisfies_sensing_and_guaranteed_portion() -> None:
    """200 seeds on a single-leg route: every placement validates, both signs occur."""
    target = _dest(LAUNCH, 75.0, 300.0)
    tasks = _tasks(target)
    solution = {"ego": _plan(0)}

    signs = set()
    fractions = []
    for seed in range(200):
        placements = place_hidden_targets(
            solution, tasks, LAUNCH, PARAMS, random.Random(seed)
        )
        _assert(len(placements) == 1, "one placement per ego route")
        p = placements[0]
        _assert(p.leg_index == 1, f"seed {seed}: expected leg 1, got {p.leg_index}")
        _assert(p.origin_assignment is None, "leg 1 originates at the launch point")
        _assert(p.tie_gap_km is None and p.tie_margin_required_km is None,
                "leg 1 carries no tie margin")
        _assert(p.origin_uncertainty_km == 0.0, "leg 1 origin is exact")
        _assert(abs(p.max_abs_offset_km - (DETECTION_KM - GUARD_KM)) < 1e-9,
                f"leg 1 cap must be {DETECTION_KM - GUARD_KM} km, got {p.max_abs_offset_km}")

        # Sensing budget and guaranteed portion, re-measured independently.
        validate_placement(p, PARAMS)
        _assert(abs(p.offset_km) <= DETECTION_KM - GUARD_KM + 1e-9,
                f"seed {seed}: offset {p.offset_km} exceeds the 40 km cap")
        _assert(0.0 < p.arc_km <= p.guaranteed_km + 1e-9,
                f"seed {seed}: projection {p.arc_km} outside guaranteed {p.guaranteed_km}")
        # The projection must stop short of the attack radius around the planned target.
        _assert(p.leg_length_km - p.arc_km > DETECTION_KM - 1e-9,
                f"seed {seed}: projection is inside the target's detection radius")

        signs.add(1 if p.offset_km > 0 else (-1 if p.offset_km < 0 else 0))
        fractions.append(p.fraction)

    _assert({1, -1} <= signs, f"both offset signs must occur over the sweep, got {signs}")
    _assert(min(fractions) >= 0.60 - 1e-12 and max(fractions) <= 0.85 + 1e-12,
            f"fractions escaped [0.60, 0.85]: [{min(fractions)}, {max(fractions)}]")


def test_po1_scripted_fraction_and_offset_boundaries() -> None:
    """Both ends of the fraction range and both extreme offsets, exactly."""
    target = _dest(LAUNCH, 20.0, 300.0)
    tasks = _tasks(target)
    solution = {"ego": _plan(0)}
    guaranteed = float(LAUNCH.distance_to(target)) - DETECTION_KM
    cap = DETECTION_KM - GUARD_KM

    cases = [
        ([0.0, 0.0], 0.60, -cap),   # low fraction, hard left-cap... (sign: full negative)
        ([1.0, 1.0], 0.85, +cap),   # high fraction, full positive offset
        ([0.5, 0.5], 0.725, 0.0),   # mid fraction, exactly on the route
    ]
    for draws, expected_fraction, expected_offset in cases:
        p = place_hidden_targets(
            solution, tasks, LAUNCH, PARAMS, _ScriptedRandom(draws)
        )[0]
        _assert(abs(p.fraction - expected_fraction) < 1e-12,
                f"fraction {p.fraction} != {expected_fraction}")
        _assert(abs(p.offset_km - expected_offset) < 1e-9,
                f"offset {p.offset_km} != {expected_offset}")
        _assert(abs(p.arc_km - expected_fraction * guaranteed) < 1e-9,
                "arc must be fraction * guaranteed")
        validate_placement(p, PARAMS)


def test_po1_later_leg_has_reduced_offset_budget() -> None:
    """On a later leg the cap is 40 km MINUS the residual origin uncertainty."""
    t1 = _dest(LAUNCH, 90.0, 200.0)
    t2 = _dest(t1, 90.0, 400.0)
    t3 = _dest(t1, 45.0, 450.0)
    tasks = _tasks(t1, t2, t3)
    solution = {"ego": _plan(0, 1, 2)}

    p = place_hidden_targets(solution, tasks, LAUNCH, PARAMS, random.Random(7))[0]
    _assert(p.leg_index >= 2, f"expected a later leg, got {p.leg_index}")
    _assert(p.origin_uncertainty_km > 0.0, "a later leg must budget origin uncertainty")
    expected_uncertainty = (1.0 - p.arc_km / p.leg_length_km) * DETECTION_KM
    _assert(abs(p.origin_uncertainty_km - expected_uncertainty) < 1e-9,
            "origin uncertainty must be (1 - s/L) * D")
    _assert(p.max_abs_offset_km < DETECTION_KM - GUARD_KM,
            "a later leg's cap must be strictly below leg 1's 40 km")
    _assert(abs(p.max_abs_offset_km
                - (DETECTION_KM - GUARD_KM - expected_uncertainty)) < 1e-9,
            "cap must be D - guard - origin uncertainty")
    _assert(abs(p.offset_km) <= p.max_abs_offset_km + 1e-9, "offset must respect its cap")
    _assert(p.min_projection_km == DETECTION_KM, "a later leg must clear the uncertain origin")
    _assert(p.arc_km > DETECTION_KM, f"projection {p.arc_km} must be beyond {DETECTION_KM} km")
    validate_placement(p, PARAMS)


def test_po1_impossible_short_leg_raises() -> None:
    """A leg shorter than the detection radius has no guaranteed portion -- fail loudly."""
    tasks = _tasks(_dest(LAUNCH, 10.0, 30.0))
    exc = _raises(
        HiddenPlacementError,
        place_hidden_targets,
        {"ego": _plan(0)}, tasks, LAUNCH, PARAMS, random.Random(0),
    )
    _assert("guaranteed portion" in str(exc), f"unexpected message: {exc}")

    # Exactly at the radius is still impossible (G == 0).
    tasks = _tasks(_dest(LAUNCH, 10.0, DETECTION_KM))
    _raises(
        HiddenPlacementError,
        place_hidden_targets,
        {"ego": _plan(0)}, tasks, LAUNCH, PARAMS, random.Random(0),
    )


def test_po1_validate_placement_rejects_tampered_metadata() -> None:
    """Validation is load-bearing: perturb a placement and it must refuse it."""
    tasks = _tasks(_dest(LAUNCH, 75.0, 300.0))
    good = place_hidden_targets({"ego": _plan(0)}, tasks, LAUNCH, PARAMS, random.Random(3))[0]
    validate_placement(good, PARAMS)  # control

    # (a) coordinate moved off the constructed geometry
    moved = dataclasses.replace(good, latitude=good.latitude + 1.5)
    _raises(HiddenPlacementError, validate_placement, moved, PARAMS)

    # (b) offset beyond the sensing budget, with a coordinate to match
    far = _dest(LAUNCH, 75.0, 100.0)
    over = dataclasses.replace(good, offset_km=DETECTION_KM * 3,
                               latitude=far.latitude, longitude=far.longitude)
    _raises(HiddenPlacementError, validate_placement, over, PARAMS)

    # (c) fraction outside the approved interval
    _raises(HiddenPlacementError, validate_placement,
            dataclasses.replace(good, fraction=0.95), PARAMS)

    # (d) a later leg claiming leg-1's zero origin uncertainty
    _raises(HiddenPlacementError, validate_placement,
            dataclasses.replace(good, leg_index=2), PARAMS)


# ---------------------------------------------------------------------------
# PO2 -- multi-leg fidelity and fallback
# ---------------------------------------------------------------------------

def test_po2_placement_lands_on_leg_two() -> None:
    """A 4-target route where ONLY leg 2 is eligible."""
    t1 = _dest(LAUNCH, 90.0, 200.0)
    t2 = _dest(t1, 90.0, 400.0)
    t3 = _dest(t2, 0.0, 520.0)
    t4 = _dest(t3, 0.0, 40.0)
    tasks = _tasks(t1, t2, t3, t4)
    solution = {"ego": _plan(0, 1, 2, 3)}

    # Fixture premises, asserted rather than assumed.
    _assert(LAUNCH.distance_to(t1) < min(LAUNCH.distance_to(x) for x in (t2, t3, t4)),
            "t1 must be the first nearest-neighbor pick")
    gap_leg2 = min(t1.distance_to(t3), t1.distance_to(t4)) - t1.distance_to(t2)
    _assert(gap_leg2 > 2 * DETECTION_KM, f"leg 2 must be stable, gap={gap_leg2}")
    gap_leg3 = t2.distance_to(t4) - t2.distance_to(t3)
    _assert(gap_leg3 <= 2 * DETECTION_KM, f"leg 3 must be unstable, gap={gap_leg3}")
    _assert(t3.distance_to(t4) < DETECTION_KM, "leg 4 must have no guaranteed portion")

    for seed in range(25):
        p = place_hidden_targets(solution, tasks, LAUNCH, PARAMS, random.Random(seed))[0]
        _assert(p.leg_index == 2, f"seed {seed}: expected leg 2, got {p.leg_index}")
        _assert(p.origin_assignment == (0, 0, 0) and p.target_assignment == (1, 0, 0),
                f"leg-2 anchors wrong: {p.origin_assignment} -> {p.target_assignment}")
        _assert(p.tie_gap_km is not None and p.tie_gap_km > 2 * DETECTION_KM,
                "leg 2 must record a passing strict margin")
        validate_placement(p, PARAMS)


def test_po2_placement_lands_on_leg_three() -> None:
    """A 3-target route where leg 2 fails the margin and ONLY leg 3 is eligible."""
    t1 = _dest(LAUNCH, 90.0, 200.0)
    t2 = _dest(t1, 90.0, 400.0)
    t3 = _dest(t1, 45.0, 450.0)
    tasks = _tasks(t1, t2, t3)
    solution = {"ego": _plan(0, 1, 2)}

    _assert(t1.distance_to(t2) < t1.distance_to(t3), "t2 must be picked before t3")
    gap_leg2 = t1.distance_to(t3) - t1.distance_to(t2)
    _assert(gap_leg2 <= 2 * DETECTION_KM, f"leg 2 must be unstable, gap={gap_leg2}")
    _assert(t2.distance_to(t3) - DETECTION_KM > 0, "leg 3 must have a guaranteed portion")

    for seed in range(25):
        p = place_hidden_targets(solution, tasks, LAUNCH, PARAMS, random.Random(seed))[0]
        _assert(p.leg_index == 3, f"seed {seed}: expected leg 3, got {p.leg_index}")
        _assert(p.origin_assignment == (1, 0, 0) and p.target_assignment == (2, 0, 0),
                f"leg-3 anchors wrong: {p.origin_assignment} -> {p.target_assignment}")
        _assert(p.single_candidate and p.tie_gap_km is None,
                "the last leg has no competitor left")
        validate_placement(p, PARAMS)


def test_po2_tie_margin_strict_at_two_detection_radii() -> None:
    """gap == 100 km is REJECTED (fallback to leg 1); gap == 100.6 km is accepted.

    The two fixtures differ ONLY in t3's distance from t1, and leg 3 is ineligible in
    both (it is ~100 km long, so its approved fraction interval cannot clear the
    uncertain origin), which isolates the margin as the single deciding factor.
    """
    t1 = _dest(LAUNCH, 90.0, 200.0)
    t2 = _dest(t1, 90.0, 400.0)

    def _run(t3_distance_km: float):
        t3 = _dest(t1, 90.0, t3_distance_km)
        tasks = _tasks(t1, t2, t3)
        gap = t1.distance_to(t3) - t1.distance_to(t2)
        legs = [place_hidden_targets({"ego": _plan(0, 1, 2)}, tasks, LAUNCH, PARAMS,
                                     random.Random(seed))[0] for seed in range(15)]
        return gap, legs

    gap_equal, legs_equal = _run(500.0)
    _assert(abs(gap_equal - 2 * DETECTION_KM) < 1e-6,
            f"fixture must sit exactly on the 100 km margin, got {gap_equal}")
    for p in legs_equal:
        _assert(p.leg_index == 1,
                f"gap == 100 km must be rejected, but placement used leg {p.leg_index}")
        validate_placement(p, PARAMS)

    gap_above, legs_above = _run(500.6)
    _assert(gap_above > 2 * DETECTION_KM, f"fixture gap must exceed 100 km, got {gap_above}")
    for p in legs_above:
        _assert(p.leg_index == 2,
                f"gap > 100 km must be accepted, but placement used leg {p.leg_index}")
        _assert(p.tie_gap_km is not None and p.tie_gap_km > 2 * DETECTION_KM,
                "the accepted leg must record its passing margin")
        validate_placement(p, PARAMS)


def test_po2_single_remaining_candidate_passes_trivially() -> None:
    """With one assignment left there is no competitor, so the margin passes trivially."""
    t1 = _dest(LAUNCH, 90.0, 200.0)
    t2 = _dest(t1, 90.0, 400.0)
    tasks = _tasks(t1, t2)
    p = place_hidden_targets({"ego": _plan(0, 1)}, tasks, LAUNCH, PARAMS, random.Random(11))[0]
    _assert(p.leg_index == 2, f"expected leg 2, got {p.leg_index}")
    _assert(p.single_candidate, "the final leg must be flagged single-candidate")
    _assert(p.tie_gap_km is None, "a single-candidate leg records no tie gap")
    _assert(p.tie_margin_required_km == PARAMS.tie_margin_km, "the requirement is still recorded")
    validate_placement(p, PARAMS)


def test_po2_predicted_route_matches_executor_queue() -> None:
    """Prediction reproduces the frozen executor's queue: levels, chaining, tie-breaks."""
    locs = [
        _dest(LAUNCH, 90.0, 220.0),
        _dest(LAUNCH, 45.0, 260.0),
        _dest(LAUNCH, 135.0, 310.0),
        _dest(LAUNCH, 20.0, 480.0),
        _dest(LAUNCH, 300.0, 540.0),
    ]
    tasks = _tasks(*locs)
    assignments = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 2)]

    agent = Agent(
        location=LAUNCH,
        capabilities=[],
        budget=0.0,
        move_cost_function=lambda a, b: 0.0,
        agent_id="ego",
        return_location=LAUNCH,
    )
    executor = BladeExecutorMinimal(
        tasks=tasks, solution={"ego": list(assignments)}, agents=[agent]
    )
    expected = [tuple(a) for a in executor.queue["ego"]]
    predicted = list(predict_route(assignments, tasks, LAUNCH))
    _assert(predicted == expected, f"route {predicted} != executor queue {expected}")
    _assert([a[2] for a in predicted] == sorted(a[2] for a in predicted),
            "levels must stay ascending")

    # Insertion order of the assignment list must not matter.
    shuffled = list(reversed(assignments))
    _assert(list(predict_route(shuffled, tasks, LAUNCH)) == expected,
            "route prediction must not depend on assignment list order")


# ---------------------------------------------------------------------------
# PO3 -- reproducibility
# ---------------------------------------------------------------------------

def _three_ego_case():
    """Three egos, three routes -- the B1 reference cardinality, multi-target per ego."""
    locs = [
        _dest(LAUNCH, 70.0, 240.0), _dest(LAUNCH, 80.0, 620.0),
        _dest(LAUNCH, 160.0, 230.0), _dest(LAUNCH, 170.0, 700.0),
        _dest(LAUNCH, 300.0, 260.0), _dest(LAUNCH, 310.0, 660.0),
    ]
    tasks = _tasks(*locs)
    routes = {
        "ego-a": _plan(0, 1),
        "ego-b": _plan(2, 3),
        "ego-c": _plan(4, 5),
    }
    return tasks, routes


def test_po3_identical_seeds_reproduce_fingerprint_and_metadata() -> None:
    tasks, routes = _three_ego_case()
    first = place_hidden_targets(routes, tasks, LAUNCH, PARAMS, random.Random(2026))
    second = place_hidden_targets(routes, tasks, LAUNCH, PARAMS, random.Random(2026))

    _assert(len(first) == 3, f"one placement per ego route, got {len(first)}")
    _assert(geometric_fingerprint(first) == geometric_fingerprint(second),
            "identical seeds must reproduce the geometric fingerprint")
    _assert(list(first) == list(second), "identical seeds must reproduce the metadata records")
    _assert([p.ego_id for p in first] == ["ego-a", "ego-b", "ego-c"],
            "egos must be iterated in sorted order")
    for p in first:
        validate_placement(p, PARAMS)

    # A different seed must actually move the geometry (the fingerprint is not constant).
    other = place_hidden_targets(routes, tasks, LAUNCH, PARAMS, random.Random(99))
    _assert(geometric_fingerprint(other) != geometric_fingerprint(first),
            "a different seed must produce different geometry")


def test_po3_insertion_order_does_not_change_the_result() -> None:
    tasks, routes = _three_ego_case()
    forward = {k: routes[k] for k in ("ego-a", "ego-b", "ego-c")}
    backward = {k: routes[k] for k in ("ego-c", "ego-b", "ego-a")}
    _assert(list(forward.keys()) != list(backward.keys()), "the two dicts must differ in order")

    a = place_hidden_targets(forward, tasks, LAUNCH, PARAMS, random.Random(5))
    b = place_hidden_targets(backward, tasks, LAUNCH, PARAMS, random.Random(5))
    _assert(geometric_fingerprint(a) == geometric_fingerprint(b),
            "dict insertion order must not change the fingerprint")
    _assert(list(a) == list(b), "dict insertion order must not change the metadata")


# ---------------------------------------------------------------------------
# Loud-failure surface and dependency purity
# ---------------------------------------------------------------------------

def test_validation_rejects_malformed_inputs() -> None:
    good_tasks = _tasks(_dest(LAUNCH, 75.0, 300.0))
    plan = _plan(0)

    def _call(solution, tasks=good_tasks, launch=LAUNCH):
        return place_hidden_targets(solution, tasks, launch, PARAMS, random.Random(0))

    _raises(HiddenPlacementError, _call, {})                       # empty solution
    _raises(HiddenPlacementError, _call, {"ego": []})              # no usable route
    _raises(HiddenPlacementError, _call, {"ego": [(9, 0, 0)]})     # task index out of range
    _raises(HiddenPlacementError, _call, {"ego": [(0, 4, 0)]})     # step index out of range
    _raises(HiddenPlacementError, _call, {"ego": [(0, 0)]})        # malformed assignment

    # missing step location
    _raises(HiddenPlacementError, _call, {"ego": plan}, [_task_at(None, "nowhere")])
    # non-finite launch point
    _raises(HiddenPlacementError, _call, {"ego": plan}, good_tasks,
            Location(float("nan"), 35.0, 0))
    # out-of-range coordinates
    _raises(HiddenPlacementError, _call, {"ego": plan}, good_tasks, Location(120.0, 35.0, 0))


def test_f1_assignment_fields_must_be_integral() -> None:
    """Review fix F1: assignment fields are never coerced.

    `int((0.9, 0, 0)[0])` used to accept a malformed assignment AS task 0, silently
    changing the predicted route this layer measures against.
    """
    tasks = _tasks(_dest(LAUNCH, 75.0, 300.0))

    def _call(assignment):
        return place_hidden_targets(
            {"ego": [assignment]}, tasks, LAUNCH, PARAMS, random.Random(0)
        )

    # positive control: a normal integer tuple still works and is unchanged
    control = _call((0, 0, 0))[0]
    _assert(control.target_assignment == (0, 0, 0),
            f"valid tuples must survive untouched, got {control.target_assignment}")
    validate_placement(control, PARAMS)

    # Rejected in every field position: fractional float, bool, numeric string, and an
    # integral-VALUED float (0.0 / 1.0 are not integers either -- no coercion at all).
    for bad in ((0.9, 0, 0), (True, 0, 0), ("0", 0, 0),
                (0, 0.5, 0), (0, 0, "0"), (0.0, 0, 0), (0, 0, 1.0)):
        exc = _raises(HiddenPlacementError, _call, bad)
        _assert("non-integral" in str(exc), f"{bad!r}: unexpected message: {exc}")


def test_f2_later_leg_requires_the_recorded_tie_margin() -> None:
    """Review fix F2: `tie_margin_required_km` is validated before the branch.

    A single-candidate later leg used to return early, so a record could drop the
    requirement entirely and still be accepted.
    """
    # A legitimate single-candidate later leg (leg 2 of a 2-target route).
    t1 = _dest(LAUNCH, 90.0, 200.0)
    t2 = _dest(t1, 90.0, 400.0)
    single = place_hidden_targets(
        {"ego": _plan(0, 1)}, _tasks(t1, t2), LAUNCH, PARAMS, random.Random(11)
    )[0]
    _assert(single.leg_index == 2 and single.single_candidate and single.tie_gap_km is None,
            "fixture must be a single-candidate later leg")
    _assert(single.tie_margin_required_km == PARAMS.tie_margin_km,
            "control: the requirement is recorded")
    validate_placement(single, PARAMS)  # still passes with tie_gap_km None

    for tampered, why in (
        (None, "missing"),
        (float("nan"), "non-finite"),
        (55.0, "incorrect finite"),
        (PARAMS.tie_margin_km + 1.0, "off-by-one"),
    ):
        _raises(HiddenPlacementError, validate_placement,
                dataclasses.replace(single, tie_margin_required_km=tampered), PARAMS)

    # A later leg WITH a competitor is held to the same requirement, plus a real gap.
    t3 = _dest(t2, 0.0, 520.0)
    t4 = _dest(t3, 0.0, 40.0)
    contested = place_hidden_targets(
        {"ego": _plan(0, 1, 2, 3)}, _tasks(t1, t2, t3, t4), LAUNCH, PARAMS, random.Random(0)
    )[0]
    _assert(contested.leg_index == 2 and not contested.single_candidate
            and contested.tie_gap_km is not None, "fixture must be a contested later leg")
    validate_placement(contested, PARAMS)
    _raises(HiddenPlacementError, validate_placement,
            dataclasses.replace(contested, tie_margin_required_km=None), PARAMS)
    _raises(HiddenPlacementError, validate_placement,
            dataclasses.replace(contested, tie_gap_km=float("nan")), PARAMS)
    # a gap exactly ON the margin is not a passing gap
    _raises(HiddenPlacementError, validate_placement,
            dataclasses.replace(contested, tie_gap_km=PARAMS.tie_margin_km), PARAMS)
    # claiming "no competitor" while still carrying a gap is incoherent
    _raises(HiddenPlacementError, validate_placement,
            dataclasses.replace(contested, single_candidate=True), PARAMS)


def test_parameters_validation_rejects_bad_geometry() -> None:
    _raises(HiddenPlacementError, PlacementParameters(detection_km=0.0).validate)
    _raises(HiddenPlacementError, PlacementParameters(detection_km=-1.0).validate)
    _raises(HiddenPlacementError,
            PlacementParameters(detection_km=50.0, guard_km=50.0).validate)
    _raises(HiddenPlacementError,
            PlacementParameters(detection_km=50.0, guard_km=-1.0).validate)
    _raises(HiddenPlacementError,
            PlacementParameters(detection_km=50.0, fraction_min=0.9, fraction_max=0.5).validate)
    _raises(HiddenPlacementError,
            PlacementParameters(detection_km=50.0, fraction_min=0.0).validate)
    _raises(HiddenPlacementError,
            PlacementParameters(detection_km=50.0, fraction_max=1.5).validate)
    _raises(HiddenPlacementError,
            PlacementParameters(detection_km=float("inf")).validate)
    PARAMS.validate()  # the reference parameters stay valid
    _assert(PARAMS.tie_margin_km == 2 * DETECTION_KM, "the tie margin is 2 x detection_km")
    _assert(PARAMS.leg1_max_abs_offset_km == DETECTION_KM - GUARD_KM, "leg-1 cap is D - guard")


def test_rng_must_be_explicit_random() -> None:
    """Module-global randomness is not an option: the rng is an explicit dependency."""
    tasks = _tasks(_dest(LAUNCH, 75.0, 300.0))

    class _NotRandom:
        def random(self):
            return 0.5

        def choice(self, seq):
            return seq[0]

    _raises(HiddenPlacementError, place_hidden_targets,
            {"ego": _plan(0)}, tasks, LAUNCH, PARAMS, _NotRandom())


_PURITY_SENTINEL = "PLACEMENT_PURITY_JSON:"
_PURITY_CHILD = (
    "import sys, json, importlib\n"
    "importlib.import_module(sys.argv[1])\n"
    "print('%s' + json.dumps(sorted(sys.modules)))\n" % _PURITY_SENTINEL
)


def _import_closure(module_name: str) -> set:
    """Modules present in a FRESH interpreter after importing `module_name`."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, "-c", _PURITY_CHILD, module_name],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    _assert(proc.returncode == 0,
            f"child import of {module_name} failed:\n{proc.stdout}\n{proc.stderr}")
    line = next((l for l in proc.stdout.splitlines() if l.startswith(_PURITY_SENTINEL)), None)
    _assert(line is not None, f"no sentinel line for {module_name}:\n{proc.stdout}")
    return set(json.loads(line[len(_PURITY_SENTINEL):]))


def test_module_has_no_blade_torch_or_solver_dependency() -> None:
    """A fresh import of the placement module pulls in no simulator and no learner.

    `pyomo` is deliberately NOT in the banned set, and the exemption is proven rather
    than assumed: `src/match_aou/__init__.py` does `from .solvers import MatchAou`, so the
    ROOT package drags pyomo into every `match_aou.*` import -- all twelve
    `tests/test_import_purity.py` entry modules included. The control below imports plain
    `match_aou` and asserts pyomo is already there, so if the root package is ever made
    lazy this test starts failing and the exemption must be re-earned instead of silently
    covering a solver import added here.
    """
    modules = _import_closure("match_aou.rl.training.graph_hidden_placement")

    _assert("match_aou.rl.training.graph_hidden_placement" in modules,
            "positive control: the module itself must be imported")
    banned_roots = {"blade", "gymnasium", "gym", "torch"}
    leaked = sorted(m for m in modules if m.split(".")[0] in banned_roots)
    _assert(not leaked, f"placement module leaked heavy dependencies: {leaked}")
    for forbidden in (
        "match_aou.rl.training.graph_episode_setup",   # would re-introduce DETECTION_KM
        "match_aou.rl.training.graph_tick_loop",
        "match_aou.utils.blade_utils.blade_graph_executor",
    ):
        _assert(forbidden not in modules, f"placement module must not import {forbidden}")

    root_only = _import_closure("match_aou")
    _assert("pyomo" in root_only,
            "control failed: pyomo no longer comes from the root package, so the placement "
            "module's pyomo exemption is no longer justified")


# ---------------------------------------------------------------------------
# __main__ runner (pytest is absent from nlp_env -- CLAUDE.md Sec 1)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("po1_leg1_sweep_satisfies_sensing_and_guaranteed_portion",
         test_po1_leg1_sweep_satisfies_sensing_and_guaranteed_portion),
        ("po1_scripted_fraction_and_offset_boundaries",
         test_po1_scripted_fraction_and_offset_boundaries),
        ("po1_later_leg_has_reduced_offset_budget",
         test_po1_later_leg_has_reduced_offset_budget),
        ("po1_impossible_short_leg_raises",
         test_po1_impossible_short_leg_raises),
        ("po1_validate_placement_rejects_tampered_metadata",
         test_po1_validate_placement_rejects_tampered_metadata),
        ("po2_placement_lands_on_leg_two",
         test_po2_placement_lands_on_leg_two),
        ("po2_placement_lands_on_leg_three",
         test_po2_placement_lands_on_leg_three),
        ("po2_tie_margin_strict_at_two_detection_radii",
         test_po2_tie_margin_strict_at_two_detection_radii),
        ("po2_single_remaining_candidate_passes_trivially",
         test_po2_single_remaining_candidate_passes_trivially),
        ("po2_predicted_route_matches_executor_queue",
         test_po2_predicted_route_matches_executor_queue),
        ("po3_identical_seeds_reproduce_fingerprint_and_metadata",
         test_po3_identical_seeds_reproduce_fingerprint_and_metadata),
        ("po3_insertion_order_does_not_change_the_result",
         test_po3_insertion_order_does_not_change_the_result),
        ("validation_rejects_malformed_inputs",
         test_validation_rejects_malformed_inputs),
        ("f1_assignment_fields_must_be_integral",
         test_f1_assignment_fields_must_be_integral),
        ("f2_later_leg_requires_the_recorded_tie_margin",
         test_f2_later_leg_requires_the_recorded_tie_margin),
        ("parameters_validation_rejects_bad_geometry",
         test_parameters_validation_rejects_bad_geometry),
        ("rng_must_be_explicit_random",
         test_rng_must_be_explicit_random),
        ("module_has_no_blade_torch_or_solver_dependency",
         test_module_has_no_blade_torch_or_solver_dependency),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"GRAPH_HIDDEN_PLACEMENT TESTS: {failures} failed")
        sys.exit(1)
    print(f"GRAPH_HIDDEN_PLACEMENT TESTS: all {len(tests)} passed")
