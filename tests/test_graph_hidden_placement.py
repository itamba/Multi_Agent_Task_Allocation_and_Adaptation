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
       unstable, and that the predicted route is byte-identical to the route the CURRENT
       `GraphPlanExecutor` actually flies -- obtained by consuming its own `_eligible`
       level by level from a live position, not by re-calling the ordering helper (level
       grouping + chained start location included).

  PO3  REPRODUCIBILITY
       Identically seeded `random.Random` instances produce identical geometric
       fingerprints AND identical metadata records, regardless of the solution dict's
       insertion order. Fingerprints are coordinates only -- never target ids or uuids
       (CLAUDE.md Sec 8: generated ids are not seed-derived).

GENERALIZED-V1 adds two more, and they are deliberately about DIFFERENT things:

  G1   HISTORICAL EXACT-PATH PRESERVATION
       `place_hidden_targets` is byte-unchanged by the bounded-backoff addition. Pinned
       against values captured from the PRE-GENERALIZED implementation at base commit
       `7b86098a` -- fingerprints, chosen leg indices, ego order, sampled fractions,
       sampled offsets, AND the episode rng's stream position AFTER the call, so a
       changed geometry, a changed selection or even one extra/reordered draw fails.

  G2   DETERMINISTIC BOUNDED BACKOFF
       Candidates are driven by STABLE AGENT ORDINALS rather than id text (relabelling
       every ego changes nothing); the same seed reproduces the candidate order, the
       selected ordinals, the audit and the fingerprint; a candidate's FAILURE cannot
       shift a later candidate's geometry (substreams are derived up front); realizing
       fewer hidden targets than requested is reported truthfully; realizing none is
       refused; and no ego route is used twice. The bounded path is also shown to reuse
       the EXACT path's own single-route geometry and draw order, not a copy of it.

Fixture geometry is built with an independent spherical destination helper (`_dest`) and
every fixture asserts its own premises (which target is nearest, what the gaps are), so a
test that stops proving what it claims fails instead of passing vacuously.

Pure: no bonmin, no BLADE `Game`, no gymnasium env, no torch, no file I/O. The only
heavier import is `GraphPlanExecutor` (pure Python -- it duck-types the observation and
imports no simulator), used as the independent oracle for route prediction. The module
UNDER TEST still must not reach it: `test_module_has_no_blade_torch_or_solver_dependency`
checks the placement module's own import closure in a child process.

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
    BACKOFF_REJECTION_REASONS,
    EARTH_RADIUS_KM,
    HIDDEN_CARDINALITY_POLICIES,
    HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    HIDDEN_POLICY_EXACT_V1,
    REASON_NO_ELIGIBLE_LEG,
    REASON_NO_ROUTE,
    BoundedBackoffAudit,
    HiddenPlacement,
    HiddenPlacementError,
    PlacementParameters,
    _candidate_substream_seeds,
    _ordinal_permutation,
    geometric_fingerprint,
    place_hidden_targets,
    place_hidden_targets_bounded,
    predict_route,
    validate_placement,
)
from match_aou.utils.blade_utils.blade_graph_executor import (  # noqa: E402
    GraphPlanExecutor,
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


class _FakeAircraft:
    """The attributes `blade_graph_executor._live_location` reads off an observation."""

    def __init__(self, ego_id: str, lat: float, lon: float) -> None:
        self.id = ego_id
        self.latitude = float(lat)
        self.longitude = float(lon)


class _FakeScenario:
    """Minimal airborne-ego observation. No BLADE `Game`, no env, no stepping."""

    def __init__(self, ego_id: str, loc: Location) -> None:
        self.aircraft = [_FakeAircraft(ego_id, loc.latitude, loc.longitude)]
        self.airbases = []


def _executor_flown_route(tasks, assignments, ego_id: str, start: Location):
    """The order the CURRENT `GraphPlanExecutor` really flies, level by level.

    Independent of `predict_route`: it drives the executor's own `_eligible` exactly the
    way `_command_for_ego` does -- recompute eligibility from the ego's LIVE position,
    take the head, record the confirmed kill in `done` (the executor's sole advance
    signal), move the live position onto that target, repeat. So this reads the
    executor's real level-min gating and private per-ego task resolution rather than
    re-calling the shared ordering helper, which would make the comparison tautological.
    """
    agent = Agent(
        location=start,
        capabilities=[],
        budget=0.0,
        move_cost_function=lambda a, b: 0.0,
        agent_id=ego_id,
        return_location=start,
    )
    executor = GraphPlanExecutor(
        tasks=list(tasks),
        solution={ego_id: [tuple(a) for a in assignments]},
        agents=[agent],
        add_return_to_base=False,
    )
    flown = []
    scenario = _FakeScenario(ego_id, start)
    for _guard in range(1000):
        eligible = executor._eligible(ego_id, scenario)
        if not eligible:
            return flown
        head = eligible[0]
        step = executor._resolve_step(ego_id, head)
        _assert(step is not None, f"assignment {head} must resolve to a step")
        flown.append(tuple(head))
        executor.done.add((ego_id, str(step.target_id)))
        scenario = _FakeScenario(ego_id, step.location)
    raise AssertionError("executor route consumption did not terminate")


def test_po2_predicted_route_matches_executor_flown_route() -> None:
    """Prediction reproduces what the CURRENT executor flies: levels, chaining, ties."""
    locs = [
        _dest(LAUNCH, 90.0, 220.0),
        _dest(LAUNCH, 45.0, 260.0),
        _dest(LAUNCH, 135.0, 310.0),
        _dest(LAUNCH, 20.0, 480.0),
        _dest(LAUNCH, 300.0, 540.0),
    ]
    tasks = _tasks(*locs)
    assignments = [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 2)]

    expected = _executor_flown_route(tasks, assignments, "ego", LAUNCH)
    _assert(len(expected) == len(assignments),
            f"the executor must fly every assignment once, got {expected}")
    _assert(sorted(expected) == sorted(tuple(a) for a in assignments),
            f"the executor must fly exactly the planned SET, got {expected}")

    predicted = list(predict_route(assignments, tasks, LAUNCH))
    _assert(predicted == expected, f"route {predicted} != executor flown route {expected}")
    _assert([a[2] for a in predicted] == sorted(a[2] for a in predicted),
            "levels must stay ascending")

    # The fixture must be non-trivial: at least one level must be REORDERED away from
    # the plain (task_idx, step_idx) order, else agreement proves nothing.
    _assert(predicted != [tuple(a) for a in assignments],
            "fixture is vacuous: nearest-neighbor ordering did not reorder anything")

    # Insertion order of the assignment list must not matter -- on either side.
    shuffled = list(reversed(assignments))
    _assert(list(predict_route(shuffled, tasks, LAUNCH)) == expected,
            "route prediction must not depend on assignment list order")
    _assert(_executor_flown_route(tasks, shuffled, "ego", LAUNCH) == expected,
            "executor execution must not depend on assignment list order")


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
# GENERALIZED-V1 fixtures
# ---------------------------------------------------------------------------

# The EXACT fixture the G1 golden values below were captured on, at base commit
# `7b86098a7573be15b0d8bfcf959b1d1f63288ffc`. Three egos, two of them with two-leg
# routes, so leg CHOICE as well as fraction and offset are all exercised.
_G1_T0 = _dest(LAUNCH, 20.0, 300.0)
_G1_T1 = _dest(_G1_T0, 20.0, 400.0)
_G1_T2 = _dest(LAUNCH, 75.0, 320.0)
_G1_T3 = _dest(_G1_T2, 70.0, 380.0)
_G1_T4 = _dest(LAUNCH, 150.0, 280.0)
_G1_TASKS = [
    _task_at(_G1_T0, "t0"), _task_at(_G1_T1, "t1"), _task_at(_G1_T2, "t2"),
    _task_at(_G1_T3, "t3"), _task_at(_G1_T4, "t4"),
]
_G1_SOLUTION = {
    "ego-a": [(0, 0, 0), (1, 0, 1)],
    "ego-b": [(2, 0, 0), (3, 0, 1)],
    "ego-c": [(4, 0, 0)],
}

# Captured from the PRE-GENERALIZED implementation. `rng_after` is the very next
# `rng.random()` the episode rng yields once `place_hidden_targets` has returned: it pins
# the number and ORDER of draws, so an extra, missing or reordered draw fails here even
# when the coordinates happen to survive.
_G1_GOLDEN = {
    0: {
        "fingerprint": ((37.702438324, 37.559472451), (34.370028122, 41.494251143),
                        (31.480639012, 36.559342226)),
        "legs": (2, 2, 1),
        "fractions": (0.7894886007350757, 0.8413662215904792, 0.8295585829462829),
        "offsets": (-3.8983668736067316, -0.7467610509317638, 26.388232292719316),
        "rng_after": 0.9677999949201714,
    },
    1: {
        "fingerprint": ((37.618693088, 37.307543469), (34.206981215, 40.887571666),
                        (31.628814058, 36.28855396)),
        "legs": (2, 2, 1),
        "fractions": (0.7423009687055531, 0.6637672564348553, 0.7123727661971845),
        "offsets": (13.58721795134759, -0.17183676988074426, 12.127437817821036),
        "rng_after": 0.7887233511355132,
    },
    7: {
        "fingerprint": ((37.836836997, 37.644370735), (34.175072126, 40.727658113),
                        (31.455539192, 35.827738607)),
        "legs": (2, 2, 1),
        "fractions": (0.8369663401522658, 0.6181090716668857, 0.6914222292281463),
        "offsets": (-5.5990243394754335, 1.2084303040275586, -35.36008601802345),
        "rng_after": 0.5074357331894203,
    },
    12345: {
        "fingerprint": ((37.745320929, 37.343775616), (34.537119973, 41.321960087),
                        (31.63307457, 36.076557576)),
        "legs": (2, 2, 1),
        "fractions": (0.7831713188629023, 0.8137845854617872, 0.6675612102560851),
        "offsets": (15.53727577557694, 21.878861306398885, -5.101727150864708),
        "rng_after": 0.3730638408978796,
    },
}


def _long_leg(bearing_deg: float, distance_km: float = 300.0):
    """A single-target route whose one leg comfortably carries a placement."""
    return _dest(LAUNCH, bearing_deg, distance_km)


def _backoff_world():
    """Four independent single-leg ego routes, all geometrically usable.

    Ordinal `i` owns task `i`, so ordinal -> route is fixed while the ego LABELS stay
    free -- which is what lets the ordinal-vs-id-text tests vary one without the other.
    """
    tasks = [
        _task_at(_long_leg(20.0), "t0"),
        _task_at(_long_leg(75.0, 320.0), "t1"),
        _task_at(_long_leg(150.0, 280.0), "t2"),
        _task_at(_long_leg(250.0, 340.0), "t3"),
    ]
    return tasks


# A route whose only leg is far shorter than the detection radius: G = L - D <= 0, so the
# approved geometry refuses it. Used as the "this candidate fails" arm.
_UNUSABLE_TASK = _task_at(_dest(LAUNCH, 200.0, 20.0), "unusable")


# ---------------------------------------------------------------------------
# G1 -- the historical exact path is byte-unchanged
# ---------------------------------------------------------------------------

def test_g1_exact_path_is_pinned_to_the_pre_generalized_geometry() -> None:
    """PO1. `place_hidden_targets` reproduces the pre-GENERALIZED-V1 values exactly.

    Geometry, chosen legs, ego ORDER, the sampled fraction and offset of every placement,
    and the episode rng's post-call stream position -- all captured from base commit
    `7b86098a` before the bounded-backoff addition existed. Extracting the shared
    `_select_leg` helper must not have moved a single draw.
    """
    for seed, golden in sorted(_G1_GOLDEN.items()):
        rng = random.Random(seed)
        placements = place_hidden_targets(_G1_SOLUTION, _G1_TASKS, LAUNCH, PARAMS, rng)

        _assert(len(placements) == 3, f"seed {seed}: expected 3 placements")
        _assert([p.ego_id for p in placements] == ["ego-a", "ego-b", "ego-c"],
                f"seed {seed}: ego order changed -> {[p.ego_id for p in placements]}")
        _assert(geometric_fingerprint(placements) == golden["fingerprint"],
                f"seed {seed}: fingerprint moved -> {geometric_fingerprint(placements)}")
        _assert(tuple(p.leg_index for p in placements) == golden["legs"],
                f"seed {seed}: leg selection moved -> "
                f"{tuple(p.leg_index for p in placements)}")
        _assert(tuple(p.fraction for p in placements) == golden["fractions"],
                f"seed {seed}: fractions moved -> {tuple(p.fraction for p in placements)}")
        _assert(tuple(p.offset_km for p in placements) == golden["offsets"],
                f"seed {seed}: offsets moved -> {tuple(p.offset_km for p in placements)}")
        # THE DRAW-COUNT PIN: one extra, missing or reordered draw shows up here.
        _assert(rng.random() == golden["rng_after"],
                f"seed {seed}: the rng stream position after the call changed")


def test_g1_exact_path_keeps_its_loud_no_route_failure() -> None:
    """PO1. An ego without a route still fails LOUDLY on the exact path.

    That refusal is the whole difference between the two policies, so it must survive the
    addition of one that backs off instead.
    """
    tasks = _backoff_world()
    for empty in ([], None):
        exc = _raises(
            HiddenPlacementError, place_hidden_targets,
            {"ego-a": _plan(0), "ego-b": empty}, tasks, LAUNCH, PARAMS, random.Random(0),
        )
        _assert("no usable assigned route" in str(exc), str(exc))

    # And an ego whose ONLY leg is geometrically impossible still raises rather than
    # being quietly skipped.
    exc = _raises(
        HiddenPlacementError, place_hidden_targets,
        {"ego-a": _plan(0)}, [_UNUSABLE_TASK], LAUNCH, PARAMS, random.Random(0),
    )
    _assert("leg 1 is unusable" in str(exc), str(exc))


# ---------------------------------------------------------------------------
# G2 -- deterministic bounded backoff
# ---------------------------------------------------------------------------

def test_g2_policy_ids_are_explicit() -> None:
    """The two policies are named, distinct and both registered."""
    _assert(HIDDEN_POLICY_EXACT_V1 == "exact_v1", HIDDEN_POLICY_EXACT_V1)
    _assert(HIDDEN_POLICY_BOUNDED_BACKOFF_V1 == "bounded_backoff_v1",
            HIDDEN_POLICY_BOUNDED_BACKOFF_V1)
    _assert(HIDDEN_CARDINALITY_POLICIES
            == (HIDDEN_POLICY_EXACT_V1, HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
            HIDDEN_CARDINALITY_POLICIES)


def test_g2_reuses_the_exact_paths_single_route_geometry() -> None:
    """PO2. The bounded walk delegates to the SAME geometry AND the same draw order.

    Re-derived independently: take the documented ordinal permutation and substream
    seeds from a freshly seeded rng, then run the HISTORICAL `place_hidden_targets` on the
    one candidate with a `random.Random` on that substream seed. The two records must be
    field-for-field identical -- which they can only be if the bounded path drew the leg,
    then the fraction, then the offset, exactly as the exact path does.
    """
    tasks = _backoff_world()
    solution = {"solo": _plan(0)}

    placements, audit = place_hidden_targets_bounded(
        solution, tasks, LAUNCH, PARAMS, random.Random(4242),
        agent_ordinals=["solo"], hidden_requested=1,
    )

    probe = random.Random(4242)
    order = _ordinal_permutation(1, probe)
    seeds = _candidate_substream_seeds(1, probe)
    _assert(order == (0,), order)
    expected = place_hidden_targets(
        solution, tasks, LAUNCH, PARAMS, random.Random(seeds[0])
    )

    _assert(dataclasses.asdict(placements[0]) == dataclasses.asdict(expected[0]),
            "the bounded path did not reproduce the exact path's single-route geometry")
    _assert(audit.policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1, audit.policy)
    _assert(audit.hidden_realized == audit.hidden_requested == 1, audit)
    validate_placement(placements[0], PARAMS)


def test_g2_candidates_follow_stable_ordinals_not_id_text() -> None:
    """PO2. Relabelling every ego changes NOTHING about the ordinal/geometric result.

    The two rosters have deliberately OPPOSITE lexical orders while keeping the same
    ordinal -> route mapping, so a candidate walk driven by sorted id text would produce a
    different selection. Generated ids are not seed-derived (CLAUDE.md Sec 8), which is
    exactly why ordering may never key on them.
    """
    tasks = _backoff_world()

    labels_a = ["zz", "yy", "xx", "ww"]          # lexically DESCENDING
    labels_b = ["aa", "bb", "cc", "dd"]          # lexically ASCENDING
    _assert(sorted(labels_a) == list(reversed(labels_a)), labels_a)
    _assert(sorted(labels_b) == list(labels_b), labels_b)

    results = []
    for labels in (labels_a, labels_b):
        solution = {label: _plan(i) for i, label in enumerate(labels)}
        results.append(
            place_hidden_targets_bounded(
                solution, tasks, LAUNCH, PARAMS, random.Random(11),
                agent_ordinals=labels, hidden_requested=3,
            )
        )

    (pl_a, au_a), (pl_b, au_b) = results
    _assert(au_a.candidate_order == au_b.candidate_order, "candidate order followed labels")
    _assert(au_a.selected_ordinals == au_b.selected_ordinals,
            f"selection followed labels: {au_a.selected_ordinals} vs "
            f"{au_b.selected_ordinals}")
    _assert(au_a.geometric_fingerprint == au_b.geometric_fingerprint,
            "geometry followed labels")
    for a, b in zip(pl_a, pl_b):
        stripped_a = {k: v for k, v in dataclasses.asdict(a).items() if k != "ego_id"}
        stripped_b = {k: v for k, v in dataclasses.asdict(b).items() if k != "ego_id"}
        _assert(stripped_a == stripped_b, "a placement record followed the ego label")
    # The labels really did differ -- otherwise this proves nothing.
    _assert([p.ego_id for p in pl_a] != [p.ego_id for p in pl_b],
            "the two rosters used the same labels, so nothing was varied")


def test_g2_same_seed_reproduces_order_selection_and_audit() -> None:
    """PO2. An episode's bounded walk is a pure function of its seed."""
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    solution = {label: _plan(i) for i, label in enumerate(labels)}

    def _run(seed: int):
        return place_hidden_targets_bounded(
            solution, tasks, LAUNCH, PARAMS, random.Random(seed),
            agent_ordinals=labels, hidden_requested=2,
        )

    pl_1, au_1 = _run(2026)
    pl_2, au_2 = _run(2026)
    _assert(dataclasses.asdict(au_1) == dataclasses.asdict(au_2), "audit is not reproducible")
    _assert([dataclasses.asdict(p) for p in pl_1] == [dataclasses.asdict(p) for p in pl_2],
            "placements are not reproducible")

    # A DIFFERENT seed really does move the walk (otherwise the rng drives nothing).
    moved = False
    for other in range(50):
        _, au_other = _run(other)
        if (au_other.candidate_order != au_1.candidate_order
                or au_other.geometric_fingerprint != au_1.geometric_fingerprint):
            moved = True
            break
    _assert(moved, "no seed in the sweep changed the walk; the rng is not driving it")


def test_g2_a_failed_candidate_cannot_shift_a_later_ones_geometry() -> None:
    """PO2. Per-candidate substreams are derived UP FRONT, before any attempt.

    Baseline: every candidate succeeds. Variant: the FIRST-VISITED candidate's route is
    replaced by a geometrically impossible one. The candidate visited SECOND must produce
    a byte-identical placement in both runs -- which is only true if its geometry stream
    was fixed before the first candidate's outcome was known.
    """
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    seed = 909

    baseline_solution = {label: _plan(i) for i, label in enumerate(labels)}
    baseline_pl, baseline_au = place_hidden_targets_bounded(
        baseline_solution, tasks, LAUNCH, PARAMS, random.Random(seed),
        agent_ordinals=labels, hidden_requested=4,
    )
    _assert(baseline_au.hidden_realized == 4, baseline_au)

    first, second = baseline_au.candidate_order[0], baseline_au.candidate_order[1]
    baseline_second = next(p for p in baseline_pl if p.ego_id == labels[second])

    # Break ONLY the first-visited candidate: its task becomes an unusable short leg.
    broken_tasks = list(tasks)
    broken_tasks[first] = _UNUSABLE_TASK
    variant_pl, variant_au = place_hidden_targets_bounded(
        baseline_solution, broken_tasks, LAUNCH, PARAMS, random.Random(seed),
        agent_ordinals=labels, hidden_requested=4,
    )

    rejected = [c for c in variant_au.candidates if c.ordinal == first]
    _assert(len(rejected) == 1 and not rejected[0].accepted,
            "the first-visited candidate was supposed to fail")
    _assert(rejected[0].reason == REASON_NO_ELIGIBLE_LEG, rejected[0].reason)
    _assert(variant_au.hidden_realized == 3, variant_au)

    variant_second = next(p for p in variant_pl if p.ego_id == labels[second])
    _assert(dataclasses.asdict(variant_second) == dataclasses.asdict(baseline_second),
            "a candidate's failure shifted a later candidate's geometry")


def test_g2_rng_stream_position_depends_only_on_the_candidate_count() -> None:
    """PO2. The episode rng ends in the same place however the walk went.

    Both runs share one roster size, so both consume exactly the same permutation draws
    and the same burst of substream seeds -- and nothing else.
    """
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]

    def _end_position(task_list, requested: int) -> float:
        rng = random.Random(77)
        place_hidden_targets_bounded(
            {label: _plan(i) for i, label in enumerate(labels)},
            task_list, LAUNCH, PARAMS, rng,
            agent_ordinals=labels, hidden_requested=requested,
        )
        return rng.random()

    broken = list(tasks)
    broken[0] = _UNUSABLE_TASK
    _assert(_end_position(tasks, 4) == _end_position(broken, 4),
            "a rejection moved the episode rng's stream position")
    _assert(_end_position(tasks, 1) == _end_position(tasks, 4),
            "the requested count moved the episode rng's stream position")


def test_g2_realizes_fewer_than_requested_and_says_so() -> None:
    """PO2. A short world is a truthful RECORDED outcome, never a repaired one."""
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    # Only two egos were allocated a route by the (allocated-only) solve.
    solution = {"e1": _plan(1), "e2": _plan(2)}

    placements, audit = place_hidden_targets_bounded(
        solution, tasks, LAUNCH, PARAMS, random.Random(5),
        agent_ordinals=labels, hidden_requested=4,
    )

    _assert(audit.hidden_requested == 4, audit.hidden_requested)
    _assert(audit.hidden_realized == 2 == len(placements), audit.hidden_realized)
    _assert(audit.realized_full_request is False, audit)
    _assert(set(audit.selected_ordinals) == {1, 2}, audit.selected_ordinals)
    # Every candidate is accounted for, and the omitted egos are named as no_route.
    _assert(tuple(c.ordinal for c in audit.candidates) == audit.considered_ordinals,
            "candidate records and the considered list disagree")
    _assert(audit.considered_ordinals == audit.candidate_order,
            "an exhausted walk must have visited every candidate")
    no_route = {c.ordinal for c in audit.candidates if c.reason == REASON_NO_ROUTE}
    _assert(no_route == {0, 3}, no_route)
    for candidate in audit.candidates:
        _assert(candidate.accepted == (candidate.reason is None), candidate)
        _assert(candidate.reason is None
                or candidate.reason in BACKOFF_REJECTION_REASONS, candidate)
    # And the request itself was never rewritten to match what was possible.
    _assert(audit.as_dict()["hidden_requested"] == 4, audit.as_dict())


def test_g2_the_walk_is_bounded_and_stops_at_the_request() -> None:
    """PO2. Meeting the request ends the walk; the rest is genuinely never visited."""
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    solution = {label: _plan(i) for i, label in enumerate(labels)}

    placements, audit = place_hidden_targets_bounded(
        solution, tasks, LAUNCH, PARAMS, random.Random(31),
        agent_ordinals=labels, hidden_requested=2,
    )
    _assert(len(placements) == 2, len(placements))
    _assert(audit.considered_ordinals == audit.candidate_order[:2], audit.considered_ordinals)
    _assert(len(audit.candidates) == 2, audit.candidates)


def test_g2_no_ego_route_is_used_twice() -> None:
    """PO2. One hidden target per ego route -- multiples are out of scope in v1."""
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    solution = {label: _plan(i) for i, label in enumerate(labels)}

    for seed in range(40):
        placements, audit = place_hidden_targets_bounded(
            solution, tasks, LAUNCH, PARAMS, random.Random(seed),
            agent_ordinals=labels, hidden_requested=4,
        )
        ordinals = list(audit.selected_ordinals)
        _assert(len(ordinals) == len(set(ordinals)), f"seed {seed}: repeated ordinal")
        egos = [p.ego_id for p in placements]
        _assert(len(egos) == len(set(egos)), f"seed {seed}: repeated ego {egos}")
        _assert(sorted(audit.candidate_order) == [0, 1, 2, 3],
                f"seed {seed}: candidate order is not a permutation")
        for placement in placements:
            validate_placement(placement, PARAMS)


def test_g2_zero_realizable_hidden_targets_is_refused() -> None:
    """PO2. `H_realized == 0` is a refusal -- an accepted world needs a hidden half."""
    labels = ["e0", "e1"]
    tasks = [_UNUSABLE_TASK, _UNUSABLE_TASK]
    solution = {label: _plan(i) for i, label in enumerate(labels)}

    exc = _raises(
        HiddenPlacementError, place_hidden_targets_bounded,
        solution, tasks, LAUNCH, PARAMS, random.Random(0),
        agent_ordinals=labels, hidden_requested=2,
    )
    _assert("realized 0 hidden targets" in str(exc), str(exc))

    # Also when the solve allocated nothing at all: every candidate is `no_route`.
    exc = _raises(
        HiddenPlacementError, place_hidden_targets_bounded,
        {}, _backoff_world(), LAUNCH, PARAMS, random.Random(0),
        agent_ordinals=labels, hidden_requested=1,
    )
    _assert(REASON_NO_ROUTE in str(exc), str(exc))


def test_g2_solver_omitted_egos_are_still_candidates() -> None:
    """PO2. The candidate population is the SCHEDULED roster, not `A_init`'s keys.

    `A_init` is allocated-only, so an ego it omits is invisible to `solution.keys()` while
    still being a real scheduled ego. It has no route, and it is RECORDED as such rather
    than silently vanishing from the accounting.
    """
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    solution = {"e2": _plan(2)}

    placements, audit = place_hidden_targets_bounded(
        solution, tasks, LAUNCH, PARAMS, random.Random(8),
        agent_ordinals=labels, hidden_requested=1,
    )
    _assert(audit.candidate_count == 4, audit.candidate_count)
    _assert(audit.selected_ordinals == (2,), audit.selected_ordinals)
    _assert(len(placements) == 1 and placements[0].ego_id == "e2", placements)
    recorded = {c.ordinal for c in audit.candidates}
    _assert(recorded <= {0, 1, 2, 3} and 2 in recorded, recorded)


def test_g2_audit_is_typed_and_json_ready() -> None:
    """The accounting is a structure, not a console line: it survives serialization."""
    tasks = _backoff_world()
    labels = ["e0", "e1", "e2", "e3"]
    solution = {"e0": _plan(0), "e3": _plan(3)}

    _placements, audit = place_hidden_targets_bounded(
        solution, tasks, LAUNCH, PARAMS, random.Random(6),
        agent_ordinals=labels, hidden_requested=3,
    )
    _assert(isinstance(audit, BoundedBackoffAudit), type(audit))
    payload = audit.as_dict()
    round_tripped = json.loads(json.dumps(payload))
    _assert(round_tripped == payload, "the audit is not JSON-round-trippable")
    for key in ("policy", "candidate_count", "candidate_order", "considered_ordinals",
                "candidates", "selected_ordinals", "hidden_requested", "hidden_realized",
                "realized_full_request", "geometric_fingerprint"):
        _assert(key in payload, f"audit is missing {key}")
    _assert(payload["geometric_fingerprint"]
            == [list(fp) for fp in geometric_fingerprint(_placements)], payload)


def test_g2_input_validation_is_loud() -> None:
    """PO2. Malformed requests are refused, never silently normalized."""
    tasks = _backoff_world()
    labels = ["e0", "e1"]
    solution = {"e0": _plan(0), "e1": _plan(1)}
    ok = dict(agent_ordinals=labels, hidden_requested=1)

    # The rng must be an explicit Random -- module-global randomness is unreproducible.
    for bad_rng in (0, "rng", random, random.SystemRandom):
        _raises(HiddenPlacementError, place_hidden_targets_bounded,
                solution, tasks, LAUNCH, PARAMS, bad_rng, **ok)

    # hidden_requested: a genuine integer >= 1 (bool rejected despite subclassing int).
    for bad in (0, -1, 1.0, "1", True, False, None):
        _raises(HiddenPlacementError, place_hidden_targets_bounded,
                solution, tasks, LAUNCH, PARAMS, random.Random(0),
                agent_ordinals=labels, hidden_requested=bad)

    # The ordinal roster must exist and address exactly one ego per ordinal.
    for bad_roster in ([], ["e0", "e0"]):
        _raises(HiddenPlacementError, place_hidden_targets_bounded,
                solution, tasks, LAUNCH, PARAMS, random.Random(0),
                agent_ordinals=bad_roster, hidden_requested=1)

    # A solution naming an ego outside the roster means the roster is not authoritative.
    _raises(HiddenPlacementError, place_hidden_targets_bounded,
            {"ghost": _plan(0)}, tasks, LAUNCH, PARAMS, random.Random(0), **ok)

    # A non-integral assignment tuple is a caller contract violation, NOT a backoff case:
    # it raises instead of being recorded as a rejected candidate.
    _raises(HiddenPlacementError, place_hidden_targets_bounded,
            {"e0": [(0.9, 0, 0)]}, tasks, LAUNCH, PARAMS, random.Random(0), **ok)


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
        ("po2_predicted_route_matches_executor_flown_route",
         test_po2_predicted_route_matches_executor_flown_route),
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
        ("g1_exact_path_is_pinned_to_the_pre_generalized_geometry",
         test_g1_exact_path_is_pinned_to_the_pre_generalized_geometry),
        ("g1_exact_path_keeps_its_loud_no_route_failure",
         test_g1_exact_path_keeps_its_loud_no_route_failure),
        ("g2_policy_ids_are_explicit",
         test_g2_policy_ids_are_explicit),
        ("g2_reuses_the_exact_paths_single_route_geometry",
         test_g2_reuses_the_exact_paths_single_route_geometry),
        ("g2_candidates_follow_stable_ordinals_not_id_text",
         test_g2_candidates_follow_stable_ordinals_not_id_text),
        ("g2_same_seed_reproduces_order_selection_and_audit",
         test_g2_same_seed_reproduces_order_selection_and_audit),
        ("g2_a_failed_candidate_cannot_shift_a_later_ones_geometry",
         test_g2_a_failed_candidate_cannot_shift_a_later_ones_geometry),
        ("g2_rng_stream_position_depends_only_on_the_candidate_count",
         test_g2_rng_stream_position_depends_only_on_the_candidate_count),
        ("g2_realizes_fewer_than_requested_and_says_so",
         test_g2_realizes_fewer_than_requested_and_says_so),
        ("g2_the_walk_is_bounded_and_stops_at_the_request",
         test_g2_the_walk_is_bounded_and_stops_at_the_request),
        ("g2_no_ego_route_is_used_twice",
         test_g2_no_ego_route_is_used_twice),
        ("g2_zero_realizable_hidden_targets_is_refused",
         test_g2_zero_realizable_hidden_targets_is_refused),
        ("g2_solver_omitted_egos_are_still_candidates",
         test_g2_solver_omitted_egos_are_still_candidates),
        ("g2_audit_is_typed_and_json_ready",
         test_g2_audit_is_typed_and_json_ready),
        ("g2_input_validation_is_loud",
         test_g2_input_validation_is_loud),
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
