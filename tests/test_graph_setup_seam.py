"""
Tests for the B3 SETUP SEAM -- `setup_episode`'s construction path.

    known-only world -> env-1 -> solve A_init -> B2 placement -> patch the scenario JSON
      -> close env-1 -> env-2 reload -> re-extract -> solve the full oracle
      -> beliefs / executor built EXCLUSIVELY from env-2 objects

The module is split into three tiers by what each tier needs, so the cheap proofs stay
cheap and the expensive ones are still reachable:

  PURE      -- no BLADE, no solver. Argument pairing, the airbase-only guard, prototype
               selection, the JSON patch, agent-id preservation, and the known-task
               re-materialization are all plain functions on plain data, so every
               failure mode of the seam is provable without paying for a solve.
  BLADE     -- needs the engine but NOT bonmin. Environment ownership, on both sides of
               `_build_env`'s return: a `reset()` failure INSIDE the builder must close
               the environment the builder had already created, and an injected failure
               between "env-1 is up" and "A_init exists" must still close env-1. Plus P3:
               what an ego-global SELF_PRESERVATION_ABORT does to a REAL aircraft that is
               already flying a stale mission route.
  SOLVER    -- needs BLADE *and* bonmin, so it runs under `nlp_env` only (P1 / P2).
               `conda run -n nlp_env --no-capture-output python tests/test_graph_setup_seam.py`

PROOF OBLIGATIONS THIS MODULE CARRIES
-------------------------------------
P1  patch/reload + reproducible source of truth: known targets preserved in order and
    identity, exactly `n_hidden` enemy airbases added, runtime agents/tasks re-extracted
    from env-2, the oracle solved from env-2 and containing every hidden target, A_init's
    positional indices still valid against the re-materialized env-2 tasks, the SAME
    geometric fingerprint for a repeated setup at the same placement seed (never compared
    by uuid), and env-1 closed on both the success and the injected-failure path.
P3  real-BLADE stale-route replacement (no solver): an airborne ego flying a real
    executor-issued mission route has its plan emptied by the ego-global abort; the real
    `GraphPlanExecutor` empty-plan branch emits exactly ONE `aircraft_return_to_base`,
    and applying it through the real `Game` action seam sets `rtb`, DROPS the stale
    waypoint and leaves a route that ends at the aircraft's actual home base.
P4  the DERIVED attack-confirmation wait (pure): the wait armed after a salvo comes from
    the weapon the engine's own 2-arg attack will select and the engagement distance the
    executor already computed; the travel term is a CONSERVATIVE full-distance bound in
    the engine's units (km -> nm -> knots -> seconds, ceiled) plus a one-tick margin, NOT
    a reconstruction of BLADE's discrete launch/update/endgame schedule; the configured
    `kill_confirm_ticks` stays the floor and the fallback; unusable weapon/speed data is
    refused rather than guessed; a peer cannot move the result; and the per-(ego, target)
    cooldown identity and the confirmed-kill bypass are exactly as they were.
P5  real-BLADE regression (no solver): with the loadout in the state the first short probe
    reached, a single AIM-9 salvo is left to fly to completion instead of being re-fired
    over -- the AGM-65 reserve survives, the target still dies, and the plan still advances
    the moment the kill is confirmed. A control arm restoring the retired flat wait
    exhibits the same premature-re-fire mechanism the probe hit. The BLADE tier is also
    where the executor's transcribed km -> nm constant is compared against the ENGINE'S
    OWN `blade.utils.constants` value.
P6  world truth vs oracle allocation (pure): both setup paths snapshot the t=0 target
    roster from the RAW pre-solve task sets, so a target the solver leaves UNSELECTED --
    absent from the allocated-only `belief_tasks` / `oracle_tasks` by contract -- is still
    in `EpisodeContext.known_target_ids` / `executed_target_ids`. That separation is what
    stops an allocation from being read as a world inventory.
P2  private sensing isolation, through the INTEGRATED setup/tick seam: a hidden target
    that the setup really put in the world is reported as sensed by ONE ego, and the real
    `run_episode` Phase-1 chain is what carries it into that ego's belief and executor
    slice. Every peer belief and executor slice must be byte-unchanged and must not
    contain the target -- which it never could, because no belief held it at t=0.
P7  GENERALIZED-V1 cardinality + accounting: the hidden-CARDINALITY policy is EXPLICIT and
    defaults to the historical `exact_v1`; selecting `bounded_backoff_v1` without the
    construction pair is refused rather than ignored; the generalized cell (A in {2,3,4},
    K == A RAW known targets, 1 <= H_requested <= A) is enforced BEFORE anything is
    solved; the JSON patch adds exactly H_REALIZED targets; env-2 stays the sole
    authority; the REQUESTED counts survive a backoff instead of being rewritten to what
    was possible; `known_target_ids` / `executed_target_ids` reconcile with the audit
    while the allocated-only `oracle_tasks` deliberately does not; and NO accounting
    quantity reaches `GraphObservation`.

`pytest` is absent from `nlp_env` (CLAUDE.md section 1), so this module keeps a
`__main__` runner that executes every tier directly.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so match_aou.* imports resolve

from match_aou.models import Location, Step, StepKind, Task  # noqa: E402
from match_aou.rl.training import graph_episode_setup as _setup  # noqa: E402
from match_aou.rl.training.graph_reward import (  # noqa: E402
    REFERENCE_POLICY_STATIC_T0_V1,
)
from match_aou.rl.training.graph_episode_setup import (  # noqa: E402
    ATTACKING_SIDE_COLOR,
    CONSTRUCTION_TARGET_CLASS,
    DETECTION_KM,
    GENERALIZED_AGENT_COUNTS,
    HIDDEN_TARGET_NAME_TEMPLATE,
    MAX_SIM_TICKS,
    ConstructionAudit,
    EpisodeContext,
    _require_generalized_cardinality,
    _finish_context,
    _rematerialize_known_tasks,
    _require_agent_ids_preserved,
    _require_airbase_only_targets,
    _resolve_construction_mode,
    _select_hidden_prototype,
    _shared_launch_point,
    _task_target_id,
    _world_target_ids,
    build_patched_scenario,
    setup_episode,
)
from match_aou.rl.training.graph_hidden_placement import (  # noqa: E402
    BACKOFF_REJECTION_REASONS,
    HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    HIDDEN_POLICY_EXACT_V1,
    REASON_NO_ROUTE,
    BackoffCandidate,
    BoundedBackoffAudit,
    HiddenPlacement,
    geometric_fingerprint,
)

BASE_SCENARIO = ROOT / "data" / "scenarios" / "strike_training_4v5.json"


# =============================================================================
# Capability probes -- what this interpreter can actually run
# =============================================================================

def _have_blade() -> bool:
    try:
        import blade  # noqa: F401
        import gymnasium  # noqa: F401
    except Exception:  # pragma: no cover - environment-dependent
        return False
    return True


def _have_bonmin() -> bool:
    return shutil.which("bonmin") is not None


HAVE_BLADE = _have_blade()
HAVE_SOLVER = HAVE_BLADE and _have_bonmin()

_needs_blade = (
    pytest.mark.skipif(not HAVE_BLADE, reason="needs the BLADE engine")
    if pytest is not None else (lambda f: f)
)
_needs_solver = (
    pytest.mark.skipif(
        not HAVE_SOLVER, reason="needs BLADE + bonmin (run under nlp_env)"
    )
    if pytest is not None else (lambda f: f)
)


# =============================================================================
# Hand-built fixtures (no engine, no solver)
# =============================================================================

class _StubUnit:
    """Base for the duck-typed BLADE units `iter_enemy_targets` walks."""

    def __init__(self, uid: str, lat: float, lon: float, side: str) -> None:
        self.id, self.latitude, self.longitude = uid, lat, lon
        self.side_color, self.altitude = side, 0


class Airbase(_StubUnit):
    """Class NAME matters: the airbase-only guard reads `type(unit).__name__`."""


class Facility(_StubUnit):
    pass


class Ship(_StubUnit):
    pass


class _StubScenario:
    def __init__(self, facilities: List[Any], airbases: List[Any], ships: List[Any]) -> None:
        self.facilities, self.airbases, self.ships = facilities, airbases, ships

    def get_target(self, target_id: str) -> Optional[Any]:
        for unit in (self.facilities + self.airbases + self.ships):
            if str(unit.id) == str(target_id):
                return unit
        return None


class _StubAgent:
    def __init__(self, uid: str, loc: Location, ret: Optional[Location]) -> None:
        self.id, self.location, self.return_location = uid, loc, ret


def _task(target_id: str, lat: float = 32.0, lon: float = 35.0) -> Task:
    return Task(
        steps=[Step(Location(lat, lon), target_id, [], 1.0, 2, StepKind.ATTACK)],
        utility=80,
    )


def _airbase_entry(
    uid: str, name: str, side_id: str, side_color: str, *, aircraft: Optional[List] = None
) -> Dict[str, Any]:
    return {
        "id": uid, "name": name, "sideId": side_id, "className": "Airfield",
        "latitude": 33.0, "longitude": 35.5, "altitude": 0,
        "sideColor": side_color, "aircraft": list(aircraft or []),
    }


def _scenario_json(airbases: List[Dict[str, Any]]) -> str:
    return json.dumps({"currentScenario": {"id": "s", "name": "n", "airbases": airbases}})


def _placement(lat: float, lon: float, ego: str = "e") -> HiddenPlacement:
    """A minimal record: the patch step reads only `latitude` / `longitude`."""
    return HiddenPlacement(
        ego_id=ego, leg_index=1, origin_assignment=None, target_assignment=(0, 0, 0),
        origin_latitude=32.0, origin_longitude=35.0,
        target_latitude=34.0, target_longitude=36.0,
        leg_length_km=300.0, guaranteed_km=250.0, fraction=0.7, arc_km=175.0,
        offset_km=10.0, origin_uncertainty_km=0.0, max_abs_offset_km=40.0,
        min_projection_km=0.0, tie_gap_km=None, tie_margin_required_km=None,
        single_candidate=False, latitude=lat, longitude=lon,
    )


def _expect_raises(exc_types, what: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_types:
        return
    raise AssertionError(f"{what}: no {exc_types} was raised")


# =============================================================================
# PURE -- construction-mode argument pairing
# =============================================================================

def test_construction_mode_requires_the_whole_pair() -> None:
    """Neither -> legacy, both -> construction, exactly one -> refused."""
    assert _resolve_construction_mode(None, None) is False
    assert _resolve_construction_mode(3, random.Random(0)) is True
    assert _resolve_construction_mode(0, random.Random(0)) is True  # a legal probe

    _expect_raises(ValueError, "n_hidden alone", _resolve_construction_mode, 3, None)
    _expect_raises(
        ValueError, "rng alone", _resolve_construction_mode, None, random.Random(0)
    )


def test_construction_mode_rejects_bad_counts_and_rngs() -> None:
    """`n_hidden` must be a genuine non-negative int; the rng must be a real Random.

    `bool` is rejected despite subclassing `int` (mirroring the locked B2 layer), and a
    non-`random.Random` rng is refused rather than silently falling back to module-global
    randomness -- which would make an episode's geometry depend on everything that ran
    before it.
    """
    rng = random.Random(0)
    for bad in (-1, 1.0, 2.5, "3", True, False):
        _expect_raises(
            ValueError, f"n_hidden={bad!r}", _resolve_construction_mode, bad, rng
        )
    for bad_rng in (0, "rng", random, random.SystemRandom):
        _expect_raises(
            ValueError, f"rng={bad_rng!r}", _resolve_construction_mode, 3, bad_rng
        )
    # A SystemRandom INSTANCE subclasses random.Random and is therefore accepted by type,
    # which is fine: the guard is about the module-global fallback, not about entropy.
    assert _resolve_construction_mode(1, random.SystemRandom()) is True


# =============================================================================
# PURE -- the airbase-only cell guard
# =============================================================================

def test_airbase_only_guard_accepts_airbases_and_rejects_anything_else() -> None:
    ours = ATTACKING_SIDE_COLOR
    ok = _StubScenario([], [Airbase("r1", 33.0, 35.0, "red"),
                            Airbase("r2", 34.0, 36.0, "red")], [])
    _require_airbase_only_targets(ok, ours)  # must not raise

    # A friendly airbase is not an enemy target and must not be judged.
    _require_airbase_only_targets(
        _StubScenario([], [Airbase("b1", 32.0, 35.0, ours)], []), ours
    )

    for bad in (
        _StubScenario([Facility("f", 33.0, 35.0, "red")], [], []),
        _StubScenario([], [Airbase("r1", 33.0, 35.0, "red")],
                      [Ship("s", 34.0, 36.0, "red")]),
    ):
        _expect_raises(
            RuntimeError, "mixed target classes", _require_airbase_only_targets, bad, ours
        )
    assert CONSTRUCTION_TARGET_CLASS == "Airbase"


# =============================================================================
# PURE -- the shared launch point
# =============================================================================

def test_shared_launch_point_verifies_the_star_origin() -> None:
    """One origin for every ego, and return_location == that origin.

    Both halves are premises B2's route geometry rests on (CLAUDE.md section 3). A fleet
    parked away from its base would still produce placements -- against the WRONG legs --
    so this must fail rather than proceed.
    """
    base = Location(32.85, 35.31)
    agents = [_StubAgent("a", Location(32.85, 35.31), Location(32.85, 35.31)),
              _StubAgent("b", Location(32.85, 35.31), Location(32.85, 35.31))]
    assert _shared_launch_point(agents) is agents[0].location
    assert base.distance_to(_shared_launch_point(agents)) < 1e-6

    _expect_raises(RuntimeError, "empty fleet", _shared_launch_point, [])
    _expect_raises(
        RuntimeError, "split origin", _shared_launch_point,
        [_StubAgent("a", Location(32.85, 35.31), Location(32.85, 35.31)),
         _StubAgent("b", Location(32.35, 34.81), Location(32.35, 34.81))],
    )
    # The exact pre-384845b defect: parked 72.7 km from the base it returns to.
    _expect_raises(
        RuntimeError, "return != launch", _shared_launch_point,
        [_StubAgent("a", Location(32.35416, 34.81240), Location(32.85416, 35.31240))],
    )
    # No return_location at all is not an error -- there is nothing to contradict.
    _shared_launch_point([_StubAgent("a", Location(32.85, 35.31), None)])


# =============================================================================
# PURE -- prototype selection + the JSON patch
# =============================================================================

def test_prototype_selection_picks_a_safe_enemy_airbase() -> None:
    red = "red-side-id"
    blue_base = _airbase_entry("b", "Blue", "blue-side-id", "blue", aircraft=[{"id": "ac"}])
    red_base = _airbase_entry("r", "Red", red, "red")

    chosen = _select_hidden_prototype(
        {"airbases": [blue_base, red_base]}, ATTACKING_SIDE_COLOR
    )
    assert chosen is red_base

    # An enemy base holding aircraft is NOT safe: cloning it would mint enemy aircraft
    # as a side effect of adding a target.
    _expect_raises(
        RuntimeError, "only prototype has an inventory", _select_hidden_prototype,
        {"airbases": [blue_base,
                      _airbase_entry("r2", "R2", red, "red", aircraft=[{"id": "x"}])]},
        ATTACKING_SIDE_COLOR,
    )
    # Only friendly airbases -> nothing to clone.
    _expect_raises(RuntimeError, "no enemy base", _select_hidden_prototype,
                   {"airbases": [blue_base]}, ATTACKING_SIDE_COLOR)
    # No airbases collection at all.
    _expect_raises(RuntimeError, "no airbases list", _select_hidden_prototype,
                   {"facilities": []}, ATTACKING_SIDE_COLOR)
    # Schema-incomplete prototypes are skipped, not repaired.
    broken = _airbase_entry("r3", "R3", red, "red")
    del broken["className"]
    _expect_raises(RuntimeError, "incomplete schema", _select_hidden_prototype,
                   {"airbases": [blue_base, broken]}, ATTACKING_SIDE_COLOR)
    # Two enemy sides -> ambiguous; the seam refuses to guess.
    _expect_raises(
        RuntimeError, "ambiguous enemy side", _select_hidden_prototype,
        {"airbases": [_airbase_entry("r", "R", "red-a", "red"),
                      _airbase_entry("o", "O", "orange-b", "orange")]},
        ATTACKING_SIDE_COLOR,
    )


def test_patch_appends_hidden_airbases_without_touching_the_known_ones() -> None:
    """Known entries keep their content AND their positions; hidden ones go at the end."""
    red = "red-side-id"
    known = [
        _airbase_entry("b", "Blue", "blue-side-id", "blue", aircraft=[{"id": "ac"}]),
        _airbase_entry("r1", "Red 1", red, "red"),
        _airbase_entry("r2", "Red 2", red, "red"),
    ]
    original = _scenario_json(known)
    before = json.loads(original)["currentScenario"]["airbases"]

    placements = (_placement(36.0, 38.0, "e1"), _placement(37.5, 39.25, "e2"))
    patched = json.loads(
        build_patched_scenario(original, placements,
                               attacking_side_color=ATTACKING_SIDE_COLOR)
    )["currentScenario"]["airbases"]

    assert len(patched) == len(before) + 2
    # Every pre-existing entry is byte-identical AND still at its original index.
    assert patched[:len(before)] == before

    new_entries = patched[len(before):]
    for index, (entry, placement) in enumerate(zip(new_entries, placements), start=1):
        assert entry["latitude"] == placement.latitude
        assert entry["longitude"] == placement.longitude
        assert entry["name"] == HIDDEN_TARGET_NAME_TEMPLATE % index
        assert entry["aircraft"] == []            # never an inventory
        assert entry["sideId"] == red             # inherited enemy ownership
        assert entry["sideColor"] == "red"
        assert entry["className"] == known[1]["className"]
        assert entry["id"] not in {e["id"] for e in before}
    assert len({e["id"] for e in patched}) == len(patched)      # ids unique
    assert len({e["name"] for e in patched}) == len(patched)    # names unique

    # The originals are untouched: the patch never mutates the caller's prototype dicts.
    assert json.loads(original)["currentScenario"]["airbases"] == before


def test_patch_with_no_placements_is_byte_identical() -> None:
    """An `n_hidden=0` probe must not perturb the world at all -- not even formatting."""
    original = _scenario_json([_airbase_entry("r", "R", "red-side", "red")])
    assert build_patched_scenario(original, ()) is original


def test_patch_fails_loudly_on_a_malformed_scenario() -> None:
    for bad in ('{"noScenario": {}}', '{"currentScenario": []}', '"a string"'):
        _expect_raises(RuntimeError, f"malformed {bad!r}", build_patched_scenario,
                       bad, (_placement(36.0, 38.0),))
    # A scenario with no safe prototype fails at patch time, not silently.
    _expect_raises(
        RuntimeError, "no prototype", build_patched_scenario,
        _scenario_json([_airbase_entry("b", "B", "blue-side", "blue")]),
        (_placement(36.0, 38.0),),
    )


def test_patch_refuses_a_name_collision() -> None:
    """A pre-existing target already called `Hidden Airbase #001` is ambiguous."""
    entries = [
        _airbase_entry("r", "Red", "red-side", "red"),
        _airbase_entry("x", HIDDEN_TARGET_NAME_TEMPLATE % 1, "red-side", "red"),
    ]
    _expect_raises(RuntimeError, "name collision", build_patched_scenario,
                   _scenario_json(entries), (_placement(36.0, 38.0),))


# =============================================================================
# PURE -- reload invariants
# =============================================================================

def test_agent_id_drift_across_reload_is_refused() -> None:
    """A_init's agent keys must still address the runtime egos, in the same ORDER."""
    _require_agent_ids_preserved(["a", "b"], ["a", "b"])
    for after in (["a"], ["a", "b", "c"], ["b", "a"], ["a", "z"]):
        _expect_raises(RuntimeError, f"drift to {after}",
                       _require_agent_ids_preserved, ["a", "b"], after)


def test_known_tasks_rematerialize_as_env2_objects_in_a_init_order() -> None:
    """Order comes from A_init's list; the OBJECTS come from the reloaded world."""
    env2 = [_task("t2"), _task("hidden"), _task("t0"), _task("t1")]
    out = _rematerialize_known_tasks(env2, ["t0", "t1", "t2"])

    assert [_task_target_id(t) for t in out] == ["t0", "t1", "t2"]
    # Identity, not equality: every returned task IS an env-2 object.
    assert all(any(t is w for w in env2) for t in out)
    # The hidden target is NOT re-materialized into the belief list -- that is the whole
    # no-communication point: it exists in the world and in the oracle, in no belief.
    assert "hidden" not in {_task_target_id(t) for t in out}


def test_known_task_rematerialization_fails_loudly() -> None:
    env2 = [_task("t0"), _task("t1")]
    _expect_raises(RuntimeError, "known target missing from the reload",
                   _rematerialize_known_tasks, env2, ["t0", "gone"])
    _expect_raises(RuntimeError, "duplicate in A_init's list",
                   _rematerialize_known_tasks, env2, ["t0", "t0"])
    _expect_raises(RuntimeError, "duplicate in the world",
                   _rematerialize_known_tasks, [_task("t0"), _task("t0")], ["t0"])
    _expect_raises(RuntimeError, "blank world target id",
                   _rematerialize_known_tasks, [Task(steps=[], utility=1)], [])
    _expect_raises(RuntimeError, "blank known target id",
                   _rematerialize_known_tasks, env2, [""])


# =============================================================================
# PURE -- P6: the t=0 world snapshots come from the RAW, PRE-SOLVE task sets
# =============================================================================
#
# `solve_and_normalize` returns an ALLOCATED-ONLY task list by contract, for BOTH solves.
# So `belief_tasks` says what the egos were planned against and `oracle_tasks` says what
# the oracle allocated -- neither is an inventory of what exists, and a target the solver
# left unselected is missing from both while still being in the world, sensible,
# attackable and confirmable. `EpisodeContext.known_target_ids` / `executed_target_ids`
# are therefore captured BEFORE either solve, and these tests pin that ordering by using a
# solver stub that DROPS a task.


class _StubEnv:
    def __init__(self, name: str) -> None:
        self.name, self.closed = name, False

    def close(self) -> None:
        self.closed = True


def _agent(agent_id: str) -> Any:
    """A minimal real `Agent` -- enough for `Belief` and `GraphPlanExecutor`."""
    from match_aou.models import Agent
    return Agent(
        location=Location(32.0, 35.0), capabilities=[], budget=1000.0,
        move_cost_function=lambda a, b: 0.0, speed=250.0,
        return_location=Location(32.0, 35.0), agent_id=agent_id, side_color="blue",
    )


class _patched:
    """Swap module attributes for the duration of a `with` block, then restore them.

    Hand-rolled rather than `monkeypatch`, so this file's `__main__` runner (pytest is
    absent from `nlp_env`) executes these tests too.
    """

    def __init__(self, module: Any, **attrs: Any) -> None:
        self._module, self._attrs, self._saved = module, attrs, {}

    def __enter__(self) -> "_patched":
        for name, value in self._attrs.items():
            self._saved[name] = getattr(self._module, name)
            setattr(self._module, name, value)
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            setattr(self._module, name, value)


def test_world_target_ids_is_raw_ordered_and_strict() -> None:
    """The snapshot helper: order preserved, duplicates collapsed, blanks refused."""
    assert _world_target_ids([_task("t0"), _task("t1"), _task("t2")], "w") == \
        ("t0", "t1", "t2")
    # Two tasks may legitimately name ONE target; first use wins and order survives.
    assert _world_target_ids([_task("b"), _task("a"), _task("b")], "w") == ("b", "a")
    assert _world_target_ids([], "w") == ()
    # A target-less task would silently SHORTEN the inventory, so it raises instead.
    _expect_raises(RuntimeError, "blank target id in the world snapshot",
                   _world_target_ids, [_task("t0"), Task(steps=[], utility=1)], "w")


def test_finish_context_requires_a_coherent_world_snapshot() -> None:
    """`_finish_context` verifies the snapshots rather than trusting its caller.

    A future third setup path must not be able to reach a context carrying an empty world
    inventory -- that is the one shape in which allocated-only data gets read as world
    truth again. `reference_policy` / `t0_reference_tasks` are REQUIRED for the same
    reason (GENERALIZED-V1 task 3): a path that omitted them could silently claim the
    historical reference policy while having deferred its reference solve.
    """
    agents = [_agent("ego_0")]
    tasks = [_task("k0")]
    common = dict(
        game=None, env=None, obs=None, agents=agents, a_init={},
        belief_tasks=tasks, oracle_solution={}, oracle_tasks=tasks, split_meta={},
        detection_km=DETECTION_KM, recording_export_path=None, placements=(),
        reference_policy=REFERENCE_POLICY_STATIC_T0_V1, t0_reference_tasks=tasks,
    )
    ctx = _finish_context(known_target_ids=("k0",),
                          executed_target_ids=("k0", "h0"), **common)
    assert ctx.known_target_ids == ("k0",)
    assert ctx.executed_target_ids == ("k0", "h0")

    _expect_raises(RuntimeError, "empty executed-world snapshot", _finish_context,
                   known_target_ids=("k0",), executed_target_ids=(), **common)
    _expect_raises(RuntimeError, "known target outside the executed world",
                   _finish_context, known_target_ids=("k0", "ghost"),
                   executed_target_ids=("k0",), **common)


def _dropping_solver(drop: str):
    """A `solve_and_normalize` stub that leaves ONE target unselected.

    Exactly the locked contract's shape: it returns the ALLOCATED-ONLY task list, so the
    dropped target is simply not in what it hands back.
    """
    def _solve(agents, tasks, precedence_relations=None):
        kept = [t for t in tasks if _task_target_id(t) != drop]
        solution = {str(a.id): [(i, 0, 0) for i in range(len(kept))] for a in agents}
        return solution, kept, []
    return _solve


def test_legacy_path_snapshots_the_world_not_the_allocation() -> None:
    """P6. Legacy: known ids come from raw `partial`, executed ids from raw `full`."""
    agents = [_agent("ego_0")]
    all_tasks = [_task("k0"), _task("k1"), _task("h0")]
    partial = all_tasks[:2]

    with _patched(
        _setup,
        _build_env=lambda *a, **k: (None, _StubEnv("env"), "obs"),
        _extract_world=lambda obs, color: (agents, all_tasks),
        split_tasks=lambda tasks, ratio, **k: (partial, all_tasks, {"outcome": "ok"}),
        # k1 (known) and h0 (hidden) are both left UNSELECTED by their solves.
        solve_and_normalize=_dropping_solver("k1"),
    ):
        ctx = setup_episode("{}", partial_ratio=0.5)

    # The allocations really are short -- otherwise this test proves nothing.
    planned = {_task_target_id(t) for t in ctx.beliefs["ego_0"].tasks}
    allocated = {_task_target_id(t) for t in ctx.oracle_tasks}
    assert planned == {"k0"} and allocated == {"k0", "h0"}

    # The WORLD is whole in both halves, in raw order.
    assert ctx.known_target_ids == ("k0", "k1")
    assert ctx.executed_target_ids == ("k0", "k1", "h0")


def test_construction_path_snapshots_the_world_not_the_allocation() -> None:
    """P6. Construction: known ids from raw env-1, executed ids from raw env-2.

    Both are taken before their own solve, and the env-2 one includes the hidden half.
    """
    agents = [_agent("ego_0")]
    known_tasks = [_task("k0"), _task("k1")]
    env2_tasks = known_tasks + [_task("h0")]
    env1, env2 = _StubEnv("env1"), _StubEnv("env2")
    built: List[Any] = []

    def _build_env(scenario_json, **kwargs):
        env = env1 if not built else env2
        built.append(env)
        return None, env, "obs%d" % len(built)

    def _extract_world(obs, color):
        return (agents, known_tasks if obs == "obs1" else env2_tasks)

    with _patched(
        _setup,
        _build_env=_build_env,
        _extract_world=_extract_world,
        _require_airbase_only_targets=lambda obs, color: None,
        _shared_launch_point=lambda agents_: Location(32.0, 35.0),
        _require_agent_ids_preserved=lambda before, after: None,
        place_hidden_targets=lambda *a, **k: ("<placement>",),
        build_patched_scenario=lambda scenario_json, placements, **k: "{}",
        geometric_fingerprint=lambda placements: (),
        # k1 is a known target the KNOWN solve does not select; the oracle solve over
        # env-2 then drops it as well. It is in neither allocated-only list.
        solve_and_normalize=_dropping_solver("k1"),
    ):
        ctx = setup_episode("{}", n_hidden=1, placement_rng=random.Random(0))

    assert env1.closed and not env2.closed, "env-1 must be closed, env-2 returned"

    planned = {_task_target_id(t) for t in ctx.beliefs["ego_0"].tasks}
    allocated = {_task_target_id(t) for t in ctx.oracle_tasks}
    assert planned == {"k0"}, planned
    assert allocated == {"k0", "h0"}, allocated

    # The WORLD, raw and in order -- the hidden target included on the executed side.
    assert ctx.known_target_ids == ("k0", "k1")
    assert ctx.executed_target_ids == ("k0", "k1", "h0")
    # And the allocated-only contracts are UNCHANGED for their own purposes.
    assert len(ctx.beliefs["ego_0"].tasks) == 1
    assert ctx.split_meta["known"] == 2 and ctx.split_meta["hidden"] == 1


# =============================================================================
# PURE -- P7: GENERALIZED-V1 cardinality policy, cell enforcement and accounting
# =============================================================================


def _backoff_audit(
    *, requested: int, realized: int, candidate_count: int = 2
) -> BoundedBackoffAudit:
    """A hand-built backoff record -- the pure tests never run real geometry."""
    accepted = tuple(
        BackoffCandidate(ordinal=i, ego_id=f"ego_{i}", accepted=True, reason=None,
                         detail=None, leg_index=1)
        for i in range(realized)
    )
    rejected = tuple(
        BackoffCandidate(ordinal=i, ego_id=f"ego_{i}", accepted=False,
                         reason=REASON_NO_ROUTE, detail="no route", leg_index=None)
        for i in range(realized, candidate_count)
    )
    return BoundedBackoffAudit(
        policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
        candidate_count=candidate_count,
        candidate_order=tuple(range(candidate_count)),
        considered_ordinals=tuple(range(candidate_count)),
        candidates=accepted + rejected,
        selected_ordinals=tuple(range(realized)),
        hidden_requested=requested,
        hidden_realized=realized,
        geometric_fingerprint=tuple((32.0 + i, 35.0 + i) for i in range(realized)),
    )


def test_hidden_policy_is_explicit_and_defaults_to_the_exact_path() -> None:
    """P7. The policy is chosen, never inferred, and the historical one is the default.

    A generalized policy that could be silently dropped -- because the construction pair
    was missing, or the id was misspelled -- would build a world nobody asked for, so both
    are refused BEFORE any BLADE object exists.
    """
    rng = random.Random(0)

    # Default and explicit-exact both behave exactly as they always did.
    assert _resolve_construction_mode(None, None) is False
    assert _resolve_construction_mode(None, None, HIDDEN_POLICY_EXACT_V1) is False
    assert _resolve_construction_mode(3, rng) is True
    assert _resolve_construction_mode(3, rng, HIDDEN_POLICY_EXACT_V1) is True
    assert _resolve_construction_mode(0, rng, HIDDEN_POLICY_EXACT_V1) is True  # legal probe

    assert _resolve_construction_mode(2, rng, HIDDEN_POLICY_BOUNDED_BACKOFF_V1) is True

    # Selecting the generalized policy on the LEGACY path would silently ignore it.
    _expect_raises(ValueError, "bounded policy without the construction pair",
                   _resolve_construction_mode, None, None,
                   HIDDEN_POLICY_BOUNDED_BACKOFF_V1)
    # `n_hidden=0` is an exact-path probe only: a generalized world needs a hidden half.
    _expect_raises(ValueError, "bounded policy with n_hidden=0",
                   _resolve_construction_mode, 0, rng,
                   HIDDEN_POLICY_BOUNDED_BACKOFF_V1)
    # An unknown id is refused on BOTH paths rather than falling back to a default.
    for args in ((None, None), (2, rng)):
        _expect_raises(ValueError, f"unknown policy with {args}",
                       _resolve_construction_mode, args[0], args[1], "bounded_backoff_v2")


def test_generalized_cardinality_cell_is_enforced() -> None:
    """P7. A in {2,3,4}, K == A, 1 <= H_requested <= A -- refused, never repaired."""
    assert GENERALIZED_AGENT_COUNTS == (2, 3, 4), GENERALIZED_AGENT_COUNTS

    for agents in GENERALIZED_AGENT_COUNTS:
        for hidden in range(1, agents + 1):
            _require_generalized_cardinality(
                agent_count=agents, known_count=agents, hidden_requested=hidden
            )

    for agents, known, hidden, what in (
        (1, 1, 1, "A below the cell"),
        (5, 5, 1, "A above the cell"),
        (3, 2, 1, "K < A"),
        (3, 4, 1, "K > A"),
        (3, 3, 0, "H_requested below 1"),
        (3, 3, 4, "H_requested above A"),
    ):
        _expect_raises(RuntimeError, what, _require_generalized_cardinality,
                       agent_count=agents, known_count=known, hidden_requested=hidden)


def _generalized_stub_context(
    *, n_hidden: int, realized: int, audit_realized: Optional[int] = None,
    num_agents: int = 2, claim_hidden_in_world: Optional[int] = None,
) -> EpisodeContext:
    """Drive the REAL generalized construction seam with stubbed env / solver / geometry.

    Only the pieces that need BLADE or bonmin are replaced. The cell guard, the accounting
    assembly, the world snapshots, `_rematerialize_known_tasks` and `_finish_context` are
    all the production code.
    """
    agents = [_agent(f"ego_{i}") for i in range(num_agents)]
    known_tasks = [_task(f"k{i}") for i in range(num_agents)]
    in_world = realized if claim_hidden_in_world is None else claim_hidden_in_world
    env2_tasks = known_tasks + [_task(f"h{i}") for i in range(in_world)]
    env1, env2 = _StubEnv("env1"), _StubEnv("env2")
    built: List[Any] = []

    def _build_env(scenario_json, **kwargs):
        env = env1 if not built else env2
        built.append(env)
        return None, env, "obs%d" % len(built)

    def _extract_world(obs, color):
        return (agents, known_tasks if obs == "obs1" else env2_tasks)

    def _solve(agents_, tasks_, precedence_relations=None):
        solution = {str(a.id): [(i, 0, 0)] for i, a in enumerate(agents_)}
        return solution, list(tasks_), []

    def _bounded(*args, **kwargs):
        return (
            tuple(f"<placement {i}>" for i in range(realized)),
            _backoff_audit(
                requested=kwargs["hidden_requested"],
                realized=realized if audit_realized is None else audit_realized,
                candidate_count=len(kwargs["agent_ordinals"]),
            ),
        )

    with _patched(
        _setup,
        _build_env=_build_env,
        _extract_world=_extract_world,
        _require_airbase_only_targets=lambda obs, color: None,
        _shared_launch_point=lambda agents_: Location(32.0, 35.0),
        _require_agent_ids_preserved=lambda before, after: None,
        place_hidden_targets_bounded=_bounded,
        build_patched_scenario=lambda scenario_json, placements, **k: "{}",
        geometric_fingerprint=lambda placements: (),
        solve_and_normalize=_solve,
    ):
        return setup_episode(
            "{}", n_hidden=n_hidden, placement_rng=random.Random(0),
            hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
        )


def test_generalized_context_reports_requested_and_realized_cardinality() -> None:
    """P7. A backed-off world states BOTH counts; the request is never rewritten."""
    ctx = _generalized_stub_context(n_hidden=2, realized=1, num_agents=2)
    audit = ctx.construction_audit

    assert isinstance(audit, ConstructionAudit), audit
    assert audit.policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1, audit.policy
    assert audit.agent_count == 2 and audit.known_requested == 2, audit
    assert audit.known_realized == 2 == len(ctx.known_target_ids), audit
    # THE POINT: the request survives the backoff.
    assert audit.hidden_requested == 2, audit.hidden_requested
    assert audit.hidden_realized == 1 == len(ctx.placements), audit
    assert audit.realized_full_request is False, audit
    assert audit.total_requested == 4, audit.total_requested
    assert audit.total_realized == 3 == len(ctx.executed_target_ids), audit

    # The world counts reconcile with the RAW snapshots, not with an allocation.
    hidden_ids = [t for t in ctx.executed_target_ids if t not in set(ctx.known_target_ids)]
    assert len(hidden_ids) == audit.hidden_realized, hidden_ids
    assert set(ctx.known_target_ids) <= set(ctx.executed_target_ids)

    # The candidate accounting is reachable from the context, and JSON-ready.
    assert audit.candidate_order == (0, 1) and audit.selected_ordinals == (0,), audit
    assert audit.considered_ordinals == (0, 1), audit
    assert [c.reason for c in audit.candidates] == [None, REASON_NO_ROUTE], audit.candidates
    payload = json.loads(json.dumps(audit.as_dict()))
    assert payload["hidden_requested"] == 2 and payload["hidden_realized"] == 1, payload

    assert ctx.split_meta["hidden_policy"] == HIDDEN_POLICY_BOUNDED_BACKOFF_V1
    assert ctx.split_meta["hidden_realized"] == 1
    assert ctx.split_meta["n_hidden_requested"] == 2
    assert ctx.split_meta["construction_audit"] == audit.as_dict()


def test_generalized_full_request_is_reported_as_such() -> None:
    """P7. Realizing everything requested is reported as such, with no backoff noise."""
    ctx = _generalized_stub_context(n_hidden=2, realized=2, num_agents=2)
    audit = ctx.construction_audit
    assert audit.hidden_requested == audit.hidden_realized == 2, audit
    assert audit.realized_full_request is True, audit
    assert audit.total_requested == audit.total_realized == 4, audit
    assert all(c.accepted for c in audit.candidates), audit.candidates


def test_generalized_cell_is_judged_before_anything_is_solved() -> None:
    """P7. An out-of-cell request costs no solve and leaves no partial construction."""
    def _forbidden(*_a: Any, **_k: Any):
        raise AssertionError("the solver ran before the generalized cell was judged")

    with _patched(_setup, solve_and_normalize=_forbidden):
        # H_requested = 3 > A = 2.
        _expect_raises(RuntimeError, "H_requested above A",
                       _generalized_stub_context, n_hidden=3, realized=1, num_agents=2)


def test_generalized_accounting_that_does_not_reconcile_is_refused() -> None:
    """P7. The audit is VERIFIED against the built world, never trusted.

    A record claiming more realized hidden targets than were actually placed would make
    every downstream requested-vs-realized statistic wrong while looking self-consistent,
    so it is refused here.
    """
    _expect_raises(
        RuntimeError, "audit disagreeing with the placements",
        _generalized_stub_context, n_hidden=2, realized=1, audit_realized=2, num_agents=2,
    )


def test_generalized_patch_adds_exactly_the_realized_hidden_targets() -> None:
    """P7. H_REALIZED targets are patched in -- never H_REQUESTED phantom ones."""
    # The reloaded world claims TWO hidden targets while only ONE was realized: the
    # existing env-2 cardinality guard refuses it, so a phantom target cannot survive.
    _expect_raises(
        RuntimeError, "a phantom hidden target in the reloaded world",
        _generalized_stub_context, n_hidden=2, realized=1, claim_hidden_in_world=2,
        num_agents=2,
    )


def test_exact_construction_path_carries_no_generalized_accounting() -> None:
    """P7 / PO1. The historical path is observably unchanged: no audit, no new keys."""
    agents = [_agent("ego_0")]
    known_tasks = [_task("k0")]
    env2_tasks = known_tasks + [_task("h0")]
    env1, env2 = _StubEnv("env1"), _StubEnv("env2")
    built: List[Any] = []

    def _build_env(scenario_json, **kwargs):
        env = env1 if not built else env2
        built.append(env)
        return None, env, "obs%d" % len(built)

    def _forbidden_bounded(*_a: Any, **_k: Any):
        raise AssertionError("the bounded policy ran on the exact path")

    with _patched(
        _setup,
        _build_env=_build_env,
        _extract_world=lambda obs, color: (
            agents, known_tasks if obs == "obs1" else env2_tasks
        ),
        _require_airbase_only_targets=lambda obs, color: None,
        _shared_launch_point=lambda agents_: Location(32.0, 35.0),
        _require_agent_ids_preserved=lambda before, after: None,
        place_hidden_targets=lambda *a, **k: ("<placement>",),
        place_hidden_targets_bounded=_forbidden_bounded,
        build_patched_scenario=lambda scenario_json, placements, **k: "{}",
        geometric_fingerprint=lambda placements: (),
        solve_and_normalize=_dropping_solver("<nothing>"),
    ):
        ctx = setup_episode("{}", n_hidden=1, placement_rng=random.Random(0))

    assert ctx.construction_audit is None, ctx.construction_audit
    for key in ("hidden_policy", "hidden_realized", "construction_audit"):
        assert key not in ctx.split_meta, f"the exact path grew a {key!r} key"
    # And the keys it always carried are unchanged.
    assert ctx.split_meta["known"] == 1 and ctx.split_meta["hidden"] == 1
    assert ctx.split_meta["n_hidden_requested"] == 1


def test_no_construction_accounting_reaches_the_actor_observation() -> None:
    """P7. NOT ONE accounting quantity enters `GraphObservation`.

    A hidden COUNT, a policy id or a backoff reason is exactly the kind of privileged
    quantity an ego cannot sense, and `CLAUDE.md` section 3 is not up for renegotiation.
    The observation's field set is pinned, so adding one would fail here.
    """
    import dataclasses as _dc
    from match_aou.rl.observation.graph_builder import GraphObservation

    fields = tuple(f.name for f in _dc.fields(GraphObservation))
    assert fields == (
        "task_features", "agent_features", "ego_index", "edge_index", "edge_type",
        "task_target_ids", "agent_ids", "agent_id", "current_time", "time_norm",
    ), fields

    banned = ("hidden", "policy", "backoff", "realized", "requested", "candidate",
              "ordinal", "audit", "construction")
    for name in fields:
        assert not any(term in name.lower() for term in banned), name


def test_legacy_context_has_an_empty_placement_audit() -> None:
    """The audit field defaults to empty, so the legacy path's contract is unchanged."""
    ctx = EpisodeContext(
        env=None, game=None, observation=None, agents=[], agent_ids=[], beliefs={},
        executor=None, a_init={}, oracle_solution={}, oracle_tasks=[], split_meta={},
    )
    assert ctx.placements == ()
    assert ctx.record is False
    assert geometric_fingerprint(ctx.placements) == ()
    # The generalized accounting field defaults to absent, so the legacy path's contract
    # is unchanged and "which policy ran?" is answerable from the context alone.
    assert ctx.construction_audit is None


# =============================================================================
# PURE -- P4: the DERIVED attack-confirmation wait (Defect B)
# =============================================================================
#
# The executor used to arm a FIXED `kill_confirm_ticks` (default 60) after every salvo.
# It now DERIVES the wait at issue time from the weapon BLADE's own 2-arg attack path
# will select and the engagement distance the executor already computed, keeping the
# configured value as the FLOOR and the fallback. These are hand-built stubs: no engine,
# no solver -- the derivation is arithmetic over duck-typed attributes.


class _StubWeapon:
    """The two attributes the derivation touches, plus the engine's own range rule.

    `Weapon.get_engagement_range` is `speed * (current_fuel / fuel_rate)` -- an endurance
    product, NOT a speed ranking. That is exactly why the selected weapon can get SLOWER
    as a loadout is spent, which is the defect being fixed.
    """

    def __init__(self, name: str, speed: Any, engagement_range: float) -> None:
        self.name, self.speed, self._range = name, speed, engagement_range

    def get_engagement_range(self) -> float:
        return self._range


class _StubArmedAircraft:
    """A duck-typed live aircraft exposing the engine's own selection method."""

    def __init__(self, uid: str, lat: float, lon: float, weapons: List[Any]) -> None:
        self.id, self.latitude, self.longitude = uid, lat, lon
        self.weapons = list(weapons)

    def get_weapon_with_highest_engagement_range(self) -> Optional[Any]:
        # Byte-for-byte the engine's rule, from `Aircraft`.
        if not self.weapons:
            return None
        return max(self.weapons, key=lambda w: w.get_engagement_range())


class _StubAirScenario:
    """Only `aircraft` is read by `_find_aircraft`, which is all the wait needs."""

    def __init__(self, aircraft: List[Any]) -> None:
        self.aircraft = list(aircraft)


def _wait_executor(kill_confirm_ticks: int = 60) -> Any:
    """A minimal executor whose ONLY job here is to expose `_confirmation_wait_ticks`."""
    from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor

    agent = _StubAgent("ego", Location(32.0, 35.0), Location(32.0, 35.0))
    return GraphPlanExecutor(
        tasks=[], solution={"ego": []}, agents=[agent],
        arrival_threshold_km=DETECTION_KM, kill_confirm_ticks=kill_confirm_ticks,
    )


def test_salvo_travel_bound_uses_the_engine_units_and_ceiling() -> None:
    """P4a: km -> nm -> hours at the weapon's KNOTS speed -> seconds, CEILED.

    What this pins is the BOUND's arithmetic: a changed conversion, a lost `3600` or a
    swapped rounding mode all fail, because the expected values are recomputed here
    rather than copied from the implementation.

    What it does NOT do is compare anything against the running engine -- the constant
    check below pins the executor's transcription against a LITERAL, which cannot see
    drift in BLADE's own value. That comparison lives in the BLADE tier, where the engine
    can be imported: see the `test_blade_transcribed_km_to_nm_...` test below.

    And the bound is a BOUND: it is not the number of engine ticks a salvo takes. BLADE
    advances a new weapon inside `launch_weapon`, may advance it again in the same
    `update_game_state`, and resolves the target once the remaining distance is under
    1 km -- so real engagements confirm EARLIER than this figure (measured in P5:
    bound 62 at ~47.2 km, real confirmation on executor call 60).
    """
    import math

    from match_aou.utils.blade_utils.blade_graph_executor import (
        KILOMETERS_TO_NAUTICAL_MILES,
        _salvo_travel_ticks,
    )

    # The executor's transcription, pinned against a LITERAL so this pure tier stays
    # engine-free. This is NOT the drift check -- see the BLADE-tier test named in the
    # docstring, which compares it with the engine's own constant.
    assert KILOMETERS_TO_NAUTICAL_MILES == 0.539957

    def expected(distance_km: float, speed_knots: float) -> int:
        return int(math.ceil(distance_km * 0.539957 / speed_knots * 3600.0))

    # The three real loadout speeds of `strike_training_4v5.json`, at the observed
    # engagement distance and at the detection radius.
    cases = [
        (47.2, 2600.0, 36),   # AIM-120 -- bound comfortably inside a fixed 60
        (47.2, 1500.0, 62),   # AIM-9   -- bound ALREADY ABOVE a fixed 60: the defect
        (47.2, 600.0, 153),   # AGM-65  -- bound far above it
        (50.0, 2600.0, 38),
        (50.0, 1500.0, 65),
        (0.0, 1500.0, 0),     # degenerate but well defined: no flight, no wait
    ]
    for distance_km, speed_knots, want in cases:
        got = _salvo_travel_ticks(_StubWeapon("w", speed_knots, 1.0), distance_km)
        assert got == want == expected(distance_km, speed_knots), (
            distance_km, speed_knots, got, want
        )

    # CEILING, not rounding and not truncation: 1 km at 1 kt is 1943.85 s -> 1944.
    assert _salvo_travel_ticks(_StubWeapon("w", 1.0, 1.0), 1.0) == 1944

    # A FINITE NEGATIVE speed is accepted and normalised with |speed|, exactly as the
    # engine normalises it in `get_next_coordinates` -- it is not a fallback case.
    assert _salvo_travel_ticks(_StubWeapon("w", -1500.0, 1.0), 47.2) == 62


def test_salvo_travel_ticks_refuses_to_guess_on_unusable_data() -> None:
    """P4b: every underivable input returns None -> the caller uses the configured wait.

    A finite NEGATIVE speed is deliberately absent from this list: it is usable, and P4a
    pins it as normalised with `abs` rather than refused.
    """
    from match_aou.utils.blade_utils.blade_graph_executor import _salvo_travel_ticks

    assert _salvo_travel_ticks(None, 47.2) is None                        # no weapon
    assert _salvo_travel_ticks(object(), 47.2) is None                    # no `speed`
    assert _salvo_travel_ticks(_StubWeapon("w", 0.0, 1.0), 47.2) is None  # zero speed
    assert _salvo_travel_ticks(_StubWeapon("w", float("nan"), 1.0), 47.2) is None
    assert _salvo_travel_ticks(_StubWeapon("w", float("inf"), 1.0), 47.2) is None
    assert _salvo_travel_ticks(_StubWeapon("w", "fast", 1.0), 47.2) is None
    assert _salvo_travel_ticks(_StubWeapon("w", None, 1.0), 47.2) is None
    # ... and an unusable DISTANCE is refused the same way.
    assert _salvo_travel_ticks(_StubWeapon("w", 1500.0, 1.0), float("nan")) is None
    assert _salvo_travel_ticks(_StubWeapon("w", 1500.0, 1.0), float("inf")) is None
    assert _salvo_travel_ticks(_StubWeapon("w", 1500.0, 1.0), -1.0) is None


def test_confirmation_wait_uses_the_highest_engagement_range_live_weapon() -> None:
    """P4c: the SELECTED weapon drives the wait -- never a name, an order or a speed rank.

    The rack is deliberately adversarial: the FASTEST weapon is first, the selected one
    sits in the MIDDLE, and the LAST entry is the slowest. Only `get_engagement_range`
    decides -- the same call `Game.handle_aircraft_attack(aircraft_id, target_id)` makes.
    """
    from match_aou.utils.blade_utils.blade_graph_executor import _salvo_travel_ticks

    executor = _wait_executor(kill_confirm_ticks=60)

    fastest = _StubWeapon("AIM-120 AMRAAM", 2600.0, 1000.0)     # fastest, SHORTEST range
    selected = _StubWeapon("AIM-9 Sidewinder", 1500.0, 3565.0)  # the engine's pick
    slowest = _StubWeapon("AGM-65 Maverick", 600.0, 1875.0)
    aircraft = _StubArmedAircraft("ego", 32.0, 35.0, [fastest, selected, slowest])
    scenario = _StubAirScenario([aircraft])

    assert aircraft.get_weapon_with_highest_engagement_range() is selected
    got = executor._confirmation_wait_ticks(scenario, "ego", 47.2)
    assert got == _salvo_travel_ticks(selected, 47.2) + 1 == 63, got
    # It is NOT the fastest, NOT the slowest and NOT a list position.
    assert got != _salvo_travel_ticks(fastest, 47.2) + 1
    assert got != _salvo_travel_ticks(slowest, 47.2) + 1

    # Reordering the identical rack cannot change the answer.
    for order in ([selected, slowest, fastest], [slowest, fastest, selected]):
        shuffled = _StubAirScenario([_StubArmedAircraft("ego", 32.0, 35.0, order)])
        assert executor._confirmation_wait_ticks(shuffled, "ego", 47.2) == got, order

    # Spend the pick and the engine's own rule hands over to the next rack entry: the
    # wait FOLLOWS it downward in speed, which a fixed constant cannot do.
    aircraft.weapons.remove(selected)
    assert aircraft.get_weapon_with_highest_engagement_range() is slowest
    assert executor._confirmation_wait_ticks(scenario, "ego", 47.2) == 154


def test_confirmation_wait_keeps_the_configured_value_as_floor_and_fallback() -> None:
    """P4d: `max(configured, travel + 1)`, and the configured value on every fallback."""
    from types import SimpleNamespace

    executor = _wait_executor(kill_confirm_ticks=60)
    quick = _StubWeapon("AIM-120 AMRAAM", 2600.0, 3565.0)  # 36 ticks at 47.2 km

    # FLOOR: a fast salvo derives 37, which is below the configured minimum.
    scenario = _StubAirScenario([_StubArmedAircraft("ego", 32.0, 35.0, [quick])])
    assert executor._confirmation_wait_ticks(scenario, "ego", 47.2) == 60

    # The floor is the CONFIGURED value, not a literal 60.
    lenient = _wait_executor(kill_confirm_ticks=5)
    assert lenient._confirmation_wait_ticks(scenario, "ego", 47.2) == 37
    strict = _wait_executor(kill_confirm_ticks=500)
    assert strict._confirmation_wait_ticks(scenario, "ego", 47.2) == 500

    # FALLBACKS -- each returns the configured value untouched.
    empty_rack = _StubAirScenario([_StubArmedAircraft("ego", 32.0, 35.0, [])])
    assert executor._confirmation_wait_ticks(empty_rack, "ego", 47.2) == 60
    bad_speed = _StubAirScenario(
        [_StubArmedAircraft("ego", 32.0, 35.0, [_StubWeapon("w", 0.0, 1.0)])]
    )
    assert executor._confirmation_wait_ticks(bad_speed, "ego", 47.2) == 60
    # A duck-typed aircraft with NO selector at all (the hand-built stub tiers).
    no_selector = _StubAirScenario(
        [SimpleNamespace(id="ego", latitude=32.0, longitude=35.0)]
    )
    assert executor._confirmation_wait_ticks(no_selector, "ego", 47.2) == 60
    # The ego is not airborne at all -> `_find_aircraft` returns None.
    assert executor._confirmation_wait_ticks(_StubAirScenario([]), "ego", 47.2) == 60
    # And the configured value survives all of them.
    assert executor.kill_confirm_ticks == 60


def test_confirmation_wait_cannot_be_moved_by_a_peer() -> None:
    """P4e: no-communication -- only the ACTING ego's own aircraft is read.

    Peers carrying wildly different loadouts, with ids sorting both before and after the
    ego, must leave the acting ego's derived wait byte-identical.
    """
    executor = _wait_executor(kill_confirm_ticks=60)
    ego_weapon = _StubWeapon("AIM-9 Sidewinder", 1500.0, 1875.0)
    alone = _StubAirScenario([_StubArmedAircraft("ego", 32.0, 35.0, [ego_weapon])])
    baseline = executor._confirmation_wait_ticks(alone, "ego", 47.2)
    assert baseline == 63, baseline

    peer_slow = _StubArmedAircraft("aaa_peer", 33.0, 36.0, [_StubWeapon("p", 1.0, 9e9)])
    peer_fast = _StubArmedAircraft("zzz_peer", 31.0, 34.0, [_StubWeapon("p", 9e5, 9e9)])
    crowded = _StubAirScenario(
        [peer_slow, _StubArmedAircraft("ego", 32.0, 35.0, [ego_weapon]), peer_fast]
    )
    assert executor._confirmation_wait_ticks(crowded, "ego", 47.2) == baseline
    # Emptying a PEER's rack changes nothing either.
    peer_slow.weapons = []
    peer_fast.weapons = []
    assert executor._confirmation_wait_ticks(crowded, "ego", 47.2) == baseline


def test_derived_wait_is_armed_only_on_attack_and_stays_per_ego_and_target() -> None:
    """P4f: the cooldown identity, the 2-arg command form and the confirmed-kill bypass.

    Everything here is hand-built: two egos share ONE stub world holding two live enemy
    airbases, both egos are in range of both, and each ego's rack differs -- so a leaked
    cooldown key or a peer-derived wait would show up immediately.
    """
    from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor

    class _Scenario(_StubAirScenario):
        def __init__(self, aircraft: List[Any], airbases: List[Any]) -> None:
            super().__init__(aircraft)
            self.airbases, self.facilities, self.ships = list(airbases), [], []

        def get_target(self, target_id: str) -> Optional[Any]:
            for unit in self.airbases:
                if str(unit.id) == str(target_id):
                    return unit
            return None

    t0 = Airbase("t0", 32.05, 35.0, "red")   # ~5.6 km from both egos
    t1 = Airbase("t1", 32.06, 35.0, "red")
    tasks = [_task("t0", 32.05, 35.0), _task("t1", 32.06, 35.0)]
    ego_a = _StubArmedAircraft("A", 32.0, 35.0, [_StubWeapon("AIM-9", 1500.0, 1875.0)])
    ego_b = _StubArmedAircraft("B", 32.0, 35.0, [_StubWeapon("AGM-65", 600.0, 1000.0)])
    scenario = _Scenario([ego_a, ego_b], [t0, t1])

    agents = [
        _StubAgent("A", Location(32.0, 35.0), Location(32.0, 35.0)),
        _StubAgent("B", Location(32.0, 35.0), Location(32.0, 35.0)),
    ]
    executor = GraphPlanExecutor(
        tasks=tasks, solution={"A": [(0, 0, 0), (1, 0, 1)], "B": [(0, 0, 0)]},
        agents=agents, arrival_threshold_km=DETECTION_KM, kill_confirm_ticks=60,
    )

    # (1) Both egos engage t0 on the same tick, with the CURRENT 2-arg command form.
    commands = executor.next_actions(scenario)
    assert commands == [
        "handle_aircraft_attack('A', 't0')", "handle_aircraft_attack('B', 't0')"
    ], commands

    # (2) The cooldown is keyed per (ego, target) and each ego got ITS OWN weapon's wait.
    #     A ~5.6 km hop is 8 ticks for the AIM-9 and 19 for the AGM-65, so BOTH sit under
    #     the configured floor -- which is exactly what `max` is for.
    assert set(executor.attack_cooldown) == {("A", "t0"), ("B", "t0")}
    assert executor.attack_cooldown[("A", "t0")] == 60
    assert executor.attack_cooldown[("B", "t0")] == 60
    # Nothing was armed for a target neither ego has engaged yet.
    assert ("A", "t1") not in executor.attack_cooldown

    # (3) While the wait runs, neither ego re-fires and each cooldown decays alone.
    assert executor.next_actions(scenario) == []
    assert executor.attack_cooldown[("A", "t0")] == 59
    assert executor.attack_cooldown[("B", "t0")] == 59

    # (4) CONFIRMED KILL BYPASSES THE REMAINING WAIT: t0 dies with 59 ticks still on the
    #     clock; A advances to t1 on the very next tick and B, whose plan is finished,
    #     goes straight to RTB. Neither waits the window out.
    scenario.airbases.remove(t0)
    commands = executor.next_actions(scenario)
    assert commands == [
        "handle_aircraft_attack('A', 't1')", "aircraft_return_to_base('B')"
    ], commands
    assert ("A", "t0") in executor.done and ("B", "t0") in executor.done
    # The confirmed target's stale cooldown is dropped for BOTH egos, independently.
    assert set(executor.attack_cooldown) == {("A", "t1")}
    assert executor.attack_cooldown[("A", "t1")] == 60


# =============================================================================
# BLADE (no solver) -- environment-1 ownership
# =============================================================================

def _known_only_scenario_json(tmp_dir: Path, seed: int = 0) -> str:
    """Generate ONE known-only reference-cell scenario and return its JSON content."""
    from match_aou.rl.training.graph_train import TrainConfig, build_variation_config
    from match_aou.utils.blade_utils.scenario_generator import ScenarioGenerator

    gen = ScenarioGenerator(
        base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_dir),
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    path = gen.generate(
        episode=seed, config=build_variation_config(TrainConfig(n_iterations=1), seed)
    )
    return path.read_text(encoding="utf-8")


def _known_only_cell_json(
    tmp_dir: Path, *, seed: int, num_agents: int, n_known: int
) -> str:
    """Generate ONE known-only world for an ARBITRARY (A, K) cell.

    Kept beside `_known_only_scenario_json` rather than replacing it: the reference-cell
    helper above is what the historical proofs are written against, and its call shape
    must not move.
    """
    from match_aou.rl.training.graph_train import TrainConfig, build_variation_config
    from match_aou.utils.blade_utils.scenario_generator import ScenarioGenerator

    gen = ScenarioGenerator(
        base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_dir),
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    cfg = TrainConfig(
        n_iterations=1, num_agents=int(num_agents), n_known=int(n_known),
        n_hidden=min(int(num_agents), 3),
    )
    path = gen.generate(episode=seed, config=build_variation_config(cfg, seed))
    return path.read_text(encoding="utf-8")


@_needs_blade
def test_build_env_closes_the_environment_when_reset_fails() -> None:
    """P1 (env ownership): a `reset()` failure must not leak the environment it made.

    `_build_env` OWNS the environment between `gymnasium.make` and its return: the
    callers' `finally` / `except` blocks are keyed on the value it hands back, so an
    exception raised before that return leaves an engine object no cleanup path can
    reach. This drives `_build_env` DIRECTLY -- and because environment 1 and
    environment 2 are both built through this single helper, the guard proven here
    covers both pre-return construction windows.

    Needs BLADE (a real `Game.load_scenario` runs before `gymnasium.make`) but NOT
    bonmin: nothing is solved.
    """
    import gymnasium
    from match_aou.rl.training import graph_episode_setup as ges

    class _ResetBoom(RuntimeError):
        """Sentinel: must reach the caller unwrapped and untranslated."""

    sentinel = _ResetBoom("INJECTED: reset failed after the env was created")
    closes: List[str] = []

    class _ExplodingEnv:
        def reset(self, *_a: Any, **_k: Any):
            raise sentinel

        def close(self) -> None:
            closes.append("closed")

    scenario_json = BASE_SCENARIO.read_text(encoding="utf-8")
    build_kwargs = dict(
        max_episode_steps=MAX_SIM_TICKS,
        attacking_side_color=ATTACKING_SIDE_COLOR,
        record_every_seconds=10,
        recording_export_path=None,
    )

    real_make = gymnasium.make
    gymnasium.make = lambda *_a, **_k: _ExplodingEnv()
    try:
        raised: Optional[BaseException] = None
        try:
            ges._build_env(scenario_json, **build_kwargs)  # type: ignore[arg-type]
        except BaseException as exc:
            raised = exc
    finally:
        gymnasium.make = real_make

    # The ORIGINAL failure propagates -- not swallowed, not wrapped, not retranslated.
    assert raised is sentinel, f"expected the sentinel itself, got {raised!r}"
    # ...and the environment it had already created was closed exactly once.
    assert closes == ["closed"], (
        f"env closed {len(closes)} time(s) on the reset-failure path; it must be closed "
        "exactly once"
    )

    # Successful construction is UNCHANGED: it returns a live env and closes nothing.
    game, env, obs = ges._build_env(scenario_json, **build_kwargs)  # type: ignore[arg-type]
    try:
        assert obs is not None and env is not None and game is not None
    finally:
        env.close()


@_needs_blade
def test_environment_one_is_closed_on_an_injected_failure() -> None:
    """P1 (env ownership): a failure after env-1 is up must still close env-1.

    The failure is injected at `solve_and_normalize`, i.e. after environment 1 has been
    built and reset but before anything downstream exists -- the exact window in which a
    leaked env would go unnoticed, because the exception looks like a solver problem.
    Needs BLADE but NOT bonmin: the solve never happens.
    """
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024
    from match_aou.rl.training import graph_episode_setup as ges

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_env1_close_"))
    try:
        scenario_json = _known_only_scenario_json(tmp_dir)

        closed: List[Any] = []
        real_build = ges._build_env

        def _spy_build(*args: Any, **kwargs: Any):
            game, env, obs = real_build(*args, **kwargs)
            real_close = env.close

            def _spy_close() -> Any:
                closed.append(env)
                return real_close()

            env.close = _spy_close
            return game, env, obs

        def _boom(*_a: Any, **_k: Any):
            raise RuntimeError("INJECTED: solver unavailable")

        ges._build_env = _spy_build
        real_solve = ges.solve_and_normalize
        ges.solve_and_normalize = _boom
        try:
            _expect_raises(
                RuntimeError, "injected solver failure", setup_episode,
                scenario_json, n_hidden=3, placement_rng=random.Random(0),
            )
        finally:
            ges._build_env = real_build
            ges.solve_and_normalize = real_solve

        assert len(closed) == 1, (
            f"env-1 was closed {len(closed)} time(s); the temporary environment must be "
            "closed exactly once on the failure path"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_blade
def test_blade_generalized_cell_is_enforced_before_the_solve() -> None:
    """P7. The generalized cell is judged against the REAL world, before any solve.

    Needs BLADE (env-1 is really built and really extracted) but NOT bonmin: the point is
    precisely that the refusal happens BEFORE `solve_and_normalize` is ever called, so an
    out-of-cell request costs no solver time and leaves no partial construction. The
    solver is replaced by a raiser, which is what makes that falsifiable.

    `K` is read from the RAW world inventory, so the `K == A` rule is a statement about
    what the scenario CONTAINS -- not about what a solver would have allocated.
    """
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    def _forbidden(*_a: Any, **_k: Any):
        raise AssertionError("the solver ran before the generalized cell was judged")

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_gen_cell_"))
    try:
        # A = K = 3, so only H_requested is out of cell.
        in_cell = _known_only_cell_json(tmp_dir, seed=0, num_agents=3, n_known=3)
        # A = 2 while K = 3: the world itself violates `K == A`.
        wrong_k = _known_only_cell_json(tmp_dir, seed=1, num_agents=2, n_known=3)

        with _patched(_setup, solve_and_normalize=_forbidden):
            _expect_raises(
                RuntimeError, "H_requested above A", setup_episode, in_cell,
                n_hidden=4, placement_rng=random.Random(0),
                hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
            )
            _expect_raises(
                RuntimeError, "K != A", setup_episode, wrong_k,
                n_hidden=1, placement_rng=random.Random(0),
                hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_blade
def test_blade_abort_replaces_a_stale_route_with_the_home_base_route() -> None:
    """P3: the ego-global abort really turns a stale BLADE route into the ride home.

    Solver-free -- the plan is hand-built, nothing is solved -- but everything else is
    real: a real `Game` from `_build_env`, real agents/tasks from `_extract_world`, a
    real launch, the real `GraphPlanExecutor` producing BOTH the stale mission route and
    the RTB command, the real `graph_effect.apply_meta_action` emptying the plan, and the
    real `Game.handle_action` applying each command.

    The ego is given TWO assignments and the abort names only the FIRST, so under the
    retired node-scoped behaviour its plan would stay non-empty, the executor would keep
    issuing moves and no RTB would ever reach the engine.

    `Game.handle_action` `exec`s and swallows exceptions (it prints them), so a command
    that silently did nothing would still "succeed". Every step below is therefore
    asserted against the ENGINE STATE it produced, never against the absence of a raise.
    """
    from match_aou.rl.action.graph_action import MetaAction
    from match_aou.rl.action.graph_effect import apply_meta_action
    from match_aou.rl.training import graph_episode_setup as ges
    from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor

    game, env, obs = ges._build_env(
        BASE_SCENARIO.read_text(encoding="utf-8"),
        max_episode_steps=MAX_SIM_TICKS,
        attacking_side_color=ATTACKING_SIDE_COLOR,
        record_every_seconds=10,
        recording_export_path=None,
    )
    try:
        agents, tasks = ges._extract_world(obs, ATTACKING_SIDE_COLOR)
        assert len(agents) >= 1 and len(tasks) >= 2, (len(agents), len(tasks))
        agent = agents[0]
        ego = str(agent.id)
        home_id = str(agent.home_base_id)

        scenario = game.current_scenario

        # (1) A REAL launch: the ego leaves its airbase inventory and goes airborne.
        launched = game.launch_aircraft_from_airbase(home_id, ego)
        assert launched is not None, "the targeted launch did not happen"
        aircraft = scenario.get_aircraft(ego)
        assert aircraft is not None, "the ego is not airborne after the launch"
        assert not aircraft.route, aircraft.route
        assert aircraft.rtb is False, aircraft.rtb

        home = scenario.get_aircraft_homebase(ego)
        assert home is not None and str(home.id) == home_id, home
        home_waypoint = [home.latitude, home.longitude]

        # (2) TWO assignments at different levels; only the level-0 one is eligible now.
        #     The level-0 target must sit OUTSIDE the ego's detection radius, or the
        #     executor would ENGAGE it instead of moving and no route would be laid --
        #     the template's nearest red unit is a SAM facility only ~45 km out.
        launch_point = Location(aircraft.latitude, aircraft.longitude)
        by_distance = sorted(
            range(len(tasks)),
            key=lambda i: launch_point.distance_to(tasks[i].steps[0].location),
        )
        far_idx, next_idx = by_distance[-1], by_distance[-2]
        assert launch_point.distance_to(tasks[far_idx].steps[0].location) > DETECTION_KM, (
            "the level-0 target is already in engagement range; no route would be laid"
        )
        solution = {ego: [(far_idx, 0, 0), (next_idx, 0, 1)]}
        executor = GraphPlanExecutor(
            tasks=tasks, solution=solution, agents=agents,
            arrival_threshold_km=DETECTION_KM,
        )

        # (3) The STALE mission route -- issued by the real executor, applied by the
        #     real engine. It points at an enemy target, not at home.
        move_commands = executor.next_actions(scenario)
        assert move_commands == ["move_aircraft('%s', [[%s, %s]])" % (
            ego, tasks[far_idx].steps[0].location.latitude,
            tasks[far_idx].steps[0].location.longitude)], move_commands
        game.handle_action(move_commands)

        stale_route = [list(wp) for wp in aircraft.route]
        assert stale_route, "the engine did not store the mission route"
        assert stale_route[-1] != home_waypoint, (
            "the mission route already ends at home; it is not a stale route"
        )

        # (4) The ego-global abort, naming only the FIRST of the two assignments.
        class _Gobs:
            task_target_ids = [str(t.steps[0].target_id) for t in tasks]

        new_solution = apply_meta_action(
            solution, _Gobs(), ego, int(MetaAction.SELF_PRESERVATION_ABORT), far_idx, tasks
        )
        assert new_solution[ego] == [], (
            "the abort left %r; a non-empty plan never reaches the RTB branch"
            % (new_solution[ego],)
        )
        executor.resync(new_solution, ego_id=ego, tasks=tasks)
        assert executor.plans[ego] == [], executor.plans[ego]

        # (5) The empty-plan branch: exactly ONE RTB command, and nothing else.
        rtb_commands = executor.next_actions(scenario)
        assert rtb_commands == ["aircraft_return_to_base('%s')" % ego], rtb_commands
        game.handle_action(rtb_commands)

        # (6) What the REAL engine did with it.
        assert aircraft.rtb is True, "Game.aircraft_return_to_base did not set rtb"
        route = [list(wp) for wp in aircraft.route]
        assert route == [home_waypoint], (
            "the route must be replaced by the ride home, got %r (home %r)"
            % (route, home_waypoint)
        )
        for waypoint in stale_route:
            assert waypoint not in route, (
                "stale waypoint %r survived the RTB" % (waypoint,)
            )
        assert str(aircraft.home_base_id) == home_id, aircraft.home_base_id

        # (7) No second RTB: a second `aircraft_return_to_base` is a TOGGLE and would
        #     cancel the one just issued. The executor's latch is what prevents it.
        assert executor.next_actions(scenario) == [], "a second RTB command was emitted"
        assert aircraft.rtb is True and [list(wp) for wp in aircraft.route] == route
    finally:
        env.close()


# =============================================================================
# BLADE (no solver) -- P5: the engine constant + the redundant-salvo regression (Defect B)
# =============================================================================

# Generous cap: the slowest rack entry (AGM-65, 600 kt) is bounded at ~153 ticks at this
# range, so a run that hits this cap has stopped advancing rather than merely taken time.
_MAX_ENGAGEMENT_CALLS = 400


def _engage_one_target_with_real_blade(
    distance_km: float, *, fixed_wait: bool
) -> Dict[str, Any]:
    """Drive ONE real ego against ONE real enemy airbase and record what happened.

    Everything is real except the loadout trim: a real `Game` from `_build_env`, real
    agents/tasks from `_extract_world`, a real targeted launch, the real
    `GraphPlanExecutor` choosing every command, and the real `Game.handle_action` +
    `Game.update_game_state` resolving them tick by tick.

    THE LOADOUT TRIM. The B-2's rack is AIM-120 (2600 kt) / AIM-9 (1500 kt) / AGM-65
    (600 kt), and the engine's 2-arg attack always picks the highest ENGAGEMENT RANGE --
    the AIM-120 -- until it is spent. The probe reached the failing state after two
    AIM-120 salvos, at which point `weaponEngagement.launch_weapon` DROPS the exhausted
    entry (`origin.weapons.remove(launched_weapon)` once `current_quantity < 1`). We
    reproduce that end state directly by removing the same entry, so BLADE's own
    selection then returns the AIM-9 with the AGM-65 reserve behind it. Nothing else
    about the aircraft, the weapons or the engine is touched.

    `fixed_wait=True` selects the RETIRED behaviour as a control arm -- the executor with
    `_confirmation_wait_ticks` forced back to the flat configured constant. It is a
    test-local subclass, never production code.
    """
    from match_aou.rl.training import graph_episode_setup as ges
    from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor
    from blade.utils.utils import (
        get_distance_between_two_points,
        get_terminal_coordinates_from_distance_and_bearing,
    )

    class _FixedWaitExecutor(GraphPlanExecutor):
        """The pre-fix contract: one flat configured wait, whatever is about to fly."""

        def _confirmation_wait_ticks(
            self, scenario: object, ego_id: str, distance_km: float
        ) -> int:
            return self.kill_confirm_ticks

    executor_cls = _FixedWaitExecutor if fixed_wait else GraphPlanExecutor

    game, env, obs = ges._build_env(
        BASE_SCENARIO.read_text(encoding="utf-8"),
        max_episode_steps=MAX_SIM_TICKS,
        attacking_side_color=ATTACKING_SIDE_COLOR,
        record_every_seconds=10,
        recording_export_path=None,
    )
    try:
        agents, tasks = ges._extract_world(obs, ATTACKING_SIDE_COLOR)
        agent = agents[0]
        ego = str(agent.id)
        scenario = game.current_scenario

        assert game.launch_aircraft_from_airbase(str(agent.home_base_id), ego) is not None
        aircraft = scenario.get_aircraft(ego)
        assert aircraft is not None, "the ego is not airborne after the launch"

        # The loadout trim described above -- the engine's own post-exhaustion state.
        aircraft.weapons = [w for w in aircraft.weapons if "AIM-120" not in w.name]

        # A real enemy AIRBASE (destroyable, and what the construction cell places).
        task_idx, target_id, target_loc = next(
            (i, str(t.steps[0].target_id), t.steps[0].location)
            for i, t in enumerate(tasks)
            if scenario.get_airbase(str(t.steps[0].target_id)) is not None
        )
        latitude, longitude = get_terminal_coordinates_from_distance_and_bearing(
            target_loc.latitude, target_loc.longitude, distance_km, 90.0
        )
        aircraft.latitude, aircraft.longitude = latitude, longitude
        aircraft.route = []  # loitering: the executor issues no move once in range

        selected = aircraft.get_weapon_with_highest_engagement_range()
        reserve = [w for w in aircraft.weapons if w is not selected]

        executor = executor_cls(
            tasks=tasks, solution={ego: [(task_idx, 0, 0)]}, agents=agents,
            arrival_threshold_km=DETECTION_KM,
        )
        engagement_km = Location(latitude, longitude).distance_to(target_loc)
        derived_wait = executor._confirmation_wait_ticks(scenario, ego, engagement_km)

        attack_calls: List[int] = []
        target_alive_at: Dict[int, bool] = {}
        weapons_in_flight_at: Dict[int, int] = {}
        rack_quantity_at: Dict[int, int] = {}
        confirmed_at: Optional[int] = None
        rtb_at: Optional[int] = None

        for call in range(0, _MAX_ENGAGEMENT_CALLS):
            target_alive_at[call] = scenario.get_airbase(target_id) is not None
            weapons_in_flight_at[call] = len(scenario.weapons)
            live = scenario.get_aircraft(ego)
            rack_quantity_at[call] = (
                sum(w.current_quantity for w in live.weapons) if live is not None else -1
            )
            commands = executor.next_actions(scenario)
            for command in commands:
                if command.startswith("handle_aircraft_attack"):
                    attack_calls.append(call)
                elif command.startswith("aircraft_return_to_base") and rtb_at is None:
                    rtb_at = call
            if (ego, target_id) in executor.done and confirmed_at is None:
                confirmed_at = call
            game.handle_action(commands)
            game.update_game_state()
            if rtb_at is not None:
                break

        live = scenario.get_aircraft(ego)
        return {
            "ego": ego,
            "target_id": target_id,
            "engagement_km": engagement_km,
            "blade_engagement_km": get_distance_between_two_points(
                latitude, longitude, target_loc.latitude, target_loc.longitude
            ),
            "selected_name": selected.name,
            "selected_speed": selected.speed,
            "selected_lethality": selected.lethality,
            "reserve_names": [w.name for w in reserve],
            "derived_wait": derived_wait,
            "configured_wait": executor.kill_confirm_ticks,
            "attack_calls": attack_calls,
            "attack_commands": [
                "handle_aircraft_attack('%s', '%s')" % (ego, target_id)
            ] * len(attack_calls),
            "confirmed_at": confirmed_at,
            "rtb_at": rtb_at,
            "target_destroyed": scenario.get_airbase(target_id) is None,
            "ego_alive": live is not None,
            "remaining_rack": (
                [(w.name, w.current_quantity) for w in live.weapons]
                if live is not None else None
            ),
            "weapons_still_in_flight": len(scenario.weapons),
            "target_alive_at": target_alive_at,
            "weapons_in_flight_at": weapons_in_flight_at,
            "rack_quantity_at": rack_quantity_at,
        }
    finally:
        env.close()


@_needs_blade
def test_blade_transcribed_km_to_nm_constant_matches_the_engine() -> None:
    """P5a: the executor's transcribed km -> nm constant IS the frozen engine's own.

    `blade_graph_executor` is an import-purity ENTRY_MODULE and stays BLADE-free at import
    time, so it transcribes `KILOMETERS_TO_NAUTICAL_MILES` instead of importing it -- the
    same trade `graph_fuel_damage` makes for NAUTICAL_MILES_TO_METERS. A transcription is
    only safe if something compares it with the source, and only the BLADE tier can: the
    engine import lives INSIDE this test body, so the pure tier still imports nothing from
    BLADE and this test is skipped wherever the engine is absent.

    This is the check that would catch drift; the literal pinned in P4a cannot.
    """
    from blade.utils.constants import (
        KILOMETERS_TO_NAUTICAL_MILES as BLADE_KM_TO_NM,
    )

    from match_aou.utils.blade_utils.blade_graph_executor import (
        KILOMETERS_TO_NAUTICAL_MILES as EXECUTOR_KM_TO_NM,
    )

    assert EXECUTOR_KM_TO_NM == BLADE_KM_TO_NM, (
        "the executor's transcribed km -> nm constant (%r) has drifted from the engine's "
        "own blade.utils.constants value (%r); every derived confirmation wait is scaled "
        "by it" % (EXECUTOR_KM_TO_NM, BLADE_KM_TO_NM)
    )

    # It is the constant the engine's own weapon-flight step divides speed into, so pin
    # the exact identity rather than merely "some float that happens to match".
    import blade.utils.utils as blade_utils

    assert blade_utils.KILOMETERS_TO_NAUTICAL_MILES == BLADE_KM_TO_NM


@_needs_blade
def test_blade_derived_wait_prevents_the_redundant_salvo() -> None:
    """P5: a slower auto-selected salvo is no longer re-fired over while it is airborne.

    THE MEASURED DEFECT. In the first short probe a B-2 launched its last AIM-9 pair at a
    hidden airbase, the flat 60-tick wait expired while that pair was still flying, the
    executor issued a second attack, BLADE auto-selected the remaining AGM-65 pair, and
    the ego reached its final known target with an empty rack. Both arms below are driven
    against the REAL engine at the executor's production default `kill_confirm_ticks=60`;
    the only difference is whether the wait is derived.

    WHAT THE CONTROL ARM DOES AND DOES NOT PROVE. It reproduces the MECHANISM -- a flat
    wait expiring while the auto-selected salvo is still airborne, a redundant command,
    and the reserve consumed -- inside the same `DETECTION_KM = 50` engagement envelope.
    It does NOT reproduce every detail of the original probe's world state, and nothing
    here should be read as a re-run of that episode.

    THE CONTROL ARM'S SCHEDULE, which fixes every call index used here: the executor arms
    the cooldown on the firing call, decrements it once per later call, so a flat 60
    reaches 0 on call 60 and the earliest re-fire is call 61. The confirm-guard runs
    FIRST on every call, so a kill visible by call 61 is confirmed instead of re-fired.

    BOUNDS ARE NOT ENGINE TICKS. `_salvo_travel_ticks` returns a conservative
    full-distance bound; BLADE resolves earlier (it advances a weapon once inside
    `launch_weapon`, may advance it again in the same `update_game_state`, and calls
    `weapon_endgame` once the remaining distance is under 1 km). Both are measured here
    and they differ, which is the point of asserting each separately.

    TWO DISTANCES, both inside the single 50 km attack envelope:
      * 47.2 km -- the distance reconstructed from the probe's artifacts. The bound is 62
        and the derived wait 63 against a flat 60: the old constant was ALREADY below the
        bound for the salvo it was covering. Real confirmation lands on call 60, one
        single tick before the flat wait would have permitted a second command -- a
        one-tick escape, not a safety margin.
      * 49.0 km -- the same envelope, far enough out that the escape is gone (bound 64,
        derived wait 65, real confirmation on call 62 against a re-fire on call 61), so
        the control arm exhibits the premature re-fire and loses the reserve.
    """
    close = _engage_one_target_with_real_blade(47.2, fixed_wait=False)
    close_control = _engage_one_target_with_real_blade(47.2, fixed_wait=True)
    far = _engage_one_target_with_real_blade(49.0, fixed_wait=False)
    far_control = _engage_one_target_with_real_blade(49.0, fixed_wait=True)

    # ---- the fixture really is the observed shape -----------------------------
    for run in (close, close_control, far, far_control):
        assert run["selected_name"].startswith("AIM-9"), run["selected_name"]
        assert run["selected_speed"] == 1500, run["selected_speed"]
        assert run["selected_lethality"] == 1.0, run["selected_lethality"]  # deterministic
        assert any(n.startswith("AGM-65") for n in run["reserve_names"]), run["reserve_names"]
        assert run["configured_wait"] == 60, run["configured_wait"]
        assert run["ego_alive"], "the ego did not survive the engagement"
    # The distance the executor derives from (our `Location.distance_to`) and the one the
    # engine flies the weapon over agree to well under a metre -- orders of magnitude away
    # from the nearest ceiling boundary, so the derived tick count cannot hinge on it.
    for run in (close, far):
        assert abs(run["engagement_km"] - run["blade_engagement_km"]) < 1e-3, run

    # ---- 47.2 km: the flat constant was already BELOW the salvo's bound ------
    # 47.2 km at 1500 kt is 61.17 s -> a conservative bound of 62, +1 margin -> 63.
    assert close["derived_wait"] == 63, close["derived_wait"]
    assert close["derived_wait"] > close["configured_wait"]
    # ... and the control arm's escape is exactly one tick wide: it confirms on call 60,
    # the very call before the flat wait would have permitted a second command.
    assert close_control["confirmed_at"] == 60, close_control["confirmed_at"]
    assert close_control["attack_calls"] == [0], close_control["attack_calls"]

    # ---- 49.0 km: the control arm exhibits the probe's mechanism --------------
    # 49.0 km at 1500 kt is 63.5 s -> a conservative bound of 64, +1 margin -> 65.
    assert far["derived_wait"] == 65, far["derived_wait"]
    # THE SALVO IS STILL AIRBORNE when the flat wait expires: on call 61 the target is
    # alive, the AIM-9 pair is still in the air, and the AGM-65 reserve is still onboard.
    assert far["target_alive_at"][61] is True
    assert far["weapons_in_flight_at"][61] == 2, far["weapons_in_flight_at"][61]
    assert far["rack_quantity_at"][61] == 2, far["rack_quantity_at"][61]
    # The control arm fires again on exactly that call and burns the whole reserve.
    assert far_control["attack_calls"] == [0, 61], far_control["attack_calls"]
    assert far_control["remaining_rack"] == [], far_control["remaining_rack"]
    assert far_control["weapons_still_in_flight"] == 1, far_control["weapons_still_in_flight"]

    # ---- the derived wait: ONE salvo, reserve intact, target still dead -------
    assert far["attack_calls"] == [0], far["attack_calls"]
    assert far["attack_commands"] == [
        "handle_aircraft_attack('%s', '%s')" % (far["ego"], far["target_id"])
    ], far["attack_commands"]
    assert far["remaining_rack"] == [("AGM-65 Maverick", 2)], far["remaining_rack"]
    assert far["weapons_still_in_flight"] == 0, far["weapons_still_in_flight"]
    assert far["target_destroyed"], "the single derived-wait salvo did not kill the target"
    # PLAN ADVANCEMENT IS NOT DELAYED: the confirm-guard advances on the same call the
    # kill becomes visible (62 -- EARLIER than the bound of 64, as expected), with 3 ticks
    # still on the derived clock, and the now empty plan issues RTB immediately: the
    # longer wait throttles RE-FIRE only.
    assert far["confirmed_at"] == 62, far["confirmed_at"]
    assert far["rtb_at"] == far["confirmed_at"], (far["rtb_at"], far["confirmed_at"])
    assert far["confirmed_at"] < far["derived_wait"], far

    # The same holds at the reconstructed distance.
    assert close["attack_calls"] == [0], close["attack_calls"]
    assert close["remaining_rack"] == [("AGM-65 Maverick", 2)], close["remaining_rack"]
    assert close["target_destroyed"] and close["weapons_still_in_flight"] == 0
    assert close["rtb_at"] == close["confirmed_at"] == 60, close


# =============================================================================
# PURE -- P6: RTB ISSUANCE vs PHYSICAL COMPLETION (Defect C)
# =============================================================================
#
# `is_done` used to read the `rtb_issued` LATCH, i.e. "the order went out", and the tick
# loop stopped there -- so an episode could end while the aircraft was still airborne and
# a doomed ego was scored as a survivor. Completion is now a PHYSICAL fact read off the
# live observation: a non-dead ego is resolved only once it is back in an airbase
# inventory, and an ego that is in neither the air nor any inventory has been removed by
# the engine and is reconciled into `dead`.
#
# These stubs model exactly the three engine states and nothing else -- an aircraft lives
# in `scenario.aircraft` (flying), in some `airbase.aircraft` (landed, where
# `Game.land_aicraft` puts it), or in neither (`Game.remove_aircraft`, the empty tank).


class _LifeAircraft:
    """A live aircraft: only identity and position are read by the lifecycle path."""

    def __init__(self, uid: str, lat: float = 32.0, lon: float = 35.0) -> None:
        self.id, self.latitude, self.longitude = uid, lat, lon
        self.route: List[Any] = []


class _LifeAirbase:
    """An airbase whose `aircraft` inventory is what "landed" actually means."""

    def __init__(self, uid: str, aircraft: Optional[List[Any]] = None) -> None:
        self.id = uid
        self.aircraft = list(aircraft or [])
        self.latitude, self.longitude = 32.0, 35.0


class _LifeWorld:
    """The two lists the lifecycle reads, plus the liveness probe the guard uses."""

    def __init__(self, aircraft: List[Any], airbases: List[Any]) -> None:
        self.aircraft, self.airbases = list(aircraft), list(airbases)
        self.facilities: List[Any] = []
        self.ships: List[Any] = []

    def get_target(self, _target_id: str) -> Optional[Any]:
        return None  # every target already gone; nothing here is about kills

    def land(self, aircraft: Any, base: Any) -> None:
        """What `Game.land_aicraft` does: inventory first, then out of the air."""
        base.aircraft.append(aircraft)
        self.aircraft.remove(aircraft)

    def burn_out(self, aircraft: Any) -> None:
        """What `Game.remove_aircraft` does on an empty tank: gone, with no home."""
        self.aircraft.remove(aircraft)


def _life_executor(solution, *, tasks=None, add_return_to_base=True, agent_ids=None):
    from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor

    ids = list(agent_ids if agent_ids is not None else solution.keys())
    agents = [_StubAgent(a, Location(32.0, 35.0), Location(32.0, 35.0)) for a in ids]
    return GraphPlanExecutor(
        tasks=list(tasks or []), solution=solution, agents=agents,
        arrival_threshold_km=DETECTION_KM, add_return_to_base=add_return_to_base,
    )


def test_rtb_issuance_is_not_physical_completion() -> None:
    """P6a: ONE RTB order, no second toggle, and completion only on the landing.

    The whole defect in one sequence: the order goes out on the first empty-plan
    airborne tick, `rtb_issued` latches, and the episode must NOT be finished -- the
    aircraft is still in the air. Completion arrives only when the engine has actually
    moved it into an airbase inventory.
    """
    ego = "ego"
    aircraft = _LifeAircraft(ego)
    base = _LifeAirbase("base")
    world = _LifeWorld([aircraft], [base])
    executor = _life_executor({ego: []})

    # (1) The first airborne tick with nothing left to do: exactly ONE RTB order.
    assert executor.next_actions(world) == ["aircraft_return_to_base('%s')" % ego]
    assert executor.rtb_issued.get(ego) is True

    # (2) THE DEFECT: the latch is set, and the episode is NOT over.
    assert executor.is_done(world) is False, (
        "issuing the RTB order completed the episode -- that is Defect C"
    )

    # (3) The ride home: no second toggle (it would CANCEL the RTB), still not done.
    for _ in range(5):
        assert executor.next_actions(world) == [], "a second RTB toggle was emitted"
        assert executor.is_done(world) is False
    assert ego not in executor.dead

    # (4) The engine lands it. NOW, and only now, the episode is over.
    world.land(aircraft, base)
    assert executor.is_done(world) is True
    assert ego not in executor.dead, "a landed ego must never be counted as lost"
    assert executor.next_actions(world) == [], "a landed ego was commanded again"


def test_a_burn_out_on_the_ride_home_is_reconciled_dead() -> None:
    """P6b: an ego removed mid-return is recorded dead by the completion check itself.

    This is the accounting half of the defect: under issuance-based completion the
    episode stopped at (1), so the engine never got the ticks in which the tank ran dry
    and `dead` stayed empty -- a doomed aircraft scored as a survivor.
    """
    ego = "ego"
    aircraft = _LifeAircraft(ego)
    world = _LifeWorld([aircraft], [_LifeAirbase("base")])
    executor = _life_executor({ego: []})

    # (1) Ordered home, still flying, not finished, not dead.
    assert executor.next_actions(world) == ["aircraft_return_to_base('%s')" % ego]
    assert executor.is_done(world) is False and not executor.dead

    # (2) The tank runs dry: removed from the air, with no inventory to land in.
    world.burn_out(aircraft)

    # (3) The completion check itself reconciles the loss -- no other call is needed,
    #     which is what makes it visible to `EpisodeResult.n_dead` and the reward.
    assert executor.is_done(world) is True
    assert executor.dead == {ego}
    assert executor.next_actions(world) == [], "a dead ego was commanded"


def test_death_reconciliation_covers_every_ego_before_any_verdict() -> None:
    """P6c: a peer that is still working cannot hide another ego's death.

    The reconciliation pass is TOTAL and runs before the verdict, so the early
    "not done" produced by `a_working` does not stop `z_burned` from being recorded.
    Ids are chosen so the unfinished ego is visited FIRST in sorted order.
    """
    working, burned = "a_working", "z_burned"
    task = _task("tgt-1", 32.0, 35.0)
    flying = _LifeAircraft(working)
    doomed = _LifeAircraft(burned)
    world = _LifeWorld([flying, doomed], [_LifeAirbase("base")])
    executor = _life_executor(
        {working: [(0, 0, 0)], burned: []}, tasks=[task]
    )

    world.burn_out(doomed)

    assert executor.is_done(world) is False, "the working ego is not finished"
    assert executor.dead == {burned}, (
        "the death was skipped because a peer returned 'not done' first"
    )


def test_no_return_to_base_contract_is_preserved() -> None:
    """P6d: `add_return_to_base=False` still requires no return of any kind.

    Callers that opted out get the pre-Defect-C behaviour exactly: no RTB command is
    ever emitted, and an airborne ego whose work is finished IS done.
    """
    ego = "ego"
    aircraft = _LifeAircraft(ego)
    world = _LifeWorld([aircraft], [_LifeAirbase("base")])
    task = _task("tgt-1", 32.0, 35.0)

    finished = _life_executor({ego: []}, add_return_to_base=False)
    assert finished.next_actions(world) == [], "an RTB was emitted with RTB disabled"
    assert finished.is_done(world) is True, (
        "airborne-but-finished must stay done when no return is required"
    )
    assert finished.rtb_issued.get(ego) is None

    # Unfinished work still blocks completion, RTB policy or not.
    busy = _life_executor({ego: [(0, 0, 0)]}, tasks=[task], add_return_to_base=False)
    assert busy.is_done(world) is False


def test_physical_classification_cannot_be_moved_by_a_peer() -> None:
    """P6e: only the ego's OWN entries decide where the ego is (no-communication).

    The classifier is handed worlds that differ ONLY in peer placement -- peers flying,
    peers landed, peers removed entirely -- and must return the same verdict for the ego
    every time, so no peer's lifecycle can leak into this ego's completion.
    """
    ego, peer = "ego", "peer"
    executor = _life_executor({ego: []}, agent_ids=[ego, peer])
    base = _LifeAirbase("base")

    def _classify(build) -> Tuple[str, bool]:
        world, ego_aircraft = build()
        state = executor._physical_state(ego, world)
        # `is_done` is asked about the ego alone (only it has a plan slice).
        return state, executor.is_done(world)

    def _airborne_ego(peer_where: str):
        def build():
            ac = _LifeAircraft(ego)
            peer_ac = _LifeAircraft(peer)
            home = _LifeAirbase("base")
            if peer_where == "air":
                return _LifeWorld([ac, peer_ac], [home]), ac
            if peer_where == "landed":
                home.aircraft.append(peer_ac)
                return _LifeWorld([ac], [home]), ac
            return _LifeWorld([ac], [home]), ac  # peer removed entirely
        return build

    verdicts = {w: _classify(_airborne_ego(w)) for w in ("air", "landed", "gone")}
    assert set(verdicts.values()) == {("airborne", False)}, verdicts
    assert not executor.dead, executor.dead

    # And the same invariance once the ego itself is home.
    landed = []
    for peer_where in ("air", "landed", "gone"):
        ac = _LifeAircraft(ego)
        peer_ac = _LifeAircraft(peer)
        home = _LifeAirbase("base", aircraft=[ac])
        air = [peer_ac] if peer_where == "air" else []
        if peer_where == "landed":
            home.aircraft.append(peer_ac)
        world = _LifeWorld(air, [home])
        landed.append((executor._physical_state(ego, world), executor.is_done(world)))
    assert set(landed) == {("landed", True)}, landed
    assert base.aircraft == []


# =============================================================================
# BLADE (no solver) -- P7: the real ride home reaches the terminal result (Defect C)
# =============================================================================
#
# Everything below is the real engine: a real `Game` from `_build_env`, real agents and
# tasks from `_extract_world`, a real targeted launch, the real `GraphPlanExecutor`
# choosing every command, the real `run_episode` two-phase tick, and real `env.step`
# physics deciding whether the aircraft gets home. Only `_wake_decision` is stubbed --
# the encoder and head decide nothing about a lifecycle, and stubbing them keeps this
# tier solver-free and deterministic.

# One degree of latitude in kilometres; used only to displace the ego from its base so
# the ride home takes a measurable number of ticks instead of resolving instantly.
# The displacement is SOUTHWARD on purpose: this template's two red SAM facilities sit
# NORTH of the blue base, and flying the return leg through their envelope would let a
# surface-to-air kill masquerade as a fuel burn-out. Staying south is a PRECAUTION, not
# the proof -- the causal evidence is the `Game.remove_aircraft` witness below, which
# records which of the engine's removal branches actually fired.
_KM_PER_DEGREE_LATITUDE = 111.19492664455873
_RETURN_DISTANCE_KM = 120.0


class _ReturnCtx:
    """The `EpisodeContext` surface `run_episode` and `compute_episode_reward` read."""

    def __init__(self, *, env, game, ego, executor, observation, tasks) -> None:
        from match_aou.rl.training.belief import Belief

        self.env, self.game = env, game
        self.agent_ids = [ego]
        self.executor = executor
        self.observation = observation
        self.record = False
        self.beliefs = {ego: Belief.independent(list(tasks), {ego: []})}
        # A hand-built oracle: `plan_value` is pure arithmetic over (solution, tasks),
        # so the reward path needs no solver to be exercised honestly.
        self.oracle_tasks = list(tasks)
        self.oracle_solution = {ego: [(i, 0, 0) for i in range(len(tasks))]}


def _fly_home_with_real_blade(*, fuel_multiplier: float) -> Dict[str, Any]:
    """Order ONE real ego home from `_RETURN_DISTANCE_KM` out and report what happened.

    `fuel_multiplier` scales the engine's OWN `get_fuel_needed_to_return_to_base`, so
    `> 1` means the aircraft can reach home and `< 1` means it cannot. Nothing else is
    tuned: the burn, the movement, the landing and the removal are all BLADE's.
    """
    from match_aou.rl.observation.graph_builder import GraphObservationConfig
    from match_aou.rl.training import graph_episode_setup as ges
    from match_aou.rl.training import graph_tick_loop
    from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor

    game, env, obs = ges._build_env(
        BASE_SCENARIO.read_text(encoding="utf-8"),
        max_episode_steps=MAX_SIM_TICKS,
        attacking_side_color=ATTACKING_SIDE_COLOR,
        record_every_seconds=10,
        recording_export_path=None,
    )

    # THE CAUSAL WITNESS. TWO of the engine's three ways to take our ego out of the
    # air call `Game.remove_aircraft`, and they are distinguishable AT THE CALL:
    #   * the empty tank -- `update_all_aircraft_position` decrements `current_fuel` by
    #     `fuel_rate / 3600` and calls `remove_aircraft` when the result is `<= 0`,
    #     with nothing appended anywhere;
    #   * the landing -- `land_aicraft` appends the replacement airframe to the homebase
    #     inventory FIRST and only then calls `remove_aircraft`.
    # So the live fuel reading plus "is this id already in an airbase inventory?"
    # separate those two exactly.
    #
    # The THIRD way -- a weapon kill -- deliberately does NOT appear here:
    # `weaponEngagement.weapon_endgame` splices the target straight out of
    # `current_scenario.aircraft` and never calls `Game.remove_aircraft`. It is still
    # excluded, and more strongly than a sampled weapon count could manage: an aircraft
    # leaves the simulation exactly ONCE, so a shot-down ego would leave NO record here
    # at all. Observing exactly one recorded removal, at `current_fuel <= 0`, is
    # therefore incompatible with having been shot down.
    #
    # This wraps the bound method on THIS Game INSTANCE only; the FROZEN engine source
    # is untouched, and the instance attribute is dropped in the `finally` below.
    removals: List[Dict[str, Any]] = []
    real_remove = game.remove_aircraft

    def _witness_remove(aircraft_id):
        live = game.current_scenario.get_aircraft(aircraft_id)
        removals.append({
            "id": str(aircraft_id),
            "fuel": None if live is None else float(live.current_fuel),
            "in_inventory": any(
                str(getattr(ac, "id", "")) == str(aircraft_id)
                for base in (getattr(game.current_scenario, "airbases", []) or [])
                for ac in (getattr(base, "aircraft", []) or [])
            ),
        })
        return real_remove(aircraft_id)

    game.remove_aircraft = _witness_remove
    try:
        agents, tasks = ges._extract_world(obs, ATTACKING_SIDE_COLOR)
        agent = agents[0]
        ego, home_id = str(agent.id), str(agent.home_base_id)

        scenario = game.current_scenario
        from blade.Doctrine import DoctrineType

        assert game.launch_aircraft_from_airbase(home_id, ego) is not None
        aircraft = scenario.get_aircraft(ego)
        assert aircraft is not None
        assert not scenario.check_side_doctrine(
            aircraft.side_id, DoctrineType.AIRCRAFT_RTB_WHEN_OUT_OF_RANGE
        ), "the engine would toggle rtb itself; the single-issue latch is not sound"

        home = scenario.get_aircraft_homebase(ego)
        assert home is not None and str(home.id) == home_id

        # Displace it straight south of home, so the ride home is a real flight that
        # stays clear of the northern SAM belt (see `_RETURN_DISTANCE_KM`).
        aircraft.latitude = home.latitude - _RETURN_DISTANCE_KM / _KM_PER_DEGREE_LATITUDE
        aircraft.longitude = home.longitude
        aircraft.current_fuel = (
            float(fuel_multiplier) * game.get_fuel_needed_to_return_to_base(aircraft)
        )

        # An EMPTY plan is the whole mission here: the executor's pre-existing
        # empty-plan branch orders the ride home on the very first tick.
        executor = GraphPlanExecutor(
            tasks=tasks, solution={ego: []}, agents=[agent],
            arrival_threshold_km=DETECTION_KM,
        )
        ctx = _ReturnCtx(env=env, game=game, ego=ego, executor=executor,
                         observation=obs, tasks=tasks)

        # `next_actions` runs exactly once per tick, so its call ordinal IS the tick.
        issued: List[Tuple[int, List[str]]] = []
        weapons_seen: List[int] = []
        fuel_seen: List[float] = []
        real_next = executor.next_actions

        def spy_next(observation):
            # `next_actions` is called exactly once per tick, so its call ordinal IS the
            # tick, and this is also a free per-tick sample of the world it acted on.
            weapons_seen.append(len(getattr(observation, "weapons", []) or []))
            live = observation.get_aircraft(ego)
            if live is not None:
                fuel_seen.append(float(live.current_fuel))
            commands = real_next(observation)
            issued.append((len(issued), list(commands)))
            return commands

        executor.next_actions = spy_next

        wakes: List[Tuple[str, int]] = []

        def spy_wake(_policy, ego_id, _obs, _belief, _executor, _cfg, tick, **_kw):
            wakes.append((str(ego_id), int(tick)))
            return graph_tick_loop.Transition(
                gobs=None, ego_id=str(ego_id), tick=int(tick),
                meta_action=0, node_v=0, log_prob=0.0, entropy=0.0,
            )

        # Generous but BOUNDED: the return leg at this aircraft's own knots speed plus
        # a wide margin, so a run that hits the cap has stopped progressing.
        km_per_tick = float(aircraft.speed) * 1.852 / 3600.0
        cap = int(4 * _RETURN_DISTANCE_KM / km_per_tick) + 50

        saved = graph_tick_loop._wake_decision
        graph_tick_loop._wake_decision = spy_wake
        try:
            result = graph_tick_loop.run_episode(
                None, ctx, GraphObservationConfig(detection_range_km=DETECTION_KM),
                max_ticks=cap,
            )
        finally:
            graph_tick_loop._wake_decision = saved

        final = scenario
        in_air = {str(getattr(ac, "id", "")) for ac in getattr(final, "aircraft", []) or []}
        in_base = {
            str(getattr(ac, "id", ""))
            for b in (getattr(final, "airbases", []) or [])
            for ac in (getattr(b, "aircraft", []) or [])
        }
        rtb_cmd = "aircraft_return_to_base('%s')" % ego
        rtb_ticks = [t for t, cmds in issued if rtb_cmd in cmds]
        return {
            "ego": ego, "ctx": ctx, "result": result, "cap": cap,
            "rtb_ticks": rtb_ticks, "wakes": wakes,
            "n_commands": sum(len(cmds) for _t, cmds in issued),
            "airborne": ego in in_air, "landed": ego in in_base,
            "removals": list(removals),
            "ego_removals": [r for r in removals if r["id"] == ego],
            "max_weapons": max(weapons_seen) if weapons_seen else 0,
            "last_fuel": fuel_seen[-1] if fuel_seen else None,
            "fuel_fell": bool(len(fuel_seen) > 1 and fuel_seen[-1] < fuel_seen[0]),
            # The engine's own per-tick burn, `fuel_rate / 3600`, unconditional.
            "burn_per_tick": float(aircraft.fuel_rate) / 3600.0,
        }
    finally:
        # Exact restore: drop the instance attribute, so the class method is live again.
        game.__dict__.pop("remove_aircraft", None)
        env.close()


@_needs_blade
def test_blade_a_fuelled_ego_flies_home_and_lands_before_the_episode_ends() -> None:
    """P7a: with fuel to spare the episode continues past the order and ends on landing.

    Under the retired contract the episode stopped on the tick the order was issued.
    Here it keeps ticking, the engine flies the aircraft home, and completion arrives
    only once BLADE has put it back in an airbase inventory.
    """
    out = _fly_home_with_real_blade(fuel_multiplier=4.0)
    ego, result = out["ego"], out["result"]

    assert out["rtb_ticks"] == [0], out["rtb_ticks"]
    assert out["n_commands"] == 1, "more than the single RTB order was issued"
    assert result.ticks > out["rtb_ticks"][0] + 1, (
        "the episode ended on the issuing tick -- that is Defect C"
    )
    assert result.ended == "done" and result.ticks < out["cap"], (result.ended, result.ticks)
    assert out["landed"] and not out["airborne"], out
    assert result.n_dead == 0 and out["ctx"].executor.dead == set()

    # The removal it DID go through is the landing branch, not the empty tank: fuel was
    # still positive, and `land_aicraft` had already put the replacement airframe in the
    # inventory before calling `remove_aircraft`.
    assert len(out["ego_removals"]) == 1, out["ego_removals"]
    landing = out["ego_removals"][0]
    assert landing["fuel"] is not None and landing["fuel"] > 0.0, landing
    assert landing["in_inventory"] is True, landing

    # Secondary guard only: no weapon was seen in any per-tick sample of this world.
    assert out["max_weapons"] == 0, "a shot was observed; this is no longer a clean flight"


@_needs_blade
def test_blade_an_ego_that_burns_out_on_the_ride_home_is_counted_dead() -> None:
    """P7b: too little fuel -> the loss is real, and the reward charges for it.

    The aircraft is given HALF the fuel the engine itself says the trip needs, so it is
    removed mid-return. The episode had to keep running for that to be able to happen at
    all, which is exactly what the retired issuance-based completion prevented.
    """
    from match_aou.rl.training.graph_reward import RewardConfig, compute_episode_reward

    out = _fly_home_with_real_blade(fuel_multiplier=0.5)
    ego, ctx, result = out["ego"], out["ctx"], out["result"]

    assert out["rtb_ticks"] == [0], out["rtb_ticks"]
    assert out["n_commands"] == 1, "more than the single RTB order was issued"
    assert result.ticks > out["rtb_ticks"][0] + 1, (
        "the episode ended on the issuing tick, so the burn-out could never occur"
    )
    assert not out["airborne"] and not out["landed"], (
        "the aircraft neither burned out nor landed: %r" % (out,)
    )
    # DIRECT CAUSAL EVIDENCE: the ego went through `Game.remove_aircraft` exactly once,
    # holding `current_fuel <= 0` and with NO replacement airframe in any inventory.
    # That is the empty-tank branch of `update_all_aircraft_position` and nothing else:
    # `land_aicraft` would have shown positive fuel and an inventory entry, and a weapon
    # kill would have produced no record here at all (it bypasses `remove_aircraft`),
    # which it cannot do while also being the one removal that WAS recorded.
    assert len(out["ego_removals"]) == 1, out["ego_removals"]
    removal = out["ego_removals"][0]
    assert removal["fuel"] is not None and removal["fuel"] <= 0.0, removal
    assert removal["in_inventory"] is False, removal

    # Supporting evidence, from the other side of the same rule: the last tank reading
    # sampled before the removal held LESS than one tick of burn, so the very next
    # `fuel_rate / 3600` decrement is exactly what took it to <= 0.
    assert out["fuel_fell"], out
    assert 0.0 < out["last_fuel"] <= out["burn_per_tick"], out

    # Secondary guard only: no weapon was seen in any per-tick sample of this world.
    assert out["max_weapons"] == 0, out["max_weapons"]
    assert ctx.executor.dead == {ego}, ctx.executor.dead
    assert result.n_dead == 1, result

    # The EXISTING reward path, unchanged, now sees the airframe that was really lost.
    cfg = RewardConfig(aircraft_penalty_coeff=2.25)
    breakdown = compute_episode_reward(ctx, result, cfg)
    assert breakdown.n_lost == 1, breakdown
    denominator = abs(breakdown.u_oracle) + cfg.regret_epsilon
    assert breakdown.penalty == 2.25 * breakdown.u_aircraft * 1 / denominator
    assert breakdown.penalty > 0.0, breakdown
    assert breakdown.reward == breakdown.ratio - breakdown.penalty


# =============================================================================
# SOLVER (BLADE + bonmin) -- P1 and P2 end to end
# =============================================================================

def _hidden_target_ids(ctx: EpisodeContext) -> List[str]:
    """World target ids that are in NO belief -- i.e. the constructed hidden half."""
    belief_ids = {
        _task_target_id(task)
        for belief in ctx.beliefs.values() for task in belief.tasks
    }
    return [
        _task_target_id(task) for task in ctx.oracle_tasks
        if _task_target_id(task) not in belief_ids
    ]


def _belief_snapshot(ctx: EpisodeContext) -> Dict[str, Tuple[Any, Any]]:
    return {
        ego: ([_task_target_id(t) for t in b.tasks], copy.deepcopy(b.solution))
        for ego, b in ctx.beliefs.items()
    }


@_needs_solver
def test_p1_construction_patch_reload_and_reproducibility() -> None:
    """P1: the construction path builds a truthful, reproducible source of truth."""
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_p1_"))
    ctx = ctx_repeat = None
    try:
        scenario_json = _known_only_scenario_json(tmp_dir, seed=0)
        # The known half, straight from the generated JSON: these ids must survive the
        # patch/reload untouched and keep their positions in the belief list.
        known_json_ids = [
            str(entry["id"])
            for entry in json.loads(scenario_json)["currentScenario"]["airbases"]
            if str(entry.get("sideColor", "")).lower() != ATTACKING_SIDE_COLOR
        ]
        assert len(known_json_ids) == 3, known_json_ids

        ctx = setup_episode(scenario_json, n_hidden=3, placement_rng=random.Random(0))
        meta = ctx.split_meta

        # (1) counts describe the EMITTED world, and say construction ran, not a split.
        assert meta["outcome"] == "construction" and meta["mode"] == "construction"
        assert (meta["known"], meta["hidden"]) == (3, 3), meta
        assert (meta["partial"], meta["full"]) == (3, 6), meta
        assert meta["n_hidden_requested"] == 3, meta
        assert len(ctx.placements) == 3, len(ctx.placements)

        # (2) the world really holds 6 targets; 3 of them are in NO belief, and every
        #     KNOWN target survived the patch/reload with its identity intact.
        assert len(ctx.oracle_tasks) == 6, len(ctx.oracle_tasks)
        hidden_ids = _hidden_target_ids(ctx)
        assert len(hidden_ids) == 3, hidden_ids
        world_target_ids = {_task_target_id(t) for t in ctx.oracle_tasks}
        assert set(known_json_ids) <= world_target_ids, (
            "a known target lost its identity across patch/reload"
        )
        assert set(hidden_ids).isdisjoint(known_json_ids)

        # (3) A_init's positional indices are valid against the env-2 belief tasks, and
        #     every belief task is an env-2 object bound to the reloaded world.
        belief_tasks = ctx.executor.tasks[ctx.agent_ids[0]]
        assert len(belief_tasks) == 3, len(belief_tasks)
        referenced = {int(a[0]) for tuples in ctx.a_init.values() for a in tuples}
        assert referenced == set(range(len(belief_tasks))), (referenced, len(belief_tasks))
        world_ids = {_task_target_id(t) for t in ctx.oracle_tasks}
        assert {_task_target_id(t) for t in belief_tasks} <= world_ids

        # (4) A_init's agent keys address the runtime egos.
        assert set(ctx.a_init) <= set(ctx.agent_ids), (set(ctx.a_init), ctx.agent_ids)
        assert set(ctx.beliefs) == set(ctx.agent_ids)

        # (5) the oracle was solved from ENV-2 and covers every hidden target.
        assert ctx.oracle_solution is not ctx.a_init
        oracle_ids = {_task_target_id(t) for t in ctx.oracle_tasks}
        assert set(hidden_ids) <= oracle_ids
        u_oracle = sum(int(t.utility) for t in ctx.oracle_tasks)
        assert u_oracle == 480, u_oracle       # 6 airbases x 80

        # (6) REPRODUCIBILITY by geometry, never by uuid: the same scenario + the same
        #     placement seed gives the same fingerprint while every hidden id differs.
        fingerprint = geometric_fingerprint(ctx.placements)
        assert fingerprint == tuple(meta["geometric_fingerprint"])
        ctx_repeat = setup_episode(
            scenario_json, n_hidden=3, placement_rng=random.Random(0)
        )
        assert geometric_fingerprint(ctx_repeat.placements) == fingerprint
        repeat_hidden = set(_hidden_target_ids(ctx_repeat))
        assert repeat_hidden.isdisjoint(set(hidden_ids)), (
            "hidden uuids repeated across runs -- the fingerprint check would be "
            "measuring ids rather than geometry"
        )
        # A different placement seed moves the geometry (the rng really drives it).
        ctx_other = setup_episode(
            scenario_json, n_hidden=3, placement_rng=random.Random(12345)
        )
        try:
            assert geometric_fingerprint(ctx_other.placements) != fingerprint
        finally:
            ctx_other.env.close()

        print("  P1 known=3 hidden=3 full=6 U_oracle=%d fingerprint=%s" %
              (u_oracle, fingerprint))
    finally:
        for c in (ctx, ctx_repeat):
            if c is not None:
                try:
                    c.env.close()
                except Exception:
                    pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_solver
def test_p1_cardinality_mismatch_fails_loudly() -> None:
    """A requested `n_hidden` that B2's one-per-route contract cannot meet must raise.

    Never truncated, padded or duplicated: the reference cell solves to three ego routes,
    so asking for two or four is a request this seam does not know how to honour.
    """
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_card_"))
    try:
        scenario_json = _known_only_scenario_json(tmp_dir, seed=0)
        for bad in (2, 4):
            _expect_raises(
                RuntimeError, f"n_hidden={bad}", setup_episode,
                scenario_json, n_hidden=bad, placement_rng=random.Random(0),
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_solver
def test_p1_zero_hidden_is_a_clean_construction_probe() -> None:
    """`n_hidden=0` places nothing, patches nothing, and never calls `split_tasks`."""
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024
    from match_aou.rl.training import graph_episode_setup as ges

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_zero_"))
    ctx = None
    real_split = ges.split_tasks

    def _forbidden(*_a: Any, **_k: Any):
        raise AssertionError("split_tasks ran on the construction path")

    try:
        scenario_json = _known_only_scenario_json(tmp_dir, seed=0)
        ges.split_tasks = _forbidden
        try:
            ctx = setup_episode(scenario_json, n_hidden=0, placement_rng=random.Random(0))
        finally:
            ges.split_tasks = real_split

        assert ctx.placements == ()
        assert ctx.split_meta["hidden"] == 0
        assert ctx.split_meta["known"] == ctx.split_meta["full"] == 3
        assert len(ctx.oracle_tasks) == 3
        assert _hidden_target_ids(ctx) == []
    finally:
        if ctx is not None:
            try:
                ctx.env.close()
            except Exception:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_solver
def test_p7_generalized_construction_reconciles_requested_and_realized() -> None:
    """P7. End to end: the generalized cell really builds, and its accounting is true.

    Runs the whole seam -- generator, env-1, a REAL bonmin solve, bounded B2 backoff, the
    JSON patch, env-2 reload and the oracle solve -- across two (A, H) cells, and checks
    what only the integrated path can show:

      * the patch adds exactly H_REALIZED targets, never H_REQUESTED phantom ones;
      * `known_target_ids` / `executed_target_ids` (the RAW pre-solve snapshots) reconcile
        with the audit, while the ALLOCATED-ONLY `oracle_tasks` deliberately need not --
        reading it as a world inventory is the defect those snapshots exist to close;
      * the requested counts survive a backoff instead of being rewritten;
      * reproducibility is by GEOMETRY at a fixed seed, never by uuid.

    `A = 2` is covered by the pure cell tests instead: two known tasks is the documented
    bonmin branch-and-bound symmetry stall (`CLAUDE.md` section 8), and paying ~15 minutes
    of solver time proves nothing this cell does not.
    """
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_p7_"))
    open_contexts: List[Any] = []
    try:
        for agents, requested in ((3, 3), (4, 2)):
            scenario_json = _known_only_cell_json(
                tmp_dir, seed=0, num_agents=agents, n_known=agents
            )
            known_json_ids = [
                str(entry["id"])
                for entry in json.loads(scenario_json)["currentScenario"]["airbases"]
                if str(entry.get("sideColor", "")).lower() != ATTACKING_SIDE_COLOR
            ]
            assert len(known_json_ids) == agents, known_json_ids

            ctx = setup_episode(
                scenario_json, n_hidden=requested, placement_rng=random.Random(0),
                hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
            )
            open_contexts.append(ctx)
            audit = ctx.construction_audit
            assert isinstance(audit, ConstructionAudit), audit

            # (1) the cell, as REQUESTED -- unchanged by whatever the backoff realized.
            assert audit.policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1, audit.policy
            assert audit.agent_count == agents == len(ctx.agent_ids), audit
            assert audit.known_requested == agents, audit
            assert audit.hidden_requested == requested, audit
            assert audit.total_requested == agents + requested, audit

            # (2) what was REALIZED, measured against the RAW world snapshots.
            realized = audit.hidden_realized
            assert 1 <= realized <= requested, realized
            assert realized == len(ctx.placements), (realized, len(ctx.placements))
            assert audit.known_realized == agents == len(ctx.known_target_ids), audit
            assert audit.total_realized == len(ctx.executed_target_ids), audit
            assert audit.total_realized == agents + realized, audit

            # (3) the patch added exactly H_REALIZED targets -- no phantoms.
            known_ids = set(ctx.known_target_ids)
            assert known_ids == set(known_json_ids), "a known target lost its identity"
            hidden_ids = [t for t in ctx.executed_target_ids if t not in known_ids]
            assert len(hidden_ids) == realized, hidden_ids
            assert known_ids <= set(ctx.executed_target_ids)

            # (4) env-2 is the authority, and the ORACLE is an allocation over it -- so it
            #     is a SUBSET of the world and is never the inventory.
            oracle_ids = {_task_target_id(t) for t in ctx.oracle_tasks}
            assert oracle_ids <= set(ctx.executed_target_ids), oracle_ids
            assert set(ctx.a_init) <= set(ctx.agent_ids)
            assert set(ctx.beliefs) == set(ctx.agent_ids)
            belief_ids = {
                _task_target_id(t)
                for belief in ctx.beliefs.values() for t in belief.tasks
            }
            assert belief_ids <= known_ids, "a hidden target leaked into a t=0 belief"

            # (5) candidate accounting covers the whole scheduled roster's ordinals.
            assert sorted(audit.candidate_order) == list(range(agents)), audit
            assert audit.considered_ordinals == audit.candidate_order[
                :len(audit.considered_ordinals)
            ], audit
            assert len(set(audit.selected_ordinals)) == realized, audit
            assert set(audit.selected_ordinals) <= set(audit.candidate_order), audit
            for candidate in audit.candidates:
                assert candidate.accepted == (candidate.reason is None), candidate
                assert (candidate.reason is None
                        or candidate.reason in BACKOFF_REJECTION_REASONS), candidate
            assert ctx.split_meta["hidden_policy"] == HIDDEN_POLICY_BOUNDED_BACKOFF_V1
            assert ctx.split_meta["hidden_realized"] == realized
            assert ctx.split_meta["n_hidden_requested"] == requested
            assert ctx.split_meta["hidden"] == realized
            assert ctx.split_meta["full"] == agents + realized

            # (6) reproducible by GEOMETRY at the same seed, while the uuids all differ.
            fingerprint = audit.geometric_fingerprint
            assert fingerprint == geometric_fingerprint(ctx.placements)
            repeat = setup_episode(
                scenario_json, n_hidden=requested, placement_rng=random.Random(0),
                hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
            )
            open_contexts.append(repeat)
            assert repeat.construction_audit.geometric_fingerprint == fingerprint
            assert repeat.construction_audit.selected_ordinals == audit.selected_ordinals
            assert repeat.construction_audit.candidate_order == audit.candidate_order
            repeat_hidden = [
                t for t in repeat.executed_target_ids
                if t not in set(repeat.known_target_ids)
            ]
            assert set(repeat_hidden).isdisjoint(hidden_ids), (
                "hidden uuids repeated across runs -- the fingerprint check would be "
                "measuring ids rather than geometry"
            )

            print("  P7 A=%d K=%d H_requested=%d H_realized=%d total=%d selected=%s"
                  % (agents, agents, requested, realized, audit.total_realized,
                     audit.selected_ordinals))
    finally:
        for c in open_contexts:
            try:
                c.env.close()
            except Exception:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_solver
def test_p7_bounded_backoff_accepts_a_world_the_exact_path_refuses() -> None:
    """P7. The two policies really do differ, on a REAL world, with real bonmin.

    Seed 2 of the reference cell is the case `CLAUDE.md` section 8 documents: the static
    solve leaves one of the three egos without a route, so B2 can only produce two
    placements. Under `exact_v1` that is a LOUD refusal and the whole episode is lost;
    under `bounded_backoff_v1` the world is ACCEPTED with `H_realized = 2 < H_requested =
    3`, and the shortfall is stated in the audit instead of being hidden.

    The two arms use the SAME scenario JSON and the SAME placement seed, so the ONLY
    difference between them is the cardinality policy.
    """
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_p7_short_"))
    ctx = None
    try:
        scenario_json = _known_only_cell_json(tmp_dir, seed=2, num_agents=3, n_known=3)

        # (1) the HISTORICAL policy still refuses this world, exactly as it always did.
        _expect_raises(
            RuntimeError, "exact cardinality on a routeless-ego world", setup_episode,
            scenario_json, n_hidden=3, placement_rng=random.Random(0),
        )

        # (2) the GENERALIZED policy accepts it -- and says what it really got.
        ctx = setup_episode(
            scenario_json, n_hidden=3, placement_rng=random.Random(0),
            hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
        )
        audit = ctx.construction_audit
        assert audit.hidden_requested == 3, audit
        assert audit.hidden_realized == 2 == len(ctx.placements), audit
        assert audit.realized_full_request is False, audit
        assert audit.total_requested == 6 and audit.total_realized == 5, audit

        # (3) the shortfall is NAMED: one candidate had no route in the allocated-only
        #     A_init, and it is recorded rather than silently dropped from the accounting.
        no_route = [c for c in audit.candidates if c.reason == REASON_NO_ROUTE]
        assert len(no_route) == 1, [c.reason for c in audit.candidates]
        assert no_route[0].ordinal not in audit.selected_ordinals, audit
        assert no_route[0].ego_id not in ctx.a_init or not ctx.a_init[no_route[0].ego_id], (
            "the candidate recorded as routeless does have a route in A_init"
        )
        assert audit.considered_ordinals == audit.candidate_order, (
            "a short walk must have exhausted every candidate"
        )

        # (4) the WORLD really holds 3 known + 2 hidden, and the patch added no phantom.
        assert len(ctx.known_target_ids) == 3, ctx.known_target_ids
        assert len(ctx.executed_target_ids) == 5, ctx.executed_target_ids
        assert ctx.split_meta["hidden"] == 2 and ctx.split_meta["full"] == 5
        assert ctx.split_meta["n_hidden_requested"] == 3
        assert ctx.split_meta["hidden_realized"] == 2

        print("  P7 seed 2: exact REFUSED, bounded accepted H_realized=2/3 selected=%s"
              % (audit.selected_ordinals,))
    finally:
        if ctx is not None:
            try:
                ctx.env.close()
            except Exception:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


@_needs_solver
def test_p2_a_hidden_world_target_enters_only_the_sensing_egos_belief() -> None:
    """P2: private sensing isolation, proven through the integrated setup/tick seam.

    The target is one the SETUP really constructed and patched into the world (not a
    hand-built stub), and the carrier is the real `run_episode` Phase-1 chain
    (`sensed_target_ids` -> `decide_triggers` -> `_wake_decision` -> `executor.resync`).
    The only thing stubbed is WHO sees it: sensing is forced to report the hidden target
    to ego A and nothing at all to the peers, which is precisely the private-sensing
    situation the no-communication claim is about and which a short deterministic episode
    could not otherwise be relied on to produce.
    """
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024
    import torch
    from match_aou.rl.training.graph_tick_loop import build_policy, run_episode

    tmp_dir = Path(tempfile.mkdtemp(prefix="b3_p2_"))
    ctx = None
    try:
        scenario_json = _known_only_scenario_json(tmp_dir, seed=0)
        ctx = setup_episode(scenario_json, n_hidden=3, placement_rng=random.Random(0))

        hidden_ids = _hidden_target_ids(ctx)
        assert len(hidden_ids) == 3, hidden_ids
        target_id = sorted(hidden_ids)[0]
        ego_a = ctx.agent_ids[0]
        peers = [e for e in ctx.agent_ids if e != ego_a]
        assert peers, "P2 needs at least one peer to prove isolation against"

        # (a) at t=0 the target is in the WORLD and the ORACLE but in NO belief.
        assert target_id in {_task_target_id(t) for t in ctx.oracle_tasks}
        for ego, belief in ctx.beliefs.items():
            assert target_id not in {_task_target_id(t) for t in belief.tasks}, ego
            assert target_id not in {
                _task_target_id(t) for t in ctx.executor.tasks[ego]
            }, ego

        before = _belief_snapshot(ctx)
        peer_slices_before = {
            e: [_task_target_id(t) for t in ctx.executor.tasks[e]] for e in peers
        }

        # (b) force PRIVATE sensing: only ego A sees the hidden target, and only once it
        #     is actually airborne -- a grounded ego senses nothing, which is exactly what
        #     the real `sensed_target_ids` does (it reads the ego's LIVE position). Keeping
        #     that half faithful is what lets the wake run through the unmodified chain.
        def _private_sensing(observation: Any, ego_id: str) -> Dict[str, Any]:
            if str(ego_id) != ego_a:
                return {}
            if observation.get_aircraft(ego_a) is None:
                return {}
            unit = observation.get_target(target_id)
            return {target_id: unit} if unit is not None else {}

        ctx.executor.sensed_target_ids = _private_sensing

        torch.manual_seed(0)
        result = run_episode(build_policy(), ctx, deterministic=True, max_ticks=10)

        # (c) ego A woke, through the real trigger path, and its belief now holds it.
        assert result.n_wakes >= 1, result
        assert {tr.ego_id for tr in result.trajectory} == {ego_a}, result.trajectory
        a_task_ids = [_task_target_id(t) for t in ctx.beliefs[ego_a].tasks]
        assert target_id in a_task_ids, a_task_ids
        # APPEND-ONLY: the pre-existing known tasks keep their positions (and indices).
        assert a_task_ids[:len(before[ego_a][0])] == before[ego_a][0]

        # (d) the resync carried it into ego A's EXECUTOR slice.
        a_slice = [_task_target_id(t) for t in ctx.executor.tasks[ego_a]]
        assert target_id in a_slice, a_slice

        # (e) every peer belief is byte-unchanged and still lacks the target.
        after = _belief_snapshot(ctx)
        for peer in peers:
            assert after[peer] == before[peer], (
                f"peer {peer} belief changed after ego {ego_a} privately sensed "
                f"{target_id} -- no-communication violated"
            )
            assert target_id not in after[peer][0], peer
            peer_slice = [_task_target_id(t) for t in ctx.executor.tasks[peer]]
            assert peer_slice == peer_slices_before[peer], peer
            assert target_id not in peer_slice, peer

        print("  P2 ego %s woke on hidden target %s; %d peer(s) byte-unchanged"
              % (ego_a[:8], target_id[:8], len(peers)))
    finally:
        if ctx is not None:
            try:
                ctx.env.close()
            except Exception:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Standalone runner (pytest is absent from nlp_env)
# =============================================================================

def _run_all() -> None:
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    print("=" * 78)
    print("test_graph_setup_seam -- B3 setup seam")
    print("BLADE=%s  bonmin=%s" % (HAVE_BLADE, _have_bonmin()))
    print("=" * 78)

    ran = skipped = 0
    for name, fn in tests:
        needs_solver = (
            name.startswith("test_p1") or name.startswith("test_p2")
            or name.startswith("test_p7")
        )
        needs_blade = (
            needs_solver or "environment_one" in name or "build_env" in name
            or name.startswith("test_blade")
        )
        if needs_solver and not HAVE_SOLVER:
            print("[SKIP] %s (needs BLADE + bonmin)" % name)
            skipped += 1
            continue
        if needs_blade and not HAVE_BLADE:
            print("[SKIP] %s (needs BLADE)" % name)
            skipped += 1
            continue
        fn()
        ran += 1
        print("[ OK ] %s" % name)

    print("-" * 78)
    print("%d passed, %d skipped" % (ran, skipped))
    if skipped:
        raise SystemExit(
            "%d test(s) skipped -- run under nlp_env for the full proof" % skipped
        )
    print("All assertions passed.")


if __name__ == "__main__":
    _run_all()
