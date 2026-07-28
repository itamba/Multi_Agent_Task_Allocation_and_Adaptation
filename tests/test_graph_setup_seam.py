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
  BLADE     -- needs the engine but NOT bonmin. Environment-1 ownership: an injected
               failure between "env-1 is up" and "A_init exists" must still close it.
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
P2  private sensing isolation, through the INTEGRATED setup/tick seam: a hidden target
    that the setup really put in the world is reported as sensed by ONE ego, and the real
    `run_episode` Phase-1 chain is what carries it into that ego's belief and executor
    slice. Every peer belief and executor slice must be byte-unchanged and must not
    contain the target -- which it never could, because no belief held it at t=0.

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
from match_aou.rl.training.graph_episode_setup import (  # noqa: E402
    ATTACKING_SIDE_COLOR,
    CONSTRUCTION_TARGET_CLASS,
    DETECTION_KM,
    HIDDEN_TARGET_NAME_TEMPLATE,
    MAX_SIM_TICKS,
    EpisodeContext,
    _rematerialize_known_tasks,
    _require_agent_ids_preserved,
    _require_airbase_only_targets,
    _resolve_construction_mode,
    _select_hidden_prototype,
    _shared_launch_point,
    _task_target_id,
    build_patched_scenario,
    setup_episode,
)
from match_aou.rl.training.graph_hidden_placement import (  # noqa: E402
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


def test_legacy_context_has_an_empty_placement_audit() -> None:
    """The audit field defaults to empty, so the legacy path's contract is unchanged."""
    ctx = EpisodeContext(
        env=None, game=None, observation=None, agents=[], agent_ids=[], beliefs={},
        executor=None, a_init={}, oracle_solution={}, oracle_tasks=[], split_meta={},
    )
    assert ctx.placements == ()
    assert ctx.record is False
    assert geometric_fingerprint(ctx.placements) == ()


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
        needs_solver = name.startswith("test_p1") or name.startswith("test_p2")
        needs_blade = needs_solver or "environment_one" in name
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
