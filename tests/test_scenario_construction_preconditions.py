"""
Unit tests for the scenario-construction preconditions (offline scenario-construction
phase, step 1 of 3).

These lock three latent-defect fixes that the new "generate known targets -> solve ->
route -> place hidden targets" order depends on:

  P1/P2  LAUNCH POINT      : the BLUE airbase's aircraft sit exactly at the airbase's
                             own (lat, lon), so `launch_aircraft_from_airbase` puts the
                             fleet airborne OVER its base instead of ~73 km away.
  P3     GENERATOR FALLBACK: `_adjust_aircraft_count`'s no-existing-aircraft branch
                             anchors a newly-built aircraft to the base too (the
                             latent source of the old ~73 km skew).
  P4/P5  `ensure_discovery_chain` SWITCH: default True is behaviour-preserving; False
                             is a true skip (Layer 1 never runs, its stats are absent).
  P6     REGRESSION         : the JSON/generator fix must not move a single RED-airbase
                             target -- target placement is a function of the BASE
                             coordinates and the rng stream only, never of the
                             aircraft's own coordinates.

B1 (known-only generator + explicit construction configuration) adds five more, all
about the generator's new `min_target_separation_km` / `strict_geometry` request:

  P7     KNOWN-ONLY WORLD  : the reference cell emits exactly 3 BLUE aircraft and 3 RED
                             airbases, no SAM facility, and Layer 1 is never called.
  P8     GEOMETRY          : every emitted target is >= 200 km from the BLUE launch base
                             and every known pair is >= 100 km apart, over a seed sweep.
  P9     LOUD FAILURE      : an impossible requested geometry raises
                             `TargetPlacementError` and writes no scenario file --
                             it never lowers the floor, relaxes the separation, or
                             leaves a target on its stale template coordinate.
  P10    REPRODUCIBILITY   : same seed + same config -> identical (lat, lon) fingerprint.
                             ID-INDEPENDENT on purpose: added red airbases get a fresh
                             uuid per generate() even at a fixed seed (CLAUDE.md Sec 8).
  P11    NO SILENT CHANGE  : the new fields default to (0.0, False), and asking for
                             0.0 separation explicitly reproduces P6's coordinates
                             exactly -- the separation feature costs no rng draw when
                             it is not requested.

The `_B1_CFG_KWARGS` below mirror what `graph_train.build_variation_config` builds from
the default `TrainConfig`; that mirroring is asserted on the trainer side, in
`tests/test_graph_train.py`, so this module stays torch-free.

Solver-free (no bonmin) for everything except P2, which needs a real BLADE `Game` +
`gymnasium` env and therefore is NOT a pytest-collected `test_*` function -- it is
driven only from this file's own `__main__` runner (the `tests/test_graph_train.py`
idiom, CLAUDE.md Sec 1), so base-env pytest collection never imports `blade`.

Run:
    pytest tests/test_scenario_construction_preconditions.py -q          (base env, P1/P3-P6)
    conda run -n nlp_env --no-capture-output \\
        python tests/test_scenario_construction_preconditions.py         (nlp_env, all incl. P2)
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so match_aou.* imports resolve

from match_aou.rl.training.graph_episode_setup import DETECTION_KM  # noqa: E402
from match_aou.utils.blade_utils.scenario_generator import (  # noqa: E402
    ScenarioGenerator,
    TargetPlacementError,
    VariationConfig,
    _haversine_km,
)

BASE_SCENARIO = ROOT / "data" / "scenarios" / "strike_training_4v5.json"

# The exact VariationConfig used by P4/P5/P6 (mirrors graph_episode_setup's own
# `_selftest_generator` Test 3a common dict).
_COMMON_CFG_KWARGS = dict(
    include_sams=False,
    num_red_airbases=(4, 4),
    randomize_red_airbase_positions=True,
    stretch_target_ratio=0.5,
    detection_km=DETECTION_KM,
    seed=7,
)

# Pinned BEFORE the EDIT-1/EDIT-2 launch-point fix landed (recorded from the
# pre-edit tree by running the exact P6 generation below). See P6 below for why
# this must never change: target placement must be a function of the BASE
# coordinates + rng stream only, never of the aircraft's own coordinates.
_P6_EXPECTED_RED_AIRBASE_COORDS = [
    (27.321954994302555, 38.539945895648835),
    (32.34347098171331, 35.15134974796372),
    (27.278498912742805, 38.25190849764679),
    (32.2861031728966, 35.28078535204865),
]

# --- B1: the offline scenario-construction reference cell -------------------------
# num_agents = n_known = 3, geometry 200/100, Layer 1 off, geometry declared STRICT.
# Mirrors `graph_train.build_variation_config(TrainConfig(...), seed)`; the mirror
# itself is asserted in tests/test_graph_train.py (this module must stay torch-free).
_B1_NUM_AGENTS = 3
_B1_N_KNOWN = 3
_B1_MIN_TARGET_DISTANCE_KM = 200.0
_B1_MIN_KNOWN_SEPARATION_KM = 100.0

_B1_CFG_KWARGS = dict(
    include_sams=False,
    num_aircraft=_B1_NUM_AGENTS,
    num_red_airbases=_B1_N_KNOWN,
    randomize_red_airbase_positions=True,
    stretch_target_ratio=0.5,
    min_target_distance_km=_B1_MIN_TARGET_DISTANCE_KM,
    min_target_separation_km=_B1_MIN_KNOWN_SEPARATION_KM,
    ensure_discovery_chain=False,
    strict_geometry=True,
    detection_km=DETECTION_KM,
)

# Great-circle distances are computed in double precision from lat/lon that were
# themselves derived from a distance, so the round trip is not exact. 1e-6 km = 1 mm:
# far tighter than any placement effect, far looser than float noise.
_DISTANCE_TOL_KM = 1e-6


def _b1_config(seed: int, **overrides: object) -> VariationConfig:
    """The B1 construction request at ``seed``, with optional field overrides."""
    return VariationConfig(**{**_B1_CFG_KWARGS, "seed": seed, **overrides})


def _generate_b1(tmp_path: Path, seed: int, **overrides: object) -> dict:
    """Generate one B1 scenario and return its parsed JSON."""
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))
    path = gen.generate(episode=0, config=_b1_config(seed, **overrides))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _blue_base_and_targets(data: dict):
    """``(base_lat, base_lon, [(lat, lon), ...])`` for the RED airbase targets."""
    sc = data["currentScenario"]
    blue = next(ab for ab in sc["airbases"] if ab.get("sideColor") == "blue")
    red = [ab for ab in sc["airbases"] if ab.get("sideColor") == "red"]
    return blue["latitude"], blue["longitude"], [
        (ab["latitude"], ab["longitude"]) for ab in red
    ]


# =============================================================================
# P1 -- launch point (data): every BLUE aircraft sits at the BLUE airbase's own
# (lat, lon) in the base template.
# =============================================================================

def test_p1_blue_aircraft_sit_at_the_blue_airbase() -> None:
    with open(BASE_SCENARIO, "r", encoding="utf-8") as f:
        data = json.load(f)
    sc = data["currentScenario"]

    blue_side_id = next(s["id"] for s in sc["sides"] if s["color"].lower() == "blue")
    blue_airbase = next(ab for ab in sc["airbases"] if ab["sideId"] == blue_side_id)

    aircraft = blue_airbase.get("aircraft", [])
    assert aircraft, "blue airbase has no aircraft -- vacuous pass guard tripped"

    for ac in aircraft:
        assert ac["latitude"] == blue_airbase["latitude"], (ac["name"], ac["latitude"])
        assert ac["longitude"] == blue_airbase["longitude"], (ac["name"], ac["longitude"])


# =============================================================================
# P2 -- launch point (behaviour): NOT a pytest test (needs BLADE + gymnasium).
# Driven only from __main__ under nlp_env.
# =============================================================================

def _p2_launch_point_behaviour() -> None:
    """Build a Game from the template, launch one BLUE aircraft, and assert it
    becomes airborne exactly OVER its base -- the property the fleet's fuel/route
    geometry actually depends on, not just the static JSON field."""
    import gymnasium
    from blade.Game import Game
    from blade.Scenario import Scenario

    with open(BASE_SCENARIO, "r", encoding="utf-8") as f:
        scenario_json = f.read()

    game = Game(
        current_scenario=Scenario(), record_every_seconds=None, recording_export_path=".",
    )
    game.load_scenario(scenario_json)
    env = gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=100)
    obs, _info = env.reset()

    blue_side = next(s for s in obs.sides if str(s.name).upper() == "BLUE")
    game.current_side_id = blue_side.id
    blue_airbase = next(ab for ab in obs.airbases if ab.side_id == blue_side.id)
    assert blue_airbase.aircraft, "blue airbase has no aircraft to launch"
    aircraft_id = str(blue_airbase.aircraft[0].id)

    launched = game.launch_aircraft_from_airbase(blue_airbase.id, aircraft_id=aircraft_id)
    assert launched is not None, "launch_aircraft_from_airbase returned None"
    assert launched.latitude == blue_airbase.latitude, (launched.latitude, blue_airbase.latitude)
    assert launched.longitude == blue_airbase.longitude, (launched.longitude, blue_airbase.longitude)

    env.close()
    print(
        "[P2] launched aircraft sits exactly at the airbase's (lat, lon) = "
        f"({blue_airbase.latitude}, {blue_airbase.longitude})   OK"
    )


# =============================================================================
# P3 -- generator fallback: `_adjust_aircraft_count`'s empty-inventory branch
# anchors the new aircraft to the base, not `base - 0.5deg`.
# =============================================================================

def test_p3_adjust_aircraft_count_fallback_anchors_to_base(tmp_path: Path) -> None:
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))

    base_lat, base_lon = 40.0, -70.0
    scenario = {
        "airbases": [
            {
                "id": "synthetic-blue-base",
                "sideId": gen._blue_side_id,
                "latitude": base_lat,
                "longitude": base_lon,
                "aircraft": [],  # empty -> only reachable path into the `else` branch
            }
        ]
    }

    gen._adjust_aircraft_count(scenario, desired=1, rng=random.Random(0))

    aircraft_list = scenario["airbases"][0]["aircraft"]
    assert len(aircraft_list) == 1, aircraft_list
    ac = aircraft_list[0]
    assert ac["latitude"] == base_lat, ac["latitude"]
    assert ac["longitude"] == base_lon, ac["longitude"]


# =============================================================================
# P4 -- default `ensure_discovery_chain=True` is behaviour-preserving.
# =============================================================================

def test_p4_default_ensure_discovery_chain_is_behaviour_preserving(tmp_path: Path) -> None:
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))
    cfg = VariationConfig(**_COMMON_CFG_KWARGS)  # ensure_discovery_chain left at default (True)

    gen.generate(episode=0, config=cfg)

    assert gen.last_generation_stats["min_radar_km"] == DETECTION_KM, gen.last_generation_stats


# =============================================================================
# P5 -- `ensure_discovery_chain=False` is a TRUE skip.
# =============================================================================

def test_p5_ensure_discovery_chain_false_is_a_true_skip(tmp_path: Path) -> None:
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_ensure_discovery_chain must not be called when disabled")

    gen._ensure_discovery_chain = _boom  # type: ignore[method-assign]

    cfg = VariationConfig(**{**_COMMON_CFG_KWARGS, "ensure_discovery_chain": False})
    gen.generate(episode=0, config=cfg)  # must NOT raise (proves _boom was never hit)

    l1_keys = (
        "easy_relocated", "easy_total", "easy_isolated",
        "stretch_relocated", "stretch_total", "stretch_isolated",
        "min_radar_km",
    )
    for key in l1_keys:
        assert key not in gen.last_generation_stats, (key, gen.last_generation_stats)


# =============================================================================
# P6 -- THE LOAD-BEARING REGRESSION: target placement is unmoved by the fix.
# =============================================================================

def test_p6_red_airbase_coordinates_unchanged_by_launch_point_fix(tmp_path: Path) -> None:
    """Target placement must be a pure function of the BASE coordinates + rng
    stream -- never of the aircraft's own coordinates. If this fails: STOP, do not
    edit the test or the generator -- report the two coordinate lists (see the
    dispatch's Sec 4, P6). It means something reads aircraft position during
    placement and the offline scenario-construction design needs revisiting."""
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))
    cfg = VariationConfig(**_COMMON_CFG_KWARGS)

    path = gen.generate(episode=0, config=cfg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sc = data["currentScenario"]
    red_airbases = [ab for ab in sc["airbases"] if ab.get("sideColor") == "red"]
    coords = [(ab["latitude"], ab["longitude"]) for ab in red_airbases]

    assert coords == _P6_EXPECTED_RED_AIRBASE_COORDS, (coords, _P6_EXPECTED_RED_AIRBASE_COORDS)


# =============================================================================
# P7 -- B1 main path: a KNOWN-ONLY world of exactly the reference cell.
# =============================================================================

def test_p7_construction_cell_emits_a_known_only_world(tmp_path: Path) -> None:
    """3 aircraft, 3 RED airbases, no SAM, and Layer 1 never runs.

    The Layer-1 half is proven the P5 way -- monkeypatch the method to RAISE. `generate`
    has no try/except around it, so a call cannot be swallowed; a silent
    `ensure_discovery_chain=True` would fail here instead of quietly re-clustering the
    known targets that B2 will place against.
    """
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Layer 1 must not run on the construction path")

    gen._ensure_discovery_chain = _boom  # type: ignore[method-assign]

    path = gen.generate(episode=0, config=_b1_config(seed=7))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sc = data["currentScenario"]

    blue = next(ab for ab in sc["airbases"] if ab.get("sideColor") == "blue")
    red = [ab for ab in sc["airbases"] if ab.get("sideColor") == "red"]

    assert len(blue.get("aircraft", [])) == _B1_NUM_AGENTS, len(blue.get("aircraft", []))
    assert len(red) == _B1_N_KNOWN, len(red)
    assert sc.get("facilities", []) == [], sc.get("facilities")
    # No red airbase carries aircraft -- they are targets, not launch sites.
    assert all(not ab.get("aircraft") for ab in red), [ab.get("name") for ab in red]

    # Layer 1's seven stats keys are absent when the pass is skipped (read with .get).
    for key in ("easy_relocated", "easy_total", "easy_isolated",
                "stretch_relocated", "stretch_total", "stretch_isolated",
                "min_radar_km"):
        assert key not in gen.last_generation_stats, (key, gen.last_generation_stats)


# =============================================================================
# P8 -- the requested geometry actually holds on every emitted target.
# =============================================================================

def test_p8_construction_geometry_holds_over_a_seed_sweep(tmp_path: Path) -> None:
    """Every target >= 200 km from the launch base; every known pair >= 100 km apart.

    Swept over ten seeds rather than one: the fleet is a random 3-of-4 subset of the
    template's aircraft, so the reachable zone boundaries -- and therefore the placement
    rings -- differ from seed to seed. A single seed could pass by luck.
    """
    for seed in range(10):
        data = _generate_b1(tmp_path / ("s%d" % seed), seed)
        base_lat, base_lon, coords = _blue_base_and_targets(data)
        assert len(coords) == _B1_N_KNOWN, (seed, len(coords))

        for lat, lon in coords:
            d = _haversine_km(base_lat, base_lon, lat, lon)
            assert d >= _B1_MIN_TARGET_DISTANCE_KM - _DISTANCE_TOL_KM, (seed, d)

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = _haversine_km(*coords[i], *coords[j])
                assert d >= _B1_MIN_KNOWN_SEPARATION_KM - _DISTANCE_TOL_KM, (
                    seed, i, j, d
                )


# =============================================================================
# P9 -- an impossible requested geometry FAILS LOUD (no silent relaxation).
# =============================================================================

def test_p9_impossible_separation_raises_and_writes_nothing(tmp_path: Path) -> None:
    """A 100 000 km separation cannot be met; the generator raises and writes no file.

    The failure mode this guards against is the quiet one: the pre-B1 loop simply gave
    up after `max_attempts` and left the target on its TEMPLATE coordinate, producing a
    scenario that looks generated but silently violates the requested geometry.
    """
    out_dir = tmp_path / "impossible"
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(out_dir))

    try:
        gen.generate(episode=0, config=_b1_config(seed=7, min_target_separation_km=100_000.0))
    except TargetPlacementError as exc:
        message = str(exc)
        assert "min_target_separation_km=100000.0" in message, message
        assert "strict_geometry" in message, message
    else:
        raise AssertionError("an unsatisfiable separation was accepted")

    # Nothing partially valid was left behind: the write is the last step of generate().
    assert list(out_dir.glob("*.json")) == [], list(out_dir.glob("*.json"))


def test_p9b_impossible_distance_floor_raises_instead_of_collapsing(tmp_path: Path) -> None:
    """A floor above the easy-zone ceiling raises; it is NOT quietly cut to 10%.

    Legacy behaviour for `min_target_distance_km >= easy_max` is
    `easy_min = easy_max * 0.1` -- a ~20x relaxation of the requested floor, applied in
    silence. Under a strict request that would put targets inside the sensing bubble the
    200 km floor exists to keep them out of.
    """
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))
    try:
        gen.generate(episode=0, config=_b1_config(seed=7, min_target_distance_km=1e6))
    except TargetPlacementError as exc:
        assert "min_target_distance_km" in str(exc), str(exc)
    else:
        raise AssertionError("an unsatisfiable distance floor was silently relaxed")


def test_p9c_legacy_path_still_degrades_quietly(tmp_path: Path) -> None:
    """Without `strict_geometry` the OLD behaviour is untouched: no raise, floor cut.

    The strict switch must add a failure mode, not change the default one -- every
    pre-B1 caller keeps the placement it had.
    """
    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))
    path = gen.generate(
        episode=0,
        config=_b1_config(seed=7, min_target_distance_km=1e6, strict_geometry=False),
    )
    base_lat, base_lon, coords = _blue_base_and_targets(json.loads(
        path.read_text(encoding="utf-8")
    ))
    assert len(coords) == _B1_N_KNOWN
    # The floor collapsed, so targets sit far inside the 1e6 km that was "requested".
    assert all(
        _haversine_km(base_lat, base_lon, lat, lon) < 1e6 for lat, lon in coords
    ), coords


# =============================================================================
# P10 -- reproducibility by GEOMETRY (never by id: uuids are not seed-derived).
# =============================================================================

def test_p10_same_seed_reproduces_the_geometric_fingerprint(tmp_path: Path) -> None:
    """Two independent generators, same seed and config -> identical coordinates.

    Compared by `(latitude, longitude)`, never by target id: `ScenarioGenerator` mints a
    fresh uuid for every red airbase it ADDS on each `generate()` even at a fixed seed
    (CLAUDE.md Sec 8), so an id-keyed comparison would fail for reasons that have nothing
    to do with placement.
    """
    first = _generate_b1(tmp_path / "a", 11)
    second = _generate_b1(tmp_path / "b", 11)
    other = _generate_b1(tmp_path / "c", 12)

    _, _, coords_a = _blue_base_and_targets(first)
    _, _, coords_b = _blue_base_and_targets(second)
    _, _, coords_c = _blue_base_and_targets(other)

    assert coords_a == coords_b, (coords_a, coords_b)
    # Guard against a vacuous pass: a different seed must actually move the targets.
    assert coords_a != coords_c, coords_a


# =============================================================================
# P11 -- the new fields cost nothing when they are not requested.
# =============================================================================

def test_p11_separation_defaults_are_behaviour_preserving(tmp_path: Path) -> None:
    """Defaults are (0.0, False), and an explicit 0.0 separation reproduces P6 exactly.

    The load-bearing half is the second assertion. The separation check is inside the
    placement accept-predicate, so a version that evaluated it unconditionally could
    still pass every P8 assertion while consuming a different number of rng draws and
    silently moving every target in every pre-B1 caller. Reproducing P6's pinned
    coordinates proves the rng stream is untouched.
    """
    fresh = VariationConfig()
    assert fresh.min_target_separation_km == 0.0, fresh.min_target_separation_km
    assert fresh.strict_geometry is False, fresh.strict_geometry

    gen = ScenarioGenerator(base_scenario_path=str(BASE_SCENARIO), output_dir=str(tmp_path))
    cfg = VariationConfig(**{**_COMMON_CFG_KWARGS, "min_target_separation_km": 0.0})
    path = gen.generate(episode=0, config=cfg)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    red = [ab for ab in data["currentScenario"]["airbases"] if ab.get("sideColor") == "red"]
    coords = [(ab["latitude"], ab["longitude"]) for ab in red]

    assert coords == _P6_EXPECTED_RED_AIRBASE_COORDS, (coords, _P6_EXPECTED_RED_AIRBASE_COORDS)


if __name__ == "__main__":
    failures = 0
    tests = [
        ("p1_blue_aircraft_sit_at_the_blue_airbase",
         test_p1_blue_aircraft_sit_at_the_blue_airbase, False),
        ("p2_launch_point_behaviour", _p2_launch_point_behaviour, False),
        ("p3_adjust_aircraft_count_fallback_anchors_to_base",
         test_p3_adjust_aircraft_count_fallback_anchors_to_base, True),
        ("p4_default_ensure_discovery_chain_is_behaviour_preserving",
         test_p4_default_ensure_discovery_chain_is_behaviour_preserving, True),
        ("p5_ensure_discovery_chain_false_is_a_true_skip",
         test_p5_ensure_discovery_chain_false_is_a_true_skip, True),
        ("p6_red_airbase_coordinates_unchanged_by_launch_point_fix",
         test_p6_red_airbase_coordinates_unchanged_by_launch_point_fix, True),
        ("p7_construction_cell_emits_a_known_only_world",
         test_p7_construction_cell_emits_a_known_only_world, True),
        ("p8_construction_geometry_holds_over_a_seed_sweep",
         test_p8_construction_geometry_holds_over_a_seed_sweep, True),
        ("p9_impossible_separation_raises_and_writes_nothing",
         test_p9_impossible_separation_raises_and_writes_nothing, True),
        ("p9b_impossible_distance_floor_raises_instead_of_collapsing",
         test_p9b_impossible_distance_floor_raises_instead_of_collapsing, True),
        ("p9c_legacy_path_still_degrades_quietly",
         test_p9c_legacy_path_still_degrades_quietly, True),
        ("p10_same_seed_reproduces_the_geometric_fingerprint",
         test_p10_same_seed_reproduces_the_geometric_fingerprint, True),
        ("p11_separation_defaults_are_behaviour_preserving",
         test_p11_separation_defaults_are_behaviour_preserving, True),
    ]
    for name, fn, needs_tmp in tests:
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))  # type: ignore[arg-type]
            else:
                fn()  # type: ignore[call-arg]
            print(f"OK   {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"SCENARIO_CONSTRUCTION_PRECONDITIONS TESTS: {failures} failed")
        sys.exit(1)
    print(f"SCENARIO_CONSTRUCTION_PRECONDITIONS TESTS: all {len(tests)} passed")
