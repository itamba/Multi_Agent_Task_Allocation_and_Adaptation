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
    VariationConfig,
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
