"""Proof obligations for the actor's ``mission_fuel_slack_norm`` input (agent column 1).

The feature is ``(current_fuel - estimated_remaining_mission_and_return_fuel) / max_fuel``
on the EGO agent row, computed by ``graph_builder.estimate_mission_fuel_slack`` from
ego-private inputs only (policy and CTDE contract §1). Each section below backs one proof
obligation of the task that introduced it:

  PO1  hand-calculated route / fuel cases (one target, several, levels, nonzero start,
       return leg, empty mission, positive / zero / negative slack, no clipping);
  PO2  private completion filtering, unassigned pop-ups excluded, no source mutation;
  PO3  no-communication: peer fuel / position / plan / completions, hidden world targets
       and unsensed destruction cannot move the value or the actor input;
  PO4  task permutation, exact ties, variable cardinalities, finite signed values;
  PO5  identical private inputs -> every legacy output unchanged, only column 1 differs;
  PO6  the column reaches assigned-task action scores and survives buffer / re-score /
       diagnostic paths;
  PO7  the actor width does not move the critic's central widths;
  PO8  checkpoint payload and failure routing (an integrity abort, never attrition);
  PO9  the audit is observational (no RNG draw, no mutation).

Engineering evidence only; nothing here measures learning.

Run: python -m pytest tests/test_graph_mission_fuel_slack.py -v
"""

from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_graph_fuel_damage as FD  # noqa: E402  (shared duck-typed BLADE stubs)

from match_aou.models.location import Location  # noqa: E402
from match_aou.rl.action.graph_action import (  # noqa: E402
    ActionHead,
    MetaAction,
    build_action_mask,
    evaluate_action,
)
from match_aou.rl.agent.graph_encoder import GraphEncoder  # noqa: E402
from match_aou.rl.observation import graph_builder as GB  # noqa: E402
from match_aou.rl.observation.central_graph_builder import (  # noqa: E402
    CENTRAL_AGENT_FEATURE_DIM,
)
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    ACTOR_OBSERVATION_ID,
    AGENT_FEATURE_DIM,
    TASK_FEATURE_DIM,
    EdgeType,
    EgoMissionInputs,
    GraphObservation,
    GraphObservationConfig,
    MissionSlackIntegrityError,
    build_graph_observation,
    estimate_mission_fuel_slack,
)
from match_aou.rl.training import graph_tick_loop as TL  # noqa: E402
from match_aou.rl.training import graph_train  # noqa: E402
from match_aou.rl.training.graph_fuel_damage import fuel_for_distance_km  # noqa: E402
from match_aou.rl.training.graph_ppo import build_central_critic  # noqa: E402
from match_aou.rl.training.graph_train import (  # noqa: E402
    ActorObservationIntegrityError,
    EpisodeAttemptError,
    MeasurementIntegrityError,
    TrainConfig,
)

_R_KM = 6371.0088  # the haversine package's mean earth radius (Location.distance_to)


def _hav(a, b) -> float:
    """An INDEPENDENT haversine (degrees in, km out) for hand calculations."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * _R_KM * math.asin(math.sqrt(h))


def _fuel(km: float, speed: float, rate: float) -> float:
    """Hand transcription of Game.get_fuel_needed_to_return_to_base, factor 1.0."""
    return km * 1000.0 / 1852.0 / speed * rate


def _task(tid: str, lat: float, lon: float):
    return FD._attack_task(tid, Location(lat, lon))


def _est(tasks, assigns, *, pos=(1.0, 0.0), home=(0.0, 0.0), fuel=8000.0,
         cap=10000.0, speed=500.0, rate=3600.0, confirmed=()):
    return estimate_mission_fuel_slack(
        position=Location(*pos), home_base=Location(*home), current_fuel=fuel,
        max_fuel=cap, speed_knots=speed, fuel_rate=rate, tasks=tasks,
        own_assignments=assigns, confirmed_target_ids=frozenset(confirmed),
    )


def _close(a: float, b: float, rel: float = 1e-5) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# =============================================================================
# PO1 -- hand-calculated route and fuel
# =============================================================================

def test_po1_one_target_nonzero_start_and_return_leg() -> None:
    tasks = [_task("A", 1.0, 3.0)]
    est = _est(tasks, [(0, 0, 0)], pos=(1.0, 1.0))
    km = _hav((1, 1), (1, 3)) + _hav((1, 3), (0, 0))
    assert _close(est.route_distance_km, km)
    assert _close(est.return_leg_km, _hav((1, 3), (0, 0)))
    assert _close(est.required_fuel, _fuel(km, 500.0, 3600.0))
    assert _close(est.slack_norm, (8000.0 - _fuel(km, 500.0, 3600.0)) / 10000.0)
    assert [r["target_id"] for r in est.route] == ["A"]


def test_po1_several_targets_same_level_use_nearest_neighbour_chain() -> None:
    tasks = [_task("FAR", 1.0, 4.0), _task("NEAR", 1.0, 2.0)]
    est = _est(tasks, [(0, 0, 0), (1, 0, 0)], pos=(1.0, 1.0))
    assert [r["target_id"] for r in est.route] == ["NEAR", "FAR"]
    km = _hav((1, 1), (1, 2)) + _hav((1, 2), (1, 4)) + _hav((1, 4), (0, 0))
    assert _close(est.route_distance_km, km)


def test_po1_levels_are_visited_in_ascending_order_before_proximity() -> None:
    # Level 0 is FAR, level 1 is NEAR: levels win over nearest neighbour, and the
    # endpoint of level 0 seeds level 1 (chained).
    tasks = [_task("NEAR_L1", 1.0, 1.5), _task("FAR_L0", 1.0, 3.0)]
    est = _est(tasks, [(0, 0, 1), (1, 0, 0)], pos=(1.0, 1.0))
    assert [r["target_id"] for r in est.route] == ["FAR_L0", "NEAR_L1"]
    km = _hav((1, 1), (1, 3)) + _hav((1, 3), (1, 1.5)) + _hav((1, 1.5), (0, 0))
    assert _close(est.route_distance_km, km)
    # negative (front-inserted OE) levels come first
    est2 = _est(tasks, [(0, 0, -1), (1, 0, 0)], pos=(1.0, 1.0))
    assert [r["target_id"] for r in est2.route] == ["NEAR_L1", "FAR_L0"]


def test_po1_empty_mission_is_the_direct_return_leg_not_zero() -> None:
    est = _est([_task("A", 1.0, 3.0)], [], pos=(1.0, 1.0))
    assert est.route == ()
    assert _close(est.route_distance_km, _hav((1, 1), (0, 0)))
    assert est.route_distance_km > 100.0
    assert _close(est.required_fuel, _fuel(_hav((1, 1), (0, 0)), 500.0, 3600.0))


def test_po1_positive_zero_and_negative_slack_are_signed_and_unclipped() -> None:
    tasks = [_task("A", 1.0, 3.0)]
    ref = _est(tasks, [(0, 0, 0)])
    zero = _est(tasks, [(0, 0, 0)], fuel=ref.required_fuel)
    assert zero.slack_fuel == 0.0 and zero.slack_norm == 0.0
    pos = _est(tasks, [(0, 0, 0)], fuel=ref.required_fuel + 2500.0)
    assert _close(pos.slack_norm, 0.25)
    # a deficit larger than max_fuel stays below -1: no clipping, no division by fuel
    neg = _est(tasks, [(0, 0, 0)], fuel=0.0, cap=ref.required_fuel / 3.0)
    assert _close(neg.slack_norm, -3.0)
    assert neg.slack_norm < -1.0


def test_po1_fuel_conversion_is_the_fuel_damage_layer_convention_at_factor_one() -> None:
    est = _est([_task("A", 2.0, 2.0)], [(0, 0, 0)], speed=1303.0, rate=6700.0)
    assert est.required_fuel == fuel_for_distance_km(
        est.route_distance_km, speed_knots=1303.0, fuel_rate=6700.0)
    # the builder's LOCAL transcription equals the training layer's copy of the same
    # engine arithmetic, bit for bit (the builder may not import the FD layer)
    rng = random.Random(1)
    for _ in range(200):
        km, v, r = rng.uniform(0, 3000), rng.uniform(100, 2000), rng.uniform(100, 9000)
        assert GB._physical_fuel_for_distance_km(km, speed_knots=v, fuel_rate=r) ==             fuel_for_distance_km(km, speed_knots=v, fuel_rate=r)
    rec = est.as_record()
    assert rec["fuel_factor"] == 1.0
    assert GB.actor_observation_definition()["mission_fuel_slack_norm"][
        "fd_rtb_margin_applied"] is False


def test_po1_invalid_inputs_fail_loud_and_never_fabricate() -> None:
    tasks = [_task("A", 1.0, 3.0)]
    bad = [
        dict(home=None), dict(fuel=float("nan")), dict(fuel=-1.0), dict(cap=0.0),
        dict(cap=float("inf")), dict(speed=0.0), dict(rate=-5.0), dict(pos=(None, 0.0)),
    ]
    for kw in bad:
        home = kw.pop("home", (0.0, 0.0))
        try:
            estimate_mission_fuel_slack(
                position=Location(*kw.get("pos", (1.0, 0.0))),
                home_base=None if home is None else Location(*home),
                current_fuel=kw.get("fuel", 8000.0), max_fuel=kw.get("cap", 1e4),
                speed_knots=kw.get("speed", 500.0), fuel_rate=kw.get("rate", 3600.0),
                tasks=tasks, own_assignments=[(0, 0, 0)], confirmed_target_ids=(),
            )
        except MissionSlackIntegrityError:
            continue
        raise AssertionError("invalid input accepted: %r / home=%r" % (kw, home))
    for assigns in ([(5, 0, 0)], [(0, 3, 0)], [(0, 0)], [("x", 0, 0)], [(True, 0, 0)]):
        try:
            _est(tasks, assigns)
        except MissionSlackIntegrityError:
            continue
        raise AssertionError("unresolvable assignment accepted: %r" % (assigns,))
    unlocated = [FD.Task(steps=[FD.Step(None, "U", [], 1.0, 1, FD.StepKind.ATTACK)],
                         utility=50)]
    try:
        _est(unlocated, [(0, 0, 0)])
    except MissionSlackIntegrityError:
        pass
    else:
        raise AssertionError("an unlocated assigned step was silently dropped")


# =============================================================================
# PO2 -- private completion filtering
# =============================================================================

def test_po2_confirmed_excluded_popups_excluded_unconfirmed_kept_no_mutation() -> None:
    tasks = [_task("DONE", 1.0, 2.0), _task("OPEN", 1.0, 3.0), _task("POPUP", 1.0, 1.2)]
    assigns = [(0, 0, 0), (1, 0, 0)]              # POPUP is known but UNASSIGNED
    confirmed = {"DONE"}
    snap = (copy.deepcopy([[(s.target_id, s.location.latitude, s.location.longitude)
                             for s in t.steps] for t in tasks]),
            list(assigns), set(confirmed))
    est = _est(tasks, assigns, pos=(1.0, 1.0), confirmed=confirmed)
    assert [r["target_id"] for r in est.route] == ["OPEN"]
    assert [r["target_id"] for r in est.excluded_confirmed] == ["DONE"]
    assert "POPUP" not in {r["target_id"] for r in est.remaining}
    km = _hav((1, 1), (1, 3)) + _hav((1, 3), (0, 0))
    assert _close(est.route_distance_km, km)
    # sources untouched
    assert [[(s.target_id, s.location.latitude, s.location.longitude) for s in t.steps]
            for t in tasks] == snap[0]
    assert assigns == snap[1] and confirmed == snap[2]


def test_po2_the_tick_loop_extracts_only_this_egos_confirmations_and_home() -> None:
    ctx = FD._FuelDamageCtx()
    ex = FD.GraphPlanExecutor(tasks=ctx.beliefs["ego0"].tasks, solution=ctx.a_init,
                              agents=ctx.agents)
    ex.done |= {("ego0", "tgt0"), ("ego1", "tgt1"), ("ego2", "tgt0")}
    inputs = TL._ego_mission_inputs(ex, "ego0")
    assert inputs.confirmed_target_ids == frozenset({"tgt0"})
    assert (inputs.home_base.latitude, inputs.home_base.longitude) == (
        FD._BASE.latitude, FD._BASE.longitude)
    # a peer's confirmation of the ego's own open target never becomes the ego's
    assert "tgt1" not in TL._ego_mission_inputs(ex, "ego0").confirmed_target_ids
    # no executor surface -> an integrity error, never "no completions"
    try:
        TL._ego_mission_inputs(object(), "ego0")
    except MissionSlackIntegrityError:
        pass
    else:
        raise AssertionError("missing executor surface defaulted silently")


# =============================================================================
# PO3 -- no communication
# =============================================================================

def _ctx_obs(ctx, ego="ego0", *, mission=None, solution=None):
    belief = ctx.beliefs[ego]
    sol = belief.solution if solution is None else solution
    return build_graph_observation(
        scenario=ctx.scenario, agent_id=ego, current_plan=sol.get(ego), current_time=7,
        tasks=belief.tasks, solution=sol, precedence_relations=[],
        config=GraphObservationConfig(detection_range_km=50.0),
        mission=mission if mission is not None else EgoMissionInputs(
            home_base=Location(FD._BASE.latitude, FD._BASE.longitude),
            confirmed_target_ids=frozenset()),
    )


def test_po3_peer_state_hidden_targets_and_unsensed_destruction_change_nothing() -> None:
    ctx = FD._FuelDamageCtx()
    ego = "ego0"
    ac = ctx.scenario.get_aircraft(ego)
    ac.latitude, ac.longitude = 33.3, 35.9    # ego away from the shared base
    base = _ctx_obs(ctx)
    base_row = base.agent_features[0].copy()
    base_audit = base.mission_slack.as_record()

    # peer fuel and position
    for peer in ("ego1", "ego2"):
        p = ctx.scenario.get_aircraft(peer)
        p.current_fuel, p.latitude, p.longitude = 17.0, 31.0, 34.0
    # peer plan (still assigned, different target) and peer confirmations
    sol = {k: list(v) for k, v in ctx.beliefs[ego].solution.items()}
    sol["ego1"] = [(2, 0, 0)]
    sol["ego2"] = [(1, 0, 5)]
    ex = FD.GraphPlanExecutor(tasks=ctx.beliefs[ego].tasks, solution=sol,
                              agents=ctx.agents)
    ex.done |= {("ego1", "tgt0"), ("ego2", "tgt0"), ("ego1", "tgt2")}
    mission = TL._ego_mission_inputs(ex, ego)
    # a hidden world target the ego has no task for, and the ego's own target destroyed
    # out of its sensor range (removed from the world, never confirmed by the ego)
    ctx.scenario.airbases.append(FD._StubAirbase(
        "hidden-x", 32.0, 36.5, side_id=FD._RED_SIDE, side_color="red"))
    ctx.scenario.airbases = [b for b in ctx.scenario.airbases if b.id != "tgt0"]

    after = _ctx_obs(ctx, mission=mission, solution=sol)
    assert np.array_equal(after.agent_features[0], base_row)
    assert after.mission_slack.as_record() == base_audit
    for i in range(1, after.agent_features.shape[0]):
        assert not after.agent_features[i].any(), "a peer row is not featureless"


def test_po3_only_the_ego_row_carries_the_feature_and_peers_are_zero() -> None:
    ctx = FD._FuelDamageCtx()
    obs = _ctx_obs(ctx)
    assert obs.agent_features.shape == (len(obs.agent_ids), AGENT_FEATURE_DIM)
    assert obs.agent_features[0, 1] == np.float32(obs.mission_slack.slack_norm)
    assert not obs.agent_features[1:].any()


# =============================================================================
# PO4 -- permutation, exact ties, cardinality, finite signed values
# =============================================================================

def test_po4_task_permutation_with_remapped_assignments_changes_nothing() -> None:
    rng = random.Random(7)
    for trial in range(25):
        k = rng.randint(1, 8)
        tasks = [_task("T%d" % j, rng.uniform(-3, 3), rng.uniform(-3, 3)) for j in range(k)]
        assigns = [(j, 0, rng.randint(-1, 2)) for j in range(k) if rng.random() < 0.7]
        confirmed = {"T%d" % j for j in range(k) if rng.random() < 0.2}
        pos = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        base = _est(tasks, assigns, pos=pos, confirmed=confirmed)
        perm = list(range(k))
        rng.shuffle(perm)                            # new index i <- old index perm[i]
        inv = {old: new for new, old in enumerate(perm)}
        p_tasks = [tasks[old] for old in perm]
        p_assigns = [(inv[t], s, lv) for (t, s, lv) in assigns]
        rng.shuffle(p_assigns)
        got = _est(p_tasks, p_assigns, pos=pos, confirmed=confirmed)
        assert got.slack_norm == base.slack_norm, trial
        assert [r["target_id"] for r in got.route] == [r["target_id"] for r in base.route]
        assert math.isfinite(got.slack_norm)


def test_po4_exact_ties_break_on_geometry_not_task_index() -> None:
    # Two targets EXACTLY equidistant from the ego (mirror images about the equator).
    north, south = _task("N", 1.0, 2.0), _task("S", -1.0, 2.0)
    a = _est([north, south], [(0, 0, 0), (1, 0, 0)], pos=(0.0, 1.0))
    b = _est([south, north], [(0, 0, 0), (1, 0, 0)], pos=(0.0, 1.0))
    assert [r["target_id"] for r in a.route] == ["S", "N"]      # lower latitude first
    assert [r["target_id"] for r in b.route] == ["S", "N"]
    assert a.slack_norm == b.slack_norm
    # identical-location targets have identical movement cost whatever their order
    twin1, twin2 = _task("X1", 1.0, 2.0), _task("X2", 1.0, 2.0)
    c = _est([twin1, twin2], [(0, 0, 0), (1, 0, 0)])
    d = _est([twin2, twin1], [(1, 0, 0), (0, 0, 0)])
    assert c.slack_norm == d.slack_norm


def test_po4_builder_permutation_moves_nothing_in_the_ego_row() -> None:
    ctx = FD._FuelDamageCtx()
    ego = "ego0"
    ac = ctx.scenario.get_aircraft(ego)
    ac.latitude, ac.longitude = 33.1, 35.6
    belief = ctx.beliefs[ego]
    belief.solution[ego] = [(0, 0, 0), (2, 0, 1)]
    base = _ctx_obs(ctx)
    perm = [2, 0, 1]
    inv = {old: new for new, old in enumerate(perm)}
    belief.tasks = [belief.tasks[o] for o in perm]
    belief.solution = {a: [(inv[t], s, lv) for (t, s, lv) in v]
                       for a, v in belief.solution.items()}
    got = _ctx_obs(ctx)
    assert np.array_equal(got.agent_features, base.agent_features)
    assert np.array_equal(got.task_features, base.task_features[perm])


# =============================================================================
# PO5 -- identical private inputs: only column 1 is new
# =============================================================================

def test_po5_mission_inputs_move_only_the_new_column() -> None:
    ctx = FD._FuelDamageCtx()
    ac = ctx.scenario.get_aircraft("ego0")
    ac.latitude, ac.longitude = 33.2, 35.7
    a = _ctx_obs(ctx)
    b = _ctx_obs(ctx, mission=EgoMissionInputs(
        home_base=Location(30.0, 34.0), confirmed_target_ids=frozenset({"tgt0"})))
    assert a.agent_features[0, 1] != b.agent_features[0, 1]
    assert np.array_equal(a.agent_features[:, 0], b.agent_features[:, 0])
    for name in ("task_features", "edge_index", "edge_type"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for name in ("ego_index", "task_target_ids", "agent_ids", "agent_id",
                 "current_time", "time_norm"):
        assert getattr(a, name) == getattr(b, name), name
    assert np.array_equal(build_action_mask(a), build_action_mask(b))
    assert a.task_features.shape[1] == TASK_FEATURE_DIM == 6


def test_po5_missing_mission_inputs_are_refused() -> None:
    ctx = FD._FuelDamageCtx()
    try:
        _ctx_obs(ctx, mission=object())
    except MissionSlackIntegrityError:
        pass
    else:
        raise AssertionError("a missing EgoMissionInputs was accepted")


# =============================================================================
# PO6 -- the column reaches assigned-task scores and every downstream path
# =============================================================================

def _synthetic_obs(slack: float) -> GraphObservation:
    """k=3: task 0 assigned to the ego, task 1 to a peer, task 2 isolated (no edge)."""
    tf = np.array([[0.8, 0.2, 1, 1, 1, 1], [0.6, 0.4, 1, 1, 1, 1],
                   [0.5, 0.3, 1, 1, 1, 1]], dtype=np.float32)
    af = np.array([[0.7, slack], [0.0, 0.0]], dtype=np.float32)
    return GraphObservation(
        task_features=tf, agent_features=af, ego_index=3,
        edge_index=np.array([[3, 4], [0, 1]], dtype=np.int64),
        edge_type=np.full((2,), int(EdgeType.ASSIGNMENT), dtype=np.int64),
        task_target_ids=["t0", "t1", "t2"], agent_ids=["ego", "peer"], agent_id="ego",
        current_time=0, time_norm=0.1,
    )


def test_po6_slack_reaches_the_assigned_task_scores_only_through_the_graph() -> None:
    torch.manual_seed(3)
    enc, head = GraphEncoder(), ActionHead(embed_dim=64)
    enc.eval()
    with torch.no_grad():
        lo = head(enc(_synthetic_obs(-0.4)))
        hi = head(enc(_synthetic_obs(+0.4)))
    assert (lo[0] - hi[0]).abs().max().item() > 1e-6, "assigned task score unmoved"
    # the isolated task has only its self-loop: the ego row cannot reach it
    assert torch.equal(lo[2], hi[2])
    # and the dependence is differentiable through agent_proj's column 1
    enc.zero_grad(set_to_none=True)
    head(enc(_synthetic_obs(0.1)))[0].sum().backward()
    assert enc.agent_proj.weight.grad[:, 1].abs().sum().item() > 0.0


def test_po6_stored_observation_rescoring_and_diagnostics_carry_the_column() -> None:
    torch.manual_seed(5)
    policy = TL.build_policy()
    obs_lo, obs_hi = _synthetic_obs(-0.5), _synthetic_obs(0.5)
    with torch.no_grad():
        lp_lo, _ = evaluate_action(policy.head(policy.encoder(obs_lo)),
                                   build_action_mask(obs_lo), int(MetaAction.PLAN_COMPLIANCE),
                                   None)
        lp_hi, _ = evaluate_action(policy.head(policy.encoder(obs_hi)),
                                   build_action_mask(obs_hi), int(MetaAction.PLAN_COMPLIANCE),
                                   None)
    assert lp_lo.item() != lp_hi.item(), "the PPO re-score ignores column 1"
    logits = policy.head(policy.encoder(obs_hi)).detach()
    rec = TL._decision_record(logits=logits, mask=build_action_mask(obs_hi), gobs=obs_hi,
                              solution={}, ego_key="ego",
                              meta_action=int(MetaAction.PLAN_COMPLIANCE), node_v=None,
                              wake_kind=TL.WAKE_KIND_IMMEDIATE_FD)
    assert rec["ego_mission_fuel_slack_norm"] == float(np.float32(0.5))
    assert rec["actor_observation_id"] == ACTOR_OBSERVATION_ID
    assert rec["mission_slack_audit"] is None   # hand-built obs carries no audit


def test_po6_wake_decision_records_the_value_the_encoder_saw() -> None:
    ctx = FD._FuelDamageCtx()
    ego = "ego0"
    ex = FD.GraphPlanExecutor(tasks=ctx.beliefs[ego].tasks, solution=ctx.a_init,
                              agents=ctx.agents)
    ac = ctx.scenario.get_aircraft(ego)
    ac.latitude, ac.longitude = 33.0, 35.5
    torch.manual_seed(0)
    tr = TL._wake_decision(TL.build_policy(), ego, ctx.scenario, ctx.beliefs[ego], ex,
                           GraphObservationConfig(detection_range_km=50.0), 3,
                           deterministic=True)
    audit = tr.decision["mission_slack_audit"]
    assert tr.decision["ego_mission_fuel_slack_norm"] == float(tr.gobs.agent_features[0, 1])
    assert np.float32(audit["mission_fuel_slack_norm"]) == tr.gobs.agent_features[0, 1]
    assert audit["route"] and audit["confirmed_target_ids"] == []
    assert _close(audit["required_fuel"],
                  _fuel(audit["route_distance_km"], audit["speed_knots"],
                        audit["fuel_rate"]))


# =============================================================================
# PO7 -- the critic keeps its own widths
# =============================================================================

def test_po7_actor_width_does_not_move_the_central_critic() -> None:
    assert AGENT_FEATURE_DIM == 2 and CENTRAL_AGENT_FEATURE_DIM == 1
    assert GraphEncoder().agent_feat_dim == AGENT_FEATURE_DIM
    critic = build_central_critic()
    assert critic.encoder.agent_feat_dim == CENTRAL_AGENT_FEATURE_DIM
    assert critic.encoder.agent_proj.in_features == CENTRAL_AGENT_FEATURE_DIM
    assert TL.build_policy().encoder.agent_proj.in_features == AGENT_FEATURE_DIM


# =============================================================================
# PO8 -- checkpoint payload and failure routing
# =============================================================================

def test_po8_checkpoint_names_the_actor_observation_and_old_widths_do_not_load(
        tmp_path=None) -> None:
    import tempfile
    from match_aou.rl.training.graph_ppo import PPOUpdater
    out = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    policy = TL.build_policy()
    path = graph_train.save_checkpoint(policy, PPOUpdater(policy), 3, out)
    payload = torch.load(path, weights_only=False)
    assert payload["actor_observation_id"] == ACTOR_OBSERVATION_ID
    assert payload["actor_observation"]["agent_feature_dim"] == 2
    assert "training_mode" not in payload and "critic_encoder" not in payload
    fresh = TL.build_policy()
    fresh.encoder.load_state_dict(payload["encoder"])
    legacy = GraphEncoder(agent_feat_dim=1)
    try:
        fresh.encoder.load_state_dict(legacy.state_dict())
    except RuntimeError:
        pass
    else:
        raise AssertionError("a one-column-agent encoder loaded into the new width")


def _run_with_raising_run_episode(exc: BaseException):
    ctx = FD._FuelDamageCtx()
    n = len(ctx.known_target_ids)
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      eval_every=0, checkpoint_every=0,
                      num_agents=n, n_known=n, n_hidden=0)

    class _Path:
        @staticmethod
        def read_text(encoding=None):
            return "{}"

    class _Gen:
        @staticmethod
        def generate(episode, config):
            return _Path()

    def _raise(*_a, **_k):
        raise exc

    saved = {"setup_episode": graph_train.setup_episode,
             "run_episode": graph_train.run_episode}
    graph_train.setup_episode = lambda *a, **k: ctx
    graph_train.run_episode = _raise
    try:
        return graph_train._run_one_episode(None, _Gen(), cfg, seed=1, episode_tag=0,
                                            deterministic=False)
    finally:
        for name, original in saved.items():
            setattr(graph_train, name, original)


def test_po8_a_feature_integrity_failure_aborts_and_is_never_run_attrition() -> None:
    try:
        _run_with_raising_run_episode(MissionSlackIntegrityError("no home base"))
    except ActorObservationIntegrityError as err:
        assert isinstance(err, MeasurementIntegrityError)
        assert not isinstance(err, EpisodeAttemptError)
        assert isinstance(err.__cause__, MissionSlackIntegrityError)
    else:
        raise AssertionError("the integrity failure was swallowed")
    # control: an ordinary run failure is still accounted as the `run` stage
    try:
        _run_with_raising_run_episode(ValueError("ordinary"))
    except EpisodeAttemptError as err:
        assert err.stage == "run"
    else:
        raise AssertionError("control did not raise")


def test_po8_run_config_names_the_actor_observation() -> None:
    src = Path(graph_train.__file__).read_text(encoding="utf-8")
    assert '"actor_observation_id": ACTOR_OBSERVATION_ID' in src
    definition = GB.actor_observation_definition()
    assert definition["actor_observation_id"] == ACTOR_OBSERVATION_ID
    assert definition["agent_feature_columns"] == ["fuel_norm", "mission_fuel_slack_norm"]
    assert graph_train._EPISODE_OUTCOME_VERSION == 5
    assert graph_train._WAKE_DIAGNOSTICS_VERSION == 3


# =============================================================================
# PO9 -- observational: no RNG draw, no mutation
# =============================================================================

def test_po9_building_the_feature_draws_no_randomness_and_mutates_nothing() -> None:
    ctx = FD._FuelDamageCtx()
    belief = ctx.beliefs["ego0"]
    sol_before = copy.deepcopy(belief.solution)
    tasks_before = list(belief.tasks)
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    t_state, r_state, n_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    obs = _ctx_obs(ctx)
    obs.mission_slack.as_record()
    assert torch.equal(torch.get_rng_state(), t_state)
    assert random.getstate() == r_state
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(np.random.get_state(), n_state))
    assert belief.solution == sol_before and belief.tasks == tasks_before


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("OK", fn.__name__)
    print("all %d passed" % len(fns))
