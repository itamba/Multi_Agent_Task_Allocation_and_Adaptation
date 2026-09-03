"""
Phase-B CTDE tests -- the centralized critic, its inputs, and its isolation.

Three proof obligations, in three sections:

  PO1  ACTOR-ONLY PRESERVATION. ``training_mode="actor_only"`` is the Phase-A reference
       path and must be structurally untouched: no critic, no central observation, the
       unchanged credit assignment, and the unchanged checkpoint payload. The
       load-bearing test POISONS every central-CTDE construction site so that touching
       one RAISES, then runs an actor_only training loop through it -- and flips the mode
       to prove the poison is live rather than vacuous.

  PO2  NO PRIVILEGED LEAKAGE / TRULY DECENTRALIZED EXECUTION. Actor and critic share no
       parameter; neither loss produces a gradient in the other's parameters; the actor's
       advantage is detached; a central state cannot even be mistaken for an actor
       observation; and CHANGING PRIVILEGED FEATURES CANNOT MOVE AN ACTOR LOGIT.

  PO3  CENTRAL VALUE / GAE ALIGNMENT. Exactly one central sample per wake, in real
       decision order, captured BEFORE the action; two same-tick wakes with no
       ``env.step`` between them, the later one seeing the earlier one's resync; physical
       death removing exactly one node kind; every live agent carrying its own real fuel;
       an unallocated-but-live target still present; ``assigned`` following the CURRENT
       executor plans; hand-computed GAE with ``V_next = 0`` at the end; fixed value
       targets; zero-wake episodes contributing nothing; and variable graph sizes staying
       finite.

Solver-free and BLADE-free: every fixture is a hand-built stub, so this file runs under
the base-env ``pytest`` AND standalone under ``nlp_env``.

Run: python -m pytest tests/test_graph_ctde.py -v
     python tests/test_graph_ctde.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.models import Agent, Capability, Location, Step, StepKind, Task  # noqa: E402
from match_aou.rl.action.graph_action import ActionHead, build_action_mask  # noqa: E402
from match_aou.rl.agent.graph_encoder import GraphEncoder  # noqa: E402
from match_aou.rl.observation.central_graph_builder import (  # noqa: E402
    CENTRAL_AGENT_FEATURE_DIM,
    CENTRAL_EDGE_ATTR_DIM,
    CENTRAL_TASK_FEATURE_DIM,
    CENTRAL_EDGE_TYPE,
    CentralGraphObservation,
    CentralStateRecorder,
    NO_EGO_INDEX,
    build_central_graph_observation,
    live_aircraft,
    plan_target_ids,
)
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    EdgeType,
    GraphObservation,
    GraphObservationConfig,
)
from match_aou.rl.training import graph_ppo, graph_train  # noqa: E402
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CONDITION_DAMAGED,
    POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
    POST_FD_WAKE_SINGLE_V1,
    FuelDamageController,
    FuelDamageMode,
    FuelDamageParameters,
    plan_fuel_damage,
    resolve_condition,
    resolve_severity,
)
from match_aou.rl.training.graph_ppo import (  # noqa: E402
    CTDEBuffer,
    CTDEConfig,
    CTDEEpisodeRecord,
    CTDEUpdater,
    CentralCritic,
    PPOConfig,
    PPOUpdater,
    build_central_critic,
    compute_ctde_advantages,
    compute_gae,
)
from match_aou.rl.training.graph_tick_loop import Policy, Transition  # noqa: E402
from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor  # noqa: E402

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore


_BLUE = "side-blue"
_RED = "side-red"

# 0.1 degrees of latitude is ~11.1 km, so these fixtures place targets at hand-checkable
# multiples of the 50 km detection radius without needing exact geodesy.
_DEG_KM = 111.0


# =============================================================================
# Stubs -- the exact BLADE surface the central builder reads, and nothing more
# =============================================================================

class _Aircraft:
    """The live-airframe fields the central builder and `create_agents_from_scenario` read."""

    def __init__(self, aid, lat, lon, *, fuel=12000.0, max_fuel=12000.0,
                 speed=500.0, rate=6700.0, home_base_id="base-blue", armed=True):
        self.id = aid
        self.name = "AC %s" % aid
        self.side_id = _BLUE
        self.side_color = "blue"
        self.class_name = "F-16 Fighting Falcon"
        self.latitude = lat
        self.longitude = lon
        self.altitude = 10000
        self.heading = 90.0
        self.speed = speed
        self.current_fuel = fuel
        self.max_fuel = max_fuel
        self.fuel_rate = rate
        self.range = 100
        # A weapon is what gives the derived MATCH-AOU Agent an "attack" capability, so
        # the `capable` edge feature is 1 for an armed aircraft and 0 for an empty rack.
        self.weapons = [_Weapon()] if armed else []
        self.home_base_id = home_base_id
        self.target_id = None
        self.route = []
        self.rtb = False

    def get_weapon_with_highest_engagement_range(self):
        return self.weapons[0] if self.weapons else None


class _Weapon:
    def __init__(self):
        self.id = "w1"
        self.class_name = "AIM-120"
        self.current_quantity = 4
        self.speed = 2000.0


class _BlueBase:
    """A friendly airbase; its `aircraft` list is where a LANDED ego lives."""

    def __init__(self, bid="base-blue", lat=32.0, lon=35.0, aircraft=None):
        self.id = bid
        self.name = "Blue Base"
        self.side_id = _BLUE
        self.side_color = "blue"
        self.latitude = lat
        self.longitude = lon
        self.altitude = 0
        self.aircraft = list(aircraft or [])


class Airbase:
    """A RED target.

    The class NAME matters: ``scenario_factory.make_attack_task`` derives per-type
    utility from ``type(unit).__name__``, so this must be spelled ``Airbase`` to get the
    real 80 rather than the unknown-type fallback.
    """

    def __init__(self, tid, lat, lon):
        self.id = tid
        self.name = "Red %s" % tid
        self.side_id = _RED
        self.side_color = "red"
        self.latitude = lat
        self.longitude = lon
        self.altitude = 0
        self.aircraft = []


class _Scenario:
    """A minimal live scenario: exactly the lookups the central builder performs."""

    def __init__(self, aircraft=None, airbases=None, facilities=None, ships=None):
        self.aircraft = list(aircraft or [])
        self.airbases = list(airbases or [])
        self.facilities = list(facilities or [])
        self.ships = list(ships or [])
        self.current_time = 0

    def get_aircraft(self, aircraft_id):
        return next((a for a in self.aircraft if str(a.id) == str(aircraft_id)), None)

    def get_airbase(self, airbase_id):
        return next((b for b in self.airbases if str(b.id) == str(airbase_id)), None)

    def get_ship(self, _ship_id):
        return None

    def get_target(self, target_id):
        for unit in self.airbases + self.facilities + self.ships:
            if str(unit.id) == str(target_id):
                return unit
        return None


def _task(target_id, lat, lon, *, utility=80.0, probability=1.0):
    """A real ``Task`` with the single ATTACK step the executor and builder read."""
    return Task(
        steps=[Step(
            location=Location(lat, lon),
            target_id=target_id,
            capabilities=[Capability(name="attack", properties={"Quantity": 2})],
            probability=probability,
            effort=2,
            step_kind=StepKind.ATTACK,
        )],
        utility=utility,
    )


def _agent(aid, lat=32.0, lon=35.0, budget=12000.0):
    return Agent(
        location=Location(lat, lon),
        capabilities=[Capability(name="attack", properties={"Quantity": 2})],
        budget=budget,
        move_cost_function=lambda s, d: 0.0,
        agent_id=aid,
        side_color="blue",
        return_location=Location(lat, lon),
    )


class _Executor:
    """The executor surface the central builder reads: plans / tasks / agent_by_id.

    The tick-loop surface (``sensed_target_ids`` / ``next_actions`` / ``resync`` /
    ``is_done``) is stubbed too, so the same object can drive a real ``run_episode``.
    """

    def __init__(self, agent_ids, tasks, plans):
        self.agent_by_id = {str(a): _agent(str(a)) for a in agent_ids}
        self.tasks = {str(a): list(tasks) for a in agent_ids}
        self.plans = {str(a): list(plans.get(str(a), [])) for a in agent_ids}
        self.arrival_threshold_km = 50.0
        self.dead = set()
        self.done = set()
        self.rtb_issued = {}

    # --- the tick-loop surface (inert: these fixtures drive decisions themselves) ---
    def sensed_target_ids(self, _observation, _ego_id):
        return {}

    def next_actions(self, _observation):
        return []

    def resync(self, new_solution, *, ego_id, tasks=None):
        self.plans[str(ego_id)] = [tuple(t) for t in (new_solution.get(str(ego_id)) or [])]
        if tasks is not None:
            self.tasks[str(ego_id)] = list(tasks)

    def is_done(self, _observation):
        return False


def _world(*, n_targets=3, agents=("ego0", "ego1"), plans=None, base_lat=32.0):
    """A small world: blue egos at the base, red targets strung out to the north."""
    acs = [_Aircraft(a, base_lat, 35.0) for a in agents]
    base = _BlueBase(lat=base_lat, lon=35.0)
    targets = [Airbase("t%d" % j, base_lat + 0.2 * (j + 1), 35.0) for j in range(n_targets)]
    scen = _Scenario(aircraft=acs, airbases=[base] + targets)
    tasks = [_task("t%d" % j, base_lat + 0.2 * (j + 1), 35.0) for j in range(n_targets)]
    ex = _Executor(agents, tasks, plans or {})
    return scen, ex, list(agents), tasks


def _central(scen, ex, agent_ids, *, t=0, config=None):
    return build_central_graph_observation(
        scen, agent_ids=agent_ids, executor=ex, current_time=t,
        config=config or GraphObservationConfig(),
    )


def _synthetic_central(k=3, a=2, *, seed=0):
    """A hand-built central state (no scenario), for shape / numeric tests."""
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for i in range(a):
        for j in range(k):
            src.append(k + i)
            dst.append(j)
    e = len(src)
    return CentralGraphObservation(
        task_features=rng.random((k, CENTRAL_TASK_FEATURE_DIM)).astype(np.float32),
        agent_features=rng.random((a, CENTRAL_AGENT_FEATURE_DIM)).astype(np.float32),
        ego_index=NO_EGO_INDEX,
        edge_index=np.array([src, dst], dtype=np.int64) if e else np.zeros((2, 0), np.int64),
        edge_type=np.full((e,), CENTRAL_EDGE_TYPE, dtype=np.int64),
        edge_attr=rng.random((e, CENTRAL_EDGE_ATTR_DIM)).astype(np.float32),
        task_target_ids=["t%d" % j for j in range(k)],
        agent_ids=["a%d" % i for i in range(a)],
        current_time=10,
        time_norm=0.01,
    )


def _actor_obs(k=3, a=2, *, ego_index=None, seed=0):
    """A hand-built ACTOR observation with the real 6 task columns."""
    rng = np.random.default_rng(seed)
    tf = rng.random((k, 6)).astype(np.float32)
    tf[:, 2:4] = 1.0   # capable + reachable
    tf[:, 5] = 1.0     # sensed
    return GraphObservation(
        task_features=tf,
        agent_features=rng.random((a, 1)).astype(np.float32),
        ego_index=k if ego_index is None else ego_index,
        edge_index=np.array([[k], [0]], dtype=np.int64),
        edge_type=np.array([int(EdgeType.ASSIGNMENT)], dtype=np.int64),
        task_target_ids=["t%d" % j for j in range(k)],
        agent_ids=["a%d" % i for i in range(a)],
        agent_id="a0",
        current_time=10,
        time_norm=0.01,
    )


def _transition(gobs, *, reward=None, meta=0, node=0):
    return Transition(
        gobs=gobs, ego_id="a0", tick=1, meta_action=meta, node_v=node,
        log_prob=-1.0, entropy=0.5, reward=reward,
    )


# =============================================================================
# PO1 -- ACTOR-ONLY PRESERVATION
# =============================================================================

class _Poison(RuntimeError):
    """Raised by any central-CTDE construction site an actor_only run touches."""


def _run_stub_training(cfg, monkey_restore):
    """Drive the REAL ``graph_train.train`` over stubbed episodes / generator / updater.

    Everything scenario-, BLADE- and solver-shaped is stubbed; what is exercised is the
    trainer's own branching -- which buffer, which updater, whether a critic and a
    central recorder are constructed at all.
    """
    saved = {name: getattr(graph_train, name) for name in monkey_restore}

    def fake_build_generator(_scen_dir):
        return object()

    def fake_run_one_episode(policy, gen, cfg_, *, seed, episode_tag, deterministic,
                            fuel_damage_mode=None, **kwargs):
        recorder = kwargs.get("central_recorder")
        gobs = _actor_obs(seed=seed % 7)
        trajectory = [_transition(gobs), _transition(gobs)]
        if recorder is not None:
            # A faithful stand-in for the tick loop: ONE central sample per decision.
            for _ in trajectory:
                recorder.samples.append(_synthetic_central(seed=seed % 7))
        trajectory[-1].reward = -0.25
        # The stub must report the cell the SCHEDULE resolved: `_ConditionTally.success`
        # refuses an episode whose plan disagrees with its own scheduling, and a stub
        # that hard-coded "clean" would be exercising that refusal instead of CTDE.
        params = cfg_.fuel_damage_parameters(fuel_damage_mode)
        condition = resolve_condition(episode_seed=seed, params=params)
        severity = resolve_severity(episode_seed=seed, params=params)
        return graph_train._EpisodeOutcome(
            trajectory=trajectory, reward=-0.25, ticks=10, ended="done", n_wakes=2,
            confirmed_kills=1, n_dead=0, seconds=0.01, targets_confirmed_unique=1,
            targets_total=6, known_target_names=("A",), hidden_target_names=("B",),
            known_confirmed_names=("A",), hidden_confirmed_names=(),
            fuel_damage_plan={"condition": condition, "severity": severity,
                              "ego_id": None},
            fuel_damage_outcome={"condition": condition, "severity": severity,
                                 "fired": False, "wake_occurred": False,
                                 "wake_meta_action": None},
            selected_ego_rtb_issued=None,
        )

    graph_train._git_provenance = lambda repo_root: {
        "repo_root": str(repo_root), "available": True, "commit": "0" * 40,
        "branch": "test", "dirty": False, "dirty_path_count": 0, "reason": None,
    }
    graph_train._run_one_episode = fake_run_one_episode
    graph_train._build_generator = fake_build_generator
    try:
        return graph_train.train(cfg)
    finally:
        for name, value in saved.items():
            setattr(graph_train, name, value)


def _poison_central(saved_names):
    """Replace every central-CTDE construction site with a raiser."""
    def boom(*_a, **_k):
        raise _Poison("a central-CTDE construction site was reached")

    for name in saved_names:
        setattr(graph_train, name, boom)


def test_actor_only_never_constructs_a_critic_or_a_central_state(tmp_path):
    """PO1: with every central-CTDE site POISONED, an actor_only run completes.

    The poison is the point: `build_central_critic`, `CentralStateRecorder`,
    `CTDEBuffer`, `CTDEUpdater` and `CTDEEpisodeRecord` all raise if touched. An
    actor_only run must never touch one -- and the companion test below flips the mode
    to prove the poison actually fires, so a green result here cannot be vacuous.
    """
    names = ["build_central_critic", "CentralStateRecorder", "CTDEBuffer",
             "CTDEUpdater", "CTDEEpisodeRecord"]
    restore = names + ["_run_one_episode", "_build_generator", "_git_provenance"]
    saved = {n: getattr(graph_train, n) for n in names}
    cfg = graph_train.TrainConfig(
        n_iterations=1, episodes_per_iteration=2, eval_every=0, eval_episodes=0,
        output_dir=str(tmp_path / "run_actor_only"), checkpoint_every=1,
    )
    assert cfg.training_mode == graph_train.TRAINING_MODE_ACTOR_ONLY
    assert not cfg.ctde_enabled
    try:
        _poison_central(names)
        summary = _run_stub_training(cfg, restore)
    finally:
        for n, v in saved.items():
            setattr(graph_train, n, v)
    assert summary is not None
    # And the run really trained: the actor-only updater ran its epochs.
    assert summary["updates_completed"] >= 1


def test_the_poison_is_live_a_ctde_run_hits_it(tmp_path):
    """PO1 control: the same poison DOES fire under `training_mode="ctde"`.

    Without this, the test above would pass just as well if the poison were misspelled.
    """
    names = ["build_central_critic", "CentralStateRecorder", "CTDEBuffer",
             "CTDEUpdater", "CTDEEpisodeRecord"]
    restore = names + ["_run_one_episode", "_build_generator", "_git_provenance"]
    saved = {n: getattr(graph_train, n) for n in names}
    cfg = graph_train.TrainConfig(
        n_iterations=1, episodes_per_iteration=2, eval_every=0, eval_episodes=0,
        output_dir=str(tmp_path / "run_ctde_poisoned"),
        training_mode=graph_train.TRAINING_MODE_CTDE,
    )
    raised = False
    try:
        _poison_central(names)
        _run_stub_training(cfg, restore)
    except _Poison:
        raised = True
    finally:
        for n, v in saved.items():
            setattr(graph_train, n, v)
    assert raised, "a ctde run did not reach any central-CTDE construction site"


def test_ctde_refuses_a_zero_value_coefficient():
    """F1: `ctde` + `value_coeff = 0` is REJECTED before any training work.

    Such a run would build central observations and take its advantages from the critic
    while never training it -- the baseline would stay a frozen random function. That is
    neither the `actor_only` reference algorithm nor the approved CTDE one, and it would
    be recorded and read as CTDE.
    """
    cfg = graph_train.TrainConfig(
        n_iterations=1,
        training_mode=graph_train.TRAINING_MODE_CTDE,
        ctde=CTDEConfig(value_coeff=0.0),
    )
    try:
        cfg.validate()
        raised = None
    except ValueError as exc:
        raised = str(exc)
    assert raised is not None, "ctde with value_coeff=0 was accepted"
    assert "value_coeff" in raised and "> 0" in raised
    # A negative coefficient is refused too, and the approved default is accepted.
    for bad in (-1e-9, -0.5):
        try:
            graph_train.TrainConfig(
                n_iterations=1, training_mode=graph_train.TRAINING_MODE_CTDE,
                ctde=CTDEConfig(value_coeff=bad),
            ).validate()
            assert False, "ctde accepted value_coeff=%r" % (bad,)
        except ValueError:
            pass
    graph_train.TrainConfig(
        n_iterations=1, training_mode=graph_train.TRAINING_MODE_CTDE,
    ).validate()


def test_the_approved_ctde_defaults_are_unchanged():
    """F1: the fix is a BOUND, not a retune -- the approved defaults still stand."""
    d = CTDEConfig()
    assert d.value_coeff == 0.5
    assert d.critic_lr == 3e-4
    assert d.gae_lambda == 0.95
    assert graph_train.TrainConfig(n_iterations=1).ctde == d


def test_actor_only_is_unaffected_by_unused_ctde_configuration():
    """F1: `actor_only` validation semantics are untouched.

    An `actor_only` run never reads the CTDE block, so a value it would refuse under
    `ctde` -- including 0 -- must not make an actor_only config invalid. That keeps
    `value_coeff` from acting as a mode selector: `ctde_enabled` reads `training_mode`
    and nothing else.
    """
    for coeff in (0.0, -3.0, 0.5):
        cfg = graph_train.TrainConfig(
            n_iterations=1,
            training_mode=graph_train.TRAINING_MODE_ACTOR_ONLY,
            ctde=CTDEConfig(value_coeff=coeff),
        )
        cfg.validate()
        assert not cfg.ctde_enabled
    # And an absurd CTDE block still does not make an actor_only run a CTDE run.
    weird = graph_train.TrainConfig(
        n_iterations=1, ctde=CTDEConfig(value_coeff=0.0, critic_lr=-1.0),
    )
    weird.validate()
    assert weird.training_mode == graph_train.TRAINING_MODE_ACTOR_ONLY
    assert not weird.ctde_enabled


def test_actor_only_checkpoint_payload_keys_are_exactly_the_phase_a_five(tmp_path):
    """PO1: `critic=None` saves the pre-CTDE payload -- no renames, no extra keys."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    updater = PPOUpdater(policy, PPOConfig())
    path = graph_train.save_checkpoint(policy, updater, 3, tmp_path)
    payload = torch.load(path, weights_only=False)
    assert set(payload) == {"iteration", "encoder", "head", "optimizer", "ppo_config"}
    assert payload["iteration"] == 3


def test_ctde_checkpoint_carries_the_actual_ctde_training_state(tmp_path):
    """PO1/12: a CTDE payload keeps the actor keys AND adds the critic's own state."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    updater = CTDEUpdater(policy, critic, PPOConfig(), CTDEConfig())
    path = graph_train.save_checkpoint(policy, updater, 5, tmp_path, critic=critic)
    payload = torch.load(path, weights_only=False)
    assert set(payload) == {
        "iteration", "encoder", "head", "optimizer", "ppo_config",
        "training_mode", "critic_encoder", "value_head", "critic_optimizer",
        "ctde_config",
    }
    assert payload["training_mode"] == graph_train.TRAINING_MODE_CTDE
    assert payload["ctde_config"] == {
        "critic_lr": 3e-4, "value_coeff": 0.5, "gae_lambda": 0.95,
    }
    # The critic state really round-trips into a fresh critic.
    fresh = build_central_critic()
    fresh.encoder.load_state_dict(payload["critic_encoder"])
    fresh.value_head.load_state_dict(payload["value_head"])
    obs = _synthetic_central()
    assert torch.allclose(fresh(obs), critic(obs))


def test_ctde_credit_does_not_call_the_actor_only_credit_function():
    """PO1: `compute_ctde_advantages` is a REPLACEMENT, not a wrapper.

    If it delegated to `compute_returns_and_advantages`, the Phase-A credit assignment
    would be on the CTDE path and any future CTDE change could move it.
    """
    saved = graph_ppo.compute_returns_and_advantages

    def boom(*_a, **_k):
        raise _Poison("the actor-only credit function was called from the CTDE path")

    graph_ppo.compute_returns_and_advantages = boom
    try:
        critic = build_central_critic()
        rec = CTDEEpisodeRecord.from_episode(
            [_transition(_actor_obs(), reward=-1.0)], [_synthetic_central()], -1.0,
        )
        batch = compute_ctde_advantages([rec], critic, PPOConfig(), CTDEConfig())
    finally:
        graph_ppo.compute_returns_and_advantages = saved
    assert batch.n_transitions == 1


# The sentinel critic diagnostics `_SentinelCTDEUpdater` returns. Deliberately values
# nothing in the pipeline could produce by accident, so a record carrying them can only
# have COPIED them out of the updater's own dict.
_CRITIC_SENTINELS = {
    "value_loss": 12.5,
    "value_mean": -3.25,
    "value_target_mean": 7.75,
    "critic_grad_norm": 0.125,
}


class _SentinelCTDEUpdater:
    """A CTDEUpdater stand-in returning KNOWN critic diagnostics.

    It runs no epochs and touches no weights: what is under test is whether
    ``train`` persists the updater's OWN numbers, not how they were produced.
    """

    def __init__(self, policy, critic, ppo, ctde):
        self.cfg = ppo
        self.ctde_cfg = ctde
        self.optimizer = torch.optim.Adam(policy.encoder.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

    def update(self, buf):
        n_eps = getattr(buf, "n_episodes", 0)
        diag = {
            "policy_loss": 0.0, "total_loss": 0.0, "entropy": 0.0, "mean_ratio": 1.0,
            "clip_fraction": 0.0, "approx_kl": 0.0, "max_ratio_dev": 0.0,
            "grad_norm": 0.0, "adv_std_raw": 0.0, "n_transitions": 0,
            "n_episodes": n_eps, "episodes_with_wakes": 0,
            "n_epochs_run": 1 if n_eps else 0, "baseline": -0.5,
        }
        diag.update(_CRITIC_SENTINELS)
        return diag


def _train_records(run_dir):
    path = Path(run_dir) / "train_records.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def test_ctde_train_records_persist_the_updaters_critic_diagnostics(tmp_path):
    """F2: a CTDE iteration record carries the four critic diagnostics VERBATIM.

    The updater is replaced by one returning sentinel values no real computation would
    produce, so a matching record proves the numbers were COPIED out of the updater's
    diagnostics rather than recomputed somewhere else.
    """
    restore = ["_run_one_episode", "_build_generator", "_git_provenance", "CTDEUpdater"]
    saved = graph_train.CTDEUpdater
    cfg = graph_train.TrainConfig(
        n_iterations=1, episodes_per_iteration=2, eval_every=0, eval_episodes=0,
        output_dir=str(tmp_path / "run_ctde_records"),
        training_mode=graph_train.TRAINING_MODE_CTDE,
    )
    try:
        graph_train.CTDEUpdater = _SentinelCTDEUpdater
        _run_stub_training(cfg, restore)
    finally:
        graph_train.CTDEUpdater = saved

    records = _train_records(cfg.output_dir)
    assert len(records) == 1
    rec = records[0]
    for key, value in _CRITIC_SENTINELS.items():
        assert key in rec, "CTDE record is missing %r" % key
        assert rec[key] == value, (
            "%s was not taken from the updater diagnostics (got %r, expected %r)"
            % (key, rec[key], value)
        )
    # The actor-side reward keys keep their existing meaning and are NOT the critic's.
    assert rec["train_reward_mean"] == rec["baseline"] == -0.5
    assert rec["baseline"] != rec["value_mean"]


def test_actor_only_train_records_carry_no_critic_keys(tmp_path):
    """F2: an actor_only record is not converted into a CTDE-shaped one.

    The keys are ABSENT, not null: an actor_only updater has no critic to describe, and
    a nullable key would invite reading "no critic" as "a critic that scored 0".
    """
    restore = ["_run_one_episode", "_build_generator", "_git_provenance"]
    cfg = graph_train.TrainConfig(
        n_iterations=1, episodes_per_iteration=2, eval_every=0, eval_episodes=0,
        output_dir=str(tmp_path / "run_actor_records"),
    )
    _run_stub_training(cfg, restore)
    rec = _train_records(cfg.output_dir)[0]
    for key in _CRITIC_SENTINELS:
        assert key not in rec, "actor_only record leaked the CTDE key %r" % key
    # ...while every actor-side key it always had is still there and still means the
    # same thing.
    for key in ("train_reward_mean", "baseline", "policy_loss", "total_loss",
                "entropy", "grad_norm", "adv_std_raw", "n_transitions"):
        assert key in rec, "actor_only record lost %r" % key


def test_the_four_critic_keys_are_exactly_what_the_real_updater_returns():
    """F2: the persisted names are the REAL `CTDEUpdater.update` keys, not new coinage.

    If the updater ever renamed one, the record writer would silently start persisting
    a stale key -- this pins the two together.
    """
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    updater = CTDEUpdater(policy, critic, PPOConfig(n_epochs=1), CTDEConfig())
    gobs = _actor_obs()
    rec = CTDEEpisodeRecord.from_episode(
        [_transition(gobs, reward=-1.0)], [_synthetic_central()], -1.0,
    )
    diag = updater.update([rec])
    for key in _CRITIC_SENTINELS:
        assert key in diag, "CTDEUpdater no longer returns %r" % key
        assert np.isfinite(diag[key]), "%s is not finite" % key


def test_actor_only_run_records_its_mode_and_no_ctde_block(tmp_path):
    """PO1: `run_config.json:/training` states the mode; `ctde` is null when unused."""
    run_dir = tmp_path / "rc"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = graph_train.TrainConfig(n_iterations=1, output_dir=str(run_dir))
    path = graph_train.write_run_config(
        run_dir, cfg, provenance={"git": {"available": True}},
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = data["training"]
    assert block["mode"] == graph_train.TRAINING_MODE_ACTOR_ONLY
    assert block["ctde_enabled"] is False
    assert block["ctde"] is None
    # Execution is a CONSTANT: CTDE never changes how actions are taken.
    assert block["execution"] == "decentralized_actor_only"


# =============================================================================
# PO2 -- NO PRIVILEGED LEAKAGE / TRULY DECENTRALIZED EXECUTION
# =============================================================================

def test_actor_and_critic_parameters_are_disjoint():
    """PO2: no shared encoder, no tied weights, no shared parameter object."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    actor_ids = {id(p) for p in
                 list(policy.encoder.parameters()) + list(policy.head.parameters())}
    critic_ids = {id(p) for p in critic.parameters()}
    assert actor_ids and critic_ids
    assert actor_ids.isdisjoint(critic_ids)
    # And the encoder OBJECTS themselves are distinct instances.
    assert policy.encoder is not critic.encoder
    assert isinstance(critic.encoder, GraphEncoder)


def test_actor_backward_leaves_critic_gradients_none_and_vice_versa():
    """PO2: the two losses are backpropagated on genuinely disjoint graphs."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()

    # Actor side only.
    logits = policy.head(policy.encoder(_actor_obs()))
    logits.sum().backward()
    assert all(p.grad is None for p in critic.parameters()), \
        "an actor backward produced critic gradients"
    actor_touched = [p for p in policy.encoder.parameters() if p.grad is not None]
    assert actor_touched, "the actor backward produced no actor gradients at all"

    # Critic side only, on a fresh pair.
    torch.manual_seed(0)
    policy2 = graph_train.build_policy()
    critic2 = build_central_critic()
    critic2(_synthetic_central()).backward()
    assert all(p.grad is None for p in
               list(policy2.encoder.parameters()) + list(policy2.head.parameters())), \
        "a critic backward produced actor gradients"
    assert any(p.grad is not None for p in critic2.parameters())


def test_actor_advantage_is_a_detached_scalar():
    """PO2: the advantage crossing critic -> actor is a plain float, not a graph."""
    critic = build_central_critic()
    rec = CTDEEpisodeRecord.from_episode(
        [_transition(_actor_obs(), reward=-1.0)], [_synthetic_central()], -1.0,
    )
    batch = compute_ctde_advantages([rec], critic, PPOConfig(), CTDEConfig())
    assert isinstance(batch.advantages, np.ndarray)
    assert isinstance(float(batch.advantages[0]), float)
    assert not isinstance(batch.advantages[0], torch.Tensor)


def test_privileged_features_cannot_move_an_actor_logit():
    """PO2, the central claim: the actor's output is a function of ITS OWN input alone.

    Two WILDLY different central states -- different live-target counts, different
    fuel, different sensing, different assignment -- against the SAME private actor
    observation and the SAME actor weights. The logits and the deterministic action must
    be bit-identical, because the central state is not an input to the actor at all.
    """
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    gobs = _actor_obs()

    logits_a = policy.head(policy.encoder(gobs)).detach().clone()
    _ = build_central_critic()(_synthetic_central(k=3, a=2, seed=1))
    _ = build_central_critic()(_synthetic_central(k=7, a=4, seed=99))
    logits_b = policy.head(policy.encoder(gobs)).detach().clone()

    assert torch.equal(logits_a, logits_b)
    mask = build_action_mask(gobs)
    from match_aou.rl.action.graph_action import sample_action
    a1 = sample_action(logits_a, mask, deterministic=True)[:2]
    a2 = sample_action(logits_b, mask, deterministic=True)[:2]
    assert a1 == a2


def test_a_central_state_is_not_an_actor_observation():
    """PO2: the two are DIFFERENT TYPES, and the central one has no ego identity."""
    central = _synthetic_central()
    assert not isinstance(central, GraphObservation)
    assert not isinstance(central, type(_actor_obs()))
    # No `agent_id`: the actor's observation is FOR an ego; this one has no ego.
    assert not hasattr(central, "agent_id")
    assert central.ego_index == NO_EGO_INDEX
    # It cannot be fed to the actor's mask -- it lacks the columns the mask reads.
    try:
        build_action_mask(central)
        raised = False
    except Exception:
        raised = True
    assert raised, "a central state was silently accepted by build_action_mask"


def test_actor_encoder_marks_no_ego_when_ego_index_is_the_sentinel():
    """PO2/3: `ego_index = -1` really leaves every agent node symmetric.

    Permuting the two agent rows of an otherwise symmetric central state must not change
    the pooled summary -- which is what "no distinguished ego" means operationally.
    """
    torch.manual_seed(0)
    critic = build_central_critic()
    base = _synthetic_central(k=2, a=2, seed=5)
    # Make the two agents identical so a permutation is a true relabeling.
    base.agent_features[:] = base.agent_features[0]
    swapped = CentralGraphObservation(
        task_features=base.task_features.copy(),
        agent_features=base.agent_features[::-1].copy(),
        ego_index=base.ego_index,
        edge_index=base.edge_index.copy(),
        edge_type=base.edge_type.copy(),
        edge_attr=base.edge_attr.copy(),
        task_target_ids=list(base.task_target_ids),
        agent_ids=list(reversed(base.agent_ids)),
        current_time=base.current_time,
        time_norm=base.time_norm,
    )
    assert torch.allclose(critic(base), critic(swapped), atol=1e-6)


def test_evaluation_never_constructs_a_critic_or_a_recorder():
    """PO2/11: `evaluate` is actor-only -- it takes no critic and builds no central state."""
    import inspect
    sig = inspect.signature(graph_train.evaluate)
    assert "critic" not in sig.parameters
    assert "central_recorder" not in sig.parameters
    src = inspect.getsource(graph_train.evaluate)
    assert "CentralStateRecorder" not in src
    assert "_ctde_kwargs" not in src
    assert "build_central_critic" not in src


def test_a_ctde_trained_actor_runs_with_the_critic_absent():
    """PO2/11: inference needs the actor half and nothing else."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    updater = CTDEUpdater(policy, critic, PPOConfig(n_epochs=1), CTDEConfig())
    gobs = _actor_obs()
    rec = CTDEEpisodeRecord.from_episode(
        [_transition(gobs, reward=-1.0)], [_synthetic_central()], -1.0,
    )
    updater.update([rec])

    # Rebuild the actor alone from its state dicts; no critic object exists here.
    encoder = GraphEncoder()
    head = ActionHead(embed_dim=encoder.embed_dim)
    encoder.load_state_dict(policy.encoder.state_dict())
    head.load_state_dict(policy.head.state_dict())
    del critic, updater
    with torch.no_grad():
        logits = head(encoder(gobs))
    assert logits.shape == (gobs.task_features.shape[0], 3)
    assert bool(torch.isfinite(logits).all())


def test_central_builder_imports_no_torch_blade_or_gym():
    """PO2: the central observation layer is a PURE projection -- no engine, no torch.

    `graph_hidden_placement`'s purity check, applied to the other new module: pyomo is
    the documented inherited root-package baggage (see CLAUDE.md), everything else must
    be absent.
    """
    child = (
        "import sys, json, importlib\n"
        "importlib.import_module('match_aou.rl.observation.central_graph_builder')\n"
        "bad = [m for m in ('torch','blade','gymnasium','gym') "
        "      if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "print('CTDE_PURITY:' + json.dumps(sorted(set(bad))))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True,
                          text=True, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("CTDE_PURITY:"))
    bad = json.loads(line[len("CTDE_PURITY:"):])
    assert bad == [], f"central_graph_builder pulled in {bad}"


# =============================================================================
# PO3 -- CENTRAL VALUE / GAE ALIGNMENT
# =============================================================================

def test_gae_matches_a_hand_computed_trajectory():
    """PO3: the exact GAE recursion, on numbers checked by hand."""
    rewards = [0.0, 0.0, -1.0]
    values = [0.5, 0.25, 0.1]
    gamma, lam = 1.0, 0.95
    adv, targets = compute_gae(rewards, values, gamma=gamma, gae_lambda=lam)

    d2 = -1.0 + gamma * 0.0 - 0.1        # V_next of the LAST decision is 0
    d1 = 0.0 + gamma * 0.1 - 0.25
    d0 = 0.0 + gamma * 0.25 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    assert np.allclose(adv, [a0, a1, a2])
    assert np.allclose(targets, [a0 + 0.5, a1 + 0.25, a2 + 0.1])


def test_the_final_decision_bootstraps_from_zero():
    """PO3: `V_next = 0` at the end -- the episode really ends, it is not truncated.

    A one-decision episode makes this exact: A_0 must be `r_0 - V_0` and nothing else.
    """
    adv, targets = compute_gae([-1.0], [0.4], gamma=1.0, gae_lambda=0.95)
    assert np.isclose(adv[0], -1.0 - 0.4)
    assert np.isclose(targets[0], -1.0)  # A_0 + V_0 == r_0


def test_gae_is_per_episode_and_never_crosses_an_episode_boundary():
    """PO3: two episodes in one batch do not bootstrap into each other."""
    critic = build_central_critic()
    gobs = _actor_obs()
    recs = []
    for r in (-1.0, -0.5):
        trs = [_transition(gobs), _transition(gobs, reward=r)]
        recs.append(CTDEEpisodeRecord.from_episode(
            trs, [_synthetic_central(seed=1), _synthetic_central(seed=2)], r,
        ))
    batch = compute_ctde_advantages(recs, critic, PPOConfig(), CTDEConfig())
    assert batch.n_transitions == 4

    # Recompute each episode independently and compare the RAW advantages.
    expected = []
    for rec in recs:
        vals = [float(critic(s).item()) for s in rec.central_states]
        rew = [float(t.reward or 0.0) for t in rec.transitions]
        expected.extend(compute_gae(rew, vals, gamma=1.0, gae_lambda=0.95)[0])
    assert np.allclose(batch.raw_advantages, expected, atol=1e-6)


def test_value_targets_are_fixed_across_update_epochs():
    """PO3: the regression target cannot chase the network fitting it."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    cfg = PPOConfig(n_epochs=3)
    updater = CTDEUpdater(policy, critic, cfg, CTDEConfig())
    gobs = _actor_obs()
    rec = CTDEEpisodeRecord.from_episode(
        [_transition(gobs), _transition(gobs, reward=-1.0)],
        [_synthetic_central(seed=1), _synthetic_central(seed=2)], -1.0,
    )
    before = compute_ctde_advantages([rec], critic, cfg, CTDEConfig())
    targets_before = before.value_targets.copy()

    seen = {}
    real = graph_ppo.compute_ctde_advantages

    def spy(source, critic_, c, cc):
        batch = real(source, critic_, c, cc)
        seen.setdefault("targets", batch.value_targets.copy())
        seen["calls"] = seen.get("calls", 0) + 1
        return batch

    graph_ppo.compute_ctde_advantages = spy
    try:
        diag = updater.update([rec])
    finally:
        graph_ppo.compute_ctde_advantages = real

    # The batch (hence its targets) is computed ONCE per update, before the epochs.
    assert seen["calls"] == 1
    assert np.allclose(seen["targets"], targets_before)
    assert diag["n_epochs_run"] == 3
    assert len(diag["per_epoch"]["value_loss"]) == 3


def test_ctde_baseline_diagnostic_is_the_mean_episode_REWARD_not_the_critic_value():
    """PO1/PO3: `baseline` means the SAME thing in both training modes.

    `graph_train` records `diag["baseline"]` as an iteration's `train_reward_mean`. If
    the CTDE updater reported the critic's mean VALUE there, one recorded field would
    mean a reward under `actor_only` and a value estimate under `ctde` -- the two modes'
    learning curves would stop being comparable while still looking as though they were.
    This was a REAL defect caught by an end-to-end smoke, not a hypothetical.
    """
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    updater = CTDEUpdater(policy, critic, PPOConfig(n_epochs=1), CTDEConfig())
    gobs = _actor_obs()
    rewards = (-0.3333, -0.8750)
    recs = [
        CTDEEpisodeRecord.from_episode(
            [_transition(gobs, reward=r)], [_synthetic_central(seed=i)], r,
        )
        for i, r in enumerate(rewards)
    ]
    diag = updater.update(recs)
    assert np.isclose(diag["baseline"], float(np.mean(rewards))), (
        "CTDE `baseline` is not the batch's mean EPISODE REWARD (got %r)"
        % (diag["baseline"],)
    )
    # The critic's own estimate is still reported -- just under its own name.
    assert "value_mean" in diag
    assert not np.isclose(diag["value_mean"], diag["baseline"]), (
        "the fixture is degenerate: value_mean and baseline coincide, so this test "
        "could not tell the two apart"
    )

    # A zero-wake episode carries baseline mass here exactly as it does actor-only.
    recs.append(CTDEEpisodeRecord.from_episode([], [], -1.0))
    diag2 = updater.update(recs)
    assert np.isclose(diag2["baseline"], float(np.mean(list(rewards) + [-1.0])))


def test_zero_wake_episodes_create_no_ctde_samples():
    """PO3: a zero-wake episode contributes nothing -- and is NOT a failure."""
    critic = build_central_critic()
    empty = CTDEEpisodeRecord.from_episode([], [], -0.5)
    assert not empty.has_wakes
    batch = compute_ctde_advantages([empty], critic, PPOConfig(), CTDEConfig())
    assert batch.n_transitions == 0
    assert batch.n_episodes == 1
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    updater = CTDEUpdater(policy, critic, PPOConfig(), CTDEConfig())
    diag = updater.update([empty])           # a clean no-op, not an exception
    assert diag["n_epochs_run"] == 0
    assert diag["n_episodes"] == 1


def test_misaligned_central_samples_fail_loud():
    """PO3: a drifted capture seam is caught at record construction, never averaged."""
    try:
        CTDEEpisodeRecord.from_episode(
            [_transition(_actor_obs())], [], -1.0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised, "a mismatched transition/central-state count was accepted"


def test_variable_graph_sizes_stay_finite_without_padding():
    """PO3: the critic is size-agnostic -- no padding, no NaN, including k = 0."""
    torch.manual_seed(0)
    critic = build_central_critic()
    for k, a in ((1, 1), (3, 2), (6, 4), (0, 3), (4, 1)):
        v = critic(_synthetic_central(k=k, a=a, seed=k * 10 + a))
        assert v.shape == (), f"V(s) must be a scalar, got {tuple(v.shape)}"
        assert bool(torch.isfinite(v)), f"non-finite V(s) at k={k}, a={a}"


def test_ctde_update_moves_both_parameter_sets():
    """PO3/10: an update really trains the actor AND the critic, on separate optimizers."""
    torch.manual_seed(0)
    policy = graph_train.build_policy()
    critic = build_central_critic()
    updater = CTDEUpdater(policy, critic, PPOConfig(n_epochs=2), CTDEConfig())
    assert updater.optimizer is not updater.critic_optimizer
    # The UPDATER's own two parameter lists must be disjoint too -- this is what a
    # future "share the encoder to save a forward pass" change would violate, and it
    # catches that semantically rather than waiting for a feature-width crash.
    assert {id(p) for p in updater.actor_parameters}.isdisjoint(
        {id(p) for p in updater.critic_parameters}
    ), "the CTDE updater shares parameters between the actor and the critic"
    a0 = [p.detach().clone() for p in updater.actor_parameters]
    c0 = [p.detach().clone() for p in updater.critic_parameters]
    gobs = _actor_obs()
    rec = CTDEEpisodeRecord.from_episode(
        [_transition(gobs), _transition(gobs, reward=-1.0)],
        [_synthetic_central(seed=1), _synthetic_central(seed=2)], -1.0,
    )
    diag = updater.update([rec])
    assert diag["n_epochs_run"] == 2
    assert any(not torch.equal(p, q)
               for p, q in zip(updater.actor_parameters, a0)), "actor did not move"
    assert any(not torch.equal(p, q)
               for p, q in zip(updater.critic_parameters, c0)), "critic did not move"
    assert np.isfinite(diag["value_loss"])


# --- the central graph's CONTENT ------------------------------------------------

def test_every_live_agent_carries_its_own_real_fuel():
    """PO3: no featureless peers here -- that asymmetry is an ACTOR-graph property."""
    scen, ex, ids, _ = _world(agents=("ego0", "ego1", "ego2"))
    scen.aircraft[0].current_fuel = 12000.0
    scen.aircraft[1].current_fuel = 6000.0
    scen.aircraft[2].current_fuel = 3000.0
    c = _central(scen, ex, ids)
    assert c.n_agents == 3
    assert np.allclose(c.agent_features[:, 0], [1.0, 0.5, 0.25])


def test_all_live_targets_appear_including_unallocated_ones():
    """PO3: the inventory is the RAW LIVE WORLD, never an allocation.

    Only ego0 is assigned, and only to t0. t1 and t2 are unallocated -- and physically
    present, sensible and attackable -- so they MUST be nodes.
    """
    scen, ex, ids, _ = _world(n_targets=3, plans={"ego0": [(0, 0, 0)]})
    c = _central(scen, ex, ids)
    assert c.task_target_ids == ["t0", "t1", "t2"]
    assert c.n_tasks == 3
    # Utility 80/100 for an Airbase, probability 1.0 -- the two central task columns.
    assert np.allclose(c.task_features[:, 0], 0.8)
    assert np.allclose(c.task_features[:, 1], 1.0)


def test_a_destroyed_target_loses_its_critic_node_and_nothing_else():
    """PO3: physical destruction removes ONE task node; agents and beliefs are untouched."""
    scen, ex, ids, tasks = _world(n_targets=3, plans={"ego0": [(0, 0, 0), (1, 0, 0)]})
    before = _central(scen, ex, ids)
    assert before.n_tasks == 3 and before.n_agents == 2

    # The engine's own removal: `weapon_endgame` drops the unit from scenario.airbases.
    scen.airbases = [b for b in scen.airbases if getattr(b, "id", None) != "t1"]
    after = _central(scen, ex, ids)
    assert after.task_target_ids == ["t0", "t2"]
    assert after.n_agents == 2, "an agent node moved when a TARGET died"
    # The ACTOR-side per-ego task lists are untouched: no belief was edited.
    assert [t.steps[0].target_id for t in ex.tasks["ego0"]] == ["t0", "t1", "t2"]
    assert [t.steps[0].target_id for t in ex.tasks["ego1"]] == ["t0", "t1", "t2"]


def test_a_dead_agent_loses_its_critic_node_and_nothing_else():
    """PO3: an aircraft absent from BOTH the air and every inventory is dead -> no node."""
    scen, ex, ids, _ = _world(n_targets=2, agents=("ego0", "ego1"))
    assert _central(scen, ex, ids).n_agents == 2
    scen.aircraft = [a for a in scen.aircraft if a.id != "ego1"]   # remove_aircraft
    after = _central(scen, ex, ids)
    assert after.agent_ids == ["ego0"]
    assert after.n_tasks == 2, "a task node moved when an AGENT died"


def test_rtb_and_landing_are_not_death():
    """PO3: an ego ordered home, and one already landed, both stay LIVE critic nodes.

    RTB issuance is an order, not an outcome, and a landed airframe is still an
    airframe. Reading either as death would delete a live entity from the team state.
    """
    scen, ex, ids, _ = _world(n_targets=2, agents=("ego0", "ego1"))
    scen.aircraft[1].rtb = True                       # ordered home, still flying
    assert _central(scen, ex, ids).agent_ids == ["ego0", "ego1"]

    # Now land it, exactly as `Game.land_aicraft` does: append to the base inventory,
    # then remove it from the air.
    landed = scen.aircraft[1]
    scen.airbases[0].aircraft.append(landed)
    scen.aircraft = [scen.aircraft[0]]
    c = _central(scen, ex, ids)
    assert c.agent_ids == ["ego0", "ego1"], "a LANDED ego was treated as dead"
    assert live_aircraft(scen, "ego1") is landed
    assert np.isclose(c.agent_features[1, 0], 1.0)    # its own real fuel


def test_assigned_follows_the_current_executor_plan_not_a_static_one():
    """PO3: `assigned` is CURRENT plan membership, and it moves when a plan is resynced."""
    scen, ex, ids, _ = _world(n_targets=3, agents=("ego0", "ego1"),
                              plans={"ego0": [(0, 0, 0)], "ego1": [(2, 0, 0)]})
    c = _central(scen, ex, ids)
    k, a = c.n_tasks, c.n_agents
    assigned = c.edge_attr[:, 4].reshape(a, k)        # rows follow agent order
    assert np.allclose(assigned[0], [1.0, 0.0, 0.0])
    assert np.allclose(assigned[1], [0.0, 0.0, 1.0])

    # A runtime adaptation: ego0 aborts (empty slice), ego1 picks up t1 as well.
    ex.plans["ego0"] = []
    ex.plans["ego1"] = [(1, 0, 0), (2, 0, 0)]
    c2 = _central(scen, ex, ids)
    assigned2 = c2.edge_attr[:, 4].reshape(a, k)
    assert np.allclose(assigned2[0], [0.0, 0.0, 0.0])
    assert np.allclose(assigned2[1], [0.0, 1.0, 1.0])


def test_sensed_is_all_agent_and_follows_the_unified_detection_radius():
    """PO3: `sensed` is privileged -- computed per agent, from that agent's position."""
    cfg = GraphObservationConfig(detection_range_km=50.0)
    # ego0 sits on the base; ego1 sits right on top of target t1.
    scen, ex, ids, _ = _world(n_targets=2, agents=("ego0", "ego1"))
    scen.aircraft[1].latitude = 32.4       # t1 is at 32.4
    c = _central(scen, ex, ids, config=cfg)
    k, a = c.n_tasks, c.n_agents
    sensed = c.edge_attr[:, 3].reshape(a, k)
    # t0 is at 32.2 (~22 km from the base) -> ego0 senses it; t1 at 32.4 (~44 km) too.
    assert sensed[0, 0] == 1.0
    # ego1 is ON t1 and ~22 km from t0 -> senses both.
    assert sensed[1, 1] == 1.0
    # Widen nothing, but move ego0 far away: its own row must go dark.
    scen.aircraft[0].latitude = 30.0
    c2 = _central(scen, ex, ids, config=cfg)
    sensed2 = c2.edge_attr[:, 3].reshape(a, k)
    assert sensed2[0].sum() == 0.0
    assert sensed2[1, 1] == 1.0, "one agent's sensing changed another's"


def test_the_central_graph_is_the_complete_bipartite_relation_on_spatial():
    """PO3: one SPATIAL edge per (live agent, live target), agent -> task."""
    scen, ex, ids, _ = _world(n_targets=3, agents=("ego0", "ego1"))
    c = _central(scen, ex, ids)
    k, a = c.n_tasks, c.n_agents
    assert c.edge_index.shape == (2, k * a)
    assert c.edge_attr.shape == (k * a, CENTRAL_EDGE_ATTR_DIM)
    assert set(c.edge_type.tolist()) == {int(EdgeType.SPATIAL)}
    # Sources are agent nodes [k, k+a), destinations are task nodes [0, k).
    assert set(c.edge_index[0].tolist()) == set(range(k, k + a))
    assert set(c.edge_index[1].tolist()) == set(range(k))


def test_plan_target_ids_matches_the_real_executor_resolution():
    """PO3: the `assigned` resolver is a TEST-ENFORCED mirror of `_resolve_step`.

    Same bounds semantics on every shape that matters, measured against a REAL
    `GraphPlanExecutor` rather than against a second hand-written expectation.
    """
    tasks = [_task("t0", 32.2, 35.0), _task("t1", 32.4, 35.0)]
    solution = {
        "ego0": [(0, 0, 0), (1, 0, 0)],
        # Deliberately degenerate: an out-of-range task index and an out-of-range step
        # index. `_resolve_step` returns None for both, so neither may contribute a
        # target id here either.
        "ego1": [(9, 0, 0), (0, 7, 0)],
    }
    ex = GraphPlanExecutor(
        tasks=tasks, solution=solution,
        agents=[_agent("ego0"), _agent("ego1")],
    )
    for ego in ("ego0", "ego1"):
        expected = set()
        for assignment in ex.plans.get(ego, []):
            step = ex._resolve_step(ego, assignment)
            if step is not None and getattr(step, "target_id", None) is not None:
                expected.add(str(step.target_id))
        assert plan_target_ids(ex, ego) == expected, ego
    assert plan_target_ids(ex, "ego0") == {"t0", "t1"}
    assert plan_target_ids(ex, "ego1") == set()
    assert plan_target_ids(ex, "no-such-ego") == set()


def test_time_norm_uses_the_actor_normalization():
    """PO3: one clock convention, shared with the actor builder."""
    scen, ex, ids, _ = _world(n_targets=1)
    cfg = GraphObservationConfig(max_sim_ticks=14400)
    c = _central(scen, ex, ids, t=1440, config=cfg)
    assert np.isclose(c.time_norm, 0.1)
    assert c.current_time == 1440
    # Clipped, never > 1.
    assert _central(scen, ex, ids, t=999999, config=cfg).time_norm == 1.0


# --- the CAPTURE seam ------------------------------------------------------------

class _TickCtx:
    """The `EpisodeContext` surface `run_episode` reads, over the stubs above."""

    def __init__(self, scen, ex, agent_ids, beliefs, env):
        self.observation = scen
        self.executor = ex
        self.agent_ids = list(agent_ids)
        self.beliefs = beliefs
        self.env = env
        self.game = None
        self.record = False


def test_one_central_sample_per_wake_in_real_decision_order():
    """PO3: the recorder is filled 1:1 with the trajectory, in the order decisions ran.

    Two egos are forced to wake on the SAME tick. The result must be two ordered samples
    with NO `env.step` between them, and one sample per transition.
    """
    from match_aou.rl.training import graph_tick_loop as tl

    scen, ex, ids, tasks = _world(n_targets=2, agents=("ego0", "ego1"),
                                  plans={"ego0": [(0, 0, 0)], "ego1": [(1, 0, 0)]})

    steps = {"n": 0}

    class _Env:
        def step(self, _commands):
            steps["n"] += 1
            return scen, 0.0, False, False, {}

    class _Belief:
        def __init__(self):
            self.tasks = list(tasks)
            self.solution = {"ego0": [(0, 0, 0)], "ego1": [(1, 0, 0)]}

    beliefs = {a: _Belief() for a in ids}
    ctx = _TickCtx(scen, ex, ids, beliefs, _Env())

    # Force exactly one wake per ego on tick 0, then never again.
    fired = {"n": 0}

    def fake_triggers(btasks, bsol, sensed, eta=None, *, ego_id, clock, fuel_damage=False):
        wake = clock == 0
        if wake:
            fired["n"] += 1
        return btasks, bsol, wake, []

    order = []

    def fake_wake(policy, ego_id, obs, belief, executor, cfg, tick, *, deterministic=False):
        order.append(("act", str(ego_id), int(tick)))
        # A real decision resyncs THIS ego's slice; the CTDE contract says a later
        # same-tick capture is allowed to see it.
        executor.plans[str(ego_id)] = []
        return _transition(_actor_obs(), meta=2)

    saved_t, saved_w = tl.decide_triggers, tl._wake_decision
    tl.decide_triggers = fake_triggers
    tl._wake_decision = fake_wake
    recorder = CentralStateRecorder()
    try:
        result = tl.run_episode(
            policy=None, ctx=ctx, max_ticks=3, central=recorder,
            cfg=GraphObservationConfig(detection_range_km=50.0),
        )
    finally:
        tl.decide_triggers, tl._wake_decision = saved_t, saved_w

    assert fired["n"] >= 2
    assert len(result.trajectory) == 2
    assert len(recorder.samples) == 2, "central samples are not 1:1 with wakes"
    # Both decisions happened on tick 0, i.e. inside ONE Phase 1, before any env.step.
    assert [o[2] for o in order] == [0, 0]
    assert [o[1] for o in order] == ["ego0", "ego1"]


def test_the_second_same_tick_sample_sees_the_first_egos_resync():
    """PO3: capture happens BEFORE each action, so sample B carries A's applied edit.

    Sample A must still show ego0 assigned (its action has not run yet); sample B, taken
    after ego0 acted and resynced but before any `env.step`, must show ego0's slice
    emptied. That is the causal same-tick ordering the CTDE contract allows.
    """
    from match_aou.rl.training import graph_tick_loop as tl

    scen, ex, ids, tasks = _world(n_targets=2, agents=("ego0", "ego1"),
                                  plans={"ego0": [(0, 0, 0)], "ego1": [(1, 0, 0)]})

    class _Env:
        def step(self, _commands):
            return scen, 0.0, False, False, {}

    class _Belief:
        def __init__(self):
            self.tasks = list(tasks)
            self.solution = {}

    ctx = _TickCtx(scen, ex, ids, {a: _Belief() for a in ids}, _Env())

    def fake_triggers(btasks, bsol, sensed, eta=None, *, ego_id, clock, fuel_damage=False):
        return btasks, bsol, clock == 0, []

    def fake_wake(policy, ego_id, obs, belief, executor, cfg, tick, *, deterministic=False):
        if str(ego_id) == "ego0":
            executor.plans["ego0"] = []      # an ABORT: the real resync effect
        return _transition(_actor_obs(), meta=2)

    saved_t, saved_w = tl.decide_triggers, tl._wake_decision
    tl.decide_triggers = fake_triggers
    tl._wake_decision = fake_wake
    recorder = CentralStateRecorder()
    try:
        tl.run_episode(policy=None, ctx=ctx, max_ticks=1, central=recorder,
                       cfg=GraphObservationConfig(detection_range_km=50.0))
    finally:
        tl.decide_triggers, tl._wake_decision = saved_t, saved_w

    assert len(recorder.samples) == 2
    a_s, b_s = recorder.samples
    k, a = a_s.n_tasks, a_s.n_agents
    assigned_a = a_s.edge_attr[:, 4].reshape(a, k)
    assigned_b = b_s.edge_attr[:, 4].reshape(a, k)
    assert assigned_a[0].sum() == 1.0, "sample A already reflected ego0's own action"
    assert assigned_b[0].sum() == 0.0, "sample B did not see ego0's applied resync"
    # ego1's own row is untouched between the two samples.
    assert np.allclose(assigned_a[1], assigned_b[1])


def _boundary_ctx():
    """One damaged ego with TWO assignments, a REAL executor, and a fired FD event.

    GENERALIZED-V1 step 2 (handoff 3l.4). The ego sits ON its first target, so the kill
    below is one it can CONFIRM with its own proximity-gated sensing -- which is what
    makes the completion a boundary rather than a peer's outcome learned some other way.
    """
    scen, _stub_ex, ids, tasks = _world(n_targets=2, agents=("ego0",))
    scen.aircraft[0].latitude = 32.2  # ON t0
    executor = GraphPlanExecutor(
        tasks=tasks, solution={"ego0": [(0, 0, 0), (1, 0, 0)]},
        agents=[_agent("ego0", 32.0, 35.0)], arrival_threshold_km=50.0,
    )

    class _Belief:
        def __init__(self):
            self.tasks = list(tasks)
            self.solution = {"ego0": [(0, 0, 0), (1, 0, 0)]}

    class _Env:
        def __init__(self):
            self.n = 0

        def step(self, _commands):
            self.n += 1
            if self.n == 1:  # t0 dies after the first tick
                scen.airbases = [b for b in scen.airbases if str(b.id) != "t0"]
            return scen, 0.0, False, False, {}

    ctx = _TickCtx(scen, executor, ids, {"ego0": _Belief()}, _Env())
    return ctx, scen, executor


def _fired_boundary_controller(scen, *, policy):
    """A controller whose fuel-damage event has ALREADY fired on ``ego0``."""
    params = FuelDamageParameters(
        mode=FuelDamageMode.FORCED_DAMAGED, post_fd_wake_policy=policy
    )
    plan = plan_fuel_damage(
        condition=CONDITION_DAMAGED, mode=params.mode, derived_seed=0,
        eligible_ego_ids=("ego0",), ego_id="ego0",
        launch_point=Location(32.0, 35.0), home_base=Location(32.0, 35.0),
        route_points=[Location(32.2, 35.0), Location(32.4, 35.0)],
        speed_knots=1303.0, fuel_rate=6700.0, max_fuel=12000.0,
        fuel_at_launch=12000.0, params=params,
    )
    controller = FuelDamageController(plan)
    assert controller.maybe_apply(scen, 0) == "ego0", "the fixture's event must fire"
    return controller


class _PlanSpyRecorder(CentralStateRecorder):
    """A recorder that also notes the EXECUTION state each capture was taken in."""

    def __init__(self):
        super().__init__()
        self.plans_at_capture = []

    def capture(self, *, scenario, agent_ids, executor, current_time, config=None):
        self.plans_at_capture.append(plan_target_ids(executor, "ego0"))
        return super().capture(
            scenario=scenario, agent_ids=agent_ids, executor=executor,
            current_time=current_time, config=config,
        )


def test_a_post_fd_boundary_sample_is_captured_after_the_local_reconciliation():
    """PO3 (GENERALIZED-V1): the boundary capture sees the RECONCILED execution state.

    The completion boundary reconciles the damaged ego's confirmed assignment out of its
    own belief and resyncs its own executor slice at the TOP of the tick, before Phase 1.
    The critic's capture then happens inside Phase 1, immediately before the action -- so
    a boundary sample must describe the plan the ego really has left, not the stale one
    the world already moved past. If the reconciliation had stayed in Phase 2 the sample
    would still show the completed assignment.

    Alignment is unchanged: still exactly ONE central sample per actor decision.
    """
    from match_aou.rl.training import graph_tick_loop as tl

    ctx, scen, executor = _boundary_ctx()
    controller = _fired_boundary_controller(
        scen, policy=POST_FD_WAKE_COMPLETION_BOUNDARY_V1
    )

    # `**_kind` absorbs the OPTIONAL `wake_kind` tag the tick loop passes for a
    # NON-ordinary wake (measurement hardening). It is reporting-only and this
    # stub asserts nothing about it; an ordinary wake is called without it.
    def fake_wake(policy, ego_id, obs, belief, ex, cfg, tick, *,
                  deterministic=False, **_kind):
        return _transition(_actor_obs(), meta=0)

    saved = tl._wake_decision
    tl._wake_decision = fake_wake
    recorder = _PlanSpyRecorder()
    try:
        result = tl.run_episode(
            policy=None, ctx=ctx, max_ticks=2, central=recorder,
            cfg=GraphObservationConfig(detection_range_km=50.0),
            fuel_damage=controller,
        )
    finally:
        tl._wake_decision = saved

    outcome = controller.post_fd_outcome
    assert outcome.boundary_wakes == 1, outcome.to_record()
    assert len(result.trajectory) == 1, result.trajectory
    assert len(recorder.samples) == 1, "central samples are not 1:1 with decisions"
    assert recorder.plans_at_capture == [{"t1"}], (
        "the boundary capture must follow the local reconciliation and the resync: "
        "expected the ego's remaining plan {'t1'}, got %r"
        % (recorder.plans_at_capture,)
    )
    # And the ACTOR's own belief lost the completed assignment before it decided.
    assert ctx.beliefs["ego0"].solution["ego0"] == [(1, 0, 0)]


def test_the_default_wake_policy_produces_no_boundary_sample():
    """PO3 CONTROL: under `single_wake_v1` the same world captures nothing at all.

    Without this the test above could not distinguish "the boundary produced a correctly
    ordered sample" from "this fixture wakes anyway".
    """
    from match_aou.rl.training import graph_tick_loop as tl

    ctx, scen, _executor = _boundary_ctx()
    controller = _fired_boundary_controller(scen, policy=POST_FD_WAKE_SINGLE_V1)

    # `**_kind` absorbs the OPTIONAL `wake_kind` tag the tick loop passes for a
    # NON-ordinary wake (measurement hardening). It is reporting-only and this
    # stub asserts nothing about it; an ordinary wake is called without it.
    def fake_wake(policy, ego_id, obs, belief, ex, cfg, tick, *,
                  deterministic=False, **_kind):
        return _transition(_actor_obs(), meta=0)

    saved = tl._wake_decision
    tl._wake_decision = fake_wake
    recorder = _PlanSpyRecorder()
    try:
        result = tl.run_episode(
            policy=None, ctx=ctx, max_ticks=2, central=recorder,
            cfg=GraphObservationConfig(detection_range_km=50.0),
            fuel_damage=controller,
        )
    finally:
        tl._wake_decision = saved

    assert result.trajectory == [] and recorder.samples == []
    assert controller.post_fd_outcome.armed is False
    # The kill still happened and the executor still confirmed it in Phase 2, exactly as
    # it always did -- what is absent is the extra DECISION, not the reconciliation.
    assert ("ego0", "t0") in _executor.done


def test_actor_only_run_episode_builds_no_central_state():
    """PO1/PO3: without a recorder the tick loop constructs nothing central."""
    from match_aou.rl.training import graph_tick_loop as tl
    from match_aou.rl.observation import central_graph_builder as cgb

    scen, ex, ids, tasks = _world(n_targets=2, agents=("ego0",))

    class _Env:
        def step(self, _commands):
            return scen, 0.0, False, False, {}

    class _Belief:
        def __init__(self):
            self.tasks = list(tasks)
            self.solution = {}

    ctx = _TickCtx(scen, ex, ids, {a: _Belief() for a in ids}, _Env())

    def fake_triggers(btasks, bsol, sensed, eta=None, *, ego_id, clock, fuel_damage=False):
        return btasks, bsol, clock == 0, []

    def boom(*_a, **_k):
        raise _Poison("a central observation was built on the actor-only path")

    saved_t, saved_w = tl.decide_triggers, tl._wake_decision
    saved_b = cgb.build_central_graph_observation
    tl.decide_triggers = fake_triggers
    tl._wake_decision = lambda *a, **k: _transition(_actor_obs())
    cgb.build_central_graph_observation = boom
    try:
        result = tl.run_episode(policy=None, ctx=ctx, max_ticks=1,
                                cfg=GraphObservationConfig(detection_range_km=50.0))
    finally:
        tl.decide_triggers, tl._wake_decision = saved_t, saved_w
        cgb.build_central_graph_observation = saved_b
    assert len(result.trajectory) == 1




# =============================================================================
# GENERALIZED-V1 EARLY STOPPING -- actor-only / CTDE PARITY
# =============================================================================

def _plateau_updaters(schedule):
    """Actor-only and CTDE updater stand-ins reporting ONE shared reward trajectory.

    The parity claim is about the STOPPING MECHANISM, so both arms must be fed the
    identical `train_reward_mean` sequence; anything else would leave "they stopped at
    the same iteration" explainable by the rewards rather than by the rule. Each stand-in
    keeps the real constructor shape (`train` checkpoints the updater, so it needs a
    genuine optimizer and a genuine PPOConfig) and replaces only `update`.
    """
    state = {"i": 0}

    def _diag(n_episodes):
        value = schedule[min(state["i"], len(schedule) - 1)]
        state["i"] += 1
        return {
            "baseline": value,
            "policy_loss": -0.01, "total_loss": -0.02, "entropy": 1.5,
            "mean_ratio": 1.0, "clip_fraction": 0.0, "approx_kl": 0.0,
            "max_ratio_dev": 0.0, "grad_norm": 0.5, "adv_std_raw": 0.1,
            "n_transitions": 2 * n_episodes, "n_episodes": n_episodes,
            "episodes_with_wakes": n_episodes,
            "n_epochs_run": 1 if n_episodes else 0,
            "value_loss": 0.0, "value_mean": 0.0, "value_target_mean": 0.0,
            "critic_grad_norm": 0.0,
        }

    class _ActorOnly:
        def __init__(self, policy, ppo):
            self.cfg = ppo
            self.optimizer = torch.optim.Adam(policy.encoder.parameters(), lr=1e-4)

        def update(self, buf):
            return _diag(int(getattr(buf, "n_episodes", 0)))

    class _Ctde:
        def __init__(self, policy, critic, ppo, ctde):
            self.cfg = ppo
            self.ctde_cfg = ctde
            self.optimizer = torch.optim.Adam(policy.encoder.parameters(), lr=1e-4)
            self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

        def update(self, buf):
            return _diag(int(getattr(buf, "n_episodes", 0)))

    return _ActorOnly, _Ctde


def _early_stopping_cfg(out_dir, mode):
    """A GENERALIZED-V1 run differing ONLY in `training_mode`.

    min=4 / window=2 / patience=2 -> checks at 4, 6, 8 completed iterations, earliest
    possible stop at 8 of the 10 scheduled: the approved 100 / 25 / 3 -> 175 machine at a
    size a stubbed test can execute.
    """
    return graph_train.TrainConfig(
        n_iterations=10, episodes_per_iteration=2, base_seed=0,
        eval_every=0, eval_episodes=0, checkpoint_every=0,
        output_dir=str(out_dir),
        episode_design="generalized_v1",
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
        generalized_max_attempts_per_iteration=2,
        training_mode=mode,
        early_stopping=True,
        early_stopping_min_iterations=4,
        early_stopping_window_iterations=2,
        early_stopping_patience_windows=2,
        early_stopping_min_delta=0.01,
    )


def test_early_stopping_is_identical_under_actor_only_and_ctde(tmp_path):
    """PO1. The two training modes stop by the SAME rule, at the SAME iteration.

    Early stopping exists to be applied to BOTH arms of the actor-only-vs-CTDE
    comparison, and the comparison's semantics are "same maximum budget + same frozen
    stopping rule + same training-population contract" -- NOT "same actual number of
    iterations". That only holds if the rule itself carries no mode-specific branch: an
    arm whose stopping rule was even slightly different would be trained under a
    different protocol while being reported under a shared one.

    Fed one shared plateau trajectory, the two runs must produce byte-identical
    early-stopping check histories and stop at the identical completed-iteration count.
    """
    restore = ["_run_one_episode", "_build_generator", "_git_provenance",
               "PPOUpdater", "CTDEUpdater"]
    schedule = [-0.5] * 20                     # a PLATEAU, identical for both arms

    results = {}
    for mode in (graph_train.TRAINING_MODE_ACTOR_ONLY, graph_train.TRAINING_MODE_CTDE):
        actor_cls, ctde_cls = _plateau_updaters(schedule)
        saved_ppo, saved_ctde = graph_train.PPOUpdater, graph_train.CTDEUpdater
        cfg = _early_stopping_cfg(tmp_path / mode, mode)
        try:
            graph_train.PPOUpdater = actor_cls
            graph_train.CTDEUpdater = ctde_cls
            summary = _run_stub_training(cfg, restore)
        finally:
            graph_train.PPOUpdater, graph_train.CTDEUpdater = saved_ppo, saved_ctde
        results[mode] = (summary, _train_records(cfg.output_dir))

    actor_summary, actor_records = results[graph_train.TRAINING_MODE_ACTOR_ONLY]
    ctde_summary, ctde_records = results[graph_train.TRAINING_MODE_CTDE]

    # Both really stopped early, at the point the state machine dictates.
    assert len(actor_records) == len(ctde_records) == 8, (
        len(actor_records), len(ctde_records))
    for summary in (actor_summary, ctde_summary):
        block = summary["early_stopping"]
        assert block["enabled"] is True and block["triggered"] is True
        assert block["termination_reason"] == graph_train.TERMINATION_REASON_PLATEAU
        assert block["stop_completed_iterations"] == 8
        assert block["planned_iterations"] == 10

    # THE PARITY: the check histories are identical, field for field.
    key = graph_train._EARLY_STOPPING_RECORD_KEY
    actor_checks = [r[key] for r in actor_records if key in r]
    ctde_checks = [r[key] for r in ctde_records if key in r]
    assert actor_checks == ctde_checks, (actor_checks, ctde_checks)
    assert [c["completed_iterations"] for c in actor_checks] == [4, 6, 8]
    assert actor_summary["early_stopping"]["checks"] == \
        ctde_summary["early_stopping"]["checks"]

    # The CTDE arm really did build a critic (so this is a parity result about two
    # different training algorithms, not about one run twice).
    assert "value_loss" in ctde_records[0]
    assert "value_loss" not in actor_records[0]


# =============================================================================
# Standalone runner (pytest is absent under nlp_env)
# =============================================================================

if __name__ == "__main__":
    import inspect
    import tempfile

    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("OK   %s" % name)
        except Exception as exc:  # noqa: BLE001 - a standalone runner reports, not raises
            failures += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
            import traceback
            traceback.print_exc()
    print("\n%d passed, %d failed (of %d)" % (len(tests) - failures, failures, len(tests)))
    sys.exit(1 if failures else 0)
