"""
Graph Observation Builder (Phase-2 RL layer)
============================================

Builds a heterogeneous graph view of a BLADE scenario for one ego agent, as an
alternative to the flat 30-feature vector produced by ``observation_builder``.
This module is **side-by-side    **: the flat builder stays the baseline; nothing
here is wired into ``train_full.py`` yet.

The graph has two node types and (currently) ONE constructed edge type:

    Node types (CANONICAL GLOBAL ORDER):
        task  nodes : global indices [0 .. k-1]   (one per Task in ``tasks``)
        agent nodes : global indices [k .. k+a-1]
                      = ego  UNION  every same-side agent assigned in ``solution``
                        (the static allocation A_init). A_init is a shared plan known to
                        every agent pre-deployment, so the ego knows the full assignment;
                        each peer node is therefore anchored by >=1 ASSIGNMENT edge by
                        construction. Currently-sensed-but-unassigned peers are NOT added:
                        peer RUNTIME state is unsensable under the no-communication
                        constraint, so peer nodes carry no features and exist only to
                        anchor the static-plan ASSIGNMENT edges.
        The ego agent is placed FIRST (``ego_index == k``); the remaining agents follow
        in deterministic id order.
        Task node index == ``task_idx`` (the stable order of ``tasks``), which keeps
        the solver's ``(task_idx, step_idx, level)`` tuples aligned with the graph.

    Edge types (over GLOBAL node indices, see :class:`EdgeType`):
        SPATIAL    (0) : RESERVED / DEFERRED. The ego's "I currently sense this target"
                         signal is no longer a constructed edge; it now lives in the
                         ``sensed`` task-feature COLUMN (``task_features[:, 5]``,
                         ego-only, recomputed from the ego's current position each build).
                         The enum member is kept (it is an IntEnum — removing it would
                         renumber ASSIGNMENT / PRECEDENCE); restoring edge construction is
                         cheap if a future design wants it back.
        ASSIGNMENT (1) : agent_node -> task_node, one per (task_idx, step_idx, level) in
                         ``solution`` (the static allocation A_init). Built over the
                         COMPLETE agent-node set, so every assignment maps to a node. This
                         is the ONLY constructed relation in the graph today.
        PRECEDENCE (2) : task_node -> task_node, one per (a, b) in precedence_relations.
                         Emitted only if precedence_relations is non-empty (current
                         scenarios produce none — expected).

Feature column layouts (every value normalized to [0, 1]):

    TASK feature columns  -> task_features[k, 6]
        [0] utility_norm     = Task.utility / 100.0                         (clipped)
        [1] dist_to_ego_norm = haversine(ego, target) / theater_scale_km    (clipped)
        [2] capable_by_ego   = 1.0 if ego Agent.has_capabilities(step.caps)  else 0.0
        [3] reachable_by_ego = 1.0 if round_trip(ego->target->return) <=
                               budget*(1 - sigma) else 0.0
        [4] probability      = step.probability (default 1.0), clipped. Degenerate
                               (==1.0) today; informative once protected / low-p
                               targets are enabled (drives the engage/skip decision).
        [5] sensed           = 1.0 if haversine(ego, target) <= detection_range_km
                               else 0.0. EGO-ONLY (the ego's own sensing); peers
                               contribute nothing. Recomputed from the ego's current
                               position every build, so a target leaving range returns to
                               0.0 automatically. Replaces the former ego->task SPATIAL
                               edge (a peer->task edge would have leaked peer runtime
                               sensing under no-communication).

    AGENT feature columns -> agent_features[a, 1]
        [0] fuel_norm        = current_fuel / max_fuel (self_features._compute_fuel_norm)
                               for the EGO row only; 0.0 for every peer row. Peers are
                               featureless on purpose: peer fuel is unsensable (a true
                               comms leak), and peer position / observation status are
                               dropped too so the no-communication property is
                               structurally auditable. Two information sources stay
                               cleanly separated: the static plan A_init reaches the graph
                               ONLY via ASSIGNMENT edges + the (featureless) peer nodes
                               that anchor them; the ego's own live sensors reach it via
                               the ego features and the task-feature columns (including the
                               ego-only ``sensed`` column). No node-type / is_ego column is
                               added here — node-typing is deferred to the (not-yet-built)
                               encoder so the builder stays a pure projection of
                               ``(world, solution)``.

Detection range
---------------
A single scenario-controlled radius (``config.detection_range_km``) gates the ego's
``sensed`` task-feature column (``sensed = 1.0`` when the target is within the radius).
This is the UNIFIED sensing = attack radius: it MUST equal the executor's
``arrival_threshold_km`` (a future orchestrator will pass one value to both), and we
deliberately do NOT use the per-aircraft radar ``range``.

Dropped ``is_done``
-------------------
An earlier ``is_done`` task column was intentionally removed: it carries no information
at any RL trigger point (a target the agent senses is by definition present, so is_done
would always be 0; a destroyed target is simply absent from the agent's view), and it
was the only feature that scanned the whole scenario — violating the no-communication
constraint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models import Agent, Location, StepKind
from ..shared_utils import clip_to_01, haversine_distance
from .self_features import _compute_fuel_norm
from ...utils.blade_utils.scenario_factory import create_agents_from_scenario, _normalize_side_color

logger = logging.getLogger(__name__)


# task_features width: utility, dist, capable, reachable, probability, sensed.
# Single source of truth for the task-feature vector width. The encoder imports
# this constant for its ``task_feat_dim`` default so the two never desync.
TASK_FEATURE_DIM = 6


# =============================================================================
# Edge types
# =============================================================================

class EdgeType(IntEnum):
    """Edge-type codes stored in ``GraphObservation.edge_type``.

    IntEnum members compare and serialize as plain ints, so ``edge_type`` stays an
    int64 array; cast with ``int(...)`` when appending to keep numpy dtypes clean.
    """

    SPATIAL = 0     # RESERVED / DEFERRED — no edge built; sensing moved to the
                    # ``sensed`` task-feature column (cheap to restore; same deferred
                    # status as PRECEDENCE). Kept as a member because this is an IntEnum:
                    # deleting it would renumber ASSIGNMENT / PRECEDENCE.
    ASSIGNMENT = 1
    PRECEDENCE = 2


# =============================================================================
# Config (local on purpose)
# =============================================================================

@dataclass
class GraphObservationConfig:
    """Tunables for the graph builder.

    Defined here rather than extending ``ObservationConfig`` because the flat
    config module is part of the frozen baseline path and must not be modified
    while the graph builder lives side-by-side.

    Attributes:
        detection_range_km: Single, scenario-controlled radius gating the ego's
            ``sensed`` task-feature column (``sensed = 1.0`` when the target is within
            the radius). This is the UNIFIED sensing = attack radius and MUST be kept
            equal to the executor's ``arrival_threshold_km`` (a future orchestrator will
            pass one value to both). We intentionally do not use per-aircraft radar range.
        theater_scale_km: Distance normalizer for the ``dist_to_ego_norm`` columns
            (task and agent). Kept separate from ``detection_range_km``.
        sigma: Risk margin on the movement budget for reachability. The cap is
            ``budget * (1 - sigma)``, matching the solver/audit convention
            (validation audit uses ``cap = budget * (1 - RISK)``). Default 0.0
            (Phase-1 ``risk_factor = 0``); Phase-2 may set sigma > 0.
        max_sim_ticks: Tick cap used to normalize ``current_time`` into ``time_norm``
            (matches the ScenarioGenerator / MAX_SIM_TICKS default).
    """

    detection_range_km: float = 50.0
    theater_scale_km: float = 150.0
    sigma: float = 0.0
    max_sim_ticks: int = 14400


# =============================================================================
# GraphObservation dataclass
# =============================================================================

@dataclass
class GraphObservation:
    """Heterogeneous graph observation for one ego agent.

    See the module docstring for the canonical node ordering and feature layouts.
    Counts are variable (no padding to a fixed ``k`` or ``a``); any batching /
    padding happens later at the buffer/batch level.
    """

    task_features: np.ndarray          # [k, TASK_FEATURE_DIM] float32, all values in [0, 1]
    agent_features: np.ndarray         # [a, 1] float32: [0] fuel_norm (ego real, peers 0.0)
    ego_index: int                     # global node index of the ego agent (== k)
    edge_index: np.ndarray             # [2, E] int   COO over GLOBAL node indices
    edge_type: np.ndarray              # [E]    int   values from EdgeType
    task_target_ids: List[str]         # length k: task-node -> target_id
    agent_ids: List[str]               # length a: agent-node -> agent/aircraft id
    agent_id: str                      # the ego agent's id
    current_time: int                  # raw simulation tick
    time_norm: float                   # current_time / max_sim_ticks, clipped to [0, 1]


# =============================================================================
# Helpers (mirror the proven flat-builder logic)
# =============================================================================

def _attack_step(task: Any):
    """Return the task's ATTACK step (its target is the task-node target).

    Tasks built by ``scenario_factory._make_task`` carry a single ATTACK step.
    We still scan for the first ``StepKind.ATTACK`` to be robust, falling back to
    ``steps[0]`` if (unexpectedly) no ATTACK step exists.
    """
    steps = getattr(task, "steps", None) or []
    for step in steps:
        if getattr(step, "step_kind", None) == StepKind.ATTACK:
            return step
    return steps[0] if steps else None


def _find_match_agent(agents_by_side: Dict[str, List[Agent]], agent_id: str) -> Optional[Agent]:
    """Locate the MATCH-AOU Agent with ``id == agent_id`` across all sides."""
    for agents in agents_by_side.values():
        for agent in agents:
            if str(getattr(agent, "id", "")) == str(agent_id):
                return agent
    return None


def _reachable_by_ego(ego_agent: Agent, target_loc: Location, sigma: float) -> bool:
    """True if the ego can round-trip to ``target_loc`` within its budget.

    Reuses the solver's ``round_trip_cost`` — the single source of truth shared by
    the movement-budget constraint and the validation audit — so reachability here
    matches the solver by construction. Cap is ``budget * (1 - sigma)`` (the audit
    convention; sigma defaults to 0 to mirror Phase-1 ``risk_factor = 0``).

    Lazy import: ``round_trip_cost`` lives in the solver module (which imports
    pyomo at load), so we defer the import to keep the observation layer light for
    callers that never compute reachability.

    Fail-safe: on any cost-computation error we mark the target unreachable, but we
    log a warning (rather than swallow it) so a genuinely reachable target wrongly
    marked unreachable is visible. The round-trip model itself is unchanged here.
    """
    from ...solvers import round_trip_cost  # lazy: avoid forcing pyomo at module load

    try:
        cost = round_trip_cost(ego_agent, target_loc)
    except Exception:
        logger.warning(
            "round_trip_cost failed for agent %s -> target %s; marking unreachable",
            getattr(ego_agent, "id", "?"),
            target_loc,
            exc_info=True,
        )
        return False

    budget = float(getattr(ego_agent, "budget", 0.0) or 0.0)
    cap = budget * (1.0 - float(sigma))
    return cost <= cap


# =============================================================================
# Main entry point
# =============================================================================

def build_graph_observation(
    scenario: Any,                                   # BLADE Scenario observation
    agent_id: str,
    current_plan=None,                               # ego's [(task_idx, step_idx, level), ...]
    current_time: int = 0,
    tasks: Optional[List[Any]] = None,               # stable Task set (task_idx == node index)
    solution: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
    precedence_relations: Optional[List[Tuple[int, int]]] = None,
    config: Optional[GraphObservationConfig] = None,
) -> GraphObservation:
    """Build a :class:`GraphObservation` for ``agent_id``.

    Inputs mirror ``observation_builder.build_observation_vector`` (scenario,
    agent_id, current_plan, current_time, tasks, solution) plus
    ``precedence_relations`` (task_idx pairs, may be None/empty).

    Args:
        scenario: BLADE Scenario observation (must expose ``get_aircraft``).
        agent_id: ego aircraft id (must be airborne — present in scenario.aircraft).
        current_plan: ego's own assignment tuples. Only used as a fallback source
            for ASSIGNMENT edges when ``solution`` is None (then treated as
            ``{agent_id: current_plan}``).
        current_time: simulation tick (stored raw; also normalized into ``time_norm``).
        tasks: the stable Task list. Task nodes are one-per-Task in this order, so
            ``task_idx`` from the solution indexes directly into the graph.
        solution: full allocation ``{agent_id: [(task_idx, step_idx, level), ...]}``.
        precedence_relations: list of (a, b) task_idx pairs (a precedes b).
        config: :class:`GraphObservationConfig` (defaults used if None).

    Returns:
        GraphObservation with variable ``k`` task nodes and ``a`` agent nodes.

    Raises:
        ValueError: if the ego aircraft is not found (not airborne) or lacks a side.
    """
    if config is None:
        config = GraphObservationConfig()
    if tasks is None:
        tasks = []
    if precedence_relations is None:
        precedence_relations = []
    if solution is None:
        # Degrade gracefully: a lone ego plan still yields ASSIGNMENT edges.
        solution = {str(agent_id): list(current_plan)} if current_plan else {}

    if not agent_id:
        raise ValueError("agent_id cannot be empty")

    if not hasattr(scenario, "get_aircraft"):
        raise ValueError("Scenario does not have get_aircraft method")

    ego_ac = scenario.get_aircraft(agent_id)
    if ego_ac is None:
        raise ValueError(
            f"Aircraft {agent_id} not found in scenario.aircraft "
            f"(graph observation requires the ego to be airborne)"
        )

    ego_side = getattr(ego_ac, "side_id", None)
    if not ego_side:
        raise ValueError(f"Aircraft {agent_id} has no side_id")

    # --- MATCH-AOU agents (single source of truth for capabilities/move_cost/budget) ---
    agents_by_side = create_agents_from_scenario(scenario)
    ego_agent = _find_match_agent(agents_by_side, agent_id)
    if ego_agent is None:
        raise ValueError(
            f"Could not build a MATCH-AOU Agent for ego {agent_id}; "
            f"cannot compute capability/reachability features"
        )

    ego_pos = (ego_ac.latitude, ego_ac.longitude)
    detection_km = config.detection_range_km  # gates the ego-only `sensed` column
    theater = config.theater_scale_km

    # =========================================================================
    # Agent nodes = ego  UNION  every same-side agent assigned in A_init (`solution`).
    #
    # A_init is a shared, globally-consistent plan known to every agent BEFORE
    # deployment, so the ego knows the full allocation — which agent is assigned to
    # which target. We give every assigned same-side agent a node so its ASSIGNMENT
    # edges exist. We deliberately do NOT add currently-sensed-but-unassigned peers:
    # a peer's RUNTIME state (fuel / position / being observed) is unsensable under the
    # no-communication constraint, so peer nodes are featureless and exist only to anchor
    # the static-plan ASSIGNMENT edges. As a result every peer node has >=1 ASSIGNMENT
    # edge by construction.
    # =========================================================================
    ego_side_color = _normalize_side_color(getattr(ego_agent, "side_color", None))

    # Same-side agents holding at least one assignment in the static allocation. Side is
    # verified against the ego when the id resolves to a MATCH-AOU Agent; an unresolvable
    # id (assigned but no live aircraft — lost / not airborne) is still kept, since the
    # solver is only ever given same-side agents.
    assigned_ids = set()
    for raw_aid in solution.keys():
        aid = str(raw_aid)
        if aid == str(agent_id) or not solution.get(raw_aid):
            continue
        match_agent = _find_match_agent(agents_by_side, aid)
        if match_agent is not None and \
                _normalize_side_color(getattr(match_agent, "side_color", None)) != ego_side_color:
            continue
        assigned_ids.add(aid)

    # Ego FIRST; remaining assigned peers in deterministic id order.
    node_ids = [str(agent_id)] + sorted(assigned_ids)

    k = len(tasks)
    a = len(node_ids)
    ego_index = k  # ego is the first agent node
    agent_ids = node_ids

    # agent_features [a, 1]: [0] fuel_norm. REAL for the ego (its own sensor); 0.0 for
    # every peer row — peers are featureless (no peer runtime state under no-communication).
    # The ego is the only node with a live position now, so the `sensed` column below
    # uses ego_pos directly; peers contribute no sensing.
    agent_features = np.zeros((a, 1), dtype=np.float32)
    agent_features[0, 0] = clip_to_01(_compute_fuel_norm(ego_ac))

    # =========================================================================
    # Task nodes: one per Task in `tasks` (stable, NOT restricted to in-range).
    # =========================================================================
    task_features = np.zeros((k, TASK_FEATURE_DIM), dtype=np.float32)
    task_target_ids: List[str] = []
    task_locs: List[Optional[Location]] = []  # cached for edge geometry

    for j, task in enumerate(tasks):
        step = _attack_step(task)
        target_id = getattr(step, "target_id", None) if step is not None else None
        target_loc = getattr(step, "location", None) if step is not None else None

        task_target_ids.append(str(target_id) if target_id is not None else "")
        task_locs.append(target_loc)

        # [0] utility
        utility_norm = clip_to_01(float(getattr(task, "utility", 0.0)) / 100.0)

        # ego->target haversine, computed ONCE and reused for [1] dist and [5] sensed.
        if target_loc is not None:
            raw_km = haversine_distance(
                ego_pos, (target_loc.latitude, target_loc.longitude)
            )
            # [1] distance to ego (normalized by theater scale, NOT the detection range)
            dist_norm = clip_to_01(raw_km / theater)
            # [5] sensed: ego-only, within the unified sensing radius (recomputed every
            # build from the ego's current position).
            sensed = 1.0 if raw_km <= detection_km else 0.0
        else:
            dist_norm = 0.0
            sensed = 0.0  # None location -> leave un-sensed

        # [2] capable_by_ego: name-based capability match (mirrors the solver)
        if step is not None and ego_agent.has_capabilities(getattr(step, "capabilities", []) or []):
            capable = 1.0
        else:
            capable = 0.0

        # [3] reachable_by_ego: round-trip fuel within budget*(1 - sigma)
        if step is not None and target_loc is not None and _reachable_by_ego(
            ego_agent, target_loc, config.sigma
        ):
            reachable = 1.0
        else:
            reachable = 0.0

        # [4] probability: ATTACK success probability (defensive default 1.0).
        # Degenerate today (always 1.0); informative once low-p targets exist.
        raw_p = getattr(step, "probability", 1.0) if step is not None else 1.0
        if raw_p is None:
            raw_p = 1.0
        probability = clip_to_01(float(raw_p))

        task_features[j, 0] = utility_norm
        task_features[j, 1] = dist_norm
        task_features[j, 2] = capable
        task_features[j, 3] = reachable
        task_features[j, 4] = probability
        task_features[j, 5] = sensed

    # =========================================================================
    # Edges (COO over global indices). Build python lists, then arrays.
    # =========================================================================
    src: List[int] = []
    dst: List[int] = []
    etype: List[int] = []

    # SPATIAL edge construction was REMOVED: the ego's sensing now lives in the
    # `sensed` task-feature column (task_features[:, 5]), so ASSIGNMENT is the only
    # constructed relation here (plus optional PRECEDENCE). See the EdgeType enum.

    # ASSIGNMENT: agent node -> task node for each assignment tuple in the static
    # allocation. The node set now covers every assigned same-side agent, so every
    # assignment maps to an existing node (the task_idx range check still guards).
    for i, aid in enumerate(node_ids):
        g = k + i
        for assignment in solution.get(aid, []) or []:
            task_idx = int(assignment[0])
            if 0 <= task_idx < k:
                src.append(g)
                dst.append(task_idx)
                etype.append(int(EdgeType.ASSIGNMENT))

    # PRECEDENCE: task node -> task node for each valid (a, b) pair.
    for pair in precedence_relations:
        a_idx, b_idx = int(pair[0]), int(pair[1])
        if 0 <= a_idx < k and 0 <= b_idx < k:
            src.append(a_idx)
            dst.append(b_idx)
            etype.append(int(EdgeType.PRECEDENCE))

    if src:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_type = np.array(etype, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)

    # Normalized time (raw tick is kept too).
    max_ticks = float(getattr(config, "max_sim_ticks", 0) or 0)
    time_norm = clip_to_01(float(current_time) / max_ticks) if max_ticks > 0 else 0.0

    return GraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=int(ego_index),
        edge_index=edge_index,
        edge_type=edge_type,
        task_target_ids=task_target_ids,
        agent_ids=agent_ids,
        agent_id=str(agent_id),
        current_time=int(current_time),
        time_norm=float(time_norm),
    )


# =============================================================================
# Self-test
# =============================================================================

def _selftest() -> None:
    """Generate one scenario, launch + solve, build a graph obs, print a summary.

    Run under nlp_env (needs bonmin) from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.observation.graph_builder
    """
    from pathlib import Path
    import tempfile

    import gymnasium
    from blade.Game import Game
    from blade.Scenario import Scenario
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # match train_full's deliberate override

    from match_aou.solvers import MatchAou
    from match_aou.utils.scheduling_utils import post_solve_filter_and_level
    from match_aou.utils.blade_utils.scenario_factory import (
        generate_all_enemy_tasks,
        create_agents_from_scenario as _make_agents,
        _normalize_side_color,
    )
    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig,
    )

    repo_root = Path(__file__).resolve().parents[4]
    base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graphobs_selftest_")
    max_sim_ticks = 14400

    # --- Generate one scenario variation (RED airbases only, no SAMs) ---
    gen = ScenarioGenerator(
        base_scenario_path=str(base_scenario),
        output_dir=out_dir,
        max_sim_ticks=max_sim_ticks,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    cfg = VariationConfig(
        include_sams=False,
        num_red_airbases=(3, 3),
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        seed=0,
    )
    scenario_path = str(gen.generate(episode=0, config=cfg))

    # --- Load into BLADE and reset ---
    game = Game(current_scenario=Scenario(), record_every_seconds=10,
                recording_export_path=out_dir)
    with open(scenario_path, "r", encoding="utf-8") as f:
        game.load_scenario(f.read())
    env = gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=max_sim_ticks)
    obs, _info = env.reset()

    # --- Launch all blue aircraft so the ego is airborne ---
    for _ in range(5):
        obs, _, _, _, _ = env.step("")
    for base in getattr(obs, "airbases", []) or []:
        if _normalize_side_color(getattr(base, "side_color", "")) != "blue":
            continue
        for _ac in list(getattr(base, "aircraft", []) or []):
            obs, _, _, _, _ = env.step(f"launch_aircraft_from_airbase('{base.id}')")
    for _ in range(10):
        obs, _, _, _, _ = env.step("")

    blue_airborne = [
        ac for ac in getattr(obs, "aircraft", []) or []
        if _normalize_side_color(getattr(ac, "side_color", "")) == "blue"
    ]
    if not blue_airborne:
        raise RuntimeError("Self-test: no blue aircraft airborne after launch")
    ego_id = str(blue_airborne[0].id)

    # Real elapsed tick (exercises time normalization with genuine data).
    elapsed = int(getattr(obs, "current_time", 0)) - int(getattr(obs, "start_time", 0))
    elapsed = max(0, elapsed)

    # --- Build tasks + solve MATCH-AOU on the airborne observation ---
    tasks = generate_all_enemy_tasks(obs, attacking_side_color="blue")
    blue_agents = _make_agents(obs).get("blue", [])
    model = MatchAou(agents=blue_agents, tasks=tasks, precedence_relations=[], risk_factor=0.0)
    raw_solution, _results, unselected = model.solve(solver_name="bonmin")
    artifacts = post_solve_filter_and_level(
        tasks=tasks, solution=raw_solution,
        precedence_relations=[], unselected_tasks=unselected,
    )
    tasks_g, solution_g = artifacts.tasks, artifacts.solution

    # --- Disperse the ego from the launch cluster ---
    # Fly the ego toward a target for a short while. This used to exercise the
    # is_observed masking path; peer features are gone now (peers are featureless), so
    # the fly-out is harmless and no longer load-bearing — kept only to move the ego off
    # the launch cluster and give the sensed/dist geometry some genuine separation.
    far = _attack_step(tasks_g[0]).location if tasks_g else None
    if far is not None:
        obs, _, _, _, _ = env.step(
            f"move_aircraft('{ego_id}', [[{far.latitude}, {far.longitude}]])"
        )
    for _ in range(60):
        obs, _, _, _, _ = env.step("")
    elapsed = max(0, int(getattr(obs, "current_time", 0)) - int(getattr(obs, "start_time", 0)))

    # --- Choose a detection radius that EXERCISES both sensed and unsensed ---
    # Recompute the ego->target haversine for every task (same helpers the builder uses:
    # obs.get_aircraft(ego_id) for ego_pos, _attack_step(task).location for each target),
    # then pick the midpoint of [min, max] so the nearest target is sensed and the
    # farthest is not. A flat detection_range_km=1.0 made every target unsensed and the
    # binary-column check passed vacuously; this forces a genuine mix.
    ego_ac_now = obs.get_aircraft(ego_id)
    ego_pos_now = (ego_ac_now.latitude, ego_ac_now.longitude)
    dists = []
    for task in tasks_g:
        tloc = _attack_step(task).location
        dists.append(
            haversine_distance(ego_pos_now, (tloc.latitude, tloc.longitude))
        )
    radius = (min(dists) + max(dists)) / 2.0  # nearest sensed, farthest not (when min<max)

    # --- Build the graph observation ---
    # Assigned peers are nodes regardless (anchored by ASSIGNMENT edges) and carry no
    # features, so the radius drives only the ego's `sensed` column, never peer behavior.
    config = GraphObservationConfig(max_sim_ticks=max_sim_ticks, detection_range_km=radius)
    go = build_graph_observation(
        scenario=obs,
        agent_id=ego_id,
        current_plan=solution_g.get(ego_id),
        current_time=elapsed,
        tasks=tasks_g,
        solution=solution_g,
        precedence_relations=[],
        config=config,
    )

    n_spatial = int((go.edge_type == int(EdgeType.SPATIAL)).sum())
    n_assign = int((go.edge_type == int(EdgeType.ASSIGNMENT)).sum())
    n_prec = int((go.edge_type == int(EdgeType.PRECEDENCE)).sum())

    print("=" * 64)
    print("GraphObservation self-test")
    print("=" * 64)
    print(f"ego_id                 : {ego_id}")
    print(f"task_features.shape    : {go.task_features.shape}  (k tasks x 6)")
    print(f"agent_features.shape   : {go.agent_features.shape}  (a agents x 1)")
    print(f"ego_index              : {go.ego_index}  (== k, ego is first agent node)")
    print(f"current_time (raw tick): {go.current_time}")
    print(f"time_norm              : {go.time_norm:.6f}  (= current_time / {max_sim_ticks})")
    print(f"edges total            : {go.edge_index.shape[1]}")
    print(f"  SPATIAL    (type 0)  : {n_spatial}")
    print(f"  ASSIGNMENT (type 1)  : {n_assign}")
    print(f"  PRECEDENCE (type 2)  : {n_prec}")
    print(f"task_target_ids (k={len(go.task_target_ids)}): "
          f"{[t[:8] for t in go.task_target_ids]}")
    print(f"agent_ids (a={len(go.agent_ids)})         : "
          f"{[a[:8] for a in go.agent_ids]}")
    print("-" * 64)
    print("TASK feature index map : "
          "[0]utility [1]dist_to_ego [2]capable [3]reachable [4]probability [5]sensed")
    print("AGENT feature index map: [0]fuel_norm  (ego real; peers featureless = 0.0)")
    print("-" * 64)
    print("task_features:")
    print(np.array2string(go.task_features, precision=3, suppress_small=True))
    print("agent_features:")
    print(np.array2string(go.agent_features, precision=3, suppress_small=True))
    print("per-agent-node (id | fuel_norm):")
    for i, aid in enumerate(go.agent_ids):
        tag = "EGO" if i == 0 else "peer (featureless, A_init anchor)"
        print(f"    {aid[:8]}  fuel={go.agent_features[i, 0]:.3f}  [{tag}]")
    print(f"probability column     : {go.task_features[:, 4].tolist()}  "
          f"(all 1.0 expected — probability is degenerate until low-p targets exist)")

    # Lightweight invariant checks.
    assert go.task_features.shape == (len(tasks_g), TASK_FEATURE_DIM)
    assert go.agent_features.shape == (len(go.agent_ids), 1)
    assert go.ego_index == len(tasks_g)
    assert go.edge_index.shape[0] == 2
    assert go.edge_index.shape[1] == go.edge_type.shape[0]
    assert float(go.task_features.min()) >= 0.0 and float(go.task_features.max()) <= 1.0
    assert float(go.agent_features.min()) >= 0.0 and float(go.agent_features.max()) <= 1.0
    assert 0.0 <= go.time_norm <= 1.0
    # Ego is the first agent node; its fuel_norm is a real, normalized value.
    assert 0.0 <= float(go.agent_features[0, 0]) <= 1.0
    # Peers are featureless: every NON-ego row is exactly 0.0 (no peer runtime state).
    for i in range(1, len(go.agent_ids)):
        assert float(go.agent_features[i, 0]) == 0.0
    # Agent-node set == ego UNION assigned same-side peers; no sensed-but-unassigned peer
    # leaked in. (All agents are blue here, so the builder's side filter is a no-op.)
    expected_nodes = {ego_id} | {
        str(sid) for sid, plan in solution_g.items() if str(sid) != ego_id and plan
    }
    assert set(go.agent_ids) == expected_nodes
    # Every assigned agent in A_init must be a node: no assignment is lost.
    assert all(str(sid) in go.agent_ids for sid in solution_g.keys())
    # SPATIAL edge construction was removed: sensing now lives in the `sensed` column.
    assert n_spatial == 0
    # The `sensed` column is strictly binary (0.0 / 1.0), ego-only.
    sensed_col = go.task_features[:, 5]
    assert set(np.unique(sensed_col).tolist()).issubset({0.0, 1.0})
    # Per-target geometric cross-check: sensed[j] reflects haversine(ego, target) <= radius.
    for j, d in enumerate(dists):
        expected = 1.0 if d <= radius else 0.0
        assert float(sensed_col[j]) == expected, (
            f"sensed[{j}]={sensed_col[j]} but dist={d:.3f} radius={radius:.3f}"
        )
    # Non-trivial coverage: with a real spread the midpoint radius forces a mix. Guarded
    # behind min<max so a degenerate equal-distance scenario does not make the test flaky.
    if min(dists) < max(dists):
        assert float(sensed_col.max()) == 1.0  # at least one sensed
        assert float(sensed_col.min()) == 0.0  # at least one unsensed
    # ASSIGNMENT edges cover every assignment tuple in the (filtered) solution.
    n_assignment_tuples = sum(len(v) for v in solution_g.values())
    assert n_assign == n_assignment_tuples
    print("\nAll invariants passed.")


if __name__ == "__main__":
    _selftest()
