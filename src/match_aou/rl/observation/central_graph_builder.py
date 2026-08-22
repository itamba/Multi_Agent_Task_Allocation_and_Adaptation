"""central_graph_builder.py -- the TRAINING-ONLY central state for the Phase-B critic.

WHAT THIS IS
------------
The centralized half of CTDE (centralized training, decentralized execution). It
projects the CURRENT GLOBAL PHYSICAL / EXECUTION state of the world into a
:class:`CentralGraphObservation` that an INDEPENDENT
:class:`~match_aou.rl.agent.graph_encoder.GraphEncoder` + value head consume to
estimate ``V(s)``.

WHAT THIS IS NOT
----------------
It is **NOT** the actor's observation, it is **NOT** the actor's observation with one
privileged ego, and nothing it produces may ever reach the acting path. The actor
keeps its private, no-communication
:class:`~match_aou.rl.observation.graph_builder.GraphObservation` EXACTLY as the
Phase-A contract left it: featureless peers, ego-only sensing, ASSIGNMENT edges from
the ego's own belief. This module adds a SECOND, disjoint view used only while
training.

That separation is structural, not a convention:

  * :class:`CentralGraphObservation` is a DISTINCT type -- it is not a
    ``GraphObservation`` and is not a subclass of one, so ``isinstance`` separates
    them and a central state can never be mistaken for an actor state;
  * it deliberately carries NO ``agent_id`` field. The actor's observation is *for*
    an ego; the central state has no ego at all, and omitting the field means the
    actor's ego-keyed code cannot even accept one of these by accident;
  * ``ego_index`` is :data:`NO_EGO_INDEX` (``-1``), which the shared encoder already
    handles: its role assignment marks a node EGO only for ``0 <= ego_index < N``, so
    with ``-1`` every agent node keeps the same (PEER) role and the graph is SYMMETRIC
    over live agents. No encoder change was needed, and none was made;
  * ``build_action_mask`` / ``ActionHead`` are never reachable from here -- a central
    state does not carry the columns the mask reads, so feeding one to the actor path
    fails loudly rather than silently training on privileged information.

THE GRAPH
---------
Node types, in the encoder's canonical order (tasks first, then agents)::

    task  nodes : [0 .. k-1]    one per LIVE enemy target physically in the world
    agent nodes : [k .. k+a-1]  one per originally-scheduled same-side agent that is
                                physically alive

PRESENCE IS LIVENESS. There is no dead/alive flag and no known/hidden label: a
destroyed target simply has no node, and a destroyed aircraft simply has no node. Both
facts are read from the live BLADE observation and nothing else:

  * a TARGET is live iff the frozen engine still lists it. ``weapon_endgame`` removes a
    destroyed facility / airbase / ship from ``scenario.facilities`` / ``.airbases`` /
    ``.ships``, which are exactly the collections ``generate_all_enemy_tasks`` /
    ``iter_enemy_targets`` enumerate -- the SAME current-world extraction episode setup
    and the executor's sensing already use. So "physically destroyed" and "absent from
    the enumeration" are the same fact, with no proxy and no ambiguity;
  * an AGENT is live iff it is airborne (present in ``scenario.aircraft``) OR landed
    (present in some ``airbase.aircraft`` inventory). Absent from BOTH means the engine
    removed it, i.e. dead -- the same three-way classification
    ``GraphPlanExecutor._physical_state`` makes. **RTB ISSUANCE IS NOT DEATH**: an ego
    that was ordered home, is flying home, or has landed is still a live entity here and
    keeps its node, even though the actor's Phase-1 loop stops processing it.

THE CRITIC SEES THE PRESENT, NEVER THE FUTURE AND NEVER THE ORACLE
-------------------------------------------------------------------
Everything below is CURRENT PHYSICAL TRUTH. The critic is deliberately NOT given
``oracle_solution`` / ``oracle_tasks`` / ``U_oracle`` / any reward component, the
episode seed, the fuel-damage severity or scheduled condition label, the known-vs-hidden
split, future RNG, or any future outcome. Privileged means "all agents, right now" --
it does not mean "the answer".

In particular the task-node inventory is the RAW LIVE WORLD, not an allocation:
``solve_and_normalize`` is allocated-only by contract, so a target the solver never
selected is absent from ``oracle_tasks`` while being physically present, sensible and
attackable. Reading an allocation as a world inventory is the exact defect the
roster-integrity correction closed, and this module does not repeat it.

FEATURES (all normalized to [0, 1], all read off the live world)
----------------------------------------------------------------
    TASK  features -> ``task_features[k, 2]``
        [0] utility_norm  = Task.utility / 100.0            (the actor's normalization)
        [1] probability   = ATTACK step probability          (1.0 in the current cell)

    AGENT features -> ``agent_features[a, 1]``
        [0] fuel_norm     = current_fuel / max_fuel for THAT agent

    UNLIKE THE ACTOR GRAPH, EVERY LIVE AGENT CARRIES ITS OWN REAL FUEL. There is no
    "one fully-featured ego + featureless peers" here -- that asymmetry exists in the
    actor graph precisely because peer fuel is unsensable under no-communication, and
    the whole point of a centralized critic is that TRAINING may read it.

EDGES
-----
One :data:`~match_aou.rl.observation.graph_builder.EdgeType.SPATIAL` edge per
(live agent -> live target) pair -- the complete bipartite relation. SPATIAL is
RESERVED / unused in the actor graph, so reusing its code here changes nothing the
actor builds. The encoder symmetrizes edges internally, so the single direction is
enough.

    ``edge_attr[E, 5]`` = ``[distance_norm, capable, reachable, sensed, assigned]``

      distance_norm : current physical agent->target great-circle distance over the
                      actor's own ``theater_scale_km``, clipped.
      capable       : the existing capability predicate for THAT agent and THAT
                      target's ATTACK step.
      reachable     : the existing ``graph_builder._reachable_by_ego`` round-trip model
                      -- IMPORTED, not reimplemented -- applied per live agent from the
                      current world state. No new reachability model is introduced here.
      sensed        : 1.0 iff that agent is currently within the ONE unified detection
                      radius of that target. This is privileged ALL-AGENT sensing and is
                      exactly the kind of thing only the critic may see.
      assigned      : CURRENT per-agent execution-plan membership, read from
                      ``GraphPlanExecutor.plans[agent]`` resolved against
                      ``GraphPlanExecutor.tasks[agent]`` -- never from
                      ``oracle_solution``, never from one ego's private belief, and
                      never from t=0 ``A_init`` after runtime adaptation. That matters
                      for same-tick sequential wakes: an earlier ego's action calls
                      ``executor.resync``, and a later central sample in the SAME tick is
                      allowed to observe that causally updated global plan state.

PURITY / IMPORTS
----------------
No torch. No solver at import time (``_reachable_by_ego`` keeps its own lazy pyomo
import). It imports the same two live-world helpers the rest of the graph path already
uses (``graph_builder`` for the actor's normalization conventions, ``scenario_factory``
for the current-world target enumeration), so it adds NOTHING to the tick loop's
existing import closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from ..shared_utils import clip_to_01, haversine_distance
from .graph_builder import (
    EdgeType,
    GraphObservationConfig,
    _attack_step,
    _compute_fuel_norm,
    _reachable_by_ego,
)
from ...utils.blade_utils.scenario_factory import (
    create_agents_from_scenario,
    generate_all_enemy_tasks,
    _normalize_side_color,
)


# Central task features: [utility_norm, probability]. Deliberately NOT the actor's six
# columns -- the critic has no ego, so ego-relative distance / capability / reachability
# / sensing are PAIRWISE facts and live on the edges instead.
CENTRAL_TASK_FEATURE_DIM = 2

# Central agent features: [fuel_norm], real for EVERY live agent.
CENTRAL_AGENT_FEATURE_DIM = 1

# Central edge features: [distance_norm, capable, reachable, sensed, assigned].
CENTRAL_EDGE_ATTR_DIM = 5

# The encoder marks a node EGO only when ``0 <= ego_index < N``; -1 therefore leaves
# every agent node in the SAME role, which is what makes the critic graph symmetric
# over live agents. It is a sentinel, not an index.
NO_EGO_INDEX = -1

# The one relation code the central graph uses. SPATIAL is RESERVED / unused in the
# actor graph, so borrowing it here cannot change anything the actor builds.
CENTRAL_EDGE_TYPE = int(EdgeType.SPATIAL)


# =============================================================================
# The central observation
# =============================================================================

@dataclass
class CentralGraphObservation:
    """The global training-only state at ONE decision point.

    Structurally distinct from the actor's ``GraphObservation`` (see the module
    docstring): different type, different feature widths, no ``agent_id``, and
    ``ego_index == NO_EGO_INDEX``. It exposes exactly the attribute names
    :class:`~match_aou.rl.agent.graph_encoder.GraphEncoder` reads, which is why the
    SAME encoder class can be instantiated a second time for the critic with no change
    to the encoder itself.

    Attributes:
        task_features: ``[k, CENTRAL_TASK_FEATURE_DIM]`` float32.
        agent_features: ``[a, CENTRAL_AGENT_FEATURE_DIM]`` float32.
        ego_index: always :data:`NO_EGO_INDEX` -- there is no distinguished ego.
        edge_index: ``[2, E]`` int64 COO over GLOBAL node indices (agent -> task).
        edge_type: ``[E]`` int64, every entry :data:`CENTRAL_EDGE_TYPE`.
        edge_attr: ``[E, CENTRAL_EDGE_ATTR_DIM]`` float32, aligned with ``edge_index``.
        task_target_ids: length ``k`` -- provenance, so a test can name what it saw.
        agent_ids: length ``a`` -- the LIVE agents, in the caller's scheduled order.
        current_time: raw simulation tick.
        time_norm: ``current_time / max_sim_ticks``, clipped -- the actor's convention.
    """

    task_features: np.ndarray
    agent_features: np.ndarray
    ego_index: int
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_attr: np.ndarray
    task_target_ids: List[str]
    agent_ids: List[str]
    current_time: int
    time_norm: float

    @property
    def n_tasks(self) -> int:
        """Number of LIVE target nodes."""
        return int(self.task_features.shape[0])

    @property
    def n_agents(self) -> int:
        """Number of LIVE agent nodes."""
        return int(self.agent_features.shape[0])


# =============================================================================
# Live-world lookups (read the observation; decide nothing)
# =============================================================================

def _airborne_aircraft(scenario: Any, agent_id: str) -> Optional[Any]:
    """The agent's live airframe from ``scenario.aircraft``, or None."""
    for ac in getattr(scenario, "aircraft", None) or []:
        if str(getattr(ac, "id", "")) == str(agent_id):
            return ac
    return None


def _landed_aircraft(scenario: Any, agent_id: str) -> Optional[Any]:
    """The agent's live airframe from an ``airbase.aircraft`` inventory, or None."""
    for base in getattr(scenario, "airbases", None) or []:
        for ac in getattr(base, "aircraft", None) or []:
            if str(getattr(ac, "id", "")) == str(agent_id):
                return ac
    return None


def live_aircraft(scenario: Any, agent_id: str) -> Optional[Any]:
    """The agent's airframe wherever the engine currently keeps it, or None if dead.

    The frozen engine keeps every aircraft it still has in exactly ONE home: flying in
    ``scenario.aircraft``, or landed inside an ``airbase.aircraft`` inventory
    (``Game.land_aicraft`` appends there, then removes it from the air). Absent from
    both means ``Game.remove_aircraft`` dropped it at ``current_fuel <= 0`` -- i.e.
    dead. This is the same three-way fact ``GraphPlanExecutor._physical_state``
    classifies; here it is collapsed to "the live object, or None", because the critic
    needs the OBJECT (for its real fuel) and treats airborne and landed identically.

    RTB issuance is NOT death and NOT absence: an ego ordered home is still airborne
    until the engine lands it, and once landed it is still live.
    """
    ac = _airborne_aircraft(scenario, agent_id)
    if ac is not None:
        return ac
    return _landed_aircraft(scenario, agent_id)


def plan_target_ids(executor: Any, agent_id: str) -> Set[str]:
    """Target ids in ``agent_id``'s CURRENT executor plan slice.

    Reads ``executor.plans[agent_id]`` and resolves each ``(task_idx, step_idx, level)``
    against ``executor.tasks[agent_id]`` -- the ego's OWN task list -- with the same
    bounds semantics ``GraphPlanExecutor._resolve_step`` uses: an out-of-range task or
    step index resolves to nothing and is skipped rather than raising.

    This is the ONE place the central graph answers "is this agent currently assigned to
    this target". It is a MIRROR of the executor's own resolution kept out of that
    frozen module's import closure, and its equivalence to ``_resolve_step`` is
    TEST-ENFORCED against a real
    :class:`~match_aou.utils.blade_utils.blade_graph_executor.GraphPlanExecutor` -- the
    same pattern ``graph_fuel_damage.rtb_command_for`` uses for the executor's RTB
    emission site.

    It is PLAN MEMBERSHIP, not eligibility: no ``done`` filter and no level gating, both
    of which are execution-ordering concerns rather than "who is on the hook for what".

    Returns:
        The set of target-id strings this agent's current plan resolves to (possibly
        empty -- an ego whose mission was aborted has an empty slice, which is the
        point).
    """
    plans = getattr(executor, "plans", None) or {}
    tasks_by_ego = getattr(executor, "tasks", None) or {}
    ego_key = str(agent_id)
    ego_tasks = tasks_by_ego.get(ego_key) or []

    out: Set[str] = set()
    for assignment in plans.get(ego_key) or []:
        try:
            t_idx = int(assignment[0])
            s_idx = int(assignment[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= t_idx < len(ego_tasks)):
            continue
        steps = getattr(ego_tasks[t_idx], "steps", None) or []
        if not (0 <= s_idx < len(steps)):
            continue
        target_id = getattr(steps[s_idx], "target_id", None)
        if target_id is not None:
            out.add(str(target_id))
    return out


def _side_color_of(executor: Any, agent_ids: Sequence[str]) -> str:
    """Our side colour, from the executor's own MATCH-AOU agents.

    Taken from ``executor.agent_by_id`` rather than from the live scenario so it names
    the side the EPISODE was set up for, exactly as the executor's own sensing does.
    """
    by_id = getattr(executor, "agent_by_id", None) or {}
    for aid in agent_ids:
        agent = by_id.get(str(aid))
        if agent is not None:
            colour = getattr(agent, "side_color", None)
            if colour is not None:
                return _normalize_side_color(colour)
    return "unknown"


# =============================================================================
# The builder
# =============================================================================

def build_central_graph_observation(
    scenario: Any,
    *,
    agent_ids: Sequence[str],
    executor: Any,
    current_time: int = 0,
    config: Optional[GraphObservationConfig] = None,
) -> CentralGraphObservation:
    """Project the CURRENT global physical / execution state into a central graph.

    Stateless, like the actor builder: nothing is cached and nothing is mutated. Called
    once per actor decision (see :class:`CentralStateRecorder`), so consecutive samples
    within one tick legitimately differ where an earlier action already changed the
    world's execution state.

    Args:
        scenario: the live BLADE Scenario observation -- the ONE source of physical
            truth for both liveness questions.
        agent_ids: the episode's ORIGINAL same-side agent ids, in their scheduled order.
            Dead agents are dropped from the graph; the order of the survivors is this
            order, so the node set is deterministic and never depends on dict iteration.
        executor: the ``GraphPlanExecutor``. Read ONLY for ``plans`` / ``tasks`` (the
            current assignment relation) and ``agent_by_id`` (our side colour). Never
            mutated.
        current_time: the simulation tick, stored raw and normalized into ``time_norm``.
        config: the SAME :class:`GraphObservationConfig` the actor builder uses, so the
            detection radius, theater scale, risk margin and tick cap can never drift
            between the two views. Defaults are used when ``None``.

    Returns:
        The :class:`CentralGraphObservation`. ``k`` and ``a`` are both variable and may
        legitimately be 0 (every target destroyed, or every agent lost); the encoder is
        size-agnostic and its self-loops keep an empty edge set safe.
    """
    if config is None:
        config = GraphObservationConfig()

    # --- LIVE TARGETS: the raw current world, via the shared enumeration -----------
    # `generate_all_enemy_tasks` is the same current-world extraction episode setup uses.
    # It enumerates the engine's own collections, from which `weapon_endgame` removes a
    # destroyed unit, so a kill removes a node with no liveness flag anywhere.
    our_side = _side_color_of(executor, agent_ids)
    live_tasks = generate_all_enemy_tasks(scenario, our_side)

    task_target_ids: List[str] = []
    task_locs: List[Any] = []
    task_steps: List[Any] = []
    task_features = np.zeros(
        (len(live_tasks), CENTRAL_TASK_FEATURE_DIM), dtype=np.float32
    )
    for j, task in enumerate(live_tasks):
        step = _attack_step(task)
        task_steps.append(step)
        task_target_ids.append(
            str(getattr(step, "target_id", "")) if step is not None else ""
        )
        task_locs.append(getattr(step, "location", None) if step is not None else None)

        # [0] utility_norm -- the actor's own /100 normalization.
        task_features[j, 0] = clip_to_01(float(getattr(task, "utility", 0.0)) / 100.0)
        # [1] probability -- the ATTACK step's success probability (1.0 in this cell).
        raw_p = getattr(step, "probability", 1.0) if step is not None else 1.0
        task_features[j, 1] = clip_to_01(1.0 if raw_p is None else float(raw_p))

    # --- LIVE AGENTS: originally scheduled, still physically present ---------------
    match_agents: Dict[str, Any] = {}
    for agents in (create_agents_from_scenario(scenario) or {}).values():
        for agent in agents:
            match_agents[str(getattr(agent, "id", ""))] = agent

    live_agent_ids: List[str] = []
    live_airframes: List[Any] = []
    for aid in agent_ids:
        airframe = live_aircraft(scenario, str(aid))
        if airframe is None:
            continue  # removed by the engine -> dead -> no node (the ONLY exclusion)
        live_agent_ids.append(str(aid))
        live_airframes.append(airframe)

    a = len(live_agent_ids)
    k = len(live_tasks)
    agent_features = np.zeros((a, CENTRAL_AGENT_FEATURE_DIM), dtype=np.float32)
    for i, airframe in enumerate(live_airframes):
        # EVERY live agent carries its OWN real fuel -- no featureless peers here.
        agent_features[i, 0] = clip_to_01(_compute_fuel_norm(airframe))

    # --- EDGES: the complete live-agent -> live-target relation --------------------
    src: List[int] = []
    dst: List[int] = []
    attrs: List[List[float]] = []
    detection_km = float(config.detection_range_km)
    theater = float(config.theater_scale_km)

    for i, aid in enumerate(live_agent_ids):
        airframe = live_airframes[i]
        agent_pos = (
            getattr(airframe, "latitude", 0.0),
            getattr(airframe, "longitude", 0.0),
        )
        match_agent = match_agents.get(aid)
        assigned_ids = plan_target_ids(executor, aid)

        for j in range(k):
            step = task_steps[j]
            target_loc = task_locs[j]

            if target_loc is not None:
                raw_km = haversine_distance(
                    agent_pos, (target_loc.latitude, target_loc.longitude)
                )
                distance_norm = clip_to_01(raw_km / theater) if theater > 0 else 0.0
                sensed = 1.0 if raw_km <= detection_km else 0.0
            else:
                distance_norm = 0.0
                sensed = 0.0

            if match_agent is not None and step is not None and \
                    match_agent.has_capabilities(
                        getattr(step, "capabilities", []) or []
                    ):
                capable = 1.0
            else:
                capable = 0.0

            # The ACTOR's round-trip model, imported rather than reimplemented, applied
            # to this particular live agent from its current position and fuel budget.
            if match_agent is not None and target_loc is not None and _reachable_by_ego(
                match_agent, target_loc, config.sigma
            ):
                reachable = 1.0
            else:
                reachable = 0.0

            assigned = 1.0 if task_target_ids[j] in assigned_ids else 0.0

            src.append(k + i)   # agent node
            dst.append(j)       # task node
            attrs.append([distance_norm, capable, reachable, sensed, assigned])

    if src:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_type = np.full((len(src),), CENTRAL_EDGE_TYPE, dtype=np.int64)
        edge_attr = np.asarray(attrs, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
        edge_attr = np.zeros((0, CENTRAL_EDGE_ATTR_DIM), dtype=np.float32)

    max_ticks = float(getattr(config, "max_sim_ticks", 0) or 0)
    time_norm = clip_to_01(float(current_time) / max_ticks) if max_ticks > 0 else 0.0

    return CentralGraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=NO_EGO_INDEX,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        task_target_ids=task_target_ids,
        agent_ids=live_agent_ids,
        current_time=int(current_time),
        time_norm=float(time_norm),
    )


# =============================================================================
# The capture seam (training only)
# =============================================================================

@dataclass
class CentralStateRecorder:
    """Collects ONE central state per actor decision, in global decision order.

    This is the CTDE companion structure the tick loop fills. It exists so privileged
    state never has to be smuggled into ``Transition`` (whose ``gobs`` is the actor's
    private observation and must stay that way): the recorder's ``samples`` list is a
    SEPARATE object, aligned 1:1 and index-for-index with ``EpisodeResult.trajectory``.

    ALIGNMENT IS THE CONTRACT. ``run_episode`` calls :meth:`capture` immediately BEFORE
    the actor action of a wake it has already decided to take, and never anywhere else.
    So sample ``i`` is the global state the team was in when decision ``i`` was made,
    before that decision changed anything. With two egos waking on the same tick the
    order is::

        capture(A) -> act(A) + resync(A) -> capture(B) -> act(B) + resync(B) -> env.step

    -- there is NO ``env.step`` between them, so B's physical world is identical to A's,
    while B's ASSIGNED edge feature legitimately reflects A's already-applied resync.
    That is causal, not a leak: the critic is centralized by design, and the actor still
    decides from B's own private observation alone.

    Nothing here is ever consulted by the actor, by evaluation, or by inference.
    """

    config: Optional[GraphObservationConfig] = None
    samples: List[CentralGraphObservation] = field(default_factory=list)

    def capture(
        self,
        *,
        scenario: Any,
        agent_ids: Sequence[str],
        executor: Any,
        current_time: int,
        config: Optional[GraphObservationConfig] = None,
    ) -> CentralGraphObservation:
        """Build and record the central state for the decision about to be taken.

        ``config`` overrides :attr:`config` for this capture. ``run_episode`` passes
        the SAME :class:`GraphObservationConfig` the actor builder is using on that
        episode, so the critic's detection radius / theater scale / tick cap are the
        actor's by construction rather than by two defaults agreeing.
        """
        sample = build_central_graph_observation(
            scenario,
            agent_ids=agent_ids,
            executor=executor,
            current_time=current_time,
            config=config if config is not None else self.config,
        )
        self.samples.append(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)
