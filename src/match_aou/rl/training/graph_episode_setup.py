"""graph_episode_setup.py — Stage 0 of the graph RL orchestrator (episode init).

This is the graph-native pre-loop setup: build the BLADE env, solve MATCH-AOU for
the static plan the egos start from, mint the N per-ego private beliefs, and stand
up the ONE executor. It hands the tick-loop (next task) a single ``EpisodeContext``.

Pipeline position:

    [THIS]  ->  tick-loop  ->  (per-trigger)  graph_builder -> encoder
      setup       execute                       -> graph_action -> graph_effect
                                                 -> executor.resync

This module is graph-native. It reuses only the independent domain helpers:
``create_agents_from_scenario`` / ``generate_all_enemy_tasks`` (scenario_factory),
``MatchAou`` + ``post_solve_filter_and_level`` (solver + post-processing),
``GraphPlanExecutor`` (the sole BLADE translation layer), ``Belief``, and — on the
construction path — the locked PURE placement layer ``graph_hidden_placement``.

TWO PATHS, SELECTED EXPLICITLY (never inferred)
-----------------------------------------------
``setup_episode`` runs exactly one of two episode constructions, chosen by whether the
``(n_hidden, placement_rng)`` PAIR was supplied:

  * LEGACY SPLIT (both omitted): one world is generated with every target in it, and
    ``split_tasks`` masks a subset as hidden. Unchanged, still the default.
  * CONSTRUCTION (both supplied): the caller generates a KNOWN-ONLY world and this
    module builds the hidden half from the SOLVED routes —

        env-1 (known only) -> solve A_init -> B2 route-relative placement
          -> patch the scenario JSON -> close env-1 -> env-2 reload
          -> re-extract env-2 agents/tasks -> solve the full oracle
          -> beliefs + executor built EXCLUSIVELY from env-2 objects

    ``split_tasks`` is NOT called: there is nothing to split, and discovery is
    guaranteed by geometry rather than by the split's adjacency chain.

TWO HIDDEN-CARDINALITY POLICIES ON THE CONSTRUCTION PATH (also never inferred)
------------------------------------------------------------------------------
``hidden_policy`` selects how many hidden targets the construction is allowed to end up
with, and it DEFAULTS to the historical exact policy:

  * ``exact_v1`` (DEFAULT) — exactly ``n_hidden`` placements, one per non-empty ego
    route, and a loud failure otherwise. ``n_hidden=0`` is a legal probe. This is the
    behaviour every approved measurement was taken on and it is UNCHANGED.
  * ``bounded_backoff_v1`` — the GENERALIZED-V1 addition. It enforces the generalized
    cell (``A`` agents in ``GENERALIZED_AGENT_COUNTS``, ``K == A`` RAW known targets,
    ``1 <= n_hidden <= A``), then walks a deterministic seed-driven permutation of STABLE
    AGENT ORDINALS, attempting the SAME approved single-route geometry on each candidate
    with its own pre-derived rng substream, and accepts any realized count ``>= 1``.
    Realizing FEWER hidden targets than requested is a legitimate, RECORDED outcome
    (``EpisodeContext.construction_audit``), never a silently reduced request; realizing
    NONE is a refusal. The agent population, the seed, the world and the requested count
    are never altered to make a world succeed.

Selecting ``bounded_backoff_v1`` without the ``(n_hidden, placement_rng)`` pair is
REFUSED, not ignored. Everything downstream of the placement step is shared: both
policies patch in exactly as many targets as were REALIZED.

ENVIRONMENT OWNERSHIP (construction path)
-----------------------------------------
Environment 1 is TEMPORARY and is closed on every success and failure path; only
environment 2 is authoritative and only it is returned. No env-1 ``Agent`` or ``Task``
object may enter the returned context — the known tasks are re-materialized from env-2
BY TARGET ID, in the exact order of the normalized env-1 belief list, so A_init's
positional ``task_idx`` values stay valid against env-2 objects.

NO-COMMUNICATION FOUNDATION (load-bearing) — enforced by construction here:
  * The N beliefs are MUTUALLY INDEPENDENT (deepcopy tasks + ``_copy_solution``);
    an edit to one ego's belief can never leak into another's.
  * Beliefs / executor see only the ALLOCATED-ONLY normalized task list
    (``solve_and_normalize`` output). Passing the raw ``all_tasks`` would seed an
    unallocated task with no ASSIGNMENT edge that the graph would misread as a pop-up.
    On the construction path that list is the KNOWN half only: a hidden target exists
    in the world but in NO belief — it can enter one only through the sensing ego's
    own trigger path.
  * The partial and full sets are solved TWICE, independently, so the oracle is
    never an alias of A_init (holds even when the split leaves partial == full).
  * ONE ``DETECTION_KM`` feeds the executor now and (later) the builder's
    ``detection_range_km`` — sensing == attack == arrival is a single radius.

WORLD INVENTORY vs ORACLE ALLOCATION (do not conflate them)
-----------------------------------------------------------
``solve_and_normalize`` returns an ALLOCATED-ONLY task list by contract, for both solves.
So ``belief_tasks`` is "the known targets the solver assigned", and ``oracle_tasks`` is
"the targets the ORACLE assigned" — neither is an inventory of what exists. A target the
solver left unselected is missing from both and is still physically in the world, still
sensible, still attackable and still confirmable.

``EpisodeContext.known_target_ids`` / ``executed_target_ids`` are therefore captured from
the RAW task sets BEFORE their solves, and they are what any consumer asking "which
targets does this episode contain?" must read. ``oracle_tasks`` / ``oracle_solution``
remain exactly as they were — correct, and unchanged, for the reward's oracle denominator,
which is a question about ALLOCATION.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import uuid
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ...models import Agent, Location, StepKind, Task
from ...solvers import MatchAou
from ...utils.scheduling_utils import post_solve_filter_and_level
from ...utils.blade_utils.scenario_factory import (
    create_agents_from_scenario,
    generate_all_enemy_tasks,
    iter_enemy_targets,
)
from ...utils.blade_utils.blade_graph_executor import GraphPlanExecutor
from .belief import Belief
# The LOCKED B2 placement layer, consumed exactly as published. This import direction
# is the only legal one: the placement layer must never import THIS module (that would
# drag a pure geometry layer into the setup/solver/executor closure).
from .graph_hidden_placement import (
    HIDDEN_CARDINALITY_POLICIES,
    HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    HIDDEN_POLICY_EXACT_V1,
    BackoffCandidate,
    BoundedBackoffAudit,
    HiddenPlacement,
    PlacementParameters,
    geometric_fingerprint,
    place_hidden_targets,
    place_hidden_targets_bounded,
)

logger = logging.getLogger(__name__)

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)

# =============================================================================
# Module constants (single source of truth).
# =============================================================================
# Sensing == attack == arrival is a SINGLE physical radius (see CLAUDE.md). This
# feeds both the executor's arrival_threshold_km and the graph builder's
# detection_range_km — keep them equal.
DETECTION_KM: float = 50.0
# Fraction of tasks the egos start knowing (the partial set); consumed by
# split_tasks (the discovery-chain sampler).
PARTIAL_RATIO: float = 2.0 / 3.0
MAX_SIM_TICKS: int = 14400
SOLVER_NAME: str = "bonmin"
ATTACKING_SIDE_COLOR: str = "blue"

# --- construction-path constants (the airbase-only cell) ----------------------------
# The ONLY enemy unit class the construction path supports. Hidden targets are placed
# as enemy AIRBASES (utility 80, see scenario_factory._UTILITY_BY_UNIT_TYPE), so a world
# containing SAM facilities or ships would mix target semantics this seam does not
# define. SAM support is a separate design task.
CONSTRUCTION_TARGET_CLASS: str = "Airbase"
# Name stamped on each patched-in hidden airbase. Deterministic (index-based) so two
# runs of the same episode produce the same names; ids are uuid4 and are NOT seed-stable
# (CLAUDE.md section 8), which is why reproducibility is judged by geometric fingerprint.
HIDDEN_TARGET_NAME_TEMPLATE: str = "Hidden Airbase #%03d"
# "All egos launch from ONE point" (CLAUDE.md section 3) is a premise B2's route
# geometry rests on. One metre of slack absorbs float round-trips without admitting a
# genuinely different origin.
LAUNCH_POINT_TOLERANCE_KM: float = 1e-3
# Schema fields a patched hidden airbase must inherit from its prototype.
_REQUIRED_AIRBASE_KEYS: Tuple[str, ...] = (
    "id", "name", "sideId", "className", "latitude", "longitude", "sideColor",
)

# --- GENERALIZED-V1 cardinality (bounded-backoff policy ONLY) -----------------------
# The approved generalized cell: A agents in {2, 3, 4}, K == A known targets, and
# 1 <= H_requested <= A, so a requested world holds A + 1 .. 2A targets. The REALIZED
# hidden count may be smaller (bounded backoff), but never zero, and the REQUESTED count
# is never quietly reduced to make a world succeed -- a silently altered request is what
# makes a denominator unreadable.
# These bounds are enforced ONLY under HIDDEN_POLICY_BOUNDED_BACKOFF_V1. The historical
# exact path carries no such cell restriction and is untouched by them.
GENERALIZED_AGENT_COUNTS: Tuple[int, ...] = (2, 3, 4)


# =============================================================================
# 1. Solve + normalize (clean rewrite of the old solve wrapper)
# =============================================================================

def solve_and_normalize(
    agents: Sequence[Agent],
    tasks: List[Task],
    precedence_relations: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[Dict[str, List[Assignment]], List[Task], List[int]]:
    """Solve MATCH-AOU and return THE normalized (allocated-only) baseline.

    Runs the MINLP (``risk_factor=0.0`` — the movement budget already charges an
    explicit round-trip, so no reserve margin) then ``post_solve_filter_and_level``,
    which drops unselected tasks, remaps ``task_idx`` to a dense ``[0..n-1]``, and
    stamps a topological ``level`` onto every 3-tuple.

    Returns:
        ``(solution, belief_tasks, unselected)`` where
        ``solution`` is the normalized allocation ``{agent_id: [(task_idx, step_idx,
        level), ...]}`` over ``belief_tasks``, ``belief_tasks`` is the ALLOCATED-ONLY
        task list (never the raw pre-filter list), and ``unselected`` is the raw
        solver-index list of tasks ``y[j] == 0`` (pre-remap; informational).

    Degenerate inputs (no tasks or no agents) and a solver that selects nothing both
    return ``({}, [], all-indices-unselected)`` — an empty normalized baseline, never
    a partially-allocated one.
    """
    precedence = list(precedence_relations or [])
    if not tasks or not agents:
        return {}, [], list(range(len(tasks)))

    model = MatchAou(
        agents=list(agents),
        tasks=tasks,
        precedence_relations=precedence,
        risk_factor=0.0,
    )
    raw_solution, _results, unselected = model.solve(solver_name=SOLVER_NAME)
    if not raw_solution:
        # Solver failed to reach acceptable optimality, or selected nothing.
        return {}, [], list(range(len(tasks)))

    artifacts = post_solve_filter_and_level(
        tasks=tasks,
        solution=raw_solution,
        precedence_relations=precedence,
        unselected_tasks=unselected,
    )
    # artifacts.solution is allocated-only + remapped; artifacts.tasks is the
    # allocated-only task list. This pair is THE normalized baseline.
    return artifacts.solution, artifacts.tasks, unselected


# =============================================================================
# 2. Partial / full split (discovery-chain aware, single-radius)
# =============================================================================

def split_tasks(
    all_tasks: List[Task],
    partial_ratio: float = PARTIAL_RATIO,
    *,
    detection_km: float = DETECTION_KM,
    max_attempts: int = 20,
) -> Tuple[List[Task], List[Task], Dict[str, Any]]:
    """Split the full task set into a partial (known) set and the full (oracle) set.

    Discovery-chain aware: every HIDDEN target keeps at least one KNOWN target
    within ``detection_km`` (great-circle between ``task.steps[0].location``
    points), so a masked target can in principle be discovered once an ego reaches
    a known neighbour and senses it. ``detection_km`` is the SAME unified
    sensing/attack/arrival radius the executor senses at (``arrival_threshold_km``)
    and the generator now builds connectivity at — it is NOT BLADE ``aircraft.range``.
    This is the fix for the old flat split, which measured adjacency at the (larger)
    fleet *radar* range and so could mark a target discoverable at a radius the ego
    never actually senses at, leaving hidden targets silently undiscoverable.

    Algorithm (Layer 2 of the two-layer discovery chain; Layer 1 lives in
    ``scenario_generator._ensure_discovery_chain``):

    1. Build undirected adjacency between tasks: ``i`` and ``j`` are neighbours iff
       ``locs[i].distance_to(locs[j]) <= detection_km``.
    2. Pin "isolated" targets (no neighbour at all) into the known set — there is no
       other path to discover them.
    3. Rejection-sample the remaining known slots up to ``max_attempts`` times. A
       draw is valid iff every hidden target has ≥1 known neighbour. ``clean`` on the
       first attempt, ``resampled`` on a later one, ``warn-fallback`` on exhaustion
       (last draw kept).

    Args:
        all_tasks: every enemy task (each with a ``steps[0].location``).
        partial_ratio: fraction of tasks in the partial (known) set.
        detection_km: the unified discovery radius (== the executor sensing radius).
        max_attempts: cap on rejection-sampling retries before giving up.

    Returns:
        ``(partial_tasks, full_tasks, split_meta)`` where ``full_tasks`` is a copy of
        ``all_tasks`` and ``partial_tasks ⊆ full_tasks``. ``split_meta`` keys:
        ``outcome`` (``"clean" | "resampled" | "exhaust" | "warn-fallback" |
        "no-chain"``), ``attempt``, and the counts ``hidden, known, isolated_pinned,
        partial, full``.
    """
    full_tasks = list(all_tasks)
    n = len(all_tasks)

    # Degenerate: 0 or 1 task — nothing to hide, everything is known.
    if n < 2:
        meta = {
            "outcome": "no-chain", "attempt": 1,
            "hidden": 0, "known": n, "isolated_pinned": 0,
            "partial": n, "full": n,
        }
        return full_tasks, full_tasks, meta

    num_partial = max(1, int(n * partial_ratio))

    # Build undirected adjacency between tasks (by index) at the discovery radius.
    locs = [task.steps[0].location for task in all_tasks]
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if locs[i].distance_to(locs[j]) <= detection_km:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Pin isolated tasks (no neighbour within the discovery radius) to the known set.
    isolated = {i for i in range(n) if not neighbors[i]}

    if len(isolated) > num_partial:
        # More isolated targets than the partial budget: force as many as fit into
        # known; the rest stay hidden and undiscoverable (there is no better draw).
        forced = sorted(isolated)[:num_partial]
        partial_tasks = [all_tasks[i] for i in forced]
        logger.warning(
            "Discovery chain (split): isolated=%d exceeds partial budget=%d; "
            "%d isolated target(s) will be hidden and undiscoverable",
            len(isolated), num_partial, len(isolated) - num_partial,
        )
        meta = {
            "outcome": "exhaust", "attempt": 0,
            "hidden": n - len(partial_tasks), "known": len(partial_tasks),
            "isolated_pinned": min(len(isolated), num_partial),
            "partial": len(partial_tasks), "full": n,
        }
        return partial_tasks, full_tasks, meta

    pool = [i for i in range(n) if i not in isolated]
    extra_needed = num_partial - len(isolated)

    last_known_set: Set[int] = set()
    for attempt in range(1, max_attempts + 1):
        sampled = random.sample(pool, extra_needed) if extra_needed > 0 else []
        known_set = isolated | set(sampled)
        hidden_set = set(range(n)) - known_set

        valid = all(bool(neighbors[h] & known_set) for h in hidden_set)
        last_known_set = known_set
        if valid:
            partial_tasks = [all_tasks[i] for i in sorted(known_set)]
            tag = "clean" if attempt == 1 else f"resampled (attempt {attempt})"
            logger.debug(
                "Discovery chain (split): %s (hidden=%d, known=%d, "
                "isolated_pinned=%d, detection=%.0f km)",
                tag, len(hidden_set), len(known_set), len(isolated), detection_km,
            )
            meta = {
                "outcome": "clean" if attempt == 1 else "resampled",
                "attempt": attempt,
                "hidden": len(hidden_set), "known": len(known_set),
                "isolated_pinned": len(isolated),
                "partial": len(partial_tasks), "full": n,
            }
            return partial_tasks, full_tasks, meta

    # Exhausted retries — keep the last draw and warn.
    partial_tasks = [all_tasks[i] for i in sorted(last_known_set)]
    logger.warning(
        "Discovery chain (split): no valid split after %d attempts; some hidden "
        "target(s) may have no known neighbour (detection=%.0f km)",
        max_attempts, detection_km,
    )
    meta = {
        "outcome": "warn-fallback", "attempt": max_attempts,
        "hidden": n - len(partial_tasks), "known": len(partial_tasks),
        "isolated_pinned": len(isolated),
        "partial": len(partial_tasks), "full": n,
    }
    return partial_tasks, full_tasks, meta


# =============================================================================
# 2b. Construction-path helpers (PURE — no BLADE, no solver, no torch)
#
# Every one of these is a hand-testable function on plain data or duck-typed
# observations, so the construction path's failure modes can be proven without a
# solver. They all fail LOUDLY: this seam decides what the world contains and what
# A_init's indices mean, and a silent correction here is a silent research bug.
# =============================================================================

def _resolve_construction_mode(
    n_hidden: Optional[int],
    placement_rng: Optional[random.Random],
    hidden_policy: str = HIDDEN_POLICY_EXACT_V1,
) -> bool:
    """Decide which episode construction ``setup_episode`` runs, and validate its request.

    ``n_hidden`` and ``placement_rng`` are a PAIR, never independently optional: the
    count without the rng would silently fall back to module-global randomness (a
    reproducibility hole), and the rng without the count would be a request with no
    number attached. The construction path is NEVER inferred from ``partial_ratio``.

    ``hidden_policy`` selects the hidden-CARDINALITY policy and DEFAULTS to the historical
    ``exact_v1``, so every pre-GENERALIZED-V1 caller keeps the behaviour every approved
    measurement was taken on. Selecting ``bounded_backoff_v1`` without the construction
    pair is REFUSED rather than ignored: a generalized policy silently dropped on the
    legacy split path would produce a world nobody asked for.

    Only ARGUMENT-level facts are judged here, because that is all that exists before
    BLADE does. The generalized CELL bounds that depend on the world -- ``A`` in
    ``GENERALIZED_AGENT_COUNTS``, ``K == A``, ``H_requested <= A`` -- are checked in
    :func:`_require_generalized_cardinality` once env-1 has been extracted.

    Returns:
        True for the construction path, False for the legacy split path.

    Raises:
        ValueError: on an unknown policy id, a bounded-backoff request without the
            construction pair, a bounded-backoff ``n_hidden < 1``, exactly one of the pair
            supplied, an ``n_hidden`` that is not a non-negative integer (``bool``
            rejected despite subclassing ``int``, matching the locked B2 layer's
            ``_as_assignment``), or a ``placement_rng`` that is not an explicit
            :class:`random.Random`. Raised BEFORE any BLADE object is built.
    """
    policy = str(hidden_policy)
    if policy not in HIDDEN_CARDINALITY_POLICIES:
        raise ValueError(
            f"setup_episode: unknown hidden_policy {hidden_policy!r}; expected one of "
            f"{list(HIDDEN_CARDINALITY_POLICIES)}"
        )
    if n_hidden is None and placement_rng is None:
        if policy != HIDDEN_POLICY_EXACT_V1:
            raise ValueError(
                f"setup_episode: hidden_policy={policy!r} selects a CONSTRUCTION-path "
                "cardinality policy, but neither n_hidden nor placement_rng was supplied, "
                "so the legacy split path would run and the policy would be silently "
                "ignored"
            )
        return False
    if (n_hidden is None) != (placement_rng is None):
        raise ValueError(
            "setup_episode: n_hidden and placement_rng are a PAIR — supply BOTH to run "
            "the construction path or NEITHER to run the legacy split path (got "
            f"n_hidden={n_hidden!r}, placement_rng={type(placement_rng).__name__})"
        )
    if isinstance(n_hidden, bool) or not isinstance(n_hidden, Integral):
        raise ValueError(
            f"setup_episode: n_hidden must be a non-negative integer, got {n_hidden!r} "
            f"of type {type(n_hidden).__name__}"
        )
    if int(n_hidden) < 0:
        raise ValueError(f"setup_episode: n_hidden must be >= 0, got {n_hidden!r}")
    if not isinstance(placement_rng, random.Random):
        raise ValueError(
            "setup_episode: placement_rng must be an explicit random.Random (module-global "
            f"randomness is not reproducible per episode), got {type(placement_rng).__name__}"
        )
    if policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1 and int(n_hidden) < 1:
        raise ValueError(
            f"setup_episode: hidden_policy={policy!r} requires n_hidden >= 1 (a "
            f"generalized world needs a hidden half), got {n_hidden!r}. n_hidden=0 is a "
            f"probe of the {HIDDEN_POLICY_EXACT_V1!r} path only."
        )
    return True


def _require_generalized_cardinality(
    *, agent_count: int, known_count: int, hidden_requested: int
) -> None:
    """Enforce the approved GENERALIZED-V1 cell, BEFORE anything is solved or placed.

    The approved shape is ``A`` agents in :data:`GENERALIZED_AGENT_COUNTS`, ``K == A``
    RAW KNOWN-world targets, and ``1 <= H_requested <= A``. ``known_count`` therefore has
    to come from the RAW world inventory (``_world_target_ids``), never from an
    allocated-only solver output, which omits every unselected target and would under-count
    the very world this rule is about.

    Refuses rather than repairs: the agent population, the seed, the world and the
    requested hidden count are NEVER silently altered to make a world succeed. Only the
    REALIZED hidden count may end up smaller than requested, and only through the recorded
    bounded-backoff walk.

    Called ONLY under :data:`HIDDEN_POLICY_BOUNDED_BACKOFF_V1`; the historical exact path
    is unrestricted and never reaches it.

    Raises:
        RuntimeError: on any violation of the cell.
    """
    if int(agent_count) not in GENERALIZED_AGENT_COUNTS:
        raise RuntimeError(
            f"setup_episode: generalized construction needs A in "
            f"{list(GENERALIZED_AGENT_COUNTS)} agents, got A={int(agent_count)}"
        )
    if int(known_count) != int(agent_count):
        raise RuntimeError(
            f"setup_episode: generalized construction needs K == A known world targets, "
            f"got K={int(known_count)} for A={int(agent_count)}"
        )
    if not (1 <= int(hidden_requested) <= int(agent_count)):
        raise RuntimeError(
            f"setup_episode: generalized construction needs 1 <= H_requested <= A, got "
            f"H_requested={int(hidden_requested)} for A={int(agent_count)}"
        )


def _task_target_id(task: Task) -> str:
    """A task's target id == its ATTACK step's ``target_id``.

    Reimplemented locally rather than importing a private helper from ``graph_builder``
    / ``graph_trigger``, matching what those modules already do to each other.
    """
    steps = getattr(task, "steps", None) or []
    for step in steps:
        if getattr(step, "step_kind", None) == StepKind.ATTACK:
            return str(getattr(step, "target_id", ""))
    return str(getattr(steps[0], "target_id", "")) if steps else ""


def _shared_launch_point(agents: Sequence[Agent]) -> Location:
    """The ONE origin every predicted route starts from, verified rather than assumed.

    B2 predicts each ego's route from a SHARED launch point, which is only meaningful
    because "Launch point == the BLUE airbase" (``CLAUDE.md`` section 3) — every ego goes
    airborne over the same base, and ``Agent.location == Agent.return_location``. Both
    halves are CHECKED here instead of trusted: a fleet parked away from its base (the
    defect `384845b` fixed) would silently make every predicted leg start in the wrong
    place, and the placements would still validate against that wrong geometry.

    Raises:
        RuntimeError: on an empty fleet, a missing location, egos at different points, or
            a return location that is not the launch point.
    """
    if not agents:
        raise RuntimeError("setup_episode: no agents to derive a launch point from")
    origin = getattr(agents[0], "location", None)
    if origin is None:
        raise RuntimeError("setup_episode: agent 0 has no location; cannot derive a launch point")
    for agent in agents:
        loc = getattr(agent, "location", None)
        if loc is None:
            raise RuntimeError(f"setup_episode: agent {agent.id} has no location")
        if origin.distance_to(loc) > LAUNCH_POINT_TOLERANCE_KM:
            raise RuntimeError(
                f"setup_episode: agent {agent.id} starts {origin.distance_to(loc):.3f} km "
                "from the fleet origin; the construction path requires ONE shared launch "
                "point (CLAUDE.md section 3) because route prediction is seeded from it"
            )
        ret = getattr(agent, "return_location", None)
        if ret is not None and origin.distance_to(ret) > LAUNCH_POINT_TOLERANCE_KM:
            raise RuntimeError(
                f"setup_episode: agent {agent.id} returns to a point "
                f"{origin.distance_to(ret):.3f} km from its launch point; the solver's "
                "round_trip_cost is only a symmetric out-and-back when the two coincide"
            )
    return origin


def _require_airbase_only_targets(scenario: Any, attacking_side_color: str) -> None:
    """Refuse a construction world that is not the airbase-only cell.

    Hidden targets are patched in as enemy AIRBASES, so a world already containing SAM
    facilities or enemy ships would mix target classes (and utilities) this seam has no
    semantics for. Checked against the LIVE observation rather than the config, so a
    caller that bypasses ``TrainConfig`` / ``RolloutConfig`` is caught too.
    """
    for target_id, _loc in iter_enemy_targets(scenario, attacking_side_color):
        unit = scenario.get_target(target_id)
        if unit is None:
            raise RuntimeError(
                f"setup_episode: enemy target {target_id!r} does not resolve to a unit"
            )
        class_name = type(unit).__name__
        if class_name != CONSTRUCTION_TARGET_CLASS:
            raise RuntimeError(
                "setup_episode: the construction path supports the airbase-only cell "
                f"(include_sams=False), but enemy target {target_id!r} is a "
                f"{class_name}. Mixed SAM / facility / ship target semantics are a "
                "separate design task and are not invented here."
            )


def _select_hidden_prototype(
    current_scenario: Dict[str, Any], attacking_side_color: str
) -> Dict[str, Any]:
    """Pick the enemy-airbase entry a hidden target is deep-copied from.

    "Safe" means: enemy-side, schema-complete, and holding NO aircraft — cloning a base
    with an inventory would mint enemy aircraft as a side effect of adding a target. The
    FIRST qualifying entry is taken (deterministic, and it consumes no rng).

    Raises:
        RuntimeError: if the scenario has no ``airbases`` list, no safe enemy prototype,
            or enemy airbases spanning more than one ``sideId`` (an ambiguous schema —
            this seam will not guess which enemy side a new target belongs to).
    """
    ours = str(attacking_side_color).lower()
    airbases = current_scenario.get("airbases")
    if not isinstance(airbases, list):
        raise RuntimeError(
            "setup_episode: scenario JSON has no 'currentScenario.airbases' list to patch"
        )

    candidates: List[Dict[str, Any]] = []
    for entry in airbases:
        if not isinstance(entry, dict):
            continue
        side_color = str(entry.get("sideColor", "")).lower()
        if not side_color or side_color == ours:
            continue
        if entry.get("aircraft"):
            continue  # never clone an inventory into the world as a side effect
        if any(entry.get(key) is None for key in _REQUIRED_AIRBASE_KEYS):
            continue
        if not str(entry.get("sideId", "")) or not str(entry.get("className", "")):
            continue
        candidates.append(entry)

    if not candidates:
        raise RuntimeError(
            "setup_episode: no safe enemy-airbase prototype in "
            "currentScenario.airbases (need an enemy-side airbase with an empty aircraft "
            f"inventory and the fields {list(_REQUIRED_AIRBASE_KEYS)}); refusing to "
            "invent one"
        )
    side_ids = {str(entry["sideId"]) for entry in candidates}
    if len(side_ids) != 1:
        raise RuntimeError(
            "setup_episode: enemy airbases span several sideIds "
            f"({sorted(side_ids)}); the schema is ambiguous about which side a hidden "
            "target belongs to"
        )
    return candidates[0]


def build_patched_scenario(
    scenario_json: str,
    placements: Sequence[HiddenPlacement],
    *,
    attacking_side_color: str = ATTACKING_SIDE_COLOR,
) -> str:
    """Append one enemy airbase per placement to the scenario JSON, ONCE.

    The scenario is patched, never regenerated: regeneration after the A_init solve would
    change ids, fleet state and known-target geometry, and A_init's positional
    ``task_idx`` values would stop meaning anything. Known targets are not edited and no
    existing collection is reordered — new entries go at the END of
    ``currentScenario.airbases``, so ``iter_enemy_targets``' facilities->airbases->ships
    enumeration keeps every known target at its original position.

    Each hidden entry is a deep copy of a safe prototype (see
    :func:`_select_hidden_prototype`) with a fresh uuid4 id, a deterministic unique name,
    the placement's coordinates, and an explicitly EMPTY aircraft inventory. Everything
    else — ``sideId``, ``className``, ``sideColor``, ``altitude`` — is inherited, so the
    new target is red exactly the way the prototype is.

    Args:
        scenario_json: the scenario JSON *content*. An empty ``placements`` returns it
            BYTE-IDENTICAL (an ``n_hidden=0`` probe must not perturb the world).
        placements: the locked B2 records, in their returned (ego-sorted) order.
        attacking_side_color: our side; anything else is "enemy".

    Returns:
        The patched JSON content, ready for environment 2.

    Raises:
        RuntimeError: on a malformed scenario, a missing/unsafe prototype, or a generated
            name / id that collides with an existing airbase.
    """
    if not placements:
        return scenario_json

    data = json.loads(scenario_json)
    current = data.get("currentScenario") if isinstance(data, dict) else None
    if not isinstance(current, dict):
        raise RuntimeError("setup_episode: scenario JSON has no 'currentScenario' object")

    prototype = _select_hidden_prototype(current, attacking_side_color)
    airbases = current["airbases"]
    taken_names = {str(e.get("name")) for e in airbases if isinstance(e, dict)}
    taken_ids = {str(e.get("id")) for e in airbases if isinstance(e, dict)}

    for index, placement in enumerate(placements, start=1):
        entry = copy.deepcopy(prototype)
        entry["id"] = str(uuid.uuid4())
        entry["name"] = HIDDEN_TARGET_NAME_TEMPLATE % index
        entry["latitude"] = float(placement.latitude)
        entry["longitude"] = float(placement.longitude)
        entry["aircraft"] = []
        if entry["name"] in taken_names:
            raise RuntimeError(
                f"setup_episode: hidden target name {entry['name']!r} already exists in "
                "the scenario; refusing to add an ambiguously-named target"
            )
        if entry["id"] in taken_ids:
            raise RuntimeError(f"setup_episode: uuid collision on {entry['id']!r}")
        taken_names.add(entry["name"])
        taken_ids.add(entry["id"])
        airbases.append(entry)

    return json.dumps(data)


def _require_agent_ids_preserved(before: Sequence[str], after: Sequence[str]) -> None:
    """A_init's agent keys must still address the runtime egos after the reload.

    Compared as an ORDERED list, not a set: the patch appends one collection entry and
    must not disturb ``create_agents_from_scenario``'s aircraft -> ships -> airbase-aircraft
    enumeration at all.
    """
    if list(before) != list(after):
        raise RuntimeError(
            "setup_episode: agent identity drifted across patch/reload — env-1 "
            f"{list(before)} vs env-2 {list(after)}; A_init's agent keys would no longer "
            "address the runtime egos"
        )


def _rematerialize_known_tasks(
    world_tasks: Sequence[Task], known_target_ids: Sequence[str]
) -> List[Task]:
    """Re-look-up the known belief tasks as ENV-2 objects, preserving A_init's order.

    A_init's ``task_idx`` is positional into the normalized env-1 belief list, so env-2's
    belief list must present the SAME targets in the SAME order — built from env-2 ``Task``
    objects, because only those are bound to the world the executor drives. Matching is by
    target id; the ORDER comes from ``known_target_ids``, never from env-2's enumeration.

    Raises:
        RuntimeError: on a duplicate/blank target id in either list, or a known target
            missing from the reloaded world.
    """
    by_target: Dict[str, Task] = {}
    for task in world_tasks:
        target_id = _task_target_id(task)
        if not target_id:
            raise RuntimeError("setup_episode: reloaded world holds a task with no target id")
        if target_id in by_target:
            raise RuntimeError(
                f"setup_episode: target id {target_id!r} appears twice in the reloaded "
                "world; positional task indices would be ambiguous"
            )
        by_target[target_id] = task

    out: List[Task] = []
    seen: Set[str] = set()
    for position, target_id in enumerate(known_target_ids):
        if not target_id:
            raise RuntimeError(
                f"setup_episode: known task_idx {position} has no target id to match on"
            )
        if target_id in seen:
            raise RuntimeError(
                f"setup_episode: known target {target_id!r} appears twice in A_init's "
                "task list; positional task indices would be ambiguous"
            )
        if target_id not in by_target:
            raise RuntimeError(
                f"setup_episode: known target {target_id!r} (A_init task_idx {position}) "
                "is absent from the reloaded world — the patch changed a known target"
            )
        seen.add(target_id)
        out.append(by_target[target_id])
    return out


def _world_target_ids(tasks: Sequence[Task], what: str) -> Tuple[str, ...]:
    """The RAW target-id roster of a task set, in order — the t=0 world inventory.

    Called on a task list as it comes out of ``_extract_world`` (or out of
    ``split_tasks``), BEFORE ``solve_and_normalize`` ever sees it. That ordering is the
    whole point: ``solve_and_normalize`` returns an ALLOCATED-ONLY list, so a task the
    solver did not select is gone from its output while the target is still very much in
    the world the executor will fly through. A snapshot taken after the solve therefore
    describes the ALLOCATION, not the WORLD, and anything that reads it as a world
    inventory under-counts by exactly the unselected targets.

    Deduplicated by target id with first occurrence winning — the roster is a statement
    about TARGETS, and two tasks may legitimately name one target. A task that names NO
    target is a different thing entirely: it means the structure this snapshot is derived
    from is not what it is assumed to be, and silently dropping it would shorten the
    inventory. That RAISES.

    Raises:
        RuntimeError: on a task with no resolvable target id.
    """
    ids: List[str] = []
    for position, task in enumerate(tasks):
        target_id = _task_target_id(task)
        if not target_id:
            raise RuntimeError(
                f"setup_episode: {what} task {position} names no target, so the t=0 "
                "world roster cannot be derived from it"
            )
        ids.append(target_id)
    return tuple(dict.fromkeys(ids))


# =============================================================================
# 3. Episode context (the handoff object the tick-loop consumes)
# =============================================================================

@dataclass(frozen=True)
class ConstructionAudit:
    """Requested vs REALIZED construction cardinality for ONE generalized episode.

    The GENERALIZED-V1 accounting surface, and deliberately a TYPED record on the returned
    context rather than a console line: requested-vs-realized is meant to be inspected as
    a DISTRIBUTION across episodes (a HIGH hidden-load stratum that quietly collapses into
    the LOW one is not a stratum), and an unstructured log cannot support that.

    THE COUNTS ARE WORLD COUNTS, NOT ALLOCATION COUNTS. ``known_realized`` and
    ``total_realized`` are taken from the RAW pre-solve snapshots
    ``EpisodeContext.known_target_ids`` / ``executed_target_ids``, never from
    ``belief_tasks`` / ``oracle_tasks``, which are allocated-only by
    ``solve_and_normalize``'s contract and omit every target the solver did not select.

    Populated ONLY for :data:`HIDDEN_POLICY_BOUNDED_BACKOFF_V1`. The legacy split path and
    the historical exact construction path leave ``EpisodeContext.construction_audit`` at
    ``None`` -- that absence IS how a reader tells which policy ran, and it is what keeps
    the historical path observably unchanged.

    NOTHING here reaches the acting path: no count, no policy id and no candidate reason
    enters ``GraphObservation``. A count of what is hidden is exactly the kind of
    privileged quantity an ego cannot sense (``CLAUDE.md`` section 3).
    """

    policy: str
    agent_count: int
    known_requested: int
    known_realized: int
    hidden_requested: int
    hidden_realized: int
    total_requested: int
    total_realized: int
    backoff: BoundedBackoffAudit

    @property
    def candidate_order(self) -> Tuple[int, ...]:
        """The deterministic candidate ORDINAL order the backoff walk used."""
        return self.backoff.candidate_order

    @property
    def considered_ordinals(self) -> Tuple[int, ...]:
        """The ordinals the bounded walk actually visited (a prefix of the order)."""
        return self.backoff.considered_ordinals

    @property
    def candidates(self) -> Tuple[BackoffCandidate, ...]:
        """Per-candidate outcome / rejection reason, in visit order."""
        return self.backoff.candidates

    @property
    def selected_ordinals(self) -> Tuple[int, ...]:
        """The ordinals whose routes really carry a realized hidden target."""
        return self.backoff.selected_ordinals

    @property
    def geometric_fingerprint(self) -> Tuple[Tuple[float, float], ...]:
        """Id-free geometric identity of the realized placements."""
        return self.backoff.geometric_fingerprint

    @property
    def realized_full_request(self) -> bool:
        """True iff every requested hidden target was realized."""
        return self.backoff.realized_full_request

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-ready view (plain builtins only)."""
        return {
            "policy": str(self.policy),
            "agent_count": int(self.agent_count),
            "known_requested": int(self.known_requested),
            "known_realized": int(self.known_realized),
            "hidden_requested": int(self.hidden_requested),
            "hidden_realized": int(self.hidden_realized),
            "total_requested": int(self.total_requested),
            "total_realized": int(self.total_realized),
            "backoff": self.backoff.as_dict(),
        }


@dataclass
class EpisodeContext:
    """Everything the tick-loop needs after Stage-0 setup.

    ``a_init`` is the static plan the egos start from (the seed the beliefs and the
    executor were built from). It is exposed for the tick-loop / reward and so the
    two-independent-solves invariant (oracle is NOT A_init) is observable; the live
    authoritative plans are the per-ego ``beliefs`` and ``executor.plans``.

    ``observation`` is the ``env.reset()`` observation captured here — the SEED the
    tick-loop needs for its very first per-ego sense (before it has stepped BLADE even
    once). The loop advances its own local ``obs`` from ``env.step`` thereafter and must
    NEVER call ``env.reset()`` again (that would restart the episode and invalidate the
    solve this context is built around); this field exists so it doesn't have to.
    """

    env: Any
    game: Any
    observation: Any
    agents: List[Agent]
    agent_ids: List[str]
    beliefs: Dict[str, Belief]
    executor: GraphPlanExecutor
    a_init: Dict[str, List[Assignment]]
    oracle_solution: Dict[str, List[Assignment]]
    oracle_tasks: List[Task]
    split_meta: Dict[str, Any]
    record: bool = False
    """True iff recording was armed at setup (a ``recording_export_path`` was given);
    the tick-loop drives the recorder (start / step / export) iff this is True."""

    construction_audit: Optional[ConstructionAudit] = None
    """Requested-vs-realized construction accounting — see :class:`ConstructionAudit`.

    Set ONLY on the GENERALIZED-V1 ``bounded_backoff_v1`` construction path. ``None`` on
    the legacy split path AND on the historical ``exact_v1`` construction path, whose
    cardinality is exact by contract and therefore has nothing to reconcile."""

    placements: Tuple[HiddenPlacement, ...] = ()
    """The locked B2 placement records behind this episode's hidden targets — the
    construction path's audit trail, EMPTY on the legacy split path (and on an
    ``n_hidden=0`` probe). Carries geometry and no target uuid, because generated ids are
    not seed-derived (``CLAUDE.md`` section 8): reproducibility is judged by
    ``geometric_fingerprint(ctx.placements)``, never by id."""

    known_target_ids: Tuple[str, ...] = ()
    """Every RAW known-world target id, snapshotted BEFORE the known solve filtered it.

    THE t=0 KNOWN-WORLD INVENTORY, and deliberately not the same thing as ``belief_tasks``
    or the belief task lists built from them: ``solve_and_normalize`` returns an
    ALLOCATED-ONLY list, so a known target the solver left unselected is absent from every
    belief while still sitting in the world the egos fly through. Anything that needs to
    know how many targets EXIST reads this; anything that needs to know what the egos were
    PLANNED against reads the beliefs."""

    executed_target_ids: Tuple[str, ...] = ()
    """Every RAW target id in the AUTHORITATIVE returned environment, snapshotted BEFORE
    the full oracle solve filtered it.

    THE t=0 EXECUTED-WORLD INVENTORY — known half plus hidden half, in the world's own
    order, with ``known_target_ids`` a subset of it. Distinct from ``oracle_tasks`` for
    exactly the reason above: the oracle is an ALLOCATION over this world, so a target the
    oracle did not select is missing from ``oracle_tasks`` and present here. Reading
    ``oracle_tasks`` as a world inventory is the defect these two fields exist to close.

    A RUNTIME SNAPSHOT, not a cross-run reproducibility identity: generated target uuids
    are not seed-derived (``CLAUDE.md`` section 8), so these ids are never a comparison
    key BETWEEN runs — that is still ``geometric_fingerprint(ctx.placements)``."""


# =============================================================================
# 4. Episode setup (env + solve + belief/executor construction)
# =============================================================================

def _build_env(
    scenario_json: str,
    *,
    max_episode_steps: int,
    attacking_side_color: str,
    record_every_seconds: Optional[int],
    recording_export_path: Optional[str],
) -> Tuple[Any, Any, Any]:
    """Build + reset ONE BLADE env exactly as the frozen integration does.

    Returns ``(game, env, reset_observation)`` with ``game.current_side_id`` already
    pointing at our side. Extracted so environment 1 and environment 2 are built by the
    SAME code — a construction path whose two envs were built differently could not
    claim the reload preserved anything.

    OWNS THE ENVIRONMENT UNTIL IT RETURNS. Between ``gymnasium.make`` and the return
    statement the new environment belongs to nobody else: the caller's ``finally`` /
    ``except`` blocks are keyed on the value this function hands back, so a failure in
    ``reset()`` (or in the side selection after it) would otherwise leave an engine
    object that NO cleanup path can reach. The guard below closes it exactly once and
    re-raises the ORIGINAL exception unchanged — a leaked environment must not become a
    masked error. Because both environment 1 and environment 2 are built here, this one
    guard covers both pre-return construction windows.
    """
    # BLADE / gymnasium imported lazily (engine boundary): importing Belief or the
    # solve helpers elsewhere must not drag in the engine.
    import gymnasium
    from blade.Game import Game
    from blade.Scenario import Scenario

    env: Any = None
    try:
        game = Game(
            current_scenario=Scenario(),
            record_every_seconds=record_every_seconds,
            recording_export_path=recording_export_path or ".",
        )
        game.load_scenario(scenario_json)
        env = gymnasium.make(
            "blade/BLADE-v0", game=game, max_episode_steps=max_episode_steps
        )
        obs, _info = env.reset()

        side_name = attacking_side_color.upper()
        for side in getattr(obs, "sides", []) or []:
            if str(getattr(side, "name", "")).upper() == side_name:
                game.current_side_id = side.id
                break
        return game, env, obs
    except BaseException:
        if env is not None:
            _close_quietly(env)
        raise


def _extract_world(obs: Any, attacking_side_color: str) -> Tuple[List[Agent], List[Task]]:
    """Extract our agents + every enemy task from ONE reset observation."""
    agents_by_side = create_agents_from_scenario(obs)
    agents = agents_by_side.get(attacking_side_color.lower(), [])
    if not agents:
        raise RuntimeError(
            f"setup_episode: no {attacking_side_color!r} agents in the scenario"
        )
    # probability=1.0: the anti-stacking task-construction default (see CLAUDE.md).
    all_tasks = generate_all_enemy_tasks(
        obs, attacking_side_color=attacking_side_color, probability=1.0
    )
    if not all_tasks:
        raise RuntimeError("setup_episode: no enemy tasks in the scenario")
    return agents, all_tasks


def _close_quietly(env: Any) -> None:
    """Close an env without letting the close mask the error that caused it."""
    try:
        env.close()
    except Exception:  # pragma: no cover - defensive; a failed close must not shadow
        logger.warning("setup_episode: env.close() failed", exc_info=True)


def setup_episode(
    scenario_json: str,
    *,
    partial_ratio: float = PARTIAL_RATIO,
    max_episode_steps: int = MAX_SIM_TICKS,
    attacking_side_color: str = ATTACKING_SIDE_COLOR,
    detection_km: float = DETECTION_KM,
    record_every_seconds: Optional[int] = 10,
    recording_export_path: Optional[str] = None,
    n_hidden: Optional[int] = None,
    placement_rng: Optional[random.Random] = None,
    hidden_policy: str = HIDDEN_POLICY_EXACT_V1,
) -> EpisodeContext:
    """Stand up one episode: BLADE env -> solve -> beliefs + executor.

    Runs the LEGACY SPLIT path by default and the CONSTRUCTION path when the
    ``(n_hidden, placement_rng)`` pair is supplied — see the module docstring. The mode
    is never inferred from ``partial_ratio``.

    Args:
        scenario_json: the scenario JSON *content* (as ``load_scenario`` expects). On the
            construction path this is a KNOWN-ONLY world; the hidden half is built here.
        partial_ratio: fraction of tasks the egos start knowing (fed to ``split_tasks``).
            LEGACY PATH ONLY — the construction path never calls ``split_tasks``.
        max_episode_steps: BLADE ``max_episode_steps`` (per-episode tick cap).
        attacking_side_color: our side (blue); selects agents and the blue side id.
        detection_km: the unified sensing/attack/arrival radius fed to the executor, and
            (construction path) to the B2 :class:`PlacementParameters`.
        record_every_seconds / recording_export_path: passed to ``Game``. Passing a
            ``recording_export_path`` ARMS recording (sets ``EpisodeContext.record``);
            setup itself does NOT start recording — the tick-loop starts / steps /
            exports it. ``record_every_seconds`` throttles the per-tick frame cadence.
            Only the RETURNED environment is ever armed; the construction path's
            temporary environment 1 never records.
        n_hidden: hidden targets REQUESTED. Non-negative; ``0`` is a legal probe that
            places nothing and patches nothing (``exact_v1`` only). Must be paired with
            ``placement_rng``.
        placement_rng: an explicit :class:`random.Random` driving B2's leg / fraction /
            offset draws. Must be paired with ``n_hidden``.
        hidden_policy: the hidden-CARDINALITY policy, DEFAULTING to the historical
            ``exact_v1`` so every pre-GENERALIZED-V1 caller keeps the behaviour the
            approved measurements were taken on. ``bounded_backoff_v1`` selects the
            GENERALIZED-V1 deterministic bounded backoff, which enforces the generalized
            cell (``A`` in :data:`GENERALIZED_AGENT_COUNTS`, ``K == A``,
            ``1 <= n_hidden <= A``), MAY realize fewer hidden targets than requested, and
            fills :attr:`EpisodeContext.construction_audit`. It is REFUSED — never
            ignored — without the construction pair.

    Returns:
        An :class:`EpisodeContext` handoff object.

    Raises:
        ValueError: on an unknown ``hidden_policy``, or an invalid / half-supplied
            construction pair — raised BEFORE any BLADE object is built.
        RuntimeError: if the scenario yields no blue agents or no enemy tasks, or (on the
            construction path) if any construction invariant fails.
    """
    construction = _resolve_construction_mode(n_hidden, placement_rng, hidden_policy)
    if construction:
        return _setup_episode_construction(
            scenario_json,
            n_hidden=int(n_hidden),                      # type: ignore[arg-type]
            placement_rng=placement_rng,                 # type: ignore[arg-type]
            hidden_policy=str(hidden_policy),
            max_episode_steps=max_episode_steps,
            attacking_side_color=attacking_side_color,
            detection_km=detection_km,
            record_every_seconds=record_every_seconds,
            recording_export_path=recording_export_path,
        )
    return _setup_episode_legacy(
        scenario_json,
        partial_ratio=partial_ratio,
        max_episode_steps=max_episode_steps,
        attacking_side_color=attacking_side_color,
        detection_km=detection_km,
        record_every_seconds=record_every_seconds,
        recording_export_path=recording_export_path,
    )


def _setup_episode_legacy(
    scenario_json: str,
    *,
    partial_ratio: float,
    max_episode_steps: int,
    attacking_side_color: str,
    detection_km: float,
    record_every_seconds: Optional[int],
    recording_export_path: Optional[str],
) -> EpisodeContext:
    """The split-based episode construction: ONE world, ``split_tasks`` masks the hidden half."""
    # --- 1-2. Build env exactly as the frozen integration does; select our side ---
    game, env, obs = _build_env(
        scenario_json,
        max_episode_steps=max_episode_steps,
        attacking_side_color=attacking_side_color,
        record_every_seconds=record_every_seconds,
        recording_export_path=recording_export_path,
    )

    # --- 3. Extract blue agents + all enemy tasks -----------------------------
    agents, all_tasks = _extract_world(obs, attacking_side_color)

    # --- 4. Partial / full split (discovery-chain aware) ----------------------
    # detection_km is the SAME radius fed to the executor below (arrival_threshold_km)
    # and the generator's connectivity — so split-adjacency == runtime-sensing by
    # construction. With a real split, partial ⊊ full: A_init covers only the known
    # targets and the hidden ones become discoverable pop-ups.
    partial, full, split_meta = split_tasks(
        all_tasks, partial_ratio, detection_km=detection_km
    )

    # --- 4b. The t=0 world snapshots, taken from the RAW split sets ------------
    # Before either solve, because both solves return ALLOCATED-ONLY task lists: a target
    # the solver leaves unselected is still in the world an ego flies through, and a
    # snapshot taken afterwards would describe the allocation instead of the world.
    known_target_ids = _world_target_ids(partial, "known (partial)")
    executed_target_ids = _world_target_ids(full, "executed (full)")

    # --- 5. Solve the PARTIAL set -> A_init (the static plan egos start from) --
    a_init, belief_tasks, _ = solve_and_normalize(agents, partial)

    # --- 6. Solve the FULL set -> oracle (for the reward chat) ----------------
    # A SEPARATE, independent solve: oracle must never be an alias of a_init, even
    # in the degenerate case where the split leaves partial == full.
    oracle_solution, oracle_tasks, _ = solve_and_normalize(agents, full)

    # --- 7-8. Beliefs + executor over the normalized (allocated-only) baseline ---
    return _finish_context(
        game=game,
        env=env,
        obs=obs,
        agents=agents,
        a_init=a_init,
        belief_tasks=belief_tasks,
        oracle_solution=oracle_solution,
        oracle_tasks=oracle_tasks,
        split_meta=split_meta,
        detection_km=detection_km,
        recording_export_path=recording_export_path,
        placements=(),
        known_target_ids=known_target_ids,
        executed_target_ids=executed_target_ids,
    )


def _setup_episode_construction(
    scenario_json: str,
    *,
    n_hidden: int,
    placement_rng: random.Random,
    max_episode_steps: int,
    attacking_side_color: str,
    detection_km: float,
    record_every_seconds: Optional[int],
    recording_export_path: Optional[str],
    hidden_policy: str = HIDDEN_POLICY_EXACT_V1,
) -> EpisodeContext:
    """The construction path: solve -> place -> patch -> reload -> oracle.

    TWO hidden-CARDINALITY policies share this one seam, and only the PLACEMENT STEP
    differs between them. ``exact_v1`` (the default) demands exactly ``n_hidden``
    placements and fails loudly otherwise — unchanged, and the behaviour every approved
    measurement was taken on. ``bounded_backoff_v1`` enforces the GENERALIZED-V1 cell,
    walks a deterministic bounded backoff over stable agent ordinals, accepts any realized
    count ``>= 1``, and records requested vs realized in
    :attr:`EpisodeContext.construction_audit`. Everything downstream of the placement step
    — the patch, the reload, env-2's authority, the world snapshots, the re-materialized
    known tasks and the oracle solve — is the SAME code for both, because both must patch
    in exactly as many targets as were REALIZED.

    Environment 1 is temporary and is closed on EVERY exit path (its ``finally``);
    environment 2 is the only one that reaches the caller, and it is closed too if
    anything downstream of its reset fails. Nothing built from environment 1 survives
    except pure data: the normalized ``a_init`` assignments, the ORDERED list of known
    target ids that A_init's positional indices refer to (``known_target_order``, an
    ALLOCATED-ONLY list), and the RAW known-world snapshot ``known_target_ids``. Those
    last two are different lists on purpose — the first says what the egos were planned
    against, the second says what exists — and only the second is a world inventory.
    """
    # ================= Environment 1 — TEMPORARY (never recorded) =================
    game1, env1, obs1 = _build_env(
        scenario_json,
        max_episode_steps=max_episode_steps,
        attacking_side_color=attacking_side_color,
        record_every_seconds=record_every_seconds,
        recording_export_path=None,
    )
    try:
        agents1, known_world_tasks = _extract_world(obs1, attacking_side_color)
        _require_airbase_only_targets(obs1, attacking_side_color)

        env1_agent_ids = [str(a.id) for a in agents1]
        # The t=0 KNOWN-WORLD snapshot, taken from the RAW env-1 extraction — before the
        # known solve below, whose output is allocated-only and therefore not a world
        # inventory. Strict: a task with no target id raises rather than shortening it.
        known_target_ids = _world_target_ids(known_world_tasks, "known world")
        known_world_ids = list(known_target_ids)

        # GENERALIZED-V1 only: judge the requested cell against the RAW world BEFORE the
        # solve, so an out-of-cell request costs no bonmin call and leaves no partial
        # construction behind. `known_world_ids` is the raw world inventory, never an
        # allocation, which is the whole point of checking `K == A` against it.
        generalized = str(hidden_policy) == HIDDEN_POLICY_BOUNDED_BACKOFF_V1
        if generalized:
            _require_generalized_cardinality(
                agent_count=len(agents1),
                known_count=len(known_world_ids),
                hidden_requested=int(n_hidden),
            )

        launch_point = _shared_launch_point(agents1)

        # --- Solve the KNOWN set -> A_init (the static plan egos start from) ------
        a_init, known_belief_tasks, _unselected = solve_and_normalize(
            agents1, known_world_tasks
        )
        if not a_init:
            raise RuntimeError(
                "setup_episode: the known-only solve allocated nothing, so there is no "
                "predicted route to place hidden targets against"
            )
        # The ONE thing A_init's positional task_idx means, captured as pure strings so
        # no env-1 Task object can survive this block.
        known_target_order = [_task_target_id(t) for t in known_belief_tasks]

        # --- B2 route-relative placement (consumed exactly as published) ----------
        placements: Tuple[HiddenPlacement, ...] = ()
        backoff: Optional[BoundedBackoffAudit] = None
        if generalized:
            # GENERALIZED-V1: the candidate population is the AUTHORITATIVE pre-solve
            # agent sequence, so an ego the allocated-only solve omitted is still a
            # candidate (it is simply recorded as having no route). Ordinals come from
            # that sequence's ORDER, never from the id strings, which are not seed-derived
            # (CLAUDE.md section 8).
            placements, backoff = place_hidden_targets_bounded(
                a_init,
                known_belief_tasks,
                launch_point,
                PlacementParameters(detection_km=float(detection_km)),
                placement_rng,
                agent_ordinals=env1_agent_ids,
                hidden_requested=int(n_hidden),
            )
        elif n_hidden > 0:
            placements = place_hidden_targets(
                a_init,
                known_belief_tasks,
                launch_point,
                PlacementParameters(detection_km=float(detection_km)),
                placement_rng,
            )
            if len(placements) != n_hidden:
                raise RuntimeError(
                    f"setup_episode: B2 produced {len(placements)} placement(s) for "
                    f"n_hidden={n_hidden}. Its locked contract is exactly ONE placement "
                    f"per non-empty ego route, so the solved allocation left "
                    f"{n_hidden - len(placements)} of the {len(agents1)} ego(s) without a "
                    "route. Distributing a different n_hidden across routes is a separate "
                    "design task — refusing to truncate, pad or duplicate."
                )

        # --- Patch ONCE; never regenerate after solving ---------------------------
        patched_json = build_patched_scenario(
            scenario_json, placements, attacking_side_color=attacking_side_color
        )
    finally:
        _close_quietly(env1)

    # ================= Environment 2 — AUTHORITATIVE =================
    game2, env2, obs2 = _build_env(
        patched_json,
        max_episode_steps=max_episode_steps,
        attacking_side_color=attacking_side_color,
        record_every_seconds=record_every_seconds,
        recording_export_path=recording_export_path,
    )
    try:
        agents, all_tasks = _extract_world(obs2, attacking_side_color)
        _require_agent_ids_preserved(env1_agent_ids, [str(a.id) for a in agents])

        # The reload must have preserved every known target and added exactly the hidden ones.
        world_ids_2 = {_task_target_id(t) for t in all_tasks}
        missing = [tid for tid in known_world_ids if tid not in world_ids_2]
        if missing:
            raise RuntimeError(
                f"setup_episode: known target(s) {missing} vanished across patch/reload"
            )
        added = len(world_ids_2) - len(set(known_world_ids))
        if len(all_tasks) != len(known_world_ids) + len(placements) or added != len(placements):
            raise RuntimeError(
                f"setup_episode: reloaded world holds {len(all_tasks)} target(s) "
                f"({added} new) — expected {len(known_world_ids)} known + "
                f"{len(placements)} hidden"
            )

        # The t=0 EXECUTED-WORLD snapshot, taken from the RAW env-2 extraction — before
        # the oracle solve below, whose output is likewise allocated-only. Env-2 is the
        # sole runtime source of truth, so this list IS the world the episode runs on.
        executed_target_ids = _world_target_ids(all_tasks, "executed world")

        # A_init's positional indices, re-pointed at ENV-2 Task objects.
        belief_tasks = _rematerialize_known_tasks(all_tasks, known_target_order)

        # --- Solve the FULL env-2 set -> oracle (the reward denominator) ----------
        # A SEPARATE, independent solve over the reloaded world, so the oracle can never
        # be an alias of a_init and every hidden target is in it.
        oracle_solution, oracle_tasks, _ = solve_and_normalize(agents, all_tasks)

        split_meta: Dict[str, Any] = {
            # Truthful provenance: `split_tasks` did NOT run. The count keys below carry
            # the legacy names because training / rollout records read them, but they
            # describe WORLD TARGETS EMITTED by the construction, not a split attempt.
            "outcome": "construction",
            "mode": "construction",
            "known": len(known_world_ids),
            "hidden": len(placements),
            "partial": len(known_world_ids),
            "full": len(all_tasks),
            "n_hidden_requested": int(n_hidden),
            "allocated_known": len(known_target_order),
            # Id-free reproducibility identity (CLAUDE.md section 8: uuids are not
            # seed-derived, so they can never be the comparison key).
            "geometric_fingerprint": geometric_fingerprint(placements),
        }

        construction_audit: Optional[ConstructionAudit] = None
        if backoff is not None:
            # Reconciled against the RAW world snapshots, never against the allocated-only
            # `belief_tasks` / `oracle_tasks`: the audit's whole job is to state what the
            # world REALLY holds beside what was REQUESTED, so deriving either count from
            # an allocation would reintroduce the exact defect the snapshots exist to
            # close. Verified rather than trusted — a mismatch means the patch, the reload
            # or the accounting is wrong, and it is refused here.
            hidden_realized = len(placements)
            expected_total = len(known_world_ids) + hidden_realized
            if len(executed_target_ids) != expected_total:
                raise RuntimeError(
                    f"setup_episode: generalized accounting does not reconcile — the raw "
                    f"executed world holds {len(executed_target_ids)} target(s) but "
                    f"{len(known_world_ids)} known + {hidden_realized} realized hidden = "
                    f"{expected_total} were constructed"
                )
            if backoff.hidden_realized != hidden_realized:
                raise RuntimeError(
                    f"setup_episode: backoff audit claims {backoff.hidden_realized} "
                    f"realized hidden target(s) but {hidden_realized} were placed"
                )
            construction_audit = ConstructionAudit(
                policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
                agent_count=len(agents),
                # K == A is the approved rule, already enforced against the raw known
                # world before the solve, so the requested known count IS the agent count.
                known_requested=len(agents),
                known_realized=len(known_target_ids),
                hidden_requested=int(n_hidden),
                hidden_realized=hidden_realized,
                total_requested=len(agents) + int(n_hidden),
                total_realized=len(executed_target_ids),
                backoff=backoff,
            )
            # Generalized-only keys. The exact path's `split_meta` is deliberately left
            # exactly as it was, so nothing reading a historical record sees new fields.
            split_meta["hidden_policy"] = HIDDEN_POLICY_BOUNDED_BACKOFF_V1
            split_meta["hidden_realized"] = hidden_realized
            split_meta["construction_audit"] = construction_audit.as_dict()

        return _finish_context(
            game=game2,
            env=env2,
            obs=obs2,
            agents=agents,
            a_init=a_init,
            belief_tasks=belief_tasks,
            oracle_solution=oracle_solution,
            oracle_tasks=oracle_tasks,
            split_meta=split_meta,
            detection_km=detection_km,
            recording_export_path=recording_export_path,
            placements=placements,
            known_target_ids=known_target_ids,
            executed_target_ids=executed_target_ids,
            construction_audit=construction_audit,
        )
    except BaseException:
        _close_quietly(env2)
        raise


def _finish_context(
    *,
    game: Any,
    env: Any,
    obs: Any,
    agents: List[Agent],
    a_init: Dict[str, List[Assignment]],
    belief_tasks: List[Task],
    oracle_solution: Dict[str, List[Assignment]],
    oracle_tasks: List[Task],
    split_meta: Dict[str, Any],
    detection_km: float,
    recording_export_path: Optional[str],
    placements: Tuple[HiddenPlacement, ...],
    known_target_ids: Tuple[str, ...],
    executed_target_ids: Tuple[str, ...],
    construction_audit: Optional[ConstructionAudit] = None,
) -> EpisodeContext:
    """Mint the N independent beliefs + the ONE executor and package the context.

    Shared by both paths so the belief/executor construction — the no-communication
    foundation — has exactly ONE implementation regardless of how the world was built.

    ``known_target_ids`` / ``executed_target_ids`` are the RAW pre-solve world snapshots
    (see :class:`EpisodeContext`). They are REQUIRED keywords rather than defaulted, so a
    future third path cannot reach a context that silently carries an empty world
    inventory — the one shape in which allocated-only data gets read as world truth again.
    Verified here rather than trusted: both must be non-empty, and the known half must be
    a subset of the executed half.
    """
    known_snapshot = tuple(str(t) for t in known_target_ids)
    executed_snapshot = tuple(str(t) for t in executed_target_ids)
    if not executed_snapshot:
        raise RuntimeError(
            "setup_episode: the executed-world target snapshot is empty, so nothing "
            "downstream could state what world this episode runs on"
        )
    outside = [t for t in known_snapshot if t not in set(executed_snapshot)]
    if outside:
        raise RuntimeError(
            f"setup_episode: {len(outside)} known target(s) {outside[:3]} are absent "
            "from the executed world snapshot; the t=0 roster would not cover what runs"
        )
    # N mutually-independent beliefs, one per ego id. All egos start byte-equal to
    # a_init at t=0, but each belief is a fresh deepcopy of the tasks + a fresh
    # _copy_solution of a_init, so a per-ego edit never leaks.
    beliefs: Dict[str, Belief] = {
        str(a.id): Belief.independent(belief_tasks, a_init) for a in agents
    }

    # ONE executor over the normalized (allocated-only) baseline. It fans belief_tasks
    # out to per-ego private task lists internally.
    # arrival_threshold_km == detection_km == the single unified radius.
    executor = GraphPlanExecutor(
        solution=a_init,
        tasks=belief_tasks,
        agents=agents,
        arrival_threshold_km=detection_km,
    )

    return EpisodeContext(
        env=env,
        game=game,
        observation=obs,  # seed for the tick-loop's first sense; loop never re-resets
        agents=agents,
        agent_ids=[str(a.id) for a in agents],
        beliefs=beliefs,
        executor=executor,
        a_init=a_init,
        oracle_solution=oracle_solution,
        oracle_tasks=oracle_tasks,
        split_meta=split_meta,
        # Single source of truth: recording is armed iff an export path was given.
        record=recording_export_path is not None,
        placements=placements,
        known_target_ids=known_snapshot,
        executed_target_ids=executed_snapshot,
        construction_audit=construction_audit,
    )


# =============================================================================
# Self-test (bonmin path; generates one real scenario, like graph_builder's)
# =============================================================================

def _selftest() -> None:
    """Run ``setup_episode`` on one generated scenario and assert the invariants.

    Run under nlp_env (needs bonmin) from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.training.graph_episode_setup
    """
    import copy
    import tempfile
    from pathlib import Path

    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # PlaybackRecorder CHARACTER_LIMIT override (historical flat-era convention)

    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig,
    )
    from match_aou.models import Location, Step, StepKind, Task as _Task

    repo_root = Path(__file__).resolve().parents[4]
    base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graph_setup_selftest_")

    print("=" * 72)
    print("graph_episode_setup self-test")
    print("=" * 72)

    # --- Generate one scenario variation (RED airbases only, no SAMs) ---
    gen = ScenarioGenerator(
        base_scenario_path=str(base_scenario),
        output_dir=out_dir,
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    # detection_km=DETECTION_KM: the generator builds discovery connectivity at the
    # SAME radius the split checks and the runtime senses at (single-radius invariant).
    cfg = VariationConfig(
        include_sams=False,
        num_red_airbases=(3, 3),
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        detection_km=DETECTION_KM,
        seed=0,
    )
    scenario_path = str(gen.generate(episode=0, config=cfg))
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_json = f.read()

    # --- Independent two-solve check on the helper itself (object non-aliasing) ---
    # setup_episode relies on solve_and_normalize returning a FRESH object per call;
    # prove that here at the unit level before trusting the wired oracle-vs-A_init.
    ctx = setup_episode(
        scenario_json,
        recording_export_path=out_dir,
    )

    agent_ids = ctx.agent_ids
    n_blue = len(agent_ids)
    print(f"[env] blue agents: {n_blue} ({[a[:8] for a in agent_ids]})")
    print(f"[solve] belief tasks (allocated-only): {len(ctx.executor.tasks[agent_ids[0]])}, "
          f"assignments in A_init: {sum(len(v) for v in ctx.a_init.values())}")
    assert n_blue >= 1, "expected at least one blue agent"

    # (1) N beliefs exist (one per blue agent id) and are ALL EQUAL at t=0.
    assert set(ctx.beliefs.keys()) == set(agent_ids), (ctx.beliefs.keys(), agent_ids)
    assert len(ctx.beliefs) == n_blue

    def _tids(tasks: List[Task]) -> List[List[str]]:
        return [[str(s.target_id) for s in t.steps] for t in tasks]

    ref_sol = ctx.beliefs[agent_ids[0]].solution
    ref_tids = _tids(ctx.beliefs[agent_ids[0]].tasks)
    for aid in agent_ids:
        assert ctx.beliefs[aid].solution == ref_sol, f"belief {aid} solution differs at t=0"
        assert _tids(ctx.beliefs[aid].tasks) == ref_tids, f"belief {aid} tasks differ at t=0"
    print(f"[1] {n_blue} beliefs, one per blue agent, all EQUAL at t=0   OK")

    # (2) INDEPENDENCE: mutate belief[A]; every other belief must be byte-unchanged.
    a_id = agent_ids[0]
    # Snapshot every OTHER ego's belief (deep) BEFORE mutating A.
    snap_sol = {aid: copy.deepcopy(ctx.beliefs[aid].solution) for aid in agent_ids}
    snap_len = {aid: len(ctx.beliefs[aid].tasks) for aid in agent_ids}
    snap_tids = {aid: _tids(ctx.beliefs[aid].tasks) for aid in agent_ids}

    # Distinct underlying objects (no shared mutable state between beliefs).
    for other in agent_ids[1:]:
        assert ctx.beliefs[a_id].tasks is not ctx.beliefs[other].tasks
        assert ctx.beliefs[a_id].solution is not ctx.beliefs[other].solution

    # deepcopy defense-in-depth: the shared t=0 Task OBJECTS must be distinct per ego,
    # not shared references (else an in-place Task edit would leak across egos).
    for other in agent_ids[1:]:
        for i in range(len(ctx.beliefs[a_id].tasks)):
            assert ctx.beliefs[a_id].tasks[i] is not ctx.beliefs[other].tasks[i], (
                f"Task object {i} is shared between {a_id} and {other} — deepcopy regressed"
            )

    # Edit A: append a dummy pop-up task AND add an assignment to A's solution.
    dummy = _Task(
        steps=[Step(Location(0.0, 0.0), "DUMMY_POPUP", [], 1.0, 1, StepKind.ATTACK)],
        utility=1,
    )
    ctx.beliefs[a_id].tasks.append(dummy)
    ctx.beliefs[a_id].solution.setdefault(a_id, []).append((999, 0, -1))

    for other in agent_ids[1:]:
        assert ctx.beliefs[other].solution == snap_sol[other], \
            f"INDEPENDENCE VIOLATED: belief {other} solution changed after editing {a_id}"
        assert len(ctx.beliefs[other].tasks) == snap_len[other], \
            f"INDEPENDENCE VIOLATED: belief {other} task count changed after editing {a_id}"
        assert _tids(ctx.beliefs[other].tasks) == snap_tids[other], \
            f"INDEPENDENCE VIOLATED: belief {other} tasks changed after editing {a_id}"
    # A itself did change (sanity: the mutation actually happened).
    assert len(ctx.beliefs[a_id].tasks) == snap_len[a_id] + 1
    assert (999, 0, -1) in ctx.beliefs[a_id].solution[a_id]
    print(f"[2] object-distinct tasks; editing belief[{a_id[:8]}] left all "
          f"{n_blue - 1} peer beliefs byte-identical   OK")

    # (3) belief_tasks is ALLOCATED-ONLY: every task_idx in A_init is a valid index,
    #     and the set of referenced task_idx == the whole belief_tasks range (no orphan
    #     selected-but-unassigned task). Uses the executor's per-ego belief list.
    belief_tasks = ctx.executor.tasks[a_id]  # fanned-out copy of the normalized list
    n_tasks = len(belief_tasks)
    referenced = {int(t[0]) for tuples in ctx.a_init.values() for t in tuples}
    assert n_tasks >= 1, "solver selected no tasks — scenario too hard for a meaningful test"
    assert all(0 <= i < n_tasks for i in referenced), (referenced, n_tasks)
    assert referenced == set(range(n_tasks)), \
        f"allocated-only violated: referenced={sorted(referenced)} vs range(0,{n_tasks})"
    print(f"[3] belief_tasks allocated-only: {n_tasks} tasks, all referenced by A_init   OK")

    # (4) oracle_solution produced and NOT the same object as A_init.
    assert ctx.oracle_solution, "oracle_solution is empty"
    assert ctx.oracle_solution is not ctx.a_init, "oracle is aliased to A_init!"
    # Also object-distinct from every belief's / the executor's live plan dict.
    for aid in agent_ids:
        assert ctx.oracle_solution is not ctx.beliefs[aid].solution
    assert ctx.oracle_solution is not ctx.executor.plans
    print(f"[4] oracle_solution produced, distinct object from A_init "
          f"({sum(len(v) for v in ctx.oracle_solution.values())} assignments)   OK")

    # (5) Executor constructed; is_done() is False at t=0 (work remains, nobody home).
    #     Completion is PHYSICAL, so the check reads the reset observation.
    assert isinstance(ctx.executor, GraphPlanExecutor)
    assert ctx.executor.is_done(ctx.observation) is False, \
        "executor claims done at t=0 (no work?!)"
    print("[5] executor constructed; is_done() is False at t=0   OK")

    # (6) REAL SPLIT (Test 2): partial ⊊ full, and A_init covers only KNOWN targets.
    #     The full enemy set is recomputed from the SAME reset observation the setup
    #     split ran on; the belief/A_init task universe must exclude every hidden id.
    meta = ctx.split_meta
    print(f"[split] meta: {meta}")
    assert meta.get("outcome") in {"clean", "resampled", "exhaust", "warn-fallback",
                                    "no-chain"}, meta
    # counts are self-consistent: known + hidden == full == partial + hidden.
    assert meta["known"] + meta["hidden"] == meta["full"], meta
    assert meta["partial"] == meta["known"], meta
    # With 3 airbases + stretch 0.5 the geometry always hides ≥1 target (num_partial=2).
    assert meta["hidden"] >= 1, f"expected hidden>0 with this geometry, got {meta}"
    assert meta["partial"] < meta["full"], f"partial not a strict subset of full: {meta}"

    # A_init covers only known targets: belief/A_init target ids ⊊ the full enemy set.
    full_ids = {
        str(s.target_id)
        for t in generate_all_enemy_tasks(
            ctx.observation, attacking_side_color=ATTACKING_SIDE_COLOR, probability=1.0
        )
        for s in t.steps
    }
    belief_ids = {str(s.target_id) for t in belief_tasks for s in t.steps}
    assert len(full_ids) == meta["full"], (len(full_ids), meta["full"])
    assert belief_ids <= full_ids, (belief_ids - full_ids)
    # allocated-only A_init sees at most the known targets, never all of them.
    assert len(belief_ids) <= meta["known"], (len(belief_ids), meta["known"])
    assert belief_ids != full_ids, "A_init covers ALL targets - hidden ones leaked in!"
    print(f"[6] real split: partial={meta['partial']} < full={meta['full']} "
          f"(hidden={meta['hidden']}); A_init sees {len(belief_ids)}/{len(full_ids)} "
          f"targets, none hidden   OK")

    # --- Bonus: solve_and_normalize non-aliasing at the unit level -------------
    # Two calls on the same inputs must return DISTINCT objects (setup relies on this
    # for the two-independent-solves invariant above).
    s1, _t1, _ = solve_and_normalize(ctx.agents, belief_tasks)
    s2, _t2, _ = solve_and_normalize(ctx.agents, belief_tasks)
    assert s1 is not s2, "solve_and_normalize returned an aliased solution object"
    print("[bonus] solve_and_normalize returns a fresh solution object per call   OK")

    ctx.env.close()
    print("-" * 72)
    print("All assertions passed.")


def _selftest_split() -> None:
    """Test 1: the discovery-chain split in isolation (no BLADE, no bonmin).

    Hand-builds tasks with known great-circle spacing and asserts the split's red
    lines directly: hidden targets keep a known neighbour, isolated targets are
    pinned, partial ⊆ full with consistent counts, and a tight radius that isolates
    everything falls into the ``exhaust`` path.
    """
    from match_aou.models import Location, Step, StepKind, Task as _Task

    print("=" * 72)
    print("split_tasks unit test (Test 1 - no BLADE/bonmin)")
    print("=" * 72)

    def _mk(lon: float) -> _Task:
        # lat=0 everywhere; at the equator Δlon deg ≈ Δlon * 111.32 km great-circle.
        return _Task(
            steps=[Step(Location(0.0, lon), f"T{lon:g}", [], 1.0, 1, StepKind.ATTACK)],
            utility=80,
        )

    DET = 50.0

    # Two well-separated pairs. Within a pair ≈33 km (≤DET); pairs ≈556 km apart.
    # A naive random draw can hide a whole pair (both hidden, no known neighbour) —
    # the rejection sampler must reject those draws.
    two_pairs = [_mk(0.0), _mk(0.3), _mk(5.0), _mk(5.3)]  # P1={0,1}, P2={2,3}
    L = [t.steps[0].location for t in two_pairs]
    d01, d23, d02 = L[0].distance_to(L[1]), L[2].distance_to(L[3]), L[0].distance_to(L[2])
    assert d01 <= DET and d23 <= DET and d02 > DET, (d01, d23, d02)

    # (a) every hidden target keeps a known neighbour within DET — across many seeds.
    last_meta = None
    for seed in range(50):
        random.seed(seed)
        partial, full, meta = split_tasks(two_pairs, 2.0 / 3.0, detection_km=DET)
        last_meta = meta
        assert full == list(two_pairs) and len(full) == 4
        assert all(any(p is f for f in full) for p in partial)  # partial subset-of full
        known = {i for i, t in enumerate(two_pairs) if any(t is p for p in partial)}
        hidden = set(range(4)) - known
        for h in hidden:
            hl = two_pairs[h].steps[0].location
            assert any(two_pairs[k].steps[0].location.distance_to(hl) <= DET
                       for k in known), \
                f"seed {seed}: hidden {h} has no known neighbour within {DET} km"
        assert meta["known"] == len(partial) == meta["partial"]
        assert meta["hidden"] == 4 - len(partial)
        assert meta["known"] + meta["hidden"] == meta["full"] == 4
        assert meta["outcome"] in ("clean", "resampled")
    print(f"[1a] hidden targets always keep a known neighbour within DET (50 seeds)  OK")
    print(f"     last meta: {last_meta}")

    # (b) isolated target is PINNED to known. Close pair + one far isolated (~2226 km).
    with_isolated = [_mk(0.0), _mk(0.3), _mk(20.0)]
    far = with_isolated[2]
    for seed in range(20):
        random.seed(seed)
        partial, full, meta = split_tasks(with_isolated, 2.0 / 3.0, detection_km=DET)
        assert any(far is p for p in partial), f"seed {seed}: isolated target not pinned"
        assert meta["isolated_pinned"] == 1
    print("[1b] isolated target pinned into the known set (20 seeds)  OK")

    # (c) partial ⊆ full and meta counts consistent.
    random.seed(0)
    partial, full, meta = split_tasks(two_pairs, 2.0 / 3.0, detection_km=DET)
    assert {id(t) for t in partial} <= {id(t) for t in full}
    assert meta["partial"] + meta["hidden"] == meta["full"] == len(full)
    print("[1c] partial subset-of full, meta counts consistent  OK")

    # (d) a tight radius isolates EVERYTHING → exhaust path (isolated > partial budget).
    random.seed(0)
    partial, full, meta = split_tasks(two_pairs, 2.0 / 3.0, detection_km=1.0)
    n = len(two_pairs)
    num_partial = max(1, int(n * 2.0 / 3.0))
    assert meta["outcome"] == "exhaust", meta
    assert meta["isolated_pinned"] == num_partial == len(partial)
    assert meta["hidden"] == n - num_partial
    assert {id(t) for t in partial} <= {id(t) for t in full}
    print(f"[1d] tight radius -> all isolated -> 'exhaust', {num_partial} pinned  OK")

    # (e) degenerate n<2 → 'no-chain', nothing hidden.
    p1, f1, m1 = split_tasks([_mk(0.0)], 2.0 / 3.0, detection_km=DET)
    assert m1["outcome"] == "no-chain" and m1["hidden"] == 0 and len(p1) == 1
    p0, f0, m0 = split_tasks([], 2.0 / 3.0, detection_km=DET)
    assert m0["hidden"] == 0 and p0 == [] and f0 == []
    print("[1e] degenerate n<2 -> 'no-chain', nothing hidden  OK")

    print("-" * 72)
    print("Test 1 (split unit) passed.")


def _selftest_generator() -> None:
    """Test 3: the generator builds discovery connectivity at DETECTION_KM (no bonmin).

    Proves the connectivity-radius SOURCE switched: with ``detection_km`` set the
    stat is exactly that radius; with it ``None`` the legacy ``aircraft.range`` value
    is used (and differs). A geometric spot-check confirms real ≤DETECTION_KM pairs.
    """
    import json
    import tempfile
    from pathlib import Path

    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig,
    )
    from match_aou.models import Location

    print("=" * 72)
    print("generator connectivity radius test (Test 3 - no bonmin)")
    print("=" * 72)

    repo_root = Path(__file__).resolve().parents[4]
    base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graph_gen_selftest_")

    gen = ScenarioGenerator(
        base_scenario_path=str(base_scenario), output_dir=out_dir,
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    common = dict(include_sams=False, num_red_airbases=(4, 4),
                  randomize_red_airbase_positions=True, stretch_target_ratio=0.5, seed=7)

    # (a) detection_km=DETECTION_KM → connectivity built at exactly that radius.
    gen.generate(episode=0, config=VariationConfig(detection_km=DETECTION_KM, **common))
    stat50 = gen.last_generation_stats["min_radar_km"]
    assert stat50 == DETECTION_KM, stat50
    print(f"[3a] detection_km={DETECTION_KM} -> connectivity radius stat == {stat50}  OK")

    # (b) detection_km=None → legacy aircraft.range-derived radius (and ≠ DETECTION_KM).
    gen.generate(episode=1, config=VariationConfig(detection_km=None, **common))
    legacy = gen.last_generation_stats["min_radar_km"]
    assert legacy > 0 and legacy != DETECTION_KM, legacy
    print(f"[3b] detection_km=None -> legacy radius {legacy:.1f} km (aircraft.range, != 50)  OK")

    # (c) geometric spot-check: with detection_km=50 the generated same-zone targets
    #     actually sit within 50 km of a neighbour (connectivity produced ≤50 km pairs).
    path50 = gen.generate(episode=2, config=VariationConfig(detection_km=DETECTION_KM, **common))
    with open(path50, "r", encoding="utf-8") as f:
        sc = json.load(f)["currentScenario"]
    airbases = gen._get_red_airbases(sc)
    locs = [Location(ab["latitude"], ab["longitude"]) for ab in airbases]
    assert len(locs) >= 2, len(locs)
    # At least one ≤50 km pair must exist (2+2 zones ⇒ each zone is a connected pair).
    close_pairs = sum(
        1 for i in range(len(locs)) for j in range(i + 1, len(locs))
        if locs[i].distance_to(locs[j]) <= DETECTION_KM
    )
    assert close_pairs >= 1, f"no <={DETECTION_KM} km neighbour pair among {len(locs)} targets"
    print(f"[3c] generated targets have {close_pairs} neighbour pair(s) <={DETECTION_KM} km  OK")

    print("-" * 72)
    print("Test 3 (generator connectivity) passed.")


if __name__ == "__main__":
    _selftest_split()       # Test 1 — pure, no BLADE/bonmin
    _selftest_generator()   # Test 3 — generation only, no bonmin
    _selftest()             # existing self-test + Test 2 (bonmin)
