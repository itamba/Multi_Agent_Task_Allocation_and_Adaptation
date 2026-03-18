"""blade_executor.py

State-based (observation-driven) execution for MATCH-AOU plans inside the Panopticon
BLADE simulator.

Why this exists
---------------
The dictionary plan approach (tick -> action) is great for deterministic playback,
but it is *not* robust under uncertainty (travel time variability, target motion,
fuel burn while waiting, etc.).

This executor is **event-driven**:
- A plan is a per-agent ordered list of assigned steps:
      (task_idx, step_idx, level_order)
- Each sim timestep we pick **at most one global action** based on the current
  observation and per-agent FSM state.
- `level_order` is treated as a precedence *layer* (0,1,2,...) rather than a
  physical timestamp.

Core requirements enforced
--------------------------
1) **Global constraint:** BLADE supports **one action per timestep**.
2) **Precedence gating:** never execute a higher `level_order` before prior layers
   are completed (configurable semantics).
3) **Launch-before-move:** if aircraft start stored in an airbase, we launch them
   only when their next eligible step is ready (reduces loiter/fuel burn).

Assumptions / conventions
-------------------------
- After `launch_aircraft_from_airbase(airbase_id)`, the aircraft appears in
  `observation.aircraft` with the same id as the MATCH-AOU Agent (`agent.id`).
- `env.step("")` is a valid No-Op.
- Attack actions are strings like `handle_aircraft_attack(...)`.
  Completion is currently marked when the action is *issued* (best-effort).

Notes on robustness
-------------------
- Some scenarios may omit `Agent.home_base_id` (or `Agent.weapon_id`). We try to
  infer them from the observation when needed.
- If `level_barrier_mode="assignment"` is used together with
  `allow_redundant_assignments=False`, we must still "count" skipped redundant
  assignments, otherwise the barrier can deadlock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Mapping

from ...models import Agent, Location, Task


# Public type used across the project:
# (task_idx, step_idx, level_order)
Assignment = Tuple[int, int, int]


# --------------------------------------------------------------------------------------
# Executor types
# --------------------------------------------------------------------------------------

class AgentPhase(str, Enum):
    """High-level FSM phases for logging/debug."""

    ON_GROUND = "ON_GROUND"
    MOVING = "MOVING"
    READY = "READY"
    ATTACKING = "ATTACKING"
    RTB = "RTB"
    DONE = "DONE"


class ActionKind(str, Enum):
    LAUNCH = "LAUNCH"
    MOVE = "MOVE"
    ATTACK = "ATTACK"
    RTB = "RTB"
    OTHER = "OTHER"


@dataclass
class AgentState:
    """Mutable per-agent execution state.

    Design principle:
    - BLADE actions are treated as **one-shot**. Once we call env.step(action), we assume it was issued.
    - Therefore we prevent re-issuing LAUNCH/MOVE for the same intent using local flags,
      without verifying autopilot state in the environment.
    """

    phase: AgentPhase = AgentPhase.ON_GROUND
    assignment_idx: int = 0

    # Launch: issue once, never retry (unless plan explicitly says to launch again).
    launch_issued: bool = False

    # Move: prevent spam by remembering that we already issued MOVE for the current (task,step) and goal.
    move_active_key: Optional[Tuple[int, int]] = None   # (task_idx, step_idx)
    move_active_goal: Optional[Tuple[float, float]] = None  # (lat, lon) if known

    # RTB bookkeeping
    rtb_issued: bool = False

    # Fairness
    ready_since_tick: Optional[int] = None

    # Debug
    last_action: str = ""


@dataclass(frozen=True)
class CandidateAction:

    """A ready-to-issue action candidate produced by an agent policy."""

    agent_id: str
    level: int
    kind: ActionKind
    action: str
    reason: str
    ready_since_tick: int


    meta: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# BLADE Scenario helpers
# --------------------------------------------------------------------------------------

def _get_sim_tick(observation: Any, *, fallback_tick: int) -> int:
    """
    BLADE Scenario does not expose `t`.
    Use (current_time - start_time) if present, else fallback to loop counter.
    """
    try:
        ct = int(getattr(observation, "current_time"))
        st = int(getattr(observation, "start_time"))
        return max(0, ct - st)
    except Exception:
        return int(fallback_tick)

def _find_aircraft_obj(scenario: Any, aircraft_id: str) -> Optional[Any]:
    for ac in getattr(scenario, "aircraft", []) or []:
        if str(getattr(ac, "id", "")) == str(aircraft_id):
            return ac
    return None

def _get_aircraft_location(scenario: Any, aircraft_id: str) -> Optional[Location]:
    """Best-effort lookup of aircraft current location by id in observation.aircraft."""

    ac = _find_aircraft_obj(scenario, aircraft_id)
    if ac is None:
        return None
    return Location(
        getattr(ac, "latitude", 0),
        getattr(ac, "longitude", 0),
        getattr(ac, "altitude", 0) or 0,
    )

def _aircraft_airborne(scenario: Any, aircraft_id: str) -> bool:
    return _get_aircraft_location(scenario, aircraft_id) is not None

def _infer_airbase_id_for_aircraft(scenario: Any, aircraft_id: str) -> Optional[str]:
    """If the aircraft is stored inside an airbase inventory, return that airbase id."""

    for base in getattr(scenario, "airbases", []) or []:
        for ac in getattr(base, "aircraft", []) or []:
            if str(getattr(ac, "id", "")) == str(aircraft_id):
                bid = getattr(base, "id", None)
                return str(bid) if bid is not None else None
    return None

def _aircraft_in_any_airbase(scenario: Any, aircraft_id: str) -> bool:
    return _infer_airbase_id_for_aircraft(scenario, aircraft_id) is not None

def _infer_weapon_id_for_unit(scenario: Any, unit_id: str) -> Optional[str]:
    """Try to infer a usable weapon id from the live unit object in the observation."""

    unit = _find_aircraft_obj(scenario, unit_id)
    if unit is None:
        # fallback: ships
        for ship in getattr(scenario, "ships", []) or []:
            if str(getattr(ship, "id", "")) == str(unit_id):
                unit = ship
                break

    if unit is None:
        return None

    if hasattr(unit, "get_weapon_with_highest_engagement_range"):
        try:
            best = unit.get_weapon_with_highest_engagement_range()
            wid = getattr(best, "id", None)
            return str(wid) if wid is not None else None
        except Exception:
            pass

    # fallback: first weapon
    weapons = getattr(unit, "weapons", []) or []
    if weapons:
        wid = getattr(weapons[0], "id", None)
        return str(wid) if wid is not None else None
    return None

def _find_unit_location(scenario: Any, unit_id: str) -> Optional[Location]:
    """Best-effort lookup by id across common BLADE unit collections."""

    for coll_name in ("aircraft", "ships", "facilities", "airbases"):
        for u in getattr(scenario, coll_name, []) or []:
            if str(getattr(u, "id", "")) == str(unit_id):
                return Location(
                    getattr(u, "latitude", 0),
                    getattr(u, "longitude", 0),
                    getattr(u, "altitude", 0) or 0,
                )
    return None

def _build_validated_launch_action_for_aircraft(
    scenario: Any,
    aircraft_id: str,
    *,
    airbase_id: Optional[str] = None,
) -> str:
    """
    Build a SINGLE BLADE action-string to launch from an airbase, with strict FIFO validation.

    Rules (project invariant):
    - Only one method-call action string is allowed (no Python statements).
    - launch_aircraft_from_airbase(airbase_id) launches FIFO: airbase.aircraft.pop(0)
    - There is no API to choose aircraft_id.
    - Therefore: we ONLY launch if the requested aircraft is FIRST in the airbase inventory.
      Otherwise, raise ValueError and DO NOT launch.

    Args:
        scenario: current Scenario observation.
        aircraft_id: the specific aircraft we intend to launch.
        airbase_id: optional. If not provided, infer the airbase containing this aircraft.

    Returns:
        A single action string: launch_aircraft_from_airbase("AIRBASE_ID")

    Raises:
        ValueError: if aircraft not found in any airbase, airbase has no inventory,
                    or the aircraft is not first in FIFO order.
    """
    ac_id = str(aircraft_id)

    # Resolve airbase
    resolved_ab_id = str(airbase_id) if airbase_id is not None else _infer_airbase_id_for_aircraft(scenario, ac_id)
    if not resolved_ab_id:
        raise ValueError(f"Aircraft {ac_id} is not present in any airbase inventory; cannot launch.")

    # Find the airbase object
    ab_obj = None
    for base in getattr(scenario, "airbases", []) or []:
        if str(getattr(base, "id", "")) == resolved_ab_id:
            ab_obj = base
            break
    if ab_obj is None:
        raise ValueError(f"Airbase {resolved_ab_id} not found in scenario.airbases; cannot launch aircraft {ac_id}.")

    inv = getattr(ab_obj, "aircraft", None) or []
    if not inv:
        raise ValueError(f"Airbase {resolved_ab_id} has empty aircraft inventory; cannot launch aircraft {ac_id}.")

    head = inv[0]
    head_id = str(getattr(head, "id", ""))
    if head_id != ac_id:
        # Helpful debug context: show first few IDs in queue
        preview = [str(getattr(a, "id", "")) for a in inv[:5]]
        raise ValueError(
            f"Cannot launch aircraft {ac_id} from airbase {resolved_ab_id}: "
            f"FIFO head is {head_id}. Queue head preview={preview}"
        )

    # Return a SINGLE method-call action string
    return f'launch_aircraft_from_airbase("{resolved_ab_id}")'


# # --------------------------------------------------------------------------------------
# # Parsing / observation helpers
# # --------------------------------------------------------------------------------------
#
# _ATTACK_RE = re.compile(
#     r"^\s*handle_(?:aircraft|ship)_attack\(\s*['\"]?([^,'\"\)]+)['\"]?\s*,\s*['\"]?([^,'\"\)]+)['\"]?\s*(?:,|\))"
# )
#
#
# def _parse_attack_target_id(action: str) -> Optional[str]:
#     """Extract target_id from handle_*_attack('unit_id','target_id',...) strings."""
#
#     m = _ATTACK_RE.match(action.strip())
#     if not m:
#         return None
#     return str(m.group(2)).strip()
#
#
# def _replace_placeholders(action: str, *, agent_id: str, weapon_id: Optional[str]) -> str:
#     """Replace AGENT_ID / WEAPON_ID placeholders inside action strings."""
#
#     out = str(action).replace("AGENT_ID", str(agent_id))
#     if "WEAPON_ID" in out:
#         if weapon_id is None:
#             raise ValueError(f"Action requires WEAPON_ID but agent {agent_id} has no weapon_id")
#         out = out.replace("WEAPON_ID", str(weapon_id))
#     return out


# ---------------------------- State-based executor ----------------------------

class BladeStateExecutor:
    """Observation-driven executor with a global one-action-per-timestep constraint."""

    def __init__(
        self,
        *,
        tasks: List[Task],
        solution: Dict[str, List[Assignment]],
        agents: Sequence[Agent],
        precedence_relations: Optional[Sequence[Tuple[int, int]]] = None,
        # Movement / gating
        arrival_threshold_km: float = 5.0,
        attack_range_km: float = 50.0,
        move_reissue_interval: int = 30,
        # Launch behavior
        post_launch_wait_ticks: int = 5,
        max_launch_attempts: int = 3,
        # Plan semantics
        replace_placeholders: bool = True,
        add_return_to_base: bool = True,
        allow_redundant_assignments: bool = False,
        # Level barrier semantics:
        # - "task": unlock next level when all *tasks* in this level are completed (recommended).
        # - "assignment": unlock next level when all *assignments* in this level are completed (strict).
        level_barrier_mode: str = "task",
        # Arbitration among ready agents (action is global):
        arbitration_policy: str = "round_robin",  # "round_robin" | "priority"
        # Logging
        logger: Optional[logging.Logger] = None,
        log_every_n_ticks: int = 10,
    ) -> None:
        self.arrival_threshold_km = float(arrival_threshold_km)
        self.attack_range_km = float(attack_range_km)
        self.move_reissue_interval = max(int(move_reissue_interval), 1)

        self.post_launch_wait_ticks = max(int(post_launch_wait_ticks), 0)
        self.max_launch_attempts = max(int(max_launch_attempts), 1)

        self.replace_placeholders = bool(replace_placeholders)
        self.add_return_to_base = bool(add_return_to_base)
        self.allow_redundant_assignments = bool(allow_redundant_assignments)

        self.level_barrier_mode = str(level_barrier_mode).lower().strip()
        if self.level_barrier_mode not in ("task", "assignment"):
            raise ValueError("level_barrier_mode must be one of: 'task', 'assignment'")

        self.arbitration_policy = str(arbitration_policy).lower().strip()
        if self.arbitration_policy not in ("round_robin", "priority"):
            raise ValueError("arbitration_policy must be one of: 'round_robin', 'priority'")

        self.log = logger or logging.getLogger("blade_executor")
        self.log_every_n_ticks = max(int(log_every_n_ticks), 0)

        self._agent_by_id: Dict[str, Agent] = {str(a.id): a for a in agents}
        self.update_plan(tasks=tasks, solution=solution, precedence_relations=precedence_relations)

    # ---------------------------- Public API ----------------------------

    def update_plan(
        self,
        *,
        tasks: List[Task],
        solution: Dict[str, List[Assignment]],
        precedence_relations: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        """Replace the active plan (supports replanning)."""

        self._tasks = tasks
        self._solution = solution
        self._precedence_relations = list(precedence_relations or [])

        # Build per-agent queues sorted by (level, task_idx, step_idx).
        self._queue: Dict[str, List[Assignment]] = {}
        all_levels: List[int] = []
        task_level: Dict[int, int] = {}
        for agent_id, assigns in solution.items():
            q = sorted(assigns, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
            self._queue[str(agent_id)] = q
            for t_idx, _s_idx, lv in q:
                all_levels.append(int(lv))
                # a task should have a single level; setdefault is fine if inputs are consistent.
                task_level.setdefault(int(t_idx), int(lv))

        self._min_level = min(all_levels) if all_levels else 0
        self._max_level = max(all_levels) if all_levels else 0
        self._current_level = int(self._min_level)

        # Per-agent mutable state
        self._state: Dict[str, AgentState] = {aid: AgentState() for aid in self._queue.keys()}

        # Global completion bookkeeping (task/step semantics)
        self._completed_task_steps: Set[Tuple[int, int]] = set()
        self._completed_tasks: Set[int] = set()

        # Precompute tasks per level to support task-based barriers
        self._tasks_by_level: Dict[int, Set[int]] = {}
        for t_idx, lv in task_level.items():
            self._tasks_by_level.setdefault(int(lv), set()).add(int(t_idx))

        # Assignment counts per level (only used when level_barrier_mode="assignment")
        self._level_assignment_totals: Dict[int, int] = {}
        for q in self._queue.values():
            for _t, _s, lv in q:
                self._level_assignment_totals[int(lv)] = self._level_assignment_totals.get(int(lv), 0) + 1
        self._level_assignment_done: Dict[int, int] = {lv: 0 for lv in self._level_assignment_totals.keys()}

        # Arbitration order
        self._agent_order: List[str] = sorted(self._queue.keys())
        self._rr_cursor: int = 0

    def reset(self) -> None:
        """Reset execution state (plan remains)."""

        for st in self._state.values():
            st.phase = AgentPhase.ON_GROUND
            st.assignment_idx = 0
            st.launch_issued = False
            st.move_active_key = None
            st.move_active_goal = None
            st.rtb_issued = False
            st.ready_since_tick = None
            st.last_action = ""

        self._current_level = int(self._min_level)
        self._completed_task_steps.clear()
        self._completed_tasks.clear()
        self._level_assignment_done = {lv: 0 for lv in self._level_assignment_totals.keys()}
        self._rr_cursor = 0

    def is_done(self) -> bool:
        """True if all agents completed their queues (and RTB if enabled)."""

        for aid, st in self._state.items():
            if st.assignment_idx < len(self._queue.get(aid, [])):
                return False
            if self.add_return_to_base and not st.rtb_issued:
                return False
        return True

    def current_level(self) -> int:
        return int(self._current_level)

    def next_action(self, observation: Any, *, fallback_tick: int = 0) -> str:
        """Return the single action string for this timestep (or "" for No-Op)."""

        tick = _get_sim_tick(observation, fallback_tick=fallback_tick)

        # Remove redundant assignments already completed globally (optional).
        for aid in self._agent_order:
            self._fast_forward_completed_assignments(aid)

        self._advance_levels_if_done()

        per_agent_notes: Dict[str, str] = {}
        candidates: List[CandidateAction] = []

        for aid in self._agent_order:
            note, cand = self._candidate_for_agent(aid, observation, tick)
            per_agent_notes[aid] = note
            if cand is not None:
                candidates.append(cand)

        chosen = self._choose_candidate(candidates)
        action = chosen.action if chosen else ""

        if self.log_every_n_ticks and (tick % self.log_every_n_ticks == 0):
            self._log_tick(tick, per_agent_notes=per_agent_notes, candidates=candidates, chosen=chosen)

        if chosen:
            self._on_action_issued(chosen, tick=tick)

        # Note: some assignments may be auto-completed (arrival, empty action, etc.)
        self._advance_levels_if_done()
        return action

    # ---------------------------- Candidate generation ----------------------------

    def _fast_forward_completed_assignments(self, agent_id: str) -> None:
        """Skip plan entries already completed globally (when redundancy isn't desired).

        Important: if level_barrier_mode="assignment", skipped redundant assignments
        must still count as "done" or the barrier can deadlock.
        """

        if self.allow_redundant_assignments:
            return

        st = self._state.get(agent_id)
        q = self._queue.get(agent_id, [])
        if st is None:
            return

        while st.assignment_idx < len(q):
            t_idx, s_idx, lv = q[st.assignment_idx]
            if (int(t_idx), int(s_idx)) not in self._completed_task_steps:
                break

            # Skip this assignment (it was already done by someone else)
            st.assignment_idx += 1
            st.phase = AgentPhase.READY
            st.ready_since_tick = None

            # Count it for strict barrier mode too.
            self._level_assignment_done[int(lv)] = self._level_assignment_done.get(int(lv), 0) + 1

    def _candidate_for_agent(self, agent_id: str, observation: Any, tick: int) -> Tuple[str, Optional[CandidateAction]]:
        st = self._state.get(agent_id)
        agent = self._agent_by_id.get(agent_id)
        q = self._queue.get(agent_id, [])

        if st is None or agent is None:
            return ("(missing agent)", None)

        # Done with assigned steps -> RTB / DONE
        if st.assignment_idx >= len(q):
            return self._candidate_rtb_or_done(agent_id, st, agent, observation, tick)

        task_idx, step_idx, level = q[st.assignment_idx]

        # Level gating: do not even launch until the level is unlocked.
        if int(level) != int(self._current_level):
            st.phase = AgentPhase.ON_GROUND if _aircraft_in_any_airbase(observation, agent_id) else AgentPhase.READY
            return (f"waiting_for_level={level} unlocked={self._current_level}", None)

        # Guard against bad indices
        if not (0 <= int(task_idx) < len(self._tasks)):
            self._mark_assignment_complete(agent_id, st, reason=f"bad task_idx={task_idx}")
            return (f"skip bad task_idx={task_idx}", None)
        if not (0 <= int(step_idx) < len(self._tasks[int(task_idx)].steps)):
            self._mark_assignment_complete(agent_id, st, reason=f"bad step_idx={step_idx}")
            return (f"skip bad step_idx={step_idx}", None)

        step = self._tasks[int(task_idx)].steps[int(step_idx)]
        step_type = getattr(getattr(step, "step_type", None), "name", None)
        action_template = getattr(step, "action", None) or ""

        # No action -> mark complete immediately
        if not action_template:
            self._mark_assignment_complete(agent_id, st, reason="empty action")
            return (f"completed(task={task_idx},step={step_idx}) empty-action", None)

        airborne = _aircraft_airborne(observation, agent_id)
        in_airbase = _aircraft_in_any_airbase(observation, agent_id)
        # Launch gating (one-shot):
        # If the aircraft starts stored in an airbase inventory, we issue LAUNCH exactly once,
        # then we never re-issue it automatically.
        if not airborne and in_airbase:
            if st.launch_issued:
                st.phase = AgentPhase.ON_GROUND
                return ("LAUNCH_ALREADY_ISSUED(no-reissue)", None)

            airbase_id = getattr(agent, "home_base_id", None) or _infer_airbase_id_for_aircraft(observation, agent_id)
            if not airbase_id:
                st.phase = AgentPhase.ON_GROUND
                return ("in_airbase but cannot infer airbase_id -> no-op", None)

            st.phase = AgentPhase.ON_GROUND
            return (
                f"READY(launch specific from {airbase_id})",
                self._mk_candidate(
                    agent_id=agent_id,
                    level=int(level),
                    kind=ActionKind.LAUNCH,
                    action=_build_launch_specific_from_airbase_action(str(airbase_id), str(agent_id)),
                    reason="aircraft on ground in airbase inventory; need launch before executing step",
                    tick=tick,
                    meta={"airbase_id": str(airbase_id)},
                ),
            )

        # If not airborne and not in airbase, we can't do much.
        if not airborne:
            st.phase = AgentPhase.ON_GROUND
            return ("not airborne and not in airbase -> no-op", None)

        # Resolve placeholders (weapon_id may be inferred from observation)
        weapon_id = getattr(agent, "weapon_id", None) or _infer_weapon_id_for_unit(observation, agent_id)
        try:
            resolved_action = (
                _replace_placeholders(action_template, agent_id=agent_id, weapon_id=weapon_id)
                if self.replace_placeholders
                else str(action_template)
            )
        except Exception as e:
            # If placeholder replacement fails, do not crash the sim loop.
            self.log.error("Failed to resolve action placeholders for agent=%s: %s | template=%r", agent_id, e, action_template)
            return ("placeholder resolution failed -> no-op", None)

        # Step handlers
        if str(step_type).lower() == "attack":
            return self._candidate_attack(agent_id, st, step, resolved_action, observation, tick, level=int(level), task_idx=int(task_idx), step_idx=int(step_idx))

        if str(step_type).lower() == "move":
            return self._candidate_move(agent_id, st, step, resolved_action, observation, tick, level=int(level), task_idx=int(task_idx), step_idx=int(step_idx))

        # Default: execute once and mark complete after issuing
        st.phase = AgentPhase.READY
        return (
            f"READY(execute type={step_type})",
            self._mk_candidate(
                agent_id=agent_id,
                level=int(level),
                kind=ActionKind.OTHER,
                action=str(resolved_action),
                reason=f"execute step_type={step_type}",
                tick=tick,
            ),
        )
    def _candidate_rtb_or_done(
        self,
        agent_id: str,
        st: AgentState,
        agent: Agent,
        observation: Any,
        tick: int,
    ) -> Tuple[str, Optional[CandidateAction]]:
        airborne = _aircraft_airborne(observation, agent_id)
        in_airbase = _aircraft_in_any_airbase(observation, agent_id)

        if not self.add_return_to_base:
            st.phase = AgentPhase.DONE
            st.rtb_issued = True
            return ("DONE(no-rtb)", None)

        if st.rtb_issued:
            st.phase = AgentPhase.DONE
            return ("DONE(rtb-issued)", None)

        # If aircraft is not airborne and is already in an airbase inventory, consider it returned.
        if (not airborne) and in_airbase:
            st.phase = AgentPhase.DONE
            st.rtb_issued = True
            return ("DONE(already in base)", None)

        if airborne:
            st.phase = AgentPhase.RTB
            return (
                "READY(RTB)",
                self._mk_candidate(
                    agent_id=agent_id,
                    level=int(self._current_level),
                    kind=ActionKind.RTB,
                    action=f"aircraft_return_to_base('{agent_id}')",
                    reason="all assigned steps complete -> RTB",
                    tick=tick,
                ),
            )

        st.phase = AgentPhase.DONE
        st.rtb_issued = True
        return ("DONE(fallback)", None)


    def _candidate_move(
        self,
        agent_id: str,
        st: AgentState,
        step: Any,
        move_action: str,
        observation: Any,
        tick: int,
        *,
        level: int,
        task_idx: int,
        step_idx: int,
    ) -> Tuple[str, Optional[CandidateAction]]:
        """MOVE is one-shot.

        We may *observe* arrival to complete the assignment, but we never re-issue the MOVE action
        for the same (task_idx, step_idx, goal).
        """
        dest = getattr(step, "location", None)
        cur = _get_aircraft_location(observation, agent_id)

        # If we can measure distance, complete once arrived
        if dest is not None and cur is not None:
            try:
                d = cur.distance_to(dest)
                if d <= self.arrival_threshold_km:
                    self._mark_assignment_complete(agent_id, st, reason=f"arrived d={d:.2f}km")
                    return (f"completed(move arrived d={d:.2f}km)", None)
            except Exception:
                pass

        goal: Optional[Tuple[float, float]] = None
        if dest is not None:
            try:
                goal = (float(dest.latitude), float(dest.longitude))
            except Exception:
                goal = None

        key = (int(task_idx), int(step_idx))
        if st.move_active_key == key:
            st.phase = AgentPhase.MOVING
            return ("MOVING(no-reissue)", None)

        st.phase = AgentPhase.MOVING
        return (
            f"READY(move goal={goal})",
            self._mk_candidate(
                agent_id=agent_id,
                level=level,
                kind=ActionKind.MOVE,
                action=str(move_action),
                reason="move toward destination (one-shot)",
                tick=tick,
                meta={"task_idx": int(task_idx), "step_idx": int(step_idx), "goal": goal},
            ),
        )
    def _candidate_attack(
        self,
        agent_id: str,
        st: AgentState,
        step: Any,
        attack_action: str,
        observation: Any,
        tick: int,
        *,
        level: int,
        task_idx: int,
        step_idx: int,
    ) -> Tuple[str, Optional[CandidateAction]]:
        """ATTACK is one-shot, with optional one-shot positioning moves.

        We do not verify BLADE autopilot state. We only avoid re-issuing the same MOVE intent by
        tracking (task_idx, step_idx, goal) locally.
        """
        key = (int(task_idx), int(step_idx))
        cur = _get_aircraft_location(observation, agent_id)
        dist_note = ""

        def issue_position_move(goal: Tuple[float, float], reason: str, note: str) -> Tuple[str, Optional[CandidateAction]]:
            if st.move_active_key == key and st.move_active_goal is not None:
                st.phase = AgentPhase.MOVING
                return (f"MOVING(no-reissue){note}", None)

            lat, lon = float(goal[0]), float(goal[1])
            move_action = f"move_aircraft('{agent_id}', [[{lat}, {lon}]])"
            st.phase = AgentPhase.MOVING
            return (
                f"READY({reason}){note}",
                self._mk_candidate(
                    agent_id=agent_id,
                    level=level,
                    kind=ActionKind.MOVE,
                    action=move_action,
                    reason=reason,
                    tick=tick,
                    meta={"task_idx": int(task_idx), "step_idx": int(step_idx), "goal": goal},
                ),
            )

        # Phase A: Move to approach point (if any)
        approach = getattr(step, "location", None)
        if approach is not None and cur is not None:
            try:
                d = cur.distance_to(approach)
                if d > self.arrival_threshold_km:
                    goal = (float(approach.latitude), float(approach.longitude))
                    return issue_position_move(goal, reason=f"move to approach (d={d:.1f}km)", note=f" d={d:.1f}km")
                # arrived to approach -> allow a second positioning move within this ATTACK assignment
                if st.move_active_key == key:
                    st.move_active_goal = None
            except Exception:
                pass

        # Phase B: If we can find the target, ensure we're within attack_range_km
        target_id = _parse_attack_target_id(str(attack_action))
        if target_id and cur is not None:
            tgt = _find_unit_location(observation, target_id)
            if tgt is not None:
                try:
                    d_tgt = cur.distance_to(tgt)
                    dist_note = f" d_tgt={d_tgt:.1f}km"
                    if d_tgt > self.attack_range_km:
                        goal = (float(tgt.latitude), float(tgt.longitude))
                        return issue_position_move(
                            goal,
                            reason=f"close range (d={d_tgt:.1f}km > {self.attack_range_km}km)",
                            note=dist_note,
                        )
                except Exception:
                    pass

        # Phase C: Attack (one-shot)
        st.phase = AgentPhase.ATTACKING
        return (
            f"READY(attack){dist_note}",
            self._mk_candidate(
                agent_id=agent_id,
                level=level,
                kind=ActionKind.ATTACK,
                action=str(attack_action),
                reason="issue attack (one-shot)",
                tick=tick,
                meta={"task_idx": int(task_idx), "step_idx": int(step_idx)},
            ),
        )


    def _mk_candidate(self, *, agent_id: str, level: int, kind: ActionKind, action: str, reason: str, tick: int, meta: Optional[Mapping[str, Any]] = None) -> CandidateAction:
        st = self._state[agent_id]
        if st.ready_since_tick is None:
            st.ready_since_tick = int(tick)
        return CandidateAction(
            agent_id=str(agent_id),
            level=int(level),
            kind=kind,
            action=str(action),
            reason=str(reason),
            ready_since_tick=int(st.ready_since_tick),
            meta=(meta or {}),
        )

    # ---------------------------- Arbitration ----------------------------

    def _choose_candidate(self, candidates: List[CandidateAction]) -> Optional[CandidateAction]:
        if not candidates:
            return None

        def kind_prio(k: ActionKind) -> int:
            # Prefer "meaningful" actions first (attack before move, etc.).
            if k == ActionKind.ATTACK:
                return 0
            if k == ActionKind.LAUNCH:
                return 1
            if k == ActionKind.MOVE:
                return 2
            if k == ActionKind.RTB:
                return 3
            return 4

        if self.arbitration_policy == "priority":
            # Deterministic: lowest level, then action type priority, then fairness, then agent id.
            return sorted(
                candidates,
                key=lambda c: (int(c.level), kind_prio(c.kind), int(c.ready_since_tick), str(c.agent_id)),
            )[0]

        # Round-robin (default): scan agents from cursor until we find a candidate.
        n = len(self._agent_order)
        if n == 0:
            return candidates[0]

        candidates_by_agent: Dict[str, List[CandidateAction]] = {}
        for c in candidates:
            candidates_by_agent.setdefault(c.agent_id, []).append(c)

        for i in range(n):
            aid = self._agent_order[(self._rr_cursor + i) % n]
            if aid not in candidates_by_agent:
                continue
            chosen = sorted(candidates_by_agent[aid], key=lambda c: (int(c.level), kind_prio(c.kind)))[0]
            self._rr_cursor = (self._rr_cursor + i + 1) % n
            return chosen

        return candidates[0]

    # ---------------------------- Completion + level advancement ----------------------------

    def _mark_assignment_complete(self, agent_id: str, st: AgentState, *, reason: str) -> None:
        """Advance the agent queue and update global completion."""

        q = self._queue.get(agent_id, [])
        if st.assignment_idx >= len(q):
            return

        t_idx, s_idx, lv = q[st.assignment_idx]
        st.assignment_idx += 1
        st.ready_since_tick = None

        # Clear one-shot MOVE guard when leaving an assignment
        st.move_active_key = None
        st.move_active_goal = None

        # Record assignment completion (strict barrier mode)
        self._level_assignment_done[int(lv)] = self._level_assignment_done.get(int(lv), 0) + 1

        # Record task-step completion (task barrier mode)
        self._completed_task_steps.add((int(t_idx), int(s_idx)))

        # If all steps of this task are completed at least once -> mark task completed
        try:
            total_steps = len(self._tasks[int(t_idx)].steps)
        except Exception:
            total_steps = 1

        if all((int(t_idx), int(k)) in self._completed_task_steps for k in range(int(total_steps))):
            self._completed_tasks.add(int(t_idx))

        self.log.info(
            "COMPLETE | agent=%s level=%d task=%s step=%s idx=%d/%d | %s",
            agent_id,
            int(lv),
            int(t_idx),
            int(s_idx),
            int(st.assignment_idx),
            int(len(q)),
            str(reason),
        )

    def _advance_levels_if_done(self) -> None:
        """Advance _current_level while completed barriers are satisfied."""

        changed = True
        while changed:
            changed = False
            lv = int(self._current_level)

            # Find next non-empty level (in case a level has no tasks for some reason)
            if lv not in self._tasks_by_level and lv not in self._level_assignment_totals:
                if lv < int(self._max_level):
                    self._current_level += 1
                    changed = True
                continue

            if self.level_barrier_mode == "assignment":
                total = int(self._level_assignment_totals.get(lv, 0))
                done = int(self._level_assignment_done.get(lv, 0))
                if total > 0 and done >= total and lv < int(self._max_level):
                    self.log.info("LEVEL COMPLETE | level=%d (assignments %d/%d) -> unlock %d", lv, done, total, lv + 1)
                    self._current_level += 1
                    changed = True
                continue

            # task barrier mode
            tasks_in_level = self._tasks_by_level.get(lv, set())
            if tasks_in_level and tasks_in_level.issubset(self._completed_tasks) and lv < int(self._max_level):
                self.log.info("LEVEL COMPLETE | level=%d (tasks %d/%d) -> unlock %d", lv, len(tasks_in_level), len(tasks_in_level), lv + 1)
                self._current_level += 1
                changed = True
    def _on_action_issued(self, chosen: CandidateAction, *, tick: int) -> None:
        """Update per-agent FSM after issuing a global action.

        One-shot semantics:
        - LAUNCH is issued once (no retries).
        - MOVE is issued once per (task_idx, step_idx, goal) intent (no reissue).
        - ATTACK/OTHER are treated as complete once issued.
        """
        st = self._state[chosen.agent_id]
        st.last_action = chosen.action
        st.ready_since_tick = None

        if chosen.kind == ActionKind.LAUNCH:
            st.phase = AgentPhase.ON_GROUND
            st.launch_issued = True
            return

        if chosen.kind == ActionKind.MOVE:
            st.phase = AgentPhase.MOVING
            meta = chosen.meta or {}
            try:
                st.move_active_key = (int(meta.get("task_idx", -1)), int(meta.get("step_idx", -1)))
            except Exception:
                st.move_active_key = None
            goal = meta.get("goal", None)
            if isinstance(goal, tuple) and len(goal) == 2:
                try:
                    st.move_active_goal = (float(goal[0]), float(goal[1]))
                except Exception:
                    st.move_active_goal = None
            else:
                st.move_active_goal = None
            return

        if chosen.kind == ActionKind.RTB:
            st.phase = AgentPhase.DONE
            st.rtb_issued = True
            return

        # ATTACK/OTHER -> mark assignment complete immediately (best-effort)
        st.phase = AgentPhase.ATTACKING if chosen.kind == ActionKind.ATTACK else AgentPhase.READY
        self._mark_assignment_complete(chosen.agent_id, st, reason=f"issued {chosen.kind.value}")



    # ---------------------------- Logging ----------------------------

    def _log_tick(self, tick: int, *, per_agent_notes: Dict[str, str], candidates: List[CandidateAction], chosen: Optional[CandidateAction]) -> None:
        self.log.info("TICK %d | unlocked_level=%d | candidates=%d", int(tick), int(self._current_level), len(candidates))
        for aid in self._agent_order:
            st = self._state.get(aid)
            q = self._queue.get(aid, [])
            nxt = q[st.assignment_idx] if (st and st.assignment_idx < len(q)) else None
            self.log.info(
                "  agent=%s phase=%s next=%s | %s",
                aid,
                getattr(st, "phase", None),
                nxt,
                per_agent_notes.get(aid, ""),
            )

        for c in sorted(candidates, key=lambda x: (int(x.level), x.kind.value, x.agent_id)):
            self.log.info(
                "    cand agent=%s lvl=%d kind=%s | %s | %s",
                c.agent_id,
                int(c.level),
                c.kind.value,
                c.action,
                c.reason,
            )

        if chosen:
            self.log.info("  CHOSEN | agent=%s lvl=%d kind=%s | %s", chosen.agent_id, chosen.level, chosen.kind.value, chosen.action)
        else:
            self.log.info("  CHOSEN | (no-op)")


class BladePlanExecutor:
    """Compatibility wrapper for older code that expects actions_for_tick() -> List[str]."""

    def __init__(self, *args, **kwargs) -> None:
        self._executor = BladeStateExecutor(*args, **kwargs)

    def reset(self) -> None:
        self._executor.reset()

    def update_plan(self, *args, **kwargs) -> None:
        self._executor.update_plan(*args, **kwargs)

    def actions_for_tick(self, observation: Any) -> List[str]:
        action = self._executor.next_action(observation, fallback_tick=0)
        return [action] if action else []

    def is_done(self) -> bool:
        return self._executor.is_done()
