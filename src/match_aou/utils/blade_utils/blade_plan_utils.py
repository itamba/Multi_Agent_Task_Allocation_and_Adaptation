"""blade_plan_utils.py

BLADE-specific post-processing for executing a MATCH-AOU plan in the BLADE simulator.

This module runs *after* MATCH-AOU post-solve processing (filtering, reindexing,
precedence layering). It takes the selected tasks and the solver solution and
produces a BLADE-friendly execution plan.

Key features
------------
- Assigns per-agent execution ticks to each Step via `step.execution_times[agent_id]`.
- Resolves action templates into concrete action strings per agent (AGENT_ID / WEAPON_ID).
- Supports "macro attack" steps (single MATCH-AOU Step) that expand into:
    launch (optional) -> move_aircraft(...) -> handle_aircraft_attack(...)
- Enforces **global** BLADE constraint: **at most one action per timestep**.

Design notes
------------
- We avoid mutating `step.action` for multi-agent steps. Resolved actions are stored in
  `step.actions_by_agent[agent_id]`.
- Timing is heuristic (demo playback). This is not an observation-driven executor.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

from ...models import Agent, Step, Task

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level_order)


@dataclass(frozen=True)
class BladePlanArtifacts:
    execution_time_to_actions: Dict[int, List[str]]
    levels: List[int]
    level_start_time: Dict[int, int]


def _ensure_execution_times(step: Step) -> None:
    if not hasattr(step, "execution_times") or step.execution_times is None:
        step.execution_times = {}


def _ensure_actions_by_agent(step: Step) -> None:
    if not hasattr(step, "actions_by_agent") or step.actions_by_agent is None:
        step.actions_by_agent = {}


def _replace_action_placeholders(action: str, *, agent_id: str, weapon_id: Optional[str]) -> str:
    out = action.replace("AGENT_ID", str(agent_id))

    if "WEAPON_ID" in out:
        if weapon_id is None:
            raise ValueError(
                f"Action requires WEAPON_ID but agent_id={agent_id} has no weapon_id. "
                f"Template was: {action!r}"
            )
        out = out.replace("WEAPON_ID", str(weapon_id))

    return out


def _serialize_events_one_action_per_tick(events: List[dict]) -> List[dict]:
    """Assign a unique global tick to each event while preserving per-agent spacing.

    Constraints enforced:
    - At most one action is executed per global tick (BLADE constraint).
    - Within each agent, the *minimum* gap between consecutive events is preserved
      based on the original heuristic desired ticks (crucial for travel-time gaps).
    """
    occupied: set[int] = set()
    per_agent_last_assigned: Dict[str, int] = {}

    assigned: List[dict] = []
    for ev in sorted(
        events,
        key=lambda e: (int(e.get("desired_tick", 0)), str(e.get("agent_id", "")), int(e.get("seq", 0))),
    ):
        agent_id = str(ev["agent_id"])
        desired = int(ev["desired_tick"])
        gap = int(ev.get("gap_from_prev", 0) or 0)

        if agent_id in per_agent_last_assigned:
            earliest = max(desired, int(per_agent_last_assigned[agent_id]) + gap)
        else:
            earliest = desired

        tick = earliest
        while tick in occupied:
            tick += 1

        new_ev = dict(ev)
        new_ev["assigned_tick"] = tick
        assigned.append(new_ev)

        occupied.add(tick)
        per_agent_last_assigned[agent_id] = tick

    return assigned


def populate_blade_fields(
    *,
    tasks: List[Task],
    solution: Dict[str, List[Assignment]],
    agents: Sequence[Agent],
    start_time: int = 100,
    time_step_duration: int = 1,
    buffer_steps: int = 10,
    stagger_agents: bool = True,
    replace_placeholders: bool = True,
    min_speed_kts: float = 450.0,
    horizon_tick: Optional[int] = None,
    add_return_to_base: bool = False,
    # NEW:
    add_launch_from_airbase: bool = True,
    launch_lead_steps: int = 7,
    airborne_aircraft_ids: Optional[Sequence[str]] = None,
    # If False (default): schedule move "just-in-time" for the precedence-gated attack moment.
    # If True: allow early approach (move may happen as soon as the agent is free).
    allow_early_approach: bool = False,
) -> BladePlanArtifacts:
    """Fill BLADE-facing fields (execution times + resolved action strings).

    Parameters
    ----------
    airborne_aircraft_ids:
        IDs of aircraft already present in `observation.aircraft` at reset.
        If an agent_id is included here, we *skip* scheduling a launch action for it.
        This prevents accidental "double launch" in scenarios where aircraft start airborne.
    allow_early_approach:
        If True, the aircraft may move toward its approach point earlier than its
        precedence-gated attack time (can cause loiter/waiting at destination).
        If False, we compute the move time so arrival is as close as possible to the
        attack time ("just-in-time"), reducing unnecessary waiting (and fuel burn).
    """

    if time_step_duration <= 0:
        raise ValueError("time_step_duration must be positive")
    if launch_lead_steps < 0:
        raise ValueError("launch_lead_steps must be >= 0")

    airborne_set = {str(x) for x in (airborne_aircraft_ids or [])}

    agent_by_id: Dict[str, Agent] = {str(a.id): a for a in agents}

    levels_set = {int(level) for assigns in solution.values() for *_rest, level in assigns}
    levels = sorted(levels_set)

    level_start_time: Dict[int, int] = {}
    current_level_start = int(start_time)

    per_agent_cursor: Dict[str, int] = {str(aid): int(start_time) for aid in solution.keys()}
    per_agent_loc: Dict[str, Optional[object]] = {
        str(aid): (agent_by_id.get(str(aid)).location if agent_by_id.get(str(aid)) else None)
        for aid in solution.keys()
    }

    # Raw mapping (may have collisions until serialization)
    execution_time_to_actions: DefaultDict[int, List[str]] = defaultdict(list)

    # Structured events so we can enforce the global one-action-per-tick rule
    events: List[dict] = []
    per_agent_last_desired: Dict[str, int] = {}
    per_agent_seq: Dict[str, int] = {}

    launch_scheduled: set[str] = set()

    def _add_event(*, agent_id: str, desired_tick: int, action: str, step: Optional[Step], kind: str) -> None:
        a = str(agent_id)
        d = int(desired_tick)

        prev_d = per_agent_last_desired.get(a)
        gap = 0 if prev_d is None else max(0, d - int(prev_d))

        seq = per_agent_seq.get(a, 0)
        per_agent_seq[a] = seq + 1
        per_agent_last_desired[a] = d

        # Optional per-step debug: keep macro actions list per agent
        if step is not None and kind in ("launch", "move", "attack"):
            if not hasattr(step, "macro_actions_by_agent") or step.macro_actions_by_agent is None:
                step.macro_actions_by_agent = {}
            step.macro_actions_by_agent.setdefault(a, []).append(str(action))

        events.append(
            {
                "agent_id": a,
                "desired_tick": d,
                "gap_from_prev": int(gap),
                "seq": int(seq),
                "action": str(action),
                "step": step,
                "kind": str(kind),
            }
        )

    def _maybe_schedule_launch(agent_id: str, *, before_tick: int, step_for_debug: Optional[Step]) -> None:
        """Schedule a launch action once per agent (if enabled and needed)."""
        a = str(agent_id)
        if not add_launch_from_airbase:
            return
        if a in airborne_set:
            return
        if a in launch_scheduled:
            return

        agent = agent_by_id.get(a)
        airbase_id = getattr(agent, "home_base_id", None) if agent is not None else None
        if not airbase_id:
            # No way to launch if we don't know the home base id.
            return

        desired_launch_tick = max(0, int(before_tick) - int(launch_lead_steps))
        if horizon_tick is not None:
            desired_launch_tick = min(desired_launch_tick, int(horizon_tick))

        launch_action = f"launch_aircraft_from_airbase('{airbase_id}')"
        execution_time_to_actions[desired_launch_tick].append(launch_action)
        _add_event(agent_id=a, desired_tick=desired_launch_tick, action=launch_action, step=step_for_debug, kind="launch")
        launch_scheduled.add(a)

    def _compute_travel_steps(agent: Optional[Agent], src_loc: Optional[object], dest_loc: object) -> int:
        travel_steps = 0
        if src_loc is None or dest_loc is None:
            return 0
        try:
            dist_km = src_loc.distance_to(dest_loc)

            observed_speed_kts = float(getattr(agent, "speed", 0) or 0) if agent is not None else 0.0
            effective_speed_kts = max(observed_speed_kts, float(min_speed_kts))
            speed_kmh = effective_speed_kts * 1.852

            time_hr = dist_km / max(speed_kmh, 1e-6)
            travel_seconds = time_hr * 3600.0
            travel_steps = int(round(travel_seconds / float(time_step_duration)))
        except Exception:
            travel_steps = 0
        return max(0, int(travel_steps))

    for level in levels:
        level_start_time[level] = current_level_start

        for agent_offset_idx, agent_id in enumerate(sorted(solution.keys())):
            agent = agent_by_id.get(str(agent_id))
            weapon_id = getattr(agent, "weapon_id", None) if agent is not None else None

            agent_assigns = [(t, s) for (t, s, lv) in solution.get(agent_id, []) if int(lv) == int(level)]
            agent_assigns.sort(key=lambda x: (x[0], x[1]))
            if not agent_assigns:
                continue

            loc_cursor = per_agent_loc.get(str(agent_id))
            t_cursor = per_agent_cursor.get(str(agent_id), int(start_time))
            stagger = (agent_offset_idx * time_step_duration) if stagger_agents else 0

            for task_idx, step_idx in agent_assigns:
                if not (0 <= int(task_idx) < len(tasks)):
                    continue
                if not (0 <= int(step_idx) < len(tasks[int(task_idx)].steps)):
                    continue

                step = tasks[int(task_idx)].steps[int(step_idx)]
                _ensure_execution_times(step)
                _ensure_actions_by_agent(step)

                action_template = getattr(step, "action", None)
                step_type_name = getattr(getattr(step, "step_type", None), "name", None)

                # ---------------- Macro attack step: launch? -> move -> attack ----------------
                if (
                    action_template
                    and step_type_name == "attack"
                    and getattr(step, "location", None) is not None
                    and "handle_" in str(action_template)
                ):
                    dest = step.location
                    travel_steps = _compute_travel_steps(agent, loc_cursor, dest)

                    # Precedence-gated earliest time this *attack* may occur.
                    level_gate = int(current_level_start) + int(stagger)

                    if allow_early_approach:
                        # Move as soon as the agent is free.
                        move_time = int(t_cursor + stagger)
                        attack_time = int(move_time + travel_steps + buffer_steps)
                        attack_time = max(attack_time, level_gate)
                    else:
                        # "Just-in-time" departure: attack happens as soon as allowed,
                        # but only after we have time to travel.
                        attack_time = max(int(t_cursor + stagger + travel_steps + buffer_steps), level_gate)
                        move_time = max(int(t_cursor + stagger), int(attack_time - (travel_steps + buffer_steps)))

                    if horizon_tick is not None:
                        move_time = min(move_time, int(horizon_tick))
                        attack_time = min(attack_time, int(horizon_tick))

                    # Ensure launch happens before first movement (if needed).
                    _maybe_schedule_launch(str(agent_id), before_tick=move_time, step_for_debug=step)

                    # 1) schedule move to approach point
                    move_action = f"move_aircraft('{agent_id}', [[{dest.latitude}, {dest.longitude}]])"
                    execution_time_to_actions[move_time].append(move_action)
                    _add_event(agent_id=str(agent_id), desired_tick=move_time, action=move_action, step=step, kind="move")

                    # 2) schedule the attack after arrival
                    if replace_placeholders:
                        resolved_attack = _replace_action_placeholders(
                            str(action_template),
                            agent_id=str(agent_id),
                            weapon_id=weapon_id,
                        )
                    else:
                        resolved_attack = str(action_template)

                    step.actions_by_agent[str(agent_id)] = resolved_attack
                    execution_time_to_actions[attack_time].append(resolved_attack)
                    _add_event(agent_id=str(agent_id), desired_tick=attack_time, action=resolved_attack, step=step, kind="attack")

                    # Record execution time as the attack moment
                    step.execution_times[str(agent_id)] = int(attack_time)

                    # Update cursors: location updates to dest; time advances past attack (+ buffer)
                    loc_cursor = dest
                    t_cursor = int(attack_time + buffer_steps)
                    if horizon_tick is not None:
                        t_cursor = min(t_cursor, int(horizon_tick))
                    continue

                # ---------------- Default (non-attack) step ----------------
                exe_time = int(max(t_cursor, current_level_start) + stagger)
                if horizon_tick is not None:
                    exe_time = min(exe_time, int(horizon_tick))

                # If this is the first action for an aircraft that needs launch, launch before it.
                # (We treat any step with an action as needing the unit to exist in the sim.)
                if action_template:
                    _maybe_schedule_launch(str(agent_id), before_tick=exe_time, step_for_debug=step)

                step.execution_times[str(agent_id)] = int(exe_time)

                if action_template:
                    if replace_placeholders:
                        resolved_action = _replace_action_placeholders(
                            str(action_template),
                            agent_id=str(agent_id),
                            weapon_id=weapon_id,
                        )
                    else:
                        resolved_action = str(action_template)

                    step.actions_by_agent[str(agent_id)] = resolved_action
                    execution_time_to_actions[exe_time].append(resolved_action)
                    _add_event(agent_id=str(agent_id), desired_tick=exe_time, action=resolved_action, step=step, kind="default")

                # Travel heuristic (only affects cursor)
                travel_steps = 0
                if getattr(step, "location", None) is not None and loc_cursor is not None:
                    travel_steps = _compute_travel_steps(agent, loc_cursor, step.location)
                    loc_cursor = step.location

                t_cursor = int(t_cursor + travel_steps + buffer_steps)
                if horizon_tick is not None:
                    t_cursor = min(t_cursor, int(horizon_tick))

            per_agent_cursor[str(agent_id)] = t_cursor
            per_agent_loc[str(agent_id)] = loc_cursor

        level_finish = max(per_agent_cursor.values()) if per_agent_cursor else current_level_start
        current_level_start = int(level_finish + buffer_steps)
        if horizon_tick is not None:
            current_level_start = min(current_level_start, int(horizon_tick))

    # Return-to-base
    if add_return_to_base:
        for agent_id, cursor in per_agent_cursor.items():
            rtb_time = int(cursor + buffer_steps)
            if horizon_tick is not None:
                rtb_time = min(rtb_time, int(horizon_tick))

            # If aircraft starts in airbase and was never launched (e.g., only RTB),
            # make sure we launch before issuing RTB.
            _maybe_schedule_launch(str(agent_id), before_tick=rtb_time, step_for_debug=None)

            rtb_action = f"aircraft_return_to_base('{agent_id}')"
            execution_time_to_actions[rtb_time].append(rtb_action)
            _add_event(agent_id=str(agent_id), desired_tick=rtb_time, action=rtb_action, step=None, kind="rtb")

    # ---------------- Global one-action-per-tick serialization ----------------
    serialized_events = _serialize_events_one_action_per_tick(events)

    execution_time_to_actions = defaultdict(list)
    for ev in serialized_events:
        tick = int(ev["assigned_tick"])
        action = str(ev["action"])
        execution_time_to_actions[tick].append(action)

        # Keep step timing aligned with final schedule (debug friendly)
        step = ev.get("step")
        if step is not None:
            agent_id = str(ev["agent_id"])
            _ensure_execution_times(step)
            if ev.get("kind") == "move":
                if not hasattr(step, "approach_execution_times") or step.approach_execution_times is None:
                    step.approach_execution_times = {}
                step.approach_execution_times[agent_id] = tick
            else:
                step.execution_times[agent_id] = tick

    # Cosmetic: if a step has exactly one resolved action, set step.action for nicer printing.
    if replace_placeholders:
        for task in tasks:
            for step in task.steps:
                actions_by_agent = getattr(step, "actions_by_agent", None)
                if isinstance(actions_by_agent, dict) and len(actions_by_agent) == 1:
                    step.action = next(iter(actions_by_agent.values()))

    for t in list(execution_time_to_actions.keys()):
        execution_time_to_actions[t] = list(execution_time_to_actions[t])

    return BladePlanArtifacts(
        execution_time_to_actions=dict(execution_time_to_actions),
        levels=levels,
        level_start_time=level_start_time,
    )
