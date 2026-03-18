from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...models import Agent, Location, Task

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level_order)


# ----------------------------
# Minimal helpers (Scenario)
# ----------------------------

def _get_sim_tick(observation: Any, *, fallback_tick: int) -> int:
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

def _find_airbase_and_inventory(scenario: Any, airbase_id: str) -> Optional[Any]:
    for base in getattr(scenario, "airbases", []) or []:
        if str(getattr(base, "id", "")) == str(airbase_id):
            return base
    return None

def _infer_airbase_id_for_aircraft(scenario: Any, aircraft_id: str) -> Optional[str]:
    for base in getattr(scenario, "airbases", []) or []:
        for ac in getattr(base, "aircraft", []) or []:
            if str(getattr(ac, "id", "")) == str(aircraft_id):
                bid = getattr(base, "id", None)
                return str(bid) if bid is not None else None
    return None

def _aircraft_airborne(scenario: Any, aircraft_id: str) -> bool:
    for ac in getattr(scenario, "aircraft", []) or []:
        if str(getattr(ac, "id", "")) == str(aircraft_id):
            return True
    return False

def _aircraft_in_any_airbase(scenario: Any, aircraft_id: str) -> bool:
    return _infer_airbase_id_for_aircraft(scenario, aircraft_id) is not None

def _infer_weapon_id_for_unit(scenario: Any, unit_id: str) -> Optional[str]:
    # aircraft
    for ac in getattr(scenario, "aircraft", []) or []:
        if str(getattr(ac, "id", "")) == str(unit_id):
            if hasattr(ac, "get_weapon_with_highest_engagement_range"):
                try:
                    w = ac.get_weapon_with_highest_engagement_range()
                    wid = getattr(w, "id", None)
                    return str(wid) if wid is not None else None
                except Exception:
                    pass
            weapons = getattr(ac, "weapons", []) or []
            if weapons:
                wid = getattr(weapons[0], "id", None)
                return str(wid) if wid is not None else None
            return None

    # ships (optional demo)
    for sh in getattr(scenario, "ships", []) or []:
        if str(getattr(sh, "id", "")) == str(unit_id):
            weapons = getattr(sh, "weapons", []) or []
            if weapons:
                wid = getattr(weapons[0], "id", None)
                return str(wid) if wid is not None else None
            return None

    return None

def _replace_placeholders(action: str, *, agent_id: str, weapon_id: Optional[str]) -> str:
    out = str(action).replace("AGENT_ID", str(agent_id))
    if "WEAPON_ID" in out:
        if weapon_id is None:
            raise ValueError(f"Action requires WEAPON_ID but agent {agent_id} has no weapon_id")
        out = out.replace("WEAPON_ID", str(weapon_id))
    return out

def _build_validated_launch_action_for_aircraft(scenario: Any, aircraft_id: str, *, airbase_id: Optional[str]=None) -> str:
    ac_id = str(aircraft_id)
    resolved_ab_id = str(airbase_id) if airbase_id else _infer_airbase_id_for_aircraft(scenario, ac_id)
    if not resolved_ab_id:
        raise ValueError(f"Aircraft {ac_id} not present in any airbase inventory; cannot launch.")

    ab = _find_airbase_and_inventory(scenario, resolved_ab_id)
    if ab is None:
        raise ValueError(f"Airbase {resolved_ab_id} not found in scenario; cannot launch aircraft {ac_id}.")

    inv = getattr(ab, "aircraft", None) or []
    if not inv:
        raise ValueError(f"Airbase {resolved_ab_id} has empty aircraft inventory; cannot launch aircraft {ac_id}.")

    head_id = str(getattr(inv[0], "id", ""))
    if head_id != ac_id:
        preview = [str(getattr(a, "id", "")) for a in inv[:5]]
        raise ValueError(
            f"Cannot launch aircraft {ac_id} from airbase {resolved_ab_id}: "
            f"FIFO head is {head_id}. Queue head preview={preview}"
        )

    return f"launch_aircraft_from_airbase('{resolved_ab_id}')"


# ----------------------------
# Minimal executor
# ----------------------------

@dataclass
class _AgentExec:
    idx: int = 0
    last_move_goal: Optional[Tuple[float, float]] = None
    rtb_issued: bool = False

@dataclass(frozen=True)
class Candidate:
    agent_id: str
    action: str
    kind: str  # "MOVE" | "LAUNCH" | "RTB" | "STEP"
    # Optional metadata for commit
    move_goal: Optional[Tuple[float, float]] = None
    # If this candidate represents executing the current step, carry indices to mark complete
    task_idx: Optional[int] = None
    step_idx: Optional[int] = None


class BladeExecutorMinimal:
    """
    Minimal state executor for demos:
    - One global action per tick.
    - Level gating by level_order.
    - Launch validated FIFO if aircraft is in airbase.
    - Execute each assigned step once (mark complete on issue).
    """

    def __init__(
        self,
        *,
        tasks: List[Task],
        solution: Dict[str, List[Assignment]],
        agents: Sequence[Agent],
        add_return_to_base: bool = False,
        arrival_threshold_km: float = 50.0,
    ) -> None:
        self.tasks = tasks
        self.solution = {str(k): list(v) for k, v in solution.items()}
        self.agent_by_id = {str(a.id): a for a in agents}
        self.add_return_to_base = bool(add_return_to_base)
        self.arrival_threshold_km = float(arrival_threshold_km)

        # per-agent queue sorted by (level, task, step)
        self.queue: Dict[str, List[Assignment]] = {
            aid: sorted(assigns, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
            for aid, assigns in self.solution.items()
        }
        self.agent_order = sorted(self.queue.keys())
        self.state: Dict[str, _AgentExec] = {aid: _AgentExec() for aid in self.agent_order}

        self.completed_task_steps: set[Tuple[int, int]] = set()
        self.current_level: int = min((lv for assigns in self.queue.values() for *_, lv in assigns), default=0)
        self.max_level: int = max((lv for assigns in self.queue.values() for *_, lv in assigns), default=0)

        self._rr_cursor = 0

    def is_done(self) -> bool:
        for aid in self.agent_order:
            st = self.state[aid]
            if st.idx < len(self.queue.get(aid, [])):
                return False
            if self.add_return_to_base and not st.rtb_issued:
                return False
        return True

    def next_action(self, observation: Any, *, fallback_tick: int = 0) -> str:
        _tick = _get_sim_tick(observation, fallback_tick=fallback_tick)

        # advance level if no remaining assignments in current level
        self._advance_level_if_empty()

        # generate candidates (at most 1 per agent)
        candidates: List[Candidate] = []

        for aid in self.agent_order:
            cand = self._candidate_for_agent(aid, observation)
            if cand is not None and cand.action:
                candidates.append(cand)

        chosen = self._choose_rr(candidates)
        if chosen is None:
            return ""

        # IMPORTANT: commit side-effects only for chosen candidate
        self._on_action_chosen(chosen)

        return chosen.action

    def _advance_level_if_empty(self) -> None:
        lv = int(self.current_level)
        while lv <= int(self.max_level):
            any_left = False
            for aid in self.agent_order:
                st = self.state[aid]
                q = self.queue.get(aid, [])
                if st.idx >= len(q):
                    continue
                _t_idx, _s_idx, level = q[st.idx]
                if int(level) == lv:
                    any_left = True
                    break
            if any_left or lv >= int(self.max_level):
                self.current_level = lv
                return
            lv += 1
        self.current_level = int(self.max_level)

    def _candidate_for_agent(self, aid: str, observation: Any) -> Optional[Candidate]:
        st = self.state[aid]
        q = self.queue.get(aid, [])
        agent = self.agent_by_id.get(aid)

        if agent is None:
            return None

        # ---- RTB if done with queue ----
        if st.idx >= len(q):
            if not self.add_return_to_base or st.rtb_issued:
                return None

            airborne = _aircraft_airborne(observation, aid)
            in_airbase = _aircraft_in_any_airbase(observation, aid)

            # Already in an airbase inventory => treat as done (commit will set rtb_issued)
            if (not airborne) and in_airbase:
                return Candidate(agent_id=aid, action="", kind="RTB")

            if airborne:
                return Candidate(agent_id=aid, action=f"aircraft_return_to_base('{aid}')", kind="RTB")

            # not airborne and not in airbase -> nothing we can do, but mark as "done" for RTB semantics
            return Candidate(agent_id=aid, action="", kind="RTB")

        task_idx, step_idx, level = q[st.idx]
        if int(level) != int(self.current_level):
            return None

        # validate indices (NOTE: do NOT advance idx here; only commit after chosen)
        if not (0 <= int(task_idx) < len(self.tasks)):
            # skip by emitting a STEP candidate with empty action and commit will advance
            return Candidate(agent_id=aid, action="", kind="STEP", task_idx=int(task_idx), step_idx=int(step_idx))
        if not (0 <= int(step_idx) < len(self.tasks[int(task_idx)].steps)):
            return Candidate(agent_id=aid, action="", kind="STEP", task_idx=int(task_idx), step_idx=int(step_idx))

        step = self.tasks[int(task_idx)].steps[int(step_idx)]
        action_template = getattr(step, "action", None) or ""
        step_type = str(getattr(getattr(step, "step_type", None), "name", "")).lower()

        airborne = _aircraft_airborne(observation, aid)
        in_airbase = _aircraft_in_any_airbase(observation, aid)

        # Launch gating
        if (not airborne) and in_airbase:
            airbase_id = getattr(agent, "home_base_id", None) or _infer_airbase_id_for_aircraft(observation, aid)
            action = _build_validated_launch_action_for_aircraft(observation, aid, airbase_id=airbase_id)
            return Candidate(agent_id=aid, action=action, kind="LAUNCH")

        if not airborne and not in_airbase:
            return None

        loc = getattr(step, "location", None)

        # ATTACK: wait until close enough before issuing attack; move is one-shot but commit only on chosen
        if step_type == "attack" and loc is not None:
            cur = _get_aircraft_location(observation, aid)
            if cur is None:
                return None

            try:
                d_km = cur.distance_to(loc)
            except Exception:
                d_km = 10**9

            goal = (float(loc.latitude), float(loc.longitude))

            if d_km > self.arrival_threshold_km:
                # propose MOVE if we haven't committed this move_goal yet
                if st.last_move_goal != goal:
                    return Candidate(
                        agent_id=aid,
                        action=f"move_aircraft('{aid}', [[{goal[0]}, {goal[1]}]])",
                        kind="MOVE",
                        move_goal=goal,
                    )
                return None  # already moved; let others act while we travel

        # Non-attack optional one-shot move (also commit only on chosen)
        if loc is not None and "move_aircraft" not in str(action_template) and step_type != "attack":
            goal = (float(loc.latitude), float(loc.longitude))
            if st.last_move_goal != goal:
                return Candidate(
                    agent_id=aid,
                    action=f"move_aircraft('{aid}', [[{goal[0]}, {goal[1]}]])",
                    kind="MOVE",
                    move_goal=goal,
                )

        # Execute step action (or empty -> skip) (commit advances idx only if chosen)
        if not action_template:
            return Candidate(agent_id=aid, action="", kind="STEP", task_idx=int(task_idx), step_idx=int(step_idx))

        weapon_id = getattr(agent, "weapon_id", None) or _infer_weapon_id_for_unit(observation, aid)
        resolved = _replace_placeholders(str(action_template), agent_id=aid, weapon_id=weapon_id)

        return Candidate(
            agent_id=aid,
            action=str(resolved),
            kind="STEP",
            task_idx=int(task_idx),
            step_idx=int(step_idx),
        )

    def _on_action_chosen(self, chosen: Candidate) -> None:
        """Commit side-effects ONLY for the chosen action."""
        st = self.state[chosen.agent_id]

        if chosen.kind == "MOVE":
            st.last_move_goal = chosen.move_goal
            return

        if chosen.kind == "RTB":
            st.rtb_issued = True
            return

        if chosen.kind == "STEP":
            # Mark step complete on issue (demo semantics)
            if st.idx < len(self.queue.get(chosen.agent_id, [])):
                st.idx += 1
            if chosen.task_idx is not None and chosen.step_idx is not None:
                self.completed_task_steps.add((int(chosen.task_idx), int(chosen.step_idx)))
            return

        if chosen.kind == "LAUNCH":
            # Nothing to commit (strict FIFO handled by validator).
            return

    def _choose_rr(self, candidates: List[Candidate]) -> Optional[Candidate]:
        if not candidates:
            return None

        candidates_by_agent: Dict[str, List[Candidate]] = {}
        for c in candidates:
            candidates_by_agent.setdefault(c.agent_id, []).append(c)

        n = len(self.agent_order)
        for i in range(n):
            aid = self.agent_order[(self._rr_cursor + i) % n]
            if aid not in candidates_by_agent:
                continue

            # If somehow multiple candidates exist for an agent, choose a deterministic priority
            def prio(c: Candidate) -> int:
                if c.kind == "LAUNCH":
                    return 0
                if c.kind == "MOVE":
                    return 1
                if c.kind == "STEP":
                    return 2
                if c.kind == "RTB":
                    return 3
                return 9

            chosen = sorted(candidates_by_agent[aid], key=prio)[0]
            self._rr_cursor = (self._rr_cursor + i + 1) % n
            return chosen

        return candidates[0]
