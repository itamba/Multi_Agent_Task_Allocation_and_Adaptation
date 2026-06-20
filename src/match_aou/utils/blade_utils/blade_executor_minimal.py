from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...models import Agent, Location, StepKind, Task

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level_order)


# ----------------------------
# Intra-level nearest-neighbor ordering (WI-4)
# ----------------------------

def nearest_neighbor_order(
    assignments: Sequence[Assignment],
    *,
    location_of: Callable[[Assignment], Optional[Location]],
    start_location: Optional[Location],
) -> Tuple[List[Assignment], Optional[Location]]:
    """Reorder same-level assignments by greedy nearest-neighbor over step target locations.

    ``location_of(assignment) -> Location | None`` resolves each assignment's target
    location. Starting at ``start_location``, repeatedly pick the remaining LOCATED
    assignment minimizing ``Location.distance_to(...)`` (haversine) from the current
    position, tie-break by ``(task_idx, step_idx)``, append it, and advance the current
    position to that location. Unlocated assignments (``location_of`` returns ``None``, or
    ``start_location`` is ``None`` so distance cannot be computed) are appended last in
    ``(task_idx, step_idx)`` order and do not move the position.

    Returns ``(ordered_assignments, end_location)`` where ``end_location`` is the position
    after the last located pick (or ``start_location`` if none were located), so callers can
    chain it as the start of the next level.
    """
    located: List[Assignment] = []
    unlocated: List[Assignment] = []
    for a in assignments:
        loc = location_of(a)
        if loc is not None:
            located.append(a)
        else:
            unlocated.append(a)

    ordered: List[Assignment] = []
    current = start_location
    remaining = list(located)

    # If we have no anchor, distance is undefined: keep located steps in deterministic
    # (task_idx, step_idx) order without advancing position.
    if current is None:
        ordered.extend(sorted(remaining, key=lambda x: (int(x[0]), int(x[1]))))
        remaining = []

    while remaining:
        def _key(a: Assignment) -> Tuple[float, int, int]:
            return (current.distance_to(location_of(a)), int(a[0]), int(a[1]))

        nxt = min(remaining, key=_key)
        ordered.append(nxt)
        current = location_of(nxt)
        remaining.remove(nxt)

    ordered.extend(sorted(unlocated, key=lambda x: (int(x[0]), int(x[1]))))
    return ordered, current


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
    # Semantic target id of an ATTACK STEP candidate (audit-only; consumed in _on_action_chosen)
    target_id: Optional[str] = None


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
        nn_ordering: bool = True,
    ) -> None:
        self.tasks = tasks
        self.solution = {str(k): list(v) for k, v in solution.items()}
        self.agent_by_id = {str(a.id): a for a in agents}
        self.add_return_to_base = bool(add_return_to_base)
        self.arrival_threshold_km = float(arrival_threshold_km)
        self.nn_ordering = bool(nn_ordering)

        # Per-agent queue. PRIMARY ordering is always by level_order (topological order
        # between levels is preserved). The SECONDARY (intra-level) ordering depends on the
        # flag:
        #   nn_ordering=True  -> greedy nearest-neighbor over step target locations
        #                        (shorter flight paths; the SET of issued steps is unchanged).
        #   nn_ordering=False -> exact legacy (level_order, task_idx, step_idx) sort,
        #                        byte-for-byte recoverable.
        if self.nn_ordering:
            self.queue: Dict[str, List[Assignment]] = {
                aid: self._build_nn_queue(aid, assigns)
                for aid, assigns in self.solution.items()
            }
        else:
            self.queue = {
                aid: sorted(assigns, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
                for aid, assigns in self.solution.items()
            }
        self.agent_order = sorted(self.queue.keys())
        self.state: Dict[str, _AgentExec] = {aid: _AgentExec() for aid in self.agent_order}

        self.completed_task_steps: set[Tuple[int, int]] = set()
        self.current_level: int = min((lv for assigns in self.queue.values() for *_, lv in assigns), default=0)
        self.max_level: int = max((lv for assigns in self.queue.values() for *_, lv in assigns), default=0)

        self._rr_cursor = 0

        # Target id of the most recent attack command actually emitted. Written in
        # _on_action_chosen for the chosen candidate only (so it reflects the agent that
        # acted this tick), and read by callers that have only the emitted BLADE string
        # (e.g. train_full's validation loop). Audit-only: never influences action selection.
        self.last_attack_target_id: Optional[str] = None

    def _location_of(self, assignment: Assignment) -> Optional[Location]:
        """Resolve the target location of an assignment's step, or None if out of range.

        Mirrors the index validation in `_candidate_for_agent` so that invalid /
        location-less assignments are treated as "unlocated" and ordered last — they remain
        executable (the candidate loop emits an empty action and advances) exactly as before.
        """
        t_idx, s_idx, _level = assignment
        if not (0 <= int(t_idx) < len(self.tasks)):
            return None
        steps = self.tasks[int(t_idx)].steps
        if not (0 <= int(s_idx) < len(steps)):
            return None
        return getattr(steps[int(s_idx)], "location", None)

    def _build_nn_queue(self, aid: str, assigns: Sequence[Assignment]) -> List[Assignment]:
        """Build an agent's queue: levels in ascending order, each level greedily ordered by
        nearest-neighbor over step target locations, chaining the start position level→level.

        Level 0 starts at the agent's start location; each later level starts from the
        end_location of the previous level (carried unchanged if a level had no located steps).
        """
        agent = self.agent_by_id.get(aid)
        current: Optional[Location] = getattr(agent, "location", None) if agent is not None else None

        by_level = sorted(assigns, key=lambda x: int(x[2]))  # stable sort by level_order
        ordered_all: List[Assignment] = []
        for _level, group in groupby(by_level, key=lambda x: int(x[2])):
            ordered, end = nearest_neighbor_order(
                list(group),
                location_of=self._location_of,
                start_location=current,
            )
            ordered_all.extend(ordered)
            current = end  # carry forward (== previous position when the level had no located steps)
        return ordered_all

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
        step_kind = getattr(step, "step_kind", None)

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
        if step_kind == StepKind.ATTACK and loc is not None:
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

        # Non-ATTACK step kinds may need a one-shot move to the step location first.
        # No such kind exists today; preserved for future kinds (commit only on chosen).
        if loc is not None and step_kind != StepKind.ATTACK:
            goal = (float(loc.latitude), float(loc.longitude))
            if st.last_move_goal != goal:
                return Candidate(
                    agent_id=aid,
                    action=f"move_aircraft('{aid}', [[{goal[0]}, {goal[1]}]])",
                    kind="MOVE",
                    move_goal=goal,
                )

        # Build the BLADE command for this step. This executor is the sole translation
        # layer: it constructs the simulator command from the semantic Step + the agent
        # assignment. The Step itself carries no command string.
        if step_kind == StepKind.ATTACK:
            weapon_id = getattr(agent, "weapon_id", None) or _infer_weapon_id_for_unit(observation, aid)
            if weapon_id is None:
                raise ValueError(f"Attack step requires a weapon but agent {aid} has no weapon_id")
            action = f"handle_aircraft_attack('{aid}', '{step.target_id}', '{weapon_id}', 2)"
            return Candidate(
                agent_id=aid,
                action=action,
                kind="STEP",
                task_idx=int(task_idx),
                step_idx=int(step_idx),
                target_id=str(step.target_id),
            )

        # Unknown / unsupported step_kind -> skip with empty action (never emit garbage).
        return Candidate(agent_id=aid, action="", kind="STEP", task_idx=int(task_idx), step_idx=int(step_idx))

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
            # Record the target of the attack command we are emitting this tick. Chosen
            # candidate only, so it reflects the agent actually acting (multiple agents may
            # have built ATTACK candidates this tick; only one is emitted). Audit-only.
            if chosen.target_id is not None:
                self.last_attack_target_id = chosen.target_id
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
