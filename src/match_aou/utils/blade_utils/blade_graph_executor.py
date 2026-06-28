"""blade_graph_executor.py — Phase-2 graph-RL executor (the SOLE BLADE layer).

This is the executor half of the Phase-2 graph-RL pipeline:

    graph_builder  ->  graph_action  ->  graph_effect  ->  [THIS]  ->  BLADE
      (observation)     (decision)        (plan edit)       (execute)

``graph_effect.apply_meta_action`` returns an edited ``solution`` dict but never
touches BLADE. This executor is the ONLY translation layer from the semantic plan
(``solution`` + ``tasks``) plus the live observation to BLADE action strings.

CORE PRINCIPLE — adapt vs execute:
  * The RL fires ONLY to ADAPT the plan (pop-up / peer-target triggers ->
    ``graph_effect`` -> a new ``solution`` -> ``resync``). There is NO per-tick RL.
  * This executor EXECUTES the current plan deterministically every tick.

HARD CONSTRAINT — no communication:
  Each ego acts ONLY on its own ego-local plan slice + its OWN sensors. The
  executor never reads peer RUNTIME state to make a decision, and never uses
  omniscient/global knowledge. The one place we read the world (``get_target``
  for the engaged target) is PROXIMITY-GATED to the ego's own sensor range — see
  the liveness-guard in ``_command_for_ego``.

DONE-ON-CONFIRMED-KILL:
  An ego marks ``(ego, target_id)`` done ONLY after it CONFIRMS, within its own
  sensor range, that the target is gone (``get_target`` -> None). Emitting an
  attack does NOT mark done. After firing one salvo the ego loiters in range and
  WAITS for the engine to resolve the kill, then advances. (An earlier model
  marked done on emit, valid only while lethality == 1.0; confirming the kill
  keeps the executor correct once task probability < 1.0, where a launch may
  miss — the ego re-engages instead of silently advancing past a survivor.)

WHY THIS IS A FRESH FILE (not a refactor of blade_executor_minimal.py):
  The frozen minimal executor was written against an older model. Applying the
  lens "would I build it this way if that file did not exist?", these structural
  choices changed:
    * NO positional queue cursor (``_AgentExec.idx``). Eligibility is DERIVED
      fresh from ``(plans, done)`` every tick, so a mid-episode ``resync`` needs
      no cursor surgery — it just swaps a plan slice.
    * LIST emission (one command per ego per tick), not a global round-robin
      single action. ``Game.handle_action`` accepts a list and execs each, so the
      round-robin (``_rr_cursor``/``_choose_rr``) that existed only to satisfy the
      old one-action-per-tick model is gone; every ego can act each tick.
    * NO ``last_move_goal`` shadow var. The "already en route?" test reads the
      LIVE ``aircraft.route`` instead, so there is no shadow state to desync after
      a resync.
    * TARGETED launch (``launch_aircraft_from_airbase(base, ego)``) instead of the
      FIFO-head validation, so each ego launches itself regardless of queue order.
    * 2-arg attack (weapon defaults to highest-engagement-range in the engine)
      instead of an explicit per-unit weapon lookup.
    * ``done`` keyed on ``(ego_id, target_id)`` (semantic, per-agent), not on
      ``(task_idx, step_idx)`` + a queue index — this is what makes per-agent
      done + no-comms + resync-survival work.
  Genuinely reusable, proven, pure helper ``nearest_neighbor_order`` IS reused
  (imported, not copied) — it survives the lens unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ...models import Agent, Location, Task
# Proven, pure haversine nearest-neighbor helper. Imported (not copied): it is a
# module-level pure function, so reusing it neither couples us to the frozen
# executor's class structure nor risks divergence.
from .blade_executor_minimal import nearest_neighbor_order

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)


# =============================================================================
# Tiny stateless world lookups (read the live observation; no decision logic)
# =============================================================================
# Rationale: keep every "where is this aircraft / which airbase" question in one
# place so the decision code below reads as pure intent. These touch ONLY the
# acting ego's own id — never peer runtime state used for a decision.

def _find_aircraft(scenario: object, ego_id: str) -> Optional[object]:
    for ac in getattr(scenario, "aircraft", []) or []:
        if str(getattr(ac, "id", "")) == str(ego_id):
            return ac
    return None


def _airborne(scenario: object, ego_id: str) -> bool:
    """True iff the ego is currently flying (present in scenario.aircraft)."""
    return _find_aircraft(scenario, ego_id) is not None


def _airbase_of(scenario: object, ego_id: str) -> Optional[str]:
    """Airbase id whose inventory currently holds the ego, or None."""
    for base in getattr(scenario, "airbases", []) or []:
        for ac in getattr(base, "aircraft", []) or []:
            if str(getattr(ac, "id", "")) == str(ego_id):
                bid = getattr(base, "id", None)
                return str(bid) if bid is not None else None
    return None


def _live_location(scenario: object, ego_id: str) -> Optional[Location]:
    """The ego's live position from the observation (None if grounded/dead)."""
    ac = _find_aircraft(scenario, ego_id)
    if ac is None:
        return None
    return Location(getattr(ac, "latitude", 0.0), getattr(ac, "longitude", 0.0))


# =============================================================================
# GraphPlanExecutor
# =============================================================================

class GraphPlanExecutor:
    """Deterministic, index-free executor that plays a graph-RL ``solution`` in BLADE.

    State is fully semantic and rebuildable — there is NO positional cursor, so a
    mid-episode ``resync`` (the channel through which OE / CR / SELF_PRESERVATION
    reach BLADE) is a plain plan-slice swap.
    """

    def __init__(
        self,
        *,
        tasks: List[Task],
        solution: Dict[str, Sequence[Assignment]],
        agents: Sequence[Agent],
        add_return_to_base: bool = True,
        arrival_threshold_km: float = 50.0,
        nn_ordering: bool = True,
        kill_confirm_ticks: int = 60,
    ) -> None:
        # The list that solution tuples index into.
        self.tasks: List[Task] = list(tasks)
        # The solution, per-ego slices. str-keyed to match graph_effect's output.
        self.plans: Dict[str, List[Assignment]] = {
            str(k): [tuple(t) for t in (v or [])] for k, v in solution.items()
        }
        self.agent_by_id: Dict[str, Agent] = {str(a.id): a for a in agents}
        self.add_return_to_base = bool(add_return_to_base)
        # arrival_threshold == attack range == detection range (unified, see CLAUDE.md).
        self.arrival_threshold_km = float(arrival_threshold_km)
        self.nn_ordering = bool(nn_ordering)
        # Ticks to WAIT for an emitted salvo to confirm a kill before re-firing.
        # MUST be >= the auto-selected weapon's flight time at arrival_threshold,
        # else salvos are wasted re-firing before the first lands. Default 60 covers
        # the highest-engagement-range pick (AIM-120, ~37 ticks from 50 km). Over-
        # sizing is harmless: the proximity confirm-guard advances the instant the
        # kill resolves, so this only throttles re-fire on a MISS (probability<1.0).
        #
        # CALIBRATION: while lethality == 1.0 this value never bites (the confirm-guard
        # always advances first, so an ego fires exactly once) — 60 is a safe upper bound.
        # TODO: calibrate when task probability < 1.0 is introduced. Set it just above the
        # AUTO-SELECTED weapon's measured flight time at arrival_threshold (the smoke
        # measured 36 ticks for AIM-120 from 50 km). Better still, DERIVE it at runtime
        # from the weapon's flight time so it tracks any weapon/range change instead of a
        # hard-coded constant.
        self.kill_confirm_ticks = int(kill_confirm_ticks)

        # done: the SOLE source of truth for "this ego CONFIRMED this target dead".
        # Set ONLY by the proximity confirm-guard (get_target -> None within the ego's
        # OWN sensor range), never on attack-emit. Survives resync (never cleared) so
        # completed work is not redone.
        self.done: set[Tuple[str, str]] = set()
        # Per-(ego,target) re-fire throttle: ticks remaining before another salvo is
        # allowed while waiting to confirm the kill. Transient; resync-agnostic (a
        # resynced-away target's stale entry is harmless; a resynced-in target fires
        # immediately on arrival).
        self.attack_cooldown: Dict[Tuple[str, str], int] = {}
        # Egos confirmed DEAD (crashed / removed from the sim). is_done() treats their
        # remaining assignments as terminally unsatisfiable -> we accept the lost task
        # utility rather than hang forever. Populated by the dead-branch below.
        self.dead: set[str] = set()
        # Single-issue RTB latch (aircraft_return_to_base is a TOGGLE in BLADE;
        # issuing it twice cancels the RTB). Also latched for landed/dead egos so
        # is_done() can terminate without an observation.
        #
        # LATENT INVARIANT: this single-issue latch is safe ONLY because the BLUE side
        # does NOT enable the engine doctrine AIRCRAFT_RTB_WHEN_OUT_OF_RANGE (confirmed
        # false in strike_training_4v5.json). If that doctrine is ever turned on, the
        # engine autonomously TOGGLES aircraft.rtb on bingo fuel behind this latch's back
        # (Game.update_all_aircraft_position issues the doctrine RTB); a later
        # executor-issued aircraft_return_to_base would then toggle rtb back OFF
        # (aircraft_return_to_base is the toggle), cancelling the RTB and desyncing our
        # single-issue assumption. Keep that doctrine off, or make RTB doctrine-aware first.
        self.rtb_issued: Dict[str, bool] = {}
        # Audit-only: target of the most recent attack emitted (validation/logs).
        self.last_attack_target_id: Optional[str] = None

    # ---- assignment resolution ------------------------------------------------

    def _resolve_step(self, assignment: Assignment) -> Optional[object]:
        """Resolve an assignment's Step (carries .target_id and .location), or None.

        Out-of-range task/step indices yield None and are treated as "nothing to
        execute" (skipped), exactly like the frozen executor's index guards.
        """
        t_idx, s_idx, _lv = assignment
        if not (0 <= int(t_idx) < len(self.tasks)):
            return None
        steps = self.tasks[int(t_idx)].steps
        if not (0 <= int(s_idx) < len(steps)):
            return None
        return steps[int(s_idx)]

    # ---- per-agent eligibility (recomputed every call; non-monotonic) ---------

    def _eligible(self, ego_id: str, scenario: object) -> List[Assignment]:
        """Ego's not-done assignments at its CURRENT level, nearest-neighbor ordered.

        PER-AGENT level: take the ego's assignments whose target is not done;
        current_level = min(level) over them (may be negative for a front-inserted
        CR/OE). Only assignments at that level are eligible this tick. A consumed
        front-insert lets min() rise, so the original plan resumes automatically.
        Each ego gates ONLY on its own plan + its own ``done`` — never on peers.
        """
        not_done: List[Assignment] = []
        for a in self.plans.get(ego_id, []):
            step = self._resolve_step(a)
            if step is None:
                continue  # invalid index -> not executable -> implicitly satisfied
            if (ego_id, str(step.target_id)) in self.done:
                continue
            not_done.append(a)
        if not not_done:
            return []

        cur_level = min(int(a[2]) for a in not_done)
        current = [a for a in not_done if int(a[2]) == cur_level]

        if not self.nn_ordering:
            # Legacy-style deterministic order within the level.
            return sorted(current, key=lambda x: (int(x[0]), int(x[1])))

        # Greedy nearest-neighbor seeded from the ego's LIVE observed position
        # (a shorter flight path; does not change WHICH steps execute). When the
        # ego is grounded (no live pos) the helper falls back to deterministic
        # (task_idx, step_idx) order.
        def _loc_of(a: Assignment) -> Optional[Location]:
            step = self._resolve_step(a)
            return getattr(step, "location", None) if step is not None else None

        ordered, _end = nearest_neighbor_order(
            current,
            location_of=_loc_of,
            start_location=_live_location(scenario, ego_id),
        )
        return ordered

    # ---- public API -----------------------------------------------------------

    def next_actions(self, observation: object) -> List[str]:
        """Build at most ONE BLADE command per ego for this tick; return the list.

        ``Game.handle_action`` accepts a list and execs each entry, so all egos
        that have something to do act in the same tick. Guarantees <=1 command per
        aircraft (each ego appends at most once). Empty list is valid (no-op tick).
        """
        commands: List[str] = []
        for ego_id in sorted(self.plans.keys()):  # deterministic id order
            cmd = self._command_for_ego(ego_id, observation)
            if cmd:
                commands.append(cmd)
        return commands

    def resync(
        self,
        new_solution: Dict[str, Sequence[Assignment]],
        *,
        ego_id: str,
        tasks: Optional[List[Task]] = None,
    ) -> None:
        """Swap ONE ego's plan slice (the OE / CR / SELF_PRESERVATION channel).

        Eligibility is derived fresh from ``(plans, done)`` every tick, so there
        is no positional state to fix here. ``done`` is deliberately NOT touched,
        so already-completed targets are not redone after the edit. Pass ``tasks``
        if the task list changed (e.g. a pop-up added a task node).
        """
        ego_key = str(ego_id)
        if tasks is not None:
            self.tasks = list(tasks)
        slice_ = new_solution.get(ego_key, new_solution.get(ego_id, []))
        self.plans[ego_key] = [tuple(t) for t in (slice_ or [])]

    def is_done(self) -> bool:
        """True iff every ego has no not-done assignments AND has RTB-resolved.

        RTB-resolved == rtb_issued latched (set by next_actions when an ego with an
        empty plan is airborne->RTB, or landed/dead->latched). When
        ``add_return_to_base`` is False, RTB is not required.
        """
        for ego_id in self.plans.keys():
            if ego_id in self.dead:
                continue  # crashed: assignments terminally unsatisfiable, RTB n/a
            for a in self.plans.get(ego_id, []):
                step = self._resolve_step(a)
                if step is None:
                    continue
                if (ego_id, str(step.target_id)) not in self.done:
                    return False
            if self.add_return_to_base and not self.rtb_issued.get(ego_id, False):
                return False
        return True

    # ---- single-ego command (the whole decision lives here) -------------------

    def _command_for_ego(self, ego_id: str, scenario: object) -> Optional[str]:
        airborne = _airborne(scenario, ego_id)
        in_airbase = airborne is False and _airbase_of(scenario, ego_id) is not None

        # Dead: not flying and not in any airbase (crashed / removed). Emit no
        # command, record the death (so is_done() can stop waiting on its targets),
        # and latch RTB so is_done()'s RTB check passes (internal state only).
        if not airborne and not in_airbase:
            self.dead.add(ego_id)
            self.rtb_issued[ego_id] = True
            return None

        eligible = self._eligible(ego_id, scenario)

        # Plan finished -> RTB phase (RTB-before-plan-end is RL's SELF_PRESERVATION,
        # which arrives via resync emptying the ego's plan; the executor is dumb on fuel).
        if not eligible:
            return self._rtb_or_latch(ego_id, airborne)

        # Grounded with work to do -> targeted launch (replaces FIFO-head check).
        if not airborne:
            return self._launch_command(ego_id, scenario)

        # --- airborne: CONFIRM-GUARD (PROXIMITY-GATED) then move/attack --------
        # No-comms: probe get_target ONLY for the single target the ego is engaging,
        # and ONLY once within the ego's own sensor range (arrival_threshold). This
        # prevents learning that a FAR target was killed by a peer (a comms leak),
        # and is now ALSO our done signal: an ego advances only after it has itself
        # CONFIRMED (in its own sensor range) that the engaged target is gone.
        # NOTE: this REFINES the locked pseudocode, which placed the guard before the
        # distance check; the locked KEY-INVARIANTS text mandates proximity gating
        # ("within arrival_threshold = its own sensor range"), so the guard is gated
        # by distance here. (Flagged for review.)
        live = _live_location(scenario, ego_id)
        if live is None:
            return None  # should not happen while airborne; be safe

        while eligible:
            step = self._resolve_step(eligible[0])
            if step is None or getattr(step, "location", None) is None:
                return None  # unexecutable head; skip this ego this tick
            target_id = str(step.target_id)
            target_loc = step.location
            d = live.distance_to(target_loc)
            if d <= self.arrival_threshold_km and scenario.get_target(target_id) is None:
                # In sensor range AND target CONFIRMED gone -> done (the SOLE done
                # signal), drop its re-fire cooldown, advance to the next eligible.
                self.done.add((ego_id, target_id))
                self.attack_cooldown.pop((ego_id, target_id), None)
                eligible = self._eligible(ego_id, scenario)
                continue
            break

        if not eligible:
            return self._rtb_or_latch(ego_id, airborne)

        step = self._resolve_step(eligible[0])
        target_id = str(step.target_id)
        target_loc = step.location
        d = live.distance_to(target_loc)

        if d > self.arrival_threshold_km:
            # MOVE (one-shot): only (re)issue if the live route is not already
            # heading to this target. move_aircraft CLEARS the route every call,
            # so re-issuing an identical move would needlessly reset progress.
            if not self._route_ends_at(scenario, ego_id, target_loc):
                return f"move_aircraft('{ego_id}', [[{target_loc.latitude}, {target_loc.longitude}]])"
            return None

        # IN RANGE, target still ALIVE -> ENGAGE, then WAIT to confirm the kill.
        # NOT done-on-emit: emitting an attack does NOT mark done (that would be
        # unsafe once task probability < 1.0, where a launch may miss). We fire ONE
        # salvo, then hold (loiter in range, no command) for up to kill_confirm_ticks
        # while the weapon flies and the engine resolves the kill; the CONFIRM-GUARD
        # above marks done the instant get_target -> None. If the window elapses with
        # the target STILL alive (a future miss), the cooldown reaches 0 and we
        # re-engage. For lethality == 1.0 the guard always confirms first, so the ego
        # fires exactly once. 2-arg form: weapon -> highest-range, quantity -> 2.
        #
        # FLAG (record only — handled later in the RL abort layer, NOT here): once task
        # probability < 1.0 lands, an ego can deplete its inventory without ever
        # confirming a kill. Today it would loiter in range forever (re-firing no-ops
        # because the engine launches nothing when out of weapons, and the guard never
        # confirms). A "weapons exhausted -> give up / RTB" path is needed then; the
        # executor stays dumb on fuel/weapons by design, so that abort belongs to the
        # RL SELF_PRESERVATION layer reaching us via resync (empty plan -> RTB).
        cd = self.attack_cooldown.get((ego_id, target_id), 0)
        if cd > 0:
            self.attack_cooldown[(ego_id, target_id)] = cd - 1
            return None  # salvo in flight; loiter and wait for confirmation
        self.attack_cooldown[(ego_id, target_id)] = self.kill_confirm_ticks
        self.last_attack_target_id = target_id
        return f"handle_aircraft_attack('{ego_id}', '{target_id}')"

    # ---- command builders / small predicates ---------------------------------

    def _rtb_or_latch(self, ego_id: str, airborne: bool) -> Optional[str]:
        """Issue a single RTB if airborne, else just latch (landed with plan done)."""
        if not self.add_return_to_base:
            return None
        if self.rtb_issued.get(ego_id, False):
            return None
        self.rtb_issued[ego_id] = True
        if airborne:
            return f"aircraft_return_to_base('{ego_id}')"
        return None  # already in an airbase: nothing to issue, latch only

    def _launch_command(self, ego_id: str, scenario: object) -> Optional[str]:
        """Targeted launch of THIS ego from its home base (verified in inventory)."""
        agent = self.agent_by_id.get(ego_id)
        base_id = getattr(agent, "home_base_id", None) if agent is not None else None
        base_id = str(base_id) if base_id else _airbase_of(scenario, ego_id)
        if not base_id:
            return None
        get_airbase = getattr(scenario, "get_airbase", None)
        airbase = get_airbase(base_id) if callable(get_airbase) else None
        if airbase is None:
            return None
        inventory = {str(getattr(a, "id", "")) for a in (getattr(airbase, "aircraft", []) or [])}
        if ego_id not in inventory:
            return None  # not in this base's inventory -> cannot launch
        return f"launch_aircraft_from_airbase('{base_id}', '{ego_id}')"

    def _route_ends_at(self, scenario: object, ego_id: str, target_loc: Location) -> bool:
        """True iff the ego's live route already terminates at ``target_loc``.

        Compared with a 1 km tolerance: move_aircraft stores the exact target
        waypoint, and update_all_aircraft_position only pops it within 0.5 km, so a
        single en-route waypoint sits at ~0 km from the target until arrival.
        """
        ac = _find_aircraft(scenario, ego_id)
        if ac is None:
            return False
        route = getattr(ac, "route", None) or []
        if not route:
            return False
        last = route[-1]
        try:
            last_loc = Location(last[0], last[1])
        except (TypeError, IndexError):
            return False
        return last_loc.distance_to(target_loc) <= 1.0
