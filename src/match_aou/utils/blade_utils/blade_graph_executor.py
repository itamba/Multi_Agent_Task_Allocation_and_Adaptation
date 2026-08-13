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

WHY THIS IS A FRESH FILE (not a refactor of the retired minimal executor):
  This module is the sole BLADE translation layer. It was written fresh rather
  than refactored out of the earlier minimal demo executor (retired and deleted
  from `main`; preserved on the historical `flat-final` branch), which had been
  built against an older model. Applying the lens "would I build it this way if
  that file did not exist?", these structural choices changed:
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
  The genuinely reusable, proven, pure helper ``nearest_neighbor_order`` survived
  the lens unchanged and now lives in ``match_aou.utils.scheduling_utils``, the
  environment-agnostic scheduling layer. It is IMPORTED, never copied.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ...models import Agent, Location, Task
# Proven, pure haversine nearest-neighbor helper. Imported (not copied) from the
# environment-agnostic scheduling layer, so this executor and the offline route
# prediction in `graph_hidden_placement` share ONE implementation and cannot diverge.
from ..scheduling_utils import nearest_neighbor_order
# Shared enemy-enumeration (single source of truth). sensed_target_ids reuses it so
# executor sensing and generate_all_enemy_tasks agree on "which units are enemy targets".
from .scenario_factory import iter_enemy_targets

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
        # The list that solution tuples index into, PER-EGO (no-comms isolation).
        # All egos start identical at t=0 (fan the provided list out to every agent);
        # per-ego divergence happens ONLY via resync (e.g. a pop-up appended to one
        # ego's belief). A single shared list would leak one ego's pop-up into a
        # peer's task-view or collide indices -> a silent cross-agent information leak.
        self.tasks: Dict[str, List[Task]] = {str(a.id): list(tasks) for a in agents}
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

    def _resolve_step(self, ego_id: str, assignment: Assignment) -> Optional[object]:
        """Resolve an assignment's Step against the EGO's OWN task list, or None.

        Reads the acting ego's PRIVATE task list (per-ego isolation): task indices
        resolve against ``self.tasks[str(ego_id)]``, never a peer's list. An unknown
        ego (absent key) is treated as an empty list, so its assignments resolve to
        None. Out-of-range task/step indices likewise yield None and are treated as
        "nothing to execute" (skipped) rather than raising.
        """
        t_idx, s_idx, _lv = assignment
        ego_tasks = self.tasks.get(str(ego_id), [])
        if not (0 <= int(t_idx) < len(ego_tasks)):
            return None
        steps = ego_tasks[int(t_idx)].steps
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
            step = self._resolve_step(ego_id, a)
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
            step = self._resolve_step(ego_id, a)
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
        if THIS ego's task list changed (e.g. a pop-up added a task node); only
        this ego's slice is updated, so a peer's task-view is never touched (no-comms).
        """
        ego_key = str(ego_id)
        if tasks is not None:
            # Update ONLY the resynced ego's task slice — a pop-up appended to this
            # ego's belief must NOT appear in any peer's task-view (no-comms).
            self.tasks[ego_key] = list(tasks)
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
                step = self._resolve_step(ego_id, a)
                if step is None:
                    continue
                if (ego_id, str(step.target_id)) not in self.done:
                    return False
            if self.add_return_to_base and not self.rtb_issued.get(ego_id, False):
                return False
        return True

    # ---- sensing exposure (the trigger layer's eyes) --------------------------

    def sensed_target_ids(self, observation: object, ego_id: str) -> Dict[str, object]:
        """Enemy targets within the ego's OWN sensor range right now, as ``{id: unit}``.

        Thin EXPOSURE of the sensing predicate the confirm-guard already uses: the
        ego's live position (``_live_location``) and the unified ``arrival_threshold_km``
        radius (== detection range). It scans the WORLD's enemy targets (via
        ``iter_enemy_targets``), NOT ``self.tasks`` — a pop-up is by definition absent
        from the plan, so a tasks-scan could never sense it.

        No-communication: uses ONLY the ego's own live position + own sensor radius;
        peer runtime state is never read. Only LIVE enemies count (the same
        ``get_target`` liveness probe the confirm-guard uses). A grounded / dead ego
        (no live location) senses nothing -> ``{}``.

        Returns the ``{id: unit}`` MAP (values are the live BLADE units from
        ``get_target``) rather than a bare id set: ``decide_triggers`` needs the unit to
        build a pop-up Task via ``make_attack_task``, and we already resolved it for the
        liveness check — so the map is free. This is the trigger layer's SOLE sensor
        input; it is pure w.r.t. executor state (reads ``self.agent_by_id`` /
        ``self.arrival_threshold_km``, mutates nothing).

        Args:
            observation: the live BLADE Scenario observation.
            ego_id: the ego aircraft id.

        Returns:
            ``dict[str, unit]`` mapping each in-range, live enemy target-id to its live
            BLADE unit (possibly empty).
        """
        ego_id = str(ego_id)
        ego_loc = _live_location(observation, ego_id)
        if ego_loc is None:
            return {}  # grounded / dead: no live position -> senses nothing
        agent = self.agent_by_id.get(ego_id)
        if agent is None:
            return {}  # unknown ego -> cannot determine "our side"
        our_side = getattr(agent, "side_color", None)

        sensed: Dict[str, object] = {}
        for target_id, target_loc in iter_enemy_targets(observation, our_side):
            # In the ego's own sensor range? Distance first (cheap) short-circuits
            # get_target's scan, so we never probe liveness for out-of-range targets.
            if ego_loc.distance_to(target_loc) <= self.arrival_threshold_km:
                unit = observation.get_target(target_id)  # confirm-guard's liveness probe
                if unit is not None:  # still alive
                    sensed[str(target_id)] = unit
        return sensed

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
            step = self._resolve_step(ego_id, eligible[0])
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

        step = self._resolve_step(ego_id, eligible[0])
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


# =============================================================================
# Self-test (hand-built stubs; NO BLADE engine, NO torch, NO solver)
# =============================================================================

def _selftest() -> None:
    """Exercise ``sensed_target_ids`` geometry / liveness / no-live-position paths.

    Run under nlp_env from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.utils.blade_utils.blade_graph_executor
    """
    from types import SimpleNamespace
    from ...models import Step, StepKind  # only the isolation section builds real Steps

    def _agent(aid: str, side: str) -> Agent:
        # Only .id / .side_color are read by sensed_target_ids; the rest are inert.
        return Agent(
            location=Location(32.0, 35.0),
            capabilities=[],
            budget=0.0,
            move_cost_function=lambda s, d: 0.0,
            agent_id=aid,
            side_color=side,
        )

    def _unit(uid: str, lat: float, lon: float, side: str) -> SimpleNamespace:
        return SimpleNamespace(id=uid, latitude=lat, longitude=lon, side_color=side, altitude=0)

    # Ego at (32.0, 35.0). arrival_threshold = 50 km. 0.1 deg lat ~= 11.1 km.
    ego_id = "ego1"
    ego_ac = SimpleNamespace(id=ego_id, latitude=32.0, longitude=35.0)

    # Enemy (red) targets:
    fac_in = _unit("fac_in", 32.10, 35.0, "red")       # ~11 km  -> in range, live
    fac_killed = _unit("fac_killed", 32.05, 35.0, "red")  # ~5.5 km -> in range but KILLED
    fac_far = _unit("fac_far", 33.0, 35.0, "red")      # ~111 km -> out of range
    ab_in = _unit("ab_in", 31.95, 35.0, "red")         # ~5.5 km -> in range, live
    # Friendly (blue) unit close by -> excluded by side (not an enemy target).
    fac_friend = _unit("fac_friend", 32.0, 35.02, "blue")

    class StubScenario:
        def __init__(self, aircraft):
            self.aircraft = aircraft
            self.facilities = [fac_in, fac_killed, fac_far, fac_friend]
            self.airbases = [ab_in]
            self.ships = []

        def get_target(self, target_id):
            # "fac_killed" is still enumerated but reads as dead (get_target -> None),
            # so it exercises the liveness guard INDEPENDENTLY of enumeration.
            if str(target_id) == "fac_killed":
                return None
            for u in (self.facilities + self.airbases + self.ships):
                if str(u.id) == str(target_id):
                    return u
            return None

    ex = GraphPlanExecutor(
        tasks=[],
        solution={},
        agents=[_agent(ego_id, "blue")],
        arrival_threshold_km=50.0,
    )

    print("=" * 72)
    print("blade_graph_executor.sensed_target_ids self-test")
    print("=" * 72)

    # (1) Airborne ego: in-range live enemies only, returned as an {id: unit} map.
    #     Out-of-range, killed, and friendly units are all excluded.
    scenario = StubScenario(aircraft=[ego_ac])
    sensed = ex.sensed_target_ids(scenario, ego_id)
    assert set(sensed.keys()) == {"fac_in", "ab_in"}, sensed
    # Values are the live BLADE units (identity-equal to the scenario's units).
    assert sensed["fac_in"] is fac_in and sensed["ab_in"] is ab_in
    print(f"[1] in-range live enemies only (id -> unit): {sorted(sensed)}   OK")
    print("    (fac_far out-of-range, fac_killed dead, fac_friend friendly -> excluded)")

    # (2) Grounded / dead ego (absent from scenario.aircraft) -> {}.
    grounded = StubScenario(aircraft=[])
    assert ex.sensed_target_ids(grounded, ego_id) == {}
    print("[2] grounded / dead ego (no live location) -> {}   OK")

    # (3) Unknown ego (airborne but not in agent_by_id -> side unknown) -> {}.
    ghost = SimpleNamespace(id="ghost", latitude=32.0, longitude=35.0)
    assert ex.sensed_target_ids(StubScenario(aircraft=[ghost]), "ghost") == {}
    print("[3] unknown ego (no MATCH-AOU agent -> no side) -> {}   OK")

    # (4) Scans the WORLD, not self.tasks: the executor has zero tasks yet still senses
    #     both in-range enemies (proves pop-ups, absent from the plan, are sensable).
    #     self.tasks is now the per-ego dict shape: {ego_id: []} (fanned out to agents).
    assert ex.tasks == {ego_id: []} and set(sensed.keys()) == {"fac_in", "ab_in"}
    print("[4] scans world enemies (self.tasks empty) -> pop-ups are sensable   OK")

    # -------------------------------------------------------------------------
    # PER-EGO TASK-LIST ISOLATION (no-comms RED LINE)
    # -------------------------------------------------------------------------
    # A 2-ego executor (A, B) with IDENTICAL initial tasks. A pop-up appended to
    # one ego's belief (via resync) must never reach the peer's task-view, and an
    # index collision between two egos' pop-ups must resolve to each ego's OWN task.
    print("-" * 72)
    print("per-ego task-list isolation (no-comms)")

    def _iso_agent(aid: str) -> Agent:
        return Agent(
            location=Location(32.0, 35.0),
            capabilities=[],
            budget=0.0,
            move_cost_function=lambda s, d: 0.0,
            agent_id=aid,
            side_color="blue",
        )

    def _iso_task(target_id: str, lat: float, lon: float) -> Task:
        # One ATTACK step carrying .target_id and .location (all the executor reads).
        step = Step(
            location=Location(lat, lon),
            target_id=target_id,
            capabilities=[],
            probability=1.0,
            effort=1,
            step_kind=StepKind.ATTACK,
        )
        return Task(steps=[step], utility=100.0)

    # Identical initial tasks: task0 -> "t0", task1 -> "t1".
    init_tasks = [_iso_task("t0", 32.10, 35.0), _iso_task("t1", 31.90, 35.0)]
    # A owns task0, B owns task1 (both at level 0).
    iso_solution = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}
    iso_ex = GraphPlanExecutor(
        tasks=init_tasks,
        solution=iso_solution,
        agents=[_iso_agent("A"), _iso_agent("B")],
        arrival_threshold_km=50.0,
    )
    # Grounded scenario: both egos absent from .aircraft, so _eligible falls back to
    # deterministic order (no live position needed for the isolation checks).
    grounded_world = SimpleNamespace(aircraft=[], facilities=[], airbases=[], ships=[])

    # Fanned-out identical lists at t=0 (distinct list objects, equal content).
    assert iso_ex.tasks["A"] == init_tasks and iso_ex.tasks["B"] == init_tasks
    assert iso_ex.tasks["A"] is not iso_ex.tasks["B"]

    # Baselines captured BEFORE any resync.
    b_tasks_before = list(iso_ex.tasks["B"])
    b_eligible_before = iso_ex._eligible("B", grounded_world)

    # (ISO-1) Append a pop-up to A ONLY. B's task list stays byte-identical; A's grows.
    popup_a = _iso_task("popupA", 32.20, 35.0)
    iso_ex.resync({"A": [(0, 0, 0), (2, 0, 0)]}, ego_id="A", tasks=init_tasks + [popup_a])
    assert iso_ex.tasks["B"] == b_tasks_before, "ISO-1: B's task-view leaked A's pop-up!"
    assert popup_a in iso_ex.tasks["A"] and len(iso_ex.tasks["A"]) == 3
    print("[ISO-1] pop-up appended to A only; B's task-view unchanged   OK")

    # (ISO-2) B resolves against its OWN list: index 2 is out-of-range for B (-> None),
    #         never A's pop-up; and B's eligibility is identical to before A's resync.
    assert iso_ex._resolve_step("B", (2, 0, 0)) is None, "ISO-2: B saw A's index-2 pop-up!"
    assert iso_ex._eligible("B", grounded_world) == b_eligible_before
    print("[ISO-2] B resolves via its own list; eligibility unchanged by A's resync   OK")

    # (ISO-3) Index collision: give A and B DIFFERENT pop-ups at the SAME index (2),
    #         each via its own resync. Each ego's index 2 resolves to ITS OWN task.
    popup_b = _iso_task("popupB", 31.80, 35.0)
    iso_ex.resync({"B": [(1, 0, 0), (2, 0, 0)]}, ego_id="B", tasks=init_tasks + [popup_b])
    assert iso_ex.tasks["A"][2] is popup_a and iso_ex.tasks["B"][2] is popup_b
    step_a = iso_ex._resolve_step("A", (2, 0, 0))
    step_b = iso_ex._resolve_step("B", (2, 0, 0))
    assert step_a is not None and str(step_a.target_id) == "popupA"
    assert step_b is not None and str(step_b.target_id) == "popupB"
    assert step_a is not step_b, "ISO-3: cross-bleed between A's and B's index-2 task!"
    print("[ISO-3] same-index pop-ups resolve to each ego's OWN task (no cross-bleed)   OK")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
