"""graph_tick_loop.py — the single-episode graph-RL rollout driver (Stages 2-6).

This is the tick-loop that plays ONE episode of the graph-RL pipeline. It consumes
the :class:`EpisodeContext` produced by ``graph_episode_setup.setup_episode``
(Stage 0) and drives, per tick:

    Stage 0 (setup)  ->  [THIS: run_episode]
                            per tick:
                              Phase 1  sense -> decide_triggers -> (on wake) RL
                                       (graph_builder -> encoder -> action -> effect)
                              Phase 2  executor.next_actions -> env.step

Pipeline position (per WAKE, inside Phase 1):

    sensed_target_ids -> decide_triggers -> build_graph_observation -> GraphEncoder
      -> ActionHead -> sample_action -> apply_meta_action -> executor.resync

This module is graph-native.

SCOPE (what this file is / is NOT)
----------------------------------
IS: the ONE-episode rollout + the reward/PPO seam (:class:`Transition`). Every wake
produces one detached ``Transition`` (inference-only; grads are recomputed by the
future PPO update).

IS NOT: no outer episode-loop, no PPO update, no reward computation, no buffer — those
are separate tasks. It also does NOT reimplement the agent lifecycle: the executor OWNS
death / RTB / done / kill-confirmation (``executor.dead`` / ``rtb_issued`` /
``executor.done`` / ``is_done()``); the loop only CALLS it.

THE TWO-PHASE NO-COMMS BACKBONE (load-bearing — do not merge the phases)
------------------------------------------------------------------------
Within a tick, Phase 1 runs EVERY ego's sense+trigger+(on-wake)RL against the SAME
``obs`` snapshot and edits ONLY that ego's private belief — with NO ``env.step`` /
BLADE mutation. Only after all egos have decided does Phase 2 issue ONE ``env.step``
for the whole tick. Because BLADE does not advance until every ego has already sensed
and decided on the identical snapshot, the Phase-1 ego ITERATION ORDER cannot affect
the outcome — that structural property is the no-communication guarantee. Merging the
phases (stepping mid-Phase-1) would let an earlier ego's action change the snapshot a
later ego senses, i.e. an implicit communication channel.

Each belief edit is confined to the acting ego's belief (``ctx.beliefs[ego_id]``),
which is mutually independent from every peer's (``Belief.independent`` — see
``belief.py`` / ``graph_episode_setup``), so no edit can leak across egos.

AN EGO THAT HAS COMMITTED TO RETURN LEAVES PHASE 1
--------------------------------------------------
Because completion is now PHYSICAL (``executor.is_done(observation)`` requires a
non-dead ego to be back in an airbase inventory, not merely to have been ordered
home), an episode keeps ticking while aircraft fly the ride home. Those extra ticks
exist ONLY to let the engine resolve the lifecycle -- landing, or running the tank
dry -- so once an ego's ``rtb_issued`` latch is set it is SKIPPED in Phase 1: no
sensing, no trigger, no wake, no policy inference, no belief edit, no transition.
Without that guard the returning ego would keep sensing targets it has already
abandoned and manufacture fresh decisions out of the return leg, which is a change of
research semantics, not a lifecycle fix. Peers are untouched and continue normally,
and Phase 2 still runs for every ego every tick, so the two-phase structure and the
one-snapshot no-comms property are exactly as before.

EXOGENOUS EVENTS ENTER AT THE TOP OF A TICK, NEVER INSIDE PHASE 1
-----------------------------------------------------------------
FD-BASELINE-v1's fuel-damage event (``graph_fuel_damage``) is the first thing a tick
does, BEFORE the per-ego Phase-1 loop begins. That placement is the whole no-comms
argument for it: the event mutates one live BLADE aircraft's ``current_fuel``, and every
ego — the damaged one included — then senses and decides against the SAME post-event
snapshot. Applying it partway through the ego loop would make some egos see the
pre-damage world and others the post-damage world, i.e. the outcome would depend on
Phase-1 ego ITERATION ORDER, which is precisely the implicit communication channel the
two-phase split exists to close. The event is ego-local twice over: only the selected ego
receives ``fuel_damage=True``, and a peer's graph row carries no fuel at all (the builder
gives peers ``fuel_norm = 0.0`` by construction). The feature is OFF by default
(``fuel_damage=None``), so a loop without it is byte-unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from ..observation.central_graph_builder import CentralStateRecorder
from ..observation.graph_builder import (
    build_graph_observation,
    GraphObservation,
    GraphObservationConfig,
)
from ..action.graph_action import ActionHead, build_action_mask, sample_action
from ..action.graph_effect import apply_meta_action
from ..action.graph_trigger import decide_triggers, never_overdue
from ..agent.graph_encoder import GraphEncoder
from .graph_episode_setup import EpisodeContext, MAX_SIM_TICKS
from .graph_fuel_damage import FuelDamageController
from .belief import Belief

# Default per-episode tick cap. Equal to the env's TimeLimit (setup passes
# ``max_episode_steps=MAX_SIM_TICKS``) so the loop and BLADE expire together — the loop
# hits ``truncated`` from the TimeLimit rather than silently out-running it.
_DEFAULT_MAX_TICKS: int = MAX_SIM_TICKS


# =============================================================================
# 1. Policy bundle (encoder + head), built ONCE and shared across episodes
# =============================================================================

@dataclass
class Policy:
    """The graph policy: the shared encoder + the per-node action head.

    Built ONCE by the (future) train loop via :func:`build_policy` and passed into
    every :func:`run_episode` — it lives ACROSS episodes and carries the learned
    weights. ``run_episode`` NEVER constructs a policy.
    """

    encoder: GraphEncoder
    head: ActionHead


def build_policy(embed_dim: int = 64, **encoder_kwargs: Any) -> Policy:
    """Construct the encoder + head as one :class:`Policy` (call ONCE per training run).

    The head's input width is pinned to the encoder's output width
    (``ActionHead(embed_dim=encoder.embed_dim)``) so the two can never desync. Extra
    keyword arguments are forwarded to :class:`GraphEncoder` (e.g. ``model_dim``,
    ``num_heads``, ``num_layers``).

    Args:
        embed_dim: encoder output / head input embedding width.
        **encoder_kwargs: forwarded to :class:`GraphEncoder`.

    Returns:
        A :class:`Policy` bundling the encoder and head.
    """
    encoder = GraphEncoder(embed_dim=embed_dim, **encoder_kwargs)
    head = ActionHead(embed_dim=encoder.embed_dim)
    return Policy(encoder=encoder, head=head)


# =============================================================================
# 2. Transition (the reward / PPO seam) + episode result
# =============================================================================

@dataclass
class Transition:
    """One recorded RL wake — enough to later recompute grads (PPO) and a reward.

    The rollout is INFERENCE-ONLY: ``log_prob`` / ``entropy`` are stored as DETACHED
    python floats (grads are recomputed from ``gobs`` + ``(meta_action, node_v)`` in the
    PPO update epochs, which live in a separate task). ``reward`` is filled in later by
    the reward task — ``None`` until then.
    """

    gobs: GraphObservation          # the GraphObservation the decision was made on
    ego_id: str                     # the deciding ego
    tick: int                       # simulation tick of the wake
    meta_action: int                # a MetaAction value (0..2)
    node_v: int                     # chosen task-node index (== task_idx)
    log_prob: float                 # detached scalar log-prob of the joint (node, meta)
    entropy: float                  # detached scalar policy entropy at this wake
    reward: Optional[float] = None  # filled later by the reward task


@dataclass
class EpisodeResult:
    """The output of one :func:`run_episode` call.

    ``trajectory`` is the ordered list of RL wakes (empty is valid — the identity
    split stub produces ~no organic wakes). The rest are diagnostics.
    """

    trajectory: List[Transition]
    ticks: int                      # number of ticks executed (Phase-2 env.steps)
    ended: str                      # "done" | "terminated" | "truncated"
    n_wakes: int                    # == len(trajectory)
    confirmed_kills: int            # len(executor.done) — proximity-confirmed kills
    n_dead: int                     # len(executor.dead) — crashed egos


# =============================================================================
# 3. The per-wake RL step (factored out so it is unit-testable in isolation)
# =============================================================================

def _wake_decision(
    policy: Policy,
    ego_id: str,
    obs: Any,
    belief: Belief,
    executor: Any,
    cfg: GraphObservationConfig,
    tick: int,
    *,
    deterministic: bool = False,
) -> Transition:
    """Run ONE RL decision for ``ego_id`` on the current ``obs`` + its ``belief``.

    Called ONLY when a trigger woke the ego (see :func:`run_episode` Phase 1), so the
    ego is guaranteed airborne (a grounded / dead ego senses ``{}`` -> never wakes),
    which is exactly ``build_graph_observation``'s airborne precondition.

    The chain (all under ``torch.no_grad`` — inference only):
      1. build the graph observation from the ego's POST-trigger belief,
      2. encode -> per-node logits,
      3. build the additive action mask,
      4. sample a joint ``(meta_action, node_v)``,
      5. apply the meta-action to the ego's belief solution (pure plan edit),
      6. resync ONLY this ego's executor slice with the edited plan + tasks,
      7. return the recorded :class:`Transition`.

    The belief edit lands ONLY in ``belief`` (this ego's private view) and the resync
    touches ONLY this ego's executor slice — never a peer's (the no-communication red
    line).

    Args:
        policy: the shared encoder + head.
        ego_id: the deciding ego's id.
        obs: the live BLADE observation (the Phase-1 snapshot — NOT stepped here).
        belief: the ego's private, already-trigger-edited :class:`Belief`.
        executor: the ``GraphPlanExecutor`` (only ``resync`` is called).
        cfg: the graph-builder config (detection radius == executor's, built once).
        tick: current simulation tick (fed as ``current_time`` + stamped on the wake).
        deterministic: argmax instead of sampling (evaluation).

    Returns:
        The :class:`Transition` for this wake.
    """
    ego_key = str(ego_id)
    with torch.no_grad():
        gobs = build_graph_observation(
            scenario=obs,
            agent_id=ego_key,
            current_plan=belief.solution.get(ego_key),
            current_time=tick,
            tasks=belief.tasks,
            solution=belief.solution,
            precedence_relations=[],
            config=cfg,
        )
        emb = policy.encoder(gobs)
        logits = policy.head(emb)
        mask = build_action_mask(gobs)
        meta_action, node_v, log_prob, entropy = sample_action(
            logits, mask, deterministic=deterministic
        )

        # --- Belief edit (pure) + executor resync (this ego's slice ONLY) ---
        belief.solution = apply_meta_action(
            belief.solution, gobs, ego_key, meta_action, node_v, belief.tasks
        )
        executor.resync(belief.solution, ego_id=ego_key, tasks=belief.tasks)

    return Transition(
        gobs=gobs,
        ego_id=ego_key,
        tick=int(tick),
        meta_action=int(meta_action),
        node_v=int(node_v),
        log_prob=float(log_prob.item()),
        entropy=float(entropy.item()),
    )


# =============================================================================
# 4. The single-episode tick-loop
# =============================================================================

def run_episode(
    policy: Policy,
    ctx: EpisodeContext,
    cfg: Optional[GraphObservationConfig] = None,
    *,
    deterministic: bool = False,
    max_ticks: Optional[int] = None,
    fuel_damage: Optional[FuelDamageController] = None,
    central: Optional[CentralStateRecorder] = None,
) -> EpisodeResult:
    """Play ONE episode of the graph-RL pipeline and return its rollout.

    Drives the two-phase per-tick loop described in the module docstring:
    Phase 1 (all egos sense + trigger + on-wake RL on the SAME snapshot; pure belief
    edits, NO env.step), then Phase 2 (ONE deterministic ``env.step`` for the whole
    tick). The env is already reset by :func:`setup_episode`; this loop NEVER re-resets
    it — the first sense uses ``ctx.observation`` (the reset snapshot) and every later
    tick uses the observation returned by ``env.step``.

    Recording contract: recording is ARMED by :func:`setup_episode` (iff a
    ``recording_export_path`` was given -> ``ctx.record``) and DRIVEN here. When
    ``ctx.record`` is True this starts the recorder before the loop (a forced t=0
    frame), records the post-step state each tick (the recorder's ``should_record``
    throttles to a fixed sim-second cadence), and forces a terminal frame + exports on
    exit. Recording is a pure READ of engine state (scenario serialization) — it never
    mutates BLADE or alters control flow, and defaults to a no-op (``ctx.record`` False).
    The artifact lands at ``{export_path}/{scenario_name} Recording {start} - {end}.jsonl``
    (``scenario_name`` from the generator, e.g. ``episode_0000``). An episode that raises
    mid-loop exports nothing — intentional (no ``try/finally``).

    Args:
        policy: the shared encoder + head (built once, lives across episodes).
        ctx: the :class:`EpisodeContext` from :func:`setup_episode`.
        cfg: graph-builder config. If ``None`` (default) it is built ONCE here with
            ``detection_range_km`` == the executor's ``arrival_threshold_km`` — the
            single unified sensing/attack radius — so the builder and executor can
            never disagree on what "in range" means.
        deterministic: pass argmax (evaluation) down to every wake.
        max_ticks: per-episode tick cap. Defaults to the env's TimeLimit
            (``MAX_SIM_TICKS``) so the loop and BLADE expire together.
        fuel_damage: optional FD-BASELINE-v1 :class:`FuelDamageController`. ``None``
            (the default) leaves the loop byte-identical to the pre-FD behaviour. When
            supplied it is consulted ONCE per tick, at the top, before any ego is
            processed (see the module docstring); a CLEAN controller is a no-op on every
            call, so there is one code path rather than two.
        central: optional Phase-B CTDE :class:`CentralStateRecorder`. ``None`` (the
            default, and what ``actor_only`` training passes) leaves the loop
            byte-identical to the pre-CTDE behaviour -- no central state is built and
            nothing privileged is computed. When supplied it collects ONE central state
            per actor decision, captured immediately BEFORE that decision, so its
            ``samples`` are aligned 1:1 with the returned ``trajectory``. It is a
            TRAINING-ONLY companion structure: the actor's ``Transition.gobs`` stays the
            ego's private observation, and evaluation / inference never construct one.

    Returns:
        An :class:`EpisodeResult` with the wake ``trajectory`` and diagnostics. The
        fuel-damage event's own record (whether it fired, when, the observed fuel before
        and after, and which meta-action its wake produced) lives on the controller the
        caller passed in, not here -- this loop reports the EPISODE, and the controller
        reports the event.
    """
    if cfg is None:
        # Detection radius == the executor's arrival/attack threshold (the ONE unified
        # radius); reading it off the executor guarantees they stay equal.
        cfg = GraphObservationConfig(
            detection_range_km=float(ctx.executor.arrival_threshold_km),
            max_sim_ticks=MAX_SIM_TICKS,
        )

    cap = int(max_ticks) if max_ticks is not None else _DEFAULT_MAX_TICKS
    obs = ctx.observation  # the reset snapshot (seed); never re-reset the env
    trajectory: List[Transition] = []
    # Default ending: exhausting the tick budget is a time-limit truncation (same
    # meaning as the env's own TimeLimit `truncated`), so it stays within the enum.
    ended = "truncated"
    tick = -1  # so an empty loop (cap == 0) reports ticks == 0 honestly

    # Recording (armed at setup): start once and stamp the t=0 layout. Reads
    # game.current_scenario (already named by the generator) -> no "New Scenario" trap.
    if ctx.record:
        ctx.game.start_recording()
        ctx.game.record_step(force=True)  # t=0 frame: initial layout, pre-launch

    for tick in range(cap):
        # --- EXOGENOUS EVENTS: applied BEFORE the per-ego loop, never inside it. ---
        # One live-aircraft mutation, at most once per episode, so that every ego below
        # reasons from the identical post-event snapshot and Phase-1 iteration order
        # stays irrelevant (module docstring). `damaged_ego` is the ONLY ego that may be
        # told about it, and only on this one tick.
        damaged_ego: Optional[str] = None
        if fuel_damage is not None:
            damaged_ego = fuel_damage.maybe_apply(obs, tick)

        # --- Phase 1: per-ego sense + trigger + (on wake) RL. NO env.step here. ---
        for ego_id in ctx.agent_ids:
            if ego_id in ctx.executor.dead:
                continue  # crashed ego senses {} anyway; cheap early-out
            if ctx.executor.rtb_issued.get(str(ego_id), False):
                # COMMITTED TO RETURN (module docstring): its mission is over and the
                # remaining ticks are the engine flying it home. Skipping the whole
                # Phase-1 chain is what keeps the ride home from producing new
                # decisions -- no sensing, no trigger, no wake, no belief edit.
                continue
            belief = ctx.beliefs[ego_id]
            sensed = ctx.executor.sensed_target_ids(obs, ego_id)
            ego_fuel_damage = damaged_ego is not None and str(ego_id) == damaged_ego
            new_tasks, new_sol, wake, _events = decide_triggers(
                belief.tasks,
                belief.solution,
                sensed,
                eta=never_overdue,  # ETA dormant for first runs (PEER-OVERDUE off)
                ego_id=ego_id,
                clock=tick,
                fuel_damage=ego_fuel_damage,
            )
            # Persist the trigger's belief edit (pop-up append / peer-overdue removal).
            belief.tasks, belief.solution = new_tasks, new_sol
            if wake:
                if central is not None:
                    # CTDE ONLY (Phase B). One central state per actor decision,
                    # captured IMMEDIATELY BEFORE the action so it is the state the
                    # decision was made in, not the state the decision produced.
                    # Placed inside `if wake` so the samples stay aligned 1:1 with the
                    # transitions appended below, and BEFORE `_wake_decision` so the
                    # belief edit + `executor.resync` it performs are not yet visible.
                    # With two egos waking on the same tick this yields two ordered
                    # samples with NO env.step between them; the later one legitimately
                    # sees the earlier one's resynced plan (the critic is centralized).
                    # Nothing captured here reaches the actor -- see
                    # `central_graph_builder`.
                    central.capture(
                        scenario=obs,
                        agent_ids=ctx.agent_ids,
                        executor=ctx.executor,
                        current_time=tick,
                        # The ACTOR's own config, so the critic's detection radius /
                        # theater scale / tick cap are the same values by construction.
                        config=cfg,
                    )
                transition = _wake_decision(
                    policy, ego_id, obs, belief, ctx.executor, cfg, tick,
                    deterministic=deterministic,
                )
                trajectory.append(transition)
                if ego_fuel_damage and fuel_damage is not None:
                    # Attribute the decision to the event that caused this wake. Latched
                    # inside the controller, so a later organic wake of the same ego
                    # cannot overwrite what the fuel-damage wake actually chose.
                    fuel_damage.note_wake(
                        ego_id=ego_id, meta_action=transition.meta_action
                    )

        # --- Phase 2: deterministic execution. ONE env.step for the whole tick. ---
        commands = ctx.executor.next_actions(obs)
        if fuel_damage is not None:
            # A read-only look at what was ACTUALLY ORDERED this tick. This is the only
            # sound source for "did the damaged ego return to base": the executor's
            # `rtb_issued` is a lifecycle LATCH that is also set True for a DEAD ego
            # (which emits no command at all), so reading it would report an ego that
            # flew its plan into the ground as having returned to base.
            fuel_damage.note_commands(commands)
        obs, _reward, terminated, truncated, _info = ctx.env.step(commands)

        # Record the post-step state (unconditional per executed tick — before the exit
        # checks). The recorder's should_record throttles to the sim-second cadence.
        if ctx.record:
            ctx.game.record_step()

        # Executor owns the lifecycle; the loop only READS is_done() to decide to stop.
        # It is handed THIS tick's POST-STEP observation, because completion is a
        # physical fact about the world the step just produced: a non-dead ego counts
        # as resolved only once it is back in an airbase inventory, and an ego the
        # engine removed mid-return is reconciled into `executor.dead` by that same
        # call -- before the loop returns, so `n_dead` (and the terminal reward's
        # `n_lost`) sees it.
        if ctx.executor.is_done(obs):
            ended = "done"
            break
        if terminated:
            ended = "terminated"
            break
        if truncated:
            ended = "truncated"
            break

    # Terminal frame + export (force so kills / RTB are visible even if the throttle
    # would have skipped this tick; a possible duplicate final frame is harmless). An
    # episode that raised mid-loop skips this entirely and exports nothing (intentional).
    if ctx.record:
        ctx.game.record_step(force=True)
        ctx.game.export_recording()

    return EpisodeResult(
        trajectory=trajectory,
        ticks=tick + 1,
        ended=ended,
        n_wakes=len(trajectory),
        confirmed_kills=len(ctx.executor.done),
        n_dead=len(ctx.executor.dead),
    )


# =============================================================================
# Self-test (bonmin path; generates one real scenario like the sibling stages)
# =============================================================================

def _selftest() -> None:
    """Two required tests: full-loop drive + wake-helper isolation.

    Run under nlp_env (needs bonmin) from the repo root, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.training.graph_tick_loop
    """
    import copy
    import glob
    import json
    import random
    import tempfile
    from pathlib import Path

    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # PlaybackRecorder CHARACTER_LIMIT override (historical flat-era convention)

    from ...models import Location, Step, StepKind, Task
    from ...utils.blade_utils.scenario_factory import _normalize_side_color
    from ...utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig,
    )
    from ..action.graph_action import MetaAction
    from .graph_episode_setup import setup_episode

    def _tids(tasks: List[Any]) -> List[List[str]]:
        return [[str(s.target_id) for s in t.steps] for t in tasks]

    repo_root = Path(__file__).resolve().parents[4]
    base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graph_tick_loop_selftest_")

    print("=" * 72)
    print("graph_tick_loop self-test")
    print("=" * 72)

    # --- Generate ONE scenario variation (RED airbases only, no SAMs); reuse for both. ---
    gen = ScenarioGenerator(
        base_scenario_path=str(base_scenario),
        output_dir=out_dir,
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    cfg_gen = VariationConfig(
        include_sams=False,
        num_red_airbases=(3, 3),
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        seed=0,
    )
    scenario_path = str(gen.generate(episode=0, config=cfg_gen))
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_json = f.read()

    # ONE policy, built once and reused (mirrors the future train loop).
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)

    # =====================================================================
    # TEST 1 — full-loop drive: run_episode terminates, steps once per tick,
    #          and never calls env.step during Phase 1.
    # =====================================================================
    print("-" * 72)
    print("[TEST 1] full-loop drive")
    # Seed the GLOBAL random so split_tasks draws the SAME split TEST 1b re-draws below
    # (split_tasks is the one global-random consumer; TEST 1b must reproduce this episode).
    random.seed(1234)
    ctx = setup_episode(scenario_json, recording_export_path=out_dir)

    # Spies: count env.step calls and executor.next_actions calls, and capture whether
    # each step's action was a list (the executor's list-emission contract).
    step_calls = {"n": 0, "saw_nonempty_cmd": False, "all_lists": True}
    real_step = ctx.env.step
    real_next_actions = ctx.executor.next_actions

    def _spy_step(action):
        step_calls["n"] += 1
        if not isinstance(action, list):
            step_calls["all_lists"] = False
        if action:  # non-empty command list -> BLADE actually driven this tick
            step_calls["saw_nonempty_cmd"] = True
        return real_step(action)

    next_actions_calls = {"n": 0}

    def _spy_next_actions(observation):
        next_actions_calls["n"] += 1
        return real_next_actions(observation)

    ctx.env.step = _spy_step
    ctx.executor.next_actions = _spy_next_actions

    result = run_episode(policy, ctx, deterministic=True, max_ticks=2500)

    print(f"  ended={result.ended!r}  ticks={result.ticks}  n_wakes={result.n_wakes}  "
          f"confirmed_kills={result.confirmed_kills}  n_dead={result.n_dead}")
    print(f"  env.step calls={step_calls['n']}  next_actions calls={next_actions_calls['n']}")

    assert result.ended in {"done", "terminated", "truncated"}, result.ended
    assert result.ended, "EpisodeResult.ended must be set"
    assert result.ticks >= 1, result.ticks
    # Exactly ONE env.step per tick -> Phase 1 never stepped (the two-phase invariant).
    assert step_calls["n"] == result.ticks, (step_calls["n"], result.ticks)
    # The executor was consulted exactly once per tick (Phase 2).
    assert next_actions_calls["n"] == result.ticks, (next_actions_calls["n"], result.ticks)
    # Commands were issued (at minimum the launch tick) and always as a list.
    assert step_calls["saw_nonempty_cmd"], "executor never issued a non-empty command"
    assert step_calls["all_lists"], "executor emitted a non-list action"
    print("  [1a] run_episode terminated with a set 'ended' and ticks >= 1   OK")
    print("  [1b] exactly one env.step per tick (Phase 1 never stepped)      OK")
    print("  [1c] executor consulted once/tick; commands issued as lists     OK")

    # [1d] recording was genuinely produced (armed via recording_export_path=out_dir).
    rec_files = sorted(glob.glob(str(Path(out_dir) / "* Recording *.jsonl")))
    assert len(rec_files) == 1, f"expected exactly ONE recording artifact, got {rec_files}"
    rec_path = Path(rec_files[0])
    assert not rec_path.name.startswith("New Scenario"), f"unnamed recording: {rec_path.name!r}"
    rec_lines = rec_path.read_text(encoding="utf-8").splitlines()
    assert len(rec_lines) >= 2, f"recording has <2 frames (forced t=0 + terminal): {len(rec_lines)}"
    json.loads(rec_lines[0])  # first frame parses as JSON (raises if not)
    print(f"  [1d] recording produced: {rec_path.name!r} ({len(rec_lines)} frames)   OK")

    ctx.env.close()

    # =====================================================================
    # TEST 1b — observational purity: recording OFF reproduces TEST 1 exactly
    #           and leaves NO new artifact (proves recording changes nothing
    #           observable, and the default path stays a no-op).
    # =====================================================================
    print("-" * 72)
    print("[TEST 1b] recording-off purity")
    rec_before = set(glob.glob(str(Path(out_dir) / "* Recording *.jsonl")))
    random.seed(1234)  # SAME split as TEST 1 -> identical episode
    ctx_norec = setup_episode(scenario_json, recording_export_path=None)
    assert ctx_norec.record is False, "recording armed despite recording_export_path=None"
    result_norec = run_episode(policy, ctx_norec, deterministic=True, max_ticks=2500)

    got = (result_norec.ended, result_norec.ticks, result_norec.n_wakes)
    want = (result.ended, result.ticks, result.n_wakes)
    assert got == want, f"recording changed the episode: {got} != {want}"
    rec_after = set(glob.glob(str(Path(out_dir) / "* Recording *.jsonl")))
    assert rec_after == rec_before, \
        f"recording-off produced a new artifact: {sorted(rec_after - rec_before)}"
    print(f"  [1b'] record off: (ended,ticks,n_wakes)={got} matches TEST 1; "
          f"no new artifact   OK")

    ctx_norec.env.close()

    # =====================================================================
    # TEST 2 — wake helper in isolation: one ego senses an UNASSIGNED pop-up;
    #          _wake_decision runs the full chain and edits ONLY that belief.
    # =====================================================================
    print("-" * 72)
    print("[TEST 2] wake helper in isolation")
    ctx2 = setup_episode(scenario_json, recording_export_path=out_dir)
    env2 = ctx2.env
    obs = ctx2.observation

    # Get an ego airborne: FIFO-launch every blue aircraft, then advance a few ticks.
    for base in getattr(obs, "airbases", []) or []:
        if _normalize_side_color(getattr(base, "side_color", "")) != "blue":
            continue
        for _ac in list(getattr(base, "aircraft", []) or []):
            obs, _, _, _, _ = env2.step(f"launch_aircraft_from_airbase('{base.id}')")
    for _ in range(10):
        obs, _, _, _, _ = env2.step([])

    airborne = [
        ac for ac in getattr(obs, "aircraft", []) or []
        if _normalize_side_color(getattr(ac, "side_color", "")) == "blue"
    ]
    assert airborne, "TEST 2: no blue aircraft airborne after launch"
    ego_id = str(airborne[0].id)
    ego_ac = obs.get_aircraft(ego_id)

    # SYNTHETIC pop-up ~1 km from the ego, EMPTY capabilities: this deterministically
    # makes the target sensed (large radius) + capable (no required caps) + reachable
    # (a hair away, ego still near its launch base), so OPPORTUNISTIC_ENGAGEMENT is
    # available in the mask and the OE-landing check below is not left to chance. It is
    # UNASSIGNED (solution empty), i.e. a genuine pop-up from the policy's view.
    popup_loc = Location(ego_ac.latitude, ego_ac.longitude + 0.01)
    popup = Task(steps=[Step(popup_loc, "POPUP_T2", [], 1.0, 1, StepKind.ATTACK)], utility=90)
    belief = Belief.independent([popup], {})
    # A mutually-independent sibling to prove the edit does not leak across egos.
    sibling = belief.independent_copy()
    sib_tasks_snap = _tids(sibling.tasks)
    sib_sol_snap = copy.deepcopy(sibling.solution)

    cfg2 = GraphObservationConfig(detection_range_km=500.0, max_sim_ticks=MAX_SIM_TICKS)
    tick = 123

    # Sanity-probe the construction: OE must be an available column for the pop-up node,
    # so forcing OE below is deterministic (fails LOUD here if sensed/capable/reachable
    # regressed, rather than silently falling back to PLAN_COMPLIANCE).
    probe = build_graph_observation(
        scenario=obs, agent_id=ego_id, current_plan=None, current_time=tick,
        tasks=belief.tasks, solution=belief.solution, precedence_relations=[], config=cfg2,
    )
    probe_mask = build_action_mask(probe)
    assert probe_mask[0, int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)] == 0.0, (
        "TEST 2: pop-up is not OE-available (sensed/capable/reachable failed) — "
        f"task_features={probe.task_features[0].tolist()}"
    )

    # Force OE: a strong bias on the head's OE output makes argmax pick it wherever OE is
    # unmasked. A DEDICATED policy (not test 1's) keeps the two tests hermetic.
    torch.manual_seed(1)
    policy2 = build_policy(embed_dim=64)
    with torch.no_grad():
        policy2.head.mlp[-1].bias[int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)] += 50.0

    tr = _wake_decision(
        policy2, ego_id, obs, belief, ctx2.executor, cfg2, tick, deterministic=True
    )

    # (a) the full chain ran and returned a valid Transition (OE, the adaptation path).
    assert isinstance(tr, Transition)
    assert tr.ego_id == ego_id, tr.ego_id
    assert tr.tick == tick, tr.tick
    assert tr.node_v == 0, tr.node_v
    assert tr.meta_action == int(MetaAction.OPPORTUNISTIC_ENGAGEMENT), tr.meta_action
    assert isinstance(tr.log_prob, float) and isinstance(tr.entropy, float)
    assert tr.reward is None
    print(f"  wake -> node_v={tr.node_v} meta={MetaAction(tr.meta_action).name} "
          f"log_prob={tr.log_prob:.4f} entropy={tr.entropy:.4f}")

    # The OE edit LANDED in this ego's belief: it now holds an assignment to the pop-up.
    assert any(int(t[0]) == 0 for t in belief.solution.get(ego_id, [])), (
        "OE chosen but the ego's belief has no assignment to the pop-up", belief.solution
    )
    print("  [2a] full chain ran; OE assignment landed in the ego's belief      OK")

    # (b) the sibling belief is byte-unchanged (the no-comms isolation red line).
    assert _tids(sibling.tasks) == sib_tasks_snap, "sibling tasks changed!"
    assert sibling.solution == sib_sol_snap, "sibling solution changed!"
    print("  [2b] sibling belief byte-unchanged after ego A's wake edit         OK")

    ctx2.env.close()

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
