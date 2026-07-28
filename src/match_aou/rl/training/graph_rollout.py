"""graph_rollout.py -- full-pipeline rollout harness (diagnostics, NOT training).

This is a PURE CONSUMER of the locked graph pipeline (CLAUDE.md sections 3 and 5).
It loops N full episodes end-to-end with a single RANDOM-WEIGHT policy and logs
per-episode diagnostics, then prints an aggregate summary. There is NO learning:
no grads, no PPO, no buffer, no weight updates. The policy is built ONCE and only
run under the tick-loop's own ``torch.no_grad`` inference path.

Pipeline position (one episode):

    ScenarioGenerator.generate  ->  setup_episode  ->  run_episode  ->  compute_episode_reward
        (scenario JSON Path)         (EpisodeContext)   (EpisodeResult)   (EpisodeReward)

PURPOSE
-------
Prove the whole chain integrates before the PPO task lands: real organic wakes fire
(the discovery-chain split is live), the action distribution is sane, rewards land in
range, and the loop is stable across N episodes. The outer loop here is deliberately
the SAME skeleton the future PPO loop will wrap (setup -> run -> reward -> [update]),
so this harness de-risks that task.

WIRING (mirrors the sibling selftests EXACTLY -- this file adds no new pipeline logic)
-------------------------------------------------------------------------------------
  * generate -> setup: ScenarioGenerator + VariationConfig(detection_km=DETECTION_KM)
    on the same base scenario JSON the selftests use (single-radius invariant: one
    radius for sensing, arrival, attack and split adjacency).
  * the GENERATED world is known-only (B1): exactly ``n_known`` targets, Layer-1
    discovery-chain relocation off, geometry requested strictly. The hidden half is
    built by ``setup_episode``'s construction path (B3) from the solved routes --
    solve A_init -> place -> patch the scenario JSON -> reload -> solve the oracle --
    so the executed world holds ``n_known + n_hidden`` targets and a rollout episode
    has real, discoverable pop-ups. ``split_tasks`` is not called on this path.
  * setup takes the scenario JSON *CONTENT* (str), not a path -- the generator returns
    a Path, so we ``read_text()`` it.
  * every episode closes its env (``ctx.env.close()``), even on failure, so BLADE
    resources never leak across the loop.

SEEDING / REPRODUCIBILITY
------------------------
``torch.manual_seed(base_seed)`` runs ONCE before ``build_policy()`` -- it pins the
random policy weights for the whole rollout. Then EVERY episode reseeds both global
``random`` and torch with ``base_seed + i`` at the top of its iteration. That second
reseed is what makes an episode reproducible in isolation: the generator has its own
``random.Random(seed)`` and never touches global ``random``, but the tick-loop's action
sampling draws from torch's global RNG -- without the per-episode reseed, episode i
would depend on how much RNG state episodes 0..i-1 happened to consume. Hidden-target
placement deliberately does NOT ride on global ``random``: setup gets its own explicit
``random.Random(seed)``, so the placement geometry of episode i is a pure function of
its seed no matter what else consumes global randomness. Each record carries
``known_target_ids`` (the t=0 known set) so that identity is externally checkable, and
``ctx.placements`` carries the id-free geometric fingerprint of the hidden half. The
PPO loop inherits this per-episode pattern.

This module imports ONLY the locked public interfaces; it modifies no existing file.
Windows-safe: pathlib paths and ASCII-only console output (cp1255 console).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from .graph_episode_setup import (
    setup_episode,
    DETECTION_KM,
    MAX_SIM_TICKS,
)
from .graph_tick_loop import build_policy, run_episode
from .graph_reward import compute_episode_reward
from ..action.graph_action import MetaAction
from ...models import StepKind
from ...utils.blade_utils.scenario_generator import (
    ScenarioGenerator,
    VariationConfig,
)

# The base template every generated variation is derived from -- the SAME scenario the
# sibling selftests use. graph_rollout.py lives at src/match_aou/rl/training/, so
# parents[4] is the repo root (training -> rl -> match_aou -> src -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BASE_SCENARIO = _REPO_ROOT / "data" / "scenarios" / "strike_training_4v5.json"

# The three meta-action columns, in enum order (0..2). Fixed key set for the counts.
_META_NAMES = [MetaAction(i).name for i in range(len(MetaAction))]


# =============================================================================
# 1. Config
# =============================================================================

@dataclass
class RolloutConfig:
    """Knobs for a diagnostic rollout (no training).

    ``deterministic`` defaults to False on purpose: a stochastic policy is what makes
    the per-episode action distribution meaningful (an argmax random-weight policy
    would pick the same column every wake). ``detection_km`` is pinned to
    ``DETECTION_KM`` per episode (not exposed) to hold the single-radius invariant.

    THE SCENARIO CELL MIRRORS ``graph_train.TrainConfig`` FIELD FOR FIELD. It used to
    diverge -- ``(3, 3)`` targets at ``PARTIAL_RATIO`` against the trainer's ``(6, 6)``
    at 0.5 -- which meant a diagnostic rollout and a training run generated different
    worlds by default and were not comparable. B1 closes that: both build the same
    explicit known-only construction cell, and an anti-drift test compares the two
    dataclasses' defaults directly (they stay STRUCTURALLY aligned rather than sharing
    an import: the trainer is a torch/PPO leaf and this harness must not depend on it).
    """

    n_episodes: int = 20
    # Episode i uses VariationConfig(seed=base_seed+i) AND reseeds global `random`
    # + torch with that same base_seed+i at the top of the iteration, so the episode
    # is a pure function of its seed given the policy weights (which are pinned once,
    # before the loop, by torch.manual_seed(base_seed)).
    base_seed: int = 0
    output_dir: Union[str, Path] = "rollouts"  # created if missing
    deterministic: bool = False              # stochastic by default (we WANT the distribution)
    max_ticks: Optional[int] = None          # pass-through to run_episode (None -> MAX_SIM_TICKS)
    record_first_episode: bool = False       # BLADE PlaybackRecorder for episode 0 only

    # --- the offline scenario-construction reference cell (mirrors TrainConfig) ---
    # The generator writes n_known targets; setup_episode's construction path places
    # n_hidden route-relative targets and patches them in, so an episode's world holds
    # n_known + n_hidden and a record's `n_hidden` (read off split_meta) is real.
    num_agents: int = 3
    n_known: int = 3
    n_hidden: int = 3
    min_target_distance_km: float = 200.0
    min_known_separation_km: float = 100.0

    # --- generator knobs (mirrors TrainConfig) ---
    include_sams: bool = False
    randomize_red_airbase_positions: bool = True
    stretch_target_ratio: float = 0.5

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Refuse an impossible construction cell BEFORE any expensive work starts.

        The construction half of ``graph_train.TrainConfig.validate``, with the same
        verdicts, because this harness drives the SAME generator and the SAME solver: a
        cell that is invalid for a training run is invalid for a diagnostic rollout, and
        a harness that discovered it 45 seconds into bonmin (or, at ``num_agents >
        n_known``, only in the shape of a reward that never moves) would be the reason
        the two surfaces drifted apart in the first place.

        ``run_rollout`` calls this as its FIRST statement -- before it creates a
        directory, builds a policy, constructs a generator, or touches BLADE -- so a bad
        config leaves nothing behind.

        Raises:
            ValueError: on a non-positive count or distance, a negative ``n_hidden`` or
                separation, or ``num_agents > n_known``.
        """
        if int(self.n_episodes) < 1:
            raise ValueError(f"n_episodes must be >= 1, got {self.n_episodes}")
        if int(self.num_agents) < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if int(self.n_known) < 1:
            raise ValueError(
                f"n_known must be >= 1 (an episode needs a target), got {self.n_known}"
            )
        if int(self.n_hidden) < 0:
            raise ValueError(f"n_hidden must be >= 0, got {self.n_hidden}")
        if int(self.num_agents) > int(self.n_known):
            raise ValueError(
                "num_agents (%d) must be <= n_known (%d): more agents than targets "
                "forces the stacking cell in which several egos share one target and "
                "every episode returns the same reward."
                % (int(self.num_agents), int(self.n_known))
            )
        if float(self.min_target_distance_km) <= 0.0:
            raise ValueError(
                f"min_target_distance_km must be > 0, got {self.min_target_distance_km}"
            )
        if float(self.min_known_separation_km) < 0.0:
            raise ValueError(
                "min_known_separation_km must be >= 0 (0 disables the constraint), "
                f"got {self.min_known_separation_km}"
            )
        if bool(self.include_sams):
            raise ValueError(
                "include_sams=True is not supported on the construction path: hidden "
                "targets are patched in as enemy AIRBASES, and setup_episode refuses a "
                "world whose enemy units are not all airbases. Mixed SAM / facility / "
                "ship target semantics are a separate design task."
            )

        # Hazard, not an error: a researcher may probe the stalling cell deliberately.
        # Same solver, same stall, so the same warning the trainer prints.
        if int(self.n_known) < 3:
            print("[WARN] n_known=%d: fewer than 3 known tasks is the bonmin "
                  "branch-and-bound SYMMETRY-STALL region (~15 min per episode "
                  "observed instead of ~45 s). Proceeding."
                  % int(self.n_known))


# =============================================================================
# 2. Small helpers (stdlib only)
# =============================================================================

def _stats(values: List[float]) -> Dict[str, float]:
    """(mean, min, max) of a list, safe on empty (returns zeros)."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _meta_action_counts(trajectory: List[Any]) -> Dict[str, int]:
    """Count meta-actions over one trajectory, keyed by the three fixed enum names."""
    counts = {name: 0 for name in _META_NAMES}
    for tr in trajectory:
        counts[MetaAction(int(tr.meta_action)).name] += 1
    return counts


def _known_target_ids(ctx: Any) -> List[str]:
    """Sorted target ids of the t=0 KNOWN task set -- the episode's known-set identity.

    Recorded so an external check can prove two runs produced the SAME split for a
    given seed (the reproducibility claim the per-episode reseed makes). Small
    strings, so this does not breach the scalar-only rule on the records.

    Read from ``ctx.beliefs``: every ego's belief is minted from the same A_init
    baseline, so all N lists are identical -- the cross-ego assert is a cheap
    invariant check (<= ~4 egos x ~9 tasks). Extraction mirrors the builder's
    canonical form (first ATTACK step -> ``target_id``, ``steps[0]`` fallback),
    duplicated here rather than importing the builder's private helper.

    CALL AT t=0 ONLY -- i.e. after ``setup_episode``, BEFORE ``run_episode``. The
    beliefs are equal only at t=0; during a rollout the trigger layer appends
    pop-ups and the effect layer edits assignments PER EGO, so the beliefs diverge
    by design (that divergence IS the no-communication guarantee) and the assert
    below would fire on any episode that woke.
    """
    per_ego: List[List[str]] = []
    for belief in ctx.beliefs.values():
        ids: List[str] = []
        for task in belief.tasks:
            steps = getattr(task, "steps", None) or []
            step = next(
                (s for s in steps
                 if getattr(s, "step_kind", None) == StepKind.ATTACK),
                steps[0] if steps else None,
            )
            target_id = getattr(step, "target_id", None) if step is not None else None
            ids.append(str(target_id) if target_id is not None else "")
        per_ego.append(sorted(ids))

    assert per_ego and all(x == per_ego[0] for x in per_ego), (
        "beliefs disagree on the t=0 known target set (all mint from one A_init)"
    )
    return per_ego[0]


# =============================================================================
# 3. The rollout
# =============================================================================

def run_rollout(cfg: RolloutConfig) -> Dict[str, Any]:
    """Loop ``cfg.n_episodes`` full episodes with one random-weight policy; log + summarize.

    One failed episode logs a traceback, counts as failed, and the loop CONTINUES
    (its env is still closed in the ``finally``). Per-episode SCALAR records are appended
    to ``<output_dir>/rollout_records.jsonl`` (never gobs/tensors; ``known_target_ids``
    is a short list of id strings, not a gob). Returns an aggregate summary dict (also
    printed as an ASCII table).

    Every episode reseeds global ``random`` + torch with ``base_seed + i`` -- see the
    module docstring's SEEDING / REPRODUCIBILITY note.

    Raises:
        ValueError: from :meth:`RolloutConfig.validate`, which runs FIRST -- before any
            directory, policy, generator or engine import -- so an impossible cell costs
            nothing and leaves nothing behind.
    """
    cfg.validate()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scen_dir = out_dir / "scenarios"
    scen_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "rollout_records.jsonl"

    # Match the selftests' deliberate PlaybackRecorder override so a recorded episode 0
    # is not silently truncated. Lazy import (engine boundary); we always run under nlp_env.
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    # ONE policy, random weights, built once and shared across every episode (the
    # tick-loop contract: run_episode NEVER builds a policy).
    torch.manual_seed(cfg.base_seed)
    policy = build_policy()

    # ONE generator, reused across episodes; the time-feasibility cap tracks the full pool.
    gen = ScenarioGenerator(
        base_scenario_path=str(_BASE_SCENARIO),
        output_dir=str(scen_dir),
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)

    print("=" * 72)
    print("graph_rollout: %d episode(s), base_seed=%d, deterministic=%s"
          % (cfg.n_episodes, cfg.base_seed, cfg.deterministic))
    print("output_dir=%s" % str(out_dir))
    print("=" * 72)

    records: List[Dict[str, Any]] = []
    n_failed = 0

    with open(records_path, "w", encoding="utf-8") as fh:
        for i in range(cfg.n_episodes):
            ctx = None
            seed = cfg.base_seed + i
            # Per-episode reseed: the generator already derives its variation from
            # its own random.Random(seed), but split_tasks consumes GLOBAL `random`
            # and run_episode's sampling consumes torch's global RNG. Reseeding both
            # here makes episode i a pure function of `seed` (given the policy
            # weights) -- independent of how much RNG state earlier episodes consumed.
            random.seed(seed)
            torch.manual_seed(seed)
            # Hidden placement gets its OWN explicit rng (never global `random`), so the
            # constructed geometry of episode i is a pure function of `seed`.
            placement_rng = random.Random(seed)
            try:
                # --- generate + setup (bonmin solves TWICE here) ---
                t_setup = time.perf_counter()
                # The B1 construction request -- structurally identical to
                # `graph_train.build_variation_config` (anti-drift test): a KNOWN-ONLY
                # world of exactly n_known targets, Layer 1 OFF (it would cluster the
                # known targets and flatten route diversity), and the requested geometry
                # declared STRICT so the generator raises instead of weakening it.
                var = VariationConfig(
                    include_sams=cfg.include_sams,
                    num_aircraft=int(cfg.num_agents),
                    num_red_airbases=int(cfg.n_known),
                    randomize_red_airbase_positions=cfg.randomize_red_airbase_positions,
                    stretch_target_ratio=float(cfg.stretch_target_ratio),
                    min_target_distance_km=float(cfg.min_target_distance_km),
                    min_target_separation_km=float(cfg.min_known_separation_km),
                    ensure_discovery_chain=False,
                    strict_geometry=True,
                    detection_km=DETECTION_KM,  # single-radius: gen connectivity == split == sensing
                    seed=seed,
                )
                scenario_path = gen.generate(episode=i, config=var)
                rec_path = (
                    str(out_dir) if (i == 0 and cfg.record_first_episode) else None
                )
                ctx = setup_episode(
                    scenario_path.read_text(encoding="utf-8"),
                    # CONSTRUCTION PATH: the generated world is known-only, and setup
                    # builds the hidden half from the solved routes (solve -> place ->
                    # patch -> reload). `partial_ratio` is the legacy split surface and
                    # is deliberately NOT passed -- `split_tasks` never runs here.
                    n_hidden=int(cfg.n_hidden),
                    placement_rng=placement_rng,
                    recording_export_path=rec_path,
                )
                setup_seconds = time.perf_counter() - t_setup

                # Snapshot the split identity NOW, at t=0. This MUST happen before
                # run_episode: the triggers append pop-ups and the effect layer edits
                # assignments, so after the rollout the N beliefs have legitimately
                # DIVERGED and the cross-ego agreement asserted below no longer holds.
                known_tids = _known_target_ids(ctx)

                # --- rollout + reward ---
                t_ep = time.perf_counter()
                result = run_episode(
                    policy, ctx,
                    deterministic=cfg.deterministic,
                    max_ticks=cfg.max_ticks,
                )
                ep_reward = compute_episode_reward(ctx, result)
                episode_seconds = time.perf_counter() - t_ep

                # --- SCALAR-ONLY per-episode record (read split_meta keys directly) ---
                meta = ctx.split_meta
                record = {
                    "episode": i,
                    "seed": seed,
                    "scenario_path": str(scenario_path),
                    "n_agents": len(ctx.agent_ids),
                    "n_tasks_full": int(meta["full"]),
                    "n_known": int(meta["known"]),
                    "n_hidden": int(meta["hidden"]),
                    "known_target_ids": known_tids,
                    # Id-free geometric identity of the constructed hidden half -- the
                    # ONLY sound cross-run comparison key (uuids are not seed-derived).
                    "hidden_fingerprint": [
                        list(pair) for pair in meta.get("geometric_fingerprint", ())
                    ],
                    "ticks": int(result.ticks),
                    "ended": result.ended,
                    "n_wakes": int(result.n_wakes),
                    "wake_ticks": [int(tr.tick) for tr in result.trajectory],
                    "meta_action_counts": _meta_action_counts(result.trajectory),
                    "confirmed_kills": int(result.confirmed_kills),
                    "n_dead": int(result.n_dead),
                    "u_achieved": float(ep_reward.u_achieved),
                    "u_oracle": float(ep_reward.u_oracle),
                    "ratio": float(ep_reward.ratio),
                    "penalty": float(ep_reward.penalty),
                    "reward": float(ep_reward.reward),
                    "setup_seconds": float(setup_seconds),
                    "episode_seconds": float(episode_seconds),
                }
                records.append(record)
                fh.write(json.dumps(record) + "\n")
                fh.flush()

                print("[ep %2d] ended=%-10s ticks=%5d wakes=%2d kills=%2d dead=%d "
                      "reward=%+.4f setup=%5.1fs ep=%5.1fs"
                      % (i, result.ended, result.ticks, result.n_wakes,
                         result.confirmed_kills, result.n_dead, ep_reward.reward,
                         setup_seconds, episode_seconds))

            except Exception as exc:  # one failed episode must not abort the loop
                n_failed += 1
                print("[ep %2d] FAILED (seed=%d): %s: %s"
                      % (i, seed, type(exc).__name__, exc))
                traceback.print_exc()
            finally:
                if ctx is not None:
                    try:
                        ctx.env.close()
                    except Exception:
                        pass

    summary = _summarize(records, n_failed, cfg, records_path, out_dir)
    _print_summary(summary)
    return summary


# =============================================================================
# 4. Aggregate + print
# =============================================================================

def _summarize(
    records: List[Dict[str, Any]],
    n_failed: int,
    cfg: RolloutConfig,
    records_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Aggregate the per-episode records into a summary dict (over OK episodes only)."""
    n_ok = len(records)

    wakes = [r["n_wakes"] for r in records]
    rewards = [r["reward"] for r in records]
    ticks = [r["ticks"] for r in records]
    setup_secs = [r["setup_seconds"] for r in records]
    ep_secs = [r["episode_seconds"] for r in records]
    episodes_with_wake = sum(1 for w in wakes if w >= 1)

    meta_totals = {name: 0 for name in _META_NAMES}
    for r in records:
        for name in _META_NAMES:
            meta_totals[name] += r["meta_action_counts"].get(name, 0)

    ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
    for r in records:
        if r["ended"] in ended_counts:
            ended_counts[r["ended"]] += 1

    return {
        "n_episodes": cfg.n_episodes,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "wakes": _stats([float(w) for w in wakes]),
        "episodes_with_wake": episodes_with_wake,
        "frac_with_wake": (episodes_with_wake / n_ok) if n_ok else 0.0,
        "meta_action_totals": meta_totals,
        "reward": _stats(rewards),
        "ended_counts": ended_counts,
        "ticks_mean": _stats([float(t) for t in ticks])["mean"],
        "setup_seconds_mean": _stats(setup_secs)["mean"],
        "episode_seconds_mean": _stats(ep_secs)["mean"],
        "records_path": str(records_path),
        "output_dir": str(out_dir),
        "records": records,
    }


def _print_summary(s: Dict[str, Any]) -> None:
    """Print the summary as an ASCII table (no unicode -- cp1255 console)."""
    print("-" * 72)
    print("ROLLOUT SUMMARY (%d episode(s))" % s["n_episodes"])
    print("-" * 72)
    print("episodes:  ok=%d  failed=%d" % (s["n_ok"], s["n_failed"]))
    w = s["wakes"]
    print("wakes:     mean=%.2f  min=%d  max=%d   with_wake=%d/%d (%.0f%%)"
          % (w["mean"], int(w["min"]), int(w["max"]),
             s["episodes_with_wake"], s["n_ok"], 100.0 * s["frac_with_wake"]))
    mt = s["meta_action_totals"]
    print("meta-acts: PLAN_COMPLIANCE=%d  OPPORTUNISTIC_ENGAGEMENT=%d  "
          "SELF_PRESERVATION_ABORT=%d"
          % (mt["PLAN_COMPLIANCE"], mt["OPPORTUNISTIC_ENGAGEMENT"],
             mt["SELF_PRESERVATION_ABORT"]))
    r = s["reward"]
    print("reward:    mean=%+.4f  min=%+.4f  max=%+.4f"
          % (r["mean"], r["min"], r["max"]))
    ec = s["ended_counts"]
    print("ended:     done=%d  terminated=%d  truncated=%d"
          % (ec["done"], ec["terminated"], ec["truncated"]))
    print("ticks:     mean=%.1f" % s["ticks_mean"])
    print("timing:    setup_mean=%.1fs  episode_mean=%.1fs"
          % (s["setup_seconds_mean"], s["episode_seconds_mean"]))
    print("records:   %s" % s["records_path"])
    print("-" * 72)


# =============================================================================
# 5. CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    d = RolloutConfig()
    p = argparse.ArgumentParser(
        description="Full-pipeline diagnostic rollout (no training)."
    )
    p.add_argument("--episodes", type=int, default=d.n_episodes,
                   help="number of episodes (default: %(default)s)")
    p.add_argument("--seed", type=int, default=d.base_seed,
                   help="base seed; episode i generates with seed+i and reseeds "
                        "global random + torch with it (default: %(default)s)")
    p.add_argument("--out", type=str, default=str(d.output_dir),
                   help="output directory (default: %(default)s)")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax instead of sampling (default: stochastic)")
    p.add_argument("--record-first", action="store_true",
                   help="record episode 0 with the BLADE PlaybackRecorder")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = RolloutConfig(
        n_episodes=args.episodes,
        base_seed=args.seed,
        output_dir=args.out,
        deterministic=args.deterministic,
        record_first_episode=args.record_first,
    )
    run_rollout(cfg)


if __name__ == "__main__":
    main()
