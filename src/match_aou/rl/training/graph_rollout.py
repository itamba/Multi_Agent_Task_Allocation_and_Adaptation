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
    on the same base scenario JSON the selftests use (single-radius invariant: the
    generator builds discovery connectivity at the SAME radius the split checks and
    the runtime senses at).
  * setup takes the scenario JSON *CONTENT* (str), not a path -- the generator returns
    a Path, so we ``read_text()`` it.
  * every episode closes its env (``ctx.env.close()``), even on failure, so BLADE
    resources never leak across the loop.

This module imports ONLY the locked public interfaces; it modifies no existing file.
Windows-safe: pathlib paths and ASCII-only console output (cp1255 console).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .graph_episode_setup import (
    setup_episode,
    DETECTION_KM,
    PARTIAL_RATIO,
    MAX_SIM_TICKS,
)
from .graph_tick_loop import build_policy, run_episode
from .graph_reward import compute_episode_reward
from ..action.graph_action import MetaAction
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
    would pick the same column every wake). The generator knobs mirror the tick-loop
    selftest's ``VariationConfig`` defaults; ``detection_km`` is pinned to
    ``DETECTION_KM`` per episode (not exposed) to hold the single-radius invariant.
    """

    n_episodes: int = 20
    base_seed: int = 0                       # episode i uses VariationConfig(seed=base_seed+i)
    output_dir: Union[str, Path] = "rollouts"  # created if missing
    partial_ratio: float = PARTIAL_RATIO
    deterministic: bool = False              # stochastic by default (we WANT the distribution)
    max_ticks: Optional[int] = None          # pass-through to run_episode (None -> MAX_SIM_TICKS)
    record_first_episode: bool = False       # BLADE PlaybackRecorder for episode 0 only

    # --- generator knobs (same defaults as the tick-loop selftest's VariationConfig) ---
    include_sams: bool = False
    num_red_airbases: Tuple[int, int] = (3, 3)
    randomize_red_airbase_positions: bool = True
    stretch_target_ratio: float = 0.5


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


# =============================================================================
# 3. The rollout
# =============================================================================

def run_rollout(cfg: RolloutConfig) -> Dict[str, Any]:
    """Loop ``cfg.n_episodes`` full episodes with one random-weight policy; log + summarize.

    One failed episode logs a traceback, counts as failed, and the loop CONTINUES
    (its env is still closed in the ``finally``). Per-episode SCALAR records are appended
    to ``<output_dir>/rollout_records.jsonl`` (never gobs/tensors). Returns an aggregate
    summary dict (also printed as an ASCII table).
    """
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
            try:
                # --- generate + setup (bonmin solves TWICE here) ---
                t_setup = time.perf_counter()
                var = VariationConfig(
                    include_sams=cfg.include_sams,
                    num_red_airbases=cfg.num_red_airbases,
                    randomize_red_airbase_positions=cfg.randomize_red_airbase_positions,
                    stretch_target_ratio=cfg.stretch_target_ratio,
                    detection_km=DETECTION_KM,  # single-radius: gen connectivity == split == sensing
                    seed=seed,
                )
                scenario_path = gen.generate(episode=i, config=var)
                rec_path = (
                    str(out_dir) if (i == 0 and cfg.record_first_episode) else None
                )
                ctx = setup_episode(
                    scenario_path.read_text(encoding="utf-8"),
                    partial_ratio=cfg.partial_ratio,
                    recording_export_path=rec_path,
                )
                setup_seconds = time.perf_counter() - t_setup

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
                   help="base seed; episode i uses seed+i (default: %(default)s)")
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
