"""
Evaluate a Trained Checkpoint (or the Solver-Only Baseline) on Held-Out Scenarios
==================================================================================

This script reuses the training-episode machinery from `train_full.py` but
runs every episode in eval mode:

- Buffer storage and PPO updates are skipped (`replay_only=True`).
- For RL checkpoints, `trainer.get_action` is monkey-patched to argmax
  (deterministic) via `ActorCriticNetwork.get_greedy_action`.
- For the `baseline` mode (`--checkpoint baseline`), `trainer.get_action`
  is patched to always return NOOP so the partial-plan executor runs
  alone, producing a pure MATCH-AOU baseline result with no RL adaptation.

Per-episode and aggregate metrics are written to a JSON file at `--output`.

Usage
-----

    python evaluate_checkpoint.py \\
        --checkpoint runs/full_3000_reward_refactor/models/checkpoint_ep600.pt \\
        --n-episodes 200 \\
        --eval-seed 1337 \\
        --output eval_results/ep600.json \\
        --fuel-damage --vary-scenarios

    # Solver-only baseline (no RL):
    python evaluate_checkpoint.py \\
        --checkpoint baseline \\
        --n-episodes 200 \\
        --eval-seed 1337 \\
        --output eval_results/baseline.json \\
        --fuel-damage --vary-scenarios

Determinism
-----------
Scenario seed for episode `e` is `eval_seed + e`. Fuel-damage seed is
`episode_num` (i.e. `e`). With deterministic argmax (and NOOP for baseline)
every config sees identical scenarios + identical fuel-damage events,
so cross-config comparisons are like-for-like.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Project-internal imports — these all come from train_full.py and the
# RL package. We deliberately reuse them rather than reimplementing the
# simulation loop.
import train_full as TF
from train_full import (
    MAX_AGENTS,
    MAX_SIM_TICKS,
    OUTPUT_DIR,
    VARY_BASE,
    INCLUDE_SAMS,
    FUEL_DAMAGE_ENABLED,
    VARY_SCENARIOS,
    setup_blade_env,
    reload_scenario,
    train_episode,
)
from match_aou.rl.agent.network import ActorCriticNetwork
from match_aou.rl.training import (
    PPOConfig,
    PPOTrainer,
    RewardConfig,
)
from match_aou.rl.training.reward import compute_episode_reward
from match_aou.rl.observation import ObservationConfig
from match_aou.utils.blade_utils.scenario_generator import (
    ScenarioGenerator,
    VariationConfig,
)

logger = logging.getLogger("evaluate_checkpoint")


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint (or solver-only baseline) on "
                    "held-out scenarios.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint, OR the literal string 'baseline' to "
             "run solver-only (no RL adaptation).",
    )
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=1337,
                        help="Seed offset; episode e uses scenario seed eval_seed+e.")
    parser.add_argument("--output", required=True,
                        help="Path to write the JSON results file.")
    parser.add_argument("--fuel-damage", action="store_true",
                        default=FUEL_DAMAGE_ENABLED)
    parser.add_argument("--vary-scenarios", action="store_true",
                        default=VARY_SCENARIOS)

    # Passthrough / defaults that match training config
    parser.add_argument("--scenario", default="data/scenarios/strike_training_4v5.json")
    parser.add_argument("--max-ticks", type=int, default=MAX_SIM_TICKS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-aircraft", type=int, default=2)
    parser.add_argument("--max-aircraft", type=int, default=3)
    parser.add_argument("--min-facilities", type=int, default=2)
    parser.add_argument("--max-facilities", type=int, default=4)
    parser.add_argument("--max-target-dist", type=float, default=2500.0)
    parser.add_argument("--min-red-airbases", type=int, default=3)
    parser.add_argument("--max-red-airbases", type=int, default=5)
    parser.add_argument("--vary-base", action="store_true", default=VARY_BASE)
    parser.add_argument("--include-sams", action="store_true", default=INCLUDE_SAMS)
    parser.add_argument("--base-shift-km", type=float, default=150.0)
    parser.add_argument("--allowed-aircraft", nargs="+", default=None)
    parser.add_argument("--stretch-ratio", type=float, default=0.5)
    parser.add_argument("--time-feasible-max-km", type=float, default=None)
    parser.add_argument("--work-dir", default="eval_results/_workdir",
                        help="Scratch directory for BLADE recordings dir / "
                             "generated scenario JSONs. Wiped per run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-episode INFO logs.")

    return parser.parse_args()


# =============================================================================
# Setup helpers
# =============================================================================

def setup_logging(quiet: bool) -> None:
    """Minimal logging — eval should print episode counter and aggregates only.

    We suppress most train_full chatter by setting INFO root level and pushing
    train_full's heavy debug loggers to WARNING. The eval script's own logger
    keeps INFO so progress lines are visible.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    root.addHandler(ch)
    # Our own logger sits above the root level so it surfaces.
    logger.setLevel(logging.INFO if not quiet else logging.WARNING)


def make_trainer(work_dir: Path) -> PPOTrainer:
    """Build a PPOTrainer with the same shape as training. We pass a fresh
    model_dir under the work_dir; checkpoints saved during eval would land
    there but we never call save_checkpoint, so the dir stays empty."""
    obs_dim = 6 + (3 * 6) + 6  # 30
    action_dim = 5
    network = ActorCriticNetwork(
        obs_dim=obs_dim, action_dim=action_dim,
        n_agents=MAX_AGENTS, hidden_size=128,
    )
    config = PPOConfig(
        obs_dim=obs_dim, action_dim=action_dim,
        n_agents=MAX_AGENTS, hidden_size=128,
        learning_rate=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
        ppo_epochs=4, batch_size=64, max_grad_norm=0.5,
        value_coef=0.5, entropy_coef=0.01, buffer_capacity=2048,
        reward_config=RewardConfig(),
        model_dir=str(work_dir / "models"),
    )
    return PPOTrainer(network, config)


def patch_get_action_baseline(trainer: PPOTrainer) -> None:
    """Baseline mode: actor is silenced. Always returns NOOP so train_episode's
    `if rl_action != 0` branch never overrides the executor."""
    def _noop_action(local_obs, global_obs, action_mask):
        return 0, 0.0, 0.0
    trainer.get_action = _noop_action  # type: ignore[assignment]


def patch_get_action_deterministic(trainer: PPOTrainer) -> None:
    """RL eval mode: deterministic argmax over the action distribution.

    Mirrors PPOTrainer.get_action's signature/return so callers don't
    notice. We compute a critic value too (in case anything downstream
    expects a finite number), but it's not stored anywhere — replay_only
    skips buffer.store().
    """
    network = trainer.network
    device = trainer.device

    def _det_action(local_obs, global_obs, action_mask):
        with torch.no_grad():
            obs_t = torch.FloatTensor(local_obs).to(device)
            global_t = torch.FloatTensor(global_obs).to(device)
            mask_t = torch.BoolTensor(action_mask).to(device)
            action = network.get_greedy_action(obs_t, mask_t)
            value = network.get_value(global_t).squeeze().item()
        return int(action), 0.0, float(value)
    trainer.get_action = _det_action  # type: ignore[assignment]


# =============================================================================
# Per-episode runner
# =============================================================================

def run_one_episode(
    trainer: PPOTrainer,
    game,
    env,
    obs_config: ObservationConfig,
    scenario_gen: Optional[ScenarioGenerator],
    ep_idx: int,
    args: argparse.Namespace,
) -> Dict:
    """Run a single eval episode and return a flat per-episode record.

    The seed for scenario generation is `eval_seed + ep_idx` — this is the
    knob that controls which scenario this config sees. All configs share
    the same eval_seed, so all configs see the same scenario sequence.
    """
    # --- Generate / select scenario ---
    if scenario_gen is not None:
        min_rab = args.min_red_airbases
        if not args.include_sams and min_rab < 1:
            min_rab = 1
        ep_config = VariationConfig(
            include_sams=args.include_sams,
            num_aircraft=(args.min_aircraft, args.max_aircraft),
            allowed_aircraft_classes=args.allowed_aircraft,
            num_facilities=(args.min_facilities, args.max_facilities),
            num_red_airbases=(min_rab, args.max_red_airbases),
            randomize_facility_positions=True,
            randomize_red_airbase_positions=True,
            max_target_distance_km=args.max_target_dist,
            stretch_target_ratio=args.stretch_ratio,
            time_feasible_max_km=args.time_feasible_max_km,
            randomize_base_position=args.vary_base,
            base_shift_radius_km=args.base_shift_km,
            seed=args.eval_seed + ep_idx,
        )
        ep_scenario_path = str(scenario_gen.generate(
            episode=ep_idx, config=ep_config,
        ))
        reload_scenario(game, ep_scenario_path)
    else:
        ep_scenario_path = args.scenario

    # --- Run the episode ---
    try:
        metrics = train_episode(
            trainer=trainer,
            game=game,
            env=env,
            scenario_path=ep_scenario_path,
            obs_config=obs_config,
            episode_num=ep_idx,
            max_ticks=args.max_ticks,
            record=False,
            fuel_damage_enabled=args.fuel_damage,
            replay_only=True,  # skips buffer.store + PPO update
        )
    except Exception as e:
        logger.warning(f"  ep{ep_idx:04d} crashed: {type(e).__name__}: {e}")
        return {
            "ep_idx": ep_idx,
            "error": f"{type(e).__name__}: {e}",
            "achieved_utility": 0.0,
            "oracle_total_utility": 0.0,
            "crashes": 0,
            "ep_reward": 0.0,
            "utility_ratio": 0.0,
            "targets_hit": 0,
            "targets_total": 0,
            "timeout": False,
        }

    # --- Reconstruct ep_reward from metrics ---
    # train_episode under replay_only=True does NOT fold the episode reward
    # into metrics["episode_reward"] (that path requires the buffer to be
    # non-empty). Recompute it from the canonical formula.
    ep_reward = compute_episode_reward(
        achieved_utility=metrics["achieved_utility"],
        max_total_utility=metrics["oracle_utility"],
        lost_aircraft_count=metrics["crashes"],
        max_target_utility=metrics["max_target_utility"],
        config=trainer.config.reward_config,
    )

    return {
        "ep_idx": ep_idx,
        "achieved_utility": float(metrics["achieved_utility"]),
        "oracle_total_utility": float(metrics["oracle_utility"]),
        "max_target_utility": float(metrics["max_target_utility"]),
        "crashes": int(metrics["crashes"]),
        "ep_reward": float(ep_reward),
        "utility_ratio": float(metrics["utility_ratio"]),
        "targets_hit": int(metrics["targets_hit_total"]),
        "targets_total": int(metrics["n_tasks"]),
        "timeout": bool(metrics.get("timeout", False)),
        "all_rtb": bool(metrics.get("all_rtb", False)),
        "ticks": int(metrics["ticks"]),
        "n_agents": int(metrics["n_agents"]),
    }


# =============================================================================
# Aggregation
# =============================================================================

def aggregate(episodes: List[Dict]) -> Dict:
    """Compute aggregate stats over a list of per-episode records.

    Skips episodes that errored (presence of an "error" key). With n=200
    the standard error on mean_reward is std/sqrt(n) ≈ std/14.1.
    """
    valid = [ep for ep in episodes if "error" not in ep]
    n = len(valid)
    if n == 0:
        return {"n_valid": 0}

    rewards = [ep["ep_reward"] for ep in valid]
    utilities = [ep["utility_ratio"] for ep in valid]
    crashes = [ep["crashes"] for ep in valid]
    timeouts = [int(ep["timeout"]) for ep in valid]

    def _std(xs: List[float]) -> float:
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    return {
        "n_valid": n,
        "n_errors": len(episodes) - n,
        # Reward distribution
        "mean_reward": statistics.fmean(rewards),
        "std_reward": _std(rewards),
        "median_reward": statistics.median(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        # Utility
        "mean_utility_ratio": statistics.fmean(utilities),
        "std_utility_ratio": _std(utilities),
        "median_utility_ratio": statistics.median(utilities),
        # Crashes
        "mean_crashes": statistics.fmean(crashes),
        "crash_rate": sum(1 for c in crashes if c >= 1) / n,
        # Categorical buckets
        "perfect_rate": sum(1 for r in rewards if r >= 0.0) / n,
        "disaster_rate": sum(1 for r in rewards if r <= -1.0) / n,
        # Timeouts
        "timeout_rate": statistics.fmean(timeouts),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    setup_logging(args.quiet)

    is_baseline = (args.checkpoint == "baseline")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir)
    recordings_dir = work_dir / "recordings"
    scenarios_dir = work_dir / "scenarios"
    models_dir = work_dir / "models"
    for d in (recordings_dir, scenarios_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Wipe stale scratch (recordings and generated scenarios only).
    for old in recordings_dir.glob("*"):
        try:
            old.unlink()
        except OSError:
            pass
    for old in scenarios_dir.glob("*.json"):
        try:
            old.unlink()
        except OSError:
            pass

    # --- Seed globally; per-episode scenario seeding is explicit below. ---
    import random
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    label = "baseline" if is_baseline else Path(args.checkpoint).stem
    logger.info(f"=== Eval start: label={label}  n_episodes={args.n_episodes}  "
                f"eval_seed={args.eval_seed}  fuel_damage={args.fuel_damage}  "
                f"vary_scenarios={args.vary_scenarios} ===")

    # --- BLADE env ---
    game, env, _ = setup_blade_env(
        args.scenario, args.max_ticks, recording_dir=str(recordings_dir),
    )

    # --- Scenario generator (mirrors main()) ---
    scenario_gen: Optional[ScenarioGenerator] = None
    if args.vary_scenarios:
        scenario_gen = ScenarioGenerator(
            base_scenario_path=args.scenario,
            output_dir=str(scenarios_dir),
            max_sim_ticks=args.max_ticks,
        )
        scenario_gen.recompute_time_feasible_cap(
            allowed_classes=args.allowed_aircraft,
        )

    # --- Trainer (loaded or empty) ---
    obs_config = ObservationConfig(top_k=3)
    trainer = make_trainer(work_dir)

    if is_baseline:
        patch_get_action_baseline(trainer)
        logger.info("Mode: baseline (NOOP — pure partial-plan executor)")
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return 2
        trainer.load_checkpoint(str(ckpt_path))
        # Eval mode for any BatchNorm/Dropout (none here, but harmless).
        trainer.network.eval()
        patch_get_action_deterministic(trainer)
        logger.info(f"Mode: checkpoint ({ckpt_path.name}) — deterministic argmax")

    # --- Episode loop ---
    t0 = time.time()
    episodes: List[Dict] = []
    for ep in range(args.n_episodes):
        rec = run_one_episode(
            trainer=trainer,
            game=game, env=env,
            obs_config=obs_config,
            scenario_gen=scenario_gen,
            ep_idx=ep,
            args=args,
        )
        episodes.append(rec)
        if (ep + 1) % 20 == 0 or ep == args.n_episodes - 1:
            elapsed = time.time() - t0
            valid_so_far = [e for e in episodes if "error" not in e]
            mean_r = (statistics.fmean(e["ep_reward"] for e in valid_so_far)
                      if valid_so_far else 0.0)
            logger.info(
                f"  [{label}] ep {ep + 1:4d}/{args.n_episodes}  "
                f"mean_r={mean_r:+.3f}  elapsed={elapsed:.1f}s"
            )

    # --- Aggregate + write ---
    agg = aggregate(episodes)
    out = {
        "label": label,
        "checkpoint": str(args.checkpoint),
        "n_episodes": args.n_episodes,
        "eval_seed": args.eval_seed,
        "fuel_damage": args.fuel_damage,
        "vary_scenarios": args.vary_scenarios,
        "wall_clock_seconds": time.time() - t0,
        "aggregate": agg,
        "episodes": episodes,
    }
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote {out_path}  (n_valid={agg.get('n_valid', 0)}, "
                f"mean_r={agg.get('mean_reward', 0):+.3f} ± "
                f"{agg.get('std_reward', 0):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
