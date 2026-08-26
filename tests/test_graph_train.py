"""
Unit tests for `graph_train` -- the outer PPO Trainer (Phase A, actor-only).

These are the PURE half of the module's proof obligations. The trainer's real
end-to-end behaviour (a short live training run, eval purity, zero-wake handling)
needs BLADE + bonmin and therefore lives in the module's own `_selftest()`, run under
`nlp_env` -- pytest stays solver-free and fast:

    conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --selftest

What these tests lock:

  T1 checkpoint round-trip : a saved checkpoint restores the encoder, the head AND the
                             optimizer bit-for-bit into a FRESH policy + updater. The
                             optimizer half is the one that silently rots: Adam's
                             moment estimates are training state, and a checkpoint that
                             dropped them would still "load" and still train -- just
                             from a different trajectory than the one it claims to
                             continue. The test therefore steps the optimizer first, so
                             its state is non-empty when saved.
  T2 seeding schedule      : g, the training seed and the eval seed are exactly the
                             documented derivations; the eval band is provably disjoint
                             from every training seed the run reaches; and a config
                             whose bands overlap is REFUSED by validate() rather than
                             quietly training on its own "held-out" set.
  T3 plotting              : plot_training() renders the THREE figures from synthetic
                             jsonl alone (no training, no torch) into <run_dir>/plots/,
                             tolerates a missing eval file, and degrades to a friendly
                             no-op when matplotlib is absent -- matplotlib is optional
                             and must never fail the suite.
  T4 the baseline cell     : the DEFAULTS are the approved Phase-A cell --
                             num_red_airbases=(6, 6), partial_ratio=0.5 -> known 3 /
                             hidden 3 -- and a range config previews both of its ends.
  T5 the split mirror      : `derived_split` is proven equal to the LOCKED authority,
                             `split_tasks`, over an n x ratio grid using real
                             `split_tasks` calls. This is the load-bearing test of this
                             group: `derived_split` is a SECOND copy of the split
                             arithmetic, and a second copy that drifts from the first
                             would make the startup header lie about the config a run
                             is really using. The truncation trap is pinned explicitly
                             (n=6: 1.0/3.0 -> known 2, but 0.333 -> known 1) so a
                             future "cleanup" into rounding fails loudly here.
  T6 CLI parsing           : --num-red-airbases takes "6" or "6,8" and REJECTS "8,6",
                             "0", "-1", "abc" and "" through argparse (a usage error,
                             not a traceback from deep inside the generator); the three
                             scenario flags' defaults are read off TrainConfig rather
                             than restated as literals (drift guard).
  T7 hazard warnings       : known < 3 (bonmin symmetry stall) and hidden == 0 (no
                             pop-up possible) are WARNED about on stdout, and
                             validate() still returns -- a researcher may probe those
                             cells deliberately, so they must not be errors. The
                             approved default cell stays silent.
  T8 run_config.json       : every run records its own resolved config + derived split.
  T9 the construction cell : the explicit surface. The DEFAULTS are the reference cell
                             (3 agents, 3 known, 3 hidden, 200/100 km -> a 6-target
                             executed world); `build_variation_config` is the single
                             site that turns a TrainConfig into a generator request and
                             it asks for a KNOWN-ONLY world with Layer 1 off and the
                             geometry strict; `num_agents > n_known` and
                             `include_sams=True` are REFUSED; the new CLI flags read
                             their defaults off the dataclass and reject bad values at
                             parse time; `run_config.json` separates what the GENERATOR
                             writes from what the episode EXECUTES; and
                             `RolloutConfig` is checked field-for-field against
                             `TrainConfig` so the two harnesses cannot drift apart
                             again (they were `(3,3)`/2-3 vs `(6,6)`/0.5 before B1),
                             and it now carries the same validation -- enforced as the
                             FIRST statement of `run_rollout`, proven by the run
                             directory still not existing after the raise.

WHY T3 GOES THROUGH A SUBPROCESS
--------------------------------
This test module imports torch (T1 needs a real policy + optimizer), and on this
Windows/Anaconda stack importing matplotlib into a torch process aborts the
interpreter outright -- duplicate OpenMP runtimes, "OMP: Error #15", `Fatal Python
error: Aborted`. Verified in both environments and both import orders; it is a
property of the machine, not of this code, and an abort cannot be caught. So the
plotting tests drive `plot_training_subprocess`, which is exactly what a real training
run does at the end (see the graph_train module docstring). The pytest interpreter
itself never touches matplotlib.

No BLADE episode, no solver, no env: T1 builds a policy and drives the optimizer with
synthetic grads; T2 is arithmetic; T3 writes its own jsonl; T5 calls the real
`split_tasks` on synthetic tasks (it is a pure geometric sampler -- no solver, no
engine); T8 calls the config writer directly. The end-to-end proof that a REAL run
emits `run_config.json` lives in the module's `_selftest` TEST 1.

Run: python -m pytest tests/test_graph_train.py -v
     python tests/test_graph_train.py
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import inspect
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import torch

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so match_aou.* imports resolve

from match_aou.models.location import Location  # noqa: E402
from match_aou.models.step import Step, StepKind  # noqa: E402
from match_aou.models.task import Task  # noqa: E402
from match_aou.rl.training.graph_episode_setup import (  # noqa: E402
    DETECTION_KM,
    _resolve_construction_mode,
    split_tasks,
)
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CONDITION_CLEAN,
    CONDITION_DAMAGED,
    FD_ELIGIBILITY_CERTIFIED_V1,
    FuelDamageController,
    FuelDamageMode,
    FuelDamageParameters,
    certify_fd_candidate,
    plan_fuel_damage,
    resolve_condition,
    resolve_severity,
)
from match_aou.rl.training.graph_ppo import PPOConfig, PPOUpdater  # noqa: E402
from match_aou.rl.training.graph_tick_loop import build_policy  # noqa: E402
from match_aou.rl.training import graph_rollout, graph_train  # noqa: E402
from match_aou.rl.training.graph_rollout import (  # noqa: E402
    RolloutConfig,
    run_rollout,
)
from match_aou.rl.training.graph_train import (  # noqa: E402
    EpisodeAttemptError,
    EpisodeRosterError,
    TrainConfig,
    _EpisodeOutcome,
    _EVAL_STAGE_POST_UPDATE,
    _EVAL_STAGE_PRE_UPDATE,
    _EXACT_CARDINALITY_POLICY,
    _PIPELINE_STAGES,
    _PLOT_DIAGNOSTICS,
    _PLOT_FILENAMES,
    _PLOT_MEASUREMENT_HEALTH,
    _PLOT_PERFORMANCE,
    _PLOTS_DIRNAME,
    _build_arg_parser,
    _explicit_cli_dests,
    _comparable_records,
    _git_provenance,
    _parse_airbase_range,
    _probe_command,
    _xy,
    build_run_summary,
    build_variation_config,
    collect_provenance,
    derived_split,
    eval_member_tag,
    eval_seed,
    global_episode_index,
    config_source_record,
    load_config_file,
    plot_training,
    plot_training_subprocess,
    resolve_train_config,
    save_checkpoint,
    seed_bands,
    train_seed,
    write_run_config,
    write_run_summary,
)
from match_aou.utils.blade_utils.scenario_generator import VariationConfig  # noqa: E402


# =============================================================================
# T1 -- checkpoint round-trip (encoder + head + optimizer)
# =============================================================================

def _assert_state_dicts_equal(a: dict, b: dict, what: str) -> None:
    """Every tensor leaf equal, every non-tensor leaf equal, same key set."""
    assert set(a.keys()) == set(b.keys()), f"{what}: key sets differ"
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, torch.Tensor):
            assert isinstance(vb, torch.Tensor), f"{what}[{key}]: type differs"
            assert torch.equal(va, vb), f"{what}[{key}]: tensors differ"
        elif isinstance(va, dict):
            _assert_state_dicts_equal(va, vb, f"{what}[{key}]")
        else:
            assert va == vb, f"{what}[{key}]: {va!r} != {vb!r}"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """Save -> fresh policy + updater -> load -> encoder, head and optimizer all match."""
    torch.manual_seed(11)
    policy = build_policy()
    updater = PPOUpdater(policy, PPOConfig(lr=1e-3))

    # Take a real optimizer step so Adam's state (step / exp_avg / exp_avg_sq) is
    # populated -- an untouched optimizer has an EMPTY state dict, which would make the
    # optimizer half of this test vacuous. Synthetic grads keep it BLADE-free; the
    # optimizer cannot tell where a .grad came from.
    torch.manual_seed(12)
    for p in updater.parameters:
        p.grad = torch.randn_like(p)
    updater.optimizer.step()
    assert updater.optimizer.state_dict()["state"], "optimizer state is still empty"

    path = save_checkpoint(policy, updater, iteration=7, ckpt_dir=tmp_path / "ckpts")
    assert path.exists() and path.name == "ckpt_iter0007.pt"

    # A genuinely FRESH policy: different seed -> different initial weights, so an
    # equality that passes cannot be an artifact of identical initialization.
    torch.manual_seed(99)
    fresh_policy = build_policy()
    fresh_updater = PPOUpdater(fresh_policy, PPOConfig(lr=1e-3))
    assert not torch.equal(
        policy.head.state_dict()[list(policy.head.state_dict())[0]],
        fresh_policy.head.state_dict()[list(fresh_policy.head.state_dict())[0]],
    ), "fresh policy accidentally initialized identically"

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    fresh_policy.encoder.load_state_dict(ckpt["encoder"])
    fresh_policy.head.load_state_dict(ckpt["head"])
    fresh_updater.optimizer.load_state_dict(ckpt["optimizer"])

    _assert_state_dicts_equal(
        policy.encoder.state_dict(), fresh_policy.encoder.state_dict(), "encoder"
    )
    _assert_state_dicts_equal(
        policy.head.state_dict(), fresh_policy.head.state_dict(), "head"
    )
    _assert_state_dicts_equal(
        updater.optimizer.state_dict(), fresh_updater.optimizer.state_dict(),
        "optimizer",
    )

    assert ckpt["iteration"] == 7
    assert ckpt["ppo_config"]["lr"] == 1e-3


# =============================================================================
# T2 -- the seeding schedule
# =============================================================================

def test_global_episode_index_and_train_seed() -> None:
    """g = iteration*episodes_per_iteration + j; training seed = base_seed + g."""
    cfg = TrainConfig(n_iterations=4, episodes_per_iteration=8, base_seed=0)
    assert global_episode_index(cfg, 0, 0) == 0
    assert global_episode_index(cfg, 0, 7) == 7
    assert global_episode_index(cfg, 1, 0) == 8
    assert global_episode_index(cfg, 3, 5) == 29
    assert train_seed(cfg, 3, 5) == 29

    shifted = TrainConfig(n_iterations=4, episodes_per_iteration=8, base_seed=1234)
    assert train_seed(shifted, 3, 5) == 1234 + 29

    # Every (iteration, j) pair maps to a DISTINCT seed -- no episode is silently
    # trained on twice within a run.
    seeds = [train_seed(cfg, it, j)
             for it in range(cfg.n_iterations)
             for j in range(cfg.episodes_per_iteration)]
    assert len(set(seeds)) == len(seeds) == cfg.total_episodes


def test_eval_seed_band_is_fixed_and_disjoint() -> None:
    """Eval seeds are the SAME every round and never collide with a training seed."""
    cfg = TrainConfig(
        n_iterations=100, episodes_per_iteration=8,
        base_seed=0, eval_base_seed=1_000_000, eval_episodes=8,
    )
    cfg.validate()

    # Fixed: eval episode e is seed eval_base_seed + e regardless of the round.
    assert [eval_seed(cfg, e) for e in range(cfg.eval_episodes)] == \
           [1_000_000 + e for e in range(8)]

    train_seeds = {train_seed(cfg, it, j)
                   for it in range(cfg.n_iterations)
                   for j in range(cfg.episodes_per_iteration)}
    eval_seeds = {eval_seed(cfg, e) for e in range(cfg.eval_episodes)}
    assert train_seeds.isdisjoint(eval_seeds)
    assert max(train_seeds) < min(eval_seeds)


def test_validate_rejects_overlapping_seed_bands() -> None:
    """A run long enough to reach into the eval band is REFUSED, not silently run."""
    cfg = TrainConfig(
        n_iterations=10, episodes_per_iteration=10,
        base_seed=0, eval_base_seed=50, eval_episodes=8,
    )
    try:
        cfg.validate()
    except ValueError as exc:
        assert "OVERLAP" in str(exc), str(exc)
    else:
        raise AssertionError("validate() accepted overlapping seed bands")

    # Exactly adjacent bands (train [0, 100), eval [100, 108)) are legal.
    ok = TrainConfig(
        n_iterations=10, episodes_per_iteration=10,
        base_seed=0, eval_base_seed=100, eval_episodes=8,
    )
    ok.validate()

    # Disabled eval means the bands are irrelevant -- no spurious refusal.
    disabled = TrainConfig(
        n_iterations=10, episodes_per_iteration=10,
        base_seed=0, eval_base_seed=50, eval_episodes=8, eval_every=0,
    )
    assert not disabled.eval_enabled
    disabled.validate()


def test_validate_rejects_degenerate_shapes() -> None:
    """n_iterations / episodes_per_iteration / partial_ratio are checked up front."""
    for kwargs in (
        {"n_iterations": 0},
        {"n_iterations": 1, "episodes_per_iteration": 0},
        {"n_iterations": 1, "partial_ratio": 0.0},
        {"n_iterations": 1, "partial_ratio": 1.5},
    ):
        try:
            TrainConfig(**kwargs).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate() accepted {kwargs}")


# =============================================================================
# T3 -- plotting from jsonl alone
# =============================================================================

def _write_synthetic_run(
    run_dir: Path,
    *,
    with_eval: bool = True,
    legacy: bool = False,
    all_failed_iteration: bool = False,
) -> None:
    """Write train/eval jsonl in EXACTLY the shape `train()` emits (scalar-only).

    ``legacy`` writes PRE-B4 records instead -- no ``updates_completed*``, no attempt
    accounting, the reward under its old ``baseline`` name. The plotter must still
    render those (a run started before this change is still a run), which is the only
    thing that flag is for.

    ``all_failed_iteration`` inserts one iteration in which every scheduled attempt
    FAILED: reward ``null``, 0 successes, 4 failures. That is the record shape the
    "never plot a total data loss as R = 0" rule exists for.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train_records.jsonl", "w", encoding="utf-8") as fh:
        for it in range(6):
            failed_all = all_failed_iteration and it == 3
            record = {
                "iteration": it,
                "baseline": None if failed_all else -0.9 + 0.1 * it,
                "entropy": 1.7 - 0.05 * it,
                "policy_loss": 0.01 * it,
                "total_loss": 0.01 * it - 0.017,
                "n_transitions": 0 if failed_all else 10 + it,
                "n_epochs_run": 0 if failed_all else 2,
                "episodes_with_wakes": 0 if failed_all else 3,
                "meta_action_counts": {
                    "PLAN_COMPLIANCE": 6, "OPPORTUNISTIC_ENGAGEMENT": 3,
                    "SELF_PRESERVATION_ABORT": 1,
                },
                "meta_action_fractions": {
                    "PLAN_COMPLIANCE": 0.6, "OPPORTUNISTIC_ENGAGEMENT": 0.3,
                    "SELF_PRESERVATION_ABORT": 0.1,
                },
            }
            if not legacy:
                record.update({
                    "updates_completed_before": it,
                    "updates_completed": it if failed_all else it + 1,
                    "n_attempted": 4,
                    "n_successful": 0 if failed_all else 4,
                    "n_failed": 4 if failed_all else 0,
                    "success_fraction": 0.0 if failed_all else 1.0,
                    "wake_fraction_of_successful": None if failed_all else 0.75,
                    "train_reward_mean": None if failed_all else -0.9 + 0.1 * it,
                    "aggregates_over": "successful_episodes",
                })
            fh.write(json.dumps(record) + "\n")
    if with_eval:
        with open(run_dir / "eval_records.jsonl", "w", encoding="utf-8") as fh:
            # The pre-update round measures the INITIAL policy: 0 updates completed,
            # no training iteration yet.
            rounds = [(None, 0, _EVAL_STAGE_PRE_UPDATE),
                      (1, 2, _EVAL_STAGE_POST_UPDATE),
                      (3, 4, _EVAL_STAGE_POST_UPDATE),
                      (5, 6, _EVAL_STAGE_POST_UPDATE)]
            for it, updates, stage in rounds:
                record = {
                    "iteration": it,
                    "n_episodes": 4,
                    "eval_reward_mean": -0.8 + 0.05 * updates,
                }
                if not legacy:
                    record.update({
                        "evaluation_stage": stage,
                        "updates_completed": updates,
                        "n_attempted": 4,
                        "n_successful": 3,
                        "n_failed": 1,
                        "success_fraction": 0.75,
                        "n_ok": 3,
                        "aggregates_over": "successful_episodes",
                        # FD-BASELINE-v1 matched pairs, in the shape `evaluate` emits.
                        # The two conditions differ so a plot that pooled them, or that
                        # drew one series twice, would be visible rather than plausible.
                        "eval_reward_mean_clean": -0.7 + 0.05 * updates,
                        "eval_reward_mean_damaged": -0.9 + 0.05 * updates,
                        "eval_paired_reward_delta": -0.2,
                        "n_pairs_attempted": 2,
                        "n_pairs_successful": 1,
                        "pair_success_fraction": 0.5,
                        # ASYMMETRIC on purpose: the damaged condition completes fewer
                        # held-out seeds than the clean one, so the two condition means
                        # are NOT over the same seeds. That is exactly the case the
                        # per-condition denominators exist to expose.
                        "eval_n_clean_attempted": 2,
                        "eval_n_clean_successful": 2,
                        "eval_n_clean_failed": 0,
                        "eval_n_damaged_attempted": 2,
                        "eval_n_damaged_successful": 1,
                        "eval_n_damaged_failed": 1,
                    })
                elif it is None:      # a legacy run has no pre-update round at all
                    continue
                fh.write(json.dumps(record) + "\n")


def _matplotlib_available() -> bool:
    """Is matplotlib importable? Probed in a CHILD -- see the module docstring."""
    proc = subprocess.run(
        [sys.executable, "-c", "import matplotlib"],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


def _skip_without_matplotlib() -> bool:
    """True if the caller should bail out (matplotlib is an optional dependency)."""
    if _matplotlib_available():
        return False
    if pytest is not None:
        pytest.skip("matplotlib not installed (optional dependency)")
    return True


def test_plot_training_writes_three_figures_into_the_plots_dir(tmp_path: Path) -> None:
    """The three figures are produced from the jsonl alone, under `<run_dir>/plots/`.

    Also the run-output-organization claim: figures are DERIVED artifacts and live in
    their own subdirectory, so they never sit among the scientific records.
    """
    if _skip_without_matplotlib():
        return
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    out = plot_training_subprocess(run_dir)

    assert [path.name for path in out] == list(_PLOT_FILENAMES)
    for path in out:
        assert path.exists(), path
        assert path.parent == run_dir / _PLOTS_DIRNAME, path
        assert path.stat().st_size > 1000, "%s is suspiciously small" % path.name
    # Nothing was written next to the records.
    assert not (run_dir / "training_plot.png").exists()


def test_plot_training_without_eval_records(tmp_path: Path) -> None:
    """An in-progress run with no eval round yet still plots (eval series are empty)."""
    if _skip_without_matplotlib():
        return
    run_dir = tmp_path / "run_no_eval"
    _write_synthetic_run(run_dir, with_eval=False)
    out = plot_training_subprocess(run_dir)
    assert [path.name for path in out] == list(_PLOT_FILENAMES)
    assert all(path.exists() for path in out)


def test_plot_training_missing_records_is_a_clean_noop(tmp_path: Path) -> None:
    """Pointing the plotter at a directory with no records returns [], never raises.

    Safe to call IN-PROCESS despite the torch/matplotlib conflict: `plot_training`
    reads the records before it touches matplotlib, so the empty path never imports it.
    """
    assert plot_training(tmp_path / "does_not_exist") == []


# =============================================================================
# T4 -- the defaults ARE the approved Phase-A baseline cell
# =============================================================================

def test_defaults_are_the_phase_a_baseline_cell() -> None:
    """(6, 6) targets and partial_ratio 0.5 -> known 3 / hidden 3.

    Pins the measured cell as the DEFAULT. The retired (3, 3) default gave 4 agents 3
    targets, forcing 2:1 stacking and a scenario-constant R = -1/3 (zero advantage,
    nothing to learn); 6 targets > the 4-agent fleet removes the forcing, and known=3
    keeps bonmin out of its symmetry stall.
    """
    cfg = TrainConfig(n_iterations=1)
    assert cfg.num_red_airbases == (6, 6), cfg.num_red_airbases
    assert cfg.partial_ratio == 0.5, cfg.partial_ratio

    assert derived_split(6, cfg.partial_ratio) == (3, 3)
    assert cfg.split_preview == [{"n": 6, "known": 3, "hidden": 3}], cfg.split_preview


def test_split_preview_covers_both_ends_of_a_range() -> None:
    """A range config previews the split at BOTH ends -- they bracket every episode."""
    cfg = TrainConfig(n_iterations=1, num_red_airbases=(4, 8), partial_ratio=0.5)
    assert cfg.split_preview == [
        {"n": 4, "known": 2, "hidden": 2},
        {"n": 8, "known": 4, "hidden": 4},
    ], cfg.split_preview


# =============================================================================
# T5 -- the split mirror does not drift from split_tasks (the locked authority)
# =============================================================================

def _synthetic_tasks(n: int) -> list:
    """``n`` tasks placed MUTUALLY within DETECTION_KM, so any split is chain-valid.

    `split_tasks` reads exactly one thing off a task -- ``steps[0].location`` -- and
    rejection-samples until every hidden target has a known neighbour. Packing all
    targets inside one detection radius makes every draw valid on the first attempt, so
    the sampler never falls back and the returned ``known`` count is the pure arithmetic
    result. The assertion is on the COUNT, not on which tasks were picked.
    """
    # ~4.7 km apart at this latitude; n <= 8 spans ~33 km, well inside DETECTION_KM=50.
    tasks = []
    for i in range(n):
        loc = Location(latitude=32.0, longitude=35.0 + 0.05 * i)
        step = Step(
            location=loc,
            target_id="tgt_%d" % i,
            capabilities=[],
            probability=1.0,
            effort=1,
            step_kind=StepKind.ATTACK,
        )
        tasks.append(Task(steps=[step], utility=80.0))
    return tasks


def test_derived_split_matches_real_split_tasks() -> None:
    """`derived_split` == `split_tasks` over an n x partial_ratio grid.

    THE anti-drift test. `derived_split` is a second copy of the locked arithmetic,
    kept only so the trainer can echo the split before generating an episode; if the
    copy diverged, the startup header would confidently report a split the run does not
    actually use. Includes the exact-fraction and decimal ratios side by side.

    The grid covers n < 2 as well, because `derived_split` has a SECOND branch there
    (mirroring `split_tasks`' "nothing to hide" degenerate path) and that branch is
    reachable from a real config: `--num-red-airbases 1` is a legal CLI value. Its
    expected values are fully determined, so they are asserted as literals rather than
    re-derived from the function under test:
      n=0 -> (0, 0)   n=1 -> (1, 0)   n=2 -> (1, 1) for ALL four ratios
    (n=2 is the interesting one: 2*(1/3) and 2*0.333 both truncate to 0, and the
    `max(1, ...)` floor lifts them back to known=1.)
    """
    degenerate_expected = {0: (0, 0), 1: (1, 0), 2: (1, 1)}

    ratios = [2.0 / 3.0, 0.5, 1.0 / 3.0, 0.333]
    for n in (0, 1, 2, 4, 5, 6, 7, 8):
        tasks = _synthetic_tasks(n)
        for ratio in ratios:
            random.seed(1234)  # split_tasks draws from global random
            partial, full, meta = split_tasks(
                tasks, ratio, detection_km=DETECTION_KM,
            )
            known, hidden = derived_split(n, ratio)
            assert known == meta["known"], (
                "derived_split DRIFTED from split_tasks at n=%d ratio=%r: "
                "mirror known=%d, split_tasks known=%d"
                % (n, ratio, known, meta["known"])
            )
            assert hidden == meta["hidden"], (n, ratio, hidden, meta["hidden"])
            # Cross-check against what split_tasks actually returned, not just its meta.
            assert len(partial) == known, (n, ratio, len(partial), known)
            assert len(full) == n, (n, ratio, len(full))
            if n in degenerate_expected:
                assert (known, hidden) == degenerate_expected[n], (
                    "degenerate branch n=%d ratio=%r: expected %r, got %r"
                    % (n, ratio, degenerate_expected[n], (known, hidden))
                )


def test_truncation_trap_is_pinned() -> None:
    """n=6: 1.0/3.0 -> known 2, but the decimal 0.333 -> known 1.

    `int()` TRUNCATES. These are two different, non-interchangeable configs, and the
    decimal one is the hazardous (known < 3) cell. Written as an explicit assertion so
    that "cleaning up" the arithmetic into rounding -- in either the mirror or the
    locked `split_tasks` -- breaks this test instead of silently changing every run's
    scenario. Never auto-corrected: the config a user typed is the config they get,
    which is why the split is echoed at startup instead.
    """
    assert derived_split(6, 1.0 / 3.0) == (2, 4)
    assert derived_split(6, 0.333) == (1, 5)

    # And the authority agrees with the mirror on both.
    tasks = _synthetic_tasks(6)
    random.seed(7)
    assert split_tasks(tasks, 1.0 / 3.0, detection_km=DETECTION_KM)[2]["known"] == 2
    random.seed(7)
    assert split_tasks(tasks, 0.333, detection_km=DETECTION_KM)[2]["known"] == 1


# =============================================================================
# T6 -- CLI parsing of the scenario knobs
# =============================================================================

def test_parse_airbase_range_accepts_int_and_range() -> None:
    """"6" -> (6, 6); "6,8" -> (6, 8); surrounding whitespace tolerated."""
    assert _parse_airbase_range("6") == (6, 6)
    assert _parse_airbase_range("6,8") == (6, 8)
    assert _parse_airbase_range(" 6 , 8 ") == (6, 8)
    assert _parse_airbase_range("1") == (1, 1)


def test_parse_airbase_range_rejects_bad_values() -> None:
    """A bad value is an argparse ERROR, never a traceback from inside the generator."""
    for bad in ("8,6", "0", "-1", "abc", "", "6,8,10", "6.5"):
        try:
            _parse_airbase_range(bad)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError("_parse_airbase_range accepted %r" % bad)


def test_cli_parses_scenario_knobs() -> None:
    """The three flags reach the namespace, and bad input exits via argparse."""
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--iterations", "2", "--num-red-airbases", "6,8",
        "--partial-ratio", "0.5", "--stretch-target-ratio", "0.25",
    ])
    assert args.num_red_airbases == (6, 8)
    assert args.partial_ratio == 0.5
    assert args.stretch_target_ratio == 0.25

    # A single integer resolves to a fixed (n, n) range through the same type function.
    assert parser.parse_args(["--iterations", "1",
                             "--num-red-airbases", "6"]).num_red_airbases == (6, 6)

    # argparse exits (SystemExit) with a usage error rather than raising through.
    for bad in ("8,6", "0", "-1", "abc", ""):
        try:
            parser.parse_args(["--iterations", "1", "--num-red-airbases", bad])
        except SystemExit:
            pass
        else:
            raise AssertionError("the CLI accepted --num-red-airbases %r" % bad)


def test_cli_defaults_equal_the_dataclass_defaults() -> None:
    """DRIFT GUARD: the CLI reads its scenario defaults off TrainConfig, never literals."""
    d = TrainConfig(n_iterations=1)
    args = _build_arg_parser().parse_args(["--iterations", "1"])
    assert args.num_red_airbases == d.num_red_airbases
    assert args.partial_ratio == d.partial_ratio
    assert args.stretch_target_ratio == d.stretch_target_ratio


# =============================================================================
# T7 -- hazard warnings: printed, never raised
# =============================================================================

def _validate_capturing_stdout(cfg) -> str:
    """Run ``cfg.validate()`` and return what it printed.

    Duck-typed: takes a ``TrainConfig`` or a ``RolloutConfig``, both of which expose a
    ``validate()`` that raises on a bad cell and prints hazards.

    Uses ``redirect_stdout`` rather than pytest's ``capsys`` so these tests work in BOTH
    modes -- pytest and this file's own ``__main__`` runner (pytest is absent in
    nlp_env). Any exception propagates: "validate() must not raise here" is part of what
    each caller asserts.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg.validate()
    return buf.getvalue()


def test_hidden_zero_warns_and_validate_still_passes() -> None:
    """partial_ratio=1.0 -> nothing hidden -> WARN, but validate() returns normally."""
    cfg = TrainConfig(n_iterations=1, partial_ratio=1.0)
    assert cfg.split_preview == [{"n": 6, "known": 6, "hidden": 0}]
    out = _validate_capturing_stdout(cfg)       # must NOT raise
    assert "[WARN]" in out, out
    assert "NO target is hidden" in out, out
    assert "OPPORTUNISTIC_ENGAGEMENT" in out, out


def test_known_below_three_warns_about_the_bonmin_stall() -> None:
    """(4, 4) at ratio 0.5 -> known 2 -> the symmetry-stall warning, no exception."""
    cfg = TrainConfig(n_iterations=1, num_red_airbases=(4, 4), partial_ratio=0.5)
    out = _validate_capturing_stdout(cfg)
    assert "[WARN]" in out, out
    assert "SYMMETRY-STALL" in out, out
    assert "known/hidden = 2/2" in out, out


def test_default_config_emits_no_hazard_warning() -> None:
    """The approved cell is quiet -- a warning must mean something is actually off."""
    assert "[WARN]" not in _validate_capturing_stdout(TrainConfig(n_iterations=1))


def test_warnings_use_the_low_end_of_the_range() -> None:
    """A range is judged by its WORST case (fewest targets), not its best."""
    cfg = TrainConfig(n_iterations=1, num_red_airbases=(4, 12), partial_ratio=0.5)
    out = _validate_capturing_stdout(cfg)
    assert "SYMMETRY-STALL" in out, out       # n=4 -> known 2, even though n=12 is fine
    assert "n=4" in out, out


# =============================================================================
# T8 -- run_config.json
# =============================================================================

def test_write_run_config_records_the_scenario_knobs(tmp_path: Path) -> None:
    """The file exists, parses, and its scenario knobs + derived split match the config.

    Called directly rather than through `train()` so the suite stays solver-free; that a
    real run writes it is asserted in the module's own `_selftest` TEST 1.
    """
    cfg = TrainConfig(
        n_iterations=2,
        episodes_per_iteration=3,
        output_dir=tmp_path / "run",
        num_red_airbases=(6, 8),
        partial_ratio=0.5,
        stretch_target_ratio=0.25,
        ppo=PPOConfig(lr=3e-4, n_epochs=2),
    )
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    path = write_run_config(run_dir, cfg)
    assert path.exists() and path.name == "run_config.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    tc = payload["train_config"]
    # Tuples round-trip through JSON as lists.
    assert tc["num_red_airbases"] == [6, 8], tc["num_red_airbases"]
    assert tc["partial_ratio"] == 0.5
    assert tc["stretch_target_ratio"] == 0.25
    assert tc["include_sams"] is False
    assert tc["n_iterations"] == 2 and tc["episodes_per_iteration"] == 3
    # The nested PPOConfig is included -- a run's optimizer knobs are part of its config.
    assert tc["ppo"]["lr"] == 3e-4 and tc["ppo"]["n_epochs"] == 2

    assert payload["derived_split"] == [
        {"n": 6, "known": 3, "hidden": 3},
        {"n": 8, "known": 4, "hidden": 4},
    ], payload["derived_split"]
    assert payload["derived_split"] == cfg.split_preview
    assert payload["base_scenario"] == "strike_training_4v5.json"


# =============================================================================
# T9 -- the B1 offline scenario-construction cell (explicit counts + geometry)
# =============================================================================

def test_construction_defaults_are_the_reference_cell() -> None:
    """3 agents, 3 known, 3 hidden, 200 km floor, 100 km separation -> a 6-target world.

    `n_targets_emitted` is asserted separately from `n_known` on purpose: those two
    numbers are the whole GENERATED-vs-EXECUTED distinction. The generator writes three
    targets; `setup_episode` patches three more in from the solved routes, so an episode
    runs on six -- `U_oracle = 6 * 80 = 480` for the reference cell.
    """
    cfg = TrainConfig(n_iterations=1)
    assert cfg.num_agents == 3, cfg.num_agents
    assert cfg.n_known == 3, cfg.n_known
    assert cfg.n_hidden == 3, cfg.n_hidden
    assert cfg.min_target_distance_km == 200.0, cfg.min_target_distance_km
    assert cfg.min_known_separation_km == 100.0, cfg.min_known_separation_km

    assert cfg.n_targets_emitted == 6, cfg.n_targets_emitted
    assert cfg.n_targets_emitted == cfg.n_known + cfg.n_hidden
    # The generator itself still writes ONLY the known half.
    assert build_variation_config(cfg, seed=0).num_red_airbases == cfg.n_known
    # 200 km is comfortably outside the sensing bubble an ego launches inside of.
    assert cfg.min_target_distance_km > DETECTION_KM


def test_build_variation_config_requests_a_known_only_world() -> None:
    """The ONE generator request: n_known targets, Layer 1 OFF, geometry STRICT.

    Compared as a whole dataclass rather than field by field, so a field ADDED to the
    request later cannot slip through unasserted.
    """
    cfg = TrainConfig(n_iterations=1)
    var = build_variation_config(cfg, seed=7)

    assert var == VariationConfig(
        include_sams=False,
        num_aircraft=3,
        num_red_airbases=3,
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        min_target_distance_km=200.0,
        min_target_separation_km=100.0,
        ensure_discovery_chain=False,
        strict_geometry=True,
        detection_km=DETECTION_KM,
        seed=7,
    ), var

    # The legacy split surface reaches the generator NOWHERE: the request carries
    # n_known, not num_red_airbases, and no ratio at all.
    legacy = TrainConfig(n_iterations=1, num_red_airbases=(9, 9), partial_ratio=1.0 / 3.0)
    assert build_variation_config(legacy, seed=7).num_red_airbases == legacy.n_known

    # Only n_known drives emission; n_hidden is planned and never generated.
    bigger = TrainConfig(n_iterations=1, num_agents=2, n_known=5, n_hidden=4)
    assert build_variation_config(bigger, seed=0).num_red_airbases == 5
    assert build_variation_config(bigger, seed=0).num_aircraft == 2


def test_both_harnesses_call_setup_in_construction_mode() -> None:
    """Both callers hand setup the `(n_hidden, placement_rng)` pair and no partial_ratio.

    Read off the SOURCE rather than executed, because executing either caller needs
    BLADE + bonmin. The claim being locked is the one that replaced the pre-B3
    `partial_ratio=1.0` compatibility call: the legacy split surface must not reach
    `setup_episode` from either harness, and the rng must be an explicit per-episode
    `random.Random(seed)` rather than module-global randomness.
    """
    for module in (graph_train, graph_rollout):
        source = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
        # GENERALIZED-V1 Task 4 moved the hidden count from the config to the RESOLVED
        # per-episode cell (`episode_cardinality` / `sample_generalized_cardinality`),
        # which under `fixed_cell_v1` is `cfg.n_hidden` verbatim. The claim being locked
        # is unchanged -- setup is handed a hidden COUNT and an explicit per-episode rng
        # -- so the literal it is read off moves with it.
        assert "n_hidden=int(cell.hidden_requested)" in source, module.__name__
        assert "random.Random(seed)" in source, module.__name__
        # The pre-B3 constant is gone from both harnesses.
        assert "_ALL_KNOWN_PARTIAL_RATIO" not in source, module.__name__

    # ... and the resolved cell IS the configured one on the historical path, so the
    # substitution above changes what is passed for exactly no fixed-cell run.
    from match_aou.rl.training.graph_generalized import fixed_cell_cardinality
    cfg = TrainConfig(n_iterations=1, num_agents=2, n_known=5, n_hidden=4)
    resolved = graph_train.episode_cardinality(cfg, seed=11)
    assert (resolved.agent_count, resolved.known_count, resolved.hidden_requested) == (
        cfg.num_agents, cfg.n_known, cfg.n_hidden
    )
    assert resolved == fixed_cell_cardinality(
        agent_count=cfg.num_agents, known_count=cfg.n_known,
        hidden_requested=cfg.n_hidden,
    )

    # `setup_episode` refuses a half-supplied pair, so neither caller can drift into
    # passing only one half without failing loudly.
    for kwargs in ({"n_hidden": 3}, {"placement_rng": random.Random(0)}):
        try:
            _resolve_construction_mode(
                kwargs.get("n_hidden"), kwargs.get("placement_rng")
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"a half-supplied construction pair was accepted: {kwargs}")

    # The legacy split arithmetic is untouched and still hides nothing at ratio 1.0.
    for n in (1, 3, 5, 8):
        assert derived_split(n, 1.0) == (n, 0)


def test_validate_rejects_more_agents_than_targets() -> None:
    """num_agents > n_known RAISES -- it is the forced-stacking cell, not a hazard."""
    try:
        TrainConfig(n_iterations=1, num_agents=4, n_known=3).validate()
    except ValueError as exc:
        assert "num_agents" in str(exc) and "n_known" in str(exc), str(exc)
    else:
        raise AssertionError("validate() accepted num_agents > n_known")

    # Equality is the reference cell and must stay legal.
    TrainConfig(n_iterations=1, num_agents=3, n_known=3).validate()
    TrainConfig(n_iterations=1, num_agents=2, n_known=3).validate()


def test_validate_rejects_degenerate_construction_values() -> None:
    """Non-positive counts / distances fail up front, not inside the generator."""
    for kwargs in (
        {"num_agents": 0},
        {"n_known": 0},
        {"n_hidden": -1},
        {"min_target_distance_km": 0.0},
        {"min_target_distance_km": -1.0},
        {"min_known_separation_km": -1.0},
    ):
        try:
            TrainConfig(n_iterations=1, **kwargs).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate() accepted {kwargs}")

    # 0 separation is legal: it means "no separation constraint", not a typo.
    TrainConfig(n_iterations=1, min_known_separation_km=0.0).validate()


def test_both_configs_reject_sams_on_the_construction_path() -> None:
    """include_sams=True RAISES in BOTH harnesses, before any generation or setup.

    Hidden targets are patched in as enemy AIRBASES and `setup_episode` refuses a world
    whose enemy units are not all airbases. Catching that in `validate()` turns a
    45-second bonmin solve followed by a RuntimeError into an instant, explained refusal
    -- and it must be refused identically in both harnesses, since they drive the same
    generator and the same seam.
    """
    for cfg in (TrainConfig(n_iterations=1, include_sams=True),
                RolloutConfig(include_sams=True)):
        try:
            cfg.validate()
        except ValueError as exc:
            assert "include_sams" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"{type(cfg).__name__}.validate() accepted include_sams=True"
            )

    # The airbase-only default stays legal in both.
    TrainConfig(n_iterations=1).validate()
    RolloutConfig().validate()


def test_low_n_known_warns_about_the_bonmin_stall() -> None:
    """The stall hazard now tracks n_known -- the count that actually reaches bonmin."""
    out = _validate_capturing_stdout(TrainConfig(n_iterations=1, num_agents=2, n_known=2))
    assert "[WARN]" in out, out
    assert "SYMMETRY-STALL" in out, out
    assert "n_known=2" in out, out


def test_cli_exposes_the_construction_cell() -> None:
    """The five flags parse, default off the dataclass, and reject bad values."""
    parser = _build_arg_parser()

    d = TrainConfig(n_iterations=1)
    args = parser.parse_args(["--iterations", "1"])
    assert args.num_agents == d.num_agents
    assert args.n_known == d.n_known
    assert args.n_hidden == d.n_hidden
    assert args.min_target_distance_km == d.min_target_distance_km
    assert args.min_known_separation_km == d.min_known_separation_km

    args = parser.parse_args([
        "--iterations", "1", "--num-agents", "2", "--n-known", "5", "--n-hidden", "0",
        "--min-target-distance-km", "250.5", "--min-known-separation-km", "0",
    ])
    assert (args.num_agents, args.n_known, args.n_hidden) == (2, 5, 0)
    assert args.min_target_distance_km == 250.5
    assert args.min_known_separation_km == 0.0

    # Bad values are argparse usage errors (SystemExit), never a later traceback.
    for flag, bad in (
        ("--num-agents", "0"), ("--num-agents", "abc"), ("--num-agents", "1.5"),
        ("--n-known", "0"), ("--n-hidden", "-1"),
        ("--min-target-distance-km", "0"), ("--min-target-distance-km", "-5"),
        ("--min-target-distance-km", "abc"), ("--min-known-separation-km", "-1"),
    ):
        try:
            parser.parse_args(["--iterations", "1", flag, bad])
        except SystemExit:
            pass
        else:
            raise AssertionError("the CLI accepted %s %r" % (flag, bad))


def test_write_run_config_separates_generated_from_executed(tmp_path: Path) -> None:
    """`run_config.json` records the cell AND the generated-vs-executed world size.

    Post-B3 the two numbers no longer differ by "what this phase has not built yet" but
    by WHERE each target comes from: the generator writes the known ones, and
    `setup_episode`'s construction path patches the hidden ones in between the two
    solves. A record that blurred them would describe a 5-target world where 9 run.
    """
    cfg = TrainConfig(
        n_iterations=1, output_dir=tmp_path / "run",
        num_agents=2, n_known=5, n_hidden=4,
        min_target_distance_km=250.0, min_known_separation_km=120.0,
    )
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(
        write_run_config(run_dir, cfg).read_text(encoding="utf-8")
    )
    con = payload["construction"]
    assert con["num_agents"] == 2 and con["n_known"] == 5
    assert con["n_hidden"] == 4
    assert con["n_targets_generated"] == 5
    assert con["n_targets_emitted"] == 9 == cfg.n_targets_emitted
    assert con["min_target_distance_km"] == 250.0
    assert con["min_known_separation_km"] == 120.0
    assert con["ensure_discovery_chain"] is False
    assert con["strict_geometry"] is True
    assert con["detection_km"] == DETECTION_KM
    assert con["setup_mode"] == "construction"

    # The construction block matches the request the generator will actually receive:
    # the GENERATED count, not the executed one (the generator never writes a hidden target).
    var = build_variation_config(cfg, seed=0)
    assert con["n_targets_generated"] == var.num_red_airbases
    assert con["num_agents"] == var.num_aircraft
    assert con["min_target_distance_km"] == var.min_target_distance_km
    assert con["min_known_separation_km"] == var.min_target_separation_km

    # The legacy split record survives untouched alongside it.
    assert payload["derived_split"] == cfg.split_preview


def test_rollout_config_mirrors_the_train_reference_cell() -> None:
    """ANTI-DRIFT: `RolloutConfig` and `TrainConfig` agree on the whole cell.

    They deliberately do NOT share an import -- `graph_train` is a torch/PPO leaf and
    `graph_rollout` is an import-purity ENTRY module, so coupling them to share five
    literals would be the wrong trade. This test is the seam instead: before B1 the two
    silently disagreed ((3, 3) + 2/3 vs (6, 6) + 0.5), which made a diagnostic rollout
    and a training run generate different worlds by default.
    """
    t = TrainConfig(n_iterations=1)
    r = RolloutConfig()
    for name in ("num_agents", "n_known", "n_hidden",
                 "min_target_distance_km", "min_known_separation_km",
                 "include_sams", "randomize_red_airbase_positions",
                 "stretch_target_ratio"):
        assert getattr(t, name) == getattr(r, name), (
            "RolloutConfig.%s (%r) drifted from TrainConfig.%s (%r)"
            % (name, getattr(r, name), name, getattr(t, name))
        )

    # The rollout no longer carries the retired split knobs at all.
    assert not hasattr(r, "partial_ratio"), "RolloutConfig still carries partial_ratio"
    assert not hasattr(r, "num_red_airbases"), "RolloutConfig still carries num_red_airbases"


def test_rollout_config_validate_accepts_the_reference_cell() -> None:
    """The default rollout cell is valid and quiet -- as the trainer's default is."""
    out = _validate_capturing_stdout(RolloutConfig())     # must NOT raise
    assert "[WARN]" not in out, out

    # A legal non-default cell: fewer agents than targets is fine, 0 separation means
    # "no separation constraint" rather than a typo.
    RolloutConfig(num_agents=2, n_known=5, n_hidden=0, min_known_separation_km=0.0).validate()


def test_rollout_config_validate_rejects_invalid_cells() -> None:
    """Same verdicts as TrainConfig: bad counts, bad distances, agents > targets."""
    for kwargs in (
        {"n_episodes": 0},
        {"num_agents": 0},
        {"n_known": 0},
        {"n_hidden": -1},
        {"num_agents": 4, "n_known": 3},
        {"min_target_distance_km": 0.0},
        {"min_target_distance_km": -1.0},
        {"min_known_separation_km": -1.0},
    ):
        try:
            RolloutConfig(**kwargs).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"RolloutConfig.validate() accepted {kwargs}")

    # The agents-vs-targets message names both quantities, like the trainer's.
    try:
        RolloutConfig(num_agents=4, n_known=3).validate()
    except ValueError as exc:
        assert "num_agents" in str(exc) and "n_known" in str(exc), str(exc)


def test_run_rollout_validates_before_touching_anything(tmp_path: Path) -> None:
    """An impossible cell costs nothing: no directory, no policy, no generator, no BLADE.

    Solver-free BY CONSTRUCTION -- `run_rollout` raises on its first statement, so this
    test never reaches the engine import, the generator, or bonmin. The surviving
    absence of the run directory is what proves the ORDERING, not merely the raise:
    validation placed after `out_dir.mkdir` would still raise and would still leave a
    half-built run behind.
    """
    out = tmp_path / "never_created"
    try:
        run_rollout(RolloutConfig(n_episodes=1, output_dir=out, num_agents=4, n_known=3))
    except ValueError as exc:
        assert "num_agents" in str(exc), str(exc)
    else:
        raise AssertionError("run_rollout accepted num_agents > n_known")

    assert not out.exists(), "run_rollout created its run directory before validating"


# =============================================================================
# T10 -- provenance: a run states what code, machine and seeds produced it
# =============================================================================

def test_provenance_block_is_complete_and_explicit() -> None:
    """Every required field is PRESENT -- unavailable facts say so, never vanish.

    The distinction this locks is the whole point of the block: a key that is absent
    and a key whose value could not be determined look identical to a reader six months
    later. So `available`/`error`/`reason`/`probe` are asserted to exist even in the
    cases where the underlying fact does not.

    `argv` and `repo_root` are injected so the assertion is about the COLLECTOR, not
    about how this test process happened to be invoked.
    """
    cfg = TrainConfig(n_iterations=3, episodes_per_iteration=4, base_seed=0,
                      eval_episodes=8, eval_base_seed=1_000_000)
    prov = collect_provenance(cfg, argv=["prog", "--iterations", "3"], repo_root=ROOT)

    assert prov["provenance_version"] >= 1
    assert prov["exact_cardinality_policy"] == _EXACT_CARDINALITY_POLICY
    assert prov["exact_cardinality_policy"] == "skip_and_account_v1"
    assert prov["collected_at"]

    # The invocation is recorded verbatim as an argv ARRAY (not a re-quoted string,
    # which could not be replayed).
    assert prov["invocation"]["argv"] == ["prog", "--iterations", "3"]
    assert prov["invocation"]["cwd"] and prov["invocation"]["python_executable"]
    assert prov["python"]["version"] and len(prov["python"]["version_info"]) == 3
    for key in ("system", "release", "machine"):
        assert key in prov["platform"], prov["platform"]

    # Targeted packages only -- never a pip freeze -- and each with an explicit verdict.
    assert set(prov["packages"]) == {"torch", "gymnasium", "blade", "match_aou"}
    for name, info in prov["packages"].items():
        assert set(info) == {"available", "version", "path", "error"}, (name, info)
        assert isinstance(info["available"], bool), (name, info)
        if info["available"]:
            assert info["path"], name          # the vendored fork carries no version
        else:
            assert info["error"], name         # ... but must then say why

    bonmin = prov["solver"]["bonmin"]
    assert set(bonmin) == {"executable", "available", "probe", "probe_output"}
    assert bonmin["probe"] is not None, bonmin
    if not bonmin["available"]:
        assert bonmin["executable"] is None and bonmin["probe"] == "not_found", bonmin

    git = prov["git"]
    assert isinstance(git["available"], bool)
    if git["available"]:
        assert len(git["commit"]) == 40 and int(git["commit"], 16) >= 0, git
        assert isinstance(git["dirty"], bool), git
    else:
        assert git["reason"], git

    assert prov["seeds"] == seed_bands(cfg)
    assert prov["train_config_location"] == "run_config.json:/train_config"


def test_git_provenance_reports_absence_explicitly(tmp_path: Path) -> None:
    """A directory that is not a repository yields available=False PLUS a reason.

    This is what makes provenance collection testable without depending on the
    developer machine's live Git state: the outcome is chosen by the test, not
    inherited from whatever the checkout looks like today.
    """
    info = _git_provenance(tmp_path)
    assert info["available"] is False, info
    assert info["commit"] is None, info
    assert info["reason"], "an unavailable commit SHA must say why"
    assert info["repo_root"] == str(tmp_path)


def test_probe_command_survives_non_utf8_output() -> None:
    """A probe that emits non-UTF-8 bytes is captured, not silently lost.

    MEASURED, not hypothetical: `bonmin -v` under nlp_env emits byte 0x81. With
    subprocess's `text=True` the decode happens on a reader THREAD, so the
    UnicodeDecodeError kills that thread, prints a traceback, and returns an EMPTY
    stdout with returncode 0 -- the probe would record "ok" with no output, losing the
    one fact it exists to capture, on every real training run.

    Driven through a child that writes the offending byte directly, so the regression is
    reproducible on any machine rather than only where bonmin is installed.
    """
    code = "import sys; sys.stdout.buffer.write(b'bonmin \\x81 v1.8')"
    returncode, stdout, stderr = _probe_command(
        [sys.executable, "-c", code], timeout=60,
    )
    assert returncode == 0, (returncode, stderr)
    assert "bonmin" in stdout and "v1.8" in stdout, repr(stdout)


def test_seed_bands_are_half_open_and_match_the_schedule() -> None:
    """The recorded bands are exactly the seeds the pure schedule functions produce."""
    cfg = TrainConfig(n_iterations=3, episodes_per_iteration=4, base_seed=100,
                      eval_episodes=5, eval_base_seed=1_000_000)
    bands = seed_bands(cfg)

    train = bands["train_band"]
    assert train["half_open"] is True
    real_train = [train_seed(cfg, it, j)
                  for it in range(cfg.n_iterations)
                  for j in range(cfg.episodes_per_iteration)]
    assert min(real_train) == train["start"]
    assert max(real_train) == train["stop"] - 1      # stop is EXCLUSIVE
    assert train["count"] == len(real_train) == 12

    ev = bands["eval_band"]
    assert ev["half_open"] is True and ev["count"] == 5
    assert [eval_seed(cfg, e) for e in range(cfg.eval_episodes)] == \
        list(range(ev["start"], ev["stop"]))
    assert bands["eval_band_is_fixed_across_rounds"] is True

    # Disabled eval records the absence explicitly rather than a phantom band.
    off = seed_bands(TrainConfig(n_iterations=1, eval_every=0))
    assert off["eval_enabled"] is False and off["eval_band"] is None


def test_write_run_config_embeds_the_provenance_block(tmp_path: Path) -> None:
    """One file describes a run: the config AND its provenance, never two manifests."""
    cfg = TrainConfig(n_iterations=1, output_dir=tmp_path / "run")
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    injected = collect_provenance(cfg, argv=["stub"], repo_root=ROOT)
    payload = json.loads(
        write_run_config(run_dir, cfg, provenance=injected).read_text(encoding="utf-8")
    )
    assert payload["provenance"] == json.loads(json.dumps(injected, default=str))
    # The resolved config still lives at the documented top-level key.
    assert payload["train_config"]["n_iterations"] == 1
    assert payload["provenance"]["train_config_location"] == \
        "run_config.json:/train_config"

    # Omitted, it is collected rather than skipped.
    auto = json.loads(
        write_run_config(run_dir, cfg).read_text(encoding="utf-8")
    )["provenance"]
    assert auto["exact_cardinality_policy"] == _EXACT_CARDINALITY_POLICY


# =============================================================================
# T11 -- the stub-driven trainer: skip-and-account + true pre-update evaluation
# =============================================================================

class _RecordingUpdater(PPOUpdater):
    """A REAL PPOUpdater whose `update` is observed and short-circuited.

    Real, not a mock, because `train()` also checkpoints it -- `save_checkpoint` needs a
    genuine optimizer state_dict and a genuine PPOConfig. Only `update` is replaced, so
    no gradient work (and no BLADE, no solver) happens while the loop's ORDERING and
    ACCOUNTING are under test.

    The returned diagnostics keep the REAL contract, which the loop's learning axis
    depends on: the batch shape is read off the buffer, and `n_epochs_run` is 0 when the
    batch holds no TRANSITIONS -- an all-zero-wake batch is a documented no-op even
    though it holds real episodes. Getting that backwards would let a batch that ran no
    gradient step advance `updates_completed`.
    """

    def __init__(self, policy, cfg, *, log):
        super().__init__(policy, cfg)
        self._log = log

    def update(self, buffer):
        rewards = [rec.episode_reward for rec in buffer.records]
        self._log.append(("update", len(rewards)))
        return {
            "baseline": (sum(rewards) / len(rewards)) if rewards else 0.0,
            "policy_loss": -0.01, "total_loss": -0.02, "entropy": 1.5,
            "mean_ratio": 1.0, "clip_fraction": 0.0, "approx_kl": 0.0,
            "max_ratio_dev": 0.0, "grad_norm": 0.5, "adv_std_raw": 0.1,
            "n_transitions": buffer.n_transitions,
            "n_episodes": buffer.n_episodes,
            "episodes_with_wakes": buffer.episodes_with_wakes,
            "n_epochs_run": 2 if buffer.n_transitions else 0,
        }


class _StubTransition:
    """The two fields the trainer reads off a wake: the acting ego and its meta-action.

    `EpisodeRecord.from_trajectory` groups on `ego_id` and `_add_meta_action_counts`
    reads `meta_action`; nothing else in the stubbed path touches a transition (the
    re-encode lives inside the update, which is replaced). Building a genuine
    `Transition` would require a real `GraphObservation`, i.e. BLADE -- the very thing
    these tests exist to avoid.
    """

    def __init__(self, ego_id: str, meta_action: int = 0):
        self.ego_id = ego_id
        self.meta_action = meta_action


def _weight_snapshot(policy) -> dict:
    """A detached copy of every encoder + head parameter, for equality comparison."""
    snap = {}
    for prefix, module in (("encoder", policy.encoder), ("head", policy.head)):
        for key, value in module.state_dict().items():
            snap["%s.%s" % (prefix, key)] = value.detach().clone()
    return snap


def _assert_weights_unchanged(before: dict, after: dict, what: str) -> None:
    assert set(before) == set(after), f"{what}: parameter set changed"
    for key in before:
        assert torch.equal(before[key], after[key]), f"{what}: {key} changed"


# A COMPLETE Git provenance verdict: both the SHA and the clean/dirty state known.
# The stub trainer installs this by default so that driving `train()` does not depend on
# the developer checkout's live state -- and so that the run-refusal gate is exercised
# only where a test asks for it.
_FAKE_GIT_OK = {
    "repo_root": "<stub>", "available": True, "commit": "a" * 40,
    "branch": "stub", "dirty": False, "dirty_path_count": 0, "reason": None,
}

# A SHA was recovered but the clean/dirty verdict was NOT -- incomplete provenance.
_FAKE_GIT_INCOMPLETE = {
    "repo_root": "<stub>", "available": False, "commit": "b" * 40,
    "branch": None, "dirty": None, "dirty_path_count": None,
    "reason": "git status --porcelain failed",
}


# The target roster every stubbed SUCCESSFUL episode reports, shaped like the reference
# cell: 3 known + 3 hidden = 6 executed targets, of which 3 distinct ones are confirmed
# (2 known + 1 hidden). Names only -- ids never reach a success block.
#
# `_STUB_CONFIRMED_KILLS` is deliberately LARGER than the 3 unique targets: it models the
# (ego, target) confirmation count that `EpisodeResult.confirmed_kills` reports, so any
# aggregate that silently reverted to it would read 4.0 instead of 3.0 and the tests
# would catch it.
_STUB_KNOWN_TARGETS = ("Enemy Airbase #1", "Enemy Airbase #2", "Enemy Airbase #3")
_STUB_HIDDEN_TARGETS = ("Hidden Airbase #001", "Hidden Airbase #002",
                        "Hidden Airbase #003")
_STUB_KNOWN_CONFIRMED = ("Enemy Airbase #1", "Enemy Airbase #2")
_STUB_HIDDEN_CONFIRMED = ("Hidden Airbase #002",)
_STUB_UNIQUE_CONFIRMED = len(_STUB_KNOWN_CONFIRMED) + len(_STUB_HIDDEN_CONFIRMED)  # 3
_STUB_TARGETS_TOTAL = len(_STUB_KNOWN_TARGETS) + len(_STUB_HIDDEN_TARGETS)         # 6
_STUB_CONFIRMED_KILLS = 4          # (ego, target) PAIRS -- must never be aggregated


# A SECOND valid roster with a different world and a different unique count. It replaces
# the former `empty_roster=True` variant, which encoded the false zero the review
# rejected: a degraded roster is no longer a successful measurement at all, so it cannot
# be used to show that the reward is independent of the roster. Two DIFFERENT VALID
# rosters make the same point without asserting that a broken one is fine.
_STUB_ALT_KNOWN_TARGETS = ("Floridistan AFB #11", "Floridistan AFB #22")
_STUB_ALT_HIDDEN_TARGETS = ("Hidden Airbase #009",)
_STUB_ALT_KNOWN_CONFIRMED = ("Floridistan AFB #22",)
_STUB_ALT_HIDDEN_CONFIRMED = ()
_STUB_ALT_UNIQUE_CONFIRMED = 1
_STUB_ALT_TARGETS_TOTAL = 3


def _stub_roster_fields(*, variant: str = "default") -> dict:
    """The observability fields a stubbed `_EpisodeOutcome` carries.

    Both variants are VALID rosters whose confirmed-name subsets reconcile with their
    unique count; they differ in world size, names and count. There is deliberately no
    "degraded" variant: a roster that cannot be established now fails the attempt, so a
    zero-roster success is not a state the trainer can be in.
    """
    if variant == "alt":
        return {
            "targets_confirmed_unique": _STUB_ALT_UNIQUE_CONFIRMED,
            "targets_total": _STUB_ALT_TARGETS_TOTAL,
            "known_target_names": _STUB_ALT_KNOWN_TARGETS,
            "hidden_target_names": _STUB_ALT_HIDDEN_TARGETS,
            "known_confirmed_names": _STUB_ALT_KNOWN_CONFIRMED,
            "hidden_confirmed_names": _STUB_ALT_HIDDEN_CONFIRMED,
        }
    return {
        "targets_confirmed_unique": _STUB_UNIQUE_CONFIRMED,
        "targets_total": _STUB_TARGETS_TOTAL,
        "known_target_names": _STUB_KNOWN_TARGETS,
        "hidden_target_names": _STUB_HIDDEN_TARGETS,
        "known_confirmed_names": _STUB_KNOWN_CONFIRMED,
        "hidden_confirmed_names": _STUB_HIDDEN_CONFIRMED,
    }


# A CLEAN fuel-damage record, for outcomes built by hand where the condition is not what
# is under test. Clean is the inert case: no ego, no window, no event.
_STUB_CLEAN_FUEL_DAMAGE = {
    "fuel_damage_plan": {"condition": CONDITION_CLEAN, "ego_id": None},
    "fuel_damage_outcome": {"condition": CONDITION_CLEAN, "fired": False,
                            "wake_occurred": False, "wake_meta_action": None},
    "selected_ego_rtb_issued": None,
}


def _stub_fuel_damage_fields(cfg, seed: int, mode) -> dict:
    """The FD-BASELINE-v1 fields a stubbed `_EpisodeOutcome` must now carry.

    The condition is resolved through the REAL `resolve_condition`, so a stubbed episode
    reports the same clean/damaged label the trainer scheduled for that seed -- otherwise
    the per-condition accounting these tests exercise would be checking the stub's
    opinion rather than the loop's.
    """
    params = cfg.fuel_damage_parameters(mode)
    condition = resolve_condition(episode_seed=int(seed), params=params)
    # The SEVERITY is resolved through the production function too, so a stubbed episode
    # reports the same CELL the loop scheduled it under. Without it a `seeded_variable`
    # run's stub plan would report `damaged` while its schedule counted `mild`, and the
    # tally would (correctly) refuse the episode -- exercising that refusal instead of
    # the feature under test. `None` under every legacy mode, which is what keeps the
    # clean/damaged tests byte-unchanged.
    severity = resolve_severity(episode_seed=int(seed), params=params)
    damaged = condition == CONDITION_DAMAGED
    return {
        "fuel_damage_plan": {"condition": condition, "severity": severity,
                             "ego_id": "ego_0" if damaged else None},
        "fuel_damage_outcome": {"condition": condition, "severity": severity,
                                "fired": damaged,
                                "wake_occurred": damaged, "wake_meta_action": None},
        "selected_ego_rtb_issued": True if damaged else None,
    }


def _run_stub_training(
    cfg: TrainConfig,
    *,
    failures=None,
    integrity_failures=(),
    wakes_per_episode: int = 0,
    git=None,
    events=None,
    roster_variant: str = "default",
    capture_stdout: bool = False,
):
    """Drive the REAL `train()` with the BLADE+solver episode body replaced by a stub.

    Everything under test is real: the loop, the seed schedule, the ledger, the record
    writers, the summary and a real policy + optimizer. Only three seams are stubbed --
    the episode body, the scenario generator and the gradient step -- because those are
    exactly the parts that need BLADE and bonmin, and none of them decides WHICH seed is
    attempted, WHEN evaluation happens, or HOW a failure is accounted for.

    `failures` maps ``seed -> (pipeline_stage, message)``; that seed raises an
    `EpisodeAttemptError` from the given stage, which is precisely what
    `_run_one_episode` raises for a real exact-cardinality construction failure.
    `integrity_failures` is a set of seeds that instead raise `EpisodeRosterError` -- a
    `MeasurementIntegrityError`, which names no stage and must ABORT the run rather than
    be accounted. The two knobs are deliberately separate: the whole correction is that
    these are different kinds of event, and a test that could not tell them apart could
    not show it.
    `wakes_per_episode` gives every SUCCESSFUL episode that many wakes -- 0 models the
    zero-wake case (real episodes, no gradient step, so no completed update).

    `git` replaces `_git_provenance`'s verdict; it defaults to `_FAKE_GIT_OK` so these
    tests neither depend on the developer checkout's live state nor trip the
    incomplete-provenance gate by accident. Pass `_FAKE_GIT_INCOMPLETE` to exercise it.

    `events` may be supplied by the caller so the log survives a `train()` that RAISES
    -- which is exactly what the provenance-gate test needs to inspect.

    Every successful stub episode reports the reference-cell target roster
    (`_stub_roster_fields`); pass `roster_variant="alt"` for a DIFFERENT valid roster.
    `capture_stdout=True` returns everything the run printed as ``state["stdout"]``,
    which is how the per-episode OK blocks are asserted.

    The stub episode body also TOUCHES the scenario file the real generator would have
    written for its `episode_tag` (`ScenarioGenerator.generate` names it
    ``episode_%04d_scenario.json``). Nothing is generated -- the point is only that the
    tag namespace is observable as filenames, so "a later eval round overwrote an earlier
    round's scenario" is testable without BLADE.

    Patched by hand with try/finally rather than via pytest's `monkeypatch` fixture, so
    these tests also run through this file's `__main__` runner (pytest is absent in
    nlp_env).

    Returns ``(summary, events, state)``. ``events`` is ONE ordered list mixing policy
    construction, every episode attempt and every update -- an interleaved log is the
    only way to assert that evaluation happened BEFORE training rather than merely that
    both happened. Each episode event carries its `episode_tag` as a fourth element.
    """
    failures = dict(failures or {})
    integrity_seeds = set(int(x) for x in integrity_failures)
    n_wakes = int(wakes_per_episode)
    git_verdict = dict(_FAKE_GIT_OK if git is None else git)
    events = [] if events is None else events
    roster_fields = _stub_roster_fields(variant=roster_variant)
    state: dict = {
        "weights_at_build": None, "at_first_train": None,
        "scen_dir": None, "stdout": "",
    }

    saved = {
        "_run_one_episode": graph_train._run_one_episode,
        "_build_generator": graph_train._build_generator,
        "PPOUpdater": graph_train.PPOUpdater,
        "build_policy": graph_train.build_policy,
        "_git_provenance": graph_train._git_provenance,
    }

    def fake_build_policy():
        policy = saved["build_policy"]()
        state["weights_at_build"] = _weight_snapshot(policy)
        events.append(("policy_built", None))
        return policy

    def fake_build_generator(scen_dir):
        events.append(("generator_built", None))
        state["scen_dir"] = Path(scen_dir)
        return object()          # the stub episode body never touches it

    def fake_run_one_episode(policy, gen, cfg_, *, seed, episode_tag, deterministic,
                             fuel_damage_mode=None, **extra):
        phase = "eval" if deterministic else "train"
        # FD-BASELINE-v1: evaluation attempts each held-out seed once per matched pair
        # member, passing the forced mode; training passes none. The mode is recorded on
        # the event so a pairing assertion can read it.
        #
        # VISUAL ARTIFACTS: `**extra` rather than `artifacts=None`, so the event can
        # record whether the keyword was passed AT ALL. A run that did not opt in must
        # call `_run_one_episode` exactly as it did before the feature existed, which is
        # a stronger claim than "it passed None".
        # GENERALIZED-V1 (Task 4): elements 7 and 8 record the CARDINALITY keyword the
        # same way -- its value, and whether it was passed AT ALL. Appended rather than
        # inserted, so every existing positional read of this tuple is unaffected.
        events.append(("episode", phase, int(seed), int(episode_tag), fuel_damage_mode,
                       extra.get("artifacts"), "artifacts" in extra,
                       extra.get("cardinality"), "cardinality" in extra))
        # Stand in for the file the real generator would have written under this tag.
        if state["scen_dir"] is not None:
            state["scen_dir"].mkdir(parents=True, exist_ok=True)
            (state["scen_dir"]
             / ("episode_%04d_scenario.json" % int(episode_tag))).write_text(
                json.dumps({"tag": int(episode_tag), "seed": int(seed)}),
                encoding="utf-8",
            )
        if phase == "train" and state["at_first_train"] is None:
            state["at_first_train"] = {
                "weights": _weight_snapshot(policy),
                "n_updates": sum(1 for e in events if e[0] == "update"),
                "n_eval_episodes": sum(
                    1 for e in events if e[0] == "episode" and e[1] == "eval"
                ),
            }
        if seed in integrity_seeds:
            raise graph_train.EpisodeRosterError(
                "stubbed measurement-integrity fault at seed %d" % int(seed)
            )
        if seed in failures:
            stage, message = failures[seed]
            raise EpisodeAttemptError(stage, ValueError(message))
        # GENERALIZED-V1: the real `_run_one_episode` resolves the scheduled cell and
        # reports what the world REALIZED, so the stub models both. On the fixed-cell
        # path no keyword arrives and the cardinality is the configured cell, which is
        # exactly what the real body resolves there too.
        stub_card = extra.get("cardinality") or graph_train.fixed_cell_cardinality(
            agent_count=int(cfg_.num_agents), known_count=int(cfg_.n_known),
            hidden_requested=int(cfg_.n_hidden),
        )
        return _EpisodeOutcome(
            trajectory=[_StubTransition("ego_%d" % (k % 2), k % 3)
                        for k in range(n_wakes)],
            reward=-0.5 + 0.01 * (seed % 7), ticks=42,
            ended="done", n_wakes=n_wakes,
            confirmed_kills=_STUB_CONFIRMED_KILLS, n_dead=0, seconds=0.01,
            **roster_fields,
            **_stub_fuel_damage_fields(cfg_, seed, fuel_damage_mode),
            cardinality=stub_card,
            hidden_realized=int(stub_card.hidden_requested),
            reward_breakdown={"u_ref": -1.0, "u_achieved": -1.0,
                              "reference_policy": cfg_.design.reference_policy},
        )

    graph_train._git_provenance = lambda repo_root: dict(git_verdict)
    graph_train._run_one_episode = fake_run_one_episode
    graph_train._build_generator = fake_build_generator
    graph_train.PPOUpdater = lambda policy, ppo: _RecordingUpdater(
        policy, ppo, log=events
    )
    graph_train.build_policy = fake_build_policy
    buf = io.StringIO()
    try:
        if capture_stdout:
            with contextlib.redirect_stdout(buf):
                summary = graph_train.train(cfg)
        else:
            summary = graph_train.train(cfg)
    finally:
        state["stdout"] = buf.getvalue()
        for name, original in saved.items():
            setattr(graph_train, name, original)
    return summary, events, state


def _episode_seeds(events, phase: str) -> list:
    return [e[2] for e in events if e[0] == "episode" and e[1] == phase]


def _episode_tags(events, phase: str) -> list:
    return [e[3] for e in events if e[0] == "episode" and e[1] == phase]


def test_failed_seeds_are_skipped_and_accounted(tmp_path: Path) -> None:
    """PO1. Every scheduled seed is attempted ONCE; failures are recorded, not replaced.

    The policy under test is `skip_and_account_v1`, and the failure it is written for is
    real: B2 places one hidden target per non-empty ego route and B3 demands
    `len(placements) == n_hidden` exactly, so a solve that leaves an ego idle fails the
    episode (measured: 2 of seeds 0..11 on the default cell). The tempting repairs --
    draw a replacement seed, retry, or slide the band -- all silently change the
    population a result is reported over, so what is locked here is that NONE of them
    happens: the attempted seed set is exactly the scheduled seed set, and the shortfall
    shows up as accounting instead.
    """
    cfg = TrainConfig(
        n_iterations=2, episodes_per_iteration=3, base_seed=0,
        output_dir=tmp_path / "run",
        eval_every=2, eval_episodes=2, eval_base_seed=1_000_000,
        checkpoint_every=0,
    )
    failures = {
        1: ("setup", "exact cardinality: 2 usable routes for 3 hidden targets"),
        4: ("run", "engine edge case"),
        1_000_001: ("generation", "strict geometry unsatisfiable"),
    }
    summary, events, _ = _run_stub_training(cfg, failures=failures,
                                            wakes_per_episode=2)

    # --- every ORIGINAL seed attempted exactly once, nothing else attempted ---
    scheduled_train = [train_seed(cfg, it, j)
                       for it in range(cfg.n_iterations)
                       for j in range(cfg.episodes_per_iteration)]
    assert _episode_seeds(events, "train") == scheduled_train == [0, 1, 2, 3, 4, 5]

    # Two eval ROUNDS (pre-update + the one at iteration 1), each attempting the FIXED
    # band once PER MATCHED PAIR MEMBER (FD-BASELINE-v1: forced clean, then forced
    # damaged, on the same seed). A failed eval seed is re-attempted on the next round --
    # that is the band being fixed, not a retry of a spent attempt.
    eval_seeds_seen = _episode_seeds(events, "eval")
    assert eval_seeds_seen == [1_000_000, 1_000_000, 1_000_001, 1_000_001] * 2, \
        eval_seeds_seen
    assert set(eval_seeds_seen) <= {1_000_000, 1_000_001}

    # --- the ledger: one record per failed attempt, with stage and reason ---
    ledger = [json.loads(line) for line in
              (Path(cfg.output_dir) / "episode_failures.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    # 2 train + 1 eval seed x 2 pair members x 2 rounds. The failure is keyed by SEED in
    # this stub, so both members of that seed's pair fail -- and each is recorded once.
    assert len(ledger) == 6, ledger

    train_failures = [r for r in ledger if r["phase"] == "train"]
    assert sorted(r["seed"] for r in train_failures) == [1, 4]
    assert {r["seed"]: r["pipeline_stage"] for r in train_failures} == \
        {1: "setup", 4: "run"}
    assert {r["seed"]: r["iteration"] for r in train_failures} == {1: 0, 4: 1}
    assert {r["seed"]: r["attempt_ordinal"] for r in train_failures} == {1: 1, 4: 1}
    assert {r["seed"]: r["episode_index"] for r in train_failures} == {1: 1, 4: 4}

    eval_failures = [r for r in ledger if r["phase"] == "eval"]
    assert [r["seed"] for r in eval_failures] == [1_000_001] * 4
    assert [r["evaluation_stage"] for r in eval_failures] == \
        [_EVAL_STAGE_PRE_UPDATE] * 2 + [_EVAL_STAGE_POST_UPDATE] * 2
    assert [r["updates_completed"] for r in eval_failures] == [0, 0, 2, 2]
    assert all(r["pipeline_stage"] == "generation" for r in eval_failures)
    # Both members of the pair are accounted, under their own conditions and their own
    # attempt ordinals -- a half-failed pair is never collapsed into one record.
    assert [r["condition"] for r in eval_failures] == \
        [CONDITION_CLEAN, CONDITION_DAMAGED] * 2
    assert [r["attempt_ordinal"] for r in eval_failures] == [2, 3, 2, 3]

    for record in ledger:
        # The ORIGINAL exception survives the attribution wrapper.
        assert record["error_type"] == "ValueError", record
        assert record["error_message"], record
        assert "ValueError" in record["traceback"], record
        assert record["pipeline_stage"] in _PIPELINE_STAGES, record

    # --- attempts = successes + failures, in every artifact ---
    train_records = _read_records(cfg.output_dir, "train_records.jsonl")
    assert [r["n_attempted"] for r in train_records] == [3, 3]
    for record in train_records:
        assert record["n_attempted"] == record["n_successful"] + record["n_failed"]
    assert [r["n_failed"] for r in train_records] == [1, 1]

    eval_records = _read_records(cfg.output_dir, "eval_records.jsonl")
    for record in eval_records:
        # 2 held-out seeds x 2 matched pair members = 4 episode attempts per round.
        assert record["n_attempted"] == record["n_successful"] + record["n_failed"] == 4
        assert record["n_failed"] == 2
        assert record["success_fraction"] == 0.5
        assert record["aggregates_over"] == "successful_episodes"
        # Seed 1_000_001 lost BOTH members, so it contributes no complete pair.
        assert record["n_pairs_attempted"] == 2
        assert record["n_pairs_successful"] == 1

    assert summary["train_episodes_attempted"] == 6
    assert summary["train_episodes_successful"] == 4
    assert summary["train_episodes_failed"] == 2
    assert summary["eval_episodes_attempted"] == 8
    assert summary["eval_episodes_successful"] == 4
    assert summary["eval_episodes_failed"] == 4
    assert summary["failures_recorded"] == 6
    assert summary["failures_by_phase"] == {"train": 2, "eval": 4}
    assert summary["failures_by_pipeline_stage"] == \
        {"setup": 1, "run": 1, "generation": 4}
    assert summary["failures_by_error_type"] == {"ValueError": 6}
    assert summary["accounting_reconciled"] is True
    assert summary["exact_cardinality_policy"] == "skip_and_account_v1"

    # And it is persisted, derived from those same records.
    persisted = json.loads(
        (Path(cfg.output_dir) / "run_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["train_episodes_attempted"] == 6
    assert persisted["failures_by_pipeline_stage"] == \
        summary["failures_by_pipeline_stage"]


def test_pre_update_evaluation_precedes_all_training(tmp_path: Path) -> None:
    """PO2. The held-out round at updates_completed=0 runs before ANY training work.

    "Before" is asserted against an interleaved event log rather than against the
    records, because the records cannot distinguish "evaluated first" from "evaluated
    later and written first". Three things must all be true at the moment the FIRST
    training episode starts: the policy exists, the whole eval band has already been
    attempted, and zero updates have run.

    The other half is that adding this round changes nothing else: the weights are
    untouched, and the training seed schedule is exactly what the pure functions say.
    """
    cfg = TrainConfig(
        n_iterations=2, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "run",
        eval_every=5, eval_episodes=3, eval_base_seed=1_000_000,
        checkpoint_every=0,
    )
    summary, events, state = _run_stub_training(cfg, wakes_per_episode=2)

    kinds = [e[0] if e[0] != "episode" else "episode:" + e[1] for e in events]
    first_train = kinds.index("episode:train")
    first_eval = kinds.index("episode:eval")
    assert kinds.index("policy_built") < first_eval < first_train
    assert "update" not in kinds[:first_train], kinds[:first_train]
    assert kinds.index("update") > first_train

    # The pre-update round used the FIXED held-out band, in order, once per MATCHED
    # PAIR MEMBER (FD-BASELINE-v1: each seed is run forced-clean then forced-damaged).
    assert _episode_seeds(events, "eval")[:6] == [
        1_000_000, 1_000_000, 1_000_001, 1_000_001, 1_000_002, 1_000_002,
    ]

    snapshot = state["at_first_train"]
    assert snapshot is not None, "no training episode ran"
    assert snapshot["n_updates"] == 0, "an optimizer update ran before training"
    assert snapshot["n_eval_episodes"] == cfg.eval_episodes * 2  # one pair per seed
    _assert_weights_unchanged(
        state["weights_at_build"], snapshot["weights"],
        "the pre-update evaluation modified the policy",
    )

    # The training seed schedule is untouched by the extra round.
    assert _episode_seeds(events, "train") == [
        train_seed(cfg, it, j)
        for it in range(cfg.n_iterations)
        for j in range(cfg.episodes_per_iteration)
    ] == [0, 1, 2, 3]

    eval_records = _read_records(cfg.output_dir, "eval_records.jsonl")
    first = eval_records[0]
    assert first["evaluation_stage"] == _EVAL_STAGE_PRE_UPDATE
    assert first["updates_completed"] == 0
    assert first["iteration"] is None, "a pre-update round is not iteration 0"
    assert first["n_attempted"] == 6 and first["n_successful"] == 6   # 3 seeds x 2
    assert first["n_pairs_attempted"] == 3 and first["n_pairs_successful"] == 3
    assert first["seed_band"] == {
        "start": 1_000_000, "stop": 1_000_003, "half_open": True,
    }

    # A post-update round states its REAL number of completed updates.
    last = eval_records[-1]
    assert last["evaluation_stage"] == _EVAL_STAGE_POST_UPDATE
    assert last["updates_completed"] == 2 and last["iteration"] == 1

    assert summary["initial_pre_update_eval"]["updates_completed"] == 0
    assert summary["initial_pre_update_eval"]["n_attempted"] == 6   # 3 seeds x 2
    assert summary["final_eval"]["updates_completed"] == 2
    assert summary["updates_completed"] == 2

    # Training iteration 0 shares the origin with the pre-update point.
    train_records = _read_records(cfg.output_dir, "train_records.jsonl")
    assert [r["updates_completed_before"] for r in train_records] == [0, 1]
    assert [r["updates_completed"] for r in train_records] == [1, 2]


def test_an_all_failed_batch_reports_a_missing_reward_not_zero(tmp_path: Path) -> None:
    """PO1/PO3. Zero successful episodes -> null reward, never 0.0 (the oracle optimum).

    The reward is oracle-normalized regret, so 0 is the best value an episode can
    report. An iteration whose every attempt failed measured nothing at all, and
    recording that as 0.0 would put a total data loss at the top of the learning curve.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "run", eval_every=0, checkpoint_every=0,
    )
    summary, _, _ = _run_stub_training(cfg, failures={
        0: ("setup", "exact cardinality"), 1: ("setup", "exact cardinality"),
    })

    record = _read_records(cfg.output_dir, "train_records.jsonl")[0]
    assert record["n_attempted"] == 2 and record["n_successful"] == 0
    assert record["n_failed"] == 2 and record["success_fraction"] == 0.0
    assert record["train_reward_mean"] is None, record
    assert record["baseline"] is None, record          # compat field lies about nothing
    assert record["wake_fraction_of_successful"] is None, record
    assert record["reward_min"] is None and record["kills_mean"] is None, record
    # No epochs ran, so the learning axis did not advance.
    assert record["updates_completed"] == record["updates_completed_before"] == 0

    assert summary["train_reward_first"] is None
    assert summary["train_reward_last"] is None
    assert summary["train_reward_mean"] is None
    assert summary["n_iterations_without_reward"] == 1
    assert summary["train_success_fraction"] == 0.0
    assert summary["train_episodes_failed"] == 2
    assert summary["accounting_reconciled"] is True

    # An iteration in which NOTHING completed is not an iteration in which episodes ran
    # and nobody woke. The two used to collide on `n_epochs_run == 0`.
    assert summary["n_all_failed_iterations"] == 1
    assert summary["n_zero_wake_iterations"] == 0
    assert summary["n_productive_iterations"] == 0


def test_a_successful_zero_wake_episode_is_not_a_failure(tmp_path: Path) -> None:
    """A real episode in which nobody woke is a SUCCESS with a real reward.

    The two are easy to collapse into one "nothing happened" bucket and they are
    opposite findings: a zero-wake episode says the event-triggered policy was never
    invoked, a failed attempt says the episode never existed. Only the second belongs
    in the ledger.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=3, base_seed=0,
        output_dir=tmp_path / "run", eval_every=0, checkpoint_every=0,
    )
    summary, _, _ = _run_stub_training(cfg)      # every episode succeeds, no wakes

    record = _read_records(cfg.output_dir, "train_records.jsonl")[0]
    assert record["n_attempted"] == record["n_successful"] == 3
    assert record["n_failed"] == 0 and record["success_fraction"] == 1.0
    assert record["episodes_with_wakes"] == 0 and record["n_transitions"] == 0
    assert record["train_reward_mean"] is not None, "a real episode has a real reward"
    assert record["wake_fraction_of_successful"] == 0.0

    assert (Path(cfg.output_dir) / "episode_failures.jsonl").read_text(
        encoding="utf-8").strip() == "", "a zero-wake success was logged as a failure"
    assert summary["failures_recorded"] == 0
    assert summary["train_zero_wake_episodes"] == 3
    assert summary["updates_completed"] == 0, "an empty batch is not a completed update"

    # The mirror image of the all-failed case: episodes DID complete, nobody woke.
    assert summary["n_zero_wake_iterations"] == 1
    assert summary["n_all_failed_iterations"] == 0
    assert summary["n_productive_iterations"] == 0


def test_console_flag_never_reports_both_failure_states(tmp_path: Path) -> None:
    """An iteration prints AT MOST ONE of the two flags -- they are different findings.

    The regression: an all-failed batch printed `[ZERO-WAKE: update skipped]` AND
    `[ALL n ATTEMPTS FAILED]` on the same line, telling the operator both that episodes
    ran and nobody woke and that no episode ran at all.
    """
    def _run(name, **kwargs):
        cfg = TrainConfig(
            n_iterations=1, episodes_per_iteration=2, base_seed=0,
            output_dir=tmp_path / name, eval_every=0, checkpoint_every=0,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _run_stub_training(cfg, **kwargs)
        return buf.getvalue()

    all_failed = _run("all_failed", failures={
        0: ("setup", "exact cardinality"), 1: ("setup", "exact cardinality"),
    })
    assert "[ALL 2 ATTEMPTS FAILED" in all_failed, all_failed
    assert "[ZERO-WAKE" not in all_failed, all_failed

    zero_wake = _run("zero_wake")            # every episode succeeds, none wakes
    assert "[ZERO-WAKE" in zero_wake, zero_wake
    assert "ATTEMPTS FAILED" not in zero_wake, zero_wake

    productive = _run("productive", wakes_per_episode=2)
    assert "[ZERO-WAKE" not in productive and "ATTEMPTS FAILED" not in productive


# =============================================================================
# T12 -- P1: one truthful OK block per successful episode
# =============================================================================

# Any RFC-4122-shaped id. A success block naming targets by uuid would be technically
# complete and practically unreadable, and generated target ids are not even seed-stable
# across runs (CLAUDE.md section 8), so a uuid there is never the right answer.
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _ok_blocks(stdout: str) -> list:
    """Every printed OK block, keyed off the ``] OK`` header and its ``ended=`` footer.

    The block is no longer a fixed line count: FD-BASELINE-v1 adds one ``fuel_damage=``
    line for a clean episode and three for a damaged one. It is delimited instead, which
    is also what makes "the footer is always last" an assertion rather than an index.
    """
    lines = stdout.splitlines()
    blocks = []
    for i, line in enumerate(lines):
        if not line.endswith("] OK"):
            continue
        end = next(
            (j for j in range(i + 1, len(lines))
             if lines[j].strip().startswith("ended=")),
            None,
        )
        assert end is not None, "an OK block has no ended= footer: %r" % (lines[i:],)
        blocks.append(lines[i:end + 1])
    return blocks


def _assert_block_is_complete(block: list) -> None:
    """Every field P1 requires is present, with the stub roster's exact values."""
    header, body = block[0], block[1:]
    assert header.endswith("] OK"), header
    text = "\n".join(block)

    assert "reward=" in body[0] and "wakes=" in body[0], body[0]
    assert ("targets_confirmed_unique=%d/%d"
            % (_STUB_UNIQUE_CONFIRMED, _STUB_TARGETS_TOTAL)) in body[0], body[0]
    for label, names in (
        ("known_targets", _STUB_KNOWN_TARGETS),
        ("known_confirmed", _STUB_KNOWN_CONFIRMED),
        ("hidden_targets", _STUB_HIDDEN_TARGETS),
        ("hidden_confirmed", _STUB_HIDDEN_CONFIRMED),
    ):
        line = next(l for l in body if l.strip().startswith(label + "="))
        assert json.loads(line.split("=", 1)[1]) == list(names), line

    # FD-BASELINE-v1: every block states the episode's condition, and a damaged one
    # states what the event did. A block that stopped reporting the difficulty factor
    # would leave an operator unable to tell the two halves of a mixture apart.
    fd_line = next(l for l in body if l.strip().startswith("fuel_damage="))
    if "fuel_damage=damaged" in fd_line:
        assert "ego=" in fd_line and "fired=" in fd_line, fd_line
        assert any("fuel_before=" in l and "fuel_after=" in l for l in body), body
        # PLANNED and LIVE bounds are reported side by side and named apart, so a reader
        # can never mistake the preflight window for the one the mutation was validated
        # against.
        bounds = next(l for l in body if "planned_rtb_floor=" in l)
        assert "planned_continue_req=" in bounds, bounds
        assert "live_rtb_floor=" in bounds and "live_continue_req=" in bounds, bounds
        # `rtb_command=`, not `rtb_issued=`: it is an emitted COMMAND, never the
        # executor's lifecycle latch (which is also set for a dead ego).
        assert any("fd_wake=" in l and "rtb_command=" in l for l in body), body
        assert not any("rtb_issued=" in l for l in body), body
    else:
        assert fd_line.strip() == "fuel_damage=clean ego=none", fd_line

    assert body[-1].strip().startswith("ended="), body[-1]
    assert "ticks=" in body[-1] and "dead=" in body[-1], body[-1]
    assert "elapsed=" in body[-1] and body[-1].rstrip().endswith("s"), body[-1]
    assert not _UUID_RE.search(text), "a target uuid reached a success block:\n" + text
    # cp1255 console: the block must survive being written to it.
    text.encode("ascii")


def test_every_successful_episode_prints_one_labelled_ok_block(tmp_path: Path) -> None:
    """P1. Train, pre-update eval and post-update eval each emit ONE correct OK block.

    The finding this closes: a run printed one aggregate line per ITERATION and one per
    eval ROUND, so while a batch of multi-minute episodes was collecting there was no way
    to tell what any individual episode had done -- or whether anything was happening at
    all. The block is emitted on RETURN from each attempt, before the next one starts.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "run",
        eval_every=1, eval_episodes=2, eval_base_seed=1_000_000,
        checkpoint_every=0,
    )
    _, _, state = _run_stub_training(cfg, wakes_per_episode=2, capture_stdout=True)
    out = state["stdout"]

    blocks = _ok_blocks(out)
    headers = [b[0] for b in blocks]
    # 2 pre-update eval SEEDS x 2 matched pair members, then 2 train, then the same 4
    # post-update eval members -- in that order. The eval headers name the condition, so
    # the two members of one seed's pair are never confused with each other.
    assert headers == [
        "[eval stage=pre_update ep=0 clean seed=1000000] OK",
        "[eval stage=pre_update ep=0 damaged seed=1000000] OK",
        "[eval stage=pre_update ep=1 clean seed=1000001] OK",
        "[eval stage=pre_update ep=1 damaged seed=1000001] OK",
        "[train iter=0 ep=0 seed=0] OK",
        "[train iter=0 ep=1 seed=1] OK",
        "[eval stage=post_update ep=0 clean seed=1000000] OK",
        "[eval stage=post_update ep=0 damaged seed=1000000] OK",
        "[eval stage=post_update ep=1 clean seed=1000001] OK",
        "[eval stage=post_update ep=1 damaged seed=1000001] OK",
    ], headers
    for block in blocks:
        _assert_block_is_complete(block)


def test_a_failed_attempt_prints_no_ok_block_and_still_accounts(tmp_path: Path) -> None:
    """P1. A failure keeps its FAILED line and its ledger entry -- and gains no OK block.

    `OK` must mean "this attempt completed", so exactly the attempts that completed may
    print one. The existing failure reporting is unchanged.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=3, base_seed=0,
        output_dir=tmp_path / "run", eval_every=0, checkpoint_every=0,
    )
    summary, _, state = _run_stub_training(
        cfg, failures={1: ("setup", "exact cardinality: 2 usable routes")},
        wakes_per_episode=1, capture_stdout=True,
    )
    out = state["stdout"]

    headers = [b[0] for b in _ok_blocks(out)]
    assert headers == ["[train iter=0 ep=0 seed=0] OK",
                       "[train iter=0 ep=2 seed=2] OK"], headers
    assert "seed=1] OK" not in out, out
    assert "[iter 0 ep 1] FAILED (seed=1, cell=damaged, stage=setup)" in out, out

    # Accounting is untouched by the new output.
    assert summary["train_episodes_attempted"] == 3
    assert summary["train_episodes_successful"] == 2
    assert summary["train_episodes_failed"] == 1
    assert summary["accounting_reconciled"]
    ledger = _read_records(cfg.output_dir, "episode_failures.jsonl")
    assert [r["seed"] for r in ledger] == [1], ledger


def test_ok_block_reports_the_real_ending_not_a_verdict() -> None:
    """P1. `OK` is "the attempt completed"; `ended` still states how the episode ended."""
    out = _EpisodeOutcome(
        trajectory=[], reward=-0.25, ticks=7, ended="truncated", n_wakes=0,
        confirmed_kills=0, n_dead=1, seconds=3.25,
        **_stub_roster_fields(),
        **_STUB_CLEAN_FUEL_DAMAGE,
    )
    block = graph_train._format_episode_block("[train iter=2 ep=9 seed=9]", out)
    lines = block.splitlines()
    assert lines[0] == "[train iter=2 ep=9 seed=9] OK"
    assert "reward=-0.2500" in lines[1] and "wakes=0" in lines[1]
    assert lines[-1] == "  ended=truncated ticks=7 dead=1 elapsed=3.2s", lines[-1]
    _assert_block_is_complete(lines)


def test_ok_block_survives_a_non_ascii_target_name() -> None:
    """P1. A stray non-ASCII target name is transliterated, never a UnicodeEncodeError.

    Target names come out of a scenario JSON this module does not own, and the Windows
    console this project runs on is cp1255. A print that raised would abort a run whose
    episode had already completed successfully.
    """
    out = _EpisodeOutcome(
        trajectory=[], reward=0.0, ticks=1, ended="done", n_wakes=0,
        confirmed_kills=0, n_dead=0, seconds=0.5,
        targets_confirmed_unique=0, targets_total=1,
        known_target_names=("Enemy Airbase ÅÜ",), hidden_target_names=(),
        known_confirmed_names=(), hidden_confirmed_names=(),
        **_STUB_CLEAN_FUEL_DAMAGE,
    )
    block = graph_train._format_episode_block("[train iter=0 ep=0 seed=0]", out)
    block.encode("ascii")            # would raise if a raw name had leaked through
    assert "Enemy Airbase" in block


# =============================================================================
# T13 -- P2: confirmations are counted UNIQUELY over target id
# =============================================================================

def test_unique_confirmed_target_ids_deduplicates_over_ego() -> None:
    """P2. Two egos confirming ONE target is ONE target.

    `GraphPlanExecutor.done` is a set of (ego_id, target_id) pairs, so its length counts
    CONFIRMATIONS. The approved first probe reported more "kills" than the world had
    targets because that length was being aggregated as a target count. The executor's
    set is correct and unchanged; this is the conversion that was missing.
    """
    done = {("ego-a", "target-1"), ("ego-b", "target-1"), ("ego-a", "target-2")}
    unique = graph_train._unique_confirmed_target_ids(done)
    assert unique == {"target-1", "target-2"}
    assert len(unique) == 2 and len(done) == 3

    # The empty / absent cases are the ones a degraded episode hits.
    assert graph_train._unique_confirmed_target_ids(set()) == set()
    assert graph_train._unique_confirmed_target_ids(None) == set()

    # Ids are stringified, so a non-str target id cannot split one target into two.
    assert graph_train._unique_confirmed_target_ids(
        {("ego-a", 7), ("ego-b", "7")}
    ) == {"7"}


def test_roster_split_totals_the_unique_count() -> None:
    """P2. total == known confirmed + hidden confirmed, over the executed denominator."""
    roster = graph_train._TargetRoster(
        known_ids=("k1", "k2", "k3"),
        known_names=("Enemy Airbase #1", "Enemy Airbase #2", "Enemy Airbase #3"),
        hidden_ids=("h1", "h2", "h3"),
        hidden_names=("Hidden Airbase #001", "Hidden Airbase #002",
                      "Hidden Airbase #003"),
    )
    assert roster.total == 6

    done = {("ego-a", "k1"), ("ego-b", "k1"), ("ego-a", "k3"), ("ego-c", "h2")}
    known, hidden = roster.confirmed(graph_train._unique_confirmed_target_ids(done))
    assert known == ("Enemy Airbase #1", "Enemy Airbase #3")
    assert hidden == ("Hidden Airbase #002",)
    # The two halves are disjoint and cover the world, so the split IS the total.
    assert len(known) + len(hidden) == 3 <= roster.total

    # Roster ORDER is preserved regardless of the set's iteration order.
    all_known, all_hidden = roster.confirmed(set(roster.known_ids) | set(roster.hidden_ids))
    assert all_known == roster.known_names and all_hidden == roster.hidden_names
    assert len(all_known) + len(all_hidden) == roster.total


def test_trainer_aggregates_use_the_unique_target_count(tmp_path: Path) -> None:
    """P2. Authoritative keys and compatibility aliases carry the SAME unique count.

    `kills_mean` / `eval_kills_mean` are kept so a pre-B4 reader still resolves, but they
    are aliases now -- fed from the unique-target count, never from the (ego, target)
    confirmation count the stub also reports.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "run",
        eval_every=1, eval_episodes=2, eval_base_seed=1_000_000,
        checkpoint_every=0,
    )
    _run_stub_training(cfg, wakes_per_episode=2)

    train_recs = _read_records(cfg.output_dir, "train_records.jsonl")
    eval_recs = _read_records(cfg.output_dir, "eval_records.jsonl")
    assert train_recs and eval_recs

    for rec in train_recs:
        assert rec["targets_confirmed_unique_mean"] == float(_STUB_UNIQUE_CONFIRMED)
        assert rec["kills_mean"] == rec["targets_confirmed_unique_mean"]
        assert rec["target_confirmation_count_semantics"] == "unique_target_id"
        # The (ego, target) pair count must not be what got aggregated.
        assert rec["kills_mean"] != float(_STUB_CONFIRMED_KILLS)
        # The reference cell can never confirm more than its 6 executed targets.
        assert rec["targets_confirmed_unique_mean"] <= _STUB_TARGETS_TOTAL

    for rec in eval_recs:
        assert rec["eval_targets_confirmed_unique_mean"] == float(_STUB_UNIQUE_CONFIRMED)
        assert rec["eval_kills_mean"] == rec["eval_targets_confirmed_unique_mean"]
        assert rec["target_confirmation_count_semantics"] == "unique_target_id"
        assert rec["eval_kills_mean"] != float(_STUB_CONFIRMED_KILLS)
        assert rec["eval_targets_confirmed_unique_mean"] <= _STUB_TARGETS_TOTAL


def test_observability_does_not_touch_reward_or_ppo_diagnostics(tmp_path: Path) -> None:
    """P2. Two DIFFERENT valid rosters leave reward, PPO, seeds and eval tags identical.

    The roster is a read-only projection of the context; if the reward or the update
    could see it, this observability task would have changed the experiment.

    This used to compare a populated roster against a DEGRADED one and assert the
    degraded run recorded `targets_confirmed_unique_mean == 0.0` -- i.e. it codified the
    false zero the review rejected. A degraded roster is no longer a successful
    measurement, so the invariance is now shown between two valid worlds instead.
    """
    reward_and_ppo = (
        "train_reward_mean", "baseline", "reward_min", "reward_max",
        "policy_loss", "total_loss", "entropy", "mean_ratio", "clip_fraction",
        "approx_kl", "max_ratio_dev", "grad_norm", "adv_std_raw",
        "n_transitions", "n_episodes", "episodes_with_wakes", "n_epochs_run",
        "meta_action_counts", "meta_action_fractions", "ended_counts", "ticks_mean",
    )

    def _run(name, *, variant):
        cfg = TrainConfig(
            n_iterations=2, episodes_per_iteration=2, base_seed=0,
            output_dir=tmp_path / name,
            eval_every=1, eval_episodes=2, eval_base_seed=1_000_000,
            checkpoint_every=0,
        )
        _, events, _ = _run_stub_training(cfg, wakes_per_episode=2,
                                          roster_variant=variant)
        return (_read_records(cfg.output_dir, "train_records.jsonl"),
                _read_records(cfg.output_dir, "eval_records.jsonl"),
                events)

    a_train, a_eval, a_events = _run("roster_default", variant="default")
    b_train, b_eval, b_events = _run("roster_alt", variant="alt")

    assert len(a_train) == len(b_train) == 2
    for x, y in zip(a_train, b_train):
        for key in reward_and_ppo:
            assert x[key] == y[key], (key, x[key], y[key])
    for x, y in zip(a_eval, b_eval):
        for key in ("eval_reward_mean", "eval_reward_min", "eval_reward_max",
                    "eval_wakes_mean", "meta_action_counts", "ended_counts",
                    "n_attempted", "n_successful", "n_failed"):
            assert x[key] == y[key], (key, x[key], y[key])

    # Seeds and eval scenario tags are unaffected too.
    for phase in ("train", "eval"):
        assert _episode_seeds(a_events, phase) == _episode_seeds(b_events, phase)
        assert _episode_tags(a_events, phase) == _episode_tags(b_events, phase)

    # ... and the only thing that DID change is the confirmation count itself, which in
    # both cases is a REAL count, never a degraded zero.
    assert a_train[0]["targets_confirmed_unique_mean"] == float(_STUB_UNIQUE_CONFIRMED)
    assert b_train[0]["targets_confirmed_unique_mean"] == float(
        _STUB_ALT_UNIQUE_CONFIRMED)
    assert a_train[0]["targets_confirmed_unique_mean"] != \
        b_train[0]["targets_confirmed_unique_mean"]


# =============================================================================
# T13b -- the FALSE-ZERO regression, at the seam that sees both inputs
# =============================================================================
#
# The rejected candidate computed the authoritative count as
# `len(known_confirmed) + len(hidden_confirmed)` -- i.e. from the names the roster
# managed to CLASSIFY -- while `_episode_target_roster` swallowed every structural
# exception and returned an empty roster. An episode with real confirmations was
# therefore recorded as a SUCCESSFUL `0/0`, and that zero flowed into
# `targets_confirmed_unique_mean` and its aliases.
#
# These tests drive the real `_run_one_episode`, which is the only place that receives
# BOTH the roster and the executor's `done` pairs. Constructing an `_EpisodeOutcome` by
# hand cannot reach the defect, because the defect was in how that outcome's count was
# derived. The BLADE/solver seams around it (`setup_episode`, `run_episode`,
# `compute_episode_reward`, the generator) are stubbed; everything between them is real.

class _FakeStep:
    def __init__(self, target_id):
        self.step_kind = StepKind.ATTACK
        self.target_id = target_id


class _FakeTask:
    """The only shape `_task_target_id` reads: `.steps[*].step_kind` / `.target_id`."""

    def __init__(self, target_id=None, *, steps=None):
        self.steps = [_FakeStep(target_id)] if steps is None else steps


class _FakeBelief:
    def __init__(self, tasks):
        self.tasks = tasks


class _FakeUnit:
    def __init__(self, name):
        self.name = name


class _FakeScenario:
    """`get_target` over a fixed id -> name map; `raise_for` models a broken lookup."""

    def __init__(self, names, *, raise_for=()):
        self._names = dict(names)
        self._raise_for = set(raise_for)

    def get_target(self, target_id):
        if target_id in self._raise_for:
            raise RuntimeError("simulated scenario lookup failure for %s" % target_id)
        name = self._names.get(target_id)
        return None if name is None else _FakeUnit(name)


class _FakeGame:
    """`current_scenario` for the roster, plus the engine's read-only scenario exporter.

    `export_calls` counts every `export_scenario()` -- which is how the OFF path proves it
    never calls it at all, and the ON path that it calls it EXACTLY once per attempt.
    `export_error` makes the exporter raise, which is what an artifact-side infrastructure
    failure looks like from this module's point of view.
    """

    def __init__(self, scenario, *, exported=None, export_error=None):
        self.current_scenario = scenario
        self.export_calls = 0
        self._exported = exported
        self._export_error = export_error

    def export_scenario(self):
        self.export_calls += 1
        if self._export_error is not None:
            raise self._export_error
        return {} if self._exported is None else self._exported


class _FakeExecutor:
    def __init__(self, done):
        self.done = set(done)


class _FakeEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeCtx:
    """The `EpisodeContext` attributes the observability path actually reads.

    `known_target_ids` / `executed_target_ids` are the RAW t=0 world snapshots setup takes
    BEFORE either solve. They are kept deliberately separate from `beliefs` (the
    allocated-only known plan) and `oracle_tasks` (the allocated-only oracle), because the
    whole defect being regressed is those two being read as world inventories.
    """

    def __init__(self, *, beliefs, oracle_tasks, scenario, done,
                 known_target_ids, executed_target_ids,
                 exported=None, export_error=None):
        self.beliefs = beliefs
        self.oracle_tasks = oracle_tasks
        self.known_target_ids = known_target_ids
        self.executed_target_ids = executed_target_ids
        self.game = _FakeGame(scenario, exported=exported, export_error=export_error)
        self.executor = _FakeExecutor(done)
        self.env = _FakeEnv()


class _FakeResult:
    def __init__(self, *, confirmed_kills, trajectory=None, n_wakes=0):
        self.trajectory = [] if trajectory is None else list(trajectory)
        self.ticks = 11
        self.ended = "done"
        self.n_wakes = n_wakes
        self.confirmed_kills = confirmed_kills
        self.n_dead = 0


class _FakeReward:
    reward = -0.25


class _FakeScenarioPath:
    @staticmethod
    def read_text(encoding=None):
        return "{}"


class _FakeGenerator:
    @staticmethod
    def generate(episode, config):
        return _FakeScenarioPath()


def _reference_ctx(*, done, raise_names=(), known=3, hidden=3,
                   belief_ids=None, oracle_ids=None,
                   known_snapshot=None, executed_snapshot=None,
                   exported=None, export_error=None):
    """A valid reference-cell context: `known` known + `hidden` hidden targets.

    Ids are `k0..`/`h0..` (never uuids, so a leak into a success block is unmistakable);
    names mimic the real generator's `Floridistan AFB #N` / `Hidden Airbase #NNN`.

    THREE INDEPENDENT LAYERS, which is the point of the fixture after the world-truth fix:

      * `known_snapshot` / `executed_snapshot` -- the RAW t=0 world, defaulting to the
        full `known` + `hidden` cell. This is what the roster must read.
      * `belief_ids` -- the ALLOCATED-ONLY known plan, defaulting to every known target.
        Pass a strict subset to model a solver that left a known target unselected.
      * `oracle_ids` -- the ALLOCATED-ONLY oracle, defaulting to the whole world. Pass a
        strict subset to model an oracle that did not select some target. The roster must
        be indifferent to it.

    `exported` / `export_error` configure the game's `export_scenario()` -- the visual
    artifact tests' only additional seam.
    """
    known_ids = ["k%d" % i for i in range(known)]
    hidden_ids = ["h%d" % i for i in range(hidden)]
    names = {t: "Floridistan AFB #%d" % (i + 1) for i, t in enumerate(known_ids)}
    names.update({t: "Hidden Airbase #%03d" % (i + 1)
                  for i, t in enumerate(hidden_ids)})
    planned = list(known_ids if belief_ids is None else belief_ids)
    allocated = list(known_ids + hidden_ids if oracle_ids is None else oracle_ids)
    belief_tasks = [_FakeTask(t) for t in planned]
    return _FakeCtx(
        beliefs={"ego_%d" % i: _FakeBelief(list(belief_tasks)) for i in range(3)},
        oracle_tasks=[_FakeTask(t) for t in allocated],
        known_target_ids=tuple(known_ids if known_snapshot is None else known_snapshot),
        executed_target_ids=tuple(
            known_ids + hidden_ids if executed_snapshot is None else executed_snapshot
        ),
        scenario=_FakeScenario(names, raise_for=raise_names),
        done=done,
        exported=exported,
        export_error=export_error,
    )


def _run_one_episode_against(ctx, *, confirmed_kills=0, reward=None):
    """Drive the REAL `_run_one_episode` against a fake context; returns its outcome.

    Only the BLADE / solver seams are replaced. The roster snapshot, the unique-id
    computation, the name split, the failure attribution and the outcome construction
    are the production code paths.

    ``reward`` replaces the stubbed ``compute_episode_reward`` outright, so a caller can
    make the REWARD STAGE raise and exercise its attribution -- which is otherwise
    unreachable, because this helper installs its own stub last.
    """
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      eval_every=0, checkpoint_every=0)
    saved = {
        "setup_episode": graph_train.setup_episode,
        "run_episode": graph_train.run_episode,
        "compute_episode_reward": graph_train.compute_episode_reward,
    }
    graph_train.setup_episode = lambda *a, **k: ctx
    graph_train.run_episode = lambda *a, **k: _FakeResult(
        confirmed_kills=confirmed_kills)
    graph_train.compute_episode_reward = (
        reward if reward is not None else (lambda *a, **k: _FakeReward())
    )
    try:
        return graph_train._run_one_episode(
            None, _FakeGenerator(), cfg,
            seed=0, episode_tag=0, deterministic=False,
        )
    finally:
        for name, original in saved.items():
            setattr(graph_train, name, original)


def test_unique_count_is_taken_directly_from_the_executor_done_set() -> None:
    """P2/P1. The count is `len(unique ids)`, and the names only have to agree with it.

    Duplicate `(ego, target)` pairs collapse; the raw pair count is larger and is not
    what is reported.
    """
    done = {
        ("ego_0", "k0"), ("ego_1", "k0"),        # one target, two confirmations
        ("ego_1", "k2"), ("ego_2", "h1"), ("ego_0", "h1"),
    }
    out = _run_one_episode_against(_reference_ctx(done=done), confirmed_kills=len(done))

    confirmed_ids = graph_train._unique_confirmed_target_ids(done)
    assert len(confirmed_ids) == 3 and len(done) == 5
    assert out.targets_confirmed_unique == len(confirmed_ids) == 3
    assert out.targets_total == 6
    # The names reconcile with the count rather than producing it.
    assert out.known_confirmed_names == ("Floridistan AFB #1", "Floridistan AFB #3")
    assert out.hidden_confirmed_names == ("Hidden Airbase #002",)
    assert (len(out.known_confirmed_names) + len(out.hidden_confirmed_names)
            == out.targets_confirmed_unique)
    # The raw pair count is preserved separately and is NOT the reported one.
    assert out.confirmed_kills == 5 != out.targets_confirmed_unique


def test_a_broken_name_lookup_keeps_the_id_the_count_and_the_denominator() -> None:
    """P1/P2. A display failure degrades TEXT only -- never an id, a count or a total.

    This is the one degradation that stays nonfatal, and the point is that it is
    inert: the target keeps its roster slot and its contribution to every number.
    """
    done = {("ego_0", "k1"), ("ego_1", "h0")}
    ctx = _reference_ctx(done=done, raise_names=("k1", "h0"))
    out = _run_one_episode_against(ctx, confirmed_kills=2)

    assert out.targets_confirmed_unique == 2          # unchanged by the failed lookups
    assert out.targets_total == 6                     # denominator unchanged
    assert len(out.known_target_names) == 3 and len(out.hidden_target_names) == 3
    # The two unresolvable targets are still present -- as placeholders, not as gaps.
    assert out.known_target_names[1] == "<unnamed target>"
    assert out.hidden_target_names[0] == "<unnamed target>"
    assert out.known_confirmed_names == ("<unnamed target>",)
    assert out.hidden_confirmed_names == ("<unnamed target>",)
    assert (len(out.known_confirmed_names) + len(out.hidden_confirmed_names)
            == out.targets_confirmed_unique)

    # A target the scenario simply does not know is the same story.
    ctx2 = _reference_ctx(done={("ego_0", "k0")})
    ctx2.game.current_scenario = _FakeScenario({})       # every lookup returns None
    out2 = _run_one_episode_against(ctx2, confirmed_kills=1)
    assert out2.targets_confirmed_unique == 1 and out2.targets_total == 6
    assert set(out2.known_target_names) == {"<unnamed target>"}


def _integrity_raise(callable_, *args, **kwargs):
    """Call something expected to abort as a measurement-integrity fault; return the error.

    Asserts the whole routing contract in one place: it is an `EpisodeRosterError`, it is
    a `MeasurementIntegrityError` (so both attempt handlers re-raise it ahead of their
    broad `except Exception`), and it is NOT an `EpisodeAttemptError` -- which is what
    keeps it out of the ledger, out of `skip_and_account_v1` and out of every scientific
    denominator.
    """
    try:
        callable_(*args, **kwargs)
    except graph_train.MeasurementIntegrityError as exc:
        assert isinstance(exc, graph_train.EpisodeRosterError), type(exc)
        assert not isinstance(exc, EpisodeAttemptError), type(exc)
        return exc
    raise AssertionError("the call did not abort as a measurement-integrity fault")


def test_a_structural_roster_failure_aborts_as_measurement_integrity() -> None:
    """P2/P3. A broken roster ABORTS -- it is never a measured zero and never a skipped setup.

    Two defects are regressed at once. The older one: each case below was once swallowed
    into an empty roster which, combined with a count derived from classified names,
    reported an episode holding real confirmations as a successful `0/0`. The newer one:
    each was then booked as an accounted `setup` episode failure, which is how the long
    baseline lost 143 training attempts to a measurement defect while reporting itself
    reconciled. A fault in the instrument is not an outcome of the experiment.
    """
    done = {("ego_0", "k0"), ("ego_1", "h0")}

    # 1. no beliefs at all
    no_beliefs = _reference_ctx(done=done)
    no_beliefs.beliefs = {}

    # 2. a belief task that names no target
    malformed = _reference_ctx(done=done)
    malformed.beliefs["ego_0"] = _FakeBelief([_FakeTask(steps=[])])

    # 3. the t=0 beliefs disagree -- a real no-communication defect
    disagree = _reference_ctx(done=done)
    disagree.beliefs["ego_2"] = _FakeBelief([_FakeTask("k0"), _FakeTask("k1")])

    # 4. the executed-world snapshot is absent entirely
    no_world = _reference_ctx(done=done)
    del no_world.executed_target_ids

    # 5. the executed-world snapshot is empty
    empty_world = _reference_ctx(done=done, executed_snapshot=())

    # 6. a known target the executed world does not contain
    uncovered = _reference_ctx(done=done, known_snapshot=("k0", "k1", "gone"))

    # 7. a PLANNED target the known world does not contain -- the egos were assigned
    #    something that does not exist, which is the one direction that is still a defect
    planned_ghost = _reference_ctx(done=done,
                                   belief_ids=("k0", "k1", "not-in-the-world"))

    for label, ctx in (("no beliefs", no_beliefs), ("malformed task", malformed),
                       ("beliefs disagree", disagree), ("no world snapshot", no_world),
                       ("empty world snapshot", empty_world),
                       ("known not executed", uncovered),
                       ("planned not known", planned_ghost)):
        exc = _integrity_raise(_run_one_episode_against, ctx, confirmed_kills=2)
        assert str(exc), label
        # The env is still closed on the aborting path.
        assert ctx.env.closed, label


def test_a_confirmed_id_outside_the_roster_cannot_produce_a_record() -> None:
    """P3. An unaccountable confirmation aborts loudly instead of being dropped or skipped.

    Silently discarding it would leave a block whose printed names no longer sum to its
    own total. Accounting it as a `setup` failure -- what the code used to do -- is worse
    in a different way: the executor confirmed a target the t=0 snapshot never listed, so
    the instrument and the world disagree, and that is not a property of this episode.
    """
    ctx = _reference_ctx(done={("ego_0", "k0"), ("ego_1", "ghost-target")})
    exc = _integrity_raise(_run_one_episode_against, ctx, confirmed_kills=2)
    assert "ghost-target" in str(exc), str(exc)
    assert ctx.env.closed

    # The roster helper says the same thing on its own.
    roster = graph_train._TargetRoster(("k0",), ("A",), ("h0",), ("B",))
    _integrity_raise(roster.confirmed, {"k0", "ghost-target"})


def test_a_setup_failure_contributes_no_false_zero_to_the_aggregates(
    tmp_path: Path,
) -> None:
    """P6. An ACCOUNTED episode failure is a FAILURE, not a zero in the mean.

    Unchanged behaviour, re-asserted so the routing correction cannot be read as having
    loosened it: a genuine `setup` episode failure (an exact-cardinality construction
    refusal, say) is still skipped and accounted, and the authoritative aggregate is still
    the successful episode's real count rather than the average of that count and a
    fabricated 0.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "run", eval_every=0, checkpoint_every=0,
    )
    summary, _, state = _run_stub_training(
        cfg,
        failures={1: ("setup", "stubbed exact-cardinality construction refusal")},
        wakes_per_episode=1, capture_stdout=True,
    )
    out = state["stdout"]

    # No OK block for the failed attempt.
    assert [b[0] for b in _ok_blocks(out)] == ["[train iter=0 ep=0 seed=0] OK"]
    assert "seed=1] OK" not in out

    record = _read_records(cfg.output_dir, "train_records.jsonl")[0]
    assert record["n_attempted"] == 2 and record["n_successful"] == 1
    assert record["n_failed"] == 1
    # THE assertion: the mean is over the one SUCCESSFUL episode, undiluted.
    assert record["targets_confirmed_unique_mean"] == float(_STUB_UNIQUE_CONFIRMED)
    assert record["kills_mean"] == record["targets_confirmed_unique_mean"]
    assert record["targets_confirmed_unique_mean"] != float(_STUB_UNIQUE_CONFIRMED) / 2
    assert summary["train_episodes_failed"] == 1 and summary["accounting_reconciled"]

    ledger = _read_records(cfg.output_dir, "episode_failures.jsonl")
    assert [r["pipeline_stage"] for r in ledger] == ["setup"], ledger


def test_an_episode_outcome_cannot_omit_what_it_measured() -> None:
    """P2. `_EpisodeOutcome` has no roster defaults, so a silent `0/0` is unconstructible.

    The defaults were how an unmeasured episode could look like a measured one.
    """
    try:
        _EpisodeOutcome(
            trajectory=[], reward=-0.5, ticks=1, ended="done", n_wakes=0,
            confirmed_kills=0, n_dead=0, seconds=0.1,
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "_EpisodeOutcome still constructs without stating what it measured"
        )


# =============================================================================
# T13b -- EXECUTED-WORLD ROSTER INTEGRITY (P1-P4, P6)
# =============================================================================
#
# The long baseline exposed one root error: `_episode_target_roster` read
# `ctx.oracle_tasks` (and the beliefs) as the episode's world. Both are ALLOCATIONS --
# `solve_and_normalize` returns an allocated-only task list by contract -- so every
# target the solver left unselected was missing from the roster while still sitting in
# the executed world. The roster then failed the episode over its own under-count, as an
# accounted `setup` failure, and 143 of 800 training attempts vanished while every
# preserved `executed_t0_scenario.json` held the full six-target world.
#
# The tests below fix the two halves separately, because they are two different mistakes:
# WHERE the world comes from (`known_target_ids` / `executed_target_ids`, raw and
# pre-solve) and WHAT a violation means (measurement integrity, which aborts -- not
# episode attrition, which is accounted).


def test_oracle_allocation_is_not_the_world_inventory() -> None:
    """P1. A target the ORACLE did not select is still in the world, and still in the roster.

    The exact shape of the defect: `oracle_tasks` omits a known target, so a roster
    derived from it would report 5 executed targets for a 6-target world -- the `3 known /
    2 hidden / 5 total` that 11 `complete` manifests carried. Here the world snapshot is
    intact, so the roster is 3 + 3 = 6 regardless of what the oracle allocated.
    """
    # The oracle allocated everything EXCEPT known target k1.
    ctx = _reference_ctx(done={("ego_0", "k0")},
                         oracle_ids=("k0", "k2", "h0", "h1", "h2"))
    assert len(ctx.oracle_tasks) == 5, "the fixture did not model a short oracle"

    out = _run_one_episode_against(ctx, confirmed_kills=1)
    assert out.targets_total == 6
    assert len(out.known_target_names) == 3 and len(out.hidden_target_names) == 3
    # k1 is in the roster under its real name, not dropped and not renamed.
    assert out.known_target_names == ("Floridistan AFB #1", "Floridistan AFB #2",
                                      "Floridistan AFB #3")

    # And an unselected HIDDEN target that is then CONFIRMED is a valid episode, and is
    # classified as hidden -- not as an out-of-world confirmation.
    ctx2 = _reference_ctx(done={("ego_0", "h2")},
                          oracle_ids=("k0", "k1", "k2", "h0", "h1"))
    assert "h2" not in {graph_train._task_target_id(t) for t in ctx2.oracle_tasks}
    out2 = _run_one_episode_against(ctx2, confirmed_kills=1)
    assert out2.targets_total == 6
    assert out2.targets_confirmed_unique == 1
    assert out2.hidden_confirmed_names == ("Hidden Airbase #003",)
    assert out2.known_confirmed_names == ()


def test_the_roster_never_reads_the_oracle_task_list() -> None:
    """P1. Structural: `oracle_tasks` is not consulted at all, so it cannot come back.

    An attribute that RAISES on access is the strongest available statement -- it fails
    whether the read is direct, incidental or reintroduced by a later refactor.
    """
    class _ExplodingOracle:
        def __get__(self, obj, owner=None):
            raise AssertionError("_episode_target_roster read ctx.oracle_tasks")

    class _Ctx(_FakeCtx):
        oracle_tasks = _ExplodingOracle()

        def __init__(self, source):
            self.beliefs = source.beliefs
            self.known_target_ids = source.known_target_ids
            self.executed_target_ids = source.executed_target_ids
            self.game = source.game
            self.executor = source.executor
            self.env = source.env

    roster = graph_train._episode_target_roster(_Ctx(_reference_ctx(done=set())))
    assert len(roster.known_ids) == 3 and len(roster.hidden_ids) == 3
    assert roster.total == 6


def test_belief_allocation_is_not_the_known_world_denominator() -> None:
    """P2. Beliefs are an allocated-only SUBSET, checked as one -- never the known count.

    A known target the solver did not allocate is in no belief and in every ego's world.
    The old roster took the belief list AS the known set, so that target became "hidden"
    (or, once the executed side was also allocated-only, vanished). The known count now
    comes from the world snapshot, and the beliefs are constrained the only way they
    truthfully can be: as a subset.
    """
    # Beliefs cover only k0 and k2; the world still holds k0, k1, k2.
    ctx = _reference_ctx(done={("ego_0", "k1")}, belief_ids=("k0", "k2"))
    out = _run_one_episode_against(ctx, confirmed_kills=1)

    assert out.targets_total == 6
    assert len(out.known_target_names) == 3, "the known count came from the beliefs"
    assert len(out.hidden_target_names) == 3, "an unallocated known target became hidden"
    # k1 -- in the world, in no belief -- is confirmed and classified KNOWN.
    assert out.known_confirmed_names == ("Floridistan AFB #2",)
    assert out.hidden_confirmed_names == ()

    # Belief AGREEMENT is still enforced, and a belief naming a target outside the known
    # world is still a real defect in the other direction.
    disagree = _reference_ctx(done=set(), belief_ids=("k0", "k2"))
    disagree.beliefs["ego_1"] = _FakeBelief([_FakeTask("k0")])
    exc = _integrity_raise(graph_train._episode_target_roster, disagree)
    assert "disagree" in str(exc), str(exc)

    outside = _reference_ctx(done=set(), belief_ids=("k0", "off-world"))
    exc2 = _integrity_raise(graph_train._episode_target_roster, outside)
    assert "off-world" in str(exc2), str(exc2)


def test_the_roster_must_match_the_scheduled_cell() -> None:
    """P1/P3. A world that is not the configured cell is a measurement fault, not an episode.

    `setup_episode` already enforces exact cardinality on its own side, so a disagreement
    here means this module measured the world wrongly -- the `3/2/5` case, stated against
    the config rather than against a manifest.
    """
    short = _reference_ctx(done={("ego_0", "k0")}, hidden=2)
    assert len(short.executed_target_ids) == 5
    exc = _integrity_raise(_run_one_episode_against, short, confirmed_kills=1)
    assert "hidden targets: observed 2, scheduled 3" in str(exc), str(exc)
    assert "executed targets: observed 5, scheduled 6" in str(exc), str(exc)

    # The valid cell passes the same gate untouched. The expectation is now the RESOLVED
    # scheduled cell rather than the config -- `_scheduled_cell` is the one site that
    # decides what "hidden" should be, and on the historical path (no construction audit)
    # it resolves to exactly the configured count.
    roster = graph_train._episode_target_roster(_reference_ctx(done=set()))
    cfg = TrainConfig(n_iterations=1, n_known=3, n_hidden=3)
    expected = graph_train._scheduled_cell(
        graph_train.episode_cardinality(cfg, seed=0), None
    )
    assert (expected.n_known, expected.n_hidden, expected.n_targets_executed) == (3, 3, 6)
    assert expected.realized_short is False
    graph_train._require_scheduled_cell(roster, expected)


def test_an_integrity_fault_aborts_the_run_and_writes_no_ledger_row(
    tmp_path: Path,
) -> None:
    """P3. The run STOPS: no ledger row, no denominator shrink, no next scheduled attempt.

    Four scheduled training episodes with the SECOND one faulting. Under the old routing
    the run would have continued through all four and recorded one `setup` failure; under
    the corrected routing it stops at the second and the schedule never reaches the third.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=4, base_seed=0,
        output_dir=tmp_path / "abort", eval_every=0, checkpoint_every=0,
    )
    events: list = []
    raised = None
    try:
        _run_stub_training(cfg, integrity_failures=(1,), wakes_per_episode=1,
                           events=events, capture_stdout=True)
    except BaseException as exc:  # noqa: BLE001 - the abort itself is under test
        raised = exc

    assert isinstance(raised, graph_train.EpisodeRosterError), raised
    assert isinstance(raised, graph_train.MeasurementIntegrityError), raised
    assert not isinstance(raised, EpisodeAttemptError), raised

    # It stopped AT the faulting attempt: seeds 0 and 1 were attempted, 2 and 3 never.
    assert _episode_seeds(events, "train") == [0, 1], events

    # Nothing was booked as science: no ledger row, and no completed record either.
    run_dir = Path(cfg.output_dir)
    assert _read_records(run_dir, "episode_failures.jsonl") == []
    assert _read_records(run_dir, "train_records.jsonl") == []


def test_an_eval_integrity_fault_aborts_the_round(tmp_path: Path) -> None:
    """P3. The EVAL handler re-raises it too -- both handlers, not just the training one.

    The pre-update round runs before any training episode, so a faulting held-out seed
    stops the run before a single training attempt exists.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "abort_eval", eval_every=1, eval_episodes=2,
        eval_base_seed=1_000_000, checkpoint_every=0,
    )
    events: list = []
    raised = None
    try:
        _run_stub_training(cfg, integrity_failures=(1_000_001,), wakes_per_episode=1,
                           events=events, capture_stdout=True)
    except BaseException as exc:  # noqa: BLE001 - the abort itself is under test
        raised = exc

    assert isinstance(raised, graph_train.MeasurementIntegrityError), raised
    assert _episode_seeds(events, "train") == [], "training started after an eval fault"
    run_dir = Path(cfg.output_dir)
    assert _read_records(run_dir, "episode_failures.jsonl") == []
    assert _read_records(run_dir, "eval_records.jsonl") == []


def test_a_post_run_integrity_fault_leaves_the_real_playback_listed(
    tmp_path: Path,
) -> None:
    """P4. A completed episode's recording is in its manifest even when validation then fails.

    THE 17-file finding: those attempts ran `run_episode` to completion, exported a real
    playback, and were then failed on a confirmed id. `finalize` was never reached, so the
    only trace of the file was the file. The recording is now synchronized into the
    manifest the moment the run returns -- while the status is still `incomplete` -- so an
    aborted attempt is truthful about what it holds.
    """
    cfg = _va_cfg(tmp_path, name="ghost", visual_artifacts=True,
                  eval_every=0, eval_episodes=0)
    _summary, _calls, state = _run_training_with_real_episode_body(
        cfg, ghost_confirmation_tags=(0,))

    assert isinstance(state["raised"], graph_train.MeasurementIntegrityError), \
        state["raised"]
    assert "ghost-target" in str(state["raised"])
    # Not science: nothing in the ledger.
    assert _read_records(Path(cfg.output_dir), "episode_failures.jsonl") == []

    bundle = _va_bundles(Path(cfg.output_dir))[0]
    manifest = _va_manifest(bundle)
    assert manifest["status"] == "incomplete"
    real = sorted(p.name for p in bundle.glob("*.jsonl"))
    assert real, "the stub run wrote no playback at all"
    assert manifest["playback_recordings"] == real, manifest["playback_recordings"]
    # The two scenarios captured before the fault are listed too.
    assert manifest["known_only_scenario"] == "known_only_scenario.json"
    assert manifest["executed_t0_scenario"] == "executed_t0_scenario.json"


def test_finalize_refuses_to_certify_a_world_its_files_contradict(
    tmp_path: Path,
) -> None:
    """P4. Expected 3/3/6 against observed 3/2/5 cannot become `complete`.

    Driven directly, because the roster gate now stops such an episode earlier: this
    asserts `finalize`'s OWN refusal, which is the last line of defence if anything ever
    reaches it with mismatched counts again.
    """
    identity = graph_train._AttemptIdentity(
        phase="train", iteration=0, updates_completed=0,
        eval_round_ordinal=None, eval_episode_index=None, eval_pair_member=None,
        attempt_ordinal=0, episode_index=0, seed=0, condition="clean", episode_tag=0,
    )
    bundle = graph_train._AttemptArtifacts(root=tmp_path / "va", identity=identity).open()
    (bundle.directory / "known_only_scenario.json").write_text("{}", encoding="utf-8")
    bundle._known_only = "known_only_scenario.json"
    bundle._executed_t0 = "executed_t0_scenario.json"
    (bundle.directory / "run Recording 000000 - 000100.jsonl").write_text(
        "{}\n", encoding="utf-8")
    assert bundle.sync_recordings() == ("run Recording 000000 - 000100.jsonl",)

    expected = {"n_known": 3, "n_hidden": 3, "n_targets_executed": 6}
    try:
        bundle.finalize(expected=dict(expected),
                        observed={"n_known": 3, "n_hidden": 2,
                                  "n_targets_executed": 5,
                                  "targets_confirmed_unique": 2})
    except graph_train._VisualArtifactError as exc:
        assert "hidden" in str(exc) and "incomplete" in str(exc), str(exc)
    else:
        raise AssertionError("finalize certified a world its own counts contradict")

    manifest = _va_manifest(bundle.directory)
    assert manifest["status"] == "incomplete"
    # The observed counts are RECORDED, not hidden -- and the playback is still listed.
    assert manifest["targets"]["observed"]["n_hidden"] == 2
    assert manifest["targets"]["expected"]["n_hidden"] == 3
    assert manifest["playback_recordings"] == ["run Recording 000000 - 000100.jsonl"]

    # The matching cell is accepted, and lists every chunk.
    (bundle.directory / "run Recording 000100 - 000200.jsonl").write_text(
        "{}\n", encoding="utf-8")
    bundle.sync_recordings()
    bundle.finalize(expected=dict(expected),
                    observed={"n_known": 3, "n_hidden": 3, "n_targets_executed": 6,
                              "targets_confirmed_unique": 2})
    done = _va_manifest(bundle.directory)
    assert done["status"] == "complete"
    assert done["playback_recordings"] == ["run Recording 000000 - 000100.jsonl",
                                           "run Recording 000100 - 000200.jsonl"]


def test_finalize_cannot_complete_without_a_synchronized_recording(
    tmp_path: Path,
) -> None:
    """P4. No playback is ever fabricated: `finalize` refuses rather than inventing one."""
    identity = graph_train._AttemptIdentity(
        phase="train", iteration=0, updates_completed=0,
        eval_round_ordinal=None, eval_episode_index=None, eval_pair_member=None,
        attempt_ordinal=0, episode_index=0, seed=0, condition="clean", episode_tag=0,
    )
    bundle = graph_train._AttemptArtifacts(root=tmp_path / "va", identity=identity).open()
    bundle._known_only = "known_only_scenario.json"
    bundle._executed_t0 = "executed_t0_scenario.json"
    try:
        bundle.finalize(expected={"n_known": 3}, observed={"n_known": 3})
    except graph_train._VisualArtifactError as exc:
        assert graph_train._ARTIFACT_RECORDING_GLOB in str(exc), str(exc)
    else:
        raise AssertionError("finalize completed a bundle with no recording")
    assert _va_manifest(bundle.directory)["status"] == "incomplete"
    assert not list(bundle.directory.glob("*.jsonl")), "a recording was fabricated"

    # And `sync_recordings` refuses too, rather than reporting an empty tuple.
    try:
        bundle.sync_recordings()
    except graph_train._VisualArtifactError as exc:
        assert "no BLADE playback" in str(exc), str(exc)
    else:
        raise AssertionError("sync_recordings accepted an empty directory")


def test_a_valid_episode_is_unchanged_by_the_world_truth_fix() -> None:
    """P6. When the oracle happens to cover the world, every observable is as before.

    The invariance claim, stated over the outcome an attempt actually produces: totals,
    name partitions, the unique-confirmed count, the reward, the trajectory and the
    fuel-damage records are identical whether the oracle allocated the whole world or only
    part of it. The roster is a read-only projection; nothing downstream may move with it.
    """
    done = {("ego_0", "k0"), ("ego_1", "k0"), ("ego_1", "k2"), ("ego_2", "h1")}
    fields = ("targets_total", "targets_confirmed_unique", "known_target_names",
              "hidden_target_names", "known_confirmed_names", "hidden_confirmed_names",
              "reward", "ticks", "ended", "n_wakes", "confirmed_kills", "n_dead",
              "trajectory", "fuel_damage_plan", "fuel_damage_outcome",
              "selected_ego_rtb_issued")

    full = _run_one_episode_against(_reference_ctx(done=done), confirmed_kills=len(done))
    # The SAME world, measured while the oracle allocated only part of it.
    partial = _run_one_episode_against(
        _reference_ctx(done=done, oracle_ids=("k0", "h1")), confirmed_kills=len(done))

    for name in fields:
        assert getattr(full, name) == getattr(partial, name), name

    # And the values are the reference-cell ones, not merely equal to each other.
    assert full.targets_total == 6 and full.targets_confirmed_unique == 3
    assert full.known_confirmed_names == ("Floridistan AFB #1", "Floridistan AFB #3")
    assert full.hidden_confirmed_names == ("Hidden Airbase #002",)
    assert full.confirmed_kills == 4 != full.targets_confirmed_unique


def test_train_and_eval_accounting_are_unchanged_on_the_valid_path(
    tmp_path: Path,
) -> None:
    """P6. A run with no integrity fault produces exactly the accounting it did before.

    The routing change adds a new abort path; it must add nothing to the ordinary one.
    """
    cfg = TrainConfig(
        n_iterations=2, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "valid", eval_every=1, eval_episodes=2,
        eval_base_seed=1_000_000, checkpoint_every=0,
    )
    summary, events, _state = _run_stub_training(cfg, wakes_per_episode=1)

    assert summary["accounting_reconciled"] is True
    assert summary["train_episodes_attempted"] == 4
    assert summary["train_episodes_successful"] == 4
    assert summary["train_episodes_failed"] == 0
    assert _read_records(Path(cfg.output_dir), "episode_failures.jsonl") == []
    assert _episode_seeds(events, "train") == [0, 1, 2, 3]
    # Matched-pair evaluation is untouched: 3 rounds x 2 members = 6 attempts per
    # held-out seed, with both forced modes present in every round.
    eval_seeds = _episode_seeds(events, "eval")
    assert eval_seeds.count(1_000_000) == 6 and eval_seeds.count(1_000_001) == 6
    assert len(eval_seeds) == 12


# =============================================================================
# T14 -- P3: every eval round keeps its own scenario artifacts
# =============================================================================

def test_eval_episode_tag_is_deterministic_and_round_disjoint() -> None:
    """P3. Round ordinal r owns [base + r*stride, base + (r+1)*stride), and only that."""
    stride = graph_train._EVAL_ROUND_TAG_STRIDE
    base = graph_train._EVAL_EPISODE_TAG_BASE

    assert graph_train.eval_episode_tag(round_ordinal=0, e=0) == base
    assert graph_train.eval_episode_tag(round_ordinal=0, e=3) == base + 3
    assert graph_train.eval_episode_tag(round_ordinal=2, e=1) == base + 2 * stride + 1

    # Deterministic: the same (round, e) is always the same tag.
    assert (graph_train.eval_episode_tag(round_ordinal=5, e=7)
            == graph_train.eval_episode_tag(round_ordinal=5, e=7))

    # No two rounds share a tag over any plausible eval band.
    bands = [
        {graph_train.eval_episode_tag(round_ordinal=r, e=e) for e in range(8)}
        for r in range(6)
    ]
    for i, first in enumerate(bands):
        for second in bands[i + 1:]:
            assert not (first & second), (first & second)

    # Out-of-band indices RAISE rather than silently reach into the next round.
    for bad in (stride, stride + 1, -1):
        try:
            graph_train.eval_episode_tag(round_ordinal=0, e=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("eval_episode_tag accepted e=%r" % (bad,))
    try:
        graph_train.eval_episode_tag(round_ordinal=-1, e=0)
    except ValueError:
        pass
    else:
        raise AssertionError("eval_episode_tag accepted a negative round_ordinal")


def test_validate_refuses_an_eval_band_wider_than_one_tag_namespace() -> None:
    """P3. A config whose rounds could collide by FILENAME is refused up front."""
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=1, base_seed=0,
        eval_every=1, eval_episodes=graph_train._EVAL_ROUND_TAG_STRIDE + 1,
        eval_base_seed=1_000_000,
    )
    try:
        cfg.validate()
    except ValueError as exc:
        assert "scenario-tag namespace" in str(exc), str(exc)
    else:
        raise AssertionError("validate() accepted an eval band wider than one namespace")


def test_eval_rounds_reuse_the_seeds_but_not_the_tags(tmp_path: Path) -> None:
    """P3. Same held-out seeds every round; a fresh tag namespace every round.

    Both halves matter and they pull in opposite directions. Fixing the seeds is what
    makes round-to-round differences attributable to the POLICY; fixing the tags is what
    destroyed the earlier rounds' scenario JSONs, because the tag names the file.
    """
    cfg = TrainConfig(
        n_iterations=2, episodes_per_iteration=1, base_seed=0,
        output_dir=tmp_path / "run",
        eval_every=1, eval_episodes=2, eval_base_seed=1_000_000,
        checkpoint_every=0,
    )
    _, events, state = _run_stub_training(cfg, wakes_per_episode=1)

    eval_seeds = _episode_seeds(events, "eval")
    eval_tags = _episode_tags(events, "eval")
    train_tags = _episode_tags(events, "train")

    # 3 rounds (pre_update + 2 post_update) x 2 held-out seeds x 2 matched pair
    # members, the SAME seeds each round.
    assert eval_seeds == [1_000_000, 1_000_000, 1_000_001, 1_000_001] * 3, eval_seeds
    # ... and 12 DISTINCT tags (each member of each pair of each round gets its own).
    assert len(set(eval_tags)) == len(eval_tags) == 12, eval_tags

    rounds = [set(eval_tags[i:i + 4]) for i in range(0, 12, 4)]
    pre_update, post_1, post_2 = rounds
    assert not (pre_update & post_1) and not (pre_update & post_2)
    assert not (post_1 & post_2)
    assert not (set(train_tags) & set(eval_tags)), (train_tags, eval_tags)

    # The eval records say which namespace each round wrote into.
    recs = _read_records(cfg.output_dir, "eval_records.jsonl")
    assert [r["eval_round_ordinal"] for r in recs] == [0, 1, 2], recs
    assert [r["episode_tag_start"] for r in recs] == [
        eval_member_tag(round_ordinal=r, e=0, member=0) for r in range(3)
    ], recs
    assert recs[0]["evaluation_stage"] == _EVAL_STAGE_PRE_UPDATE
    assert all(r["evaluation_stage"] == _EVAL_STAGE_POST_UPDATE for r in recs[1:])


def test_pre_and_post_update_scenario_files_coexist(tmp_path: Path) -> None:
    """P3. Every round's scenario artifact is still on disk when the run finishes.

    The stub episode body writes the file `ScenarioGenerator.generate` would have written
    for its tag (`episode_%04d_scenario.json`), so the FILENAME consequence of the tag
    namespace is testable without BLADE or a solver. Before this change every round wrote
    `episode_900000_scenario.json` and only the last round's world survived the run.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=1, base_seed=0,
        output_dir=tmp_path / "run",
        eval_every=1, eval_episodes=2, eval_base_seed=1_000_000,
        checkpoint_every=0,
    )
    _run_stub_training(cfg, wakes_per_episode=1)

    scen_dir = Path(cfg.output_dir) / "scenarios"
    names = sorted(p.name for p in scen_dir.glob("episode_*_scenario.json"))

    def _expected(round_ordinal):
        return {
            "episode_%04d_scenario.json"
            % eval_member_tag(round_ordinal=round_ordinal, e=e, member=m)
            for e in range(cfg.eval_episodes) for m in (0, 1)
        }

    pre_update, post_update = _expected(0), _expected(1)
    assert pre_update <= set(names), (pre_update, names)
    assert post_update <= set(names), (post_update, names)
    assert not (pre_update & post_update)
    # 4 pre-update + 1 training + 4 post-update, all present SIMULTANEOUSLY: both
    # members of every matched pair keep their own world.
    assert len(names) == 9, names

    # Each file still records the seed it was written for -- the seeds really are reused,
    # and BOTH members of a pair record the SAME one (that is what makes it a pair).
    for round_files in (pre_update, post_update):
        seeds = sorted(
            json.loads((scen_dir / n).read_text(encoding="utf-8"))["seed"]
            for n in round_files
        )
        assert seeds == [1_000_000, 1_000_000, 1_000_001, 1_000_001], seeds


# =============================================================================
# T11b -- provenance is a precondition, not a log line
# =============================================================================

def test_provenance_is_collected_before_any_run_artifact_exists(tmp_path: Path) -> None:
    """PO2. The tree is inspected BEFORE this run creates its own files.

    `output_dir` may legitimately point inside the repository. Anything the run creates
    there is untracked, so collecting provenance after `mkdir` would let the run's own
    scenario files and ledger be reported as pre-existing DIRTY SOURCE STATE -- a run
    contaminating the record of what produced it.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=1, base_seed=0,
        output_dir=tmp_path / "run", eval_every=0, checkpoint_every=0,
    )
    seen: dict = {}
    real_collect = graph_train.collect_provenance

    def spy(cfg_, **kwargs):
        run_dir = Path(cfg_.output_dir)
        seen["existed"] = run_dir.exists()
        seen["contents"] = (
            sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else []
        )
        return real_collect(cfg_, **kwargs)

    graph_train.collect_provenance = spy
    try:
        _run_stub_training(cfg)
    finally:
        graph_train.collect_provenance = real_collect

    assert seen["existed"] is False, (
        "the run directory already existed when provenance was collected: %r"
        % (seen["contents"],)
    )
    assert seen["contents"] == []
    # The run really did produce those artifacts afterwards -- the check above is about
    # ORDER, not about the artifacts being absent altogether.
    assert (Path(cfg.output_dir) / "run_config.json").exists()
    assert (Path(cfg.output_dir) / "episode_failures.jsonl").exists()


def test_git_provenance_requires_both_the_sha_and_the_dirty_state(tmp_path: Path) -> None:
    """PO2. A recovered SHA with an UNKNOWN clean/dirty verdict is not available.

    A commit alone names a revision the run may or may not have executed; without the
    dirty verdict it cannot be said that the run used it. So the failure mode being
    locked out is `available=True` alongside `dirty=None`.

    Driven by replacing the probe transport, so the outcome is chosen here rather than
    inherited from the developer checkout.
    """
    real_probe = graph_train._probe_command

    def fake_probe(args, *, timeout, cwd=None):
        if args[1:] == ["rev-parse", "HEAD"]:
            return 0, "c" * 40 + "\n", ""
        if args[1:] == ["status", "--porcelain"]:
            return 128, "", "fatal: unable to read index\n"
        return 0, "stub\n", ""

    graph_train._probe_command = fake_probe
    try:
        info = _git_provenance(tmp_path)
    finally:
        graph_train._probe_command = real_probe

    assert info["available"] is False, info
    assert info["dirty"] is None, info
    assert info["commit"] == "c" * 40, "the recovered SHA should still be reported"
    assert "unable to read index" in info["reason"], info

    # A status probe that RAISES (timeout / no git) is the same verdict, not a crash.
    def raising_probe(args, *, timeout, cwd=None):
        if args[1:] == ["rev-parse", "HEAD"]:
            return 0, "d" * 40 + "\n", ""
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    graph_train._probe_command = raising_probe
    try:
        info = _git_provenance(tmp_path)
    finally:
        graph_train._probe_command = real_probe
    assert info["available"] is False and info["dirty"] is None, info
    assert info["reason"], info


def test_training_stops_when_git_provenance_is_incomplete(tmp_path: Path) -> None:
    """PO2. Incomplete provenance refuses the run before any policy/generator/episode.

    Compute is not the point -- attributability is. A run that cannot name the code that
    produced it yields records nobody can tie to a revision, so it must not start rather
    than finish and be discovered unusable. The attempted `run_config.json` is still
    written, so the refusal itself is inspectable.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=2, base_seed=0,
        output_dir=tmp_path / "run", eval_every=2, eval_episodes=2,
        eval_base_seed=1_000_000, checkpoint_every=0,
    )
    events: list = []
    try:
        _run_stub_training(cfg, git=_FAKE_GIT_INCOMPLETE, events=events)
    except RuntimeError as exc:
        assert "provenance" in str(exc), str(exc)
        assert "run_config.json" in str(exc), str(exc)
    else:
        raise AssertionError("train() ran with incomplete Git provenance")

    # Nothing expensive happened: no policy, no generator, no episode, no update.
    assert events == [], events

    run_dir = Path(cfg.output_dir)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert payload["provenance"]["git"]["available"] is False
    assert payload["provenance"]["git"]["dirty"] is None
    # The refused attempt did not write, truncate or fabricate any record stream.
    for name in ("train_records.jsonl", "eval_records.jsonl",
                 "episode_failures.jsonl", "run_summary.json"):
        assert not (run_dir / name).exists(), name


def test_a_dirty_tree_warns_but_still_runs(tmp_path: Path) -> None:
    """A dirty tree is a hazard a researcher may choose -- warned about, never blocked.

    The counterpart to the refusal above: complete provenance that happens to say
    "dirty" is still complete. Hiding or normalizing it would be the real failure.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=1, base_seed=0,
        output_dir=tmp_path / "run", eval_every=0, checkpoint_every=0,
    )
    dirty = dict(_FAKE_GIT_OK, dirty=True, dirty_path_count=3)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary, _, _ = _run_stub_training(cfg, git=dirty)
    out = buf.getvalue()

    assert "[WARN]" in out and "DIRTY" in out, out
    assert summary["train_episodes_attempted"] == 1
    assert (Path(cfg.output_dir) / "run_summary.json").exists()


# =============================================================================
# T12 -- derived artifacts: run_summary.json and the 4-panel plot
# =============================================================================

def _read_records(run_dir, name: str) -> list:
    """Read one jsonl artifact out of a run directory."""
    path = Path(run_dir) / name
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_failure_ledger(run_dir: Path, entries) -> None:
    """Write a synthetic `episode_failures.jsonl` from (phase, stage, error) triples."""
    with open(run_dir / "episode_failures.jsonl", "w", encoding="utf-8") as fh:
        for i, (phase, stage, error) in enumerate(entries):
            fh.write(json.dumps({
                "phase": phase, "evaluation_stage": None, "updates_completed": 0,
                "iteration": 0, "attempt_ordinal": i, "episode_index": i,
                "eval_tag": None, "seed": 1000 + i, "pipeline_stage": stage,
                "error_type": error, "error_message": "synthetic",
                "traceback": "synthetic traceback",
            }) + "\n")


def test_run_summary_is_derived_from_the_jsonl_records(tmp_path: Path) -> None:
    """PO3. The summary is a pure function of the recorded files -- no second path.

    Built from fixtures written by hand, with no training involved at all: if the
    summary could only be produced by a live run, then "the summary agrees with the
    records" would be an assumption rather than a checked property.
    """
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, all_failed_iteration=True)
    _write_failure_ledger(run_dir, [
        ("train", "setup", "RuntimeError"),
        ("train", "setup", "RuntimeError"),
        ("train", "setup", "RuntimeError"),
        ("train", "setup", "RuntimeError"),
        ("eval", "generation", "ValueError"),
        ("eval", "run", "KeyError"),
        ("eval", "run", "KeyError"),
        ("eval", "run", "KeyError"),
    ])

    summary = build_run_summary(run_dir)

    # 6 iterations x 4 attempts; iteration 3 lost all four.
    assert summary["train_episodes_attempted"] == 24
    assert summary["train_episodes_successful"] == 20
    assert summary["train_episodes_failed"] == 4
    assert summary["train_success_fraction"] == 20 / 24
    # 4 eval rounds x 4 attempts, 1 failure each.
    assert summary["eval_episodes_attempted"] == 16
    assert summary["eval_episodes_successful"] == 12
    assert summary["eval_episodes_failed"] == 4
    assert summary["n_eval_rounds"] == 4

    assert summary["failures_by_phase"] == {"train": 4, "eval": 4}
    assert summary["failures_by_pipeline_stage"] == \
        {"setup": 4, "generation": 1, "run": 3}
    assert summary["failures_by_error_type"] == \
        {"RuntimeError": 4, "ValueError": 1, "KeyError": 3}
    assert summary["accounting_reconciled"] is True

    # The pre-update result is exposed AS SUCH, with its denominator attached.
    pre = summary["initial_pre_update_eval"]
    assert pre["evaluation_stage"] == _EVAL_STAGE_PRE_UPDATE
    assert pre["updates_completed"] == 0 and pre["iteration"] is None
    assert pre["n_attempted"] == 4 and pre["n_successful"] == 3
    assert summary["final_eval"]["updates_completed"] == 6

    # The all-failed iteration is EXCLUDED from the reward aggregates, and counted.
    assert summary["n_iterations_without_reward"] == 1
    # ... and classified as all-failed, not as zero-wake. Same distinction as the live
    # trainer makes, proven here on a derived summary built from files alone.
    assert summary["n_all_failed_iterations"] == 1
    assert summary["n_zero_wake_iterations"] == 0
    assert summary["n_productive_iterations"] == 5
    assert (summary["n_all_failed_iterations"] + summary["n_zero_wake_iterations"]
            + summary["n_productive_iterations"]) == summary["n_iterations"] == 6
    assert summary["train_reward_first"] == -0.9
    assert abs(summary["train_reward_last"] - (-0.4)) < 1e-9
    assert summary["aggregates_over"] == "successful_episodes"
    assert summary["total_transitions"] == 10 + 11 + 12 + 0 + 14 + 15

    for key in ("run_dir", "train_records_path", "eval_records_path",
                "failures_path", "run_config_path", "run_summary_path", "plots_dir"):
        assert summary[key], key
    # Every figure is discoverable by NAME, and the legacy single-path key survives as a
    # documented ALIAS of the performance figure rather than pointing at a dead file.
    assert set(summary["plot_paths"]) == set(_PLOT_FILENAMES)
    for name, path in summary["plot_paths"].items():
        assert path.endswith(name) and _PLOTS_DIRNAME in path
    assert summary["plot_path"] == summary["plot_paths"][_PLOT_PERFORMANCE]


def test_run_summary_flags_a_ledger_that_disagrees(tmp_path: Path) -> None:
    """A lost or double-counted failure is REPORTED, not silently smoothed over.

    Two independent counts of the same thing -- the per-record `n_failed` fields and the
    ledger -- are only useful if a disagreement is surfaced. Negative control for
    `accounting_reconciled` in the test above.
    """
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, all_failed_iteration=True)
    _write_failure_ledger(run_dir, [("train", "setup", "RuntimeError")])  # 1, not 4+4

    summary = build_run_summary(run_dir)
    assert summary["train_episodes_failed"] == 4      # what the records say
    assert summary["failures_by_phase"] == {"train": 1}   # what the ledger says
    assert summary["accounting_reconciled"] is False


def test_run_summary_json_omits_the_embedded_record_lists(tmp_path: Path) -> None:
    """The persisted summary is a summary: the jsonl files stay the single record."""
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    summary = build_run_summary(run_dir)
    assert "train_records" in summary and "eval_records" in summary   # in process

    payload = json.loads(
        write_run_summary(run_dir, summary).read_text(encoding="utf-8")
    )
    for key in ("train_records", "eval_records", "failure_records"):
        assert key not in payload, key
    assert payload["train_episodes_attempted"] == summary["train_episodes_attempted"]


def test_xy_drops_missing_rewards_and_anchors_pre_update_at_zero() -> None:
    """PO3. The plot's data path skips nulls and puts the pre-update point at x=0.

    Asserted on the pure series builder rather than on pixels: what must never happen is
    an all-failed batch being DRAWN, and at x=0 a missing point and a perfect score
    would be visually indistinguishable.
    """
    train = [
        {"updates_completed_before": 0, "train_reward_mean": -0.9},
        {"updates_completed_before": 1, "train_reward_mean": None},   # all failed
        {"updates_completed_before": 1, "train_reward_mean": -0.7},
    ]
    xs, ys = _xy(train, "updates_completed_before", "train_reward_mean")
    assert xs == [0.0, 1.0] and ys == [-0.9, -0.7]
    assert 0.0 not in ys, "a missing reward was drawn at the oracle optimum"

    ev = [
        {"updates_completed": 0, "evaluation_stage": _EVAL_STAGE_PRE_UPDATE,
         "eval_reward_mean": -0.8},
        {"updates_completed": 4, "evaluation_stage": _EVAL_STAGE_POST_UPDATE,
         "eval_reward_mean": None},
        {"updates_completed": 6, "evaluation_stage": _EVAL_STAGE_POST_UPDATE,
         "eval_reward_mean": -0.4},
    ]
    ex, ey = _xy(ev, "updates_completed", "eval_reward_mean")
    assert ex == [0.0, 6.0] and ey == [-0.8, -0.4]
    assert ex[0] == 0.0, "the pre-update held-out point must anchor the curve at x=0"

    # Pre-B4 records have no `updates_completed*` key and fall back to `iteration`.
    legacy = [{"iteration": 2, "baseline": -0.5}]
    assert _xy(legacy, "updates_completed_before", "baseline") == ([2.0], [-0.5])


def test_plot_separates_performance_from_diagnostics_and_health() -> None:
    """THREE figures, each built at its own site, all on the same x quantity.

    Structural rather than pixel-based: a PNG cannot be asked how many axes it has, and
    the facts worth locking -- that performance, diagnostics and denominators are three
    SEPARATE files, and that every one of them labels the shared x-axis -- are visible
    at the construction sites. Each figure's data is proven separately, from jsonl.
    """
    source = Path(inspect.getsourcefile(graph_train)).read_text(encoding="utf-8")
    # Exactly three figures are saved: one per claim, no fourth artifact.
    assert source.count("fig.savefig(") == 3
    assert "plt.subplots(3, 1" in inspect.getsource(
        graph_train._plot_training_performance), "performance is not 3 panels"
    # Diagnostics gained the FD-wake severity-response panel (mix / entropy / response);
    # it stayed INSIDE the diagnostics figure rather than becoming a fourth file.
    assert "plt.subplots(3, 1" in inspect.getsource(
        graph_train._plot_policy_diagnostics), "diagnostics is not 3 panels"
    # MEASUREMENT HEALTH is 3 panels for a fixed-cell run and gains a FOURTH -- the
    # GENERALIZED-V1 requested-vs-realized hidden load -- only when the run has a
    # cardinality to show. It stayed INSIDE the health figure rather than becoming a
    # fourth FILE, for the same reason the severity-response panel stayed inside
    # diagnostics: requested-vs-realized is a DENOMINATOR question, and the denominators
    # are read together.
    health_src = inspect.getsource(graph_train._plot_measurement_health)
    assert "n_panels = 4 if cardinality_rows else 3" in health_src, (
        "health must default to 3 panels and only conditionally add the fourth"
    )
    assert "plt.subplots(\n        n_panels, 1" in health_src, "health is not n_panels"
    # The retired single dashboard is gone as an OUTPUT (its name survives only in the
    # prose explaining why).
    assert 'savefig(out_path' in source
    assert '"training_plot.png"' not in source
    # One x-axis quantity, stamped on every figure.
    assert source.count("ax.set_xlabel(_PLOT_X_LABEL)") == 3
    assert source.count("_annotate_x_semantics(fig)") == 3
    # The honest placement survives: training at updates_completed_before, eval at
    # updates_completed. Neither was moved to make a curve look better.
    assert '_xy(\n        train_records, "updates_completed_before", "train_reward_mean"\n    )' in source
    assert '_xy(eval_records, "updates_completed", "eval_reward_mean_%s" % cell)' in source


def test_performance_plot_draws_every_cell_and_delta_distinctly() -> None:
    """PO2. The held-out panel is PER CELL, and the deltas are the matched fields.

    The series must come from different record fields, one per reporting cell, and the
    deltas from the declared matched-difference keys. The specific thing forbidden here
    is the retired dashboard's behaviour: drawing the pooled `eval_reward_mean` as THE
    held-out signal, which averages across the very factor the matched groups exist to
    isolate.

    The cell list is read from the RECORDS (`_record_cells`), so the same function draws
    a legacy clean/damaged pair round and a clean/mild/severe triad round without being
    told which it is -- `--plot <run_dir>` has no TrainConfig to ask.
    """
    source = inspect.getsource(graph_train._plot_training_performance)
    assert "_record_cells(eval_records)" in source
    assert "_record_delta_keys(eval_records)" in source
    assert '"eval_reward_mean_%s" % cell' in source
    # Training reward and the held-out series are not the same panel: three axes, and
    # the train series is drawn on the first.
    assert "plt.subplots(3, 1" in source
    assert (source.index('"train_reward_mean"')
            < source.index('"eval_reward_mean_%s" % cell'))
    # The pooled series appears ONLY inside the legacy fallback branch -- i.e. after the
    # per-cell series have been found empty -- and is labelled as pooled.
    assert "if not drew_any:" in source
    assert source.index("if not drew_any:") < source.index('"eval_reward_mean")')
    assert "POOLED" in source


def test_measurement_health_keeps_every_denominator() -> None:
    """PO2. Coverage moved to its own figure, and NOTHING was dropped on the way.

    The fractions the packet requires, plus the matched-GROUP fraction that the
    episode-level one cannot express: two surviving members of two different groups are
    two successful episodes and ZERO complete groups.
    """
    source = inspect.getsource(graph_train._plot_measurement_health)
    for key in ("success_fraction", "wake_fraction_of_successful",
                "group_success_fraction"):
        assert '"%s"' % key in source, key
    # Both record streams are read, so "eval success_fraction" is really eval's.
    assert "(train_records, \"updates_completed_before\", \"success_fraction\"" in source
    assert "(eval_records, \"updates_completed\", \"success_fraction\"" in source
    # It says what it is: a health figure, explicitly not a performance claim.
    assert "MEASUREMENT HEALTH" in source and "NOT performance" in source


def test_plot_renders_every_figure_from_jsonl(tmp_path: Path) -> None:
    """PO3. All three figures, including an all-failed iteration, render from records."""
    if _skip_without_matplotlib():
        return
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, all_failed_iteration=True)
    out = plot_training_subprocess(run_dir)
    assert [path.name for path in out] == list(_PLOT_FILENAMES)
    for path in out:
        assert path.exists() and path.stat().st_size > 1000, path


def test_plot_still_renders_pre_b4_records(tmp_path: Path) -> None:
    """A run started before this change is still a run -- its records still plot.

    Pre-B4 records carry no per-condition eval means at all, so the held-out panel falls
    back to the pooled series. That path must still produce all three figures.
    """
    if _skip_without_matplotlib():
        return
    run_dir = tmp_path / "run_legacy"
    _write_synthetic_run(run_dir, legacy=True)
    out = plot_training_subprocess(run_dir)
    assert [path.name for path in out] == list(_PLOT_FILENAMES)
    assert all(path.exists() for path in out)


# =============================================================================
# T15 -- FINAL-CELL VISUAL ARTIFACTS (opt-in inspection bundles)
# =============================================================================
#
# THREE PROOF OBLIGATIONS, and they pull in different directions on purpose:
#
#   PO1  default-OFF invariance      : the feature is invisible unless asked for. No
#                                      directory, no identity, no scenario copy, no
#                                      `Game.export_scenario()`, and `setup_episode` is
#                                      called WITHOUT the recording keyword -- exactly as
#                                      the pre-feature trainer called it.
#   PO2  snapshot fidelity + identity: an enabled successful attempt preserves the
#                                      generator's known-only file BYTE for BYTE, the
#                                      object the authoritative env-2 game exported BEFORE
#                                      the controller and the run, the recording the tick
#                                      loop emitted, and a complete manifest that places
#                                      the attempt without reference to console order.
#                                      Two members of one held-out seed, and the same seed
#                                      across two rounds, get DISJOINT bundles.
#   PO3  isolation + failure honesty : capture adds writes and one read-only export and
#                                      nothing else -- policy inputs, result, reward,
#                                      trajectory, buffer content, seeds, tags and
#                                      condition accounting are identical to the disabled
#                                      run. A normal EPISODE failure stays in the existing
#                                      stage taxonomy and leaves an `incomplete` bundle; an
#                                      ARTIFACT failure aborts loudly through
#                                      `_VisualArtifactError` and never reaches the ledger.
#
# The BLADE / solver seams are stubbed as everywhere else in this file; everything between
# them -- the bundle lifecycle, the identity, the manifest, the ordering of the export
# against the controller, the re-raise ahead of the broad handlers -- is production code.

_VA_KNOWN_TARGET_NAMES = ("Floridistan AFB #1", "Floridistan AFB #2",
                          "Floridistan AFB #3")
_VA_HIDDEN_TARGET_NAMES = ("Hidden Airbase #001", "Hidden Airbase #002",
                           "Hidden Airbase #003")


def _va_known_only_bytes(tag: int, seed: int) -> bytes:
    """The exact bytes the generator "wrote" for one attempt: a 3-target known-only world.

    Deliberately IRREGULAR (odd spacing, hand-ordered keys, a trailing newline): a copy
    that went through `json.load` + `json.dump` would come back normalized, so byte
    equality is a real test of "copied, not reserialized".
    """
    airbases = ", ".join('{"name": "%s"}' % n for n in _VA_KNOWN_TARGET_NAMES)
    return (
        '{"currentScenario": {"name": "episode_%04d",   "airbases": [ %s ]},'
        '  "tag": %d,   "seed": %d}\n' % (int(tag), airbases, int(tag), int(seed))
    ).encode("utf-8")


def _va_executed_t0_object(tag: int, seed: int) -> dict:
    """What the AUTHORITATIVE env-2 game exports at t=0: the 6-target executed world."""
    return {
        "currentScenario": {
            "name": "episode_%04d" % int(tag),
            "airbases": [{"name": n} for n in
                         _VA_KNOWN_TARGET_NAMES + _VA_HIDDEN_TARGET_NAMES],
        },
        "currentSideId": "side-blue",
        "selectedUnitId": "",
        "mapView": {"defaultCenter": [0.0, 0.0], "currentCameraZoom": 5},
        "tag": int(tag),
        "seed": int(seed),
    }


class _VaFuelDamagePlan:
    def __init__(self, condition):
        self._condition = condition

    def to_record(self):
        return {"condition": self._condition, "ego_id": None}


class _VaFuelDamageOutcome:
    rtb_command_issued = None

    def __init__(self, condition):
        self._condition = condition

    def to_record(self):
        return {"condition": self._condition, "fired": False,
                "wake_occurred": False, "wake_meta_action": None}


class _VaFuelDamageController:
    """A clean, inert controller -- the FD mechanism is not what these tests measure."""

    def __init__(self, condition):
        self.plan = _VaFuelDamagePlan(condition)
        self.outcome = _VaFuelDamageOutcome(condition)


def _unrealized_certified_controller() -> FuelDamageController:
    """A REAL certified DAMAGED controller whose event never fired.

    Built through the PURE certification path -- `certify_fd_candidate` then
    `plan_fuel_damage` -- so the exception the routing tests below observe is the one the
    PRODUCTION terminal invariant really raises
    (`FuelDamageController.require_certified_event_realized`, called at `run_episode`'s
    episode-exit seam), not a hand-made stand-in that could drift from it.
    """
    params = FuelDamageParameters(
        mode=FuelDamageMode.FORCED_SEVERE,
        eligibility_policy=FD_ELIGIBILITY_CERTIFIED_V1,
    )
    certificate, rejection = certify_fd_candidate(
        ego_id="ego0",
        launch_point=Location(32.0, 35.0),
        route_points=[Location(34.0, 35.0)],
        home_base=Location(32.0, 35.0),
        speed_knots=1303.0, fuel_rate=6700.0, fuel_at_launch=12000.0,
        params=params, detection_km=50.0,
        world_targets=[("t0", Location(34.0, 35.0))],
        belief_target_ids=["t0"],
    )
    assert rejection is None, "the fixture world must certify: %r" % (rejection,)
    plan = plan_fuel_damage(
        condition=CONDITION_DAMAGED, mode=params.mode, derived_seed=0,
        eligible_ego_ids=("ego0",), ego_id="ego0",
        launch_point=Location(32.0, 35.0), home_base=Location(32.0, 35.0),
        route_points=[Location(34.0, 35.0)],
        speed_knots=1303.0, fuel_rate=6700.0, max_fuel=12000.0,
        fuel_at_launch=12000.0, params=params,
        severity="severe", certificate=certificate,
    )
    controller = FuelDamageController(plan)
    assert controller.outcome.fired is False
    return controller


def _run_training_with_real_episode_body(
    cfg: TrainConfig,
    *,
    setup_error_seeds=(),
    export_error_tags=(),
    ghost_confirmation_tags=(),
    fd_integrity_seeds=(),
    emit_recording: bool = True,
):
    """Drive the REAL `train()` AND the REAL `_run_one_episode` over stubbed engine seams.

    `_run_stub_training` replaces `_run_one_episode` wholesale, which is right for the
    schedule and the ledger but reaches none of the artifact code. Here the four seams
    that genuinely need BLADE / bonmin are replaced instead:

      * `_build_generator`           -> writes a real known-only scenario file per tag;
      * `setup_episode`              -> a fake context; RECORDS the recording keyword it
                                        was (or was not) given;
      * `run_episode`                -> writes a playback `.jsonl` into that path, exactly
                                        as the locked tick-loop contract does on a
                                        completed run;
      * `compute_episode_reward` /
        `build_fuel_damage_controller`-> inert.

    Everything else -- the bundle lifecycle, the manifest, the ordering, the failure
    routing -- is production code.

    `setup_error_seeds` makes those seeds raise inside setup (a NORMAL episode failure).
    `export_error_tags` makes `Game.export_scenario()` raise for those tags (an ARTIFACT
    failure). `fd_integrity_seeds` makes `run_episode` raise the GENERALIZED-V1
    `FuelDamageIntegrityError` -- a world CERTIFIED FD-capable that then failed to deliver
    its own certificate, which is an INSTRUMENT fault and must abort exactly as a
    `MeasurementIntegrityError` does. The exception comes from the REAL controller
    predicate (`_unrealized_certified_controller`), not from a stub, so the routing is
    tested against what production actually raises. `emit_recording=False` models a run
    that produced no playback file.

    Returns ``(summary, calls, state)``; `calls` is the ordered per-attempt log.
    """
    calls: list = []
    state: dict = {"scen_dir": None, "raised": None}

    saved = {
        "_build_generator": graph_train._build_generator,
        "setup_episode": graph_train.setup_episode,
        "run_episode": graph_train.run_episode,
        "compute_episode_reward": graph_train.compute_episode_reward,
        "build_fuel_damage_controller": graph_train.build_fuel_damage_controller,
        "PPOUpdater": graph_train.PPOUpdater,
        "build_policy": graph_train.build_policy,
        "_git_provenance": graph_train._git_provenance,
    }

    class _VaGenerator:
        def generate(self, episode, config):
            path = state["scen_dir"] / ("episode_%04d_scenario.json" % int(episode))
            path.write_bytes(_va_known_only_bytes(int(episode), int(config.seed)))
            calls.append({"kind": "generate", "tag": int(episode),
                          "seed": int(config.seed)})
            return path

    def fake_build_generator(scen_dir):
        state["scen_dir"] = Path(scen_dir)
        state["scen_dir"].mkdir(parents=True, exist_ok=True)
        return _VaGenerator()

    def fake_setup_episode(scenario_json, **kwargs):
        # The tag / seed the generator just recorded identify this attempt.
        tag = calls[-1]["tag"]
        seed = calls[-1]["seed"]
        recording_path = kwargs.get("recording_export_path", "<absent>")
        calls.append({"kind": "setup", "tag": tag, "seed": seed,
                      "recording_kwarg_present": "recording_export_path" in kwargs,
                      "recording_export_path": recording_path})
        if seed in set(setup_error_seeds):
            raise RuntimeError("stubbed exact-cardinality failure at seed %d" % seed)
        export_error = (ValueError("stubbed export failure")
                        if tag in set(export_error_tags) else None)
        # `ghost_confirmation_tags` puts a target OUTSIDE the executed world into the
        # executor's confirmed set. The episode still runs to completion and still writes
        # its playback -- the fault only surfaces in the post-run reconciliation, which is
        # exactly the shape of the long baseline's 17 "completed then failed" attempts.
        done = {("ego_0", "k0"), ("ego_1", "h1")}
        if tag in set(ghost_confirmation_tags):
            done.add(("ego_2", "ghost-target"))
        ctx = _reference_ctx(done=done,
                             exported=_va_executed_t0_object(tag, seed),
                             export_error=export_error)
        ctx.recording_export_path = (
            None if recording_path == "<absent>" else recording_path)
        ctx.tag = tag
        ctx.seed = seed
        return ctx

    def fake_run_episode(policy, ctx, **kwargs):
        # The locked tick-loop contract: a COMPLETED run exports the playback into the
        # armed export path (and none at all when recording was never armed).
        calls.append({"kind": "run", "tag": ctx.tag,
                      "export_calls_before_run": ctx.game.export_calls})
        if int(getattr(ctx, "seed", -1)) in set(int(x) for x in fd_integrity_seeds):
            # Raised from INSIDE `run_episode`, which is where the production controller
            # raises it -- both from `maybe_apply` (a live contradiction) and from the
            # episode-exit seam (a certificate that never materialized). The exception is
            # produced by the REAL predicate on a REAL certified damaged controller, so
            # these routing tests cannot pass against a stand-in the production path has
            # drifted away from. The point under test is that `_run_one_episode` does not
            # wrap it as `EpisodeAttemptError("run", ...)` on the way out.
            _unrealized_certified_controller().require_certified_event_realized()
            raise AssertionError(  # pragma: no cover - the line above always raises
                "the terminal invariant did not raise for an unrealized certificate"
            )
        if emit_recording and getattr(ctx, "recording_export_path", None):
            out = Path(ctx.recording_export_path)
            (out / ("episode_%04d Recording 000000 - 000100.jsonl" % ctx.tag)).write_text(
                json.dumps({"tag": ctx.tag}) + "\n", encoding="utf-8")
        # One wake, so a training iteration really runs an update and
        # `updates_completed` advances -- the identity field the eval bundles carry.
        return _FakeResult(confirmed_kills=2,
                           trajectory=[_StubTransition("ego_0", 0)], n_wakes=1)

    def fake_build_fuel_damage_controller(ctx, *, episode_seed, params):
        calls.append({"kind": "fuel_damage_controller", "tag": ctx.tag,
                      "export_calls_before_controller": ctx.game.export_calls})
        return _VaFuelDamageController(
            resolve_condition(episode_seed=int(episode_seed), params=params))

    graph_train._git_provenance = lambda repo_root: dict(_FAKE_GIT_OK)
    graph_train._build_generator = fake_build_generator
    graph_train.setup_episode = fake_setup_episode
    graph_train.run_episode = fake_run_episode
    graph_train.compute_episode_reward = lambda *a, **k: _FakeReward()
    graph_train.build_fuel_damage_controller = fake_build_fuel_damage_controller
    graph_train.PPOUpdater = lambda policy, ppo: _RecordingUpdater(policy, ppo, log=[])
    buf = io.StringIO()
    summary = None
    try:
        with contextlib.redirect_stdout(buf):
            summary = graph_train.train(cfg)
    except BaseException as exc:  # noqa: BLE001 - the abort itself is under test
        state["raised"] = exc
    finally:
        state["stdout"] = buf.getvalue()
        for name, original in saved.items():
            setattr(graph_train, name, original)
    return summary, calls, state


# --- GENERALIZED-V1 step 2: `FuelDamageIntegrityError` routing -----------------
# A world CERTIFIED FD-capable at setup that then contradicts its own certificate is an
# INSTRUMENT fault, not an experimental outcome (handoff 3l.3). It must be routed exactly
# as `MeasurementIntegrityError` and `_VisualArtifactError` already are: re-raised ahead
# of every broad handler, never wrapped as an `EpisodeAttemptError`, never written to the
# ledger, never counted against a condition, never entered into `skip_and_account_v1`.
# The whole lesson of the invalid long baseline is that a defect which silently shrinks a
# scientific denominator is worse than one that stops the run.

def test_a_fuel_damage_integrity_fault_is_not_wrapped_as_an_episode_failure(
    tmp_path: Path,
) -> None:
    """GEN-V1. `_run_one_episode` re-raises it instead of attributing it to `run`.

    The REAL `_run_one_episode` runs here, so the assertion is about its own handler
    ordering: the broad `except Exception -> EpisodeAttemptError("run", ...)` sits right
    below, and if the re-raise were removed this fault would be booked as ordinary
    episode attrition at a named pipeline stage.

    The exception is produced by the REAL terminal invariant on a REAL certified damaged
    controller whose event never fired, which is one of the two production sources (the
    other is a live contradiction inside `maybe_apply`). Both raise from inside
    `run_episode`, so this one seam covers both.
    """
    cfg = _va_cfg(tmp_path, name="fd_integrity", visual_artifacts=False,
                  eval_every=0, eval_episodes=0)
    _summary, _calls, state = _run_training_with_real_episode_body(
        cfg, fd_integrity_seeds=(0,))

    raised = state["raised"]
    assert isinstance(raised, graph_train.FuelDamageIntegrityError), raised
    assert not isinstance(raised, EpisodeAttemptError), raised
    assert not hasattr(raised, "stage"), (
        "an instrument fault names no pipeline stage -- it did not happen in one"
    )
    assert "CERTIFICATE NOT REALIZED" in str(raised), str(raised)
    # It is a SIBLING of the other two, not a subclass of either, and emphatically not a
    # `FuelDamageError`: anything catching one of those would swallow it.
    assert not isinstance(raised, graph_train.MeasurementIntegrityError)
    assert not isinstance(raised, graph_train._VisualArtifactError)


def test_a_fuel_damage_integrity_fault_aborts_the_run_and_never_reaches_the_ledger(
    tmp_path: Path,
) -> None:
    """GEN-V1. The TRAIN handler re-raises it: the run stops, the ledger stays empty.

    Four scheduled training episodes with the SECOND faulting. If it were accounted, the
    run would continue through all four and one `setup`/`run` row would appear -- exactly
    the silent denominator shrink the roster correction closed.
    """
    cfg = _va_cfg(tmp_path, name="fd_abort", visual_artifacts=False,
                  episodes_per_iteration=4, eval_every=0, eval_episodes=0)
    _summary, calls, state = _run_training_with_real_episode_body(
        cfg, fd_integrity_seeds=(1,))

    assert isinstance(state["raised"], graph_train.FuelDamageIntegrityError),         state["raised"]
    attempted = [c["seed"] for c in calls if c["kind"] == "generate"]
    assert attempted == [0, 1], (
        "the run must stop AT the faulting attempt, never reach seeds 2 and 3: %r"
        % (attempted,)
    )
    run_dir = Path(cfg.output_dir)
    assert _read_records(run_dir, "episode_failures.jsonl") == []
    assert _read_records(run_dir, "train_records.jsonl") == []


def test_an_eval_fuel_damage_integrity_fault_aborts_the_round(tmp_path: Path) -> None:
    """GEN-V1. The EVAL handler re-raises it too -- both handlers, not just one.

    The `pre_update` round runs before any training episode, so a faulting held-out seed
    stops the run before a single training attempt exists.
    """
    cfg = _va_cfg(tmp_path, name="fd_abort_eval", visual_artifacts=False,
                  episodes_per_iteration=2, eval_every=1, eval_episodes=2,
                  eval_base_seed=1_000_000)
    _summary, calls, state = _run_training_with_real_episode_body(
        cfg, fd_integrity_seeds=(1_000_001,))

    assert isinstance(state["raised"], graph_train.FuelDamageIntegrityError),         state["raised"]
    train_seeds = [c["seed"] for c in calls
                   if c["kind"] == "generate" and c["seed"] < 1_000_000]
    assert train_seeds == [], "training started after an eval instrument fault"
    run_dir = Path(cfg.output_dir)
    assert _read_records(run_dir, "episode_failures.jsonl") == []
    assert _read_records(run_dir, "eval_records.jsonl") == []


def test_an_ordinary_fuel_damage_failure_is_still_accounted_attrition(
    tmp_path: Path,
) -> None:
    """GEN-V1 CONTROL. The routing above must not have widened to ordinary FD failures.

    A `FuelDamageError` -- no eligible ego, no valid strict window -- is a SCIENTIFIC
    outcome and keeps its `setup` row and its place in `skip_and_account_v1`. If the new
    re-raise had been written against the wrong class, this run would abort instead.
    """
    cfg = _va_cfg(tmp_path, name="fd_ordinary", visual_artifacts=False,
                  episodes_per_iteration=2, eval_every=0, eval_episodes=0)
    summary, calls, state = _run_training_with_real_episode_body(
        cfg, setup_error_seeds=(0,))

    assert state["raised"] is None, state["raised"]
    attempted = [c["seed"] for c in calls if c["kind"] == "generate"]
    assert attempted == [0, 1], "the schedule continued past an accounted failure"
    rows = _read_records(Path(cfg.output_dir), "episode_failures.jsonl")
    assert len(rows) == 1 and rows[0]["pipeline_stage"] == "setup", rows
    assert summary is not None and summary["accounting_reconciled"] is True


def _va_cfg(tmp_path: Path, *, name: str, visual_artifacts: bool, **kwargs) -> TrainConfig:
    """A tiny 1x1 run with one held-out pair, so a full run has 4 attempts (2+1+... )."""
    defaults = dict(
        n_iterations=1, episodes_per_iteration=1, base_seed=0,
        output_dir=tmp_path / name, eval_every=1, eval_episodes=1,
        eval_base_seed=1_000_000, checkpoint_every=0,
        visual_artifacts=visual_artifacts,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def _va_bundles(run_dir: Path) -> list:
    """Every attempt bundle under `<run_dir>/visual_artifacts`, sorted by name."""
    root = Path(run_dir) / "visual_artifacts"
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)


def _va_manifest(bundle: Path) -> dict:
    return json.loads((bundle / "artifact_manifest.json").read_text(encoding="utf-8"))


# --- PO1: default OFF, at both surfaces and on the whole call path -------------

def test_visual_artifacts_default_off_at_both_surfaces() -> None:
    """PO1. The dataclass default and the CLI default are both OFF; the flag turns it on.

    The CLI default is READ OFF `TrainConfig`, so this also guards the drift the other
    CLI-default test guards for the scenario knobs.
    """
    assert TrainConfig(n_iterations=1).visual_artifacts is False

    parser = _build_arg_parser()
    off = parser.parse_args(["--iterations", "1"])
    assert off.visual_artifacts is False
    assert TrainConfig(n_iterations=1).visual_artifacts == off.visual_artifacts

    on = parser.parse_args(["--iterations", "1", "--visual-artifacts"])
    assert on.visual_artifacts is True


def test_disabled_run_builds_no_bundle_and_no_directory(tmp_path: Path) -> None:
    """PO1. Off means nothing is constructed: no identity, no directory, no files.

    Driven through the stub trainer, which records the `artifacts` value the loop resolved
    for every attempt -- so this asserts the ABSENCE at the call boundary, not merely that
    no file happened to appear.
    """
    cfg = _va_cfg(tmp_path, name="off", visual_artifacts=False)
    summary, events, _state = _run_stub_training(cfg)

    episodes = [e for e in events if e[0] == "episode"]
    assert episodes, "the stub run collected no attempts"
    assert all(e[6] is False for e in episodes), \
        "the disabled run passed an `artifacts` keyword at all"
    assert all(e[5] is None for e in episodes), \
        "the disabled run constructed an artifact bundle"
    assert not (Path(cfg.output_dir) / "visual_artifacts").exists()
    assert summary["n_iterations"] == cfg.n_iterations


def test_disabled_run_passes_no_recording_path_and_never_exports(tmp_path: Path) -> None:
    """PO1. `setup_episode` gets NO recording keyword and `export_scenario` is not called.

    The strongest form of the claim: the keyword is ABSENT from the call, not present and
    `None`, so the disabled path is the pre-feature call.
    """
    cfg = _va_cfg(tmp_path, name="off_body", visual_artifacts=False)
    _summary, calls, state = _run_training_with_real_episode_body(cfg)
    assert state["raised"] is None, state["raised"]

    setups = [c for c in calls if c["kind"] == "setup"]
    assert setups, "no episode reached setup"
    assert all(c["recording_kwarg_present"] is False for c in setups), \
        "the disabled path armed recording"
    # `export_scenario` is only ever reached through the artifact layer, so the count
    # observed by the controller (which runs after the export site) stays 0.
    controllers = [c for c in calls if c["kind"] == "fuel_damage_controller"]
    assert controllers and all(
        c["export_calls_before_controller"] == 0 for c in controllers)
    assert not (Path(cfg.output_dir) / "visual_artifacts").exists()


def test_enabling_artifacts_changes_no_seed_tag_or_condition(tmp_path: Path) -> None:
    """PO1/PO3. The schedule is identical with the feature on and off.

    Same seeds, in the same order, under the same scenario tags, with the same forced
    evaluation modes. Artifacts are an observation surface; if they moved any of these the
    two runs would not be comparable at all.
    """
    def _schedule(name, enabled):
        cfg = _va_cfg(tmp_path, name=name, visual_artifacts=enabled,
                      n_iterations=2, episodes_per_iteration=2, eval_episodes=2)
        _summary, events, _state = _run_stub_training(cfg)
        return [(e[1], e[2], e[3], e[4]) for e in events if e[0] == "episode"]

    assert _schedule("sched_off", False) == _schedule("sched_on", True)


# --- PO2: fidelity of the three snapshots, and the identity that names them ----

def test_enabled_attempt_preserves_all_three_snapshots(tmp_path: Path) -> None:
    """PO2. One bundle per attempt, holding the exact three files and a complete manifest.

    A 1x1 run with one held-out pair schedules four attempts: pre_update clean,
    pre_update damaged, one training episode, post_update clean, post_update damaged --
    covering every phase and both matched-pair members in one run.
    """
    cfg = _va_cfg(tmp_path, name="on", visual_artifacts=True)
    _summary, calls, state = _run_training_with_real_episode_body(cfg)
    assert state["raised"] is None, state["raised"]

    run_dir = Path(cfg.output_dir)
    bundles = _va_bundles(run_dir)
    attempts = [c for c in calls if c["kind"] == "generate"]
    assert len(bundles) == len(attempts) == 5, [b.name for b in bundles]

    phases = sorted({_va_manifest(b)["identity"]["phase"] for b in bundles})
    assert phases == ["post_update", "pre_update", "train"]

    for bundle in bundles:
        manifest = _va_manifest(bundle)
        tag = manifest["identity"]["episode_tag"]
        seed = manifest["identity"]["seed"]

        assert manifest["status"] == "complete"
        assert manifest["schema"] == "final_cell_visual_artifacts"
        assert manifest["version"] == 1
        assert manifest["source_episode_tag"] == tag

        # 1. the generator's known-only world, BYTE for byte (3 targets).
        known_only = bundle / manifest["known_only_scenario"]
        assert manifest["known_only_scenario"] == "known_only_scenario.json"
        assert known_only.read_bytes() == _va_known_only_bytes(tag, seed)
        # ... and the run's own scenario file is untouched by the copy.
        original = run_dir / "scenarios" / ("episode_%04d_scenario.json" % tag)
        assert original.read_bytes() == known_only.read_bytes()

        # 2. the AUTHORITATIVE executed world (6 targets), equal to what env-2 exported.
        executed = bundle / manifest["executed_t0_scenario"]
        assert manifest["executed_t0_scenario"] == "executed_t0_scenario.json"
        assert json.loads(executed.read_text(encoding="utf-8")) == \
            _va_executed_t0_object(tag, seed)
        assert len(json.loads(executed.read_text(encoding="utf-8"))
                   ["currentScenario"]["airbases"]) == 6

        # 3. the BLADE playback the tick loop emitted into this bundle.
        recordings = manifest["playback_recordings"]
        assert recordings and all((bundle / r).exists() for r in recordings)
        assert all(r.endswith(".jsonl") for r in recordings)
        assert sorted(p.name for p in bundle.glob("*.jsonl")) == sorted(recordings)

        # target-count expectations vs observations, both explicit.
        assert manifest["targets"]["expected"] == {
            "n_known": cfg.n_known, "n_hidden": cfg.n_hidden,
            "n_targets_executed": cfg.n_targets_emitted}
        assert manifest["targets"]["observed"]["n_known"] == 3
        assert manifest["targets"]["observed"]["n_hidden"] == 3
        assert manifest["targets"]["observed"]["n_targets_executed"] == 6


def test_manifest_identity_places_every_phase_exactly(tmp_path: Path) -> None:
    """PO2. Phase, iteration/update state, ordinals, seed, condition and tag are explicit.

    Read from the manifests alone -- no console order, no directory order.
    """
    cfg = _va_cfg(tmp_path, name="identity", visual_artifacts=True)
    _summary, _calls, state = _run_training_with_real_episode_body(cfg)
    assert state["raised"] is None, state["raised"]

    ids = {}
    for bundle in _va_bundles(Path(cfg.output_dir)):
        ident = _va_manifest(bundle)["identity"]
        ids[(ident["phase"], ident["seed"], ident["condition"])] = ident

    train_id = ids[("train", 0, resolve_condition(
        episode_seed=0, params=cfg.fuel_damage_parameters()))]
    assert train_id["iteration"] == 0
    assert train_id["episode_index"] == 0
    assert train_id["attempt_ordinal"] == 0
    assert train_id["updates_completed"] == 0
    assert train_id["episode_tag"] == 0
    assert train_id["eval_round_ordinal"] is None
    assert train_id["eval_episode_index"] is None
    assert train_id["eval_pair_member"] is None

    pre_clean = ids[("pre_update", 1_000_000, CONDITION_CLEAN)]
    pre_damaged = ids[("pre_update", 1_000_000, CONDITION_DAMAGED)]
    assert pre_clean["iteration"] is None and pre_clean["updates_completed"] == 0
    assert (pre_clean["eval_round_ordinal"], pre_clean["eval_pair_member"]) == (0, 0)
    assert (pre_damaged["eval_round_ordinal"], pre_damaged["eval_pair_member"]) == (0, 1)
    assert pre_clean["eval_episode_index"] == pre_damaged["eval_episode_index"] == 0
    assert pre_clean["attempt_ordinal"] == 0 and pre_damaged["attempt_ordinal"] == 1
    assert pre_clean["episode_index"] is None

    post_clean = ids[("post_update", 1_000_000, CONDITION_CLEAN)]
    # The SAME held-out seed as the pre-update round, one round later -- the case that
    # must not overwrite anything.
    assert post_clean["eval_round_ordinal"] == 1
    assert post_clean["updates_completed"] == 1
    assert post_clean["iteration"] == 0
    assert post_clean["episode_tag"] != pre_clean["episode_tag"]


def test_one_seed_cannot_overwrite_another_bundle(tmp_path: Path) -> None:
    """PO2. The same held-out seed x 2 members x 2 rounds gives FOUR distinct bundles.

    This is the exact collision the eval tag namespace exists to prevent, now applied to
    artifact directories: four attempts share one seed and one world, and each keeps its
    own files.
    """
    cfg = _va_cfg(tmp_path, name="collide", visual_artifacts=True)
    _summary, _calls, state = _run_training_with_real_episode_body(cfg)
    assert state["raised"] is None, state["raised"]

    eval_bundles = [b for b in _va_bundles(Path(cfg.output_dir))
                    if _va_manifest(b)["identity"]["seed"] == 1_000_000]
    assert len(eval_bundles) == 4
    assert len({b.name for b in eval_bundles}) == 4
    keys = {(m["phase"], m["eval_round_ordinal"], m["eval_pair_member"])
            for m in (_va_manifest(b)["identity"] for b in eval_bundles)}
    assert keys == {("pre_update", 0, 0), ("pre_update", 0, 1),
                    ("post_update", 1, 0), ("post_update", 1, 1)}
    # Every bundle really holds its own three artifacts (nothing was merged away).
    for bundle in eval_bundles:
        manifest = _va_manifest(bundle)
        assert manifest["status"] == "complete"
        assert (bundle / "known_only_scenario.json").exists()
        assert (bundle / "executed_t0_scenario.json").exists()
        assert manifest["playback_recordings"]


def test_a_bundle_directory_collision_fails_loudly(tmp_path: Path) -> None:
    """PO2. A pre-existing attempt directory RAISES; nothing is overwritten or merged."""
    root = tmp_path / "artifacts"
    identity = graph_train._AttemptIdentity(
        phase="train", iteration=0, updates_completed=0, eval_round_ordinal=None,
        eval_episode_index=None, eval_pair_member=None, attempt_ordinal=0,
        episode_index=0, seed=0, condition=CONDITION_CLEAN, episode_tag=0,
    )
    first = graph_train._AttemptArtifacts(root=root, identity=identity).open()
    (first.directory / "known_only_scenario.json").write_bytes(b"original")

    second = graph_train._AttemptArtifacts(root=root, identity=identity)
    try:
        second.open()
    except graph_train._VisualArtifactError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("a colliding bundle directory was accepted")
    assert (first.directory / "known_only_scenario.json").read_bytes() == b"original"


def test_export_happens_before_the_controller_and_exactly_once(tmp_path: Path) -> None:
    """PO2/PO3. `export_scenario()` runs ONCE, before the FD controller and the run.

    The controller is what plans (and therefore can fire) the fuel-damage mutation, so
    "before the controller exists" is also "before the top-of-tick mutation, before any
    policy decision and before any env.step".
    """
    cfg = _va_cfg(tmp_path, name="order", visual_artifacts=True)
    _summary, calls, state = _run_training_with_real_episode_body(cfg)
    assert state["raised"] is None, state["raised"]

    controllers = [c for c in calls if c["kind"] == "fuel_damage_controller"]
    runs = [c for c in calls if c["kind"] == "run"]
    assert controllers and runs
    assert all(c["export_calls_before_controller"] == 1 for c in controllers), \
        "the executed t=0 scenario was not exported exactly once before the controller"
    assert all(c["export_calls_before_run"] == 1 for c in runs)

    # Ordering in the call log itself: setup -> controller -> run, per attempt.
    kinds = [c["kind"] for c in calls]
    assert kinds == ["generate", "setup", "fuel_damage_controller", "run"] * len(runs)


def test_recording_is_armed_only_on_the_bundle_directory(tmp_path: Path) -> None:
    """PO2. Recording is armed through `setup_episode`, at the attempt's own directory."""
    cfg = _va_cfg(tmp_path, name="armed", visual_artifacts=True)
    _summary, calls, state = _run_training_with_real_episode_body(cfg)
    assert state["raised"] is None, state["raised"]

    root = Path(cfg.output_dir) / "visual_artifacts"
    setups = [c for c in calls if c["kind"] == "setup"]
    assert setups and all(c["recording_kwarg_present"] for c in setups)
    armed = [Path(c["recording_export_path"]) for c in setups]
    assert len(armed) == len({str(p) for p in armed}), "two attempts shared a directory"
    assert all(p.parent == root for p in armed)
    assert {p.name for p in armed} == {b.name for b in _va_bundles(Path(cfg.output_dir))}


# --- PO3: observational isolation, and honest failure routing ------------------

def test_artifacts_change_no_outcome_record_or_ppo_input(tmp_path: Path) -> None:
    """PO3. An enabled run and a disabled run produce identical scientific records.

    Same seeds, tags, conditions, rewards, wake counts, target counts, meta-action mixes,
    endings, failure accounting and PPO batch shapes. Only the wall-clock fields and the
    run directory legitimately differ.
    """
    def _run(name, enabled):
        cfg = _va_cfg(tmp_path, name=name, visual_artifacts=enabled,
                      n_iterations=2, episodes_per_iteration=2, eval_episodes=1)
        _summary, calls, state = _run_training_with_real_episode_body(cfg)
        assert state["raised"] is None, state["raised"]
        run_dir = Path(cfg.output_dir)
        return (
            _comparable_records(_read_records(run_dir, "train_records.jsonl")),
            _comparable_records(_read_records(run_dir, "eval_records.jsonl")),
            _read_records(run_dir, "episode_failures.jsonl"),
            [(c["kind"], c.get("tag"), c.get("seed")) for c in calls],
            run_dir,
        )

    off_train, off_eval, off_fail, off_calls, off_dir = _run("iso_off", False)
    on_train, on_eval, on_fail, on_calls, on_dir = _run("iso_on", True)

    assert off_train == on_train
    assert off_eval == on_eval
    assert off_fail == on_fail == []
    assert off_calls == on_calls
    assert not (off_dir / "visual_artifacts").exists()
    assert _va_bundles(on_dir), "the enabled run wrote no bundles"


def test_a_normal_episode_failure_stays_scientific_and_leaves_an_incomplete_bundle(
    tmp_path: Path,
) -> None:
    """PO3. A `setup` failure is accounted as before; its bundle stays `incomplete`.

    The pre-failure artifact that WAS valid (the known-only scenario) is kept, the two
    that could not exist are absent, and no recording is fabricated -- the tick loop
    exports none when the loop never ran.
    """
    cfg = _va_cfg(tmp_path, name="epfail", visual_artifacts=True,
                  n_iterations=1, episodes_per_iteration=1, eval_every=0,
                  eval_episodes=0)
    _summary, _calls, state = _run_training_with_real_episode_body(
        cfg, setup_error_seeds=(0,))
    assert state["raised"] is None, state["raised"]

    run_dir = Path(cfg.output_dir)
    ledger = _read_records(run_dir, "episode_failures.jsonl")
    assert len(ledger) == 1
    assert ledger[0]["pipeline_stage"] == "setup"
    assert ledger[0]["pipeline_stage"] in _PIPELINE_STAGES
    assert ledger[0]["seed"] == 0 and ledger[0]["phase"] == "train"

    bundles = _va_bundles(run_dir)
    assert len(bundles) == 1
    manifest = _va_manifest(bundles[0])
    assert manifest["status"] == "incomplete"
    assert manifest["known_only_scenario"] == "known_only_scenario.json"
    assert (bundles[0] / "known_only_scenario.json").exists()
    assert manifest["executed_t0_scenario"] is None
    assert manifest["playback_recordings"] == []
    assert not list(bundles[0].glob("*.jsonl")), "a recording was fabricated"


def test_an_artifact_failure_aborts_and_never_enters_the_ledger(tmp_path: Path) -> None:
    """PO3. An artifact failure raises `_VisualArtifactError` and stops the run.

    It must not be written as a `generation` / `setup` / `run` / `reward` failure, must
    not enter `skip_and_account_v1`, and therefore cannot shrink a scientific denominator
    by masquerading as an episode failure.
    """
    # The FIRST attempt of a run with eval enabled is a pre_update member (tag 900000),
    # so this exercises the eval handler's re-raise.
    cfg = _va_cfg(tmp_path, name="artfail_eval", visual_artifacts=True)
    _summary, _calls, state = _run_training_with_real_episode_body(
        cfg, export_error_tags=(900_000,))
    assert isinstance(state["raised"], graph_train._VisualArtifactError), state["raised"]
    assert not isinstance(state["raised"], EpisodeAttemptError)
    assert _read_records(Path(cfg.output_dir), "episode_failures.jsonl") == []

    # And the training handler's re-raise, on the training tag.
    cfg2 = _va_cfg(tmp_path, name="artfail_train", visual_artifacts=True,
                   eval_every=0, eval_episodes=0)
    _summary2, _calls2, state2 = _run_training_with_real_episode_body(
        cfg2, export_error_tags=(0,))
    assert isinstance(state2["raised"], graph_train._VisualArtifactError), state2["raised"]
    assert _read_records(Path(cfg2.output_dir), "episode_failures.jsonl") == []


def test_a_missing_recording_is_an_artifact_failure_not_a_silent_pass(
    tmp_path: Path,
) -> None:
    """PO3. A completed episode with no playback file aborts; the bundle stays incomplete.

    A bundle without a recording is not the deliverable, and reporting it as `complete`
    would let a probe finish with un-inspectable episodes.
    """
    cfg = _va_cfg(tmp_path, name="norec", visual_artifacts=True,
                  eval_every=0, eval_episodes=0)
    _summary, _calls, state = _run_training_with_real_episode_body(
        cfg, emit_recording=False)
    assert isinstance(state["raised"], graph_train._VisualArtifactError), state["raised"]
    assert "no BLADE playback" in str(state["raised"])
    assert _read_records(Path(cfg.output_dir), "episode_failures.jsonl") == []
    manifest = _va_manifest(_va_bundles(Path(cfg.output_dir))[0])
    assert manifest["status"] == "incomplete"


def test_run_config_records_the_resolved_visual_artifacts_flag(tmp_path: Path) -> None:
    """PO1/PO2. The resolved flag is in `run_config.json` through the existing path."""
    for enabled in (False, True):
        run_dir = tmp_path / ("cfg_%s" % enabled)
        run_dir.mkdir()
        cfg = TrainConfig(n_iterations=1, output_dir=run_dir,
                          visual_artifacts=enabled)
        path = write_run_config(run_dir, cfg, provenance={"stub": True})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["train_config"]["visual_artifacts"] is enabled


def test_an_attempt_identity_cannot_omit_what_names_it() -> None:
    """PO2. An identity that could not distinguish its attempt is refused outright."""
    base = dict(phase="train", iteration=0, updates_completed=0,
                eval_round_ordinal=None, eval_episode_index=None,
                eval_pair_member=None, attempt_ordinal=0, episode_index=0,
                seed=0, condition=CONDITION_CLEAN, episode_tag=0)
    graph_train._AttemptIdentity(**base)          # the complete form is accepted

    for field_name in ("iteration", "episode_index"):
        broken = dict(base, **{field_name: None})
        try:
            graph_train._AttemptIdentity(**broken)
        except ValueError as exc:
            assert field_name in str(exc)
        else:
            raise AssertionError("a train identity without %s was accepted" % field_name)

    eval_base = dict(base, phase="pre_update", iteration=None, episode_index=None,
                     eval_round_ordinal=0, eval_episode_index=0, eval_pair_member=0,
                     episode_tag=900_000)
    graph_train._AttemptIdentity(**eval_base)
    for field_name in ("eval_round_ordinal", "eval_episode_index", "eval_pair_member"):
        broken = dict(eval_base, **{field_name: None})
        try:
            graph_train._AttemptIdentity(**broken)
        except ValueError as exc:
            assert field_name in str(exc)
        else:
            raise AssertionError("an eval identity without %s was accepted" % field_name)

    try:
        graph_train._AttemptIdentity(**dict(base, phase="not_a_phase"))
    except ValueError as exc:
        assert "phase" in str(exc)
    else:
        raise AssertionError("an unknown artifact phase was accepted")



# =============================================================================
# T16 -- JSON presets: --config, override precedence, and the recorded source
# =============================================================================

_PROBE_PRESET = ROOT / "configs" / "graph_train" / "final_cell_probe.json"


def _resolve(argv, *, config_path=None):
    """Resolve argv the way `main` does: defaults < preset < EXPLICIT flags."""
    parser = _build_arg_parser()
    values = None if config_path is None else load_config_file(config_path)
    return resolve_train_config(
        parser.parse_args(argv),
        explicit=_explicit_cli_dests(argv),
        config_values=values,
        config_path=config_path,
    )


def test_the_repository_probe_preset_resolves_a_valid_train_config() -> None:
    """PO1. The shipped preset loads, resolves and VALIDATES as the bounded probe.

    The preset is the artifact that makes the short probe reproducible from the
    repository rather than from a shell history, so its exact declared shape is pinned
    here: a wrong number in that file is a differently-sized experiment.
    """
    assert _PROBE_PRESET.exists(), str(_PROBE_PRESET)
    argv = ["--config", str(_PROBE_PRESET)]
    cfg, source = _resolve(argv, config_path=_PROBE_PRESET)
    cfg.validate()      # must not raise: the preset is a runnable configuration

    # The bounded short probe, exactly.
    assert cfg.n_iterations == 2
    assert cfg.episodes_per_iteration == 4
    assert cfg.base_seed == 0
    assert cfg.eval_every == 2
    assert cfg.eval_episodes == 4
    assert cfg.eval_base_seed == 1_000_000
    assert cfg.total_episodes == 8, "the probe must stay bounded"
    # The final cell, unchanged.
    assert (cfg.num_agents, cfg.n_known, cfg.n_hidden) == (3, 3, 3)
    assert cfg.n_targets_emitted == 6
    assert cfg.min_target_distance_km == 200.0
    assert cfg.min_known_separation_km == 100.0
    assert cfg.include_sams is False
    # FD-BASELINE-v1 as merged on main.
    assert cfg.fuel_damage_mode == FuelDamageMode.SEEDED_MIXTURE
    assert cfg.fuel_damage_probability == 0.5
    assert cfg.fuel_damage_leg_progress == 0.30
    assert cfg.fuel_damage_rtb_margin == 1.10
    assert cfg.aircraft_penalty_coeff == 2.25
    # The inspection surface is ON for this probe.
    assert cfg.visual_artifacts is True

    assert source["resolved_from"] == "config_file"
    assert source["cli_overrides"] == [], "the preset alone must need no flags"


def test_the_probe_preset_carries_the_fd_and_cell_defaults_of_the_dataclass() -> None:
    """The preset RESTATES the approved defaults; it never quietly retunes the cell.

    Anything the preset sets that also has a dataclass default must AGREE with it. The
    probe is meant to measure the merged cell, so a preset that silently changed the
    geometry or the difficulty factor would measure something else under the same name.
    """
    d = TrainConfig(n_iterations=1)
    values = load_config_file(_PROBE_PRESET)
    for field_name in ("num_agents", "n_known", "n_hidden", "min_target_distance_km",
                       "min_known_separation_km", "include_sams", "fuel_damage_mode",
                       "fuel_damage_probability", "fuel_damage_leg_progress",
                       "fuel_damage_rtb_margin", "aircraft_penalty_coeff",
                       "base_seed", "eval_base_seed"):
        assert values[field_name] == getattr(d, field_name), field_name
    # The two knobs the preset deliberately CHANGES are the run's size, nothing else.
    changed = {k for k in values
               if k in {f.name for f in dataclasses.fields(TrainConfig)}
               and values[k] != getattr(d, k, object())}
    assert changed <= {"n_iterations", "episodes_per_iteration", "eval_every",
                       "eval_episodes", "visual_artifacts"}, sorted(changed)


def test_explicit_cli_flags_beat_the_json_preset(tmp_path: Path) -> None:
    """PO1. Defaults < preset < EXPLICIT flag -- and a DEFAULT never counts as explicit.

    The load-bearing half is the negative one. If "explicit" were inferred from the
    parsed value, every flag whose default happens to differ from the preset would
    silently override the preset, and a preset would be unusable.
    """
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({
        "n_iterations": 5,
        "episodes_per_iteration": 6,
        "base_seed": 11,
        "n_hidden": 2,
        "ppo": {"lr": 0.005, "n_epochs": 7},
    }), encoding="utf-8")

    # (a) preset alone: every declared field is taken, nothing else moves.
    argv = ["--config", str(preset)]
    cfg, source = _resolve(argv, config_path=preset)
    assert (cfg.n_iterations, cfg.episodes_per_iteration) == (5, 6)
    assert cfg.base_seed == 11 and cfg.n_hidden == 2
    assert cfg.ppo.lr == 0.005 and cfg.ppo.n_epochs == 7
    # ... and the fields the preset did NOT set are still the dataclass defaults, even
    # though argparse handed `resolve_train_config` a value for every one of them.
    d = TrainConfig(n_iterations=1)
    assert cfg.n_known == d.n_known
    assert cfg.eval_episodes == d.eval_episodes
    assert cfg.ppo.clip_ratio == PPOConfig().clip_ratio
    assert source["cli_overrides"] == []

    # (b) explicit flags win, including inside the nested PPO block.
    argv = ["--config", str(preset), "--episodes", "2", "--seed", "99", "--lr", "0.001"]
    cfg, source = _resolve(argv, config_path=preset)
    assert cfg.episodes_per_iteration == 2 and cfg.base_seed == 99
    assert cfg.ppo.lr == 0.001
    assert cfg.n_iterations == 5, "an unmentioned preset field must survive"
    assert cfg.ppo.n_epochs == 7
    assert source["cli_overrides"] == ["base_seed", "episodes_per_iteration", "ppo.lr"]

    # (c) a flag passed its OWN DEFAULT value is still explicit -- it was typed.
    argv = ["--config", str(preset), "--seed", str(d.base_seed)]
    cfg, _ = _resolve(argv, config_path=preset)
    assert cfg.base_seed == d.base_seed != 11

    # (d) a store_true flag: absent means "the preset decides", present means ON.
    preset.write_text(json.dumps({"n_iterations": 1, "visual_artifacts": True}),
                      encoding="utf-8")
    cfg, _ = _resolve(["--config", str(preset)], config_path=preset)
    assert cfg.visual_artifacts is True, "the absent flag overrode the preset"
    cfg, _ = _resolve(["--config", str(preset), "--visual-artifacts"],
                      config_path=preset)
    assert cfg.visual_artifacts is True


def test_no_config_reproduces_the_pre_preset_cli_resolution() -> None:
    """With no --config, resolution is exactly the argparse defaults plus what was typed.

    The invariance claim for the whole feature: presets must not have changed what an
    existing command line already produced.
    """
    argv = ["--iterations", "3", "--n-known", "4", "--num-agents", "2"]
    cfg, source = _resolve(argv)
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    assert cfg.n_iterations == 3 and cfg.n_known == 4 and cfg.num_agents == 2
    # Every mapped flag equals what argparse resolved -- no third source of values.
    # `output_dir` is excluded: TrainConfig.__post_init__ turns the empty default into
    # `training_output_<timestamp>`, which is the dataclass's own behaviour and not a
    # resolution step.
    for dest, field_name in graph_train._CLI_FIELD_BY_DEST.items():
        if field_name == "output_dir":
            continue
        assert getattr(cfg, field_name) == getattr(args, dest), field_name
    assert args.out == "" and str(cfg.output_dir).startswith("training_output_")
    for dest, field_name in graph_train._CLI_PPO_FIELD_BY_DEST.items():
        assert getattr(cfg.ppo, field_name) == getattr(args, dest), field_name
    assert source["path"] is None and source["resolved_from"] == "cli_defaults"
    assert source["config_fields"] == [] and source["cli_overrides"] == []


def test_a_config_may_supply_iterations_and_its_absence_still_fails(
    tmp_path: Path,
) -> None:
    """`n_iterations` may come from the preset, but it is never DEFAULTED into existence."""
    preset = tmp_path / "iters.json"
    preset.write_text(json.dumps({"n_iterations": 4}), encoding="utf-8")
    cfg, _ = _resolve(["--config", str(preset)], config_path=preset)
    assert cfg.n_iterations == 4

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"base_seed": 3}), encoding="utf-8")
    try:
        _resolve(["--config", str(empty)], config_path=empty)
    except ValueError as exc:
        assert "n_iterations" in str(exc)
    else:
        raise AssertionError("a config with no n_iterations was accepted")


def test_load_config_file_refuses_what_it_cannot_honour(tmp_path: Path) -> None:
    """A typo is REFUSED, not ignored: a silently dropped knob is a mismeasured run."""
    bad = tmp_path / "bad.json"

    bad.write_text(json.dumps({"n_iterations": 1, "n_hiddenn": 3}), encoding="utf-8")
    try:
        load_config_file(bad)
    except ValueError as exc:
        assert "n_hiddenn" in str(exc)
    else:
        raise AssertionError("an unknown TrainConfig field was accepted")

    bad.write_text(json.dumps({"n_iterations": 1, "ppo": {"learning_rate": 0.1}}),
                   encoding="utf-8")
    try:
        load_config_file(bad)
    except ValueError as exc:
        assert "learning_rate" in str(exc)
    else:
        raise AssertionError("an unknown PPOConfig field was accepted")

    bad.write_text(json.dumps({"n_iterations": 1, "ppo": 3}), encoding="utf-8")
    try:
        load_config_file(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-object ppo block was accepted")

    bad.write_text("[1, 2, 3]", encoding="utf-8")
    try:
        load_config_file(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-object config was accepted")

    bad.write_text("{not json", encoding="utf-8")
    try:
        load_config_file(bad)
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("malformed JSON was accepted")

    try:
        load_config_file(tmp_path / "nope.json")
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("a missing config file was accepted")


def test_config_comments_are_ignored_and_tuples_round_trip(tmp_path: Path) -> None:
    """Underscore keys are prose; a list reloads as the tuple the dataclass holds.

    The tuple half matters because `asdict` writes `num_red_airbases` as a LIST, so a
    preset lifted out of a previous run's `run_config.json:/train_config` must load back
    into the config it came from.
    """
    preset = tmp_path / "commented.json"
    preset.write_text(json.dumps({
        "_comment": "why this preset exists",
        "_anything": {"nested": "prose"},
        "n_iterations": 1,
        "num_red_airbases": [6, 8],
        "ppo": {"_note": "prose here too", "lr": 0.002},
    }), encoding="utf-8")
    values = load_config_file(preset)
    assert "_comment" not in values and "_anything" not in values
    assert values["num_red_airbases"] == (6, 8)
    assert values["ppo"] == {"lr": 0.002}

    cfg, _ = _resolve(["--config", str(preset)], config_path=preset)
    assert cfg.num_red_airbases == (6, 8)
    # ... and that IS the shape the dataclass round-trips through `asdict`.
    assert tuple(dataclasses.asdict(cfg)["num_red_airbases"]) == (6, 8)


def test_run_config_records_the_effective_config_and_its_preset(tmp_path: Path) -> None:
    """PO1. `run_config.json` states the resolved config AND where it came from.

    Without the source, two runs of the same preset with different one-off flags are
    indistinguishable after the fact except by comparing every number by eye.
    """
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"n_iterations": 5, "base_seed": 11, "n_hidden": 2}),
                      encoding="utf-8")
    argv = ["--config", str(preset), "--seed", "99"]
    cfg, source = _resolve(argv, config_path=preset)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = write_run_config(run_dir, cfg, provenance={"stub": True},
                            config_source=source)
    payload = json.loads(path.read_text(encoding="utf-8"))

    # The EFFECTIVE config, with the override applied.
    assert payload["train_config"]["base_seed"] == 99
    assert payload["train_config"]["n_iterations"] == 5
    assert payload["train_config"]["n_hidden"] == 2
    assert payload["construction"]["n_hidden"] == 2

    # ... and the preset that produced it, by path, with what the CLI took back off it.
    recorded = payload["config_source"]
    assert recorded["path"] == str(preset)
    assert Path(recorded["absolute_path"]) == preset.resolve()
    assert recorded["format"] == "json"
    assert recorded["resolved_from"] == "config_file"
    assert set(recorded["config_fields"]) == {"n_iterations", "base_seed", "n_hidden"}
    assert recorded["cli_overrides"] == ["base_seed"]


def test_config_source_is_always_a_structured_object(tmp_path: Path) -> None:
    """PO1. ONE schema, THREE truthful kinds, and never `null`.

    `null` would have collapsed two different facts into one value: "this run used no
    preset" and "whoever wrote this file did not record where the config came from". But
    a single non-null fallback is not enough either -- see
    `test_a_direct_train_call_is_not_recorded_as_cli_provenance`: the three ways a config
    can arise must stay distinguishable under the same schema.

    Checked at every site that can produce the record, because a contract that only holds
    at one of them is not a contract.
    """
    expected_keys = {"path", "absolute_path", "format", "config_fields",
                     "cli_overrides", "resolved_from"}

    # (1) the constructor, for each kind. The shape never varies.
    cli = config_source_record(resolved_from="cli_defaults")
    direct = config_source_record(resolved_from="direct_config")
    for record in (cli, direct):
        assert set(record) == expected_keys
        assert record["path"] is None and record["absolute_path"] is None
        assert record["format"] is None
        assert record["config_fields"] == [] and record["cli_overrides"] == []
    assert cli["resolved_from"] == "cli_defaults"
    assert direct["resolved_from"] == "direct_config"
    assert cli != direct, "the two provenances must not be the same record"

    # (2) `resolve_train_config` on a CLI-only invocation -- a command line really did
    # resolve this one, so `cli_defaults` is the truthful value here.
    _cfg, source = _resolve(["--iterations", "2"])
    assert source == cli

    # (3) the preset case is the SAME shape, only differently filled.
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"n_iterations": 1}), encoding="utf-8")
    _cfg, file_source = _resolve(["--config", str(preset)], config_path=preset)
    assert set(file_source) == expected_keys
    assert file_source["resolved_from"] == "config_file"
    assert file_source["format"] == "json"
    assert file_source["path"] == str(preset)

    # (4) the record must be internally consistent: only a `config_file` may name a
    # file, and it MUST name one. A record that could claim a preset it cannot identify
    # is a provenance defect, so it fails at construction rather than reaching disk.
    for bad in (
        {"resolved_from": "not_a_kind"},
        {"resolved_from": "config_file"},                       # claims a file, names none
        {"resolved_from": "cli_defaults", "config_path": preset},
        {"resolved_from": "direct_config", "config_path": preset},
    ):
        try:
            config_source_record(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError("config_source_record accepted %r" % bad)


def test_a_direct_train_call_is_not_recorded_as_cli_provenance(tmp_path: Path) -> None:
    """PO1. A `TrainConfig` built in PYTHON is `direct_config`, never `cli_defaults`.

    This is a real repository path, not a hypothetical: `_selftest` calls `train(cfg)`
    directly three times, as would any notebook or script importing the trainer. Such a
    run never saw a command line, so recording it as `cli_defaults` would assert that
    argparse defaults resolved it -- a plausible-looking false statement in exactly the
    field a reader consults to find out what produced the run. A provenance value that
    can be wrong in a believable way is worse than one that is absent.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = TrainConfig(n_iterations=1, output_dir=str(run_dir))

    payload = json.loads(
        write_run_config(run_dir, cfg, provenance={"stub": True})
        .read_text(encoding="utf-8")
    )
    recorded = payload["config_source"]
    assert recorded is not None, "the fallback must still be structured"
    assert recorded["resolved_from"] == "direct_config"
    assert recorded["resolved_from"] != "cli_defaults", "a direct call claimed a CLI"
    assert recorded["path"] is None and recorded["config_fields"] == []

    # The three kinds are mutually exclusive and exhaustive at the schema level.
    kinds = {graph_train._CONFIG_SOURCE_CLI_DEFAULTS,
             graph_train._CONFIG_SOURCE_FILE,
             graph_train._CONFIG_SOURCE_DIRECT}
    assert len(kinds) == 3
    assert set(graph_train._CONFIG_SOURCE_KINDS) == kinds
    assert recorded["resolved_from"] in kinds

    # A caller that DID resolve a real source still wins -- the fallback only applies
    # when nothing was supplied.
    supplied = config_source_record(resolved_from="cli_defaults")
    payload = json.loads(
        write_run_config(run_dir, cfg, provenance={"stub": True},
                         config_source=supplied).read_text(encoding="utf-8")
    )
    assert payload["config_source"]["resolved_from"] == "cli_defaults"


def test_cli_exposes_config_and_plot_targets_the_plots_dir() -> None:
    """`--config` parses, and `--plot` documents where the figures land."""
    parser = _build_arg_parser()
    args = parser.parse_args(["--config", "some/preset.json", "--iterations", "1"])
    assert args.config == "some/preset.json"
    assert parser.parse_args(["--iterations", "1"]).config is None
    # `--config` is not a training knob and must never reach TrainConfig as a field.
    assert "config" not in {f.name for f in dataclasses.fields(TrainConfig)}



def test_main_hands_train_the_resolved_config_and_its_source(tmp_path: Path) -> None:
    """PO1. End to end through `main`: the preset reaches `train`, and so does its origin.

    Everything below `main` is stubbed -- no policy, no generator, no engine, no solver
    -- because what is under test is the WIRING: that `--config` is resolved once, that
    an explicit flag still wins at that level, and that `config_source` is handed to
    `train` (which records it in run_config.json) rather than being computed and dropped.
    """
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"n_iterations": 3, "base_seed": 11, "n_hidden": 2}),
                      encoding="utf-8")

    seen = {}

    def fake_train(cfg, *, config_source=None):
        seen["cfg"] = cfg
        seen["config_source"] = config_source
        return {"run_dir": str(tmp_path / "run")}

    def fake_plot(run_dir, **kwargs):
        seen["plotted"] = str(run_dir)
        return []

    real_train = graph_train.train
    real_plot = graph_train.plot_training_subprocess
    graph_train.train = fake_train                     # type: ignore[assignment]
    graph_train.plot_training_subprocess = fake_plot   # type: ignore[assignment]
    try:
        graph_train.main(["--config", str(preset), "--seed", "99"])
    finally:
        graph_train.train = real_train                 # type: ignore[assignment]
        graph_train.plot_training_subprocess = real_plot  # type: ignore[assignment]

    cfg = seen["cfg"]
    assert cfg.n_iterations == 3 and cfg.n_hidden == 2   # from the preset
    assert cfg.base_seed == 99                            # the explicit flag won
    source = seen["config_source"]
    assert source is not None, "train was called without the config source"
    assert source["path"] == str(preset)
    assert source["cli_overrides"] == ["base_seed"]
    # The figures are still drawn in the child, from the run directory `train` reported.
    assert seen["plotted"] == str(tmp_path / "run")



def test_main_with_argv_none_still_sees_the_real_command_line(tmp_path: Path) -> None:
    """PO1. `main()` called with NO argument resolves the REAL `sys.argv`, both passes.

    This is the ordinary invocation -- PyCharm, a terminal, `python -m ...` all reach
    `main()` with `argv=None` -- and it is where the override precedence silently broke:
    argparse reads `None` as `sys.argv[1:]`, so a probe pass that read `None` as `[]`
    concluded the operator had typed nothing and let the preset overwrite a flag that
    was really given. The regression is asserted on the VALUE that reaches `train`, not
    on the helper, because that is what a run would have been configured with.
    """
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"n_iterations": 3, "base_seed": 11, "n_hidden": 2}),
                      encoding="utf-8")

    seen = {}

    def fake_train(cfg, *, config_source=None):
        seen["cfg"] = cfg
        seen["config_source"] = config_source
        return {"run_dir": str(tmp_path / "run")}

    real_train = graph_train.train
    real_plot = graph_train.plot_training_subprocess
    real_argv = sys.argv
    graph_train.train = fake_train                        # type: ignore[assignment]
    graph_train.plot_training_subprocess = lambda run_dir, **kw: []   # type: ignore
    sys.argv = ["graph_train.py", "--config", str(preset), "--seed", "7"]
    try:
        graph_train.main()          # <- NO argv argument: the real entry point
    finally:
        graph_train.train = real_train                    # type: ignore[assignment]
        graph_train.plot_training_subprocess = real_plot  # type: ignore[assignment]
        sys.argv = real_argv

    cfg = seen["cfg"]
    assert cfg.base_seed == 7, "the typed --seed lost to the preset (finding 1)"
    assert cfg.n_iterations == 3 and cfg.n_hidden == 2, "the preset stopped applying"
    assert seen["config_source"]["cli_overrides"] == ["base_seed"]


def test_explicit_dests_read_argv_none_as_the_real_command_line() -> None:
    """The helper itself: `None` means `sys.argv[1:]`, exactly as argparse reads it.

    Unit-level companion to the entry-point test above -- it pins the CAUSE, so a future
    change that reintroduces `[] if argv is None` fails here with a readable message and
    not only through a stubbed `main`.
    """
    real_argv = sys.argv
    sys.argv = ["graph_train.py", "--iterations", "2", "--seed", "5"]
    try:
        assert _explicit_cli_dests(None) == {"iterations", "seed"}
        # An explicitly EMPTY list is still an empty command line -- the two must not be
        # conflated in the other direction either.
        assert _explicit_cli_dests([]) == set()
    finally:
        sys.argv = real_argv


def test_health_plot_exposes_per_cell_eval_denominators() -> None:
    """PO3. Every cell mean carries its own completion count.

    `eval_reward_mean_<cell>` is a mean over THAT cell's successful episodes, so when one
    cell fails more held-out seeds the curves are not averages over the same seeds and
    their gaps are not within-seed effects. The per-cell attempted/successful counts are
    what make that inspectable, and they come from existing record fields -- no
    evaluation semantics change.
    """
    source = inspect.getsource(graph_train._plot_measurement_health)
    assert '"eval_n_%s_%s" % (cell, suffix)' in source
    assert "attempted" in source and "successful" in source
    assert "_record_cells(eval_records)" in source
    # The abort-rate denominator is drawn too: a rate over one FD wake and the same rate
    # over eight are different findings, and only this panel shows which it was.
    assert '"eval_n_%s_fd_wakes" % cell' in source
    # It says WHY the panel exists: the cell means are not a within-seed comparison,
    # while the matched deltas are.
    assert "denominator" in source.lower()

    # The performance figure must label its cell series honestly, and reserve the
    # within-seed claim for the matched deltas.
    perf = inspect.getsource(graph_train._plot_training_performance)
    assert "mean over SUCCESSFUL forced_%s episodes" in perf
    assert "each mean over THAT cell's successful" in perf
    assert "WITHIN-SEED" in perf
    # The deltas still come from the declared matched keys only -- unchanged semantics.
    assert "_record_delta_keys(eval_records)" in perf


def test_per_condition_denominators_survive_an_asymmetric_round(tmp_path: Path) -> None:
    """PO3. With clean 2/2 and damaged 1/2 completed, both denominators are plottable.

    Data-level proof on the `_xy` series builder that feeds the panel: the asymmetry is
    present in the records and is picked up as two DISTINCT series, so a reader can see
    that the damaged mean rests on fewer seeds than the clean one.
    """
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    records = [json.loads(line) for line in
               (run_dir / "eval_records.jsonl").read_text(encoding="utf-8").splitlines()]

    clean_att = _xy(records, "updates_completed", "eval_n_clean_attempted")
    clean_ok = _xy(records, "updates_completed", "eval_n_clean_successful")
    dmg_att = _xy(records, "updates_completed", "eval_n_damaged_attempted")
    dmg_ok = _xy(records, "updates_completed", "eval_n_damaged_successful")

    assert clean_att[1] and dmg_att[1], "the per-condition denominators are missing"
    assert clean_att[1] == dmg_att[1], "both conditions attempt the same seeds"
    assert clean_ok[1] != dmg_ok[1], "the asymmetry the panel exists for is not present"
    # ... and the paired delta is still taken over COMPLETE pairs only, untouched.
    assert all(r["n_pairs_successful"] <= r["n_pairs_attempted"] for r in records)
    assert all(r["eval_paired_reward_delta"] is not None for r in records)



def test_probe_preset_claims_scheduled_iterations_not_productive_updates() -> None:
    """PO2. The preset promises two scheduled ITERATIONS, never two productive UPDATES.

    `updates_completed` advances only when the updater actually runs epochs, and a
    successful zero-wake iteration completes with `n_epochs_run == 0` and leaves it
    unchanged. So "after both updates" was a claim the schedule cannot guarantee -- and
    productive-update yield is one of the quantities this probe is run to MEASURE, which
    makes assuming it in the preset's own description exactly backwards.

    Asserted on the shipped file's prose because the prose is what a reader takes the
    run's shape from; the schedule fields are pinned separately and are unchanged.
    """
    raw = json.loads(_PROBE_PRESET.read_text(encoding="utf-8"))
    text = raw["_evaluation"]

    assert "after both updates" not in text, "the preset still promises two updates"
    assert "SCHEDULED TRAINING ITERATIONS" in text
    # It says what updates_completed actually depends on, and that 2 is not guaranteed.
    assert "0, 1 or 2" in text
    assert "productive" in text.lower()
    assert "no wakes" in text or "zero-wake" in text

    # The SCHEDULE itself is untouched -- this finding was about wording only.
    assert raw["n_iterations"] == 2
    assert raw["eval_every"] == 2
    assert raw["eval_episodes"] == 4
    assert raw["episodes_per_iteration"] == 4


# =============================================================================
# GENERALIZED-V1 TASK 4 -- harness selection, cardinality, benchmark, persistence
# =============================================================================

from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    SEVERITY_MILD,
    SEVERITY_SEVERE,
)
from match_aou.rl.training.graph_generalized import (  # noqa: E402
    BenchmarkIdentityError,
    BenchmarkManifestError,
    BENCHMARK_GROUP_SIZE,
    BENCHMARK_STRATUM_KEYS,
    EPISODE_DESIGN_FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1,
    LOAD_HIGH,
    LOAD_LOW,
    WorldIdentity,
    build_benchmark_manifest,
    sample_generalized_cardinality,
    write_benchmark_manifest,
)
from match_aou.rl.training.graph_reward import (  # noqa: E402
    REFERENCE_FAULT_MISSING,
    REFERENCE_FAULT_SOLVE_UNACCEPTABLE,
    ReferenceIntegrityError,
)

_GEN_MODE = FuelDamageMode.SEEDED_VARIABLE


def _gen_cfg(tmp_path: Path, **kwargs) -> TrainConfig:
    """A minimal GENERALIZED-V1 training config with evaluation OFF.

    Evaluation is off because a generalized run REQUIRES a frozen benchmark to evaluate,
    and the benchmark tests below drive `evaluate_benchmark` directly rather than through
    a whole training run.
    """
    base = dict(
        n_iterations=1, episodes_per_iteration=4, base_seed=0,
        output_dir=tmp_path / "gen", eval_every=0, checkpoint_every=0,
        episode_design=EPISODE_DESIGN_GENERALIZED_V1, fuel_damage_mode=_GEN_MODE,
    )
    base.update(kwargs)
    return TrainConfig(**base)


def _episode_cardinalities(events) -> list:
    """`(cardinality, was_the_keyword_passed)` for every stubbed attempt, in order."""
    return [(e[7], e[8]) for e in events if e[0] == "episode"]


def test_gen_a_fixed_cell_run_passes_no_generalized_keyword(tmp_path: Path) -> None:
    """PO1. The historical path calls `_run_one_episode` EXACTLY as it did before.

    The keyword-OMISSION claim, not a "passes None" claim: a fixed-cell run must not
    acquire a new argument at all, which is both the stronger invariance statement and
    what keeps the independent stubs in the sibling test files working. The same
    discipline `_artifact_kwargs` and `_ctde_kwargs` already hold themselves to.
    """
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=3, base_seed=0,
        output_dir=tmp_path / "fixed", eval_every=0, checkpoint_every=0,
    )
    _summary, events, _state = _run_stub_training(cfg, wakes_per_episode=1)

    passed = _episode_cardinalities(events)
    assert passed, "the stub recorded no attempt"
    assert all(was_passed is False for _card, was_passed in passed), (
        "a fixed_cell_v1 run must not pass a cardinality keyword at all"
    )
    # And the two setup keywords are omitted for the same reason.
    assert graph_train._generalized_setup_kwargs(cfg) == {}
    assert graph_train._cardinality_kwargs(None) == {}


def test_gen_a_generalized_run_passes_the_sampled_cardinality(tmp_path: Path) -> None:
    """PO1. Every generalized training attempt carries the cell its SEED determines.

    Re-derived independently from the seed through the public sampler, so the loop is
    proven to pass the sampler's answer rather than something it computed its own way.
    """
    cfg = _gen_cfg(tmp_path, episodes_per_iteration=6)
    _summary, events, _state = _run_stub_training(cfg, wakes_per_episode=1)

    seeds = _episode_seeds(events, "train")
    passed = _episode_cardinalities(events)
    assert len(seeds) == 6 and len(passed) == 6
    for seed, (card, was_passed) in zip(seeds, passed):
        assert was_passed is True, "a generalized run must pass its cardinality"
        expected = sample_generalized_cardinality(episode_seed=int(seed))
        assert card == expected, seed
        assert card.known_count == card.agent_count
        assert 1 <= card.hidden_requested <= card.agent_count

    # The two setup keywords ARE supplied on this path, as the whole bundle.
    kwargs = graph_train._generalized_setup_kwargs(cfg)
    assert kwargs == {
        "hidden_policy": cfg.design.hidden_policy,
        "reference_policy": cfg.design.reference_policy,
    }


def test_gen_the_fd_parameters_carry_the_designs_two_policies() -> None:
    """PO1. The FD seams are resolved from the ONE selector, never set independently.

    And on the historical path the constructed object is IDENTICAL to the pre-Task-4 one
    -- proven by equality against a parameter object built with no policy arguments at
    all, which is what every pre-Task-4 call site produced.
    """
    fixed = TrainConfig(n_iterations=1)
    historical = FuelDamageParameters(
        mode=fixed.fuel_damage_mode,
        probability=fixed.fuel_damage_probability,
        leg_progress_threshold=fixed.fuel_damage_leg_progress,
        rtb_safety_margin=fixed.fuel_damage_rtb_margin,
        mild_probability=fixed.fuel_damage_mild_probability,
    )
    assert fixed.fuel_damage_parameters() == historical

    gen = TrainConfig(n_iterations=1, episode_design=EPISODE_DESIGN_GENERALIZED_V1,
                      fuel_damage_mode=_GEN_MODE)
    params = gen.fuel_damage_parameters()
    assert params.eligibility_policy == gen.design.eligibility_policy
    assert params.post_fd_wake_policy == gen.design.post_fd_wake_policy
    assert params.certified_eligibility is True
    assert params.completion_boundary_wakes is True


def test_gen_validate_refuses_a_half_configured_generalized_run(tmp_path: Path) -> None:
    """PO1/PO2. Every way of reaching a MISLABELLED generalized run is refused.

    Each of these would produce a run whose records claim the generalized design while
    measuring something else -- which is the failure mode the selector exists to prevent.
    """
    def _refuses(**kwargs) -> str:
        cfg = TrainConfig(n_iterations=1, output_dir=tmp_path / "x", **kwargs)
        try:
            cfg.validate()
        except ValueError as exc:
            return str(exc)
        raise AssertionError("accepted: %r" % (kwargs,))

    # An unknown design.
    assert "episode_design must be one of" in _refuses(episode_design="generalized")
    # Generalized on a mixture that makes every damaged episode structurally severe.
    assert "seeded_variable" in _refuses(
        episode_design=EPISODE_DESIGN_GENERALIZED_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_MIXTURE, eval_every=0)
    # Generalized WITH evaluation but no frozen benchmark: the held-out band carries no
    # stratum, so evaluating on it would measure an unstratified population.
    assert "benchmark_manifest" in _refuses(
        episode_design=EPISODE_DESIGN_GENERALIZED_V1, fuel_damage_mode=_GEN_MODE,
        eval_every=1, eval_episodes=2)
    # A benchmark on a FIXED-CELL run: every world would come from the fixed cell while
    # the record carried stratum labels the run never varied.
    assert "18-stratum" in _refuses(benchmark_manifest="somewhere.json")

    # ... and the two legitimate shapes are accepted.
    TrainConfig(n_iterations=1, output_dir=tmp_path / "ok1").validate()
    TrainConfig(n_iterations=1, output_dir=tmp_path / "ok2", eval_every=0,
                episode_design=EPISODE_DESIGN_GENERALIZED_V1,
                fuel_damage_mode=_GEN_MODE).validate()


def test_gen_short_realization_is_accounted_and_a_lying_audit_aborts() -> None:
    """PO1/PO3. A legitimately short world is ACCOUNTED; a contradictory audit ABORTS.

    The whole point of bounded backoff is that realizing fewer hidden targets than were
    requested is a recorded OUTCOME, not a failure -- so the roster expectation must
    follow the construction's own realized count. What must NOT be tolerated is an audit
    that disagrees with the schedule that produced it: that means the episode and its
    accounting describe different worlds.
    """
    card = graph_train.episode_cardinality(
        TrainConfig(n_iterations=1, episode_design=EPISODE_DESIGN_GENERALIZED_V1,
                    fuel_damage_mode=_GEN_MODE),
        seed=0,
        benchmark_cardinality=graph_train.fixed_cell_cardinality(
            agent_count=4, known_count=4, hidden_requested=4),
    )

    class _Audit:
        def __init__(self, agents, requested, realized):
            self.agent_count = agents
            self.hidden_requested = requested
            self.hidden_realized = realized

    # Realized SHORT: expected hidden follows the audit, the REQUEST is preserved, and
    # the shortfall is flagged rather than repaired.
    short = graph_train._scheduled_cell(card, _Audit(4, 4, 2))
    assert (short.n_known, short.n_hidden, short.n_targets_executed) == (4, 2, 6)
    assert short.hidden_requested == 4, "the request must never be rewritten"
    assert short.realized_short is True

    # Realized in FULL.
    full = graph_train._scheduled_cell(card, _Audit(4, 4, 4))
    assert (full.n_hidden, full.realized_short) == (4, False)

    # An audit that contradicts the schedule is a measurement-integrity fault.
    for bad in (_Audit(3, 4, 2), _Audit(4, 2, 2)):
        exc = _integrity_raise(graph_train._scheduled_cell, card, bad)
        assert "contradicts the schedule" in str(exc)


def test_gen_the_outcome_record_states_the_design_and_the_cardinality(
    tmp_path: Path,
) -> None:
    """PO3. `episode_outcomes.jsonl` carries the design and the world, on BOTH paths.

    The durable per-attempt stream is where a distributional question is answered, so the
    design, the requested-vs-realized cardinality and the policy ids must be readable per
    episode rather than only from the run config.
    """
    gen = _gen_cfg(tmp_path, episodes_per_iteration=3)
    _run_stub_training(gen, wakes_per_episode=1)
    rows = _read_records(gen.output_dir, "episode_outcomes.jsonl")
    assert len(rows) == 3
    for row in rows:
        assert row["schema_version"] == 2
        assert row["episode_design"] == EPISODE_DESIGN_GENERALIZED_V1
        assert row["generalized"] is True
        assert row["hidden_policy"] == gen.design.hidden_policy
        assert row["eligibility_policy"] == gen.design.eligibility_policy
        assert row["post_fd_wake_policy"] == gen.design.post_fd_wake_policy
        assert row["reference_policy"] == gen.design.reference_policy
        assert row["target_destruction_probability"] == 1.0
        expected = sample_generalized_cardinality(episode_seed=int(row["seed"]))
        assert row["agent_count"] == expected.agent_count
        assert row["known_requested"] == expected.known_count
        assert row["hidden_requested"] == expected.hidden_requested
        assert row["cardinality_source"] == "generalized_sampler"
        # The benchmark identity keys exist on EVERY row, as null on a non-benchmark one.
        assert row["benchmark_stratum"] is None
        assert row["benchmark_manifest_id"] is None

    # The HISTORICAL path states its own design just as explicitly -- a reader must never
    # have to infer it from the absence of keys.
    fixed = TrainConfig(n_iterations=1, episodes_per_iteration=2, base_seed=0,
                        output_dir=tmp_path / "fixed", eval_every=0, checkpoint_every=0)
    _run_stub_training(fixed, wakes_per_episode=1)
    fixed_rows = _read_records(fixed.output_dir, "episode_outcomes.jsonl")
    assert fixed_rows and all(
        r["episode_design"] == EPISODE_DESIGN_FIXED_CELL_V1
        and r["generalized"] is False
        and r["reference_policy"] == "static_t0_v1"
        for r in fixed_rows
    )


def test_gen_the_failure_ledger_keeps_the_scheduled_world(tmp_path: Path) -> None:
    """PO3. A FAILED attempt still records the world it was scheduled to build.

    It matters precisely because it never built one: without the scheduled cardinality a
    failed HIGH-load attempt would be invisible in the requested-vs-realized
    distribution, and that stratum's denominator would quietly shrink to the attempts
    that happened to succeed.
    """
    cfg = _gen_cfg(tmp_path, episodes_per_iteration=4)
    summary, _events, _state = _run_stub_training(
        cfg, wakes_per_episode=1, failures={1: ("setup", "no strict window")})

    ledger = _read_records(cfg.output_dir, "episode_failures.jsonl")
    assert len(ledger) == 1
    row = ledger[0]
    expected = sample_generalized_cardinality(episode_seed=int(row["seed"]))
    assert row["agent_count"] == expected.agent_count
    assert row["hidden_requested"] == expected.hidden_requested
    assert row["cardinality_source"] == "generalized_sampler"
    assert row["pipeline_stage"] == "setup"
    assert row["cell"] in ("clean", "mild", "severe")
    assert row["reference_fault_reason"] is None
    # Disjoint streams: the failed attempt appears in NEITHER outcome file.
    outcomes = _read_records(cfg.output_dir, "episode_outcomes.jsonl")
    assert row["seed"] not in [o["seed"] for o in outcomes]
    assert summary["train_episodes_attempted"] == 4
    assert summary["train_episodes_successful"] == 3
    assert summary["train_episodes_failed"] == 1
    assert summary["accounting_reconciled"] is True


def test_gen_the_summary_reports_requested_vs_realized(tmp_path: Path) -> None:
    """PO3. The roll-up is DERIVED from the canonical jsonl, with explicit denominators.

    And it reports the requested-vs-realized DISTRIBUTION without judging it: no verdict
    key, no threshold, no "degenerate" flag -- the handoff requires a human to inspect
    that distribution, and a threshold invented here would pre-empt that decision.
    """
    cfg = _gen_cfg(tmp_path, episodes_per_iteration=6)
    summary, _events, _state = _run_stub_training(
        cfg, wakes_per_episode=1, failures={2: ("setup", "boom")})

    block = summary["generalized"]
    assert summary["episode_design"] == EPISODE_DESIGN_GENERALIZED_V1
    assert summary["episode_design_policies"]["hidden_policy"] == \
        cfg.design.hidden_policy
    assert block["episode_design"] == EPISODE_DESIGN_GENERALIZED_V1
    assert block["cardinality_sampler"]["rng_domain"] == "generalized_cardinality_v1"

    # attempted == successful + failed in EVERY bucket, by construction.
    for key in ("train_by_agent_count", "train_by_hidden_requested"):
        buckets = block[key]
        assert buckets, key
        for name, entry in buckets.items():
            assert entry["n_attempted"] == entry["n_successful"] + entry["n_failed"], (
                key, name)
        assert sum(e["n_attempted"] for e in buckets.values()) == 6

    # The distribution is REPORTED, and nothing in the block renders a VERDICT on it:
    # no threshold, no "degenerate" flag, no accept/reject decision. Checked over the
    # KEY names (a rejection-REASON tally is data about the backoff, not a verdict on
    # the benchmark, so the word "rejections" is legitimate and appears as a key).
    assert "cardinality_requested_vs_realized" in block

    def _keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k).lower()
                for sub in _keys(v):
                    yield sub

    for key in _keys(block):
        # `n_solver_accepted` is deliberately NOT in this list: whether a SOLVER
        # reached acceptable optimality is solver health, not a verdict on the
        # benchmark's stratification.
        for verdict_word in ("degenerate", "collapse", "threshold", "verdict",
                             "benchmark_ok", "should_reject"):
            assert verdict_word not in key, (key, verdict_word)


_BENCH_CELL_BY_MODE = {
    FuelDamageMode.FORCED_CLEAN: CONDITION_CLEAN,
    FuelDamageMode.FORCED_MILD: SEVERITY_MILD,
    FuelDamageMode.FORCED_SEVERE: SEVERITY_SEVERE,
}


def _bench_outcome(mode, hidden_realized=1, identity=None, reward=-0.25):
    """A stub `_EpisodeOutcome` shaped like a completed benchmark member.

    The plan reports the cell its FORCED MODE really produces, so the tally's
    scheduled-vs-executed guard sees agreement and these tests exercise the benchmark
    accounting rather than that guard's refusal.
    """
    cell = _BENCH_CELL_BY_MODE[mode]
    damaged = cell != CONDITION_CLEAN
    condition = CONDITION_DAMAGED if damaged else CONDITION_CLEAN
    return _EpisodeOutcome(
        trajectory=[_StubTransition("ego_0", 0)], reward=reward, ticks=10,
        ended="done", n_wakes=1, confirmed_kills=1, n_dead=0, seconds=0.01,
        targets_confirmed_unique=1, targets_total=1 + hidden_realized,
        known_target_names=("K",), hidden_target_names=("H",),
        known_confirmed_names=("K",), hidden_confirmed_names=(),
        fuel_damage_plan={"condition": condition,
                          "severity": cell if damaged else None,
                          "ego_id": "ego_0" if damaged else None},
        fuel_damage_outcome={"condition": condition,
                             "severity": cell if damaged else None,
                             "fired": damaged, "wake_occurred": damaged,
                             "wake_meta_action": None},
        selected_ego_rtb_issued=True if damaged else None,
        hidden_realized=hidden_realized,
        world_identity=identity,
    )


def _run_benchmark_round(cfg, manifest, tmp_path, *, body):
    """Drive the REAL `evaluate_benchmark` with a stubbed episode body."""
    saved = graph_train._run_one_episode
    graph_train._run_one_episode = body
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return graph_train.evaluate_benchmark(
                None, object(), cfg, manifest,
                iteration=None, stage="post_update", updates_completed=1,
                round_ordinal=0,
                failures_path=tmp_path / "episode_failures.jsonl",
                outcomes_path=tmp_path / "episode_outcomes.jsonl",
            )
    finally:
        graph_train._run_one_episode = saved


def _one_world_manifest():
    """A one-world-per-base-cell manifest -- 6 groups, 18 member episodes, 18 strata."""
    return build_benchmark_manifest(worlds_per_cell=1, benchmark_base_seed=2_000_000)


def test_gen_benchmark_round_reports_all_18_strata_with_denominators(
    tmp_path: Path,
) -> None:
    """PO2. A manifest round accounts EVERY requested stratum, not only the ones that ran.

    Per-stratum denominators are the point: a mean over `severe` pools six team-size /
    load combinations, which is exactly the pooling the stratification exists to avoid.
    """
    cfg = _gen_cfg(tmp_path, eval_every=0)
    manifest = _one_world_manifest()
    seen = []

    def body(policy, gen, cfg_, *, seed, episode_tag, deterministic,
             fuel_damage_mode=None, cardinality=None, **extra):
        seen.append((seed, fuel_damage_mode, cardinality))
        return _bench_outcome(
            fuel_damage_mode,
            hidden_realized=cardinality.hidden_requested,
            identity=WorldIdentity(
                hidden_realized=cardinality.hidden_requested,
                known_realized=cardinality.known_count,
                geometric_fingerprint=((float(seed), 1.0),),
                fd_selected_ordinal=0, fd_certificate_fingerprint="c%d" % seed,
            ),
        )

    record = _run_benchmark_round(cfg, manifest, tmp_path, body=body)

    assert record["eval_population"] == "benchmark_manifest"
    assert record["benchmark_manifest_id"] == manifest.manifest_id
    assert record["n_attempted"] == manifest.n_members == 18
    assert record["n_successful"] == 18 and record["n_failed"] == 0
    assert record["n_groups_attempted"] == 6 == record["n_groups_successful"]

    strata = record["benchmark_strata"]
    assert set(strata) == set(BENCHMARK_STRATUM_KEYS)
    assert len(strata) == 18
    for key, entry in strata.items():
        assert entry["n_attempted"] == 1 and entry["n_successful"] == 1
        assert entry["success_fraction"] == 1.0
        assert entry["reward_mean"] is not None
        assert entry["n_short_realized"] == 0
        assert entry["hidden_realized_histogram"], key

    # Every member of a group ran the SAME seed and the SAME requested cardinality, and
    # the three differ ONLY in the forced mode.
    by_seed = {}
    for seed, mode, card in seen:
        by_seed.setdefault(seed, []).append((mode, card))
    assert len(by_seed) == 6
    for members in by_seed.values():
        assert len(members) == BENCHMARK_GROUP_SIZE == 3
        assert len({m for m, _c in members}) == 3
        assert len({c for _m, c in members}) == 1, "one cardinality per world group"

    # LOW requests 1 hidden target, HIGH requests A -- readable per stratum.
    low = [k for k in strata if ("-%s-" % LOW_KEY) in k]
    assert low and all(strata[k]["hidden_requested_total"] == 1 for k in low)
    high = [k for k in strata if ("-%s-" % HIGH_KEY) in k]
    assert high and all(strata[k]["hidden_requested_total"] >= 2 for k in high)


LOW_KEY = LOAD_LOW
HIGH_KEY = LOAD_HIGH


def test_gen_benchmark_deltas_use_complete_groups_only(tmp_path: Path) -> None:
    """PO2. A group with a failed member contributes to NO within-world delta.

    Nor is it repaired from its surviving members: a clean+mild pair inside a failed
    triad yields no mild-minus-clean delta either, because the GROUP it belonged to is
    incomplete. It stays visible in the attempt counts.
    """
    cfg = _gen_cfg(tmp_path, eval_every=0)
    manifest = _one_world_manifest()
    doomed = manifest.worlds[0].seed
    rewards = {"clean": -0.1, "mild": -0.2, "severe": -0.5}

    def body(policy, gen, cfg_, *, seed, episode_tag, deterministic,
             fuel_damage_mode=None, cardinality=None, **extra):
        cell = {"forced_clean": "clean", "forced_mild": "mild",
                "forced_severe": "severe"}[fuel_damage_mode]
        if seed == doomed and cell == "severe":
            raise EpisodeAttemptError("setup", ValueError("no strict window"))
        return _bench_outcome(
            fuel_damage_mode,
            hidden_realized=cardinality.hidden_requested, reward=rewards[cell],
            identity=WorldIdentity(
                hidden_realized=cardinality.hidden_requested,
                known_realized=cardinality.known_count,
                geometric_fingerprint=((float(seed), 1.0),),
                fd_selected_ordinal=0, fd_certificate_fingerprint="c%d" % seed,
            ),
        )

    record = _run_benchmark_round(cfg, manifest, tmp_path, body=body)

    assert record["n_attempted"] == 18
    assert record["n_successful"] == 17 and record["n_failed"] == 1
    # FIVE complete groups, not six -- the failed member's group contributes to none.
    assert record["n_groups_attempted"] == 6
    assert record["n_groups_successful"] == 5
    for key in record["benchmark_delta_keys"]:
        assert record["%s_n" % key] == 5, key
    for key, expected in (("eval_delta_severe_minus_clean", -0.4),
                          ("eval_delta_mild_minus_clean", -0.1),
                          ("eval_delta_severe_minus_mild", -0.3)):
        assert abs(record[key] - expected) < 1e-9, (key, record[key])

    # The surviving members of the incomplete group ARE still counted as episodes.
    strata = record["benchmark_strata"]
    doomed_world = manifest.worlds[0]
    assert strata[doomed_world.stratum_key("clean")]["n_successful"] == 1
    assert strata[doomed_world.stratum_key("severe")]["n_successful"] == 0
    assert strata[doomed_world.stratum_key("severe")]["n_failed"] == 1
    assert strata[doomed_world.stratum_key("severe")]["reward_mean"] is None
    # NOTHING was substituted for the failed member.
    ledger = _read_records(tmp_path, "episode_failures.jsonl")
    assert len(ledger) == 1
    assert ledger[0]["seed"] == doomed
    assert ledger[0]["benchmark_stratum"] == doomed_world.stratum_key("severe")
    assert ledger[0]["benchmark_group_key"] == doomed_world.key


def test_gen_a_matched_group_that_built_two_worlds_aborts(tmp_path: Path) -> None:
    """PO2. Members that disagree about their world are an INSTRUMENT fault.

    Their delta would be a between-worlds comparison wearing a within-world label, so the
    round stops rather than reporting it.
    """
    cfg = _gen_cfg(tmp_path, eval_every=0)
    manifest = _one_world_manifest()
    target = manifest.worlds[0].seed

    def body(policy, gen, cfg_, *, seed, episode_tag, deterministic,
             fuel_damage_mode=None, cardinality=None, **extra):
        # The SEVERE member of one world reports a different certified ego ordinal.
        ordinal = 3 if (seed == target and fuel_damage_mode == "forced_severe") else 0
        return _bench_outcome(
            fuel_damage_mode,
            hidden_realized=cardinality.hidden_requested,
            identity=WorldIdentity(
                hidden_realized=cardinality.hidden_requested,
                known_realized=cardinality.known_count,
                geometric_fingerprint=((float(seed), 1.0),),
                fd_selected_ordinal=ordinal,
                fd_certificate_fingerprint="c%d" % seed,
            ),
        )

    try:
        _run_benchmark_round(cfg, manifest, tmp_path, body=body)
    except BenchmarkIdentityError as exc:
        assert "did not produce the same world" in str(exc)
    else:
        raise AssertionError("a mismatched matched group was accepted")

    # An instrument fault is NEVER written to the scientific ledger.
    assert not (tmp_path / "episode_failures.jsonl").exists() or not _read_records(tmp_path, "episode_failures.jsonl")


def test_gen_a_frozen_preflight_mismatch_aborts(tmp_path: Path) -> None:
    """PO2. A preflighted world that comes out different is REFUSED, not substituted."""
    cfg = _gen_cfg(tmp_path, eval_every=0)
    preflight = {
        "hidden_realized": 99, "known_realized": 99,
        "geometric_fingerprint": [[0.0, 0.0]],
        "fd_selected_ordinal": 0, "fd_certificate_fingerprint": "frozen",
    }
    from match_aou.rl.training.graph_generalized import BENCHMARK_BASE_CELLS
    manifest = build_benchmark_manifest(worlds=[
        {"agent_count": a, "load_bucket": bucket, "seed": 3_000_000 + i,
         **({"preflight": preflight} if i == 0 else {})}
        for i, (a, bucket) in enumerate(BENCHMARK_BASE_CELLS)
    ])

    def body(policy, gen, cfg_, *, seed, episode_tag, deterministic,
             fuel_damage_mode=None, cardinality=None, **extra):
        return _bench_outcome(
            fuel_damage_mode,
            hidden_realized=cardinality.hidden_requested,
            identity=WorldIdentity(
                hidden_realized=cardinality.hidden_requested,
                known_realized=cardinality.known_count,
                geometric_fingerprint=((1.0, 2.0),),
                fd_selected_ordinal=0, fd_certificate_fingerprint="observed",
            ),
        )

    try:
        _run_benchmark_round(cfg, manifest, tmp_path, body=body)
    except BenchmarkIdentityError as exc:
        assert "frozen manifest" in str(exc)
        assert "REFUSED" in str(exc)
    else:
        raise AssertionError("a world differing from its frozen manifest was accepted")


def test_gen_a_tampered_manifest_never_reaches_a_run(tmp_path: Path) -> None:
    """PO2. `train` verifies the frozen population BEFORE it creates anything.

    A manifest that fails its own content hash must cost nothing and leave nothing
    behind -- not a run directory, not a config, not a ledger.
    """
    manifest = _one_world_manifest()
    path = write_benchmark_manifest(manifest, tmp_path / "bench.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["worlds"][0]["seed"] = 999_999
    path.write_text(json.dumps(record), encoding="utf-8")

    run_dir = tmp_path / "run"
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=1, base_seed=0, output_dir=run_dir,
        eval_every=1, eval_episodes=1, checkpoint_every=0,
        episode_design=EPISODE_DESIGN_GENERALIZED_V1, fuel_damage_mode=_GEN_MODE,
        benchmark_manifest=str(path),
    )
    try:
        _run_stub_training(cfg, wakes_per_episode=1)
    except BenchmarkManifestError as exc:
        assert "identity MISMATCH" in str(exc)
    else:
        raise AssertionError("a tampered manifest was accepted by train()")
    assert not run_dir.exists(), "a refused manifest must leave no run directory"


def test_gen_run_config_records_the_design_the_sampler_and_the_manifest(
    tmp_path: Path,
) -> None:
    """PO3. `run_config.json` states the population, its rule, and the frozen benchmark.

    The manifest's content HASH is what makes "the two arms ran the same benchmark" a
    checkable claim rather than an assertion.
    """
    manifest = _one_world_manifest()
    path = write_benchmark_manifest(manifest, tmp_path / "bench.json")
    cfg = TrainConfig(
        n_iterations=1, episodes_per_iteration=1, base_seed=0,
        output_dir=tmp_path / "run", eval_every=1, eval_episodes=1, checkpoint_every=0,
        episode_design=EPISODE_DESIGN_GENERALIZED_V1, fuel_damage_mode=_GEN_MODE,
        benchmark_manifest=str(path),
    )
    _run_stub_training(cfg, wakes_per_episode=1)
    payload = json.loads(
        (Path(cfg.output_dir) / "run_config.json").read_text(encoding="utf-8"))

    block = payload["episode_design"]
    assert block["design"] == EPISODE_DESIGN_GENERALIZED_V1
    assert block["generalized"] is True
    assert block["hidden_policy"] == cfg.design.hidden_policy
    assert block["reference_policy"] == cfg.design.reference_policy
    assert block["target_destruction_probability"] == 1.0
    assert block["cardinality_sampler"]["policy"] == \
        "generalized_cardinality_uniform_v1"
    assert block["fixed_cell"] is None
    assert block["benchmark_manifest"]["manifest_id"] == manifest.manifest_id
    assert block["benchmark_manifest"]["n_strata"] == 18
    assert block["benchmark_manifest"]["n_worlds"] == manifest.n_worlds

    # The HISTORICAL run records the same block truthfully, the other way round.
    fixed = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                        output_dir=tmp_path / "fixed", eval_every=0, checkpoint_every=0)
    _run_stub_training(fixed, wakes_per_episode=1)
    fixed_block = json.loads(
        (Path(fixed.output_dir) / "run_config.json").read_text(encoding="utf-8")
    )["episode_design"]
    assert fixed_block["design"] == EPISODE_DESIGN_FIXED_CELL_V1
    assert fixed_block["generalized"] is False
    assert fixed_block["cardinality_sampler"] is None, (
        "a configured cell was not sampled, and must not claim a sampler"
    )
    assert fixed_block["fixed_cell"] == {"num_agents": 3, "n_known": 3, "n_hidden": 3}
    assert fixed_block["benchmark_manifest"] is None


def test_gen_reference_attrition_is_accounted_and_contradiction_aborts() -> None:
    """PO3. The reference-fault routing, through the REAL `_run_one_episode`.

    An unanswered solve is a fact about ONE attempt: it is wrapped as an ordinary
    `reward`-stage episode failure and enters `skip_and_account_v1`. An instrument
    contradiction implicates every episode the reference layer touched, so it propagates
    unwrapped and ABORTS. Neither can ever become a successful zero-reference episode.
    """
    ctx = _reference_ctx(done=set())

    def _raise(reason):
        def boom(*a, **k):
            raise ReferenceIntegrityError("identical text", reason=reason)

        return _run_one_episode_against(ctx, reward=boom)

    # (a) ATTRITION: wrapped, staged, and therefore accountable.
    try:
        _raise(REFERENCE_FAULT_SOLVE_UNACCEPTABLE)
    except EpisodeAttemptError as exc:
        assert exc.stage == "reward"
        assert isinstance(exc.original, ReferenceIntegrityError)
        assert exc.original.reason == REFERENCE_FAULT_SOLVE_UNACCEPTABLE
    else:
        raise AssertionError("an unanswered solve must be accounted, not silent")

    # (b) INTEGRITY: unwrapped, so both attempt handlers re-raise it and the run stops.
    try:
        _raise(REFERENCE_FAULT_MISSING)
    except EpisodeAttemptError:
        raise AssertionError("an instrument contradiction must NOT be accounted")
    except ReferenceIntegrityError as exc:
        assert exc.reason == REFERENCE_FAULT_MISSING
        assert exc.is_measurement_integrity is True
    else:
        raise AssertionError("an instrument contradiction must abort")


def test_gen_cli_and_rollout_expose_the_selector_without_drift() -> None:
    """PO1. Both harnesses expose the selector, and neither CLI default drifts.

    The defaults are read OFF the dataclasses rather than restated as literals, which is
    the same anti-drift discipline every other flag in this parser holds to.
    """
    from match_aou.rl.training import graph_rollout as gr

    train_args = graph_train._build_arg_parser().parse_args(["--iterations", "1"])
    assert train_args.episode_design == TrainConfig(n_iterations=1).episode_design
    assert train_args.benchmark_manifest == TrainConfig(n_iterations=1).benchmark_manifest

    roll_args = gr._build_arg_parser().parse_args([])
    assert roll_args.episode_design == gr.RolloutConfig().episode_design
    assert roll_args.fuel_damage_mode == gr.RolloutConfig().fuel_damage_mode

    # The selector is a real choice on both, and both DEFAULT to the historical design.
    for parser in (graph_train._build_arg_parser(), gr._build_arg_parser()):
        actions = {a.dest: a for a in parser._actions}
        assert set(actions["episode_design"].choices) == {
            EPISODE_DESIGN_FIXED_CELL_V1, EPISODE_DESIGN_GENERALIZED_V1
        }
        assert actions["episode_design"].default == EPISODE_DESIGN_FIXED_CELL_V1

    # A preset may set it, through the ONE dest->field mapping.
    assert graph_train._CLI_FIELD_BY_DEST["episode_design"] == "episode_design"
    assert graph_train._CLI_FIELD_BY_DEST["benchmark_manifest"] == "benchmark_manifest"

    # And a generalized ROLLOUT stays diagnostic: it exposes no benchmark at all.
    assert not hasattr(gr.RolloutConfig(), "benchmark_manifest")


if __name__ == "__main__":
    import tempfile

    failures = 0
    tests = [
        ("checkpoint_round_trip", test_checkpoint_round_trip, True),
        ("global_episode_index_and_train_seed",
         test_global_episode_index_and_train_seed, False),
        ("eval_seed_band_is_fixed_and_disjoint",
         test_eval_seed_band_is_fixed_and_disjoint, False),
        ("validate_rejects_overlapping_seed_bands",
         test_validate_rejects_overlapping_seed_bands, False),
        ("validate_rejects_degenerate_shapes",
         test_validate_rejects_degenerate_shapes, False),
        ("plot_training_writes_three_figures_into_the_plots_dir",
         test_plot_training_writes_three_figures_into_the_plots_dir, True),
        ("plot_training_without_eval_records",
         test_plot_training_without_eval_records, True),
        ("plot_training_missing_records_is_a_clean_noop",
         test_plot_training_missing_records_is_a_clean_noop, True),
        ("defaults_are_the_phase_a_baseline_cell",
         test_defaults_are_the_phase_a_baseline_cell, False),
        ("split_preview_covers_both_ends_of_a_range",
         test_split_preview_covers_both_ends_of_a_range, False),
        ("derived_split_matches_real_split_tasks",
         test_derived_split_matches_real_split_tasks, False),
        ("truncation_trap_is_pinned", test_truncation_trap_is_pinned, False),
        ("parse_airbase_range_accepts_int_and_range",
         test_parse_airbase_range_accepts_int_and_range, False),
        ("parse_airbase_range_rejects_bad_values",
         test_parse_airbase_range_rejects_bad_values, False),
        ("cli_parses_scenario_knobs", test_cli_parses_scenario_knobs, False),
        ("cli_defaults_equal_the_dataclass_defaults",
         test_cli_defaults_equal_the_dataclass_defaults, False),
        ("hidden_zero_warns_and_validate_still_passes",
         test_hidden_zero_warns_and_validate_still_passes, False),
        ("known_below_three_warns_about_the_bonmin_stall",
         test_known_below_three_warns_about_the_bonmin_stall, False),
        ("default_config_emits_no_hazard_warning",
         test_default_config_emits_no_hazard_warning, False),
        ("warnings_use_the_low_end_of_the_range",
         test_warnings_use_the_low_end_of_the_range, False),
        ("write_run_config_records_the_scenario_knobs",
         test_write_run_config_records_the_scenario_knobs, True),
        ("construction_defaults_are_the_reference_cell",
         test_construction_defaults_are_the_reference_cell, False),
        ("build_variation_config_requests_a_known_only_world",
         test_build_variation_config_requests_a_known_only_world, False),
        ("both_harnesses_call_setup_in_construction_mode",
         test_both_harnesses_call_setup_in_construction_mode, False),
        ("validate_rejects_more_agents_than_targets",
         test_validate_rejects_more_agents_than_targets, False),
        ("validate_rejects_degenerate_construction_values",
         test_validate_rejects_degenerate_construction_values, False),
        ("both_configs_reject_sams_on_the_construction_path",
         test_both_configs_reject_sams_on_the_construction_path, False),
        ("low_n_known_warns_about_the_bonmin_stall",
         test_low_n_known_warns_about_the_bonmin_stall, False),
        ("cli_exposes_the_construction_cell",
         test_cli_exposes_the_construction_cell, False),
        ("write_run_config_separates_generated_from_executed",
         test_write_run_config_separates_generated_from_executed, True),
        ("rollout_config_mirrors_the_train_reference_cell",
         test_rollout_config_mirrors_the_train_reference_cell, False),
        ("rollout_config_validate_accepts_the_reference_cell",
         test_rollout_config_validate_accepts_the_reference_cell, False),
        ("rollout_config_validate_rejects_invalid_cells",
         test_rollout_config_validate_rejects_invalid_cells, False),
        ("run_rollout_validates_before_touching_anything",
         test_run_rollout_validates_before_touching_anything, True),
        # --- B4: provenance, skip-and-account, pre-update eval, derived artifacts ---
        ("provenance_block_is_complete_and_explicit",
         test_provenance_block_is_complete_and_explicit, False),
        ("git_provenance_reports_absence_explicitly",
         test_git_provenance_reports_absence_explicitly, True),
        ("probe_command_survives_non_utf8_output",
         test_probe_command_survives_non_utf8_output, False),
        ("seed_bands_are_half_open_and_match_the_schedule",
         test_seed_bands_are_half_open_and_match_the_schedule, False),
        ("write_run_config_embeds_the_provenance_block",
         test_write_run_config_embeds_the_provenance_block, True),
        ("failed_seeds_are_skipped_and_accounted",
         test_failed_seeds_are_skipped_and_accounted, True),
        ("pre_update_evaluation_precedes_all_training",
         test_pre_update_evaluation_precedes_all_training, True),
        ("an_all_failed_batch_reports_a_missing_reward_not_zero",
         test_an_all_failed_batch_reports_a_missing_reward_not_zero, True),
        ("a_successful_zero_wake_episode_is_not_a_failure",
         test_a_successful_zero_wake_episode_is_not_a_failure, True),
        ("console_flag_never_reports_both_failure_states",
         test_console_flag_never_reports_both_failure_states, True),
        # --- P1: one truthful OK block per successful episode ---
        ("every_successful_episode_prints_one_labelled_ok_block",
         test_every_successful_episode_prints_one_labelled_ok_block, True),
        ("a_failed_attempt_prints_no_ok_block_and_still_accounts",
         test_a_failed_attempt_prints_no_ok_block_and_still_accounts, True),
        ("ok_block_reports_the_real_ending_not_a_verdict",
         test_ok_block_reports_the_real_ending_not_a_verdict, False),
        ("ok_block_survives_a_non_ascii_target_name",
         test_ok_block_survives_a_non_ascii_target_name, False),
        # --- P2: confirmations counted UNIQUELY over target id ---
        ("unique_confirmed_target_ids_deduplicates_over_ego",
         test_unique_confirmed_target_ids_deduplicates_over_ego, False),
        ("roster_split_totals_the_unique_count",
         test_roster_split_totals_the_unique_count, False),
        ("trainer_aggregates_use_the_unique_target_count",
         test_trainer_aggregates_use_the_unique_target_count, True),
        ("observability_does_not_touch_reward_or_ppo_diagnostics",
         test_observability_does_not_touch_reward_or_ppo_diagnostics, True),
        # --- the false-zero regression, at the seam that sees both inputs ---
        ("unique_count_is_taken_directly_from_the_executor_done_set",
         test_unique_count_is_taken_directly_from_the_executor_done_set, False),
        ("a_broken_name_lookup_keeps_the_id_the_count_and_the_denominator",
         test_a_broken_name_lookup_keeps_the_id_the_count_and_the_denominator, False),
        ("a_structural_roster_failure_aborts_as_measurement_integrity",
         test_a_structural_roster_failure_aborts_as_measurement_integrity, False),
        ("a_confirmed_id_outside_the_roster_cannot_produce_a_record",
         test_a_confirmed_id_outside_the_roster_cannot_produce_a_record, False),
        ("a_setup_failure_contributes_no_false_zero_to_the_aggregates",
         test_a_setup_failure_contributes_no_false_zero_to_the_aggregates, True),
        # --- T13b: executed-world roster integrity (P1-P4, P6) ---
        ("oracle_allocation_is_not_the_world_inventory",
         test_oracle_allocation_is_not_the_world_inventory, False),
        ("the_roster_never_reads_the_oracle_task_list",
         test_the_roster_never_reads_the_oracle_task_list, False),
        ("belief_allocation_is_not_the_known_world_denominator",
         test_belief_allocation_is_not_the_known_world_denominator, False),
        ("the_roster_must_match_the_scheduled_cell",
         test_the_roster_must_match_the_scheduled_cell, False),
        ("an_integrity_fault_aborts_the_run_and_writes_no_ledger_row",
         test_an_integrity_fault_aborts_the_run_and_writes_no_ledger_row, True),
        ("an_eval_integrity_fault_aborts_the_round",
         test_an_eval_integrity_fault_aborts_the_round, True),
        ("a_post_run_integrity_fault_leaves_the_real_playback_listed",
         test_a_post_run_integrity_fault_leaves_the_real_playback_listed, True),
        ("finalize_refuses_to_certify_a_world_its_files_contradict",
         test_finalize_refuses_to_certify_a_world_its_files_contradict, True),
        ("finalize_cannot_complete_without_a_synchronized_recording",
         test_finalize_cannot_complete_without_a_synchronized_recording, True),
        ("a_valid_episode_is_unchanged_by_the_world_truth_fix",
         test_a_valid_episode_is_unchanged_by_the_world_truth_fix, False),
        ("train_and_eval_accounting_are_unchanged_on_the_valid_path",
         test_train_and_eval_accounting_are_unchanged_on_the_valid_path, True),
        ("an_episode_outcome_cannot_omit_what_it_measured",
         test_an_episode_outcome_cannot_omit_what_it_measured, False),
        # --- P3: every eval round keeps its own scenario artifacts ---
        ("eval_episode_tag_is_deterministic_and_round_disjoint",
         test_eval_episode_tag_is_deterministic_and_round_disjoint, False),
        ("validate_refuses_an_eval_band_wider_than_one_tag_namespace",
         test_validate_refuses_an_eval_band_wider_than_one_tag_namespace, False),
        ("eval_rounds_reuse_the_seeds_but_not_the_tags",
         test_eval_rounds_reuse_the_seeds_but_not_the_tags, True),
        ("pre_and_post_update_scenario_files_coexist",
         test_pre_and_post_update_scenario_files_coexist, True),
        ("provenance_is_collected_before_any_run_artifact_exists",
         test_provenance_is_collected_before_any_run_artifact_exists, True),
        ("git_provenance_requires_both_the_sha_and_the_dirty_state",
         test_git_provenance_requires_both_the_sha_and_the_dirty_state, True),
        ("training_stops_when_git_provenance_is_incomplete",
         test_training_stops_when_git_provenance_is_incomplete, True),
        ("a_dirty_tree_warns_but_still_runs",
         test_a_dirty_tree_warns_but_still_runs, True),
        ("run_summary_is_derived_from_the_jsonl_records",
         test_run_summary_is_derived_from_the_jsonl_records, True),
        ("run_summary_flags_a_ledger_that_disagrees",
         test_run_summary_flags_a_ledger_that_disagrees, True),
        ("run_summary_json_omits_the_embedded_record_lists",
         test_run_summary_json_omits_the_embedded_record_lists, True),
        ("xy_drops_missing_rewards_and_anchors_pre_update_at_zero",
         test_xy_drops_missing_rewards_and_anchors_pre_update_at_zero, False),
        ("plot_separates_performance_from_diagnostics_and_health",
         test_plot_separates_performance_from_diagnostics_and_health, False),
        ("performance_plot_draws_every_cell_and_delta_distinctly",
         test_performance_plot_draws_every_cell_and_delta_distinctly, False),
        ("measurement_health_keeps_every_denominator",
         test_measurement_health_keeps_every_denominator, False),
        ("plot_renders_every_figure_from_jsonl",
         test_plot_renders_every_figure_from_jsonl, True),
        ("plot_still_renders_pre_b4_records",
         test_plot_still_renders_pre_b4_records, True),
        # --- T15: the opt-in visual-artifact bundles (PO1 / PO2 / PO3) ---
        ("visual_artifacts_default_off_at_both_surfaces",
         test_visual_artifacts_default_off_at_both_surfaces, False),
        ("disabled_run_builds_no_bundle_and_no_directory",
         test_disabled_run_builds_no_bundle_and_no_directory, True),
        ("disabled_run_passes_no_recording_path_and_never_exports",
         test_disabled_run_passes_no_recording_path_and_never_exports, True),
        ("enabling_artifacts_changes_no_seed_tag_or_condition",
         test_enabling_artifacts_changes_no_seed_tag_or_condition, True),
        ("enabled_attempt_preserves_all_three_snapshots",
         test_enabled_attempt_preserves_all_three_snapshots, True),
        ("manifest_identity_places_every_phase_exactly",
         test_manifest_identity_places_every_phase_exactly, True),
        ("one_seed_cannot_overwrite_another_bundle",
         test_one_seed_cannot_overwrite_another_bundle, True),
        ("a_bundle_directory_collision_fails_loudly",
         test_a_bundle_directory_collision_fails_loudly, True),
        ("export_happens_before_the_controller_and_exactly_once",
         test_export_happens_before_the_controller_and_exactly_once, True),
        ("recording_is_armed_only_on_the_bundle_directory",
         test_recording_is_armed_only_on_the_bundle_directory, True),
        ("artifacts_change_no_outcome_record_or_ppo_input",
         test_artifacts_change_no_outcome_record_or_ppo_input, True),
        ("a_normal_episode_failure_stays_scientific_and_leaves_an_incomplete_bundle",
         test_a_normal_episode_failure_stays_scientific_and_leaves_an_incomplete_bundle,
         True),
        ("an_artifact_failure_aborts_and_never_enters_the_ledger",
         test_an_artifact_failure_aborts_and_never_enters_the_ledger, True),
        ("a_missing_recording_is_an_artifact_failure_not_a_silent_pass",
         test_a_missing_recording_is_an_artifact_failure_not_a_silent_pass, True),
        ("run_config_records_the_resolved_visual_artifacts_flag",
         test_run_config_records_the_resolved_visual_artifacts_flag, True),
        ("an_attempt_identity_cannot_omit_what_names_it",
         test_an_attempt_identity_cannot_omit_what_names_it, False),
        # --- T16: JSON presets ---
        ("the_repository_probe_preset_resolves_a_valid_train_config",
         test_the_repository_probe_preset_resolves_a_valid_train_config, False),
        ("the_probe_preset_carries_the_fd_and_cell_defaults_of_the_dataclass",
         test_the_probe_preset_carries_the_fd_and_cell_defaults_of_the_dataclass, False),
        ("explicit_cli_flags_beat_the_json_preset",
         test_explicit_cli_flags_beat_the_json_preset, True),
        ("no_config_reproduces_the_pre_preset_cli_resolution",
         test_no_config_reproduces_the_pre_preset_cli_resolution, False),
        ("a_config_may_supply_iterations_and_its_absence_still_fails",
         test_a_config_may_supply_iterations_and_its_absence_still_fails, True),
        ("load_config_file_refuses_what_it_cannot_honour",
         test_load_config_file_refuses_what_it_cannot_honour, True),
        ("config_comments_are_ignored_and_tuples_round_trip",
         test_config_comments_are_ignored_and_tuples_round_trip, True),
        ("run_config_records_the_effective_config_and_its_preset",
         test_run_config_records_the_effective_config_and_its_preset, True),
        ("config_source_is_always_a_structured_object",
         test_config_source_is_always_a_structured_object, True),
        ("a_direct_train_call_is_not_recorded_as_cli_provenance",
         test_a_direct_train_call_is_not_recorded_as_cli_provenance, True),
        ("probe_preset_claims_scheduled_iterations_not_productive_updates",
         test_probe_preset_claims_scheduled_iterations_not_productive_updates, False),
        ("cli_exposes_config_and_plot_targets_the_plots_dir",
         test_cli_exposes_config_and_plot_targets_the_plots_dir, False),
        ("main_hands_train_the_resolved_config_and_its_source",
         test_main_hands_train_the_resolved_config_and_its_source, True),
        ("main_with_argv_none_still_sees_the_real_command_line",
         test_main_with_argv_none_still_sees_the_real_command_line, True),
        ("explicit_dests_read_argv_none_as_the_real_command_line",
         test_explicit_dests_read_argv_none_as_the_real_command_line, False),
        ("health_plot_exposes_per_cell_eval_denominators",
         test_health_plot_exposes_per_cell_eval_denominators, False),
        ("per_condition_denominators_survive_an_asymmetric_round",
         test_per_condition_denominators_survive_an_asymmetric_round, True),
        # --- GENERALIZED-V1 step 2: FuelDamageIntegrityError routing ---
        ("a_fuel_damage_integrity_fault_is_not_wrapped_as_an_episode_failure",
         test_a_fuel_damage_integrity_fault_is_not_wrapped_as_an_episode_failure, True),
        ("a_fuel_damage_integrity_fault_aborts_the_run_and_never_reaches_the_ledger",
         test_a_fuel_damage_integrity_fault_aborts_the_run_and_never_reaches_the_ledger,
         True),
        ("an_eval_fuel_damage_integrity_fault_aborts_the_round",
         test_an_eval_fuel_damage_integrity_fault_aborts_the_round, True),
        ("an_ordinary_fuel_damage_failure_is_still_accounted_attrition",
         test_an_ordinary_fuel_damage_failure_is_still_accounted_attrition, True),
        # --- GENERALIZED-V1 Task 4: selection, cardinality, benchmark, persistence ---
        ("gen_a_fixed_cell_run_passes_no_generalized_keyword",
         test_gen_a_fixed_cell_run_passes_no_generalized_keyword, True),
        ("gen_a_generalized_run_passes_the_sampled_cardinality",
         test_gen_a_generalized_run_passes_the_sampled_cardinality, True),
        ("gen_the_fd_parameters_carry_the_designs_two_policies",
         test_gen_the_fd_parameters_carry_the_designs_two_policies, False),
        ("gen_validate_refuses_a_half_configured_generalized_run",
         test_gen_validate_refuses_a_half_configured_generalized_run, True),
        ("gen_short_realization_is_accounted_and_a_lying_audit_aborts",
         test_gen_short_realization_is_accounted_and_a_lying_audit_aborts, False),
        ("gen_the_outcome_record_states_the_design_and_the_cardinality",
         test_gen_the_outcome_record_states_the_design_and_the_cardinality, True),
        ("gen_the_failure_ledger_keeps_the_scheduled_world",
         test_gen_the_failure_ledger_keeps_the_scheduled_world, True),
        ("gen_the_summary_reports_requested_vs_realized",
         test_gen_the_summary_reports_requested_vs_realized, True),
        ("gen_benchmark_round_reports_all_18_strata_with_denominators",
         test_gen_benchmark_round_reports_all_18_strata_with_denominators, True),
        ("gen_benchmark_deltas_use_complete_groups_only",
         test_gen_benchmark_deltas_use_complete_groups_only, True),
        ("gen_a_matched_group_that_built_two_worlds_aborts",
         test_gen_a_matched_group_that_built_two_worlds_aborts, True),
        ("gen_a_frozen_preflight_mismatch_aborts",
         test_gen_a_frozen_preflight_mismatch_aborts, True),
        ("gen_a_tampered_manifest_never_reaches_a_run",
         test_gen_a_tampered_manifest_never_reaches_a_run, True),
        ("gen_run_config_records_the_design_the_sampler_and_the_manifest",
         test_gen_run_config_records_the_design_the_sampler_and_the_manifest, True),
        ("gen_reference_attrition_is_accounted_and_contradiction_aborts",
         test_gen_reference_attrition_is_accounted_and_contradiction_aborts, False),
        ("gen_cli_and_rollout_expose_the_selector_without_drift",
         test_gen_cli_and_rollout_expose_the_selector_without_drift, False),
    ]
    for name, fn, needs_tmp in tests:
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))  # type: ignore[arg-type]
            else:
                fn()  # type: ignore[call-arg]
            print(f"OK   {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"GRAPH_TRAIN TESTS: {failures} failed")
        sys.exit(1)
    print(f"GRAPH_TRAIN TESTS: all {len(tests)} passed")
