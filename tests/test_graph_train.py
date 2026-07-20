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
  T3 plotting              : plot_training() renders a PNG from synthetic jsonl alone
                             (no training, no torch), tolerates a missing eval file,
                             and degrades to a friendly no-op when matplotlib is absent
                             -- matplotlib is optional and must never fail the suite.

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
synthetic grads; T2 is arithmetic; T3 writes its own jsonl.

Run: python -m pytest tests/test_graph_train.py -v
     python tests/test_graph_train.py
"""

from __future__ import annotations

import json
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

from match_aou.rl.training.graph_ppo import PPOConfig, PPOUpdater  # noqa: E402
from match_aou.rl.training.graph_tick_loop import build_policy  # noqa: E402
from match_aou.rl.training.graph_train import (  # noqa: E402
    TrainConfig,
    eval_seed,
    global_episode_index,
    plot_training,
    plot_training_subprocess,
    save_checkpoint,
    train_seed,
)


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

def _write_synthetic_run(run_dir: Path, *, with_eval: bool = True) -> None:
    """Write train/eval jsonl in EXACTLY the shape `train()` emits (scalar-only)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train_records.jsonl", "w", encoding="utf-8") as fh:
        for it in range(6):
            fh.write(json.dumps({
                "iteration": it,
                "baseline": -0.9 + 0.1 * it,
                "entropy": 1.7 - 0.05 * it,
                "policy_loss": 0.01 * it,
                "total_loss": 0.01 * it - 0.017,
                "n_transitions": 10 + it,
                "n_epochs_run": 2,
                "meta_action_counts": {
                    "PLAN_COMPLIANCE": 6, "OPPORTUNISTIC_ENGAGEMENT": 3,
                    "SELF_PRESERVATION_ABORT": 1,
                },
                "meta_action_fractions": {
                    "PLAN_COMPLIANCE": 0.6, "OPPORTUNISTIC_ENGAGEMENT": 0.3,
                    "SELF_PRESERVATION_ABORT": 0.1,
                },
            }) + "\n")
    if with_eval:
        with open(run_dir / "eval_records.jsonl", "w", encoding="utf-8") as fh:
            for it in (1, 3, 5):
                fh.write(json.dumps({
                    "iteration": it,
                    "n_episodes": 4,
                    "eval_reward_mean": -0.8 + 0.1 * it,
                }) + "\n")


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


def test_plot_training_from_synthetic_jsonl(tmp_path: Path) -> None:
    """A PNG is produced from the jsonl alone -- no training, no policy involved."""
    if _skip_without_matplotlib():
        return
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    out = plot_training_subprocess(run_dir)
    assert out is not None and out.exists()
    assert out.name == "training_plot.png"
    assert out.stat().st_size > 1000, "PNG is suspiciously small"


def test_plot_training_without_eval_records(tmp_path: Path) -> None:
    """An in-progress run with no eval round yet still plots (panel 1 loses a series)."""
    if _skip_without_matplotlib():
        return
    run_dir = tmp_path / "run_no_eval"
    _write_synthetic_run(run_dir, with_eval=False)
    out = plot_training_subprocess(run_dir)
    assert out is not None and out.exists()


def test_plot_training_missing_records_is_a_clean_noop(tmp_path: Path) -> None:
    """Pointing the plotter at a directory with no records returns None, never raises.

    Safe to call IN-PROCESS despite the torch/matplotlib conflict: `plot_training`
    reads the records before it touches matplotlib, so the empty path never imports it.
    """
    assert plot_training(tmp_path / "does_not_exist") is None


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
        ("plot_training_from_synthetic_jsonl",
         test_plot_training_from_synthetic_jsonl, True),
        ("plot_training_without_eval_records",
         test_plot_training_without_eval_records, True),
        ("plot_training_missing_records_is_a_clean_noop",
         test_plot_training_missing_records_is_a_clean_noop, True),
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
