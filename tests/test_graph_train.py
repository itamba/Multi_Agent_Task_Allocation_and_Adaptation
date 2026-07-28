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
import io
import inspect
import json
import random
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
from match_aou.rl.training.graph_ppo import PPOConfig, PPOUpdater  # noqa: E402
from match_aou.rl.training.graph_tick_loop import build_policy  # noqa: E402
from match_aou.rl.training import graph_rollout, graph_train  # noqa: E402
from match_aou.rl.training.graph_rollout import (  # noqa: E402
    RolloutConfig,
    run_rollout,
)
from match_aou.rl.training.graph_train import (  # noqa: E402
    TrainConfig,
    _build_arg_parser,
    _parse_airbase_range,
    build_variation_config,
    derived_split,
    eval_seed,
    global_episode_index,
    plot_training,
    plot_training_subprocess,
    save_checkpoint,
    train_seed,
    write_run_config,
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
        assert "n_hidden=int(cfg.n_hidden)" in source, module.__name__
        assert "random.Random(seed)" in source, module.__name__
        # The pre-B3 constant is gone from both harnesses.
        assert "_ALL_KNOWN_PARTIAL_RATIO" not in source, module.__name__

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
