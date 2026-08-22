"""
Unit tests for `graph_ppo` — the PPO core (Phase A: actor-only).

The module under test is the first consumer of `EpisodeResult.trajectory`: it groups
each episode's wakes into per-ego chains, turns the episode's terminal reward into
per-transition returns and normalized advantages, and runs the clipped-surrogate
update through `evaluate_action`.

What these tests lock (the proof obligations, P1-P6):

  P1 epoch-0 identity      : with unchanged weights every ratio is exactly 1.0, and the
                             epoch-0 clipped policy loss equals -mean(A_norm) ~ 0. This
                             is the load-bearing consequence of `sample_action` and
                             `evaluate_action` sharing one distribution construction
                             site — drift there would silently corrupt the gradient.
  P2 learning direction    : an action given a clearly positive advantage becomes MORE
                             probable after repeated updates (and the negative one less).
  P3 clipping is live      : the clamped branch is hand-checked on known numbers, and a
                             real update with a large lr drives clip_fraction > 0.
  P4 grouping correctness  : an interleaved two-ego trajectory splits into per-ego
                             chains preserving each ego's order; a zero-wake episode
                             counts toward n_episodes and the baseline, contributes no
                             transitions, and never crashes the update.
  P5 degenerate batches    : an all-same-R batch yields ~0 advantages and NO NaNs; an
                             empty buffer is a clean documented no-op.
  P6 grads + import purity : every exercised encoder + head parameter receives a finite
                             grad from a real update (`edge_attr_proj` exempt by exact
                             name — the builder emits no edge_attr); and importing the
                             module in a fresh interpreter drags in no flat-only module.

No BLADE, no solver, no env: hand-built synthetic `GraphObservation`s throughout, and
transitions produced by the real policy under `no_grad` exactly as the tick-loop's
`_wake_decision` stores them.

Run: python -m pytest tests/test_graph_ppo.py -v
     python tests/test_graph_ppo.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so match_aou.* imports resolve

from match_aou.rl.action.graph_action import (  # noqa: E402
    MetaAction,
    build_action_mask,
    evaluate_action,
    sample_action,
)
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    EdgeType,
    GraphObservation,
)
from match_aou.rl.training.graph_ppo import (  # noqa: E402
    AdvantageBatch,
    EpisodeRecord,
    PPOBuffer,
    PPOConfig,
    PPOUpdater,
    clipped_surrogate,
    compute_returns_and_advantages,
)
from match_aou.rl.training.graph_tick_loop import Transition, build_policy  # noqa: E402

OE = int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)
PC = int(MetaAction.PLAN_COMPLIANCE)


# =============================================================================
# Synthetic fixtures (mirrors tests/test_graph_action_evaluate.py::_make_obs)
# =============================================================================

def _make_obs(seed_shift: float = 0.0) -> GraphObservation:
    """A k=4 / a=3 graph with a KNOWN mask: node0 has ABORT, node2 has ENGAGEMENT.

      task 0: assigned to ego  (4->0), sensed          -> COMPLIANCE + ABORT
      task 1: assigned to peer1(5->1), sensed          -> COMPLIANCE
      task 2: UNASSIGNED, sensed, capable, reachable   -> COMPLIANCE + ENGAGEMENT
      task 3: assigned to peer2(6->3), NOT sensed      -> COMPLIANCE

    `seed_shift` perturbs the utility column so distinct "states" can be built without
    changing the mask topology.
    """
    task_features = np.array(
        [
            # [utility, dist_to_ego, capable, reachable, probability, sensed]
            [0.80 + seed_shift, 0.20, 1.0, 1.0, 1.0, 1.0],
            [0.60 + seed_shift, 0.40, 1.0, 1.0, 1.0, 1.0],
            [0.50 + seed_shift, 0.30, 1.0, 1.0, 1.0, 1.0],
            [0.70 + seed_shift, 0.50, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    agent_features = np.array([[0.90], [0.00], [0.00]], dtype=np.float32)
    return GraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=4,  # == k; ego is the first agent node
        edge_index=np.array([[4, 5, 6],
                             [0, 1, 3]], dtype=np.int64),
        edge_type=np.array([int(EdgeType.ASSIGNMENT)] * 3, dtype=np.int64),
        task_target_ids=["t0", "t1", "t2", "t3"],
        agent_ids=["ego", "peer1", "peer2"],
        agent_id="ego",
        current_time=0,
        time_norm=0.0,
    )


def _make_transition(
    policy,
    gobs: GraphObservation,
    ego_id: str,
    tick: int,
    *,
    force_cell: Optional[Tuple[int, int]] = None,
) -> Transition:
    """One real `Transition`, storing exactly what `_wake_decision` stores.

    Synthetic ROLLOUT data: `no_grad` here mirrors the inference-only rollout path.
    `force_cell=(meta_action, node_v)` pins the stored action and scores it through
    `evaluate_action` — the SAME distribution `sample_action` draws from (one shared
    `_masked_dist` site), so the stored log-prob is exactly what a rollout that
    sampled that cell would have stored. That makes P2/P3 deterministic.
    """
    mask = build_action_mask(gobs)
    with torch.no_grad():
        logits = policy.head(policy.encoder(gobs))
        if force_cell is None:
            meta_action, node_v, log_prob, entropy = sample_action(
                logits, mask, deterministic=False
            )
        else:
            meta_action, node_v = int(force_cell[0]), int(force_cell[1])
            log_prob, entropy = evaluate_action(logits, mask, meta_action, node_v)
    return Transition(
        gobs=gobs,
        ego_id=str(ego_id),
        tick=int(tick),
        meta_action=int(meta_action),
        node_v=int(node_v),
        log_prob=float(log_prob.item()),
        entropy=float(entropy.item()),
    )


def _action_prob(policy, gobs: GraphObservation, meta_action: int, node_v: int) -> float:
    """Current probability of `(meta_action, node_v)` on `gobs` (fresh forward, no-grad)."""
    mask = build_action_mask(gobs)
    with torch.no_grad():
        logits = policy.head(policy.encoder(gobs))
        log_prob, _ = evaluate_action(logits, mask, meta_action, node_v)
    return float(torch.exp(log_prob).item())


def _two_outcome_buffer(policy, gobs: GraphObservation) -> PPOBuffer:
    """Two one-wake episodes on the SAME state: a good action (R=0) and a bad one (R=-1).

    Sharing the state is what makes the comparison clean — the only difference between
    the episodes is the action taken and the reward it earned, so the resulting
    advantages (+1 / -1 after normalization) point unambiguously at one cell.
    """
    tr_good = _make_transition(policy, gobs, "egoA", 1, force_cell=(OE, 2))
    tr_bad = _make_transition(policy, gobs, "egoA", 1, force_cell=(PC, 0))
    buf = PPOBuffer()
    buf.add(EpisodeRecord.from_trajectory([tr_good], 0.0, seed=0, episode_index=0))
    buf.add(EpisodeRecord.from_trajectory([tr_bad], -1.0, seed=1, episode_index=1))
    return buf


# =============================================================================
# P4 — grouping correctness
# =============================================================================

def test_grouping_preserves_per_ego_order() -> None:
    """An INTERLEAVED trajectory splits into per-ego chains, each in temporal order.

    The tick-loop appends wakes in tick order across egos, so a two-ego episode is
    interleaved. Grouping must be a stable partition (never a sort): each ego's chain
    is its own decisions in the order it made them — the shape the Phase-B GAE needs.
    """
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    obs = _make_obs()
    interleaved = [
        _make_transition(policy, obs, "egoA", 1),
        _make_transition(policy, obs, "egoB", 1),
        _make_transition(policy, obs, "egoA", 5),
        _make_transition(policy, obs, "egoB", 5),
        _make_transition(policy, obs, "egoA", 9),
    ]
    rec = EpisodeRecord.from_trajectory(interleaved, -0.25, seed=7, episode_index=3)

    assert list(rec.chains.keys()) == ["egoA", "egoB"], "ego keys not in first-wake order"
    assert [t.tick for t in rec.chains["egoA"]] == [1, 5, 9]
    assert [t.tick for t in rec.chains["egoB"]] == [1, 5]
    assert all(t.ego_id == "egoA" for t in rec.chains["egoA"])
    assert all(t.ego_id == "egoB" for t in rec.chains["egoB"])
    assert rec.n_transitions == 5 and rec.has_wakes
    assert rec.episode_reward == -0.25 and rec.seed == 7 and rec.episode_index == 3

    # The flattened view is chain-by-chain and index-aligned with the arrays.
    flat = rec.transitions()
    assert [t.tick for t in flat] == [1, 5, 9, 1, 5]


def test_zero_wake_episode_counts_but_adds_no_transitions() -> None:
    """A zero-wake episode is VALID: it moves the baseline but contributes nothing else.

    Roughly a quarter of rollout episodes produce no organic wake (the trigger layer is
    event-driven), so this is a normal operating condition, not an error. It must count
    toward n_episodes and the baseline mean, add zero transitions, and not crash an
    update.
    """
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    obs = _make_obs()
    rec = EpisodeRecord.from_trajectory(
        [_make_transition(policy, obs, "egoA", 1)], -0.25, seed=0, episode_index=0
    )
    zero_wake = EpisodeRecord.from_trajectory([], -1.0, seed=1, episode_index=1)
    assert zero_wake.chains == {} and zero_wake.n_transitions == 0
    assert not zero_wake.has_wakes

    buf = PPOBuffer()
    buf.add(rec)
    buf.add(zero_wake)
    assert buf.n_episodes == 2
    assert buf.n_transitions == 1
    assert buf.episodes_with_wakes == 1

    batch = compute_returns_and_advantages(buf)
    # Baseline is the mean over EPISODES: (-0.25 + -1.0)/2. The zero-wake episode pulls
    # it with full weight despite contributing no transitions — that is the point.
    assert abs(batch.baseline - (-0.625)) < 1e-12
    assert batch.n_episodes == 2 and batch.n_transitions == 1

    diag = PPOUpdater(policy, PPOConfig(n_epochs=1)).update(buf)
    assert diag["n_transitions"] == 1 and diag["n_epochs_run"] == 1
    assert diag["episodes_with_wakes"] == 1
    assert np.isfinite(diag["policy_loss"])


def test_buffer_clear_resets_everything() -> None:
    """`clear()` empties the buffer so iterations stay strictly on-policy."""
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    buf = PPOBuffer()
    buf.add(EpisodeRecord.from_trajectory(
        [_make_transition(policy, _make_obs(), "egoA", 1)], -0.3
    ))
    assert buf.n_episodes == 1 and buf.n_transitions == 1
    buf.clear()
    assert buf.n_episodes == 0 and buf.n_transitions == 0
    assert buf.transitions() == [] and len(buf) == 0


# =============================================================================
# Returns: gamma is dormant at 1.0 but genuinely honoured below it
# =============================================================================

def test_returns_are_the_episode_reward_at_gamma_one() -> None:
    """gamma == 1.0 -> the return of EVERY transition is its episode's R.

    This is the Phase-A semantics: a single terminal reward, chains of 1-4 decisions,
    no per-step cost. Discounting would arbitrarily penalize the earliest decisions.
    """
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    obs = _make_obs()
    buf = PPOBuffer()
    buf.add(EpisodeRecord.from_trajectory(
        [_make_transition(policy, obs, "egoA", t) for t in (1, 2, 3)], -0.4
    ))
    buf.add(EpisodeRecord.from_trajectory(
        [_make_transition(policy, obs, "egoB", t) for t in (1, 2)], -0.9
    ))
    batch = compute_returns_and_advantages(buf, PPOConfig(gamma=1.0))
    assert np.allclose(batch.returns, [-0.4, -0.4, -0.4, -0.9, -0.9])


def test_gamma_below_one_discounts_along_each_chain() -> None:
    """gamma < 1 discounts the terminal reward back along EACH ego's own chain.

    Proves the per-ego grouping is structurally load-bearing rather than decorative:
    the two chains have different lengths, so a flat list could not produce this.
    """
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    obs = _make_obs()
    traj = [
        _make_transition(policy, obs, "egoA", 1),
        _make_transition(policy, obs, "egoB", 1),
        _make_transition(policy, obs, "egoA", 5),
        _make_transition(policy, obs, "egoB", 5),
        _make_transition(policy, obs, "egoA", 9),
    ]
    rec = EpisodeRecord.from_trajectory(traj, -0.25)
    batch = compute_returns_and_advantages([rec], PPOConfig(gamma=0.5))
    # egoA (len 3): R*0.25, R*0.5, R    egoB (len 2): R*0.5, R
    expected = [-0.25 * 0.25, -0.25 * 0.5, -0.25, -0.25 * 0.5, -0.25]
    assert np.allclose(batch.returns, expected), (batch.returns, expected)


def test_advantages_are_normalized() -> None:
    """Normalized advantages have mean ~0 and std ~1 across ALL batch transitions."""
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    buf = PPOBuffer()
    for i, R in enumerate([-0.1, -0.55, -0.9, -0.3]):
        obs = _make_obs(seed_shift=0.01 * i)
        buf.add(EpisodeRecord.from_trajectory(
            [_make_transition(policy, obs, "egoA", i),
             _make_transition(policy, obs, "egoB", i)],
            R, seed=i, episode_index=i,
        ))
    batch = compute_returns_and_advantages(buf)
    assert isinstance(batch, AdvantageBatch)
    assert batch.n_transitions == 8 and len(batch.transitions) == 8
    assert abs(float(batch.advantages.mean())) < 1e-9
    assert abs(float(batch.advantages.std()) - 1.0) < 1e-6
    # raw advantage == return - baseline, and the baseline is the episode mean.
    assert np.allclose(batch.raw_advantages, batch.returns - batch.baseline)
    assert abs(batch.baseline - np.mean([-0.1, -0.55, -0.9, -0.3])) < 1e-12


def test_baseline_is_per_episode_not_per_transition() -> None:
    """The baseline weights EPISODES equally, regardless of how many wakes each had.

    A 4-wake episode must not pull the baseline four times harder than a 1-wake one:
    wake count is a property of the scenario (how much was hidden, what got sensed),
    not of the policy's merit.
    """
    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    obs = _make_obs()
    many = EpisodeRecord.from_trajectory(
        [_make_transition(policy, obs, "egoA", t) for t in range(4)], 0.0
    )
    one = EpisodeRecord.from_trajectory(
        [_make_transition(policy, obs, "egoB", 0)], -1.0
    )
    batch = compute_returns_and_advantages([many, one])
    # Per-episode mean: (0.0 + -1.0)/2 = -0.5. A per-TRANSITION mean would be
    # (0*4 + -1)/5 = -0.2 — this asserts we are not doing that.
    assert abs(batch.baseline - (-0.5)) < 1e-12, batch.baseline


# =============================================================================
# P1 — epoch-0 identity
# =============================================================================

def test_epoch0_ratio_is_one_and_loss_is_minus_mean_advantage() -> None:
    """P1: unchanged weights -> every ratio exactly 1.0, epoch-0 policy loss ~ 0.

    With ratio == 1 the clamp is inactive, so the clipped surrogate collapses to
    -mean(A_norm), and normalized advantages have mean 0 -> the loss is ~0 before the
    first optimizer step. Any deviation would mean the update is re-scoring actions
    under a DIFFERENT distribution than the rollout sampled from — a silent corruption
    of the policy gradient, which is exactly what the shared `_masked_dist`
    construction site exists to prevent.
    """
    torch.manual_seed(1)
    policy = build_policy(embed_dim=64)
    buf = PPOBuffer()
    for i, R in enumerate([-0.10, -0.55, -0.90, -0.30]):
        obs = _make_obs(seed_shift=0.01 * i)
        buf.add(EpisodeRecord.from_trajectory(
            [_make_transition(policy, obs, "egoA", 1 + i),
             _make_transition(policy, obs, "egoB", 2 + i)],
            R, seed=i, episode_index=i,
        ))

    batch = compute_returns_and_advantages(buf)
    diag = PPOUpdater(policy, PPOConfig(n_epochs=1)).update(buf)

    max_dev = diag["per_epoch"]["max_ratio_dev"][0]
    loss0 = diag["per_epoch"]["policy_loss"][0]
    expected = -float(batch.advantages.mean())
    print(f"\n  epoch-0 max |ratio - 1| = {max_dev:.3e}  "
          f"mean_ratio={diag['per_epoch']['mean_ratio'][0]:.12f}")
    print(f"  epoch-0 policy_loss={loss0:.3e}  expected -mean(A_norm)={expected:.3e}")

    assert max_dev <= 1e-6, f"epoch-0 ratio deviates from 1.0 by {max_dev}"
    assert abs(diag["per_epoch"]["mean_ratio"][0] - 1.0) <= 1e-6
    assert abs(loss0 - expected) <= 1e-6, (loss0, expected)
    assert abs(loss0) <= 1e-6, f"epoch-0 policy loss is not ~0: {loss0}"
    # approx_kl = mean(lp_old - lp_new) is 0 when nothing moved yet.
    assert abs(diag["per_epoch"]["approx_kl"][0]) <= 1e-6
    # No clipping can engage at ratio == 1.
    assert diag["per_epoch"]["clip_fraction"][0] == 0.0


def test_diagnostics_keys_and_shapes() -> None:
    """The diagnostics contract the outer training loop will log against."""
    torch.manual_seed(2)
    policy = build_policy(embed_dim=64)
    buf = _two_outcome_buffer(policy, _make_obs())
    cfg = PPOConfig(n_epochs=3)
    diag = PPOUpdater(policy, cfg).update(buf)

    scalar_keys = ["policy_loss", "total_loss", "entropy", "mean_ratio",
                   "clip_fraction", "approx_kl", "max_ratio_dev", "grad_norm",
                   "baseline", "adv_std_raw"]
    for key in scalar_keys:
        assert key in diag, f"missing diagnostic {key}"
        assert np.isfinite(diag[key]), f"{key} is not finite: {diag[key]}"
    assert diag["n_episodes"] == 2
    assert diag["episodes_with_wakes"] == 2
    assert diag["n_transitions"] == 2
    assert diag["n_epochs_run"] == 3
    for key, values in diag["per_epoch"].items():
        assert len(values) == 3, f"per_epoch[{key}] has {len(values)} entries, want 3"
    # Scalars are the mean over epochs.
    assert abs(diag["policy_loss"]
               - float(np.mean(diag["per_epoch"]["policy_loss"]))) < 1e-12


# =============================================================================
# P2 — learning direction
# =============================================================================

def test_positive_advantage_action_becomes_more_probable() -> None:
    """P2: repeated updates raise P(good action) and lower P(bad action).

    Two one-wake episodes on the SAME state: the engaged pop-up scored R=0, the
    compliant one scored R=-1. Normalized advantages are +1 / -1, so the gradient
    points unambiguously. This is the end-to-end proof that the sign conventions
    (surrogate negation, entropy subtraction, advantage direction) all line up — any
    single sign error would move the probabilities the wrong way.
    """
    torch.manual_seed(2)
    policy = build_policy(embed_dim=64)
    obs = _make_obs()
    buf = _two_outcome_buffer(policy, obs)

    batch = compute_returns_and_advantages(buf)
    assert batch.advantages[0] > 0.5 > batch.advantages[1], batch.advantages

    p_good_before = _action_prob(policy, obs, OE, 2)
    p_bad_before = _action_prob(policy, obs, PC, 0)

    updater = PPOUpdater(policy, PPOConfig(lr=1e-3, n_epochs=4))
    for _ in range(6):
        updater.update(buf)

    p_good_after = _action_prob(policy, obs, OE, 2)
    p_bad_after = _action_prob(policy, obs, PC, 0)
    print(f"\n  P(good=(node2,OE)): {p_good_before:.6f} -> {p_good_after:.6f}")
    print(f"  P(bad =(node0,PC)): {p_bad_before:.6f} -> {p_bad_after:.6f}")

    assert p_good_after > p_good_before, (p_good_before, p_good_after)
    assert p_bad_after < p_bad_before, (p_bad_before, p_bad_after)


# =============================================================================
# P3 — clipping is live
# =============================================================================

def test_clipped_surrogate_branches_by_hand() -> None:
    """P3(a): the clamped branch on hand-computed numbers, in isolation.

    Sign convention: PPO maximizes the surrogate, so the LOSS is its negation. The
    clamp binds when it makes the surrogate SMALLER — i.e. when a positive advantage
    would otherwise reward pushing the ratio far above 1, or a negative advantage
    would reward pushing it far below.
    """
    clip = 0.2
    # A > 0, ratio above the trust region -> saturates at -(1+clip)*A.
    assert abs(float(clipped_surrogate(torch.tensor(1.5), 2.0, clip))
               - (-(1.0 + clip) * 2.0)) < 1e-6
    # A < 0, ratio below the trust region -> saturates at -(1-clip)*A.
    assert abs(float(clipped_surrogate(torch.tensor(0.5), -2.0, clip))
               - (-(1.0 - clip) * -2.0)) < 1e-6
    # Inside the trust region -> the vanilla policy-gradient term.
    assert abs(float(clipped_surrogate(torch.tensor(1.05), 2.0, clip))
               - (-1.05 * 2.0)) < 1e-6
    # A > 0 with ratio BELOW 1: the unclipped term is smaller, so no clamping.
    assert abs(float(clipped_surrogate(torch.tensor(0.5), 2.0, clip))
               - (-0.5 * 2.0)) < 1e-6


def test_clip_fraction_becomes_positive_in_a_real_update() -> None:
    """P3(b): a real multi-epoch update with a large lr drives |ratio - 1| past the clip."""
    torch.manual_seed(3)
    policy = build_policy(embed_dim=64)
    buf = _two_outcome_buffer(policy, _make_obs())
    diag = PPOUpdater(policy, PPOConfig(lr=0.05, n_epochs=8, clip_ratio=0.2)).update(buf)

    print(f"\n  per-epoch max|ratio-1|: "
          f"{[round(v, 4) for v in diag['per_epoch']['max_ratio_dev']]}")
    print(f"  per-epoch clip_fraction: "
          f"{[round(v, 4) for v in diag['per_epoch']['clip_fraction']]}")
    assert max(diag["per_epoch"]["max_ratio_dev"]) > 0.2, "ratio never left the clip band"
    assert max(diag["per_epoch"]["clip_fraction"]) > 0.0, "clipping never engaged"
    assert diag["clip_fraction"] > 0.0


# =============================================================================
# P5 — degenerate batches
# =============================================================================

def test_all_same_reward_batch_is_a_safe_near_no_op() -> None:
    """P5(a): zero advantage variance -> ~0 advantages, NO NaNs, update still runs.

    A batch where every episode scored the same R carries no information about which
    action to prefer, so a near-no-op update is the CORRECT behavior — inventing a
    direction would be pure noise. The eps guard is what keeps 0/0 from becoming NaN.
    """
    torch.manual_seed(4)
    policy = build_policy(embed_dim=64)
    buf = PPOBuffer()
    for i in range(3):
        obs = _make_obs(seed_shift=0.02 * i)
        buf.add(EpisodeRecord.from_trajectory(
            [_make_transition(policy, obs, "egoA", i)], -0.5, seed=i, episode_index=i
        ))

    batch = compute_returns_and_advantages(buf)
    assert batch.adv_std_raw == 0.0
    assert np.all(np.isfinite(batch.advantages))
    assert float(np.max(np.abs(batch.advantages))) < 1e-6

    diag = PPOUpdater(policy, PPOConfig(n_epochs=2)).update(buf)
    for key in ("policy_loss", "total_loss", "entropy", "mean_ratio",
                "approx_kl", "grad_norm"):
        assert np.isfinite(diag[key]), f"{key} is not finite: {diag[key]}"
    # Only the entropy bonus moves the policy here.
    assert abs(diag["per_epoch"]["policy_loss"][0]) < 1e-6


def test_empty_batch_is_a_clean_no_op() -> None:
    """P5(b): an empty buffer returns empty arrays and a no-op update (documented).

    An iteration in which no ego ever woke is a legitimate outcome of the
    event-triggered design, so the outer loop should be able to log it and move on
    rather than catch an exception.
    """
    batch = compute_returns_and_advantages(PPOBuffer())
    assert batch.n_transitions == 0 and batch.n_episodes == 0
    assert batch.transitions == []
    assert batch.returns.shape == (0,) and batch.advantages.shape == (0,)
    assert batch.baseline == 0.0

    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)
    before = [p.detach().clone() for p in policy.head.parameters()]
    diag = PPOUpdater(policy, PPOConfig(n_epochs=4)).update(PPOBuffer())
    assert diag["n_epochs_run"] == 0 and diag["n_transitions"] == 0
    assert diag["policy_loss"] == 0.0 and diag["grad_norm"] == 0.0
    assert diag["per_epoch"]["policy_loss"] == []
    # A no-op must not have touched the weights.
    for p_before, p_now in zip(before, policy.head.parameters()):
        assert torch.equal(p_before, p_now), "empty update modified the policy"


def test_batch_of_only_zero_wake_episodes_is_a_no_op_with_a_baseline() -> None:
    """Episodes that all produced no wakes: no transitions, but the baseline is real."""
    records = [EpisodeRecord.from_trajectory([], R, seed=i, episode_index=i)
               for i, R in enumerate([-0.2, -0.8])]
    batch = compute_returns_and_advantages(records)
    assert batch.n_episodes == 2 and batch.n_transitions == 0
    assert abs(batch.baseline - (-0.5)) < 1e-12  # still loggable

    torch.manual_seed(0)
    diag = PPOUpdater(build_policy(embed_dim=64)).update(records)
    assert diag["n_epochs_run"] == 0
    assert diag["episodes_with_wakes"] == 0
    assert abs(diag["baseline"] - (-0.5)) < 1e-12


# =============================================================================
# P6 — grads + import purity
# =============================================================================

def test_update_produces_finite_grads_on_every_exercised_parameter() -> None:
    """P6(a): a real update leaves finite grads on all encoder + head params.

    `edge_attr_proj` is the encoder's RESERVED edge-attribute path — the builder emits
    no edge_attr, so it is legitimately unexercised (exempt by exact name, as in
    tests/test_graph_action_evaluate.py). Every other parameter must receive one.
    """
    torch.manual_seed(5)
    policy = build_policy(embed_dim=64)
    buf = PPOBuffer()
    for i, R in enumerate([-0.2, -0.8]):
        obs = _make_obs(seed_shift=0.03 * i)
        buf.add(EpisodeRecord.from_trajectory(
            [_make_transition(policy, obs, "egoA", i),
             _make_transition(policy, obs, "egoB", i)],
            R, seed=i, episode_index=i,
        ))
    PPOUpdater(policy, PPOConfig(n_epochs=1)).update(buf)

    _EDGE_ATTR_PARAMS = {"encoder.edge_attr_proj.weight", "encoder.edge_attr_proj.bias"}
    with_grad, without_grad = [], []
    for mod_name, module in (("encoder", policy.encoder), ("head", policy.head)):
        for name, p in module.named_parameters():
            full = f"{mod_name}.{name}"
            if p.grad is None:
                without_grad.append(full)
                continue
            assert torch.isfinite(p.grad).all(), f"{full} has a non-finite grad"
            with_grad.append(full)

    unexpected = [n for n in without_grad if n not in _EDGE_ATTR_PARAMS]
    print(f"\n  grads: {len(with_grad)} params with finite grad, "
          f"{len(without_grad)} without ({without_grad or 'none'})")
    assert not unexpected, f"parameters got NO grad from the update: {unexpected}"
    assert len(with_grad) >= 30, f"suspiciously few grads: {len(with_grad)}"
    any_nonzero = any(
        bool(p.grad.abs().sum() > 0)
        for _, p in list(policy.encoder.named_parameters())
        + list(policy.head.named_parameters())
        if p.grad is not None
    )
    assert any_nonzero, "every grad is exactly zero — the gradient path is dead"


def test_update_actually_changes_the_policy_weights() -> None:
    """A non-degenerate update must move the weights (guards against a silent no-op)."""
    torch.manual_seed(6)
    policy = build_policy(embed_dim=64)
    buf = _two_outcome_buffer(policy, _make_obs())
    before = [p.detach().clone() for p in policy.head.parameters()]
    PPOUpdater(policy, PPOConfig(n_epochs=2)).update(buf)
    changed = any(
        not torch.equal(b, p) for b, p in zip(before, policy.head.parameters())
    )
    assert changed, "a non-degenerate update left the head weights untouched"


# `graph_ppo`'s runtime closure must stay free of the deleted flat path. This mirrors
# tests/test_import_purity.py (which cannot list this module without editing it) and
# also proves the import has no torch global-state side effects worth noticing.
_DENY_MODULES = [
    "match_aou.rl.plan_editor",
    "match_aou.rl.training.ppo_trainer",
    "match_aou.rl.training.rollout_buffer",
    "match_aou.rl.training.reward",
    "match_aou.rl.agent.network",
    "match_aou.rl.observation.observation_builder",
    "match_aou.utils.blade_utils.blade_plan_utils",
]

# The child imports the LOCKED action layer first, snapshots the closure, then imports
# graph_ppo — so the delta is exactly what this new module adds on top of what its
# dependency already dragged in.
_CHILD = (
    "import sys, json, importlib, torch\n"
    "seed_before = torch.initial_seed()\n"
    "importlib.import_module('match_aou.rl.action.graph_action')\n"
    "base = set(m for m in sys.modules if m.startswith('match_aou'))\n"
    "importlib.import_module('match_aou.rl.training.graph_ppo')\n"
    "present = sorted(m for m in sys.modules if m.startswith('match_aou'))\n"
    "print('PPO_PURITY:' + json.dumps({\n"
    "    'entry': 'match_aou.rl.training.graph_ppo' in sys.modules,\n"
    "    'modules': present,\n"
    "    'added': sorted(set(present) - base),\n"
    "    'blade': sorted(m for m in sys.modules if m.split('.')[0] == 'blade'),\n"
    "    'gymnasium': sorted(\n"
    "        m for m in sys.modules if m.split('.')[0] == 'gymnasium'),\n"
    "    'seed_stable': torch.initial_seed() == seed_before,\n"
    "}))\n"
)


def test_import_is_clean() -> None:
    """P6(b): importing `graph_ppo` adds only torch/numpy-level modules to its closure.

    `Policy` / `Transition` / `CentralGraphObservation` are TYPE_CHECKING-only imports
    (the `graph_reward` idiom), so the runtime closure is `graph_action`'s plus this
    module, its package, and the two the Phase-B critic really constructs
    (`graph_encoder`, `central_graph_builder`) — no BLADE engine, no gymnasium env, no
    episode-setup, no tick-loop. That is what lets the whole PPO core, CTDE included,
    be exercised on synthetic data.

    Note what is NOT claimed: `match_aou.solvers.match_aou_MINLP_solver` IS in the
    closure. It arrives through the LOCKED `graph_builder -> scenario_factory` chain
    and is present for `graph_action` too (this test measures the delta precisely so
    that inherited baggage is not mistaken for something this module introduced). The
    solver module is an inert pyomo model definition at import time — it starts no
    process and needs no bonmin — which is why the base env can run these tests.

    The seed check confirms the import touches no torch global state.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        f"importing graph_ppo failed (rc={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    line = next(
        (l for l in proc.stdout.splitlines() if l.startswith("PPO_PURITY:")), None
    )
    assert line is not None, f"no sentinel line.\nSTDOUT:\n{proc.stdout}"
    result = json.loads(line[len("PPO_PURITY:"):])

    assert result["entry"], "graph_ppo not in the child's sys.modules"
    present = set(result["modules"])
    leaked = [m for m in _DENY_MODULES if m in present]
    assert not leaked, f"graph_ppo leaked flat-only module(s): {leaked}"
    assert result["seed_stable"], "importing graph_ppo mutated torch's global RNG"

    # graph_ppo adds ONLY itself, its package, and the two modules the Phase-B critic
    # genuinely needs. Still an EXACT-equality lock (not a subset check), so any further
    # widening still fails here.
    #
    # Why each addition is benign -- and why it is an addition at all: `CentralCritic`
    # IS a network, so it constructs a real `GraphEncoder`, and it needs the central
    # feature widths to construct it with. Both new modules are torch/numpy-only and
    # every dependency they have of their own (`graph_builder`, `scenario_factory`,
    # `shared_utils`) was ALREADY in `graph_action`'s inherited closure, which is why
    # the delta is exactly three module names and no engine or env came with them. The
    # four hard assertions below are unchanged and are what actually guard that.
    assert set(result["added"]) == {
        "match_aou.rl.training",
        "match_aou.rl.training.graph_ppo",
        "match_aou.rl.agent",
        "match_aou.rl.agent.graph_encoder",
        "match_aou.rl.observation.central_graph_builder",
    }, f"graph_ppo widened the import closure: {result['added']}"
    # No engine, no env: the tick-loop types are TYPE_CHECKING-only.
    assert "match_aou.rl.training.graph_episode_setup" not in present
    assert "match_aou.rl.training.graph_tick_loop" not in present
    assert result["blade"] == [], f"BLADE engine imported: {result['blade']}"
    assert result["gymnasium"] == [], f"gymnasium imported: {result['gymnasium']}"
    print(f"\n  graph_ppo closure: {len(present)} match_aou modules; adds only "
          f"{result['added']} over graph_action; no BLADE, no gymnasium, "
          f"no episode-setup, no tick-loop")


if __name__ == "__main__":
    tests = [
        test_grouping_preserves_per_ego_order,
        test_zero_wake_episode_counts_but_adds_no_transitions,
        test_buffer_clear_resets_everything,
        test_returns_are_the_episode_reward_at_gamma_one,
        test_gamma_below_one_discounts_along_each_chain,
        test_advantages_are_normalized,
        test_baseline_is_per_episode_not_per_transition,
        test_epoch0_ratio_is_one_and_loss_is_minus_mean_advantage,
        test_diagnostics_keys_and_shapes,
        test_positive_advantage_action_becomes_more_probable,
        test_clipped_surrogate_branches_by_hand,
        test_clip_fraction_becomes_positive_in_a_real_update,
        test_all_same_reward_batch_is_a_safe_near_no_op,
        test_empty_batch_is_a_clean_no_op,
        test_batch_of_only_zero_wake_episodes_is_a_no_op_with_a_baseline,
        test_update_produces_finite_grads_on_every_exercised_parameter,
        test_update_actually_changes_the_policy_weights,
        test_import_is_clean,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"All {len(tests)} graph_ppo tests passed.")
