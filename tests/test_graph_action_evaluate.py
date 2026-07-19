"""
Unit tests for `evaluate_action` — the PPO-ratio half of the graph action layer.

The rollout is inference-only: `graph_tick_loop._wake_decision` runs under
`torch.no_grad` and stores each wake as a `Transition` holding the
`GraphObservation`, the chosen `(meta_action, node_v)`, and DETACHED log_prob /
entropy floats. The PPO update must recompute the log-prob WITH grad to form the
ratio `pi_new / pi_old`.

The load-bearing property: the update must re-score the action under EXACTLY the
distribution the rollout sampled from. `sample_action` and `evaluate_action` share
one construction site (`_masked_dist`), so that identity holds BY CONSTRUCTION.
Drift there would not crash — it would silently corrupt the policy gradient — so
these tests assert BITWISE equality (`torch.equal`), not `allclose`.

Coverage:
  (i)   exact agreement: evaluate_action's (log_prob, entropy) are bitwise equal to
        sample_action's, for both stochastic and deterministic draws.
  (ii)  end-to-end ratio == 1.0: a real GraphEncoder + ActionHead (build_policy),
        forward under no_grad -> sample -> store floats (mirroring Transition), then
        re-forward WITH grad on the same gobs / weights -> evaluate_action.
  (iii) grad flow: backward from evaluate_action's log_prob reaches every exercised
        encoder + head parameter with a finite grad.
  (iv)  the masked-cell and out-of-bounds guards raise ValueError (fail LOUD).

No BLADE, no solver, no env: a hand-built synthetic GraphObservation throughout.

Run: python -m pytest tests/test_graph_action_evaluate.py -v
     python tests/test_graph_action_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # so match_aou.* imports resolve

from match_aou.rl.action.graph_action import (  # noqa: E402
    MetaAction,
    NUM_META_ACTIONS,
    build_action_mask,
    evaluate_action,
    sample_action,
)
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    EdgeType,
    GraphObservation,
)
from match_aou.rl.training.graph_tick_loop import build_policy  # noqa: E402


# =============================================================================
# Synthetic fixtures (no BLADE / solver / env)
# =============================================================================

def _make_obs() -> GraphObservation:
    """A k=4 / a=3 graph with a KNOWN mask: node0 has ABORT, node2 has ENGAGEMENT.

    Mirrors the graph_action selftest topology but makes task 2 REACHABLE, so the
    OPPORTUNISTIC_ENGAGEMENT column is live somewhere — the mask under test is then
    genuinely mixed (valid and -inf cells in both dimensions) rather than a single
    always-valid column.

      task 0: assigned to ego  (4->0), sensed          -> COMPLIANCE + ABORT
      task 1: assigned to peer1(5->1), sensed          -> COMPLIANCE
      task 2: UNASSIGNED, sensed, capable, reachable   -> COMPLIANCE + ENGAGEMENT
      task 3: assigned to peer2(6->3), NOT sensed      -> COMPLIANCE
    """
    task_features = np.array(
        [
            # [utility, dist_to_ego, capable, reachable, probability, sensed]
            [0.80, 0.20, 1.0, 1.0, 1.0, 1.0],   # task 0
            [0.60, 0.40, 1.0, 1.0, 1.0, 1.0],   # task 1
            [0.50, 0.30, 1.0, 1.0, 1.0, 1.0],   # task 2 (pop-up, engageable)
            [0.70, 0.50, 1.0, 1.0, 1.0, 0.0],   # task 3
        ],
        dtype=np.float32,
    )
    # agent_features are [a, 1] = [fuel_norm]: ego real, peers 0.0 (featureless).
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


def _fixed_logits(k: int, seed: int = 0) -> torch.Tensor:
    """Deterministic [k, 3] logits, independent of any module (pure torch RNG)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(k, NUM_META_ACTIONS, generator=gen)


# =============================================================================
# (i) Exact agreement between sample_action and evaluate_action
# =============================================================================

def test_evaluate_matches_sample_bitwise() -> None:
    """evaluate_action reproduces sample_action's log_prob / entropy BITWISE.

    Both build the distribution via the shared `_masked_dist`, so the results come
    from the same ops in the same order — equality is exact, not approximate.
    """
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])

    for deterministic in (False, True):
        torch.manual_seed(7)
        meta, node, lp_sample, ent_sample = sample_action(
            logits, mask, deterministic=deterministic
        )
        lp_eval, ent_eval = evaluate_action(logits, mask, meta, node)

        assert torch.equal(lp_sample, lp_eval), (
            f"log_prob drift (deterministic={deterministic}): "
            f"sample={lp_sample.item()!r} eval={lp_eval.item()!r}"
        )
        assert torch.equal(ent_sample, ent_eval), (
            f"entropy drift (deterministic={deterministic}): "
            f"sample={ent_sample.item()!r} eval={ent_eval.item()!r}"
        )
        # The sampled cell must be a valid one (sanity on the fixture itself).
        assert mask[node, meta] == 0.0


def test_evaluate_scores_every_valid_cell() -> None:
    """Every UNMASKED cell is scorable, and the probabilities sum to 1.

    Sweeping all valid cells proves the flat-index decode (`flat = v*3 + m`) is the
    same convention on both sides: a transposed index would still return a finite
    log-prob, but the total mass would not sum to 1.
    """
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0], seed=3)

    total = 0.0
    n_valid = 0
    for v in range(mask.shape[0]):
        for m in range(NUM_META_ACTIONS):
            if not np.isfinite(mask[v, m]):
                continue
            log_prob, entropy = evaluate_action(logits, mask, m, v)
            assert torch.isfinite(log_prob), (v, m)
            assert torch.isfinite(entropy), (v, m)
            total += float(torch.exp(log_prob).item())
            n_valid += 1

    assert n_valid >= 5, f"fixture is degenerate: only {n_valid} valid cells"
    assert abs(total - 1.0) < 1e-5, (
        f"valid-cell probabilities sum to {total!r}, not 1.0; the flat-index "
        "convention likely diverged between mask and distribution"
    )


# =============================================================================
# (ii) + (iii) End-to-end: real encoder + head, ratio == 1, grads flow
# =============================================================================

def test_ppo_ratio_is_one_and_grads_flow() -> None:
    """The full PPO round-trip on a real policy: store -> re-score -> ratio == 1.

    Mirrors what the PPO update will do: the rollout half runs under `no_grad` and
    keeps only python floats (as `Transition` does); the update half re-runs the
    forward WITH grad on the same observation and the same (unchanged) weights.
    With unchanged weights the ratio pi_new/pi_old must be exactly 1.0 — that is the
    invariant proving no drift between the two paths.
    """
    obs = _make_obs()
    mask = build_action_mask(obs)

    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)

    # --- Rollout half: no_grad, keep DETACHED floats (exactly like Transition). ---
    with torch.no_grad():
        emb = policy.encoder(obs)
        logits_rollout = policy.head(emb)
        torch.manual_seed(11)
        meta_action, node_v, lp_t, ent_t = sample_action(
            logits_rollout, mask, deterministic=False
        )
        stored_log_prob = float(lp_t.item())     # Transition.log_prob
        stored_entropy = float(ent_t.item())     # Transition.entropy
    assert isinstance(stored_log_prob, float) and isinstance(stored_entropy, float)

    # --- Update half: SAME gobs, SAME weights, grad ENABLED. ---
    emb_new = policy.encoder(obs)
    logits_new = policy.head(emb_new)
    log_prob_new, entropy_new = evaluate_action(
        logits_new, mask, meta_action, node_v
    )

    # No no_grad inside evaluate_action -> the caller's grad mode wins.
    assert log_prob_new.requires_grad, "evaluate_action returned a detached log_prob"

    ratio = torch.exp(log_prob_new - torch.tensor(stored_log_prob))
    deviation = float(abs(ratio.item() - 1.0))
    print(f"\n  ratio=exp(lp_new - lp_old)={ratio.item():.12f}  "
          f"max deviation from 1.0 = {deviation:.3e}")
    assert deviation <= 1e-6, f"PPO ratio deviates from 1.0 by {deviation!r}"

    # Entropy round-trips through the float store too.
    assert abs(float(entropy_new.item()) - stored_entropy) <= 1e-6

    # --- Grad sweep over encoder + head (mirrors the encoder selftest). ---
    policy.encoder.zero_grad(set_to_none=True)
    policy.head.zero_grad(set_to_none=True)
    log_prob_new.backward()

    with_grad, without_grad = [], []
    for module_name, module in (("encoder", policy.encoder), ("head", policy.head)):
        for name, p in module.named_parameters():
            full = f"{module_name}.{name}"
            if p.grad is None:
                without_grad.append(full)
                continue
            assert torch.isfinite(p.grad).all(), f"{full} has a non-finite grad"
            with_grad.append(full)

    # `edge_attr_proj` is the encoder's RESERVED edge-attribute path: the builder
    # emits no edge_attr today (graph_tick_loop passes none), so it is legitimately
    # not exercised. Every OTHER parameter must receive a grad.
    _EDGE_ATTR_PARAMS = {"encoder.edge_attr_proj.weight", "encoder.edge_attr_proj.bias"}
    unexpected = [n for n in without_grad if n not in _EDGE_ATTR_PARAMS]
    print(f"  grads: {len(with_grad)} params with finite grad, "
          f"{len(without_grad)} without ({without_grad or 'none'})")
    assert not unexpected, f"parameters got NO grad from evaluate_action: {unexpected}"
    assert len(with_grad) >= 30, f"suspiciously few grads: {len(with_grad)}"

    # At least one grad must be genuinely non-zero (an all-zero sweep would pass the
    # finiteness check while proving nothing about the gradient path).
    any_nonzero = any(
        bool(p.grad.abs().sum() > 0)
        for _, p in list(policy.encoder.named_parameters())
        + list(policy.head.named_parameters())
        if p.grad is not None
    )
    assert any_nonzero, "every grad is exactly zero — the gradient path is dead"


# =============================================================================
# (iv) Guards: fail LOUD, never return -inf
# =============================================================================

def test_masked_cell_raises() -> None:
    """A MASKED stored action raises ValueError instead of returning -inf.

    A masked stored action means the mask rebuilt at update time diverged from the
    rollout-time mask (the rollout can never sample a masked cell). Returning -inf
    would silently corrupt the ratio, so we fail loud.
    """
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])

    # node1 is peer-assigned: both ENGAGEMENT and ABORT are masked there.
    masked_cells = [
        (int(MetaAction.OPPORTUNISTIC_ENGAGEMENT), 1),
        (int(MetaAction.SELF_PRESERVATION_ABORT), 1),
    ]
    for meta, node in masked_cells:
        assert not np.isfinite(mask[node, meta]), f"fixture: ({node},{meta}) not masked"
        try:
            evaluate_action(logits, mask, meta, node)
        except ValueError as exc:
            assert "MASKED" in str(exc), f"unclear error message: {exc}"
        else:
            raise AssertionError(
                f"evaluate_action accepted MASKED cell (node={node}, meta={meta})"
            )


def test_out_of_bounds_raises() -> None:
    """Out-of-range node / meta indices raise ValueError, not an IndexError."""
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])
    k = mask.shape[0]

    for meta, node in [(0, k), (0, -1), (NUM_META_ACTIONS, 0), (-1, 0)]:
        try:
            evaluate_action(logits, mask, meta, node)
        except ValueError as exc:
            assert "out of bounds" in str(exc), f"unclear error message: {exc}"
        else:
            raise AssertionError(
                f"evaluate_action accepted out-of-bounds (node={node}, meta={meta})"
            )


if __name__ == "__main__":
    tests = [
        test_evaluate_matches_sample_bitwise,
        test_evaluate_scores_every_valid_cell,
        test_ppo_ratio_is_one_and_grads_flow,
        test_masked_cell_raises,
        test_out_of_bounds_raises,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"All {len(tests)} evaluate_action tests passed.")
