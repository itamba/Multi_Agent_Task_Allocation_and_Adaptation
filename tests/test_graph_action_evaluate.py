"""
Unit tests for `evaluate_action` — the PPO-ratio half of the graph action layer.

The rollout is inference-only: `graph_tick_loop._wake_decision` runs under
`torch.no_grad` and stores each wake as a `Transition` holding the
`GraphObservation`, the chosen SEMANTIC `(meta_action, node_v)` identity, and DETACHED
log_prob / entropy floats. The PPO update must recompute the log-prob WITH grad to form
the ratio `pi_new / pi_old`.

The load-bearing property: the update must re-score the action under EXACTLY the
distribution the rollout sampled from. `sample_action` and `evaluate_action` share
one construction site (`_semantic_dist`), so that identity holds BY CONSTRUCTION.
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
  (iv)  the malformed-identity, masked-leaf and out-of-bounds guards raise ValueError.

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
    GLOBAL_META_ACTIONS,
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

PLAN = int(MetaAction.PLAN_COMPLIANCE)
ENGAGE = int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)
ABORT = int(MetaAction.SELF_PRESERVATION_ABORT)


# =============================================================================
# Synthetic fixtures (no BLADE / solver / env)
# =============================================================================

def _make_obs(*, ego_assigned: bool = True) -> GraphObservation:
    """A k=4 / a=3 graph with a KNOWN mask: node0 has ABORT, node2 has ENGAGEMENT.

      task 0: assigned to ego  (4->0), sensed          -> PLAN + ABORT cells
      task 1: assigned to peer1(5->1), sensed          -> PLAN cell
      task 2: UNASSIGNED, sensed, capable, reachable   -> PLAN + ENGAGE cells
      task 3: assigned to peer2(6->3), NOT sensed      -> PLAN cell

    ``ego_assigned=False`` re-points task 0 to peer1, so NO abort cell is legal and the
    one semantic ABORT leaf is masked.
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
    # ego row [fuel_norm, mission_fuel_slack_norm]; peers featureless
    agent_features = np.array([[0.90, 0.10], [0.00, 0.00], [0.00, 0.00]], dtype=np.float32)
    src0 = 4 if ego_assigned else 5
    return GraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=4,  # == k; ego is the first agent node
        edge_index=np.array([[src0, 5, 6],
                             [0, 1, 3]], dtype=np.int64),
        edge_type=np.array([int(EdgeType.ASSIGNMENT)] * 3, dtype=np.int64),
        task_target_ids=["t0", "t1", "t2", "t3"],
        agent_ids=["ego", "peer1", "peer2"],
        agent_id="ego",
        current_time=0,
        time_norm=0.0,
    )


def _fixed_logits(k: int, seed: int = 0) -> torch.Tensor:
    """Deterministic [k, 3] scores, independent of any module (pure torch RNG)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(k, NUM_META_ACTIONS, generator=gen)


def _legal_identities(mask: np.ndarray):
    """Every LEGAL semantic identity of a mask: PLAN, ABORT if legal, legal ENGAGE(i)."""
    out = [(PLAN, None)]
    if np.isfinite(mask[:, ABORT]).any():
        out.append((ABORT, None))
    out.extend((ENGAGE, v) for v in range(mask.shape[0]) if np.isfinite(mask[v, ENGAGE]))
    return out


# =============================================================================
# (i) Exact agreement between sample_action and evaluate_action
# =============================================================================

def test_evaluate_matches_sample_bitwise() -> None:
    """evaluate_action reproduces sample_action's log_prob / entropy BITWISE."""
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])

    for deterministic in (False, True):
        for seed in range(25):
            torch.manual_seed(seed)
            meta, node, lp_sample, ent_sample = sample_action(
                logits, mask, deterministic=deterministic
            )
            lp_eval, ent_eval = evaluate_action(logits, mask, meta, node)
            assert torch.equal(lp_sample, lp_eval), (deterministic, seed)
            assert torch.equal(ent_sample, ent_eval), (deterministic, seed)
            # The sampled identity is a legal, well-formed semantic one.
            assert (meta, node) in _legal_identities(mask)
            assert (node is None) == (meta in GLOBAL_META_ACTIONS)


def test_evaluate_scores_every_legal_semantic_leaf() -> None:
    """Every LEGAL semantic leaf is scorable, and the probabilities sum to 1.

    A leaf-order or legality mistake would still return finite log-probs, but the total
    mass over the legal identities would not be 1.
    """
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0], seed=3)

    identities = _legal_identities(mask)
    assert identities == [(PLAN, None), (ABORT, None), (ENGAGE, 2)], identities
    total = 0.0
    for meta, node in identities:
        log_prob, entropy = evaluate_action(logits, mask, meta, node)
        assert torch.isfinite(log_prob) and torch.isfinite(entropy), (meta, node)
        total += float(torch.exp(log_prob).item())
    assert abs(total - 1.0) < 1e-5, total


# =============================================================================
# (ii) + (iii) End-to-end: real encoder + head, ratio == 1, grads flow
# =============================================================================

def test_ppo_ratio_is_one_and_grads_flow() -> None:
    """The full PPO round-trip on a real policy: store -> re-score -> ratio == 1."""
    obs = _make_obs()
    mask = build_action_mask(obs)

    torch.manual_seed(0)
    policy = build_policy(embed_dim=64)

    with torch.no_grad():
        logits_rollout = policy.head(policy.encoder(obs))
        torch.manual_seed(11)
        meta_action, node_v, lp_t, ent_t = sample_action(
            logits_rollout, mask, deterministic=False
        )
        stored_log_prob = float(lp_t.item())     # Transition.log_prob
        stored_entropy = float(ent_t.item())     # Transition.entropy

    logits_new = policy.head(policy.encoder(obs))
    log_prob_new, entropy_new = evaluate_action(logits_new, mask, meta_action, node_v)
    assert log_prob_new.requires_grad, "evaluate_action returned a detached log_prob"

    # The float store is exact for a float32 log-prob: the ratio is exactly 1.0.
    ratio = torch.exp(log_prob_new - stored_log_prob)
    assert float(ratio.item()) == 1.0, ratio.item()
    assert float(entropy_new.item()) == stored_entropy

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
    _EDGE_ATTR_PARAMS = {"encoder.edge_attr_proj.weight", "encoder.edge_attr_proj.bias"}
    unexpected = [n for n in without_grad if n not in _EDGE_ATTR_PARAMS]
    assert not unexpected, f"parameters got NO grad from evaluate_action: {unexpected}"
    assert len(with_grad) >= 30, f"suspiciously few grads: {len(with_grad)}"
    assert any(bool(p.grad.abs().sum() > 0)
               for _, p in list(policy.encoder.named_parameters())
               + list(policy.head.named_parameters()) if p.grad is not None)


# =============================================================================
# (iv) Guards: fail LOUD, never return -inf
# =============================================================================

def _raises(fn, *args, contains: str) -> None:
    try:
        fn(*args)
    except ValueError as exc:
        assert contains in str(exc), f"unclear error message: {exc}"
    else:
        raise AssertionError(f"accepted {args[2:]!r}")


def test_masked_semantic_leaf_raises() -> None:
    """A MASKED stored ENGAGE, or an ABORT with no legal abort cell, raises."""
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])
    for node in (0, 1, 3):  # ENGAGE is legal on node 2 only
        _raises(evaluate_action, logits, mask, ENGAGE, node, contains="MASKED")

    no_abort = build_action_mask(_make_obs(ego_assigned=False))
    assert not np.isfinite(no_abort[:, ABORT]).any()
    _raises(evaluate_action, logits, no_abort, ABORT, None, contains="MASKED")


def test_malformed_semantic_identity_raises() -> None:
    """A global action carrying a node, or an ENGAGE without an integer node, raises."""
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])
    for meta in (PLAN, ABORT):
        for node in (0, 2):
            _raises(evaluate_action, logits, mask, meta, node, contains="malformed")
    _raises(evaluate_action, logits, mask, ENGAGE, None, contains="malformed")
    _raises(evaluate_action, logits, mask, ENGAGE, True, contains="malformed")
    _raises(evaluate_action, logits, mask, ENGAGE, 2.0, contains="malformed")


def test_out_of_bounds_raises() -> None:
    """Out-of-range ENGAGE nodes / meta-actions raise ValueError, not an IndexError."""
    obs = _make_obs()
    mask = build_action_mask(obs)
    logits = _fixed_logits(mask.shape[0])
    k = mask.shape[0]
    for meta, node in [(ENGAGE, k), (ENGAGE, -1), (NUM_META_ACTIONS, None), (-1, None)]:
        _raises(evaluate_action, logits, mask, meta, node, contains="out of bounds")


if __name__ == "__main__":
    tests = [
        test_evaluate_matches_sample_bitwise,
        test_evaluate_scores_every_legal_semantic_leaf,
        test_ppo_ratio_is_one_and_grads_flow,
        test_masked_semantic_leaf_raises,
        test_malformed_semantic_identity_raises,
        test_out_of_bounds_raises,
    ]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"All {len(tests)} evaluate_action tests passed.")
