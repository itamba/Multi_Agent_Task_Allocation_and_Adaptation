"""
Graph Action Module (Phase-2 RL layer)
=======================================

The decision core of the Phase-2 graph + Transformer RL layer: the node-wise
k x 4 meta-action mechanism. This module is **side-by-side** with the flat action
path (``action_config`` / ``action_utils`` / ``action_validation`` / ``plan_editor``),
which stays the frozen baseline; nothing here is wired into ``train_full.py`` yet.

It consumes the :class:`GraphObservation` produced by
``observation/graph_builder.py``: a heterogeneous graph with ``k`` task nodes
(global indices ``[0 .. k-1]``) and ``a`` agent nodes (global indices
``[k .. k+a-1]``, ego first so ``ego_index == k``), plus typed COO edges over the
:class:`EdgeType` codes (SPATIAL, ASSIGNMENT, PRECEDENCE).

The mechanism (FROM THE PAPER, MATCH-AOU paper §4.2.2)
------------------------------------------------------
For each of the ``k`` task nodes the policy chooses one of four meta-actions: a
node-wise **k x 4** decision head with weights SHARED across nodes, scored under
a masked softmax with an ADDITIVE mask ``M in {0, -inf}^{k x 4}``. The joint
``(node, meta-action)`` choice is a single Categorical over the flattened ``k*4``
logits.

What we KEEP / DROP relative to the paper
-----------------------------------------
- OUR CHOICE: we drop the paper's "Local Queue Optimization" meta-action and keep
  §3.3's "Self-Preservation Abort", giving the locked 4-action set in
  :class:`MetaAction`.
- OUR CHOICE: Self-Preservation Abort is **node-indexed** (it targets the ego's own
  assigned task node), not a global action.
- OUR CHOICE: the exact per-cell mask rules in :func:`build_action_mask`.
- OUR CHOICE: "sensed" means the EGO's own sensing only, read from the ego-only
  ``sensed`` task-feature column (``task_features[:, 5]``). Under no-communication the
  ego can act only on what IT senses.
- EXTENDS: Cooperative Recovery is NOT conditioned on "the peer has failed". Under
  no-comms the ego cannot observe peer failure; that inference is **learned** by the
  policy (from ``time_norm`` / graph structure), not encoded in the mask. The mask
  encodes hard physical / structural constraints only.

Mask provenance boundary
------------------------
``capable``, ``reachable``, and ``sensed`` are read from the task-feature COLUMNS only
(``task_features[:, 2]``, ``[:, 3]``, and ``[:, 5]``); reachability is NEVER recomputed
here, and sensing is no longer derived from SPATIAL edges. When ``reachable_by_ego``'s
model is later swapped (round-trip -> marginal-detour) that changes ONLY
``graph_builder``; this mask stays untouched.

Framework: PyTorch (same as ``agent/network.py``).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..observation.graph_builder import GraphObservation, EdgeType


# =============================================================================
# Meta-action set (FROM THE PAPER §4.2.2 names; OUR CHOICE of which to keep)
# =============================================================================

class MetaAction(IntEnum):
    """The locked per-node meta-action set.

    FROM THE PAPER (§4.2.2 / §3.3): the meta-action *names* below.
    OUR CHOICE: we keep these four and drop the paper's "Local Queue Optimization".

    The integer value of each member IS its column index in the ``[k, 4]`` mask /
    logit matrix (e.g. ``mask[v, MetaAction.COOPERATIVE_RECOVERY]``), so member value
    and column index are one and the same by construction.
    """

    PLAN_COMPLIANCE = 0
    COOPERATIVE_RECOVERY = 1
    OPPORTUNISTIC_ENGAGEMENT = 2
    SELF_PRESERVATION_ABORT = 3


NUM_META_ACTIONS = 4  # number of columns in the k x 4 head (== len(MetaAction))


# =============================================================================
# Additive action mask (pure function of the graph — no torch)
# =============================================================================

def build_action_mask(
    obs: GraphObservation,
    capable_threshold: float = 0.5,
    reachable_threshold: float = 0.5,
    sensed_threshold: float = 0.5,
) -> np.ndarray:
    """Build the additive per-node meta-action mask ``M in {0, -inf}^{k x 4}``.

    FROM THE PAPER (§4.2.2): masked softmax with an additive mask in
    ``{0, -inf}``, ready to add to the logits before the softmax. OUR CHOICE: the
    exact per-cell validity rules below.

    Pure function of the graph: no torch, and reachability/capability are read from
    the task-feature COLUMNS only (never recomputed — see module docstring).

    Per-task-node ``v in [0 .. k-1]`` predicates, derived from edges + columns:

    - ``assigned_to_ego[v]``  : an ASSIGNMENT edge with ``src == obs.ego_index`` and
                                ``dst == v``.
    - ``assigned_to_peer[v]`` : an ASSIGNMENT edge with ``dst == v`` and ``src`` an
                                agent node (``src >= k``) that is NOT ``obs.ego_index``.
    - ``unassigned[v]``       : NO ASSIGNMENT edge has ``dst == v``.
    - ``sensed[v]``           : ``task_features[v, 5] >= sensed_threshold`` (the EGO's
                                own sensing only — ego-only column, recomputed each build
                                from the ego's current position; under no-comms the ego can
                                act only on what IT sees).
    - ``capable[v]``          : ``task_features[v, 2] >= capable_threshold``.
    - ``reachable[v]``        : ``task_features[v, 3] >= reachable_threshold``.

    Per-column validity (``0.0`` = valid, ``-inf`` = invalid):

    - PLAN_COMPLIANCE          : ALWAYS valid. Invariant: guarantees >= 1 valid action
                                 per node, so the masked softmax is never all ``-inf``.
    - COOPERATIVE_RECOVERY     : ``assigned_to_peer & sensed & capable & reachable``.
                                 NOT conditioned on "peer is dead" — that is learned
                                 by the policy (EXTENDS), not masked.
    - OPPORTUNISTIC_ENGAGEMENT : ``unassigned & sensed & capable & reachable``.
    - SELF_PRESERVATION_ABORT  : ``assigned_to_ego`` (reachability / capability
                                 irrelevant — abandoning an assignment to preserve
                                 the airframe is always physically available).

    Args:
        obs: the :class:`GraphObservation` to mask.
        capable_threshold: threshold on ``task_features[:, 2]`` for ``capable``.
        reachable_threshold: threshold on ``task_features[:, 3]`` for ``reachable``.
        sensed_threshold: threshold on ``task_features[:, 5]`` for ``sensed``.

    Returns:
        ``np.ndarray`` of shape ``[k, 4]``, dtype ``float32``, values in
        ``{0.0, -inf}``. Column index == :class:`MetaAction` value.

    Edge cases:
        ``k == 0`` -> ``np.zeros((0, 4), float32)``. No edges -> recovery / engagement
        / abort all ``-inf`` while compliance stays valid.
    """
    k = int(obs.task_features.shape[0])
    if k == 0:
        return np.zeros((0, 4), dtype=np.float32)

    ego_index = int(obs.ego_index)

    # --- Derive per-node structural predicates in one pass over the edges ---
    assigned_to_ego = np.zeros(k, dtype=bool)
    has_assignment = np.zeros(k, dtype=bool)   # any ASSIGNMENT edge into v
    assigned_to_peer = np.zeros(k, dtype=bool)

    edge_index = obs.edge_index
    edge_type = obs.edge_type
    num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

    for e in range(num_edges):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])
        etype = int(edge_type[e])

        if etype == int(EdgeType.ASSIGNMENT) and 0 <= dst < k:
            has_assignment[dst] = True
            if src == ego_index:
                assigned_to_ego[dst] = True
            elif src >= k:  # an agent node that is not the ego -> a peer
                assigned_to_peer[dst] = True

    unassigned = ~has_assignment

    # --- Capability / reachability / sensing from the task-feature COLUMNS only ---
    capable = obs.task_features[:, 2] >= capable_threshold
    reachable = obs.task_features[:, 3] >= reachable_threshold
    sensed = obs.task_features[:, 5] >= sensed_threshold  # ego-only sensing column

    # --- Per-column validity -> additive mask ---
    recovery_valid = assigned_to_peer & sensed & capable & reachable
    engagement_valid = unassigned & sensed & capable & reachable
    abort_valid = assigned_to_ego  # capability / reachability irrelevant

    neg_inf = np.float32(-np.inf)
    mask = np.zeros((k, 4), dtype=np.float32)
    # PLAN_COMPLIANCE column stays 0.0 everywhere (always valid; the invariant).
    mask[~recovery_valid, int(MetaAction.COOPERATIVE_RECOVERY)] = neg_inf
    mask[~engagement_valid, int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)] = neg_inf
    mask[~abort_valid, int(MetaAction.SELF_PRESERVATION_ABORT)] = neg_inf
    return mask


# =============================================================================
# Action head (the shared per-node policy MLP) + sampling
# =============================================================================

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init for a linear layer (mirrors ``agent/network.py``).

    std ``sqrt(2)`` for hidden layers, ``0.01`` for the policy output layer so the
    initial policy is close to uniform — the standard PPO convention used by
    ``ActorCriticNetwork``.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActionHead(nn.Module):
    """Shared per-node policy MLP producing the k x 4 meta-action logits.

    FROM THE PAPER (§4.2.2): a node-wise k x 4 head whose weights are SHARED across
    task nodes. Here that sharing is by construction — the head is a plain MLP over
    the last (feature) dimension, so applying it to ``node_embeddings`` of shape
    ``[k, embed_dim]`` yields ``[k, num_meta_actions]`` with the same weights for
    every node.

    The head is decoupled from the graph encoder (which comes later); it takes node
    embeddings as input and knows nothing about how they were produced.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64, num_meta_actions: int = 4):
        """Build the shared head.

        Args:
            embed_dim: per-node embedding dimension (input).
            hidden_dim: hidden width of the shared MLP.
            num_meta_actions: number of output columns (default 4, == len(MetaAction)).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_meta_actions = num_meta_actions

        self.mlp = nn.Sequential(
            _layer_init(nn.Linear(embed_dim, hidden_dim)),
            nn.Tanh(),
            # Small std (0.01) on the output layer -> initial policy near-uniform.
            _layer_init(nn.Linear(hidden_dim, num_meta_actions), std=0.01),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Map node embeddings to per-node meta-action logits.

        Args:
            node_embeddings: ``[k, embed_dim]`` tensor of task-node embeddings.

        Returns:
            ``[k, num_meta_actions]`` logits (weights shared across the ``k`` nodes).
        """
        return self.mlp(node_embeddings)


def sample_action(
    logits: torch.Tensor,
    mask_np: np.ndarray,
    deterministic: bool = False,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """Sample a joint ``(meta_action, node)`` decision under the additive mask.

    FROM THE PAPER (§4.2.2): masked softmax over the k x 4 head. We flatten the
    masked logits row-major to ``[k*4]`` (``flat = v*4 + m``, so ``node = flat // 4``
    and ``meta_action = flat % 4``) and build a single Categorical over the joint
    decision.

    Numerical safety: the additive ``{0, -inf}`` mask is exact for ``sample()`` and
    ``log_prob()`` (invalid cells get exactly zero mass, and we only ever sample a
    valid cell). ``entropy()``, however, sums ``p * log p`` over masked cells where
    ``0 * (-inf) = NaN`` on older torch versions. To stay version-independent we
    compute entropy from a copy of the masked logits clamped to
    ``torch.finfo(dtype).min`` (a large finite negative), so masked cells contribute
    a finite ``~0`` term. A NaN entropy would silently poison the PPO entropy bonus.

    Args:
        logits: ``[k, 4]`` raw logits from :class:`ActionHead`.
        mask_np: ``[k, 4]`` additive mask from :func:`build_action_mask`
            (``{0.0, -inf}``).
        deterministic: if True, take the argmax cell instead of sampling.

    Returns:
        ``(meta_action, node_index, log_prob, entropy)`` where ``meta_action`` and
        ``node_index`` are python ints and ``log_prob`` / ``entropy`` are scalar
        tensors (kept in the autograd graph for the PPO update).

    Raises:
        ValueError: if EVERY entry is ``-inf`` after masking. This should be
            impossible given the Plan-Compliance invariant; we raise a clear error
            rather than produce NaNs.
    """
    mask_t = torch.as_tensor(mask_np, dtype=logits.dtype, device=logits.device)
    masked = logits + mask_t                      # -inf in invalid cells
    flat = masked.reshape(-1)                      # row-major: flat = v*4 + m

    # Guard on the RAW masked logits (before any clamp): the Plan-Compliance
    # invariant should make this unreachable, but failing loud beats NaNs.
    if not torch.isfinite(flat).any():
        raise ValueError(
            "build_action_mask produced an all -inf mask (no valid action); "
            "the Plan-Compliance invariant should make this impossible"
        )

    # Sampling / log_prob: exact distribution with -inf -> zero mass on invalid cells.
    dist = Categorical(logits=flat)
    if deterministic:
        flat_action = torch.argmax(flat)
    else:
        flat_action = dist.sample()
    log_prob = dist.log_prob(flat_action)

    # Entropy: version-independent masked-safe form (clamp -inf -> finfo.min so the
    # masked cells contribute a finite ~0 term instead of 0 * -inf = NaN).
    safe_flat = torch.clamp(flat, min=torch.finfo(flat.dtype).min)
    entropy = Categorical(logits=safe_flat).entropy()

    flat_idx = int(flat_action.item())
    node_index = flat_idx // NUM_META_ACTIONS
    meta_action = flat_idx % NUM_META_ACTIONS
    return meta_action, node_index, log_prob, entropy


# =============================================================================
# Self-test
# =============================================================================

def _selftest() -> None:
    """Hand-crafted graph (no solver/bonmin) with a KNOWN topology to assert exact mask cells.

    Run under nlp_env from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.action.graph_action
    """
    # --- Build a synthetic GraphObservation with a known topology ---
    #   k = 4 task nodes, a = 3 agents (ego_index = 4, peer1 = 5, peer2 = 6)
    #   task 0: assigned to ego (4->0), sensed by ego (col [5]=1), capable=1, reachable=1
    #   task 1: assigned to peer1 (5->1), sensed by ego (col [5]=1), capable=1, reachable=1
    #   task 2: unassigned, sensed by ego (col [5]=1), capable=1, reachable=0 (pop-up, out of fuel range)
    #   task 3: assigned to peer2 (6->3), NOT sensed by ego (col [5]=0),
    #           capable=1, reachable=1 -> REGRESSION case (unsensed in-plan peer target).
    # Sensing is now the ego-only `sensed` column [5] = [1, 1, 1, 0] (replaces the SPATIAL edges).
    task_features = np.array(
        [
            # [utility, dist_to_ego, capable, reachable, probability, sensed]
            [0.80, 0.20, 1.0, 1.0, 1.0, 1.0],   # task 0 (sensed)
            [0.60, 0.40, 1.0, 1.0, 1.0, 1.0],   # task 1 (sensed)
            [0.50, 0.90, 1.0, 0.0, 1.0, 1.0],   # task 2 (sensed, unreachable)
            [0.70, 0.50, 1.0, 1.0, 1.0, 0.0],   # task 3 (assigned to peer2, NOT sensed)
        ],
        dtype=np.float32,
    )
    # agent_features live contract is [a, 1] = [fuel_norm]: ego real, peers 0.0 (featureless).
    agent_features = np.array(
        [
            [0.90],   # ego  (real fuel_norm)
            [0.00],   # peer1 (featureless)
            [0.00],   # peer2 (featureless)
        ],
        dtype=np.float32,
    )
    ego_index = 4  # == k, ego is the first agent node
    # Edges: ASSIGNMENT only — 4->0, 5->1, 6->3 (complete static allocation, incl. peer2).
    # SPATIAL edges are gone; ego sensing is carried by task_features[:, 5].
    edge_index = np.array(
        [[4, 5, 6],
         [0, 1, 3]],
        dtype=np.int64,
    )
    edge_type = np.array(
        [int(EdgeType.ASSIGNMENT), int(EdgeType.ASSIGNMENT), int(EdgeType.ASSIGNMENT)],
        dtype=np.int64,
    )
    obs = GraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=ego_index,
        edge_index=edge_index,
        edge_type=edge_type,
        task_target_ids=["t0", "t1", "t2", "t3"],
        agent_ids=["ego", "peer1", "peer2"],
        agent_id="ego",
        current_time=0,
        time_norm=0.0,
    )

    mask = build_action_mask(obs)

    # --- Expected mask (0.0 = valid, -inf = invalid) ---
    NINF = float("-inf")
    expected = np.array(
        [
            [0.0, NINF, NINF, 0.0],    # node0: PLAN_COMPLIANCE + SELF_PRESERVATION_ABORT
            [0.0, 0.0, NINF, NINF],    # node1: PLAN_COMPLIANCE + COOPERATIVE_RECOVERY
            [0.0, NINF, NINF, NINF],   # node2: PLAN_COMPLIANCE only (unreachable -> no engagement)
            # node3 REGRESSION: an unsensed in-plan peer target is NOT a pop-up. Engagement is
            # masked because task3 HAS an ASSIGNMENT edge (-> not unassigned), and recovery is
            # masked because the ego does not sense it (-> not sensed). Only PLAN_COMPLIANCE.
            [0.0, NINF, NINF, NINF],   # node3: PLAN_COMPLIANCE only
        ],
        dtype=np.float32,
    )

    # --- Print the mask with a column legend ---
    col_names = ["PLAN_COMPLIANCE", "COOPERATIVE_RECOVERY",
                 "OPPORTUNISTIC_ENGAGEMENT", "SELF_PRESERVATION_ABORT"]
    print("=" * 72)
    print("graph_action self-test")
    print("=" * 72)
    print("MetaAction columns (index == MetaAction value):")
    for m in MetaAction:
        print(f"  [{int(m)}] {m.name}")
    print("-" * 72)
    print("action mask  (.  = valid 0.0,  -inf = masked):")
    header = "        " + "  ".join(f"{n[:10]:>10}" for n in col_names)
    print(header)
    for v in range(mask.shape[0]):
        cells = "  ".join(
            f"{'.':>10}" if np.isfinite(mask[v, m]) else f"{'-inf':>10}"
            for m in range(NUM_META_ACTIONS)
        )
        print(f"  node{v}  {cells}")
    print("-" * 72)

    # --- Assert the mask matches the expected topology EXACTLY ---
    same_inf_pattern = np.array_equal(np.isneginf(mask), np.isneginf(expected))
    finite_mask = np.where(np.isfinite(mask), mask, 0.0)
    finite_exp = np.where(np.isfinite(expected), expected, 0.0)
    same_finite = np.array_equal(finite_mask, finite_exp)
    assert same_inf_pattern and same_finite, (
        f"mask mismatch:\n{mask}\nexpected:\n{expected}"
    )

    # Every node must keep >= 1 valid action (the Plan-Compliance invariant).
    assert np.all(np.isfinite(mask).any(axis=1)), "a node has no valid action"
    print("Mask matches expected topology; every node has >= 1 valid action.")

    # --- Exercise the ActionHead + sampling (both stochastic and deterministic) ---
    torch.manual_seed(0)
    head = ActionHead(embed_dim=16)
    embeddings = torch.randn(mask.shape[0], 16)
    logits = head(embeddings)
    assert logits.shape == (mask.shape[0], NUM_META_ACTIONS), logits.shape

    for deterministic in (False, True):
        meta_action, node_index, log_prob, entropy = sample_action(
            logits, mask, deterministic=deterministic
        )
        kind = "deterministic" if deterministic else "stochastic   "
        print(
            f"sample ({kind}): node={node_index} "
            f"meta={MetaAction(meta_action).name} "
            f"log_prob={log_prob.item():.4f} entropy={entropy.item():.4f}"
        )
        # The chosen cell must be a VALID (0.0) cell — never a masked -inf cell.
        assert 0 <= node_index < mask.shape[0]
        assert 0 <= meta_action < NUM_META_ACTIONS
        assert mask[node_index, meta_action] == 0.0, (
            f"sampled a masked cell: node={node_index} meta={meta_action}"
        )
        # log_prob / entropy must be finite (a NaN entropy would poison the PPO bonus).
        assert torch.isfinite(log_prob).all(), "log_prob is not finite"
        assert torch.isfinite(entropy).all(), "entropy is not finite"

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
