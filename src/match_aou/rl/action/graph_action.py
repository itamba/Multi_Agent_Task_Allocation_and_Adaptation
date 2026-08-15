"""
Graph Action Module (Phase-2 RL layer)
=======================================

The decision core of the Phase-2 graph + Transformer RL layer: the node-wise
k x 3 meta-action mechanism. It replaced the retired flat action path; its consumer
is the graph tick-loop (``training/graph_tick_loop.py``).

It consumes the :class:`GraphObservation` produced by
``observation/graph_builder.py``: a heterogeneous graph with ``k`` task nodes
(global indices ``[0 .. k-1]``) and ``a`` agent nodes (global indices
``[k .. k+a-1]``, ego first so ``ego_index == k``), plus typed COO edges over the
:class:`EdgeType` codes (SPATIAL, ASSIGNMENT, PRECEDENCE).

The mechanism (FROM THE PAPER, MATCH-AOU paper §4.2.2)
------------------------------------------------------
For each of the ``k`` task nodes the policy chooses one of three meta-actions: a
node-wise **k x 3** decision head with weights SHARED across nodes, scored under
a masked softmax with an ADDITIVE mask ``M in {0, -inf}^{k x 3}``. The joint
``(node, meta-action)`` choice is a single Categorical over the flattened ``k*3``
logits.

What we KEEP / DROP relative to the paper
-----------------------------------------
- OUR CHOICE: we drop the paper's "Local Queue Optimization" meta-action and keep
  §3.3's "Self-Preservation Abort", giving the locked 3-action set in
  :class:`MetaAction`.
- OUR CHOICE: Self-Preservation Abort keeps the shared node-indexed SELECTION
  identity — it is chosen as a ``k x 3`` cell on one of the ego's own assigned task
  nodes, and that cell is what is sampled, stored and re-scored by PPO. Its EFFECT
  SCOPE is a different question and is EGO-GLOBAL: ``graph_effect.apply_meta_action``
  clears the acting ego's whole remaining plan, which the executor turns into RTB. The
  mask below governs SELECTION only; it is unchanged by that.
- OUR CHOICE: the exact per-cell mask rules in :func:`build_action_mask`.
- OUR CHOICE: "sensed" means the EGO's own sensing only, read from the ego-only
  ``sensed`` task-feature column (``task_features[:, 5]``). Under no-communication the
  ego can act only on what IT senses.

Mask provenance boundary
------------------------
``capable``, ``reachable``, and ``sensed`` are read from the task-feature COLUMNS only
(``task_features[:, 2]``, ``[:, 3]``, and ``[:, 5]``); reachability is NEVER recomputed
here, and sensing is no longer derived from SPATIAL edges. When ``reachable_by_ego``'s
model is later swapped (round-trip -> marginal-detour) that changes ONLY
``graph_builder``; this mask stays untouched.

Framework: PyTorch.
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
    OUR CHOICE: we keep three and drop the paper's "Local Queue Optimization";
    Cooperative Recovery is also removed (4->3) — peer-failure recovery is handled
    upstream by the trigger layer (a peer-overdue sensed target becomes a pop-up the
    policy may OPPORTUNISTIC_ENGAGEMENT), so a CR column would be dead.

    The integer value of each member IS its column index in the ``[k, 3]`` mask /
    logit matrix (e.g. ``mask[v, MetaAction.OPPORTUNISTIC_ENGAGEMENT]``), so member value
    and column index are one and the same by construction.

    SELECTION vs EFFECT — two separate things. EVERY member keeps the same node-indexed
    SELECTION identity: what is sampled, stored and re-scored by PPO is a ``(node, meta)``
    cell. What the chosen cell then DOES to the plan differs per member:

    - PLAN_COMPLIANCE          : no plan edit at all (the node is selection only).
    - OPPORTUNISTIC_ENGAGEMENT : NODE-LOCAL effect — it assigns the ego to THAT task node.
    - SELF_PRESERVATION_ABORT  : EGO-GLOBAL effect — it clears the acting ego's whole
      remaining plan, so every legal cell produces the same result.

    The effects themselves live in ``graph_effect.apply_meta_action``.
    """

    PLAN_COMPLIANCE = 0
    OPPORTUNISTIC_ENGAGEMENT = 1
    SELF_PRESERVATION_ABORT = 2


NUM_META_ACTIONS = 3  # number of columns in the k x 3 head (== len(MetaAction))


# =============================================================================
# Additive action mask (pure function of the graph — no torch)
# =============================================================================

def build_action_mask(
    obs: GraphObservation,
    capable_threshold: float = 0.5,
    reachable_threshold: float = 0.5,
    sensed_threshold: float = 0.5,
) -> np.ndarray:
    """Build the additive per-node meta-action mask ``M in {0, -inf}^{k x 3}``.

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
    - OPPORTUNISTIC_ENGAGEMENT : ``unassigned & sensed & capable & reachable``.
    - SELF_PRESERVATION_ABORT  : ``assigned_to_ego`` (reachability / capability
                                 irrelevant — abandoning the mission to preserve the
                                 airframe is always physically available). This LEGALITY
                                 rule is per-node and unchanged; the chosen cell's EFFECT
                                 is ego-global and belongs to ``graph_effect``.

    Args:
        obs: the :class:`GraphObservation` to mask.
        capable_threshold: threshold on ``task_features[:, 2]`` for ``capable``.
        reachable_threshold: threshold on ``task_features[:, 3]`` for ``reachable``.
        sensed_threshold: threshold on ``task_features[:, 5]`` for ``sensed``.

    Returns:
        ``np.ndarray`` of shape ``[k, 3]``, dtype ``float32``, values in
        ``{0.0, -inf}``. Column index == :class:`MetaAction` value.

    Edge cases:
        ``k == 0`` -> ``np.zeros((0, 3), float32)``. No edges -> engagement / abort
        all ``-inf`` while compliance stays valid.
    """
    k = int(obs.task_features.shape[0])
    if k == 0:
        return np.zeros((0, 3), dtype=np.float32)

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
    engagement_valid = unassigned & sensed & capable & reachable
    abort_valid = assigned_to_ego  # capability / reachability irrelevant

    neg_inf = np.float32(-np.inf)
    mask = np.zeros((k, 3), dtype=np.float32)
    # PLAN_COMPLIANCE column stays 0.0 everywhere (always valid; the invariant).
    mask[~engagement_valid, int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)] = neg_inf
    mask[~abort_valid, int(MetaAction.SELF_PRESERVATION_ABORT)] = neg_inf
    return mask


# =============================================================================
# Action head (the shared per-node policy MLP) + sampling
# =============================================================================

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init for a linear layer — the standard PPO scheme.

    std ``sqrt(2)`` for hidden layers, ``0.01`` for the policy output layer so the
    initial policy is close to uniform — the standard PPO convention.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActionHead(nn.Module):
    """Shared per-node policy MLP producing the k x 3 meta-action logits.

    FROM THE PAPER (§4.2.2): a node-wise k x 3 head whose weights are SHARED across
    task nodes. Here that sharing is by construction — the head is a plain MLP over
    the last (feature) dimension, so applying it to ``node_embeddings`` of shape
    ``[k, embed_dim]`` yields ``[k, num_meta_actions]`` with the same weights for
    every node.

    The head is decoupled from the graph encoder (which comes later); it takes node
    embeddings as input and knows nothing about how they were produced.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64, num_meta_actions: int = 3):
        """Build the shared head.

        Args:
            embed_dim: per-node embedding dimension (input).
            hidden_dim: hidden width of the shared MLP.
            num_meta_actions: number of output columns (default 3, == len(MetaAction)).
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


def _masked_dist(
    logits: torch.Tensor,
    mask_np: np.ndarray,
) -> Tuple[torch.Tensor, Categorical, torch.Tensor]:
    """Build THE joint masked distribution over the flattened ``k*3`` decision.

    THE single construction site. :func:`sample_action` (rollout, no-grad) and
    :func:`evaluate_action` (PPO update, with grad) both route through here, so the
    distribution they act on is identical BY CONSTRUCTION rather than by two code
    paths agreeing. That identity is load-bearing: any drift between the rollout
    distribution and the update distribution would corrupt the PPO ratio
    ``pi_new / pi_old`` SILENTLY — no crash, just poisoned learning. Do not
    reimplement any part of this in a caller.

    Encapsulates exactly: mask -> tensor, additive masking, row-major flatten
    (``flat = v*3 + m``), the all ``-inf`` guard, the Categorical, and the
    clamped-logits entropy (see :func:`sample_action` for the entropy rationale).

    Args:
        logits: ``[k, 3]`` raw logits from :class:`ActionHead`. Grad-attached or not
            — this helper never detaches and never touches grad mode.
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`
            (``{0.0, -inf}``).

    Returns:
        ``(flat, dist, entropy)`` — the flattened masked logits ``[k*3]``, the
        Categorical over them, and the masked-safe scalar entropy.

    Raises:
        ValueError: if EVERY entry is ``-inf`` after masking.
    """
    mask_t = torch.as_tensor(mask_np, dtype=logits.dtype, device=logits.device)
    masked = logits + mask_t                      # -inf in invalid cells
    flat = masked.reshape(-1)                      # row-major: flat = v*3 + m

    # Guard on the RAW masked logits (before any clamp): the Plan-Compliance
    # invariant should make this unreachable, but failing loud beats NaNs.
    if not torch.isfinite(flat).any():
        raise ValueError(
            "build_action_mask produced an all -inf mask (no valid action); "
            "the Plan-Compliance invariant should make this impossible"
        )

    # Exact distribution: -inf -> zero mass on invalid cells.
    dist = Categorical(logits=flat)

    # Entropy: version-independent masked-safe form (clamp -inf -> finfo.min so the
    # masked cells contribute a finite ~0 term instead of 0 * -inf = NaN).
    safe_flat = torch.clamp(flat, min=torch.finfo(flat.dtype).min)
    entropy = Categorical(logits=safe_flat).entropy()

    return flat, dist, entropy


def sample_action(
    logits: torch.Tensor,
    mask_np: np.ndarray,
    deterministic: bool = False,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """Sample a joint ``(meta_action, node)`` decision under the additive mask.

    FROM THE PAPER (§4.2.2): masked softmax over the k x 3 head. We flatten the
    masked logits row-major to ``[k*3]`` (``flat = v*3 + m``, so ``node = flat // 3``
    and ``meta_action = flat % 3``) and build a single Categorical over the joint
    decision.

    Numerical safety: the additive ``{0, -inf}`` mask is exact for ``sample()`` and
    ``log_prob()`` (invalid cells get exactly zero mass, and we only ever sample a
    valid cell). ``entropy()``, however, sums ``p * log p`` over masked cells where
    ``0 * (-inf) = NaN`` on older torch versions. To stay version-independent we
    compute entropy from a copy of the masked logits clamped to
    ``torch.finfo(dtype).min`` (a large finite negative), so masked cells contribute
    a finite ``~0`` term. A NaN entropy would silently poison the PPO entropy bonus.

    The distribution itself is built by :func:`_masked_dist` — the SHARED
    construction site this function and :func:`evaluate_action` both call, so the
    PPO update re-scores an action under exactly the distribution it was sampled
    from. Everything below the helper call is sampling + flat-index decode only.

    Args:
        logits: ``[k, 3]`` raw logits from :class:`ActionHead`.
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`
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
    flat, dist, entropy = _masked_dist(logits, mask_np)

    # Sampling / log_prob: -inf cells carry exactly zero mass, so a sampled cell is
    # always a valid one.
    if deterministic:
        flat_action = torch.argmax(flat)
    else:
        flat_action = dist.sample()
    log_prob = dist.log_prob(flat_action)

    flat_idx = int(flat_action.item())
    node_index = flat_idx // NUM_META_ACTIONS
    meta_action = flat_idx % NUM_META_ACTIONS
    return meta_action, node_index, log_prob, entropy


def evaluate_action(
    logits: torch.Tensor,
    mask_np: np.ndarray,
    meta_action: int,
    node_v: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-score an ALREADY-CHOSEN ``(meta_action, node_v)`` — the PPO-ratio half.

    Purpose (PPO). The rollout is inference-only: ``graph_tick_loop._wake_decision``
    runs under ``torch.no_grad`` and stores each wake as a ``Transition`` holding the
    ``GraphObservation``, the chosen ``(meta_action, node_v)``, and DETACHED
    ``log_prob`` / ``entropy`` floats. The PPO update must recompute the log-prob WITH
    grad to form the ratio ``exp(log_prob_new - log_prob_old) = pi_new / pi_old``. This
    function is that recomputation: re-encode the stored ``gobs``, re-run the head, and
    call this with the stored action.

    Identity BY CONSTRUCTION. The distribution is built by the SAME
    :func:`_masked_dist` helper :func:`sample_action` used at rollout time — same
    additive masking, same row-major flatten (``flat = node_v*3 + meta_action``), same
    Categorical, same clamped-logits entropy. So on the first PPO epoch (unchanged
    weights) the returned ``log_prob`` is BITWISE equal to the stored one and the ratio
    is exactly ``1.0``. This is not a coincidence to be re-verified per call — it holds
    because there is only one construction path. Never reimplement the construction
    here; that would reintroduce the drift this design exists to prevent (a drift that
    fails SILENTLY — it only shows up as a corrupted policy gradient).

    Grad contract: NO ``torch.no_grad`` anywhere inside. The CALLER controls grad mode.
    Gradients flow from the returned tensors back through ``logits`` to whatever
    produced them (head + encoder).

    Args:
        logits: ``[k, 3]`` raw logits from :class:`ActionHead`, normally grad-attached
            (the PPO update re-runs the forward pass with grad enabled).
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`, rebuilt from
            the STORED observation so it reproduces the rollout-time mask.
        meta_action: the stored :class:`MetaAction` value (column, ``0..2``).
        node_v: the stored task-node index (row, ``0..k-1``).

    Returns:
        ``(log_prob, entropy)`` — scalar tensors. ``log_prob`` is the joint log-prob of
        the requested cell; ``entropy`` is the SAME masked-safe policy entropy
        :func:`sample_action` would have reported for this state.

    Raises:
        ValueError: if the mask is all ``-inf`` (via :func:`_masked_dist`); if
            ``(node_v, meta_action)`` is out of bounds; or if the requested cell is
            MASKED. A masked stored action means the mask reconstructed at update time
            diverged from the rollout-time mask (a stale/mismatched observation), which
            would otherwise silently feed ``-inf`` into the ratio. We fail LOUD instead.
    """
    k = int(mask_np.shape[0])
    node_v = int(node_v)
    meta_action = int(meta_action)

    if not (0 <= node_v < k) or not (0 <= meta_action < NUM_META_ACTIONS):
        raise ValueError(
            f"evaluate_action: action cell (node_v={node_v}, meta_action={meta_action}) "
            f"is out of bounds for a [{k}, {NUM_META_ACTIONS}] mask"
        )

    # Guard BEFORE building the distribution: a masked stored action is a mask
    # reconstruction bug, not a legitimate zero-probability action.
    if not np.isfinite(mask_np[node_v, meta_action]):
        raise ValueError(
            f"evaluate_action: the stored action (node_v={node_v}, "
            f"meta_action={MetaAction(meta_action).name}) is MASKED (-inf) in the "
            "supplied mask. The rollout could not have sampled it, so the mask "
            "rebuilt at update time diverged from the rollout-time mask; check that "
            "the stored GraphObservation is the one the action was sampled on."
        )

    flat, dist, entropy = _masked_dist(logits, mask_np)

    # Row-major flatten, identical to sample_action's decode (flat = v*3 + m).
    flat_idx = node_v * NUM_META_ACTIONS + meta_action
    # 0-dim long tensor: exactly the shape/dtype dist.sample() returns, so log_prob
    # takes the same gather path and the result is bitwise identical.
    flat_action = torch.as_tensor(flat_idx, dtype=torch.long, device=flat.device)
    log_prob = dist.log_prob(flat_action)

    return log_prob, entropy


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
            [0.0, NINF, 0.0],    # node0: PLAN_COMPLIANCE + SELF_PRESERVATION_ABORT
            [0.0, NINF, NINF],   # node1: PLAN_COMPLIANCE only (peer-assigned+sensed, but CR removed 4->3)
            [0.0, NINF, NINF],   # node2: PLAN_COMPLIANCE only (unreachable -> no engagement)
            # node3 REGRESSION: an unsensed in-plan peer target is NOT a pop-up. Engagement is
            # masked because task3 HAS an ASSIGNMENT edge (-> not unassigned). Only PLAN_COMPLIANCE.
            [0.0, NINF, NINF],   # node3: PLAN_COMPLIANCE only
        ],
        dtype=np.float32,
    )

    # --- Print the mask with a column legend ---
    col_names = ["PLAN_COMPLIANCE",
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

    # -------------------------------------------------------------------------
    # evaluate_action (the PPO-ratio half) — the shared-construction identity.
    # -------------------------------------------------------------------------
    print("-" * 72)
    print("evaluate_action (PPO re-scoring):")

    # [E1] EXACT agreement with sample_action. Both route through _masked_dist, so
    #      the two log-probs / entropies are BITWISE equal (torch.equal, not allclose).
    #      This is the property the whole refactor exists to guarantee: if these ever
    #      diverge, the PPO ratio pi_new/pi_old is wrong on epoch 0 and learning is
    #      silently poisoned.
    for deterministic in (False, True):
        meta_s, node_s, lp_s, ent_s = sample_action(
            logits, mask, deterministic=deterministic
        )
        lp_e, ent_e = evaluate_action(logits, mask, meta_s, node_s)
        kind = "deterministic" if deterministic else "stochastic   "
        assert torch.equal(lp_s, lp_e), (
            f"log_prob drift ({kind}): sample={lp_s.item()!r} eval={lp_e.item()!r}"
        )
        assert torch.equal(ent_s, ent_e), (
            f"entropy drift ({kind}): sample={ent_s.item()!r} eval={ent_e.item()!r}"
        )
        print(f"  [E1] {kind}: node={node_s} meta={MetaAction(meta_s).name} "
              f"log_prob={lp_e.item():.6f} entropy={ent_e.item():.6f}  "
              f"BITWISE == sample_action   OK")

    # [E2] The ratio a PPO epoch-0 update would form is exactly 1.0.
    ratio = torch.exp(lp_e - lp_s)
    assert torch.equal(ratio, torch.ones_like(ratio)), f"ratio != 1: {ratio.item()!r}"
    print(f"  [E2] exp(log_prob_new - log_prob_old) == {ratio.item():.1f} exactly   OK")

    # [E3] Grad flows: NO no_grad inside evaluate_action, so the caller's grad mode
    #      wins and every head parameter receives a finite grad. (The full
    #      encoder+head sweep lives in tests/test_graph_action_evaluate.py.)
    head.zero_grad(set_to_none=True)
    logits_grad = head(embeddings)  # fresh forward, grad-attached
    lp_g, ent_g = evaluate_action(logits_grad, mask, meta_s, node_s)
    assert lp_g.requires_grad, "evaluate_action returned a detached log_prob"
    lp_g.backward()
    n_params = 0
    for name, p in head.named_parameters():
        assert p.grad is not None, f"parameter {name} has no grad"
        assert torch.isfinite(p.grad).all(), f"parameter {name} has non-finite grad"
        n_params += 1
    print(f"  [E3] backward through evaluate_action: all {n_params} head params "
          f"have finite grads   OK")

    # [E4] A MASKED stored action fails LOUD (mask reconstruction diverged), rather
    #      than quietly returning -inf and corrupting the ratio. node1/OE is -inf.
    assert not np.isfinite(mask[1, int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)]), \
        "test setup: node1/OE was expected to be masked"
    try:
        evaluate_action(logits, mask, int(MetaAction.OPPORTUNISTIC_ENGAGEMENT), 1)
    except ValueError as exc:
        print(f"  [E4] masked cell (node1, OE) -> ValueError   OK\n"
              f"       {str(exc).splitlines()[0][:96]}...")
    else:
        raise AssertionError("evaluate_action accepted a MASKED action cell")

    # [E5] Out-of-bounds cells are rejected too (node index past k).
    try:
        evaluate_action(logits, mask, 0, mask.shape[0])
    except ValueError:
        print("  [E5] out-of-bounds node index -> ValueError   OK")
    else:
        raise AssertionError("evaluate_action accepted an out-of-bounds node index")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
