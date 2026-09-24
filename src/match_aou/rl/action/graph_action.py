"""
Graph Action Module (Phase-2 RL layer)
=======================================

The decision core of the Phase-2 graph + Transformer RL layer. It replaced the retired
flat action path; its consumer is the graph tick-loop (``training/graph_tick_loop.py``).

It consumes the :class:`GraphObservation` produced by
``observation/graph_builder.py``: a heterogeneous graph with ``k`` task nodes
(global indices ``[0 .. k-1]``) and ``a`` agent nodes (global indices
``[k .. k+a-1]``, ego first so ``ego_index == k``), plus typed COO edges over the
:class:`EdgeType` codes (SPATIAL, ASSIGNMENT, PRECEDENCE).

The mechanism
-------------
FROM THE PAPER (MATCH-AOU paper §4.2.2): a node-wise **k x 3** decision head with
weights SHARED across nodes, and an ADDITIVE legality mask ``M in {0, -inf}^{k x 3}``.
Both are kept: :class:`ActionHead` still emits one score per ``(task node, meta-action)``
cell and :func:`build_action_mask` still states per-cell legality.

OUR CHOICE — THE SEMANTIC ACTION SPACE (:data:`ACTION_REPRESENTATION_ID`). The policy
does NOT act on the flattened ``k*3`` cells. Two of the three meta-actions have no
node-scoped meaning — ``PLAN_COMPLIANCE`` edits nothing and
``SELF_PRESERVATION_ABORT`` clears the ego's whole plan — so giving each of them ``k``
node-indexed aliases made one semantic action look like ``k`` actions. The categorical
the actor samples, stores and re-scores is therefore over ``k + 2`` SEMANTIC LEAVES in a
fixed order::

    leaf 0        global PLAN_COMPLIANCE           (node None)
    leaf 1        global SELF_PRESERVATION_ABORT   (node None)
    leaf 2 + i    OPPORTUNISTIC_ENGAGEMENT(task i) (node i), i in [0, k)

derived from the EXISTING ``k x 3`` scores ``z`` by count-normalized collapse
(``logmeanexp``) — no new head and no new actor input:

    s_PLAN     = logsumexp_v z[v, PLAN]                   - log(k)
    s_ABORT    = logsumexp_{v abort-legal} z[v, ABORT]    - log(n_abort_legal)
    s_ENGAGE_i = z[i, ENGAGE]

Count normalization is load-bearing: duplicating equal PLAN / ABORT evidence over more
nodes creates no multiplicity bonus. ``ENGAGE`` stays node-local because its effect is.
Illegal leaves are masked exactly (``-inf``): ABORT is legal iff at least one ABORT cell
is legal; ``ENGAGE(i)`` is legal iff its cell is.

What we KEEP / DROP relative to the paper
-----------------------------------------
- OUR CHOICE: we drop the paper's "Local Queue Optimization" meta-action and keep
  §3.3's "Self-Preservation Abort", giving the locked 3-action set in
  :class:`MetaAction`.
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

import math
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..observation.graph_builder import GraphObservation, EdgeType


# =============================================================================
# Meta-action set (FROM THE PAPER §4.2.2 names; OUR CHOICE of which to keep)
# =============================================================================

class MetaAction(IntEnum):
    """The locked meta-action set.

    FROM THE PAPER (§4.2.2 / §3.3): the meta-action *names* below.
    OUR CHOICE: we keep three and drop the paper's "Local Queue Optimization";
    Cooperative Recovery is also removed (4->3) — peer-failure recovery is handled
    upstream by the trigger layer (a peer-overdue sensed target becomes a pop-up the
    policy may OPPORTUNISTIC_ENGAGEMENT), so a CR column would be dead.

    The integer value of each member IS its column index in the ``[k, 3]`` mask /
    score matrix (e.g. ``mask[v, MetaAction.OPPORTUNISTIC_ENGAGEMENT]``).

    SEMANTIC SELECTION IDENTITY (:data:`ACTION_REPRESENTATION_ID`). What is sampled,
    stored and re-scored is a ``(meta_action, node_v)`` pair whose node is NULLABLE:

    - PLAN_COMPLIANCE          : ``node_v is None`` — one global action, no plan edit.
    - OPPORTUNISTIC_ENGAGEMENT : ``node_v`` = the task index — NODE-LOCAL effect (it
      assigns the ego to THAT task node).
    - SELF_PRESERVATION_ABORT  : ``node_v is None`` — one global action, EGO-GLOBAL
      effect (it clears the acting ego's whole remaining plan).

    The effects themselves live in ``graph_effect.apply_meta_action``.
    """

    PLAN_COMPLIANCE = 0
    OPPORTUNISTIC_ENGAGEMENT = 1
    SELF_PRESERVATION_ABORT = 2


NUM_META_ACTIONS = 3  # number of columns in the k x 3 score head (== len(MetaAction))

#: The ONE identifier of the action representation this module implements. Persisted in
#: run configuration, checkpoints, per-wake diagnostics and credit diagnostics, so no
#: artifact can be read under the wrong action semantics.
ACTION_REPRESENTATION_ID = "semantic_k_plus_2_logmeanexp_v1"

#: Fixed semantic leaf layout: ``[PLAN, ABORT, ENGAGE(0), ..., ENGAGE(k-1)]``.
SEMANTIC_PLAN_LEAF = 0
SEMANTIC_ABORT_LEAF = 1
SEMANTIC_ENGAGE_LEAF_OFFSET = 2
#: The two meta-actions whose semantic identity carries NO node.
GLOBAL_META_ACTIONS = (int(MetaAction.PLAN_COMPLIANCE),
                       int(MetaAction.SELF_PRESERVATION_ABORT))

_PLAN = int(MetaAction.PLAN_COMPLIANCE)
_ENGAGE = int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)
_ABORT = int(MetaAction.SELF_PRESERVATION_ABORT)


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
    ``{0, -inf}``. OUR CHOICE: the exact per-cell validity rules below. The mask is the
    SOURCE of semantic-leaf legality (:func:`_semantic_dist`); it is not itself the
    action space.

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

    - PLAN_COMPLIANCE          : ALWAYS valid. Invariant: guarantees the semantic PLAN
                                 leaf is legal, so the semantic softmax is never all
                                 ``-inf`` for ``k > 0``.
    - OPPORTUNISTIC_ENGAGEMENT : ``unassigned & sensed & capable & reachable``.
    - SELF_PRESERVATION_ABORT  : ``assigned_to_ego`` (reachability / capability
                                 irrelevant — abandoning the mission to preserve the
                                 airframe is always physically available). The ONE
                                 semantic ABORT leaf is legal iff any of these is.

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
# Action head (the shared per-node score MLP)
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
    """Shared per-node MLP producing the k x 3 source scores.

    FROM THE PAPER (§4.2.2): a node-wise k x 3 head whose weights are SHARED across
    task nodes. Here that sharing is by construction — the head is a plain MLP over
    the last (feature) dimension, so applying it to ``node_embeddings`` of shape
    ``[k, embed_dim]`` yields ``[k, num_meta_actions]`` with the same weights for
    every node. Its output is the SOURCE of the semantic distribution
    (:func:`_semantic_dist`), not a distribution over cells.

    The head is decoupled from the graph encoder; it takes node embeddings as input and
    knows nothing about how they were produced.
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
        """Map node embeddings to per-node meta-action scores.

        Args:
            node_embeddings: ``[k, embed_dim]`` tensor of task-node embeddings.

        Returns:
            ``[k, num_meta_actions]`` scores (weights shared across the ``k`` nodes).
        """
        return self.mlp(node_embeddings)


# =============================================================================
# Semantic leaf identity
# =============================================================================

def semantic_leaf_identity(leaf: int, k: int) -> Tuple[int, Optional[int]]:
    """Decode a semantic leaf index into its ``(meta_action, node_v)`` identity."""
    leaf = int(leaf)
    if leaf == SEMANTIC_PLAN_LEAF:
        return _PLAN, None
    if leaf == SEMANTIC_ABORT_LEAF:
        return _ABORT, None
    node = leaf - SEMANTIC_ENGAGE_LEAF_OFFSET
    if not (0 <= node < int(k)):
        raise ValueError("semantic leaf %d is out of range for k=%d" % (leaf, int(k)))
    return _ENGAGE, node


def semantic_leaf_index(meta_action: Any, node_v: Any, k: int) -> int:
    """The semantic leaf of a ``(meta_action, node_v)`` identity, validated LOUDLY.

    Refuses every malformed identity rather than coercing it into a leaf: a global
    action carrying a node, an ENGAGE carrying none, a non-integer or boolean node, an
    ENGAGE node out of ``[0, k)``, and an unknown meta-action. Legality against a mask is
    a separate question (:func:`evaluate_action`).
    """
    if isinstance(meta_action, bool) or not isinstance(meta_action, (int, np.integer)):
        raise ValueError("meta_action %r is not a MetaAction integer" % (meta_action,))
    meta = int(meta_action)
    if meta not in (_PLAN, _ENGAGE, _ABORT):
        raise ValueError("meta_action %r is out of bounds for %d meta-actions"
                         % (meta_action, NUM_META_ACTIONS))
    if meta in GLOBAL_META_ACTIONS:
        if node_v is not None:
            raise ValueError(
                "malformed semantic identity: global %s carries node_v=%r; its semantic "
                "identity has NO node (node_v must be None)"
                % (MetaAction(meta).name, node_v))
        return SEMANTIC_PLAN_LEAF if meta == _PLAN else SEMANTIC_ABORT_LEAF
    if node_v is None:
        raise ValueError("malformed semantic identity: OPPORTUNISTIC_ENGAGEMENT carries "
                         "no node (node_v must be a task index)")
    if isinstance(node_v, bool) or not isinstance(node_v, (int, np.integer)):
        raise ValueError("malformed semantic identity: node_v %r is not an integer "
                         "task index" % (node_v,))
    node = int(node_v)
    if not (0 <= node < int(k)):
        raise ValueError("OPPORTUNISTIC_ENGAGEMENT node_v=%d is out of bounds for k=%d "
                         "task node(s)" % (node, int(k)))
    return SEMANTIC_ENGAGE_LEAF_OFFSET + node


# =============================================================================
# THE shared semantic distribution + sampling / re-scoring
# =============================================================================

def _semantic_dist(
    logits: torch.Tensor,
    mask_np: np.ndarray,
) -> Tuple[torch.Tensor, Categorical, torch.Tensor]:
    """Build THE masked semantic distribution over the ``k + 2`` leaves.

    THE single construction site. :func:`sample_action` (rollout, no-grad),
    :func:`evaluate_action` (PPO update, with grad) and :func:`summarize_decision`
    (reporting, detached) all route through here, so the distribution they act on is
    identical BY CONSTRUCTION rather than by several code paths agreeing. Any drift
    between the rollout distribution and the update distribution would corrupt the PPO
    ratio ``pi_new / pi_old`` SILENTLY. Do not reimplement any part of this in a caller.

    Encapsulates exactly: legality from the ``[k, 3]`` mask, the count-normalized
    ``logmeanexp`` collapse of the PLAN column (over all ``k`` nodes) and of the ABORT
    column (over the abort-LEGAL nodes only), the node-local ENGAGE leaves under their
    additive cell mask, the fixed leaf order, the Categorical, and the clamped-logits
    entropy (see :func:`sample_action` for the entropy rationale). Dtype and device
    follow ``logits``; gradients flow to every source score that enters a legal leaf.

    Args:
        logits: ``[k, 3]`` raw scores from :class:`ActionHead`. Grad-attached or not —
            this helper never detaches and never touches grad mode.
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`.

    Returns:
        ``(semantic_logits, dist, entropy)`` — the masked semantic logits ``[k + 2]``
        (``-inf`` on illegal leaves), the Categorical over them, and the masked-safe
        scalar entropy of the SEMANTIC leaf distribution.

    Raises:
        ValueError: if ``k == 0`` (no task node, hence no semantic action space — the
            acting path fails loud there rather than inventing a pooled score); if the
            shapes disagree; or if any PLAN cell is masked (the Plan-Compliance
            invariant that keeps the semantic softmax from being all ``-inf``).
    """
    mask_np = np.asarray(mask_np)
    k = int(mask_np.shape[0]) if mask_np.ndim == 2 else -1
    if k <= 0 or mask_np.shape[1] != NUM_META_ACTIONS:
        raise ValueError(
            "build_action_mask produced no valid action (mask shape %r): a semantic "
            "action space needs k >= 1 task node" % (tuple(mask_np.shape),))
    if tuple(logits.shape) != (k, NUM_META_ACTIONS):
        raise ValueError("score shape %r does not match the [%d, %d] mask"
                         % (tuple(logits.shape), k, NUM_META_ACTIONS))
    if not np.isfinite(mask_np[:, _PLAN]).all():
        raise ValueError("a PLAN_COMPLIANCE cell is masked; the Plan-Compliance "
                         "invariant keeps the semantic PLAN leaf always legal")

    abort_legal = np.isfinite(mask_np[:, _ABORT])
    n_abort = int(abort_legal.sum())

    # PLAN: exact logmeanexp over ALL k nodes -- equal duplicated evidence gets no bonus.
    plan = torch.logsumexp(logits[:, _PLAN], dim=0) - math.log(k)
    if n_abort:
        # ABORT: exact logmeanexp over the abort-LEGAL nodes only.
        idx = torch.as_tensor(np.flatnonzero(abort_legal), dtype=torch.long,
                              device=logits.device)
        abort = (torch.logsumexp(logits[:, _ABORT].index_select(0, idx), dim=0)
                 - math.log(n_abort))
    else:
        # The ONE semantic ABORT leaf is masked when no abort cell is legal.
        abort = torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device)
    engage_mask = torch.as_tensor(mask_np[:, _ENGAGE], dtype=logits.dtype,
                                  device=logits.device)
    engage = logits[:, _ENGAGE] + engage_mask      # node-local score, -inf if illegal

    semantic = torch.cat([plan.reshape(1), abort.reshape(1), engage])

    # Exact distribution: -inf -> zero mass on illegal leaves.
    dist = Categorical(logits=semantic)

    # Entropy: version-independent masked-safe form (clamp -inf -> finfo.min so an
    # illegal leaf contributes a finite ~0 term instead of 0 * -inf = NaN).
    safe = torch.clamp(semantic, min=torch.finfo(semantic.dtype).min)
    entropy = Categorical(logits=safe).entropy()

    return semantic, dist, entropy


def sample_action(
    logits: torch.Tensor,
    mask_np: np.ndarray,
    deterministic: bool = False,
) -> Tuple[int, Optional[int], torch.Tensor, torch.Tensor]:
    """Sample a SEMANTIC ``(meta_action, node_v)`` decision under the legality mask.

    The distribution is the ``k + 2``-leaf semantic Categorical built by
    :func:`_semantic_dist` — the SHARED construction site this function and
    :func:`evaluate_action` both call, so the PPO update re-scores an action under
    exactly the distribution it was sampled from.

    Deterministic selection is ``torch.argmax`` over the SEMANTIC leaves — never a
    source ``k x 3`` cell and never an aggregate reconstructed afterwards. On an exact
    tie ``torch.argmax`` returns the FIRST maximal leaf, so ties resolve in the fixed
    leaf order PLAN, ABORT, ENGAGE(0), ENGAGE(1), ...

    Numerical safety: the additive ``{0, -inf}`` masking is exact for ``sample()`` and
    ``log_prob()``. ``entropy()``, however, sums ``p * log p`` over masked leaves where
    ``0 * (-inf) = NaN`` on older torch versions, so the entropy is computed from a copy
    of the semantic logits clamped to ``torch.finfo(dtype).min``. A NaN entropy would
    silently poison the PPO entropy bonus.

    Args:
        logits: ``[k, 3]`` raw scores from :class:`ActionHead`.
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`.
        deterministic: if True, take the argmax semantic leaf instead of sampling.

    Returns:
        ``(meta_action, node_v, log_prob, entropy)``: ``meta_action`` a python int,
        ``node_v`` ``None`` for PLAN / ABORT and the task index for ENGAGE, and
        ``log_prob`` / ``entropy`` scalar tensors of the SEMANTIC distribution.

    Raises:
        ValueError: as :func:`_semantic_dist`.
    """
    semantic, dist, entropy = _semantic_dist(logits, mask_np)
    if deterministic:
        leaf = torch.argmax(semantic)
    else:
        leaf = dist.sample()
    log_prob = dist.log_prob(leaf)
    meta_action, node_v = semantic_leaf_identity(int(leaf.item()), int(logits.shape[0]))
    return meta_action, node_v, log_prob, entropy


def summarize_decision(
    logits: torch.Tensor,
    mask_np: np.ndarray,
    meta_action: int,
    node_v: Optional[int],
) -> Dict[str, Any]:
    """REPORTING-ONLY summary of ONE semantic decision, from the SAME scores and mask.

    MEASUREMENT, NOT AN ALGORITHM CHANGE. A pure function of the arguments
    :func:`sample_action` was already called with, so it needs no second encoder/head
    forward pass and cannot describe a distribution the actor did not act on. Its output
    feeds artifacts and plots ONLY.

    FOUR PROPERTIES ARE LOAD-BEARING, and all four are structural:

    * **THE ACTOR'S OWN DISTRIBUTION.** Every probability, entropy and argmax comes from
      :func:`_semantic_dist` — the SAME construction site the actor samples and PPO
      re-scores through — on a DETACHED copy of the same scores, in their ORIGINAL dtype.
    * **NO RANDOMNESS.** Nothing here samples, so the torch RNG state cannot move.
    * **NO GRADIENT.** ``logits.detach()`` is taken BEFORE the distribution is built.
    * **THE SAME ARGMAX AS THE ACTOR.** The deterministic leaf is literally
      ``torch.argmax(semantic_logits)``, the expression the deterministic branch of
      :func:`sample_action` evaluates, so an exact tie resolves identically.

    SCHEMA. The record names its representation (``action_representation_id``). The
    per-meta-action probability is DIRECTLY a policy action probability for PLAN and
    ABORT (one leaf each) and the sum over legal ENGAGE leaves for ENGAGE. There is no
    joint-cell-versus-aggregate field: that quantity described the retired node-indexed
    alias geometry and is not a measurement of this representation.

    Args:
        logits: ``[k, 3]`` raw scores from :class:`ActionHead` — the SAME tensor passed
            to :func:`sample_action`.
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`.
        meta_action: the meta-action the actor actually selected.
        node_v: the node the actor actually selected (``None`` for PLAN / ABORT).

    Returns:
        A JSON-ready dict of plain builtins (no tensor, no numpy scalar, no NaN).

    Raises:
        ValueError: as :func:`_semantic_dist`, or on a malformed selected identity.
    """
    semantic, dist, entropy_t = _semantic_dist(logits.detach(), mask_np)
    probs_t = dist.probs.reshape(-1)
    k = int(logits.shape[0])
    legal_t = torch.isfinite(semantic)
    n_valid = int(legal_t.sum().item())
    if n_valid == 0:                                                # pragma: no cover
        raise ValueError("summarize_decision received no legal semantic leaf; the "
                         "Plan-Compliance invariant should make this impossible")
    selected_leaf = semantic_leaf_index(meta_action, node_v, k)

    i1 = int(torch.argmax(semantic).item())
    i2: Optional[int] = None
    if n_valid > 1:
        rest = semantic.clone()
        rest[i1] = float("-inf")
        i2 = int(torch.argmax(rest).item())
    margin_t = None if i2 is None else (probs_t[i1] - probs_t[i2])

    engage_legal_t = legal_t[SEMANTIC_ENGAGE_LEAF_OFFSET:]
    engage_mass_t = probs_t[SEMANTIC_ENGAGE_LEAF_OFFSET:][engage_legal_t].sum()
    raw_entropy = float(entropy_t.item())
    norm_entropy: Optional[float] = (
        float(raw_entropy / math.log(n_valid)) if n_valid > 1 else None)

    # ---- JSON conversion ONLY from here down: every value above is already final ----
    probs = [float(v) for v in probs_t.cpu().tolist()]
    legal = [bool(v) for v in legal_t.cpu().tolist()]
    scores = [float(v) if ok else None
              for v, ok in zip(semantic.cpu().tolist(), legal)]
    mask_arr = np.asarray(mask_np)
    n_abort_legal = int(np.isfinite(mask_arr[:, _ABORT]).sum())

    def _leaf(idx: int) -> Dict[str, Any]:
        meta, node = semantic_leaf_identity(idx, k)
        return {"leaf": int(idx), "meta_action": int(meta),
                "meta_action_name": MetaAction(meta).name, "node": node,
                "probability": probs[idx]}

    leaves = []
    for idx in range(k + SEMANTIC_ENGAGE_LEAF_OFFSET):
        entry = _leaf(idx)
        entry.update({"legal": legal[idx], "score": scores[idx]})
        leaves.append(entry)
    sel_meta, sel_node = semantic_leaf_identity(selected_leaf, k)
    argmax_meta = semantic_leaf_identity(i1, k)[0]
    return {
        "action_representation_id": ACTION_REPRESENTATION_ID,
        "n_task_nodes": k,
        "n_meta_actions": int(NUM_META_ACTIONS),
        "n_semantic_leaves": k + SEMANTIC_ENGAGE_LEAF_OFFSET,
        "n_valid_semantic_leaves": n_valid,
        "n_abort_legal_nodes": n_abort_legal,
        "n_engage_legal_leaves": int(sum(legal[SEMANTIC_ENGAGE_LEAF_OFFSET:])),
        "source_scores": [[float(v) for v in row]
                          for row in logits.detach().cpu().tolist()],
        "source_cell_legal": [[int(v) for v in row]
                              for row in np.isfinite(mask_arr).astype(np.int64).tolist()],
        "semantic_leaves": leaves,
        "semantic_probabilities": probs,
        "selected_leaf": int(selected_leaf),
        "selected_meta_action": int(sel_meta),
        "selected_meta_action_name": MetaAction(sel_meta).name,
        "selected_node": sel_node,
        "selected_action_probability": probs[selected_leaf],
        "semantic_probability_per_meta_action": {
            MetaAction.PLAN_COMPLIANCE.name: probs[SEMANTIC_PLAN_LEAF],
            MetaAction.OPPORTUNISTIC_ENGAGEMENT.name: float(engage_mass_t.item()),
            MetaAction.SELF_PRESERVATION_ABORT.name: probs[SEMANTIC_ABORT_LEAF],
        },
        "deterministic_argmax_leaf": _leaf(i1),
        "deterministic_argmax_meta_action": int(argmax_meta),
        "deterministic_argmax_meta_action_name": MetaAction(argmax_meta).name,
        "top_two_semantic_leaves": [_leaf(i1)] + ([_leaf(i2)] if i2 is not None else []),
        "top_two_probability_margin": (
            None if margin_t is None else float(margin_t.item())),
        "semantic_entropy_raw": raw_entropy,
        "semantic_entropy_normalized": norm_entropy,
    }


def evaluate_action(
    logits: torch.Tensor,
    mask_np: np.ndarray,
    meta_action: int,
    node_v: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-score an ALREADY-CHOSEN semantic ``(meta_action, node_v)`` — the PPO-ratio half.

    Purpose (PPO). The rollout is inference-only: ``graph_tick_loop._wake_decision``
    stores each wake as a ``Transition`` holding the ``GraphObservation``, the semantic
    ``(meta_action, node_v)`` identity and DETACHED ``log_prob`` / ``entropy`` floats.
    The PPO update re-encodes the stored ``gobs``, re-runs the head, and calls this with
    the stored identity to recompute the log-prob WITH grad.

    Identity BY CONSTRUCTION. The distribution is built by the SAME
    :func:`_semantic_dist` helper :func:`sample_action` used at rollout time, and the
    leaf is gathered with the same 0-dim long tensor ``dist.sample()`` returns. So on
    the first PPO epoch (unchanged weights) the returned ``log_prob`` is BITWISE equal to
    the stored one and the ratio is exactly ``1.0``. Never reimplement the construction
    here.

    Grad contract: NO ``torch.no_grad`` anywhere inside. The CALLER controls grad mode.

    Args:
        logits: ``[k, 3]`` raw scores from :class:`ActionHead`, normally grad-attached.
        mask_np: ``[k, 3]`` additive mask from :func:`build_action_mask`, rebuilt from
            the STORED observation so it reproduces the rollout-time mask.
        meta_action: the stored :class:`MetaAction` value.
        node_v: ``None`` for PLAN / ABORT; the stored task index for ENGAGE.

    Returns:
        ``(log_prob, entropy)`` — scalar tensors: the semantic log-prob of the stored
        leaf, and the SAME masked-safe semantic entropy :func:`sample_action` reports.

    Raises:
        ValueError: on a malformed semantic identity (a global action carrying a node,
            an ENGAGE with no node, a non-integer node, an out-of-bounds node or
            meta-action); if the stored ENGAGE cell is MASKED, or an ABORT is stored while
            no abort cell is legal — either means the mask rebuilt at update time diverged
            from the rollout-time mask, which would otherwise silently feed ``-inf`` into
            the ratio; or as :func:`_semantic_dist`.
    """
    mask_arr = np.asarray(mask_np)
    k = int(mask_arr.shape[0]) if mask_arr.ndim == 2 else 0
    leaf = semantic_leaf_index(meta_action, node_v, k)

    # Guard BEFORE building the distribution: a masked stored action is a mask
    # reconstruction bug, not a legitimate zero-probability action.
    if leaf == SEMANTIC_ABORT_LEAF and not np.isfinite(mask_arr[:, _ABORT]).any():
        raise ValueError(
            "evaluate_action: the stored SELF_PRESERVATION_ABORT is MASKED -- no abort "
            "cell is legal in the supplied mask, so the semantic ABORT leaf is illegal. "
            "The mask rebuilt at update time diverged from the rollout-time mask.")
    if leaf >= SEMANTIC_ENGAGE_LEAF_OFFSET and not np.isfinite(
            mask_arr[leaf - SEMANTIC_ENGAGE_LEAF_OFFSET, _ENGAGE]):
        raise ValueError(
            "evaluate_action: the stored OPPORTUNISTIC_ENGAGEMENT (node_v=%d) is MASKED "
            "(-inf) in the supplied mask. The rollout could not have sampled it, so the "
            "mask rebuilt at update time diverged from the rollout-time mask; check that "
            "the stored GraphObservation is the one the action was sampled on."
            % (leaf - SEMANTIC_ENGAGE_LEAF_OFFSET))

    semantic, dist, entropy = _semantic_dist(logits, mask_arr)
    # 0-dim long tensor: exactly the shape/dtype dist.sample() returns, so log_prob
    # takes the same gather path and the result is bitwise identical.
    leaf_t = torch.as_tensor(leaf, dtype=torch.long, device=semantic.device)
    log_prob = dist.log_prob(leaf_t)
    return log_prob, entropy


# =============================================================================
# Self-test
# =============================================================================

def _selftest() -> None:
    """Hand-crafted graph (no solver/bonmin) with a KNOWN topology.

    Run under nlp_env from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.action.graph_action
    """
    #   k = 4 task nodes, a = 3 agents (ego_index = 4, peer1 = 5, peer2 = 6)
    #   task 0: assigned to ego, sensed, capable, reachable       -> PLAN + ABORT cells
    #   task 1: assigned to peer1, sensed                         -> PLAN cell
    #   task 2: unassigned, sensed, capable, reachable (pop-up)   -> PLAN + ENGAGE cells
    #   task 3: assigned to peer2, NOT sensed                     -> PLAN cell
    task_features = np.array(
        [
            # [utility, dist_to_ego, capable, reachable, probability, sensed]
            [0.80, 0.20, 1.0, 1.0, 1.0, 1.0],
            [0.60, 0.40, 1.0, 1.0, 1.0, 1.0],
            [0.50, 0.30, 1.0, 1.0, 1.0, 1.0],
            [0.70, 0.50, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # ego row [fuel_norm, mission_fuel_slack_norm]; peers featureless
    agent_features = np.array([[0.90, 0.10], [0.00, 0.00], [0.00, 0.00]], dtype=np.float32)
    obs = GraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=4,
        edge_index=np.array([[4, 5, 6], [0, 1, 3]], dtype=np.int64),
        edge_type=np.array([int(EdgeType.ASSIGNMENT)] * 3, dtype=np.int64),
        task_target_ids=["t0", "t1", "t2", "t3"],
        agent_ids=["ego", "peer1", "peer2"],
        agent_id="ego",
        current_time=0,
        time_norm=0.0,
    )

    mask = build_action_mask(obs)
    NINF = float("-inf")
    expected = np.array(
        [[0.0, NINF, 0.0], [0.0, NINF, NINF], [0.0, 0.0, NINF], [0.0, NINF, NINF]],
        dtype=np.float32,
    )
    assert np.array_equal(np.isneginf(mask), np.isneginf(expected)), mask
    print("=" * 72)
    print("graph_action self-test (%s)" % ACTION_REPRESENTATION_ID)
    print("=" * 72)
    print("[M] k x 3 source legality matches the expected topology   OK")

    torch.manual_seed(0)
    head = ActionHead(embed_dim=16)
    embeddings = torch.randn(4, 16)
    logits = head(embeddings)
    semantic, dist, _ent = _semantic_dist(logits, mask)
    assert semantic.shape == (6,)
    assert torch.isfinite(semantic).tolist() == [True, True, False, False, True, False]
    exp_plan = torch.logsumexp(logits[:, _PLAN], 0) - math.log(4)
    assert torch.equal(semantic[SEMANTIC_PLAN_LEAF], exp_plan)
    assert torch.equal(semantic[SEMANTIC_ABORT_LEAF], logits[0, _ABORT])  # one legal node
    assert abs(float(dist.probs.sum()) - 1.0) < 1e-6
    print("[S] k + 2 semantic leaves, logmeanexp PLAN, single-node ABORT   OK")

    for deterministic in (False, True):
        meta_s, node_s, lp_s, ent_s = sample_action(logits, mask, deterministic)
        lp_e, ent_e = evaluate_action(logits, mask, meta_s, node_s)
        assert torch.equal(lp_s, lp_e) and torch.equal(ent_s, ent_e)
        assert (node_s is None) == (meta_s in GLOBAL_META_ACTIONS)
        print("[E1] %s: meta=%s node=%r  BITWISE == sample_action   OK"
              % ("deterministic" if deterministic else "stochastic   ",
                 MetaAction(meta_s).name, node_s))

    head.zero_grad(set_to_none=True)
    lp_g, _ = evaluate_action(head(embeddings), mask, _ABORT, None)
    lp_g.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters())
    print("[E3] backward through evaluate_action: finite head grads   OK")

    for meta, node in ((_PLAN, 0), (_ABORT, 0), (_ENGAGE, None), (_ENGAGE, 1), (_ENGAGE, 4)):
        try:
            evaluate_action(logits, mask, meta, node)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted malformed/masked identity %r" % ((meta, node),))
    print("[E4] malformed / masked semantic identities -> ValueError   OK")
    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
