"""
Graph Encoder (Phase-2 RL layer)
================================

The encoder of the Phase-2 graph + Transformer RL layer: an edge-aware,
permutation-invariant, size-agnostic Transformer that turns a
:class:`GraphObservation` into **per-task-node** embeddings the actor's
``ActionHead`` consumes directly. This module is **side-by-side** with the flat
``agent/network.py`` (the frozen baseline actor-critic); nothing here is wired
into ``train_full.py`` yet.

Mandatory properties (the design intent — these are not optional)
-----------------------------------------------------------------
(a) **Permutation-invariance** over nodes: relabeling task nodes (and the edges /
    feature rows accordingly) permutes the output identically.
(b) **Size-agnosticism**: native to any ``(k, a, E)`` with no retraining and no
    padding to fixed sizes. We never inherit the flat path's ``MAX_AGENTS``
    zero-padding — size-agnosticism is the whole point.
(c) **Per-task-node output**, NOT a single pooled graph vector: this encoder emits
    one ``[embed_dim]`` embedding per task node (output ``[k, embed_dim]``) and knows
    NOTHING about meta-actions. The DOWNSTREAM ``ActionHead`` turns each per-task
    embedding into the ``k x 3`` meta-action logits (3 after Cooperative Recovery was
    removed) independently per task node — which is why the encoder must produce one
    embedding per task node rather than a pooled summary. A pooled summary is wanted
    only later, for the future centralized critic — hence the
    :meth:`GraphEncoder.pool` HOOK with NO value head built now.

What is prescribed vs. our engineering choice
---------------------------------------------
The three properties above are mandated by the source design. The attention
INTERNALS below (masked-over-real-edges rather than all-pairs, directed edges
SYMMETRIZED, explicit self-loops, a per-relation additive bias) are OUR locked
engineering choices; nothing in the source design forbids them.

Architecture (locked defaults: model_dim=64, embed_dim=64, num_heads=4, num_layers=2)
-------------------------------------------------------------------------------------
1. Type-specific input projections: ``Linear(task_feat_dim -> model_dim)`` over
   ``task_features`` and ``Linear(agent_feat_dim -> model_dim)`` over
   ``agent_features``. Feature columns are treated as OPAQUE (we never special-case
   ``capable`` / ``reachable`` — masking is the action layer's job). The
   ``*_feat_dim`` are constructor params so a future builder column drops in
   without reopening the encoder.
2. Node-typing / ``is_ego`` is handled HERE (the builder defers it on purpose): a
   learned role embedding over {TASK, EGO, PEER} (MISSION reserved) added per node.
   Roles are derived deterministically from indices: ``idx < k`` -> TASK;
   ``idx == ego_index`` -> EGO; else -> PEER.
3. The global ``time_norm`` scalar is projected ``Linear(1 -> model_dim)`` and
   broadcast-added to ALL node features once, before layer 1.
4. ``num_layers`` hand-rolled multi-head attention layers, RESTRICTED to the actual
   graph edges (NOT all-pairs). Per forward pass we build an AUGMENTED edge set:
   the forward edges as-is, their reverses (SYMMETRIZATION, so 2 layers do real
   2-hop flow and agent/ego nodes actually update — safe under no-comms because
   peers are featureless), and one self-loop per node (guarantees >= 1 incoming
   edge per node, so the softmax is never empty, and lets a node retain itself).
   Per head, per augmented edge ``j -> i`` (aggregated at the DESTINATION ``i``):
       ``score = (Q_i . K_j)/sqrt(d_head) + type_bias[relation, head] + edge_attr_term``
   ``type_bias`` is a learnable ``[num_relations=7, num_heads]`` table (3 edge types
   x {fwd, rev} + self-loop). The bias + connectivity are materialized as a DENSE
   ``[N, N, num_heads]`` matrix (N = k + a is small — no padding), with non-edges
   filled with a LARGE FINITE NEGATIVE (not literal ``-inf``, to avoid ``0 * -inf``
   NaN) so a standard softmax masks them. Each layer is pre-LN: an attention
   sublayer (residual + LayerNorm) then a position-wise FFN sublayer (residual +
   LayerNorm).
5. EDGE-FEATURE-READY. ``edge_attr`` is an OPTIONAL forward argument, ``None`` today
   (the builder does not emit it yet — there is NO ``obs.edge_attr`` field). When
   present it is ``[E, edge_attr_dim]`` aligned with the original ``edge_index``,
   projected ``Linear(edge_attr_dim -> num_heads)`` and added per-head to that
   edge's score on BOTH its forward and reverse augmented copies (self-loops carry
   none). The encoder is agnostic to whether the future value is a static
   expected-exec-time or a dynamic relative-slack — that is a BUILDER decision.
   Reserved use: a normalized expected-execution-time on ASSIGNMENT edges driving
   Cooperative-Recovery timing.
6. OUTPUT = per-task-node embeddings ``[k, embed_dim]`` (NOT pooled). After the
   final layer we slice the task rows ``[0:k]`` and apply a final
   ``Linear(model_dim -> embed_dim)`` (so ``embed_dim`` may differ from
   ``model_dim``).
7. POOLING HOOK :meth:`GraphEncoder.pool` (mean over ALL node embeddings) for the
   future centralized critic. NO value head is built now.

Dependencies: ``torch`` + ``numpy`` only (no PyG/DGL — only torch/numpy are
installed). The body imports ONLY :class:`GraphObservation` / :class:`EdgeType`
from the observation layer; the action layer is touched only by ``_selftest``.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..observation.graph_builder import GraphObservation, EdgeType, TASK_FEATURE_DIM


# =============================================================================
# Conventions borrowed from the flat path (RE-DEFINED locally on purpose)
# =============================================================================

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init for a linear layer — the standard PPO scheme.

    Re-defined LOCALLY (rather than imported from ``action/graph_action.py``) to
    avoid an ``agent -> action`` dependency; it is byte-identical to the action
    layer's helper and to ``agent/network.py``'s. ``std=sqrt(2)`` for hidden layers
    is the established convention in this repo.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# Large FINITE negative used to mask non-edges before softmax. NOT literal -inf:
# scores at non-edges are ``QK + LARGE_NEG`` and ``0 * -inf = NaN`` is avoided while
# ``exp(LARGE_NEG - row_max)`` still underflows cleanly to 0. Kept well inside the
# float32 range so adding the (small) QK term never overflows to -inf.
_LARGE_NEG = -1.0e9


# =============================================================================
# Node roles (handled HERE; builder defers node-typing on purpose)
# =============================================================================

class NodeRole(IntEnum):
    """Learned-role codes added per node after its input projection.

    Derived deterministically from node indices (``idx < k`` -> TASK;
    ``idx == ego_index`` -> EGO; else PEER). ``MISSION`` is RESERVED for a future
    node type (the 4th role) and is unused today — ``num_roles`` defaults to 4 so
    that slot already exists in the embedding table.
    """

    TASK = 0
    EGO = 1
    PEER = 2
    MISSION = 3  # reserved (future); unused now


# Augmented-edge relation codes for the per-relation ``type_bias`` table.
# 3 edge types x {forward, reverse} + 1 self-loop = 7 relations.
_NUM_BASE_RELATIONS = 3                  # SPATIAL, ASSIGNMENT, PRECEDENCE (== len(EdgeType))
_SELF_LOOP_RELATION = 2 * _NUM_BASE_RELATIONS  # == 6
NUM_RELATIONS = _SELF_LOOP_RELATION + 1        # == 7


# =============================================================================
# One pre-LN multi-head attention + FFN layer (edge-masked)
# =============================================================================

class _GraphAttentionLayer(nn.Module):
    """A single pre-LN edge-masked multi-head attention + position-wise FFN layer.

    The per-relation bias and edge-feature projection live on the encoder and are
    materialized ONCE per forward pass into the dense ``adj_bias`` matrix this layer
    consumes, so the layer itself only owns Q/K/V/out projections, two LayerNorms,
    and the FFN. Attention is restricted to the augmented graph edges via
    ``adj_bias`` (real edges carry their bias; non-edges carry ``_LARGE_NEG``).
    """

    def __init__(self, model_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        assert model_dim % num_heads == 0, (
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = _layer_init(nn.Linear(model_dim, model_dim))
        self.k_proj = _layer_init(nn.Linear(model_dim, model_dim))
        self.v_proj = _layer_init(nn.Linear(model_dim, model_dim))
        self.out_proj = _layer_init(nn.Linear(model_dim, model_dim))

        self.ln_attn = nn.LayerNorm(model_dim)
        self.ln_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            _layer_init(nn.Linear(model_dim, ff_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(ff_dim, model_dim)),
        )

    def forward(self, x: torch.Tensor, adj_bias: torch.Tensor) -> torch.Tensor:
        """Update node features.

        Args:
            x: ``[N, model_dim]`` node features.
            adj_bias: ``[N, N, num_heads]`` additive attention bias indexed
                ``[dst, src, head]``; non-edges are ``_LARGE_NEG``.

        Returns:
            ``[N, model_dim]`` updated node features.
        """
        n = x.shape[0]
        h, d = self.num_heads, self.head_dim

        # --- Attention sublayer (pre-LN) ---
        xn = self.ln_attn(x)
        q = self.q_proj(xn).view(n, h, d)
        k = self.k_proj(xn).view(n, h, d)
        v = self.v_proj(xn).view(n, h, d)

        # score[i, j, head] = (Q_i . K_j) / sqrt(d) + adj_bias[i, j, head]
        qk = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(d)
        scores = qk + adj_bias                       # non-edges ~ _LARGE_NEG
        attn = torch.softmax(scores, dim=1)          # normalize over sources j
        ctx = torch.einsum("ijh,jhd->ihd", attn, v)  # [N, H, d]
        attn_out = self.out_proj(ctx.reshape(n, h * d))
        x = x + attn_out

        # --- Position-wise FFN sublayer (pre-LN) ---
        x = x + self.ffn(self.ln_ffn(x))
        return x


# =============================================================================
# The encoder
# =============================================================================

class GraphEncoder(nn.Module):
    """Edge-aware permutation-invariant Transformer over a :class:`GraphObservation`.

    Stable interface (internals swappable)::

        forward(obs, edge_attr=None) -> Tensor[k, embed_dim]   # per-task-node, NOT pooled
        pool(obs, edge_attr=None)    -> Tensor[embed_dim]      # future-critic hook

    The output ``embed_dim`` MUST equal the ``embed_dim`` the actor's
    :class:`~match_aou.rl.action.graph_action.ActionHead` is constructed with.
    """

    def __init__(
        self,
        model_dim: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        task_feat_dim: int = TASK_FEATURE_DIM,
        agent_feat_dim: int = 1,
        edge_attr_dim: int = 1,
        ff_dim: Optional[int] = None,
        num_roles: int = 4,
    ):
        """Build the encoder.

        Args:
            model_dim: internal node-feature width (must be divisible by num_heads).
            embed_dim: output per-task-node embedding width (may differ from
                model_dim). MUST match the actor ``ActionHead``'s ``embed_dim``.
            num_heads: number of attention heads.
            num_layers: number of attention + FFN layers.
            task_feat_dim: width of ``task_features`` columns (param so a future
                builder column drops in without reopening the encoder).
            agent_feat_dim: width of ``agent_features`` columns (same rationale).
            edge_attr_dim: width of the optional ``edge_attr`` (projected to
                num_heads). Used only when ``edge_attr`` is passed to ``forward``.
            ff_dim: FFN hidden width; defaults to ``4 * model_dim``.
            num_roles: size of the role-embedding table. Defaults to 4 so the
                reserved MISSION slot already exists (only TASK/EGO/PEER used now).
        """
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * model_dim

        self.model_dim = model_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.task_feat_dim = task_feat_dim
        self.agent_feat_dim = agent_feat_dim
        self.edge_attr_dim = edge_attr_dim

        # --- Type-specific input projections (columns treated as OPAQUE) ---
        self.task_proj = _layer_init(nn.Linear(task_feat_dim, model_dim))
        self.agent_proj = _layer_init(nn.Linear(agent_feat_dim, model_dim))

        # --- Node-typing handled here (builder defers it) ---
        self.role_embed = nn.Embedding(num_roles, model_dim)

        # --- Global time scalar injection ---
        self.time_proj = _layer_init(nn.Linear(1, model_dim))

        # --- Per-relation additive bias (shared across layers) ---
        # Zero-init: initial attention is plain content-based with no relational
        # preference; the table learns per-relation, per-head structure.
        self.type_bias = nn.Parameter(torch.zeros(NUM_RELATIONS, num_heads))

        # --- Edge-feature projection (shared; used only when edge_attr is given) ---
        self.edge_attr_proj = _layer_init(nn.Linear(edge_attr_dim, num_heads))

        # --- Stacked attention layers ---
        self.layers = nn.ModuleList(
            _GraphAttentionLayer(model_dim, num_heads, ff_dim) for _ in range(num_layers)
        )

        # --- Final per-node read-out (model_dim -> embed_dim) ---
        self.out_proj = _layer_init(nn.Linear(model_dim, embed_dim))

    # ------------------------------------------------------------------ helpers

    @property
    def _device(self) -> torch.device:
        return self.task_proj.weight.device

    def _node_inputs(self, obs: GraphObservation) -> torch.Tensor:
        """Project task/agent features, add role + time, return ``[N, model_dim]``."""
        device = self._device
        k = int(obs.task_features.shape[0])
        a = int(obs.agent_features.shape[0])
        n = k + a

        task_feats = torch.as_tensor(obs.task_features, dtype=torch.float32, device=device)
        agent_feats = torch.as_tensor(obs.agent_features, dtype=torch.float32, device=device)

        task_h = self.task_proj(task_feats)    # [k, model_dim]
        agent_h = self.agent_proj(agent_feats)  # [a, model_dim]
        x = torch.cat([task_h, agent_h], dim=0)  # [N, model_dim]; tasks first, agents after

        # Roles: tasks -> TASK; the ego agent node -> EGO; the rest -> PEER.
        roles = torch.full((n,), int(NodeRole.PEER), dtype=torch.long, device=device)
        roles[:k] = int(NodeRole.TASK)
        ego_index = int(obs.ego_index)
        if 0 <= ego_index < n:
            roles[ego_index] = int(NodeRole.EGO)
        x = x + self.role_embed(roles)

        # Global time scalar, broadcast-added once to every node.
        time_term = self.time_proj(
            torch.tensor([float(obs.time_norm)], dtype=torch.float32, device=device)
        )
        x = x + time_term.unsqueeze(0)
        return x

    def _build_adj_bias(
        self, obs: GraphObservation, n: int, edge_attr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Materialize the dense ``[N, N, num_heads]`` additive attention bias.

        Builds the AUGMENTED edge set (forward + reverse + self-loops), gathers the
        per-relation bias (plus the edge-feature term on fwd/rev when ``edge_attr``
        is given), scatters it into a dense ``[dst, src, head]`` matrix, and fills
        every non-edge cell with ``_LARGE_NEG``. Self-loops guarantee >= 1 edge per
        destination row, so no softmax row is ever all-masked.
        """
        device = self._device
        ei = torch.as_tensor(obs.edge_index, dtype=torch.long, device=device).reshape(2, -1)
        et = torch.as_tensor(obs.edge_type, dtype=torch.long, device=device).reshape(-1)
        e = ei.shape[1]

        # Forward edges (src->dst), relation == edge_type in {0,1,2}.
        fwd_src, fwd_dst, fwd_rel = ei[0], ei[1], et
        # Reverse edges (dst->src), relation == edge_type + num_base (SYMMETRIZATION).
        rev_src, rev_dst, rev_rel = ei[1], ei[0], et + _NUM_BASE_RELATIONS
        # Self-loops (i->i), relation == SELF_LOOP.
        loop = torch.arange(n, dtype=torch.long, device=device)
        loop_rel = torch.full((n,), _SELF_LOOP_RELATION, dtype=torch.long, device=device)

        aug_src = torch.cat([fwd_src, rev_src, loop])
        aug_dst = torch.cat([fwd_dst, rev_dst, loop])
        aug_rel = torch.cat([fwd_rel, rev_rel, loop_rel])

        per_edge = self.type_bias[aug_rel]  # [E_aug, num_heads]

        # Edge-feature term: applies to BOTH fwd and rev copies of each original
        # edge; self-loops carry none. Aligned with the original edge_index.
        if edge_attr is not None and e > 0:
            ea = torch.as_tensor(edge_attr, dtype=torch.float32, device=device).reshape(e, -1)
            if ea.shape[0] != e:
                raise ValueError(
                    f"edge_attr has {ea.shape[0]} rows but edge_index has {e} edges"
                )
            ea_term = self.edge_attr_proj(ea)  # [E, num_heads]
            add = torch.zeros((aug_src.shape[0], self.num_heads), device=device)
            add[:e] = ea_term        # forward copies
            add[e:2 * e] = ea_term   # reverse copies
            per_edge = per_edge + add

        # Scatter into a dense [dst, src, head] matrix (out-of-place index_add keeps
        # autograd flowing through per_edge -> type_bias / edge_attr_proj).
        flat = aug_dst * n + aug_src  # row-major [dst, src]
        adj = torch.zeros((n * n, self.num_heads), device=device).index_add(0, flat, per_edge)
        present = torch.zeros(n * n, dtype=torch.bool, device=device)
        present[flat] = True

        adj = adj.view(n, n, self.num_heads)
        present = present.view(n, n)
        adj = adj.masked_fill(~present.unsqueeze(-1), _LARGE_NEG)
        return adj

    def _encode(self, obs: GraphObservation, edge_attr: Optional[torch.Tensor]) -> torch.Tensor:
        """Run the full stack, returning ALL node embeddings ``[N, model_dim]``."""
        x = self._node_inputs(obs)
        n = x.shape[0]
        adj_bias = self._build_adj_bias(obs, n, edge_attr)
        for layer in self.layers:
            x = layer(x, adj_bias)
        return x

    # ------------------------------------------------------------------- public

    def forward(
        self, obs: GraphObservation, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode ``obs`` into per-task-node embeddings.

        Args:
            obs: the :class:`GraphObservation` to encode (single graph, no batch dim).
            edge_attr: optional ``[E, edge_attr_dim]`` features aligned with
                ``obs.edge_index``. ``None`` today (the builder emits none).

        Returns:
            ``[k, embed_dim]`` per-task-node embeddings (NOT pooled).
        """
        node_emb = self._encode(obs, edge_attr)
        k = int(obs.task_features.shape[0])
        task_emb = node_emb[:k]                 # task rows are the first k nodes
        return self.out_proj(task_emb)

    def pool(
        self, obs: GraphObservation, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pooled graph summary ``[embed_dim]`` (mean over ALL node embeddings).

        HOOK for the future centralized critic; unused in the actor path. NO value
        head is built here.
        """
        node_emb = self._encode(obs, edge_attr)
        return self.out_proj(node_emb).mean(dim=0)


# =============================================================================
# Self-test (disposable, one-shot; NO solver/bonmin)
# =============================================================================

def _make_obs(
    task_features: np.ndarray,
    agent_features: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    ego_index: int,
) -> GraphObservation:
    """Construct a synthetic GraphObservation directly (no build_graph_observation)."""
    k = task_features.shape[0]
    a = agent_features.shape[0]
    return GraphObservation(
        task_features=task_features.astype(np.float32),
        agent_features=agent_features.astype(np.float32),
        ego_index=int(ego_index),
        edge_index=edge_index.astype(np.int64).reshape(2, -1),
        edge_type=edge_type.astype(np.int64).reshape(-1),
        task_target_ids=[f"t{j}" for j in range(k)],
        agent_ids=[f"ag{i}" for i in range(a)],
        agent_id="ag0",
        current_time=123,
        time_norm=0.25,
    )


def _selftest() -> None:
    """Hand-built synthetic graphs; assert the three properties + grad finiteness.

    Run under nlp_env from the repo root, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.agent.graph_encoder
    """
    from ..action.graph_action import ActionHead, build_action_mask, sample_action, NUM_META_ACTIONS

    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 72)
    print("graph_encoder self-test")
    print("=" * 72)

    EMBED_DIM = 32  # deliberately != model_dim to exercise the out_proj read-out
    # NOTE: task_feat_dim is deliberately NOT overridden here — the default value is
    # what we regression-guard, so the encoder must consume the builder's real column
    # width (TASK_FEATURE_DIM == 6) with no help from the test.
    encoder = GraphEncoder(model_dim=64, embed_dim=EMBED_DIM, num_heads=4, num_layers=2)
    encoder.eval()  # deterministic forward (no dropout, but explicit for clarity)

    # Contract pin: the default MUST track the builder's single source of truth. A
    # future revert of the default back to a magic number breaks this immediately.
    assert encoder.task_feat_dim == TASK_FEATURE_DIM, (
        f"default task_feat_dim {encoder.task_feat_dim} != builder "
        f"TASK_FEATURE_DIM {TASK_FEATURE_DIM}"
    )
    print(f"[contract] default task_feat_dim == TASK_FEATURE_DIM == {TASK_FEATURE_DIM}")

    # -------------------------------------------------------------------------
    # Topology A: k=4 tasks, a=2 agents (ego index 4 + one peer index 5).
    #   Builder-faithful: ASSIGNMENT is the only constructed relation, and the ego's
    #   sensing lives in the `sensed` COLUMN (task_features[:, 5]), NOT in edges.
    #   ASSIGNMENT: ego(4)->0, peer(5)->1 ; PRECEDENCE: none. Tasks 2 and 3 have NO
    #   ASSIGNMENT in-edge -> the pop-up-like attention-stress / regression nodes:
    #   self-loops alone must keep them finite. task_features columns are 6-wide now
    #   ([5] = sensed in {0.0, 1.0}); the encoder's default task_feat_dim must consume
    #   them with no override.
    # -------------------------------------------------------------------------
    task_feats_A = np.array(
        #    util  dist  cap  reach prob  sensed
        [
            [0.80, 0.20, 1.0, 1.0, 1.0, 1.0],
            [0.60, 0.40, 1.0, 1.0, 1.0, 0.0],
            [0.50, 0.90, 1.0, 0.0, 1.0, 1.0],   # sensed pop-up (no ASSIGNMENT in-edge)
            [0.70, 0.55, 1.0, 1.0, 1.0, 0.0],   # isolated: no ASSIGNMENT in-edge, unsensed
        ],
        dtype=np.float32,
    )
    agent_feats_A = np.array([[0.90], [0.0]], dtype=np.float32)  # ego real fuel, peer featureless
    edge_index_A = np.array([[4, 5],
                             [0, 1]], dtype=np.int64)
    edge_type_A = np.array(
        [int(EdgeType.ASSIGNMENT), int(EdgeType.ASSIGNMENT)],
        dtype=np.int64,
    )
    obs_A = _make_obs(task_feats_A, agent_feats_A, edge_index_A, edge_type_A, ego_index=4)
    k_A = task_feats_A.shape[0]

    # --- end-to-end: encoder -> ActionHead -> mask -> sample ---
    emb_A = encoder(obs_A)
    assert emb_A.shape == (k_A, EMBED_DIM), emb_A.shape
    assert torch.isfinite(emb_A).all(), "encoder output has non-finite values (topology A)"
    print(f"[A] a=2  encoder(obs) -> {tuple(emb_A.shape)}  all-finite OK "
          f"(tasks 2,3 have no ASSIGNMENT in-edge; self-loops kept them finite)")

    head = ActionHead(embed_dim=encoder.embed_dim)
    logits_A = head(emb_A)
    assert logits_A.shape == (k_A, NUM_META_ACTIONS), logits_A.shape
    mask_A = build_action_mask(obs_A)
    meta, node, log_prob, entropy = sample_action(logits_A, mask_A, deterministic=False)
    assert isinstance(meta, int) and isinstance(node, int)
    assert 0 <= node < k_A and 0 <= meta < NUM_META_ACTIONS
    assert torch.isfinite(log_prob).all() and torch.isfinite(entropy).all()
    print(f"[A] ActionHead+mask+sample -> node={node} meta={meta} "
          f"log_prob={log_prob.item():.4f} entropy={entropy.item():.4f}")

    # --- pool() hook returns a single [embed_dim] vector ---
    pooled = encoder.pool(obs_A)
    assert pooled.shape == (EMBED_DIM,), pooled.shape
    assert torch.isfinite(pooled).all()
    print(f"[A] pool() -> {tuple(pooled.shape)}  all-finite OK")

    # -------------------------------------------------------------------------
    # Topology B: k=2 tasks, a=1 agent (EGO ONLY, index 2). No peers, no PRECEDENCE.
    #   Builder-faithful: ASSIGNMENT ego(2)->0 only (sensing is the `sensed` column,
    #   not an edge). Task 1 has NO ASSIGNMENT in-edge -> a=1 + a no-in-edge task,
    #   again proving self-loops prevent NaN. Columns are 6-wide ([5] = sensed).
    # -------------------------------------------------------------------------
    task_feats_B = np.array(
        [[0.65, 0.30, 1.0, 1.0, 1.0, 1.0],
         [0.55, 0.70, 1.0, 1.0, 1.0, 0.0]],   # no ASSIGNMENT in-edge, unsensed
        dtype=np.float32,
    )
    agent_feats_B = np.array([[0.80]], dtype=np.float32)
    edge_index_B = np.array([[2],
                             [0]], dtype=np.int64)
    edge_type_B = np.array([int(EdgeType.ASSIGNMENT)], dtype=np.int64)
    obs_B = _make_obs(task_feats_B, agent_feats_B, edge_index_B, edge_type_B, ego_index=2)
    emb_B = encoder(obs_B)
    assert emb_B.shape == (2, EMBED_DIM), emb_B.shape
    assert torch.isfinite(emb_B).all(), "encoder output non-finite (topology B, a=1)"
    print(f"[B] a=1 (ego only)  encoder(obs) -> {tuple(emb_B.shape)}  all-finite OK")

    # -------------------------------------------------------------------------
    # Empty-edge stress: NO original edges at all (only self-loops survive).
    # -------------------------------------------------------------------------
    obs_empty = _make_obs(
        task_feats_B, agent_feats_B,
        np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64),
        ego_index=2,
    )
    emb_empty = encoder(obs_empty)
    assert torch.isfinite(emb_empty).all(), "encoder output non-finite with zero edges"
    print(f"[B'] zero original edges -> {tuple(emb_empty.shape)}  all-finite OK (self-loops only)")

    # -------------------------------------------------------------------------
    # Reserved-relation machinery ONLY (NOT builder-emitted). The builder no longer
    # emits SPATIAL edges — sensing moved to the `sensed` COLUMN — but EdgeType.SPATIAL
    # and its fwd/rev rows in the encoder's `type_bias` table are kept reserved. Feed a
    # lone SPATIAL edge to prove the multi-relation attention path stays finite if that
    # relation is ever re-introduced. This topology is DELIBERATELY not builder-faithful.
    # -------------------------------------------------------------------------
    task_feats_R = np.array(
        [[0.60, 0.40, 1.0, 1.0, 1.0, 1.0],
         [0.50, 0.60, 1.0, 1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    agent_feats_R = np.array([[0.75]], dtype=np.float32)
    edge_index_R = np.array([[2, 2],
                             [0, 1]], dtype=np.int64)
    edge_type_R = np.array(
        [int(EdgeType.ASSIGNMENT), int(EdgeType.SPATIAL)], dtype=np.int64
    )
    obs_R = _make_obs(task_feats_R, agent_feats_R, edge_index_R, edge_type_R, ego_index=2)
    emb_R = encoder(obs_R)
    assert torch.isfinite(emb_R).all(), "encoder non-finite on reserved SPATIAL relation"
    print(f"[R] reserved-relation (SPATIAL) machinery -> {tuple(emb_R.shape)}  "
          f"all-finite OK (NOT builder-emitted)")

    # -------------------------------------------------------------------------
    # Gradient: every encoder parameter gets a finite grad. Run WITH edge_attr so
    # edge_attr_proj participates (otherwise it would have a None grad).
    # -------------------------------------------------------------------------
    encoder.zero_grad(set_to_none=True)
    e_A = edge_index_A.shape[1]
    edge_attr_A = torch.randn(e_A, encoder.edge_attr_dim, requires_grad=False)
    emb_grad = encoder(obs_A, edge_attr=edge_attr_A)
    emb_grad.sum().backward()
    n_params = 0
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"parameter {name} has no grad (not exercised)"
        assert torch.isfinite(p.grad).all(), f"parameter {name} has non-finite grad"
        n_params += 1
    print(f"[grad] backward OK: all {n_params} encoder params have finite grads "
          f"(edge_attr path exercised)")

    # -------------------------------------------------------------------------
    # Permutation-invariance over TASK nodes (ego held FIXED). Permuting task rows
    # + remapping edges must permute the output rows identically.
    # -------------------------------------------------------------------------
    encoder.zero_grad(set_to_none=True)
    with torch.no_grad():
        base = encoder(obs_A)  # [k, embed]

        perm = np.array([2, 0, 3, 1], dtype=np.int64)  # new row i <- old task perm[i]
        n_A = k_A + agent_feats_A.shape[0]
        old2new = np.arange(n_A, dtype=np.int64)       # identity for agent nodes
        for new_i, old_t in enumerate(perm):
            old2new[old_t] = new_i                     # task remap only

        perm_task_feats = task_feats_A[perm]
        perm_edge_index = np.array(
            [[old2new[s] for s in edge_index_A[0]],
             [old2new[d] for d in edge_index_A[1]]],
            dtype=np.int64,
        )
        obs_perm = _make_obs(
            perm_task_feats, agent_feats_A, perm_edge_index, edge_type_A, ego_index=4
        )
        permuted = encoder(obs_perm)

        # Output row i (a permuted task) must equal base row perm[i].
        expected = base[torch.as_tensor(perm, dtype=torch.long)]
        max_dev = (permuted - expected).abs().max().item()
        assert max_dev < 1e-5, f"task permutation-invariance violated (max dev {max_dev:.2e})"
    print(f"[perm-task] task-node permutation invariant (max dev {max_dev:.2e}, ego fixed)")

    # -------------------------------------------------------------------------
    # Permutation-invariance over PEER nodes (ego + 2 peers). Swapping the two peer
    # nodes + remapping their ASSIGNMENT edges must leave TASK embeddings unchanged.
    # -------------------------------------------------------------------------
    with torch.no_grad():
        task_feats_C = np.array(
            [[0.7, 0.3, 1.0, 1.0, 1.0, 1.0],
             [0.6, 0.5, 1.0, 1.0, 1.0, 0.0],
             [0.5, 0.7, 1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        agent_feats_C = np.array([[0.9], [0.0], [0.0]], dtype=np.float32)  # ego + 2 featureless peers
        # Builder-faithful: ASSIGNMENT only (sensing is the `sensed` column, not an edge).
        # ego=3, peer1=4, peer2=5 ; ASSIGN ego->0, peer1->1, peer2->2
        edge_index_C = np.array([[3, 4, 5],
                                 [0, 1, 2]], dtype=np.int64)
        edge_type_C = np.array(
            [int(EdgeType.ASSIGNMENT), int(EdgeType.ASSIGNMENT),
             int(EdgeType.ASSIGNMENT)],
            dtype=np.int64,
        )
        obs_C = _make_obs(task_feats_C, agent_feats_C, edge_index_C, edge_type_C, ego_index=3)
        base_C = encoder(obs_C)  # [3, embed]

        # Swap peer1 (4) and peer2 (5): remap their edges, keep ego (3) and tasks fixed.
        swap = {4: 5, 5: 4}
        swapped_src = np.array([swap.get(int(s), int(s)) for s in edge_index_C[0]], dtype=np.int64)
        edge_index_Cs = np.array([swapped_src, edge_index_C[1]], dtype=np.int64)
        obs_Cs = _make_obs(task_feats_C, agent_feats_C, edge_index_Cs, edge_type_C, ego_index=3)
        swapped_C = encoder(obs_Cs)

        max_dev_peer = (swapped_C - base_C).abs().max().item()
        assert max_dev_peer < 1e-5, (
            f"peer permutation-invariance violated (max dev {max_dev_peer:.2e})"
        )
    print(f"[perm-peer] peer-node swap leaves task embeddings invariant "
          f"(max dev {max_dev_peer:.2e})")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
