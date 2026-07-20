"""graph_ppo.py — the PPO core (Phase A: actor-only) for the graph-RL pipeline.

This is the consumer the pipeline has been building toward. Today the chain ends at
``compute_episode_reward``:

    setup_episode(scenario)                  -> EpisodeContext
    run_episode(policy, ctx)                 -> EpisodeResult(.trajectory)
    compute_episode_reward(ctx, result)      -> EpisodeReward   # fills Transition.reward
    [THIS FILE]                              -> records -> advantages -> PPO update

SCOPE (what this file is / is NOT)
----------------------------------
IS: the PURE half of the PPO consumer — episode records + buffer, returns /
advantages, and the clipped-surrogate update step. Everything here is provable on
synthetic data: NO BLADE, NO solver, NO env. The whole module is exercisable from a
hand-built :class:`GraphObservation` plus a random-weight policy.

IS NOT:
  * the outer training loop (generate -> setup -> run -> reward -> update -> log -> repeat).
    That is the NEXT task; this module is what it will call.
  * the critic / value head (Phase B, CTDE). There is NO value loss here — see the
    clearly marked PHASE-B SEAM comments in :func:`compute_returns_and_advantages`
    and :meth:`PPOUpdater.update`.

THE CREDIT STRUCTURE (locked in planning — read before changing anything)
-------------------------------------------------------------------------
1. **Per-ego grouping.** An :class:`EpisodeRecord` stores the episode's transitions
   grouped BY EGO, each ego's own list in temporal order. In Phase A this grouping
   does not change a single number (see gamma below) — it exists as the SEAM
   CONTRACT for the Phase-B critic/GAE, which needs each ego's own temporal chain to
   propagate ``delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`` backwards. Doing the
   grouping NOW means Phase B replaces only the returns component, without reshaping
   the data the buffer already holds. A flat "all transitions of the episode" list
   would have to be re-derived (and the interleaved two-ego ordering un-mixed) at
   that point.

2. **gamma = 1.0, and it is DORMANT.** An ego's chain is 1-4 decisions long and the
   episode carries a SINGLE terminal reward. Discounting would arbitrarily penalize
   the earliest decisions — the ones that matter most — for no modelling reason:
   there is no per-step cost and no infinite horizon to tame. So in Phase A the
   return of EVERY transition of an episode is that episode's scalar ``R``. gamma is
   an explicit config field (not a hard-coded 1) so the generalization is visible and
   testable: with ``gamma < 1`` a chain of length ``n`` discounts the terminal reward
   back as ``R * gamma**(n-1-t)``. gamma becomes REAL when Phase B lands dense /
   per-wake rewards and GAE.

3. **Baseline = mean R across the EPISODES of the batch**, NOT across transitions.
   An episode with 4 wakes must not pull the baseline four times harder than an
   episode with 1 wake — the baseline is a statement about episode quality, and
   wake-count is a property of the scenario (how much was hidden, what got sensed),
   not of the policy's merit. A zero-wake episode is a legitimate outcome and DOES
   count toward the baseline while contributing no transitions.

4. **Advantages are normalized across ALL transitions of the batch** (mean 0, std 1,
   with an epsilon guard) — the standard PPO variance reduction. ``R`` is already
   scenario-normalized by the oracle denominator in ``graph_reward`` (``R`` in
   ~[-1, 0] regardless of scenario size), so mixing episodes from DIFFERENT scenarios
   in one batch is meaningful by construction; the normalization then only removes
   the batch's own scale/offset.

MASKS ARE REBUILT, NEVER STORED
-------------------------------
``build_action_mask`` is a pure function of the stored ``GraphObservation``, so the
update rebuilds the mask from ``tr.gobs`` rather than carrying a second copy that
could drift. ``evaluate_action`` is the ONLY way a stored action is re-scored: it and
``sample_action`` share ONE distribution construction site (``_masked_dist``), which
is what makes the epoch-0 ratio exactly 1.0 BY CONSTRUCTION rather than by two code
paths agreeing (see ``graph_action`` and ``tests/test_graph_action_evaluate.py``). If
a rebuilt mask ever masks a stored action, ``evaluate_action`` raises — that is a mask
reconstruction bug and it fails LOUD instead of feeding ``-inf`` into the ratio.

SINGLE-GRAPH ENCODER
--------------------
``GraphEncoder`` is single-graph (no batch dimension) and episodes have different
``k``, so the update iterates transition-by-transition and accumulates per-transition
losses, taking ONE mean and ONE backward per epoch. Batching heterogeneous graphs is
deliberately not attempted here; it is the buffer's concern if it ever becomes one.

Import contract: torch + numpy + the action layer only. ``Policy`` / ``Transition``
are TYPE_CHECKING-only imports (the ``graph_reward`` idiom), so importing this module
adds NOTHING to ``graph_action``'s own closure beyond this module and its package —
no BLADE engine, no gymnasium env, no episode-setup, no tick-loop. (The pyomo solver
MODULE does ride along, inherited from the locked ``graph_builder ->
scenario_factory`` chain; it is an inert model definition at import time and needs no
bonmin, which is why this module's tests run in the base env.) No torch global state
(seeds, default dtype) is touched at import time.

Framework: PyTorch. Windows-safe: ASCII-only console output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import torch

from ..action.graph_action import build_action_mask, evaluate_action

if TYPE_CHECKING:  # Types only — keeps the runtime closure BLADE-free / solver-free.
    from .graph_tick_loop import Policy, Transition


# =============================================================================
# 1. EpisodeRecord — one finished episode, grouped per ego (torch-free)
# =============================================================================

@dataclass
class EpisodeRecord:
    """One finished episode: per-ego transition chains + the episode's scalar reward.

    Torch-free by construction (it holds ``Transition`` objects, which are plain
    dataclasses of detached floats plus a ``GraphObservation`` of numpy arrays).

    Attributes:
        chains: ``{ego_id: [Transition, ...]}`` — each ego's OWN decisions in temporal
            order. Ego keys are in first-wake order. An episode where nobody woke has
            an EMPTY dict; that is valid and is not an error (roughly a quarter of
            rollout episodes produce no organic wake).
        episode_reward: the episode's scalar terminal reward ``R`` from
            ``graph_reward.compute_episode_reward`` (``EpisodeReward.reward``),
            normalized by the oracle denominator to ~[-1, 0]. Recorded even for a
            zero-wake episode — such an episode still counts toward the batch baseline.
        seed: the episode's seed (``base_seed + i`` in the rollout convention).
        episode_index: the episode's index within its rollout / batch.

    Why per-ego and not one flat list: see the module docstring, "THE CREDIT
    STRUCTURE" (1). It is the Phase-B GAE seam.
    """

    chains: Dict[str, List["Transition"]] = field(default_factory=dict)
    episode_reward: float = 0.0
    seed: int = 0
    episode_index: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_trajectory(
        cls,
        trajectory: Sequence["Transition"],
        episode_reward: float,
        *,
        seed: int = 0,
        episode_index: int = 0,
    ) -> "EpisodeRecord":
        """Build a record by grouping a flat ``EpisodeResult.trajectory`` on ``ego_id``.

        The tick-loop appends wakes in TICK order across all egos, so a two-ego episode
        yields an INTERLEAVED list (``egoA@t1, egoB@t1, egoA@t5, ...``). Grouping splits
        it into each ego's own chain while preserving that ego's relative order — a
        stable partition, never a sort.

        Args:
            trajectory: the flat wake list from ``EpisodeResult.trajectory``. Empty is
                valid -> an empty ``chains`` dict.
            episode_reward: the episode's scalar ``R`` (``EpisodeReward.reward``).
            seed: the episode's seed, for provenance.
            episode_index: the episode's index, for provenance.

        Returns:
            The :class:`EpisodeRecord`.
        """
        chains: Dict[str, List["Transition"]] = {}
        for tr in trajectory:
            chains.setdefault(str(tr.ego_id), []).append(tr)
        return cls(
            chains=chains,
            episode_reward=float(episode_reward),
            seed=int(seed),
            episode_index=int(episode_index),
        )

    # ------------------------------------------------------------------
    def transitions(self) -> List["Transition"]:
        """Flatten the chains into one list, chain by chain (ego order, then time).

        The flattening order is deterministic (dicts preserve insertion order, and
        insertion order is first-wake order), so the returns / advantages arrays this
        module produces stay aligned with this list across calls.
        """
        out: List["Transition"] = []
        for chain in self.chains.values():
            out.extend(chain)
        return out

    @property
    def n_transitions(self) -> int:
        """Total number of wakes in this episode (0 for a zero-wake episode)."""
        return sum(len(c) for c in self.chains.values())

    @property
    def has_wakes(self) -> bool:
        """True iff this episode produced at least one RL decision."""
        return self.n_transitions > 0


# =============================================================================
# 2. PPOBuffer — accumulates records across the episodes of one PPO iteration
# =============================================================================

class PPOBuffer:
    """Accumulates :class:`EpisodeRecord`s for ONE PPO iteration, then is cleared.

    Torch-free. Deliberately thin: it owns storage and the flattened view, and NOT the
    credit assignment (that is :func:`compute_returns_and_advantages`, the replaceable
    component) — so Phase B can swap the credit math without touching the buffer.

    Typical outer-loop use::

        buf = PPOBuffer()
        for i in range(episodes_per_iteration):
            ...                                  # setup -> run -> reward
            buf.add(EpisodeRecord.from_trajectory(result.trajectory, er.reward,
                                                  seed=seed, episode_index=i))
        diag = updater.update(buf)
        buf.clear()
    """

    def __init__(self) -> None:
        self.records: List[EpisodeRecord] = []

    # ------------------------------------------------------------------
    def add(self, record: EpisodeRecord) -> None:
        """Append one finished episode's record (zero-wake records included)."""
        self.records.append(record)

    def clear(self) -> None:
        """Drop every record — call after each update so iterations stay on-policy."""
        self.records.clear()

    # ------------------------------------------------------------------
    def transitions(self) -> List["Transition"]:
        """Every transition in the buffer, in (episode, ego-chain, time) order."""
        out: List["Transition"] = []
        for rec in self.records:
            out.extend(rec.transitions())
        return out

    @property
    def n_episodes(self) -> int:
        """Number of episodes recorded — INCLUDING zero-wake ones (baseline mass)."""
        return len(self.records)

    @property
    def n_transitions(self) -> int:
        """Total number of wakes across every recorded episode."""
        return sum(rec.n_transitions for rec in self.records)

    @property
    def episodes_with_wakes(self) -> int:
        """How many recorded episodes produced at least one decision (diagnostic)."""
        return sum(1 for rec in self.records if rec.has_wakes)

    def __len__(self) -> int:
        return len(self.records)


# Anything the credit / update functions accept as their batch.
RecordSource = Union[PPOBuffer, Sequence[EpisodeRecord]]


def _as_records(source: RecordSource) -> List[EpisodeRecord]:
    """Normalize a buffer OR a bare record sequence to a list of records."""
    if isinstance(source, PPOBuffer):
        return list(source.records)
    return list(source)


# =============================================================================
# 3. Config
# =============================================================================

@dataclass(frozen=True)
class PPOConfig:
    """PPO hyper-parameters. Frozen: an update's settings must not drift mid-run.

    Attributes:
        clip_ratio: the PPO clipping parameter ``eps``. The surrogate uses
            ``min(ratio*A, clamp(ratio, 1-eps, 1+eps)*A)``, so a single update can
            never move the policy more than ~20% (in probability ratio) in the
            direction the advantage favours. 0.2 is the canonical value.
        entropy_coeff: weight of the entropy bonus. Entropy is MAXIMIZED, so it enters
            the loss with a MINUS sign (``loss -= entropy_coeff * entropy``). It keeps
            the masked-softmax policy from collapsing onto one meta-action early,
            which matters here because PLAN_COMPLIANCE is always valid and is the easy
            local optimum.
        lr: Adam learning rate over encoder + head parameters (one optimizer, built
            once, so its moment estimates persist across updates).
        n_epochs: how many passes over the SAME batch per update. >1 is the whole point
            of the clipped surrogate (it is what makes off-by-a-few-steps reuse safe).
        gamma: discount. DORMANT at 1.0 in Phase A — see the module docstring, "THE
            CREDIT STRUCTURE" (2). Kept explicit and honoured by
            :func:`compute_returns_and_advantages` so the generalization is testable
            before Phase B needs it.
        max_grad_norm: global grad-norm clip applied to encoder + head together before
            each optimizer step. Guards against a single outlier graph producing a
            step that wrecks the shared encoder.
        adv_norm_eps: epsilon in the advantage normalization denominator
            (``std + eps``). A batch whose episodes all scored the SAME R has zero
            advantage variance; this makes that case produce ~0 advantages instead of
            NaNs (see :func:`compute_returns_and_advantages`).
    """

    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    lr: float = 3e-4
    n_epochs: int = 4
    gamma: float = 1.0
    max_grad_norm: float = 0.5
    adv_norm_eps: float = 1e-8


# =============================================================================
# 4. Returns + advantages — THE REPLACEABLE COMPONENT (Phase-B swaps this)
# =============================================================================

@dataclass
class AdvantageBatch:
    """The flattened, aligned view the update consumes.

    ``transitions[i]`` corresponds to ``returns[i]`` / ``advantages[i]`` /
    ``raw_advantages[i]`` — the alignment is the contract.

    Attributes:
        transitions: flattened (episode, ego-chain, time) transition list.
        returns: per-transition return ``G_t``. Phase A: the episode's ``R``
            (gamma-discounted back from the chain end when ``gamma < 1``).
        advantages: NORMALIZED advantages (mean ~0, std ~1) — what the update uses.
        raw_advantages: pre-normalization ``return - baseline``, kept for logging.
        baseline: mean episode reward across the batch's EPISODES (not transitions).
        adv_mean_raw / adv_std_raw: moments of ``raw_advantages`` (logging; a
            ``adv_std_raw`` of 0.0 flags the degenerate all-same-R batch).
        n_episodes / n_transitions: batch shape, for diagnostics.
    """

    transitions: List["Transition"]
    returns: np.ndarray
    advantages: np.ndarray
    raw_advantages: np.ndarray
    baseline: float
    adv_mean_raw: float
    adv_std_raw: float
    n_episodes: int
    n_transitions: int


def _chain_returns(chain_len: int, episode_reward: float, gamma: float) -> List[float]:
    """Per-transition returns for ONE ego chain under a single TERMINAL reward.

    The episode's scalar ``R`` is realized at the END of the chain, so the return of
    the ``t``-th decision (0-based, chain length ``n``) is ``R * gamma**(n-1-t)``.

    At the Phase-A default ``gamma == 1.0`` this is exactly ``R`` for every element —
    which is why the per-ego grouping does not change any number today. The
    generalization is written out rather than hard-coded to ``R`` so that ``gamma``
    is a real, testable field and the chain structure is already load-bearing when
    Phase B introduces dense rewards.

    PHASE-B SEAM: this is where GAE replaces the plain return. With a critic, this
    function becomes ``delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`` accumulated
    backwards with ``gamma*lambda``, consuming EXACTLY the same per-ego chain — the
    reason the buffer stores chains rather than a flat list.
    """
    if chain_len <= 0:
        return []
    if gamma == 1.0:  # exact, no float drift from pow(1.0, m)
        return [float(episode_reward)] * chain_len
    return [
        float(episode_reward) * (gamma ** (chain_len - 1 - t))
        for t in range(chain_len)
    ]


def compute_returns_and_advantages(
    source: RecordSource,
    cfg: PPOConfig = PPOConfig(),
) -> AdvantageBatch:
    """Turn per-ego-grouped episode records into aligned returns + normalized advantages.

    THE REPLACEABLE COMPONENT. The update calls this and nothing else for credit
    assignment; Phase B swaps the body (critic + GAE) without changing the signature,
    the buffer, or the update loop.

    Phase-A semantics (all four decisions justified in the module docstring):
      1. return of each transition = its episode's ``R``, discounted back from the
         chain end by ``gamma`` (a no-op at the default ``gamma == 1.0``);
      2. baseline = mean ``episode_reward`` over the batch's EPISODES — zero-wake
         episodes included, transition counts irrelevant;
      3. raw advantage = return - baseline;
      4. advantages = ``(raw - mean(raw)) / (std(raw) + adv_norm_eps)``.

    Args:
        source: a :class:`PPOBuffer` or a bare sequence of :class:`EpisodeRecord`.
        cfg: supplies ``gamma`` and ``adv_norm_eps``.

    Returns:
        An :class:`AdvantageBatch` whose arrays are index-aligned with its
        ``transitions`` list.

    Edge cases (both are NORMAL operating conditions, not errors):
      * **Zero-variance batch** (every episode scored the same ``R``, e.g. a batch of
        total failures early in training): ``std == 0``, so the normalized advantages
        are ``0 / adv_norm_eps == 0``. NO NaN. The subsequent update is then a
        near-no-op apart from the entropy bonus — which is CORRECT: a batch in which
        no episode did better than any other carries no signal about which action to
        prefer, and inventing one would be noise.
      * **Zero transitions** (empty batch, or every episode zero-wake): empty float64
        arrays and an empty transition list are returned. The baseline is still the
        mean over whatever episodes exist (0.0 if there are none), so it stays
        loggable.
    """
    records = _as_records(source)
    n_episodes = len(records)

    # --- Baseline: mean over EPISODES (zero-wake episodes carry full weight). ---
    if n_episodes > 0:
        baseline = float(np.mean([rec.episode_reward for rec in records]))
    else:
        baseline = 0.0

    # --- Per-ego chain returns, flattened in the record's own flatten order. ---
    transitions: List["Transition"] = []
    returns_list: List[float] = []
    for rec in records:
        for chain in rec.chains.values():
            transitions.extend(chain)
            returns_list.extend(
                _chain_returns(len(chain), rec.episode_reward, cfg.gamma)
            )

    if not transitions:
        empty = np.zeros(0, dtype=np.float64)
        return AdvantageBatch(
            transitions=[],
            returns=empty,
            advantages=empty.copy(),
            raw_advantages=empty.copy(),
            baseline=baseline,
            adv_mean_raw=0.0,
            adv_std_raw=0.0,
            n_episodes=n_episodes,
            n_transitions=0,
        )

    returns = np.asarray(returns_list, dtype=np.float64)
    raw_adv = returns - baseline

    adv_mean = float(raw_adv.mean())
    adv_std = float(raw_adv.std())  # population std (ddof=0), the PPO convention
    # eps guard: an all-same-R batch has adv_std == 0 -> 0/eps == 0, never NaN.
    advantages = (raw_adv - adv_mean) / (adv_std + cfg.adv_norm_eps)

    return AdvantageBatch(
        transitions=transitions,
        returns=returns,
        advantages=advantages,
        raw_advantages=raw_adv,
        baseline=baseline,
        adv_mean_raw=adv_mean,
        adv_std_raw=adv_std,
        n_episodes=n_episodes,
        n_transitions=len(transitions),
    )


# =============================================================================
# 5. The clipped surrogate (factored out so it is hand-checkable in isolation)
# =============================================================================

def clipped_surrogate(
    ratio: torch.Tensor,
    advantage: float,
    clip_ratio: float,
) -> torch.Tensor:
    """The per-transition PPO policy loss ``-min(r*A, clamp(r, 1-eps, 1+eps)*A)``.

    Factored out of :meth:`PPOUpdater.update` so a test can hand-check the clamped
    branch on known numbers rather than inferring it from an aggregate diagnostic.

    Sign convention: PPO MAXIMIZES the surrogate objective, so the LOSS is its
    negation. A positive advantage with an unclipped ratio therefore yields a loss
    that DECREASES as the ratio grows — i.e. gradient descent pushes the action's
    probability UP, which is the whole mechanism.

    Which branch binds:
      * ``A > 0`` and ``ratio > 1+eps``: the clamped term is smaller -> the loss
        saturates at ``-(1+eps)*A``, so no further gradient rewards pushing the
        probability higher. This is the trust region.
      * ``A < 0`` and ``ratio < 1-eps``: the clamped term is again the smaller of the
        two -> the loss saturates at ``-(1-eps)*A``.
      * otherwise: the unclipped term binds and the gradient is the vanilla PG one.

    Args:
        ratio: scalar tensor ``exp(log_prob_new - log_prob_old)`` (grad-attached).
        advantage: the transition's NORMALIZED advantage (a python float — it is a
            constant w.r.t. the parameters, never a gradient path).
        clip_ratio: ``eps``.

    Returns:
        A scalar tensor: the per-transition policy loss.
    """
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    return -torch.min(unclipped, clipped)


# =============================================================================
# 6. PPOUpdater — the clipped update step
# =============================================================================

class PPOUpdater:
    """Owns the optimizer and performs the clipped PPO update on a batch of records.

    Built ONCE per training run from the ONE :class:`Policy` that lives across
    episodes (``graph_tick_loop.build_policy``), because Adam's moment estimates are
    state: rebuilding the optimizer each iteration would silently throw them away.

    Actor-only (Phase A): the loss is the clipped surrogate minus the entropy bonus.
    There is NO value loss — see the PHASE-B SEAM comment in :meth:`update`.
    """

    def __init__(self, policy: "Policy", cfg: PPOConfig = PPOConfig()) -> None:
        """Bind the updater to a policy and build its single Adam optimizer.

        Args:
            policy: the encoder + head bundle. Both modules' parameters go into ONE
                optimizer — the encoder is trained by the policy gradient alone in
                Phase A (Phase B adds the critic's gradient through the same encoder).
            cfg: the :class:`PPOConfig` (frozen for the run).
        """
        self.policy = policy
        self.cfg = cfg
        self.parameters: List[torch.nn.Parameter] = (
            list(policy.encoder.parameters()) + list(policy.head.parameters())
        )
        self.optimizer = torch.optim.Adam(self.parameters, lr=cfg.lr)

    # ------------------------------------------------------------------
    def _forward_logits(self, gobs: Any) -> torch.Tensor:
        """Re-run encoder -> head on a STORED observation, WITH grad.

        Deliberately no ``torch.no_grad`` anywhere on this path: the whole point of
        the update is that gradients flow from the re-scored log-prob back through the
        head into the shared encoder.
        """
        emb = self.policy.encoder(gobs)
        return self.policy.head(emb)

    # ------------------------------------------------------------------
    def update(self, source: RecordSource) -> Dict[str, Any]:
        """Run ``cfg.n_epochs`` clipped-PPO epochs over ``source`` and step the optimizer.

        Per epoch, per transition (the encoder is SINGLE-GRAPH, so this is a python
        loop over transitions — see the module docstring):

          1. re-encode ``tr.gobs`` WITH grad -> ``[k, 3]`` logits;
          2. REBUILD the mask from ``tr.gobs`` (pure function; never stored);
          3. ``evaluate_action`` -> ``(log_prob_new, entropy)`` under the SAME masked
             distribution the rollout sampled from;
          4. ``ratio = exp(log_prob_new - tr.log_prob)`` (``tr.log_prob`` is the stored,
             detached rollout value -> a constant, so the ratio's grad is the new
             log-prob's);
          5. ``clipped_surrogate(ratio, A, clip)`` minus ``entropy_coeff * entropy``.

        Then ONE mean over the batch's transitions, ONE backward, a global grad-norm
        clip, and ONE optimizer step per epoch.

        PHASE-B SEAM: the critic's value loss joins the per-transition loss right where
        the entropy bonus is subtracted, as
        ``+ value_coeff * (V(pool(emb)) - batch.returns[i])**2`` — the encoder forward
        above already produces the embedding ``GraphEncoder.pool()`` consumes, so the
        critic costs no extra forward pass. ``batch.returns`` is already computed and
        aligned for exactly that use.

        Epoch 0 is diagnostically special: the weights are UNCHANGED from the rollout,
        so every ratio is exactly 1.0 and the reported ``per_epoch`` entries at index 0
        are PRE-first-step values (the loss is computed before ``backward``). That is
        what makes the epoch-0 identity assertable from the returned diagnostics.

        Args:
            source: a :class:`PPOBuffer` or a sequence of :class:`EpisodeRecord`.

        Returns:
            A diagnostics dict. Scalar entries are MEANS OVER EPOCHS (except
            ``baseline`` / batch-shape fields, which are per-update constants);
            ``per_epoch`` holds the per-epoch lists, so ``per_epoch["policy_loss"][0]``
            is the epoch-0, pre-step value. Keys:
            ``policy_loss``, ``total_loss``, ``entropy``, ``mean_ratio``,
            ``clip_fraction``, ``approx_kl``, ``max_ratio_dev``, ``grad_norm``,
            ``baseline``, ``adv_std_raw``, ``n_transitions``, ``n_episodes``,
            ``episodes_with_wakes``, ``n_epochs_run``, ``per_epoch``.

        Empty-batch contract: a batch with ZERO transitions is a clean NO-OP — no
        forward, no backward, no optimizer step — and returns the same dict shape with
        ``n_epochs_run == 0`` and empty ``per_epoch`` lists. It is NOT an error: a
        rollout iteration in which no ego ever woke is a legitimate outcome of the
        event-triggered design, and the outer loop should be able to log it and move on
        rather than catch an exception.
        """
        cfg = self.cfg
        records = _as_records(source)
        batch = compute_returns_and_advantages(records, cfg)

        n_ep_with_wakes = sum(1 for rec in records if rec.has_wakes)
        per_epoch: Dict[str, List[float]] = {
            "policy_loss": [], "total_loss": [], "entropy": [], "mean_ratio": [],
            "clip_fraction": [], "approx_kl": [], "max_ratio_dev": [], "grad_norm": [],
        }
        diagnostics: Dict[str, Any] = {
            "baseline": batch.baseline,
            "adv_std_raw": batch.adv_std_raw,
            "n_transitions": batch.n_transitions,
            "n_episodes": batch.n_episodes,
            "episodes_with_wakes": n_ep_with_wakes,
            "n_epochs_run": 0,
            "per_epoch": per_epoch,
        }

        if batch.n_transitions == 0:
            # Clean no-op (documented above): nothing to learn from.
            for key in ("policy_loss", "total_loss", "entropy", "mean_ratio",
                        "clip_fraction", "approx_kl", "max_ratio_dev", "grad_norm"):
                diagnostics[key] = 0.0
            return diagnostics

        advantages = batch.advantages
        n = batch.n_transitions

        for _epoch in range(cfg.n_epochs):
            policy_losses: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []
            ratios_detached: List[float] = []
            kl_terms: List[float] = []

            for i, tr in enumerate(batch.transitions):
                logits = self._forward_logits(tr.gobs)
                # Pure function of the stored observation -> rebuilt, never stored.
                mask = build_action_mask(tr.gobs)
                log_prob_new, entropy = evaluate_action(
                    logits, mask, tr.meta_action, tr.node_v
                )
                # tr.log_prob is the DETACHED rollout value (a constant here).
                ratio = torch.exp(log_prob_new - float(tr.log_prob))

                policy_losses.append(
                    clipped_surrogate(ratio, float(advantages[i]), cfg.clip_ratio)
                )
                entropies.append(entropy)

                r = float(ratio.detach().item())
                ratios_detached.append(r)
                # approx_kl = mean(log_prob_old - log_prob_new), the standard
                # cheap KL proxy; positive means the new policy moved away.
                kl_terms.append(float(tr.log_prob) - float(log_prob_new.detach().item()))

            policy_loss = torch.stack(policy_losses).mean()
            entropy_mean = torch.stack(entropies).mean()
            # Entropy is MAXIMIZED -> it enters the loss with a MINUS sign.
            # PHASE-B SEAM: the critic's value loss is ADDED here.
            total_loss = policy_loss - cfg.entropy_coeff * entropy_mean

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, cfg.max_grad_norm
            )
            self.optimizer.step()

            ratios_arr = np.asarray(ratios_detached, dtype=np.float64)
            per_epoch["policy_loss"].append(float(policy_loss.detach().item()))
            per_epoch["total_loss"].append(float(total_loss.detach().item()))
            per_epoch["entropy"].append(float(entropy_mean.detach().item()))
            per_epoch["mean_ratio"].append(float(ratios_arr.mean()))
            per_epoch["clip_fraction"].append(
                float(np.mean(np.abs(ratios_arr - 1.0) > cfg.clip_ratio))
            )
            per_epoch["approx_kl"].append(float(np.mean(kl_terms)))
            per_epoch["max_ratio_dev"].append(float(np.max(np.abs(ratios_arr - 1.0))))
            per_epoch["grad_norm"].append(float(grad_norm))

        diagnostics["n_epochs_run"] = cfg.n_epochs
        for key, values in per_epoch.items():
            diagnostics[key] = float(np.mean(values)) if values else 0.0
        assert diagnostics["n_transitions"] == n  # alignment sanity
        return diagnostics


# =============================================================================
# Self-test — synthetic data only (no BLADE, no solver, no env)
# =============================================================================

def _make_obs(
    k: int = 4,
    *,
    sensed_all: bool = False,
    seed_shift: float = 0.0,
) -> Any:
    """A hand-built ``GraphObservation`` with a KNOWN mask (mirrors the action tests).

    Topology (k=4, a=3, ego_index == k == 4):
      task 0: assigned to ego  (4->0), sensed          -> COMPLIANCE + ABORT
      task 1: assigned to peer1(5->1), sensed          -> COMPLIANCE
      task 2: UNASSIGNED, sensed, capable, reachable   -> COMPLIANCE + ENGAGEMENT
      task 3: assigned to peer2(6->3), NOT sensed      -> COMPLIANCE

    ``seed_shift`` perturbs the utility column so two "different" states can be built
    without changing the mask topology.
    """
    from ..observation.graph_builder import EdgeType, GraphObservation

    rows = [
        # [utility, dist_to_ego, capable, reachable, probability, sensed]
        [0.80 + seed_shift, 0.20, 1.0, 1.0, 1.0, 1.0],   # task 0
        [0.60 + seed_shift, 0.40, 1.0, 1.0, 1.0, 1.0],   # task 1
        [0.50 + seed_shift, 0.30, 1.0, 1.0, 1.0, 1.0],   # task 2 (engageable)
        [0.70 + seed_shift, 0.50, 1.0, 1.0, 1.0, 1.0 if sensed_all else 0.0],
    ][:k]
    task_features = np.asarray(rows, dtype=np.float32)
    agent_features = np.array([[0.90], [0.00], [0.00]], dtype=np.float32)
    return GraphObservation(
        task_features=task_features,
        agent_features=agent_features,
        ego_index=k,
        edge_index=np.array([[k, k + 1, k + 2],
                             [0, 1, 3]], dtype=np.int64),
        edge_type=np.array([int(EdgeType.ASSIGNMENT)] * 3, dtype=np.int64),
        task_target_ids=[f"t{i}" for i in range(k)],
        agent_ids=["ego", "peer1", "peer2"],
        agent_id="ego",
        current_time=0,
        time_norm=0.0,
    )


def _make_transition(
    policy: "Policy",
    gobs: Any,
    ego_id: str,
    tick: int,
    *,
    force_cell: Optional[Tuple[int, int]] = None,
) -> "Transition":
    """Produce ONE real :class:`Transition`, storing exactly what ``_wake_decision`` stores.

    This is SYNTHETIC ROLLOUT DATA construction — the one place ``torch.no_grad`` is
    legitimate in this module's test path (it mirrors the inference-only rollout).

    ``force_cell=(meta_action, node_v)`` pins the stored action instead of sampling it,
    scoring it through ``evaluate_action``. That is the SAME distribution
    ``sample_action`` would have drawn from (one shared ``_masked_dist`` construction
    site, proven bitwise in ``tests/test_graph_action_evaluate.py``), so the stored
    log-prob is exactly the one a rollout that happened to sample this cell would have
    stored. It makes the learning-direction / clipping proofs deterministic instead of
    leaving them to the sampler.
    """
    from .graph_tick_loop import Transition

    mask = build_action_mask(gobs)
    with torch.no_grad():
        logits = policy.head(policy.encoder(gobs))
        if force_cell is None:
            from ..action.graph_action import sample_action
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


def _action_prob(policy: "Policy", gobs: Any, meta_action: int, node_v: int) -> float:
    """Current probability of ``(meta_action, node_v)`` on ``gobs`` (fresh, no-grad)."""
    mask = build_action_mask(gobs)
    with torch.no_grad():
        logits = policy.head(policy.encoder(gobs))
        log_prob, _ = evaluate_action(logits, mask, meta_action, node_v)
    return float(torch.exp(log_prob).item())


def _selftest() -> None:
    """Proofs P1-P6 on synthetic data. No BLADE, no bonmin, no env.

    Run from the repo root, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.training.graph_ppo
    """
    from ..action.graph_action import MetaAction
    from .graph_tick_loop import build_policy

    print("=" * 72)
    print("graph_ppo self-test (PPO core, Phase A: actor-only)")
    print("=" * 72)

    OE = int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)
    PC = int(MetaAction.PLAN_COMPLIANCE)

    # =====================================================================
    # P4 — grouping correctness (pure data; no policy needed for the shape)
    # =====================================================================
    print("-" * 72)
    print("[P4] grouping: interleaved trajectory -> per-ego chains; zero-wake episode")
    torch.manual_seed(0)
    policy_g = build_policy(embed_dim=64)
    obs_g = _make_obs()
    # An INTERLEAVED trajectory exactly as the tick-loop emits it (tick order,
    # both egos each tick): A@1, B@1, A@5, B@5, A@9.
    interleaved = [
        _make_transition(policy_g, obs_g, "egoA", 1),
        _make_transition(policy_g, obs_g, "egoB", 1),
        _make_transition(policy_g, obs_g, "egoA", 5),
        _make_transition(policy_g, obs_g, "egoB", 5),
        _make_transition(policy_g, obs_g, "egoA", 9),
    ]
    rec = EpisodeRecord.from_trajectory(interleaved, -0.25, seed=7, episode_index=3)
    assert list(rec.chains.keys()) == ["egoA", "egoB"], rec.chains.keys()
    assert [t.tick for t in rec.chains["egoA"]] == [1, 5, 9]
    assert [t.tick for t in rec.chains["egoB"]] == [1, 5]
    assert rec.n_transitions == 5 and rec.has_wakes
    print(f"  egoA chain ticks={[t.tick for t in rec.chains['egoA']]}  "
          f"egoB chain ticks={[t.tick for t in rec.chains['egoB']]}   OK")

    zero_wake = EpisodeRecord.from_trajectory([], -1.0, seed=8, episode_index=4)
    assert zero_wake.chains == {} and zero_wake.n_transitions == 0
    assert not zero_wake.has_wakes

    buf_g = PPOBuffer()
    buf_g.add(rec)
    buf_g.add(zero_wake)
    assert buf_g.n_episodes == 2 and buf_g.n_transitions == 5
    assert buf_g.episodes_with_wakes == 1
    batch_g = compute_returns_and_advantages(buf_g)
    # Baseline is the mean over EPISODES: (-0.25 + -1.0)/2 = -0.625 — the zero-wake
    # episode pulls it fully even though it contributed no transitions.
    assert abs(batch_g.baseline - (-0.625)) < 1e-12, batch_g.baseline
    assert batch_g.n_episodes == 2 and batch_g.n_transitions == 5
    # gamma == 1.0 -> every return is the episode's R.
    assert np.allclose(batch_g.returns, -0.25), batch_g.returns
    print(f"  zero-wake episode: n_episodes={batch_g.n_episodes} "
          f"n_transitions={batch_g.n_transitions} baseline={batch_g.baseline:.4f} "
          f"(mean over EPISODES)   OK")

    # A gamma < 1.0 sanity check: the chain structure is genuinely load-bearing.
    batch_disc = compute_returns_and_advantages(buf_g, PPOConfig(gamma=0.5))
    # egoA chain (len 3): R*0.25, R*0.5, R ; egoB chain (len 2): R*0.5, R
    want = np.array([-0.25 * 0.25, -0.25 * 0.5, -0.25, -0.25 * 0.5, -0.25])
    assert np.allclose(batch_disc.returns, want), (batch_disc.returns, want)
    print(f"  gamma=0.5 discounts back along each chain: "
          f"{[round(float(x), 4) for x in batch_disc.returns]}   OK")

    # The zero-wake episode must not crash an update either.
    upd_g = PPOUpdater(policy_g, PPOConfig(n_epochs=1))
    diag_g = upd_g.update(buf_g)
    assert diag_g["n_transitions"] == 5 and diag_g["n_epochs_run"] == 1
    assert np.isfinite(diag_g["policy_loss"])
    print(f"  update over a batch containing a zero-wake episode ran: "
          f"policy_loss={diag_g['policy_loss']:.6e}   OK")

    # =====================================================================
    # P1 — epoch-0 identity: ratio == 1.0 exactly; policy loss == -mean(A) ~ 0
    # =====================================================================
    print("-" * 72)
    print("[P1] epoch-0 identity (unchanged weights -> ratio 1.0, loss ~ 0)")
    torch.manual_seed(1)
    policy1 = build_policy(embed_dim=64)
    buf1 = PPOBuffer()
    for i, R in enumerate([-0.10, -0.55, -0.90, -0.30]):
        obs_i = _make_obs(seed_shift=0.01 * i)
        traj = [
            _make_transition(policy1, obs_i, "egoA", 1 + i),
            _make_transition(policy1, obs_i, "egoB", 2 + i),
        ]
        buf1.add(EpisodeRecord.from_trajectory(traj, R, seed=i, episode_index=i))

    batch1 = compute_returns_and_advantages(buf1)
    print(f"  batch: n_episodes={batch1.n_episodes} n_transitions={batch1.n_transitions} "
          f"baseline={batch1.baseline:.6f} adv_std_raw={batch1.adv_std_raw:.6f}")
    print(f"  advantages (normalized): "
          f"{[round(float(a), 4) for a in batch1.advantages]}")
    assert abs(float(batch1.advantages.mean())) < 1e-9
    assert abs(float(batch1.advantages.std()) - 1.0) < 1e-6

    upd1 = PPOUpdater(policy1, PPOConfig(n_epochs=1, entropy_coeff=0.01))
    diag1 = upd1.update(buf1)
    max_dev = diag1["per_epoch"]["max_ratio_dev"][0]
    mean_ratio0 = diag1["per_epoch"]["mean_ratio"][0]
    loss0 = diag1["per_epoch"]["policy_loss"][0]
    expected_loss0 = -float(batch1.advantages.mean())
    print(f"  epoch-0 mean_ratio={mean_ratio0:.12f}  "
          f"max |ratio - 1| = {max_dev:.3e}")
    print(f"  epoch-0 policy_loss={loss0:.3e}  expected -mean(A_norm)="
          f"{expected_loss0:.3e}  |diff|={abs(loss0 - expected_loss0):.3e}")
    assert max_dev <= 1e-6, f"epoch-0 ratio deviates by {max_dev}"
    assert abs(loss0 - expected_loss0) <= 1e-6, (loss0, expected_loss0)
    assert abs(loss0) <= 1e-6, f"epoch-0 policy loss is not ~0: {loss0}"
    assert abs(diag1["per_epoch"]["approx_kl"][0]) <= 1e-6
    print("  ratio == 1.0 (within 1e-6), loss == -mean(A_norm) ~ 0   OK")

    # =====================================================================
    # P2 — learning direction: a clearly-positive-advantage action gets likelier
    # =====================================================================
    print("-" * 72)
    print("[P2] learning direction (positive advantage -> action probability rises)")
    torch.manual_seed(2)
    policy2 = build_policy(embed_dim=64)
    obs_shared = _make_obs()  # the SAME state in both episodes -> directly comparable
    good_cell = (OE, 2)   # engage the unassigned pop-up (node 2)
    bad_cell = (PC, 0)    # comply on the ego's own assigned task

    tr_good = _make_transition(policy2, obs_shared, "egoA", 1, force_cell=good_cell)
    tr_bad = _make_transition(policy2, obs_shared, "egoA", 1, force_cell=bad_cell)
    buf2 = PPOBuffer()
    buf2.add(EpisodeRecord.from_trajectory([tr_good], 0.0, seed=0, episode_index=0))
    buf2.add(EpisodeRecord.from_trajectory([tr_bad], -1.0, seed=1, episode_index=1))
    batch2 = compute_returns_and_advantages(buf2)
    print(f"  baseline={batch2.baseline:.4f}  raw_adv="
          f"{[round(float(a), 4) for a in batch2.raw_advantages]}  "
          f"norm_adv={[round(float(a), 4) for a in batch2.advantages]}")
    assert batch2.advantages[0] > 0.5 > batch2.advantages[1], batch2.advantages

    p_good_before = _action_prob(policy2, obs_shared, *good_cell)
    p_bad_before = _action_prob(policy2, obs_shared, *bad_cell)
    upd2 = PPOUpdater(policy2, PPOConfig(lr=1e-3, n_epochs=4))
    for _ in range(6):
        upd2.update(buf2)
    p_good_after = _action_prob(policy2, obs_shared, *good_cell)
    p_bad_after = _action_prob(policy2, obs_shared, *bad_cell)
    print(f"  P(good=(node2,OE)): {p_good_before:.6f} -> {p_good_after:.6f}  "
          f"(delta {p_good_after - p_good_before:+.6f})")
    print(f"  P(bad =(node0,PC)): {p_bad_before:.6f} -> {p_bad_after:.6f}  "
          f"(delta {p_bad_after - p_bad_before:+.6f})")
    assert p_good_after > p_good_before, (p_good_before, p_good_after)
    assert p_bad_after < p_bad_before, (p_bad_before, p_bad_after)
    print("  positive-advantage action rose, negative-advantage action fell   OK")

    # =====================================================================
    # P3 — clip is live (both the aggregate diagnostic and the branch itself)
    # =====================================================================
    print("-" * 72)
    print("[P3] clipping engages")
    # (a) numeric hand-check of the clamped branch, in isolation.
    clip = 0.2
    r_hi = torch.tensor(1.5)   # A > 0 and ratio > 1+clip -> clamped branch binds
    loss_hi = clipped_surrogate(r_hi, 2.0, clip)
    assert abs(float(loss_hi) - (-(1.0 + clip) * 2.0)) < 1e-6, float(loss_hi)
    r_lo = torch.tensor(0.5)   # A < 0 and ratio < 1-clip -> clamped branch binds
    loss_lo = clipped_surrogate(r_lo, -2.0, clip)
    assert abs(float(loss_lo) - (-(1.0 - clip) * -2.0)) < 1e-6, float(loss_lo)
    r_in = torch.tensor(1.05)  # inside the trust region -> unclipped branch
    loss_in = clipped_surrogate(r_in, 2.0, clip)
    assert abs(float(loss_in) - (-1.05 * 2.0)) < 1e-6, float(loss_in)
    print(f"  hand-check: A=+2 r=1.5 -> loss={float(loss_hi):.4f} (== -(1+clip)*A)")
    print(f"              A=-2 r=0.5 -> loss={float(loss_lo):.4f} (== -(1-clip)*A)")
    print(f"              A=+2 r=1.05-> loss={float(loss_in):.4f} (== -r*A, unclipped)")

    # (b) a real update with a large lr drives |ratio - 1| past the clip.
    torch.manual_seed(3)
    policy3 = build_policy(embed_dim=64)
    tr_g3 = _make_transition(policy3, obs_shared, "egoA", 1, force_cell=good_cell)
    tr_b3 = _make_transition(policy3, obs_shared, "egoA", 1, force_cell=bad_cell)
    buf3 = PPOBuffer()
    buf3.add(EpisodeRecord.from_trajectory([tr_g3], 0.0, seed=0, episode_index=0))
    buf3.add(EpisodeRecord.from_trajectory([tr_b3], -1.0, seed=1, episode_index=1))
    upd3 = PPOUpdater(policy3, PPOConfig(lr=0.05, n_epochs=8, clip_ratio=0.2))
    diag3 = upd3.update(buf3)
    print(f"  per-epoch max|ratio-1|: "
          f"{[round(v, 4) for v in diag3['per_epoch']['max_ratio_dev']]}")
    print(f"  per-epoch clip_fraction: "
          f"{[round(v, 4) for v in diag3['per_epoch']['clip_fraction']]}")
    assert max(diag3["per_epoch"]["clip_fraction"]) > 0.0, "clipping never engaged"
    assert max(diag3["per_epoch"]["max_ratio_dev"]) > 0.2
    print(f"  clip_fraction (mean over epochs)={diag3['clip_fraction']:.4f} > 0   OK")

    # =====================================================================
    # P5 — degenerate batches
    # =====================================================================
    print("-" * 72)
    print("[P5] degenerate batches (all-same-R; empty)")
    torch.manual_seed(4)
    policy5 = build_policy(embed_dim=64)
    buf5 = PPOBuffer()
    for i in range(3):
        obs_i = _make_obs(seed_shift=0.02 * i)
        buf5.add(EpisodeRecord.from_trajectory(
            [_make_transition(policy5, obs_i, "egoA", i)], -0.5,
            seed=i, episode_index=i,
        ))
    batch5 = compute_returns_and_advantages(buf5)
    assert np.all(np.isfinite(batch5.advantages)), batch5.advantages
    assert float(np.max(np.abs(batch5.advantages))) < 1e-6, batch5.advantages
    assert batch5.adv_std_raw == 0.0
    upd5 = PPOUpdater(policy5, PPOConfig(n_epochs=2))
    diag5 = upd5.update(buf5)
    assert all(np.isfinite(v) for v in diag5["per_epoch"]["policy_loss"])
    assert np.isfinite(diag5["entropy"]) and np.isfinite(diag5["grad_norm"])
    print(f"  all-same-R: adv_std_raw={batch5.adv_std_raw} "
          f"max|A_norm|={float(np.max(np.abs(batch5.advantages))):.3e} "
          f"policy_loss={diag5['policy_loss']:.3e} (near-no-op, no NaN)   OK")

    empty_batch = compute_returns_and_advantages(PPOBuffer())
    assert empty_batch.n_transitions == 0 and empty_batch.returns.shape == (0,)
    diag_empty = PPOUpdater(build_policy(embed_dim=64)).update(PPOBuffer())
    assert diag_empty["n_epochs_run"] == 0 and diag_empty["n_transitions"] == 0
    assert diag_empty["policy_loss"] == 0.0
    print(f"  empty buffer: clean no-op, n_epochs_run={diag_empty['n_epochs_run']}, "
          f"empty arrays, no crash   OK")

    # =====================================================================
    # P6 — finite grads on every exercised encoder + head parameter
    # =====================================================================
    print("-" * 72)
    print("[P6] finite grads on every exercised encoder + head parameter")
    torch.manual_seed(5)
    policy6 = build_policy(embed_dim=64)
    buf6 = PPOBuffer()
    for i, R in enumerate([-0.2, -0.8]):
        obs_i = _make_obs(seed_shift=0.03 * i)
        buf6.add(EpisodeRecord.from_trajectory(
            [_make_transition(policy6, obs_i, "egoA", i),
             _make_transition(policy6, obs_i, "egoB", i)],
            R, seed=i, episode_index=i,
        ))
    upd6 = PPOUpdater(policy6, PPOConfig(n_epochs=1))
    upd6.update(buf6)  # grads from the last (only) epoch survive the step

    # `edge_attr_proj` is the encoder's RESERVED edge-attribute path (the builder emits
    # no edge_attr), so it is legitimately unexercised — every OTHER param must have one.
    _EDGE_ATTR_PARAMS = {"encoder.edge_attr_proj.weight", "encoder.edge_attr_proj.bias"}
    with_grad, without_grad = [], []
    for mod_name, module in (("encoder", policy6.encoder), ("head", policy6.head)):
        for name, p in module.named_parameters():
            full = f"{mod_name}.{name}"
            if p.grad is None:
                without_grad.append(full)
                continue
            assert torch.isfinite(p.grad).all(), f"{full} has a non-finite grad"
            with_grad.append(full)
    unexpected = [nm for nm in without_grad if nm not in _EDGE_ATTR_PARAMS]
    print(f"  {len(with_grad)} params with finite grad, "
          f"{len(without_grad)} without ({without_grad or 'none'})")
    assert not unexpected, f"parameters got NO grad from the update: {unexpected}"
    assert len(with_grad) >= 30, f"suspiciously few grads: {len(with_grad)}"
    any_nonzero = any(
        bool(p.grad.abs().sum() > 0)
        for _, p in list(policy6.encoder.named_parameters())
        + list(policy6.head.named_parameters())
        if p.grad is not None
    )
    assert any_nonzero, "every grad is exactly zero — the gradient path is dead"
    print("  all exercised params have finite grads; at least one is non-zero   OK")

    # =====================================================================
    # Diagnostics printout (the dict the outer training loop will log)
    # =====================================================================
    print("-" * 72)
    print("diagnostics dict (from the P1 batch, n_epochs=4):")
    upd_d = PPOUpdater(build_policy(embed_dim=64), PPOConfig(n_epochs=4))
    torch.manual_seed(6)
    diag_d = upd_d.update(buf1)
    for key in ("n_episodes", "episodes_with_wakes", "n_transitions", "n_epochs_run",
                "baseline", "adv_std_raw", "policy_loss", "total_loss", "entropy",
                "mean_ratio", "clip_fraction", "approx_kl", "max_ratio_dev",
                "grad_norm"):
        value = diag_d[key]
        shown = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key:<20} {shown}")
    print("  per_epoch:")
    for key, values in diag_d["per_epoch"].items():
        print(f"    {key:<18} {[round(v, 6) for v in values]}")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
