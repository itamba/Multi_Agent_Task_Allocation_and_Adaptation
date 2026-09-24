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

PHASE A AND PHASE B LIVE SIDE BY SIDE HERE
-------------------------------------------
Sections 1-6 are the ACTOR-ONLY Phase-A path and are exactly what the approved Phase-A
baseline was measured on: ``EpisodeRecord`` / ``PPOBuffer`` /
``compute_returns_and_advantages`` / ``PPOUpdater``, with NO value loss and an
episode-mean baseline. Section 7 adds the Phase-B CTDE path — ``CTDEConfig``,
``CentralCritic`` (its OWN encoder + a :class:`ValueHead` off ``GraphEncoder.pool``),
``CTDEEpisodeRecord`` / ``CTDEBuffer``, ``compute_ctde_advantages`` (GAE over the
episode's GLOBAL decision sequence) and ``CTDEUpdater`` (separate actor and critic
optimizers).

The two are DISJOINT, not layered: a run selects one by ``training_mode``, and an
``actor_only`` run constructs nothing from section 7. Sections 1-6 were not refactored
to make room for it, so the Phase-A credit semantics cannot have moved. The historical
"PHASE-B SEAM" comments below mark where the critic was expected to join; section 7
records where it actually did.

OBSERVATIONAL CREDIT REPORTING (both updaters). ``update(source, credit_sink=None)``
hands an optional sink ONE :class:`CreditReport` after a productive update: the SAME
batch object the update consumed -- its returns / baseline or its ``V_old`` / GAE
advantages / TD residuals / targets, plus the normalized advantages -- never a
recomputation. The sink adds no forward, no GAE pass, no RNG draw and no gradient, and
nothing in this module reads anything back from it.

OBSERVATIONAL STEP INSTRUMENTATION (actor-only, opt-in). ``PPOUpdater.update`` also
takes ``step_group_ids`` / ``step_sink`` / ``step_contrast_ids``: every epoch's group
gradients, contrast gradient, pre- / post-clip ``.grad`` and the actual parameter
displacement of the ONE existing ``optimizer.step()``, handed to the sink as ONE
:class:`ActorStepReport`. It is the only addition to sections 1-6; with it off the
update runs exactly the historical operations, and with it on the learning semantics
(losses, backward, clip, step, RNG) are unchanged -- proven against the base updater by
``tests/test_graph_actor_step_diagnostics.py``, not asserted as byte-identity.

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
``sample_action`` share ONE distribution construction site (``_semantic_dist``), which
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
    TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
    Union,
)

import numpy as np
import torch

from ..action.graph_action import (
    SEMANTIC_ABORT_LEAF, _semantic_dist, build_action_mask, evaluate_action,
)
from ..agent.graph_encoder import GraphEncoder
from ..observation.central_graph_builder import (
    CENTRAL_AGENT_FEATURE_DIM,
    CENTRAL_EDGE_ATTR_DIM,
    CENTRAL_TASK_FEATURE_DIM,
)

if TYPE_CHECKING:  # Types only — keeps the runtime closure BLADE-free / solver-free.
    from ..observation.central_graph_builder import CentralGraphObservation
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
        record_positions / chain_ordinals: for transition ``i``, the index of its
            :class:`EpisodeRecord` in the batch and its position in that record's ego
            chain. IDENTITY ONLY, recorded in the same flattening loop so a report can
            join a credit value to its episode; the credit math never reads them.
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
    record_positions: List[int] = field(default_factory=list)
    chain_ordinals: List[int] = field(default_factory=list)


def _chain_returns(chain_len: int, episode_reward: float, gamma: float) -> List[float]:
    """Per-transition returns for ONE ego chain under a single TERMINAL reward.

    The episode's scalar ``R`` is realized at the END of the chain, so the return of
    the ``t``-th decision (0-based, chain length ``n``) is ``R * gamma**(n-1-t)``.

    At the Phase-A default ``gamma == 1.0`` this is exactly ``R`` for every element —
    which is why the per-ego grouping does not change any number today. The
    generalization is written out rather than hard-coded to ``R`` so that ``gamma``
    is a real, testable field and the chain structure is already load-bearing when
    Phase B introduces dense rewards.

    PHASE-B SEAM (historical note, now resolved): this was where GAE was expected to
    replace the plain return, consuming the same per-ego chain. Phase B in fact runs GAE
    over the episode's GLOBAL decision sequence rather than per ego — the centralized
    value is a statement about the TEAM state at each decision event, so bootstrapping
    along one ego's chain would bootstrap from a state that is not the successor of the
    one before it. That path is :func:`compute_ctde_advantages`, a SEPARATE function; it
    does not call this one, and this one is unchanged.
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
    record_positions: List[int] = []
    chain_ordinals: List[int] = []
    for rec_pos, rec in enumerate(records):
        for chain in rec.chains.values():
            transitions.extend(chain)
            returns_list.extend(
                _chain_returns(len(chain), rec.episode_reward, cfg.gamma)
            )
            record_positions.extend([rec_pos] * len(chain))
            chain_ordinals.extend(range(len(chain)))

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
        record_positions=record_positions,
        chain_ordinals=chain_ordinals,
    )


# =============================================================================
# 4b. The observational credit report (shared by both updaters)
# =============================================================================

@dataclass(frozen=True)
class CreditReport:
    """What ONE productive update used for credit -- handed to an optional sink.

    ``batch`` is THE object the update consumed (an :class:`AdvantageBatch` on
    ``actor_only``, a :class:`CTDEAdvantageBatch` on ``ctde``), not a copy and not a
    recomputation, so a report cannot describe credit the update did not use.
    ``records`` is the record list the batch's ``record_positions`` index into.
    Everything a reader needs is a numpy array, a python scalar or a stored
    ``Transition``; nothing here is a tensor, so no gradient can cross it.
    """

    training_mode: str
    records: Sequence[Any]
    batch: Any
    cfg: "PPOConfig"
    ctde_cfg: Optional["CTDEConfig"] = None


CreditSink = Callable[[CreditReport], None]


@dataclass(frozen=True)
class ActorGradientReport:
    """Epoch-0 actor-gradient decomposition of ONE productive CTDE update -- OBSERVATIONAL.

    Built by :meth:`CTDEUpdater.update` from the per-transition policy losses epoch 0
    ALREADY built, with ``torch.autograd.grad`` over the retained graph: no forward, no
    GAE pass, no RNG draw, no ``.grad`` write, no optimizer touch. The group ids are
    OPAQUE integers the caller supplied, index-aligned with ``batch.transitions``; this
    module attaches no meaning to them.

    Every gradient is a flat float64 numpy vector over ``CTDEUpdater.actor_parameters``
    (a parameter the loss does not reach contributes zeros). A group's policy-surrogate
    gradient is that of ``sum(policy_loss_i for i in group) / n_transitions`` -- the
    group's REAL share of the batch mean, never re-normalized by the group's own size --
    so the group gradients sum to ``total_policy_surrogate_grad``.
    ``total_actor_loss_grad`` is the gradient of the real epoch-0 ``actor_loss``
    (surrogate minus the entropy bonus). Nothing here is a tensor.

    CONTRAST (optional): given a ``(positive_id, negative_id)`` pair of the same opaque
    ids, ``contrast_value`` is ``mean P(ABORT)`` over the positive-id transitions minus
    ``mean P(ABORT)`` over the negative-id transitions -- the semantic ABORT-leaf
    probability of :func:`_semantic_dist`, built from the SAME epoch-0 logits the losses
    used -- and ``contrast_grad`` is its actor-parameter gradient. Both are ``None`` when
    no pair was given or either id has no transition in the batch.
    """

    records: Sequence[Any]
    batch: Any
    group_ids: Tuple[int, ...]
    group_counts: Dict[int, int]
    group_policy_surrogate_grads: Dict[int, np.ndarray]
    total_policy_surrogate_grad: np.ndarray
    total_actor_loss_grad: np.ndarray
    entropy_coeff: float
    epoch: int = 0
    contrast_ids: Optional[Tuple[int, int]] = None
    contrast_value: Optional[float] = None
    contrast_grad: Optional[np.ndarray] = None


GradientSink = Callable[[ActorGradientReport], None]


def _flat_actor_grad(
    loss: torch.Tensor, params: Sequence[torch.nn.Parameter],
) -> np.ndarray:
    """``d loss / d params`` as one flat float64 vector, WITHOUT touching ``.grad``.

    ``retain_graph=True`` keeps the graph alive for the real ``backward()`` that follows.
    """
    grads = torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)
    return np.concatenate([
        np.zeros(p.numel(), dtype=np.float64) if g is None
        else g.detach().reshape(-1).to(torch.float64).cpu().numpy()
        for g, p in zip(grads, params)
    ])


def _flat_tensors(
    tensors: Sequence[Optional[torch.Tensor]], params: Sequence[torch.nn.Parameter],
) -> np.ndarray:
    """Detached float64 copies of per-parameter tensors, concatenated in parameter order.

    ``None`` (an unreached parameter's ``.grad``) contributes zeros. Reading a tensor this
    way writes nothing back to it.
    """
    return np.concatenate([
        np.zeros(p.numel(), dtype=np.float64) if t is None
        else t.detach().reshape(-1).to(torch.float64).cpu().numpy()
        for t, p in zip(tensors, params)
    ])


@dataclass(frozen=True)
class ActorStepEpoch:
    """ONE epoch of an instrumented actor-only update -- OBSERVATIONAL, all numpy / float.

    Built by :meth:`PPOUpdater.update` around the epoch's ONE real ``backward()`` /
    clip / ``optimizer.step()``. Every vector is flat float64 over
    ``PPOUpdater.parameters`` in its own order:

      * ``group_policy_surrogate_grads[g]`` -- ``d(sum(policy_loss_i, i in g) / n)``, the
        group's REAL share of this epoch's mean clipped surrogate (``n`` = the full batch);
      * ``policy_surrogate_grad`` / ``total_loss_grad`` -- the gradients of this epoch's
        actual ``policy_loss`` and ``total_loss``, by ``autograd.grad`` before the backward;
      * ``backward_grad`` -- the ``.grad`` the real backward left, BEFORE clipping;
      * ``clipped_grad`` -- the ``.grad`` after the real clip, i.e. what Adam consumed;
      * ``delta_theta`` -- the parameters after the ONE real ``optimizer.step()`` minus
        the parameters before it.

    CONTRAST (only when the caller named two opaque contrast ids and both occur in the
    batch; otherwise every contrast field is ``None``): ``contrast_before`` is ``mean
    P(ABORT)`` over the positive-id transitions minus that over the negative-id ones --
    the semantic ABORT leaf of :func:`_semantic_dist` on THIS epoch's pre-step logits --
    with ``contrast_side_means_before = (positive mean, negative mean)`` and
    ``contrast_grad`` its gradient at the pre-step parameters. ``contrast_after`` /
    ``contrast_side_means_after`` re-read the SAME stored observations after the step,
    under ``torch.no_grad``.
    """

    epoch: int
    policy_loss: float
    total_loss: float
    entropy: float
    ratios: Tuple[float, ...]
    group_policy_surrogate_grads: Dict[int, np.ndarray]
    policy_surrogate_grad: np.ndarray
    total_loss_grad: np.ndarray
    backward_grad: np.ndarray
    clipped_grad: np.ndarray
    pre_clip_grad_norm: float
    delta_theta: np.ndarray
    contrast_before: Optional[float] = None
    contrast_side_means_before: Optional[Tuple[float, float]] = None
    contrast_grad: Optional[np.ndarray] = None
    contrast_after: Optional[float] = None
    contrast_side_means_after: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ActorStepReport:
    """Every epoch of ONE productive, instrumented actor-only update -- OBSERVATIONAL.

    ``batch`` / ``records`` are the objects the update consumed (as on
    :class:`CreditReport`); ``group_ids`` are the OPAQUE integers the caller supplied,
    index-aligned with ``batch.transitions``, and this module attaches no meaning to
    them. ``parameter_layout`` names every flat-vector slice: ``(name, shape, dtype)`` in
    ``PPOUpdater.parameters`` order. Nothing here is a tensor.
    """

    records: Sequence[Any]
    batch: Any
    group_ids: Tuple[int, ...]
    group_counts: Dict[int, int]
    contrast_ids: Optional[Tuple[int, int]]
    contrast_counts: Optional[Tuple[int, int]]
    parameter_layout: Tuple[Tuple[str, Tuple[int, ...], str], ...]
    max_grad_norm: float
    clip_ratio: float
    entropy_coeff: float
    epochs: Tuple[ActorStepEpoch, ...]


StepSink = Callable[[ActorStepReport], None]


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
    def update(
        self,
        source: RecordSource,
        credit_sink: Optional[CreditSink] = None,
        step_group_ids: Optional[Sequence[int]] = None,
        step_sink: Optional[StepSink] = None,
        step_contrast_ids: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Run ``cfg.n_epochs`` clipped-PPO epochs over ``source`` and step the optimizer.

        ``credit_sink``, when given, receives ONE :class:`CreditReport` after the last
        epoch of a PRODUCTIVE update (``n_epochs_run > 0``) and nothing otherwise. It is
        observational: it is handed the already-computed batch, and nothing in this
        method reads anything back from it.

        ``step_group_ids`` / ``step_sink`` (OBSERVATIONAL, both or neither; OFF by
        default): one OPAQUE integer per transition, in the batch's transition order
        (records in order, each record's ``transitions()``). In EVERY epoch, after the
        per-transition losses and ``total_loss`` exist and BEFORE the real backward, the
        updater differentiates each id's summed policy loss over the FULL batch size and
        both totals with ``torch.autograd.grad`` on the retained graph; around the real
        backward / clip / step it reads (never writes) the pre- and post-clip ``.grad``
        and the parameters, and after a productive update hands the sink ONE
        :class:`ActorStepReport`. ``step_contrast_ids`` (optional, requires the two
        above) names a ``(positive, negative)`` id pair: when both occur, each epoch
        also forms ``mean P(ABORT | positive) - mean P(ABORT | negative)`` from that
        epoch's own logits, its gradient, and its value after the step from a
        ``torch.no_grad`` re-read of the same stored observations (the encoder and head
        hold no dropout, no RNG and no mutable buffer, and nothing changes train / eval
        mode). The ids never reach an advantage, a loss weight, the clip or the step.

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

        PHASE-B SEAM (historical note, now resolved): the critic's value loss was
        expected to join the per-transition loss here, reusing this forward's embedding.
        Phase B deliberately did NOT do that. The critic reads a CENTRAL state — all
        live agents and targets — which this actor forward does not produce and must
        never see, so sharing the embedding would have shared the encoder and made
        "privileged information never reaches the actor's weights" unprovable. The
        critic therefore owns a separate encoder and a separate optimizer in
        :class:`CTDEUpdater`, and THIS updater is unchanged: actor-only, no value loss.

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

        if (step_group_ids is None) != (step_sink is None):
            raise ValueError("step_group_ids and step_sink must be given together")
        step_ids: Optional[Tuple[int, ...]] = None
        if step_group_ids is not None:
            step_ids = tuple(int(g) for g in step_group_ids)
            if len(step_ids) != batch.n_transitions:
                raise ValueError(
                    "step_group_ids has %d entries but the batch has %d transitions"
                    % (len(step_ids), batch.n_transitions))
        step_contrast: Optional[Tuple[int, int]] = None
        if step_contrast_ids is not None:
            if step_ids is None:
                raise ValueError("step_contrast_ids requires step_group_ids")
            if len(step_contrast_ids) != 2:
                raise ValueError("step_contrast_ids must be two distinct ids")
            step_contrast = (int(step_contrast_ids[0]), int(step_contrast_ids[1]))
            if step_contrast[0] == step_contrast[1]:
                raise ValueError("step_contrast_ids must be two distinct ids")

        if batch.n_transitions == 0:
            # Clean no-op (documented above): nothing to learn from.
            for key in ("policy_loss", "total_loss", "entropy", "mean_ratio",
                        "clip_fraction", "approx_kl", "max_ratio_dev", "grad_norm"):
                diagnostics[key] = 0.0
            return diagnostics

        advantages = batch.advantages
        n = batch.n_transitions
        step_epochs: List[ActorStepEpoch] = []
        step_members: Dict[int, List[int]] = {}
        contrast_rows: Tuple[List[int], List[int]] = ([], [])
        if step_ids is not None:
            for i, g in enumerate(step_ids):
                step_members.setdefault(g, []).append(i)
            if step_contrast is not None:
                for side, cid in enumerate(step_contrast):
                    contrast_rows[side].extend(step_members.get(cid, []))
        contrast_on = bool(contrast_rows[0]) and bool(contrast_rows[1])

        for _epoch in range(cfg.n_epochs):
            policy_losses: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []
            ratios_detached: List[float] = []
            kl_terms: List[float] = []
            # (row -> this epoch's grad-attached logits, mask) -- kept only when requested.
            contrast_inputs: Dict[int, Tuple[torch.Tensor, np.ndarray]] = {}

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
                if contrast_on:
                    contrast_inputs[i] = (logits, mask)

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

            # OBSERVATIONAL step instrumentation, part 1: gradients over the graph built
            # above. `autograd.grad` writes no `.grad` and retains the graph, so the real
            # backward below runs exactly as it would without this block.
            if step_ids is not None:
                step_group_grads = {
                    g: _flat_actor_grad(
                        torch.stack([policy_losses[i] for i in idx]).sum() / n,
                        self.parameters)
                    for g, idx in step_members.items()
                }
                step_policy_grad = _flat_actor_grad(policy_loss, self.parameters)
                step_total_grad = _flat_actor_grad(total_loss, self.parameters)
                step_contrast_grad: Optional[np.ndarray] = None
                step_before: Optional[Tuple[float, float, float]] = None
                if contrast_on:
                    contrast_t, means_t = self._abort_contrast(
                        contrast_inputs, contrast_rows)
                    step_before = (float(contrast_t.detach().item()),) + means_t
                    step_contrast_grad = _flat_actor_grad(contrast_t, self.parameters)
                contrast_inputs.clear()
                theta_before = _flat_tensors(list(self.parameters), self.parameters)

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if step_ids is not None:
                step_backward_grad = _flat_tensors(
                    [p.grad for p in self.parameters], self.parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, cfg.max_grad_norm
            )
            if step_ids is not None:
                step_clipped_grad = _flat_tensors(
                    [p.grad for p in self.parameters], self.parameters)
            self.optimizer.step()

            # OBSERVATIONAL step instrumentation, part 2: the actual displacement of the
            # ONE step above, and the contrast re-read on the same stored observations.
            if step_ids is not None:
                step_after: Optional[Tuple[float, float, float]] = None
                if contrast_on:
                    step_after = self._abort_contrast_now(batch, contrast_rows)
                step_epochs.append(ActorStepEpoch(
                    epoch=int(_epoch),
                    policy_loss=float(policy_loss.detach().item()),
                    total_loss=float(total_loss.detach().item()),
                    entropy=float(entropy_mean.detach().item()),
                    ratios=tuple(ratios_detached),
                    group_policy_surrogate_grads=step_group_grads,
                    policy_surrogate_grad=step_policy_grad,
                    total_loss_grad=step_total_grad,
                    backward_grad=step_backward_grad,
                    clipped_grad=step_clipped_grad,
                    pre_clip_grad_norm=float(grad_norm),
                    delta_theta=(_flat_tensors(list(self.parameters), self.parameters)
                                 - theta_before),
                    contrast_before=None if step_before is None else step_before[0],
                    contrast_side_means_before=(
                        None if step_before is None else step_before[1:]),
                    contrast_grad=step_contrast_grad,
                    contrast_after=None if step_after is None else step_after[0],
                    contrast_side_means_after=(
                        None if step_after is None else step_after[1:]),
                ))

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
        if credit_sink is not None and cfg.n_epochs > 0:
            credit_sink(CreditReport(
                training_mode="actor_only", records=records, batch=batch, cfg=cfg,
            ))
        if step_sink is not None and cfg.n_epochs > 0:
            step_sink(ActorStepReport(
                records=records,
                batch=batch,
                group_ids=step_ids,
                group_counts={g: len(idx) for g, idx in step_members.items()},
                contrast_ids=step_contrast,
                contrast_counts=(None if step_contrast is None
                                 else (len(contrast_rows[0]), len(contrast_rows[1]))),
                parameter_layout=self._parameter_layout(),
                max_grad_norm=float(cfg.max_grad_norm),
                clip_ratio=float(cfg.clip_ratio),
                entropy_coeff=float(cfg.entropy_coeff),
                epochs=tuple(step_epochs),
            ))
        return diagnostics

    # ------------------------------------------------------------------
    # Observational helpers of the opt-in step instrumentation (never on the OFF path).
    @staticmethod
    def _abort_contrast(
        inputs: Mapping[int, Tuple[torch.Tensor, np.ndarray]],
        rows: Tuple[Sequence[int], Sequence[int]],
    ) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """``mean P(ABORT | rows[0]) - mean P(ABORT | rows[1])`` from given logits.

        The semantic ABORT leaf of the same masked ``_semantic_dist`` the losses use;
        each mean is over its own rows, whichever action was stored.
        """
        means = []
        for side in rows:
            probs = [_semantic_dist(inputs[i][0], inputs[i][1])[1].probs[SEMANTIC_ABORT_LEAF]
                     for i in side]
            means.append(torch.stack(probs).mean())
        return means[0] - means[1], (float(means[0].detach().item()),
                                     float(means[1].detach().item()))

    def _abort_contrast_now(
        self, batch: AdvantageBatch, rows: Tuple[Sequence[int], Sequence[int]],
    ) -> Tuple[float, float, float]:
        """The contrast at the CURRENT parameters, re-read under ``torch.no_grad``."""
        with torch.no_grad():
            inputs = {i: (self._forward_logits(batch.transitions[i].gobs),
                          build_action_mask(batch.transitions[i].gobs))
                      for side in rows for i in side}
            value, means = self._abort_contrast(inputs, rows)
        return (float(value.item()),) + means

    def _parameter_layout(self) -> Tuple[Tuple[str, Tuple[int, ...], str], ...]:
        """``(name, shape, dtype)`` for every entry of :attr:`parameters`, in order."""
        named = ([("encoder." + k, p) for k, p in self.policy.encoder.named_parameters()]
                 + [("head." + k, p) for k, p in self.policy.head.named_parameters()])
        if len(named) != len(self.parameters) or any(
                p is not q for (_, p), q in zip(named, self.parameters)):
            raise RuntimeError("the named parameters do not match the optimized parameters")
        return tuple((name, tuple(int(s) for s in p.shape), str(p.dtype))
                     for name, p in named)


# =============================================================================
# 7. PHASE B (CTDE) -- the centralized critic, its credit math and its updater
# =============================================================================
#
# Everything from here down is TRAINING-ONLY and is reached ONLY when the run's
# ``training_mode`` is ``ctde``. Sections 1-6 above are the actor-only Phase-A path:
# ``EpisodeRecord`` / ``PPOBuffer`` / ``compute_returns_and_advantages`` / ``PPOUpdater``
# keep their exact learning semantics (the one later addition, the opt-in observational
# step instrumentation of ``PPOUpdater.update``, changes no number it trains on), and an
# ``actor_only`` run never constructs a single object defined below. That separation is
# deliberate -- emulating actor-only by running the CTDE path with ``value_coeff = 0``
# would still build a critic, still sample central states, still replace the
# episode-mean baseline with a learned one, and would therefore NOT be the Phase-A
# baseline the approved measurement was taken on.
#
# WHAT IS CENTRALIZED, AND WHAT IS NOT
# ------------------------------------
# CENTRALIZED (training only): the value estimate ``V(s)``, computed from the global
# ``CentralGraphObservation`` -- all live targets, all live agents with their real fuel,
# all-agent sensing, and the current global executor plans.
#
# DECENTRALIZED (unchanged, always): every action. The actor still consumes ONLY its own
# private ``GraphObservation`` through the SAME encoder + head + mask + sampling path,
# and this module never feeds it anything else. The critic exists to reduce the variance
# of the actor's advantage; it never appears in the acting path, in evaluation, or at
# inference. A CTDE-trained actor is runnable with the critic object absent.
#
# ACTOR AND CRITIC SHARE NOTHING
# ------------------------------
# The critic owns its OWN :class:`~match_aou.rl.agent.graph_encoder.GraphEncoder`
# instance and its own optimizer. There is no shared encoder, no shared parameter
# object and no shared optimizer, so "did privileged information reach the actor's
# weights?" has a structural answer: the actor's parameter set and the critic's are
# disjoint, the two losses are backpropagated SEPARATELY, and the advantage the actor
# consumes is a detached python float.

def _ctde_layer_init(
    layer: torch.nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> torch.nn.Linear:
    """Orthogonal init for a linear layer -- the standard PPO scheme.

    Re-defined LOCALLY, the same way ``graph_encoder`` and ``graph_action`` each carry
    their own byte-identical copy, so this module does not acquire a dependency on
    either of them for two lines of initialization.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass(frozen=True)
class CTDEConfig:
    """Phase-B CTDE hyper-parameters. Frozen, like :class:`PPOConfig`.

    ``gamma`` is deliberately NOT duplicated here -- CTDE reuses ``PPOConfig.gamma``, so
    a run has exactly one discount factor and the two configs cannot disagree about it.

    Attributes:
        critic_lr: Adam learning rate for the critic (its own encoder + value head).
            Separate from ``PPOConfig.lr`` because the critic is a regression problem
            and the actor is not; sharing one rate would make them tune together for no
            reason.
        value_coeff: weight on the critic's MSE value loss. It scales the CRITIC's own
            loss only -- it is not a knob that turns CTDE off (see the section header),
            and ``TrainConfig.validate`` REFUSES ``0`` under ``training_mode='ctde'``:
            a zero coefficient would leave the critic untrained while its advantages
            were still driving the actor, which is neither training mode.
        gae_lambda: the GAE trace-decay ``lambda``. 1.0 would be the plain Monte-Carlo
            advantage (maximum variance, zero bias); 0 would be the one-step TD residual
            (minimum variance, maximum bias). 0.95 is the canonical middle.
    """

    critic_lr: float = 3e-4
    value_coeff: float = 0.5
    gae_lambda: float = 0.95


# -----------------------------------------------------------------------------
# 7a. The critic network
# -----------------------------------------------------------------------------

class ValueHead(torch.nn.Module):
    """Scalar state-value head over a POOLED graph summary.

    Consumes ``GraphEncoder.pool(...)`` -- the ``[embed_dim]`` mean over ALL node
    embeddings, which is the size-agnostic hook the encoder has carried since Phase A
    for exactly this purpose. Pooling is what makes the value estimator native to a
    varying number of targets and agents with no padding: the graph can shrink as
    targets are destroyed and aircraft are lost, and the head's input width never
    changes.

    INITIALIZATION, stated as the code has it: both layers are orthogonally initialized
    through :func:`_ctde_layer_init` with zero bias -- the hidden layer at its default
    gain ``sqrt(2)``, and the OUTPUT layer at ``std=1.0``, the conventional gain for a
    value head (an actor's policy head uses a deliberately small ``0.01`` instead, to
    start near-uniform; a value head does not).

    So the UNTRAINED critic is an arbitrary small-magnitude function of the state, NOT
    zero everywhere. Nothing here relies on it being zero: the advantages the actor
    consumes are normalized across the whole batch in
    :func:`compute_ctde_advantages`, which subtracts the batch mean, so an offset the
    untrained critic applies uniformly across a batch cancels there rather than here.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mlp = torch.nn.Sequential(
            _ctde_layer_init(torch.nn.Linear(embed_dim, hidden_dim)),
            torch.nn.Tanh(),
            _ctde_layer_init(torch.nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Map a ``[embed_dim]`` pooled summary to a SCALAR value (shape ``[]``)."""
        return self.mlp(pooled).squeeze(-1)


class CentralCritic(torch.nn.Module):
    """The centralized value estimator: its OWN encoder + a :class:`ValueHead`.

    ``V(s) = value_head(critic_encoder.pool(central_obs, edge_attr))``.

    THE ENCODER IS A DISTINCT INSTANCE. It is constructed here, from the same
    ``GraphEncoder`` CLASS the actor uses, with the CENTRAL feature widths
    (``task_feat_dim = 2``, ``agent_feat_dim = 1``, ``edge_attr_dim = 5``) -- all three
    already constructor parameters, which is why no encoder change was needed. Its
    parameters are disjoint from the actor's; nothing is shared, tied or copied.

    THE STATE IS DECISION-CONDITIONED BY ROLE ONLY. At a decision capture
    ``CentralGraphObservation.ego_index`` is the node of the live agent that owns the
    current decision, so the encoder's EXISTING role mechanism marks that node EGO and
    every other live agent PEER: the critic values ``V(s, acting_ego)``. No
    agent-identity or agent-order feature is invented, the architecture and pooling are
    unchanged, and the encoder's role semantics are untouched.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            embed_dim=embed_dim,
            task_feat_dim=CENTRAL_TASK_FEATURE_DIM,
            agent_feat_dim=CENTRAL_AGENT_FEATURE_DIM,
            edge_attr_dim=CENTRAL_EDGE_ATTR_DIM,
            **encoder_kwargs,
        )
        self.value_head = ValueHead(self.encoder.embed_dim, hidden_dim)

    def forward(self, central_obs: "CentralGraphObservation") -> torch.Tensor:
        """Estimate ``V(s)`` for one central state; returns a SCALAR tensor.

        A state with NO nodes at all (every target destroyed AND every agent lost)
        would make the pooling mean over an empty set, i.e. NaN. It cannot arise at a
        capture point -- a decision requires an airborne ego, so there is always at
        least one agent node -- but the guard is here anyway so the critic's output is
        finite by construction rather than by an argument about the caller.
        """
        n_nodes = int(central_obs.task_features.shape[0]) + int(
            central_obs.agent_features.shape[0]
        )
        if n_nodes == 0:
            return torch.zeros((), dtype=torch.float32)
        pooled = self.encoder.pool(central_obs, edge_attr=central_obs.edge_attr)
        return self.value_head(pooled)


def build_central_critic(embed_dim: int = 64, **kwargs: Any) -> CentralCritic:
    """Construct the run's ONE :class:`CentralCritic` (call once, like ``build_policy``).

    Kept as a function so the CTDE side mirrors ``graph_tick_loop.build_policy``: the
    critic is training state and must live across episodes, exactly like the actor and
    its Adam moments.
    """
    return CentralCritic(embed_dim=embed_dim, **kwargs)


# -----------------------------------------------------------------------------
# 7b. CTDE episode record + buffer
# -----------------------------------------------------------------------------

@dataclass
class CTDEEpisodeRecord:
    """One finished episode as CTDE consumes it: the GLOBAL decision sequence.

    DELIBERATELY NOT PER-EGO. :class:`EpisodeRecord` groups transitions by ego because
    the Phase-A credit structure is per-ego and gamma is dormant there. CTDE's value
    function is a statement about the TEAM state at each decision event, so its GAE runs
    over the episode's single ordered decision sequence -- the order
    ``EpisodeResult.trajectory`` already has, which is the order the decisions actually
    happened in. Re-grouping it per ego would ask the critic to bootstrap from a state
    that is not the successor of the one before it.

    Attributes:
        transitions: the episode's wakes in GLOBAL order (``EpisodeResult.trajectory``).
        central_states: the central state captured immediately BEFORE each of those
            decisions, index-for-index. The 1:1 alignment is validated on construction,
            not assumed.
        episode_reward: the episode's scalar terminal ``R`` from ``graph_reward``.
        seed / episode_index: provenance, as on :class:`EpisodeRecord`.
    """

    transitions: List["Transition"] = field(default_factory=list)
    central_states: List["CentralGraphObservation"] = field(default_factory=list)
    episode_reward: float = 0.0
    seed: int = 0
    episode_index: int = 0

    def __post_init__(self) -> None:
        if len(self.transitions) != len(self.central_states):
            raise ValueError(
                "CTDE alignment broken: %d transition(s) but %d central state(s). "
                "Exactly one central state is captured per actor decision."
                % (len(self.transitions), len(self.central_states))
            )

    @classmethod
    def from_episode(
        cls,
        trajectory: Sequence["Transition"],
        central_states: Sequence["CentralGraphObservation"],
        episode_reward: float,
        *,
        seed: int = 0,
        episode_index: int = 0,
    ) -> "CTDEEpisodeRecord":
        """Build a record from a finished episode's trajectory + its central samples."""
        return cls(
            transitions=list(trajectory),
            central_states=list(central_states),
            episode_reward=float(episode_reward),
            seed=int(seed),
            episode_index=int(episode_index),
        )

    @property
    def n_transitions(self) -> int:
        """Number of decisions in this episode (0 for a zero-wake episode)."""
        return len(self.transitions)

    @property
    def has_wakes(self) -> bool:
        """True iff this episode produced at least one decision."""
        return bool(self.transitions)


class CTDEBuffer:
    """Accumulates :class:`CTDEEpisodeRecord`s for ONE CTDE iteration, then is cleared.

    The CTDE sibling of :class:`PPOBuffer`, kept separate rather than bolted onto it so
    that the actor-only buffer stays exactly what the Phase-A contract describes. Like
    its sibling it owns storage only -- the credit math lives in
    :func:`compute_ctde_advantages`.
    """

    def __init__(self) -> None:
        self.records: List[CTDEEpisodeRecord] = []

    def add(self, record: CTDEEpisodeRecord) -> None:
        """Append one finished episode's record (zero-wake records included)."""
        self.records.append(record)

    def clear(self) -> None:
        """Drop every record -- call after each update so iterations stay on-policy."""
        self.records.clear()

    @property
    def n_episodes(self) -> int:
        """Number of episodes recorded, INCLUDING zero-wake ones."""
        return len(self.records)

    @property
    def n_transitions(self) -> int:
        """Total number of decisions across every recorded episode."""
        return sum(rec.n_transitions for rec in self.records)

    @property
    def episodes_with_wakes(self) -> int:
        """How many recorded episodes produced at least one decision (diagnostic)."""
        return sum(1 for rec in self.records if rec.has_wakes)

    def __len__(self) -> int:
        return len(self.records)


CTDERecordSource = Union[CTDEBuffer, Sequence[CTDEEpisodeRecord]]


def _as_ctde_records(source: CTDERecordSource) -> List[CTDEEpisodeRecord]:
    """Normalize a CTDE buffer OR a bare record sequence to a list of records."""
    if isinstance(source, CTDEBuffer):
        return list(source.records)
    return list(source)


# -----------------------------------------------------------------------------
# 7c. CTDE credit assignment -- GAE over the global decision sequence
# -----------------------------------------------------------------------------

@dataclass
class CTDEAdvantageBatch:
    """The flattened, aligned view :class:`CTDEUpdater` consumes.

    ``transitions[i]`` / ``central_states[i]`` / ``values[i]`` / ``advantages[i]`` /
    ``value_targets[i]`` all describe the SAME decision -- the alignment is the
    contract, exactly as on :class:`AdvantageBatch`.

    Attributes:
        transitions: the batch's decisions, episode by episode, in global order.
        central_states: their central states, index-for-index.
        values: ``V_old`` -- the critic's estimate BEFORE any update epoch ran. Fixed
            for the whole update, which is what makes the PPO epochs off-policy-safe on
            the critic side too.
        advantages: NORMALIZED GAE advantages -- what the ACTOR consumes.
        raw_advantages: pre-normalization GAE advantages, kept for logging.
        value_targets: ``A_t + V_old[t]`` -- the DETACHED regression target the critic
            is fitted to across all epochs.
        episode_reward_mean: mean ``episode_reward`` over the batch's EPISODES,
            zero-wake episodes included. It is NOT used in the CTDE credit math at all
            (the critic is the baseline here) -- it exists so the CTDE diagnostics can
            report the same REWARD quantity the actor-only path reports under
            ``baseline``. See :meth:`CTDEUpdater.update` for why that matters.
        adv_mean_raw / adv_std_raw: moments of ``raw_advantages`` (logging).
        n_episodes / n_transitions: batch shape.
        rewards: the realized ``r_t`` the GAE pass consumed, per decision.
        td_residuals: the one-step ``delta_t = r_t + gamma*V_next[t] - V_old[t]`` from
            the SAME GAE pass that produced ``raw_advantages`` (no second critic call).
        record_positions / decision_ordinals: for decision ``i``, the index of its
            :class:`CTDEEpisodeRecord` and its position in that episode's GLOBAL
            decision sequence. IDENTITY ONLY; the credit math never reads them.
    """

    transitions: List["Transition"]
    central_states: List["CentralGraphObservation"]
    values: np.ndarray
    advantages: np.ndarray
    raw_advantages: np.ndarray
    value_targets: np.ndarray
    episode_reward_mean: float
    adv_mean_raw: float
    adv_std_raw: float
    n_episodes: int
    n_transitions: int
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    td_residuals: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    record_positions: List[int] = field(default_factory=list)
    decision_ordinals: List[int] = field(default_factory=list)


def episode_rewards_sequence(record: CTDEEpisodeRecord) -> List[float]:
    """The REALIZED per-decision rewards ``r_t`` of one episode, in global order.

    ``graph_reward.compute_episode_reward`` is terminal: it writes the episode's scalar
    ``R`` onto the LAST transition of the trajectory and leaves every earlier one at
    ``0.0``. This function reads those realized values off the transitions rather than
    reconstructing them, so the CTDE credit math consumes exactly what the (unchanged)
    reward layer produced. A transition whose reward was never filled in is read as
    ``0.0`` -- the same "no reward realized here" the reward layer means by it.
    """
    return [float(getattr(tr, "reward", 0.0) or 0.0) for tr in record.transitions]


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Generalized Advantage Estimation over ONE episode's decision sequence.

    The whole of CTDE's credit assignment, factored out so it is hand-checkable on
    known numbers instead of inferred from an aggregate.

    For ``N`` decisions, with ``V_old[t]`` the pre-update critic estimate of decision
    ``t``'s central state::

        V_next[t]  = V_old[t+1]   for t < N-1
        V_next[N-1] = 0           # the final decision bootstraps from NOTHING
        delta_t     = r_t + gamma * V_next[t] - V_old[t]
        A_t         = delta_t + gamma * gae_lambda * A_{t+1}      (A_N = 0)
        target_t    = A_t + V_old[t]

    ``V_next`` of the LAST decision is 0 rather than a bootstrapped estimate because the
    episode genuinely ends there: there is no successor state, and the terminal reward
    ``R`` is the realized outcome of the whole episode.

    Args:
        rewards: realized ``r_t`` per decision (all 0 but the last, under the current
            terminal reward).
        values: ``V_old[t]`` per decision, same length as ``rewards``.
        gamma: the discount, taken from ``PPOConfig.gamma`` (1.0 today).
        gae_lambda: the GAE trace decay, from :attr:`CTDEConfig.gae_lambda`.

    Returns:
        ``(advantages, value_targets)``, both length ``N`` and index-aligned with the
        inputs. Both are empty for an empty episode. The one-step residuals of the same
        pass are available from :func:`_gae_pass`.
    """
    advantages, targets, _deltas = _gae_pass(
        rewards, values, gamma=gamma, gae_lambda=gae_lambda)
    return advantages, targets


def _gae_pass(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float], List[float]]:
    """THE single GAE pass: ``(advantages, value_targets, td_residuals)``.

    :func:`compute_gae` returns the first two; :func:`compute_ctde_advantages` keeps
    all three, so the reported ``delta_t`` is the residual this very loop accumulated.
    """
    n = len(rewards)
    if n != len(values):
        raise ValueError(
            "GAE alignment broken: %d reward(s) vs %d value(s)" % (n, len(values))
        )
    if n == 0:
        return [], [], []

    advantages = [0.0] * n
    deltas = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        # The FINAL decision has no successor state -> V_next = 0 (see compute_gae).
        v_next = float(values[t + 1]) if t + 1 < n else 0.0
        delta = float(rewards[t]) + gamma * v_next - float(values[t])
        running = delta + gamma * gae_lambda * running
        advantages[t] = running
        deltas[t] = delta
    targets = [advantages[t] + float(values[t]) for t in range(n)]
    return advantages, targets, deltas


def compute_ctde_advantages(
    source: CTDERecordSource,
    critic: CentralCritic,
    cfg: PPOConfig = PPOConfig(),
    ctde_cfg: CTDEConfig = CTDEConfig(),
) -> CTDEAdvantageBatch:
    """Evaluate ``V_old``, run GAE per episode, and normalize across the whole batch.

    THE CTDE REPLACEMENT for :func:`compute_returns_and_advantages`. It does NOT call
    that function, does not touch it, and is never called on an ``actor_only`` run.

    Three properties are load-bearing:

    1. **``V_old`` is computed ONCE, under ``torch.no_grad``, before any update epoch.**
       Every epoch of the update then regresses the critic toward targets derived from
       those fixed values, so the target cannot chase the network that is fitting it.
    2. **GAE runs PER EPISODE over the GLOBAL decision sequence** -- never regrouped per
       ego, and never across an episode boundary (each episode's last decision
       bootstraps from 0).
    3. **Advantages are normalized across ALL decision samples of the batch**, with the
       same ``adv_norm_eps`` guard the actor-only path uses, so a degenerate batch
       yields ~0 advantages rather than NaN.

    Args:
        source: a :class:`CTDEBuffer` or a bare sequence of :class:`CTDEEpisodeRecord`.
        critic: the run's :class:`CentralCritic`.
        cfg: supplies ``gamma`` and ``adv_norm_eps``.
        ctde_cfg: supplies ``gae_lambda``.

    Returns:
        A :class:`CTDEAdvantageBatch` whose arrays are index-aligned with its
        ``transitions`` / ``central_states``. An empty batch (no episode woke) returns
        empty arrays -- a normal operating condition, not an error.
    """
    records = _as_ctde_records(source)
    n_episodes = len(records)
    # The same REWARD quantity the actor-only path reports as its baseline: the mean
    # over the batch's EPISODES, zero-wake episodes included. CTDE does not USE it for
    # credit (the critic is the baseline), but the run record must keep reporting a
    # reward under a reward-named key -- see `CTDEUpdater.update`.
    episode_reward_mean = (
        float(np.mean([rec.episode_reward for rec in records])) if records else 0.0
    )

    transitions: List["Transition"] = []
    central_states: List["CentralGraphObservation"] = []
    values_list: List[float] = []
    adv_list: List[float] = []
    target_list: List[float] = []
    reward_list: List[float] = []
    delta_list: List[float] = []
    record_positions: List[int] = []
    decision_ordinals: List[int] = []

    was_training = critic.training
    critic.eval()
    try:
        with torch.no_grad():
            for rec_pos, rec in enumerate(records):
                if not rec.transitions:
                    # A zero-wake episode contributes NOTHING to a CTDE update: no actor
                    # sample, no critic sample, no baseline mass. It is still a valid
                    # scientific episode outcome and is never a failure.
                    continue
                ep_values = [
                    float(critic(state).item()) for state in rec.central_states
                ]
                ep_rewards = episode_rewards_sequence(rec)
                ep_adv, ep_targets, ep_deltas = _gae_pass(
                    ep_rewards,
                    ep_values,
                    gamma=cfg.gamma,
                    gae_lambda=ctde_cfg.gae_lambda,
                )
                transitions.extend(rec.transitions)
                central_states.extend(rec.central_states)
                values_list.extend(ep_values)
                adv_list.extend(ep_adv)
                target_list.extend(ep_targets)
                reward_list.extend(ep_rewards)
                delta_list.extend(ep_deltas)
                record_positions.extend([rec_pos] * len(rec.transitions))
                decision_ordinals.extend(range(len(rec.transitions)))
    finally:
        critic.train(was_training)

    if not transitions:
        empty = np.zeros(0, dtype=np.float64)
        return CTDEAdvantageBatch(
            transitions=[],
            central_states=[],
            values=empty,
            advantages=empty.copy(),
            raw_advantages=empty.copy(),
            value_targets=empty.copy(),
            episode_reward_mean=episode_reward_mean,
            adv_mean_raw=0.0,
            adv_std_raw=0.0,
            n_episodes=n_episodes,
            n_transitions=0,
        )

    raw_adv = np.asarray(adv_list, dtype=np.float64)
    adv_mean = float(raw_adv.mean())
    adv_std = float(raw_adv.std())  # population std (ddof=0), the PPO convention
    # Same eps guard as the actor-only path: a zero-variance batch gives 0, never NaN.
    advantages = (raw_adv - adv_mean) / (adv_std + cfg.adv_norm_eps)

    return CTDEAdvantageBatch(
        transitions=transitions,
        central_states=central_states,
        values=np.asarray(values_list, dtype=np.float64),
        advantages=advantages,
        raw_advantages=raw_adv,
        value_targets=np.asarray(target_list, dtype=np.float64),
        episode_reward_mean=episode_reward_mean,
        adv_mean_raw=adv_mean,
        adv_std_raw=adv_std,
        n_episodes=n_episodes,
        n_transitions=len(transitions),
        rewards=np.asarray(reward_list, dtype=np.float64),
        td_residuals=np.asarray(delta_list, dtype=np.float64),
        record_positions=record_positions,
        decision_ordinals=decision_ordinals,
    )


# -----------------------------------------------------------------------------
# 7d. The CTDE updater
# -----------------------------------------------------------------------------

class CTDEUpdater:
    """Clipped-PPO actor update + centralized value regression, on SEPARATE optimizers.

    The CTDE sibling of :class:`PPOUpdater`. It is a separate class on purpose: the
    actor-only updater is part of the locked Phase-A contract and its learning semantics
    are left unchanged, so an ``actor_only`` run cannot be affected by anything here. The actor half of the
    loss is identical in form to the actor-only one and reuses the SAME factored
    :func:`clipped_surrogate`, the SAME rebuilt-never-stored mask and the SAME
    ``evaluate_action`` distribution site, so the ratio semantics do not fork.

    GRADIENT ISOLATION IS STRUCTURAL, AND IT IS ALSO EXPLICIT:

      * the two parameter sets are DISJOINT (no shared encoder, no tied weights);
      * the actor loss and the value loss are backpropagated in TWO SEPARATE
        ``backward()`` calls onto two separate graphs, each followed by its own
        grad-norm clip and its own ``optimizer.step()``. So "the actor's backward
        produced no critic gradient" is true by observation, not only by argument;
      * the advantage the actor consumes is a plain python float taken from a numpy
        array, so no gradient can flow from the actor's loss into the critic through it.
    """

    def __init__(
        self,
        policy: "Policy",
        critic: CentralCritic,
        cfg: PPOConfig = PPOConfig(),
        ctde_cfg: CTDEConfig = CTDEConfig(),
    ) -> None:
        """Bind the updater to an actor policy + a critic and build BOTH optimizers.

        Args:
            policy: the actor's encoder + head bundle. Its parameters go into the ACTOR
                optimizer and nothing else.
            critic: the :class:`CentralCritic`. Its encoder + value head go into the
                CRITIC optimizer and nothing else.
            cfg: the frozen :class:`PPOConfig` (actor lr, clipping, epochs, gamma).
            ctde_cfg: the frozen :class:`CTDEConfig` (critic lr, value coeff, lambda).
        """
        self.policy = policy
        self.critic = critic
        self.cfg = cfg
        self.ctde_cfg = ctde_cfg

        self.actor_parameters: List[torch.nn.Parameter] = (
            list(policy.encoder.parameters()) + list(policy.head.parameters())
        )
        self.critic_parameters: List[torch.nn.Parameter] = list(critic.parameters())
        self.optimizer = torch.optim.Adam(self.actor_parameters, lr=cfg.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=ctde_cfg.critic_lr
        )

    # ------------------------------------------------------------------
    def _forward_logits(self, gobs: Any) -> torch.Tensor:
        """Re-run the ACTOR's encoder -> head on a stored PRIVATE observation, WITH grad.

        ``gobs`` is the ego's own ``GraphObservation`` and nothing else -- the central
        state is never passed here, and the actor's forward has no access to it.
        """
        emb = self.policy.encoder(gobs)
        return self.policy.head(emb)

    # ------------------------------------------------------------------
    def update(
        self,
        source: CTDERecordSource,
        credit_sink: Optional[CreditSink] = None,
        gradient_group_ids: Optional[Sequence[int]] = None,
        gradient_sink: Optional[GradientSink] = None,
        gradient_contrast_ids: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Run ``cfg.n_epochs`` CTDE epochs over ``source`` and step BOTH optimizers.

        ``credit_sink``: as on :meth:`PPOUpdater.update` -- ONE :class:`CreditReport`
        carrying the pre-update ``V_old`` / GAE batch, after a productive update only.

        ``gradient_group_ids`` / ``gradient_sink`` (OBSERVATIONAL, both or neither): one
        OPAQUE integer per transition, in the batch's transition order (the records'
        transitions concatenated in record order). At epoch 0, after the per-transition
        losses exist and BEFORE the real ``backward()``, the updater differentiates each
        id's summed policy loss (over the full batch size) and the two totals with
        ``torch.autograd.grad`` on the retained graph, and after a productive update hands
        the sink ONE :class:`ActorGradientReport`. No forward, GAE pass, RNG draw,
        ``.grad`` write or optimizer change is added; the real backward, clipping and
        steps are unchanged.

        ``gradient_contrast_ids`` (optional, requires the two above): a
        ``(positive_id, negative_id)`` pair of those opaque ids. When both occur in the
        batch, epoch 0 also forms the differentiable contrast ``mean P(ABORT | positive)
        - mean P(ABORT | negative)`` from the SAME epoch-0 logits through
        :func:`_semantic_dist` (no forward) and reports its value and gradient.

        Per epoch:

          ACTOR (per decision) -- re-encode the stored PRIVATE ``tr.gobs``, rebuild the
          mask, ``evaluate_action``, form the ratio against the stored rollout log-prob,
          and take the clipped surrogate against the DETACHED normalized GAE advantage,
          minus the entropy bonus.

          CRITIC (per decision) -- re-encode the stored CENTRAL state, pool it, read
          ``V(s)``, and take the squared error against the FIXED ``value_target``
          computed before epoch 0. Scaled by ``ctde_cfg.value_coeff``. No value
          clipping in v1.

        Then ONE mean and ONE backward PER SIDE, each with its own grad-norm clip and
        its own optimizer step.

        Empty-batch contract, unchanged in meaning from :meth:`PPOUpdater.update`: a
        batch with ZERO decisions is a clean no-op -- no forward, no backward, no step
        on either optimizer -- reported with ``n_epochs_run == 0``. An iteration in
        which no ego woke is a legitimate outcome, not an error.

        Returns:
            A diagnostics dict shaped like :meth:`PPOUpdater.update`'s, plus the CTDE
            keys ``value_loss``, ``critic_grad_norm``, ``value_mean`` and
            ``value_target_mean``.

            ``baseline`` DELIBERATELY KEEPS ITS ACTOR-ONLY MEANING -- the batch's mean
            EPISODE REWARD, zero-wake episodes included -- even though the CTDE baseline
            is really the critic. ``graph_train`` records that key as an iteration's
            ``train_reward_mean``, so reporting the critic's mean value there would make
            one recorded field mean a reward under ``actor_only`` and a value estimate
            under ``ctde``: the two modes' learning curves would stop being comparable
            while still looking as though they were. The critic's own estimate is
            reported under ``value_mean`` instead.
        """
        cfg = self.cfg
        ctde_cfg = self.ctde_cfg
        records = _as_ctde_records(source)
        batch = compute_ctde_advantages(records, self.critic, cfg, ctde_cfg)

        n_ep_with_wakes = sum(1 for rec in records if rec.has_wakes)
        per_epoch: Dict[str, List[float]] = {
            "policy_loss": [], "total_loss": [], "entropy": [], "mean_ratio": [],
            "clip_fraction": [], "approx_kl": [], "max_ratio_dev": [], "grad_norm": [],
            "value_loss": [], "critic_grad_norm": [],
        }
        diagnostics: Dict[str, Any] = {
            # `baseline` KEEPS ITS ACTOR-ONLY MEANING: the batch's mean EPISODE REWARD.
            # It is not the CTDE baseline -- the critic is -- but this key is what
            # `graph_train` records as an iteration's `train_reward_mean`, so putting
            # the critic's mean value here would silently make one field mean a reward
            # in one training mode and a value estimate in the other, and the two modes'
            # records would stop being comparable while still looking it. The critic's
            # own estimate is reported separately, as `value_mean`.
            "baseline": batch.episode_reward_mean,
            "adv_std_raw": batch.adv_std_raw,
            "n_transitions": batch.n_transitions,
            "n_episodes": batch.n_episodes,
            "episodes_with_wakes": n_ep_with_wakes,
            "n_epochs_run": 0,
            "value_mean": (
                float(np.mean(batch.values)) if batch.n_transitions else 0.0
            ),
            "value_target_mean": (
                float(np.mean(batch.value_targets)) if batch.n_transitions else 0.0
            ),
            "per_epoch": per_epoch,
        }

        if (gradient_group_ids is None) != (gradient_sink is None):
            raise ValueError(
                "gradient_group_ids and gradient_sink must be given together")
        group_ids: Optional[Tuple[int, ...]] = None
        if gradient_group_ids is not None:
            group_ids = tuple(int(g) for g in gradient_group_ids)
            if len(group_ids) != batch.n_transitions:
                raise ValueError(
                    "gradient_group_ids has %d entries but the batch has %d transitions"
                    % (len(group_ids), batch.n_transitions))
        contrast_ids: Optional[Tuple[int, int]] = None
        if gradient_contrast_ids is not None:
            if group_ids is None:
                raise ValueError("gradient_contrast_ids requires gradient_group_ids")
            if len(gradient_contrast_ids) != 2:
                raise ValueError("gradient_contrast_ids must be two distinct ids")
            contrast_ids = (int(gradient_contrast_ids[0]), int(gradient_contrast_ids[1]))
            if contrast_ids[0] == contrast_ids[1]:
                raise ValueError("gradient_contrast_ids must be two distinct ids")

        if batch.n_transitions == 0:
            for key in ("policy_loss", "total_loss", "entropy", "mean_ratio",
                        "clip_fraction", "approx_kl", "max_ratio_dev", "grad_norm",
                        "value_loss", "critic_grad_norm"):
                diagnostics[key] = 0.0
            return diagnostics

        advantages = batch.advantages
        value_targets = batch.value_targets
        n = batch.n_transitions
        gradient_report: Optional[ActorGradientReport] = None

        for _epoch in range(cfg.n_epochs):
            policy_losses: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []
            value_losses: List[torch.Tensor] = []
            ratios_detached: List[float] = []
            kl_terms: List[float] = []
            # (contrast side, the SAME epoch-0 logits, mask) -- kept only when requested.
            contrast_inputs: List[Tuple[int, torch.Tensor, np.ndarray]] = []

            for i, tr in enumerate(batch.transitions):
                # --- ACTOR: the ego's PRIVATE observation only ---
                logits = self._forward_logits(tr.gobs)
                mask = build_action_mask(tr.gobs)
                log_prob_new, entropy = evaluate_action(
                    logits, mask, tr.meta_action, tr.node_v
                )
                ratio = torch.exp(log_prob_new - float(tr.log_prob))
                policy_losses.append(
                    clipped_surrogate(ratio, float(advantages[i]), cfg.clip_ratio)
                )
                entropies.append(entropy)
                if (contrast_ids is not None and _epoch == 0
                        and group_ids[i] in contrast_ids):
                    contrast_inputs.append((contrast_ids.index(group_ids[i]), logits, mask))

                r = float(ratio.detach().item())
                ratios_detached.append(r)
                kl_terms.append(float(tr.log_prob) - float(log_prob_new.detach().item()))

                # --- CRITIC: the CENTRAL state only, against a FIXED target ---
                value = self.critic(batch.central_states[i])
                value_losses.append((value - float(value_targets[i])) ** 2)

            policy_loss = torch.stack(policy_losses).mean()
            entropy_mean = torch.stack(entropies).mean()
            actor_loss = policy_loss - cfg.entropy_coeff * entropy_mean
            value_loss = torch.stack(value_losses).mean()
            critic_loss = ctde_cfg.value_coeff * value_loss

            # OBSERVATIONAL epoch-0 gradient decomposition over the graph built above.
            # `autograd.grad` writes no `.grad` and retains the graph, so the real
            # backward below runs exactly as it would without this block.
            if group_ids is not None and _epoch == 0:
                members: Dict[int, List[int]] = {}
                for i, g in enumerate(group_ids):
                    members.setdefault(g, []).append(i)
                contrast_value: Optional[float] = None
                contrast_grad: Optional[np.ndarray] = None
                sides: Tuple[List[torch.Tensor], List[torch.Tensor]] = ([], [])
                for side, c_logits, c_mask in contrast_inputs:
                    _, c_dist, _ = _semantic_dist(c_logits, c_mask)
                    sides[side].append(c_dist.probs[SEMANTIC_ABORT_LEAF])
                if sides[0] and sides[1]:
                    contrast = torch.stack(sides[0]).mean() - torch.stack(sides[1]).mean()
                    contrast_value = float(contrast.detach().item())
                    contrast_grad = _flat_actor_grad(contrast, self.actor_parameters)
                gradient_report = ActorGradientReport(
                    records=records,
                    batch=batch,
                    group_ids=group_ids,
                    group_counts={g: len(idx) for g, idx in members.items()},
                    group_policy_surrogate_grads={
                        g: _flat_actor_grad(
                            torch.stack([policy_losses[i] for i in idx]).sum() / n,
                            self.actor_parameters)
                        for g, idx in members.items()
                    },
                    total_policy_surrogate_grad=_flat_actor_grad(
                        policy_loss, self.actor_parameters),
                    total_actor_loss_grad=_flat_actor_grad(
                        actor_loss, self.actor_parameters),
                    entropy_coeff=float(cfg.entropy_coeff),
                    contrast_ids=contrast_ids,
                    contrast_value=contrast_value,
                    contrast_grad=contrast_grad,
                )

            # TWO independent backwards on two disjoint graphs. The actor's backward
            # cannot reach a critic parameter and the critic's cannot reach an actor
            # parameter, because neither loss references the other's modules and the
            # advantage crossing between them is a python float.
            self.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            critic_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_parameters, cfg.max_grad_norm
            )
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, cfg.max_grad_norm
            )
            self.optimizer.step()
            self.critic_optimizer.step()

            ratios_arr = np.asarray(ratios_detached, dtype=np.float64)
            per_epoch["policy_loss"].append(float(policy_loss.detach().item()))
            # `total_loss` stays the ACTOR-side total, so it means the same thing it
            # does in the actor-only diagnostics; the critic reports `value_loss`.
            per_epoch["total_loss"].append(float(actor_loss.detach().item()))
            per_epoch["entropy"].append(float(entropy_mean.detach().item()))
            per_epoch["mean_ratio"].append(float(ratios_arr.mean()))
            per_epoch["clip_fraction"].append(
                float(np.mean(np.abs(ratios_arr - 1.0) > cfg.clip_ratio))
            )
            per_epoch["approx_kl"].append(float(np.mean(kl_terms)))
            per_epoch["max_ratio_dev"].append(float(np.max(np.abs(ratios_arr - 1.0))))
            per_epoch["grad_norm"].append(float(grad_norm))
            per_epoch["value_loss"].append(float(value_loss.detach().item()))
            per_epoch["critic_grad_norm"].append(float(critic_grad_norm))

        diagnostics["n_epochs_run"] = cfg.n_epochs
        for key, values in per_epoch.items():
            diagnostics[key] = float(np.mean(values)) if values else 0.0
        assert diagnostics["n_transitions"] == n  # alignment sanity
        if credit_sink is not None and cfg.n_epochs > 0:
            credit_sink(CreditReport(
                training_mode="ctde", records=records, batch=batch, cfg=cfg,
                ctde_cfg=ctde_cfg,
            ))
        if gradient_sink is not None and gradient_report is not None:
            gradient_sink(gradient_report)
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
    # ego row [fuel_norm, mission_fuel_slack_norm]; peers featureless
    agent_features = np.array([[0.90, 0.10], [0.00, 0.00], [0.00, 0.00]], dtype=np.float32)
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
    force_cell: Optional[Tuple[int, Optional[int]]] = None,
) -> "Transition":
    """Produce ONE real :class:`Transition`, storing exactly what ``_wake_decision`` stores.

    This is SYNTHETIC ROLLOUT DATA construction — the one place ``torch.no_grad`` is
    legitimate in this module's test path (it mirrors the inference-only rollout).

    ``force_cell=(meta_action, node_v)`` pins the stored SEMANTIC action (``node_v``
    ``None`` for PLAN / ABORT) instead of sampling it, scoring it through
    ``evaluate_action``. That is the SAME distribution ``sample_action`` would have drawn
    from (one shared ``_semantic_dist`` construction site), so the stored log-prob is
    exactly the one a rollout that happened to sample this action would have stored.
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
            meta_action, node_v = int(force_cell[0]), force_cell[1]
            log_prob, entropy = evaluate_action(logits, mask, meta_action, node_v)

    return Transition(
        gobs=gobs,
        ego_id=str(ego_id),
        tick=int(tick),
        meta_action=int(meta_action),
        node_v=node_v,
        log_prob=float(log_prob.item()),
        entropy=float(entropy.item()),
    )


def _action_prob(
    policy: "Policy", gobs: Any, meta_action: int, node_v: Optional[int],
) -> float:
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
    good_cell = (OE, 2)     # engage the unassigned pop-up (node 2)
    bad_cell = (PC, None)   # the one global PLAN_COMPLIANCE action

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
    print(f"  P(bad =PLAN): {p_bad_before:.6f} -> {p_bad_after:.6f}  "
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
