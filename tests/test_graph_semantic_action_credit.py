"""GENERALIZED-V2 SEMANTIC ACTION REPRESENTATION + OBSERVATIONAL CREDIT DIAGNOSTICS.

Solver-free and BLADE-free. Every test drives the real production symbols.

SEMANTIC ACTIONS (``semantic_k_plus_2_logmeanexp_v1``)
  S1  leaf order and legality follow the k x 3 source mask exactly
  S2  PLAN is exact logmeanexp over all k nodes; duplicated evidence buys nothing
  S3  ABORT is exact logmeanexp over the abort-LEGAL nodes only
  S4  no canonical node; task permutation leaves PLAN / ABORT invariant and permutes
      ENGAGE (construction level AND through a real encoder + head)
  S5  stochastic sampling follows the semantic distribution
  S6  deterministic selection is the semantic argmax, with stable exact-tie order
  S7  the entropy bonus is the semantic-leaf entropy (no alias spread)
  S8  sample / re-score bitwise agreement, dtype preservation, epoch-0 ratio == 1 in
      BOTH updaters, gradient routing, malformed identities fail loud in BOTH updaters
  S9  effect semantics and nullable-node guards
  S10 the V2 primary endpoint reads the semantic ABORT leaf; historical records stay
      readable; a pair across representations is not measurable

CREDIT DIAGNOSTICS (``train_credit_diagnostics.jsonl``)
  C1  actor-only rows carry exactly the credit the update used
  C2  CTDE rows carry V_old, the TD residual, the GAE advantage and the target used
  C3  observational: no RNG, no extra forward, no gradient, identical update on / off
  C4  measurement tags cannot reach any actor / critic / optimizer input
  C5  no control path reads the artifact back
  C6  persistence failures fail loud; provenance names the representation
  C7  end to end through the real trainer and the real updaters
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_aou.models import Step, StepKind, Task  # noqa: E402
from match_aou.rl.action import graph_action as GA  # noqa: E402
from match_aou.rl.action.graph_action import (  # noqa: E402
    ACTION_REPRESENTATION_ID,
    MetaAction,
    _semantic_dist,
    build_action_mask,
    evaluate_action,
    sample_action,
    summarize_decision,
)
from match_aou.rl.action.graph_effect import apply_meta_action  # noqa: E402
from match_aou.rl.observation import central_graph_builder as CGB  # noqa: E402
from match_aou.rl.observation.graph_builder import EdgeType, GraphObservation  # noqa: E402
from match_aou.rl.training import graph_ppo as GP  # noqa: E402
from match_aou.rl.training import graph_tick_loop as TL  # noqa: E402
from match_aou.rl.training import graph_train as GT  # noqa: E402
from match_aou.rl.training.graph_ppo import (  # noqa: E402
    CTDEConfig,
    CTDEEpisodeRecord,
    CTDEUpdater,
    EpisodeRecord,
    PPOBuffer,
    PPOConfig,
    PPOUpdater,
    build_central_critic,
)

import test_graph_ctde as CT  # noqa: E402  (stub trainer + synthetic central states)

PLAN = int(MetaAction.PLAN_COMPLIANCE)
ENGAGE = int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)
ABORT = int(MetaAction.SELF_PRESERVATION_ABORT)
NEG = float("-inf")
ABORT_NAME = MetaAction.SELF_PRESERVATION_ABORT.name


# =============================================================================
# Fixtures
# =============================================================================

def _row(utility, *, sensed=1.0, reachable=1.0):
    # [utility, dist_to_ego, capable, reachable, probability, sensed]
    return [utility, 0.3, 1.0, reachable, 1.0, sensed]


_ROWS = [_row(0.8), _row(0.6), _row(0.5), _row(0.7)]


def _obs(rows=None, *, ego_nodes=(0, 1), peer_nodes=()):
    """k task nodes; ``ego_nodes`` assigned to the ego (abort-legal), the rest unassigned
    unless in ``peer_nodes``. Unassigned sensed/capable/reachable nodes are ENGAGE-legal."""
    rows = _ROWS if rows is None else rows
    k = len(rows)
    src = [k] * len(ego_nodes) + [k + 1] * len(peer_nodes)
    dst = list(ego_nodes) + list(peer_nodes)
    return GraphObservation(
        task_features=np.asarray(rows, dtype=np.float32),
        agent_features=np.array([[0.9], [0.0]], dtype=np.float32),
        ego_index=k,
        edge_index=(np.array([src, dst], dtype=np.int64) if src
                    else np.zeros((2, 0), dtype=np.int64)),
        edge_type=np.full((len(src),), int(EdgeType.ASSIGNMENT), dtype=np.int64),
        task_target_ids=["t%d" % i for i in range(k)],
        agent_ids=["ego", "peer"],
        agent_id="ego",
        current_time=5,
        time_norm=0.05,
    )


def _logits(k, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(k, 3, generator=gen).to(dtype)


def _policy(seed=0):
    torch.manual_seed(seed)
    return TL.build_policy()


def _sampled_transition(policy, gobs, *, ego="egoA", tick=1, seed=0, reward=None,
                        force=None):
    mask = build_action_mask(gobs)
    with torch.no_grad():
        logits = policy.head(policy.encoder(gobs))
        if force is None:
            torch.manual_seed(seed)
            meta, node, lp, ent = sample_action(logits, mask)
        else:
            meta, node = force
            lp, ent = evaluate_action(logits, mask, meta, node)
    return TL.Transition(gobs=gobs, ego_id=ego, tick=tick, meta_action=int(meta),
                         node_v=node, log_prob=float(lp.item()),
                         entropy=float(ent.item()), reward=reward)


def _actor_only_records(policy):
    """Three episodes with different rewards; two egos in the first (interleaved)."""
    records = []
    for i, reward in enumerate([-0.1, -0.6, -0.9]):
        gobs = _obs([_row(0.8 - 0.05 * i), _row(0.6), _row(0.5), _row(0.7)])
        traj = [_sampled_transition(policy, gobs, ego="egoA", tick=1, seed=10 * i + 1),
                _sampled_transition(policy, gobs, ego="egoB", tick=1, seed=10 * i + 2),
                _sampled_transition(policy, gobs, ego="egoA", tick=4, seed=10 * i + 3)]
        traj[-1].reward = reward
        records.append(EpisodeRecord.from_trajectory(traj, reward, seed=100 + i,
                                                     episode_index=i))
    return records


def _ctde_records(policy):
    records = []
    for i, reward in enumerate([-0.2, -0.7]):
        gobs = _obs([_row(0.8), _row(0.6 - 0.1 * i), _row(0.5), _row(0.7)])
        traj = [_sampled_transition(policy, gobs, ego="egoA", tick=t, seed=7 * i + t)
                for t in (1, 3, 6)]
        traj[-1].reward = reward
        states = [CT._synthetic_central(k=3, a=2, seed=31 * i + j) for j in range(3)]
        records.append(CTDEEpisodeRecord.from_episode(traj, states, reward,
                                                      seed=200 + i, episode_index=i))
    return records


# =============================================================================
# S1 -- S3: semantic construction
# =============================================================================

def test_s1_leaf_order_and_legality_follow_the_source_mask():
    gobs = _obs()
    mask = build_action_mask(gobs)
    sem, dist, _ent = _semantic_dist(_logits(4), mask)
    assert sem.shape == (4 + 2,)
    # PLAN, ABORT (ego holds nodes 0 and 1), ENGAGE(0..3): legal on the unassigned 2, 3
    assert torch.isfinite(sem).tolist() == [True, True, False, False, True, True]
    assert [GA.semantic_leaf_identity(i, 4) for i in range(6)] == [
        (PLAN, None), (ABORT, None), (ENGAGE, 0), (ENGAGE, 1), (ENGAGE, 2), (ENGAGE, 3)]
    for i in range(6):
        assert GA.semantic_leaf_index(*GA.semantic_leaf_identity(i, 4), 4) == i
    assert float(dist.probs[~torch.isfinite(sem)].sum()) == 0.0

    # no ego assignment -> the ONE ABORT leaf is masked
    sem2, _d, _e = _semantic_dist(_logits(4), build_action_mask(_obs(ego_nodes=())))
    assert not bool(torch.isfinite(sem2[1]))


def test_s1_k_zero_fails_loud():
    empty = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        _semantic_dist(torch.zeros((0, 3)), empty)
    with pytest.raises(ValueError):
        sample_action(torch.zeros((0, 3)), empty)


def test_s2_plan_is_exact_logmeanexp_and_duplication_gives_no_bonus():
    logits = _logits(4, seed=3)
    sem, _d, _e = _semantic_dist(logits, build_action_mask(_obs()))
    assert torch.equal(sem[0], torch.logsumexp(logits[:, PLAN], 0) - math.log(4))

    s = 0.7
    one = torch.tensor([[s, -9.0, s]])
    one_mask = np.array([[0.0, NEG, 0.0]], dtype=np.float32)
    three = torch.tensor([[s, -9.0, s], [s, -9.0, -9.0], [s, -9.0, -9.0]])
    three_mask = np.array([[0.0, NEG, 0.0], [0.0, NEG, NEG], [0.0, NEG, NEG]],
                          dtype=np.float32)
    p_one = _semantic_dist(one, one_mask)[1].probs
    p_three = _semantic_dist(three, three_mask)[1].probs
    assert float(p_one[0]) == pytest.approx(0.5) and float(p_one[1]) == pytest.approx(0.5)
    # three equal PLAN aliases: still 50/50 (a logsumexp collapse would give PLAN 0.75)
    assert float(p_three[0]) == pytest.approx(0.5, abs=1e-6)
    assert float(p_three[1]) == pytest.approx(0.5, abs=1e-6)


def test_s3_abort_is_logmeanexp_over_abort_legal_nodes_only():
    gobs = _obs()
    mask = build_action_mask(gobs)
    logits = _logits(4, seed=4)
    sem, dist, _e = _semantic_dist(logits, mask)
    expect = torch.logsumexp(logits[[0, 1], ABORT], 0) - math.log(2)
    assert torch.equal(sem[1], expect)

    # abort scores on ABORT-ILLEGAL nodes move nothing at all
    moved = logits.clone()
    moved[2, ABORT] += 50.0
    moved[3, ABORT] -= 50.0
    assert torch.equal(_semantic_dist(moved, mask)[1].probs, dist.probs)

    # equal abort evidence on two legal nodes == the same evidence on one
    two = logits.clone()
    two[0, ABORT] = two[1, ABORT] = 1.25
    single = build_action_mask(_obs(ego_nodes=(0,)))
    one = logits.clone()
    one[0, ABORT] = 1.25
    assert float(_semantic_dist(two, mask)[0][1]) == pytest.approx(
        float(_semantic_dist(one, single)[0][1]), abs=1e-6)


# =============================================================================
# S4: permutation semantics, no canonical node
# =============================================================================

_PERM = [2, 0, 3, 1]   # new node i holds old node _PERM[i]


def test_s4_permutation_construction_level():
    logits = _logits(4, seed=8)
    logits[3, ENGAGE] = 6.0             # make ENGAGE(3) the deterministic choice
    mask = build_action_mask(_obs())
    sem, dist, ent = _semantic_dist(logits, mask)
    sem_p, dist_p, ent_p = _semantic_dist(logits[_PERM], mask[_PERM])

    assert float(sem_p[0]) == pytest.approx(float(sem[0]), abs=1e-6)
    assert float(sem_p[1]) == pytest.approx(float(sem[1]), abs=1e-6)
    for i, old in enumerate(_PERM):
        assert torch.equal(sem_p[2 + i], sem[2 + old])
        assert float(dist_p.probs[2 + i]) == pytest.approx(float(dist.probs[2 + old]),
                                                           abs=1e-6)
    assert float(ent_p) == pytest.approx(float(ent), abs=1e-6)

    meta, node, _lp, _e = sample_action(logits, mask, deterministic=True)
    meta_p, node_p, _lp2, _e2 = sample_action(logits[_PERM], mask[_PERM], True)
    assert (meta, node) == (ENGAGE, 3)
    assert (meta_p, _PERM[node_p]) == (ENGAGE, 3), "the selected task did not remap"


def test_s4_permutation_through_a_real_encoder_and_head():
    policy = _policy(1)
    rows = [_row(0.8), _row(0.6), _row(0.5), _row(0.7)]
    inv = {old: new for new, old in enumerate(_PERM)}
    base = _obs(rows, ego_nodes=(0, 1))
    perm = _obs([rows[p] for p in _PERM], ego_nodes=[inv[0], inv[1]])
    with torch.no_grad():
        p = _semantic_dist(policy.head(policy.encoder(base)), build_action_mask(base))[1].probs
        q = _semantic_dist(policy.head(policy.encoder(perm)), build_action_mask(perm))[1].probs
    assert float(q[0]) == pytest.approx(float(p[0]), abs=1e-5)
    assert float(q[1]) == pytest.approx(float(p[1]), abs=1e-5)
    for i, old in enumerate(_PERM):
        assert float(q[2 + i]) == pytest.approx(float(p[2 + old]), abs=1e-5)


# =============================================================================
# S5 -- S7: sampling, argmax, entropy
# =============================================================================

def test_s5_stochastic_sampling_follows_the_semantic_distribution():
    logits = torch.tensor([[0.2, 0.0, 0.9], [0.1, 0.0, 0.3],
                           [0.4, 0.8, 0.0], [0.0, -0.3, 0.0]])
    mask = build_action_mask(_obs())
    probs = _semantic_dist(logits, mask)[1].probs
    torch.manual_seed(1234)
    n = 6000
    counts = np.zeros(6)
    for _ in range(n):
        meta, node, _lp, _e = sample_action(logits, mask)
        counts[GA.semantic_leaf_index(meta, node, 4)] += 1
    assert counts[2] == counts[3] == 0, "an illegal leaf was sampled"
    assert np.allclose(counts / n, probs.numpy(), atol=0.03), (counts / n, probs)


def test_s6_deterministic_selection_is_the_semantic_argmax_with_stable_ties():
    mask = build_action_mask(_obs())
    for seed in range(20):
        logits = _logits(4, seed=seed)
        sem = _semantic_dist(logits, mask)[0]
        meta, node, _lp, _e = sample_action(logits, mask, deterministic=True)
        assert GA.semantic_leaf_index(meta, node, 4) == int(torch.argmax(sem))
        assert [sample_action(logits, mask, True)[:2] for _ in range(3)] == [(meta, node)] * 3

    # ENGAGE exact tie -> the LOWEST task index
    tie = torch.tensor([[-5.0, 2.0, 0.0], [-5.0, 2.0, 0.0]])
    tie_mask = np.array([[0.0, 0.0, NEG], [0.0, 0.0, NEG]], dtype=np.float32)
    assert sample_action(tie, tie_mask, True)[:2] == (ENGAGE, 0)
    # PLAN == ABORT exact tie (k = 1, so logmeanexp is the score itself) -> PLAN
    pa = torch.tensor([[1.0, 0.0, 1.0]])
    assert sample_action(pa, np.array([[0.0, NEG, 0.0]], dtype=np.float32), True)[:2] == (
        PLAN, None)


def test_s7_entropy_is_the_semantic_leaf_entropy_without_alias_spread():
    mask = build_action_mask(_obs())
    logits = _logits(4, seed=5)
    sem, dist, ent = _semantic_dist(logits, mask)
    p = dist.probs[torch.isfinite(sem)]
    assert float(ent) == pytest.approx(float(-(p * torch.log(p)).sum()), abs=1e-6)
    assert math.isfinite(float(ent))

    # the same semantic scores stated through 1 or 3 PLAN aliases -> the same entropy
    one = torch.tensor([[0.3, -9.0, 1.1]])
    one_mask = np.array([[0.0, NEG, 0.0]], dtype=np.float32)
    three = torch.tensor([[0.3, -9.0, 1.1], [0.3, -9.0, 0.0], [0.3, -9.0, 0.0]])
    three_mask = np.array([[0.0, NEG, 0.0], [0.0, NEG, NEG], [0.0, NEG, NEG]],
                          dtype=np.float32)
    assert float(_semantic_dist(three, three_mask)[2]) == pytest.approx(
        float(_semantic_dist(one, one_mask)[2]), abs=1e-6)


# =============================================================================
# S8: PPO identity, gradients, malformed identities, both updaters
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_s8_sample_and_rescore_agree_bitwise_and_preserve_dtype(dtype):
    mask = build_action_mask(_obs())
    for seed in range(15):
        logits = _logits(4, seed=seed, dtype=dtype)
        for det in (False, True):
            torch.manual_seed(seed)
            meta, node, lp, ent = sample_action(logits, mask, det)
            lp2, ent2 = evaluate_action(logits, mask, meta, node)
            assert lp.dtype == dtype and ent.dtype == dtype
            assert torch.equal(lp, lp2) and torch.equal(ent, ent2)


def test_s8_epoch_zero_ratio_is_exactly_one_in_both_updaters():
    policy = _policy(2)
    diag = PPOUpdater(copy.deepcopy(policy), PPOConfig(n_epochs=1)).update(
        _actor_only_records(policy))
    assert diag["per_epoch"]["max_ratio_dev"][0] == 0.0
    assert diag["per_epoch"]["mean_ratio"][0] == 1.0

    critic = build_central_critic()
    diag_c = CTDEUpdater(copy.deepcopy(policy), critic, PPOConfig(n_epochs=1),
                         CTDEConfig()).update(_ctde_records(policy))
    assert diag_c["per_epoch"]["max_ratio_dev"][0] == 0.0
    assert np.isfinite(diag_c["per_epoch"]["grad_norm"][0])
    # ONE shared re-scoring function for both updaters
    assert GP.evaluate_action is GA.evaluate_action


def test_s8_gradients_route_only_through_legal_source_scores():
    mask = build_action_mask(_obs())
    for meta, node in ((ABORT, None), (PLAN, None), (ENGAGE, 2)):
        logits = _logits(4, seed=6).requires_grad_(True)
        lp, ent = evaluate_action(logits, mask, meta, node)
        (lp + ent).backward()
        g = logits.grad
        assert torch.isfinite(g).all()
        legal = torch.as_tensor(np.isfinite(mask))
        assert torch.all(g[~legal] == 0.0), (meta, node, g)
        assert bool((g[:, PLAN] != 0).all()), "PLAN normalizes over all k nodes"
        assert bool((g[[0, 1], ABORT] != 0).all())


def test_s8_malformed_stored_identity_fails_loud_in_both_updaters():
    policy = _policy(3)
    gobs = _obs()
    bad = TL.Transition(gobs=gobs, ego_id="egoA", tick=1, meta_action=PLAN, node_v=0,
                        log_prob=-1.0, entropy=1.0, reward=-0.5)
    with pytest.raises(ValueError, match="malformed"):
        PPOUpdater(policy, PPOConfig(n_epochs=1)).update(
            [EpisodeRecord.from_trajectory([bad], -0.5)])
    with pytest.raises(ValueError, match="malformed"):
        CTDEUpdater(policy, build_central_critic(), PPOConfig(n_epochs=1),
                    CTDEConfig()).update(
            [CTDEEpisodeRecord.from_episode([bad], [CT._synthetic_central()], -0.5)])


# =============================================================================
# S9: effects
# =============================================================================

def _tasks(n=3):
    return [Task(steps=[Step(None, "t%d" % i, [], 1.0, 1, StepKind.ATTACK)], utility=50)
            for i in range(n)]


def test_s9_effects_and_nullable_node_guards():
    tasks = _tasks()
    gobs = _obs(_ROWS[:3], ego_nodes=(0,))
    base = {"ego": [(0, 0, 0), (2, 0, 1)], "peer": [(1, 0, 0)]}
    assert apply_meta_action(base, gobs, "ego", PLAN, None, tasks) == base
    aborted = apply_meta_action(base, gobs, "ego", ABORT, None, tasks)
    assert aborted == {"ego": [], "peer": [(1, 0, 0)]}
    engaged = apply_meta_action(base, gobs, "ego", ENGAGE, 1, tasks)
    assert engaged["ego"] == [(0, 0, 0), (2, 0, 1), (1, 0, -1)]
    assert base == {"ego": [(0, 0, 0), (2, 0, 1)], "peer": [(1, 0, 0)]}
    for meta, node in ((PLAN, 0), (ABORT, 1), (ENGAGE, None), (ENGAGE, True),
                       (ENGAGE, 3), (ENGAGE, -1), (True, None)):
        with pytest.raises(ValueError):
            apply_meta_action(base, gobs, "ego", meta, node, tasks)


# =============================================================================
# S10: the V2 endpoint and historical readability
# =============================================================================

def _semantic_fd_wake(abort_score):
    logits = torch.tensor([[0.0, 0.0, abort_score], [0.0, 0.0, abort_score],
                           [0.0, 0.0, 0.0]])
    mask = np.array([[0.0, NEG, 0.0], [0.0, NEG, 0.0], [0.0, NEG, NEG]], dtype=np.float32)
    meta, node, _lp, _e = sample_action(logits, mask, True)
    wake = summarize_decision(logits, mask, meta, node)
    wake["wake_kind"] = TL.WAKE_KIND_IMMEDIATE_FD
    return wake


def _legacy_fd_wake(mass, selected=ABORT_NAME):
    return {"wake_kind": TL.WAKE_KIND_IMMEDIATE_FD, "selected_meta_action_name": selected,
            "aggregate_probability_per_meta_action": {
                ABORT_NAME: mass, MetaAction.PLAN_COMPLIANCE.name: 1.0 - mass,
                MetaAction.OPPORTUNISTIC_ENGAGEMENT.name: 0.0}}


def test_s10_v2_endpoint_reads_the_semantic_abort_leaf_with_unchanged_aggregation():
    groups, expected = [], {}
    for c, bc in enumerate(GT.V2_BENCHMARK_BASE_CELL_KEYS):
        mild, severe = _semantic_fd_wake(-1.0 + 0.1 * c), _semantic_fd_wake(0.5 + 0.1 * c)
        p_m = mild["semantic_probability_per_meta_action"][ABORT_NAME]
        p_s = severe["semantic_probability_per_meta_action"][ABORT_NAME]
        assert p_m == mild["semantic_probabilities"][GA.SEMANTIC_ABORT_LEAF]
        expected[bc] = p_s - p_m
        groups.append({"group_key": "g%d" % c, "base_cell": bc, "complete": True,
                       "member_decisions": {GT.SEVERITY_MILD: [mild],
                                            GT.SEVERITY_SEVERE: [severe]}})
    out = GT._v2_behaviour_summary(groups)
    assert out["metric"] == "severe_minus_mild_aggregate_abort_mass"
    assert out["action_representation_ids_observed"] == [ACTION_REPRESENTATION_ID]
    assert out["aggregate_mass_is_not_selected_action_probability"] is False
    assert out["n_groups_metric_eligible"] == len(expected)
    for row in out["groups"]:
        assert row["severe_minus_mild_abort_mass"] == expected[row["base_cell"]]
    assert out["macro_mean_over_base_cells"] == pytest.approx(
        sum(expected.values()) / len(expected))


def test_s10_historical_records_stay_readable_and_pairs_never_mix_representations():
    assert GT._v2_immediate_fd_member([_legacy_fd_wake(0.3)], "mild") == (0.3, ABORT_NAME,
                                                                         None)
    assert GT._immediate_fd_abort_mass({"wake_decisions": [_legacy_fd_wake(0.4)]}) == 0.4
    sem = _semantic_fd_wake(0.2)
    assert GT._immediate_fd_abort_mass({"wake_decisions": [sem]}) == \
        sem["semantic_probability_per_meta_action"][ABORT_NAME]

    mixed = GT._v2_behaviour_summary([{
        "group_key": "g0", "base_cell": GT.V2_BENCHMARK_BASE_CELL_KEYS[0],
        "complete": True,
        "member_decisions": {GT.SEVERITY_MILD: [_legacy_fd_wake(0.3)],
                             GT.SEVERITY_SEVERE: [sem]}}])
    assert mixed["groups"][0]["not_measurable_reason"] == "mixed_action_representations"
    assert mixed["n_groups_metric_eligible"] == 0


# =============================================================================
# C1 / C2: credit rows carry the credit the update used
# =============================================================================

def _capture_surrogate(monkeypatch):
    seen = []
    real = GP.clipped_surrogate

    def spy(ratio, advantage, clip_ratio):
        seen.append(float(advantage))
        return real(ratio, advantage, clip_ratio)

    monkeypatch.setattr(GP, "clipped_surrogate", spy)
    return seen


def test_c1_actor_only_rows_carry_the_credit_the_update_used(monkeypatch):
    policy = _policy(4)
    records = _actor_only_records(policy)
    seen = _capture_surrogate(monkeypatch)
    batches = []
    real_credit = GP.compute_returns_and_advantages
    monkeypatch.setattr(GP, "compute_returns_and_advantages",
                        lambda *a, **k: batches.append(real_credit(*a, **k)) or batches[-1])
    reports = []
    diag = PPOUpdater(policy, PPOConfig(n_epochs=2)).update(records, credit_sink=reports.append)

    assert len(reports) == 1 and reports[0].batch is batches[0]
    batch = batches[0]
    tags = {(r.episode_index, r.seed): {"cell": "mild", "severity": "mild",
                                        "condition": "damaged",
                                        "fd_selected_ego_id": "egoA", "fd_event_tick": 4}
            for r in records}
    rows = GT._credit_rows(reports[0], iteration=3, updates_completed_before=2,
                           measurement_tags=tags)
    json.dumps(rows)
    assert len(rows) == diag["n_transitions"] == 9
    n = len(rows)
    assert [r["normalized_advantage"] for r in rows] == seen[:n] == seen[n:]
    baseline = float(np.mean([-0.1, -0.6, -0.9]))
    for i, row in enumerate(rows):
        assert row["training_mode"] == "actor_only"
        assert row["action_representation_id"] == ACTION_REPRESENTATION_ID
        assert row["return"] == float(batch.returns[i])
        assert row["actor_only_episode_baseline"] == baseline == float(batch.baseline)
        assert row["raw_advantage"] == float(batch.raw_advantages[i])
        assert row["raw_advantage"] == row["return"] - baseline
        assert row["batch_raw_advantage_std"] == float(batch.adv_std_raw)
        for key in ("value_old", "td_residual", "value_target", "gae_lambda",
                    "episode_decision_ordinal"):
            assert row[key] is None, key          # null, never a fabricated 0
        tr = batch.transitions[i]
        assert (row["ego_id"], row["tick"], row["selected_node"]) == (
            tr.ego_id, tr.tick, tr.node_v)
        assert row["measurement_join"]["joined"] is True
        assert row["measurement_join"]["is_fd_selected_ego"] == (tr.ego_id == "egoA")
    assert [r["ego_chain_ordinal"] for r in rows[:3]] == [0, 1, 0]


def test_c2_ctde_rows_carry_v_old_td_residual_gae_and_target(monkeypatch):
    policy = _policy(5)
    records = _ctde_records(policy)
    seen = _capture_surrogate(monkeypatch)
    batches = []
    real = GP.compute_ctde_advantages
    monkeypatch.setattr(GP, "compute_ctde_advantages",
                        lambda *a, **k: batches.append(real(*a, **k)) or batches[-1])
    cfg, ccfg = PPOConfig(n_epochs=1, gamma=0.9), CTDEConfig(gae_lambda=0.8)
    reports = []
    CTDEUpdater(policy, build_central_critic(), cfg, ccfg).update(
        records, credit_sink=reports.append)
    batch = batches[0]
    assert reports[0].batch is batch
    rows = GT._credit_rows(reports[0], iteration=0, updates_completed_before=0,
                           measurement_tags={})
    assert [r["normalized_advantage"] for r in rows] == seen
    for e in range(2):
        idx = [i for i, r in enumerate(rows) if r["episode_index"] == e]
        v = [rows[i]["value_old"] for i in idx]
        rew = [rows[i]["transition_reward"] for i in idx]
        assert rew == [0.0, 0.0, [-0.2, -0.7][e]]
        running = 0.0
        for pos in reversed(range(3)):
            v_next = v[pos + 1] if pos < 2 else 0.0
            delta = rew[pos] + 0.9 * v_next - v[pos]
            running = delta + 0.9 * 0.8 * running
            row = rows[idx[pos]]
            assert row["td_residual"] == pytest.approx(delta, abs=1e-12)
            assert row["raw_advantage"] == pytest.approx(running, abs=1e-12)
            assert row["value_target"] == pytest.approx(running + v[pos], abs=1e-12)
            assert row["episode_decision_ordinal"] == pos
    for i, row in enumerate(rows):
        assert row["value_old"] == float(batch.values[i])
        assert row["td_residual"] == float(batch.td_residuals[i])
        assert row["value_target"] == float(batch.value_targets[i])
        assert row["gae_lambda"] == 0.8
        assert row["return"] is None and row["actor_only_episode_baseline"] is None
        assert row["ego_chain_ordinal"] is None
        assert row["measurement_join"]["joined"] is False


# =============================================================================
# C3: observational guarantees
# =============================================================================

def _count_forwards(*modules):
    counter = {"n": 0}

    def hook(_m, _i, _o):
        counter["n"] += 1

    handles = [m.register_forward_hook(hook) for m in modules]
    return counter, handles


def _state_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(
        (torch.equal(sa[k], sb[k]) if torch.is_tensor(sa[k]) else sa[k] == sb[k])
        for k in sa)


def _optim_equal(a, b):
    sa, sb = a.state_dict()["state"], b.state_dict()["state"]
    return sa.keys() == sb.keys() and all(
        all(torch.equal(sa[k][f], sb[k][f]) for f in sa[k]) for k in sa)


@pytest.mark.parametrize("mode", ["actor_only", "ctde"])
def test_c3_sink_on_off_changes_nothing_and_adds_no_forward_rng_or_gradient(mode):
    base = _policy(6)
    records = _actor_only_records(base) if mode == "actor_only" else _ctde_records(base)
    torch.manual_seed(99)
    critic0 = build_central_critic()

    results = {}
    for sink_on in (False, True):
        policy, critic = copy.deepcopy(base), copy.deepcopy(critic0)
        if mode == "actor_only":
            updater = PPOUpdater(policy, PPOConfig(n_epochs=3))
            watched = (policy.encoder, policy.head)
        else:
            updater = CTDEUpdater(policy, critic, PPOConfig(n_epochs=3), CTDEConfig())
            watched = (policy.encoder, policy.head, critic)
        counter, handles = _count_forwards(*watched)
        reports = []
        torch.manual_seed(7)
        np.random.seed(7)
        diag = updater.update(records, **({"credit_sink": reports.append} if sink_on else {}))
        forwards_in_update = counter["n"]
        grads_before = [None if p.grad is None else p.grad.clone()
                        for m in watched for p in m.parameters()]
        rows = (GT._credit_rows(reports[0], iteration=0, updates_completed_before=0,
                                measurement_tags={}) if sink_on else [])
        for h in handles:
            h.remove()
        grads_after = [None if p.grad is None else p.grad.clone()
                       for m in watched for p in m.parameters()]
        results[sink_on] = dict(
            diag=diag, policy=policy, critic=critic, updater=updater,
            forwards=forwards_in_update, forwards_total=counter["n"],
            rng=torch.get_rng_state().clone(), np_rng=np.random.get_state()[1].copy(),
            rows=rows, grads=(grads_before, grads_after))

    off, on = results[False], results[True]
    assert off["diag"] == on["diag"]
    assert _state_equal(off["policy"].encoder, on["policy"].encoder)
    assert _state_equal(off["policy"].head, on["policy"].head)
    assert _optim_equal(off["updater"].optimizer, on["updater"].optimizer)
    if mode == "ctde":
        assert _state_equal(off["critic"], on["critic"])
        assert _optim_equal(off["updater"].critic_optimizer, on["updater"].critic_optimizer)
    # no extra forward: during the update, and none at all while building the rows
    assert off["forwards"] == on["forwards"] == on["forwards_total"]
    # no RNG consumed by the sink or the row builder
    assert torch.equal(off["rng"], on["rng"])
    assert np.array_equal(off["np_rng"], on["np_rng"])
    # no gradient created by building rows, and nothing tensor-valued in them
    before, after = on["grads"]
    assert all((b is None and a is None) or torch.equal(b, a) for b, a in zip(before, after))
    json.dumps(on["rows"])
    assert on["rows"] and all(not torch.is_tensor(v) for r in on["rows"] for v in r.values())


# =============================================================================
# C4 / C5: tag isolation and control-path isolation (AST, not substring)
# =============================================================================

_TAG_NAMES = {"severity", "condition", "cell", "fd_selected_ego_id", "fd_event_tick",
              "measurement_join", "measurement_tags", "credit_tags", "benchmark_profile"}


def test_c4_no_learning_object_carries_a_measurement_tag_field():
    for cls in (TL.Transition, GraphObservation, CGB.CentralGraphObservation,
                EpisodeRecord, CTDEEpisodeRecord, GP.AdvantageBatch,
                GP.CTDEAdvantageBatch, GP.CreditReport):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not names & _TAG_NAMES, (cls.__name__, names & _TAG_NAMES)


def _names(source):
    tree = ast.parse(source)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def test_c4_actor_critic_and_credit_code_never_reads_a_measurement_tag():
    from match_aou.rl.agent import graph_encoder
    from match_aou.rl.observation import graph_builder
    for mod in (GA, GP, graph_encoder, graph_builder, CGB):
        leaked = _names(inspect.getsource(mod)) & {"severity", "measurement_join",
                                                   "measurement_tags", "credit_tags",
                                                   "fd_selected_ego_id", "fd_event_tick"}
        assert not leaked, (mod.__name__, leaked)

    # In `train`, the tag map reaches ONLY the post-update persistence call.
    n_calls_with_tags = 0
    for call in [n for n in ast.walk(ast.parse(_dedent(inspect.getsource(GT.train))))
                 if isinstance(n, ast.Call)]:
        func = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(
            call.func, "id", "")
        uses = {n.id for a in list(call.args) + [k.value for k in call.keywords]
                for n in ast.walk(a) if isinstance(n, ast.Name)}
        if "credit_tags" in uses:
            assert func == "_persist_credit_diagnostics", func
            n_calls_with_tags += 1
    assert n_calls_with_tags == 1

    # the credit VALUES do not depend on the tags at all
    policy = _policy(8)
    reports = []
    PPOUpdater(policy, PPOConfig(n_epochs=1)).update(_actor_only_records(policy),
                                                     credit_sink=reports.append)
    a = GT._credit_rows(reports[0], iteration=0, updates_completed_before=0,
                        measurement_tags={})
    b = GT._credit_rows(reports[0], iteration=0, updates_completed_before=0,
                        measurement_tags={(r.episode_index, r.seed): {"severity": "severe"}
                                          for r in reports[0].records})
    strip = [{k: v for k, v in r.items() if k != "measurement_join"} for r in a]
    assert strip == [{k: v for k, v in r.items() if k != "measurement_join"} for r in b]


def _dedent(source):
    import textwrap
    return textwrap.dedent(source)


_CREDIT_SYMBOLS = {"_CREDIT_DIAGNOSTICS_FILENAME", "train_credit_diagnostics.jsonl",
                   "_credit_rows", "_persist_credit_diagnostics",
                   "_observed_credit_diagnostics", "credit_reports", "credit_sink",
                   "observed_credit_diagnostics"}


def test_c5_no_control_path_reads_the_credit_artifact():
    from match_aou.rl.training import graph_reward
    for mod in (graph_reward, TL):
        assert not _names(inspect.getsource(mod)) & _CREDIT_SYMBOLS, mod.__name__
    for fn in (GT._EarlyStoppingMonitor.observe, GT._EarlyStoppingMonitor.is_due,
               GT.save_checkpoint, GT._iteration_outcome, GT.evaluate,
               GT.evaluate_benchmark, GT._run_one_episode):
        assert not _names(_dedent(inspect.getsource(fn))) & _CREDIT_SYMBOLS, fn.__name__
    # graph_ppo HANDS the report to a sink and never reads anything back from it
    upd_src = _dedent(inspect.getsource(PPOUpdater.update))
    assert "credit_sink(" in upd_src and "= credit_sink" not in upd_src
    # inside graph_train the filename constant appears only in the writer / observer
    tree = ast.parse(inspect.getsource(GT))
    users = set()
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.ClassDef)):
            if "_CREDIT_DIAGNOSTICS_FILENAME" in _names(ast.unparse(top)):
                users.add(top.name)
    assert users == {"train", "build_run_summary", "_persist_credit_diagnostics",
                     "_observed_credit_diagnostics", "write_run_config"}, users


# =============================================================================
# C6: fail loud; provenance
# =============================================================================

def test_c6_credit_persistence_fails_loud(tmp_path):
    policy = _policy(9)
    reports = []
    diag = PPOUpdater(policy, PPOConfig(n_epochs=1)).update(_actor_only_records(policy),
                                                            credit_sink=reports.append)
    kw = dict(iteration=0, updates_completed_before=0, measurement_tags={})
    with pytest.raises(GT.CreditDiagnosticsError):          # the path is a directory
        GT._persist_credit_diagnostics(tmp_path, reports, diag, **kw)
    with pytest.raises(GT.CreditDiagnosticsError):          # productive, no report
        GT._persist_credit_diagnostics(tmp_path / "c.jsonl", [], diag, **kw)
    with pytest.raises(GT.CreditDiagnosticsError):          # rows do not cover the batch
        GT._persist_credit_diagnostics(tmp_path / "c.jsonl", reports,
                                       dict(diag, n_transitions=1), **kw)
    assert GT._persist_credit_diagnostics(
        tmp_path / "c.jsonl", [], {"n_epochs_run": 0, "n_transitions": 0}, **kw) == 0
    assert GT._persist_credit_diagnostics(tmp_path / "c.jsonl", reports, diag, **kw) == 9


def test_c6_run_config_names_the_representation_and_the_credit_schema(tmp_path):
    cfg = GT.TrainConfig(n_iterations=1, output_dir=str(tmp_path))
    data = json.loads(Path(GT.write_run_config(
        tmp_path, cfg, provenance={"git": {"available": True}})).read_text("utf-8"))
    assert data["training"]["action_representation_id"] == ACTION_REPRESENTATION_ID
    assert data["training"]["credit_diagnostics"] == {
        "artifact": "train_credit_diagnostics.jsonl",
        "schema": GT._CREDIT_DIAGNOSTICS_SCHEMA, "schema_version": 1}


# =============================================================================
# C7: end to end through the real trainer and the REAL updaters
# =============================================================================

@pytest.mark.parametrize("mode", [GT.TRAINING_MODE_ACTOR_ONLY, GT.TRAINING_MODE_CTDE])
def test_c7_training_writes_one_credit_row_per_updated_transition(tmp_path, mode):
    cfg = GT.TrainConfig(n_iterations=2, episodes_per_iteration=2, eval_every=0,
                         eval_episodes=0, output_dir=str(tmp_path / mode),
                         training_mode=mode)
    CT._run_stub_training(cfg, ["_run_one_episode", "_build_generator", "_git_provenance"])
    run = Path(cfg.output_dir)
    train = [json.loads(line) for line in (run / "train_records.jsonl").read_text(
        "utf-8").splitlines() if line]
    rows = [json.loads(line) for line in (run / "train_credit_diagnostics.jsonl").read_text(
        "utf-8").splitlines() if line]
    assert len(rows) == sum(r["n_transitions"] for r in train if r["n_epochs_run"]) > 0
    assert {r["training_mode"] for r in rows} == {mode}
    assert {r["iteration"] for r in rows} == {0, 1}
    null_keys = (("value_old", "td_residual", "value_target") if mode == "actor_only"
                 else ("return", "actor_only_episode_baseline"))
    assert all(r[k] is None for r in rows for k in null_keys)
    assert all(r["measurement_join"]["joined"] for r in rows)
    summary = json.loads((run / "run_summary.json").read_text("utf-8"))
    assert summary["observed_credit_diagnostics"]["n_rows"] == len(rows)
    assert summary["observed_credit_diagnostics"]["action_representation_ids_observed"] == [
        ACTION_REPRESENTATION_ID]
    config = json.loads((run / "run_config.json").read_text("utf-8"))
    assert config["training"]["action_representation_id"] == ACTION_REPRESENTATION_ID


if __name__ == "__main__":  # pragma: no cover - direct runner
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
