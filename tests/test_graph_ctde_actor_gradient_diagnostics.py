"""OBSERVATIONAL EPOCH-0 CTDE ACTOR-GRADIENT DIAGNOSTICS (``train_actor_gradient_diagnostics.jsonl``).

Solver-free and BLADE-free; synthetic records only. Every test drives the real
production symbols (``CTDEUpdater.update``, ``graph_train._actor_gradient_*``).

  G1  diagnostic ON vs OFF: identical actor / critic parameters, optimizer states, RNG,
      forward counts, final ``.grad``, clipping and optimizer steps
  G2  the group ids never change the update (any opaque labelling -> the same update)
  G3  the total surrogate / actor-loss gradients are the REAL epoch-0 gradients
  G4  group gradients carry real batch mass and reconstruct the total
  G5  fd == mild + severe, non_fd == post_fd + ordinary
  G6  empty groups: explicit counts, null cosines / projections, no NaN
  G7  measurement labels cannot reach an actor / critic / credit input
  G8  fail loud: misuse, misalignment, unattributable FD wakes, persistence
  G9  configuration: OFF by default, CTDE only, recorded in run_config.json, CLI flag
  G10 end to end through the real trainer
  G11 the separation contrast is the semantic ABORT difference of the SAME epoch-0 logits
  G12 separation pressure / alignment match an independent autograd reference
  G13 sign semantics: positive pressure == a raw descent step raises the contrast
  G14 a missing MILD or SEVERE group makes every contrast field null (not 0 / NaN)
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_aou.rl.action import graph_action as GA  # noqa: E402
from match_aou.rl.training import graph_ppo as GP  # noqa: E402
from match_aou.rl.training import graph_tick_loop as TL  # noqa: E402
from match_aou.rl.training import graph_train as GT  # noqa: E402
from match_aou.rl.training.graph_ppo import (  # noqa: E402
    CTDEConfig,
    CTDEEpisodeRecord,
    CTDEUpdater,
    PPOConfig,
    build_central_critic,
)

import test_graph_ctde as CT  # noqa: E402
import test_graph_semantic_action_credit as SC  # noqa: E402

MILD, SEVERE, POST, ORD = range(4)
IMM = TL.WAKE_KIND_IMMEDIATE_FD
POSTK = TL.WAKE_KIND_POST_FD_BOUNDARY
ORDK = TL.WAKE_KIND_ORDINARY


# =============================================================================
# Fixtures
# =============================================================================

def _records(policy, *, with_fd=True):
    """Four episodes, one of them zero-wake, mixing all four measurement groups."""
    specs = ([  # (episode_index, seed, reward, [(ego, wake_kind)])
        (0, 300, -0.2, [("egoB", ORDK), ("egoA", IMM), ("egoA", POSTK)]),
        (1, 301, -0.8, []),
        (2, 302, -0.6, [("egoA", IMM), ("egoB", ORDK), ("egoA", POSTK), ("egoB", ORDK)]),
        (3, 303, -0.1, [("egoA", ORDK), ("egoB", ORDK)]),
    ] if with_fd else [
        (0, 300, -0.2, [("egoB", ORDK), ("egoA", ORDK)]),
        (1, 301, -0.7, [("egoA", ORDK), ("egoB", ORDK), ("egoA", ORDK)]),
    ])
    records = []
    for e, seed, reward, wakes in specs:
        gobs = SC._obs([SC._row(0.8 - 0.1 * e), SC._row(0.6), SC._row(0.5 + 0.05 * e),
                        SC._row(0.7)])
        traj = []
        for t, (ego, kind) in enumerate(wakes):
            tr = SC._sampled_transition(policy, gobs, ego=ego, tick=t + 1,
                                        seed=13 * e + t)
            tr.wake_kind = kind
            traj.append(tr)
        if traj:
            traj[-1].reward = reward
        states = [CT._synthetic_central(k=3, a=2, seed=17 * e + j) for j in range(len(traj))]
        records.append(CTDEEpisodeRecord.from_episode(traj, states, reward, seed=seed,
                                                      episode_index=e))
    return records


_TAGS = {
    (0, 300): {"cell": "mild", "condition": "damaged", "severity": "mild",
               "fd_selected_ego_id": "egoA", "fd_event_tick": 2},
    (1, 301): {"cell": "clean", "condition": "clean", "severity": None,
               "fd_selected_ego_id": None, "fd_event_tick": None},
    (2, 302): {"cell": "severe", "condition": "damaged", "severity": "severe",
               "fd_selected_ego_id": "egoA", "fd_event_tick": 1},
    (3, 303): {"cell": "clean", "condition": "clean", "severity": None,
               "fd_selected_ego_id": None, "fd_event_tick": None},
}


def _setup(seed=21):
    policy = SC._policy(seed)
    torch.manual_seed(seed + 1)
    critic = build_central_critic()
    return policy, critic


def _run(policy, critic, records, *, ids=None, cfg=None):
    cfg = cfg or PPOConfig(n_epochs=3)
    updater = CTDEUpdater(policy, critic, cfg, CTDEConfig())
    reports = []
    kwargs = {} if ids is None else {"gradient_group_ids": ids,
                                     "gradient_sink": reports.append,
                                     "gradient_contrast_ids": GT._actor_gradient_contrast_ids()}
    torch.manual_seed(7)
    np.random.seed(7)
    diag = updater.update(records, **kwargs)
    return updater, diag, reports


def _clip_spy(monkeypatch):
    calls = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        params = list(params)
        calls.append(([None if p.grad is None else p.grad.detach().clone() for p in params],
                      float(max_norm)))
        out = real(params, max_norm, *a, **k)
        calls[-1] = calls[-1] + (float(out),)
        return out

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    return calls


def _np_dot(a, b):
    # elementwise only: BLAS-backed numpy aborts next to torch on this Windows stack
    return float(np.sum(a * b))


def _np_norm(v):
    return _np_dot(v, v) ** 0.5


def _flat(grads, params):
    return np.concatenate([np.zeros(p.numel()) if g is None
                           else g.reshape(-1).double().numpy() for g, p in zip(grads, params)])


# =============================================================================
# G1 / G2: behaviour preservation
# =============================================================================

def test_g1_on_off_identical_update_state_rng_forwards_clipping_and_steps(monkeypatch):
    base_policy, base_critic = _setup()
    records = _records(base_policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    out = {}
    for on in (False, True):
        policy, critic = copy.deepcopy(base_policy), copy.deepcopy(base_critic)
        clip_calls = _clip_spy(monkeypatch)
        steps = {"n": 0}
        real_step = torch.optim.Adam.step

        def counting_step(self, *a, **k):
            steps["n"] += 1
            return real_step(self, *a, **k)

        monkeypatch.setattr(torch.optim.Adam, "step", counting_step)
        counter, handles = SC._count_forwards(policy.encoder, policy.head, critic)
        updater, diag, reports = _run(policy, critic, records, ids=ids if on else None)
        for h in handles:
            h.remove()
        monkeypatch.undo()
        out[on] = dict(policy=policy, critic=critic, updater=updater, diag=diag,
                       reports=reports, forwards=counter["n"], steps=steps["n"],
                       clip=clip_calls, rng=torch.get_rng_state().clone(),
                       np_rng=np.random.get_state()[1].copy(),
                       grads=[None if p.grad is None else p.grad.clone()
                              for p in updater.actor_parameters + updater.critic_parameters])
    off, on = out[False], out[True]
    assert off["reports"] == [] and len(on["reports"]) == 1
    assert off["diag"] == on["diag"]
    assert SC._state_equal(off["policy"].encoder, on["policy"].encoder)
    assert SC._state_equal(off["policy"].head, on["policy"].head)
    assert SC._state_equal(off["critic"], on["critic"])
    assert SC._optim_equal(off["updater"].optimizer, on["updater"].optimizer)
    assert SC._optim_equal(off["updater"].critic_optimizer, on["updater"].critic_optimizer)
    assert torch.equal(off["rng"], on["rng"])
    assert np.array_equal(off["np_rng"], on["np_rng"])
    assert off["forwards"] == on["forwards"] > 0
    assert off["steps"] == on["steps"] == 2 * 3
    assert len(off["clip"]) == len(on["clip"]) == 2 * 3
    for (g_off, m_off, n_off), (g_on, m_on, n_on) in zip(off["clip"], on["clip"]):
        assert (m_off, n_off) == (m_on, n_on)
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(g_off, g_on))
    assert all((a is None and b is None) or torch.equal(a, b)
               for a, b in zip(off["grads"], on["grads"]))


def test_g2_any_opaque_labelling_yields_the_identical_update():
    base_policy, base_critic = _setup(22)
    records = _records(base_policy)
    n = sum(len(r.transitions) for r in records)
    states = []
    for ids in ([0] * n, list(range(n)), [7 - (i % 3) for i in range(n)]):
        policy, critic = copy.deepcopy(base_policy), copy.deepcopy(base_critic)
        updater, diag, reports = _run(policy, critic, records, ids=ids)
        assert sum(reports[0].group_counts.values()) == n
        states.append((policy, critic, updater, diag))
    for policy, critic, updater, diag in states[1:]:
        assert diag == states[0][3]
        assert SC._state_equal(policy.encoder, states[0][0].encoder)
        assert SC._state_equal(policy.head, states[0][0].head)
        assert SC._state_equal(critic, states[0][1])
        assert SC._optim_equal(updater.optimizer, states[0][2].optimizer)


# =============================================================================
# G3 -- G6: the numbers
# =============================================================================

def test_g3_totals_are_the_real_epoch_zero_gradients(monkeypatch):
    policy, critic = _setup(23)
    records = _records(policy)
    clip_calls = _clip_spy(monkeypatch)
    updater, _, reports = _run(policy, critic, records,
                               ids=GT._actor_gradient_group_ids(records, _TAGS))
    rep = reports[0]
    real_actor_epoch0 = _flat(clip_calls[0][0], updater.actor_parameters)
    assert rep.epoch == 0
    np.testing.assert_allclose(rep.total_actor_loss_grad, real_actor_epoch0,
                               rtol=1e-6, atol=1e-9)
    assert not np.allclose(rep.total_policy_surrogate_grad, 0.0)
    assert rep.total_policy_surrogate_grad.size == sum(
        p.numel() for p in updater.actor_parameters)

    # with no entropy bonus the actor-loss gradient IS the surrogate gradient
    policy0, critic0 = _setup(23)
    records0 = _records(policy0)
    _, _, rep0 = _run(policy0, critic0, records0, cfg=PPOConfig(n_epochs=1, entropy_coeff=0.0),
                      ids=GT._actor_gradient_group_ids(records0, _TAGS))
    np.testing.assert_allclose(rep0[0].total_actor_loss_grad,
                               rep0[0].total_policy_surrogate_grad, rtol=0, atol=1e-12)
    rec = GT._actor_gradient_record(rep0[0], iteration=0, updates_completed_before=0,
                                    measurement_tags=_TAGS)
    assert rec["cosine_policy_surrogate_vs_actor_loss"] == pytest.approx(1.0, abs=1e-12)


def test_g4_group_gradients_keep_batch_mass_and_reconstruct_the_total(monkeypatch):
    policy, critic = _setup(24)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    assert ids == [ORD, MILD, POST, SEVERE, ORD, POST, ORD, ORD, ORD]

    # capture the per-transition losses the update itself built, then differentiate a
    # group's share independently on a copy of the same state
    probe_policy = copy.deepcopy(policy)
    updater, _, reports = _run(policy, critic, records, ids=ids)
    rep = reports[0]
    n = len(ids)
    assert rep.group_counts == {ORD: 5, MILD: 1, POST: 2, SEVERE: 1}
    total = sum(rep.group_policy_surrogate_grads.values())
    # per element: float32 autograd round-off only; over the whole vector: tight
    np.testing.assert_allclose(total, rep.total_policy_surrogate_grad, rtol=1e-5, atol=1e-7)
    assert (_np_norm(total - rep.total_policy_surrogate_grad)
            <= 1e-6 * _np_norm(rep.total_policy_surrogate_grad))

    batch = rep.batch
    params = list(probe_policy.encoder.parameters()) + list(probe_policy.head.parameters())
    losses = []
    for i, tr in enumerate(batch.transitions):
        logits = probe_policy.head(probe_policy.encoder(tr.gobs))
        lp, _ = GP.evaluate_action(logits, GP.build_action_mask(tr.gobs), tr.meta_action,
                                   tr.node_v)
        losses.append(GP.clipped_surrogate(torch.exp(lp - float(tr.log_prob)),
                                           float(batch.advantages[i]), 0.2))
    post_idx = [i for i, g in enumerate(ids) if g == POST]
    manual = torch.autograd.grad(torch.stack([losses[i] for i in post_idx]).sum() / n,
                                 params, allow_unused=True)
    np.testing.assert_allclose(rep.group_policy_surrogate_grads[POST], _flat(manual, params),
                               rtol=1e-5, atol=1e-8)
    # NOT the group-size-normalized mean (which would be n / |G| = 4.5x larger)
    assert not np.allclose(rep.group_policy_surrogate_grads[POST] * n / len(post_idx),
                           rep.group_policy_surrogate_grads[POST])

    rec = GT._actor_gradient_record(rep, iteration=2, updates_completed_before=1,
                                    measurement_tags=_TAGS)
    assert rec["reconstruction_error_norm"] <= 1e-6 * rec["total_policy_surrogate_grad_norm"]
    assert rec["reconstruction_relative_error"] < 1e-6
    projections = sum(b["projection_on_total"] for b in rec["groups"].values())
    assert projections == pytest.approx(rec["total_policy_surrogate_grad_norm"], rel=1e-6)
    assert [rec["groups"][g]["n_transitions"] for g in GT._ACTOR_GRADIENT_GROUPS] == [1, 1, 2, 5]
    assert rec["groups"]["ordinary"]["batch_fraction"] == pytest.approx(5 / 9)
    for b in rec["groups"].values():
        assert -1.0 - 1e-9 <= b["cosine_vs_total"] <= 1.0 + 1e-9
    assert rec["total_actor_loss_grad_norm"] > 0
    json.dumps(rec, allow_nan=False)


def test_g5_derived_fd_and_non_fd_are_sums_of_their_primary_groups():
    policy, critic = _setup(25)
    records = _records(policy)
    _, _, reports = _run(policy, critic, records,
                         ids=GT._actor_gradient_group_ids(records, _TAGS))
    rep = reports[0]
    g = rep.group_policy_surrogate_grads
    total = rep.total_policy_surrogate_grad
    fd, non_fd = g[MILD] + g[SEVERE], g[POST] + g[ORD]
    rec = GT._actor_gradient_record(rep, iteration=0, updates_completed_before=0,
                                    measurement_tags=_TAGS)
    d = rec["derived"]
    assert d["fd"]["n_transitions"] == (rec["groups"]["immediate_fd_mild"]["n_transitions"]
                                        + rec["groups"]["immediate_fd_severe"]["n_transitions"])
    assert d["non_fd"]["n_transitions"] == (rec["groups"]["post_fd"]["n_transitions"]
                                            + rec["groups"]["ordinary"]["n_transitions"])
    assert rec["fd_grad_norm"] == d["fd"]["grad_norm"] == pytest.approx(_np_norm(fd))
    assert rec["non_fd_grad_norm"] == pytest.approx(_np_norm(non_fd))
    assert d["fd"]["projection_on_total"] == pytest.approx(
        rec["groups"]["immediate_fd_mild"]["projection_on_total"]
        + rec["groups"]["immediate_fd_severe"]["projection_on_total"], abs=1e-9)
    cos = _np_dot(fd, non_fd) / (_np_norm(fd) * _np_norm(non_fd))
    assert rec["cosine_fd_vs_non_fd"] == pytest.approx(cos, abs=1e-9)
    assert rec["projection_non_fd_on_fd"] == pytest.approx(
        _np_dot(non_fd, fd) / _np_norm(fd), abs=1e-9)
    assert d["fd"]["cosine_vs_total"] == pytest.approx(
        _np_dot(fd, total) / (_np_norm(fd) * _np_norm(total)), abs=1e-9)


def test_g6_empty_groups_are_explicit_counts_with_null_cosines_and_no_nan():
    policy, critic = _setup(26)
    records = _records(policy, with_fd=False)
    ids = GT._actor_gradient_group_ids(records, {})
    assert set(ids) == {ORD}
    _, _, reports = _run(policy, critic, records, ids=ids)
    rep = reports[0]
    assert set(rep.group_counts) == {ORD}
    rec = GT._actor_gradient_record(rep, iteration=0, updates_completed_before=0,
                                    measurement_tags={})
    text = json.dumps(rec, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    for name in ("immediate_fd_mild", "immediate_fd_severe", "post_fd"):
        b = rec["groups"][name]
        assert b["n_transitions"] == 0 and b["batch_fraction"] == 0.0
        assert b["grad_norm"] == 0.0
        assert b["cosine_vs_total"] is None and b["projection_on_total"] is None
    assert rec["derived"]["fd"]["n_transitions"] == 0
    assert rec["derived"]["fd"]["cosine_vs_total"] is None
    assert rec["cosine_fd_vs_non_fd"] is None and rec["projection_non_fd_on_fd"] is None
    ordinary = rec["groups"]["ordinary"]
    assert ordinary["n_transitions"] == rec["batch_n_transitions"] == 5
    assert ordinary["cosine_vs_total"] == pytest.approx(1.0, abs=1e-9)
    assert rec["reconstruction_error_norm"] == pytest.approx(0.0, abs=1e-9)

    # the undefined-value helpers never invent a zero
    z, v = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert GT._cosine(z, v) is None and GT._cosine(v, z) is None
    assert GT._projection(v, z) is None and GT._projection(z, v) == 0.0


def test_g6_empty_batch_emits_no_report():
    policy, critic = _setup(27)
    empty = [CTDEEpisodeRecord.from_episode([], [], -0.5, seed=1, episode_index=0)]
    _, diag, reports = _run(policy, critic, empty, ids=[])
    assert reports == [] and diag["n_epochs_run"] == 0
    assert GT._persist_actor_gradient_diagnostics(
        Path("unused.jsonl"), [], diag, iteration=0, updates_completed_before=0,
        measurement_tags={}) == 0


# =============================================================================
# G7: label isolation
# =============================================================================

_GROUP_WORDS = {"severity", "mild", "severe", "immediate_fd_mild", "immediate_fd_severe",
                "post_fd", "measurement_join", "measurement_tags", "credit_tags",
                "fd_selected_ego_id", "fd_event_tick", "is_fd_selected_ego"}


def test_g7_graph_ppo_and_learning_objects_never_see_a_measurement_label():
    ppo_names = {n.lower() for n in SC._names(inspect.getsource(GP))}
    assert not ppo_names & _GROUP_WORDS, ppo_names & _GROUP_WORDS
    for cls in (GP.ActorGradientReport, TL.Transition, CTDEEpisodeRecord,
                GP.CTDEAdvantageBatch):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not names & _GROUP_WORDS, (cls.__name__, names & _GROUP_WORDS)


def test_g7_in_train_the_tags_reach_only_the_trainer_side_join_and_writers():
    tree = ast.parse(SC._dedent(inspect.getsource(GT.train)))
    consumers = []
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(
            call.func, "id", "")
        direct = {n.id for a in list(call.args) + [k.value for k in call.keywords]
                  for n in ast.walk(a) if isinstance(n, ast.Name)}
        if "credit_tags" in direct:
            consumers.append(func)
        if func == "update":
            # the updater call is handed the buffer, sinks and the kwargs dict only
            assert "credit_tags" not in direct
    # (+ the actor-only step diagnostic's two trainer-side functions, which likewise hand
    # the updater opaque integer ids only; tests/test_graph_actor_step_diagnostics.py)
    assert sorted(consumers) == sorted(["_persist_credit_diagnostics",
                                        "_actor_gradient_group_ids",
                                        "_persist_actor_gradient_diagnostics",
                                        "_actor_step_group_ids",
                                        "_persist_actor_step_diagnostics"]), consumers


def test_g7_group_ids_are_plain_ints_and_do_not_feed_credit_or_losses(monkeypatch):
    policy, critic = _setup(28)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    assert all(type(i) is int for i in ids)

    seen_adv, seen_obs = [], []
    real_surrogate = GP.clipped_surrogate
    real_eval = GP.evaluate_action
    monkeypatch.setattr(GP, "clipped_surrogate",
                        lambda r, a, c: seen_adv.append(a) or real_surrogate(r, a, c))
    monkeypatch.setattr(GP, "evaluate_action",
                        lambda lo, m, a, v: seen_obs.append((lo.detach().clone(), a, v))
                        or real_eval(lo, m, a, v))
    out = []
    for labels in (ids, [ORD] * len(ids)):
        seen_adv.clear()
        seen_obs.clear()
        p, c = copy.deepcopy(policy), copy.deepcopy(critic)
        creds = []
        CTDEUpdater(p, c, PPOConfig(n_epochs=2), CTDEConfig()).update(
            records, credit_sink=creds.append, gradient_group_ids=labels,
            gradient_sink=lambda r: None)
        rows = GT._credit_rows(creds[0], iteration=0, updates_completed_before=0,
                               measurement_tags={})
        out.append((list(seen_adv), [(lo, a, v) for lo, a, v in seen_obs], rows))
    assert out[0][0] == out[1][0]
    assert all(torch.equal(x[0], y[0]) and x[1:] == y[1:] for x, y in zip(out[0][1], out[1][1]))
    assert out[0][2] == out[1][2]


# =============================================================================
# G8: fail loud
# =============================================================================

def test_g8_updater_refuses_half_or_misaligned_wiring():
    policy, critic = _setup(29)
    records = _records(policy)
    upd = CTDEUpdater(policy, critic, PPOConfig(n_epochs=1), CTDEConfig())
    with pytest.raises(ValueError):
        upd.update(records, gradient_group_ids=[0] * 9)
    with pytest.raises(ValueError):
        upd.update(records, gradient_sink=lambda r: None)
    with pytest.raises(ValueError):
        upd.update(records, gradient_group_ids=[0] * 8, gradient_sink=lambda r: None)


def test_g8_unattributable_immediate_fd_wakes_fail_loud():
    policy = SC._policy(30)
    records = _records(policy)
    for bad in ({k: v for k, v in _TAGS.items() if k != (2, 302)},
                {**_TAGS, (2, 302): dict(_TAGS[(2, 302)], fd_selected_ego_id="egoB")},
                {**_TAGS, (0, 300): dict(_TAGS[(0, 300)], severity=None)}):
        with pytest.raises(GT.ActorGradientDiagnosticsError):
            GT._actor_gradient_group_ids(records, bad)


def test_g8_misaligned_ids_and_persistence_failures_fail_loud(tmp_path):
    policy, critic = _setup(31)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    _, diag, reports = _run(policy, critic, records, ids=ids)
    kw = dict(iteration=0, updates_completed_before=0, measurement_tags=_TAGS)
    swapped = dataclasses.replace(reports[0], group_ids=tuple(reversed(ids)))
    with pytest.raises(GT.ActorGradientDiagnosticsError):
        GT._actor_gradient_record(swapped, **kw)
    with pytest.raises(GT.ActorGradientDiagnosticsError):         # path is a directory
        GT._persist_actor_gradient_diagnostics(tmp_path, reports, diag, **kw)
    with pytest.raises(GT.ActorGradientDiagnosticsError):         # productive, no report
        GT._persist_actor_gradient_diagnostics(tmp_path / "g.jsonl", [], diag, **kw)
    with pytest.raises(GT.ActorGradientDiagnosticsError):         # several reports
        GT._persist_actor_gradient_diagnostics(tmp_path / "g.jsonl", reports * 2, diag, **kw)
    with pytest.raises(GT.ActorGradientDiagnosticsError):         # count mismatch
        GT._persist_actor_gradient_diagnostics(tmp_path / "g.jsonl", reports,
                                               dict(diag, n_transitions=3), **kw)
    path = tmp_path / "g.jsonl"
    assert GT._persist_actor_gradient_diagnostics(path, reports, diag, **kw) == 1
    assert GT._persist_actor_gradient_diagnostics(path, reports, diag, **kw) == 1
    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["schema_version"] == 1


# =============================================================================
# G9 / G10: configuration and end to end
# =============================================================================

def test_g9_off_by_default_ctde_only_and_recorded_in_run_config(tmp_path):
    assert GT.TrainConfig(n_iterations=1).actor_gradient_diagnostics is False
    with pytest.raises(ValueError, match="actor_gradient_diagnostics"):
        GT.TrainConfig(n_iterations=1, actor_gradient_diagnostics=True).validate()
    GT.TrainConfig(n_iterations=1, training_mode=GT.TRAINING_MODE_CTDE,
                   actor_gradient_diagnostics=True).validate()
    for on in (False, True):
        cfg = GT.TrainConfig(n_iterations=1, output_dir=str(tmp_path),
                             training_mode=GT.TRAINING_MODE_CTDE,
                             actor_gradient_diagnostics=on)
        data = json.loads(Path(GT.write_run_config(
            tmp_path, cfg, provenance={"git": {"available": True}})).read_text("utf-8"))
        block = data["training"]["actor_gradient_diagnostics"]
        assert block["enabled"] is on
        assert block["artifact"] == "train_actor_gradient_diagnostics.jsonl"
        assert (block["schema"], block["schema_version"]) == (
            "graph_train_actor_gradient_diagnostics", 1)
        assert block["groups"] == ["immediate_fd_mild", "immediate_fd_severe", "post_fd",
                                   "ordinary"]
        assert data["train_config"]["actor_gradient_diagnostics"] is on
    parser = GT._build_arg_parser()
    assert parser.parse_args(["--iterations", "1"]).actor_gradient_diagnostics is False
    assert parser.parse_args(["--iterations", "1", "--actor-gradient-diagnostics"]
                             ).actor_gradient_diagnostics is True
    assert GT._CLI_FIELD_BY_DEST["actor_gradient_diagnostics"] == "actor_gradient_diagnostics"


@pytest.mark.parametrize("on", [False, True])
def test_g10_training_writes_one_record_per_productive_update_only_when_enabled(tmp_path, on):
    cfg = GT.TrainConfig(n_iterations=2, episodes_per_iteration=2, eval_every=0,
                         eval_episodes=0, output_dir=str(tmp_path / str(on)),
                         training_mode=GT.TRAINING_MODE_CTDE,
                         actor_gradient_diagnostics=on)
    CT._run_stub_training(cfg, ["_run_one_episode", "_build_generator", "_git_provenance"])
    run = Path(cfg.output_dir)
    path = run / "train_actor_gradient_diagnostics.jsonl"
    if not on:
        assert not path.exists()
        return
    train = [json.loads(line) for line in (run / "train_records.jsonl").read_text(
        "utf-8").splitlines() if line]
    recs = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    productive = [r for r in train if r["n_epochs_run"]]
    assert len(recs) == len(productive) == 2
    assert [r["iteration"] for r in recs] == [0, 1]
    for rec, tr in zip(recs, productive):
        assert rec["batch_n_transitions"] == tr["n_transitions"]
        assert rec["groups"]["ordinary"]["n_transitions"] == tr["n_transitions"]
        assert rec["reconstruction_relative_error"] < 1e-6
        assert rec["separation_contrast"] is None      # the stub batch has no FD wake


# =============================================================================
# G11 -- G14: local severity-separation pressure
# =============================================================================

def _p_abort(logits, mask):
    _, dist, _ = GA._semantic_dist(logits, mask)
    return dist.probs[GA.SEMANTIC_ABORT_LEAF]


def _reference(policy, batch, ids):
    """Independent autograd reference on a COPY of the pre-update actor."""
    params = list(policy.encoder.parameters()) + list(policy.head.parameters())
    n = len(ids)
    losses, pos, neg = [], [], []
    for i, tr in enumerate(batch.transitions):
        logits = policy.head(policy.encoder(tr.gobs))
        mask = GP.build_action_mask(tr.gobs)
        lp, ent = GP.evaluate_action(logits, mask, tr.meta_action, tr.node_v)
        losses.append((GP.clipped_surrogate(torch.exp(lp - float(tr.log_prob)),
                                            float(batch.advantages[i]), 0.2), ent))
        if ids[i] == SEVERE:
            pos.append(_p_abort(logits, mask))
        elif ids[i] == MILD:
            neg.append(_p_abort(logits, mask))
    contrast = torch.stack(pos).mean() - torch.stack(neg).mean()

    def grad(t):
        return _flat(torch.autograd.grad(t, params, retain_graph=True, allow_unused=True),
                     params)

    h = grad(contrast)
    comps = {name: grad(sum(losses[i][0] for i, g in enumerate(ids) if g == gid) / n)
             for gid, name in enumerate(GT._ACTOR_GRADIENT_GROUPS)}
    surrogate = torch.stack([l for l, _ in losses]).mean()
    comps["total"] = grad(surrogate)
    comps["actor"] = grad(surrogate - 0.01 * torch.stack([e for _, e in losses]).mean())
    comps["fd"] = comps["immediate_fd_mild"] + comps["immediate_fd_severe"]
    comps["non_fd"] = comps["post_fd"] + comps["ordinary"]
    return float(contrast.item()), h, comps, params


def test_g11_contrast_is_the_semantic_abort_difference_of_the_same_epoch0_logits(
        monkeypatch):
    policy, critic = _setup(41)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    seen = []
    real = GP.evaluate_action

    def spy(logits, mask, meta, node):
        seen.append((logits, mask))
        return real(logits, mask, meta, node)

    monkeypatch.setattr(GP, "evaluate_action", spy)
    _, _, reports = _run(policy, critic, records, ids=ids)
    epoch0 = seen[:len(ids)]                    # the calls epoch 0 itself made
    with torch.no_grad():
        sev = [float(_p_abort(lo, m)) for (lo, m), g in zip(epoch0, ids) if g == SEVERE]
        mild = [float(_p_abort(lo, m)) for (lo, m), g in zip(epoch0, ids) if g == MILD]
    expected = sum(sev) / len(sev) - sum(mild) / len(mild)
    rep = reports[0]
    assert rep.contrast_ids == (SEVERE, MILD)
    assert rep.contrast_value == pytest.approx(expected, abs=1e-7)
    assert rep.contrast_grad is not None and _np_norm(rep.contrast_grad) > 0
    rec = GT._actor_gradient_record(rep, iteration=0, updates_completed_before=0,
                                    measurement_tags=_TAGS)
    assert rec["separation_contrast"] == rep.contrast_value
    assert rec["separation_contrast_grad_norm"] == pytest.approx(_np_norm(rep.contrast_grad))
    # no gradient vector is persisted: every leaf is a scalar, string or null
    def leaves(obj):
        for v in (obj.values() if isinstance(obj, dict) else [obj]):
            yield from (leaves(v) if isinstance(v, dict) else [v])
    assert all(v is None or isinstance(v, (int, float, str)) for v in leaves(rec))


def test_g12_pressure_and_alignment_match_an_independent_autograd_reference():
    policy, critic = _setup(42)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    probe = copy.deepcopy(policy)
    _, _, reports = _run(policy, critic, records, ids=ids)
    rec = GT._actor_gradient_record(reports[0], iteration=0, updates_completed_before=0,
                                    measurement_tags=_TAGS)
    contrast, h, comps, _ = _reference(probe, reports[0].batch, ids)
    assert rec["separation_contrast"] == pytest.approx(contrast, abs=1e-6)
    assert rec["separation_contrast_grad_norm"] == pytest.approx(_np_norm(h), rel=1e-4)

    def check(block_pressure, block_alignment, g):
        assert block_pressure == pytest.approx(-_np_dot(h, g), rel=1e-4, abs=1e-9)
        assert block_alignment == pytest.approx(
            _np_dot(-g, h) / (_np_norm(g) * _np_norm(h)), rel=1e-4, abs=1e-6)

    for name in GT._ACTOR_GRADIENT_GROUPS:
        b = rec["groups"][name]
        check(b["separation_pressure"], b["separation_alignment"], comps[name])
    for name in ("fd", "non_fd"):
        b = rec["derived"][name]
        check(b["separation_pressure"], b["separation_alignment"], comps[name])
    check(rec["total_policy_surrogate_separation_pressure"],
          rec["total_policy_surrogate_separation_alignment"], comps["total"])
    check(rec["total_actor_loss_separation_pressure"],
          rec["total_actor_loss_separation_alignment"], comps["actor"])
    # additivity: pressure is linear in g
    assert sum(rec["groups"][g]["separation_pressure"] for g in GT._ACTOR_GRADIENT_GROUPS) \
        == pytest.approx(rec["total_policy_surrogate_separation_pressure"], rel=1e-5)
    json.dumps(rec, allow_nan=False)


def test_g13_positive_pressure_means_a_raw_descent_step_raises_the_contrast():
    policy, critic = _setup(43)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    probe = copy.deepcopy(policy)
    _, _, reports = _run(policy, critic, records, ids=ids)
    rec = GT._actor_gradient_record(reports[0], iteration=0, updates_completed_before=0,
                                    measurement_tags=_TAGS)
    batch = reports[0].batch
    contrast0, _, comps, _ = _reference(copy.deepcopy(probe), batch, ids)

    def contrast_after_step(g, eps):
        stepped = copy.deepcopy(probe)
        params = list(stepped.encoder.parameters()) + list(stepped.head.parameters())
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p -= eps * torch.as_tensor(g[offset:offset + n], dtype=p.dtype).view_as(p)
                offset += n
        return _reference(stepped, batch, ids)[0]

    signs = set()
    cases = [(rec["groups"][g]["separation_pressure"], comps[g], g)
             for g in GT._ACTOR_GRADIENT_GROUPS]
    cases += [(rec["derived"][g]["separation_pressure"], comps[g], g) for g in ("fd", "non_fd")]
    cases.append((rec["total_policy_surrogate_separation_pressure"], comps["total"], "total"))
    for pressure, g, name in cases:
        if abs(pressure) < 1e-5:
            continue
        eps = 1e-2 / max(_np_norm(g), 1e-12)          # a small raw descent step
        delta = contrast_after_step(g, eps) - contrast0
        assert np.sign(delta) == np.sign(pressure), (name, pressure, delta)
        assert delta / eps == pytest.approx(pressure, rel=0.25), (name, pressure, delta / eps)
        signs.add(np.sign(pressure))
    for b in list(rec["groups"].values()) + list(rec["derived"].values()):
        if b["separation_alignment"] is not None:
            assert np.sign(b["separation_alignment"]) == np.sign(b["separation_pressure"])
    assert signs == {-1.0, 1.0}, "fixture must exercise both signs"


@pytest.mark.parametrize("keep", ["severe", "mild"])
def test_g14_missing_mild_or_severe_nulls_every_contrast_field(tmp_path, keep):
    policy, critic = _setup(44)
    records = _records(policy)
    tags = {k: dict(v) for k, v in _TAGS.items()}
    tags[(0, 300)]["severity"] = keep                   # both FD episodes share one severity
    tags[(2, 302)]["severity"] = keep
    ids = GT._actor_gradient_group_ids(records, tags)
    assert (MILD in ids) != (SEVERE in ids)
    _, diag, reports = _run(policy, critic, records, ids=ids)
    rep = reports[0]
    assert rep.contrast_value is None and rep.contrast_grad is None
    rec = GT._actor_gradient_record(rep, iteration=0, updates_completed_before=0,
                                    measurement_tags=tags)
    for key in ("separation_contrast", "separation_contrast_grad_norm",
                "total_policy_surrogate_separation_alignment",
                "total_policy_surrogate_separation_pressure",
                "total_actor_loss_separation_alignment", "total_actor_loss_separation_pressure"):
        assert rec[key] is None, key
    for b in list(rec["groups"].values()) + list(rec["derived"].values()):
        assert b["separation_alignment"] is None and b["separation_pressure"] is None
    # the decomposition itself is still written normally
    assert rec["reconstruction_relative_error"] < 1e-6
    assert rec["groups"]["immediate_fd_%s" % keep]["n_transitions"] == 2
    text = json.dumps(rec, allow_nan=False)
    assert "NaN" not in text
    path = tmp_path / "g.jsonl"
    assert GT._persist_actor_gradient_diagnostics(path, reports, diag, iteration=0,
                                                  updates_completed_before=0,
                                                  measurement_tags=tags) == 1


def test_g14_contrast_wiring_is_checked_both_ways():
    policy, critic = _setup(45)
    records = _records(policy)
    ids = GT._actor_gradient_group_ids(records, _TAGS)
    upd = CTDEUpdater(policy, critic, PPOConfig(n_epochs=1), CTDEConfig())
    with pytest.raises(ValueError):
        upd.update(records, gradient_contrast_ids=(1, 0))
    with pytest.raises(ValueError):
        upd.update(records, gradient_group_ids=ids, gradient_sink=lambda r: None,
                   gradient_contrast_ids=(1, 1))
    _, _, reports = _run(copy.deepcopy(policy), copy.deepcopy(critic), records, ids=ids)
    kw = dict(iteration=0, updates_completed_before=0, measurement_tags=_TAGS)
    for bad in (dataclasses.replace(reports[0], contrast_ids=None),
                dataclasses.replace(reports[0], contrast_ids=(0, 1)),
                dataclasses.replace(reports[0], contrast_grad=None, contrast_value=None)):
        with pytest.raises(GT.ActorGradientDiagnosticsError):
            GT._actor_gradient_record(bad, **kw)


if __name__ == "__main__":  # pragma: no cover - direct runner
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
