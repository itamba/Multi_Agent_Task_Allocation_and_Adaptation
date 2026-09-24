"""OBSERVATIONAL EVERY-EPOCH ACTOR-ONLY STEP DIAGNOSTICS (``train_actor_step_diagnostics.jsonl``).

Solver-free and BLADE-free; synthetic records only. Every test drives the real production
symbols (``PPOUpdater.update``, ``graph_train._actor_step_*``).

  A1  instrumentation OFF == ON == the BASE updater (extracted from the verified base
      commit): final parameters, Adam state, updater outputs and RNG after all epochs of
      two consecutive updates, with positive / negative advantages and later-epoch clipping
  A2  any opaque labelling or contrast pair yields the identical update
  A3  attribution follows the ACTUAL batch identities (interleaved egos, several chains);
      every transition counted once; trajectory-order ids are refused
  A4  empty batch, one missing severity, empty groups and a zero contrast gradient
  A5  group gradients sum to the real surrogate gradient, adding the entropy component
      gives the real backward gradient before clipping, at EVERY epoch (independent
      autograd reference at the reconstructed pre-step parameters)
  A6  the displacement is the ACTUAL Adam step (clip + moments), epoch deltas telescope
  A7  sign conventions: pressure == raw-descent first-order change; the displacement
      projection predicts the observed contrast change for a small step
  A8  fail loud: misuse, misalignment, non-finite values, disagreement with the update,
      persistence and write-once vector files
  A9  configuration: OFF by default, actor_only only, recorded, CLI flags
  A10 end to end through the real trainer: ON / OFF identical learning, one record per
      productive update, vector files, faults stop the run (never attrition)
  A11 labels never reach graph_ppo or a learning object; nothing reads the artifact back
"""
from __future__ import annotations

import __future__
import ast
import copy
import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_aou.rl.action import graph_action as GA  # noqa: E402
from match_aou.rl.action.graph_action import MetaAction  # noqa: E402
from match_aou.rl.training import graph_ppo as GP  # noqa: E402
from match_aou.rl.training import graph_tick_loop as TL  # noqa: E402
from match_aou.rl.training import graph_train as GT  # noqa: E402
from match_aou.rl.training.graph_ppo import EpisodeRecord, PPOConfig, PPOUpdater  # noqa: E402
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    resolve_condition,
    resolve_severity,
)

import test_graph_ctde as CT  # noqa: E402
import test_graph_semantic_action_credit as SC  # noqa: E402

MILD, SEVERE, POST, ORD = range(4)
IMM = TL.WAKE_KIND_IMMEDIATE_FD
POSTK = TL.WAKE_KIND_POST_FD_BOUNDARY
ORDK = TL.WAKE_KIND_ORDINARY
PLAN = int(MetaAction.PLAN_COMPLIANCE)
ABORT = int(MetaAction.SELF_PRESERVATION_ABORT)

# The verified base of this task: the base updater is extracted from it (A1).
BASE_SHA = "bcb1746fbc677b3109f73b36699ac3b3780c32a4"


# =============================================================================
# Fixtures
# =============================================================================

# (episode_index, seed, reward, [(ego, wake_kind, forced action or None, ego_nodes)])
_SPECS = [
    (0, 500, -0.15, [("egoA", ORDK, None, (0, 1)), ("egoB", ORDK, None, (0, 1)),
                     ("egoA", IMM, ABORT, (0, 1)), ("egoB", ORDK, None, (0, 1)),
                     ("egoA", POSTK, None, (0, 1))]),
    (1, 501, -0.9, []),
    (2, 502, -0.7, [("egoB", ORDK, None, (0, 1)), ("egoA", IMM, PLAN, (0, 1)),
                    ("egoB", ORDK, None, (0, 1)), ("egoA", POSTK, None, (0, 1))]),
    (3, 503, -0.05, [("egoA", ORDK, None, (0, 1)), ("egoB", ORDK, None, (0, 1))]),
    (4, 504, -0.55, [("egoB", IMM, PLAN, (0, 1)), ("egoA", ORDK, None, (0, 1))]),
]
_TAGS = {
    (0, 500): {"cell": "severe", "condition": "damaged", "severity": "severe",
               "fd_selected_ego_id": "egoA", "fd_event_tick": 2},
    (1, 501): {"cell": "clean", "condition": "clean", "severity": None,
               "fd_selected_ego_id": None, "fd_event_tick": None},
    (2, 502): {"cell": "mild", "condition": "damaged", "severity": "mild",
               "fd_selected_ego_id": "egoA", "fd_event_tick": 1},
    (3, 503): {"cell": "clean", "condition": "clean", "severity": None,
               "fd_selected_ego_id": None, "fd_event_tick": None},
    (4, 504): {"cell": "mild", "condition": "damaged", "severity": "mild",
               "fd_selected_ego_id": "egoB", "fd_event_tick": 0},
}


def _records(policy, specs=_SPECS, *, fd_ego_nodes=None):
    records = []
    for e, seed, reward, wakes in specs:
        traj = []
        for t, (ego, kind, force, ego_nodes) in enumerate(wakes):
            nodes = fd_ego_nodes if (fd_ego_nodes is not None and kind == IMM) else ego_nodes
            gobs = SC._obs([SC._row(0.8 - 0.07 * e - 0.02 * t), SC._row(0.6 + 0.03 * t),
                            SC._row(0.5 + 0.05 * e), SC._row(0.7)], ego_nodes=nodes)
            tr = SC._sampled_transition(policy, gobs, ego=ego, tick=t + 1,
                                        seed=19 * e + t,
                                        force=None if force is None else (force, None))
            tr.wake_kind = kind
            traj.append(tr)
        if traj:
            traj[-1].reward = reward
        records.append(EpisodeRecord.from_trajectory(traj, reward, seed=seed,
                                                     episode_index=e))
    return records


def _ids(records, tags=_TAGS):
    return GT._actor_step_group_ids(records, tags)


def _kwargs(records, tags=_TAGS, sink=None):
    return {"step_group_ids": _ids(records, tags),
            "step_sink": sink if sink is not None else (lambda r: None),
            "step_contrast_ids": GT._actor_gradient_contrast_ids()}


def _base_updater_class():
    """The BASE ``PPOUpdater`` class, compiled from the verified base commit's source in
    the current module's namespace (every helper it calls is unchanged by this task)."""
    try:
        src = subprocess.run(
            ["git", "show", "%s:src/match_aou/rl/training/graph_ppo.py" % BASE_SHA],
            cwd=str(ROOT), capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip("base commit %s unavailable: %s" % (BASE_SHA, exc))
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "PPOUpdater")
    namespace = dict(vars(GP))
    code = compile(ast.get_source_segment(src, node), "<base PPOUpdater>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)
    return namespace["PPOUpdater"]


def _flat_params(policy):
    return np.concatenate([p.detach().reshape(-1).double().numpy()
                           for p in list(policy.encoder.parameters())
                           + list(policy.head.parameters())])


def _np_dot(a, b):
    return float(np.sum(a * b))


def _np_norm(v):
    return _np_dot(v, v) ** 0.5


_HIGH_LR = PPOConfig(lr=0.03, n_epochs=4)   # forces later-epoch clipping on tiny batches


def _update_twice(updater_cls, policy, records, cfg, *, instrument, reports=None):
    upd = updater_cls(policy, cfg)
    diags = []
    torch.manual_seed(11)
    np.random.seed(11)
    for _ in range(2):
        kw = {}
        if instrument:
            kw = _kwargs(records, sink=(reports.append if reports is not None else None))
        diags.append(upd.update(records, **kw))
    return upd, diags


# =============================================================================
# A1 / A2: behaviour preservation against the BASE updater
# =============================================================================

def test_a1_off_on_and_base_updaters_are_identical():
    base_cls = _base_updater_class()
    policy0 = SC._policy(31)
    records = _records(policy0)
    batch = GP.compute_returns_and_advantages(records, _HIGH_LR)
    assert (batch.advantages > 0).any() and (batch.advantages < 0).any()
    out = {}
    for name, cls, instrument in (("base", base_cls, False), ("off", PPOUpdater, False),
                                  ("on", PPOUpdater, True)):
        policy = copy.deepcopy(policy0)
        reports = []
        upd, diags = _update_twice(cls, policy, records, _HIGH_LR, instrument=instrument,
                                   reports=reports)
        out[name] = dict(policy=policy, upd=upd, diags=diags, reports=reports,
                         rng=torch.get_rng_state().clone(),
                         np_rng=np.random.get_state()[1].copy())
    # later epochs really clip, and the clamped branch binds for some transitions
    assert max(out["base"]["diags"][0]["per_epoch"]["clip_fraction"][1:]) > 0.0
    for name in ("off", "on"):
        assert out[name]["diags"] == out["base"]["diags"], name
        assert SC._state_equal(out[name]["policy"].encoder, out["base"]["policy"].encoder)
        assert SC._state_equal(out[name]["policy"].head, out["base"]["policy"].head)
        assert SC._optim_equal(out[name]["upd"].optimizer, out["base"]["upd"].optimizer)
        assert (out[name]["upd"].optimizer.state_dict()["param_groups"]
                == out["base"]["upd"].optimizer.state_dict()["param_groups"])
        assert torch.equal(out[name]["rng"], out["base"]["rng"])
        assert np.array_equal(out[name]["np_rng"], out["base"]["np_rng"])
    assert out["off"]["reports"] == [] and len(out["on"]["reports"]) == 2
    assert all(len(r.epochs) == 4 for r in out["on"]["reports"])
    rec = GT._actor_step_record(out["on"]["reports"][0], out["on"]["diags"][0], iteration=0,
                                updates_completed_before=0, measurement_tags=_TAGS)
    binding = [ep["ratio"]["all"]["fraction_clip_binding"] for ep in rec["epochs"]]
    assert binding[0] == 0.0 and max(binding[1:]) > 0.0


def test_a1_extra_reads_consume_no_rng_and_leave_no_grad_difference(monkeypatch):
    policy0 = SC._policy(32)
    records = _records(policy0)
    grads = {}
    for instrument in (False, True):
        policy = copy.deepcopy(policy0)
        upd = PPOUpdater(policy, _HIGH_LR)
        torch.manual_seed(5)
        state = torch.get_rng_state().clone()
        upd.update(records, **(_kwargs(records) if instrument else {}))
        assert torch.equal(torch.get_rng_state(), state)       # nothing drew from RNG
        assert policy.encoder.training == policy0.encoder.training
        assert policy.head.training == policy0.head.training
        grads[instrument] = [None if p.grad is None else p.grad.clone()
                             for p in upd.parameters]
    assert all((a is None and b is None) or torch.equal(a, b)
               for a, b in zip(grads[False], grads[True]))


def test_a2_any_opaque_labelling_or_contrast_pair_gives_the_identical_update():
    policy0 = SC._policy(33)
    records = _records(policy0)
    n = sum(r.n_transitions for r in records)
    states = []
    for ids, contrast in (([0] * n, None), (list(range(n)), (0, 1)),
                          ([7 - (i % 3) for i in range(n)], (7, 6)),
                          (_ids(records), GT._actor_gradient_contrast_ids())):
        policy = copy.deepcopy(policy0)
        upd = PPOUpdater(policy, _HIGH_LR)
        got = []
        diag = upd.update(records, step_group_ids=ids, step_sink=got.append,
                          step_contrast_ids=contrast)
        assert len(got) == 1 and sum(got[0].group_counts.values()) == n
        states.append((policy, upd, diag))
    for policy, upd, diag in states[1:]:
        assert diag == states[0][2]
        assert SC._state_equal(policy.encoder, states[0][0].encoder)
        assert SC._state_equal(policy.head, states[0][0].head)
        assert SC._optim_equal(upd.optimizer, states[0][1].optimizer)


# =============================================================================
# A3 / A4: attribution and edge cases
# =============================================================================

def test_a3_ids_follow_the_actual_batch_identities_not_the_trajectory_order():
    policy = SC._policy(34)
    records = _records(policy)
    batch = GP.compute_returns_and_advantages(records, PPOConfig())
    ids = _ids(records)
    # the batch is chain-ordered (egoA's chain, then egoB's), NOT tick-interleaved
    assert [tr.ego_id for tr in batch.transitions[:5]] == ["egoA"] * 3 + ["egoB"] * 2
    assert ids[:5] == [ORD, SEVERE, POST, ORD, ORD]
    by_identity = []
    for i, tr in enumerate(batch.transitions):
        rec = records[batch.record_positions[i]]
        assert rec.chains[tr.ego_id][batch.chain_ordinals[i]] is tr
        by_identity.append(GT._ACTOR_GRADIENT_GROUPS.index(GT._actor_step_group(
            tr, _TAGS[(rec.episode_index, rec.seed)])))
    assert ids == by_identity
    assert len({id(tr) for tr in batch.transitions}) == len(ids) == sum(
        r.n_transitions for r in records)
    assert {g: ids.count(g) for g in range(4)} == {MILD: 2, SEVERE: 1, POST: 2, ORD: 8}

    # ids in the tick-interleaved TRAJECTORY order are misaligned and refused
    traj_ids = []
    for e, seed, _, wakes in _SPECS:
        for ego, kind, _, _ in wakes:
            fake = TL.Transition(gobs=None, ego_id=ego, tick=0, meta_action=0, node_v=None,
                                 log_prob=0.0, entropy=0.0, wake_kind=kind)
            traj_ids.append(GT._ACTOR_GRADIENT_GROUPS.index(
                GT._actor_step_group(fake, _TAGS[(e, seed)])))
    assert traj_ids != ids
    upd = PPOUpdater(copy.deepcopy(policy), PPOConfig(n_epochs=1))
    got = []
    diag = upd.update(records, step_group_ids=traj_ids, step_sink=got.append,
                      step_contrast_ids=GT._actor_gradient_contrast_ids())
    with pytest.raises(GT.ActorStepDiagnosticsError, match="aligned"):
        GT._actor_step_record(got[0], diag, iteration=0, updates_completed_before=0,
                              measurement_tags=_TAGS)


def test_a4_empty_batch_is_a_no_op_with_no_report(tmp_path):
    policy = SC._policy(35)
    empty = [EpisodeRecord.from_trajectory([], -0.4, seed=1, episode_index=0)]
    got = []
    diag = PPOUpdater(policy, PPOConfig()).update(
        empty, step_group_ids=[], step_sink=got.append,
        step_contrast_ids=GT._actor_gradient_contrast_ids())
    assert got == [] and diag["n_epochs_run"] == 0
    assert GT._persist_actor_step_diagnostics(
        tmp_path / "s.jsonl", [], diag, iteration=0, updates_completed_before=0,
        measurement_tags={}) == 0
    assert not (tmp_path / "s.jsonl").exists()


@pytest.mark.parametrize("keep", ["severe", "mild"])
def test_a4_one_missing_severity_nulls_every_contrast_field(keep):
    policy = SC._policy(36)
    records = _records(policy)
    tags = {k: dict(v) for k, v in _TAGS.items()}
    for key in ((0, 500), (2, 502), (4, 504)):
        tags[key]["severity"] = keep
    got = []
    diag = PPOUpdater(policy, _HIGH_LR).update(records, **_kwargs(records, tags, got.append))
    rep = got[0]
    assert all(e.contrast_grad is None and e.contrast_before is None for e in rep.epochs)
    rec = GT._actor_step_record(rep, diag, iteration=3, updates_completed_before=3,
                                measurement_tags=tags)
    other = "mild" if keep == "severe" else "severe"
    for ep in rec["epochs"]:
        assert ep["contrast"] is None
        assert ep["contrast_undefined_reason"] == "no_immediate_fd_%s_transition" % other
        assert ep["gradient"]["group_sum_relative_residual"] < 1e-5   # still decomposed
        assert ep["step"]["delta_theta_norm"] > 0
    s = rec["update_summary"]
    assert s["contrast_defined"] is False and s["full_update_delta"] is None
    assert s["n_immediate_fd_%s" % keep] == 3 and s["n_immediate_fd_%s" % other] == 0
    assert rec["groups"]["immediate_fd_%s" % other]["n_transitions"] == 0
    assert rec["groups"]["immediate_fd_%s" % other]["normalized_advantage"] is None
    json.dumps(rec, allow_nan=False)


def test_a4_empty_groups_have_zero_gradient_and_null_cosines():
    policy = SC._policy(37)
    specs = [(0, 600, -0.2, [("egoA", ORDK, None, (0, 1)), ("egoB", ORDK, None, (0, 1))]),
             (1, 601, -0.8, [("egoA", ORDK, None, (0, 1))])]
    records = _records(policy, specs)
    got = []
    diag = PPOUpdater(policy, PPOConfig()).update(records, **_kwargs(records, {}, got.append))
    rec = GT._actor_step_record(got[0], diag, iteration=0, updates_completed_before=0,
                                measurement_tags={})
    for ep in rec["epochs"]:
        for name in ("immediate_fd_mild", "immediate_fd_severe", "post_fd"):
            b = ep["gradient"]["groups"][name]
            assert b["n_transitions"] == 0 and b["grad_norm"] == 0.0
            assert b["cosine_vs_policy_surrogate"] is None
        assert ep["gradient"]["cosine_fd_vs_non_fd"] is None
        assert ep["gradient"]["groups"]["ordinary"]["cosine_vs_policy_surrogate"] == \
            pytest.approx(1.0, abs=1e-9)
        assert ep["contrast_undefined_reason"] == \
            "no_immediate_fd_severe_and_no_immediate_fd_mild_transition"
        assert ep["ratio"]["groups"]["post_fd"] is None


def test_a4_zero_contrast_gradient_keeps_pressure_defined_and_alignment_null():
    """ABORT illegal at every FD row: P(ABORT) is identically 0, so h == 0."""
    policy = SC._policy(38)
    specs = [(e, s, r, [(ego, kind, (PLAN if kind == IMM else f), n)
                        for ego, kind, f, n in w]) for e, s, r, w in _SPECS]
    records = _records(policy, specs, fd_ego_nodes=())
    got = []
    diag = PPOUpdater(policy, PPOConfig()).update(records, **_kwargs(records, sink=got.append))
    rec = GT._actor_step_record(got[0], diag, iteration=0, updates_completed_before=0,
                                measurement_tags=_TAGS)
    for ep in rec["epochs"]:
        c = ep["contrast"]
        assert c["before"] == 0.0 and c["after"] == 0.0 and c["grad_norm"] == 0.0
        assert c["pressure"]["total_loss"]["pressure"] == 0.0
        assert c["pressure"]["total_loss"]["alignment"] is None
        assert c["cosine_delta_vs_contrast_grad"] is None
        assert c["actual_delta"] == 0.0 and c["linearization_residual"] == 0.0
    json.dumps(rec, allow_nan=False)


# =============================================================================
# A5 / A6: the numbers are the real ones
# =============================================================================

def _reference(policy, batch, ids, cfg):
    """Independent autograd reference on ``policy`` at its CURRENT parameters."""
    params = list(policy.encoder.parameters()) + list(policy.head.parameters())
    n = len(ids)
    losses, ents, pos, neg = [], [], [], []
    for i, tr in enumerate(batch.transitions):
        logits = policy.head(policy.encoder(tr.gobs))
        mask = GP.build_action_mask(tr.gobs)
        lp, ent = GP.evaluate_action(logits, mask, tr.meta_action, tr.node_v)
        losses.append(GP.clipped_surrogate(torch.exp(lp - float(tr.log_prob)),
                                           float(batch.advantages[i]), cfg.clip_ratio))
        ents.append(ent)
        p_abort = GA._semantic_dist(logits, mask)[1].probs[GA.SEMANTIC_ABORT_LEAF]
        (pos if ids[i] == SEVERE else neg if ids[i] == MILD else []).append(p_abort)

    def grad(t):
        return np.concatenate([
            np.zeros(p.numel()) if g is None else g.reshape(-1).double().numpy()
            for g, p in zip(torch.autograd.grad(t, params, retain_graph=True,
                                                allow_unused=True), params)])

    surrogate = torch.stack(losses).mean()
    total = surrogate - cfg.entropy_coeff * torch.stack(ents).mean()
    comps = {g: grad(sum(losses[i] for i, x in enumerate(ids) if x == g) / n)
             for g in set(ids)}
    contrast = torch.stack(pos).mean() - torch.stack(neg).mean()
    return dict(groups=comps, policy=grad(surrogate), total=grad(total),
                contrast=float(contrast.item()), h=grad(contrast))


def _set_flat(policy, flat):
    params = list(policy.encoder.parameters()) + list(policy.head.parameters())
    off = 0
    with torch.no_grad():
        for p in params:
            k = p.numel()
            p.copy_(torch.as_tensor(flat[off:off + k]).view_as(p).to(p.dtype))
            off += k


def test_a5_every_epoch_decomposition_matches_an_independent_reference(monkeypatch):
    policy0 = SC._policy(39)
    records = _records(policy0)
    ids = _ids(records)
    clip_seen = CT_clip_spy(monkeypatch)
    policy = copy.deepcopy(policy0)
    got = []
    diag = PPOUpdater(policy, _HIGH_LR).update(records, **_kwargs(records, sink=got.append))
    rep = got[0]
    theta = _flat_params(policy0)
    for e, ep in enumerate(rep.epochs):
        probe = copy.deepcopy(policy0)
        _set_flat(probe, theta)                        # the reconstructed pre-step state
        ref = _reference(probe, rep.batch, ids, _HIGH_LR)
        for g, v in ep.group_policy_surrogate_grads.items():
            np.testing.assert_allclose(v, ref["groups"][g], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(ep.policy_surrogate_grad, ref["policy"], rtol=1e-4,
                                   atol=1e-8)
        np.testing.assert_allclose(ep.total_loss_grad, ref["total"], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(ep.contrast_grad, ref["h"], rtol=1e-4, atol=1e-8)
        assert ep.contrast_before == pytest.approx(ref["contrast"], abs=1e-6)
        # sum of groups == surrogate; + entropy component == the real pre-clip backward
        group_sum = sum(ep.group_policy_surrogate_grads.values())
        assert _np_norm(group_sum - ep.policy_surrogate_grad) <= 1e-6 * _np_norm(
            ep.policy_surrogate_grad)
        real_pre_clip = clip_seen[e]
        np.testing.assert_allclose(ep.backward_grad, real_pre_clip, rtol=0, atol=0)
        entropy = ep.total_loss_grad - ep.policy_surrogate_grad
        assert _np_norm(group_sum + entropy - real_pre_clip) <= 1e-5 * _np_norm(real_pre_clip)
        theta = theta + ep.delta_theta
    np.testing.assert_array_equal(theta, _flat_params(policy))   # exact reconstruction
    rec = GT._actor_step_record(rep, diag, iteration=0, updates_completed_before=0,
                                measurement_tags=_TAGS)
    for ep in rec["epochs"]:
        assert ep["gradient"]["group_sum_relative_residual"] < 1e-5
        assert ep["gradient"]["total_loss_vs_backward_relative_residual"] < 1e-5
        assert ep["gradient"]["groups_plus_entropy_vs_backward_relative_residual"] < 1e-5
        pressures = ep["contrast"]["pressure"]
        assert sum(pressures["groups"][g]["pressure"] for g in GT._ACTOR_GRADIENT_GROUPS) \
            == pytest.approx(pressures["policy_surrogate"]["pressure"], rel=1e-4, abs=1e-12)
        assert (pressures["policy_surrogate"]["pressure"]
                + pressures["entropy_component"]["pressure"]) == pytest.approx(
                    pressures["total_loss"]["pressure"], rel=1e-9, abs=1e-15)


def CT_clip_spy(monkeypatch):
    """Record the real ``.grad`` handed to ``clip_grad_norm_`` (i.e. before clipping)."""
    seen = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        params = list(params)
        seen.append(np.concatenate([
            np.zeros(p.numel()) if p.grad is None else p.grad.detach().reshape(-1).double()
            .numpy() for p in params]))
        return real(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    return seen


def test_a6_displacement_is_the_actual_adam_step_and_epoch_deltas_telescope(monkeypatch):
    policy0 = SC._policy(40)
    records = _records(policy0)
    policy = copy.deepcopy(policy0)
    upd = PPOUpdater(policy, _HIGH_LR)
    observed = []
    real_step = torch.optim.Adam.step

    def spy(self, *a, **k):
        before = _flat_params(policy)
        grad = np.concatenate([np.zeros(p.numel()) if p.grad is None
                               else p.grad.detach().reshape(-1).double().numpy()
                               for p in upd.parameters])
        out = real_step(self, *a, **k)
        observed.append((before, grad, _flat_params(policy) - before))
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", spy)
    got = []
    diag = upd.update(records, **_kwargs(records, sink=got.append))
    diag2 = upd.update(records, **_kwargs(records, sink=got.append))  # moments carried over
    monkeypatch.undo()
    epochs = list(got[0].epochs) + list(got[1].epochs)
    assert len(observed) == len(epochs) == 8
    for ep, (_, grad, delta) in zip(epochs, observed):
        np.testing.assert_array_equal(ep.delta_theta, delta)
        np.testing.assert_array_equal(ep.clipped_grad, grad)    # what Adam consumed
    # the first step of a fresh Adam, by its formula, on the CLIPPED gradient
    b1, b2 = 0.9, 0.999
    g = epochs[0].clipped_grad
    m_hat, v_hat = (1 - b1) * g / (1 - b1), (1 - b2) * g * g / (1 - b2)
    expected = -_HIGH_LR.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # float32 Adam arithmetic vs this float64 formula differs only where |g| ~ eps
    np.testing.assert_allclose(epochs[0].delta_theta, expected, rtol=1e-4, atol=1e-6)
    # clipping really acted: the step used the scaled gradient
    assert epochs[0].pre_clip_grad_norm > _HIGH_LR.max_grad_norm
    assert _np_norm(epochs[0].clipped_grad) == pytest.approx(_HIGH_LR.max_grad_norm, rel=1e-5)
    for rep, d in ((got[0], diag), (got[1], diag2)):
        rec = GT._actor_step_record(rep, d, iteration=0, updates_completed_before=0,
                                    measurement_tags=_TAGS)
        s = rec["update_summary"]
        assert s["full_update_delta"] == pytest.approx(s["sum_epoch_deltas"], abs=1e-12)
        assert abs(s["telescoping_residual"]) <= 1e-12
        assert s["max_epoch_boundary_mismatch"] <= 1e-7
        assert len(s["epoch_deltas"]) == 4
        assert s["contrast_after_update"] - s["contrast_before_update"] == \
            pytest.approx(s["full_update_delta"], abs=1e-15)
        for ep in rec["epochs"]:
            st = ep["step"]
            assert st["clipped"] is True
            assert st["post_clip_grad_norm"] == pytest.approx(_HIGH_LR.max_grad_norm,
                                                              rel=1e-5)


# =============================================================================
# A7: sign conventions
# =============================================================================

def test_a7_positive_pressure_means_a_raw_descent_step_raises_the_contrast():
    policy0 = SC._policy(41)
    records = _records(policy0)
    ids = _ids(records)
    got = []
    diag = PPOUpdater(copy.deepcopy(policy0), PPOConfig()).update(
        records, **_kwargs(records, sink=got.append))
    rep = got[0]
    ep0 = rep.epochs[0]
    rec = GT._actor_step_record(rep, diag, iteration=0, updates_completed_before=0,
                                measurement_tags=_TAGS)
    pressures = rec["epochs"][0]["contrast"]["pressure"]
    theta0 = _flat_params(policy0)
    c0 = _reference(copy.deepcopy(policy0), rep.batch, ids, PPOConfig())["contrast"]
    cases = [(pressures["groups"][name]["pressure"],
              ep0.group_policy_surrogate_grads[gid], name)
             for gid, name in enumerate(GT._ACTOR_GRADIENT_GROUPS)]
    cases += [(pressures["total_loss"]["pressure"], ep0.total_loss_grad, "total_loss"),
              (pressures["entropy_component"]["pressure"],
               ep0.total_loss_grad - ep0.policy_surrogate_grad, "entropy")]
    signs = set()
    for pressure, g, name in cases:
        if abs(pressure) < 1e-6:
            continue
        eps = 1e-3 / _np_norm(g)                     # a small RAW descent step
        probe = copy.deepcopy(policy0)
        _set_flat(probe, theta0 - eps * g)
        delta = _reference(probe, rep.batch, ids, PPOConfig())["contrast"] - c0
        assert np.sign(delta) == np.sign(pressure), (name, pressure, delta)
        assert delta / eps == pytest.approx(pressure, rel=0.3), (name, pressure, delta / eps)
        signs.add(np.sign(pressure))
    assert signs == {-1.0, 1.0}, "fixture must exercise both signs"


def test_a7_displacement_projection_predicts_a_small_actual_step():
    policy0 = SC._policy(42)
    records = _records(policy0)
    got = []
    cfg = PPOConfig(lr=1e-5, n_epochs=4)
    diag = PPOUpdater(copy.deepcopy(policy0), cfg).update(
        records, **_kwargs(records, sink=got.append))
    rec = GT._actor_step_record(got[0], diag, iteration=0, updates_completed_before=0,
                                measurement_tags=_TAGS)
    for ep in rec["epochs"]:
        c = ep["contrast"]
        assert abs(c["predicted_delta_from_displacement"]) > 0
        assert np.sign(c["actual_delta"]) == np.sign(c["predicted_delta_from_displacement"])
        assert c["actual_delta"] == pytest.approx(c["predicted_delta_from_displacement"],
                                                  rel=0.05)
        assert c["linearization_residual"] == pytest.approx(
            c["actual_delta"] - c["predicted_delta_from_displacement"], abs=1e-18)


def test_a7_hand_example_of_the_pressure_convention():
    """``C(theta) = theta_0``; a loss gradient ``g = (-2, 1)`` descends to ``theta - lr*g``,
    raising ``C`` by ``2*lr``: the pressure ``-dot(h, g)`` is ``+2`` and the alignment
    ``cosine(-g, h)`` is positive -- the helpers the record uses."""
    h, g = np.array([1.0, 0.0]), np.array([-2.0, 1.0])
    assert -GT._dot(h, g) == 2.0
    assert GT._cosine(-g, h) == pytest.approx(2.0 / 5 ** 0.5)
    lr = 0.1
    assert (np.array([0.3, 0.4]) - lr * g)[0] - 0.3 == pytest.approx(lr * -GT._dot(h, g))


# =============================================================================
# A8: fail loud
# =============================================================================

def test_a8_updater_refuses_half_or_misaligned_wiring():
    policy = SC._policy(43)
    records = _records(policy)
    upd = PPOUpdater(policy, PPOConfig(n_epochs=1))
    n = sum(r.n_transitions for r in records)
    with pytest.raises(ValueError):
        upd.update(records, step_group_ids=[0] * n)
    with pytest.raises(ValueError):
        upd.update(records, step_sink=lambda r: None)
    with pytest.raises(ValueError):
        upd.update(records, step_group_ids=[0] * (n - 1), step_sink=lambda r: None)
    with pytest.raises(ValueError):
        upd.update(records, step_contrast_ids=(1, 0))
    with pytest.raises(ValueError):
        upd.update(records, step_group_ids=[0] * n, step_sink=lambda r: None,
                   step_contrast_ids=(1, 1))


def _one(seed=44, cfg=None):
    policy = SC._policy(seed)
    records = _records(policy)
    got = []
    diag = PPOUpdater(policy, cfg or PPOConfig()).update(
        records, **_kwargs(records, sink=got.append))
    return got, diag


def test_a8_record_faults_fail_loud():
    got, diag = _one()
    rep = got[0]
    kw = dict(iteration=0, updates_completed_before=0, measurement_tags=_TAGS)
    GT._actor_step_record(rep, diag, **kw)
    bad_epoch = dataclasses.replace(rep.epochs[1], delta_theta=rep.epochs[1].delta_theta * np.nan)
    faults = [
        dataclasses.replace(rep, group_ids=tuple(reversed(rep.group_ids))),
        dataclasses.replace(rep, contrast_ids=None),
        dataclasses.replace(rep, contrast_ids=(0, 1)),
        dataclasses.replace(rep, contrast_counts=(9, 9)),
        dataclasses.replace(rep, epochs=rep.epochs[:3]),
        dataclasses.replace(rep, epochs=(rep.epochs[0], bad_epoch) + rep.epochs[2:]),
        dataclasses.replace(rep, epochs=(dataclasses.replace(
            rep.epochs[0], contrast_grad=None),) + rep.epochs[1:]),
        dataclasses.replace(rep, epochs=(dataclasses.replace(
            rep.epochs[0], pre_clip_grad_norm=rep.epochs[0].pre_clip_grad_norm + 1.0),)
            + rep.epochs[1:]),
        dataclasses.replace(rep, epochs=(dataclasses.replace(
            rep.epochs[0], policy_surrogate_grad=rep.epochs[0].policy_surrogate_grad * 1.1),)
            + rep.epochs[1:]),
        dataclasses.replace(rep, epochs=(dataclasses.replace(
            rep.epochs[0], backward_grad=rep.epochs[0].backward_grad * 1.1),)
            + rep.epochs[1:]),
    ]
    for i, bad in enumerate(faults):
        with pytest.raises(GT.ActorStepDiagnosticsError):
            GT._actor_step_record(bad, diag, **kw)
            pytest.fail("fault %d was accepted" % i)
    # a transition swapped between two records' chains breaks identity attribution
    swapped = copy.copy(rep.batch)
    swapped.transitions = list(rep.batch.transitions)
    swapped.transitions[0], swapped.transitions[1] = (swapped.transitions[1],
                                                      swapped.transitions[0])
    with pytest.raises(GT.ActorStepDiagnosticsError, match="identity"):
        GT._actor_step_record(dataclasses.replace(rep, batch=swapped), diag, **kw)
    # an unattributable immediate-FD wake
    with pytest.raises(GT.ActorStepDiagnosticsError):
        GT._actor_step_group_ids(rep.records, {k: v for k, v in _TAGS.items()
                                               if k != (2, 502)})


def test_a8_persistence_fails_loud_and_vector_files_are_write_once(tmp_path):
    got, diag = _one(45)
    kw = dict(iteration=7, updates_completed_before=7, measurement_tags=_TAGS)
    with pytest.raises(GT.ActorStepDiagnosticsError):              # path is a directory
        GT._persist_actor_step_diagnostics(tmp_path, got, diag, **kw)
    with pytest.raises(GT.ActorStepDiagnosticsError):              # productive, no report
        GT._persist_actor_step_diagnostics(tmp_path / "s.jsonl", [], diag, **kw)
    with pytest.raises(GT.ActorStepDiagnosticsError):              # several reports
        GT._persist_actor_step_diagnostics(tmp_path / "s.jsonl", got * 2, diag, **kw)
    with pytest.raises(GT.ActorStepDiagnosticsError):              # count mismatch
        GT._persist_actor_step_diagnostics(tmp_path / "s.jsonl", got,
                                           dict(diag, n_transitions=3), **kw)
    vdir = tmp_path / GT._ACTOR_STEP_VECTORS_DIRNAME
    vdir.mkdir()
    path = tmp_path / "s.jsonl"
    assert GT._persist_actor_step_diagnostics(path, got, diag, vector_dir=vdir, **kw) == 1
    with pytest.raises(GT.ActorStepDiagnosticsError, match="already exists"):
        GT._persist_actor_step_diagnostics(path, got, diag, vector_dir=vdir, **kw)
    lines = path.read_text("utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    vf = rec["vector_file"]
    data = (tmp_path / vf["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == vf["sha256"] and len(data) == vf["bytes"]
    with np.load(tmp_path / vf["path"], allow_pickle=False) as z:
        assert sorted(z.files) == sorted(vf["arrays"] + ["layout_json"])
        layout = json.loads(str(z["layout_json"]))
        ep0 = got[0].epochs[0]
        np.testing.assert_array_equal(z["contrast_grad"], ep0.contrast_grad)
        np.testing.assert_array_equal(z["delta_theta"], ep0.delta_theta)
        np.testing.assert_array_equal(z["total_loss_grad"], ep0.total_loss_grad)
        for gid, name in enumerate(GT._ACTOR_GRADIENT_GROUPS):
            np.testing.assert_array_equal(z["group_" + name],
                                          ep0.group_policy_surrogate_grads[gid])
        assert sum(int(np.prod(e["shape"])) for e in layout) == z["delta_theta"].size
        assert [e["name"] for e in layout][:1] == [
            "encoder." + next(iter(dict(SC._policy(0).encoder.named_parameters())))]
    assert vf["contrast_grad_absent_reason"] is None


# =============================================================================
# A9: configuration
# =============================================================================

def test_a9_off_by_default_actor_only_only_and_recorded(tmp_path):
    cfg = GT.TrainConfig(n_iterations=1)
    assert cfg.actor_step_diagnostics is False and tuple(cfg.actor_step_vector_iterations) == ()
    with pytest.raises(ValueError, match="actor_step_diagnostics"):
        GT.TrainConfig(n_iterations=1, training_mode=GT.TRAINING_MODE_CTDE,
                       actor_step_diagnostics=True).validate()
    with pytest.raises(ValueError, match="requires actor_step_diagnostics"):
        GT.TrainConfig(n_iterations=1, actor_step_vector_iterations=(0,)).validate()
    for bad in ((3, 1), (1, 1), (-1,), (True,)):
        with pytest.raises(ValueError, match="strictly increasing"):
            GT.TrainConfig(n_iterations=5, actor_step_diagnostics=True,
                           actor_step_vector_iterations=bad).validate()
    GT.TrainConfig(n_iterations=5, actor_step_diagnostics=True,
                   actor_step_vector_iterations=[0, 4]).validate()
    for on in (False, True):
        cfg = GT.TrainConfig(n_iterations=1, output_dir=str(tmp_path),
                             actor_step_diagnostics=on,
                             actor_step_vector_iterations=(0,) if on else ())
        data = json.loads(Path(GT.write_run_config(
            tmp_path, cfg, provenance={"git": {"available": True}})).read_text("utf-8"))
        block = data["training"]["actor_step_diagnostics"]
        assert block["enabled"] is on
        assert block["artifact"] == "train_actor_step_diagnostics.jsonl"
        assert (block["schema"], block["schema_version"]) == (
            "graph_train_actor_step_diagnostics", 1)
        assert block["vector_iterations"] == ([0] if on else [])
        assert data["train_config"]["actor_step_diagnostics"] is on
        # the CTDE diagnostic's provenance block is untouched
        assert data["training"]["actor_gradient_diagnostics"]["scope"] == "ctde_updater_epoch_0"
    parser = GT._build_arg_parser()
    args = parser.parse_args(["--iterations", "1"])
    assert args.actor_step_diagnostics is False and args.actor_step_vector_iterations == ()
    args = parser.parse_args(["--iterations", "1", "--actor-step-diagnostics",
                              "--actor-step-vector-iterations", "0,24,49,74,99"])
    assert args.actor_step_diagnostics is True
    assert args.actor_step_vector_iterations == (0, 24, 49, 74, 99)
    assert GT._CLI_FIELD_BY_DEST["actor_step_diagnostics"] == "actor_step_diagnostics"
    assert GT._CLI_FIELD_BY_DEST["actor_step_vector_iterations"] == \
        "actor_step_vector_iterations"


# =============================================================================
# A10: end to end through the real trainer
# =============================================================================

def _fd_stub(policy_holder):
    """A stub episode whose FD-damaged episodes carry a REAL immediate-FD wake of the
    scheduled ego (``a0``), so the trainer-side join, the contrast and the step all run."""
    def fake_run_one_episode(policy, gen, cfg_, *, seed, episode_tag, deterministic,
                             fuel_damage_mode=None, **kwargs):
        params = cfg_.fuel_damage_parameters(fuel_damage_mode)
        condition = resolve_condition(episode_seed=seed, params=params)
        severity = resolve_severity(episode_seed=seed, params=params)
        damaged = condition == GT.CONDITION_DAMAGED
        kinds = [ORDK, IMM, POSTK] if damaged else [ORDK, ORDK]
        traj = []
        for t, kind in enumerate(kinds):
            gobs = SC._obs([SC._row(0.8 - 0.01 * (seed % 11)), SC._row(0.6 + 0.02 * t),
                            SC._row(0.5), SC._row(0.7)])
            tr = SC._sampled_transition(policy, gobs, ego="a0", tick=t + 1,
                                        seed=seed * 7 + t)
            tr.wake_kind = kind
            traj.append(tr)
        reward = -0.1 - 0.05 * (seed % 9)
        traj[-1].reward = reward
        return GT._EpisodeOutcome(
            trajectory=traj, reward=reward, ticks=10, ended="done", n_wakes=len(traj),
            confirmed_kills=1, n_dead=0, seconds=0.01, targets_confirmed_unique=1,
            targets_total=6, known_target_names=("A",), hidden_target_names=("B",),
            known_confirmed_names=("A",), hidden_confirmed_names=(),
            fuel_damage_plan={"condition": condition, "severity": severity,
                              "ego_id": "a0" if damaged else None},
            fuel_damage_outcome={"condition": condition, "severity": severity,
                                 "fired": damaged, "wake_occurred": damaged,
                                 "wake_meta_action": None, "event_tick": 2 if damaged
                                 else None},
            selected_ego_rtb_issued=None,
        )
    return fake_run_one_episode


def _run_fd_stub_training(cfg):
    saved = {n: getattr(GT, n) for n in ("_run_one_episode", "_build_generator",
                                         "_git_provenance")}
    GT._git_provenance = lambda repo_root: {
        "repo_root": str(repo_root), "available": True, "commit": "0" * 40,
        "branch": "test", "dirty": False, "dirty_path_count": 0, "reason": None}
    GT._run_one_episode = _fd_stub(None)
    GT._build_generator = lambda _d: object()
    try:
        return GT.train(cfg)
    finally:
        for n, v in saved.items():
            setattr(GT, n, v)


def _lines(path):
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x]


_TIMING = ("seconds", "elapsed", "timestamp", "_at", "wall")


def _strip_timing(obj):
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items()
                if not any(t in k for t in _TIMING)}
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    return obj


def _cfg(tmp_path, name, on, **kw):
    return GT.TrainConfig(n_iterations=3, episodes_per_iteration=4, eval_every=0,
                          eval_episodes=0, output_dir=str(tmp_path / name),
                          fuel_damage_mode="seeded_variable", checkpoint_every=1,
                          actor_step_diagnostics=on,
                          actor_step_vector_iterations=(0, 2) if on else (), **kw)


def test_a10_on_off_identical_learning_and_one_record_per_productive_update(tmp_path):
    runs = {}
    for on in (False, True):
        cfg = _cfg(tmp_path, str(on), on)
        _run_fd_stub_training(cfg)
        runs[on] = Path(cfg.output_dir)
    off, on = runs[False], runs[True]
    assert not (off / "train_actor_step_diagnostics.jsonl").exists()
    assert not (off / GT._ACTOR_STEP_VECTORS_DIRNAME).exists()
    # identical learning: training records, credit rows, outcomes and checkpoints
    assert _strip_timing(_lines(off / "train_records.jsonl")) == \
        _strip_timing(_lines(on / "train_records.jsonl"))
    assert _lines(off / "train_credit_diagnostics.jsonl") == \
        _lines(on / "train_credit_diagnostics.jsonl")
    assert _strip_timing(_lines(off / "episode_outcomes.jsonl")) == \
        _strip_timing(_lines(on / "episode_outcomes.jsonl"))
    ckpts = sorted(p.name for p in (off / "checkpoints").iterdir())
    assert ckpts and ckpts == sorted(p.name for p in (on / "checkpoints").iterdir())
    for name in ckpts:
        a = torch.load(off / "checkpoints" / name, weights_only=False)
        b = torch.load(on / "checkpoints" / name, weights_only=False)
        for key in ("encoder", "head"):
            assert all(torch.equal(a[key][k], b[key][k]) for k in a[key])
        sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
        assert all(torch.equal(sa[k][f], sb[k][f]) for k in sa for f in sa[k])
    train = _lines(on / "train_records.jsonl")
    recs = _lines(on / "train_actor_step_diagnostics.jsonl")
    productive = [r for r in train if r["n_epochs_run"]]
    assert len(recs) == len(productive) == 3
    assert [r["iteration"] for r in recs] == [0, 1, 2]
    credit = _lines(on / "train_credit_diagnostics.jsonl")
    for rec, tr in zip(recs, productive):
        assert rec["batch_n_transitions"] == tr["n_transitions"]
        assert len(rec["epochs"]) == 4 == tr["n_epochs_run"]
        # group counts agree with the credit rows' own wake kinds and joins
        rows = [c for c in credit if c["iteration"] == rec["iteration"]]
        fd = [c for c in rows if c["wake_kind"] == IMM]
        assert rec["groups"]["immediate_fd_mild"]["n_transitions"] + \
            rec["groups"]["immediate_fd_severe"]["n_transitions"] == len(fd)
        assert rec["groups"]["post_fd"]["n_transitions"] == sum(
            1 for c in rows if c["wake_kind"] == POSTK)
    assert any(r["update_summary"]["contrast_defined"] for r in recs), \
        "the stub schedule must produce at least one batch with both severities"
    vec = [r["vector_file"] for r in recs]
    assert vec[1] is None and vec[0] is not None and vec[2] is not None
    for v in (vec[0], vec[2]):
        data = (on / v["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == v["sha256"]
    config = json.loads((on / "run_config.json").read_text("utf-8"))
    assert config["training"]["actor_step_diagnostics"]["enabled"] is True


@pytest.mark.parametrize("fault", ["raise", "nan", "no_report"])
def test_a10_a_diagnostic_fault_stops_the_run_and_is_never_attrition(tmp_path, monkeypatch,
                                                                     fault):
    cfg = _cfg(tmp_path, fault, True)
    if fault == "raise":
        def broken(*a, **k):
            raise GT.ActorStepDiagnosticsError("injected")
        monkeypatch.setattr(GT, "_actor_step_record", broken)
    elif fault == "nan":
        real_record = GT._actor_step_record

        def poisoned(report, diag, **k):
            ep = dataclasses.replace(report.epochs[0],
                                     total_loss_grad=report.epochs[0].total_loss_grad * np.nan)
            return real_record(dataclasses.replace(report, epochs=(ep,) + report.epochs[1:]),
                               diag, **k)
        monkeypatch.setattr(GT, "_actor_step_record", poisoned)
    else:
        real_update = PPOUpdater.update

        def silent(self, source, credit_sink=None, **k):
            k.pop("step_sink", None)
            k.pop("step_group_ids", None)
            k.pop("step_contrast_ids", None)
            return real_update(self, source, credit_sink=credit_sink)
        monkeypatch.setattr(PPOUpdater, "update", silent)
    with pytest.raises(GT.ActorStepDiagnosticsError):
        _run_fd_stub_training(cfg)
    run = Path(cfg.output_dir)
    assert (run / "episode_failures.jsonl").read_text("utf-8") == ""
    assert _lines(run / "train_actor_step_diagnostics.jsonl") == []


# =============================================================================
# A11: isolation
# =============================================================================

_GROUP_WORDS = {"severity", "mild", "severe", "immediate_fd_mild", "immediate_fd_severe",
                "post_fd", "measurement_join", "measurement_tags", "credit_tags",
                "fd_selected_ego_id", "fd_event_tick", "is_fd_selected_ego"}


def test_a11_no_measurement_label_in_graph_ppo_or_a_learning_object():
    ppo_names = {n.lower() for n in SC._names(inspect.getsource(GP))}
    assert not ppo_names & _GROUP_WORDS, ppo_names & _GROUP_WORDS
    for cls in (GP.ActorStepReport, GP.ActorStepEpoch, GP.AdvantageBatch, TL.Transition,
                EpisodeRecord):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not names & _GROUP_WORDS, (cls.__name__, names & _GROUP_WORDS)
    # the updater's ids feed nothing but the diagnostic: they never reach an advantage,
    # the surrogate, the clip or the step (checked by source, beyond A1 / A2's numbers)
    src = textwrap.dedent(inspect.getsource(PPOUpdater.update))
    tree = ast.parse(src)
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        func = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(
            call.func, "id", "")
        if func in ("clipped_surrogate", "evaluate_action", "clip_grad_norm_", "step",
                    "backward", "compute_returns_and_advantages", "zero_grad"):
            used = {n.id for a in list(call.args) + [k.value for k in call.keywords]
                    for n in ast.walk(a) if isinstance(n, ast.Name)}
            assert not used & {"step_ids", "step_group_ids", "step_contrast",
                               "step_members", "contrast_rows"}, (func, used)


def test_a11_nothing_reads_the_step_artifact_back():
    tree = ast.parse(inspect.getsource(GT))
    users = set()
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.ClassDef)):
            names = SC._names(ast.unparse(top))
            if names & {"_ACTOR_STEP_DIAGNOSTICS_FILENAME", "_ACTOR_STEP_VECTORS_DIRNAME",
                        "train_actor_step_diagnostics.jsonl"}:
                users.add(top.name)
    assert users == {"train", "write_run_config", "_persist_actor_step_diagnostics",
                     "_build_arg_parser"}, users
    train_tree = ast.parse(SC._dedent(inspect.getsource(GT.train)))
    readers = [n for n in ast.walk(train_tree) if isinstance(n, ast.Name)
               and n.id == "step_reports"]
    # created, handed to the sink, handed to the writer -- nothing else
    assert len(readers) == 3


if __name__ == "__main__":  # pragma: no cover - direct runner
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
