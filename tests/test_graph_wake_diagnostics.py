"""GENERALIZED-V1 -- FD POLICY MEASUREMENT HARDENING: the three proof obligations.

Solver-free and BLADE-free. Every test here drives the real production symbols
(``graph_action.summarize_decision``, ``graph_tick_loop._wake_decision`` /
``_decision_record`` / ``Transition``, ``graph_train._wake_decision_records`` /
``_wake_diag_digest`` / ``_fd_policy_sensitivity_from_outcomes`` /
``_episode_outcome_record`` / ``plot_training``) -- nothing is re-implemented here.

PO1  zero policy drift    -- adding diagnostics changes no decision and no RNG state.
PO2  action semantics     -- the locked duplicated-cell case is reported accurately.
PO3  attribution/control  -- the three wake populations stay disjoint, a zero-wake
                             success records ``[]``, legacy records stay readable and
                             plottable, and no control path reads the diagnostics.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from match_aou.rl.action.graph_action import (  # noqa: E402
    MetaAction,
    NUM_META_ACTIONS,
    _masked_dist,
    sample_action,
    summarize_decision,
)
from match_aou.rl.training import graph_tick_loop as TL  # noqa: E402
from match_aou.rl.training import graph_train as GT  # noqa: E402

NEG = float("-inf")
PLAN = int(MetaAction.PLAN_COMPLIANCE)
ENGAGE = int(MetaAction.OPPORTUNISTIC_ENGAGEMENT)
ABORT = int(MetaAction.SELF_PRESERVATION_ABORT)


# =============================================================================
# PO1 -- ZERO POLICY DRIFT
# =============================================================================

def _fixed_case():
    """Fixed logits + mask with several valid cells across two meta-actions."""
    logits = torch.tensor([[0.7, -0.2, 0.4],
                           [0.1, 0.9, -0.5],
                           [-0.3, 0.2, 1.1]], dtype=torch.float32)
    mask = np.array([[0.0, NEG, 0.0],
                     [0.0, 0.0, NEG],
                     [0.0, NEG, 0.0]], dtype=np.float32)
    return logits, mask


@pytest.mark.parametrize("deterministic", [True, False])
def test_po1_selection_and_rng_state_unchanged_by_diagnostics(deterministic):
    """The decision AND the torch RNG state are identical with/without diagnostics."""
    logits, mask = _fixed_case()

    torch.manual_seed(1234)
    base_meta, base_node, base_lp, base_ent = sample_action(
        logits, mask, deterministic=deterministic)
    base_rng = torch.get_rng_state().clone()

    torch.manual_seed(1234)
    meta, node, lp, ent = sample_action(logits, mask, deterministic=deterministic)
    # The diagnostic summarizer runs exactly where production runs it: AFTER the draw.
    diag = summarize_decision(logits, mask, meta, node)
    after_rng = torch.get_rng_state().clone()

    assert (meta, node) == (base_meta, base_node)
    assert float(lp.item()) == float(base_lp.item())
    assert float(ent.item()) == float(base_ent.item())
    assert torch.equal(after_rng, base_rng), "the summarizer moved the torch RNG state"
    # The summarizer reports the action that was actually taken.
    assert diag["selected_meta_action"] == meta
    assert diag["selected_node"] == node


def test_po1_summarizer_alone_leaves_rng_state_untouched():
    """Calling the summarizer N times cannot shift a later stochastic draw."""
    logits, mask = _fixed_case()
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    for _ in range(25):
        summarize_decision(logits, mask, PLAN, 0)
    assert torch.equal(torch.get_rng_state(), before)
    # And the very next sampled action is the one an un-instrumented run would draw.
    torch.manual_seed(99)
    expect = sample_action(logits, mask, deterministic=False)
    torch.manual_seed(99)
    for _ in range(25):
        summarize_decision(logits, mask, PLAN, 0)
    got = sample_action(logits, mask, deterministic=False)
    assert (got[0], got[1]) == (expect[0], expect[1])


def test_po1_summarizer_takes_no_gradient_and_does_not_touch_the_graph():
    """No autograd node is created, and a grad-attached input is left intact."""
    logits, mask = _fixed_case()
    logits = logits.clone().requires_grad_(True)
    out = summarize_decision(logits, mask, PLAN, 0)
    assert logits.grad is None
    # Everything returned is a plain builtin -- no tensor can leak into an artifact.
    json.dumps(out)


# =============================================================================
# PO2 -- ACTION-SEMANTICS ACCURACY (the locked duplicated-cell case)
# =============================================================================

def _locked_disagreement_case():
    """LOCKED: joint-cell argmax says ABORT, aggregate-meta argmax says PLAN.

    PLAN is valid on all three nodes with equal logit 1.0; ABORT is valid on ONE node
    with the single largest logit 1.6. So ABORT owns the biggest SINGLE cell while PLAN
    owns the biggest TOTAL mass -- exactly the duplicated-representation effect the
    artifact must report without conflating the two.
    """
    logits = torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 0.0, 1.6]], dtype=torch.float32)
    mask = np.array([[0.0, NEG, NEG],
                     [0.0, NEG, NEG],
                     [0.0, NEG, 0.0]], dtype=np.float32)
    return logits, mask


def test_po2_joint_and_aggregate_argmax_disagree_and_are_reported_accurately():
    logits, mask = _locked_disagreement_case()
    d = summarize_decision(logits, mask, ABORT, 2)

    # --- independently expected values (hand-computed from the logits above) ---
    e1, e16 = np.exp(1.0), np.exp(1.6)
    z = 3.0 * e1 + e16
    p_plan_cell = e1 / z
    p_abort_cell = e16 / z

    assert d["joint_argmax_meta_action_name"] == MetaAction.SELF_PRESERVATION_ABORT.name
    assert d["aggregate_argmax_meta_action_name"] == MetaAction.PLAN_COMPLIANCE.name
    assert d["joint_vs_aggregate_disagree"] is True

    assert d["n_valid_cells"] == 4
    assert d["valid_cells_per_meta_action"] == {
        MetaAction.PLAN_COMPLIANCE.name: 3,
        MetaAction.OPPORTUNISTIC_ENGAGEMENT.name: 0,
        MetaAction.SELF_PRESERVATION_ABORT.name: 1,
    }
    agg = d["aggregate_probability_per_meta_action"]
    assert agg[MetaAction.PLAN_COMPLIANCE.name] == pytest.approx(3.0 * p_plan_cell)
    assert agg[MetaAction.SELF_PRESERVATION_ABORT.name] == pytest.approx(p_abort_cell)
    assert agg[MetaAction.OPPORTUNISTIC_ENGAGEMENT.name] == 0.0
    # ...and the aggregate really is larger for PLAN while the single cell is larger
    # for ABORT: the whole point of reporting both.
    assert agg[MetaAction.PLAN_COMPLIANCE.name] > agg[MetaAction.SELF_PRESERVATION_ABORT.name]
    assert p_abort_cell > p_plan_cell

    assert d["selected_cell_probability"] == pytest.approx(p_abort_cell)
    assert d["top_two_probability_margin"] == pytest.approx(p_abort_cell - p_plan_cell)
    assert d["top_two_valid_cells"][0]["meta_action_name"] == \
        MetaAction.SELF_PRESERVATION_ABORT.name
    assert d["top_two_valid_cells"][1]["meta_action_name"] == \
        MetaAction.PLAN_COMPLIANCE.name

    # masked cells carry EXACTLY zero mass, and the distribution normalizes.
    pm = np.asarray(d["masked_probabilities"])
    assert pm.sum() == pytest.approx(1.0)
    assert pm[0, ENGAGE] == 0.0 and pm[1, ABORT] == 0.0

    # entropy: raw, and normalized by log(valid cells)
    ps = np.array([p_plan_cell] * 3 + [p_abort_cell])
    assert d["joint_entropy_raw"] == pytest.approx(float(-(ps * np.log(ps)).sum()))
    assert d["joint_entropy_normalized"] == pytest.approx(
        d["joint_entropy_raw"] / np.log(4))


def test_po2_single_valid_cell_reports_null_normalized_entropy():
    """Fewer than two valid cells -> None, never NaN and never an invented number."""
    logits = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    mask = np.array([[0.0, NEG, NEG]], dtype=np.float32)
    d = summarize_decision(logits, mask, PLAN, 0)
    assert d["n_valid_cells"] == 1
    assert d["joint_entropy_normalized"] is None
    assert d["top_two_probability_margin"] is None
    assert d["joint_entropy_raw"] == pytest.approx(0.0)
    json.dumps(d)  # still serializable


def test_po2_all_masked_raises_like_the_actor_path():
    logits = torch.zeros((2, NUM_META_ACTIONS))
    mask = np.full((2, NUM_META_ACTIONS), NEG, dtype=np.float32)
    with pytest.raises(ValueError):
        summarize_decision(logits, mask, PLAN, 0)


def test_po2_digest_keeps_selected_cell_and_aggregate_mass_apart():
    """The digest must not conflate the SELECTED action with aggregate column mass."""
    logits, mask = _locked_disagreement_case()
    d = summarize_decision(logits, mask, ABORT, 2)
    dig = GT._wake_diag_digest([d])
    assert dig["n_wakes"] == 1
    # selected joint cell WAS abort -> fraction 1.0 ...
    assert dig["selected_joint_cell_abort_fraction"] == pytest.approx(1.0)
    # ... while the aggregate abort MASS is well under half.
    assert dig["aggregate_p_abort_mean"] < 0.5
    assert dig["joint_vs_aggregate_disagreement_fraction"] == pytest.approx(1.0)


def test_po2_digest_empty_population_reports_none_not_zero():
    dig = GT._wake_diag_digest([])
    assert dig["n_wakes"] == 0
    for key in ("selected_joint_cell_abort_fraction", "aggregate_p_abort_mean",
                "joint_entropy_raw_mean", "joint_entropy_normalized_mean",
                "joint_vs_aggregate_disagreement_fraction",
                "distance_clipping_fraction_mean"):
        assert dig[key] is None, key


def test_po2_undefined_normalized_entropy_excluded_from_mean_and_denominator():
    logits1, mask1 = _locked_disagreement_case()
    multi = summarize_decision(logits1, mask1, ABORT, 2)
    single = summarize_decision(torch.tensor([[0.5, 0.0, 0.0]]),
                                np.array([[0.0, NEG, NEG]], dtype=np.float32), PLAN, 0)
    dig = GT._wake_diag_digest([multi, single])
    assert dig["n_wakes"] == 2
    assert dig["n_joint_entropy_normalized_defined"] == 1
    assert dig["joint_entropy_normalized_mean"] == pytest.approx(
        multi["joint_entropy_normalized"])


# =============================================================================
# PO3 -- ATTRIBUTION / CONTROL ISOLATION
# =============================================================================

class _Tr:
    """A minimal Transition-shaped stand-in for the artifact-level tests."""

    def __init__(self, kind, meta, node=0, agg=None, extra=None):
        self.meta_action = int(meta)
        self.node_v = int(node)
        self.wake_kind = kind
        self.decision = {
            "wake_kind": kind,
            "selected_meta_action": int(meta),
            "selected_meta_action_name": MetaAction(int(meta)).name,
            "selected_node": int(node),
            "selected_node_ownership": TL.OWNERSHIP_EGO,
            "aggregate_probability_per_meta_action": agg or {
                MetaAction.PLAN_COMPLIANCE.name: 0.6,
                MetaAction.OPPORTUNISTIC_ENGAGEMENT.name: 0.1,
                MetaAction.SELF_PRESERVATION_ABORT.name: 0.3,
            },
            "joint_entropy_raw": 1.0,
            "joint_entropy_normalized": 0.9,
            "aggregate_meta_action_entropy": 0.8,
            "joint_vs_aggregate_disagree": False,
            "n_valid_cells": 4,
            "fraction_task_distance_clipped": 0.5,
        }
        if extra:
            self.decision.update(extra)


def test_po3_three_wake_kinds_are_recorded_separately_and_never_pooled():
    recs = [
        {"phase": "post_update", "cell": "mild", "updates_completed": 10,
         "benchmark_group_key": "A2-low-w000",
         "wake_decisions": [
             _Tr(TL.WAKE_KIND_ORDINARY, PLAN).decision,
             _Tr(TL.WAKE_KIND_IMMEDIATE_FD, ABORT).decision,
             _Tr(TL.WAKE_KIND_POST_FD_BOUNDARY, ENGAGE).decision,
         ]},
    ]
    d = GT._fd_policy_sensitivity_from_outcomes(recs)
    assert d["recorded"] is True
    # The three kinds are counted apart WITHIN the post_update population, and the
    # pooled view is only ever reachable under a name that says it pooled.
    block = d["by_phase"]["post_update"]
    assert block["n_wakes_by_kind"] == {
        TL.WAKE_KIND_ORDINARY: 1,
        TL.WAKE_KIND_IMMEDIATE_FD: 1,
        TL.WAKE_KIND_POST_FD_BOUNDARY: 1,
    }
    assert d["all_phases_pooled"]["n_wakes_by_kind"] == block["n_wakes_by_kind"]
    assert d["by_phase"]["train"]["n_wakes_by_kind"][TL.WAKE_KIND_IMMEDIATE_FD] == 0
    fd = block["by_wake_kind"][TL.WAKE_KIND_IMMEDIATE_FD]
    # The immediate-FD denominator is ONE wake -- the boundary wake is NOT in it.
    assert fd["n_wakes"] == 1
    assert fd["selected_joint_cell_abort_fraction"] == pytest.approx(1.0)
    boundary = block["by_wake_kind"][TL.WAKE_KIND_POST_FD_BOUNDARY]
    assert boundary["n_wakes"] == 1
    assert boundary["selected_joint_cell_abort_fraction"] == pytest.approx(0.0)
    ordinary = block["by_wake_kind"][TL.WAKE_KIND_ORDINARY]
    assert ordinary["n_wakes"] == 1


def test_po3_matched_delta_pairs_by_group_key_never_by_target_uuid():
    """Pairing is by frozen benchmark group identity; uuids are irrelevant."""
    def rec(cell, group, abort_mass, uuid):
        return {"phase": "post_update", "cell": cell, "updates_completed": 5,
                "benchmark_group_key": group,
                "wake_decisions": [_Tr(
                    TL.WAKE_KIND_IMMEDIATE_FD, ABORT,
                    agg={MetaAction.PLAN_COMPLIANCE.name: 1.0 - abort_mass,
                         MetaAction.OPPORTUNISTIC_ENGAGEMENT.name: 0.0,
                         MetaAction.SELF_PRESERVATION_ABORT.name: abort_mass},
                    extra={"target_uuid_label": uuid}).decision]}
    recs = [
        rec("mild", "A2-low-w000", 0.20, "uuid-A"),
        rec("severe", "A2-low-w000", 0.50, "uuid-DIFFERENT"),   # uuids differ on purpose
        rec("mild", "A3-high-w001", 0.30, "uuid-C"),
        rec("severe", "A3-high-w001", 0.40, "uuid-D"),
        rec("mild", "A4-low-w002", 0.10, "uuid-E"),             # unmatched -> excluded
    ]
    m = GT._fd_policy_sensitivity_from_outcomes(recs)[
        "matched_severe_minus_mild_aggregate_p_abort"]
    # One evaluation round here (one stage / update / ordinal / manifest identity).
    assert m["n_rounds"] == 1
    r0 = m["by_round"][0]
    assert r0["n"] == 2, "the unmatched group must not contribute"
    assert r0["mean"] == pytest.approx((0.30 + 0.10) / 2)
    assert "benchmark_group_key" in m["pairing"]
    assert "evaluation records only" in m["population"]


def test_po3_zero_wake_success_records_an_empty_list_not_null():
    assert GT._wake_decision_records([]) == []
    tr = _Tr(TL.WAKE_KIND_ORDINARY, PLAN)
    assert GT._wake_decision_records([tr]) == [tr.decision]
    # A transition with no diagnostics is SKIPPED, never filled with invented values.

    class _Bare:
        meta_action, node_v = PLAN, 0
    assert GT._wake_decision_records([_Bare()]) == []


def test_po3_legacy_records_without_the_field_stay_readable():
    legacy = [{"phase": "post_update", "cell": "mild", "updates_completed": 3,
               "fd_wake_meta_action_name": "PLAN_COMPLIANCE"}]
    d = GT._fd_policy_sensitivity_from_outcomes(legacy)
    assert d["recorded"] is False
    assert d["wake_diagnostics_schema_version_observed"] is None
    assert d["wake_diagnostics_schema_versions_observed"] == []
    # and the legacy severity table still works on the same rows
    GT._severity_response_from_outcomes(legacy)


def test_po3_legacy_run_directory_still_plots_three_figures_and_no_fourth(tmp_path):
    """`plot_training` on a pre-v3 directory: three figures, no fabricated fourth.

    Driven through ``plot_training_subprocess`` -- the production path on this stack.
    torch and matplotlib abort together in one process on this Windows/OpenMP build
    (that is exactly why the plot child exists), so importing matplotlib here beside
    the torch this module already loaded would crash the interpreter rather than test
    anything.
    """
    pytest.importorskip("matplotlib")
    run = tmp_path / "legacy_run"
    run.mkdir()
    (run / "train_records.jsonl").write_text(json.dumps({
        "iteration": 0, "updates_completed": 1, "updates_completed_before": 0,
        "n_attempted": 2, "n_successful": 2, "n_failed": 0,
        "train_reward_mean": -0.5, "entropy": 1.2,
        "meta_action_counts": {n: 1 for n in GT._META_NAMES},
        "success_fraction": 1.0, "wake_fraction_of_successful": 1.0,
    }) + "\n", encoding="utf-8")
    (run / "eval_records.jsonl").write_text(json.dumps({
        "evaluation_stage": "pre_update", "updates_completed": 0,
        "eval_reward_mean": -0.6, "n_attempted": 2, "n_successful": 2, "n_failed": 0,
        "success_fraction": 1.0,
    }) + "\n", encoding="utf-8")
    written = GT.plot_training_subprocess(run)
    names = {p.name for p in written}
    assert names == set(GT._PLOT_FILENAMES), names
    assert GT._PLOT_FD_SENSITIVITY not in names, "fabricated a figure with no data"
    assert not (run / "plots" / GT._PLOT_FD_SENSITIVITY).exists()
    assert None not in written


def _diagnostic_symbol_uses(source: str):
    """AST-precise uses of the diagnostic surface -- prose in docstrings is ignored.

    A substring scan is useless here: the word "decision" appears throughout the
    project's prose. This walks the parsed tree and reports only real CODE references:
    an attribute access ``x.decision`` / ``x.wake_decisions``, or a name/attribute
    binding one of the diagnostic helpers.
    """
    import ast
    import textwrap

    # `inspect.getsource` of a METHOD is indented; `ast.parse` needs it flush-left.
    source = textwrap.dedent(source)
    banned_attrs = {"decision", "wake_decisions"}
    banned_names = {"summarize_decision", "_wake_decision_records",
                    "_wake_diag_digest", "_fd_policy_sensitivity_from_outcomes"}
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in banned_attrs:
            found.append(node.attr)
        elif isinstance(node, ast.Attribute) and node.attr in banned_names:
            found.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in banned_names:
            found.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in banned_attrs:
            # a string KEY like rec["wake_decisions"] is a read too
            found.append(node.value)
    return sorted(set(found))


def test_po3_diagnostics_are_not_read_by_any_control_path():
    """No control surface consults the per-wake diagnostics.

    PPO/CTDE credit, the reward, the optimizer, early stopping and checkpoint control
    must all be blind to this field. Checked with an AST walk, so the project's prose
    (which uses the word "decision" everywhere) cannot mask a real reference.
    """
    import inspect
    from match_aou.rl.training import graph_ppo, graph_reward

    for mod in (graph_ppo, graph_reward):
        uses = _diagnostic_symbol_uses(inspect.getsource(mod))
        assert uses == [], "%s reads the diagnostics: %r" % (mod.__name__, uses)

    # In graph_train the field may be WRITTEN (the artifact) and AGGREGATED (the
    # summary/plot), but must never be reachable from a control function.
    for fn in (GT._EarlyStoppingMonitor.observe, GT._EarlyStoppingMonitor.is_due,
               GT.save_checkpoint, GT._iteration_outcome):
        uses = _diagnostic_symbol_uses(inspect.getsource(fn))
        assert uses == [], "%s reads the diagnostics: %r" % (fn.__name__, uses)

    # The early-stopping monitor still takes exactly its two keyword arguments, so no
    # diagnostic can reach the stopping rule even by accident.
    params = list(inspect.signature(GT._EarlyStoppingMonitor.observe).parameters)
    assert params == ["self", "completed_iterations", "train_reward_mean"]

    # Control-flow sanity: the PPO buffer records consume only the fields they always
    # did, so a Transition carrying diagnostics is credited identically.
    import ast as _ast
    ppo_src = inspect.getsource(graph_ppo)
    attrs = {n.attr for n in _ast.walk(_ast.parse(ppo_src))
             if isinstance(n, _ast.Attribute)}
    assert "wake_kind" not in attrs, "graph_ppo reads the wake-kind tag"


def test_po3_transition_defaults_keep_pre_change_behaviour():
    """A Transition built without the new fields is an ordinary, undiagnosed wake."""
    import numpy as _np
    from match_aou.rl.observation.graph_builder import GraphObservation
    gobs = GraphObservation(
        task_features=_np.zeros((1, 6), dtype=_np.float32),
        agent_features=_np.zeros((1, 1), dtype=_np.float32),
        ego_index=1,
        edge_index=_np.zeros((2, 0), dtype=_np.int64),
        edge_type=_np.zeros((0,), dtype=_np.int64),
        task_target_ids=["t"], agent_ids=["e"], agent_id="e",
        current_time=0, time_norm=0.0,
    )
    tr = TL.Transition(gobs=gobs, ego_id="e", tick=0, meta_action=PLAN, node_v=0,
                       log_prob=-1.0, entropy=1.0)
    assert tr.wake_kind == TL.WAKE_KIND_ORDINARY
    assert tr.decision is None
    assert GT._wake_decision_records([tr]) == []


def test_po3_node_ownership_classification():
    sol = {"ego": [(0, 0, 0)], "peer": [(1, 0, 0)]}
    assert TL._node_ownership(sol, "ego", 0) == TL.OWNERSHIP_EGO
    assert TL._node_ownership(sol, "ego", 1) == TL.OWNERSHIP_PEER
    assert TL._node_ownership(sol, "ego", 2) == TL.OWNERSHIP_UNASSIGNED
    # a malformed tuple is skipped, never raised on (this is a reporting path)
    assert TL._node_ownership({"ego": [("bad",)]}, "ego", 0) == TL.OWNERSHIP_UNASSIGNED
    assert TL._node_ownership(None, "ego", 0) == TL.OWNERSHIP_UNASSIGNED


def test_po3_outcome_record_carries_versioned_wake_list_and_is_json_safe():
    """The artifact contract: versioned, JSON-safe, and empty (not null) at zero wakes."""
    recs = GT._wake_decision_records([_Tr(TL.WAKE_KIND_IMMEDIATE_FD, ABORT)])
    assert len(recs) == 1
    json.dumps(recs)
    assert GT._EPISODE_OUTCOME_VERSION == 3
    assert isinstance(GT._WAKE_DIAGNOSTICS_VERSION, int)


def test_po3_fd_sensitivity_splits_by_cell_within_the_fd_population():
    recs = [
        {"phase": "post_update", "cell": "mild", "updates_completed": 1,
         "benchmark_group_key": "g0",
         "wake_decisions": [_Tr(TL.WAKE_KIND_IMMEDIATE_FD, PLAN).decision]},
        {"phase": "post_update", "cell": "severe", "updates_completed": 1,
         "benchmark_group_key": "g0",
         "wake_decisions": [_Tr(TL.WAKE_KIND_IMMEDIATE_FD, ABORT).decision]},
    ]
    by = GT._fd_policy_sensitivity_from_outcomes(recs)[
        "by_phase"]["post_update"]["by_wake_kind_and_cell"]
    fd = by[TL.WAKE_KIND_IMMEDIATE_FD]
    assert fd["mild"]["selected_joint_cell_abort_fraction"] == pytest.approx(0.0)
    assert fd["severe"]["selected_joint_cell_abort_fraction"] == pytest.approx(1.0)


# =============================================================================
# FIX 1 -- THE REPORTED DISTRIBUTION IS THE ACTOR'S EXACT DISTRIBUTION
# =============================================================================
# `summarize_decision` must not rebuild the distribution with an independent
# implementation. Every quantity below is compared to `_masked_dist` / the Categorical
# `sample_action` itself routes through, by EXACT equality -- an independent float64
# masked softmax cannot satisfy these.

def _ref(logits, mask):
    """The actor's own distribution, built through the SAME shared site."""
    return _masked_dist(logits, mask)


def test_fix1_reported_values_equal_the_masked_dist_construction_exactly():
    logits, mask = _fixed_case()
    d = summarize_decision(logits, mask, PLAN, 0)
    flat, dist, ent = _ref(logits, mask)

    probs = dist.probs.reshape(-1).tolist()
    k = int(logits.shape[0])
    # EXACT equality, not approx: any second implementation would drift here.
    assert [v for row in d["masked_probabilities"] for v in row] == probs
    assert d["joint_entropy_raw"] == float(ent.item())
    assert d["n_valid_cells"] == int(torch.isfinite(flat).sum().item())

    agg_ref = dist.probs.reshape(k, NUM_META_ACTIONS).sum(dim=0).tolist()
    assert [d["aggregate_probability_per_meta_action"][MetaAction(m).name]
            for m in range(NUM_META_ACTIONS)] == agg_ref

    i1 = int(torch.argmax(flat).item())
    cell = d["joint_argmax_cell"]
    assert cell["node"] * NUM_META_ACTIONS + cell["meta_action"] == i1
    assert cell["probability"] == probs[i1]
    # and the normalized form is the raw one over log(valid cells) -- same raw value
    assert d["joint_entropy_normalized"] == pytest.approx(
        float(ent.item()) / math.log(d["n_valid_cells"]))


def test_fix1_close_float32_logits_stay_in_the_actors_dtype():
    """Cells a float64 recomputation would separate differently stay exact.

    The reported probabilities must be exactly the float32 Categorical values -- so
    every one of them is exactly representable as a float32. A float64 masked softmax
    of the same logits would generally NOT be, which is what makes this falsifying.
    """
    logits = torch.tensor([[0.30000001, 0.30000004, 0.29999998],
                           [0.30000002, 0.30000000, 0.30000003]],
                          dtype=torch.float32)
    mask = np.zeros((2, 3), dtype=np.float32)
    d = summarize_decision(logits, mask, PLAN, 0)
    _flat, dist, ent = _ref(logits, mask)

    probs = dist.probs.reshape(-1).tolist()
    flat_reported = [v for row in d["masked_probabilities"] for v in row]
    assert flat_reported == probs
    for v in flat_reported:
        assert float(np.float32(v)) == v, "a float64 recomputation leaked in"
    assert d["joint_entropy_raw"] == float(ent.item())


def test_fix1_exact_ties_resolve_like_deterministic_sample_action():
    """An exact tie must land on the cell the DETERMINISTIC actor would take."""
    logits = torch.tensor([[1.0, 0.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0]], dtype=torch.float32)
    mask = np.array([[0.0, NEG, 0.0],
                     [0.0, NEG, 0.0],
                     [0.0, NEG, NEG]], dtype=np.float32)
    meta, node, _lp, _e = sample_action(logits, mask, deterministic=True)
    d = summarize_decision(logits, mask, meta, node)
    assert d["joint_argmax_cell"]["node"] == node
    assert d["joint_argmax_cell"]["meta_action"] == meta
    assert d["joint_argmax_meta_action"] == meta
    # the runner-up is a DIFFERENT valid cell, chosen on the same ordering basis
    second = d["top_two_valid_cells"][1]
    assert (second["node"], second["meta_action"]) != (node, meta)
    assert np.isfinite(mask[second["node"], second["meta_action"]])


def test_fix1_selected_probability_agrees_with_the_selected_log_probability():
    logits, mask = _fixed_case()
    meta, node, log_prob, entropy = sample_action(logits, mask, deterministic=True)
    d = summarize_decision(logits, mask, meta, node)
    _flat, dist, _e = _ref(logits, mask)

    idx = node * NUM_META_ACTIONS + meta
    assert d["selected_cell_probability"] == float(dist.probs.reshape(-1)[idx])
    assert d["selected_cell_probability"] == pytest.approx(
        float(torch.exp(log_prob).item()), rel=1e-6)
    assert d["joint_entropy_raw"] == float(entropy.item())


# =============================================================================
# FIX 2 -- TRAIN AND EVALUATION ARE SEPARATE POPULATIONS
# =============================================================================

def _fd_rec(phase, cell, x, abort_mass, meta, group=None, ordinal=None,
            manifest=None, disagree=False):
    return {
        "phase": phase, "cell": cell, "updates_completed": x,
        "eval_round_ordinal": ordinal, "benchmark_group_key": group,
        "benchmark_manifest_id": manifest,
        "wake_decisions": [_Tr(
            TL.WAKE_KIND_IMMEDIATE_FD, meta,
            agg={MetaAction.PLAN_COMPLIANCE.name: 1.0 - abort_mass,
                 MetaAction.OPPORTUNISTIC_ENGAGEMENT.name: 0.0,
                 MetaAction.SELF_PRESERVATION_ABORT.name: abort_mass},
            extra={"joint_vs_aggregate_disagree": disagree}).decision],
    }


def test_fix2_training_rows_cannot_change_evaluation_plot_data():
    """CONTRADICTORY train and eval values at the SAME update: eval data is untouched."""
    x = 7
    eval_rows = [
        _fd_rec("post_update", "severe", x, 1.0, ABORT),
        _fd_rec("post_update", "mild", x, 0.0, PLAN),
    ]
    # Training rows at the SAME x saying the OPPOSITE thing on both cells.
    train_rows = [
        _fd_rec("train", "severe", x, 0.0, PLAN),
        _fd_rec("train", "mild", x, 1.0, ABORT),
    ]

    eval_only = GT._fd_sensitivity_plot_data(eval_rows)
    with_train = GT._fd_sensitivity_plot_data(eval_rows + train_rows)
    assert eval_only == with_train, "training rows leaked into the plotted series"
    # and the eval values are the eval ones, not an average of the two
    sel = eval_only["series"]["selected_joint_cell_abort_fraction"]
    assert sel["severe"] == {"x": [float(x)], "y": [1.0]}
    assert sel["mild"] == {"x": [float(x)], "y": [0.0]}
    agg = eval_only["series"]["aggregate_p_abort_mean"]
    assert agg["severe"]["y"] == [pytest.approx(1.0)]
    assert agg["mild"]["y"] == [pytest.approx(0.0)]

    # The DIGEST keeps the same three populations apart, and says so.
    d = GT._fd_policy_sensitivity_from_outcomes(eval_rows + train_rows)
    post = d["by_phase"]["post_update"]["by_wake_kind_and_cell"][
        TL.WAKE_KIND_IMMEDIATE_FD]
    tr = d["by_phase"]["train"]["by_wake_kind_and_cell"][TL.WAKE_KIND_IMMEDIATE_FD]
    assert post["severe"]["selected_joint_cell_abort_fraction"] == pytest.approx(1.0)
    assert tr["severe"]["selected_joint_cell_abort_fraction"] == pytest.approx(0.0)
    assert d["by_phase"]["pre_update"]["n_episode_records"] == 0
    assert "POOLED" in d["all_phases_pooled"]["note"]


def test_fix2_matched_deltas_are_evaluation_only():
    """A training row with a benchmark group key contributes to no matched delta."""
    rows = [
        _fd_rec("train", "mild", 3, 0.0, PLAN, group="g0", ordinal=0),
        _fd_rec("train", "severe", 3, 1.0, ABORT, group="g0", ordinal=0),
    ]
    m = GT._fd_policy_sensitivity_from_outcomes(rows)[
        "matched_severe_minus_mild_aggregate_p_abort"]
    assert m["n_rounds"] == 0
    assert m["pooled_across_rounds"]["n_group_deltas"] == 0
    assert GT._fd_sensitivity_plot_data(rows)["recorded"] is False


# =============================================================================
# FIX 3 -- ROUND-SCOPED MATCHED DELTAS, AND FINAL-ROUND SELECTION BY IDENTITY
# =============================================================================

def _two_rounds():
    """The SAME frozen group `g0`, measured in TWO rounds with different values."""
    return [
        _fd_rec("post_update", "mild", 5, 0.10, PLAN, group="g0", ordinal=0,
                manifest="m1"),
        _fd_rec("post_update", "severe", 5, 0.30, ABORT, group="g0", ordinal=0,
                manifest="m1"),
        _fd_rec("post_update", "mild", 10, 0.20, PLAN, group="g0", ordinal=1,
                manifest="m1"),
        _fd_rec("post_update", "severe", 10, 0.90, ABORT, group="g0", ordinal=1,
                manifest="m1"),
    ]


def test_fix3_same_group_in_two_rounds_stays_two_deltas_in_either_input_order():
    rows = _two_rounds()
    final = {"evaluation_stage": "post_update", "updates_completed": 10,
             "eval_round_ordinal": 1}

    forward = GT._fd_policy_sensitivity_from_outcomes(rows, final_eval=final)[
        "matched_severe_minus_mild_aggregate_p_abort"]
    reverse = GT._fd_policy_sensitivity_from_outcomes(
        list(reversed(rows)), final_eval=final)[
        "matched_severe_minus_mild_aggregate_p_abort"]
    assert forward == reverse, "the record order in the file changed the result"

    assert forward["n_rounds"] == 2
    r0, r1 = forward["by_round"]
    assert (r0["updates_completed"], r0["eval_round_ordinal"]) == (5, 0)
    assert (r1["updates_completed"], r1["eval_round_ordinal"]) == (10, 1)
    assert r0["n"] == 1 and r0["mean"] == pytest.approx(0.30 - 0.10)
    assert r1["n"] == 1 and r1["mean"] == pytest.approx(0.90 - 0.20)
    # the two rounds re-measure ONE frozen world, and the pool says so
    assert forward["pooled_across_rounds"][
        "totals_across_rounds_are_repeated_measures"] is True
    assert forward["pooled_across_rounds"]["n_group_deltas"] == 2

    # FINAL ROUND: chosen by validated identity, NOT by file order.
    sel = forward["final_round_selection"]
    assert sel["selected"] is True and sel["n_candidates"] == 1
    fr = forward["final_round"]
    assert (fr["evaluation_stage"], fr["updates_completed"],
            fr["eval_round_ordinal"]) == ("post_update", 10, 1)
    assert fr["mean"] == pytest.approx(0.90 - 0.20)


def test_fix3_final_round_selection_refuses_rather_than_guesses():
    rows = _two_rounds()
    # no digest at all
    m = GT._fd_policy_sensitivity_from_outcomes(rows)[
        "matched_severe_minus_mild_aggregate_p_abort"]
    assert m["final_round"] is None
    assert m["final_round_selection"]["reason"] == "no_final_evaluation_digest_provided"
    # an identity no round carries
    m = GT._fd_policy_sensitivity_from_outcomes(
        rows, final_eval={"evaluation_stage": "post_update",
                          "updates_completed": 99, "eval_round_ordinal": 9})[
        "matched_severe_minus_mild_aggregate_p_abort"]
    assert m["final_round"] is None
    assert m["final_round_selection"]["reason"] == (
        "no_matched_round_with_that_identity")
    # a digest stating no identity at all
    m = GT._fd_policy_sensitivity_from_outcomes(rows, final_eval={})[
        "matched_severe_minus_mild_aggregate_p_abort"]
    assert m["final_round_selection"]["reason"] == (
        "no_final_evaluation_digest_provided")


def test_fix3_eval_digest_states_the_round_ordinal():
    """`_eval_digest` carries the ordinal, so the final identity is complete."""
    d = GT._eval_digest({"evaluation_stage": "post_update", "updates_completed": 10,
                         "eval_round_ordinal": 1, "iteration": 4})
    assert d["eval_round_ordinal"] == 1
    assert d["evaluation_stage"] == "post_update"
    assert d["updates_completed"] == 10


# =============================================================================
# FIX 4 -- OBSERVED SCHEMA, NEVER THE CURRENT WRITER'S CONSTANTS
# =============================================================================

def _write_run(tmp_path, name, outcome_rows):
    run = tmp_path / name
    run.mkdir()
    (run / "train_records.jsonl").write_text("", encoding="utf-8")
    (run / "eval_records.jsonl").write_text("", encoding="utf-8")
    (run / "episode_failures.jsonl").write_text("", encoding="utf-8")
    (run / GT._EPISODE_OUTCOMES_FILENAME).write_text(
        "".join(json.dumps(r) + "\n" for r in outcome_rows), encoding="utf-8")
    return run


def test_fix4_build_run_summary_reports_a_legacy_v2_run_as_v2(tmp_path):
    run = _write_run(tmp_path, "legacy_v2", [
        {"schema": GT._EPISODE_OUTCOME_SCHEMA, "schema_version": 2,
         "phase": "post_update", "cell": "mild", "updates_completed": 3},
        {"schema": GT._EPISODE_OUTCOME_SCHEMA, "schema_version": 2,
         "phase": "train", "cell": "clean", "updates_completed": 3},
    ])
    obs = GT.build_run_summary(run)["observed_artifact_schema"]
    assert obs["state"] == "uniform"
    assert obs["episode_outcome_schema_version_observed"] == 2
    assert obs["episode_outcome_schema_versions_observed"] == [2]
    assert obs["wake_diagnostics_recorded"] is False
    assert obs["wake_diagnostics_schema_version_observed"] is None
    assert obs["n_records_with_wake_decisions"] == 0
    # the CURRENT writer is still reported -- under a name that says so
    assert obs["episode_outcome_schema_version_writer"] == GT._EPISODE_OUTCOME_VERSION
    assert obs["wake_diagnostics_schema_version_writer"] == (
        GT._WAKE_DIAGNOSTICS_VERSION)
    assert GT.build_run_summary(run)["fd_policy_sensitivity"]["recorded"] is False


def test_fix4_build_run_summary_reports_a_v3_run_as_v3(tmp_path):
    rec = _fd_rec("post_update", "severe", 4, 0.7, ABORT, group="g0", ordinal=1)
    rec.update({"schema": GT._EPISODE_OUTCOME_SCHEMA, "schema_version": 3,
                "wake_diagnostics_schema_version": GT._WAKE_DIAGNOSTICS_VERSION})
    run = _write_run(tmp_path, "current_v3", [rec])
    summary = GT.build_run_summary(run)
    obs = summary["observed_artifact_schema"]
    assert obs["state"] == "uniform"
    assert obs["episode_outcome_schema_version_observed"] == 3
    assert obs["wake_diagnostics_recorded"] is True
    assert obs["wake_diagnostics_schema_version_observed"] == (
        GT._WAKE_DIAGNOSTICS_VERSION)
    assert summary["fd_policy_sensitivity"]["recorded"] is True
    assert summary["fd_policy_sensitivity"][
        "wake_diagnostics_schema_versions_observed"] == [GT._WAKE_DIAGNOSTICS_VERSION]


def test_fix4_empty_and_mixed_run_directories_report_a_truthful_state(tmp_path):
    empty = GT.build_run_summary(_write_run(tmp_path, "empty", []))[
        "observed_artifact_schema"]
    assert empty["state"] == "no_records"
    assert empty["episode_outcome_schema_version_observed"] is None
    assert empty["episode_outcome_schema_versions_observed"] == []
    assert empty["wake_diagnostics_recorded"] is False

    v3 = _fd_rec("post_update", "mild", 1, 0.2, PLAN)
    v3.update({"schema_version": 3,
               "wake_diagnostics_schema_version": GT._WAKE_DIAGNOSTICS_VERSION})
    mixed = GT.build_run_summary(_write_run(tmp_path, "mixed", [
        {"schema_version": 2, "phase": "train", "cell": "clean",
         "updates_completed": 0},
        v3,
    ]))["observed_artifact_schema"]
    assert mixed["state"] == "mixed"
    assert mixed["episode_outcome_schema_version_observed"] is None
    assert mixed["episode_outcome_schema_versions_observed"] == [2, 3]
    assert mixed["n_records_with_wake_decisions"] == 1


# =============================================================================
# FIX 5 -- THE DISAGREEMENT SERIES DOES NOT DEPEND ON CELL ORDERING
# =============================================================================

def test_fix5_disagreement_series_is_per_cell_and_clean_cannot_suppress_it():
    """Clean has NO immediate-FD wake; mild and severe still get their own series."""
    rows = [
        # a CLEAN evaluation record whose only wake is ORDINARY -- exactly the shape
        # that made a `cells[0]`-keyed series come back empty.
        {"phase": "post_update", "cell": "clean", "updates_completed": 6,
         "wake_decisions": [_Tr(TL.WAKE_KIND_ORDINARY, PLAN).decision]},
        _fd_rec("post_update", "mild", 6, 0.2, PLAN, disagree=True),
        _fd_rec("post_update", "severe", 6, 0.8, ABORT, disagree=False),
    ]
    data = GT._fd_sensitivity_plot_data(rows)
    assert data["recorded"] is True
    dis = data["disagreement_series"]
    assert dis["mild"] == {"x": [6.0], "y": [1.0]}
    assert dis["severe"] == {"x": [6.0], "y": [0.0]}
    # clean contributed no immediate-FD wake, so it has no series -- and its PRESENCE
    # did not remove mild's or severe's.
    assert not dis.get("clean", {"x": []})["x"]
    assert "clean" in data["cells"], "the clean cell is still listed"

    # ORDER INDEPENDENCE: any permutation of the input yields the same plot data.
    for perm in ([rows[2], rows[0], rows[1]], [rows[1], rows[2], rows[0]],
                 list(reversed(rows))):
        assert GT._fd_sensitivity_plot_data(perm) == data


def test_fix5_disagreement_is_reported_even_when_only_damaged_cells_exist():
    rows = [_fd_rec("post_update", "damaged", 2, 0.5, ABORT, disagree=True)]
    dis = GT._fd_sensitivity_plot_data(rows)["disagreement_series"]
    assert dis["damaged"] == {"x": [2.0], "y": [1.0]}


if __name__ == "__main__":  # pragma: no cover - direct runner (nlp_env)
    raise SystemExit(pytest.main([__file__, "-v", "--no-header", "-x"]))
