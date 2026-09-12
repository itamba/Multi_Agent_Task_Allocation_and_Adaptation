"""GENERALIZED-V2 BENCHMARK / EVALUATION -- the frozen ten-cell matched benchmark.

SOLVER-FREE and BLADE-free: every test is a statement about a deterministic function, a
validation verdict, a serialized record, or the REAL selection / evaluation loop driven
through stubbed episode seams. Runs under the base-env ``pytest`` AND under the standalone
``__main__`` runner (``nlp_env`` has no pytest).

WHAT IT PROVES, mapped to the task's proof obligations:

  PO1  HISTORICAL ISOLATION + FROZEN-POPULATION CORRECTNESS
       -- a deterministic V1 manifest keeps its PRE-TASK canonical identity (golden);
          the V1 18-stratum construct is unchanged and V1 / V2 manifests refuse each
          other; exactly ten V2 base cells and no LOW/HIGH V2 construct; twelve worlds per
          cell, development 0..1 / confirmatory 2..11, disjoint and exhaustive; membership
          never balances on R, H or H/R.
  PO2  EXACT V2 WORLD IDENTITY + MATCHED-TRIAD FIDELITY
       -- the identity carries every required field; the allocation fingerprint is
          UUID-free and structural; a mismatch on R, allocation, geometry, FD ego ordinal
          or certificate ABORTS (frozen-vs-observed and across members), never replaces.
  PO3  COMPARATOR-SAFE EVALUATION, PROFILES AND ATTRITION
       -- independent per-cell windows; a recognized rejection replaces once and spends
          the seed; an unknown exception ABORTS; the probe freezes the ACTUAL stage-2
          record; evaluation runs matched triads on identical seeds with no replacement;
          a profile selects exactly its groups; the WHOLE manifest is held out against
          max_training_attempts; early stopping stays refused; the primary
          severe-minus-mild immediate-FD abort-mass metric is paired per group, then
          macro-averaged over the ten cells, with explicit switch-rate denominators; no
          ordinary / post-FD-boundary / training row reaches it; incomplete groups
          contribute no delta and stay visible.

Run: python -m pytest tests/test_graph_generalized_v2_benchmark.py -v
     python tests/test_graph_generalized_v2_benchmark.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.solvers.match_aou_backend import (  # noqa: E402
    MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
    MATCH_AOU_BACKEND_P1_MILP_V1,
    MatchAouBackendError,
)
from match_aou.utils.blade_utils.scenario_generator import (  # noqa: E402
    TargetPlacementError,
)
from match_aou.rl.training import graph_benchmark_preflight as pf  # noqa: E402
from match_aou.rl.training import graph_generalized as gg  # noqa: E402
from match_aou.rl.training import graph_train as gt  # noqa: E402
from match_aou.rl.training.graph_episode_setup import (  # noqa: E402
    ROUTE_RELATIVE_NO_ROUTES,
    RouteRelativeNoRoutesError,
    RouteRelativePopulationRecorder,
)
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CONDITION_CLEAN,
    CONDITION_DAMAGED,
    NO_FD_ELIGIBLE_EGO,
    SEVERITY_MILD,
    SEVERITY_SEVERE,
    FuelDamageError,
    FuelDamageIntegrityError,
    FuelDamageMode,
)
from match_aou.rl.training.graph_generalized import (  # noqa: E402
    BENCHMARK_BASE_CELLS,
    BENCHMARK_SCHEMA,
    BENCHMARK_STRATA,
    CARDINALITY_SOURCE_V2_BENCHMARK,
    CARDINALITY_SOURCE_V2_BENCHMARK_PRE_SOLVE,
    CARDINALITY_SOURCE_V2_ROUTE_RELATIVE,
    EPISODE_DESIGN_GENERALIZED_V1,
    EPISODE_DESIGN_GENERALIZED_V2,
    GENERALIZED_V1,
    GENERALIZED_V2,
    HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
    LOAD_BUCKETS,
    V2_BENCHMARK_BASE_CELLS,
    V2_BENCHMARK_BASE_CELL_KEYS,
    V2_BENCHMARK_PROFILES,
    V2_BENCHMARK_SCHEMA,
    V2_BENCHMARK_WORLDS_PER_CELL,
    V2_PROFILE_CONFIRMATORY,
    V2_PROFILE_DEVELOPMENT,
    BenchmarkIdentityError,
    BenchmarkManifestError,
    RouteRelativeHiddenLoad,
    V2BenchmarkManifest,
    V2WorldIdentity,
    V2WorldPreflight,
    WorldIdentity,
    build_benchmark_manifest,
    build_v2_benchmark_manifest,
    load_benchmark_manifest,
    load_benchmark_manifest_for_design,
    load_v2_benchmark_manifest,
    manifest_from_record,
    manifest_identity,
    resolved_v2_cardinality,
    v2_allocation_fingerprint,
    v2_benchmark_pre_solve_cardinality,
    v2_manifest_from_record,
    v2_profile_world_ordinals,
    write_benchmark_manifest,
)
from match_aou.rl.training.graph_hidden_placement import (  # noqa: E402
    HiddenPlacementError,
)
from match_aou.rl.training.graph_tick_loop import (  # noqa: E402
    WAKE_KIND_IMMEDIATE_FD,
    WAKE_KIND_ORDINARY,
    WAKE_KIND_POST_FD_BOUNDARY,
)

ABORT = "SELF_PRESERVATION_ABORT"
PLAN = "PLAN_COMPLIANCE"

_FAKE_GIT_OK = {
    "available": True, "commit": "0" * 40, "branch": "test",
    "dirty": False, "dirty_path_count": 0, "reason": None,
}


# =============================================================================
# Helpers
# =============================================================================

def _raises(exc_types, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_types as exc:
        return exc
    raise AssertionError("expected %r" % (exc_types,))


def _identity_for(seed: int, agent_count: int, known_offset: int, **override):
    """A deterministic stand-in V2 identity: a pure function of the world seed."""
    route_count = 1 + (seed % agent_count)
    fields = dict(
        seed=int(seed), agent_count=int(agent_count),
        known_count=int(agent_count) + int(known_offset),
        match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1,
        route_count=route_count,
        allocation_fingerprint="alloc-%d" % seed,
        hidden_requested=1 + (seed % route_count),
        hidden_realized=1 + (seed % route_count),
        known_realized=int(agent_count) + int(known_offset),
        geometric_fingerprint=((float(seed % 90), float(seed % 180)),),
        fd_selected_ordinal=0,
        fd_certificate_fingerprint="cert-%d" % seed,
    )
    fields.update(override)
    return V2WorldIdentity(**fields)


def _load_record(identity: V2WorldIdentity) -> dict:
    return RouteRelativeHiddenLoad(
        route_count=identity.route_count, hidden_requested=identity.hidden_requested,
        derived_seed=7,
    ).to_record()


def _preflight_for(seed, agent_count, known_offset, **override) -> V2WorldPreflight:
    ident = _identity_for(seed, agent_count, known_offset, **override)
    return V2WorldPreflight(identity=ident, hidden_load=_load_record(ident),
                            construction_audit={"policy": "bounded_backoff_v1"})


def _entries(base_seed: int = 4_000_000, *, override=None, confirmatory_base=None):
    """120 preflighted world entries, 12 per base cell.

    Seeds are consecutive; with ``confirmatory_base`` the confirmatory worlds take their
    own consecutive band starting there, far from the development seeds.
    """
    out = []
    seed = int(base_seed)
    conf_seed = None if confirmatory_base is None else int(confirmatory_base)
    for (a, d) in V2_BENCHMARK_BASE_CELLS:
        for ordinal in range(V2_BENCHMARK_WORLDS_PER_CELL):
            extra = (override or (lambda *_: {}))(a, d, ordinal) or {}
            if conf_seed is not None and ordinal >= 2:
                this, conf_seed = conf_seed, conf_seed + 1
            else:
                this, seed = seed, seed + 1
            out.append({"agent_count": a, "known_offset": d, "world_ordinal": ordinal,
                        "seed": this, "preflight": _preflight_for(this, a, d, **extra)})
    return out


def _manifest(base_seed: int = 4_000_000, **kw) -> V2BenchmarkManifest:
    return build_v2_benchmark_manifest(worlds=_entries(base_seed, **kw), label="test")


def _v2_cfg(**kw):
    base = dict(
        n_iterations=2, episode_design=EPISODE_DESIGN_GENERALIZED_V2,
        match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
        eval_every=0, eval_episodes=0, episodes_per_iteration=8,
        generalized_max_attempts_per_iteration=12,
    )
    base.update(kw)
    return gt.TrainConfig(**base)


class _Tr:
    """The three fields the evaluation round reads off a stub transition."""

    def __init__(self, ego_id, meta_action, decision):
        self.ego_id = ego_id
        self.meta_action = meta_action
        self.decision = decision


def _wake(kind, abort_mass, selected):
    return {"wake_kind": kind, "selected_meta_action_name": selected,
            "aggregate_probability_per_meta_action": {
                ABORT: abort_mass, PLAN: 1.0 - abort_mass,
                "OPPORTUNISTIC_ENGAGEMENT": 0.0}}


_CELL_BY_MODE = {
    FuelDamageMode.FORCED_CLEAN: CONDITION_CLEAN,
    FuelDamageMode.FORCED_MILD: SEVERITY_MILD,
    FuelDamageMode.FORCED_SEVERE: SEVERITY_SEVERE,
}


def _stub_outcome(mode, *, pre, load, identity, decisions, reward=-0.25):
    cell = _CELL_BY_MODE[mode]
    damaged = cell != CONDITION_CLEAN
    condition = CONDITION_DAMAGED if damaged else CONDITION_CLEAN
    trajectory = [_Tr("ego_0", 0, d) for d in decisions]
    return gt._EpisodeOutcome(
        trajectory=trajectory, reward=reward, ticks=10, ended="done",
        n_wakes=len(trajectory), confirmed_kills=1, n_dead=0, seconds=0.01,
        targets_confirmed_unique=1, targets_total=identity.known_realized
        + identity.hidden_realized,
        known_target_names=("K",), hidden_target_names=("H",),
        known_confirmed_names=("K",), hidden_confirmed_names=(),
        fuel_damage_plan={"condition": condition,
                          "severity": cell if damaged else None,
                          "ego_id": "ego_0" if damaged else None},
        fuel_damage_outcome={"condition": condition,
                             "severity": cell if damaged else None,
                             "fired": damaged, "wake_occurred": damaged,
                             "wake_meta_action": None},
        selected_ego_rtb_issued=True if damaged else None,
        cardinality=resolved_v2_cardinality(pre, load),
        pre_solve_cardinality=pre,
        route_relative_load=load,
        hidden_realized=identity.hidden_realized,
        construction_audit={"policy": "bounded_backoff_v1"},
        world_identity=WorldIdentity(
            hidden_realized=identity.hidden_realized,
            known_realized=identity.known_realized,
            geometric_fingerprint=identity.geometric_fingerprint,
            fd_selected_ordinal=identity.fd_selected_ordinal,
            fd_certificate_fingerprint=identity.fd_certificate_fingerprint),
        allocation_fingerprint=identity.allocation_fingerprint,
    )


def _stub_body(manifest, *, calls, fail=None, drift=None, decisions=None):
    """A stubbed `_run_one_episode` that rebuilds each member as its frozen world.

    `fail(seed, cell)` -> exception to raise AFTER stage 2; `drift(seed, cell)` -> identity
    overrides for this member; `decisions(seed, cell)` -> its wake records.
    """
    by_seed = {int(w.seed): w for w in manifest.worlds}

    def body(policy, gen, cfg_, *, seed, episode_tag, deterministic,
             fuel_damage_mode=None, pre_solve_cardinality=None,
             population_recorder=None, **extra):
        assert deterministic is True
        assert "cardinality" not in extra, "a V2 member must never be pre-resolved"
        cell = _CELL_BY_MODE[fuel_damage_mode]
        calls.append((int(seed), cell, int(episode_tag), pre_solve_cardinality))
        frozen = by_seed[int(seed)].preflight.identity
        ident = _identity_for(seed, frozen.agent_count, frozen.known_offset,
                              **((drift or (lambda *_: {}))(seed, cell) or {}))
        load = RouteRelativeHiddenLoad(route_count=ident.route_count,
                                       hidden_requested=ident.hidden_requested)
        population_recorder.record(load)
        failure = (fail or (lambda *_: None))(seed, cell)
        if failure is not None:
            raise failure
        wakes = (decisions or (lambda s, c: [
            _wake(WAKE_KIND_IMMEDIATE_FD, 0.2 if c == SEVERITY_MILD else 0.7,
                  PLAN if c == SEVERITY_MILD else ABORT)] if c != CONDITION_CLEAN
            else []))(seed, cell)
        return _stub_outcome(fuel_damage_mode, pre=pre_solve_cardinality, load=load,
                             identity=ident, decisions=wakes)

    return body


def _run_v2_round(cfg, manifest, tmp_path, *, body):
    saved = gt._run_one_episode
    gt._run_one_episode = body
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return gt.evaluate_benchmark(
                None, object(), cfg, manifest, iteration=None, stage="post_update",
                updates_completed=1, round_ordinal=0,
                failures_path=Path(tmp_path) / "episode_failures.jsonl",
                outcomes_path=Path(tmp_path) / "episode_outcomes.jsonl")
    finally:
        gt._run_one_episode = saved


def _jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


# =============================================================================
# PO1 -- historical isolation and frozen-population correctness
# =============================================================================

# Captured from `fe7b449c94281bb12fdadc12be89ee36f447c79a` BEFORE this task edited
# anything: (manifest_id, SHA-256 of the canonical to_record() bytes).
_V1_GOLDEN_SCALE = ("cae55fb3c01682b32ec02bbd43c05045f7c3fdc1ed5d076e5ebaa2acdea0e65b",
                    "3e593945acb3dac75cae4b8d599795f0987048fd4cf8485b700930d7a68a4073")
_V1_GOLDEN_PREFLIGHTED = (
    "1c42760350abc58f84863046029083005ffe7d5f023ed4dde9b06864322b1d2e",
    "86e9509cc6ad050122d0d65393f4e1185abf25d11fc186799eefc4377e76bcaa")


def test_po1_a_v1_manifest_keeps_its_pre_task_canonical_identity() -> None:
    """GOLDEN: two deterministic V1 manifests hash exactly as they did at the base SHA."""
    m = build_benchmark_manifest(worlds_per_cell=2, benchmark_base_seed=5_000_000)
    assert m.manifest_id == _V1_GOLDEN_SCALE[0]
    assert hashlib.sha256(gg._canonical_json(m.to_record()).encode()).hexdigest() \
        == _V1_GOLDEN_SCALE[1]
    pre = {"hidden_realized": 1, "known_realized": 2,
           "geometric_fingerprint": [[31.5, 34.25]], "fd_selected_ordinal": 1,
           "fd_certificate_fingerprint": "abc",
           "construction_audit": {"policy": "bounded_backoff_v1"}}
    m2 = build_benchmark_manifest(worlds=[
        {"agent_count": a, "load_bucket": b, "seed": 7_000_000 + i, "preflight": pre}
        for i, (a, b) in enumerate(BENCHMARK_BASE_CELLS)], label="golden", notes="n")
    assert m2.manifest_id == _V1_GOLDEN_PREFLIGHTED[0]
    assert hashlib.sha256(gg._canonical_json(m2.to_record()).encode()).hexdigest() \
        == _V1_GOLDEN_PREFLIGHTED[1]
    assert manifest_from_record(m2.to_record()).manifest_id == m2.manifest_id


def test_po1_the_v1_construct_is_unchanged() -> None:
    assert len(BENCHMARK_STRATA) == 18 and len(BENCHMARK_BASE_CELLS) == 6
    assert LOAD_BUCKETS == ("low", "high")
    assert BENCHMARK_SCHEMA == "generalized_v1_benchmark_manifest"
    assert V2_BENCHMARK_SCHEMA != BENCHMARK_SCHEMA


def test_po1_exactly_ten_v2_base_cells_and_no_low_high_construct() -> None:
    assert len(V2_BENCHMARK_BASE_CELLS) == 10
    assert V2_BENCHMARK_BASE_CELLS == tuple(
        (a, d) for a in (2, 3, 4, 5, 6) for d in (0, 2))
    assert V2_BENCHMARK_BASE_CELL_KEYS[0] == "A2-D0"
    assert V2_BENCHMARK_BASE_CELL_KEYS[-1] == "A6-D2"
    for key in V2_BENCHMARK_BASE_CELL_KEYS:
        assert "low" not in key.lower() and "high" not in key.lower()
    # No V2 LOW/HIGH alias exists anywhere in the population module.
    assert not [n for n in dir(gg) if n.startswith("V2") and ("LOW" in n or "HIGH" in n)]
    payload_text = json.dumps(_manifest().payload())
    assert "load_bucket" not in payload_text
    assert "not_strata" in payload_text


def test_po1_twelve_worlds_per_cell_and_two_disjoint_exhaustive_profiles() -> None:
    assert V2_BENCHMARK_WORLDS_PER_CELL == 12
    dev = v2_profile_world_ordinals(V2_PROFILE_DEVELOPMENT)
    conf = v2_profile_world_ordinals(V2_PROFILE_CONFIRMATORY)
    assert dev == (0, 1) and conf == tuple(range(2, 12))
    assert not set(dev) & set(conf) and sorted(dev + conf) == list(range(12))
    m = _manifest()
    assert m.n_worlds == 120 and m.n_members == 360
    d, c = m.profile_worlds(V2_PROFILE_DEVELOPMENT), m.profile_worlds(V2_PROFILE_CONFIRMATORY)
    assert len(d) == 20 and len(c) == 100
    assert {w.key for w in d}.isdisjoint({w.key for w in c})
    assert {w.key for w in d} | {w.key for w in c} == {w.key for w in m.worlds}
    for profile, worlds in ((V2_PROFILE_DEVELOPMENT, d), (V2_PROFILE_CONFIRMATORY, c)):
        per_cell = {}
        for w in worlds:
            per_cell[w.base_cell_key] = per_cell.get(w.base_cell_key, 0) + 1
        assert len(per_cell) == 10 and len(set(per_cell.values())) == 1, profile
    _raises(BenchmarkManifestError, v2_profile_world_ordinals, "dev")


def test_po1_a_cell_without_exactly_twelve_worlds_is_refused() -> None:
    entries = _entries()
    _raises(BenchmarkManifestError, build_v2_benchmark_manifest, worlds=entries[1:])
    dup = _entries()
    dup[1] = dict(dup[1], seed=dup[0]["seed"],
                  preflight=_preflight_for(dup[0]["seed"], 2, 0))
    _raises(BenchmarkManifestError, build_v2_benchmark_manifest, worlds=dup)
    bad = _entries()
    bad[0] = dict(bad[0], world_ordinal=12)
    _raises(BenchmarkManifestError, build_v2_benchmark_manifest, worlds=bad)


def test_po1_membership_is_never_balanced_on_route_count_or_hidden_load() -> None:
    """A manifest whose worlds all have R=1, H=1 is exactly as valid as a varied one."""
    flat = _manifest(override=lambda a, d, o: {"route_count": 1, "hidden_requested": 1,
                                                "hidden_realized": 1})
    varied = _manifest()
    assert flat.n_worlds == varied.n_worlds == 120
    assert flat.manifest_id != varied.manifest_id      # R / H are part of the IDENTITY
    for name in ("route_count", "hidden_requested"):
        assert name in json.loads(json.dumps(flat.payload()))["not_strata"] or \
            name == "hidden_requested"


def test_po1_v1_and_v2_manifests_refuse_each_other(tmp_path=None) -> None:
    v1 = build_benchmark_manifest(worlds_per_cell=1, benchmark_base_seed=1_000_000)
    v2 = _manifest()
    exc = _raises(BenchmarkManifestError, v2_manifest_from_record, v1.to_record())
    assert "generalized_v1 18-stratum manifest" in str(exc)
    _raises(BenchmarkManifestError, manifest_from_record, v2.to_record())
    assert v2_manifest_from_record(v2.to_record()).manifest_id == v2.manifest_id


def test_po1_a_v2_manifest_is_content_addressed_and_tampering_is_refused() -> None:
    import tempfile
    m = _manifest()
    assert m.manifest_id == manifest_identity(m.payload())
    with tempfile.TemporaryDirectory() as tmp:
        path = write_benchmark_manifest(m, Path(tmp) / "v2.json")
        assert load_v2_benchmark_manifest(path).manifest_id == m.manifest_id
        assert load_benchmark_manifest_for_design(path, GENERALIZED_V2).manifest_id \
            == m.manifest_id
        _raises(BenchmarkManifestError, load_benchmark_manifest_for_design, path,
                GENERALIZED_V1)
        _raises(BenchmarkManifestError, load_benchmark_manifest, path)
    tampered = m.to_record()
    tampered["worlds"][0]["preflight"]["identity"]["route_count"] = 99
    _raises(BenchmarkManifestError, v2_manifest_from_record, tampered)
    forged = m.to_record()
    forged["injected"] = True
    forged.pop("manifest_id")
    forged["manifest_id"] = manifest_identity(forged)
    _raises(BenchmarkManifestError, v2_manifest_from_record, forged)


def test_po1_profiles_are_refused_outside_generalized_v2() -> None:
    _raises(ValueError, gt.TrainConfig(n_iterations=1,
                                       benchmark_profile="development").validate)
    _raises(ValueError, gt.TrainConfig(
        n_iterations=2, episode_design=EPISODE_DESIGN_GENERALIZED_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE, eval_every=0, eval_episodes=0,
        generalized_max_attempts_per_iteration=8,
        benchmark_profile="development").validate)
    _raises(ValueError, _v2_cfg(benchmark_profile="development").validate)  # no manifest
    _raises(ValueError, _v2_cfg(benchmark_manifest="m.json",
                                benchmark_profile="final").validate)


def test_po1_the_benchmark_pre_solve_half_is_stage_correct() -> None:
    pre = v2_benchmark_pre_solve_cardinality(agent_count=4, known_offset=2)
    assert (pre.agent_count, pre.known_count) == (4, 6)
    assert pre.source == CARDINALITY_SOURCE_V2_BENCHMARK_PRE_SOLVE
    assert pre.rng_domain is None and pre.derived_seed is None
    assert pre.to_record()["rng_domain"] is None
    assert not hasattr(pre, "hidden_requested")
    load = RouteRelativeHiddenLoad(route_count=3, hidden_requested=2)
    assert resolved_v2_cardinality(pre, load).source == CARDINALITY_SOURCE_V2_BENCHMARK
    sampled = gt.sample_generalized_v2_pre_solve_cardinality(episode_seed=5)
    assert resolved_v2_cardinality(sampled, load).source \
        == CARDINALITY_SOURCE_V2_ROUTE_RELATIVE
    for bad in ((8, 0), (2, 1), (True, 0)):
        _raises(BenchmarkManifestError, v2_benchmark_pre_solve_cardinality,
                agent_count=bad[0], known_offset=bad[1])


# =============================================================================
# PO2 -- exact V2 world identity and matched-triad fidelity
# =============================================================================

def test_po2_the_identity_carries_every_required_component() -> None:
    record = _identity_for(123, 3, 2).to_record()
    for name in ("seed", "agent_count", "known_count", "known_offset",
                 "match_aou_backend", "route_count", "allocation_fingerprint",
                 "hidden_requested", "hidden_realized", "geometric_fingerprint",
                 "fd_selected_ordinal", "fd_certificate_fingerprint"):
        assert name in record, name
    assert record["known_offset"] == 2
    assert V2WorldIdentity.from_record(record) == _identity_for(123, 3, 2)
    world_record = _manifest().worlds[0].to_record()
    assert "profile" in world_record and "base_cell" in world_record
    assert "load_bucket" not in world_record


def test_po2_the_allocation_fingerprint_is_uuid_free_and_structural() -> None:
    known = ["uuid-a", "uuid-b", "uuid-c"]
    a_init = {"e0": [(0, 0, 0)], "e1": [(1, 0, 0)]}
    base = v2_allocation_fingerprint(a_init, agent_ids=["e0", "e1", "e2"],
                                     belief_target_ids=["uuid-a", "uuid-c"],
                                     known_target_ids=known)
    # The same structure under DIFFERENT generated uuids -> the same fingerprint.
    renamed = v2_allocation_fingerprint(
        {"x0": [(0, 0, 0)], "x1": [(1, 0, 0)]}, agent_ids=["x0", "x1", "x2"],
        belief_target_ids=["Q", "S"], known_target_ids=["Q", "R", "S"])
    assert renamed == base
    # A different routed ego, target, or level -> a different fingerprint.
    other_ego = v2_allocation_fingerprint(
        {"e0": [(0, 0, 0)], "e2": [(1, 0, 0)]}, agent_ids=["e0", "e1", "e2"],
        belief_target_ids=["uuid-a", "uuid-c"], known_target_ids=known)
    other_target = v2_allocation_fingerprint(
        a_init, agent_ids=["e0", "e1", "e2"], belief_target_ids=["uuid-a", "uuid-b"],
        known_target_ids=known)
    other_level = v2_allocation_fingerprint(
        {"e0": [(0, 0, 1)], "e1": [(1, 0, 0)]}, agent_ids=["e0", "e1", "e2"],
        belief_target_ids=["uuid-a", "uuid-c"], known_target_ids=known)
    assert len({base, other_ego, other_target, other_level}) == 4
    _raises(ValueError, v2_allocation_fingerprint, {"ghost": [(0, 0, 0)]},
            agent_ids=["e0"], belief_target_ids=["uuid-a"], known_target_ids=known)
    _raises(ValueError, v2_allocation_fingerprint, {"e0": [(5, 0, 0)]},
            agent_ids=["e0"], belief_target_ids=["uuid-a"], known_target_ids=known)


def test_po2_a_mismatch_on_any_frozen_component_aborts() -> None:
    m = _manifest()
    world = m.worlds[5]
    frozen = world.preflight.identity
    gg.require_v2_world_matches_manifest(world, frozen)
    for override in ({"route_count": frozen.route_count + 1},
                     {"allocation_fingerprint": "other"},
                     {"geometric_fingerprint": ((0.0, 0.0),)},
                     {"fd_selected_ordinal": 3},
                     {"fd_certificate_fingerprint": "other"},
                     {"hidden_requested": frozen.hidden_requested + 1}):
        observed = _identity_for(world.seed, world.agent_count, world.known_offset,
                                 **override)
        exc = _raises(BenchmarkIdentityError, gg.require_v2_world_matches_manifest,
                      world, observed)
        assert "REFUSED" in str(exc)
        _raises(BenchmarkIdentityError, gg.require_v2_matched_group_identity, world,
                {"clean": frozen, "severe": observed})


def test_po2_a_world_whose_frozen_identity_contradicts_it_is_refused() -> None:
    import dataclasses
    for override in ({"seed": 1}, {"agent_count": 3}, {"known_count": 3},
                     {"match_aou_backend": MATCH_AOU_BACKEND_LEGACY_MINLP_V1}):
        ident = dataclasses.replace(_identity_for(4_000_000, 2, 0), **override)
        _raises(BenchmarkManifestError, gg.V2BenchmarkWorld, agent_count=2,
                known_offset=0, world_ordinal=0, seed=4_000_000,
                preflight=V2WorldPreflight(identity=ident,
                                           hidden_load=_load_record(ident)))


# =============================================================================
# PO3 -- the V2 preflight: deterministic, fail-closed, population-blind
# =============================================================================

def _stub_probe(*, reject=None, calls=None, identity_override=None):
    seen = [] if calls is None else calls

    def probe(cfg, gen, *, seed, agent_count, known_offset, population_recorder):
        seen.append(int(seed))
        failure = (reject or {}).get(int(seed))
        if failure is not None:
            raise failure
        pre = _preflight_for(int(seed), agent_count, known_offset,
                             **((identity_override or (lambda *_: {}))(seed) or {}))
        population_recorder.record(RouteRelativeHiddenLoad(
            route_count=pre.identity.route_count,
            hidden_requested=pre.identity.hidden_requested))
        return pre

    return probe, seen


def _run_v2_preflight(*, probe, base_seed=900_000, max_candidates=14, output_dir=None,
                      cfg=None, worlds_per_cell=12):
    return pf.run_benchmark_preflight(
        cfg or _v2_cfg(), worlds_per_cell=worlds_per_cell,
        benchmark_base_seed=base_seed, max_candidates_per_cell=max_candidates,
        output_dir=output_dir, probe=probe, generator=object(),
        provenance=_FAKE_GIT_OK)


def test_po3_v2_windows_are_independent_per_cell() -> None:
    windows = pf.v2_cell_windows(benchmark_base_seed=900_000, max_candidates_per_cell=14)
    assert len(windows) == 10
    assert [w.key for w in windows] == list(V2_BENCHMARK_BASE_CELL_KEYS)
    for c, w in enumerate(windows):
        assert (w.start, w.stop) == (900_000 + 14 * c, 900_000 + 14 * (c + 1))
    plain = _run_v2_preflight(probe=_stub_probe()[0])
    rejects = {900_000 + i: pf.EpisodeAttemptError(
        "setup", RouteRelativeNoRoutesError("no routes")) for i in range(2)}
    with_rejects = _run_v2_preflight(probe=_stub_probe(reject=rejects)[0])
    a1 = [w.seed for w in plain.manifest.worlds if w.base_cell_key == "A2-D2"]
    b1 = [w.seed for w in with_rejects.manifest.worlds if w.base_cell_key == "A2-D2"]
    assert a1 == b1, "a rejection in cell A2-D0 moved cell A2-D2's accepted seeds"


def test_po3_a_recognized_rejection_replaces_once_and_spends_the_seed() -> None:
    no_routes = pf.EpisodeAttemptError("setup", RouteRelativeNoRoutesError("zero"))
    probe, seen = _stub_probe(reject={900_001: no_routes})
    result = _run_v2_preflight(probe=probe)
    assert seen.count(900_001) == 1 and len(seen) == len(set(seen))
    cell0 = [w.seed for w in result.manifest.worlds if w.base_cell_key == "A2-D0"]
    assert 900_001 not in cell0 and cell0 == [900_000] + list(range(900_002, 900_013))
    rejected = [c for c in result.candidates if not c.accepted]
    assert len(rejected) == 1 and rejected[0].reason == ROUTE_RELATIVE_NO_ROUTES
    assert result.report["status"] == pf.PREFLIGHT_STATUS_COMPLETE
    assert result.report["design"] == EPISODE_DESIGN_GENERALIZED_V2
    assert result.manifest.n_worlds == 120


def test_po3_every_recognized_world_level_reason_is_replacement_eligible() -> None:
    cases = {
        900_000: pf.EpisodeAttemptError("generation", TargetPlacementError("geom")),
        900_001: pf.EpisodeAttemptError("setup", RouteRelativeNoRoutesError("R=0")),
        900_002: pf.EpisodeAttemptError("setup", HiddenPlacementError("none placed")),
        900_003: pf.EpisodeAttemptError("setup", FuelDamageError(
            "%s: reasons: pre_event_popup_risk" % NO_FD_ELIGIBLE_EGO)),
    }
    result = _run_v2_preflight(probe=_stub_probe(reject=cases)[0], max_candidates=16)
    reasons = {c.seed: c.reason for c in result.candidates if not c.accepted}
    assert reasons == {900_000: pf.V2_REJECTION_GENERATOR_PLACEMENT,
                       900_001: ROUTE_RELATIVE_NO_ROUTES,
                       900_002: pf.V2_REJECTION_HIDDEN_PLACEMENT,
                       900_003: NO_FD_ELIGIBLE_EGO}


def test_po3_an_unknown_or_unclassified_exception_aborts_the_preflight() -> None:
    import tempfile
    for failure in (
        pf.EpisodeAttemptError("setup", RuntimeError("an unrelated setup failure")),
        pf.EpisodeAttemptError("setup", ValueError("unknown")),
        pf.EpisodeAttemptError("setup", FuelDamageError("a window fault, no marker")),
        pf.EpisodeAttemptError("run", RouteRelativeNoRoutesError("wrong stage")),
        pf.EpisodeAttemptError("generation", RuntimeError("not a placement refusal")),
        RuntimeError("bare"),
        gt.EpisodeRosterError("roster"),
        BenchmarkIdentityError("members disagree"),
        FuelDamageIntegrityError("certificate"),
        MatchAouBackendError("backend"),
    ):
        probe, seen = _stub_probe(reject={900_003: failure})
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _run_v2_preflight(probe=probe, output_dir=Path(tmp))
            except BaseException as exc:          # noqa: BLE001
                assert exc is failure, (type(exc), failure)
            else:
                raise AssertionError("unclassified %r was replaced" % (failure,))
            assert not (Path(tmp) / "benchmark_manifest.json").exists()
        assert max(seen) == 900_003, "no seed after an unclassified fault was tried"


def test_po3_acceptance_never_reads_route_count_or_hidden_load() -> None:
    """Every eligible candidate is accepted in seed order, whatever its R and H are."""
    flat = _run_v2_preflight(probe=_stub_probe(identity_override=lambda s: {
        "route_count": 1, "hidden_requested": 1, "hidden_realized": 1})[0])
    varied = _run_v2_preflight(probe=_stub_probe()[0])
    assert [w.seed for w in flat.manifest.worlds] == [w.seed for w in varied.manifest.worlds]
    short = _run_v2_preflight(probe=_stub_probe(identity_override=lambda s: {
        "hidden_realized": 1})[0])
    assert short.manifest.n_worlds == 120


def test_po3_window_exhaustion_writes_no_manifest_but_a_failed_report() -> None:
    import tempfile
    rejects = {900_000 + i: pf.EpisodeAttemptError(
        "setup", HiddenPlacementError("x")) for i in range(3)}
    with tempfile.TemporaryDirectory() as tmp:
        exc = _raises(pf.BenchmarkPreflightError, _run_v2_preflight,
                      probe=_stub_probe(reject=rejects)[0], max_candidates=14,
                      output_dir=Path(tmp))
        assert exc.report["status"] == pf.PREFLIGHT_STATUS_FAILED
        assert exc.report["manifest"] is None
        assert exc.report["failure"]["base_cell"] == "A2-D0"
        assert not (Path(tmp) / "benchmark_manifest.json").exists()
        assert (Path(tmp) / "benchmark_preflight_report.json").exists()


def test_po3_the_v2_preflight_config_requires_p1_and_twelve_worlds() -> None:
    probe = _stub_probe()[0]
    _raises(pf.BenchmarkPreflightError, _run_v2_preflight, probe=probe,
            cfg=_v2_cfg(match_aou_backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1))
    _raises(pf.BenchmarkPreflightError, _run_v2_preflight, probe=probe, worlds_per_cell=2)
    _raises(pf.BenchmarkPreflightError, _run_v2_preflight, probe=probe, max_candidates=11)


def test_po3_the_probe_freezes_the_actual_stage_two_record_never_a_redraw() -> None:
    """`probe_v2_world` calls the PRODUCTION route-relative setup and freezes ctx's draw."""
    import tempfile

    class _Ctx:
        def __init__(self, load):
            self.route_relative_load = load
            self.construction_audit = None
            self.env = type("E", (), {"close": lambda self: None})()

    class _Plan:
        def to_record(self):
            return {"eligibility_audit": {"selected_ordinal": 1}, "certificate": {"t": 5}}

    # A load deliberately DIFFERENT from what the production rule would draw for this
    # seed: the probe must freeze what the context carries, never recompute it.
    seed = 910_000
    production = gg.resolve_route_relative_hidden_load(episode_seed=seed, route_count=4)
    actual = RouteRelativeHiddenLoad(
        route_count=4, hidden_requested=1 + (production.hidden_requested % 4))
    assert actual.hidden_requested != production.hidden_requested
    seen_kwargs = {}

    def fake_setup(text, **kwargs):
        seen_kwargs.update(kwargs)
        kwargs["population_recorder"].record(actual)
        return _Ctx(actual)

    with tempfile.TemporaryDirectory() as tmp:
        scen = Path(tmp) / "s.json"
        scen.write_text("{}", encoding="utf-8")
        gen = type("G", (), {"generate": lambda self, episode, config: scen})()
        saved = {n: getattr(pf, n) for n in (
            "setup_episode", "build_variation_config", "_episode_target_roster",
            "_scheduled_cell", "_require_scheduled_cell", "_v2_allocation_fingerprint",
            "build_fuel_damage_controller", "_observe_world_identity")}
        pf.setup_episode = fake_setup
        pf.build_variation_config = lambda cfg, seed, cardinality: cardinality
        pf._episode_target_roster = lambda ctx: None
        pf._scheduled_cell = lambda cell, audit: cell
        pf._require_scheduled_cell = lambda roster, scheduled: None
        pf._v2_allocation_fingerprint = lambda ctx: "alloc-frozen"
        pf.build_fuel_damage_controller = lambda ctx, episode_seed, params: type(
            "C", (), {"plan": _Plan()})()
        pf._observe_world_identity = lambda ctx, roster, fd_plan_record: WorldIdentity(
            hidden_realized=2, known_realized=6, geometric_fingerprint=((1.0, 2.0),),
            fd_selected_ordinal=1, fd_certificate_fingerprint="cert")
        try:
            recorder = RouteRelativePopulationRecorder()
            pre = pf.probe_v2_world(_v2_cfg(), gen, seed=seed, agent_count=4,
                                    known_offset=2, population_recorder=recorder)
            # ... and a context whose record is NOT the recorder's aborts.
            def split_setup(text, **kwargs):
                kwargs["population_recorder"].record(actual)
                return _Ctx(RouteRelativeHiddenLoad(route_count=4, hidden_requested=4))
            pf.setup_episode = split_setup
            _raises(gt.MeasurementIntegrityError, pf.probe_v2_world, _v2_cfg(), gen,
                    seed=seed, agent_count=4, known_offset=2,
                    population_recorder=RouteRelativePopulationRecorder())
        finally:
            for n, v in saved.items():
                setattr(pf, n, v)

    assert pre.identity.hidden_requested == actual.hidden_requested
    assert pre.identity.route_count == 4 and pre.hidden_load == actual.to_record()
    assert (pre.identity.agent_count, pre.identity.known_count) == (4, 6)
    assert pre.identity.allocation_fingerprint == "alloc-frozen"
    assert "n_hidden" not in seen_kwargs
    assert seen_kwargs["hidden_load_policy"] == HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2
    assert seen_kwargs["hidden_load_seed"] == seed
    assert seen_kwargs["known_requested"] == 6
    assert seen_kwargs["match_aou_backend"] == MATCH_AOU_BACKEND_P1_MILP_V1


def test_po3_the_typed_no_routes_refusal_is_still_an_ordinary_failed_attempt() -> None:
    """Outside the preflight the typed R=0 refusal is an accounted `setup` failure."""
    import tempfile

    def fake_setup(text, **kwargs):
        raise RouteRelativeNoRoutesError("the known-only solve allocated nothing")

    with tempfile.TemporaryDirectory() as tmp:
        scen = Path(tmp) / "s.json"
        scen.write_text("{}", encoding="utf-8")
        gen = type("G", (), {"generate": lambda self, episode, config: scen})()
        saved = gt.setup_episode
        gt.setup_episode = fake_setup
        try:
            exc = _raises(gt.EpisodeAttemptError, gt._run_one_episode, None, gen,
                          _v2_cfg(), seed=5, episode_tag=5, deterministic=False,
                          pre_solve_cardinality=gt.v2_pre_solve_cardinality(_v2_cfg(), 5))
        finally:
            gt.setup_episode = saved
    assert exc.stage == "setup" and isinstance(exc.original, RouteRelativeNoRoutesError)


# =============================================================================
# PO3 -- the V2 evaluation round
# =============================================================================

def _eval_cfg(profile=V2_PROFILE_DEVELOPMENT, **kw):
    return _v2_cfg(eval_every=1, eval_episodes=1, benchmark_manifest="m.json",
                   benchmark_profile=profile, **kw)


def test_po3_evaluation_runs_matched_triads_on_identical_seeds() -> None:
    import tempfile
    m = _manifest()
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        record = _run_v2_round(_eval_cfg(), m, tmp, body=_stub_body(m, calls=calls))
        outcomes = _jsonl(Path(tmp) / "episode_outcomes.jsonl")
    dev = m.profile_worlds(V2_PROFILE_DEVELOPMENT)
    assert record["n_attempted"] == 60 and record["n_successful"] == 60
    assert record["n_groups_successful"] == 20
    assert record["benchmark_profile"] == V2_PROFILE_DEVELOPMENT
    assert record["benchmark_design"] == EPISODE_DESIGN_GENERALIZED_V2
    by_seed = {}
    for seed, cell, tag, pre in calls:
        by_seed.setdefault(seed, []).append((cell, tag, pre))
    assert sorted(by_seed) == sorted(w.seed for w in dev)
    for seed, members in by_seed.items():
        assert [c for c, _t, _p in members] == [CONDITION_CLEAN, SEVERITY_MILD,
                                                SEVERITY_SEVERE]
        assert len({t for _c, t, _p in members}) == 3
        assert len({(p.agent_count, p.known_count, p.source)
                    for _c, _t, p in members}) == 1
    assert len({t for _s, _c, t, _p in calls}) == 60
    # The outcome record retains the V2 population identity of each member.
    rec = outcomes[0]
    v2 = rec["benchmark_v2"]
    assert rec["benchmark_manifest_id"] == m.manifest_id
    assert rec["benchmark_group_key"] == dev[0].key
    assert rec["benchmark_stratum"] == dev[0].base_cell_key
    assert v2["profile"] == V2_PROFILE_DEVELOPMENT and v2["base_cell"] == "A2-D0"
    assert (v2["agent_count"], v2["known_count"], v2["known_offset"]) == (2, 2, 0)
    assert v2["frozen_route_count"] == dev[0].preflight.identity.route_count
    pop = rec["generalized_v2_population"]
    assert pop["route_count_at_hidden_resolution"] == dev[0].preflight.identity.route_count
    assert pop["hidden_load"]["hidden_requested"] \
        == dev[0].preflight.identity.hidden_requested
    assert pop["pre_solve_rng_domain"] is None
    assert rec["cardinality_source"] == CARDINALITY_SOURCE_V2_BENCHMARK
    assert rec["hidden_realized"] == dev[0].preflight.identity.hidden_realized
    assert "construction_audit" in rec and "wake_decisions" in rec
    assert rec["benchmark_world_identity"]["allocation_fingerprint"] \
        == dev[0].preflight.identity.allocation_fingerprint


def test_po3_a_profile_selects_exactly_its_frozen_groups() -> None:
    import tempfile
    m = _manifest()
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        record = _run_v2_round(_eval_cfg(V2_PROFILE_CONFIRMATORY), m, tmp,
                               body=_stub_body(m, calls=calls))
    groups = record["v2_benchmark_groups"]
    expected = [w.key for w in m.profile_worlds(V2_PROFILE_CONFIRMATORY)]
    assert groups["evaluated_group_keys"] == expected and len(expected) == 100
    assert all(int(k.split("-w")[1]) >= 2 for k in expected)
    assert {s for s, *_ in calls} == {w.seed for w in m.profile_worlds(
        V2_PROFILE_CONFIRMATORY)}
    assert groups["benchmark_profile_identity"]["group_keys"] == expected
    assert groups["evaluated_group_keys_sha256"] == gg.canonical_digest(
        {"group_keys": expected})


def test_po3_a_failed_member_is_never_replaced_and_its_group_stays_visible() -> None:
    import tempfile
    m = _manifest()
    doomed = m.profile_worlds(V2_PROFILE_DEVELOPMENT)[3]
    calls = []
    body = _stub_body(m, calls=calls, fail=lambda s, c: (
        gt.EpisodeAttemptError("setup", HiddenPlacementError("none"))
        if (s == doomed.seed and c == SEVERITY_SEVERE) else None))
    with tempfile.TemporaryDirectory() as tmp:
        record = _run_v2_round(_eval_cfg(), m, tmp, body=body)
        ledger = _jsonl(Path(tmp) / "episode_failures.jsonl")
    assert record["n_failed"] == 1 and record["n_groups_successful"] == 19
    assert len(calls) == 60, "a member was retried or replaced"
    groups = record["v2_benchmark_groups"]
    assert groups["incomplete_group_keys"] == [doomed.key]
    assert doomed.key not in groups["metric_eligible_group_keys"]
    behaviour = record["v2_behaviour"]
    assert behaviour["n_groups_incomplete"] == 1
    assert behaviour["n_groups_metric_eligible"] == 19
    assert record["eval_delta_severe_minus_mild_n"] == 19
    row = [r for r in behaviour["groups"] if r["group_key"] == doomed.key][0]
    assert row["not_measurable_reason"] == "incomplete_group"
    assert row["severe_minus_mild_abort_mass"] is None
    assert len(ledger) == 1
    entry = ledger[0]
    assert entry["benchmark_group_key"] == doomed.key
    assert entry["benchmark_v2"]["profile"] == V2_PROFILE_DEVELOPMENT
    assert entry["generalized_v2_population"]["stage_resolved"] == "route_relative"
    assert entry["cardinality_source"] == CARDINALITY_SOURCE_V2_BENCHMARK


def test_po3_an_identity_mismatch_during_evaluation_aborts() -> None:
    import tempfile
    m = _manifest()
    target = m.profile_worlds(V2_PROFILE_DEVELOPMENT)[0].seed
    for drift in (
        lambda s, c: {"route_count": 99} if (s == target and c == SEVERITY_SEVERE) else {},
        lambda s, c: {"allocation_fingerprint": "moved"} if s == target else {},
        lambda s, c: {"fd_selected_ordinal": 2} if s == target else {},
    ):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _run_v2_round(_eval_cfg(), m, tmp,
                              body=_stub_body(m, calls=[], drift=drift))
            except (BenchmarkIdentityError, ValueError) as exc:
                assert isinstance(exc, BenchmarkIdentityError) or "route_count" in str(exc)
            else:
                raise AssertionError("a drifted V2 world was accepted")
            assert not _jsonl(Path(tmp) / "episode_failures.jsonl")


def test_po3_a_v1_manifest_is_never_evaluated_under_v2_and_vice_versa() -> None:
    import tempfile
    v1 = build_benchmark_manifest(worlds_per_cell=1, benchmark_base_seed=1_000_000)
    _raises(BenchmarkManifestError, gt.evaluate_benchmark, None, object(), _eval_cfg(),
            v1, iteration=None)
    v1_cfg = gt.TrainConfig(
        n_iterations=1, episode_design=EPISODE_DESIGN_GENERALIZED_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
        generalized_max_attempts_per_iteration=8)
    _raises(BenchmarkManifestError, gt.evaluate_benchmark, None, object(), v1_cfg,
            _manifest(), iteration=None)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_benchmark_manifest(v1, Path(tmp) / "v1.json")
        cfg = _eval_cfg(output_dir=Path(tmp) / "run")
        cfg.benchmark_manifest = str(path)
        _raises(BenchmarkManifestError, gt.train, cfg)
        assert not (Path(tmp) / "run").exists()


def test_po3_the_whole_manifest_is_held_out_against_max_training_attempts() -> None:
    m = _manifest(base_seed=10_000, confirmatory_base=50_000)
    first_confirmatory = m.profile_worlds(V2_PROFILE_CONFIRMATORY)[0].seed
    assert first_confirmatory == 50_000
    # A development-only run whose MAXIMUM attempt band reaches a CONFIRMATORY seed while
    # its successful-episode quota band does not.
    cfg = _eval_cfg(V2_PROFILE_DEVELOPMENT, base_seed=first_confirmatory - 20,
                    n_iterations=2, episodes_per_iteration=10,
                    generalized_max_attempts_per_iteration=12)
    assert cfg.max_training_attempts == 24 and cfg.total_episodes == 20
    assert all(w.seed >= cfg.base_seed + 24 or w.seed < cfg.base_seed
               for w in m.profile_worlds(V2_PROFILE_DEVELOPMENT))
    exc = _raises(ValueError, gt._require_benchmark_seeds_held_out, m, cfg)
    assert "NOT held out" in str(exc)
    # The successful-episode quota alone would NOT have reached it: the attempt band did.
    assert first_confirmatory >= cfg.base_seed + cfg.total_episodes
    clear = _eval_cfg(V2_PROFILE_DEVELOPMENT, base_seed=0)
    gt._require_benchmark_seeds_held_out(m, clear)
    bands = gt.seed_bands(clear, benchmark=m)
    ev = bands["benchmark_evaluation"]
    assert ev["held_out_checked_over"] == "entire_manifest_all_profiles"
    assert ev["n_world_seeds"] == 120 and ev["n_member_episodes_per_round"] == 60
    assert ev["evaluation_profile"]["profile"] == V2_PROFILE_DEVELOPMENT


def test_po3_early_stopping_is_still_refused_under_v2() -> None:
    exc = _raises(ValueError, _eval_cfg(early_stopping=True, n_iterations=400).validate)
    assert "training_reward_plateau_v1" in str(exc)


def _group(bc, key, complete=True, mild=None, severe=None):
    return {"group_key": key, "base_cell": bc, "complete": complete,
            "member_decisions": {SEVERITY_MILD: mild, SEVERITY_SEVERE: severe}}


def test_po3_the_primary_metric_is_paired_per_group_then_macro_averaged() -> None:
    fd = WAKE_KIND_IMMEDIATE_FD
    groups = []
    # A2-D0: THREE eligible groups, delta 0.1 each; every other cell ONE group, delta 0.5.
    for i in range(3):
        groups.append(_group("A2-D0", "A2-D0-w%03d" % i,
                             mild=[_wake(fd, 0.3, PLAN)], severe=[_wake(fd, 0.4, PLAN)]))
    for bc in V2_BENCHMARK_BASE_CELL_KEYS[1:]:
        groups.append(_group(bc, bc + "-w000",
                             mild=[_wake(fd, 0.1, PLAN)], severe=[_wake(fd, 0.6, ABORT)]))
    s = gt._v2_behaviour_summary(groups)
    assert s["n_groups_metric_eligible"] == 12
    assert abs(s["by_base_cell"]["A2-D0"]["severe_minus_mild_abort_mass_mean"] - 0.1) < 1e-12
    # EQUAL WEIGHT over the ten cells: (0.1 + 9 * 0.5) / 10, NOT the pooled group mean.
    assert abs(s["macro_mean_over_base_cells"] - 0.46) < 1e-12
    assert abs(s["pooled_mean_over_groups"] - (0.3 + 4.5) / 12) < 1e-12
    assert s["macro_n_base_cells_defined"] == 10 and s["macro_undefined_base_cells"] == []
    # Directional switch: mild != ABORT and severe == ABORT, over metric-eligible groups.
    assert s["directional_switch_count"] == 9 and s["reverse_switch_count"] == 0
    assert abs(s["directional_switch_rate"] - 9 / 12) < 1e-12
    assert s["by_base_cell"]["A2-D0"]["directional_switch_rate"] == 0.0
    assert s["by_base_cell"]["A3-D0"]["directional_switch_rate"] == 1.0
    assert s["switch_rates_over"] == "metric_eligible_groups"
    # A reverse switch is counted separately.
    rev = gt._v2_behaviour_summary([_group(
        "A2-D0", "k", mild=[_wake(fd, 0.9, ABORT)], severe=[_wake(fd, 0.2, PLAN)])])
    assert rev["reverse_switch_count"] == 1 and rev["directional_switch_count"] == 0
    # A cell with no eligible group makes the macro mean UNDEFINED, never re-weighted.
    assert rev["macro_mean_over_base_cells"] is None
    assert len(rev["macro_undefined_base_cells"]) == 9
    empty = gt._v2_behaviour_summary([])
    assert empty["pooled_mean_over_groups"] is None
    assert empty["directional_switch_rate"] is None


def test_po3_only_immediate_fd_wakes_of_complete_groups_reach_the_metric() -> None:
    fd = WAKE_KIND_IMMEDIATE_FD
    clean = gt._v2_behaviour_summary([_group(
        "A2-D0", "g", mild=[_wake(fd, 0.2, PLAN)], severe=[_wake(fd, 0.5, PLAN)])])
    noisy = gt._v2_behaviour_summary([_group(
        "A2-D0", "g",
        mild=[_wake(WAKE_KIND_ORDINARY, 1.0, ABORT), _wake(fd, 0.2, PLAN),
              _wake(WAKE_KIND_POST_FD_BOUNDARY, 1.0, ABORT)],
        severe=[_wake(WAKE_KIND_POST_FD_BOUNDARY, 0.0, PLAN), _wake(fd, 0.5, PLAN)])])
    for key in ("pooled_mean_over_groups", "directional_switch_count",
                "reverse_switch_count", "n_groups_metric_eligible"):
        assert clean[key] == noisy[key], key
    none = gt._v2_behaviour_summary([_group(
        "A2-D0", "g", mild=[_wake(WAKE_KIND_ORDINARY, 0.9, ABORT)],
        severe=[_wake(fd, 0.5, ABORT)])])
    assert none["n_groups_metric_eligible"] == 0
    assert none["not_measurable_reasons"] == {"mild_no_immediate_fd_wake": 1}
    many = gt._v2_behaviour_summary([_group(
        "A2-D0", "g", mild=[_wake(fd, 0.1, PLAN), _wake(fd, 0.2, PLAN)],
        severe=[_wake(fd, 0.5, ABORT)])])
    assert many["not_measurable_reasons"] == {"mild_multiple_immediate_fd_wakes": 1}
    incomplete = gt._v2_behaviour_summary([_group(
        "A2-D0", "g", complete=False, mild=[_wake(fd, 0.1, PLAN)],
        severe=[_wake(fd, 0.9, ABORT)])])
    assert incomplete["n_groups_incomplete"] == 1
    assert incomplete["pooled_mean_over_groups"] is None
    assert incomplete["not_measurable_reasons"] == {}


def test_po3_training_rows_cannot_contaminate_the_v2_summary_block() -> None:
    """The run-summary V2 block is read off EVALUATION round records only."""
    import tempfile
    m = _manifest()
    with tempfile.TemporaryDirectory() as tmp:
        record = _run_v2_round(_eval_cfg(), m, tmp, body=_stub_body(m, calls=[]))
        outcomes = _jsonl(Path(tmp) / "episode_outcomes.jsonl")
    assert outcomes and all(r["generalized"] for r in outcomes)
    # TRAINING rows cloned from real V2 outcome rows, with adversarial immediate-FD wakes.
    train_rows = []
    for r in outcomes[:6]:
        row = dict(r, phase="train", benchmark_group_key=None,
                   wake_decisions=[_wake(WAKE_KIND_IMMEDIATE_FD, 1.0, ABORT)])
        train_rows.append(row)
    block = gt._generalized_summary(outcomes + train_rows, [], [record])["v2_benchmark"]
    alone = gt._generalized_summary(outcomes, [], [record])["v2_benchmark"]
    assert block == alone
    assert block["final_round_behaviour"] == record["v2_behaviour"]
    assert block["totals_across_rounds_are_repeated_measures"] is True
    # No V2 round -> no V2 key: a V1 / fixed-cell summary keeps its shape.
    assert "v2_benchmark" not in gt._generalized_summary(outcomes, [], [])


# =============================================================================
# Standalone runner (nlp_env has no pytest)
# =============================================================================

def _run_all() -> None:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("[ OK ] %s" % name)
    print("%d passed" % len(tests))


if __name__ == "__main__":
    _run_all()
