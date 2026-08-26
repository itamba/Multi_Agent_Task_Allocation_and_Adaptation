"""GENERALIZED-V1 Task 4 -- the POPULATION layer: design selector, training cardinality
sampler, frozen 18-stratum benchmark manifest, and reference-fault routing.

PURE: no BLADE, no gymnasium, no torch, no solver CALL, no training run. Every test here
is a statement about a deterministic function or a serialized record, so the whole file
runs in milliseconds and can be driven by the base-env ``pytest`` AND by the standalone
``__main__`` runner under ``nlp_env``.

WHAT IT PROVES, mapped to the task's proof obligations:

  PO1  historical preservation + deterministic generalized sampling
       -- ``fixed_cell_v1`` is the default of BOTH harnesses and resolves the four
          historical policy ids; the sampler's support is exactly
          ``A ~ U{2,3,4}`` and ``H | A ~ U{1..A}`` with ``K == A``; it draws from an
          ISOLATED seed domain and perturbs no other stream.
  PO2  matched 18-stratum benchmark integrity
       -- exactly ``3 x 2 x 3`` strata; matched CLEAN/MILD/SEVERE members share one
          world-group identity; no identity rests on a uuid; a frozen manifest detects
          tampering, reordering and preflight mismatch instead of substituting silently.
  PO3  integrity routing
       -- an unacceptable reference solve is ORDINARY attrition; every other reference
          fault is a measurement-integrity abort; the routing reads a SLUG, never a
          message.

Run: python -m pytest tests/test_graph_generalized.py -v
     python tests/test_graph_generalized.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CONDITION_CLEAN,
    FD_ELIGIBILITY_CERTIFIED_V1,
    FD_ELIGIBILITY_LEGACY_V1,
    FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN,
    FUEL_DAMAGE_RNG_DOMAIN,
    FUEL_DAMAGE_SEVERITY_RNG_DOMAIN,
    POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
    POST_FD_WAKE_SINGLE_V1,
    SEVERITY_MILD,
    SEVERITY_SEVERE,
    FuelDamageMode,
    FuelDamageParameters,
    derive_fuel_damage_eligibility_seed,
    derive_fuel_damage_seed,
    derive_fuel_damage_severity_seed,
)
from match_aou.rl.training.graph_generalized import (  # noqa: E402
    BENCHMARK_BASE_CELLS,
    BENCHMARK_CELLS,
    BENCHMARK_DELTAS,
    BENCHMARK_GROUP_SIZE,
    BENCHMARK_MEMBERS,
    BENCHMARK_SCHEMA,
    BENCHMARK_SCHEMA_VERSION,
    BENCHMARK_STRATA,
    BENCHMARK_STRATUM_KEYS,
    CARDINALITY_RNG_DOMAIN,
    CARDINALITY_SAMPLER_POLICY,
    CARDINALITY_SOURCE_BENCHMARK,
    CARDINALITY_SOURCE_FIXED_CELL,
    CARDINALITY_SOURCE_SAMPLER,
    EPISODE_DESIGN_FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1,
    EPISODE_DESIGNS,
    FIXED_CELL_V1,
    GENERALIZED_AGENT_COUNTS,
    GENERALIZED_V1,
    LOAD_HIGH,
    LOAD_LOW,
    TARGET_DESTRUCTION_PROBABILITY,
    BenchmarkIdentityError,
    BenchmarkManifestError,
    EpisodeCardinality,
    WorldIdentity,
    WorldPreflight,
    build_benchmark_manifest,
    cardinality_sampler_record,
    certificate_fingerprint,
    derive_cardinality_seed,
    fixed_cell_cardinality,
    hidden_requested_for,
    load_benchmark_manifest,
    manifest_from_record,
    manifest_identity,
    manifest_seed_overlap,
    require_matched_group_identity,
    require_world_matches_manifest,
    resolve_episode_design,
    sample_generalized_cardinality,
    write_benchmark_manifest,
)
from match_aou.rl.training.graph_hidden_placement import (  # noqa: E402
    HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    HIDDEN_POLICY_EXACT_V1,
)
from match_aou.rl.training.graph_reward import (  # noqa: E402
    REFERENCE_ATTRITION_REASONS,
    REFERENCE_FAULT_ARITHMETIC,
    REFERENCE_FAULT_MISSING,
    REFERENCE_FAULT_NO_UNIVERSE,
    REFERENCE_FAULT_REASONS,
    REFERENCE_FAULT_SOLVE_UNACCEPTABLE,
    REFERENCE_FAULT_UNKNOWN_KIND,
    REFERENCE_POLICY_EVENT_CONDITIONED_V1,
    REFERENCE_POLICY_STATIC_T0_V1,
    ReferenceIntegrityError,
    reference_fault_aborts,
)

# A benchmark large enough to exercise ordering and per-cell structure, small enough to
# stay instant. NOTHING here is a scientific scale -- the manifest mechanism refuses to
# default one, and choosing the real one is a later task's job.
_TEST_WORLDS_PER_CELL = 2
_TEST_BASE_SEED = 7_000_000


def _manifest(**kwargs):
    kwargs.setdefault("worlds_per_cell", _TEST_WORLDS_PER_CELL)
    kwargs.setdefault("benchmark_base_seed", _TEST_BASE_SEED)
    return build_benchmark_manifest(**kwargs)


def _identity(hidden=2, known=3, fingerprint=((1.0, 2.0),), ordinal=1, cert="abc"):
    return WorldIdentity(
        hidden_realized=hidden, known_realized=known,
        geometric_fingerprint=tuple(fingerprint),
        fd_selected_ordinal=ordinal, fd_certificate_fingerprint=cert,
    )


# =============================================================================
# PO1 -- HISTORICAL PRESERVATION
# =============================================================================

def test_po1_fixed_cell_is_the_default_of_both_harnesses() -> None:
    """The historical design is what a caller that says NOTHING gets.

    This is the load-bearing preservation claim of the whole task: every approved
    measurement was taken on the fixed cell and its four historical policies, so a run
    that does not opt in must resolve exactly those and nothing else.
    """
    from match_aou.rl.training.graph_rollout import RolloutConfig
    from match_aou.rl.training.graph_train import TrainConfig

    for cfg in (TrainConfig(n_iterations=1), RolloutConfig()):
        assert cfg.episode_design == EPISODE_DESIGN_FIXED_CELL_V1
        assert cfg.generalized is False
        assert cfg.design is FIXED_CELL_V1


def test_po1_the_historical_bundle_is_the_four_layer_defaults() -> None:
    """`fixed_cell_v1` resolves the DEFAULT of every layer that owns a policy.

    Not "some historical-looking values": each id is compared against the constant the
    owning layer exports, so a layer that ever changed its own default would fail here
    rather than silently redefining what "historical" means.
    """
    assert FIXED_CELL_V1.hidden_policy == HIDDEN_POLICY_EXACT_V1
    assert FIXED_CELL_V1.eligibility_policy == FD_ELIGIBILITY_LEGACY_V1
    assert FIXED_CELL_V1.post_fd_wake_policy == POST_FD_WAKE_SINGLE_V1
    assert FIXED_CELL_V1.reference_policy == REFERENCE_POLICY_STATIC_T0_V1
    assert FIXED_CELL_V1.generalized is False

    # And they really ARE the defaults the layers would have chosen unaided.
    legacy_params = FuelDamageParameters(mode=FuelDamageMode.SEEDED_MIXTURE)
    assert legacy_params.eligibility_policy == FIXED_CELL_V1.eligibility_policy
    assert legacy_params.post_fd_wake_policy == FIXED_CELL_V1.post_fd_wake_policy


def test_po1_the_generalized_bundle_is_the_four_approved_policies() -> None:
    """`generalized_v1` is the WHOLE approved bundle -- never three of the four."""
    assert GENERALIZED_V1.hidden_policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1
    assert GENERALIZED_V1.eligibility_policy == FD_ELIGIBILITY_CERTIFIED_V1
    assert GENERALIZED_V1.post_fd_wake_policy == POST_FD_WAKE_COMPLETION_BOUNDARY_V1
    assert GENERALIZED_V1.reference_policy == REFERENCE_POLICY_EVENT_CONDITIONED_V1
    assert GENERALIZED_V1.generalized is True
    # Every one of the four DIFFERS from the historical bundle: a "generalized" design
    # that silently kept a historical policy would be a design nobody approved.
    for field in ("hidden_policy", "eligibility_policy", "post_fd_wake_policy",
                  "reference_policy"):
        assert getattr(GENERALIZED_V1, field) != getattr(FIXED_CELL_V1, field), field


def test_po1_an_unknown_design_raises_and_never_falls_back() -> None:
    """A mislabelled measurement is worse than a crash, so resolution is strict."""
    for bad in ("generalized", "GENERALIZED_V1", "fixed_cell", "", None, 1):
        try:
            resolve_episode_design(bad)
        except ValueError:
            continue
        raise AssertionError("resolve_episode_design(%r) must raise" % (bad,))
    assert set(EPISODE_DESIGNS) == {
        EPISODE_DESIGN_FIXED_CELL_V1, EPISODE_DESIGN_GENERALIZED_V1
    }


def test_po1_p_destroy_is_unchanged_and_mirrors_the_pipeline_default() -> None:
    """Target destruction stays DETERMINISTIC, and the recorded value is not a guess.

    The constant is a MIRROR of `scenario_factory`'s own default, so a future change to
    the pipeline's probability would fail here instead of leaving `run_config.json`
    quietly asserting 1.0 about a world that is no longer deterministic.
    """
    import inspect

    from match_aou.utils.blade_utils import scenario_factory

    assert TARGET_DESTRUCTION_PROBABILITY == 1.0
    for fn in (scenario_factory.make_attack_task,
               scenario_factory.generate_all_enemy_tasks):
        default = inspect.signature(fn).parameters["probability"].default
        assert default == TARGET_DESTRUCTION_PROBABILITY, fn.__name__


# =============================================================================
# PO1 -- THE DETERMINISTIC CARDINALITY SAMPLER
# =============================================================================

def test_po1_sampler_support_is_exactly_the_approved_cell() -> None:
    """`A in {2,3,4}`, `K == A`, `1 <= H <= A` -- and every one of the 9 cells occurs."""
    seen = set()
    for seed in range(6000):
        card = sample_generalized_cardinality(episode_seed=seed)
        assert card.agent_count in GENERALIZED_AGENT_COUNTS
        assert card.known_count == card.agent_count, "K must equal A"
        assert 1 <= card.hidden_requested <= card.agent_count
        assert card.targets_requested == card.known_count + card.hidden_requested
        assert card.source == CARDINALITY_SOURCE_SAMPLER
        seen.add((card.agent_count, card.hidden_requested))
    assert seen == {(a, h) for a in GENERALIZED_AGENT_COUNTS
                    for h in range(1, a + 1)}, sorted(seen)


def test_po1_sampler_is_uniform_in_A_and_in_H_given_A() -> None:
    """The DISTRIBUTION is the approved one, not merely the support.

    A sampler that produced the right values with the wrong weights would stratify the
    training population incorrectly while passing every support test, so the marginal of
    ``A`` and the conditional of ``H | A`` are both checked, with a band wide enough to
    be robust and tight enough to catch a real skew.
    """
    n = 30_000
    a_counts = {a: 0 for a in GENERALIZED_AGENT_COUNTS}
    h_counts = {a: {h: 0 for h in range(1, a + 1)}
                for a in GENERALIZED_AGENT_COUNTS}
    for seed in range(n):
        card = sample_generalized_cardinality(episode_seed=seed)
        a_counts[card.agent_count] += 1
        h_counts[card.agent_count][card.hidden_requested] += 1

    expected_a = 1.0 / len(GENERALIZED_AGENT_COUNTS)
    for a, count in a_counts.items():
        assert abs(count / n - expected_a) < 0.02, (a, count / n)
    for a, hist in h_counts.items():
        total = sum(hist.values())
        for h, count in hist.items():
            assert abs(count / total - 1.0 / a) < 0.03, (a, h, count / total)


def test_po1_sampler_is_reproducible_across_processes() -> None:
    """The draw is a pure function of the SEED -- not of a per-process hash salt.

    Run in a FRESH interpreter with a different `PYTHONHASHSEED`, because that is exactly
    the failure `hash()` would produce and no in-process test could ever catch: a
    sampler that reproduced perfectly within one run and differed between two.
    """
    local = [tuple(sample_generalized_cardinality(episode_seed=s).to_record().values())
             for s in range(50)]
    child = subprocess.run(
        [sys.executable, "-c",
         "import json,sys;"
         "sys.path.insert(0, r'%s');"
         "from match_aou.rl.training.graph_generalized import "
         "sample_generalized_cardinality as f;"
         "print(json.dumps([list(f(episode_seed=s).to_record().values()) "
         "for s in range(50)]))" % str(SRC)],
        capture_output=True, text=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "", "SYSTEMROOT":
             __import__("os").environ.get("SYSTEMROOT", "")},
    )
    assert child.returncode == 0, child.stderr
    remote = [tuple(row) for row in json.loads(child.stdout.strip().splitlines()[-1])]
    assert remote == local


def test_po1_sampler_perturbs_no_other_rng_stream() -> None:
    """RNG ISOLATION, from both sides.

    The sampler must neither consume global ``random`` / torch state (which would make an
    episode's cardinality depend on what ran before it) nor be movable by them (which
    would make it irreproducible). Both directions are checked, and torch is included
    because action sampling draws from its global generator.
    """
    import torch

    # (a) the sampler consumes NOTHING global.
    random.seed(4242)
    torch.manual_seed(4242)
    expected_random = random.random()
    expected_torch = torch.rand(1).item()

    random.seed(4242)
    torch.manual_seed(4242)
    for seed in range(200):
        sample_generalized_cardinality(episode_seed=seed)
    assert random.random() == expected_random, "global random state moved"
    assert torch.rand(1).item() == expected_torch, "torch global state moved"

    # (b) the sampler is not MOVED by them either.
    baseline = [sample_generalized_cardinality(episode_seed=s) for s in range(40)]
    for global_seed in (0, 1, 999):
        random.seed(global_seed)
        torch.manual_seed(global_seed)
        for _ in range(17):
            random.random()
        assert [sample_generalized_cardinality(episode_seed=s)
                for s in range(40)] == baseline


def test_po1_sampler_seed_domain_is_disjoint_from_the_three_fd_domains() -> None:
    """Its own SHA-256 domain, and provably not one of the fuel-damage ones.

    Sharing a domain would insert draws between the fuel-damage stream's mixture bit and
    its ego selection and CHANGE which ego every damaged episode picks -- invalidating the
    approved measurements rather than extending them.
    """
    domains = (FUEL_DAMAGE_RNG_DOMAIN, FUEL_DAMAGE_SEVERITY_RNG_DOMAIN,
               FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN)
    assert CARDINALITY_RNG_DOMAIN not in domains
    assert len(set(domains + (CARDINALITY_RNG_DOMAIN,))) == 4

    for seed in range(500):
        mine = derive_cardinality_seed(seed)
        assert mine != derive_fuel_damage_seed(seed)
        assert mine != derive_fuel_damage_severity_seed(seed)
        assert mine != derive_fuel_damage_eligibility_seed(seed)


def test_po1_sampler_draws_exactly_two_values_in_a_stated_order() -> None:
    """The draw sequence is a CONTRACT: A first, then H conditional on it.

    Re-derived here from a bare ``random.Random`` on the same derived seed, so an added,
    removed or reordered draw inside the sampler fails.
    """
    for seed in (0, 1, 7, 12345):
        rng = random.Random(derive_cardinality_seed(seed))
        counts = tuple(int(a) for a in GENERALIZED_AGENT_COUNTS)
        expected_a = counts[rng.randrange(len(counts))]
        expected_h = 1 + rng.randrange(expected_a)
        card = sample_generalized_cardinality(episode_seed=seed)
        assert (card.agent_count, card.hidden_requested) == (expected_a, expected_h)


def test_po1_the_requested_cardinality_is_never_rewritten() -> None:
    """A REQUEST is immutable. Short realization is recorded, never repaired.

    ``EpisodeCardinality`` is frozen, so nothing downstream can quietly lower
    ``hidden_requested`` to match a world that realized fewer -- which is the mutation
    that would make a requested-vs-realized denominator unreadable.
    """
    import dataclasses

    card = sample_generalized_cardinality(episode_seed=3)
    try:
        card.hidden_requested = 0  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("a scheduled cardinality must not be mutable")


def test_po1_fixed_cell_cardinality_is_verbatim_and_labelled() -> None:
    """The historical path takes the config VERBATIM -- no draw, no seed involved."""
    card = fixed_cell_cardinality(agent_count=3, known_count=3, hidden_requested=3)
    assert (card.agent_count, card.known_count, card.hidden_requested) == (3, 3, 3)
    assert card.source == CARDINALITY_SOURCE_FIXED_CELL
    assert card.targets_requested == 6


def test_po1_cardinality_source_is_a_closed_set() -> None:
    """Where a cardinality came from is a recorded FACT, and an unknown one raises."""
    try:
        EpisodeCardinality(agent_count=3, known_count=3, hidden_requested=1,
                           source="made-up")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown cardinality source must raise")


def test_po1_the_sampler_record_states_the_rule_not_just_a_name() -> None:
    """`run_config.json` must let a reader CHECK the distribution without this module."""
    record = cardinality_sampler_record()
    assert record["policy"] == CARDINALITY_SAMPLER_POLICY
    assert record["rng_domain"] == CARDINALITY_RNG_DOMAIN
    assert record["agent_counts"] == [int(a) for a in GENERALIZED_AGENT_COUNTS]
    assert record["known_rule"] == "K == A"
    assert "Uniform" in record["hidden_requested_rule"]
    # The two facts a reader most needs in order not to misread a short realization.
    assert record["realized_hidden_may_be_short"] is True
    assert record["retry_on_short_realization"] is False
    json.dumps(record)  # must be JSON-ready


# =============================================================================
# PO2 -- THE MATCHED 18-STRATUM BENCHMARK
# =============================================================================

def test_po2_there_are_exactly_eighteen_requested_strata() -> None:
    """`3 agent counts x 2 load buckets x 3 conditions = 18`, and they are unique."""
    assert len(BENCHMARK_STRATA) == 18
    assert len(set(BENCHMARK_STRATUM_KEYS)) == 18
    assert len(BENCHMARK_BASE_CELLS) == 6
    assert len(BENCHMARK_CELLS) == BENCHMARK_GROUP_SIZE == 3
    assert set(BENCHMARK_CELLS) == {CONDITION_CLEAN, SEVERITY_MILD, SEVERITY_SEVERE}
    assert {s.agent_count for s in BENCHMARK_STRATA} == set(GENERALIZED_AGENT_COUNTS)
    assert {s.load_bucket for s in BENCHMARK_STRATA} == {LOAD_LOW, LOAD_HIGH}


def test_po2_low_requests_one_hidden_and_high_requests_A() -> None:
    """The two REQUESTED load buckets, and they really differ for every team size."""
    for a in GENERALIZED_AGENT_COUNTS:
        assert hidden_requested_for(a, LOAD_LOW) == 1
        assert hidden_requested_for(a, LOAD_HIGH) == a
    # At A=2 the buckets are 1 and 2 -- still distinct, which is what makes A=2 a usable
    # stratum rather than one where LOW and HIGH coincide.
    assert hidden_requested_for(2, LOAD_LOW) != hidden_requested_for(2, LOAD_HIGH)
    try:
        hidden_requested_for(3, "medium")
    except BenchmarkManifestError:
        pass
    else:
        raise AssertionError("an unknown load bucket must raise")


def test_po2_the_benchmark_scale_is_never_defaulted() -> None:
    """The world count is a SCIENTIFIC decision this module refuses to make.

    Choosing it needs bounded runtime / solver validation that has not happened, so both
    the no-argument and the ambiguous both-arguments forms raise rather than silently
    picking a population.
    """
    for kwargs in (
        {},
        {"worlds_per_cell": 2, "worlds": [{"agent_count": 2,
                                           "load_bucket": LOAD_LOW, "seed": 1}]},
        {"worlds_per_cell": 2},          # a count with no seed base
        {"worlds_per_cell": 0, "benchmark_base_seed": 1},
        {"worlds_per_cell": -1, "benchmark_base_seed": 1},
    ):
        try:
            build_benchmark_manifest(**kwargs)
        except BenchmarkManifestError:
            continue
        raise AssertionError("build_benchmark_manifest(%r) must raise" % (kwargs,))


def test_po2_a_manifest_is_content_addressed_and_reproducible() -> None:
    """Same inputs -> same id. That is what makes "the same benchmark" checkable."""
    a = _manifest()
    b = _manifest()
    assert a.manifest_id == b.manifest_id
    assert a.canonical_json() == b.canonical_json()
    assert len(a.manifest_id) == 64  # sha256 hex
    # A different population is a different identity.
    assert _manifest(benchmark_base_seed=_TEST_BASE_SEED + 1).manifest_id != a.manifest_id
    assert _manifest(worlds_per_cell=3).manifest_id != a.manifest_id
    assert _manifest(label="arm-A").manifest_id != a.manifest_id


def test_po2_a_manifest_round_trips_through_a_file(tmp_path: Path) -> None:
    """Written canonically, it hashes to its own id when read back."""
    manifest = _manifest(label="round-trip")
    path = write_benchmark_manifest(manifest, tmp_path / "bench.json")
    loaded = load_benchmark_manifest(path)
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.seeds() == manifest.seeds()
    assert loaded.label == "round-trip"
    assert loaded.n_worlds == 6 * _TEST_WORLDS_PER_CELL
    assert loaded.n_members == loaded.n_worlds * BENCHMARK_GROUP_SIZE
    assert loaded.schema == BENCHMARK_SCHEMA
    assert loaded.schema_version == BENCHMARK_SCHEMA_VERSION


def test_po2_a_tampered_manifest_is_refused_not_rehashed(tmp_path: Path) -> None:
    """Editing a frozen population after the fact is REFUSED, in every field.

    The point is that a manifest cannot be quietly re-pointed at different worlds while
    still claiming the identity two arms compared themselves by.
    """
    manifest = _manifest()
    for mutate in (
        lambda r: r["worlds"][0].__setitem__("seed", 424242),
        lambda r: r["worlds"][2].__setitem__("agent_count", 4),
        lambda r: r["worlds"].pop(),
        lambda r: r.__setitem__("label", "something-else"),
        lambda r: r.__setitem__("manifest_id", "0" * 64),
    ):
        record = json.loads(json.dumps(manifest.to_record()))
        mutate(record)
        try:
            manifest_from_record(record)
        except BenchmarkManifestError:
            continue
        raise AssertionError("a tampered manifest was accepted")


#: Every CANONICAL field of a stored manifest payload, with a value that differs from
#: the one this schema produces. `manifest_id` is excluded (it is the hash, not payload)
#: and `notes` is excluded only because the fixture leaves it `None` and the mutation
#: below sets a string -- it is covered by the `label` case, which has the same shape.
_CANONICAL_FIELD_MUTATIONS = (
    ("group_size", 99),
    ("group_cells", ["clean", "mild"]),
    ("group_modes", ["forced_clean"]),
    ("n_strata", 3),
    ("strata", []),
    ("n_worlds", 1),
    ("n_members", 1),
    ("schema_version", 2),
    ("label", "a-different-population"),
    ("worlds", []),
)


def test_fix1_every_canonical_field_is_covered_by_the_manifest_id() -> None:
    """REVIEW FIX 1. The id authenticates the EXACT STORED payload, not a reconstruction.

    THE DEFECT THIS CLOSES. The loader used to rebuild the canonical fields from CURRENT
    CONSTANTS and hash THAT, so every field it rebuilt -- `group_size`, `group_cells`,
    `group_modes`, `n_strata`, `strata`, `n_worlds`, `n_members` -- could be edited in a
    stored file and still pass, because the edit was discarded before the hash was taken.
    A frozen, content-addressed manifest has to authenticate the bytes a reviewer can
    actually read, or "the two arms ran the same benchmark" is not a checkable claim.

    Each mutation here is a single stored field changed and NOTHING else -- in particular
    the id is left alone, so what is proven is that the stored id no longer matches the
    stored content.
    """
    manifest = _manifest()
    record = manifest.to_record()

    for field, value in _CANONICAL_FIELD_MUTATIONS:
        assert field in record, "%r is not a canonical field any more" % field
        tampered = json.loads(json.dumps(record))
        assert tampered[field] != value, field
        tampered[field] = value
        try:
            manifest_from_record(tampered)
        except BenchmarkManifestError as exc:
            # `schema_version` is caught EARLIER, by the dedicated version check, which
            # is the more informative refusal for a file this code cannot read at all.
            # Every other field must fail on the identity itself.
            if field != "schema_version":
                assert "MISMATCH" in str(exc), (field, str(exc))
            continue
        raise AssertionError(
            "a manifest with %r edited to %r was ACCEPTED" % (field, value)
        )


def test_fix1_an_injected_or_missing_canonical_field_is_refused() -> None:
    """REVIEW FIX 1. Unknown and missing canonical content is REFUSED, never ignored."""
    record = _manifest().to_record()

    injected = json.loads(json.dumps(record))
    injected["injected_field"] = {"anything": 1}
    for mutate, what in (
        (lambda r: r.__setitem__("injected_field", {"anything": 1}), "extra field"),
        (lambda r: r.pop("n_strata"), "missing n_strata"),
        (lambda r: r.pop("group_modes"), "missing group_modes"),
        (lambda r: r.pop("worlds"), "missing worlds"),
    ):
        tampered = json.loads(json.dumps(record))
        mutate(tampered)
        try:
            manifest_from_record(tampered)
        except BenchmarkManifestError:
            continue
        raise AssertionError("a manifest with %s was ACCEPTED" % what)


def test_fix1_a_self_consistently_rehashed_forgery_is_still_refused() -> None:
    """REVIEW FIX 1. The hash check ALONE is not enough, and both layers are proven.

    A forger who edits a canonical field AND re-hashes produces a document that is
    internally consistent -- it passes the identity check by construction. It is still
    not the payload this schema defines, and it is refused by the SECOND check: the
    stored payload must EQUAL the canonical payload the parsed manifest produces.

    The two checks are independent and neither implies the other, which is why both
    exist: this test would pass with only the equality check, and
    :func:`test_fix1_every_canonical_field_is_covered_by_the_manifest_id` would pass with
    only the hash check.
    """
    record = _manifest().to_record()

    # The DERIVED fields only -- the ones the old loader rebuilt from current constants
    # and therefore could not authenticate. `label` / `notes` are deliberately excluded:
    # they are free-form pass-throughs, so a re-hashed document with a different label is
    # a legitimately DIFFERENT population carrying a DIFFERENT id, not a forgery of this
    # one (and the no-re-hash case above already proves editing it in place is refused).
    # `worlds` is excluded because an empty world list is refused for its own, equally
    # correct reason (no base cell is populated), which would not exercise this check.
    derived = tuple(
        (f, v) for f, v in _CANONICAL_FIELD_MUTATIONS
        if f not in ("label", "notes", "worlds")
    )
    for field, value in derived + (("injected", "x"),):
        tampered = json.loads(json.dumps(record))
        tampered[field] = value
        # Re-hash so the document authenticates its OWN edited content.
        payload = {k: v for k, v in tampered.items() if k != "manifest_id"}
        tampered["manifest_id"] = manifest_identity(payload)
        try:
            manifest_from_record(tampered)
        except BenchmarkManifestError as exc:
            # `schema_version` is again caught by the earlier version check; everything
            # else must be refused by the CANONICAL-PAYLOAD comparison, since a re-hashed
            # document passes the identity check by construction.
            if field != "schema_version":
                assert "canonical payload" in str(exc), (field, str(exc))
            continue
        raise AssertionError(
            "a self-consistently re-hashed forgery of %r was ACCEPTED" % field
        )


def test_fix1_a_manifest_without_a_usable_id_is_refused() -> None:
    """REVIEW FIX 1. An unidentified population is not the frozen one."""
    record = _manifest().to_record()
    for mutate, what in (
        (lambda r: r.pop("manifest_id"), "no manifest_id"),
        (lambda r: r.__setitem__("manifest_id", ""), "empty manifest_id"),
        (lambda r: r.__setitem__("manifest_id", None), "null manifest_id"),
        (lambda r: r.__setitem__("manifest_id", 12345), "non-string manifest_id"),
    ):
        tampered = json.loads(json.dumps(record))
        mutate(tampered)
        try:
            manifest_from_record(tampered)
        except BenchmarkManifestError:
            continue
        raise AssertionError("a manifest with %s was ACCEPTED" % what)


def test_fix1_a_valid_manifest_round_trips_content_identically(tmp_path: Path) -> None:
    """REVIEW FIX 1. Tightening the loader did not break the honest path.

    The stored document, the reloaded manifest's canonical payload and a freshly built
    one all agree exactly -- so the stricter equality check is satisfied by the writer
    this module ships, not merely by a hand-built fixture.
    """
    manifest = _manifest(label="round-trip", notes="engineering fixture")
    path = write_benchmark_manifest(manifest, tmp_path / "bench.json")

    stored = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_benchmark_manifest(path)

    assert loaded.manifest_id == manifest.manifest_id
    # The STORED payload is exactly the canonical payload, key for key and value for
    # value -- which is the property the loader now enforces.
    stored_payload = {k: v for k, v in stored.items() if k != "manifest_id"}
    assert stored_payload == json.loads(json.dumps(manifest.payload()))
    assert stored_payload == json.loads(json.dumps(loaded.payload()))
    assert manifest_identity(stored_payload) == manifest.manifest_id
    # And re-writing the reloaded manifest reproduces the same bytes.
    again = write_benchmark_manifest(loaded, tmp_path / "again.json")
    assert again.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_po2_a_reordered_manifest_is_refused_not_resorted() -> None:
    """The stored ORDER is part of the identity, so it is never silently re-sorted."""
    record = json.loads(json.dumps(_manifest().to_record()))
    record["worlds"][0], record["worlds"][-1] = record["worlds"][-1], record["worlds"][0]
    try:
        manifest_from_record(record)
    except BenchmarkManifestError as exc:
        assert "canonical order" in str(exc) or "identity MISMATCH" in str(exc)
    else:
        raise AssertionError("a reordered manifest was accepted")


def test_po2_a_manifest_missing_a_base_cell_is_refused() -> None:
    """Missing one base cell means three empty strata -- not this benchmark."""
    worlds = [{"agent_count": a, "load_bucket": bucket, "seed": 100 + i}
              for i, (a, bucket) in enumerate(BENCHMARK_BASE_CELLS)]
    build_benchmark_manifest(worlds=worlds)          # complete: accepted
    try:
        build_benchmark_manifest(worlds=worlds[:-1])  # one base cell short
    except BenchmarkManifestError as exc:
        assert "strata" in str(exc)
    else:
        raise AssertionError("an incomplete benchmark was accepted")


def test_po2_a_duplicate_seed_is_refused() -> None:
    """A seed IS a world's generator identity, so two worlds may not share one."""
    worlds = [{"agent_count": a, "load_bucket": bucket, "seed": 500}
              for (a, bucket) in BENCHMARK_BASE_CELLS]
    try:
        build_benchmark_manifest(worlds=worlds)
    except BenchmarkManifestError as exc:
        assert "twice" in str(exc)
    else:
        raise AssertionError("duplicate benchmark seeds were accepted")


def test_po2_no_identity_in_the_manifest_is_a_uuid() -> None:
    """Generated uuids are not seed-derived, so NO identity may rest on one.

    Checked structurally over the whole serialized document rather than field by field,
    so a future field carrying a uuid fails here too.
    """
    import re

    text = _manifest().canonical_json()
    uuid_like = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    assert not uuid_like.search(text), "a benchmark manifest must contain no uuid"
    # What it identifies worlds by instead: an ordinal group key and a seed.
    for world in _manifest().worlds:
        assert world.key.startswith("A%d-%s-w" % (world.agent_count, world.load_bucket))
        assert isinstance(world.seed, int)


def test_po2_a_world_group_is_three_members_on_one_world() -> None:
    """CLEAN / MILD / SEVERE share the seed, the cardinality and the group key.

    This is the matched comparison unit: one world, one factor varied. The members differ
    ONLY in the forced fuel-damage mode.
    """
    manifest = _manifest()
    for world in manifest.worlds:
        members = world.members()
        assert members == BENCHMARK_MEMBERS
        assert [cell for cell, _m in members] == list(BENCHMARK_CELLS)
        modes = [mode for _c, mode in members]
        assert modes == [FuelDamageMode.FORCED_CLEAN, FuelDamageMode.FORCED_MILD,
                         FuelDamageMode.FORCED_SEVERE]
        assert len(set(modes)) == 3, "the three members must differ in the event alone"
        card = world.cardinality()
        assert card.source == CARDINALITY_SOURCE_BENCHMARK
        assert card.known_count == world.agent_count
        assert card.hidden_requested == hidden_requested_for(
            world.agent_count, world.load_bucket)
        # One cardinality object for the whole group -- every member is built from it.
        assert all(world.cardinality() == card for _c, _m in members)
        assert {world.stratum_key(cell) for cell, _m in members} <= set(
            BENCHMARK_STRATUM_KEYS)


def test_po2_every_stratum_is_covered_by_the_generated_manifest() -> None:
    """A generated manifest really does populate all 18 requested strata."""
    manifest = _manifest()
    covered = {
        world.stratum_key(cell)
        for world in manifest.worlds for cell, _m in world.members()
    }
    assert covered == set(BENCHMARK_STRATUM_KEYS)
    assert manifest.worlds_per_base_cell() == {
        "A%d-%s" % cell: _TEST_WORLDS_PER_CELL for cell in BENCHMARK_BASE_CELLS
    }


def test_po2_the_three_within_group_deltas_are_the_approved_ones() -> None:
    """mild-clean, severe-clean, severe-mild -- and nothing pooled across worlds."""
    assert BENCHMARK_DELTAS == (
        (SEVERITY_MILD, CONDITION_CLEAN),
        (SEVERITY_SEVERE, CONDITION_CLEAN),
        (SEVERITY_SEVERE, SEVERITY_MILD),
    )


def test_fix2_seed_overlap_names_the_offending_worlds() -> None:
    """REVIEW FIX 2. The held-out test for a manifest is about ITS OWN seeds.

    The legacy check compares two CONFIGURED intervals, which is the right test exactly
    when the evaluation schedule IS that band -- and simply not this run's schedule when
    the seeds come from a frozen manifest. This helper answers the question that actually
    applies, and returns the offending seeds rather than a bool so a refusal can name
    which worlds collide.
    """
    manifest = build_benchmark_manifest(worlds=[
        {"agent_count": a, "load_bucket": bucket, "seed": 100 + i}
        for i, (a, bucket) in enumerate(BENCHMARK_BASE_CELLS)
    ])
    assert manifest.seeds() == (100, 101, 102, 103, 104, 105)

    # Half-open, exactly like every other band in this project: `stop` is EXCLUSIVE.
    assert manifest_seed_overlap(manifest, start=0, stop=100) == ()
    assert manifest_seed_overlap(manifest, start=106, stop=200) == ()
    assert manifest_seed_overlap(manifest, start=100, stop=101) == (100,)
    assert manifest_seed_overlap(manifest, start=102, stop=105) == (102, 103, 104)
    assert manifest_seed_overlap(manifest, start=0, stop=1000) == manifest.seeds()
    # An empty band excludes nothing.
    assert manifest_seed_overlap(manifest, start=50, stop=50) == ()


def test_fix2_the_seed_digest_is_ordered_and_stable() -> None:
    """REVIEW FIX 2. Provenance can re-check the executed seeds without the manifest.

    ORDERED on purpose: two manifests holding the same seeds in a different order are
    different populations, because the order is part of a manifest's identity.
    """
    a = _manifest()
    b = _manifest()
    assert a.seed_digest() == b.seed_digest()
    assert len(a.seed_digest()) == 64
    # A different population -> a different digest.
    assert _manifest(benchmark_base_seed=_TEST_BASE_SEED + 1).seed_digest() !=         a.seed_digest()
    # The digest really is over the ORDER, not the set.
    from match_aou.rl.training.graph_generalized import _content_hash
    forward = _content_hash({"seeds": [1, 2, 3]})
    backward = _content_hash({"seeds": [3, 2, 1]})
    assert forward != backward


def test_po2_a_preflighted_world_that_comes_out_different_is_refused() -> None:
    """A frozen world is VERIFIED, and a mismatch REFUSES rather than substituting."""
    observed = _identity()
    preflight = WorldPreflight(
        hidden_realized=observed.hidden_realized,
        known_realized=observed.known_realized,
        geometric_fingerprint=observed.geometric_fingerprint,
        fd_selected_ordinal=observed.fd_selected_ordinal,
        fd_certificate_fingerprint=observed.fd_certificate_fingerprint,
    )
    manifest = build_benchmark_manifest(worlds=[
        {"agent_count": a, "load_bucket": bucket, "seed": 900 + i,
         **({"preflight": preflight.to_record()} if i == 0 else {})}
        for i, (a, bucket) in enumerate(BENCHMARK_BASE_CELLS)
    ])
    preflighted = manifest.worlds[0]
    assert preflighted.preflight is not None

    require_world_matches_manifest(preflighted, observed)          # agrees: silent

    for wrong in (
        _identity(hidden=1),
        _identity(known=2),
        _identity(fingerprint=((9.9, 9.9),)),
        _identity(ordinal=2),
        _identity(cert="different"),
    ):
        try:
            require_world_matches_manifest(preflighted, wrong)
        except BenchmarkIdentityError as exc:
            assert "REFUSED" in str(exc)
            continue
        raise AssertionError("a world differing from its manifest was accepted")

    # A world with NO preflight has nothing to contradict, and is not invented.
    require_world_matches_manifest(manifest.worlds[-1], _identity(hidden=99))


def test_po2_matched_members_that_built_different_worlds_abort() -> None:
    """Two members that disagree about their world may never be differenced."""
    world = _manifest().worlds[0]
    same = _identity()

    require_matched_group_identity(world, {})                      # nothing to compare
    require_matched_group_identity(world, {CONDITION_CLEAN: same})  # one member only
    require_matched_group_identity(world, {
        CONDITION_CLEAN: same, SEVERITY_MILD: same, SEVERITY_SEVERE: same,
    })

    try:
        require_matched_group_identity(world, {
            CONDITION_CLEAN: same,
            SEVERITY_MILD: same,
            SEVERITY_SEVERE: _identity(ordinal=3),
        })
    except BenchmarkIdentityError as exc:
        assert "did not produce the same world" in str(exc)
        assert "severe" in str(exc)
    else:
        raise AssertionError("a mismatched matched group was accepted")


def test_po2_the_certificate_fingerprint_excludes_the_ego_uuid() -> None:
    """The certified event is compared by PHYSICS, never by the ego's generated id.

    Two runs of the same seed mint different ego uuids, so a fingerprint that included
    one would report the same certified event as two different ones.
    """
    base = {"ego_id": "aaaaaaaa-1111-2222-3333-444444444444", "event_tick": 137,
            "progress": 0.3, "fuel_before": 1000.0, "mild_target": 900.0}
    other_ego = dict(base, ego_id="bbbbbbbb-5555-6666-7777-888888888888")
    moved_event = dict(base, event_tick=138)

    assert certificate_fingerprint(base) == certificate_fingerprint(other_ego)
    assert certificate_fingerprint(base) != certificate_fingerprint(moved_event)
    assert certificate_fingerprint(None) is None, "no certificate -> no fingerprint"


# =============================================================================
# PO3 -- REFERENCE-FAULT ROUTING
# =============================================================================

def test_po3_reference_reasons_split_into_attrition_and_integrity() -> None:
    """EXACTLY ONE reason is ordinary attrition; every other one aborts."""
    assert set(REFERENCE_ATTRITION_REASONS) == {REFERENCE_FAULT_SOLVE_UNACCEPTABLE}
    assert set(REFERENCE_FAULT_REASONS) == {
        REFERENCE_FAULT_SOLVE_UNACCEPTABLE, REFERENCE_FAULT_MISSING,
        REFERENCE_FAULT_NO_UNIVERSE, REFERENCE_FAULT_UNKNOWN_KIND,
        REFERENCE_FAULT_ARITHMETIC,
    }
    for reason in REFERENCE_FAULT_REASONS:
        exc = ReferenceIntegrityError("x", reason=reason)
        aborts = reason not in REFERENCE_ATTRITION_REASONS
        assert exc.is_measurement_integrity is aborts, reason
        assert reference_fault_aborts(exc) is aborts, reason


def test_po3_an_unanswered_solve_is_ordinary_attrition() -> None:
    """The solver was ASKED and did not ANSWER: one attempt is spent, nothing aborts."""
    exc = ReferenceIntegrityError("solver stalled",
                                  reason=REFERENCE_FAULT_SOLVE_UNACCEPTABLE)
    assert reference_fault_aborts(exc) is False
    assert exc.is_measurement_integrity is False


def test_po3_an_instrument_contradiction_aborts() -> None:
    """A reference that contradicts itself implicates every episode the layer touched."""
    for reason in (REFERENCE_FAULT_MISSING, REFERENCE_FAULT_NO_UNIVERSE,
                   REFERENCE_FAULT_UNKNOWN_KIND, REFERENCE_FAULT_ARITHMETIC):
        assert reference_fault_aborts(
            ReferenceIntegrityError("x", reason=reason)) is True


def test_po3_the_reason_is_required_and_closed() -> None:
    """No unlabelled fault, and no invented label.

    A default reason would let a future raise site skip the classification by omission
    and fall into whichever routing happened to be the default -- the same class of
    defect that once let a roster fault read as ordinary episode attrition.
    """
    try:
        ReferenceIntegrityError("no reason")  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("reason must be a REQUIRED keyword")

    try:
        ReferenceIntegrityError("bad", reason="made-up")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown reason must be refused")


def test_po3_routing_reads_the_slug_and_never_the_message() -> None:
    """Two faults with IDENTICAL text route differently -- so nothing parses strings."""
    text = "the reference could not be established"
    attrition = ReferenceIntegrityError(text,
                                        reason=REFERENCE_FAULT_SOLVE_UNACCEPTABLE)
    integrity = ReferenceIntegrityError(text, reason=REFERENCE_FAULT_MISSING)
    assert str(attrition) == str(integrity)
    assert reference_fault_aborts(attrition) is not reference_fault_aborts(integrity)
    # An exception this layer does not own is not its to classify.
    assert reference_fault_aborts(ValueError(text)) is False
    assert reference_fault_aborts(RuntimeError(text)) is False


def test_po3_the_raise_sites_carry_a_reason() -> None:
    """Every `ReferenceIntegrityError(` construction in the source names a reason.

    Read off the SOURCE because several of the sites need a live solver or a full episode
    to reach. A site that forgot the keyword would raise ``TypeError`` at runtime, but
    only on the path that reaches it -- which is exactly the path a test suite is least
    likely to cover.
    """
    for name in ("graph_reward.py", "graph_episode_setup.py"):
        text = (SRC / "match_aou" / "rl" / "training" / name).read_text(
            encoding="utf-8")
        constructions = text.count("raise ReferenceIntegrityError(")
        reasons = text.count("reason=REFERENCE_FAULT_")
        assert constructions > 0, name
        assert reasons == constructions, (
            "%s: %d raise site(s) but %d reason(s)" % (name, constructions, reasons)
        )


# =============================================================================
# Import purity -- the population layer stays pure
# =============================================================================

def test_the_agent_count_mirror_matches_the_construction_constraint() -> None:
    """The mirrored `A in {2,3,4}` IS the cell `setup_episode` enforces.

    ``graph_generalized`` mirrors the constant instead of importing it, because importing
    the setup layer would drag torch into a module whose value is that it needs no engine
    (proved by the purity test below). A mirror is only safe while it is CHECKED, so the
    two are compared here against the authority that enforces the constraint.

    This test deliberately lives on the impure side of the boundary -- it imports the
    setup layer -- so the pure module never has to.
    """
    from match_aou.rl.training.graph_episode_setup import (
        GENERALIZED_AGENT_COUNTS as CONSTRUCTION_COUNTS,
    )

    assert tuple(GENERALIZED_AGENT_COUNTS) == tuple(CONSTRUCTION_COUNTS), (
        "the sampler could draw a team size the construction path refuses"
    )
    # And the sampler really is bounded by it.
    drawn = {sample_generalized_cardinality(episode_seed=s).agent_count
             for s in range(2000)}
    assert drawn <= set(CONSTRUCTION_COUNTS)


def test_the_population_layer_imports_no_blade_gym_or_torch() -> None:
    """A fresh import of ``graph_generalized`` pulls in no engine and no torch.

    Same criterion the placement layer holds itself to: this module is deterministic
    arithmetic over seeds and records, and it must stay hand-testable without an engine.
    pyomo is inherited from the ROOT package (``match_aou/__init__`` imports the solver
    eagerly), so it is exempted with a CONTROL that would fail if that ever stopped being
    true and the exemption became a real leak.
    """
    child = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "import match_aou;"
         "root = 'pyomo' in sys.modules;"
         "import match_aou.rl.training.graph_generalized;"
         "banned = [m for m in ('blade', 'gymnasium', 'gym', 'torch') "
         "if m in sys.modules];"
         "print(root, banned)" % str(SRC)],
        capture_output=True, text=True,
    )
    assert child.returncode == 0, child.stderr
    root_has_pyomo, banned = child.stdout.strip().split(" ", 1)
    assert root_has_pyomo == "True", (
        "the root package no longer imports pyomo eagerly -- the exemption below is now "
        "hiding a real dependency and must be re-examined"
    )
    assert banned == "[]", "graph_generalized leaked %s" % banned


def test_the_population_layer_never_imports_a_harness() -> None:
    """The import direction is one-way: the harnesses import THIS, never the reverse."""
    text = (SRC / "match_aou" / "rl" / "training" / "graph_generalized.py").read_text(
        encoding="utf-8")
    for forbidden in ("graph_train", "graph_rollout"):
        assert "import %s" % forbidden not in text, forbidden
        assert "from .%s" % forbidden not in text, forbidden


if __name__ == "__main__":
    import inspect as _inspect

    _tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    _tmp = Path(__import__("tempfile").mkdtemp(prefix="gen_v1_"))
    _failures = 0
    for _name, _fn in _tests:
        try:
            if "tmp_path" in _inspect.signature(_fn).parameters:
                _sub = _tmp / _name
                _sub.mkdir(parents=True, exist_ok=True)
                _fn(_sub)
            else:
                _fn()
            print("OK   %s" % _name)
        except Exception as _exc:  # noqa: BLE001 -- a standalone runner reports all
            _failures += 1
            print("FAIL %s: %s: %s" % (_name, type(_exc).__name__, _exc))
    print("-" * 70)
    print("%d passed, %d failed (of %d)"
          % (len(_tests) - _failures, _failures, len(_tests)))
    sys.exit(1 if _failures else 0)
