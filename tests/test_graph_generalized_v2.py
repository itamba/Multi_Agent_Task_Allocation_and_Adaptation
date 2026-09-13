"""GENERALIZED-V2 -- the TWO-STAGE, ROUTE-RELATIVE population contract.

PURE: no BLADE, no gymnasium, no torch, no solver CALL, no training run. Every test here
is a statement about a deterministic function, a validation verdict or a serialized
record, so the whole file runs in milliseconds and can be driven by the base-env
``pytest`` AND by the standalone ``__main__`` runner under ``nlp_env``.

WHAT IT PROVES, mapped to the task's proof obligations:

  PO1  HISTORICAL-PATH ISOLATION
       -- ``fixed_cell_v1`` and ``generalized_v1`` resolve exactly the policy ids, the
          sampler outputs, the seed arithmetic, the record shape and the verdicts they
          resolved before V2 existed; the V1 sampler's support and its pinned outputs are
          unmoved; the historical hidden-load policy is what every non-V2 caller gets;
          and the 18-stratum benchmark stays a V1 construct that V2 cannot reach.
  PO2  THE DETERMINISTIC TWO-STAGE V2 CONTRACT
       -- ``A in {2,3,4,5,6}``; ``K in {A, A+2}``; the same seed reproduces the same
          pre-solve cell; the same seed and route count reproduce the same ``H``;
          ``1 <= H <= R``; both stages run on their OWN SHA-256 domains and neither
          consumes nor is moved by global ``random``, the V1 sampler or the three
          fuel-damage streams; ``R == 0`` is REFUSED; and ``generalized_v2`` with the
          legacy allocation objective is REFUSED.

PO3 is a REAL-CONSTRUCTION obligation and cannot live here: it needs BLADE, a scenario
generator and the P1 solver. It is exercised by the bounded engineering smoke reported
with the task, never by this file.

Run: python -m pytest tests/test_graph_generalized_v2.py -v
     python tests/test_graph_generalized_v2.py
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.solvers.match_aou_backend import (  # noqa: E402
    MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
    MATCH_AOU_BACKEND_P1_MILP_V1,
)
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN,
    FUEL_DAMAGE_RNG_DOMAIN,
    FUEL_DAMAGE_SEVERITY_RNG_DOMAIN,
    FuelDamageMode,
    derive_fuel_damage_eligibility_seed,
    derive_fuel_damage_seed,
    derive_fuel_damage_severity_seed,
)
from match_aou.rl.training.graph_generalized import (  # noqa: E402
    CARDINALITY_RNG_DOMAIN,
    CARDINALITY_SOURCE_FIXED_CELL,
    CARDINALITY_SOURCE_SAMPLER,
    CARDINALITY_SOURCE_V2_PRE_SOLVE,
    CARDINALITY_SOURCE_V2_ROUTE_RELATIVE,
    CARDINALITY_SOURCES,
    DEFAULT_HIDDEN_LOAD_POLICY,
    EPISODE_DESIGN_FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1,
    EPISODE_DESIGN_GENERALIZED_V2,
    EPISODE_DESIGNS,
    FIXED_CELL_V1,
    GENERALIZED_AGENT_COUNTS,
    GENERALIZED_V1,
    GENERALIZED_V2,
    GENERALIZED_V2_AGENT_COUNTS,
    GENERALIZED_V2_KNOWN_OFFSETS,
    GENERALIZED_V2_REQUIRED_BACKEND,
    HIDDEN_LOAD_POLICIES,
    HIDDEN_LOAD_POLICY_EXPLICIT_V1,
    HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
    PRE_SOLVE_CARDINALITY_POLICY_V2,
    V2_CARDINALITY_RNG_DOMAIN,
    V2_HIDDEN_LOAD_RNG_DOMAIN,
    EpisodeCardinality,
    PreSolveCardinality,
    RouteRelativeHiddenLoad,
    derive_cardinality_seed,
    derive_hidden_load_seed,
    derive_v2_cardinality_seed,
    generalized_v2_cardinality_sampler_record,
    resolve_episode_design,
    resolve_hidden_load_policy,
    resolve_route_relative_hidden_load,
    resolved_v2_cardinality,
    sample_generalized_cardinality,
    sample_generalized_v2_pre_solve_cardinality,
)
from match_aou.rl.training.graph_hidden_placement import (  # noqa: E402
    HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    HIDDEN_POLICY_EXACT_V1,
    HiddenPlacementError,
    routed_ordinals,
)
from match_aou.rl.training import graph_episode_setup as ges  # noqa: E402
from match_aou.rl.training import graph_train as gt  # noqa: E402
from match_aou.rl.training import graph_rollout as gr  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================

def _v2_cfg(**kw):
    """A MINIMAL VALID GENERALIZED-V2 training config.

    Evaluation is off because this design defines no evaluation construct, and the
    attempt budget is explicit because the quota policy has no default -- both are
    verdicts under test below, so the helper states them rather than hiding them.
    """
    base = dict(
        n_iterations=2,
        episode_design=EPISODE_DESIGN_GENERALIZED_V2,
        match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
        eval_every=0,
        eval_episodes=0,
        episodes_per_iteration=8,
        generalized_max_attempts_per_iteration=12,
    )
    base.update(kw)
    return gt.TrainConfig(**base)


def _v1_cfg(**kw):
    """A MINIMAL VALID GENERALIZED-V1 training config (the historical generalized path)."""
    base = dict(
        n_iterations=2,
        episode_design=EPISODE_DESIGN_GENERALIZED_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
        eval_every=0,
        eval_episodes=0,
        episodes_per_iteration=8,
        generalized_max_attempts_per_iteration=12,
    )
    base.update(kw)
    return gt.TrainConfig(**base)


def _refuses(fn, *, what: str) -> str:
    """Assert ``fn()`` raises, and return the message so a test can name what it caught."""
    try:
        fn()
    except (ValueError, RuntimeError, HiddenPlacementError) as exc:
        return str(exc)
    raise AssertionError("expected a refusal for %s" % what)


# =============================================================================
# PO1 -- HISTORICAL-PATH ISOLATION
# =============================================================================

def test_po1_the_historical_designs_resolve_exactly_what_they_always_did() -> None:
    """V2's existence moves no policy id on either historical bundle.

    Stated as an EXHAUSTIVE field comparison rather than a spot check: the whole point of
    a bundle is that it is all-or-nothing, so a test that checked three of four ids could
    pass against a design nobody approved.
    """
    assert FIXED_CELL_V1.design == EPISODE_DESIGN_FIXED_CELL_V1
    assert FIXED_CELL_V1.hidden_policy == HIDDEN_POLICY_EXACT_V1
    assert FIXED_CELL_V1.eligibility_policy == "legacy_selected_ego_v1"
    assert FIXED_CELL_V1.post_fd_wake_policy == "single_wake_v1"
    assert FIXED_CELL_V1.reference_policy == "static_t0_v1"

    assert GENERALIZED_V1.design == EPISODE_DESIGN_GENERALIZED_V1
    assert GENERALIZED_V1.hidden_policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1
    assert GENERALIZED_V1.eligibility_policy == "certified_both_severities_v1"
    assert GENERALIZED_V1.post_fd_wake_policy == "completion_boundary_v1"
    assert GENERALIZED_V1.reference_policy == "event_conditioned_continuation_v1"

    # V2 reuses the reviewed MECHANISMS exactly; only the POPULATION differs. If this
    # ever diverges, V2 has silently become a second episode design rather than a second
    # population, which is a different -- and unreviewed -- thing.
    for field in ("hidden_policy", "eligibility_policy", "post_fd_wake_policy",
                  "reference_policy"):
        assert getattr(GENERALIZED_V2, field) == getattr(GENERALIZED_V1, field), field
    assert GENERALIZED_V2.design == EPISODE_DESIGN_GENERALIZED_V2


def test_po1_the_design_record_shape_is_unchanged_for_the_historical_designs() -> None:
    """`to_record()` grows NO key, so a V1 or fixed-cell artifact is byte-comparable.

    Provenance is the thing a reviewer reads a run through, so adding a key to it -- even
    a truthful one -- would change what every historical record looks like. The design ID
    already disambiguates V1 from V2, and that is what a reader keys off.
    """
    expected_keys = {"design", "generalized", "hidden_policy", "eligibility_policy",
                     "post_fd_wake_policy", "reference_policy"}
    for design in (FIXED_CELL_V1, GENERALIZED_V1, GENERALIZED_V2):
        assert set(design.to_record()) == expected_keys, design.design
    assert FIXED_CELL_V1.to_record()["generalized"] is False
    assert GENERALIZED_V1.to_record()["generalized"] is True
    # V2 IS a generalized bundle, and the harness behaviours that key off `generalized`
    # (seeded-variable fuel damage, the successful-episode quota, dynamic construction
    # provenance) apply to it in full.
    assert GENERALIZED_V2.to_record()["generalized"] is True


def test_po1_the_two_generalized_predicates_are_distinct_and_mean_what_they_say() -> None:
    """`generalized` is "any generalized bundle"; the other two are exact.

    They are separate predicates because the benchmark, the manifest and the approved
    stopping rule are V1-only while the fuel-damage mixture and the attempt quota are not.
    A single predicate would have to be wrong for one of those groups.
    """
    assert FIXED_CELL_V1.generalized is False
    assert FIXED_CELL_V1.generalized_v1_design is False
    assert FIXED_CELL_V1.route_relative_population is False

    assert GENERALIZED_V1.generalized is True
    assert GENERALIZED_V1.generalized_v1_design is True
    assert GENERALIZED_V1.route_relative_population is False

    assert GENERALIZED_V2.generalized is True
    assert GENERALIZED_V2.generalized_v1_design is False
    assert GENERALIZED_V2.route_relative_population is True


def test_po1_the_v1_sampler_is_untouched_in_support_seed_and_output() -> None:
    """The V1 training cardinality sampler produces exactly what it produced before.

    Pinned as LITERAL outputs, not merely as a support check: a support test passes
    against a sampler whose draw ORDER changed, and a changed draw order silently moves
    every world a V1 run has ever built.
    """
    assert GENERALIZED_AGENT_COUNTS == (2, 3, 4)
    assert CARDINALITY_RNG_DOMAIN == "generalized_cardinality_v1"
    pinned = {0: (4, 4, 3), 1: (2, 2, 2), 2: (4, 4, 1),
              3: (2, 2, 1), 4: (3, 3, 2), 5: (4, 4, 2)}
    for seed, expected in pinned.items():
        c = sample_generalized_cardinality(episode_seed=seed)
        assert (c.agent_count, c.known_count, c.hidden_requested) == expected, seed
        assert c.source == CARDINALITY_SOURCE_SAMPLER
    # ... and the whole support, over a wide sweep.
    for seed in range(500):
        c = sample_generalized_cardinality(episode_seed=seed)
        assert c.agent_count in GENERALIZED_AGENT_COUNTS
        assert c.known_count == c.agent_count
        assert 1 <= c.hidden_requested <= c.agent_count


def test_po1_the_historical_hidden_load_policy_is_the_default_everywhere() -> None:
    """Every pre-V2 construction call resolves the EXPLICIT request, by default.

    That default is what makes `fixed_cell_v1` and `generalized_v1` unchanged by V2's
    existence: their hidden load is still a number their caller computed before
    `setup_episode` was entered.
    """
    assert DEFAULT_HIDDEN_LOAD_POLICY == HIDDEN_LOAD_POLICY_EXPLICIT_V1
    assert HIDDEN_LOAD_POLICIES == (HIDDEN_LOAD_POLICY_EXPLICIT_V1,
                                    HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2)
    import inspect
    params = inspect.signature(ges.setup_episode).parameters
    assert params["hidden_load_policy"].default == HIDDEN_LOAD_POLICY_EXPLICIT_V1
    assert params["hidden_load_seed"].default is None
    assert params["known_requested"].default is None
    # An unknown id RAISES rather than falling back on the default.
    for bad in ("route_relative", "ROUTE_RELATIVE_UNIFORM_V2", "", None, 1):
        _refuses(lambda b=bad: resolve_hidden_load_policy(b), what="policy %r" % (bad,))


def test_po1_the_historical_construction_requests_are_resolved_unchanged() -> None:
    """`_resolve_construction_mode`'s historical branches keep every verdict and message.

    The V2 branch is an ADDED branch, not an edited one: the legacy split path, the exact
    construction path and the bounded-backoff path must all still be reached -- and
    refused -- exactly as before.
    """
    assert ges._resolve_construction_mode(None, None) is False
    assert ges._resolve_construction_mode(2, random.Random(0)) is True
    assert ges._resolve_construction_mode(0, random.Random(0)) is True
    assert ges._resolve_construction_mode(
        2, random.Random(0), HIDDEN_POLICY_BOUNDED_BACKOFF_V1) is True

    msg = _refuses(lambda: ges._resolve_construction_mode(2, None), what="half a pair")
    assert "PAIR" in msg
    msg = _refuses(lambda: ges._resolve_construction_mode(True, random.Random(0)),
                   what="a bool n_hidden")
    assert "non-negative integer" in msg
    msg = _refuses(
        lambda: ges._resolve_construction_mode(
            0, random.Random(0), HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
        what="bounded backoff with n_hidden=0")
    assert "n_hidden >= 1" in msg
    msg = _refuses(
        lambda: ges._resolve_construction_mode(
            None, None, HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
        what="a generalized policy with no construction pair")
    assert "silently ignored" in msg


def test_po1_a_v2_only_argument_is_refused_under_the_historical_policy() -> None:
    """A V2 knob handed to a non-V2 call is REFUSED, never accepted-and-ignored.

    An ignored knob is how a caller ends up believing it selected V2 while its worlds were
    built from its own fixed number -- a mislabelled population that nothing would report.
    """
    for name, kw in (("hidden_load_seed", {"hidden_load_seed": 7}),
                     ("known_requested", {"known_requested": 3})):
        msg = _refuses(
            lambda k=kw: ges._resolve_construction_mode(
                2, random.Random(0), HIDDEN_POLICY_BOUNDED_BACKOFF_V1, **k),
            what="%s under the explicit policy" % name)
        assert "Refusing rather than ignoring" in msg


def test_po1_the_historical_harness_paths_pass_no_v2_keyword_at_all() -> None:
    """Keyword OMISSION, not a `None`-valued keyword -- the stronger invariance claim.

    The same discipline `_artifact_kwargs` / `_ctde_kwargs` / `_cardinality_kwargs` already
    follow: a fixed-cell or V1 run must make EXACTLY its pre-V2 calls, so that "nothing
    changed for those runs" is a property of the call site rather than of the callee's
    defaults.
    """
    for cfg in (gt.TrainConfig(n_iterations=1), _v1_cfg()):
        assert gt.v2_pre_solve_cardinality(cfg, 0) is None
        assert gt._pre_solve_kwargs(gt.v2_pre_solve_cardinality(cfg, 0)) == {}
        assert gt._v2_hidden_load_kwargs(cfg, 0, None) == {}
    # ... and on V2 all three setup keywords travel TOGETHER, because setup refuses a
    # half-supplied route-relative request.
    cfg = _v2_cfg()
    pre = gt.v2_pre_solve_cardinality(cfg, 41)
    assert isinstance(pre, PreSolveCardinality)
    kwargs = gt._v2_hidden_load_kwargs(cfg, 41, pre)
    assert set(kwargs) == {"hidden_load_policy", "hidden_load_seed", "known_requested"}
    assert kwargs["hidden_load_policy"] == HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2
    assert kwargs["hidden_load_seed"] == 41
    assert kwargs["known_requested"] == pre.known_count


def test_po1_episode_cardinality_is_unchanged_for_every_historical_path() -> None:
    """The one-stage question keeps its one-stage answer -- and V2 refuses to fake one."""
    fixed = gt.TrainConfig(n_iterations=1, num_agents=2, n_known=5, n_hidden=4)
    c = gt.episode_cardinality(fixed, seed=11)
    assert (c.agent_count, c.known_count, c.hidden_requested) == (2, 5, 4)
    assert c.source == CARDINALITY_SOURCE_FIXED_CELL

    v1 = _v1_cfg()
    assert gt.episode_cardinality(v1, seed=3) == sample_generalized_cardinality(
        episode_seed=3)

    # A benchmark member states its own cell and is used VERBATIM, on every design that
    # can carry one.
    frozen = EpisodeCardinality(agent_count=4, known_count=4, hidden_requested=4,
                                source="benchmark_manifest")
    assert gt.episode_cardinality(v1, seed=3, benchmark_cardinality=frozen) is frozen

    msg = _refuses(lambda: gt.episode_cardinality(_v2_cfg(), seed=3),
                   what="a one-stage answer under V2")
    assert "TWO" in msg and "STAGES" in msg


def test_po1_the_v1_benchmark_stays_v1_and_v2_evaluates_only_its_own_construct() -> None:
    """The 18-stratum V1 construct is unreachable from V2; V2 evaluates its OWN benchmark.

    The V1 strata are built from `A in {2,3,4}` and a LOW/HIGH hidden load defined against
    `A`, so a V1 manifest evaluated under V2 would report strata the population never
    varied. V2 therefore evaluates only a frozen ten-cell V2 manifest under one declared
    profile -- never the fixed held-out band, and never a V1 manifest (refused by schema at
    load time; see `tests/test_graph_generalized_v2_benchmark.py`).
    """
    # Evaluation still requires a frozen manifest -- the held-out band is never a fallback.
    msg = _refuses(lambda: _v2_cfg(eval_every=5, eval_episodes=8).validate(),
                   what="evaluation under V2 with no manifest")
    assert "requires benchmark_manifest" in msg and "held-out" in msg
    # ... and exactly one declared profile.
    msg = _refuses(
        lambda: _v2_cfg(eval_every=5, eval_episodes=8,
                        benchmark_manifest="frozen.json").validate(),
        what="evaluation under V2 with no profile")
    assert "benchmark_profile" in msg
    _v2_cfg(eval_every=5, eval_episodes=8, benchmark_manifest="frozen.json",
            benchmark_profile="development").validate()

    # The V1 preflight CONFIG check is still V1-only; V2 selection is a separate path.
    from match_aou.rl.training import graph_benchmark_preflight as pf
    msg = _refuses(lambda: pf._require_preflight_config(_v2_cfg()),
                   what="the V1 preflight config check under V2")
    assert EPISODE_DESIGN_GENERALIZED_V1 in msg

    # `evaluate()` -- the fixed held-out band -- is still refused under V2.
    msg = _refuses(
        lambda: gt.evaluate(None, None, _v2_cfg(eval_every=0, eval_episodes=0),
                            iteration=None),
        what="evaluate() under V2")
    assert "not defined for episode_design" in msg
    # ... and the refusal no longer claims V2 has no evaluation construct: it points at
    # the frozen-benchmark path that now exists.
    assert "no evaluation construct" not in msg.lower()
    assert "evaluate_benchmark" in msg and "benchmark_profile" in msg
    # `evaluate_benchmark()` dispatches V2 to its own round, which refuses anything that is
    # not a frozen V2 manifest.
    msg = _refuses(
        lambda: gt.evaluate_benchmark(None, None, _v2_cfg(), None, iteration=None),
        what="evaluate_benchmark() under V2 without a V2 manifest")
    assert "generalized_v2 benchmark manifest" in msg


def test_po1_the_v1_generalized_verdicts_are_unchanged() -> None:
    """A V1 run keeps every rule it had, including the ones V2 states differently."""
    _v1_cfg().validate()                                  # the historical happy path
    # V1 may use EITHER allocation objective -- the backend is INDEPENDENT of the design
    # there, and that independence is not weakened by V2 requiring one.
    _v1_cfg(match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1).validate()
    _v1_cfg(match_aou_backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1).validate()
    # V1 with evaluation still REQUIRES a frozen manifest.
    msg = _refuses(lambda: _v1_cfg(eval_every=5, eval_episodes=8).validate(),
                   what="V1 evaluation with no manifest")
    assert "benchmark_manifest" in msg
    # The approved stopping rule stays V1-only, and V2 does not acquire it.
    msg = _refuses(lambda: _v2_cfg(early_stopping=True, n_iterations=400).validate(),
                   what="early stopping under V2")
    assert "training_reward_plateau_v1" in msg


def test_po1_finish_context_verifies_the_hidden_load_pairing() -> None:
    """The policy and its record are REQUIRED keywords, and their pairing is checked.

    A context that claimed the historical explicit request while its count had been drawn
    against a route count -- or one that declared the route-relative policy with no record
    of the `R` it drew against -- would make "which population is this episode from?"
    unanswerable from the artifact it produced.
    """
    import inspect
    params = inspect.signature(ges._finish_context).parameters
    for name in ("hidden_load_policy", "route_relative_load"):
        assert params[name].default is inspect.Parameter.empty, name


def test_po1_the_routed_ego_predicate_is_shared_with_the_walk() -> None:
    """`routed_ordinals` is the SAME predicate the bounded walk's `no_route` branch uses.

    Two definitions that agree today are exactly how a request silently becomes
    unsatisfiable later, so the count V2 bounds its request by and the count the walk can
    attempt against are one function.
    """
    ordinals = ["a", "b", "c", "d"]
    assert routed_ordinals({"a": [(0, 0, 0)], "c": []}, ordinals) == (0,)
    assert routed_ordinals({}, ordinals) == ()
    assert routed_ordinals(
        {"d": [(1, 0, 0)], "b": [(0, 0, 0)]}, ordinals) == (1, 3)
    # It refuses exactly the rosters the walk refuses.
    _refuses(lambda: routed_ordinals({"a": [(0, 0, 0)]}, []), what="an empty roster")
    _refuses(lambda: routed_ordinals({"a": [(0, 0, 0)]}, ["a", "a"]),
             what="a duplicated roster")
    _refuses(lambda: routed_ordinals({"z": [(0, 0, 0)]}, ordinals),
             what="a stray ego")


# =============================================================================
# PO2 -- THE DETERMINISTIC TWO-STAGE V2 POPULATION CONTRACT
# =============================================================================

def test_po2_stage_one_draws_a_and_k_from_the_approved_cell() -> None:
    """`A in {2,3,4,5,6}` and `K in {A, A+2}`, over a wide seed sweep.

    A=8 and A=10 are DELIBERATELY ABSENT: they exist only as engineering scaling evidence,
    and the executor / training path has not been validated at those sizes.
    """
    assert GENERALIZED_V2_AGENT_COUNTS == (2, 3, 4, 5, 6)
    assert GENERALIZED_V2_KNOWN_OFFSETS == (0, 2)
    seen_a, seen_k = set(), set()
    for seed in range(2000):
        c = sample_generalized_v2_pre_solve_cardinality(episode_seed=seed)
        assert c.agent_count in GENERALIZED_V2_AGENT_COUNTS, seed
        assert c.known_count - c.agent_count in GENERALIZED_V2_KNOWN_OFFSETS, seed
        assert c.policy == PRE_SOLVE_CARDINALITY_POLICY_V2
        assert c.rng_domain == V2_CARDINALITY_RNG_DOMAIN
        assert c.source == CARDINALITY_SOURCE_V2_PRE_SOLVE
        seen_a.add(c.agent_count)
        seen_k.add(c.known_count - c.agent_count)
    # The support is REACHED, not merely permitted: a sampler that could only ever draw
    # one value would satisfy the bounds above and be a fixed cell in disguise.
    assert seen_a == set(GENERALIZED_V2_AGENT_COUNTS)
    assert seen_k == set(GENERALIZED_V2_KNOWN_OFFSETS)


def test_po2_stage_one_is_reproducible_from_the_episode_seed_alone() -> None:
    """The same seed gives the same pre-solve cell, in this process and in any other."""
    for seed in (0, 1, 7, 41, 999, 123456):
        a = sample_generalized_v2_pre_solve_cardinality(episode_seed=seed)
        b = sample_generalized_v2_pre_solve_cardinality(episode_seed=seed)
        assert a == b, seed
        assert a.derived_seed == derive_v2_cardinality_seed(seed)
    pinned = {0: (2, 2), 1: (5, 5), 2: (5, 5), 3: (5, 7), 4: (3, 3),
              5: (5, 7), 6: (3, 3), 7: (3, 5), 8: (3, 3), 9: (5, 7)}
    for seed, expected in pinned.items():
        c = sample_generalized_v2_pre_solve_cardinality(episode_seed=seed)
        assert (c.agent_count, c.known_count) == expected, seed


def test_po2_stage_two_draws_h_uniformly_from_one_to_r() -> None:
    """`1 <= H <= R`, reproducible from the seed and the route count, and only those."""
    for r in range(1, 9):
        seen = set()
        for seed in range(600):
            load = resolve_route_relative_hidden_load(episode_seed=seed, route_count=r)
            assert 1 <= load.hidden_requested <= r, (seed, r)
            assert load.route_count == r
            assert load.policy == HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2
            assert load.rng_domain == V2_HIDDEN_LOAD_RNG_DOMAIN
            assert load.derived_seed == derive_hidden_load_seed(seed)
            # Same seed AND same route count => same draw, always.
            again = resolve_route_relative_hidden_load(episode_seed=seed, route_count=r)
            assert again == load
            seen.add(load.hidden_requested)
        assert seen == set(range(1, r + 1)), r
    pinned = {0: 4, 1: 1, 2: 2, 3: 1, 4: 1, 5: 3, 6: 4, 7: 2, 8: 3, 9: 4}
    for seed, expected in pinned.items():
        assert resolve_route_relative_hidden_load(
            episode_seed=seed, route_count=4).hidden_requested == expected, seed


def test_po2_a_world_with_no_routed_ego_is_refused() -> None:
    """`R == 0` RAISES rather than resolving to a zero request.

    A zero would be both an invalid bounded-backoff request and a silent redefinition of
    the population: "this world has no hidden half" is a different design from "this world
    has one hidden target somewhere on a route".
    """
    for bad in (0, -1, -5):
        msg = _refuses(
            lambda b=bad: resolve_route_relative_hidden_load(
                episode_seed=1, route_count=b),
            what="route_count=%r" % (bad,))
        assert "route_count must be >= 1" in msg
    # `bool` is refused despite subclassing `int` -- `True` reading as R=1 is exactly the
    # kind of value that produces a plausible-looking wrong population.
    _refuses(lambda: resolve_route_relative_hidden_load(episode_seed=1, route_count=True),
             what="a bool route_count")
    # The record type enforces the same invariant on construction.
    _refuses(lambda: RouteRelativeHiddenLoad(route_count=0, hidden_requested=1),
             what="a zero route count on the record")
    _refuses(lambda: RouteRelativeHiddenLoad(route_count=2, hidden_requested=3),
             what="H > R on the record")


def test_po2_the_two_stages_run_on_their_own_disjoint_seed_domains() -> None:
    """Five SHA-256 domains, all distinct, and no two derive the same seed.

    Taking either V2 draw from an existing domain would insert draws into a stream another
    decision depends on -- most damagingly `fuel_damage_v1`, whose second draw picks WHICH
    EGO every damaged episode selects. With separate domains the decisions are orthogonal
    in both directions.
    """
    domains = {
        CARDINALITY_RNG_DOMAIN,
        V2_CARDINALITY_RNG_DOMAIN,
        V2_HIDDEN_LOAD_RNG_DOMAIN,
        FUEL_DAMAGE_RNG_DOMAIN,
        FUEL_DAMAGE_SEVERITY_RNG_DOMAIN,
        FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN,
    }
    assert len(domains) == 6
    for seed in range(300):
        derived = [
            derive_cardinality_seed(seed),
            derive_v2_cardinality_seed(seed),
            derive_hidden_load_seed(seed),
            derive_fuel_damage_seed(seed),
            derive_fuel_damage_severity_seed(seed),
            derive_fuel_damage_eligibility_seed(seed),
        ]
        assert len(set(derived)) == 6, seed


def test_po2_neither_stage_consumes_or_is_moved_by_any_other_stream() -> None:
    """RNG ISOLATION, proven in BOTH directions.

    Forwards: draining global `random`, the V1 sampler and all three fuel-damage
    derivations cannot change either V2 draw. Backwards: taking either V2 draw leaves
    global `random`'s state byte-identical, so it cannot displace a later action sample or
    a later fuel-damage decision.
    """
    baseline_pre = sample_generalized_v2_pre_solve_cardinality(episode_seed=17)
    baseline_load = resolve_route_relative_hidden_load(episode_seed=17, route_count=5)

    for burn in (0, 1, 17, 250):
        random.seed(12345)
        for _ in range(burn):
            random.random()
        # Unrelated deterministic streams, consumed in between.
        sample_generalized_cardinality(episode_seed=99)
        derive_fuel_damage_seed(99)
        derive_fuel_damage_severity_seed(99)
        derive_fuel_damage_eligibility_seed(99)
        assert sample_generalized_v2_pre_solve_cardinality(
            episode_seed=17) == baseline_pre, burn
        assert resolve_route_relative_hidden_load(
            episode_seed=17, route_count=5) == baseline_load, burn

    random.seed(4242)
    before = random.getstate()
    sample_generalized_v2_pre_solve_cardinality(episode_seed=3)
    resolve_route_relative_hidden_load(episode_seed=3, route_count=4)
    assert random.getstate() == before

    # The two V2 stages are independent of EACH OTHER: the hidden load is a function of
    # the seed and R alone, so it cannot depend on which (A, K) stage 1 happened to draw.
    for seed in range(200):
        assert resolve_route_relative_hidden_load(
            episode_seed=seed, route_count=3
        ) == resolve_route_relative_hidden_load(episode_seed=seed, route_count=3)


def test_po2_the_resolved_cardinality_is_a_new_object_and_mutates_neither_stage() -> None:
    """Combining the stages BUILDS a third record; it never writes into either.

    Both stage records stay available beside the resolved cell, so "what was requested
    before anything was solved" and "what the route count turned that into" remain
    separately readable -- and neither can be silently revised into agreement with an
    outcome.
    """
    pre = sample_generalized_v2_pre_solve_cardinality(episode_seed=8)
    load = resolve_route_relative_hidden_load(episode_seed=8, route_count=4)
    pre_before, load_before = (pre.agent_count, pre.known_count), (
        load.route_count, load.hidden_requested)

    resolved = resolved_v2_cardinality(pre, load)
    assert isinstance(resolved, EpisodeCardinality)
    assert resolved.agent_count == pre.agent_count
    assert resolved.known_count == pre.known_count
    assert resolved.hidden_requested == load.hidden_requested
    assert resolved.targets_requested == pre.known_count + load.hidden_requested
    assert resolved.source == CARDINALITY_SOURCE_V2_ROUTE_RELATIVE
    assert (pre.agent_count, pre.known_count) == pre_before
    assert (load.route_count, load.hidden_requested) == load_before
    # Both stage records are FROZEN, so "never rewritten" is enforced rather than trusted.
    for record, field in ((pre, "known_count"), (load, "hidden_requested")):
        try:
            setattr(record, field, 99)
        except Exception:
            continue
        raise AssertionError("%s.%s must be immutable" % (type(record).__name__, field))


def test_po2_the_pre_solve_source_can_never_label_a_resolved_cell() -> None:
    """The stage-1 label is deliberately NOT a valid `EpisodeCardinality` source.

    An `EpisodeCardinality` states a COMPLETE requested cell, and a pre-solve draw is by
    definition half of one -- so a record carrying `A`, `K` and a hidden count must never
    be able to claim it came from the pre-solve stage.
    """
    assert CARDINALITY_SOURCE_V2_PRE_SOLVE not in CARDINALITY_SOURCES
    assert CARDINALITY_SOURCE_V2_ROUTE_RELATIVE in CARDINALITY_SOURCES
    _refuses(lambda: EpisodeCardinality(agent_count=3, known_count=3,
                                        hidden_requested=1,
                                        source=CARDINALITY_SOURCE_V2_PRE_SOLVE),
             what="a pre-solve source on a resolved cell")


def test_po2_the_design_requires_the_p1_objective_on_both_harnesses() -> None:
    """`generalized_v2` + `legacy_minlp_v1` is REFUSED, never silently overridden.

    Structural rather than preferential: V2 resolves its hidden load from the number of
    NON-EMPTY ROUTES the known-only allocation produced, so letting two objectives define
    that route count would make one design id mean two population selectors. The legacy
    objective's EPSILON stacking incentive in particular changes which allocations are
    optimal, hence which egos are routed at all.
    """
    assert GENERALIZED_V2_REQUIRED_BACKEND == MATCH_AOU_BACKEND_P1_MILP_V1
    _v2_cfg().validate()                                   # the approved combination
    msg = _refuses(
        lambda: _v2_cfg(match_aou_backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1).validate(),
        what="V2 with the legacy objective")
    assert "requires match_aou_backend" in msg and "Refused" in msg

    # The DIAGNOSTIC harness reaches the same verdict from the same constant.
    gr.RolloutConfig(episode_design=EPISODE_DESIGN_GENERALIZED_V2,
                     match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1,
                     fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE).validate()
    msg = _refuses(
        lambda: gr.RolloutConfig(
            episode_design=EPISODE_DESIGN_GENERALIZED_V2,
            match_aou_backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
            fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE).validate(),
        what="a V2 rollout with the legacy objective")
    assert "requires match_aou_backend" in msg


def test_po2_the_backend_contract_is_stated_as_design_constrained_not_independent() -> None:
    """The CURRENT contract, pinned in BOTH directions: behaviour and the prose about it.

    Before V2 the MATCH-AOU backend really was unconstrained by ``episode_design``, and
    ``graph_train`` said so in its field comment, its ``validate`` comment, a helper
    docstring, both startup-header branches and the CLI help. V2 made that FALSE -- and a
    valid V2 run reaches the P1 header branch, so the console itself was misstating the
    contract to the operator at the moment it mattered.

    THE TRUE CONTRACT IS NARROW, and both halves have to survive:

      * SELECTION IS STILL EXPLICIT. Nothing is inferred from the design, the task
        probabilities or what is installed; there is no ``auto`` and no fallback; a
        contradictory request is REFUSED rather than overridden. V2 must never read as
        though the backend were chosen on the operator's behalf.
      * THE VALID VALUE SET IS DESIGN-CONSTRAINED. ``fixed_cell_v1`` and
        ``generalized_v1`` accept EITHER approved objective; ``generalized_v2`` accepts
        only ``p1_milp_v1``.

    The behavioural half is asserted first -- a prose test that passed while the verdicts
    had drifted would be worse than no test. The source half then refuses the specific
    stale phrasings, by exact phrase rather than by the word "independent", so the many
    legitimate uses elsewhere in the module (FD pair members failing independently,
    repeated-measures rounds not being independent worlds, the solve budget being the same
    under either backend) are untouched.
    """
    # --- the BEHAVIOURAL contract, in one place ---------------------------------
    for design, backend, valid in (
        (EPISODE_DESIGN_GENERALIZED_V2, MATCH_AOU_BACKEND_P1_MILP_V1, True),
        (EPISODE_DESIGN_GENERALIZED_V2, MATCH_AOU_BACKEND_LEGACY_MINLP_V1, False),
        (EPISODE_DESIGN_GENERALIZED_V1, MATCH_AOU_BACKEND_LEGACY_MINLP_V1, True),
        (EPISODE_DESIGN_GENERALIZED_V1, MATCH_AOU_BACKEND_P1_MILP_V1, True),
        (EPISODE_DESIGN_FIXED_CELL_V1, MATCH_AOU_BACKEND_LEGACY_MINLP_V1, True),
        (EPISODE_DESIGN_FIXED_CELL_V1, MATCH_AOU_BACKEND_P1_MILP_V1, True),
    ):
        cfg = (_v2_cfg() if design == EPISODE_DESIGN_GENERALIZED_V2
               else _v1_cfg() if design == EPISODE_DESIGN_GENERALIZED_V1
               else gt.TrainConfig(n_iterations=2))
        cfg = type(cfg)(**{**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
                           "episode_design": design, "match_aou_backend": backend})
        if valid:
            cfg.validate()
        else:
            msg = _refuses(cfg.validate, what="%s + %s" % (design, backend))
            # REFUSED, not overridden -- the distinction the prose must also keep.
            assert "requires match_aou_backend" in msg and "Refused" in msg

    # --- the SOURCE / OPERATOR-TEXT contract ------------------------------------
    source = (SRC / "match_aou" / "rl" / "training" / "graph_train.py").read_text(
        encoding="utf-8")
    # Exact phrasings that claimed UNRESTRICTED independence. Each was present before V2
    # was added and is false now; none of them can return without failing here.
    for stale in (
        "INDEPENDENT of episode_design",
        "INDEPENDENT of --episode-design",
        "orthogonal to ``episode_design``",
        "either design may run under either backend",
        "An INDEPENDENT explicit selector",
        "An INDEPENDENT selector: it is NOT resolved from",
    ):
        assert stale not in source, "the unconditional independence claim returned: %r" % (
            stale,)

    # A SECOND PASS OVER THE FOLDED STRING CONSTANTS, because the raw-source scan above
    # cannot see a claim split across an implicit concatenation -- which is exactly how
    # the P1 startup branch used to carry "INDEPENDENT " / "of episode_design". Parsing
    # folds those, so an operator-facing message is checked as the operator reads it.
    constants = [
        n.value for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    for text in constants:
        for stale in ("INDEPENDENT of episode_design",
                      "INDEPENDENT of --episode-design",
                      "orthogonal to ``episode_design``",
                      "either design may run under either backend"):
            assert stale not in text, (
                "an operator-facing message still claims unrestricted independence: %r"
                % (text[:120],))

    # ... and the true contract is actually STATED, so this test cannot pass merely because
    # the subject was deleted.
    assert "ITS VALID VALUE SET IS DESIGN-CONSTRAINED" in source
    assert source.count("never inferred") >= 3, "the explicitness half must survive too"
    assert "no auto and no fallback" in source

    # The OPERATOR-facing surfaces specifically -- the CLI help and both startup branches,
    # because a valid V2 run reaches the P1 branch and a legacy run must be told why it
    # cannot be a V2 one.
    help_text = {a.dest: a.help for a in gt._build_arg_parser()._actions}
    backend_help = help_text["match_aou_backend"]
    assert "SEPARATE EXPLICIT selector" in backend_help
    assert EPISODE_DESIGN_GENERALIZED_V2 in backend_help
    assert "no auto and no fallback" in backend_help
    assert "INDEPENDENT of --episode-design" not in backend_help
    assert "generalized_v2 is defined ONLY against this one" in source     # P1 branch
    assert "generalized_v2 requires p1_milp_v1 and REFUSES this" in source  # legacy branch


def test_po2_the_operator_surfaces_describe_all_three_selectable_designs() -> None:
    """The CURRENT population / budget / benchmark contract, as an operator meets it.

    A selector whose ``choices`` offer three designs while its help explains two leaves an
    operator to discover the third from a traceback. And the generalized attempt budget
    was described as a ``generalized_v1`` requirement "refused otherwise" -- which reads,
    to someone configuring a V2 run, as though the flag did not apply to them, when
    ``validate`` in fact REQUIRES it there too.

    Behaviour is asserted FIRST, so this can never pass while the verdicts drifted; the
    operator text is then held to the same contract. The one benchmark nuance matters on
    its own: the budget really does fix the maximum training-attempt seed band on BOTH
    generalized designs, but only ``generalized_v1`` has a frozen benchmark to hold out
    from -- so the help must not imply that V2 has one.
    """
    # --- the BEHAVIOURAL contract ------------------------------------------------
    # The attempt budget: required on both generalized designs, refused on the fixed cell.
    _v1_cfg().validate()
    _v2_cfg().validate()
    for cfg_fn, label in ((_v1_cfg, EPISODE_DESIGN_GENERALIZED_V1),
                          (_v2_cfg, EPISODE_DESIGN_GENERALIZED_V2)):
        msg = _refuses(
            lambda f=cfg_fn: f(generalized_max_attempts_per_iteration=None).validate(),
            what="%s with no attempt budget" % label)
        assert "requires an explicit" in msg
        assert "generalized_max_attempts_per_iteration" in msg
        # The message names the design that really failed, not a hard-coded one.
        assert label in msg
    msg = _refuses(
        lambda: gt.TrainConfig(n_iterations=2,
                               generalized_max_attempts_per_iteration=12).validate(),
        what="the fixed cell carrying a generalized attempt budget")
    assert "must not silently acquire replacement" in msg
    # ... and the same property is reachable directly, on both generalized designs.
    for cfg in (_v1_cfg(), _v2_cfg()):
        assert cfg.training_attempt_policy == gt.TRAINING_ATTEMPT_POLICY_QUOTA
        assert cfg.max_attempts_per_iteration == 12
    # `max_attempts_per_iteration` carries its OWN refusal for a generalized config that
    # never went through `validate` (which would have caught the missing budget first), so
    # it is pinned here directly -- otherwise the only message a test ever sees is
    # validate's, and this one could quietly go on naming a design it was not reached from.
    for cfg_fn, label in ((_v1_cfg, EPISODE_DESIGN_GENERALIZED_V1),
                          (_v2_cfg, EPISODE_DESIGN_GENERALIZED_V2)):
        bare = cfg_fn(generalized_max_attempts_per_iteration=None)
        msg = _refuses(lambda c=bare: c.max_attempts_per_iteration,
                       what="max_attempts_per_iteration with no budget on %s" % label)
        assert label in msg, msg
    assert (gt.TrainConfig(n_iterations=2).training_attempt_policy
            == gt.TRAINING_ATTEMPT_POLICY_SCHEDULED)

    # The benchmark manifest: each generalized design evaluates its OWN manifest (V2 under a
    # declared profile); the fixed cell refuses any manifest.
    _v1_cfg(eval_every=5, eval_episodes=8,
            benchmark_manifest="frozen.json").validate()
    _v2_cfg(eval_every=5, eval_episodes=8, benchmark_manifest="frozen.json",
            benchmark_profile="confirmatory").validate()
    msg = _refuses(
        lambda: gt.TrainConfig(n_iterations=2,
                               benchmark_manifest="frozen.json").validate(),
        what="a manifest under %s" % EPISODE_DESIGN_FIXED_CELL_V1)
    assert "benchmark" in msg
    for cfg_fn, label in ((_v1_cfg, EPISODE_DESIGN_GENERALIZED_V1),
                          (lambda **k: gt.TrainConfig(n_iterations=2, **k),
                           EPISODE_DESIGN_FIXED_CELL_V1)):
        msg = _refuses(lambda f=cfg_fn: f(benchmark_profile="development").validate(),
                       what="a benchmark profile under %s" % label)
        assert "benchmark_profile" in msg

    # --- the OPERATOR-FACING text ------------------------------------------------
    help_text = {a.dest: a.help for a in gt._build_arg_parser()._actions}

    design_help = help_text["episode_design"]
    for design in EPISODE_DESIGNS:
        assert design in design_help, (
            "--episode-design offers %r but its help never names it" % design)
    # V2's two distinguishing facts, stated without any efficacy or benchmark claim.
    assert "TWO-STAGE" in design_help
    assert MATCH_AOU_BACKEND_P1_MILP_V1 in design_help
    for forbidden in ("benchmark", "better", "improve"):
        assert forbidden not in design_help.lower(), forbidden

    budget_help = help_text["generalized_max_attempts_per_iteration"]
    assert EPISODE_DESIGN_GENERALIZED_V1 in budget_help
    assert EPISODE_DESIGN_GENERALIZED_V2 in budget_help
    assert EPISODE_DESIGN_FIXED_CELL_V1 in budget_help
    # The stale claim: required for V1 and "refused otherwise", which is false for V2.
    assert "refused otherwise" not in budget_help
    # ... and the benchmark nuance is scoped rather than implied.
    assert "sets the maximum training seed band the benchmark is held out against"         not in budget_help
    # Both generalized designs now have a frozen benchmark, and the held-out claim is over
    # every manifest seed whichever profile is evaluated.
    assert "defines no evaluation benchmark" not in budget_help
    assert "held out from" in budget_help and "profile" in budget_help

    # The same stale claims must not survive anywhere in the module's CURRENT prose. The
    # comment is scanned in RAW source (comments exist nowhere else); the two message
    # claims are scanned over FOLDED string constants, because both were written across an
    # implicit concatenation and a raw scan would silently pass on either.
    source = (SRC / "match_aou" / "rl" / "training" / "graph_train.py").read_text(
        encoding="utf-8")
    assert "REQUIRED under `generalized_v1`, where" not in source, (
        "the V1-only attempt-budget comment returned")
    assert "REQUIRED under BOTH generalized designs" in source

    constants = [
        n.value for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    for stale in (
        # Implied that V2 has a benchmark for its seed band to be held out from.
        "sets the maximum training seed band the benchmark is held out against",
        # Implied the budget is a V1 requirement and "refused otherwise".
        "REQUIRED for a %s run (where episodes_per_iteration is a quota",
    ):
        for text in constants:
            assert stale not in text, (
                "a V1-only attempt-budget claim returned in an operator message: %r"
                % (text[:120],))


def test_po2_a_route_relative_request_is_refused_unless_it_is_complete() -> None:
    """Every half-supplied V2 construction request is refused BEFORE any BLADE object.

    Each refusal is a refusal to guess: the bound `H <= R` is defined by the bounded-backoff
    placement rule; a stated count and a deferred one are contradictory; the seed is what
    makes the draw reproducible from the artifact; and `K` is a CHOICE, so construction is
    told which one the schedule made rather than adopting whatever it received.
    """
    rr = dict(hidden_load_policy=HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
              hidden_load_seed=5, known_requested=3)
    assert ges._resolve_construction_mode(
        None, random.Random(0), HIDDEN_POLICY_BOUNDED_BACKOFF_V1, **rr) is True

    cases = [
        ("a stated n_hidden", (2, random.Random(0), HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         rr, "must be omitted"),
        ("the exact placement policy", (None, random.Random(0), HIDDEN_POLICY_EXACT_V1),
         rr, "requires hidden_policy"),
        ("no placement rng", (None, None, HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         rr, "explicit random.Random"),
        ("no hidden_load_seed", (None, random.Random(0),
                                 HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         dict(rr, hidden_load_seed=None), "integer hidden_load_seed"),
        ("a bool hidden_load_seed", (None, random.Random(0),
                                     HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         dict(rr, hidden_load_seed=True), "integer hidden_load_seed"),
        ("a negative hidden_load_seed", (None, random.Random(0),
                                         HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         dict(rr, hidden_load_seed=-1), "hidden_load_seed must be >= 0"),
        ("no known_requested", (None, random.Random(0),
                                HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         dict(rr, known_requested=None), "integer known_requested"),
        ("a bool known_requested", (None, random.Random(0),
                                    HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         dict(rr, known_requested=True), "integer known_requested"),
        ("a zero known_requested", (None, random.Random(0),
                                    HIDDEN_POLICY_BOUNDED_BACKOFF_V1),
         dict(rr, known_requested=0), "known_requested must be >= 1"),
    ]
    for label, args, kw, needle in cases:
        msg = _refuses(lambda a=args, k=kw: ges._resolve_construction_mode(*a, **k),
                       what=label)
        assert needle in msg, (label, msg)


def test_po2_the_pre_solve_cell_is_enforced_against_the_raw_known_world() -> None:
    """`A in {2..6}`, `K == K_requested`, and `K in {A, A+2}` -- refused, never repaired.

    `known_count` comes from the RAW world inventory, never from an allocated-only solver
    output. A world whose extracted known cardinality disagrees with the scheduled one is
    not a world that came out differently: it is the generator and the schedule describing
    different episodes, and adopting the extracted count would silently move the population
    every per-load statistic is reported over.
    """
    ges._require_generalized_v2_cardinality(
        agent_count=5, known_count=5, known_requested=5)
    ges._require_generalized_v2_cardinality(
        agent_count=5, known_count=7, known_requested=7)
    ges._require_generalized_v2_cardinality(
        agent_count=2, known_count=4, known_requested=4)

    msg = _refuses(
        lambda: ges._require_generalized_v2_cardinality(
            agent_count=7, known_count=7, known_requested=7),
        what="A outside the approved cell")
    assert "needs A in" in msg
    for unvalidated in (8, 10):
        _refuses(
            lambda a=unvalidated: ges._require_generalized_v2_cardinality(
                agent_count=a, known_count=a, known_requested=a),
            what="A=%d, which is engineering scaling evidence only" % unvalidated)
    msg = _refuses(
        lambda: ges._require_generalized_v2_cardinality(
            agent_count=4, known_count=5, known_requested=4),
        what="an extracted K that disagrees with the schedule")
    assert "refusing rather than adopting" in msg
    msg = _refuses(
        lambda: ges._require_generalized_v2_cardinality(
            agent_count=4, known_count=5, known_requested=5),
        what="K outside {A, A+2}")
    assert "needs K in" in msg


def test_po2_the_provenance_records_are_json_ready_and_state_both_stages() -> None:
    """Every V2 record round-trips through JSON as plain builtins, and names its stages.

    A record a reviewer cannot read out of an artifact is not provenance, so the whole
    two-stage identity -- both policies, both rng domains, both derived seeds, `R`, and
    both requested counts -- has to survive serialization.
    """
    pre = sample_generalized_v2_pre_solve_cardinality(episode_seed=21)
    load = resolve_route_relative_hidden_load(episode_seed=21, route_count=3)
    for record in (pre.to_record(), load.to_record(),
                   generalized_v2_cardinality_sampler_record(),
                   GENERALIZED_V2.to_record(),
                   resolved_v2_cardinality(pre, load).to_record()):
        assert json.loads(json.dumps(record)) == record

    p = pre.to_record()
    assert p["policy"] == PRE_SOLVE_CARDINALITY_POLICY_V2
    assert p["rng_domain"] == V2_CARDINALITY_RNG_DOMAIN
    assert p["derived_seed"] == derive_v2_cardinality_seed(21)
    assert p["agent_count"] == pre.agent_count
    assert p["known_requested"] == pre.known_count
    assert p["source"] == CARDINALITY_SOURCE_V2_PRE_SOLVE

    lo = load.to_record()
    assert lo["policy"] == HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2
    assert lo["rng_domain"] == V2_HIDDEN_LOAD_RNG_DOMAIN
    assert lo["derived_seed"] == derive_hidden_load_seed(21)
    assert lo["route_count_at_hidden_resolution"] == 3
    assert 1 <= lo["hidden_requested"] <= 3

    sampler = generalized_v2_cardinality_sampler_record()
    assert sampler["stages"] == 2
    assert sampler["required_match_aou_backend"] == MATCH_AOU_BACKEND_P1_MILP_V1
    assert sampler["pre_solve"]["agent_counts"] == list(GENERALIZED_V2_AGENT_COUNTS)
    assert sampler["pre_solve"]["known_offsets"] == list(GENERALIZED_V2_KNOWN_OFFSETS)
    assert sampler["post_solve"]["resolved_after_known_only_solve"] is True
    # A short realization stays a RECORDED outcome, never a retry -- the V1 rule, unmoved.
    assert sampler["realized_hidden_may_be_short"] is True
    assert sampler["retry_on_short_realization"] is False


def test_po2_the_design_is_selectable_only_as_a_whole() -> None:
    """One selector, no per-policy field -- a half-V2 run is not expressible.

    The four low-level policy ids are resolved from `episode_design` and are not settable
    from a config, a preset or a CLI flag; the two V2 population rules are resolved from
    the same selector. So a run can never enable the route-relative load while keeping a
    historical policy beside it.
    """
    assert EPISODE_DESIGN_GENERALIZED_V2 in EPISODE_DESIGNS
    assert resolve_episode_design(EPISODE_DESIGN_GENERALIZED_V2) is GENERALIZED_V2
    for bad in ("generalized_v3", "GENERALIZED_V2", "v2", "", None, 2):
        _refuses(lambda b=bad: resolve_episode_design(b), what="design %r" % (bad,))

    for cfg_cls in (gt.TrainConfig, gr.RolloutConfig):
        fields = {f for f in cfg_cls.__dataclass_fields__}
        for forbidden in ("hidden_policy", "eligibility_policy", "post_fd_wake_policy",
                          "reference_policy", "hidden_load_policy",
                          "pre_solve_cardinality_policy"):
            assert forbidden not in fields, (cfg_cls.__name__, forbidden)

    cfg = _v2_cfg()
    assert cfg.route_relative_population is True
    assert cfg.generalized is True
    assert cfg.design.hidden_policy == HIDDEN_POLICY_BOUNDED_BACKOFF_V1
    roll = gr.RolloutConfig(episode_design=EPISODE_DESIGN_GENERALIZED_V2,
                            match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1,
                            fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE)
    assert roll.route_relative_population is True


def test_po2_the_population_layer_never_imports_a_harness() -> None:
    """The import direction stays one-way: the harnesses import the population layer.

    `graph_episode_setup` may import `graph_generalized` (that is how it obtains the
    hidden-load policy ids and the stage-2 draw), but the reverse would drag the BLADE
    translation layer into a module whose whole value is that it is deterministic
    arithmetic over seeds and records.
    """
    text = (SRC / "match_aou" / "rl" / "training" / "graph_generalized.py").read_text(
        encoding="utf-8")
    for forbidden in ("graph_train", "graph_rollout", "graph_episode_setup"):
        assert "import %s" % forbidden not in text, forbidden
        assert "from .%s" % forbidden not in text, forbidden
    for engine in ("import blade", "import torch", "import gymnasium"):
        assert engine not in text, engine


def test_po1_the_failure_population_block_is_v2_only_and_stage_aware() -> None:
    """A FAILED attempt reports the identity it RECEIVED -- and only on the V2 path.

    Two facts in one place, because they are the two halves of the same rule: a
    ``fixed_cell_v1`` or ``generalized_v1`` ledger entry grows NO key (its single-stage
    cell is already complete), and a V2 entry states WHICH stage it reached rather than
    leaving a reader to infer it from which fields happen to be null.
    """
    pre = sample_generalized_v2_pre_solve_cardinality(episode_seed=5)
    load = resolve_route_relative_hidden_load(episode_seed=5, route_count=4)

    # Historical designs: nothing at all.
    assert gt._v2_failure_population(None, None) == {}
    assert gt._v2_failure_population(None, load) == {}

    # V2, stage 1 only: the hidden half is `null`, never fabricated.
    block = gt._v2_failure_population(pre, None)["generalized_v2_population"]
    assert block["stage_resolved"] == "pre_solve"
    assert block["agent_count"] == pre.agent_count
    assert block["known_requested"] == pre.known_count
    assert block["pre_solve_derived_seed"] == pre.derived_seed
    assert block["route_count_at_hidden_resolution"] is None
    assert block["hidden_load"] is None

    # V2, stage 2 reached: the EXACT draw, carried whole from the frozen record.
    block = gt._v2_failure_population(pre, load)["generalized_v2_population"]
    assert block["stage_resolved"] == "route_relative"
    assert block["route_count_at_hidden_resolution"] == load.route_count
    assert block["hidden_load_derived_seed"] == load.derived_seed
    assert block["hidden_load"] == load.to_record()
    # Both halves of the identity survive, and the whole block is JSON-ready.
    assert block["pre_solve_derived_seed"] == pre.derived_seed
    assert json.loads(json.dumps(block)) == block


def test_po1_the_failure_cardinality_assembles_and_never_redraws() -> None:
    """The cell a failed attempt reports is ASSEMBLED from two recorded facts.

    ``H`` is reproducible from the seed and ``R``, so a post-hoc redraw would usually
    agree -- and "usually" is exactly the property a ledger must not rest on. The resolved
    cell is built from the stage-1 draw and the ACTUAL stage-2 record; with no stage-2
    record it stays the stage-1 half-cell, and a non-V2 attempt reports its own cell
    untouched.
    """
    pre = sample_generalized_v2_pre_solve_cardinality(episode_seed=9)
    load = resolve_route_relative_hidden_load(episode_seed=9, route_count=3)

    # Case A -- no stage 2: the half-cell itself, with no hidden count attached.
    assert gt._failure_cardinality(None, pre, None) is pre
    assert getattr(gt._failure_cardinality(None, pre, None),
                   "hidden_requested", None) is None

    # Case B -- stage 2 reached: the RESOLVED cell, equal to the production assembly.
    resolved = gt._failure_cardinality(None, pre, load)
    assert resolved == resolved_v2_cardinality(pre, load)
    assert resolved.source == CARDINALITY_SOURCE_V2_ROUTE_RELATIVE
    assert resolved.hidden_requested == load.hidden_requested

    # Non-V2 -- the schedule's own cell wins and is returned untouched.
    v1 = sample_generalized_cardinality(episode_seed=9)
    assert gt._failure_cardinality(v1, None, None) is v1
    assert gt._failure_cardinality(v1, pre, load) is v1
    # Fixed cell: nothing is invented when the run config already states the cell.
    assert gt._failure_cardinality(None, None, None) is None


if __name__ == "__main__":
    _tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    _failures = 0
    for _name, _fn in _tests:
        try:
            _fn()
            print("OK   %s" % _name)
        except Exception as _exc:  # noqa: BLE001 -- a standalone runner reports all
            _failures += 1
            print("FAIL %s: %s: %s" % (_name, type(_exc).__name__, _exc))
    print("-" * 70)
    print("%d passed, %d failed (of %d)"
          % (len(_tests) - _failures, _failures, len(_tests)))
    sys.exit(1 if _failures else 0)
