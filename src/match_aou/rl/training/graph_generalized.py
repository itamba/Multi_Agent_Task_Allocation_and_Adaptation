"""GENERALIZED-V1 POPULATION: the episode-design selector, the deterministic training
cardinality sampler, and the FROZEN STRATIFIED BENCHMARK MANIFEST.

WHAT THIS MODULE OWNS, AND WHAT IT DELIBERATELY DOES NOT
========================================================
It owns three questions that are about the POPULATION an episode is drawn from, and
nothing about how an episode RUNS:

  1. **Which design is this run?** ``fixed_cell_v1`` (the DEFAULT and the historical
     behaviour) or ``generalized_v1`` (the complete approved GENERALIZED-V1 bundle of the
     already-reviewed Task-1/2/3 policy seams). ONE selector, resolved in ONE place, so a
     harness never has to coordinate four independent low-level policy knobs and can
     never resolve half a bundle.
  2. **What cardinality does a scheduled TRAINING episode have?** ``A ~ Uniform({2,3,4})``
     and ``H_requested | A ~ Uniform({1..A})`` with ``K == A``, drawn from an ISOLATED
     deterministic seed domain of this layer's own.
  3. **Which worlds does the frozen benchmark hold?** The 18-stratum matched
     CLEAN / MILD / SEVERE evaluation manifest, its canonical serialization, its content
     hash, and the identity checks that make a member refusable instead of silently
     substitutable.

It owns NONE of the mechanisms those choices select. Bounded-backoff placement geometry,
FD certification physics and the continuation-reference arithmetic are the LOCKED Task-1,
Task-2 and Task-3 contracts, and this module only names their policy ids.

PURITY
======
No BLADE, no gymnasium, no torch, no file I/O beyond reading/writing a manifest JSON that
a caller names explicitly, and NO module-global randomness: every draw runs on a
``random.Random`` this module constructs from a derived seed. It must never import
``graph_train`` or ``graph_rollout`` -- the harnesses import THIS.

RNG ISOLATION IS THE LOAD-BEARING PROPERTY
==========================================
The cardinality sampler has its OWN SHA-256 seed domain,
:data:`CARDINALITY_RNG_DOMAIN`, constructed exactly like the three fuel-damage domains
and disjoint from all of them. That separation is not tidiness:

  * taking the cardinality draw from ``fuel_damage_v1`` would insert draws between that
    stream's mixture bit and its ego selection and CHANGE WHICH EGO every damaged episode
    picks -- silently invalidating the approved FD measurements instead of extending them;
  * taking it from the episode's global ``random`` would make an episode's cardinality
    depend on how many global draws happened to run before it;
  * taking it from the hidden-placement rng would make it depend on how many placement
    candidates were rejected;
  * taking it from torch's generator would couple the world's SHAPE to the actor's
    action sampling.

With its own domain the decisions are orthogonal in both directions: the cardinality draw
cannot move the fuel-damage condition, severity, eligibility walk, hidden geometry or
action sampling, and none of them can move the cardinality.

NO UUID EQUALITY, ANYWHERE
==========================
Generated agent and target uuids are not seed-derived (``CLAUDE.md`` section 8), so no
identity in this module is ever a uuid: a world is identified by its ``(A, load bucket,
world ordinal)`` group key and its SEED, a candidate by its stable ORDINAL, and a realized
world by its id-free geometric fingerprint plus a certificate fingerprint taken over
scalars with the ego uuid REMOVED.

NOTHING HERE REACHES THE ACTING PATH
====================================
No design id, no cardinality, no stratum label, no load bucket and no manifest field
enters ``GraphObservation`` or the central critic's ``CentralGraphObservation``. A count of
what is hidden, and a label saying how hard the world is, are exactly the privileged
quantities an ego cannot sense (``CLAUDE.md`` section 3).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .graph_fuel_damage import (
    CONDITION_CLEAN,
    FD_ELIGIBILITY_CERTIFIED_V1,
    FD_ELIGIBILITY_LEGACY_V1,
    POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
    POST_FD_WAKE_SINGLE_V1,
    SEVERITY_MILD,
    SEVERITY_SEVERE,
    FuelDamageMode,
)
from .graph_hidden_placement import (
    HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    HIDDEN_POLICY_EXACT_V1,
)
from .graph_reward import (
    REFERENCE_POLICY_EVENT_CONDITIONED_V1,
    REFERENCE_POLICY_STATIC_T0_V1,
)

# =============================================================================
# 0. Constants that are MIRRORS of decisions taken elsewhere
# =============================================================================

# Target destruction is DETERMINISTIC and stays that way. `p(destroy) < 1` is a separate,
# still-deferred Grade-A research task; recording the value in `run_config.json` is how a
# generalized run states -- rather than implies -- that the redesign did not touch it.
#
# A MIRROR of `scenario_factory.make_attack_task`'s / `generate_all_enemy_tasks`'s default
# `probability` argument, kept here so this module needs no BLADE-adjacent import, and
# TEST-ENFORCED against those defaults (the same discipline
# `graph_fuel_damage.rtb_command_for` and `graph_train.derived_split` already use).
TARGET_DESTRUCTION_PROBABILITY: float = 1.0

# The approved generalized team sizes: `A in {2, 3, 4}`.
#
# A MIRROR of `graph_episode_setup.GENERALIZED_AGENT_COUNTS`, and the equivalence is
# TEST-ENFORCED against it -- the same discipline `graph_fuel_damage.rtb_command_for`
# and `graph_train.derived_split` already use, and for the same reason: importing the
# setup layer here would drag torch (through the BLADE translation layer) into a module
# whose whole value is that it is deterministic arithmetic over seeds and records, and
# can be exercised with no engine present.
#
# The mirrored value is the CONSTRUCTION CONSTRAINT, so the sampler below can never draw
# a team size construction would refuse. If the two ever diverge the mirror test fails
# rather than the sampler silently producing worlds that cannot be built.
GENERALIZED_AGENT_COUNTS: Tuple[int, ...] = (2, 3, 4)


# =============================================================================
# 1. THE EPISODE-DESIGN SELECTOR -- one knob, one resolution site
# =============================================================================

# `fixed_cell_v1` is the DEFAULT and is the HISTORICAL behaviour in full: the exact
# 3/3/3 construction cell, exact hidden cardinality, the legacy fuel-damage eligibility
# and single post-FD wake, and the static t=0 reward reference. Every approved
# measurement was taken on it (`CLAUDE.md` section 7), so it is preserved rather than
# reproduced.
EPISODE_DESIGN_FIXED_CELL_V1: str = "fixed_cell_v1"

# `generalized_v1` selects the COMPLETE approved GENERALIZED-V1 bundle in one word. It is
# deliberately a bundle rather than four independent knobs: the four policies were
# designed, reviewed and locked together, and a run that enabled three of them would be a
# design nobody approved while still recording itself as generalized.
EPISODE_DESIGN_GENERALIZED_V1: str = "generalized_v1"

EPISODE_DESIGNS: Tuple[str, ...] = (
    EPISODE_DESIGN_FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1,
)


@dataclass(frozen=True)
class EpisodeDesign:
    """The four LOW-LEVEL policy ids one high-level design resolves to.

    A frozen record rather than four loose strings so "which bundle did this run use?" is
    answered by one object that a record can serialize whole, and so a partially-resolved
    bundle is not expressible.
    """

    design: str
    hidden_policy: str
    eligibility_policy: str
    post_fd_wake_policy: str
    reference_policy: str

    @property
    def generalized(self) -> bool:
        """True iff this is the GENERALIZED-V1 bundle."""
        return self.design == EPISODE_DESIGN_GENERALIZED_V1

    def to_record(self) -> Dict[str, Any]:
        """A JSON-ready view (plain builtins only)."""
        return {
            "design": str(self.design),
            "generalized": bool(self.generalized),
            "hidden_policy": str(self.hidden_policy),
            "eligibility_policy": str(self.eligibility_policy),
            "post_fd_wake_policy": str(self.post_fd_wake_policy),
            "reference_policy": str(self.reference_policy),
        }


# The HISTORICAL bundle. Every id here is the DEFAULT of the layer that owns it, so a
# `fixed_cell_v1` run resolves exactly the values those layers would have chosen alone.
FIXED_CELL_V1: EpisodeDesign = EpisodeDesign(
    design=EPISODE_DESIGN_FIXED_CELL_V1,
    hidden_policy=HIDDEN_POLICY_EXACT_V1,
    eligibility_policy=FD_ELIGIBILITY_LEGACY_V1,
    post_fd_wake_policy=POST_FD_WAKE_SINGLE_V1,
    reference_policy=REFERENCE_POLICY_STATIC_T0_V1,
)

# The approved GENERALIZED-V1 bundle (handoff 3l.1-3l.5, `CLAUDE.md` section 5).
GENERALIZED_V1: EpisodeDesign = EpisodeDesign(
    design=EPISODE_DESIGN_GENERALIZED_V1,
    hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
    eligibility_policy=FD_ELIGIBILITY_CERTIFIED_V1,
    post_fd_wake_policy=POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
    reference_policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1,
)

_DESIGNS: Dict[str, EpisodeDesign] = {
    EPISODE_DESIGN_FIXED_CELL_V1: FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1: GENERALIZED_V1,
}


def resolve_episode_design(design: Any) -> EpisodeDesign:
    """The ONE site that turns a design id into its four policy ids.

    An unknown id RAISES rather than falling back on the historical bundle: a run that
    quietly measured the fixed cell while its config said ``generalized_v1`` would be a
    mislabelled measurement, which is worse than a crash. It is never case-folded into a
    match either -- a design id is a stored fact, not a spelling.

    Raises:
        ValueError: the id is not one of :data:`EPISODE_DESIGNS`.
    """
    key = str(design)
    if key not in _DESIGNS:
        raise ValueError(
            "unknown episode_design %r; expected one of %r"
            % (design, list(EPISODE_DESIGNS))
        )
    return _DESIGNS[key]


# =============================================================================
# 2. THE TRAINING CARDINALITY SAMPLER -- its own deterministic seed domain
# =============================================================================

# The sampler's identity, recorded in `run_config.json` so a run states which sampling
# rule produced its population rather than leaving it to be inferred from the results.
CARDINALITY_SAMPLER_POLICY: str = "generalized_cardinality_uniform_v1"

# The PRIVATE rng domain. Disjoint from `fuel_damage_v1`, `fuel_damage_severity_v1`,
# `fuel_damage_eligibility_v1`, the per-episode hidden-placement rng, global `random` and
# torch's generator -- see the module docstring for why each separation is load-bearing.
CARDINALITY_RNG_DOMAIN: str = "generalized_cardinality_v1"

# Where a resolved cardinality came from. Recorded per episode, because "A=3" produced by
# the fixed cell and "A=3" drawn by the sampler are different facts about the population.
CARDINALITY_SOURCE_FIXED_CELL: str = "fixed_cell"
CARDINALITY_SOURCE_SAMPLER: str = "generalized_sampler"
CARDINALITY_SOURCE_BENCHMARK: str = "benchmark_manifest"
CARDINALITY_SOURCES: Tuple[str, ...] = (
    CARDINALITY_SOURCE_FIXED_CELL,
    CARDINALITY_SOURCE_SAMPLER,
    CARDINALITY_SOURCE_BENCHMARK,
)


def derive_cardinality_seed(episode_seed: int) -> int:
    """A 64-bit seed for the cardinality draw, from the EPISODE SEED alone.

    ``SHA-256("generalized_cardinality_v1:<episode_seed>")``, truncated to 64 bits -- the
    same construction as ``graph_fuel_damage.derive_fuel_damage_seed`` and its two
    siblings, and deliberately NOT ``hash()``, which is salted per process and would make
    an episode's cardinality irreproducible between runs of the same seed.
    """
    payload = "%s:%d" % (CARDINALITY_RNG_DOMAIN, int(episode_seed))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class EpisodeCardinality:
    """ONE scheduled attempt's REQUESTED cardinality, and where it came from.

    ``hidden_requested`` is a REQUEST, never a promise: under the generalized bounded
    backoff a world may legitimately realize fewer hidden targets, and that shortfall is a
    recorded outcome rather than a failure (handoff 3l.1). Nothing in this object is ever
    rewritten to match what a world realized -- a silently reduced request is exactly what
    makes a denominator unreadable.
    """

    agent_count: int
    known_count: int
    hidden_requested: int
    source: str

    def __post_init__(self) -> None:
        if int(self.agent_count) < 1:
            raise ValueError("agent_count must be >= 1, got %r" % (self.agent_count,))
        if int(self.known_count) < 1:
            raise ValueError("known_count must be >= 1, got %r" % (self.known_count,))
        if int(self.hidden_requested) < 0:
            raise ValueError(
                "hidden_requested must be >= 0, got %r" % (self.hidden_requested,)
            )
        if str(self.source) not in CARDINALITY_SOURCES:
            raise ValueError(
                "unknown cardinality source %r; expected one of %r"
                % (self.source, list(CARDINALITY_SOURCES))
            )

    @property
    def targets_requested(self) -> int:
        """``K + H_requested`` -- the world size the schedule asked for."""
        return int(self.known_count) + int(self.hidden_requested)

    def to_record(self) -> Dict[str, Any]:
        """A JSON-ready view (plain builtins only)."""
        return {
            "agent_count": int(self.agent_count),
            "known_count": int(self.known_count),
            "hidden_requested": int(self.hidden_requested),
            "targets_requested": int(self.targets_requested),
            "source": str(self.source),
        }


def fixed_cell_cardinality(
    *, agent_count: int, known_count: int, hidden_requested: int
) -> EpisodeCardinality:
    """The HISTORICAL cardinality, taken verbatim from a fixed-cell config.

    No draw, no derivation and no seed: on the fixed-cell path the cardinality IS the
    configuration, and stating it through the same object the generalized path uses is
    what lets one downstream check serve both designs.
    """
    return EpisodeCardinality(
        agent_count=int(agent_count),
        known_count=int(known_count),
        hidden_requested=int(hidden_requested),
        source=CARDINALITY_SOURCE_FIXED_CELL,
    )


def sample_generalized_cardinality(*, episode_seed: int) -> EpisodeCardinality:
    """The approved GENERALIZED-V1 TRAINING draw: ``A``, then ``H | A``, with ``K == A``.

    Exactly, and in this order:

      1. ``A ~ Uniform(GENERALIZED_AGENT_COUNTS)`` -- ``{2, 3, 4}``, the cell the
         construction path accepts, IMPORTED rather than restated so the sampler cannot
         draw a team size construction would refuse;
      2. ``H_requested | A ~ Uniform({1, ..., A})`` -- drawn CONDITIONAL on ``A``, so the
         hidden load tracks the team size;
      3. ``K = A`` -- the known load tracks the team size too, by definition, not by a
         draw.

    Both draws run on a ``random.Random`` constructed HERE from
    :func:`derive_cardinality_seed`, so this function consumes NOTHING from global
    ``random``, from torch, or from any fuel-damage or placement stream, and nothing any
    of those consume can move its result. Two calls with the same ``episode_seed`` return
    the same cardinality, in this process and in any other.

    The index draws are written as explicit ``randrange`` calls rather than delegated to
    ``random.choice`` / ``random.randint`` so the exact number and order of draws is a
    stated contract a test can pin.
    """
    rng = random.Random(derive_cardinality_seed(int(episode_seed)))
    counts = tuple(int(a) for a in GENERALIZED_AGENT_COUNTS)
    agent_count = counts[rng.randrange(len(counts))]
    hidden_requested = 1 + rng.randrange(agent_count)
    return EpisodeCardinality(
        agent_count=int(agent_count),
        known_count=int(agent_count),
        hidden_requested=int(hidden_requested),
        source=CARDINALITY_SOURCE_SAMPLER,
    )


def cardinality_sampler_record() -> Dict[str, Any]:
    """The sampler's identity as a JSON-ready block for ``run_config.json``.

    States the RULE, not merely the policy name, so a reader can check the distribution a
    run sampled against the approved design without reading this module.
    """
    return {
        "policy": CARDINALITY_SAMPLER_POLICY,
        "rng_domain": CARDINALITY_RNG_DOMAIN,
        "seed_construction": "sha256('%s:<episode_seed>')[:8]" % CARDINALITY_RNG_DOMAIN,
        "agent_counts": [int(a) for a in GENERALIZED_AGENT_COUNTS],
        "agent_count_rule": "A ~ Uniform(agent_counts)",
        "known_rule": "K == A",
        "hidden_requested_rule": "H_requested | A ~ Uniform({1, ..., A})",
        "realized_hidden_may_be_short": True,
        "retry_on_short_realization": False,
    }


# =============================================================================
# 3. THE FROZEN STRATIFIED BENCHMARK -- 18 requested strata, matched world groups
# =============================================================================

class BenchmarkManifestError(ValueError):
    """A manifest is malformed, incomplete, out of order, or fails its own hash."""


class BenchmarkIdentityError(RuntimeError):
    """A benchmark world did not come out as its manifest (or its own group) says.

    An INSTRUMENT fault, not an episode outcome: the whole point of a matched world group
    is that its three members are the same world with one factor varied, so members that
    disagree about the world -- or a world that disagrees with the frozen manifest it was
    loaded from -- mean the benchmark is not measuring what it claims. The harness routes
    it as a measurement-integrity abort, never as accounted attrition.
    """


# The two REQUESTED hidden-load buckets. LOW is one hidden target whatever the team size;
# HIGH is one per agent. They are REQUESTED loads: bounded backoff may realize fewer, and
# whether HIGH systematically collapses toward LOW is a question the requested-vs-realized
# distribution answers -- inspected by a human before any measurement, never decided here
# by a threshold this module invented (handoff 3l.6).
LOAD_LOW: str = "low"
LOAD_HIGH: str = "high"
LOAD_BUCKETS: Tuple[str, ...] = (LOAD_LOW, LOAD_HIGH)

# The three CONDITION cells of a matched world group, in attempt order, each with the
# forced fuel-damage mode that realizes it. They reuse the existing forced modes rather
# than adding new ones -- a benchmark member is an ordinary evaluation member with a
# stratified world behind it.
BENCHMARK_MEMBERS: Tuple[Tuple[str, str], ...] = (
    (CONDITION_CLEAN, FuelDamageMode.FORCED_CLEAN),
    (SEVERITY_MILD, FuelDamageMode.FORCED_MILD),
    (SEVERITY_SEVERE, FuelDamageMode.FORCED_SEVERE),
)
BENCHMARK_CELLS: Tuple[str, ...] = tuple(cell for cell, _mode in BENCHMARK_MEMBERS)
BENCHMARK_GROUP_SIZE: int = len(BENCHMARK_MEMBERS)

# The within-group differences the benchmark reports, as `(cell, reference_cell)`. EVERY
# one is averaged over COMPLETE groups only.
BENCHMARK_DELTAS: Tuple[Tuple[str, str], ...] = (
    (SEVERITY_MILD, CONDITION_CLEAN),
    (SEVERITY_SEVERE, CONDITION_CLEAN),
    (SEVERITY_SEVERE, SEVERITY_MILD),
)

BENCHMARK_SCHEMA: str = "generalized_v1_benchmark_manifest"
BENCHMARK_SCHEMA_VERSION: int = 1


def hidden_requested_for(agent_count: int, load_bucket: str) -> int:
    """The REQUESTED hidden count of a stratum: ``1`` for LOW, ``A`` for HIGH."""
    bucket = str(load_bucket)
    if bucket == LOAD_LOW:
        return 1
    if bucket == LOAD_HIGH:
        return int(agent_count)
    raise BenchmarkManifestError(
        "unknown load bucket %r; expected one of %r" % (load_bucket, list(LOAD_BUCKETS))
    )


@dataclass(frozen=True)
class Stratum:
    """ONE of the 18 REQUESTED strata: a team size, a hidden load and a condition."""

    agent_count: int
    load_bucket: str
    cell: str

    @property
    def key(self) -> str:
        """The stable flat key a record and a plot label this stratum by."""
        return stratum_key(self.agent_count, self.load_bucket, self.cell)

    @property
    def hidden_requested(self) -> int:
        return hidden_requested_for(self.agent_count, self.load_bucket)

    def to_record(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "agent_count": int(self.agent_count),
            "load_bucket": str(self.load_bucket),
            "cell": str(self.cell),
            "known_requested": int(self.agent_count),
            "hidden_requested": int(self.hidden_requested),
        }


def stratum_key(agent_count: int, load_bucket: str, cell: str) -> str:
    """The flat stratum key for one member, without constructing a :class:`Stratum`."""
    return "A%d-%s-%s" % (int(agent_count), str(load_bucket), str(cell))


def group_key(agent_count: int, load_bucket: str, world_ordinal: int) -> str:
    """The stable identity of one matched WORLD GROUP.

    ``(A, load bucket, world ordinal)`` and nothing else. Deliberately NOT a uuid and
    deliberately not the seed alone: the ordinal is what makes a group's position in the
    manifest readable, and the seed travels beside it as the world's generator key.
    """
    return "A%d-%s-w%03d" % (int(agent_count), str(load_bucket), int(world_ordinal))


def base_cell_key(agent_count: int, load_bucket: str) -> str:
    """The flat key of an ``(A, load bucket)`` base cell -- a group's home."""
    return "A%d-%s" % (int(agent_count), str(load_bucket))


# The BASE CELLS -- the `(A, load bucket)` pairs a matched world group belongs to. Six of
# them; each contributes three strata (one per condition).
BENCHMARK_BASE_CELLS: Tuple[Tuple[int, str], ...] = tuple(
    (int(a), bucket)
    for a in GENERALIZED_AGENT_COUNTS
    for bucket in LOAD_BUCKETS
)
BENCHMARK_BASE_CELL_KEYS: Tuple[str, ...] = tuple(
    base_cell_key(a, bucket) for (a, bucket) in BENCHMARK_BASE_CELLS
)

# THE 18 REQUESTED STRATA, in canonical order: agent count, then load bucket, then
# condition. `3 x 2 x 3 = 18`, and the tuple is built from that product rather than
# listed, so the count cannot drift from the design.
BENCHMARK_STRATA: Tuple[Stratum, ...] = tuple(
    Stratum(agent_count=a, load_bucket=bucket, cell=cell)
    for (a, bucket) in BENCHMARK_BASE_CELLS
    for cell in BENCHMARK_CELLS
)
BENCHMARK_STRATUM_KEYS: Tuple[str, ...] = tuple(s.key for s in BENCHMARK_STRATA)


@dataclass(frozen=True)
class WorldIdentity:
    """What a benchmark world REALLY came out as, in id-free terms.

    Observed once per member episode and compared two ways: against the manifest's frozen
    preflight (when it has one), and across the three members of the group (always). Every
    field is either a count, an ORDINAL, or a hash of scalars with uuid text removed -- so
    an identity claim can never rest on a generated uuid, which is not seed-derived.
    """

    hidden_realized: int
    known_realized: int
    geometric_fingerprint: Tuple[Tuple[float, float], ...]
    fd_selected_ordinal: Optional[int]
    fd_certificate_fingerprint: Optional[str]

    def to_record(self) -> Dict[str, Any]:
        return {
            "hidden_realized": int(self.hidden_realized),
            "known_realized": int(self.known_realized),
            "geometric_fingerprint": [list(p) for p in self.geometric_fingerprint],
            "fd_selected_ordinal": self.fd_selected_ordinal,
            "fd_certificate_fingerprint": self.fd_certificate_fingerprint,
        }


# Certificate fields that carry a generated uuid and are therefore EXCLUDED from the
# certificate fingerprint. Everything else on the certificate is a tick, a count, a
# coordinate or a fuel quantity -- reproducible from the seed, and comparable across runs.
_CERTIFICATE_ID_FIELDS: Tuple[str, ...] = ("ego_id",)


def certificate_fingerprint(certificate: Optional[Mapping[str, Any]]) -> Optional[str]:
    """A stable content hash of an FD event certificate, WITHOUT its ego uuid.

    ``None`` in, ``None`` out -- a legacy (uncertified) episode has no certificate, and
    fabricating a fingerprint for it would make "these two worlds certified the same
    event" answerable where it is not.
    """
    if certificate is None:
        return None
    payload = {
        k: v for k, v in dict(certificate).items() if k not in _CERTIFICATE_ID_FIELDS
    }
    return _content_hash(payload)


@dataclass(frozen=True)
class WorldPreflight:
    """A benchmark world's REALIZED identity, frozen into the manifest by a preflight.

    OPTIONAL by design. A manifest may be frozen BEFORE any world has been constructed
    (the mechanism this task delivers), in which case every world carries ``None`` here
    and the harness records what it observes. Once a preflight has run and its results are
    frozen in, the harness VERIFIES against them and REFUSES a world that came out
    differently, rather than regenerating or substituting one.
    """

    hidden_realized: int
    known_realized: int
    geometric_fingerprint: Tuple[Tuple[float, float], ...]
    fd_selected_ordinal: Optional[int] = None
    fd_certificate_fingerprint: Optional[str] = None
    construction_audit: Optional[Dict[str, Any]] = None

    @property
    def identity(self) -> WorldIdentity:
        return WorldIdentity(
            hidden_realized=int(self.hidden_realized),
            known_realized=int(self.known_realized),
            geometric_fingerprint=self.geometric_fingerprint,
            fd_selected_ordinal=self.fd_selected_ordinal,
            fd_certificate_fingerprint=self.fd_certificate_fingerprint,
        )

    def to_record(self) -> Dict[str, Any]:
        record = self.identity.to_record()
        record["construction_audit"] = self.construction_audit
        return record

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> "WorldPreflight":
        return WorldPreflight(
            hidden_realized=int(record["hidden_realized"]),
            known_realized=int(record["known_realized"]),
            geometric_fingerprint=_as_fingerprint(
                record.get("geometric_fingerprint") or ()
            ),
            fd_selected_ordinal=(
                None if record.get("fd_selected_ordinal") is None
                else int(record["fd_selected_ordinal"])
            ),
            fd_certificate_fingerprint=record.get("fd_certificate_fingerprint"),
            construction_audit=record.get("construction_audit"),
        )


@dataclass(frozen=True)
class BenchmarkWorld:
    """ONE matched WORLD GROUP of the frozen benchmark: three members, one world.

    The scientific comparison unit. CLEAN, MILD and SEVERE members share the SAME seed --
    hence the same generated world, the same requested cardinality, the same solved
    ``A_init``, the same hidden geometry and, under the certified eligibility policy whose
    walk depends on the episode seed alone, the SAME certified damaged ego and the same
    certified event point. Only the damage condition differs.
    """

    agent_count: int
    load_bucket: str
    world_ordinal: int
    seed: int
    preflight: Optional[WorldPreflight] = None

    def __post_init__(self) -> None:
        if int(self.agent_count) not in tuple(
            int(a) for a in GENERALIZED_AGENT_COUNTS
        ):
            raise BenchmarkManifestError(
                "benchmark world A=%r is outside the approved generalized cell %r"
                % (self.agent_count, list(GENERALIZED_AGENT_COUNTS))
            )
        if str(self.load_bucket) not in LOAD_BUCKETS:
            raise BenchmarkManifestError(
                "unknown load bucket %r; expected one of %r"
                % (self.load_bucket, list(LOAD_BUCKETS))
            )
        if int(self.world_ordinal) < 0:
            raise BenchmarkManifestError(
                "world_ordinal must be >= 0, got %r" % (self.world_ordinal,)
            )
        if int(self.seed) < 0:
            raise BenchmarkManifestError("seed must be >= 0, got %r" % (self.seed,))

    @property
    def known_count(self) -> int:
        """``K == A``."""
        return int(self.agent_count)

    @property
    def hidden_requested(self) -> int:
        return hidden_requested_for(self.agent_count, self.load_bucket)

    @property
    def key(self) -> str:
        return group_key(self.agent_count, self.load_bucket, self.world_ordinal)

    @property
    def base_cell(self) -> Tuple[int, str]:
        return (int(self.agent_count), str(self.load_bucket))

    @property
    def base_cell_key(self) -> str:
        return base_cell_key(self.agent_count, self.load_bucket)

    def cardinality(self) -> EpisodeCardinality:
        """The REQUESTED cardinality every member of this group is built with."""
        return EpisodeCardinality(
            agent_count=int(self.agent_count),
            known_count=self.known_count,
            hidden_requested=self.hidden_requested,
            source=CARDINALITY_SOURCE_BENCHMARK,
        )

    def members(self) -> Tuple[Tuple[str, str], ...]:
        """``((cell, forced mode), ...)`` in attempt order -- always the three cells."""
        return BENCHMARK_MEMBERS

    def stratum_key(self, cell: str) -> str:
        return stratum_key(self.agent_count, self.load_bucket, cell)

    def to_record(self) -> Dict[str, Any]:
        return {
            "group_key": self.key,
            "agent_count": int(self.agent_count),
            "load_bucket": str(self.load_bucket),
            "world_ordinal": int(self.world_ordinal),
            "seed": int(self.seed),
            "known_requested": self.known_count,
            "hidden_requested": self.hidden_requested,
            "targets_requested": self.known_count + self.hidden_requested,
            "preflight": None if self.preflight is None else self.preflight.to_record(),
        }

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> "BenchmarkWorld":
        preflight_record = record.get("preflight")
        return BenchmarkWorld(
            agent_count=int(record["agent_count"]),
            load_bucket=str(record["load_bucket"]),
            world_ordinal=int(record["world_ordinal"]),
            seed=int(record["seed"]),
            preflight=(
                None if preflight_record is None
                else WorldPreflight.from_record(preflight_record)
            ),
        )


@dataclass(frozen=True)
class BenchmarkManifest:
    """A FROZEN, ordered, content-addressed benchmark population.

    IMMUTABLE AND AUDITABLE, which is what makes "the actor-only arm and the CTDE arm ran
    the same benchmark" a checkable claim rather than an assertion: two runs quote the
    same :attr:`manifest_id`, or they did not.

    THE SCALE IS NOT DECIDED HERE. :func:`build_benchmark_manifest` refuses to invent a
    world count: a caller must state ``worlds_per_cell`` with an explicit
    ``benchmark_base_seed``, or hand in an explicit world list. Choosing the final
    scientific scale is a later task that owns bounded runtime / solver validation first
    (handoff 3l.8 step 5), and a default here would quietly make that decision.
    """

    schema: str
    schema_version: int
    design: str
    worlds: Tuple[BenchmarkWorld, ...]
    manifest_id: str
    label: Optional[str] = None
    notes: Optional[str] = None

    # ---- reads ------------------------------------------------------------------
    @property
    def n_worlds(self) -> int:
        """Matched world GROUPS -- not episodes."""
        return len(self.worlds)

    @property
    def n_members(self) -> int:
        """Scheduled EPISODE attempts per evaluation round: ``3 x n_worlds``."""
        return len(self.worlds) * BENCHMARK_GROUP_SIZE

    @property
    def base_cells(self) -> Tuple[Tuple[int, str], ...]:
        return BENCHMARK_BASE_CELLS

    @property
    def strata(self) -> Tuple[Stratum, ...]:
        return BENCHMARK_STRATA

    def worlds_per_base_cell(self) -> Dict[str, int]:
        """How many world groups each ``(A, bucket)`` base cell holds."""
        counts = {key: 0 for key in BENCHMARK_BASE_CELL_KEYS}
        for world in self.worlds:
            counts[world.base_cell_key] = counts.get(world.base_cell_key, 0) + 1
        return counts

    # ---- serialization ----------------------------------------------------------
    def payload(self) -> Dict[str, Any]:
        """The CANONICAL content of this manifest -- everything except its own hash.

        ``manifest_id`` is excluded because it is the hash OF this payload; including it
        would make the identity self-referential and unverifiable.
        """
        return {
            "schema": str(self.schema),
            "schema_version": int(self.schema_version),
            "design": str(self.design),
            "label": self.label,
            "notes": self.notes,
            "group_size": BENCHMARK_GROUP_SIZE,
            "group_cells": list(BENCHMARK_CELLS),
            "group_modes": [mode for _cell, mode in BENCHMARK_MEMBERS],
            "n_strata": len(BENCHMARK_STRATA),
            "strata": [s.to_record() for s in BENCHMARK_STRATA],
            "n_worlds": self.n_worlds,
            "n_members": self.n_members,
            "worlds": [w.to_record() for w in self.worlds],
        }

    def to_record(self) -> Dict[str, Any]:
        """The full JSON document: the canonical payload plus its identity."""
        record = self.payload()
        record["manifest_id"] = str(self.manifest_id)
        return record

    def canonical_json(self) -> str:
        """The exact byte string :attr:`manifest_id` is the SHA-256 of."""
        return _canonical_json(self.payload())

    def identity_record(self) -> Dict[str, Any]:
        """A short provenance block for ``run_config.json`` (no world list)."""
        return {
            "schema": str(self.schema),
            "schema_version": int(self.schema_version),
            "design": str(self.design),
            "label": self.label,
            "manifest_id": str(self.manifest_id),
            "n_worlds": self.n_worlds,
            "n_members": self.n_members,
            "n_strata": len(BENCHMARK_STRATA),
            "group_size": BENCHMARK_GROUP_SIZE,
            "group_cells": list(BENCHMARK_CELLS),
            "worlds_per_base_cell": self.worlds_per_base_cell(),
            "preflighted": all(w.preflight is not None for w in self.worlds),
        }

    def seeds(self) -> Tuple[int, ...]:
        return tuple(int(w.seed) for w in self.worlds)


# =============================================================================
# 3b. Manifest construction, hashing and loading
# =============================================================================

def _canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, ASCII-escaped.

    ONE serialization site behind both the hash and the written file, so a manifest's
    identity cannot depend on which of the two produced the bytes.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _content_hash(payload: Any) -> str:
    """SHA-256 of :func:`_canonical_json` -- the content address of a payload."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def manifest_identity(payload: Mapping[str, Any]) -> str:
    """The manifest id of a canonical payload (the payload must exclude the id itself)."""
    return _content_hash(dict(payload))


def _as_fingerprint(raw: Any) -> Tuple[Tuple[float, float], ...]:
    """Normalize a geometric fingerprint to a tuple of ``(lat, lon)`` float pairs."""
    out: List[Tuple[float, float]] = []
    for pair in raw or ():
        lat, lon = tuple(pair)
        out.append((float(lat), float(lon)))
    return tuple(out)


def _canonical_world_order(
    worlds: Sequence[BenchmarkWorld],
) -> Tuple[BenchmarkWorld, ...]:
    """Sort worlds into the canonical order: base cell, then world ordinal, then seed.

    Base-cell order is :data:`BENCHMARK_BASE_CELLS`, not lexical -- a manifest's order is
    part of its identity, and sorting by a string would make it depend on how a bucket
    happens to be spelled.
    """
    cell_rank = {cell: i for i, cell in enumerate(BENCHMARK_BASE_CELLS)}
    return tuple(sorted(
        worlds,
        key=lambda w: (cell_rank[w.base_cell], int(w.world_ordinal), int(w.seed)),
    ))


def build_benchmark_manifest(
    *,
    worlds_per_cell: Optional[int] = None,
    benchmark_base_seed: Optional[int] = None,
    worlds: Optional[Sequence[Mapping[str, Any]]] = None,
    label: Optional[str] = None,
    notes: Optional[str] = None,
) -> BenchmarkManifest:
    """Freeze a benchmark population -- from an EXPLICIT scale or an EXPLICIT world list.

    TWO mutually exclusive ways to say which worlds the benchmark holds, and NEITHER has a
    default:

      * ``worlds_per_cell`` + ``benchmark_base_seed`` -- ``worlds_per_cell`` matched world
        groups in each of the six ``(A, load bucket)`` base cells, their seeds allocated
        consecutively from ``benchmark_base_seed`` in canonical order. Both are required
        together: a seed base without a count says nothing, and a count without a seed
        base would need this module to invent a seed band.
      * ``worlds`` -- an explicit list of ``{"agent_count", "load_bucket", "seed"}``
        mappings (optionally with ``"world_ordinal"`` and a frozen ``"preflight"``).

    THE SCALE IS DELIBERATELY NOT DEFAULTED. How many worlds a scientific benchmark needs
    depends on bounded runtime / solver validation that has not been done (handoff 3l.8
    step 5), and a default here would silently make that decision. Calling this with
    neither form -- or with both -- RAISES.

    Every base cell must end up with at least one world: a manifest missing a base cell is
    missing three of the eighteen strata, and is not this benchmark.

    Raises:
        BenchmarkManifestError: on a missing/ambiguous scale, a non-positive count, an
            out-of-cell team size, an unknown bucket, a duplicate seed, or an empty base
            cell.
    """
    if (worlds is None) == (worlds_per_cell is None):
        raise BenchmarkManifestError(
            "build_benchmark_manifest requires EXACTLY ONE of: worlds_per_cell (with "
            "benchmark_base_seed), or an explicit worlds list. The benchmark scale is "
            "never defaulted -- how many worlds a scientific run needs is decided by "
            "bounded runtime validation, not by this module."
        )

    built: List[BenchmarkWorld] = []
    if worlds_per_cell is not None:
        if benchmark_base_seed is None:
            raise BenchmarkManifestError(
                "worlds_per_cell requires an explicit benchmark_base_seed: a seed band "
                "this module invented would not be auditable."
            )
        count = int(worlds_per_cell)
        if count < 1:
            raise BenchmarkManifestError(
                "worlds_per_cell must be >= 1, got %r" % (worlds_per_cell,)
            )
        seed = int(benchmark_base_seed)
        if seed < 0:
            raise BenchmarkManifestError(
                "benchmark_base_seed must be >= 0, got %r" % (benchmark_base_seed,)
            )
        for (agent_count, bucket) in BENCHMARK_BASE_CELLS:
            for ordinal in range(count):
                built.append(BenchmarkWorld(
                    agent_count=int(agent_count),
                    load_bucket=str(bucket),
                    world_ordinal=int(ordinal),
                    seed=seed,
                ))
                seed += 1
    else:
        per_cell_next: Dict[Tuple[int, str], int] = {}
        for entry in worlds or ():
            agent_count = int(entry["agent_count"])
            bucket = str(entry["load_bucket"])
            cell = (agent_count, bucket)
            ordinal = entry.get("world_ordinal")
            if ordinal is None:
                ordinal = per_cell_next.get(cell, 0)
            per_cell_next[cell] = max(int(ordinal) + 1, per_cell_next.get(cell, 0))
            preflight_record = entry.get("preflight")
            built.append(BenchmarkWorld(
                agent_count=agent_count,
                load_bucket=bucket,
                world_ordinal=int(ordinal),
                seed=int(entry["seed"]),
                preflight=(
                    None if preflight_record is None
                    else WorldPreflight.from_record(preflight_record)
                ),
            ))

    ordered = _canonical_world_order(built)
    _require_well_formed_worlds(ordered)
    draft = BenchmarkManifest(
        schema=BENCHMARK_SCHEMA,
        schema_version=BENCHMARK_SCHEMA_VERSION,
        design=EPISODE_DESIGN_GENERALIZED_V1,
        worlds=ordered,
        manifest_id="",
        label=label,
        notes=notes,
    )
    return BenchmarkManifest(
        schema=draft.schema,
        schema_version=draft.schema_version,
        design=draft.design,
        worlds=draft.worlds,
        manifest_id=manifest_identity(draft.payload()),
        label=draft.label,
        notes=draft.notes,
    )


def _require_well_formed_worlds(worlds: Sequence[BenchmarkWorld]) -> None:
    """Every base cell populated, every seed unique, every group key unique."""
    if not worlds:
        raise BenchmarkManifestError("a benchmark manifest holds no worlds")
    seen_seeds: Dict[int, str] = {}
    seen_keys: Dict[str, int] = {}
    per_cell: Dict[Tuple[int, str], int] = {cell: 0 for cell in BENCHMARK_BASE_CELLS}
    for world in worlds:
        if int(world.seed) in seen_seeds:
            raise BenchmarkManifestError(
                "benchmark seed %d appears twice (%s and %s): a seed IS a world's "
                "generator identity, so two worlds sharing one is ambiguous."
                % (world.seed, seen_seeds[int(world.seed)], world.key)
            )
        seen_seeds[int(world.seed)] = world.key
        if world.key in seen_keys:
            raise BenchmarkManifestError(
                "benchmark world group %s appears twice" % world.key
            )
        seen_keys[world.key] = int(world.seed)
        per_cell[world.base_cell] = per_cell.get(world.base_cell, 0) + 1
    empty = sorted(base_cell_key(a, b) for (a, b), n in per_cell.items() if n < 1)
    if empty:
        raise BenchmarkManifestError(
            "benchmark base cell(s) %r hold no world, so %d of the %d requested strata "
            "would be empty; a manifest missing a stratum is not this benchmark."
            % (empty, len(empty) * BENCHMARK_GROUP_SIZE, len(BENCHMARK_STRATA))
        )


def manifest_from_record(record: Mapping[str, Any]) -> BenchmarkManifest:
    """Rebuild and VERIFY a manifest from its JSON document.

    Verifies, in order: the schema and version; that the design is ``generalized_v1``;
    that the stored world order is exactly the canonical order (a reordered manifest is a
    different population presented as the same one); that the worlds are well formed and
    cover every base cell; that each world's stated ``known_requested`` /
    ``hidden_requested`` match what its stratum requires; and finally that the stored
    ``manifest_id`` is the hash of the payload the file actually carries.

    THE HASH IS CHECKED LAST AND IS NOT ADVISORY. A manifest whose content no longer
    matches its identity is REFUSED -- not repaired, not re-hashed, and not loaded with a
    warning -- because silently accepting it is exactly the substitution the frozen
    manifest exists to prevent.

    Raises:
        BenchmarkManifestError: on any of the above.
    """
    schema = str(record.get("schema"))
    if schema != BENCHMARK_SCHEMA:
        raise BenchmarkManifestError(
            "not a benchmark manifest: schema=%r, expected %r"
            % (record.get("schema"), BENCHMARK_SCHEMA)
        )
    version = int(record.get("schema_version", -1))
    if version != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkManifestError(
            "benchmark manifest schema_version=%r, this code reads %r"
            % (record.get("schema_version"), BENCHMARK_SCHEMA_VERSION)
        )
    design = str(record.get("design"))
    if design != EPISODE_DESIGN_GENERALIZED_V1:
        raise BenchmarkManifestError(
            "benchmark manifest design=%r; the stratified benchmark is defined for %r "
            "only" % (record.get("design"), EPISODE_DESIGN_GENERALIZED_V1)
        )
    raw_worlds = list(record.get("worlds") or ())
    worlds = tuple(BenchmarkWorld.from_record(w) for w in raw_worlds)
    if worlds != _canonical_world_order(worlds):
        raise BenchmarkManifestError(
            "benchmark manifest worlds are not in canonical order (base cell, then "
            "world ordinal, then seed); the stored order is part of the manifest's "
            "identity and is never re-sorted on load."
        )
    _require_well_formed_worlds(worlds)
    for raw, world in zip(raw_worlds, worlds):
        stated_known = raw.get("known_requested")
        stated_hidden = raw.get("hidden_requested")
        if stated_known is not None and int(stated_known) != world.known_count:
            raise BenchmarkManifestError(
                "benchmark world %s states known_requested=%r but K == A == %d"
                % (world.key, stated_known, world.known_count)
            )
        if stated_hidden is not None and int(stated_hidden) != world.hidden_requested:
            raise BenchmarkManifestError(
                "benchmark world %s states hidden_requested=%r but its %s stratum "
                "requires %d"
                % (world.key, stated_hidden, world.load_bucket, world.hidden_requested)
            )
    manifest = BenchmarkManifest(
        schema=schema,
        schema_version=version,
        design=design,
        worlds=worlds,
        manifest_id=str(record.get("manifest_id", "")),
        label=record.get("label"),
        notes=record.get("notes"),
    )
    recomputed = manifest_identity(manifest.payload())
    if manifest.manifest_id != recomputed:
        raise BenchmarkManifestError(
            "benchmark manifest identity MISMATCH: the file states manifest_id=%r but "
            "its content hashes to %r. The population was edited after it was frozen, "
            "so it is refused rather than silently re-hashed."
            % (manifest.manifest_id, recomputed)
        )
    return manifest


def load_benchmark_manifest(path: Union[str, Path]) -> BenchmarkManifest:
    """Read and verify a frozen manifest from disk (stdlib ``json`` only)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        record = json.loads(text)
    except ValueError as exc:
        raise BenchmarkManifestError(
            "benchmark manifest %s is not valid JSON: %s" % (str(path), exc)
        ) from exc
    if not isinstance(record, dict):
        raise BenchmarkManifestError(
            "benchmark manifest %s must be a JSON object" % str(path)
        )
    return manifest_from_record(record)


def write_benchmark_manifest(
    manifest: BenchmarkManifest, path: Union[str, Path]
) -> Path:
    """Write a manifest as its CANONICAL bytes, so the file hashes to its own id."""
    out = Path(path)
    if out.parent and str(out.parent):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_canonical_json(manifest.to_record()), encoding="utf-8")
    return out


# =============================================================================
# 3c. Matched-world identity checks
# =============================================================================

def require_world_matches_manifest(
    world: BenchmarkWorld, observed: WorldIdentity
) -> None:
    """A preflighted world must come out as its frozen manifest says, or the run ABORTS.

    A no-op when the manifest carries NO preflight for this world -- there is nothing to
    contradict, and inventing an expectation would be worse than having none.

    Raises:
        BenchmarkIdentityError: the world differs from the frozen record.
    """
    expected = world.preflight
    if expected is None:
        return
    wrong = identity_differences(expected.identity, observed)
    if wrong:
        raise BenchmarkIdentityError(
            "benchmark world %s (seed %d) does not match the frozen manifest: %s. The "
            "member is REFUSED rather than regenerated or substituted -- a benchmark "
            "whose worlds can be quietly replaced measures a population nobody froze."
            % (world.key, world.seed, "; ".join(wrong))
        )


def require_matched_group_identity(
    world: BenchmarkWorld, identities: Mapping[str, WorldIdentity]
) -> None:
    """The completed members of one matched group must all describe the SAME world.

    The comparison unit is "one world with one factor varied", so CLEAN, MILD and SEVERE
    must agree on the realized cardinality, the id-free hidden geometry and the certified
    damaged ego's ORDINAL. If they do not, the group is not matched and its within-group
    deltas would compare two different worlds.

    Compares only the members that actually completed: an incomplete group is a
    denominator question, not an integrity fault, and it is accounted separately.

    Raises:
        BenchmarkIdentityError: two completed members disagree about their world.
    """
    completed = list(identities.items())
    if len(completed) < 2:
        return
    reference_cell, reference = completed[0]
    for cell, ident in completed[1:]:
        wrong = identity_differences(reference, ident)
        if wrong:
            raise BenchmarkIdentityError(
                "benchmark world %s (seed %d): matched members %r and %r did not "
                "produce the same world (%s). Their within-group delta would compare "
                "two different worlds, so the run stops."
                % (world.key, world.seed, reference_cell, cell, "; ".join(wrong))
            )


def identity_differences(
    expected: WorldIdentity, observed: WorldIdentity
) -> List[str]:
    """Human-readable differences between two world identities (empty when equal)."""
    wrong: List[str] = []
    if int(expected.hidden_realized) != int(observed.hidden_realized):
        wrong.append("hidden_realized %d != %d"
                     % (expected.hidden_realized, observed.hidden_realized))
    if int(expected.known_realized) != int(observed.known_realized):
        wrong.append("known_realized %d != %d"
                     % (expected.known_realized, observed.known_realized))
    if expected.geometric_fingerprint != observed.geometric_fingerprint:
        wrong.append("geometric fingerprint differs")
    if expected.fd_selected_ordinal != observed.fd_selected_ordinal:
        wrong.append("certified ego ordinal %r != %r"
                     % (expected.fd_selected_ordinal, observed.fd_selected_ordinal))
    if expected.fd_certificate_fingerprint != observed.fd_certificate_fingerprint:
        wrong.append("fd certificate fingerprint differs")
    return wrong


# =============================================================================
# 4. Self-test (pure: no BLADE, no solver call, no torch)
# =============================================================================

def _selftest() -> None:
    """Prove the design resolution, the sampler's domain and the manifest's identity."""
    print("=" * 70)
    print("graph_generalized selftest")
    print("=" * 70)

    # --- 1. design resolution -------------------------------------------------
    assert resolve_episode_design(EPISODE_DESIGN_FIXED_CELL_V1) is FIXED_CELL_V1
    assert resolve_episode_design(EPISODE_DESIGN_GENERALIZED_V1) is GENERALIZED_V1
    assert not FIXED_CELL_V1.generalized and GENERALIZED_V1.generalized
    try:
        resolve_episode_design("generalized")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown design must raise, never fall back")
    print("[1] design resolution OK (unknown ids raise)")

    # --- 2. the sampler's support and determinism ------------------------------
    seen_a, seen_h = set(), set()
    for seed in range(4000):
        card = sample_generalized_cardinality(episode_seed=seed)
        assert card.agent_count in GENERALIZED_AGENT_COUNTS
        assert card.known_count == card.agent_count
        assert 1 <= card.hidden_requested <= card.agent_count
        assert card == sample_generalized_cardinality(episode_seed=seed)
        seen_a.add(card.agent_count)
        seen_h.add((card.agent_count, card.hidden_requested))
    assert seen_a == set(int(a) for a in GENERALIZED_AGENT_COUNTS), seen_a
    expected_support = {(a, h) for a in GENERALIZED_AGENT_COUNTS
                        for h in range(1, a + 1)}
    assert seen_h == expected_support, sorted(seen_h)
    print("[2] sampler: full (A, H) support, %d cell(s), reproducible per seed"
          % len(seen_h))

    # --- 3. RNG isolation ------------------------------------------------------
    random.seed(1234)
    before = random.random()
    random.seed(1234)
    baseline = sample_generalized_cardinality(episode_seed=99)
    after = random.random()
    assert before == after, "the sampler consumed global random state"
    assert baseline == sample_generalized_cardinality(episode_seed=99)
    print("[3] rng isolation OK (global random untouched)")

    # --- 4. the 18 strata ------------------------------------------------------
    assert len(BENCHMARK_STRATA) == 18, len(BENCHMARK_STRATA)
    assert len(set(BENCHMARK_STRATUM_KEYS)) == 18
    assert len(BENCHMARK_BASE_CELLS) == 6
    print("[4] strata: %d (%s ...)" % (len(BENCHMARK_STRATA),
                                       ", ".join(BENCHMARK_STRATUM_KEYS[:3])))

    # --- 5. manifest identity, order and refusal -------------------------------
    m = build_benchmark_manifest(worlds_per_cell=2, benchmark_base_seed=5_000_000)
    assert m.n_worlds == 12 and m.n_members == 36
    again = build_benchmark_manifest(worlds_per_cell=2, benchmark_base_seed=5_000_000)
    assert m.manifest_id == again.manifest_id
    assert manifest_from_record(m.to_record()).manifest_id == m.manifest_id
    tampered = m.to_record()
    tampered["worlds"][0]["seed"] = 42
    try:
        manifest_from_record(tampered)
    except BenchmarkManifestError:
        pass
    else:
        raise AssertionError("a tampered manifest must be refused")
    try:
        build_benchmark_manifest()
    except BenchmarkManifestError:
        pass
    else:
        raise AssertionError("the benchmark scale must never be defaulted")
    print("[5] manifest %s: %d world(s), %d member(s), identity verified"
          % (m.manifest_id[:12], m.n_worlds, m.n_members))
    print("ALL GENERALIZED-V1 POPULATION CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
