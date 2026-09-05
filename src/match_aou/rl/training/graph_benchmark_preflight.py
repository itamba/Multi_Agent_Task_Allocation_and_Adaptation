"""
GENERALIZED-V1 Task 5C: the deterministic BENCHMARK PREFLIGHT.

WHAT THIS MODULE IS FOR
=======================
A frozen benchmark manifest is the comparator: two campaign arms quote the same
``manifest_id``, and their within-world deltas are therefore taken over the same
population. That only works if the population is COMPLETE -- every world in it can
actually be built. Under ``generalized_v1`` a candidate world may legitimately fail:
bounded-backoff construction can refuse it, and the certified FD eligibility walk can
find no ego that supports BOTH severity bands at a predicted event state. A manifest
frozen without checking would carry such worlds forever, and every validation round of
every arm would fail the SAME member again -- a permanently missing stratum member
wearing the label of ordinary attrition.

So the replacement happens ONCE, HERE, BEFORE the freeze:

  * this module SCANS a bounded, deterministic candidate-seed window per base cell and
    accepts the first ``worlds_per_cell`` worlds that satisfy the construction and
    certified-FD contract;
  * every rejected candidate is recorded once, its seed is SPENT, and the next candidate
    replaces it;
  * the accepted worlds are frozen with their ``WorldPreflight`` identities and written
    as a canonical, content-addressed manifest.

AND IT NEVER HAPPENS AGAIN. ``graph_train.evaluate_benchmark`` is deliberately NOT
touched: a frozen manifest stays immutable, a failed member is never substituted at
runtime, and the identity check (``graph_generalized.require_world_matches_manifest``)
still REFUSES a world that comes out differently from what this module froze. Population
selection lives here; scientific evaluation lives there; the two must not merge.

WHAT IT DELIBERATELY DOES NOT DO
================================
It tests WORLD ELIGIBILITY, never policy performance. No policy is built, no episode is
run, no reward is computed, and no reward, return or behaviour influences which worlds
are accepted -- selecting benchmark worlds by outcome would build the comparator out of
the very quantity the comparison measures. It also does not choose the SCIENTIFIC SCALE:
``worlds_per_cell``, ``benchmark_base_seed`` and ``max_candidates_per_cell`` are ALL
required and none is defaulted, exactly as ``build_benchmark_manifest`` refuses to invent
a world count.

A contract-successful bounded-backoff world is NOT rejected merely because
``hidden_realized < hidden_requested``. The shortfall is a RECORDED outcome -- frozen into
the world's ``WorldPreflight`` and reported in the build report -- and whether the
resulting distribution is acceptable is a human / GPT scientific-review decision taken
before any measurement, never a threshold invented here.

INDEPENDENT WINDOWS, SO ONE CELL CANNOT MOVE ANOTHER
====================================================
Each of the six canonical ``BENCHMARK_BASE_CELLS`` owns its OWN half-open candidate-seed
window::

    start = benchmark_base_seed + c * max_candidates_per_cell
    stop  = start + max_candidates_per_cell

with ``c`` the cell's ordinal in ``BENCHMARK_BASE_CELLS``. Candidates are scanned in
ascending seed order inside that window and the walk stops at ``worlds_per_cell``
acceptances. A rejection in cell ``c`` therefore cannot shift the accepted seeds of cell
``c+1`` -- which is what makes "we re-ran the preflight and the A=2/LOW worlds are the
same worlds" checkable rather than hoped for.

Exhausting a cell's window before its quota is filled ABORTS
(:class:`BenchmarkPreflightError`), stops before any later cell is scanned, and writes NO
manifest: a benchmark missing a stratum member is not this benchmark, and a partial
manifest that presented itself as complete would be worse than none.

THE AUDIT SURVIVES THE ABORT. Every candidate already attempted has SPENT its seed, so
its identity and its rejection reason are evidence an operator needs -- to decide whether
to raise ``max_candidates_per_cell``, lower ``worlds_per_cell``, or investigate the
attrition itself -- and re-running the same window cannot produce them again differently.
So a failed preflight still WRITES ``benchmark_preflight_report.json`` before raising,
carrying the completed cells, the exhausted cell and every accepted and rejected
candidate inside them. The two outcomes are told apart by ONE field, ``status``
(:data:`PREFLIGHT_STATUS_COMPLETE` / :data:`PREFLIGHT_STATUS_FAILED`), never by the shape
of the document: a failed report carries ``manifest: null``, ``manifest_written: false``
and a ``failure`` block naming the exhausted cell. With ``output_dir=None`` no file can be
written, so the same report travels on the exception's ``report`` attribute instead.

REPLACEMENT-ELIGIBILITY IS THE SAME DISTINCTION THE TRAINER MAKES
================================================================
An ordinary construction / certification rejection is replacement-eligible. A
MEASUREMENT-INTEGRITY fault is not, and never will be: ``MeasurementIntegrityError``,
``FuelDamageIntegrityError``, ``BenchmarkIdentityError`` and an ABORTING
``ReferenceIntegrityError`` propagate and stop the preflight, exactly as they stop a
training run. A world that contradicts its own certificate says the instrument is wrong,
and replacing it would freeze a population selected by a defect.

PURITY
======
This module is a CONSUMER of :mod:`graph_train` and :mod:`graph_generalized`; the import
direction is one-way and neither of them may ever import this one. It needs BLADE and
BONMIN because it really constructs worlds, so it is not an import-purity entry module --
the same standing as ``graph_train`` and ``graph_rollout``.

Run:
    python -m match_aou.rl.training.graph_benchmark_preflight \
        --worlds-per-cell 2 --benchmark-base-seed 800000 \
        --max-candidates-per-cell 12 --out preflight_out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .graph_episode_setup import setup_episode
from .graph_fuel_damage import (
    FD_ELIGIBILITY_REJECTION_REASONS,
    NO_FD_ELIGIBLE_EGO,
    FuelDamageIntegrityError,
    build_fuel_damage_controller,
)
from .graph_generalized import (
    BENCHMARK_BASE_CELLS,
    BENCHMARK_MEMBERS,
    CARDINALITY_SOURCE_BENCHMARK,
    EPISODE_DESIGN_GENERALIZED_V1,
    BenchmarkIdentityError,
    BenchmarkManifest,
    EpisodeCardinality,
    WorldPreflight,
    base_cell_key,
    build_benchmark_manifest,
    hidden_requested_for,
    identity_differences,
    write_benchmark_manifest,
)
from ...solvers.match_aou_backend import (
    MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
    MATCH_AOU_BACKENDS,
    MatchAouBackendError,
    resolve_match_aou_backend,
)
from .graph_reward import ReferenceIntegrityError, reference_fault_aborts
from .graph_train import (
    _REPO_ROOT,
    EpisodeAttemptError,
    MeasurementIntegrityError,
    TrainConfig,
    _build_generator,
    _episode_target_roster,
    _backend_setup_kwargs,
    _generalized_setup_kwargs,
    _git_provenance,
    _observe_world_identity,
    _require_scheduled_cell,
    _scheduled_cell,
    build_variation_config,
)

# The build report's own schema. SEPARATE from the manifest schema on purpose: the
# manifest is the frozen POPULATION and the report is the AUDIT TRAIL of how that
# population was selected, including every candidate that was rejected. Rejected
# candidates are evidence; they are not benchmark members and never appear in a manifest.
PREFLIGHT_SCHEMA: str = "generalized_v1_benchmark_preflight_report"
PREFLIGHT_SCHEMA_VERSION: int = 1

# The selection policy this module implements, recorded verbatim so a report says which
# rule produced it rather than leaving it to be inferred from the numbers.
PREFLIGHT_POLICY: str = "deterministic_per_cell_window_v1"

_REPORT_FILENAME: str = "benchmark_preflight_report.json"
_MANIFEST_FILENAME: str = "benchmark_manifest.json"

# THE TWO OUTCOMES A BUILD REPORT CAN DESCRIBE, stated in the report itself rather than
# left to be inferred from whether `manifest` happens to be null. A failed preflight
# writes a report too -- its spent candidate seeds and rejection reasons are the evidence
# the failure exists to produce -- so "is this a frozen benchmark?" must be answerable
# from ONE unambiguous field and never from the shape of the document.
PREFLIGHT_STATUS_COMPLETE: str = "complete"
PREFLIGHT_STATUS_FAILED: str = "failed_incomplete"
PREFLIGHT_STATUSES: Tuple[str, ...] = (
    PREFLIGHT_STATUS_COMPLETE, PREFLIGHT_STATUS_FAILED,
)

# The stable slug for the ONE way a selection walk can fail after it has started.
PREFLIGHT_FAILURE_WINDOW_EXHAUSTED: str = "candidate_window_exhausted"

# The closed set of candidate outcomes. `accepted` is not a rejection; it is listed here
# so a reader has both slugs in one place.
CANDIDATE_ACCEPTED: str = "accepted"
CANDIDATE_REJECTED: str = "rejected"
CANDIDATE_OUTCOMES: Tuple[str, ...] = (CANDIDATE_ACCEPTED, CANDIDATE_REJECTED)


class BenchmarkPreflightError(RuntimeError):
    """A preflight could not produce a COMPLETE population (or was asked for a bad one).

    NOT an ordinary rejection: a cell's candidate window is the operator's explicit
    bound, so exhausting it means the requested scale cannot be met at this attrition
    rate. It ABORTS and writes NO MANIFEST -- a benchmark missing a stratum member is not
    this benchmark, and a partial manifest that presented itself as complete would be
    worse than none.

    THE AUDIT IS NOT ABORTED WITH IT. Every candidate the walk already attempted has SPENT
    its seed, and those seeds and their rejection reasons are exactly the evidence an
    operator needs to decide whether to raise ``max_candidates_per_cell``, lower
    ``worlds_per_cell``, or investigate the attrition itself. Throwing that away because
    the last cell failed would make a failed preflight unreadable and would silently
    invite re-running the same window blind. So a failure carries its report:

      * ``report`` -- the build report as a dict, ALWAYS present for an exhaustion
        failure, so an in-memory caller (``output_dir=None``) has the full audit even
        though no file could be written;
      * ``report_path`` -- where that report was written, or ``None`` when the caller
        supplied no ``output_dir``.

    Both are ``None`` for the input-validation raises (a bad scale, a non-generalized
    config, missing provenance), which fail before any candidate is attempted and
    therefore have no audit to carry.
    """

    def __init__(
        self,
        message: str,
        *,
        report: Optional[Dict[str, Any]] = None,
        report_path: Optional[Path] = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.report_path = report_path


# =============================================================================
# 1. What one candidate produced
# =============================================================================

@dataclass(frozen=True)
class CandidateOutcome:
    """The durable record of ONE attempted candidate world.

    Recorded for accepted AND rejected candidates alike, because the rejected ones are
    exactly what makes the accepted population auditable: without them, "the A=2/LOW
    worlds are seeds 800001 and 800003" is a claim with no way to check why 800000 and
    800002 are missing.

    ``reason`` is a STABLE SLUG where the raising layer publishes one (today
    :data:`NO_FD_ELIGIBLE_EGO`, and a reference fault's own reason);
    ``detail_reasons`` carries the per-candidate FD eligibility slugs the walk reported.
    ``message`` is the human text and is never parsed for routing.
    """

    base_cell_key: str
    agent_count: int
    load_bucket: str
    candidate_ordinal: int
    seed: int
    outcome: str
    seconds: float
    world_ordinal: Optional[int] = None
    hidden_requested: int = 0
    hidden_realized: Optional[int] = None
    known_realized: Optional[int] = None
    fd_selected_ordinal: Optional[int] = None
    fd_certificate_fingerprint: Optional[str] = None
    construction_audit: Optional[Dict[str, Any]] = None
    pipeline_stage: Optional[str] = None
    error_type: Optional[str] = None
    reason: Optional[str] = None
    detail_reasons: Tuple[str, ...] = ()
    message: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.outcome == CANDIDATE_ACCEPTED

    def to_record(self) -> Dict[str, Any]:
        return {
            "base_cell": self.base_cell_key,
            "agent_count": int(self.agent_count),
            "load_bucket": str(self.load_bucket),
            "candidate_ordinal": int(self.candidate_ordinal),
            "seed": int(self.seed),
            "outcome": str(self.outcome),
            "world_ordinal": self.world_ordinal,
            "hidden_requested": int(self.hidden_requested),
            "hidden_realized": self.hidden_realized,
            "known_realized": self.known_realized,
            # Recorded, never judged: a short realization is a legitimate accepted world,
            # and the decision about the resulting distribution is a human one.
            "hidden_short_realized": (
                None if self.hidden_realized is None
                else bool(int(self.hidden_realized) < int(self.hidden_requested))
            ),
            "fd_selected_ordinal": self.fd_selected_ordinal,
            "fd_certificate_fingerprint": self.fd_certificate_fingerprint,
            "construction_audit": self.construction_audit,
            "pipeline_stage": self.pipeline_stage,
            "error_type": self.error_type,
            "reason": self.reason,
            "detail_reasons": list(self.detail_reasons),
            "message": self.message,
            "seconds": float(self.seconds),
        }


@dataclass(frozen=True)
class CellWindow:
    """One base cell's INDEPENDENT half-open candidate-seed window."""

    ordinal: int
    agent_count: int
    load_bucket: str
    start: int
    stop: int

    @property
    def key(self) -> str:
        return base_cell_key(self.agent_count, self.load_bucket)

    @property
    def hidden_requested(self) -> int:
        return hidden_requested_for(self.agent_count, self.load_bucket)

    def seeds(self) -> Tuple[int, ...]:
        return tuple(range(int(self.start), int(self.stop)))

    def to_record(self) -> Dict[str, Any]:
        return {
            "base_cell": self.key,
            "base_cell_ordinal": int(self.ordinal),
            "agent_count": int(self.agent_count),
            "load_bucket": str(self.load_bucket),
            "hidden_requested": self.hidden_requested,
            "known_requested": int(self.agent_count),
            "candidate_window": {
                "start": int(self.start),
                "stop": int(self.stop),
                "half_open": True,
                "size": int(self.stop) - int(self.start),
            },
        }


@dataclass
class PreflightResult:
    """Everything one preflight produced: the frozen population and its audit trail."""

    manifest: BenchmarkManifest
    manifest_path: Optional[Path]
    report: Dict[str, Any]
    report_path: Optional[Path]
    candidates: List[CandidateOutcome] = field(default_factory=list)


def cell_windows(
    *, benchmark_base_seed: int, max_candidates_per_cell: int
) -> Tuple[CellWindow, ...]:
    """The six INDEPENDENT candidate windows, in canonical base-cell order.

    Independence is the whole point: cell ``c`` owns ``[base + c*M, base + (c+1)*M)`` and
    scans only inside it, so however many candidates cell ``c`` rejects, cell ``c+1``
    starts where it always would. A single shared stream would make every cell's accepted
    seeds a function of every earlier cell's attrition, and a preflight re-run at a
    different scale would silently re-select worlds nobody changed.
    """
    base = int(benchmark_base_seed)
    width = int(max_candidates_per_cell)
    if base < 0:
        raise BenchmarkPreflightError(
            "benchmark_base_seed must be >= 0, got %r" % (benchmark_base_seed,)
        )
    if width < 1:
        raise BenchmarkPreflightError(
            "max_candidates_per_cell must be >= 1, got %r" % (max_candidates_per_cell,)
        )
    return tuple(
        CellWindow(
            ordinal=c,
            agent_count=int(agent_count),
            load_bucket=str(bucket),
            start=base + c * width,
            stop=base + (c + 1) * width,
        )
        for c, (agent_count, bucket) in enumerate(BENCHMARK_BASE_CELLS)
    )


# =============================================================================
# 2. Probing ONE candidate world
# =============================================================================

def _rejection_reason(original: BaseException) -> Tuple[Optional[str], Tuple[str, ...]]:
    """The STABLE slug of a rejection, plus whatever detail slugs it published.

    :data:`NO_FD_ELIGIBLE_EGO` is documented as a stable machine-readable MARKER carried
    in the exception's text (``graph_fuel_damage``), and the per-candidate reasons the
    walk reports are drawn from the closed
    :data:`FD_ELIGIBILITY_REJECTION_REASONS` set -- so both are recognized by matching
    those PUBLISHED CONSTANTS, never by parsing free prose. Anything unrecognized yields
    ``None``, which is the truthful "this layer published no slug" rather than a guess.
    """
    reason = getattr(original, "reason", None)
    if isinstance(original, ReferenceIntegrityError) and reason:
        return str(reason), ()
    text = str(original)
    primary = NO_FD_ELIGIBLE_EGO if NO_FD_ELIGIBLE_EGO in text else None
    details = tuple(slug for slug in FD_ELIGIBILITY_REJECTION_REASONS if slug in text)
    return primary, details


def probe_world(
    cfg: TrainConfig,
    gen: Any,
    *,
    seed: int,
    agent_count: int,
    load_bucket: str,
) -> WorldPreflight:
    """CONSTRUCT one candidate world and certify it, or RAISE.

    Exactly the pipeline prefix ``graph_train._run_one_episode`` runs -- the same reseed,
    the same ``build_variation_config``, the same ``setup_episode`` call with the same
    generalized policy keywords, the same roster and scheduled-cell checks -- and then
    ONE further step: the fuel-damage plan is built for ALL THREE benchmark members
    (clean, mild, severe). That is what "this world is a benchmark WORLD" means: a group
    is matched only if every member can be planned, and the certified eligibility walk
    depends on the episode seed alone, so all three must certify the SAME ego and produce
    the SAME id-free identity. They are compared here rather than assumed, and a
    disagreement is a :class:`BenchmarkIdentityError` -- an instrument fault, never a
    replacement-eligible rejection.

    NO POLICY IS BUILT AND NO EPISODE IS RUN. Nothing about reward, return or behaviour
    is computed, so no outcome can influence which worlds the benchmark holds.

    Returns the frozen :class:`WorldPreflight` the manifest will carry.
    """
    card = EpisodeCardinality(
        agent_count=int(agent_count),
        known_count=int(agent_count),                     # K == A
        hidden_requested=hidden_requested_for(agent_count, load_bucket),
        source=CARDINALITY_SOURCE_BENCHMARK,
    )
    # The SAME reseed `_run_one_episode` performs at the episode head, so this probe
    # builds bit-for-bit the world an evaluation member will build from the same seed.
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    try:
        var = build_variation_config(cfg, int(seed), cardinality=card)
        scenario_path = gen.generate(episode=int(seed), config=var)
    except Exception as exc:
        raise EpisodeAttemptError("generation", exc) from exc

    ctx = None
    try:
        try:
            ctx = setup_episode(
                Path(scenario_path).read_text(encoding="utf-8"),
                n_hidden=int(card.hidden_requested),
                placement_rng=random.Random(int(seed)),
                **_generalized_setup_kwargs(cfg),
                # THE SAME MATCH-AOU objective the later training / evaluation run will
                # use, and NOT a separately chosen one. The backend decides `A_init`, and
                # `A_init` is what route-relative hidden placement predicts routes from,
                # so a population selected under one objective and evaluated under
                # another would fail `require_world_matches_manifest` on its own frozen
                # geometry. Omitted entirely on the historical backend, exactly as in
                # `_run_one_episode`.
                **_backend_setup_kwargs(cfg),
            )
        except MatchAouBackendError:
            # BACKEND / CONFIGURATION fault: never a replacement-eligible rejection, for
            # the same reason an integrity fault is not one. Re-raised ahead of the broad
            # wrap so it propagates out of `_scan_cell` and stops the preflight instead of
            # quietly discarding a candidate the instrument could not evaluate.
            raise
        except Exception as exc:
            raise EpisodeAttemptError("setup", exc) from exc

        # The roster and the scheduled-cell check are MEASUREMENT structure, not an
        # episode outcome: they raise `EpisodeRosterError`, which is never wrapped and
        # never replacement-eligible.
        roster = _episode_target_roster(ctx)
        scheduled = _scheduled_cell(card, getattr(ctx, "construction_audit", None))
        _require_scheduled_cell(roster, scheduled)

        identities: Dict[str, Any] = {}
        for cell, mode in BENCHMARK_MEMBERS:
            try:
                controller = build_fuel_damage_controller(
                    ctx, episode_seed=int(seed),
                    params=cfg.fuel_damage_parameters(mode),
                )
            except FuelDamageIntegrityError:
                raise
            except Exception as exc:
                raise EpisodeAttemptError("setup", exc) from exc
            identities[cell] = _observe_world_identity(
                ctx, roster=roster, fd_plan_record=controller.plan.to_record()
            )

        cells = list(identities)
        reference_cell = cells[0]
        for cell in cells[1:]:
            wrong = identity_differences(identities[reference_cell], identities[cell])
            if wrong:
                raise BenchmarkIdentityError(
                    "candidate seed %d: benchmark members %r and %r did not describe "
                    "the same world (%s). A matched group whose members build different "
                    "worlds cannot be a benchmark world, and this is an instrument "
                    "contradiction rather than a rejection: the certified walk depends "
                    "on the episode seed alone, so the three members must agree."
                    % (int(seed), reference_cell, cell, "; ".join(wrong))
                )

        identity = identities[reference_cell]
        audit = getattr(ctx, "construction_audit", None)
        return WorldPreflight(
            hidden_realized=int(identity.hidden_realized),
            known_realized=int(identity.known_realized),
            geometric_fingerprint=identity.geometric_fingerprint,
            fd_selected_ordinal=identity.fd_selected_ordinal,
            fd_certificate_fingerprint=identity.fd_certificate_fingerprint,
            construction_audit=(None if audit is None else audit.as_dict()),
        )
    finally:
        if ctx is not None:
            try:
                ctx.env.close()
            except Exception:
                pass


# The default probe, injectable so the deterministic SELECTION logic can be tested
# without BLADE or BONMIN. The signature above is the contract; an injected form takes
# the same keywords.
ProbeFn = Callable[..., WorldPreflight]


# =============================================================================
# 3. The bounded, deterministic walk
# =============================================================================

def _require_preflight_config(cfg: TrainConfig) -> None:
    """The config must describe a GENERALIZED world, or refuse before any compute.

    Deliberately NARROWER than ``TrainConfig.validate``: a preflight TRAINS NOTHING, so
    it neither reads nor validates the training schedule (``n_iterations``,
    ``episodes_per_iteration``, the attempt budget, the eval band, the benchmark path).
    Requiring those would make an operator invent a training schedule in order to build a
    population, and a schedule invented to satisfy a check is exactly the kind of
    plausible-but-meaningless record a run artifact exists to prevent.

    What IS checked is everything the probe actually uses.
    """
    if not cfg.generalized:
        raise BenchmarkPreflightError(
            "benchmark preflight requires episode_design=%r, got %r: the 18-stratum "
            "benchmark and the certified eligibility contract are defined for that "
            "design only."
            % (EPISODE_DESIGN_GENERALIZED_V1, cfg.episode_design)
        )
    # WHICH MATCH-AOU objective the candidate worlds will be built under. Refused here,
    # before any compute, for the same reason the design is: a population selected under
    # an objective nobody named is a population nobody chose. It is NOT constrained to a
    # particular backend -- either is legal -- only to a KNOWN one.
    resolve_match_aou_backend(cfg.match_aou_backend)
    if bool(cfg.include_sams):
        raise BenchmarkPreflightError(
            "include_sams=True is not supported on the construction path: hidden "
            "targets are patched in as enemy AIRBASES and setup_episode refuses a mixed "
            "world."
        )
    if float(cfg.min_target_distance_km) <= 0.0:
        raise BenchmarkPreflightError(
            "min_target_distance_km must be > 0, got %r" % (cfg.min_target_distance_km,)
        )
    if float(cfg.min_known_separation_km) < 0.0:
        raise BenchmarkPreflightError(
            "min_known_separation_km must be >= 0, got %r"
            % (cfg.min_known_separation_km,)
        )
    # The FD parameter object owns its own verdicts, so the trainer, the rollout harness
    # and this module cannot disagree about what is legal.
    cfg.fuel_damage_parameters().validate()


def _rejected(
    window: CellWindow, candidate_ordinal: int, seed: int,
    exc: BaseException, seconds: float,
) -> CandidateOutcome:
    """One rejected candidate, recorded once and never retried."""
    original = getattr(exc, "original", exc)
    reason, details = _rejection_reason(original)
    return CandidateOutcome(
        base_cell_key=window.key,
        agent_count=int(window.agent_count),
        load_bucket=str(window.load_bucket),
        candidate_ordinal=int(candidate_ordinal),
        seed=int(seed),
        outcome=CANDIDATE_REJECTED,
        seconds=float(seconds),
        hidden_requested=window.hidden_requested,
        pipeline_stage=str(getattr(exc, "stage", "unknown")),
        error_type=type(original).__name__,
        reason=reason,
        detail_reasons=details,
        message=str(original),
    )


def _scan_cell(
    cfg: TrainConfig,
    gen: Any,
    window: CellWindow,
    *,
    worlds_per_cell: int,
    probe: ProbeFn,
) -> Tuple[List[Tuple[int, WorldPreflight]], List[CandidateOutcome]]:
    """Scan ONE cell's window and accept the first ``worlds_per_cell`` valid worlds.

    Candidates are attempted in ascending seed order, each EXACTLY ONCE: a rejected seed
    is spent and never revisited, and the next seed replaces it. The walk stops at the
    quota; the remaining seeds in the window are simply never attempted, which is what
    makes a smaller ``worlds_per_cell`` a strict PREFIX of a larger one.

    IT DOES NOT DECIDE WHETHER THE CELL SUCCEEDED. It returns its COMPLETE audit --
    every candidate it attempted, accepted and rejected alike -- and the caller judges
    the quota. That split is the point: the exhaustion verdict used to be raised from
    here, which discarded the outcomes on the way out and left a failed preflight with
    nothing to inspect but an exception message. Selection logic lives here; the verdict
    and the report live in :func:`run_benchmark_preflight`, and neither is duplicated.
    """
    accepted: List[Tuple[int, WorldPreflight]] = []
    outcomes: List[CandidateOutcome] = []
    for candidate_ordinal, seed in enumerate(window.seeds()):
        if len(accepted) >= int(worlds_per_cell):
            break
        t0 = time.perf_counter()
        try:
            preflight = probe(
                cfg, gen,
                seed=int(seed),
                agent_count=int(window.agent_count),
                load_bucket=str(window.load_bucket),
            )
        except (MeasurementIntegrityError, FuelDamageIntegrityError,
                BenchmarkIdentityError, MatchAouBackendError):
            # INSTRUMENT faults. Never replacement-eligible, for the same reason they
            # abort a training run: a world that contradicts its own certificate, or a
            # roster that misreads the world, implicates every world this preflight
            # touched. Replacing it would freeze a population selected by a defect.
            raise
        except ReferenceIntegrityError as exc:
            # The reference layer states WHY it refused, as a stable slug, and the
            # routing reads that slug and NOTHING ELSE. An instrument contradiction
            # aborts; a solver that was ASKED and did not ANSWER is a fact about this one
            # candidate and is an ordinary rejection.
            if reference_fault_aborts(exc):
                raise
            outcomes.append(_rejected(window, candidate_ordinal, seed, exc,
                                      time.perf_counter() - t0))
            continue
        except Exception as exc:
            outcomes.append(_rejected(window, candidate_ordinal, seed, exc,
                                      time.perf_counter() - t0))
            continue

        world_ordinal = len(accepted)
        accepted.append((int(seed), preflight))
        outcomes.append(CandidateOutcome(
            base_cell_key=window.key,
            agent_count=int(window.agent_count),
            load_bucket=str(window.load_bucket),
            candidate_ordinal=int(candidate_ordinal),
            seed=int(seed),
            outcome=CANDIDATE_ACCEPTED,
            seconds=time.perf_counter() - t0,
            world_ordinal=int(world_ordinal),
            hidden_requested=window.hidden_requested,
            hidden_realized=int(preflight.hidden_realized),
            known_realized=int(preflight.known_realized),
            fd_selected_ordinal=preflight.fd_selected_ordinal,
            fd_certificate_fingerprint=preflight.fd_certificate_fingerprint,
            construction_audit=preflight.construction_audit,
        ))

    return accepted, outcomes


# =============================================================================
# 4. The entry point
# =============================================================================

def run_benchmark_preflight(
    cfg: TrainConfig,
    *,
    worlds_per_cell: int,
    benchmark_base_seed: int,
    max_candidates_per_cell: int,
    output_dir: Optional[Path] = None,
    manifest_name: str = _MANIFEST_FILENAME,
    report_name: str = _REPORT_FILENAME,
    probe: Optional[ProbeFn] = None,
    generator: Optional[Any] = None,
    label: Optional[str] = None,
    notes: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> PreflightResult:
    """Select a COMPLETE benchmark population deterministically, then freeze it.

    Every scale input is REQUIRED and none is defaulted: the scientific scale is a later
    decision that owns bounded runtime validation first, and a default here would make it
    silently. ``probe`` and ``generator`` are injection seams for tests; production
    leaves both ``None`` and gets the real BLADE + BONMIN construction path.

    ``output_dir`` omitted writes nothing and returns the manifest and report in memory --
    which is what makes "the same inputs produce the same population" checkable without
    touching a filesystem.

    Raises:
        BenchmarkPreflightError: a bad scale, a non-generalized config, incomplete Git
            provenance, or a cell whose candidate window was exhausted before its quota
            was filled. ONLY the exhaustion case has an audit to carry, and it carries
            it: the FAILED build report is written to ``output_dir`` (when one was given)
            BEFORE the raise, and is attached to the exception as ``report`` /
            ``report_path`` either way. No manifest is written and no later cell is
            scanned. The input-validation cases fail before any candidate is attempted
            and leave both attributes ``None``.
        MeasurementIntegrityError / FuelDamageIntegrityError / BenchmarkIdentityError:
            an instrument fault -- never replaced, never recorded as a rejection.
    """
    _require_preflight_config(cfg)
    if int(worlds_per_cell) < 1:
        raise BenchmarkPreflightError(
            "worlds_per_cell must be >= 1, got %r. The scientific scale is never "
            "defaulted: how many worlds a benchmark needs is decided by bounded runtime "
            "validation, not by this module." % (worlds_per_cell,)
        )
    windows = cell_windows(
        benchmark_base_seed=int(benchmark_base_seed),
        max_candidates_per_cell=int(max_candidates_per_cell),
    )
    if int(max_candidates_per_cell) < int(worlds_per_cell):
        raise BenchmarkPreflightError(
            "max_candidates_per_cell (%d) must be >= worlds_per_cell (%d): a window "
            "smaller than the quota could never fill a cell even with no rejections."
            % (int(max_candidates_per_cell), int(worlds_per_cell))
        )

    t_run = time.perf_counter()
    out_dir = None if output_dir is None else Path(output_dir)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # PROVENANCE IS A PRECONDITION, exactly as it is for a training run: the benchmark
    # population is the comparator two campaign arms are measured against, so it must be
    # attributable to an exact code state. A DIRTY tree is a hazard and warns; INCOMPLETE
    # provenance refuses before anything is built.
    git_info = dict(provenance if provenance is not None
                    else _git_provenance(_REPO_ROOT))
    if not git_info.get("available"):
        raise BenchmarkPreflightError(
            "provenance: complete Git provenance is UNAVAILABLE (%s). The benchmark "
            "population is the comparator two campaign arms are measured against, so it "
            "must be attributable to an exact code state. Nothing was built."
            % git_info.get("reason")
        )
    if git_info.get("dirty"):
        print("[WARN] provenance: the working tree is DIRTY at %s (%s uncommitted "
              "path(s)). The exact code that selected this population exists only on "
              "this machine."
              % (git_info.get("commit"), git_info.get("dirty_path_count")))

    gen = generator
    if gen is None:
        scen_dir = (Path(out_dir) if out_dir is not None else Path(".")) / "scenarios"
        scen_dir.mkdir(parents=True, exist_ok=True)
        gen = _build_generator(scen_dir)
    probe_fn: ProbeFn = probe_world if probe is None else probe

    all_candidates: List[CandidateOutcome] = []
    cell_blocks: List[Dict[str, Any]] = []
    world_entries: List[Dict[str, Any]] = []

    for window in windows:
        print("[preflight] %s: window [%d, %d), need %d world(s)"
              % (window.key, window.start, window.stop, int(worlds_per_cell)))
        accepted, outcomes = _scan_cell(
            cfg, gen, window,
            worlds_per_cell=int(worlds_per_cell), probe=probe_fn,
        )
        all_candidates.extend(outcomes)
        for world_ordinal, (seed, preflight) in enumerate(accepted):
            world_entries.append({
                "agent_count": int(window.agent_count),
                "load_bucket": str(window.load_bucket),
                "world_ordinal": int(world_ordinal),
                "seed": int(seed),
                "preflight": preflight.to_record(),
            })
        block = _cell_block(window, accepted, outcomes,
                            worlds_per_cell=int(worlds_per_cell))
        cell_blocks.append(block)
        print("           accepted %d/%d after %d candidate(s): seeds %s"
              % (block["n_accepted"], int(worlds_per_cell),
                 block["n_candidates_attempted"], block["accepted_seeds"]))

        # THE QUOTA VERDICT, taken HERE rather than inside the walk, so the cell's
        # COMPLETE audit is already in `cell_blocks` and `all_candidates` before the
        # decision is made. A failed preflight writes NO MANIFEST -- but its spent
        # candidate seeds and their rejection reasons ARE the evidence it exists to
        # produce, and raising from inside the walk discarded them on the way out,
        # leaving an operator nothing to inspect but an exception message that pointed
        # at a build report which had never been written.
        if len(accepted) < int(worlds_per_cell):
            failure = _failure_block(
                window, accepted, outcomes,
                worlds_per_cell=int(worlds_per_cell), windows=windows,
            )
            report = _build_report(
                cfg,
                status=PREFLIGHT_STATUS_FAILED,
                git_info=git_info, windows=windows,
                worlds_per_cell=int(worlds_per_cell),
                benchmark_base_seed=int(benchmark_base_seed),
                max_candidates_per_cell=int(max_candidates_per_cell),
                cell_blocks=cell_blocks, all_candidates=all_candidates,
                world_entries=world_entries,
                manifest=None, manifest_path=None,
                stale_manifest_path=_existing_manifest(out_dir, manifest_name),
                failure=failure, seconds=time.perf_counter() - t_run,
            )
            report_path = _write_report(out_dir, report_name, report)
            raise BenchmarkPreflightError(
                "base cell %s exhausted its candidate window [%d, %d): %d candidate(s) "
                "attempted, only %d of the %d requested world(s) accepted. NO manifest "
                "is written and NO later cell is scanned -- a benchmark missing a "
                "stratum member is not this benchmark, and a partial manifest that read "
                "as complete would be worse than none. The spent candidate seeds and "
                "their rejection reasons are preserved %s. Raise "
                "max_candidates_per_cell, lower worlds_per_cell, or investigate the "
                "attrition; re-running the same window cannot give a different answer."
                % (window.key, window.start, window.stop, len(outcomes),
                   len(accepted), int(worlds_per_cell),
                   ("in %s" % report_path) if report_path is not None
                   else "on this exception's `report` attribute (no output_dir was "
                        "supplied, so no file could be written)"),
                report=report, report_path=report_path,
            )

    manifest = build_benchmark_manifest(worlds=world_entries, label=label, notes=notes)

    manifest_path = None
    if out_dir is not None:
        manifest_path = write_benchmark_manifest(manifest, out_dir / manifest_name)

    report = _build_report(
        cfg,
        status=PREFLIGHT_STATUS_COMPLETE,
        git_info=git_info, windows=windows,
        worlds_per_cell=int(worlds_per_cell),
        benchmark_base_seed=int(benchmark_base_seed),
        max_candidates_per_cell=int(max_candidates_per_cell),
        cell_blocks=cell_blocks, all_candidates=all_candidates,
        world_entries=world_entries,
        manifest=manifest, manifest_path=manifest_path,
        stale_manifest_path=None, failure=None,
        seconds=time.perf_counter() - t_run,
    )
    report_path = _write_report(out_dir, report_name, report)

    return PreflightResult(
        manifest=manifest,
        manifest_path=manifest_path,
        report=report,
        report_path=report_path,
        candidates=all_candidates,
    )


def _cell_block(
    window: CellWindow,
    accepted: Sequence[Tuple[int, WorldPreflight]],
    outcomes: Sequence[CandidateOutcome],
    *,
    worlds_per_cell: int,
) -> Dict[str, Any]:
    """ONE base cell's audit block -- identical on the successful and failed paths.

    ``worlds_missing`` and ``window_exhausted`` are carried on BOTH paths (``0`` and
    ``False`` on a cell that filled), so the failing cell is identifiable from the cell
    list itself rather than only by cross-referencing the failure block.
    """
    block = window.to_record()
    n_accepted = sum(1 for o in outcomes if o.accepted)
    block.update({
        "worlds_requested": int(worlds_per_cell),
        "n_candidates_attempted": len(outcomes),
        "n_accepted": n_accepted,
        "n_rejected": sum(1 for o in outcomes if not o.accepted),
        "worlds_missing": max(int(worlds_per_cell) - n_accepted, 0),
        "window_exhausted": bool(n_accepted < int(worlds_per_cell)),
        "accepted_seeds": [int(seed) for seed, _p in accepted],
        "rejection_reasons": _tally(
            [o.reason or o.error_type or "unknown"
             for o in outcomes if not o.accepted]
        ),
        "rejection_detail_reasons": _tally(
            [slug for o in outcomes if not o.accepted for slug in o.detail_reasons]
        ),
        "hidden_realized": _tally(
            [str(o.hidden_realized) for o in outcomes if o.accepted]
        ),
        "candidates": [o.to_record() for o in outcomes],
    })
    return block


def _failure_block(
    window: CellWindow,
    accepted: Sequence[Tuple[int, WorldPreflight]],
    outcomes: Sequence[CandidateOutcome],
    *,
    worlds_per_cell: int,
    windows: Sequence[CellWindow],
) -> Dict[str, Any]:
    """WHY the preflight stopped, and exactly where.

    The cells that COMPLETED and the cells that were NEVER ATTEMPTED are listed
    separately: "this cell was not scanned" and "this cell was scanned and filled" are
    different facts, and a reader who could not tell them apart would not know whether a
    later cell's absence from the report meant it was skipped or that it failed too.
    """
    scanned_through = int(window.ordinal)
    return {
        "reason": PREFLIGHT_FAILURE_WINDOW_EXHAUSTED,
        "base_cell": window.key,
        "base_cell_ordinal": scanned_through,
        "agent_count": int(window.agent_count),
        "load_bucket": str(window.load_bucket),
        "hidden_requested": window.hidden_requested,
        "candidate_window": {
            "start": int(window.start),
            "stop": int(window.stop),
            "half_open": True,
            "size": int(window.stop) - int(window.start),
        },
        "worlds_requested": int(worlds_per_cell),
        "worlds_accepted": len(accepted),
        "worlds_missing": max(int(worlds_per_cell) - len(accepted), 0),
        "n_candidates_attempted": len(outcomes),
        "accepted_seeds": [int(seed) for seed, _p in accepted],
        "attempted_seeds": [int(o.seed) for o in outcomes],
        "rejection_reasons": _tally(
            [o.reason or o.error_type or "unknown"
             for o in outcomes if not o.accepted]
        ),
        "rejection_detail_reasons": _tally(
            [slug for o in outcomes if not o.accepted for slug in o.detail_reasons]
        ),
        "cells_completed": [w.key for w in windows if w.ordinal < scanned_through],
        "cells_not_attempted": [w.key for w in windows if w.ordinal > scanned_through],
        "manifest_written": False,
    }


def _existing_manifest(
    out_dir: Optional[Path], manifest_name: str
) -> Optional[str]:
    """A manifest file ALREADY sitting at the target path, if any.

    REPORTED, never deleted: this is a failure path, and removing a file it did not write
    would destroy an earlier run's artifact. Naming it is what stops a reader from finding
    a stale manifest beside a failure report and taking the two for one run.
    """
    if out_dir is None:
        return None
    candidate = Path(out_dir) / manifest_name
    return str(candidate) if candidate.exists() else None


def _write_report(
    out_dir: Optional[Path], report_name: str, report: Dict[str, Any]
) -> Optional[Path]:
    """Write the build report, or return ``None`` when there is nowhere to write it."""
    if out_dir is None:
        return None
    path = Path(out_dir) / report_name
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def _build_report(
    cfg: TrainConfig,
    *,
    status: str,
    git_info: Dict[str, Any],
    windows: Sequence[CellWindow],
    worlds_per_cell: int,
    benchmark_base_seed: int,
    max_candidates_per_cell: int,
    cell_blocks: List[Dict[str, Any]],
    all_candidates: List[CandidateOutcome],
    world_entries: List[Dict[str, Any]],
    manifest: Optional[BenchmarkManifest],
    manifest_path: Optional[Path],
    stale_manifest_path: Optional[str],
    failure: Optional[Dict[str, Any]],
    seconds: float,
) -> Dict[str, Any]:
    """ONE build-report construction site, for BOTH outcomes.

    Shared deliberately: a failure report assembled by a second, parallel writer would
    drift from the successful one exactly where it matters -- the candidate audit -- and
    the whole reason for writing one is that the audit is identical evidence either way.
    What differs is stated explicitly, in one place: ``status``, ``manifest_written``, the
    ``manifest`` block, and ``failure``.

    ``accepted_seeds`` on a FAILED report lists the worlds accepted before the walk
    stopped. THEY ARE NOT A BENCHMARK: no manifest was built from them, no ``manifest_id``
    exists, ``manifest`` is ``null``, and ``status`` says so.
    """
    if str(status) not in PREFLIGHT_STATUSES:
        raise BenchmarkPreflightError(
            "unknown preflight status %r; expected one of %r"
            % (status, list(PREFLIGHT_STATUSES))
        )
    complete = str(status) == PREFLIGHT_STATUS_COMPLETE
    return {
        "schema": PREFLIGHT_SCHEMA,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "policy": PREFLIGHT_POLICY,
        "design": EPISODE_DESIGN_GENERALIZED_V1,
        # THE FIRST THREE KEYS A READER NEEDS, and the ONE unambiguous answer to "is this
        # a frozen benchmark?". Never inferred from whether `manifest` happens to be
        # null: a failed preflight writes a report too, and a reader who had to deduce
        # its standing from the document's shape could mistake a partial candidate audit
        # for a population.
        "status": str(status),
        "complete": bool(complete),
        "manifest_written": bool(complete and manifest_path is not None),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {"git": git_info},
        "request": {
            "worlds_per_cell": int(worlds_per_cell),
            "benchmark_base_seed": int(benchmark_base_seed),
            "max_candidates_per_cell": int(max_candidates_per_cell),
            "n_base_cells": len(windows),
            "candidate_seed_span": {
                "start": int(benchmark_base_seed),
                "stop": (int(benchmark_base_seed)
                         + len(windows) * int(max_candidates_per_cell)),
                "half_open": True,
            },
        },
        # WHAT THE WORLDS WERE BUILT WITH -- so a report can be checked against the
        # manifest it produced without re-deriving the construction inputs.
        "episode_design": cfg.design.to_record(),
        # WHICH allocation objective SELECTED these worlds. Recorded on the REPORT rather
        # than inside the manifest, deliberately: the manifest already carries each
        # world's id-free frozen identity, and `require_world_matches_manifest` REFUSES a
        # member whose reconstructed geometry disagrees -- so a manifest frozen under one
        # backend cannot be silently reused under the other, without a schema change.
        # This key is what lets a reader see WHY, instead of only that it happened.
        "match_aou_backend": resolve_match_aou_backend(cfg.match_aou_backend),
        "fuel_damage": cfg.fuel_damage_parameters().to_record(),
        "geometry": {
            "min_target_distance_km": float(cfg.min_target_distance_km),
            "min_known_separation_km": float(cfg.min_known_separation_km),
            "stretch_target_ratio": float(cfg.stretch_target_ratio),
            "include_sams": bool(cfg.include_sams),
            "randomize_red_airbase_positions":
                bool(cfg.randomize_red_airbase_positions),
        },
        "cells": cell_blocks,
        "totals": {
            "n_candidates_attempted": len(all_candidates),
            "n_accepted": sum(1 for o in all_candidates if o.accepted),
            "n_rejected": sum(1 for o in all_candidates if not o.accepted),
            "rejection_reasons": _tally(
                [o.reason or o.error_type or "unknown"
                 for o in all_candidates if not o.accepted]
            ),
            "rejection_detail_reasons": _tally(
                [slug for o in all_candidates if not o.accepted
                 for slug in o.detail_reasons]
            ),
            # REPORTED, never judged. Whether the accepted distribution is acceptable is
            # a human / GPT scientific-review decision taken before any measurement, so
            # there is deliberately no threshold and no verdict key here.
            "hidden_requested_vs_realized": _tally(
                ["%d->%d" % (o.hidden_requested, o.hidden_realized)
                 for o in all_candidates if o.accepted]
            ),
        },
        # On a FAILED report these are the worlds accepted before the walk stopped --
        # spent candidate seeds, not a population. `status` and a `null` manifest say so.
        "accepted_seeds": [int(w["seed"]) for w in world_entries],
        # `null` on a failure, so nothing in the document can be read as a frozen
        # benchmark: there is no `manifest_id` to quote and no file hash to check.
        "manifest": (
            None if manifest is None else {
                "manifest_id": manifest.manifest_id,
                "n_worlds": manifest.n_worlds,
                "n_members": manifest.n_members,
                "seed_list_sha256": manifest.seed_digest(),
                "path": None if manifest_path is None else str(manifest_path),
                "file_sha256": (
                    None if manifest_path is None else _file_sha256(manifest_path)
                ),
            }
        ),
        # A manifest file this preflight did NOT write, already sitting at the target
        # path from an earlier run. Named rather than deleted (see `_existing_manifest`),
        # so a stale artifact beside a failure report cannot be taken for its output.
        "stale_manifest_path": stale_manifest_path,
        # `null` on the successful path -- the truthful statement that nothing failed,
        # rather than an absent key a reader has to interpret.
        "failure": failure,
        "seconds": float(seconds),
    }


def _tally(values: Sequence[str]) -> Dict[str, int]:
    """A sorted count of stable slugs -- ``{}`` when there is nothing to count."""
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =============================================================================
# 5. CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="graph_benchmark_preflight",
        description=(
            "Select a COMPLETE GENERALIZED-V1 benchmark population deterministically "
            "and freeze it as a content-addressed manifest. Tests world ELIGIBILITY "
            "only -- no policy is built, no episode is run, and no reward influences "
            "which worlds are accepted."
        ),
    )
    p.add_argument("--config", type=str, default=None,
                   help="optional JSON preset naming TrainConfig FIELDS (the same "
                        "loader graph_train uses); supplies the geometry and the "
                        "fuel-damage knobs the worlds are built with")
    p.add_argument("--worlds-per-cell", type=int, required=True,
                   help="matched world GROUPS to accept in each of the six (A, load) "
                        "base cells. REQUIRED -- the scientific scale is never "
                        "defaulted")
    p.add_argument("--benchmark-base-seed", type=int, required=True,
                   help="first candidate seed; cell c owns "
                        "[base + c*M, base + (c+1)*M). REQUIRED")
    p.add_argument("--max-candidates-per-cell", type=int, required=True,
                   help="M -- the bounded candidate window per base cell. Exhausting it "
                        "before a cell's quota is filled ABORTS and writes no manifest. "
                        "REQUIRED")
    p.add_argument("--out", type=str, required=True,
                   help="output directory for the manifest, the build report and the "
                        "generated candidate scenarios")
    p.add_argument("--match-aou-backend", type=str, choices=list(MATCH_AOU_BACKENDS),
                   default=None,
                   help="which MATCH-AOU allocation objective SELECTS these worlds. It "
                        "MUST be the one the later training / evaluation run uses: the "
                        "backend decides A_init, and A_init decides the route-relative "
                        "hidden geometry the manifest freezes. Omitted -> whatever the "
                        "--config preset says, else the historical %s"
                        % MATCH_AOU_BACKEND_LEGACY_MINLP_V1)
    p.add_argument("--manifest-name", type=str, default=_MANIFEST_FILENAME)
    p.add_argument("--report-name", type=str, default=_REPORT_FILENAME)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--notes", type=str, default=None)
    return p


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build the construction config, forcing the generalized design.

    A preset may set any ``TrainConfig`` field (the same loader ``graph_train`` uses), but
    the design and the training mixture are PINNED here: the 18-stratum benchmark is
    defined for ``generalized_v1`` alone, and a preflight run under any other design
    would build the wrong worlds while its report claimed this one.

    The training-schedule fields are pinned to inert values and are NEITHER read NOR
    recorded: a preflight trains nothing, and making an operator invent a schedule in
    order to build a population would put a meaningless number in a scientific artifact.
    """
    from .graph_fuel_damage import FuelDamageMode
    from .graph_train import load_config_file

    fields: Dict[str, Any] = {}
    if args.config:
        fields.update(load_config_file(args.config))
    fields.pop("ppo", None)
    fields.pop("ctde", None)
    fields["episode_design"] = EPISODE_DESIGN_GENERALIZED_V1
    fields["fuel_damage_mode"] = FuelDamageMode.SEEDED_VARIABLE
    # The backend is an operator choice, not a preflight constant: whichever objective the
    # later run will train and evaluate under must be the one that selects these worlds.
    # A preset may set it; the flag overrides it; the default stays historical.
    if args.match_aou_backend is not None:
        fields["match_aou_backend"] = str(args.match_aou_backend)
    fields.setdefault("n_iterations", 1)
    fields["eval_every"] = 0
    fields["eval_episodes"] = 0
    fields["benchmark_manifest"] = None
    fields["generalized_max_attempts_per_iteration"] = None
    fields["output_dir"] = str(args.out)
    return TrainConfig(**fields)


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)
    print("=" * 78)
    print("benchmark preflight: policy=%s design=%s backend=%s"
          % (PREFLIGHT_POLICY, cfg.episode_design, cfg.match_aou_backend))
    print("worlds_per_cell=%d benchmark_base_seed=%d max_candidates_per_cell=%d"
          % (args.worlds_per_cell, args.benchmark_base_seed,
             args.max_candidates_per_cell))
    print("=" * 78)
    result = run_benchmark_preflight(
        cfg,
        worlds_per_cell=int(args.worlds_per_cell),
        benchmark_base_seed=int(args.benchmark_base_seed),
        max_candidates_per_cell=int(args.max_candidates_per_cell),
        output_dir=Path(args.out),
        manifest_name=str(args.manifest_name),
        report_name=str(args.report_name),
        label=args.label,
        notes=args.notes,
    )
    print("=" * 78)
    print("status=%s  manifest_id=%s  worlds=%d  members=%d"
          % (result.report["status"], result.manifest.manifest_id,
             result.manifest.n_worlds, result.manifest.n_members))
    print("manifest: %s" % result.manifest_path)
    print("report  : %s" % result.report_path)
    print("candidates attempted=%d accepted=%d rejected=%d"
          % (result.report["totals"]["n_candidates_attempted"],
             result.report["totals"]["n_accepted"],
             result.report["totals"]["n_rejected"]))


if __name__ == "__main__":       # pragma: no cover - CLI
    main(sys.argv[1:])
