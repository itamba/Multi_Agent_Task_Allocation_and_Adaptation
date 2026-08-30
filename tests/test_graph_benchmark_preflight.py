"""GENERALIZED-V1 Task 5C -- the deterministic BENCHMARK PREFLIGHT (PO3).

SOLVER-FREE AND BLADE-FREE BY CONSTRUCTION. Every test here injects a stub ``probe`` and
a stub generator, because what is under test is the SELECTION LOGIC -- which candidate
seeds are attempted, in what order, which cell they belong to, what is recorded, and when
the walk aborts -- and none of that needs a world to be built. The real
``probe_world`` is exercised by the separate bounded engineering smoke under ``nlp_env``.

WHAT IT PROVES (task PO3):

  * each base cell owns its own INDEPENDENT half-open candidate window, so a rejection in
    one cell cannot shift another cell's accepted seed stream;
  * every rejected candidate is recorded ONCE, its seed is SPENT, and the NEXT candidate
    replaces it;
  * exactly ``worlds_per_cell`` worlds are accepted per cell, each carrying a frozen
    ``WorldPreflight``;
  * a manifest written from the preflight RELOADS to the same ``manifest_id``;
  * candidate-window exhaustion ABORTS with NO manifest, stops before any later cell,
    and STILL writes a report that says it failed and preserves every spent candidate
    seed and rejection reason (the review fix);
  * a fixed candidate seed reproduces the same preflight decision;
  * measurement-integrity faults are NEVER replacement-eligible;
  * the preflight population-selection policy lives HERE and not inside
    ``evaluate_benchmark``, which still performs no substitution.

Run: python -m pytest tests/test_graph_benchmark_preflight.py -v
     python tests/test_graph_benchmark_preflight.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.rl.training.graph_benchmark_preflight import (  # noqa: E402
    CANDIDATE_ACCEPTED,
    CANDIDATE_REJECTED,
    PREFLIGHT_FAILURE_WINDOW_EXHAUSTED,
    PREFLIGHT_POLICY,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS_COMPLETE,
    PREFLIGHT_STATUS_FAILED,
    BenchmarkPreflightError,
    cell_windows,
    run_benchmark_preflight,
)
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    NO_FD_ELIGIBLE_EGO,
    FuelDamageError,
    FuelDamageIntegrityError,
    FuelDamageMode,
)
from match_aou.rl.training.graph_generalized import (  # noqa: E402
    BENCHMARK_BASE_CELLS,
    BENCHMARK_GROUP_SIZE,
    EPISODE_DESIGN_FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1,
    BenchmarkIdentityError,
    WorldPreflight,
    base_cell_key,
    hidden_requested_for,
    load_benchmark_manifest,
)
from match_aou.rl.training.graph_train import (  # noqa: E402
    EpisodeAttemptError,
    EpisodeRosterError,
    TrainConfig,
)

# A COMPLETE git verdict, injected so these tests neither depend on the developer
# checkout's live state nor trip the incomplete-provenance gate by accident.
_FAKE_GIT_OK = {
    "available": True, "commit": "0" * 40, "branch": "test",
    "dirty": False, "dirty_path_count": 0, "reason": None,
}
_FAKE_GIT_INCOMPLETE = {
    "available": False, "commit": None, "branch": None,
    "dirty": None, "dirty_path_count": None, "reason": "git unavailable",
}


def _cfg(tmp_path: Path, **kwargs) -> TrainConfig:
    """A minimal GENERALIZED construction config.

    The training-schedule fields are irrelevant here and deliberately untouched: a
    preflight trains nothing, and `_require_preflight_config` does not read them.
    """
    base = dict(
        n_iterations=1,
        output_dir=tmp_path / "preflight",
        episode_design=EPISODE_DESIGN_GENERALIZED_V1,
        fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
    )
    base.update(kwargs)
    return TrainConfig(**base)


def _preflight_for(seed: int, agent_count: int, load_bucket: str) -> WorldPreflight:
    """A deterministic stand-in identity: a pure function of the candidate seed."""
    return WorldPreflight(
        hidden_realized=hidden_requested_for(agent_count, load_bucket),
        known_realized=int(agent_count),
        geometric_fingerprint=((float(seed % 90), float(seed % 180)),),
        fd_selected_ordinal=int(seed % max(1, agent_count)),
        fd_certificate_fingerprint="cert-%d" % int(seed),
        construction_audit={"policy": "bounded_backoff_v1", "seed": int(seed)},
    )


def _stub_probe(*, reject=(), integrity=(), fd_integrity=(), identity=(), calls=None,
                short_hidden=()):
    """A probe that accepts everything except the seeds it is told to fail.

    `reject` raises an ORDINARY, replacement-eligible rejection carrying the real
    `no_fd_eligible_ego` marker and a real eligibility detail slug -- the exact shape the
    Task-5A observation reported. The other three knobs raise the three INSTRUMENT faults,
    which must never be replaced.
    """
    rejected = set(int(s) for s in reject)
    integrity_seeds = set(int(s) for s in integrity)
    fd_seeds = set(int(s) for s in fd_integrity)
    identity_seeds = set(int(s) for s in identity)
    short = set(int(s) for s in short_hidden)
    seen = [] if calls is None else calls

    def probe(cfg, gen, *, seed, agent_count, load_bucket):
        seen.append(int(seed))
        if int(seed) in integrity_seeds:
            raise EpisodeRosterError("stub roster fault at seed %d" % int(seed))
        if int(seed) in fd_seeds:
            raise FuelDamageIntegrityError("stub certificate fault at %d" % int(seed))
        if int(seed) in identity_seeds:
            raise BenchmarkIdentityError("stub member disagreement at %d" % int(seed))
        if int(seed) in rejected:
            raise EpisodeAttemptError("setup", FuelDamageError(
                "%s: 2 candidate(s) considered, reasons: pre_event_popup_risk"
                % NO_FD_ELIGIBLE_EGO
            ))
        world = _preflight_for(int(seed), int(agent_count), str(load_bucket))
        if int(seed) in short:
            world = WorldPreflight(
                hidden_realized=1,
                known_realized=world.known_realized,
                geometric_fingerprint=world.geometric_fingerprint,
                fd_selected_ordinal=world.fd_selected_ordinal,
                fd_certificate_fingerprint=world.fd_certificate_fingerprint,
                construction_audit=world.construction_audit,
            )
        return world

    return probe, seen


def _run(tmp_path: Path, *, worlds_per_cell=2, base_seed=800_000,
         max_candidates=6, probe=None, output_dir=None, cfg=None,
         provenance=None):
    return run_benchmark_preflight(
        cfg or _cfg(tmp_path),
        worlds_per_cell=worlds_per_cell,
        benchmark_base_seed=base_seed,
        max_candidates_per_cell=max_candidates,
        output_dir=output_dir,
        probe=probe or _stub_probe()[0],
        generator=object(),                      # the stub probe never touches it
        provenance=provenance or _FAKE_GIT_OK,
    )


# =============================================================================
# PO3.1 -- independent per-cell windows
# =============================================================================

def test_po3_each_base_cell_owns_an_independent_candidate_window() -> None:
    """Cell ``c`` owns ``[base + c*M, base + (c+1)*M)`` -- disjoint, ordered, complete."""
    windows = cell_windows(benchmark_base_seed=800_000, max_candidates_per_cell=10)
    assert len(windows) == len(BENCHMARK_BASE_CELLS) == 6
    assert [w.key for w in windows] == [
        base_cell_key(a, b) for a, b in BENCHMARK_BASE_CELLS
    ]
    for c, w in enumerate(windows):
        assert w.start == 800_000 + c * 10
        assert w.stop == w.start + 10
        assert w.hidden_requested == hidden_requested_for(w.agent_count, w.load_bucket)
    # Pairwise disjoint: no seed can belong to two cells, so a candidate's cell is a
    # property of the seed alone.
    seen = set()
    for w in windows:
        span = set(w.seeds())
        assert not (span & seen)
        seen |= span


def test_po3_a_rejection_cannot_shift_another_cells_accepted_seeds(
    tmp_path: Path,
) -> None:
    """THE INDEPENDENCE CLAIM. Reject inside cell 0 only; cells 1..5 must not move.

    A single shared candidate stream would make every cell's accepted seeds a function of
    every earlier cell's attrition -- so re-running the preflight after one world stopped
    certifying would silently re-select worlds nobody changed.
    """
    clean = _run(tmp_path, worlds_per_cell=2, max_candidates=6)
    # Reject the first two candidates of the FIRST cell only (800000, 800001).
    probe, _calls = _stub_probe(reject=(800_000, 800_001))
    perturbed = _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe)

    def by_cell(result):
        out = {}
        for world in result.manifest.worlds:
            out.setdefault(world.base_cell_key, []).append(int(world.seed))
        return out

    a, b = by_cell(clean), by_cell(perturbed)
    first = base_cell_key(*BENCHMARK_BASE_CELLS[0])
    assert a[first] == [800_000, 800_001]
    assert b[first] == [800_002, 800_003], "the rejected seeds were not replaced in-cell"
    for key in a:
        if key == first:
            continue
        assert a[key] == b[key], (
            "cell %s moved because a DIFFERENT cell rejected candidates" % key
        )


# =============================================================================
# PO3.2 -- rejection is recorded, the seed is spent, the next candidate replaces it
# =============================================================================

def test_po3_a_rejected_candidate_is_recorded_once_and_never_retried(
    tmp_path: Path,
) -> None:
    """Recorded ONCE, spent, replaced by the NEXT seed -- with its stable reason slug."""
    calls: list = []
    probe, _ = _stub_probe(reject=(800_000, 800_002), calls=calls)
    result = _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe)

    first = base_cell_key(*BENCHMARK_BASE_CELLS[0])
    cell = [c for c in result.candidates if c.base_cell_key == first]
    # Four candidates attempted for two worlds: 800000 (X) 800001 (ok) 800002 (X)
    # 800003 (ok).
    assert [c.seed for c in cell] == [800_000, 800_001, 800_002, 800_003]
    assert [c.outcome for c in cell] == [
        CANDIDATE_REJECTED, CANDIDATE_ACCEPTED, CANDIDATE_REJECTED, CANDIDATE_ACCEPTED,
    ]
    # Each seed attempted EXACTLY once across the whole preflight.
    assert len(calls) == len(set(calls)), "a candidate seed was attempted twice"
    # The rejection carries the STABLE published slug, not parsed prose.
    bad = [c for c in cell if c.outcome == CANDIDATE_REJECTED]
    assert all(c.reason == NO_FD_ELIGIBLE_EGO for c in bad)
    assert all("pre_event_popup_risk" in c.detail_reasons for c in bad)
    assert all(c.pipeline_stage == "setup" for c in bad)
    # A rejected candidate is NEVER a benchmark member.
    assert 800_000 not in result.manifest.seeds()
    assert 800_002 not in result.manifest.seeds()
    # ... and the accepted ones took the ordinals in acceptance order.
    accepted = [c for c in cell if c.outcome == CANDIDATE_ACCEPTED]
    assert [c.world_ordinal for c in accepted] == [0, 1]


def test_po3_exactly_worlds_per_cell_are_accepted_and_carry_a_preflight(
    tmp_path: Path,
) -> None:
    """Every cell is filled to its quota, and every accepted world is FROZEN with its id."""
    result = _run(tmp_path, worlds_per_cell=3, max_candidates=8)
    manifest = result.manifest
    assert manifest.n_worlds == 3 * len(BENCHMARK_BASE_CELLS)
    assert manifest.n_members == manifest.n_worlds * BENCHMARK_GROUP_SIZE
    assert manifest.worlds_per_base_cell() == {
        base_cell_key(a, b): 3 for a, b in BENCHMARK_BASE_CELLS
    }
    for world in manifest.worlds:
        assert world.preflight is not None, world.key
        expected = _preflight_for(world.seed, world.agent_count, world.load_bucket)
        assert world.preflight.identity == expected.identity
        assert world.preflight.construction_audit == expected.construction_audit


def test_po3_a_short_realization_is_accepted_and_recorded_not_rejected(
    tmp_path: Path,
) -> None:
    """A contract-successful bounded-backoff world is NOT rejected for realizing fewer.

    The shortfall is a RECORDED outcome; whether the resulting distribution is acceptable
    is a human scientific-review decision, so nothing here applies a threshold.
    """
    high = [w for w in cell_windows(benchmark_base_seed=800_000,
                                    max_candidates_per_cell=6)
            if w.load_bucket == "high"]
    short_seed = high[0].start          # an A=2/HIGH world that realizes only 1 hidden
    probe, _ = _stub_probe(short_hidden=(short_seed,))
    result = _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe)

    assert short_seed in result.manifest.seeds(), "a short world was wrongly rejected"
    row = next(c for c in result.candidates if c.seed == short_seed)
    assert row.outcome == CANDIDATE_ACCEPTED
    assert row.to_record()["hidden_short_realized"] is True
    assert row.hidden_realized == 1 and row.hidden_requested == 2
    # REPORTED, never judged. Checked over the KEY names the preflight itself invents --
    # `fuel_damage` and `geometry` are verbatim copies of other layers' records, and
    # `leg_progress_threshold` there is a construction parameter, not a verdict.
    totals = result.report["totals"]
    assert "2->1" in totals["hidden_requested_vs_realized"]

    def _keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k).lower()
                for sub in _keys(v):
                    yield sub
        elif isinstance(node, list):
            for item in node:
                for sub in _keys(item):
                    yield sub

    for key in list(_keys(totals)) + list(_keys(result.report["cells"])):
        for verdict_word in ("degenerate", "collapse", "threshold", "verdict",
                             "benchmark_ok", "should_reject", "acceptable"):
            assert verdict_word not in key, (key, verdict_word)


# =============================================================================
# PO3.3 -- the frozen artifact
# =============================================================================

def test_po3_the_written_manifest_reloads_to_the_same_identity(tmp_path: Path) -> None:
    """A manifest written by the preflight verifies and reloads to the same id."""
    out = tmp_path / "out"
    result = _run(tmp_path, worlds_per_cell=2, max_candidates=6, output_dir=out)
    assert result.manifest_path is not None and result.manifest_path.exists()
    reloaded = load_benchmark_manifest(result.manifest_path)
    assert reloaded.manifest_id == result.manifest.manifest_id
    assert reloaded.seeds() == result.manifest.seeds()
    assert [w.preflight.identity for w in reloaded.worlds] == \
        [w.preflight.identity for w in result.manifest.worlds]


def test_po3_the_build_report_is_the_audit_trail(tmp_path: Path) -> None:
    """The report names the policy, the windows, EVERY candidate, and the manifest hash.

    Rejected candidates live HERE, never in the manifest: they are evidence about how the
    population was selected, and they are not benchmark members.
    """
    out = tmp_path / "out"
    probe, _ = _stub_probe(reject=(800_000,))
    result = _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe,
                  output_dir=out)
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert report["schema"] == PREFLIGHT_SCHEMA
    assert report["policy"] == PREFLIGHT_POLICY
    assert report["design"] == EPISODE_DESIGN_GENERALIZED_V1
    # THE SUCCESSFUL OUTCOME, pinned against the failed one: one field answers "is this a
    # frozen benchmark?", and a complete report carries no failure block.
    assert report["status"] == PREFLIGHT_STATUS_COMPLETE
    assert report["complete"] is True
    assert report["manifest_written"] is True
    assert report["failure"] is None
    assert report["stale_manifest_path"] is None
    assert all(c["window_exhausted"] is False for c in report["cells"])
    assert all(c["worlds_missing"] == 0 for c in report["cells"])
    assert report["provenance"]["git"]["commit"] == _FAKE_GIT_OK["commit"]
    assert report["request"] == {
        "worlds_per_cell": 2, "benchmark_base_seed": 800_000,
        "max_candidates_per_cell": 6, "n_base_cells": 6,
        "candidate_seed_span": {"start": 800_000, "stop": 800_036, "half_open": True},
    }
    assert [c["base_cell"] for c in report["cells"]] == [
        base_cell_key(a, b) for a, b in BENCHMARK_BASE_CELLS
    ]
    first = report["cells"][0]
    assert first["candidate_window"] == {
        "start": 800_000, "stop": 800_006, "half_open": True, "size": 6,
    }
    assert first["n_candidates_attempted"] == 3
    assert first["n_accepted"] == 2 and first["n_rejected"] == 1
    assert first["accepted_seeds"] == [800_001, 800_002]
    assert first["rejection_reasons"] == {NO_FD_ELIGIBLE_EGO: 1}
    assert first["rejection_detail_reasons"] == {"pre_event_popup_risk": 1}
    assert [c["seed"] for c in first["candidates"]] == [800_000, 800_001, 800_002]

    # The manifest identity and the FILE hash, so a report and a manifest can be tied
    # together after the fact.
    import hashlib
    assert report["manifest"]["manifest_id"] == result.manifest.manifest_id
    assert report["manifest"]["file_sha256"] == hashlib.sha256(
        result.manifest_path.read_bytes()).hexdigest()
    assert report["accepted_seeds"] == [int(s) for s in result.manifest.seeds()]


# =============================================================================
# PO3.4 -- determinism
# =============================================================================

def test_po3_the_same_inputs_reproduce_the_same_population(tmp_path: Path) -> None:
    """A fixed candidate seed reproduces the same decision, twice over."""
    probe_a, calls_a = _stub_probe(reject=(800_000, 800_013))
    probe_b, calls_b = _stub_probe(reject=(800_000, 800_013))
    first = _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe_a)
    second = _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe_b)
    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert calls_a == calls_b, "the candidate ATTEMPT order is not reproducible"
    assert [c.to_record() for c in first.candidates] != [], "no candidate recorded"
    assert ([(c.seed, c.outcome) for c in first.candidates]
            == [(c.seed, c.outcome) for c in second.candidates])


def test_po3_a_smaller_scale_is_a_prefix_of_a_larger_one(tmp_path: Path) -> None:
    """Accepting fewer worlds per cell selects a PREFIX of the larger population.

    The walk stops at the quota and never looks further, so growing the scale ADDS worlds
    rather than re-selecting them -- which is what lets a pilot population be a subset of
    the eventual one.
    """
    probe_small, _ = _stub_probe(reject=(800_001,))
    probe_big, _ = _stub_probe(reject=(800_001,))
    small = _run(tmp_path, worlds_per_cell=1, max_candidates=6, probe=probe_small)
    big = _run(tmp_path, worlds_per_cell=3, max_candidates=6, probe=probe_big)
    for key, seeds in _seeds_by_cell(small).items():
        assert seeds == _seeds_by_cell(big)[key][:len(seeds)], key


def _seeds_by_cell(result) -> dict:
    out: dict = {}
    for world in result.manifest.worlds:
        out.setdefault(world.base_cell_key, []).append(int(world.seed))
    return out


# =============================================================================
# PO3.5 -- refusals: exhaustion, instrument faults, bad inputs
# =============================================================================

def test_po3_window_exhaustion_aborts_with_no_manifest_but_a_failed_report(
    tmp_path: Path,
) -> None:
    """A cell that cannot be filled ABORTS with NO MANIFEST -- and KEEPS its audit.

    THE DEFECT THIS CLOSES (review fix). The exhaustion verdict used to be raised from
    inside `_scan_cell`, so the candidate outcomes never returned to the caller and the
    report writer was never reached: a failed preflight left an exception message and
    nothing else -- while that message told the operator to inspect a build report which
    had never been written.

    That is wrong for the Task-5C contract. Every candidate already attempted has SPENT
    its seed, so its identity and its rejection reason cannot be recovered by re-running
    the same window; they ARE the evidence the failure exists to produce, and they are
    what an operator reads to decide whether to raise `max_candidates_per_cell`, lower
    `worlds_per_cell`, or investigate the attrition. So the manifest is still refused and
    the walk still stops before any later cell, but the FAILED report is written first.

    The exhaustion is placed in the SECOND base cell on purpose: that is the only shape
    in which "the completed cell's audit survived" and "no later cell was probed" are
    both observable.
    """
    out = tmp_path / "out"
    calls: list = []
    # A2-low fills from 800000; A2-high's whole window [800003, 800006) is rejected.
    probe, _ = _stub_probe(reject=range(800_003, 800_006), calls=calls)
    try:
        _run(tmp_path, worlds_per_cell=1, max_candidates=3, probe=probe, output_dir=out)
    except BenchmarkPreflightError as exc:
        assert "exhausted its candidate window" in str(exc)
        assert "NO manifest" in str(exc)
        assert "NO later cell is scanned" in str(exc)
        # The audit travels ON the exception too, so a caller never has to go looking.
        assert exc.report is not None
        assert exc.report_path == out / "benchmark_preflight_report.json"
        raised = exc
    else:
        raise AssertionError("an unfillable cell produced a benchmark")

    # --- NO MANIFEST, EVER -------------------------------------------------------
    assert not (out / "benchmark_manifest.json").exists(), (
        "a failed preflight wrote a benchmark manifest"
    )
    assert list(out.glob("*manifest*.json")) == [], list(out.glob("*"))

    # --- THE REPORT DOES EXIST, AND SAYS IT FAILED --------------------------------
    report_path = out / "benchmark_preflight_report.json"
    assert report_path.exists(), "the failed preflight discarded its candidate audit"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == raised.report, "the written report differs from the raised one"
    assert report["schema"] == PREFLIGHT_SCHEMA
    assert report["status"] == PREFLIGHT_STATUS_FAILED
    assert report["complete"] is False
    assert report["manifest_written"] is False
    assert report["manifest"] is None
    assert report["stale_manifest_path"] is None
    # NOTHING in the document may read as a frozen benchmark.
    assert "manifest_id" not in json.dumps(report)

    # --- IT NAMES THE EXHAUSTED CELL ----------------------------------------------
    failure = report["failure"]
    exhausted = base_cell_key(*BENCHMARK_BASE_CELLS[1])
    assert failure["reason"] == PREFLIGHT_FAILURE_WINDOW_EXHAUSTED
    assert failure["base_cell"] == exhausted
    assert failure["base_cell_ordinal"] == 1
    assert failure["candidate_window"] == {
        "start": 800_003, "stop": 800_006, "half_open": True, "size": 3,
    }
    assert failure["worlds_requested"] == 1
    assert failure["worlds_accepted"] == 0
    assert failure["worlds_missing"] == 1
    assert failure["n_candidates_attempted"] == 3
    assert failure["accepted_seeds"] == []
    assert failure["manifest_written"] is False

    # --- EVERY SPENT SEED IS PRESENT, EXACTLY ONCE, WITH ITS REASON ---------------
    assert failure["attempted_seeds"] == [800_003, 800_004, 800_005]
    assert len(failure["attempted_seeds"]) == len(set(failure["attempted_seeds"]))
    assert failure["rejection_reasons"] == {NO_FD_ELIGIBLE_EGO: 3}
    assert failure["rejection_detail_reasons"] == {"pre_event_popup_risk": 3}

    # --- THE COMPLETED CELL'S AUDIT SURVIVED --------------------------------------
    completed = base_cell_key(*BENCHMARK_BASE_CELLS[0])
    assert failure["cells_completed"] == [completed]
    assert failure["cells_not_attempted"] == [
        base_cell_key(a, b) for a, b in BENCHMARK_BASE_CELLS[2:]
    ]
    assert [c["base_cell"] for c in report["cells"]] == [completed, exhausted]
    first, second = report["cells"]
    assert first["n_accepted"] == 1 and first["window_exhausted"] is False
    assert first["worlds_missing"] == 0
    assert [c["seed"] for c in first["candidates"]] == [800_000]
    assert first["candidates"][0]["outcome"] == CANDIDATE_ACCEPTED
    assert second["window_exhausted"] is True and second["worlds_missing"] == 1
    assert [c["seed"] for c in second["candidates"]] == [800_003, 800_004, 800_005]
    assert all(c["outcome"] == CANDIDATE_REJECTED for c in second["candidates"])

    # Every seed the preflight spent appears EXACTLY ONCE across the whole report.
    spent = [c["seed"] for cell in report["cells"] for c in cell["candidates"]]
    assert spent == [800_000, 800_003, 800_004, 800_005]
    assert len(spent) == len(set(spent))
    assert report["totals"]["n_candidates_attempted"] == 4
    assert report["totals"]["n_accepted"] == 1
    assert report["totals"]["n_rejected"] == 3
    assert report["accepted_seeds"] == [800_000]

    # --- NO LATER CELL WAS PROBED -------------------------------------------------
    assert calls == [800_000, 800_003, 800_004, 800_005], calls


def test_po3_an_in_memory_failure_carries_its_audit_without_inventing_a_file(
    tmp_path: Path,
) -> None:
    """With no ``output_dir`` there is nowhere to write -- so the report rides the raise.

    The programmatic contract stays clean: `report` is the full audit and `report_path`
    is `None`, which is the truthful "no file could be written" rather than a path to
    something that does not exist. It must NOT invent an output directory, because a
    caller that deliberately asked for an in-memory run has not consented to one.
    """
    calls: list = []
    probe, _ = _stub_probe(reject=range(800_000, 800_003), calls=calls)
    try:
        _run(tmp_path, worlds_per_cell=1, max_candidates=3, probe=probe,
             output_dir=None)
    except BenchmarkPreflightError as exc:
        assert exc.report_path is None
        assert "no output_dir was supplied" in str(exc)
        report = exc.report
    else:
        raise AssertionError("an unfillable cell produced a benchmark")

    assert report is not None
    assert report["status"] == PREFLIGHT_STATUS_FAILED
    assert report["manifest"] is None and report["manifest_written"] is False
    assert report["failure"]["attempted_seeds"] == [800_000, 800_001, 800_002]
    assert report["failure"]["cells_completed"] == []
    assert report["failure"]["cells_not_attempted"] == [
        base_cell_key(a, b) for a, b in BENCHMARK_BASE_CELLS[1:]
    ]
    assert calls == [800_000, 800_001, 800_002]
    # Nothing was created anywhere: the run had no output directory to create.
    assert list(tmp_path.rglob("benchmark_*.json")) == []


def test_po3_a_stale_manifest_beside_a_failure_is_named_not_adopted(
    tmp_path: Path,
) -> None:
    """A manifest left by an EARLIER run is reported, never deleted and never claimed.

    Deleting a file this preflight did not write would destroy another run's artifact;
    staying silent about it would let a reader find a manifest beside a failure report
    and take the two for one run. So it is NAMED, and `manifest_written` still says
    `false`.
    """
    out = tmp_path / "out"
    out.mkdir(parents=True)
    stale = out / "benchmark_manifest.json"
    stale.write_text('{"stale": true}', encoding="utf-8")

    probe, _ = _stub_probe(reject=range(800_000, 800_003))
    try:
        _run(tmp_path, worlds_per_cell=1, max_candidates=3, probe=probe, output_dir=out)
    except BenchmarkPreflightError as exc:
        report = exc.report
    else:
        raise AssertionError("an unfillable cell produced a benchmark")

    assert report["stale_manifest_path"] == str(stale)
    assert report["manifest_written"] is False and report["manifest"] is None
    # NOT deleted, and NOT overwritten.
    assert stale.read_text(encoding="utf-8") == '{"stale": true}'


def test_po3_measurement_integrity_faults_are_never_replacement_eligible(
    tmp_path: Path,
) -> None:
    """The three instrument faults propagate; none is recorded as a rejection.

    A world that contradicts its own certificate says the INSTRUMENT is wrong, so
    replacing it would freeze a population selected by a defect -- the same reason these
    abort a training run.
    """
    for knob, exc_type in (("integrity", EpisodeRosterError),
                           ("fd_integrity", FuelDamageIntegrityError),
                           ("identity", BenchmarkIdentityError)):
        calls: list = []
        probe, _ = _stub_probe(calls=calls, **{knob: (800_001,)})
        try:
            _run(tmp_path, worlds_per_cell=2, max_candidates=6, probe=probe)
        except exc_type:
            pass
        else:
            raise AssertionError("%s was silently replaced" % exc_type.__name__)
        # It stopped AT the faulting candidate: nothing after it was attempted.
        assert calls[-1] == 800_001, (knob, calls)


def test_po3_the_scale_is_never_defaulted_and_a_bad_one_refuses(
    tmp_path: Path,
) -> None:
    """Every scale input is REQUIRED, and an impossible one is refused before compute."""
    import inspect
    sig = inspect.signature(run_benchmark_preflight)
    for name in ("worlds_per_cell", "benchmark_base_seed", "max_candidates_per_cell"):
        assert sig.parameters[name].default is inspect.Parameter.empty, name

    def _refuses(**kwargs) -> str:
        try:
            _run(tmp_path, **kwargs)
        except BenchmarkPreflightError as exc:
            return str(exc)
        raise AssertionError("accepted an impossible scale: %r" % (kwargs,))

    assert "worlds_per_cell must be >= 1" in _refuses(worlds_per_cell=0)
    assert "max_candidates_per_cell must be >= 1" in _refuses(max_candidates=0)
    assert "must be >= worlds_per_cell" in _refuses(worlds_per_cell=4, max_candidates=3)
    assert "benchmark_base_seed must be >= 0" in _refuses(base_seed=-1)


def test_po3_a_fixed_cell_config_is_refused(tmp_path: Path) -> None:
    """The 18-stratum benchmark is defined for the generalized design ONLY."""
    cfg = TrainConfig(n_iterations=1, output_dir=tmp_path / "x",
                      episode_design=EPISODE_DESIGN_FIXED_CELL_V1)
    try:
        _run(tmp_path, cfg=cfg)
    except BenchmarkPreflightError as exc:
        assert EPISODE_DESIGN_GENERALIZED_V1 in str(exc)
    else:
        raise AssertionError("a fixed-cell config built a stratified benchmark")


def test_po3_incomplete_provenance_refuses_before_anything_is_built(
    tmp_path: Path,
) -> None:
    """The comparator must be attributable to an exact code state, or nothing is built."""
    out = tmp_path / "out"
    calls: list = []
    probe, _ = _stub_probe(calls=calls)
    try:
        _run(tmp_path, probe=probe, output_dir=out,
             provenance=_FAKE_GIT_INCOMPLETE)
    except BenchmarkPreflightError as exc:
        assert "Git provenance is UNAVAILABLE" in str(exc)
    else:
        raise AssertionError("a population was selected without provenance")
    assert calls == [], "a candidate was probed despite incomplete provenance"
    assert not (out / "benchmark_manifest.json").exists()


def test_po3_a_preflight_does_not_read_the_training_schedule(tmp_path: Path) -> None:
    """A preflight TRAINS NOTHING, so it must not demand a training schedule.

    `TrainConfig.validate` would require `generalized_max_attempts_per_iteration` and a
    benchmark path; requiring them here would make an operator invent a schedule in order
    to build a population, and a number invented to satisfy a check is exactly the kind of
    plausible-but-meaningless value a scientific artifact must not carry.
    """
    cfg = _cfg(tmp_path)                       # no budget, no manifest, eval defaults on
    assert cfg.generalized_max_attempts_per_iteration is None
    try:
        cfg.validate()
    except ValueError:
        pass                                    # ... a TRAINING run would indeed refuse
    else:
        raise AssertionError("expected TrainConfig.validate to require the budget")
    # ... and the preflight nevertheless runs, because it reads none of that.
    result = _run(tmp_path, cfg=cfg, worlds_per_cell=1, max_candidates=3)
    assert result.manifest.n_worlds == len(BENCHMARK_BASE_CELLS)


# =============================================================================
# PO3.6 -- the boundary with evaluation
# =============================================================================

def test_po3_evaluation_still_performs_no_substitution() -> None:
    """Population SELECTION lives here; scientific EVALUATION stays substitution-free.

    Checked structurally rather than by prose: `evaluate_benchmark` must not reach for
    this module, and this module must not be imported by the harness it consumes. A
    preflight that ran inside an evaluation round would make the frozen manifest mutable
    in exactly the way its content hash exists to prevent.
    """
    trainer = (SRC / "match_aou" / "rl" / "training" / "graph_train.py").read_text(
        encoding="utf-8")
    assert "graph_benchmark_preflight" not in trainer, (
        "graph_train imports the preflight: population selection has leaked into the "
        "evaluation harness"
    )
    module = (SRC / "match_aou" / "rl" / "training"
              / "graph_benchmark_preflight.py").read_text(encoding="utf-8")
    # The import direction is one-way: the preflight consumes the harness, never back.
    assert "from .graph_train import" in module


def test_po3_the_preflight_builds_no_policy_and_runs_no_episode() -> None:
    """World ELIGIBILITY, never policy performance.

    Selecting benchmark worlds by outcome would build the comparator out of the very
    quantity the comparison measures, so the module must not reach for the policy, the
    tick loop or the reward at all.
    """
    module = (SRC / "match_aou" / "rl" / "training"
              / "graph_benchmark_preflight.py").read_text(encoding="utf-8")
    for forbidden in ("build_policy(", "run_episode(", "compute_episode_reward(",
                      "PPOUpdater", "CTDEUpdater", "evaluate_benchmark("):
        assert forbidden not in module, forbidden


if __name__ == "__main__":
    import inspect as _inspect

    _tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    _tmp = Path(__import__("tempfile").mkdtemp(prefix="preflight_5c_"))
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
