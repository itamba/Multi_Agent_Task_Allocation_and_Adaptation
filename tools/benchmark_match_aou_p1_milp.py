"""Same-input engineering comparison: frozen BONMIN MINLP vs the new P1 MILP.

ENGINEERING VALIDATION ONLY. This is NOT a scientific measurement: it schedules no
population, defines no comparator, trains nothing and produces no research verdict. It
exists to answer three engineering questions about the specialized solver:

  1. Does it return the SAME covered utility as the frozen solver on the same inputs?
  2. Is it faster, and by how much?
  3. Does HiGHS stack agents redundantly when redundancy is free at ``p = 1``?

METHOD
------
For every cell in the matrix a REAL project world is generated with the same
``ScenarioGenerator`` / ``VariationConfig`` request the live pipeline uses
(``graph_rollout.run_rollout`` / ``graph_train.build_variation_config``): known-only,
Layer-1 discovery chain OFF, ``strict_geometry=True``, the 200 km / 100 km geometry, and
``detection_km = DETECTION_KM``. The agents and tasks are then extracted ONCE through the
pipeline's own ``_build_env`` / ``_extract_world`` and the SAME OBJECTS are handed to both
solvers -- so no difference can come from two different worlds.

This is exactly the input the live path's first ``solve_and_normalize`` receives.

TIMING
------
Build time, solve time and total call time are measured separately, after a warm-up
repetition that is discarded. Median, p95 and cumulative totals are reported.

An honest caveat that belongs with any speedup number here: **the legacy path spawns
BONMIN as an out-of-process executable per solve and exchanges files with it, while HiGHS
runs in-process.** A large part of any measured gap is that process/IO overhead rather
than search efficiency. The comparison is still the right one -- it is what the pipeline
actually pays per episode -- but it must not be reported as "the MILP search is N times
faster than branch-and-bound".

QUALITY
-------
Per cell the report records, for BOTH solvers: the selected task indices, the unselected
tasks, the total assignment count, the redundant assignment count
``sum_j max(0, assigned_j - 1)``, the exact-P1 covered utility, the legacy-EPSILON
objective, and capability / fuel feasibility of every assignment.

Four verdicts, and only the first is a finding to investigate:

  * ``MISMATCH``               -- covered utility or covered task SET differs. Never to
                                  be waved away as solver tie-breaking.
  * ``REDUNDANCY_DIFFERENCE``  -- same covered set and same exact-P1 utility, but legacy
                                  used MORE agents. This is the EPSILON stacking
                                  incentive, and it is EXPECTED: at ``p = 1`` the legacy
                                  objective pays ``utility * (EPSILON - EPSILON**2)`` per
                                  redundant agent, so each allocation is optimal FOR ITS
                                  OWN objective. The two objectives do NOT share an
                                  optimal allocation set.
  * ``ALTERNATE_OPTIMUM``      -- same covered set, same utility, same redundancy, but a
                                  different agent->task pairing.
  * ``IDENTICAL``              -- byte-identical assignments.

Agreement on the covered task set is the ONLY equivalence this tool can evidence.
It is not evidence of a shared optimal allocation set, and must not be reported as one.

ENVIRONMENT FACTS, AND WHO JUDGES THEM
--------------------------------------
**This tool never certifies its own run.** It records environment FACTS -- Python, SciPy,
Pyomo and BONMIN versions/paths, ``CONDA_DEFAULT_ENV`` and ``PYTHONNOUSERSITE`` -- into
the printed header and into ``report.json``. Whether a given run counts as Grade-A
evidence is an **orchestrator decision**, taken by inspecting those recorded facts
against ``CLAUDE.md`` section 1. That contract is deliberately NOT duplicated here: a
second copy would drift, and a benchmark that graded itself would be worthless.

Accordingly there are only two evidence labels, and neither of them says "validated":

  * ``diagnostic_cross_environment`` -- ``--bonmin-executable`` was supplied, so this
    interpreter reached into some OTHER environment's solver. The two arms did not share
    one environment.
  * ``same_environment_unverified`` -- bonmin came from PATH, so both arms ran from this
    one process. **That is ALL it establishes.** It does not verify that the interpreter
    is one of the repository's validated execution contexts, nor -- on the cluster --
    that ``PYTHONNOUSERSITE=1`` was set.

The invocations a validated run is expected to use (the orchestrator still judges the
RECORDED FACTS, never the command line):

    # BGU cluster (contract pins SciPy 1.17.1 / Pyomo 6.10.1 / coin-or-bonmin 1.8.9)
    PYTHONNOUSERSITE=1 conda run -n graph_rl_cluster --no-capture-output python tools/benchmark_match_aou_p1_milp.py

    # LOCAL, only if `nlp_env` can import BOTH scipy.optimize.milp AND run bonmin
    conda run -n nlp_env --no-capture-output python tools/benchmark_match_aou_p1_milp.py

At the time of writing the LOCAL ``nlp_env`` carries a BROKEN SciPy (cp39-ABI extension
modules under a Python 3.12 interpreter), so ``scipy.optimize.milp`` cannot be imported
there and the LOCAL same-process comparison is not currently available.

USAGE
-----
    python tools/benchmark_match_aou_p1_milp.py --bonmin-executable <path-to-bonmin>

``--bonmin-executable`` may be omitted when ``bonmin`` is already on PATH, which is how
it is reached inside a validated environment; omitting it does not by itself make a run
validated (see above). Scenario files are written to a temporary directory unless
``--output-dir`` is given; nothing is written into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.solvers.match_aou_MINLP_solver import EPSILON as LEGACY_EPSILON  # noqa: E402
from match_aou.solvers.match_aou_MINLP_solver import MatchAou, round_trip_cost  # noqa: E402
from match_aou.solvers.match_aou_p1_milp_solver import (  # noqa: E402
    TERMINATION_OPTIMAL,
    MatchAouP1MILP,
)

BASE_SCENARIO = REPO_ROOT / "data" / "scenarios" / "strike_training_4v5.json"


#: Evidence labels. Deliberately CONSERVATIVE: this tool records environment FACTS and
#: never certifies its own run.
#:
#: `--bonmin-executable` was supplied, so this interpreter is reaching into some other
#: environment's solver. Both arms did not share one environment.
EVIDENCE_DIAGNOSTIC_CROSS_ENVIRONMENT = "diagnostic_cross_environment"
#: No override: bonmin came from PATH, so both solver calls CAN run from this one
#: process. That is all it establishes. It does NOT establish that this interpreter is
#: one of the repository's validated execution contexts (`CLAUDE.md` section 1), nor --
#: on the cluster -- that `PYTHONNOUSERSITE=1` was set. Hence "unverified".
EVIDENCE_SAME_ENVIRONMENT_UNVERIFIED = "same_environment_unverified"

EVIDENCE_CLASSES = (
    EVIDENCE_DIAGNOSTIC_CROSS_ENVIRONMENT,
    EVIDENCE_SAME_ENVIRONMENT_UNVERIFIED,
)


def classify_evidence(*, bonmin_executable_overridden: bool) -> str:
    """Label how the two solver arms were reached -- and nothing beyond that.

    There is deliberately NO "validated" outcome. Whether a run counts as Grade-A
    evidence is an ORCHESTRATOR decision, taken by inspecting the recorded environment
    facts (python / scipy / pyomo / bonmin path / ``CONDA_DEFAULT_ENV`` /
    ``PYTHONNOUSERSITE``) against ``CLAUDE.md`` section 1. Duplicating that contract in
    here would create a second, drifting copy of it, and would let a benchmark certify
    itself.
    """
    return (
        EVIDENCE_DIAGNOSTIC_CROSS_ENVIRONMENT
        if bonmin_executable_overridden
        else EVIDENCE_SAME_ENVIRONMENT_UNVERIFIED
    )


def _pyomo_version() -> str:
    """Pyomo's version, for the environment record. Never fatal to the run."""
    try:
        import pyomo

        return str(pyomo.__version__)
    except Exception:  # pragma: no cover - environment dependent
        return "<unavailable>"

#: (agent_count, known_target_count). Covers the current agent cardinalities A in
#: {2, 3, 4} against representative target counts, including the A == T square case the
#: live cell uses and target-rich cases where redundancy would be free.
DEFAULT_MATRIX: Tuple[Tuple[int, int], ...] = (
    (2, 2), (2, 3), (2, 4),
    (3, 3), (3, 4), (3, 5),
    (4, 4), (4, 5), (4, 6),
)


# ======================================================================================
# World construction -- the pipeline's own request, and ONE extraction per cell
# ======================================================================================

def build_world(
    *, num_agents: int, n_known: int, seed: int, scenario_dir: Path
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Generate a real known-only world and extract ``(agents, tasks)`` ONCE."""
    from match_aou.rl.training.graph_episode_setup import (
        DETECTION_KM,
        MAX_SIM_TICKS,
        _build_env,
        _close_quietly,
        _extract_world,
    )
    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator,
        VariationConfig,
    )

    generator = ScenarioGenerator(
        base_scenario_path=str(BASE_SCENARIO),
        output_dir=str(scenario_dir),
        max_sim_ticks=MAX_SIM_TICKS,
    )
    generator.recompute_time_feasible_cap(allowed_classes=None)

    # Structurally the B1 construction request (see graph_train.build_variation_config).
    variation = VariationConfig(
        include_sams=False,
        num_aircraft=num_agents,
        num_red_airbases=n_known,
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.0,
        min_target_distance_km=200.0,
        min_target_separation_km=100.0,
        ensure_discovery_chain=False,
        strict_geometry=True,
        detection_km=DETECTION_KM,
        seed=seed,
    )
    scenario_path = generator.generate(episode=0, config=variation)

    env = None
    try:
        _game, env, observation = _build_env(
            scenario_path.read_text(encoding="utf-8"),
            max_episode_steps=MAX_SIM_TICKS,
            attacking_side_color="blue",
            record_every_seconds=None,
            recording_export_path=None,
        )
        agents, tasks = _extract_world(observation, "blue")
    finally:
        if env is not None:
            _close_quietly(env)

    meta = {
        "scenario_file": scenario_path.name,
        "n_agents_extracted": len(agents),
        "n_tasks_extracted": len(tasks),
        "agent_ids": [str(a.id) for a in agents],
        "target_ids": [str(t.steps[0].target_id) for t in tasks],
        "utilities": [float(t.utility) for t in tasks],
    }
    return agents, tasks, meta


# ======================================================================================
# Solution accounting -- one implementation, applied to BOTH solvers' output
# ======================================================================================

def assigned_counts(n_tasks: int, solution: Optional[Dict[Any, List[Tuple[int, int]]]]) -> List[int]:
    """Agents assigned per task index."""
    counts = [0] * n_tasks
    for entries in (solution or {}).values():
        for task_index, _step in entries:
            counts[task_index] += 1
    return counts


def exact_p1_covered_utility(tasks: Sequence[Any], counts: Sequence[int]) -> float:
    """``sum_j utility_j`` over tasks with at least one assignment. No EPSILON."""
    return float(sum(float(tasks[j].utility) for j, c in enumerate(counts) if c > 0))


def legacy_epsilon_objective(tasks: Sequence[Any], counts: Sequence[int]) -> float:
    """The frozen MINLP objective evaluated at an allocation.

    ``sum_j utility_j * prod_k [1 - (1 - p_k + EPSILON) ** m_j]``. With one step at
    ``p = 1`` this is ``utility_j * (1 - EPSILON ** m_j)``, which is ``0`` at ``m_j = 0``.
    Computed for BOTH solvers' allocations so the two are compared on the legacy scale as
    well as on the exact-P1 scale.
    """
    total = 0.0
    for j, task in enumerate(tasks):
        m = counts[j]
        product = 1.0
        for step in task.steps:
            product *= 1.0 - (1.0 - float(step.probability) + LEGACY_EPSILON) ** m
        total += float(task.utility) * product
    return total


def feasibility_report(
    agents: Sequence[Any],
    tasks: Sequence[Any],
    solution: Optional[Dict[Any, List[Tuple[int, int]]]],
) -> Dict[str, Any]:
    """Check every assignment against the capability and round-trip fuel rules."""
    by_id = {a.id: a for a in agents}
    capability_violations: List[str] = []
    fuel_violations: List[str] = []
    for agent_id, entries in (solution or {}).items():
        agent = by_id.get(agent_id)
        if agent is None:
            capability_violations.append(f"unknown agent id {agent_id!r}")
            continue
        spent = 0.0
        for task_index, _step in entries:
            step = tasks[task_index].steps[0]
            if not agent.has_capabilities(step.capabilities):
                capability_violations.append(f"{agent_id} -> task {task_index}")
            if getattr(agent, "location", None) is not None and getattr(step, "location", None) is not None:
                spent += round_trip_cost(agent, step.location)
        if spent > float(agent.budget) + 1e-6:
            fuel_violations.append(f"{agent_id} spent {spent:.3f} > budget {float(agent.budget):.3f}")
    return {
        "capability_ok": not capability_violations,
        "fuel_ok": not fuel_violations,
        "capability_violations": capability_violations,
        "fuel_violations": fuel_violations,
    }


def describe(
    agents: Sequence[Any],
    tasks: Sequence[Any],
    solution: Optional[Dict[Any, List[Tuple[int, int]]]],
    unselected: Sequence[int],
) -> Dict[str, Any]:
    """The quality record PO3 requires, for one solver's output."""
    counts = assigned_counts(len(tasks), solution)
    selected = [j for j, c in enumerate(counts) if c > 0]
    return {
        "solved": solution is not None,
        "selected_task_indices": selected,
        "selected_target_ids": [str(tasks[j].steps[0].target_id) for j in selected],
        "unselected_tasks": list(unselected),
        "assigned_counts_per_task": counts,
        "total_assignments": int(sum(counts)),
        "redundant_assignments": int(sum(max(0, c - 1) for c in counts)),
        "exact_p1_covered_utility": exact_p1_covered_utility(tasks, counts),
        "legacy_epsilon_objective": legacy_epsilon_objective(tasks, counts),
        "assignments": {str(k): sorted(v) for k, v in (solution or {}).items()},
        **feasibility_report(agents, tasks, solution),
    }


# ======================================================================================
# Timing
# ======================================================================================

def _summarize(samples: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    if not ordered:
        return {"median": float("nan"), "p95": float("nan"), "total": 0.0, "n": 0}
    # Nearest-rank p95: with the small repetition counts used here, an interpolated
    # quantile would invent a value between two real observations.
    rank = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
    return {
        "median": statistics.median(ordered),
        "p95": ordered[rank],
        "total": float(sum(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
        "n": len(ordered),
    }


def time_legacy(
    agents: Sequence[Any],
    tasks: Sequence[Any],
    *,
    repeats: int,
    warmup: int,
    bonmin_executable: Optional[str],
) -> Dict[str, Any]:
    """Time the frozen MINLP: model build, bonmin solve, and total call."""
    import match_aou.solvers.match_aou_MINLP_solver as legacy_module
    from pyomo.environ import SolverFactory

    original_factory = legacy_module.SolverFactory
    if bonmin_executable:
        def _factory(name: str, **kwargs: Any) -> Any:
            return SolverFactory(name, executable=bonmin_executable, **kwargs)
        legacy_module.SolverFactory = _factory  # type: ignore[assignment]

    build_times: List[float] = []
    solve_times: List[float] = []
    total_times: List[float] = []
    last: Tuple[Any, Any, Any] = (None, None, [])
    try:
        for index in range(warmup + repeats):
            start = time.perf_counter()
            model = MatchAou(
                agents=list(agents), tasks=list(tasks),
                precedence_relations=[], risk_factor=0.0,
            )
            built = time.perf_counter()
            solution, results, unselected = model.solve(solver_name="bonmin")
            done = time.perf_counter()
            if index >= warmup:  # discard warm-up
                build_times.append(built - start)
                solve_times.append(done - built)
                total_times.append(done - start)
            last = (solution, results, unselected)
    finally:
        legacy_module.SolverFactory = original_factory  # type: ignore[assignment]

    solution, results, unselected = last
    termination = str(getattr(getattr(results, "solver", None), "termination_condition", "unavailable"))
    return {
        "solution": solution,
        "unselected": unselected,
        "termination": termination,
        "build": _summarize(build_times),
        "solve": _summarize(solve_times),
        "total": _summarize(total_times),
    }


def time_p1(
    agents: Sequence[Any], tasks: Sequence[Any], *, repeats: int, warmup: int
) -> Dict[str, Any]:
    """Time the P1 MILP: model build, HiGHS solve, and total call."""
    build_times: List[float] = []
    solve_times: List[float] = []
    total_times: List[float] = []
    last: Tuple[Any, Any, Any] = (None, None, [])
    for index in range(warmup + repeats):
        start = time.perf_counter()
        model = MatchAouP1MILP(
            agents=list(agents), tasks=list(tasks),
            precedence_relations=[], risk_factor=0.0, mip_rel_gap=0.0,
        )
        built = time.perf_counter()
        solution, results, unselected = model.solve()
        done = time.perf_counter()
        if index >= warmup:
            build_times.append(built - start)
            solve_times.append(done - built)
            total_times.append(done - start)
        last = (solution, results, unselected)

    solution, results, unselected = last
    return {
        "solution": solution,
        "unselected": unselected,
        "termination": results.solver.termination_condition,
        "n_variables": results.n_variables,
        "n_constraint_rows": results.n_constraint_rows,
        "mip_rel_gap": results.mip_rel_gap,
        "build": _summarize(build_times),
        "solve": _summarize(solve_times),
        "total": _summarize(total_times),
    }


# ======================================================================================
# Comparison
# ======================================================================================

def compare_cell(
    *,
    num_agents: int,
    n_known: int,
    seed: int,
    scenario_dir: Path,
    repeats: int,
    warmup: int,
    bonmin_executable: Optional[str],
) -> Dict[str, Any]:
    """Run one matrix cell end to end on ONE extracted world."""
    agents, tasks, world = build_world(
        num_agents=num_agents, n_known=n_known, seed=seed, scenario_dir=scenario_dir
    )

    legacy = time_legacy(
        agents, tasks, repeats=repeats, warmup=warmup, bonmin_executable=bonmin_executable
    )
    p1 = time_p1(agents, tasks, repeats=repeats, warmup=warmup)

    legacy_quality = describe(agents, tasks, legacy["solution"], legacy["unselected"])
    p1_quality = describe(agents, tasks, p1["solution"], p1["unselected"])

    same_covered_set = (
        legacy_quality["selected_task_indices"] == p1_quality["selected_task_indices"]
    )
    same_utility = abs(
        legacy_quality["exact_p1_covered_utility"] - p1_quality["exact_p1_covered_utility"]
    ) < 1e-9
    same_assignments = legacy_quality["assignments"] == p1_quality["assignments"]

    legacy_redundant = legacy_quality["redundant_assignments"]
    p1_redundant = p1_quality["redundant_assignments"]

    if not same_utility or not same_covered_set:
        # The only verdict that is a FINDING TO INVESTIGATE. Everything below is a
        # difference in WHICH optimal allocation was returned, not in how good it is.
        verdict = "MISMATCH"
    elif same_assignments:
        verdict = "IDENTICAL"
    elif legacy_redundant > p1_redundant:
        # The documented EPSILON effect (CLAUDE.md section 8, "Solver 2:1 stacking"):
        # the legacy objective scores a task at `utility * (1 - EPSILON**m)`, so a
        # SECOND agent is worth a tiny but strictly positive `utility * (EPSILON -
        # EPSILON**2)`, and the MINLP correctly chases it. At exact p = 1 that
        # incentive does not exist, so the P1 MILP covers the same tasks with fewer
        # assignments. Both allocations are optimal FOR THEIR OWN objective.
        verdict = "REDUNDANCY_DIFFERENCE"
    else:
        verdict = "ALTERNATE_OPTIMUM"

    speedup = (
        legacy["total"]["median"] / p1["total"]["median"]
        if p1["total"]["median"] > 0 else float("inf")
    )

    return {
        "cell": {"num_agents": num_agents, "n_known": n_known, "seed": seed},
        "world": world,
        "legacy": {k: v for k, v in legacy.items() if k != "solution"},
        "p1": {k: v for k, v in p1.items() if k != "solution"},
        "legacy_quality": legacy_quality,
        "p1_quality": p1_quality,
        "verdict": verdict,
        "same_covered_task_set": same_covered_set,
        "same_exact_p1_covered_utility": same_utility,
        "same_raw_assignments": same_assignments,
        "median_total_speedup_legacy_over_p1": speedup,
    }


def print_report(results: List[Dict[str, Any]]) -> None:
    """Human-readable summary. Every number is also in the JSON."""
    print("=" * 100)
    print("MATCH-AOU P1 MILP vs frozen BONMIN MINLP -- ENGINEERING VALIDATION ONLY")
    print("=" * 100)
    header = (
        f"{'cell':>10} {'T':>3} {'verdict':>18} "
        f"{'legacy tot med':>15} {'p1 tot med':>12} {'speedup':>9} "
        f"{'util L':>8} {'util P':>8} {'redun L':>8} {'redun P':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        cell = row["cell"]
        print(
            f"{cell['num_agents']}a/{cell['n_known']}k".rjust(10)
            + f" {row['world']['n_tasks_extracted']:>3}"
            + f" {row['verdict']:>18}"
            + f" {row['legacy']['total']['median']:>15.6f}"
            + f" {row['p1']['total']['median']:>12.6f}"
            + f" {row['median_total_speedup_legacy_over_p1']:>8.1f}x"
            + f" {row['legacy_quality']['exact_p1_covered_utility']:>8.1f}"
            + f" {row['p1_quality']['exact_p1_covered_utility']:>8.1f}"
            + f" {row['legacy_quality']['redundant_assignments']:>8}"
            + f" {row['p1_quality']['redundant_assignments']:>8}"
        )

    print("-" * len(header))
    legacy_total = sum(r["legacy"]["total"]["total"] for r in results)
    p1_total = sum(r["p1"]["total"]["total"] for r in results)
    print(f"cumulative total call time: legacy {legacy_total:.4f}s   p1 {p1_total:.4f}s")
    if p1_total > 0:
        print(f"cumulative speedup (legacy/p1): {legacy_total / p1_total:.1f}x")

    medians = [r["median_total_speedup_legacy_over_p1"] for r in results]
    print(f"median-of-cells speedup: {statistics.median(medians):.1f}x")

    print("\nper-phase medians (seconds):")
    for row in results:
        cell = row["cell"]
        print(
            f"  {cell['num_agents']}a/{cell['n_known']}k  "
            f"legacy build {row['legacy']['build']['median']:.6f} "
            f"solve {row['legacy']['solve']['median']:.6f} | "
            f"p1 build {row['p1']['build']['median']:.6f} "
            f"solve {row['p1']['solve']['median']:.6f} "
            f"(vars {row['p1']['n_variables']}, rows {row['p1']['n_constraint_rows']})"
        )

    mismatches = [r for r in results if r["verdict"] == "MISMATCH"]
    alternates = [r for r in results if r["verdict"] == "ALTERNATE_OPTIMUM"]
    redundant = [r for r in results if r["p1_quality"]["redundant_assignments"] > 0]
    infeasible = [
        r for r in results
        if not (r["p1_quality"]["capability_ok"] and r["p1_quality"]["fuel_ok"])
    ]

    print("\nfindings:")
    print(f"  MISMATCH cells (covered utility or covered set differs): {len(mismatches)}")
    for row in mismatches:
        print(f"    {row['cell']}: legacy {row['legacy_quality']} vs p1 {row['p1_quality']}")
    print(f"  ALTERNATE_OPTIMUM cells (same utility + redundancy, different pairing): {len(alternates)}")
    for row in alternates:
        print(
            f"    {row['cell']}: legacy {row['legacy_quality']['assignments']} "
            f"vs p1 {row['p1_quality']['assignments']}"
        )

    stacking = [r for r in results if r["verdict"] == "REDUNDANCY_DIFFERENCE"]
    print(
        f"  REDUNDANCY_DIFFERENCE cells (same covered set + utility, legacy used MORE "
        f"agents): {len(stacking)}"
    )
    for row in stacking:
        legacy_q, p1_q = row["legacy_quality"], row["p1_quality"]
        gain = legacy_q["legacy_epsilon_objective"] - p1_q["legacy_epsilon_objective"]
        print(
            f"    {row['cell']}: counts legacy {legacy_q['assigned_counts_per_task']} "
            f"({legacy_q['total_assignments']} assignments) vs p1 "
            f"{p1_q['assigned_counts_per_task']} ({p1_q['total_assignments']}); "
            f"exact-P1 utility identical at {p1_q['exact_p1_covered_utility']}; "
            f"legacy-EPSILON objective favours legacy by {gain:.3e}"
        )
    if stacking:
        print(
            "    -> this is the EPSILON incentive documented in CLAUDE.md section 8: at\n"
            "       p = 1 the legacy objective still pays utility*(EPSILON - EPSILON**2)\n"
            "       for a second agent. Each allocation is optimal for its OWN objective."
        )
    print(f"  cells where P1 stacked redundantly: {len(redundant)}")
    for row in redundant:
        print(f"    {row['cell']}: counts {row['p1_quality']['assigned_counts_per_task']}")
    print(f"  cells with a P1 capability/fuel violation: {len(infeasible)}")
    print(
        "\nNOTE: the legacy arm spawns BONMIN out-of-process per solve; HiGHS runs "
        "in-process.\n      A large share of the gap is process/IO overhead, not search "
        "efficiency."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="fixed world seed (default 0)")
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions per cell")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warm-up repetitions")
    parser.add_argument(
        "--bonmin-executable", default=None,
        help="path to bonmin; omit when it is on PATH",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="where to write scenarios + report.json (default: a temp dir)",
    )
    parser.add_argument(
        "--matrix", default=None,
        help="override the cell matrix, e.g. '2x2,3x3,4x4'",
    )
    args = parser.parse_args(argv)

    bonmin = args.bonmin_executable or shutil.which("bonmin")
    if not bonmin:
        print(
            "BLOCKED: no bonmin executable. Pass --bonmin-executable or put bonmin on "
            "PATH. Per CLAUDE.md section 1 the LOCAL solver environment is `nlp_env`.",
            file=sys.stderr,
        )
        return 2
    if not Path(bonmin).exists() and shutil.which(bonmin) is None:
        print(f"BLOCKED: bonmin executable not found at {bonmin!r}", file=sys.stderr)
        return 2

    if args.matrix:
        matrix = tuple(
            (int(part.split("x")[0]), int(part.split("x")[1]))
            for part in args.matrix.split(",")
        )
    else:
        matrix = DEFAULT_MATRIX

    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="p1_milp_benchmark_")
        out_dir = Path(temp_dir.name)

    try:
        import scipy

        # How the two arms were reached. This tool RECORDS environment facts; it never
        # certifies its own run as validated -- see `classify_evidence`.
        cross_environment = args.bonmin_executable is not None
        evidence_class = classify_evidence(
            bonmin_executable_overridden=cross_environment
        )

        print(f"python  : {sys.version.split()[0]}")
        print(f"scipy   : {scipy.__version__}")
        print(f"pyomo   : {_pyomo_version()}")
        print(f"bonmin  : {bonmin}")
        print(f"conda   : {os.environ.get('CONDA_DEFAULT_ENV', '<not a conda env>')}")
        print(f"nousersite: {os.environ.get('PYTHONNOUSERSITE', '<unset>')}")
        print(f"seed    : {args.seed}   repeats: {args.repeats}   warmup: {args.warmup}")
        print(f"outdir  : {out_dir}")
        print(f"evidence: {evidence_class}")
        if cross_environment:
            print(
                "  !! DIAGNOSTIC ONLY: --bonmin-executable was supplied, so this "
                "interpreter is\n"
                "     reaching another environment's bonmin. The two arms did not share "
                "one\n"
                "     environment."
            )
        else:
            print(
                "  NOTE: bonmin came from PATH, so both arms ran from this one process.\n"
                "        That is ALL this establishes -- it does not verify that this\n"
                "        interpreter is a validated execution context (CLAUDE.md section "
                "1),\n"
                "        nor, on cluster, that PYTHONNOUSERSITE=1 was set. Grade-A "
                "validation\n"
                "        is an orchestrator decision over the environment facts printed "
                "above."
            )

        results: List[Dict[str, Any]] = []
        for num_agents, n_known in matrix:
            scenario_dir = out_dir / f"scenarios_{num_agents}a_{n_known}k"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- cell {num_agents} agents / {n_known} known targets ---", flush=True)
            try:
                row = compare_cell(
                    num_agents=num_agents,
                    n_known=n_known,
                    seed=args.seed,
                    scenario_dir=scenario_dir,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    bonmin_executable=bonmin,
                )
            except Exception as exc:  # noqa: BLE001
                # A cell the GENERATOR refuses (strict geometry) is a world-construction
                # outcome, not a solver comparison result. Record and continue; it is
                # reported rather than silently dropped.
                print(f"  cell skipped: {type(exc).__name__}: {exc}", flush=True)
                results_entry = {
                    "cell": {"num_agents": num_agents, "n_known": n_known, "seed": args.seed},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (out_dir / "report_partial.json").write_text(
                    json.dumps(results_entry, indent=2), encoding="utf-8"
                )
                continue
            print(
                f"  verdict={row['verdict']}  "
                f"legacy_total_med={row['legacy']['total']['median']:.6f}s  "
                f"p1_total_med={row['p1']['total']['median']:.6f}s  "
                f"speedup={row['median_total_speedup_legacy_over_p1']:.1f}x",
                flush=True,
            )
            results.append(row)

        if not results:
            print("BLOCKED: no comparison cell completed.", file=sys.stderr)
            return 3

        print_report(results)

        report_path = out_dir / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "python": sys.version,
                    "scipy": scipy.__version__,
                    "pyomo": _pyomo_version(),
                    "bonmin_executable": str(bonmin),
                    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                    "pythonnousersite": os.environ.get("PYTHONNOUSERSITE"),
                    "seed": args.seed,
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "engineering_validation_only": True,
                    # See the module docstring: only a run inside a validated environment
                    # (both arms, one interpreter) is Grade-A same-input evidence.
                    "evidence_class": evidence_class,
                    "bonmin_executable_overridden": cross_environment,
                    "results": results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nreport written: {report_path}")

        mismatches = [r for r in results if r["verdict"] == "MISMATCH"]
        return 1 if mismatches else 0
    finally:
        if temp_dir is not None and args.output_dir is None:
            # Keep the directory: the report is the evidence. Detach the cleanup.
            temp_dir._finalizer.detach()  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
