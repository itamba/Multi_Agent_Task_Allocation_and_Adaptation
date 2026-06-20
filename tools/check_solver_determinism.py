"""
WI-3 byte-identity support: prove solve_match_aou is deterministic on the fixed 4v5 fixture.

Since the before-state solution can't be re-captured mid-refactor, we instead show the solver
returns the SAME solution dict on repeated runs over identical input. The WI-3 refactor changes
none of the solver's inputs (location / capabilities / probability / utility are unchanged; effort
is not consumed by the MINLP; step_kind / target_id are not solver inputs), so identical input +
deterministic solve => identical solution => any string diff is attributable to the executor alone.

Run via nlp_env:
    conda run -n nlp_env python tools/check_solver_determinism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import train_full as tf  # noqa: E402
from match_aou.utils.blade_utils import create_agents_from_scenario  # noqa: E402
from match_aou.utils.blade_utils.scenario_factory import generate_all_enemy_tasks  # noqa: E402

SCENARIO = str(ROOT / "data" / "scenarios" / "strike_training_4v5.json")


def _normalize(solution) -> str:
    norm = {
        str(aid): sorted(tuple(int(x) for x in a) for a in assigns)
        for aid, assigns in solution.items()
    }
    return "\n".join(f"{aid}: {norm[aid]}" for aid in sorted(norm))


def _solve_once() -> str:
    _game, _env, observation = tf.setup_blade_env(
        SCENARIO, max_steps=tf.MAX_SIM_TICKS, recording_dir=str(ROOT / "tools")
    )
    agents_by_side = create_agents_from_scenario(observation)
    attacking_agents = agents_by_side.get(tf.ATTACKING_SIDE_COLOR, [])
    all_tasks = generate_all_enemy_tasks(observation, tf.ATTACKING_SIDE_COLOR)
    solution, _tasks_filtered, _ = tf.solve_match_aou(
        attacking_agents, all_tasks, tf.SOLVER_NAME
    )
    return _normalize(solution)


def main() -> None:
    a = _solve_once()
    b = _solve_once()
    print("=== solution run 1 ===", file=sys.stderr)
    print(a, file=sys.stderr)
    if a == b:
        print("SOLVER_DETERMINISTIC: identical solution on both runs")
    else:
        print("SOLVER_NONDETERMINISTIC: solutions differ")
        print("--- run 2 ---")
        print(b)
        sys.exit(2)


if __name__ == "__main__":
    main()
