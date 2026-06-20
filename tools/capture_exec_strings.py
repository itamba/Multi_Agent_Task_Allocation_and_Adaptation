"""
WI-3 byte-identical harness.

Captures every non-empty BLADE command string emitted by BladeExecutorMinimal on the
FIXED 4v5 scenario, so the WI-3 decoupling refactor can be verified byte-for-byte.

This mirrors run_validation_episode() in train_full.py (manual pre-launch + settle, then
the executor loop with the same RTB termination), but uses the *fixed* scenario JSON
(data/scenarios/strike_training_4v5.json) so unit IDs are identical before and after the
refactor (ScenarioGenerator assigns fresh uuid4 IDs per run, which would make a cross-run
diff meaningless).

Run via nlp_env (the base env lacks bonmin and fails silently):
    conda run -n nlp_env python tools/capture_exec_strings.py tools/exec_strings_before.txt
    # ... after the refactor ...
    conda run -n nlp_env python tools/capture_exec_strings.py tools/exec_strings_after.txt
    # then diff the two files -- must be byte-identical.

Throwaway verification tool; not part of the product, not committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import train_full` works
sys.path.insert(0, str(ROOT / "src"))  # so match_aou.* import works

import train_full as tf  # noqa: E402  (also sets up sys.path / BLADE imports)
from match_aou.utils.blade_utils import create_agents_from_scenario  # noqa: E402
from match_aou.utils.blade_utils.scenario_factory import (  # noqa: E402
    _normalize_side_color,
    generate_all_enemy_tasks,
)
from match_aou.utils.blade_utils.blade_executor_minimal import BladeExecutorMinimal  # noqa: E402

SCENARIO = str(ROOT / "data" / "scenarios" / "strike_training_4v5.json")
MAX_TICKS = 14400


def main(out_path: str, nn_ordering: bool = True) -> None:
    game, env, observation = tf.setup_blade_env(
        SCENARIO, max_steps=MAX_TICKS, recording_dir=str(ROOT / "tools")
    )

    # Mirror validation: current side = BLUE
    for side in observation.sides:
        if str(getattr(side, "name", "")).upper() == "BLUE":
            game.current_side_id = side.id
            break

    agents_by_side = create_agents_from_scenario(observation)
    attacking_agents = agents_by_side.get(tf.ATTACKING_SIDE_COLOR, [])
    all_tasks = generate_all_enemy_tasks(observation, tf.ATTACKING_SIDE_COLOR)

    solution, tasks_filtered, _ = tf.solve_match_aou(
        attacking_agents, all_tasks, tf.SOLVER_NAME
    )
    if not solution:
        Path(out_path).write_text("EMPTY_SOLUTION\n", encoding="utf-8")
        print("EMPTY_SOLUTION", file=sys.stderr)
        return

    # Record the (normalized) solution for provenance, in a sidecar file so the strings
    # file itself stays byte-comparable across runs.
    sol_norm = {
        aid: sorted(tuple(int(x) for x in a) for a in assigns)
        for aid, assigns in sorted(solution.items())
    }
    Path(out_path + ".solution.txt").write_text(
        "\n".join(f"{aid}: {sol_norm[aid]}" for aid in sol_norm) + "\n",
        encoding="utf-8",
    )

    emitted = []  # (tick, action) for every non-empty emitted string

    # --- Launch aircraft (mirror run_validation_episode, no recording) ---
    for _ in range(5):
        observation, _, _, _, _ = env.step("")

    for airbase in getattr(observation, "airbases", []) or []:
        if _normalize_side_color(getattr(airbase, "side_color", "")) != tf.ATTACKING_SIDE_COLOR:
            continue
        ab_id = str(airbase.id)
        for _ac in list(getattr(airbase, "aircraft", []) or []):
            observation, _, _, _, _ = env.step(f"launch_aircraft_from_airbase('{ab_id}')")

    for _ in range(10):
        observation, _, _, _, _ = env.step("")

    # --- Executor loop (full plan, oracle only) ---
    executor = BladeExecutorMinimal(
        tasks=tasks_filtered,
        solution=solution,
        agents=attacking_agents,
        add_return_to_base=True,
        arrival_threshold_km=50.0,
        nn_ordering=nn_ordering,
    )

    agent_ids = [str(a.id) for a in attacking_agents]
    returned: set = set()

    for tick in range(MAX_TICKS):
        try:
            action = executor.next_action(observation, fallback_tick=tick) or ""
        except ValueError:
            action = ""

        if action:
            emitted.append((tick, action))

        observation, _, terminated, truncated, _ = env.step(action)

        airborne_ids = {
            str(getattr(ac, "id", "")) for ac in getattr(observation, "aircraft", []) or []
        }
        for aid in agent_ids:
            if aid not in returned and aid not in airborne_ids:
                returned.add(aid)

        if terminated or truncated:
            break
        if tick > 100 and len(returned) == len(agent_ids):
            break

    lines = [f"{tick}\t{action}" for tick, action in emitted]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} emitted strings to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tools" / "exec_strings.txt")
    # Optional second arg toggles WI-4 nearest-neighbor ordering (default True = product default).
    nn = True
    if len(sys.argv) > 2:
        nn = sys.argv[2].strip().lower() not in {"0", "false", "off", "no"}
    main(out, nn_ordering=nn)
