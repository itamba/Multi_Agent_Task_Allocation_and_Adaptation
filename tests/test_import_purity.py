"""
Import-purity test (CLEANUP Phase 2, Step 2).

Absolute criterion: importing any GRAPH entry module in a FRESH interpreter must
NOT drag in ANY flat-only ("old model") module. After Step 1 severed the last
source-level graph->flat import, the only remaining leak vectors were the
package-level re-exports in the four rl/*/__init__.py and the two dead eager
imports in utils/blade_utils/__init__.py. Step 2 stripped all of them; this test
locks the result and guards against future regressions.

For each entry module E we spawn a clean `python -c "import E"` subprocess with
PYTHONPATH=src (so match_aou.* resolves) and assert:
  (a) POSITIVE control: E itself is in the child's sys.modules (the import ran).
  (b) ABSOLUTE criterion: NONE of the DENY (flat-only) modules is in sys.modules.

The denylist is explicit (not a heuristic) so it stays maintainable and
Step-3-proof: once Step 3 deletes these files the denylist entries simply become
"never imported" (absent modules trivially pass) and the test keeps guarding.

Fast by design: module imports only -- no episodes, no solver, no env.reset.

Run: python -m pytest tests/test_import_purity.py -v
     python tests/test_import_purity.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest

    _parametrize = pytest.mark.parametrize
except ImportError:  # standalone mode (python tests/test_import_purity.py)
    def _parametrize(_argname, _argvalues):
        def _decorator(func):
            return func

        return _decorator

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# --- Graph entry modules: the public import surfaces a consumer would touch. ---
ENTRY_MODULES = [
    "match_aou.rl.training.graph_rollout",
    "match_aou.rl.training.graph_tick_loop",
    "match_aou.rl.training.graph_episode_setup",
    "match_aou.rl.training.graph_reward",
    "match_aou.rl.training.belief",
    "match_aou.rl.observation.graph_builder",
    "match_aou.rl.agent.graph_encoder",
    "match_aou.rl.action.graph_action",
    "match_aou.rl.action.graph_effect",
    "match_aou.rl.action.graph_trigger",
    "match_aou.utils.blade_utils.blade_graph_executor",
    "match_aou.utils.blade_utils.scenario_generator",
]

# --- Flat-only ("old model") modules: none of these may appear in the closure
#     of any graph entry module. Explicit denylist (Step-3-proof). ---
DENY_MODULES = [
    "match_aou.rl.plan_editor",
    "match_aou.rl.training.ppo_trainer",
    "match_aou.rl.training.rollout_buffer",
    "match_aou.rl.training.reward",
    "match_aou.rl.training.fuel_damage",
    "match_aou.rl.agent.network",
    "match_aou.rl.action.action_config",
    "match_aou.rl.action.action_validation",
    "match_aou.rl.action.action_utils",
    "match_aou.rl.observation.observation_builder",
    "match_aou.rl.observation.target_extraction",
    "match_aou.rl.observation.plan_parsing",
    "match_aou.rl.observation.plan_context",
    "match_aou.rl.observation.self_features",
    "match_aou.rl.observation.observation_types",
    "match_aou.rl.observation.config",
    "match_aou.rl.observation.observation_utils",
    "match_aou.utils.blade_utils.blade_plan_utils",
    "match_aou.utils.blade_utils.observation_utils",
]

# Child prints one sentinel line so library warnings on stdout can't corrupt it.
_SENTINEL = "PURITY_JSON:"
_CHILD = (
    "import sys, json, importlib\n"
    "E = sys.argv[1]\n"
    "importlib.import_module(E)\n"
    "present = sorted(m for m in sys.modules if m.startswith('match_aou'))\n"
    "print('%s' + json.dumps({'entry': E in sys.modules, 'modules': present}))\n"
    % _SENTINEL
)


def _closure_of(entry_module: str) -> dict:
    """Import `entry_module` in a fresh interpreter; return {entry, modules}."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)  # so match_aou.* resolves (the validated vector)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, entry_module],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, (
        f"import of {entry_module} failed (rc={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    line = next(
        (l for l in proc.stdout.splitlines() if l.startswith(_SENTINEL)), None
    )
    assert line is not None, (
        f"no sentinel line for {entry_module}.\nSTDOUT:\n{proc.stdout}"
    )
    return json.loads(line[len(_SENTINEL):])


@_parametrize("entry_module", ENTRY_MODULES)
def test_graph_entry_imports_no_flat_module(entry_module: str) -> None:
    """A fresh import of a graph entry module pulls in ZERO flat-only modules."""
    result = _closure_of(entry_module)

    # (a) positive control: the entry module actually imported.
    assert result["entry"], f"{entry_module} not in child sys.modules"

    # (b) absolute criterion: no flat-only module in the closure.
    present = set(result["modules"])
    leaked = [m for m in DENY_MODULES if m in present]
    assert not leaked, (
        f"{entry_module} leaked {len(leaked)} flat-only module(s): {leaked}"
    )


if __name__ == "__main__":
    failures = 0
    for E in ENTRY_MODULES:
        res = _closure_of(E)
        leaked = [m for m in DENY_MODULES if m in set(res["modules"])]
        ok = res["entry"] and not leaked
        print(f"{'OK ' if ok else 'FAIL'} {E}"
              + ("" if ok else f"  leaked={leaked} entry={res['entry']}"))
        failures += 0 if ok else 1
    if failures:
        print(f"IMPORT-PURITY: {failures} entry module(s) FAILED")
        sys.exit(1)
    print(f"IMPORT-PURITY: all {len(ENTRY_MODULES)} entry modules clean "
          f"(0 flat-only modules in any closure)")
