"""Launcher + walltime watchdog + live prefix monitor for the ONE authorized credit-to-update
diagnostic (derived from the mission-slack task's ``launch_with_watchdog.py``).

Standard library only (``compare_prefix`` is too). It starts exactly one training process from
the frozen measured worktree, streams its stdout / stderr into ``<out>__launcher/
training_console.log`` and keeps ``launcher_record.json`` current (start, every heartbeat, end):
process identity, timestamps, walltime, the child's exit code and the termination reason.

  * WALLTIME: on the hard cap (4 h by plan) it kills the whole process tree
    (``taskkill /T /F``) and records ``walltime_cap_exceeded``.
  * PREFIX MONITOR: every heartbeat it runs the pre-declared ``compare_prefix.compare`` of the
    live run against the original; a ``policy_side`` first divergence (the declared STOP rule)
    kills the process tree and records ``prefix_policy_divergence_stop``. The latest report is
    kept in ``prefix_monitor.json``; the first divergence, once seen, is kept in
    ``prefix_first_divergence.json`` and never overwritten.

It never restarts, resumes or retries anything, and writes the exit code itself
(``native_exit_code.txt``), never through a ``cmd`` redirect.

Usage:

    python launch_with_watchdog.py --worktree C:/gcud1src --out C:/gruns/<run_id> \
        --original C:/gruns/<original_run> --cap-hours 4 -- <training argv after "python">
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_prefix  # noqa: E402


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat()


def _write(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--conda-env", default="nlp_env")
    ap.add_argument("--cap-hours", type=float, default=4.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=120.0)
    ap.add_argument("train_argv", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    train_argv = [a for a in args.train_argv if a != "--"]
    worktree = Path(args.worktree).resolve()
    out = Path(args.out)
    side = Path(str(out) + "__launcher")
    if out.exists() or side.exists():
        print("refusing: output or launcher directory already exists", file=sys.stderr)
        return 3
    side.mkdir(parents=True)
    original = Path(args.original)
    t_load = time.monotonic()
    orig_cache = compare_prefix.load_original(original)
    load_seconds = time.monotonic() - t_load

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONNOUSERSITE"] = "1"
    cmd = ["conda", "run", "-n", args.conda_env, "--no-capture-output", "python"] + train_argv
    record = {
        "record": "launcher_record", "record_version": 2,
        "cwd": str(worktree), "env": {"PYTHONPATH": env["PYTHONPATH"],
                                      "PYTHONNOUSERSITE": "1"},
        "command": cmd, "output_dir": str(out), "launcher_dir": str(side),
        "original_run_dir": str(original), "original_load_seconds": load_seconds,
        "cap_hours": args.cap_hours, "heartbeat_seconds": args.heartbeat_seconds,
        "started_at": _now(), "launcher_pid": os.getpid(), "child_pid": None,
        "status": "starting", "monitor_checks": 0, "monitor_errors": 0,
    }
    rec_path = side / "launcher_record.json"
    _write(rec_path, record)
    (side / "invocation_start_local.txt").write_text(record["started_at"] + "\n",
                                                     encoding="utf-8")
    log = open(side / "training_console.log", "wb")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(worktree), env=env, stdout=log,
                            stderr=subprocess.STDOUT, shell=False,
                            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    record.update(child_pid=proc.pid, status="running")
    _write(rec_path, record)
    cap_s = args.cap_hours * 3600.0
    reason = "exited"
    first_path = side / "prefix_first_divergence.json"

    def monitor() -> bool:
        """Run the declared comparison; True when the STOP rule fired."""
        try:
            rep = compare_prefix.compare(out, original, original=orig_cache)
        except Exception:  # a monitor fault is recorded, never silently ignored
            record["monitor_errors"] += 1
            (side / "prefix_monitor_errors.log").open("a", encoding="utf-8").write(
                "%s\n%s\n" % (_now(), traceback.format_exc()))
            return False
        record["monitor_checks"] += 1
        rep["checked_at"] = _now()
        _write(side / "prefix_monitor.json", rep)
        if rep["first_divergence"] is not None and not first_path.exists():
            _write(first_path, {"seen_at": _now(), "first_divergence": rep["first_divergence"],
                                "stop": rep["stop"]})
        return bool(rep["stop"])

    rc = None
    while True:
        try:
            rc = proc.wait(timeout=args.heartbeat_seconds)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            stop = monitor()
            record.update(heartbeat_at=_now(), elapsed_seconds=elapsed)
            _write(rec_path, record)
            if stop:
                reason = "prefix_policy_divergence_stop"
            elif elapsed > cap_s:
                reason = "walltime_cap_exceeded"
            if reason != "exited":
                _kill_tree(proc.pid)
                try:
                    rc = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    rc = None
                break
    log.close()
    if reason == "exited":
        monitor()                      # the final state of the completed run
    record.update(status="finished", termination_reason=reason, exit_code=rc,
                  ended_at=_now(), walltime_seconds=time.monotonic() - t0)
    _write(rec_path, record)
    (side / "native_exit_code.txt").write_text("%s\n" % (rc,), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
