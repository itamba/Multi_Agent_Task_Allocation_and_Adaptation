"""Launcher + walltime watchdog for the ONE authorized mission-slack development run.

Standard library only. It starts exactly one training process from the frozen measured
worktree, streams its stdout/stderr into ``<out>/training_console.log``, and writes a
machine-readable ``<out>/launcher_record.json`` at start, on every heartbeat and at the end:
process identity, start / end timestamps, walltime, the child's exit code and the
termination reason (``exited`` or ``walltime_cap_exceeded``). On the 24-hour cap it kills
the whole process tree (``taskkill /T /F``) and records that; it never restarts, resumes or
retries anything.

The exit code is written by this Python process (``native_exit_code.txt``), not by a
``cmd`` redirect, so the historical ``echo %RC%> file`` defect cannot recur.

Usage (from anywhere; paths are absolute):

    python launch_with_watchdog.py --worktree C:/gms1src --out C:/gruns/<run_id> \
        --cap-hours 24 -- <training argv after "python">
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat()


def _write(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conda-env", default="nlp_env")
    ap.add_argument("--cap-hours", type=float, default=24.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=300.0)
    ap.add_argument("train_argv", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    train_argv = [a for a in args.train_argv if a != "--"]
    worktree = Path(args.worktree).resolve()
    out = Path(args.out)
    # The training process creates the run directory itself and refuses an existing one
    # only through its own logic; the launcher keeps its files beside it so it never
    # writes INTO a directory the trainer has not yet validated.
    side = Path(str(out) + "__launcher")
    if out.exists() or side.exists():
        print("refusing: output or launcher directory already exists", file=sys.stderr)
        return 3
    side.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONNOUSERSITE"] = "1"
    cmd = ["conda", "run", "-n", args.conda_env, "--no-capture-output", "python"] + train_argv
    record = {
        "record": "launcher_record", "record_version": 1,
        "cwd": str(worktree), "env": {"PYTHONPATH": env["PYTHONPATH"],
                                      "PYTHONNOUSERSITE": "1"},
        "command": cmd, "output_dir": str(out), "launcher_dir": str(side),
        "cap_hours": args.cap_hours, "started_at": _now(),
        "launcher_pid": os.getpid(), "child_pid": None, "status": "starting",
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
    while True:
        try:
            rc = proc.wait(timeout=args.heartbeat_seconds)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            record.update(heartbeat_at=_now(), elapsed_seconds=elapsed)
            _write(rec_path, record)
            if elapsed > cap_s:
                reason = "walltime_cap_exceeded"
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True)
                try:
                    rc = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    rc = None
                break
    log.close()
    record.update(status="finished", termination_reason=reason, exit_code=rc,
                  ended_at=_now(), walltime_seconds=time.monotonic() - t0)
    _write(rec_path, record)
    (side / "native_exit_code.txt").write_text("%s\n" % (rc,), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
