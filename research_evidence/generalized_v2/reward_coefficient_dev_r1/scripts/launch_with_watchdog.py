"""Launcher + walltime watchdog for ONE arm of the REWARD-01 coefficient comparison.

Derived from the mission-slack task's ``launch_with_watchdog.py``. Standard library only. It
starts exactly one training process from the frozen measured worktree, streams its stdout /
stderr into ``<out>__launcher/training_console.log`` and keeps
``<out>__launcher/launcher_record.json`` current (start, every heartbeat, end): process
identity, timestamps, walltime, the child's exit code and the termination reason.

  * ENVIRONMENT PROBE: before the training process starts, ``env_probe.py`` runs under the
    SAME conda env, PYTHONPATH and environment block and writes ``env_probe.json`` (package
    versions, device, thread settings). It solves and trains nothing.
  * WALLTIME: the hard cap (4 h per arm by plan) is enforced to the heartbeat; on the cap the
    whole process tree is killed (``taskkill /T /F``) and ``walltime_cap_exceeded`` recorded.
  * PRE-UPDATE MONITOR (arm B only, ``--reference-run``): once arm B's pre_update round is
    complete, ``compare_pre_update.compare`` checks it against arm A's recorded round, ONCE.
    The report is kept in ``pre_update_identity.json``; if the pre-declared rule says STOP,
    the process tree is killed and ``pre_update_mismatch_stop`` recorded.

It never restarts, resumes or retries anything, and writes the exit code itself
(``native_exit_code.txt``), never through a ``cmd`` redirect.

Usage:

    python launch_with_watchdog.py --worktree C:/grc1src --out C:/gruns/<run> --cap-hours 4 \
        [--reference-run C:/gruns/<arm A run>] -- <training argv after "python">
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_pre_update  # noqa: E402


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
    ap.add_argument("--reference-run", default=None)
    ap.add_argument("--conda-env", default="nlp_env")
    ap.add_argument("--cap-hours", type=float, default=4.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0)
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
    reference = Path(args.reference_run) if args.reference_run else None

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONNOUSERSITE"] = "1"
    conda = ["conda", "run", "-n", args.conda_env, "--no-capture-output", "python"]
    probe = subprocess.run(conda + [str(worktree / "research_evidence/generalized_v2/"
                                         "reward_coefficient_dev_r1/scripts/env_probe.py"),
                                    str(side / "env_probe.json")],
                           cwd=str(worktree), env=env, capture_output=True, text=True)
    if probe.returncode != 0 or not (side / "env_probe.json").exists():
        (side / "env_probe_failure.txt").write_text(probe.stdout + "\n" + probe.stderr,
                                                    encoding="utf-8")
        print("refusing: environment probe failed", file=sys.stderr)
        return 4

    cmd = conda + train_argv
    record = {
        "record": "launcher_record", "record_version": 3,
        "cwd": str(worktree), "env": {"PYTHONPATH": env["PYTHONPATH"],
                                      "PYTHONNOUSERSITE": "1"},
        "command": cmd, "output_dir": str(out), "launcher_dir": str(side),
        "reference_run": None if reference is None else str(reference),
        "cap_hours": args.cap_hours, "heartbeat_seconds": args.heartbeat_seconds,
        "started_at": _now(), "launcher_pid": os.getpid(), "child_pid": None,
        "status": "starting", "pre_update_check": "not_applicable" if reference is None
        else "pending", "monitor_errors": 0,
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

    def pre_update_monitor() -> bool:
        """Run the pre-declared comparison once arm B's round exists; True means STOP."""
        if record["pre_update_check"] != "pending":
            return False
        try:
            rep = compare_pre_update.compare(out, reference)
        except Exception:  # a monitor fault is recorded, never silently ignored
            record["monitor_errors"] += 1
            with open(side / "pre_update_monitor_errors.log", "a", encoding="utf-8") as fh:
                fh.write("%s\n%s\n" % (_now(), traceback.format_exc()))
            return False
        if rep is None:
            return False
        rep["checked_at"] = _now()
        rep["elapsed_seconds_at_check"] = time.monotonic() - t0
        _write(side / "pre_update_identity.json", rep)
        record["pre_update_check"] = "stop" if rep["stop"] else "pass"
        return bool(rep["stop"])

    rc = None
    while True:
        remaining = cap_s - (time.monotonic() - t0)
        try:
            rc = proc.wait(timeout=max(1.0, min(args.heartbeat_seconds, remaining)))
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            stop = reference is not None and pre_update_monitor()
            record.update(heartbeat_at=_now(), elapsed_seconds=elapsed)
            _write(rec_path, record)
            if stop:
                reason = "pre_update_mismatch_stop"
            elif elapsed >= cap_s:
                reason = "walltime_cap_exceeded"
            if reason != "exited":
                _kill_tree(proc.pid)
                try:
                    rc = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    rc = None
                break
    log.close()
    if reason == "exited" and reference is not None and record["pre_update_check"] == "pending":
        pre_update_monitor()           # a run that finished between heartbeats
    record.update(status="finished", termination_reason=reason, exit_code=rc,
                  ended_at=_now(), walltime_seconds=time.monotonic() - t0)
    _write(rec_path, record)
    (side / "native_exit_code.txt").write_text("%s\n" % (rc,), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
