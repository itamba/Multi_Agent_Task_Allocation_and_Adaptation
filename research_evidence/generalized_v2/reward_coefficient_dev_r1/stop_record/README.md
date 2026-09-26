# REWARD-01 R1 — arm-B pre-update STOP record

> **Status: STOPPED under the plan's stop rule; awaiting a user decision.** Arm A completed.
> Arm B's single authorized launch attempt was terminated by its own pre-declared monitor. No
> relaunch, repair or replacement has been made. No behavioural or outcome result of either arm
> had been read when this record was written. Not a verdict.

## What happened

| Item | Arm A (c = 2.25) | Arm B (c = 4.5) |
|---|---|---|
| Run directory | `C:\gruns\reward_c225_r1_s3000000_2b57019` | `C:\gruns\reward_c450_r1_s3000000_2b57019` (partial) |
| Measured SHA | `2b570194dea3612f3796999fc589b73d7082ae31` | same |
| Start → end | 2026-09-26 18:43:37 → 21:04:08 (+03:00) | 21:04:35 → 21:06:35 |
| Walltime / exit | 8431 s / 0, `exited` | 120 s / 1, **`pre_update_mismatch_stop`** (process tree killed) |
| Progress | 375 / 375 updates; 3000 / 3008 training attempts (8 accounted `no_fd_eligible_ego`); 960 / 960 evaluation episodes | pre-update round complete (60 / 60); 3 updates recorded, then killed |

Arm A passed the extractor's validity pre-check (provenance, configuration, accounting, failure
classes, console, launcher, non-finite values) before arm B was launched.

## Why the monitor stopped arm B

The rule in `authorized_plan.json:/pre_update_mismatch_rule`, written by this task before
launch, stops on any wake-sequence difference in **ego id** or **tick**, in addition to frozen-world
identity, selected action / leaf and probability. `pre_update_stop_audit.json` (produced by
`scripts/audit_pre_update_stop.py`, read-only, which reproduces the live report exactly)
classifies all 36 mismatches over 161 compared wakes:

- **Frozen-world identity: 0 mismatches.** Member sets identical (60 / 60); FD ego ids equal in
  every member.
- **Selected wake kind, meta-action and leaf: identical at all 161 wakes.**
- **Probabilities: max |Δ| = 1.79e−7**, below the declared 1e−6 tolerance.
- **24 `ego_id` mismatches**, all in `A5-*` / `A6-*` worlds. Those ids are **not stable even within
  arm A**: every one of the 21 affected members shows 16 distinct ego-id sets over arm A's own
  16 rounds of the same frozen world. They are per-episode generated ids, not a world or policy
  identity, so comparing them across runs cannot detect a mismatch. This is a **defect in the
  rule this task wrote**; the repository already records that generated ids are not seed-derived
  ([training and benchmarks §11](../../../../docs/contracts/training_benchmarks.md#11-known-limitations-and-open-items)).
- **12 `tick` mismatches** of ±1 (10) and ±2 (2) ticks, in the same worlds — the simulator
  timing class already recorded for same-seed runs (measurements §18: one-tick wake differences).
  Recorded episode outcomes differ only in `ticks` in 6 members; utility, deaths, confirmations,
  reference terms and every other recorded outcome field agree, and arm B's penalty is exactly
  2 × arm A's in all 60 members.

Against the packet's own criterion — *materially different initial policy probabilities / actions
or incompatible frozen-world inputs* — no such difference is observed. The stop fired because the
pre-declared rule was stricter than that criterion (and, for ego ids, incapable of passing in
these worlds).

## Files

| File | Content |
|---|---|
| `pre_update_stop_audit.json` | the read-only classification above, with source hashes |
| `arm_b_launcher/` | arm B's launcher record, exit code, start time, env probe, live `pre_update_identity.json` |
| `arm_a_launcher/` | arm A's launcher record, exit code, start time, env probe |
| `original_artifacts_sha256.txt` | SHA-256 and size of every file in both run directories and launcher directories at the time of the stop; nothing was moved, modified or deleted |

## Not done, pending decision

No relaunch of arm B (one attempt per arm), no change to the stop rule, no extraction or reading
of either arm's behavioural results, no documentation of a measurement. The comparison as
planned cannot be completed without a new user authorization.
