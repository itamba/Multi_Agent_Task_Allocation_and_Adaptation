# Execution environments and authorized cleanup

> **Read this when** you set up or use an environment, run anything that solves or simulates,
> work on the BGU cluster, or retire branches, worktrees or other refs.
>
> **Status:** §1 and §2 are normative (§1 moved verbatim from the former `CLAUDE.md` §1 at base
> `ae42cb01677f94868b2873008d87be677e31f0c8`). §3 is a **dated observation**, not live state.
> §4 is a normative cleanup procedure plus a registry verified on 2026-09-14. A dated
> environment validation says what was observed then; it is not a claim about any environment
> now.

## 1. Execution contexts

- **Environment — TWO VALIDATED EXECUTION CONTEXTS, and BOTH are CURRENT.** The LOCAL
  Windows context is the historical scientific execution context, and it is unchanged.
  *(Dated: at the 2026-08-31 cluster validation, the then-approved measurements were
  recorded as LOCAL. Later measurements are checked from their own evidence: the preserved
  GENERALIZED-V2 development arms record Windows and `nlp_env` in their `run_config.json`,
  while the fresh deterministic-P1 measurement records only its measured code SHA, so its
  execution environment is not inferred.)* The BGU Slurm cluster context beside it was validated LATER,
  against exact `main` SHA `926aba66fcaf2b99fc58685eb202888d8deeaf5f`. **Neither supersedes
  the other.** Establish which context you are in FIRST, then use only that context's
  commands — the two have different environment names, different interpreters and different
  isolation rules, and mixing them silently produces a run in the wrong environment.
- **LOCAL context:** Windows + PyCharm terminal + `nlp_env` conda env, Python 3.10+. Avoid POSIX-only idioms; use Python/PowerShell equivalents. Run from repo root.
- **LOCAL — `pytest` is NOT installed in `nlp_env`** — it lives in the base env, so `python -m pytest` under `nlp_env` fails with `No module named pytest`. Solver-free suites can be run with the base-env `pytest`; test files carrying a `__main__` runner (e.g. `tests/test_graph_train.py`) should ALSO be run directly under `nlp_env`.
- **LOCAL — `blade` and `gymnasium` DO resolve in the base env** (measured at `dd14ab4`): `import blade` returns the SAME vendored fork the editable install points at (`src/match_aou/integrations/panopticon-main/gym/blade/__init__.py`), and `gymnasium` imports cleanly. This CLOSES the former "is the base env the same fork?" question — a base-env test may build a `Game`, `gymnasium.make("blade/BLADE-v0", …)` and `env.reset()`, which is what lets `tests/test_graph_setup_seam.py`'s BLADE tier run under plain `pytest`. **The missing base-env dependency is BONMIN, not BLADE**: `shutil.which("bonmin")` is `None` in the base env and `…/envs/nlp_env/Library/bin/bonmin.EXE` under `nlp_env`. Nothing here relaxes the solver rule below — anything that SOLVES still runs under `nlp_env`.
- **🛑 LOCAL — Solver/bonmin commands MUST run under `nlp_env`** (`conda run -n nlp_env ...`; add `--no-capture-output` to avoid Windows cp1255 re-encode crashes on Unicode prints). The base env lacks `bonmin` and fails **silently** (exits 0). **Never trust the exit code alone** — verify no `CRASH`/`Traceback` and that the run actually solved before claiming success.
- **BGU CLUSTER context:** Linux + Slurm + the `graph_rl_cluster` conda env, **Python
  3.12.14**, validated against exact `main` SHA
  `926aba66fcaf2b99fc58685eb202888d8deeaf5f` with a clean working tree. Its DIRECT
  dependency surface is owned by **`environment.cluster.yml`** (conda-forge only, no
  defaults) — NumPy 1.26.4, SciPy 1.17.1, PyTorch 2.13.0 (**CPU build**), Pyomo 6.10.1,
  `coin-or-bonmin` 1.8.9, Gymnasium 0.29.1, Shapely 2.0.6, Haversine 2.9.0. The vendored
  BLADE engine is installed EDITABLE from
  `src/match_aou/integrations/panopticon-main/gym`, and `import blade` was confirmed to
  resolve to that vendored fork's `blade/__init__.py`. **The validated environment is
  CPU-only: GPU execution is NOT validated and is NOT required by this record**, so do
  not infer a GPU requirement, a CUDA dependency or a GPU-shaped run from it.
- **🛑 CLUSTER — every cluster validation and scientific command MUST set
  `PYTHONNOUSERSITE=1`.** This is LOAD-BEARING, not hygiene: without it an unrelated
  user-site PyTorch under `~/.local` was OBSERVED to SHADOW the conda environment, so the
  run would import a torch the environment never declared. With user-site disabled,
  PyTorch resolves inside `graph_rl_cluster`. Example:
  `PYTHONNOUSERSITE=1 conda run -n graph_rl_cluster --no-capture-output python -m ...`.
- **🛑 CLUSTER — solver/bonmin commands run under `graph_rl_cluster`** (the exact
  counterpart of the LOCAL `nlp_env` rule above, and it does NOT relax it). Validated at
  that SHA: the project imports succeeded, BLADE resolved from the vendored editable
  checkout, Pyomo's `SolverFactory("bonmin")` reported available, and a small MINLP solved
  through Pyomo → BONMIN with `termination_condition == optimal`. **Never trust the exit
  code alone here either.**
- **What that cluster smoke IS and IS NOT.** It is ENGINEERING / RUNTIME validation of the
  environment — **never scientific evidence, and no result may be drawn from it.** The
  real `graph_train` selftest progressed through real BONMIN allocation, BLADE execution,
  fuel-damage/reward processing and a real PPO update, but the long selftest process was
  **externally terminated**, so it must **NOT** be recorded as a full PASS. When reading
  cluster output, do not convert **expected fixed-cell episode attrition** or **synthetic
  test tracebacks** into environment failures — they are neither.

## 2. Dependency declarations

- [`requirements.txt`](../../requirements.txt) is the Python dependency surface, deliberately
  not an exact lock. The P1 MILP backend needs SciPy's `milp` (HiGHS, SciPy ≥ 1.9); BONMIN is a
  non-pip solver dependency that Pyomo only drives.
- [`environment.cluster.yml`](../../environment.cluster.yml) owns the validated cluster
  environment's direct dependency surface; it is not a transitive lockfile.
- The vendored BLADE engine is installed editable from
  `src/match_aou/integrations/panopticon-main/gym` **without** the `[gym]` extra. Its source
  stays frozen in every context ([`CLAUDE.md` §2](../../CLAUDE.md#2-do-not-touch-without-explicit-discussion)).

## 3. BGU cluster operations observed on 2026-08-31

The following was recorded in the former handoff §3m.6. Its statements about R1 being pending and
about which task was writable were true on that date only; see the
[handoff](../../graph_rl_project_handoff.md) for current state.

**THIS SUBSECTION IS VOLATILE OPERATIONAL STATE, NOT A SOFTWARE CONTRACT.** The durable
environment contract lives in `CLAUDE.md` §1 and in `environment.cluster.yml`; the Slurm
numbers below are **OBSERVED CLUSTER POLICY that may change without notice**, and they are
recorded so a future launch design starts from measurement rather than from guesswork.

**READINESS IS NOT AUTHORIZATION.** Everything here says a job COULD run on the cluster. It
authorizes **no run, no campaign, no launcher and no scientific decision.**

**THE ENVIRONMENT RECORD IS NOW INTEGRATED INTO `main`.** `environment.cluster.yml`, the
`requirements.txt` alignment and the `CLAUDE.md` §1 two-context contract landed through
**PR #46** — reviewed candidate `cbc227450067d96c630eed208e22b3a5a20efc1b` → merge
`e9f9f4f93412c8c6c3dd8ba81a7e784dc52cc68b` (`2026-08-31 16:40:13 +0300`), a NORMAL merge
commit preserving its reviewed candidate as its second parent, with the integrated tree
verified equal to the reviewed tree. **THE VALIDATION SHA BELOW IS A DURABLE OBSERVATION AND
DOES NOT MOVE WITH `main`:** the environment was validated against `926aba66…`, and the later
PR-#46 merge that RECORDED that validation does not retroactively change where it was taken.

**VALIDATED ENVIRONMENT IDENTITY**, observed against exact `main` SHA
`926aba66fcaf2b99fc58685eb202888d8deeaf5f`, with the cluster checkout's `HEAD` equal to
`origin/main` and the working tree CLEAN at the final environment smoke:

- conda env **`graph_rl_cluster`**, Linux / BGU Slurm, rebuilt core from **conda-forge only**;
- **Python 3.12.14**; NumPy 1.26.4; SciPy 1.17.1; **PyTorch 2.13.0, CPU build**; Pyomo 6.10.1;
  **`coin-or-bonmin` 1.8.9**; Gymnasium 0.29.1; Shapely 2.0.6; Haversine 2.9.0;
- the direct surface is owned by **`environment.cluster.yml`**, which is a small reference
  environment and deliberately **NOT** a full transitive lockfile;
- **the validated environment is CPU-only. GPU execution is NOT validated and NOT required**,
  so no GPU requirement may be inferred from this record.

**VENDORED BLADE — EDITABLE, FROM THE REPOSITORY.** Installed editable from
`src/match_aou/integrations/panopticon-main/gym`, and `import blade` was confirmed to resolve
to that vendored fork's `blade/__init__.py`. BLADE's `setup.py` pins `shapely==2.0.6`, which
the cluster environment matches exactly. **BLADE stays FROZEN** (`CLAUDE.md` §2) and was not
modified.

**🛑 `PYTHONNOUSERSITE=1` IS MANDATORY FOR EVERY CLUSTER VALIDATION AND SCIENTIFIC
COMMAND, AND IT IS LOAD-BEARING.** Without it, an unrelated user-site PyTorch under
`~/.local` was **OBSERVED to SHADOW** the conda environment — the run would import a torch
the environment never declared. With user-site disabled, PyTorch resolved inside
`graph_rl_cluster`. This is the single most easily lost fact in this subsection.

**ENVIRONMENT SMOKE — ENGINEERING / RUNTIME VALIDATION, NEVER SCIENTIFIC EVIDENCE.** Observed:
project imports succeeded; BLADE resolved from the vendored editable checkout; Pyomo's
`SolverFactory("bonmin")` reported available; and a small MINLP solved through
Pyomo → BONMIN with `termination_condition == optimal`. The real `graph_train` selftest
progressed through real BONMIN allocation, BLADE execution, fuel-damage / reward processing
and a real PPO update **before the long selftest process was EXTERNALLY TERMINATED** — so it
**MUST NOT be recorded as a full PASS**, and no completion may be inferred from it.
**No reward, learning, convergence, attrition or performance claim may be drawn from any of
this**, and it is emphatically not a measurement.

**TWO READING RULES FOR CLUSTER OUTPUT, both of which prevent a false alarm.** Do **NOT**
convert **expected fixed-cell episode attrition** into an environment failure — B2
exact-cardinality and fuel-window failures are EXPECTED SCIENTIFIC OUTCOMES of the current
contract (`CLAUDE.md` §8). Do **NOT** convert **synthetic test tracebacks** into an
environment failure either — several suites deliberately INJECT faults and print tracebacks
on the passing path.

**OBSERVED SLURM LIMITS — `course` ACCOUNT / QoS, OBSERVED 2026-08-31, VOLATILE.** Recorded as
a handoff for a future launch design, and explicitly **not** a permanent compatibility
guarantee:

| Observed | Value |
|---|---|
| usable account / QoS | `course` |
| `course` QoS `MaxWall` | `1-00:00:00` (24 h per job) |
| `MaxTRESPU` | `cpu=66`, `gres/gpu=1`, `mem=64G` |
| `course` partition `MaxMemPerCPU` | 4096 MB |
| `course` partition `JobDefaults` | `DefCpuPerGPU=6` |
| `course` partition `MaxNodes` | 1 |

**`sinteractive` — INSPECTED, NO CHANGE REQUIRED.** It does **not** itself rewrite a 1-CPU
request into 6 CPUs; **`DefCpuPerGPU=6` explains the 6-CPU allocation when one GPU is
requested.** **No change to the global `sinteractive` wrapper is required.** It does create a
temporary `interactive.sbatch` in its current working directory, **so launch it OUTSIDE the
repository** to avoid dropping a stray file into the tree.

**NO COMPUTE-NODE PERFORMANCE BENCHMARK IS REQUIRED AS A CLOSURE GATE.** Existing engineering
evidence already identifies solver / runtime dominance sufficiently for the current planning
decision (§3m.3: BONMIN dominated runtime, `A4-high` showed very large variance, and one
legitimate training solve of roughly 998 s terminated `optimal` — which is exactly why **no
short solver timeout was adopted**). A benchmark may still be chosen later as a design input;
it is **not** a prerequisite for closing this environment record.

**WHAT STILL DOES NOT EXIST ON THE CLUSTER.** **No scientific `sbatch` script, no job array,
no launcher, no queue/partition/walltime decision and no monitoring runbook** — none is
written, designed, reviewed or authorized, and **none may be invented from this record.**
*(Dated note: on 2026-08-31 the actor-only GENERALIZED-V1 R1 was a LOCAL run awaiting its result;
it has since been reviewed — see [`measurements.md`](../history/measurements.md#1-run-registry).)*

## 4. Authorized cleanup

### 4.1 Rules

- Cleanup happens **only with explicit user authorization** for the specific refs, as its own
  task, never bundled into another.
- **Safe deletion only:** delete a branch only after verifying that its tip is reachable from
  `main` or from its merged PR's `refs/pull/<n>/head`. Never force-delete unreachable work, and
  never delete a branch whose PR is open.
- A documentation branch becomes cleanup-eligible only after its own candidate is reviewed and
  integrated.
- Protected refs (§4.2) and preserved artifacts (§4.4) are **never** cleanup-eligible and never
  move.

### 4.2 Protected refs

Verified against `origin` on 2026-09-14. Their roles are distinct and never interchangeable.

| Ref | Commit | Role |
|---|---|---|
| `main` | resolve live | integration branch |
| `phase-a-baseline` | `4f0068847b017795717c5f0e331f647bcfc30547` | code state of the original Phase-A reference |
| `pre-ctde-actor-only` | `d437084c5fb1a22c21596a48c58e03f7e15a0115` | the actor-only state Phase-B CTDE was merged onto (first parent of the CTDE merge) |
| `flat-final` | `4d44c3454a5561a6cb9d7aed593d59a40068d6d7` | the retired flat-RL path |
| tag `pre-cleanup` | tag object `cce4e1e6c08878340e64543fa612a4435e11ae17`, peels to `561b7cb7f2d873e584a8c0dabe71df8050f1b4ed` | the last commit before the flat-path cleanup |

### 4.3 Evidence refs

Leave untouched; the user deferred decisions about them.

| Ref | Head | PR |
|---|---|---|
| `evidence/generalized-v2-actor-only-dev-r1` | `1375a881637a9a32721a1630f598adc571422a47` | #61 (draft) |
| `evidence/generalized-v2-ctde-dev-r1` | `b2bbe7a6235c3b9255106826cfb268af7e73f72d` | #62 (draft) |

### 4.4 Preserved run directories and external artifacts

**Protect the originals:** never modify, move, delete, regenerate or normalize them. Authorized
non-destructive copies, lossless packaging and reconstruction checks leave the originals untouched
and are allowed ([`experiments.md` §5](experiments.md#5-evidence-preservation)). Locations are as
recorded in the repository or in a run's own `run_config.json`; **a recorded location is not a
verification that the path still exists**, and where nothing is recorded nothing is invented.

| Artifact | Location as recorded |
|---|---|
| first final-cell short probe | `training_output_20260815_173029` |
| corrected-cell short probe | `training_output_20260816_162130` |
| first long baseline (inconclusive) | `training_output_long_baseline_100x8_seed0` |
| Phase-A long baseline (valid) | `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` |
| FD-VARIABLE-SEVERITY-v1 baseline (valid) | `C:\Users\Itama\f7r2` |
| FD-VARIABLE-SEVERITY-v1 precursor (invalid) | `…\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640` |
| GENERALIZED-V1 R1 run tree and diagnostic bundle | location not recorded; bundle SHA-256 `812ff43322e134e9a7ca31720007393ff1220ba50c35955b2a724b30d4d5d792` |
| GENERALIZED-V2 benchmark manifest (external) | `C:/Users/Itama/PycharmProjects/graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0/benchmark_manifest.json`, SHA-256 `dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103` — as recorded in PR #62's `artifact_sha256.txt` |
| GENERALIZED-V2 development R1 — actor-only | `C:\Users\Itama\PycharmProjects\graph_rl_v2_actor_only_dev_r1_seed3000000_ae42cb0` — `train_config.output_dir` in the run's `run_config.json` (PR #61 head) |
| GENERALIZED-V2 development R1 — CTDE | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_r1_seed3000000_ae42cb0` — `train_config.output_dir` in the run's `run_config.json` (PR #62 head) |

### 4.5 Retired branches observed on `origin` on 2026-09-14

Cleanup-eligible only after the §4.1 verification and explicit authorization.

| Branch | Tip | Merged by |
|---|---|---|
| `task/generalized-v2-benchmark-doc-lock` | `6cbc4a60d3022b51776a4c027bdfc55beedbadac` | PR #60 |
| `task/generalized-v2-benchmark-evaluation` | `786e8218a00954f7a7f20fe1dfca93ec71a400d4` | PR #59 |

### 4.6 Local worktrees observed on 2026-09-14

These detached worktrees exist on the user's machine. Only the FD-VARIABLE-SEVERITY-v1 snapshot's
role is recorded; do not remove any of them without a user decision.

| Path | Commit |
|---|---|
| `C:/Users/Itama/PycharmProjects/fd_variable_severity_v1_bf1e045f_snapshot` | `bf1e045` — the pinned measurement snapshot |
| `C:/g1src` | `4af6c5a` |
| `C:/p1src` | `8f0d250` |
| `C:/Users/Itama/ct1s` | `76abdc4` |
| `C:/Users/Itama/PycharmProjects/flat-baseline` | `4d44c34` (`flat-final`) |

### 4.7 Cleanup already performed

- After PR #11 and PR #12: `task/repo-code-hygiene`, `task/repo-doc-hygiene` and
  `task/final-cell-visual-artifacts` were deleted locally and remotely by safe deletion.
- Before the Phase-A rerun: `task/roster-world-truth-fix`, `task/long-baseline-execution` and
  `task/roster-world-truth-doc-lock` were deleted after their tips were verified reachable;
  `task/variable-fd-severity-baseline` and `task/variable-fd-severity-doc-lock` were later
  observed absent from the remote.
- The branches the former handoff listed as cleanup-eligible after PR #33 through PR #58 are no
  longer present on `origin` as of 2026-09-14; who removed them, and when, is not recorded.
