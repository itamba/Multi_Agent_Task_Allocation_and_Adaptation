# Execution environments and authorized cleanup

> **Read this when** you set up or use an environment, run anything that solves or simulates,
> work on the BGU cluster, or retire branches, worktrees or other refs.
>
> **Status:** §1 and §2 are normative (§1 moved verbatim from the former `CLAUDE.md` §1 at base
> `ae42cb01677f94868b2873008d87be677e31f0c8`). §3 is a **dated observation**, not live state.
> §4 is a normative cleanup procedure plus a registry verified on 2026-09-15. A dated
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

- Cleanup happens **only with explicit user authorization** for the specific refs, worktrees or
  artifacts, as its own task, never bundled into another.
- **Safe deletion of a merged branch:** delete it only after verifying that its exact tip is an
  ancestor of live `main` or reachable from its merged PR's `refs/pull/<n>/head`. Never delete a
  branch whose PR is open, and never force-delete unreachable work except under the next rule.
- **Temporary evidence and review branches are intentionally unmerged transport**
  ([`decisions.md` §1](../history/decisions.md#1-decision-log), 2026-09-15 lifecycle decision).
  One may be deleted only under authorization that names it, and only after all of: its PR is
  closed, not merged, at its verified exact head; the branch is checked out in no worktree and
  has no local-only commits; its source artifacts are in the local archive (§4.4) with their key
  identities reverified; and its reviewed conclusions are durably recorded on `main`. The deletion
  is recorded as branch removal only; nothing is claimed about GitHub's internal retention of
  pull-request refs.
- A documentation branch becomes cleanup-eligible only after its own candidate is reviewed and
  integrated.
- **Worktree removal:** `git worktree remove` **without `--force`**, only after verifying that
  the worktree's HEAD matches its recorded role, that it has no tracked modification and no
  untracked file, that its ignored files hold no run output which removal would delete, and that
  its commit is reachable from `main` or a protected ref. Removing a worktree never deletes its
  commit or a branch it carries. `git worktree prune` is used only for stale administrative
  metadata.
- Protected refs (§4.2) are **never** cleanup-eligible. Preserved artifacts (§4.4) are **never**
  deleted, rewritten, regenerated or normalized, and are relocated only under §4.4's archival
  relocation rule.

### 4.2 Protected refs

Verified against `origin` on 2026-09-15, before and after that day's cleanup (§4.7), with
identical identities. Their roles are distinct and never interchangeable.

| Ref | Commit | Role |
|---|---|---|
| `main` | resolve live | integration branch |
| `phase-a-baseline` | `4f0068847b017795717c5f0e331f647bcfc30547` | code state of the original Phase-A reference (remote-only; no local branch is needed) |
| `pre-ctde-actor-only` | `d437084c5fb1a22c21596a48c58e03f7e15a0115` | the actor-only state Phase-B CTDE was merged onto (first parent of the CTDE merge) |
| `flat-final` | `4d44c3454a5561a6cb9d7aed593d59a40068d6d7` | the retired flat-RL path |
| tag `pre-cleanup` | tag object `cce4e1e6c08878340e64543fa612a4435e11ae17`, peels to `561b7cb7f2d873e584a8c0dabe71df8050f1b4ed` | the last commit before the flat-path cleanup |

### 4.3 Temporary evidence and review refs

**Open now** (observed 2026-09-19 against `origin`, live `main`
`ed33b7e24a652fa00b13c708517012f8b3302496`; each branch exists on `origin` and locally at the same
head). Neither is cleanup-eligible yet: under §4.1 its source run must first be archived with its
key identities reverified. Neither may be modified or closed without explicit authorization.

| PR | Branch | Head | Durable record on `main` | Source run (original location, not archived) | Status |
|---|---|---|---|---|---|
| #71 | `evidence/generalized-v2-semantic-action-actor-only-dev-r1` | `0d136fa89286c4bbd9e89dfb6bd0a3326c70b670` | [`measurements.md` §10](../history/measurements.md#10-generalized-v2-semantic-action-actor-only-development-r1) | `C:\Users\Itama\PycharmProjects\graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37` | open draft; verdict durable on `main`; **not cleanup-eligible** until the source run is archived |
| #73 | `evidence/generalized-v2-semantic-action-ctde-dev-r1` | `ad9b545034670a7c7a9d8ff98012d56c0be07f46` | [`measurements.md` §16](../history/measurements.md#16-generalized-v2-semantic-action-ctde-development-r1--retrospective-review-closure), made durable by the 2026-09-19 closure documentation | `C:\Users\Itama\PycharmProjects\graph_rl_v2_semantic_action_ctde_dev_r1_seed3000000_8056266` | open draft; **not cleanup-eligible** until that documentation is integrated and the source run is archived |

**Closed on 2026-09-15.** Under explicit authorization and after the §4.1 gates, each PR
below was closed without merge at its verified head and its branch deleted from `origin` and
locally. Its conclusions survive in the repository record named here and its sources in the local
archive (§4.4, paths relative to `C:\gra\`).

| PR | Branch | Verified head | Durable record on `main` | Archived sources |
|---|---|---|---|---|
| #61 | `evidence/generalized-v2-actor-only-dev-r1` | `1375a881637a9a32721a1630f598adc571422a47` | [`measurements.md` §7–§8](../history/measurements.md#7-generalized-v2-development-r1-arms) | `runs\development\v2_actor_only_r1_seed3000000_ae42cb0` |
| #62 | `evidence/generalized-v2-ctde-dev-r1` | `b2bbe7a6235c3b9255106826cfb268af7e73f72d` | [`measurements.md` §7–§8](../history/measurements.md#7-generalized-v2-development-r1-arms) | `runs\development\v2_ctde_r1_seed3000000_ae42cb0` |
| #64 | `evidence/ctde-overnight-diagnostics` | `90516d51beeddacded2b89a321d14291e411f2b0` | [`measurements.md` §8](../history/measurements.md#8-generalized-v2-development-closure) | `diagnostics\v2_ctde_smallbatch_seed3000000_ae42cb0`, `…largebatch…`, `…fd80…`, `diagnostics\v2_ctde_sweep_driver_ae42cb0` |
| #65 | `review/v2-wake-pair-diagnostics` | `d565174e4ecc25eb60a4dd021e1a20025f55f07f` | [`measurements.md` §8.6](../history/measurements.md#86-matched-immediate-fd-wake-analysis) | the five V2 run directories it read; the derived extraction package itself was not copied into the archive |
| #67 | `review/v2-benchmark-preflight-provenance` | `7f56338cde6aacfa59a52399b2378b98a62ea3aa` | [`measurements.md` §8.11](../history/measurements.md#811-generalized-v2-benchmark-preflight-provenance-review) | `benchmarks\v2_preflight_seed2000000_ae42cb0` |

**Ledger check before deletion.** Every file SHA-256 in the `artifact_sha256.txt` ledgers of #61,
#62, #64 and #67 matched archived bytes. The only unmatched entries were derived packaging — the
committed `episode_outcomes` shards and shard index of #61 and #62, whose recorded original
`episode_outcomes.jsonl` hashes do match the archive — and values that are not file hashes
(manifest id, seed-list hash, ledger self-hashes).

### 4.4 Preserved run directories and external artifacts

- **Protection.** Original artifact bytes and provenance are protected: never modify, delete,
  regenerate or normalize them, even to correct a known defect. Authorized non-destructive copies,
  lossless packaging and reconstruction checks are allowed
  ([`experiments.md` §5](experiments.md#5-evidence-preservation)).
- **Authorized archival relocation.** An artifact may change location only under explicit
  authorization, by a same-volume rename that copies and rewrites no bytes, and only when: its
  original path is recorded; its current path is indexed; its file count, total bytes and
  key-file SHA-256 values are verified before and after the move; and the historical paths
  embedded inside it (`output_dir`, `repo_root`, ledger and script paths) are **not** rewritten.
  Those embedded paths are stale by design; the index resolves them.
- **Current local archive** (closure identities reviewed 2026-09-15):

  | Item | Path | SHA-256 |
  |---|---|---|
  | archive root | `C:\gra\` | — |
  | machine-readable index (authoritative; 28 rows) | `C:\gra\metadata\ARTIFACT_INDEX.jsonl` | `de96d9ba4c04e10c075549d37a1b445a6e513d133ef55591a1849bd0b0b80552` |
  | human-readable projection (not independent) | `C:\gra\metadata\ARTIFACT_INDEX.md` | `a34d69a52c7f99f93abf402e516345f6c2c9eca0fa213f7bf6efb8c8a5f211c6` |
  | archive move ledger (28 rows) | `C:\gra\metadata\ARCHIVE_MOVE_LEDGER.jsonl` | `15287e8fa7b1051f48cd2b4d1d629f61d687c567d0c4858c5248569d8b6f9eb7` |

  Per-artifact identities and recovered provenance:
  [`measurements.md` §9](../history/measurements.md#9-local-artifact-archive-closure).
- **Reading the index.** Its `evidence_ref_status` strings and caveats are archive-time text:
  they still describe PRs #61, #62, #64 and #67 as open and several source worktrees as left in
  place. Since 2026-09-15 those PRs are closed, their branches deleted and those worktrees removed
  (§4.3, §4.6); the index is deliberately not rewritten. For `v2_ctde_r1_seed3000000_ae42cb0` and
  the three diagnostic arms, `file_count` and `total_bytes` exclude the added `sidecars/` child,
  whose file hashes are listed in each row's caveats.
- **Retention.** Nothing under `C:\gra\` is cleanup-eligible under this registry — including the
  B4 engineering smokes, invalid precursor runs, the aborted P1 arm, the Task 5 engineering
  artifacts, the unclassified `ct1`, `rollouts` and `generated_scenarios`, and the review bundles.
  Pruning any of them requires its own separate authorization.

Principal artifacts, original location → current location (the index lists all 28):

| Artifact | Original location | Current location (under `C:\gra\`) |
|---|---|---|
| first real post-B3 probe (archive-time identity match, [`measurements.md` §9.3](../history/measurements.md#93-recovered-local-identities)) | `training_output_b4_probe_20260730_182528` (repository checkout) | `runs\legacy_measurements\b4_probe_a3f0838_unconfirmed_registry_match` |
| first final-cell short probe | `training_output_20260815_173029` (repository checkout) | `runs\legacy_measurements\probe_20260815_238062d` |
| corrected-cell short probe | `training_output_20260816_162130` (repository checkout) | `runs\legacy_measurements\probe_20260816_900ff0b` |
| first long baseline (inconclusive) | `training_output_long_baseline_100x8_seed0` (repository checkout) | `runs\legacy_measurements\phase_a_first_long_c30b698` |
| Phase-A long baseline (valid) | `training_output_long_baseline_100x8_seed0_rerun_20260818_737b4bf` (repository checkout) | `runs\measurements\phase_a_rerun_737b4bf` |
| FD-VARIABLE-SEVERITY-v1 baseline (valid) | `C:\Users\Itama\f7r2` | `runs\measurements\fd_variable_severity_valid_bf1e045f` |
| FD-VARIABLE-SEVERITY-v1 precursor (invalid) | `C:\Users\Itama\PycharmProjects\fd_variable_severity_v1_measurement_bf1e045f_20260822_150640` | `runs\legacy_measurements\fd_variable_severity_invalid_precursor_bf1e045f` |
| GENERALIZED-V1 R1 run tree and diagnostic bundle | `C:\g1r1` | `runs\measurements\generalized_v1_r1_4af6c5a` |
| aborted P1 arm (`DO NOT RESUME`) | `C:\p1r1` | `runs\legacy_measurements\p1_aborted_8f0d250_DO_NOT_RESUME` |
| fresh deterministic-P1 arm | `C:\p1_fresh_ae194103` | `runs\measurements\p1_fresh_ae194103` |
| GENERALIZED-V2 benchmark preflight and manifest | `C:\Users\Itama\PycharmProjects\graph_rl_v2_benchmark_preflight_seed2000000_ae42cb0` | `benchmarks\v2_preflight_seed2000000_ae42cb0` |
| GENERALIZED-V2 development R1 — actor-only | `C:\Users\Itama\PycharmProjects\graph_rl_v2_actor_only_dev_r1_seed3000000_ae42cb0` | `runs\development\v2_actor_only_r1_seed3000000_ae42cb0` |
| GENERALIZED-V2 development R1 — CTDE | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_r1_seed3000000_ae42cb0` | `runs\development\v2_ctde_r1_seed3000000_ae42cb0` |
| GENERALIZED-V2 CTDE diagnostic arms and sweep driver | `C:\Users\Itama\PycharmProjects\graph_rl_v2_ctde_dev_diag_{smallbatch,largebatch,fd80}_seed3000000_ae42cb0`, `…\graph_rl_v2_ctde_dev_diag_sweep_ae42cb0` | `diagnostics\v2_ctde_{smallbatch,largebatch,fd80}_seed3000000_ae42cb0`, `diagnostics\v2_ctde_sweep_driver_ae42cb0` |

### 4.5 Retired merged branches

**Merged branch awaiting cleanup** (observed 2026-09-19):

| Branch | Head | Merged by | Status |
|---|---|---|---|
| `task/v2-ctde-gradient-pressure-diagnostics` | `3e29a57dac54361c1a71f43f9487a6860fcae7c3` | PR #74 (merge `adc213670ce4844a7cf60943ecf50150318e40b1`) | present on `origin` and locally; verified ancestor of live `main`; not checked out in any worktree; the §4.1 safe-deletion gates are otherwise satisfied, but deletion **requires separate explicit authorization naming this branch** |

The branch of PR #75, `task/ctde-acting-ego-conditioning`, is already absent from `origin` and
locally.

**Deleted on 2026-09-15.** Each branch below was deleted from `origin` and locally after its
PR was verified `MERGED` at the listed head and that head was verified an ancestor of live `main`.

| Branch | Head | Merged by |
|---|---|---|
| `task/generalized-v2-benchmark-evaluation` | `786e8218a00954f7a7f20fe1dfca93ec71a400d4` | PR #59 |
| `task/generalized-v2-benchmark-doc-lock` | `6cbc4a60d3022b51776a4c027bdfc55beedbadac` | PR #60 |
| `docs/project-guidance-restructure` | `7936e97af4549231da0148620689d0326e88a799` | PR #63 |
| `docs/generalized-v2-development-closure` | `157dd9fdbb92fe2835ef2268f485e58b92523a4d` | PR #66 |
| `docs/v2-preflight-provenance-closure` | `0378114f5de3629854c72a446220219178b502a5` | PR #68 |

### 4.6 Local worktrees

**Removed on 2026-09-15** (`git worktree remove` without `--force`, after the §4.1 gates; every
commit stays reachable from `main`; no branch was deleted by these removals):

| Path | HEAD | Role |
|---|---|---|
| `C:/g1src` | `4af6c5aa5dd28072692bfda63282964b55010aae` | GENERALIZED-V1 R1 execution source |
| `C:/p1src` | `8f0d250cd9f96e6b8bce635065701dc47a5ee87e` | aborted P1 arm execution source |
| `C:/Users/Itama/ct1s` | `76abdc480e80a84f1503208730d4525cd5e89b69` | source checkout matching the unclassified `ct1` artifact's recorded commit |
| `C:/Users/Itama/PycharmProjects/fd_variable_severity_v1_bf1e045f_snapshot` | `bf1e045f90f74361e4ee944f7bd683a3ea72d04b` | FD-VARIABLE-SEVERITY-v1 pinned measurement snapshot |

`git worktree prune --dry-run` then reported no stale metadata, so no prune was run.

**Later lifecycle — observation of 2026-09-19** (not a cleanup action): the detached worktree
`C:/grolelambda1` (at `68055e39768d5fa601e5960a9f08823b9e65c08f`, the source checkout of the
role-only `gae_lambda = 1.0` diagnostic) is absent from disk and not registered, and no stale
worktree metadata remains for it. The local scratch directory `C:\tmp\l1v` (λ = 1 evidence
verification; scratch, never a Git worktree) is also absent. Who removed them, and when, is
unknown. The only registered worktrees are the main checkout and `flat-baseline`.

**Retained:** `C:/Users/Itama/PycharmProjects/flat-baseline`, carrying protected branch
`flat-final` at `4d44c3454a5561a6cb9d7aed593d59a40068d6d7`. Its tracked tree is clean, but its
ignored files hold flat-RL training outputs that are **not** in the local archive —
`training_output_5k_util_and_match/` (10 514 files, 3 281 368 218 bytes) and
`training_output_RL_util/` (6 283 files, 1 961 549 611 bytes) — which removing the worktree would
delete. Its removal needs a separate decision about those outputs. Branch `flat-final` stays
protected either way.

### 4.7 Cleanup already performed

- After PR #11 and PR #12: `task/repo-code-hygiene`, `task/repo-doc-hygiene` and
  `task/final-cell-visual-artifacts` were deleted locally and remotely by safe deletion.
- Before the Phase-A rerun: `task/roster-world-truth-fix`, `task/long-baseline-execution` and
  `task/roster-world-truth-doc-lock` were deleted after their tips were verified reachable;
  `task/variable-fd-severity-baseline` and `task/variable-fd-severity-doc-lock` were later
  observed absent from the remote.
- The branches the former handoff listed as cleanup-eligible after PR #33 through PR #58 are no
  longer present on `origin` as of 2026-09-14; who removed them, and when, is not recorded.
- **2026-09-15, research-chapter closure** (live `main` `81d37049845087e970d9416abc99adcbb40aef61`
  before and after): the local archive of §4.4 was verified (index, ledger and projection hashes;
  all 28 rows' key hashes, file counts and bytes); PRs #61, #62, #64, #65 and #67 were closed and
  their branches deleted (§4.3); the five merged branches of §4.5 were deleted; four worktrees
  were removed and `flat-baseline` retained (§4.6); the protected refs of §4.2 were unchanged.
  Nothing under `C:\gra\` was modified or deleted.
