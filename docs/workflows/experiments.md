# Experiments — planning, run review and evidence preservation

> **Read this when** you plan, authorize or configure a scientific run, review a completed run,
> cite a measurement, or preserve run evidence.
>
> **Status: normative procedure.** It consolidates the validity gate and interpretation rules of
> the former handoff §4, the standing prohibitions of the former handoff §6 and §8, the
> engineering-versus-measurement labels, and the comparator discipline of the former `CLAUDE.md`
> §8. The technical contracts it relies on are
> [training and benchmarks](../contracts/training_benchmarks.md) and
> [artifacts and metrics](../contracts/artifacts_metrics.md); past runs are recorded in
> [`measurements.md`](../history/measurements.md).

## 1. What counts as a measurement

- A **measurement** is an explicitly authorized scientific run with a frozen contract (design,
  backend, training mode, schedule and budget, seeds, comparator), an exact measured code SHA
  with complete clean provenance, and an independent review verdict.
- **Engineering validation is never a measurement**, because of its designated purpose — even
  when it carries seed bands, a transient manifest or real solver and engine execution. Bounded
  smokes, selftests, test suites, the Task-5A / Task-5B validations, replays and reconstructions
  support no reward, learning, performance or comparison claim.
- A Grade-A **implementation** approval is never projected onto a future measurement.
- A valid **negative** result is a result. A run whose denominator an instrument defect shrank
  is not a result at all.

## 2. Planning a run

Before execution, all of the following hold and are written down:

- **explicit user authorization** for this specific run;
- **a frozen contract**: `episode_design`, `match_aou_backend`, `training_mode`,
  `n_iterations`, `episodes_per_iteration`, `generalized_max_attempts_per_iteration` where
  required, base seeds, evaluation and checkpoint cadence, `early_stopping`, `fuel_damage_mode`,
  and the benchmark manifest and profile;
- **benchmark identity resolved explicitly**: never silently rebuild, replace or regenerate a
  population; seed windows are never chosen for convenience; population selection is policy-
  and reward-blind;
- **held-outness** checked against the maximum possible training-attempt band, over every seed of
  the manifest;
- **arms meant to be compared share one frozen manifest**, and under early stopping share the
  maximum budget, the frozen stopping rule and the training-population contract — not the actual
  iteration count;
- **provenance**: a clean checkout, complete Git provenance, one invocation driven by a config
  file, `cli_overrides` recorded;
- **a fresh, non-overwriting output directory** — on Windows under a short root, because a
  267-character playback path once removed a whole evaluation arm (the FD-VARIABLE-SEVERITY-v1
  precursor);
- **the environment rules** of [`environments_cleanup.md` §1](environments_cleanup.md#1-execution-contexts);
- **design limits**: GENERALIZED-V2 requires `p1_milp_v1`, supports `A ≤ 6` only and refuses
  early stopping.

## 3. Run review — validity before performance

### 3.1 Validity gate

The gate below is carried forward verbatim from the former handoff §4. Section references inside
it (§3e, §3f, §3h, §3i, §3j) are former handoff sections, now in
[`measurements.md`](../history/measurements.md).

**What makes a run VALID — carried forward unchanged, and it now has a passing precedent.**
A run counts as a valid measurement when ALL of:

- Git provenance is COMPLETE and the checkout was clean;
- `run_summary.json:accounting_reconciled` is true;
- no INFRASTRUCTURE or DATA-INTEGRITY failure occurred. A `_VisualArtifactError`, a
  `MeasurementIntegrityError` / `EpisodeRosterError` — including the scheduled-vs-executed
  CELL mismatch PR #27 added (§3i) — or any crash outside the
  `generation` / `setup` / `run` / `reward` episode taxonomy ABORTS the run and is not a
  scientific result;
- **at least one COMPLETED matched group exists in BOTH the `pre_update` and the
  `post_update` round.** A group counts only when EVERY member completed — two members for
  a legacy pair, **all three for a variable-severity triad.**

The Phase-A rerun (§3h) satisfied all four, and so did the variable-severity baseline
(§3j); the two runs before the Phase-A rerun did not, on the data-integrity clause (§3e,
§3f), and the variable-severity precursor did not either, on the same clause.

For generalized runs the matched group is the CLEAN / MILD / SEVERE triad, and the
infrastructure and data-integrity aborts also include `TrainingQuotaError`,
`EarlyStoppingIntegrityError`, `FuelDamageIntegrityError`, `BenchmarkIdentityError`, an aborting
`ReferenceIntegrityError` and `MatchAouBackendError` (routing:
[training and benchmarks](../contracts/training_benchmarks.md),
[reward and solvers](../contracts/reward_solvers.md)).

### 3.2 Interpretation rules

**A negative result is still a valid result — and §3j is now the worked example.** No
improvement, no severity-conditioned behavioural difference, or zero productive PPO updates,
is a valid NEGATIVE SCIENTIFIC OBSERVATION — not a technical failure, and not grounds to
re-run, re-tune or re-seed. The variable-severity baseline measured exactly that: productive
training and no severity-conditioned separation.

**Interpretation rules survive unchanged:** a held-out mean is never read without its
denominator; an all-failed batch reports `null`, never `0.0`; an empty successful-group
population is `null` too; the per-condition / per-cell means are each over their own
successful subset, so the within-seed claims are the matched deltas over COMPLETE groups
alone (`CLAUDE.md` §5); and FD-wake meta-action rates are reported over FD WAKES, never over
episodes. **Do not reuse §2's, §3e's or §3f's numbers as any expectation** — §2 measured a
different, easier cell, and §3e and §3f are both scientifically INCONCLUSIVE, as is the
variable-severity `MAX_PATH` precursor (§3j). **TWO valid scientific baselines now exist and
they measure DIFFERENT cells: §3h is the LEGACY FD-BASELINE-v1 baseline, and §3j is the
FD-VARIABLE-SEVERITY-v1 baseline. Neither is an expectation for the other, and neither is an
expectation for any CTDE comparison.**

Additional rules for generalized and per-wake evidence:

- **Repeated measures:** every evaluation round re-measures the same frozen worlds; cross-round
  totals describe a trajectory, never independent worlds. The statistical unit for a final policy
  is the final round's complete matched groups.
- **Aggregate probability mass is not the selected action's probability**, and the three wake
  kinds and the `train` / `pre_update` / `post_update` populations are never pooled silently
  ([artifacts and metrics §5](../contracts/artifacts_metrics.md#5-per-wake-fd-policy-diagnostics)).
- **The final evaluation round is selected semantically**, never as the last record.
- **Requested versus realized hidden cardinality** is reported for human or GPT inspection; no
  threshold is computed.
- **The GENERALIZED-V2 primary endpoint** is undefined unless every base cell has a
  metric-eligible group; the code implements no inference procedure.
- **Known artifact defects** are read around, never normalized
  ([artifacts and metrics §6](../contracts/artifacts_metrics.md#6-reading-preserved-artifacts)).

### 3.3 Comparator discipline

- State each record's **measured code SHA**; never present two SHAs as a one-config-field
  comparison.
- **Non-equivalent populations support no causal inference.** R1 (legacy objective at
  `4af6c5aa…`) and the fresh deterministic-P1 arm (at `ae194103…`) ran over different worlds, so
  no solver-quality or reward-difference inference is authorized.
- **Historical fixed-cell measurements are not generalized comparators** and not expectations.
- **A fixed-cell actor-only versus CTDE comparison, if ever resumed,** takes its actor-only arm
  from the approved Phase-A baseline without re-running it, matches that baseline's cell,
  schedule, seed policy, held-out band and evaluation construct, names the factor as actor-only
  versus centralized-critic training, acknowledges the distinct measured SHAs, and bundles no
  other change. The full specification is preserved in
  [`decisions.md` §4](../history/decisions.md#4-research-ordering-ctde-comparison-specification-and-difficulty-selection).
- **The old fixed-cell CTDE measurement is out of scope** and is not reviewed or compared unless
  the user explicitly asks.
- **No CTDE benefit is established** by any repository document.

## 4. Evidence preservation

- **Preserved run directories and external artifacts** are never modified, moved, copied,
  repackaged, deleted or regenerated
  ([registry](environments_cleanup.md#4-authorized-cleanup)).
- **Evidence commits** follow the practice used by PR #61 and PR #62 (both unreviewed on
  2026-09-14):
  - an evidence-only branch whose parent is the measured code SHA, adding files under
    `research_evidence/<design>/<run>/` only;
  - byte-identical copies of `run_config.json`, `run_summary.json`, the record streams, the
    failure ledger, console logs and timing files;
  - a large `episode_outcomes.jsonl` split only at existing line ends into shards, with an index
    and a reconstruction hash equal to the source's SHA-256;
  - an `artifact_sha256.txt` listing every committed file, the source hashes, checkpoint hashes
    (checkpoints are not committed) and any external manifest's path, hash and id;
  - staging with `git -c core.autocrlf=false add`, blob identity verified with `git cat-file`,
    size limits respected, and ignored files added only when declared;
  - archived bytes never normalized, even to correct a known defect.
- Record the **measured code SHA** and the **evidence-commit SHA** separately.
- Documentation links immutable evidence commits and never copies evidence blobs.

## 5. Standing prohibitions

Unless the user explicitly authorizes otherwise:

- do not re-run, resume, repair, extend or retune an approved measurement (Phase-A,
  FD-VARIABLE-SEVERITY-v1, R1, the fresh P1 arm); never resume the aborted P1 arm;
- do not relax, retry, retune or reclassify expected setup failures (B2 exact-cardinality,
  fuel-window, `NO_FD_ELIGIBLE_EGO`), including held-out seed `1000005`;
- do not reopen the closed Defects A, B, C or the roster defect, or act on the recorded
  over-safety hypothesis, without a new research decision;
- do not change code, tests, configs or presets in response to a negative result;
- do not claim more than a result establishes — every measurement record carries its
  non-claims;
- do not treat engineering validation as measurement;
- do not bundle deferred research changes (`p(destroy) < 1`, SAMs / hostile fire, dense
  per-wake reward, solver or reward-formula changes, a new difficulty factor) into a comparison;
- checkpoint resume is out of scope; a low-known solver timeout, ETA / peer-dropout, the
  reachability model and legacy-split retirement are separate future work;
- no severity, cardinality, stratum or other privileged label may reach the acting path.
