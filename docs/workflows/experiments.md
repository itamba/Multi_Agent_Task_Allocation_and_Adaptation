# Experiments — execution authority, planning, run review and evidence preservation

> **Read this when** you plan, authorize, run or review a scientific or engineering run, cite a
> measurement, or preserve run evidence. Read the section the step needs, plus the contract
> sections it names.
>
> **Status: normative procedure.** It consolidates, and in places generalizes, the validity gate
> and interpretation rules of the former handoff §4, the standing prohibitions of the former
> handoff §6 and §8, the engineering-versus-measurement labels, and the comparator discipline of
> the former `CLAUDE.md` §8, as revised in the PR #63 review. It defines **no** statistical
> acceptance threshold and **no** rerun policy.
>
> Depends on: [training and benchmarks](../contracts/training_benchmarks.md) (§5–§9 for designs,
> quotas and benchmarks) · [artifacts and metrics](../contracts/artifacts_metrics.md) (§3–§6 for
> records and reading rules) · the per-run records in [`measurements.md`](../history/measurements.md).

## 1. Status vocabulary

- **Engineering validation** exists to check wiring, runtime, solver cost, attrition in a bounded
  sample, or the reproducibility of a mechanism. It can establish **scoped engineering facts**
  (for example "a solve of roughly 998 seconds terminated optimal", "the P1 backend kept
  reference solving cheap on these cells") and never **policy-quality, learning, reward-level or
  comparative** claims — even when it carries seed bands, a transient manifest or real solver and
  engine execution. Bounded smokes, selftests, test suites, the Task-5A / Task-5B validations,
  replays and reconstructions are engineering validation.
- **Executed measurement.** A scientific run executed under an authorized plan, identified by its
  measured code SHA, resolved configuration and artifacts. It exists as a measurement once it
  has executed; its **validity is unknown until reviewed**.
- **Reviewed measurement.** An executed measurement with a recorded review verdict (for example
  `APPROVE — VALID MEASUREMENT`, `INCONCLUSIVE`, `INVALID`). Cite the verdict with its provenance
  — who recorded it and where. **An absent accessible verdict record does not establish that no
  review happened**; say which it is.
- **Development versus confirmatory.** Development-profile results may inform design choices;
  they are not confirmatory evidence. Confirmatory evidence comes from a pre-declared plan whose
  choices were not tuned on the data it confirms.
- A valid **negative** result is a result. A run whose denominator an instrument defect shrank is
  not a result at all. A Grade-A **implementation** approval is never projected onto a
  measurement.

## 2. Execution authority — the authorized bounded plan

Scientific execution — training, evaluation, benchmark preflight, replay, resume or repair — runs
only under an **authorized bounded plan** from the user. The plan states:

- the **question** the run answers;
- the **population and comparator**: episode design and backend, the benchmark manifest and
  profile or the rule that builds them, and what the result will be compared with;
- the **primary endpoint** and the review focus;
- **resources and attempt budget**: iterations, successful-episode quota and attempt budget,
  execution context and walltime;
- **stop conditions** and how integrity aborts or unexpected attrition are handled;
- the **output location** and what evidence will be preserved.

**Steps the plan covers proceed without asking again.** Escalate before proceeding on a material
deviation: a changed population, comparator, endpoint or budget; attrition outside what the plan
expected; any integrity abort; or a need to rerun, replace or extend a run. A documentation task
authorizes no scientific execution, and a dated authorization in history authorizes nothing now.

## 3. Planning checklist

- **An inspectable resolved configuration and its actual invocation.** The run's
  `run_config.json` records `train_config` and `config_source`, whose `resolved_from` may be
  `config_file`, `cli_defaults` or `direct_config`
  ([training and benchmarks §4](../contracts/training_benchmarks.md#4-configuration-presets));
  record the actual command or call, and require complete Git provenance on a clean checkout.
- **Benchmark identity is explicit.** Never silently rebuild, replace or regenerate a population;
  never choose a seed window for convenience; population selection stays policy- and reward-blind
  ([training and benchmarks §6](../contracts/training_benchmarks.md#6-training-quota-and-benchmark-preflight),
  [§9](../contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation)).
- **Held-outness** is checked against the run's maximum possible training-attempt band over every
  seed of the manifest.
- **Comparisons.** Where the approved design contract requires it — arms of the same generalized
  comparison — the arms use the identical frozen manifest and, under early stopping, the same
  maximum budget, stopping rule and training-population contract. A cross-version comparison is
  allowed only with an explicit list of its consequential differences (code SHA, backend,
  population, profile) and the causal limits those differences impose.
- **A fresh, non-overwriting output directory**; on Windows under a short root, because a
  267-character playback path once removed a whole evaluation arm.
- **The environment rules** of [`environments_cleanup.md` §1](environments_cleanup.md#1-execution-contexts).
- **Design limits**: GENERALIZED-V2 requires `p1_milp_v1`, supports `A ≤ 6` only and refuses early
  stopping; early stopping is approved for `generalized_v1` only.

## 4. Run review — validity before performance

### 4.1 Review order

Review applies the design's own contract, in this order, and records each step's result:

1. **Provenance and configuration** match the authorized plan (measured code SHA, clean tree,
   resolved configuration, invocation).
2. **Accounting and completion**: `accounting_reconciled`; attempted, successful and failed counts
   by phase and stage; quota fill on generalized designs; expected attrition identified; **no
   infrastructure or integrity abort** (`_VisualArtifactError`, `MeasurementIntegrityError` /
   `EpisodeRosterError` including the scheduled-versus-executed cell check, `TrainingQuotaError`,
   `EarlyStoppingIntegrityError`, `FuelDamageIntegrityError`, `BenchmarkIdentityError`, an
   aborting `ReferenceIntegrityError`, `MatchAouBackendError`, or a crash outside the
   `generation` / `setup` / `run` / `reward` taxonomy).
3. **Frozen identity**, for benchmark-evaluated designs: manifest id and profile as planned,
   held-outness verified, member identity verified, complete versus incomplete groups visible.
4. **Endpoint eligibility**: the matched groups and wakes the primary endpoint needs exist — for
   GENERALIZED-V2, metric-eligible groups **in every base cell**, since the macro endpoint is
   undefined otherwise ([training and benchmarks §9](../contracts/training_benchmarks.md#9-generalized-v2-benchmark-and-evaluation)).
5. **Scientific interpretation**, under §4.3, stating the non-claims.

### 4.2 Historical note — the fixed-cell four-clause gate (2026-08-16 to 2026-08-23)

The fixed-cell runs were judged valid when provenance was complete on a clean checkout,
`accounting_reconciled` was true, no infrastructure or data-integrity failure occurred, and at
least one completed matched group (a pair, or a triad for FD-VARIABLE-SEVERITY-v1) existed in both
the `pre_update` and the `post_update` round. The Phase-A rerun and the FD-VARIABLE-SEVERITY-v1
baseline passed it; the corrected short probe, the first long baseline and the variable-severity
precursor failed it on the data-integrity clause. It is recorded here as that scope's gate; it is
**not** sufficient for the generalized designs, which follow §4.1.

### 4.3 Interpretation rules

- A mean is never read without its denominator; an all-failed batch or an empty group population
  is `null`, never `0.0`.
- Per-condition and per-cell means are each over their own successful subset; **within-world
  claims come only from matched deltas over complete groups**.
- FD-wake meta-action rates are reported over FD wakes, never over episodes.
- **Repeated measures:** every evaluation round re-measures the same frozen worlds; cross-round
  totals describe a trajectory, never independent worlds. The unit for a final policy is the
  final round's complete matched groups.
- Aggregate probability mass is not the selected action's probability; the three wake kinds and
  the `train` / `pre_update` / `post_update` populations are never pooled silently
  ([artifacts and metrics §5](../contracts/artifacts_metrics.md#5-per-wake-fd-policy-diagnostics)).
- The final evaluation round is selected semantically, never as the last record.
- Requested versus realized hidden cardinality is reported for human or GPT inspection; no
  threshold is computed.
- Known artifact defects are read around, never normalized
  ([artifacts and metrics §6](../contracts/artifacts_metrics.md#6-reading-preserved-artifacts)).
- **Each reviewed measurement is scoped to its own cell or design.** Inconclusive runs and runs
  on the easier pre-FD cell are never expectations, and no valid baseline is an expectation for
  another design or for a CTDE comparison. The current list is in
  [`measurements.md` §1](../history/measurements.md#1-run-registry).

### 4.4 Comparator discipline

- State each record's **measured code SHA**; never present two SHAs as a one-config-field
  comparison.
- **Non-equivalent populations support no causal inference.** R1 (legacy objective at
  `4af6c5aa…`) and the fresh deterministic-P1 arm (at `ae194103…`) ran over different worlds, so no
  solver-quality or reward-difference inference is drawn from them.
- Historical fixed-cell measurements are not generalized comparators and not expectations.
- A fixed-cell actor-only versus CTDE comparison, if a plan ever authorizes one, reuses the
  approved Phase-A baseline as its actor-only arm, matches that baseline's cell, schedule, seed
  policy, held-out band and evaluation construct, names the factor as actor-only versus
  centralized-critic training, acknowledges the distinct measured SHAs, and bundles no other
  change ([`decisions.md` §4](../history/decisions.md#4-research-ordering-ctde-comparison-specification-and-difficulty-selection)).
- The old fixed-cell CTDE measurement is out of scope and is not reviewed or compared unless the
  user asks.
- No CTDE benefit is established by any repository document.

## 5. Evidence preservation

- **Protect the originals.** Original run directories and external artifacts keep their bytes and
  identity: never modify, move, delete, regenerate or normalize them, even to correct a known
  defect. **Authorized non-destructive copies, lossless packaging** (such as line-aligned sharding
  with a reconstruction hash) **and reconstruction checks are allowed** and leave the originals
  untouched ([registry](environments_cleanup.md#4-authorized-cleanup)).
- **Preserve enough to inspect the review question**: at least the resolved configuration and
  provenance, the summary and accounting records, and the records the question needs. Anything
  not committed (checkpoints, manifests, streams the question does not need) is identified by
  location and SHA-256.
- **Record the measured code SHA and the evidence SHA separately.**
- Documentation links immutable evidence commits and never copies evidence blobs.

**Scoped example — PR #61 and PR #62 (both unreviewed on 2026-09-14).** Each is an evidence-only
commit whose parent is the measured code SHA, adding `research_evidence/generalized_v2/<run>/`:
byte-identical copies of `run_config.json`, `run_summary.json`, the record streams, the failure
ledger, console logs and timing files; `episode_outcomes.jsonl` split at existing line ends into
shards with an index and a reconstruction hash; an `artifact_sha256.txt` listing every committed
file, source hashes, checkpoint hashes and the external manifest's path, hash and id; staging with
`git -c core.autocrlf=false add` and blob identity checked with `git cat-file`. Each package is
roughly 80 MB. This is one adequate shape for a full development-arm review, **not a universal
requirement**.

## 6. Standing constraints

- **Reviewed measurements are reused as recorded by default** (Phase-A, FD-VARIABLE-SEVERITY-v1,
  R1, the fresh P1 arm). New execution touching them — a rerun, extension, repair or resume —
  happens only under an authorized plan that names it. **The aborted P1 arm is
  `DO NOT RESUME`.**
- Expected setup failures (B2 exact-cardinality, fuel-window, `NO_FD_ELIGIBLE_EGO`, including
  held-out seed `1000005`) are accounted, never relaxed, retried or reclassified inside a run.
- Closed Defects A, B, C and the roster defect, and the recorded over-safety hypothesis, are not
  reopened or acted on without a new research decision.
- A negative result is not a defect report: code, tests, configs and presets change only in a
  separately authorized task.
- A result is never claimed beyond its record's non-claims.
- Engineering validation stays within §1's scope.
- Deferred research changes (`p(destroy) < 1`, SAMs / hostile fire, dense per-wake reward, solver
  or reward-formula changes, a new difficulty factor) are never bundled into a comparison.
- The code has no checkpoint resume. A low-known solver timeout, ETA / peer-dropout, the
  reachability model and legacy-split retirement are separate future work.
- No severity, cardinality, stratum or other privileged label may reach the acting path.
