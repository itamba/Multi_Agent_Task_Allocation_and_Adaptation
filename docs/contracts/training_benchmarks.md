# Training and benchmark contract — trainer, run integrity, presets, episode designs, quotas, early stopping and benchmarks

> **Read this when** you change or review `graph_train` (provenance, failure accounting,
> integrity aborts, presets and `config_source`), episode designs (`fixed_cell_v1`,
> `generalized_v1`, `generalized_v2`), training cardinality samplers, the successful-episode
> quota and attempt budget, held-out seed bands, early stopping, benchmark manifests and the
> benchmark preflight, GENERALIZED-V2 population or evaluation, or the diagnostic rollout's
> parity with training. Planning or reviewing a run also needs
> [`docs/workflows/experiments.md`](../workflows/experiments.md).
>
> **Status: normative, current technical contract** for the code on `main`. Lock history is in
> [`implementation.md`](../history/implementation.md), block provenance in
> [`documentation_migration.md`](../documentation_migration.md), and current run, manifest and
> evidence state in the [handoff](../../graph_rl_project_handoff.md). Statements below that
> name what a specific PR did or did not ship are scoped to that PR.
>
> Related contracts: [artifacts and metrics](artifacts_metrics.md) (visual artifacts, figures,
> outcome streams, generalized persistence, per-wake diagnostics) ·
> [construction and fuel damage](construction_fuel_damage.md) · [reward and solvers](reward_solvers.md) ·
> [policy and CTDE](policy_ctde.md).

## 1. Trainer and run auditability

**Trainer + run auditability (B4) — `rl/training/graph_train.py`.**
The outer PPO loop's *research-validity* contract. It changed NO pipeline layer: PPO
objectives/hyperparameters/checkpoint payload, reward and oracle normalization, the
solver, construction/geometry/exact cardinality, the seed formulas and the fixed
held-out band are all exactly as B1–B3 left them.

- **Exact-cardinality policy = `skip_and_account_v1`.** Every scheduled train/eval seed
  is attempted **at most once**; a failure is never retried, never replaced by another
  seed, and never shifts a band. Failures never enter a PPO buffer or a reward
  aggregate, and each is recorded exactly once. Attempts, successes, failures and
  denominators stay explicit, so every reward statistic describes the SUCCESSFUL /
  exact-cardinality-feasible subset — and says so (`aggregates_over`).
- **Git provenance is a training PRECONDITION.** `collect_provenance` runs before the
  run creates ANY artifact (not merely before the engine/policy/solver) — `output_dir`
  may sit inside the repo and its own untracked files would otherwise read as dirty
  source state. `_git_provenance` sets `available=True` only when BOTH the full commit
  SHA and the clean/dirty verdict were determined (a SHA alone does not say what ran).
  Incomplete provenance writes the attempted `run_config.json`, then `train` REFUSES
  before policy, generator, episode or optimizer work. A KNOWN-dirty tree warns loudly
  and may run.
- **Run artifacts** (a run directory is the record): `run_config.json` carrying the
  versioned `provenance` block, `train_records.jsonl`, `eval_records.jsonl`, the
  append-only immediately-flushed `episode_failures.jsonl` (phase, eval stage, updates
  completed, iteration, attempt ordinal, episode index / eval tag, exact seed, pipeline
  stage `generation|setup|run|reward`, original exception + traceback), the derived
  `run_summary.json` (`build_run_summary` reads the jsonl — ONE metric path, with
  `accounting_reconciled` cross-checking record counts against the ledger), and the
  THREE figures under `plots/` (`plot_training`, jsonl-only, torch-free child — see the
  harness contract below). A run root holds records, `scenarios/`, `checkpoints/`,
  optional `visual_artifacts/` and `plots/` as separate things.
- **Evaluation timing.** A deterministic held-out `pre_update` round runs after the
  initial policy is built and BEFORE the first training episode, buffer insert and
  optimizer step, recorded with `updates_completed = 0` and `iteration = null`. Later
  rounds carry their REAL completed-update count, so none can be read as "iteration 0".
- **Classification (`_iteration_outcome`) — three DISJOINT states:** `all_failed`
  (nothing completed; measured nothing), `zero_wake` (episodes completed, no ego woke)
  and `productive`. Both of the first two end at `n_epochs_run == 0`, so the classifier
  judges episode counts, not the updater. An all-failed batch or eval round reports its
  reward as `null`, **never `0.0`** — the reward is oracle-normalized regret, so 0 is
  the OPTIMUM. A successful zero-wake episode is a real successful episode.
- `TrainConfig` gains no scenario semantics here; `evaluate` gained `stage` /
  `updates_completed` / `failures_path`. `updates_completed` counts only updates that
  actually ran epochs, and is the learning-curve x-axis.
- **Per-episode observability and confirmation semantics (PR #7).** Every successful
  train / `pre_update` / `post_update` attempt prints one immediate, labelled `OK` block
  with its phase, indices, exact seed, reward, wakes, ending, ticks, dead count, elapsed
  time and the known/hidden target roster by BLADE name. `GraphPlanExecutor.done` remains
  a set of `(ego_id, target_id)` CONFIRMATIONS and may exceed the world target count;
  trainer target metrics instead count unique `target_id` values directly through
  `_unique_confirmed_target_ids(ctx.executor.done)`. The authoritative aggregates are
  `targets_confirmed_unique_mean` / `eval_targets_confirmed_unique_mean`;
  `kills_mean` / `eval_kills_mean` are compatibility aliases fed from the same corrected
  number. Names are presentation only. A name lookup may degrade to `<unnamed target>`
  without changing an id or count, while malformed/inconsistent roster structure raises
  `EpisodeRosterError` and never contributes a false successful zero. Reward and PPO
  semantics are unchanged; the reward already deduplicated by target id.
  *(Dated note: PR #7 routed that error as an accounted `setup` failure; since `36365f2`
  `EpisodeRosterError` is a `MeasurementIntegrityError` and ABORTS the run as INFRASTRUCTURE —
  see [§2](#2-roster-and-world-truth-integrity).)*
- **Per-round eval scenario preservation (PR #7).** `eval_episode_tag` gives every eval
  round a deterministic, disjoint file-tag namespace. Tags affect artifact names only:
  every round still evaluates the same fixed held-out seed band. `TrainConfig.validate`
  rejects tag ranges that could collide, so pre- and post-update scenario JSONs coexist.

The opt-in visual-artifact bullet of this block is contracted in
[artifacts and metrics §1](artifacts_metrics.md#1-visual-artifacts).

## 2. Roster and world-truth integrity

**Roster / world-truth integrity (PR #24) — `rl/training/graph_train.py` +
`rl/training/graph_episode_setup.py`.**
The *measurement-integrity* half of the trainer contract, and the correction that a real
long baseline forced. It changed NO pipeline layer: the reward formula, PPO,
oracle allocation, fuel-damage semantics, B2 placement, seed formulas, the evaluation
schedule, the tick loop, the executor, the generator and vendored BLADE are all exactly as
their own locks left them.

- **The roster's WORLD comes from the two raw pre-solve snapshots, never from an
  allocation.** `_episode_target_roster(ctx)` reads `ctx.known_target_ids` (KNOWN) and
  `ctx.executed_target_ids` (EXECUTED) through the validating accessor
  `_world_snapshot_ids`, and derives HIDDEN by SUBTRACTION — executed minus known, in
  EXECUTED-WORLD ORDER (`ctx.placements` is deliberately id-free, so it cannot supply
  hidden ids). **`ctx.oracle_tasks` IS NOT READ THERE AT ALL** and must not be
  reintroduced. The BELIEFS are still checked, but only in the role they can play: they
  are allocated-only too, so they are a **SUBSET** constraint on the known world, never
  its denominator — a belief naming a target the known snapshot does not hold still
  raises, because the egos would have been planned against something the world does not
  contain. The t=0 belief-agreement check is unchanged.
- **`MeasurementIntegrityError` — INFRASTRUCTURE, never a scientific outcome.**
  `EpisodeRosterError` is now a subclass of it (the NAME is retained so an audit trail
  keeps reading; what changed is where it GOES, not what it means). It is a sibling of
  `_VisualArtifactError` and routed identically: it names no pipeline stage because it did
  not happen in one, and it **ABORTS the run**. It is NEVER wrapped in
  `EpisodeAttemptError`, never appended to `episode_failures.jsonl`, never counted against
  a condition tally, and never enters `skip_and_account_v1` — so it can no longer shrink a
  scientific denominator while reading as ordinary episode attrition. Both attempt handlers
  re-raise it AHEAD of their broad `except Exception` (`except (_VisualArtifactError,
  MeasurementIntegrityError)`), and an UNEXPECTED exception raised inside the roster code is
  normalized into the same loud path with its cause preserved. **This is a deliberate
  reversal of PR #7's routing**, and the reason is that a data-integrity fault is a property
  of the INSTRUMENT, not of the episode: every episode it touches is suspect, and the ones
  it does not touch cannot be assumed unaffected.
- **`_require_scheduled_cell(roster, cfg)` — the roster must describe the cell the
  schedule asked for.** `n_known` known, `n_hidden` hidden, `n_targets_emitted` executed,
  or `EpisodeRosterError`. `setup_episode`'s construction path already enforces that
  cardinality loudly on its own side, so a roster that disagrees is not a scenario that
  came out differently — it is this module measuring the world wrongly. Checked BEFORE the
  fuel-damage plan and BEFORE `run_episode`, so nothing is paid for and no partial
  measurement exists.
- **ORDER AFTER `run_episode` IS CONTRACTUAL: synchronize the playback, validate the
  world, and only then compute a reward.** `_AttemptArtifacts.sync_recordings()` runs
  immediately after `run_episode` returns and DISCOVERS the playback chunks the completed
  run really wrote into the still-`incomplete` manifest — nothing is created, renamed or
  fabricated; a completed run with no playback file is itself a `_VisualArtifactError`,
  because the tick-loop contract exports one on every completed run and none when the loop
  raised. Only then is the confirmed-id set reconciled against the executed-world snapshot,
  and only a world that validated is allowed to produce a reward and a successful outcome.
  The long baseline ran it the other way round, which is how 17 episodes exported a real
  recording that no manifest listed.
- **A manifest cannot claim `complete` against its own files.** `_AttemptArtifacts.finalize`
  requires all three artifacts AND reconciles expected vs observed target counts; on a
  mismatch it still WRITES the observed counts, leaves the status `incomplete`, and raises
  `_VisualArtifactError`. `complete` is a CLAIM, so a manifest certifying a world its own
  `executed_t0_scenario.json` contradicts is worse than no manifest.
- **The authoritative count is unchanged**, and is still
  `len(_unique_confirmed_target_ids(executor.done))` and nothing else — never derived from
  how many ids the roster managed to NAME. A name that will not resolve still degrades to
  `<unnamed target>` and changes no id and no count.

## 3. Scheduled versus executed cell

**SCHEDULED CELL vs EXECUTED CELL — a measurement-integrity abort
(`_ConditionTally.success`, the approved review fix `eecc9b5`).**
`success(out, *, expected_cell)` takes the SCHEDULE's cell as a **REQUIRED keyword** and
requires `executed_cell == expected_cell` — **equality, not membership**. Membership alone
cannot see the fault: under FD-VARIABLE-SEVERITY-v1 a scheduled `mild` that executed as
`severe` names a cell the run legitimately reports, so a membership test ACCEPTS it and
books the ATTEMPT in one cell's denominator and the REWARD in another. **That corrupts
BOTH denominators at once** — the scheduled cell reads as a failure that never happened,
the executed cell as a success that was never scheduled — and a triad's within-seed delta
would be taken between two members the schedule never paired. The keyword is required
deliberately: an optional one would let a future call site skip the check by omission.
Three disjoint faults are named separately (the SCHEDULE names an unreported cell; the
EXECUTION reports an unreported cell; both reportable but DISAGREEING), all are
`MeasurementIntegrityError`, and every check runs BEFORE any state is mutated, so a
rejected episode leaves the tally byte-unchanged. **BOTH production call sites pass their
scheduled cell and the guard runs FIRST**, so a mismatched episode reaches NEITHER the
per-cell counters and rewards, NOR a matched-group member reward or delta, NOR
`episode_outcomes.jsonl`, NOR the PPO buffer. It ABORTS the run as INFRASTRUCTURE exactly
as a roster fault does — never an accounted scientific episode failure. **This has NOT
been observed in the real simulator**: the regression test INJECTS the divergence through
a stub, because normal production does not currently generate it.

## 4. Configuration presets

**Experiment harness: JSON presets, run layout and the three figures (PR #14) —
`rl/training/graph_train.py` + `configs/graph_train/final_cell_probe.json`.**
The OPERATOR-facing surface. It changed no pipeline layer, no scenario semantics, no seed
schedule, no PPO/reward/fuel-damage behaviour and no evaluation record field; what it
changes is how a run is CONFIGURED and how its results are PRESENTED.

- **JSON presets — `--config <path>`.** Stdlib `json` only, no new dependency. A preset
  names `TrainConfig` FIELDS (nested PPO knobs under `"ppo"`), never CLI flag spellings,
  so `TrainConfig` remains the one configuration authority and there is no second naming
  scheme to drift from it. Keys beginning with `_` are comments; an UNRECOGNIZED key
  RAISES rather than being ignored, because a knob silently left at its default produces
  a run whose file says one thing and whose behaviour is another. Resolution is
  **dataclass/CLI defaults < preset < EXPLICITLY typed CLI flags**. "Explicit" is
  measured by re-parsing argv through a throwaway parser whose defaults are
  `argparse.SUPPRESS` (`_explicit_cli_dests`) — a parsed namespace cannot tell an absent
  flag from one passed its own default, and inferring it from the VALUE would let every
  differing default silently override the preset. Both parse passes consume ONE vector
  (`_effective_argv`): `argparse` reads `None` as `sys.argv[1:]`, and `main()` is normally
  called with no argument, so reading `None` as `[]` anywhere would make every typed flag
  look un-typed. Symbols: `load_config_file`, `resolve_train_config`, `_effective_argv`,
  `_explicit_cli_dests`, `_CLI_FIELD_BY_DEST` / `_CLI_PPO_FIELD_BY_DEST` (the ONE
  dest→field mapping), `_CONFIG_TUPLE_FIELDS`.
- **The repository short-probe preset — `configs/graph_train/final_cell_probe.json`.**
  The bounded short probe, and deliberately the ONLY preset the repository owns: 2
  scheduled training iterations × 4 scheduled attempts, `base_seed = 0`, `eval_every = 2`,
  4 fixed held-out seeds from `1_000_000`, giving one `pre_update` and one `post_update`
  matched round; the final 3-agent / 3-known / 3-hidden cell with its 200 km / 100 km
  geometry and `include_sams = false`; FD-BASELINE-v1 unchanged; `visual_artifacts = true`.
  Every field it sets that also has a dataclass default AGREES with that default, so the
  preset RESTATES the approved cell instead of retuning it (test-enforced). **TWO
  SCHEDULED ITERATIONS DO NOT IMPLY TWO PRODUCTIVE PPO UPDATES:** `updates_completed`
  advances only when the updater actually runs epochs, so a successful zero-wake iteration
  leaves it unchanged and the value may be 0, 1 or 2. Productive-update yield is one of
  the things the probe MEASURES; nothing may assume it. No long-baseline preset exists.
- **`run_config.json:/config_source` — ALWAYS a structured object, never `null`.** One
  schema, one construction site (`config_source_record`, whose `resolved_from` is a
  REQUIRED argument, never inferred), and exactly THREE truthful kinds
  (`_CONFIG_SOURCE_KINDS`): `config_file` (a command line naming a preset; `path` /
  `absolute_path` say which), `cli_defaults` (a command line with no `--config`), and
  `direct_config` (a `TrainConfig` built in Python and handed straight to `train()` — what
  `_selftest` and any importing script do). The record also carries `config_fields` (what
  the preset supplied) and `cli_overrides` (what an explicit flag took back off it), and
  it is validated for internal consistency: only `config_file` may carry a path, and it
  must carry one. The three kinds exist because a provenance field that is WRONG in a
  believable way — a direct call recorded as `cli_defaults` — is worse than one that is
  absent.

The figure and presentation bullets of this block are contracted in
[artifacts and metrics §2](artifacts_metrics.md#2-figures-and-presentation-invariants).

## 5. Episode designs, GENERALIZED-V1 sampler and 18-stratum benchmark

**GENERALIZED-V1 EPISODE-DESIGN SELECTOR, TRAINING CARDINALITY SAMPLER, FROZEN STRATIFIED
BENCHMARK MANIFEST AND RUN-LEVEL PERSISTENCE — `rl/training/graph_generalized.py` (NEW) +
`rl/training/graph_train.py` + `rl/training/graph_rollout.py` + `rl/training/graph_reward.py`
+ `rl/training/graph_episode_setup.py` (`db79013`, integrated `b4daa8c`, PR #40).**

This is the HARNESS-and-POPULATION layer over the already-locked Task-1/2/3 policy seams
(FOUR low-level policy ids, opening the five pipeline seam sites
[runtime §1](runtime.md#1-the-end-to-end-pipeline) enumerates). It adds NO
new episode mechanism: the bounded-backoff placement geometry, the FD certification physics,
the post-FD boundary semantics and the continuation-reference arithmetic are exactly the
contracts above, and this layer only names their policy ids, decides which POPULATION an
episode is drawn from, and makes what those layers already produce durable and aggregable.

**ONE SELECTOR, ONE RESOLUTION SITE, AND THE BUNDLE IS ALL-OR-NOTHING.**
`graph_generalized.EPISODE_DESIGNS = (EPISODE_DESIGN_FIXED_CELL_V1,
EPISODE_DESIGN_GENERALIZED_V1, EPISODE_DESIGN_GENERALIZED_V2)`, spelled `fixed_cell_v1` /
`generalized_v1` / `generalized_v2`, and
`resolve_episode_design(id) -> EpisodeDesign` is the ONE site that turns an id into the four
low-level policy ids. `EpisodeDesign` is a frozen record carrying the design id plus
**EXACTLY FOUR** low-level policy ids and no others — `design`, then `hidden_policy`,
`eligibility_policy`, `post_fd_wake_policy`, `reference_policy` — plus its three
predicates (`generalized`, `generalized_v1_design`, `route_relative_population`) and
`to_record()`; deliberately a record rather than four loose strings, **so a
partially-resolved bundle is not expressible.**

**`generalized` MEANS V1 *OR* V2, AND THE THREE PREDICATES ARE DELIBERATELY DISTINCT.**
`EpisodeDesign.generalized` is true for `generalized_v1` AND `generalized_v2`, because the
two share all four low-level policy ids and every generalized HARNESS behaviour that keys
off it — the `seeded_variable` fuel-damage mixture, the successful-episode training quota
and its bounded attempt budget, the dynamic construction provenance.
`EpisodeDesign.generalized_v1_design` is true for EXACTLY `generalized_v1`, and it is what
the V1-only constructs test against — the 18-stratum benchmark, the V1 frozen manifest, the
V1 benchmark-preflight path (`graph_benchmark_preflight._require_preflight_config` checks
`cfg.design.generalized_v1_design`, not `cfg.generalized`) and the approved
training-reward plateau stopping rule. **Since PR #59 the SEPARATE ten-cell GENERALIZED-V2
benchmark is dispatched on `route_relative_population` instead**:
`run_benchmark_preflight` delegates a V2 config to `_run_v2_benchmark_preflight` BEFORE
`_require_preflight_config` is ever reached, `evaluate_benchmark` delegates to
`_evaluate_v2_benchmark`, and `train` loads `load_v2_benchmark_manifest` — so neither
benchmark construct is ever reached through the other's predicate (the GENERALIZED-V2
BENCHMARK block below). `EpisodeDesign.route_relative_population` is true
for EXACTLY `generalized_v2` and is the ONE predicate behind the two-stage population, so
the pre-solve `(A, K)` draw, the post-solve `H | R` draw and the P1-backend requirement
can never be reached one without the others. **A check that means "V1" must say
`generalized_v1_design`; writing `generalized` there would silently admit V2.**

**WHAT THE SELECTOR DOES *NOT* CARRY, AND MUST NEVER BE DESCRIBED AS CARRYING.**
`EpisodeDesign` holds FOUR low-level policy ids — **not five** — and two further
generalized-path behaviours sit BESIDE it rather than inside it:

- **`fuel_damage_mode` IS A SEPARATE `TrainConfig` / `RolloutConfig` FIELD.** Under
  `generalized_v1` `validate()` REQUIRES it to be `seeded_variable` (below), but the
  selector neither carries it, sets it, nor returns it, and on the fixed-cell path it keeps
  its own independent value. A required companion setting is not a resolved policy id.
- **THE GENERALIZED TRAINING CARDINALITY SAMPLER IS HARNESS / POPULATION BEHAVIOUR**, not a
  fifth policy id. `episode_cardinality` consults it because `cfg.generalized` is true;
  `EpisodeDesign` neither names nor returns it, and `cardinality_sampler_record()` is a
  separate `run_config.json` block.

- **`fixed_cell_v1` IS THE DEFAULT AND IS THE PRESERVED HISTORICAL BEHAVIOUR IN FULL.**
  `FIXED_CELL_V1` = (`exact_v1`, `legacy_selected_ego_v1`, `single_wake_v1`,
  `static_t0_v1`) — every id is the DEFAULT of the layer that owns it, so a fixed-cell run
  resolves exactly what those layers would have chosen alone. **The fixed-cell measurements
  (`737b4bf`, `bf1e045f`) were taken on this bundle and remain measurements OF IT.**
- **`generalized_v1` IS THE COMPLETE APPROVED BUNDLE IN ONE WORD.** `GENERALIZED_V1` =
  (`bounded_backoff_v1`, `certified_both_severities_v1`, `completion_boundary_v1`,
  `event_conditioned_continuation_v1`). It is one word rather than four knobs because the
  four policies were designed, reviewed and locked together: a run that enabled three of
  them would be a design nobody approved while still recording itself as generalized.
- **AN UNKNOWN ID RAISES**, never falls back on the historical bundle and is never
  case-folded into a match — a run that quietly measured the fixed cell while its config
  said `generalized_v1` is a mislabelled measurement, which is worse than a crash.
- **`training_mode` IS AN ORTHOGONAL SELECTOR AND IS UNCHANGED.** `actor_only` / `ctde`
  ([policy and CTDE §4](policy_ctde.md#4-phase-b-ctde)) selects the TRAINING ALGORITHM; `episode_design` selects the EPISODE
  POPULATION. Neither reads the other, `TrainConfig.ctde_enabled` still reads
  `training_mode` and nothing else, and **selecting a training mode does not alter the
  episode-design contract in any way.**
- **THE HARNESS FIELDS ARE `TrainConfig.episode_design` / `RolloutConfig.episode_design`
  (both defaulting to `fixed_cell_v1`) AND `TrainConfig.benchmark_manifest`, AND THERE IS NO
  PER-POLICY FIELD.** `hidden_policy`, `eligibility_policy`, `post_fd_wake_policy` and
  `reference_policy` are resolved from the selector and are not independently settable from
  a config, a preset or a CLI flag. `--episode-design` and `--benchmark-manifest` are the
  two new trainer flags; `graph_rollout` exposes `--episode-design` (and `--fuel-damage-mode`
  beside it, because `generalized_v1` REQUIRES `seeded_variable` and a flag that could only
  ever be rejected would be a trap).

**`TrainConfig.validate()` REFUSES A HALF-CONFIGURED GENERALIZED RUN, BEFORE ANY COMPUTE.**
A `generalized_v1` run MUST use `fuel_damage_mode = seeded_variable` (the approved
0.50 clean / 0.25 mild / 0.25 severe mixture — `seeded_mixture` would make every damaged
episode structurally SEVERE while the record claimed the generalized design, and `off` would
leave the certified eligibility policy nothing to certify), and — **with evaluation
enabled** — MUST name a `benchmark_manifest`, because the fixed held-out seed band carries
no stratum and evaluating a generalized run on it would measure an UNSTRATIFIED population
under a stratified label. Conversely a `benchmark_manifest` set on a `fixed_cell_v1` run is
REFUSED: that run would build every world from the fixed cell while reporting stratum labels
it never varied. A generalized run also prints a loud `[WARN]` that `num_agents` / `n_known`
/ `n_hidden` are NOT read. `RolloutConfig.validate()` applies the same design and mixture
verdicts, from the same resolution site.

**THE GENERALIZED TRAINING POPULATION, AND ITS OWN PRIVATE RNG DOMAIN.**
`sample_generalized_cardinality(*, episode_seed)` draws, in this order and no other:
`A ~ Uniform(GENERALIZED_AGENT_COUNTS)` = `{2,3,4}`; then `H_requested | A ~ Uniform({1..A})`
CONDITIONAL on `A`; and `K = A` by definition, not by a draw. So a requested world holds
`A + 1` … `2A` targets. `CARDINALITY_SAMPLER_POLICY = "generalized_cardinality_uniform_v1"`.

- **A FOURTH SHA-256 SEED DOMAIN, AND THE SEPARATION IS LOAD-BEARING.**
  `CARDINALITY_RNG_DOMAIN = "generalized_cardinality_v1"`, via `derive_cardinality_seed`
  (`SHA-256("generalized_cardinality_v1:<episode_seed>")[:8]` — the same construction as the
  three fuel-damage domains and disjoint from all of them). Taking the draw from
  `fuel_damage_v1` would insert draws between that stream's mixture bit and its ego selection
  and CHANGE WHICH EGO every damaged episode picks; taking it from global `random` would make
  a world's shape depend on how many global draws ran before it; taking it from the
  hidden-placement rng would make it depend on how many placement candidates were rejected;
  taking it from torch's generator would couple the world's SHAPE to the actor's action
  sampling. **The sampler constructs its own `random.Random` and consumes NOTHING from global
  `random`, from torch, or from any fuel-damage or placement stream**, and the two index
  draws are written as explicit `randrange` calls so their number and order are a pinned
  contract.
- **`EpisodeCardinality` CARRIES THE REQUEST AND ITS SOURCE, AND THE REQUEST IS NEVER
  REWRITTEN.** `agent_count` / `known_count` / `hidden_requested` / `source`, where `source`
  ∈ `CARDINALITY_SOURCES` = (`fixed_cell`, `generalized_sampler`, `benchmark_manifest`) —
  because "A=3" produced by the fixed cell, drawn by the sampler, and frozen into a stratum
  are three different facts about the population. Under bounded backoff a world may
  legitimately realize FEWER hidden targets; that shortfall is a RECORDED outcome, never a
  rewrite of the request, never a retry and never a replacement. `hidden_short_realized` is
  recorded per episode beside both numbers.
- **`episode_cardinality(cfg, seed, *, benchmark_cardinality=None)` IS THE ONE SITE that
  answers "what cardinality is this scheduled attempt?"** — a benchmark member states its own
  (verbatim from its frozen stratum; re-deriving it from the seed would silently leave the
  stratum it was frozen into), a generalized TRAINING episode samples one, and a fixed-cell
  episode reads the configured cell verbatim with no draw and no seed. **The sampler's rng
  domain is not even touched on the fixed-cell path.**
- **`_scheduled_cell(cardinality, construction_audit)` RESOLVES WHAT THE ROSTER MUST HOLD.**
  With no audit (`exact_v1`) the expected hidden count IS the requested one, exactly as
  before. With a `bounded_backoff_v1` audit the expectation is the audit's `hidden_realized`,
  and the audit is VERIFIED rather than trusted: it must agree with the schedule about `A`
  and about `H_requested`, else `EpisodeRosterError`. `_require_scheduled_cell` is otherwise
  unchanged, so a short realization is ACCOUNTED, not rejected.

**THE FROZEN STRATIFIED BENCHMARK — TASK 4 DELIVERED THE MECHANISM ONLY.** Task 4 delivered
the SCHEMA, the BUILDER, the canonical serialization, the content hash, the LOADER and its
verification, the CONSUMER (`evaluate_benchmark`) and the identity checks, and **it did NOT
select a worlds-per-cell SCALE and did NOT generate, commit or freeze a benchmark
POPULATION** — choosing a scale was left to the later bounded runtime / solver validation
step. **AT TASK 4 `build_benchmark_manifest` HAD NO PRODUCTION CALLER AND WAS EXERCISED BY
TESTS ALONE.**

**THAT CALLER STATEMENT IS HISTORICAL, AND TASK 5 CHANGED IT.** Since PR #43 the production
caller is `graph_benchmark_preflight.run_benchmark_preflight`, which invokes
`build_benchmark_manifest` **only after EVERY required base-cell quota has successfully
filled** — a failed preflight creates NO manifest at all (the Task-5 contract below). **A
PRODUCTION CALLER EXISTING IS NOT A POPULATION EXISTING**: the preflight refuses to invent
the scale just as firmly as the builder does, and **no benchmark manifest is committed or
tracked in the repository.**

**SCOPE OF THAT STATEMENT.** No benchmark manifest is committed or tracked in the repository.
Tests and engineering validation legitimately build transient manifests in memory and in
temporary directories, and manifests consumed by recorded runs live outside the repository;
their identities are recorded in the [measurement history](../history/measurements.md#2-measurement-records) and the [handoff](../../graph_rl_project_handoff.md).

- **18 REQUESTED STRATA, built from a product rather than listed** so the count cannot drift
  from the design: `A ∈ {2,3,4}` × load bucket ∈ `LOAD_BUCKETS = (low, high)` × cell ∈
  `BENCHMARK_CELLS = (clean, mild, severe)` = `3 × 2 × 3 = 18` (`BENCHMARK_STRATA`).
  `hidden_requested_for(A, bucket)` is `1` for LOW and `A` for HIGH. The six `(A, bucket)`
  pairs are the `BENCHMARK_BASE_CELLS`, and a manifest missing one is REFUSED — it would be
  missing three of the eighteen strata and is not this benchmark.
- **THE MATCHED UNIT IS ONE WORLD GROUP OF THREE MEMBERS.** `BenchmarkWorld` =
  `(agent_count, load_bucket, world_ordinal, seed, optional preflight)`; its three members
  are `BENCHMARK_MEMBERS` = `((clean, forced_clean), (mild, forced_mild),
  (severe, forced_severe))` — the EXISTING forced fuel-damage modes, reused rather than newly
  invented, so a benchmark member is an ordinary evaluation member with a stratified world
  behind it. All three members share the SAME seed, hence the same generated world, the same
  requested cardinality, the same solved `A_init`, the same hidden geometry and — under
  `certified_both_severities_v1`, whose walk depends on the episode seed ALONE — the SAME
  certified damaged ego at the same certified event point. **Only the condition / severity
  varies.**
- **EVERY IDENTITY IS ID-FREE; GENERATED UUID EQUALITY IS USED NOWHERE.** Generated agent and
  target uuids are not seed-derived
  ([§11](#11-known-limitations-and-open-items)), so a world's identity is its
  `(A, load bucket, world ordinal)` `group_key` plus its SEED, a candidate's identity is its
  ORDINAL, and a REALIZED world's identity is `WorldIdentity` = the realized known/hidden
  counts (taken from the RAW pre-solve world snapshots through the roster, never from an
  allocated-only task list), the coordinates-only `geometric_fingerprint(ctx.placements)`,
  the certified ego's ORDINAL, and `certificate_fingerprint(...)` — a content hash of the FD
  event certificate **with the ego uuid REMOVED** (`_CERTIFICATE_ID_FIELDS`). `None`
  components are truthful absences: a legacy uncertified plan has no certificate and no
  ordinal, and fabricating either would make "these two worlds certified the same event"
  answerable where it is not.
- **MATCHED IDENTITY IS VERIFIED, NOT ASSUMED, AND A DISAGREEMENT ABORTS.**
  `require_world_matches_manifest(world, observed)` compares a completed member against the
  manifest's FROZEN preflight when it has one (a no-op when it does not — inventing an
  expectation would be worse than having none), and `require_matched_group_identity(world,
  identities)` compares the members that COMPLETED against each other. Either raises
  `BenchmarkIdentityError`, which `graph_train` routes as a MEASUREMENT-INTEGRITY ABORT
  alongside `_VisualArtifactError` / `MeasurementIntegrityError` / `FuelDamageIntegrityError`
  / an aborting `ReferenceIntegrityError`. **A member is REFUSED, never regenerated or
  substituted:** two members that built different worlds would make their delta a
  between-worlds comparison wearing a within-world label. An INCOMPLETE group is a
  denominator question, not an integrity fault, and is accounted separately.
- **COMPLETE GROUPS ONLY.** `BENCHMARK_DELTAS` = `mild − clean`, `severe − clean`,
  `severe − mild`, each averaged over world groups whose EVERY member succeeded
  (`benchmark_delta_over = "world_groups_with_all_members_successful"`). A failed member is
  recorded once in `episode_failures.jsonl` — carrying the world it was SCHEDULED to build,
  so its stratum's denominator stays complete — its group becomes incomplete and contributes
  to NO delta, and **no other world, seed or stratum takes its place**. Per-stratum and
  per-cell reward MEANS remain each over that bucket's own successful subset, so the only
  within-world claims are the three deltas.

**THE MANIFEST IS CANONICAL AND CONTENT-ADDRESSED, AND THE LOADER AUTHENTICATES THE EXACT
STORED PAYLOAD.** `_canonical_json` (sorted keys, no insignificant whitespace,
ASCII-escaped, `allow_nan=False`) is the ONE serialization site behind both the hash and the
written file, so a manifest's identity cannot depend on which of the two produced the bytes.

**WHAT `manifest_id` IS THE HASH OF — STATED EXACTLY, BECAUSE THE SHORTHAND IS FALSE.**
`BenchmarkManifest.payload()` is the canonical content and **EXCLUDES `manifest_id`
itself**, because a self-referential identity is unverifiable; `manifest_id` is the SHA-256
of `_canonical_json(payload())`. `to_record()` is `payload()` **PLUS** `manifest_id`, and
`write_benchmark_manifest` writes `_canonical_json(to_record())`. **So the manifest_id is
the hash of the canonical PAYLOAD, and it is NOT the SHA-256 of the full serialized file
bytes** — the file additionally contains the id itself, so hashing the whole file cannot
reproduce it. `manifest_from_record` closes exactly that gap by RECONSTRUCTING the payload
from the document: it removes `manifest_id` to form `stored_payload`, hashes THAT and
compares it against the stored id (step 2), and then independently requires
`stored_payload` to equal the canonical payload the parsed manifest produces (step 4).
*(The source-code docstrings describe this as "the file hashes to its own id"; that shorthand
is false, and this contract and the code's behaviour are authoritative.)*

`manifest_from_record` verifies in this order, and **repairs nothing**:

1. the document must carry a non-empty `manifest_id` and declare THIS `schema`
   (`generalized_v1_benchmark_manifest`), `schema_version` (`1`) and `design`
   (`generalized_v1`);
2. **the STORED payload — the document minus `manifest_id` — is hashed EXACTLY AS FOUND**,
   with nothing normalized, re-sorted, filled in or dropped before the check, and must equal
   the stored id;
3. the semantic manifest is parsed and validated: canonical world order (base cell, then
   world ordinal, then seed — **never re-sorted on load**, because the stored order is part
   of the identity), well-formed and complete base cells, unique seeds, unique group keys,
   and each world's STATED `known_requested` / `hidden_requested` against its stratum;
4. **the stored payload must EQUAL the canonical payload the parsed manifest produces**
   (`_payload_differences` reports key-set differences first, then per-key value differences
   compared through `_canonical_json`), so an injected, missing or altered canonical field is
   REFUSED rather than quietly ignored.

**STEPS 2 AND 4 ARE BOTH REQUIRED AND NEITHER IMPLIES THE OTHER** — step 2 alone would accept
a **self-consistently rehashed forgery** that is not this schema's payload, and step 4 alone
would accept a correctly-shaped document carrying a different population's id. Both a
TAMPERED manifest and such a forgery are refused rather than re-hashed. **THE SCALE IS NEVER
DEFAULTED:** `build_benchmark_manifest` requires EXACTLY ONE of `worlds_per_cell` (with an
explicit `benchmark_base_seed`) or an explicit `worlds` list, and calling it with neither —
or with both — RAISES `BenchmarkManifestError`. A default there would silently make the
scientific-scale decision the runtime-validation step owns.

**THE BENCHMARK'S SEEDS ARE THE RUN'S ACTUAL EVALUATION SEEDS, AND HELD-OUTNESS IS CHECKED
AGAINST THEM.** `_require_benchmark_seeds_held_out(manifest, cfg)` uses
`manifest_seed_overlap(manifest, start=cfg.base_seed, stop=cfg.base_seed +
cfg.max_training_attempts)` and REFUSES the run — naming every offending seed, offering no
repair — if any manifest world seed lies inside the TRAINING band. **THE UPPER BOUND IS THE
MAXIMUM POSSIBLE TRAINING ATTEMPT BAND, WHICH IS THE TASK-5 CONTRACT AND IS AUTHORITATIVE
HERE TOO** — under `generalized_v1` that is `n_iterations *
generalized_max_attempts_per_iteration`, because a FAILED replacement attempt still SPENDS a
seed and therefore belongs inside the exclusion band. **On the fixed-cell path
`max_training_attempts` EQUALS `total_episodes`**, so the preserved historical statement is
unchanged there: one attempt per scheduled episode, and the same band this check always
used. This is a DIFFERENT
question from the legacy check, which compares the training band against
`eval_base_seed .. + eval_episodes`: a manifest-driven run does not evaluate that band at
all, so letting the legacy check stand in would be wrong in BOTH directions — an unused
configured band could falsely REJECT a properly held-out manifest, and could falsely VALIDATE
one that contains a training seed. `_require_benchmark_tag_namespace(manifest, cfg)` bounds
the scenario-tag namespace against the MANIFEST's member count rather than `eval_episodes`,
which `TrainConfig.validate` cannot know because it holds a path, not a population. **Both
run at LOAD time — after the manifest is loaded and BEFORE the run directory, the provenance,
the policy, the generator or any solver work exists** — so a refused run costs nothing and
leaves nothing behind. The two legacy bounds are deliberately NOT applied to a manifest run
(the training-tag bound still is, because benchmark tags share that one namespace), and that
is a correctness decision, not a relaxation. **HISTORICAL FIXED-CELL EVAL-BAND SEMANTICS ARE
UNCHANGED** — a `fixed_cell_v1` run keeps exactly the band, the formula and the checks it
always had.

**PROVENANCE RECORDS THE MANIFEST AS THE ACTUAL SEED SOURCE, AND THE UNUSED BAND AS UNUSED.**
`seed_bands(cfg, *, benchmark=None)` PRODUCES A BYTE-UNCHANGED BLOCK when `benchmark` is
omitted — every fixed-cell run and every pre-Task-4 caller — so the historical provenance
block keeps exactly the shape every existing reader and every preserved run artifact
already has, and the
historical source is identified by the ABSENCE of `benchmark_evaluation` rather than by a new
key. Supplied, the block sets `eval_seed_source = EVAL_SEED_SOURCE_MANIFEST`, EMPTIES
`eval_band` and `eval_seed_formula` so nothing in it can be read as the executed schedule,
and adds `benchmark_evaluation` (manifest id, world-seed count, members per round, the
ORDERED `seed_list_sha256` digest, the seeds themselves, the training band they were held out
against, the overlap count and `held_out_verified`) plus `unused_legacy_eval_band` marked
`executed: false`. The configured band is RETAINED rather than deleted — a reader must be
able to see it was configured AND not executed. `run_config.json` additionally gains an
`episode_design` block: the four resolved policy ids, `target_destruction_probability`, the
`cardinality_sampler` record (`null` under `fixed_cell_v1`, which is the truthful statement
that the cell was CONFIGURED rather than sampled), the `fixed_cell` counts (`null` under
`generalized_v1`) and the `benchmark_manifest` identity record with its content hash — so
"the two arms ran the same benchmark" is a CHECKABLE claim rather than an assertion.

**THE CONSTRUCTION PROVENANCE IS HONEST ABOUT A DYNAMIC CELL.** `_construction_record(cfg)`
returns the HISTORICAL block unchanged under `fixed_cell_v1`, where the configured cell IS
the executed one. Under `generalized_v1` the three count fields are read by NOTHING —
training cardinality is sampled per episode and benchmark cardinality comes from the
manifest — so writing them in the historical shape would let the artifact be read as "this
run executed 3/3/3", a plausible-looking FALSE statement about the population. The
generalized block therefore states `cardinality_source =
"per_episode_sampler_and_benchmark_manifest"`, `fixed_cell_config_used = false`,
`n_targets_emitted = null`, names where each half really comes from, points at
`episode_outcomes.jsonl` for the REALIZED per-episode counts, and keeps the configured
numbers ONLY under an explicitly-labelled `unused_fixed_cell_config` marked
`executed: false`. The geometry half is unchanged, because it really is configured and really
is applied to every generated world.

The persistence and aggregate paragraphs of this block are contracted in
[artifacts and metrics §4](artifacts_metrics.md#4-generalized-persistence-and-aggregates).

**THE DIAGNOSTIC ROLLOUT HAS SELECTOR PARITY AND STAYS DIAGNOSTIC.** `RolloutConfig` mirrors
`episode_design` field for field, resolves the same four policy ids from the same site,
samples the same TRAINING cardinality per seed, and records the design, the requested cell,
the realized hidden count, the construction audit and the reference decomposition per episode
(`u_oracle` is recorded as `null` — never coerced to `0.0` — under the event-conditioned
policy, with `u_ref` as the denominator source under BOTH). **It does NOT become a matched
scientific benchmark harness:** it runs the seeded MIXTURE only, trains nothing, evaluates no
matched group, and carries no `benchmark_manifest` field — matched clean/mild/severe worlds
and the frozen 18-stratum benchmark are an EVALUATION construct and live in
`graph_train.evaluate` / `graph_train.evaluate_benchmark`. Selecting `generalized_v1` there
samples the same training population a generalized training batch is drawn from, and makes NO
benchmark claim.

**WHAT IS PRESERVED, AND HOW.** `fixed_cell_v1` is BYTE-INVARIANT at the call boundary, by
the same keyword-OMISSION discipline `_artifact_kwargs` and `_ctde_kwargs` already use:
`_cardinality_kwargs` omits `cardinality` from the `_run_one_episode` call and
`_generalized_setup_kwargs` omits `hidden_policy` / `reference_policy` from the
`setup_episode` call, so a fixed-cell run makes exactly its pre-Task-4 calls rather than
passing a new keyword carrying a value the callee could have derived itself — which is also
the stronger invariance claim. `seed_bands` and `_construction_record` are byte-unchanged on
that path, and `evaluate_benchmark` is a SEPARATE function beside `evaluate` rather than a
rewrite of it. **`evaluate` — the legacy pair / triad round — keeps its population, its
schedule, its matched-group logic and every reported field**, and changes in exactly three
mechanical places: `ReferenceIntegrityError` joins its abort re-raise tuple, its failure
records now also carry the scheduled `cell`, and its outcome records now also carry
`design=cfg.design`. It is not byte-unchanged, and this contract does not claim it is.

**PURITY AND THE ACTING PATH.** `graph_generalized` imports no BLADE, gymnasium or torch,
holds no module-global randomness, and does no file I/O beyond a manifest JSON a caller names
explicitly; it must never import `graph_train` or `graph_rollout` — the harnesses import IT.
`GENERALIZED_AGENT_COUNTS` and `TARGET_DESTRUCTION_PROBABILITY` are TRANSCRIBED MIRRORS of
`graph_episode_setup.GENERALIZED_AGENT_COUNTS` and the pipeline's `probability` default, kept
here so the layer needs no BLADE-adjacent import, and TEST-ENFORCED against the originals —
the same discipline `graph_fuel_damage.rtb_command_for` and `graph_train.derived_split`
already use. **NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no design id, no
cardinality, no stratum label, no load bucket, no manifest field and no world identity enters
`GraphObservation` or the critic's `CentralGraphObservation`. A count of what is hidden, and
a label saying how hard a world is, are exactly the privileged quantities an ego cannot sense
([`CLAUDE.md` §3](../../CLAUDE.md#3-architecture--the-load-bearing-invariants)).

**WHAT IS EXPLICITLY NOT IN THIS TASK.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate future
Grade-A research task**. The frozen solver / BONMIN and the vendored BLADE engine are
untouched. No actor, encoder, `ActionHead`, PPO, GAE or critic architecture change; **no new
`MetaAction`**; no change to terminal-on-last credit placement or to `graph_reward`'s static
`static_t0_v1` formula; no change to the no-communication boundary; no peer behaviour change
and no communication channel of any kind; `DETECTION_KM`, the B2 geometry and the training
seed formulas are unchanged. **TASK 4 SELECTED NO worlds-per-cell SCALE and generated,
committed or froze NO benchmark POPULATION**, and **no repository preset selects
`generalized_v1`** — those are statements about TASK 4's SCOPE and stay accurate as such.
Current scale, manifest and measurement state is in the [handoff](../../graph_rl_project_handoff.md). The approved Phase-A
(`737b4bf`) and FD-VARIABLE-SEVERITY-v1 (`bf1e045f`) measurements are untouched and remain
measurements of the `fixed_cell_v1` bundle.

## 6. Training quota and benchmark preflight

**GENERALIZED-V1 TASK 5 — SUMMARY-POPULATION CORRECTION, THE SUCCESSFUL-EPISODE TRAINING
QUOTA, AND THE DETERMINISTIC BENCHMARK PREFLIGHT —
`rl/training/graph_train.py` + `rl/training/graph_benchmark_preflight.py` (NEW)
(`312f586` / `5dfcd8b`, PR #42 and `4af6c5a` / `b3c2e01f`, PR #43).**

This layer adds NO episode mechanism. The bounded-backoff geometry, the FD certification
physics, the post-FD boundary semantics and the continuation-reference arithmetic are
exactly the Task-1/2/3 contracts above, and the Task-4 selector, sampler, manifest schema
and persistence are exactly the Task-4 contract. What changes is (1) which population a
persisted TRAINING aggregate is taken over, (2) what `episodes_per_iteration` COUNTS and
how many attempts obtaining it may cost, and (3) that a benchmark population is now
SELECTED — once, before the freeze — instead of being frozen unchecked.

**1. `train_by_*` SUMMARY BUCKETS COUNT TRAINING ATTEMPTS ONLY (PR #42).**
`_generalized_summary`'s two named TRAINING aggregates — `train_by_agent_count` and
`train_by_hidden_requested` — are derived from the TRAINING-phase rows of both canonical
streams alone. Both streams mix phases BY DESIGN: an outcome row carries one of
`_ARTIFACT_PHASES` = (`pre_update`, `train`, `post_update`) and a failure row carries
`train` or `eval`, so the function first filters each to `phase == _ARTIFACT_PHASE_TRAIN`
(`train_successes` / `train_failures`) and passes the two populations to `_by` as EXPLICIT
ARGUMENTS rather than closing over them — the phase a bucket is taken over is stated at
the call site and cannot drift back to whatever was in scope.

- **THE FAULT IT CLOSES IS A DENOMINATOR FAULT, NOT A COSMETIC ONE.** A bucket built from
  the unfiltered streams folded held-out EVALUATION attempts into a denominator whose NAME
  says training. The two populations are scheduled independently — and under a frozen
  benchmark an evaluation round RE-MEASURES the same worlds every round — so their sum
  describes nothing, and `attempted` in a `train_by_*` bucket would grow with the number of
  evaluation rounds a run happened to perform.
- **TRAINING FAILURES REMAIN REPRESENTED.** The filter is on PHASE, never on outcome: a
  failed TRAINING attempt still carries the world it was SCHEDULED to build and still
  contributes its `failed` count, so a HIGH-load stratum's training denominator stays
  complete. `attempted == successful + failed` per bucket still holds by construction.
- **EVERY OTHER GENERALIZED AGGREGATE KEEPS ITS OWN INTENDED POPULATION, UNCHANGED.**
  `cardinality_requested_vs_realized`, `construction_backoff_rejections`,
  `fd_eligibility_rejections`, `fd_eligibility_selected_ordinals`, `post_fd_adaptation`
  and the reference block are still taken over ALL successful generalized episodes;
  `reference_fault_attrition` is still taken over the whole failure ledger; and the
  `benchmark` block is still built from the eval records, with its FINAL round kept
  STRICTLY APART from `strata_attempt_totals_across_rounds` and still flagged
  `totals_across_rounds_are_repeated_measures`.
- **THIS IS A PERSISTED-SUMMARY CORRECTION AND NOTHING ELSE.** It is a change to ONE
  DERIVED aggregation site. No scenario, world-construction, reward, solver, PPO, CTDE,
  fuel-damage, seed, evaluation-schedule or population-SELECTION semantics changed, the
  canonical `episode_outcomes.jsonl` / `episode_failures.jsonl` streams are byte-unchanged,
  and no episode behaves differently. What a completed run DID is unchanged; what its
  summary SAYS about the training population is corrected.

**2. WHAT `episodes_per_iteration` COUNTS — TWO ATTEMPT POLICIES, SELECTED BY
`episode_design` AND BY NOTHING ELSE (PR #43).**
`TRAINING_ATTEMPT_POLICIES` = (`TRAINING_ATTEMPT_POLICY_SCHEDULED` =
`scheduled_attempts_v1`, `TRAINING_ATTEMPT_POLICY_QUOTA` =
`successful_quota_with_deterministic_replacement_v1`), and
`TrainConfig.training_attempt_policy` is the ONE predicate behind the choice — it reads
`cfg.generalized` and nothing else.

- **`scheduled_attempts_v1` IS THE FIXED-CELL PATH AND IS THE PRESERVED HISTORICAL
  BEHAVIOUR.** `episodes_per_iteration` is a count of scheduled ATTEMPTS; each is made
  exactly once; a failure is recorded and its slot is simply LOST; and the batch that
  reaches the updater is whatever survived. The loop still calls `global_episode_index` and
  `train_seed`, so the historical seed formula is the one that runs. **The fixed-cell
  measurements (`737b4bf`, `bf1e045f`) were taken under this policy and nothing about it
  moves.**
- **`successful_quota_with_deterministic_replacement_v1` IS THE GENERALIZED PATH.**
  `episodes_per_iteration` is a count of SUCCESSFUL episodes the PPO/CTDE batch must hold.
  The generalized population is drawn from a sampler whose worlds legitimately fail
  construction or FD certification, so under a fixed attempt count the PPO batch SIZE would
  be a function of world attrition and two arms' learning curves would not be comparable.
- **THE BUDGET IS EXPLICIT, REQUIRED AND NEVER DEFAULTED — UNDER *BOTH* GENERALIZED
  DESIGNS.** `TrainConfig.generalized_max_attempts_per_iteration` is `None` on the
  historical path, where `validate()` REFUSES it if set (a fixed-cell run must not silently
  acquire replacement behaviour), and REQUIRED whenever `design.generalized` — that is,
  under `generalized_v1` **and** `generalized_v2` alike — where `validate()` also
  rejects a non-`int` (a `bool` included) and any value `< episodes_per_iteration`. There is
  deliberately no default: the number decides how much world attrition the campaign
  tolerates — the same reason `build_benchmark_manifest` refuses to invent `worlds_per_cell`
  — and it is also the bound the run's MAXIMUM POSSIBLE training-attempt seed band is
  computed from. **THE BENCHMARK HALF OF THAT BOUND APPLIES UNDER BOTH GENERALIZED DESIGNS
  WHENEVER A RUN EVALUATES:** the maximum band is what the run's frozen manifest is verified
  to be held out FROM — the 18-stratum manifest under `generalized_v1`, and under
  `generalized_v2` EVERY world seed of the ten-cell manifest, BOTH profiles, whichever
  profile the run evaluates. A run that evaluates nothing still treats the band as its own
  seed-consumption bound.
  `TrainConfig.max_attempts_per_iteration` is the ONE property behind it and RAISES rather
  than guessing when a generalized config reaches it without one. The CLI exposes
  `--generalized-max-attempts-per-iteration`.
- **WHAT AN ORDINARY EPISODE FAILURE DOES, EXACTLY.** It is recorded ONCE in
  `episode_failures.jsonl` (phase `train`, with its scheduled cell and cardinality); its
  attempt SPENDS its seed and its run-wide ordinal, both of which are advanced BEFORE the
  attempt is made, so the failure handler's `continue` can never re-use either; it is NEVER
  retried at that seed; it never reaches `tally.success`, the durable outcome stream or the
  PPO/CTDE buffer; and it is REPLACED by the NEXT deterministic attempt. Nothing is
  reseeded, substituted, reclassified or band-shifted — `skip_and_account_v1` is unchanged
  in every one of its parts except that the lost slot is now refilled.
- **ONE RUN-WIDE MONOTONE ATTEMPT ORDINAL, AND THE SEED IS DERIVED FROM IT.**
  `global_attempt_ordinal` starts at `0` and advances on EVERY attempted training episode of
  the run — successful or failed, in every iteration — and
  `train_attempt_seed(cfg, attempt_ordinal)` is `base_seed + attempt_ordinal`. That is what
  makes a replacement DETERMINISTIC (it is simply the next ordinal), a failed seed
  UNRECOVERABLE, and every training artifact tag unique even when an iteration takes more
  attempts than it collects episodes — so no replacement can overwrite the artifacts of the
  attempt it replaced. `train_attempt_seed` is a GENERALIZATION of `train_seed`, not a
  competitor: where every slot is attempted exactly once,
  `train_attempt_seed(cfg, global_episode_index(cfg, i, j)) == train_seed(cfg, i, j)`
  identically. On the fixed-cell path the loop still uses the historical formulas AND
  VERIFIES the run-wide counter agrees with them, raising `MeasurementIntegrityError` on a
  divergence rather than measuring an unknown population.
- **EXHAUSTING THE BUDGET ABORTS — IT NEVER UPDATES ON A PARTIAL BATCH.** Reaching
  `max_attempts_per_iteration` before the quota is full raises `TrainingQuotaError`, a
  SUBCLASS of `MeasurementIntegrityError` (so every existing abort re-raise already routes
  it correctly and no future handler can account it as episode attrition). It is raised at
  the TOP of the collect loop, before any further attempt and before the updater, so no
  partial PPO/CTDE update occurs, no seed is retried, no failure is reclassified and no
  budget is raised mid-run. It is NOT a verdict on the worlds: it says the attrition rate is
  higher than the operator planned for, which is a scheduling fact to inspect in
  `episode_failures.jsonl`.
- **IT CHANGES NOTHING ABOUT ACTOR-VS-CTDE EXECUTION SEMANTICS.** The attempt policy is
  resolved from `episode_design`; `training_mode` remains the ORTHOGONAL selector, the two
  buffer kinds and the two updaters are exactly as Phase-B left them, and both modes share
  ONE attempt-seed derivation. Evaluation and inference remain actor-only in both modes.
- **OBSERVABILITY, ON BOTH DESIGNS.** A training record now carries
  `training_attempt_policy`, `successful_episodes_required`, `max_attempts_per_iteration`
  and `n_replacement_attempts` (`max(0, n_attempted_iter - quota)`; always `0` under the
  historical policy, which replaces nothing); `run_config.json:/training/attempt_policy`
  states what `episodes_per_iteration` counts and the `max_possible_training_attempts` every
  held-out claim is made against; and `run_summary.json` derives
  `training_attempt_policy` / `train_replacement_attempts` / `train_iterations_at_full_quota`
  from `train_records.jsonl` — ONE metric path, present and truthful on BOTH designs, so a
  reader never infers the policy from an absence. The extra console line prints only under
  the quota policy.

**3. HELD-OUTNESS IS CHECKED AGAINST THE MAXIMUM POSSIBLE ATTEMPT BAND.**
`TrainConfig.max_training_attempts` is `n_iterations * max_attempts_per_iteration` — the
MOST training seeds a run can possibly consume — and it, never `total_episodes`, is the band
every held-out claim is made against:
`[base_seed, base_seed + n_iterations * generalized_max_attempts_per_iteration)`.
**A FAILED REPLACEMENT ATTEMPT STILL CONSUMES ONE SEED AND THEREFORE BELONGS TO THE
EXCLUSION BAND.** Checking against the successful-episode quota instead would leave a
corridor of seeds a run with ordinary attrition really does train on while its benchmark was
certified held out — a held-out failure that produces entirely normal-looking numbers.
Three sites consume it, with the same reasoning at each: `TrainConfig.validate`'s legacy
train-vs-eval overlap test and its scenario-tag namespace bound;
`_require_benchmark_seeds_held_out`, through `manifest_seed_overlap(manifest,
start=base_seed, stop=base_seed + max_training_attempts)`, which REFUSES the run naming
every offending seed and offering no repair; and `seed_bands`, whose `train_band` now counts
`max_training_attempts`. **ON THE FIXED-CELL PATH `max_training_attempts` EQUALS
`total_episodes`**, so all three checks and the historical `seed_bands` block are
byte-unchanged there; the generalized path ADDS a `train_attempt_policy` block and restates
`train_seed_formula` over the run-wide ordinal, so `count` is never misread as the number of
episodes the run collected.
**UNDER `generalized_v2` THE BAND IS COMPUTED THE SAME WAY AND MEANS THE SAME THING — the
MOST training seeds the run can possibly consume — AND, SINCE PR #59, AN EVALUATING V2 RUN
HOLDS ITS FROZEN TEN-CELL MANIFEST OUT FROM IT.** `train` loads the V2 manifest and calls
the SAME `_require_benchmark_seeds_held_out` over `manifest.seeds()` — EVERY world seed,
BOTH profiles, not merely the profile the run evaluates — before the run directory, the
provenance, the policy, the generator or any solver work exists (the GENERALIZED-V2
BENCHMARK block below).

**4. THE DETERMINISTIC BENCHMARK PREFLIGHT — POPULATION SELECTION, ONCE, BEFORE THE FREEZE —
`rl/training/graph_benchmark_preflight.py`.**
`PREFLIGHT_POLICY = "deterministic_per_cell_window_v1"`. **THIS IS THE `generalized_v1`
PATH.** Since PR #59 `run_benchmark_preflight` is DESIGN-AWARE and delegates a
`generalized_v2` config to the SEPARATE fail-closed ten-cell V2 path (the GENERALIZED-V2
BENCHMARK block below) BEFORE this path's checks run; on this path
`_require_preflight_config` still tests `cfg.design.generalized_v1_design`, not
`cfg.generalized`, and still refuses every non-V1 config that reaches it. So a
V2 config can never be
frozen into V1 strata its population never varied. Under `generalized_v1` a candidate
world may legitimately fail: bounded-backoff construction can refuse it, and the certified
FD eligibility walk can find no ego supporting BOTH severity bands at a predicted event
state. A manifest frozen without checking would carry such a world FOREVER, and every
validation round of every arm would fail the SAME member again — a permanently missing
stratum member wearing the label of ordinary attrition. So the replacement happens ONCE,
HERE, BEFORE the freeze.

- **THE SCALE IS NEVER DEFAULTED.** `run_benchmark_preflight` requires `worlds_per_cell`,
  `benchmark_base_seed` and `max_candidates_per_cell` EXPLICITLY; it refuses
  `worlds_per_cell < 1` and `max_candidates_per_cell < worlds_per_cell` (a window smaller
  than the quota could never fill a cell even with no rejections), and `cell_windows`
  refuses a negative base seed or a window narrower than one candidate. **No scientific
  scale is chosen here** — that decision owns bounded runtime validation first, exactly as
  `build_benchmark_manifest` refuses to invent a world count.
- **SIX INDEPENDENT DETERMINISTIC WINDOWS, ONE PER `BENCHMARK_BASE_CELLS` ENTRY.**
  `cell_windows` gives cell ordinal `c` the half-open window
  `[benchmark_base_seed + c*M, benchmark_base_seed + (c+1)*M)` with
  `M = max_candidates_per_cell`. Independence is the point: however many candidates cell `c`
  rejects, cell `c+1` starts where it always would, so "we re-ran the preflight and the
  A=2/LOW worlds are the same worlds" is CHECKABLE rather than hoped for. A single shared
  stream would make every cell's accepted seeds a function of every earlier cell's
  attrition.
- **FIRST VALID CANDIDATES, IN ASCENDING CANDIDATE ORDER, EACH ATTEMPTED EXACTLY ONCE.**
  `_scan_cell` walks the window in ascending seed order, accepts the first
  `worlds_per_cell` worlds that satisfy the contract, and STOPS at the quota — the remaining
  seeds are simply never attempted, which is what makes a smaller `worlds_per_cell` a strict
  PREFIX of a larger one. A rejected candidate's seed is SPENT exactly once and never
  revisited, and the next seed replaces it. Both accepted and rejected candidates are
  recorded as `CandidateOutcome` (`CANDIDATE_OUTCOMES` = `accepted` / `rejected`), because
  the rejected ones are exactly what makes the accepted population auditable.
- **WHAT A CANDIDATE MUST SATISFY.** `probe_world` runs exactly the pipeline PREFIX
  `_run_one_episode` runs — the same reseed, the same `build_variation_config`, the same
  `setup_episode` with the same `_generalized_setup_kwargs`, the same
  `_episode_target_roster` and `_require_scheduled_cell` — and then ONE further step: the
  fuel-damage plan is built for ALL THREE `BENCHMARK_MEMBERS`, whose id-free world
  identities must AGREE (a disagreement is a `BenchmarkIdentityError`, an instrument fault,
  never a replacement-eligible rejection). The accepted world is frozen as its
  `WorldPreflight`.
- **NO POLICY IS BUILT, NO EPISODE IS RUN, AND NO OUTCOME MAY INFLUENCE ACCEPTANCE.**
  Nothing about reward, return or learned behaviour is computed anywhere in this module —
  selecting benchmark worlds by outcome would build the comparator out of the very quantity
  the comparison measures. **Population selection lives here; scientific evaluation lives in
  `graph_train.evaluate_benchmark`; the two must not merge.**
- **A SHORT REALIZATION IS AUDIT DATA, NOT AN AUTOMATIC FAILURE.** A contract-successful
  bounded-backoff world with `hidden_realized < hidden_requested` is ACCEPTED and is NOT
  rejected solely for that shortfall. The shortfall is RECORDED — in the candidate's
  `hidden_short_realized`, in the world's frozen `WorldPreflight`, and in the report's
  `totals.hidden_requested_vs_realized` histogram — and **no threshold and no verdict is
  invented here**: whether the resulting distribution is acceptable is a human / GPT
  scientific-review decision taken before any measurement.
- **REPLACEMENT-ELIGIBILITY IS THE SAME DISTINCTION THE TRAINER MAKES.** An ordinary
  world-construction / certified-FD-ineligibility rejection is replaceable BEFORE the
  freeze. `MeasurementIntegrityError`, `FuelDamageIntegrityError`, `BenchmarkIdentityError`
  and an ABORTING `ReferenceIntegrityError` PROPAGATE and stop the preflight, exactly as
  they stop a training run — a world that contradicts its own certificate implicates every
  world this preflight touched, and replacing it would freeze a population selected by a
  defect. The reference split is read through `graph_reward.reference_fault_aborts` and its
  stable SLUG, never through message text; `_rejection_reason` likewise recognizes
  `NO_FD_ELIGIBLE_EGO` and the closed `FD_ELIGIBILITY_REJECTION_REASONS` set by matching
  PUBLISHED CONSTANTS, and yields `None` — the truthful "this layer published no slug" —
  for anything else.
- **PROVENANCE IS A PRECONDITION.** Incomplete Git provenance REFUSES before anything is
  built; a DIRTY tree warns and proceeds. `_require_preflight_config` is deliberately
  NARROWER than `TrainConfig.validate`: a preflight TRAINS NOTHING, so it neither reads nor
  validates the training schedule, the attempt budget, the eval band or the benchmark path.

**5. THE COMPLETE-MANIFEST RULE, AND IMMUTABLE POST-FREEZE EVALUATION.**
A manifest is created ONLY after EVERY requested base-cell quota has been filled:
`build_benchmark_manifest` is reached only once the loop over all six windows has completed
without a shortfall. After the freeze, `graph_train.evaluate_benchmark` performs **NO
SUBSTITUTION** — a failed member is recorded once and skipped, its group becomes incomplete
and contributes to NO delta, no other world / seed / stratum takes its place, **a later seed
does not replace it**, and the manifest is never regenerated in order to route around an
evaluation failure. **A member failure is MEASURED as a member failure**, and matched-world
identity remains VERIFIED rather than assumed (`require_world_matches_manifest` against the
frozen preflight, and `require_matched_group_identity` across the members that completed).
`evaluate_benchmark` was NOT touched by this task.

**6. A FAILED PREFLIGHT PRODUCES NO POPULATION AND A DURABLE AUDIT (the PR #43 review fix).**
The quota verdict is taken in `run_benchmark_preflight`, AFTER `_scan_cell` has returned its
COMPLETE candidate audit — not inside the walk, which discarded the outcomes on the way out
and left a failed preflight with nothing to inspect but an exception message. When a cell
exhausts its window before its quota is filled, in this order: the exhausted cell's audit is
appended and preserved alongside the completed cells'; **NO later cell is scanned**; **NO
manifest is created or written**; a FAILED report is assembled through the SAME
`_build_report` site the successful path uses; that report is WRITTEN before the raise
whenever an output directory exists; and only then is `BenchmarkPreflightError` raised.

- **ONE FIELD TELLS THE TWO OUTCOMES APART, NEVER THE SHAPE OF THE DOCUMENT.** `status` ∈
  `PREFLIGHT_STATUSES` = (`PREFLIGHT_STATUS_COMPLETE` = `complete`,
  `PREFLIGHT_STATUS_FAILED` = `failed_incomplete`). On the failure path
  `complete = false`, `manifest_written = false` and `manifest = null` — there is no
  `manifest_id` to quote and no file hash to check.
- **THE FAILURE BLOCK NAMES WHAT FAILED AND WHERE.** `reason =
  PREFLIGHT_FAILURE_WINDOW_EXHAUSTED` (`candidate_window_exhausted`), the exhausted base
  cell and its ordinal, its half-open candidate window, and
  `worlds_requested` / `worlds_accepted` / `worlds_missing` / `n_candidates_attempted`.
  **The attempted candidate seeds and the rejection tallies SURVIVE** (`attempted_seeds`,
  `accepted_seeds`, `rejection_reasons`, `rejection_detail_reasons`, and every
  `CandidateOutcome` inside `cells` and `totals`). **Completed earlier cells survive**
  (`cells_completed`) and **unattempted later cells are NAMED** (`cells_not_attempted`) —
  "this cell was not scanned" and "this cell was scanned and filled" are different facts.
- **A FAILED REPORT'S `accepted_seeds` IS NOT A BENCHMARK POPULATION.** They are worlds
  accepted before the walk stopped — spent candidate seeds — and `status`, `complete` and
  the `null` manifest say so. **Do not describe a failed candidate audit as a benchmark
  population.**
- **WITH `output_dir=None` NO FILE IS INVENTED.** The same report travels on
  `BenchmarkPreflightError.report`, with `BenchmarkPreflightError.report_path` `None`. Both
  attributes are `None` for the input-validation raises (a bad scale, a non-generalized
  config, missing provenance), which fail before any candidate is attempted and therefore
  have no audit to carry.
- **A MANIFEST FILE ALREADY SITTING AT THE TARGET PATH IS NAMED, NEVER DELETED AND NEVER
  ADOPTED AS THE FAILED RUN'S OUTPUT.** `_existing_manifest` reports it as
  `stale_manifest_path`: removing a file this run did not write would destroy an earlier
  run's artifact, and naming it is what stops a reader finding a stale manifest beside a
  failure report and taking the two for one run.

**WHAT IS EXPLICITLY NOT IN THIS TASK.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate
future Grade-A research task**. The frozen solver / BONMIN and the vendored BLADE engine are
untouched. No actor, encoder, `ActionHead`, PPO, GAE or critic architecture change; **no new
`MetaAction`**; no change to terminal-on-last credit placement, to `graph_reward`'s
`static_t0_v1` formula, or to the no-communication boundary; no peer behaviour change and no
communication channel of any kind; `DETECTION_KM`, the B2 geometry and the fixed-cell seed
formulas are unchanged. `evaluate_benchmark`, `evaluate`, the manifest SCHEMA, the
`episode_design` selector, the cardinality sampler and the per-episode persistence are all
exactly as Task 4 left them. **NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no attempt
ordinal, quota, budget, candidate outcome, rejection slug, preflight status or report field
enters `GraphObservation` or `CentralGraphObservation`. **THIS TASK ITSELF SELECTED NO
worlds-per-cell SCALE and COMMITTED NO benchmark POPULATION — it delivered the SELECTION
MECHANISM, and the scale remains a REQUIRED operator input with no default** — and **no
repository preset selects `generalized_v1` and no benchmark manifest is committed or tracked
in the repository.** Current scale, manifest and measurement state is in the [handoff](../../graph_rl_project_handoff.md).

## 7. Early stopping

**GENERALIZED-V1 OPT-IN TRAINING-REWARD EARLY STOPPING — `training_reward_plateau_v1` —
`rl/training/graph_train.py` (`bdfd80d`, integrated `0b9a1d6`, PR #48).**

ONE OPT-IN stopping policy over TRAINING reward alone. It adds NO episode mechanism: the
bounded-backoff geometry, the FD certification physics, the post-FD boundary semantics, the
continuation-reference arithmetic, the Task-4 selector / sampler / manifest / persistence and
the Task-5 quota, budget and preflight are exactly the contracts above. What it changes is
WHEN a run may stop consuming its budget, and nothing else.

| knob | DEFAULT (historical, preserved) | GENERALIZED-V1 addition |
|---|---|---|
| `TrainConfig.early_stopping` | `False` — fixed budget | `True` — `training_reward_plateau_v1` |

`EARLY_STOPPING_POLICIES` is the closed set `(EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU,)`
= (`training_reward_plateau_v1`,). **`TrainConfig.early_stopping_enabled` IS THE ONE
PREDICATE BEHIND EVERY BRANCH**, so "is this run early-stopped?" has a single answer and
cannot be re-derived differently at the loop, the header and the summary. It reads the opt-in
flag and **NOTHING ELSE — notably NOT `training_mode`**.

**OFF IS THE DEFAULT AND IS THE PRESERVED FIXED-BUDGET PATH.** With `early_stopping` false no
monitor is constructed, no check is computed, **no key is added to any training record**
(ABSENT, not null — the discipline the CTDE critic diagnostics already follow), and the loop
cannot exit early. **The fixed-cell measurements (`737b4bf`, `bf1e045f`) were taken on
fixed-budget runs, and this policy is REFUSED on that path**: `validate()` raises unless
`episode_design` is `generalized_v1`, because a fixed-cell run that ended early would no
longer be the fixed-budget contract those measurements were taken on, while its records would
carry the same schedule fields and read as though it were.

**THE STOPPING SIGNAL IS `train_reward_mean` AND NOTHING ELSE**
(`EARLY_STOPPING_METRIC`), and that exclusion is the whole research-validity content of the
feature. The rule MUST NOT consume, and structurally CANNOT consume:

- benchmark or held-out EVALUATION reward of any kind;
- success, feasibility, attrition or completion rates;
- PPO diagnostics (`entropy`, `approx_kl`, `clip_fraction`, `grad_norm`, `policy_loss`, …);
- CTDE critic / value diagnostics (`value_loss`, `value_mean`, `value_target_mean`,
  `critic_grad_norm`);
- checkpoint state;
- any final-comparator result, manifest field, stratum label or reference quantity.

**THE SEPARATION IS MECHANICAL, NOT A CONVENTION.** The frozen benchmark is the COMPARATOR
two arms are judged by, so letting it decide when an arm stops training would let each arm
pick its own stopping point on the very population the comparison is made over, and the
measured difference would stop being attributable to the training algorithm.
`_EarlyStoppingMonitor` is PURE: `observe` takes exactly two KEYWORD-ONLY arguments —
`completed_iterations` and `train_reward_mean` — and the class holds no reference to the
policy, the critic, the buffer, the updater, an evaluation record, the benchmark manifest or
the config, so there is no channel through which a forbidden quantity could reach the
decision even by accident. **ONE METRIC PATH:** the value handed to the monitor is read OFF
the iteration's own completed training record, so the decision and the artifact cannot
describe different numbers.

**ACTOR-ONLY AND CTDE SHARE THIS MECHANISM WITH NO MODE-SPECIFIC BRANCH.** `training_mode` is
read nowhere in it. Two arms compared under this policy therefore share

```text
the same maximum budget + the same frozen stopping rule
                        + the same training-population contract
```

which is deliberately **NOT** "the same actual number of completed iterations": the actual
count is an OUTCOME of the rule, and forcing the two to match would defeat the rule's
purpose. Fed one shared plateau trajectory the two modes produce byte-identical check
histories and stop at the identical completed-iteration count.

**THE APPROVED STATE MACHINE, IN COMPLETED-ITERATION COUNTS — NEVER ZERO-BASED INDICES.**
The approved defaults are `min_iterations = 100`, `window_iterations = 25`,
`patience_windows = 3`, `min_delta = 0.01`.

- checks fall at `min_iterations` and every `window_iterations` afterwards — **100, 125, 150,
  175, …** at the defaults;
- each check averages `train_reward_mean` over the most recent `window_iterations` completed
  iterations, so **consecutive monitored windows do NOT overlap** and the first
  `min_iterations - window_iterations` completed iterations (75 at the defaults) fall outside
  every window;
- the **FIRST check is the BASELINE** (`EARLY_STOPPING_CHECK_BASELINE`): it only establishes
  the best window mean and cannot stop. `best_window_mean_before` stays `null` there — never
  a fabricated zero — so a reader is never left comparing against a measured-looking `0.0`;
- a later window (`EARLY_STOPPING_CHECK_COMPARISON`) is a MEANINGFUL IMPROVEMENT **iff
  `window_mean >= best_window_mean + min_delta`** — the boundary is INCLUSIVE, so improving
  by exactly `min_delta` counts. It then becomes the new best and RESETS the stale counter;
- otherwise the stale counter INCREMENTS;
- the run stops when **`stale_windows >= patience_windows`**.

**THEREFORE, AT THE APPROVED DEFAULTS AND AT THE INTENDED 8 SUCCESSFUL EPISODES PER
ITERATION:** monitoring begins after **800 successful episodes** (100 completed iterations),
and the **EARLIEST POSSIBLE STOP is 175 COMPLETED ITERATIONS = 1400 SUCCESSFUL EPISODES**
(`TrainConfig.early_stopping_earliest_stop_iterations` = `min_iterations + patience_windows *
window_iterations`). **175 IS THE EARLIEST POSSIBLE STOP, NOT A PROMISED OR EXPECTED STOPPING
POINT**, and **the 1400 figure is the CAMPAIGN INTERPRETATION at 8 successful episodes per
iteration ONLY — it must not be generalized to an arbitrary `episodes_per_iteration`.**

**FIRING THE RULE MEANS EXACTLY ONE THING: "the configured training-reward plateau rule
fired."** It is **NOT** proof of convergence, **NOT** a claim of global optimality, **NOT**
evidence that training reward has provably converged, and **NOT** a performance claim of any
kind. Nothing about it may be reported as a convergence result.

**A MISSING `train_reward_mean` INSIDE A MONITORED WINDOW ABORTS THE RUN.**
`EarlyStoppingIntegrityError` is a SUBCLASS of `MeasurementIntegrityError`, so every existing
abort re-raise already routes it correctly and no future handler can account it as episode
attrition. `train_reward_mean` is `None` only when EVERY attempt of an iteration failed, and
under `successful_quota_with_deterministic_replacement_v1` — the only attempt policy this
feature is approved beside — that cannot happen: an iteration either fills its quota of
SUCCESSFUL episodes or raises `TrainingQuotaError` first. So a `None` inside a monitored
window means the instrument contradicts its own attempt contract. **Both alternatives are
refused deliberately:** averaging the window's remaining values would fabricate a window mean
over a population nobody chose, and reading the missing value as `0.0` would insert the
ORACLE OPTIMUM (the reward is normalized regret) into a plateau test and could stop a run by
declaring a total data loss the best window it ever saw. A `None` OUTSIDE every monitored
window is never consumed by the rule and is therefore not judged — what is refused is
fabricating a window mean, never an iteration the mechanism never reads. A PARTIAL window
raises for the same reason (unreachable while `validate` requires
`min_iterations >= window_iterations`, and kept so one can never be averaged as though full).

**THE ORDERING INSIDE `train` IS THE CONTRACT, AND IT IS LOAD-BEARING.** Per iteration:

```text
  training iteration / PPO or CTDE update
    → CONSTRUCT the iteration's training record
    → COMPUTE the early-stopping check FROM THAT RECORD'S `train_reward_mean`
    → ATTACH the due check to that same record (`early_stopping_check`)
    → PERSIST and FLUSH the training record
    → if the rule fired: EXIT the loop BEFORE that boundary's periodic
      evaluation and periodic checkpoint
    → the FINAL evaluation, when enabled, is strictly POST-DECISION
    → the FINAL SAVE-only checkpoint uses the ACTUAL final iteration
```

- **THE COMPARATOR NEVER PARTICIPATES IN DECIDING WHEN TRAINING STOPS.** Exiting before the
  boundary's periodic evaluation is what makes that true even when the stopping boundary is
  ALSO an evaluation boundary.
- **FINALIZATION HAPPENS ONCE.** A stopping boundary that is also an evaluation and
  checkpoint boundary produces exactly ONE final evaluation and ONE final checkpoint — never
  a periodic pair plus a duplicate final pair.
- **`final_iteration` IS THE ACTUAL LAST COMPLETED ITERATION, NOT `n_iterations - 1`.** On a
  full-budget run the two are the same number and finalization is byte-unchanged; on an
  early-stopped run `n_iterations - 1` would name an iteration that never ran, so the final
  evaluation and the final checkpoint would both be labelled with a point the policy never
  reached.
- **A FULL-BUDGET RUN FINALIZES EXACTLY AS BEFORE**, in both shapes — a last iteration that
  is also a periodic boundary, and one that is not.

**PLANNED VERSUS ACTUAL BUDGET — EARLY STOPPING CHANGES ACTUAL CONSUMPTION ONLY.**
`n_iterations` still declares the run's MAXIMUM budget, and
**`TrainConfig.max_training_attempts` (`n_iterations * max_attempts_per_iteration`) and every
held-out / seed-band claim made against it are UNCHANGED — they NEVER shrink dynamically
because a run stopped early.** All three consumers keep the PLANNED bound: `validate()`'s
train-vs-eval overlap test and its scenario-tag namespace bound;
`_require_benchmark_seeds_held_out` through `manifest_seed_overlap`; and `seed_bands`'
`train_band`. Shrinking the band to what a run happened to spend would leave a corridor of
seeds a longer run of the same config really trains on while its benchmark was certified held
out — a held-out failure that produces entirely normal-looking numbers. **SCIENTIFIC
COMPARISON SEMANTICS UNDER THIS POLICY ARE `same maximum budget + same frozen stopping rule +
same training-population contract`, and NOT `same actual number of completed iterations`.**

**CHECKPOINTS STAY SAVE-ONLY.** `save_checkpoint`'s payload contract is UNCHANGED — the
actor-only payload still holds exactly its five keys and the CTDE payload its documented
additions — and **NO loader and NO resume semantics were introduced.** What changed is only
the ITERATION the final checkpoint is written at. **Restoring or continuing a run remains
DEFERRED and out of scope**: an early-stopped run must not quietly acquire continuation
semantics through the checkpoint it now writes at a different iteration.

**OBSERVABILITY — THE EXISTING ARTIFACTS CARRY IT, AND NO NEW EARLY-STOPPING FILE EXISTS.**

- **`run_config.json`** carries the five resolved fields (`early_stopping`,
  `early_stopping_min_iterations`, `early_stopping_window_iterations`,
  `early_stopping_patience_windows`, `early_stopping_min_delta`) through the existing
  `asdict(cfg)` path under `/train_config`. There is deliberately no separate block.
- **`train_records.jsonl:/early_stopping_check`** carries every DUE check, attached to the
  iteration it was computed from — `policy`, `metric`, `check_kind`, both
  `completed_iterations` and `iteration` forms, the window bounds, `window_mean`,
  `best_window_mean_before` / `_after`, `min_delta`, `improvement`,
  `meaningful_improvement`, `stale_windows_before` / `stale_windows`, `patience_windows` and
  `stop_triggered` — so **every decision is reconstructable from `train_records.jsonl`
  alone.** The key is added ONLY on an iteration that really took a check, and never at all
  on a run with the feature off.
- **`run_summary.json:/early_stopping`** (`_early_stopping_summary`) is DERIVED from those
  durable records — ONE metric path, exactly as `severity_response` already is — so the
  summary cannot describe a decision the artifacts do not contain. It states the configured
  shape, `triggered`, a `termination_reason` from the closed set `TERMINATION_REASONS`
  (`training_reward_plateau` / `maximum_budget_reached` / `disabled_fixed_budget`), the
  verbatim check history with `checks_source`, and **every planned/actual budget pair side by
  side**: `planned_iterations` vs `completed_iterations`, `planned_successful_episodes` vs
  `actual_successful_episodes`, and `planned_max_training_attempts` vs
  `actual_training_attempts` — so "this run is short" never has to be inferred by comparing
  two numbers from different files. `stop_completed_iterations` / `stop_iteration_index` are
  `null` — never `0` — when nothing fired, because `0` is a real completed-iteration count
  and a real iteration index. **The block is present on EVERY run**: a fixed-budget one
  reports `disabled_fixed_budget`, which STATES the contract rather than leaving it to be
  inferred from an absent key.

**CONFIGURATION SURFACE.** The five fields are settable from a JSON preset and from the CLI
(`--early-stopping`, `--early-stopping-min-iterations`, `--early-stopping-window-iterations`,
`--early-stopping-patience-windows`, `--early-stopping-min-delta`), through the ONE
`_CLI_FIELD_BY_DEST` mapping, with every flag default read off the dataclass. `validate()`
checks the block **ONLY when the feature is enabled** — an unused block may hold any value,
exactly as the unused `ctde` block may — and refuses, before any compute: a
non-`generalized_v1` design; a non-`int` (a `bool` included) or `< 1` count;
`min_iterations < window_iterations` (the first check must average a FULL window); a
non-numeric or negative `min_delta`; and an `n_iterations` shorter than
`early_stopping_earliest_stop_iterations`, because a run too short to ever reach a stopping
decision would record an ACTIVE stopping policy whose mechanism was structurally inert.
**No repository preset enables this policy** — `configs/graph_train/final_cell_probe.json`
remains the ONLY repository preset and is still `fixed_cell_v1`.

**WHAT IS EXPLICITLY NOT IN THIS TASK.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate future
Grade-A research task**. The frozen solver / BONMIN and the vendored BLADE engine are
untouched. No actor, encoder, `ActionHead`, PPO, GAE or critic architecture change; **no new
`MetaAction`**; no change to terminal-on-last credit placement, to `graph_reward`'s
`static_t0_v1` formula, or to the no-communication boundary; no scenario, world-construction,
reward, solver, fuel-damage, seed-formula, episode-design, cardinality-sampler, manifest,
preflight or evaluation-schedule change; no peer behaviour change and no communication channel
of any kind. **`evaluate`, `evaluate_benchmark` and `save_checkpoint`'s payload are
untouched.** **NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no policy id, metric name,
window mean, stale count, check record or termination reason enters `GraphObservation` or
`CentralGraphObservation`. *(Dated note, 2026-09-14: no recorded measurement has used this mechanism, and no result may
be pre-claimed for it; the GENERALIZED-V1 R1 run used a fixed budget with no early stopping —
see the [measurement history](../history/measurements.md#2-measurement-records).)*

## 8. GENERALIZED-V2 population

**GENERALIZED-V2 — THE TWO-STAGE ROUTE-RELATIVE POPULATION —
`rl/training/graph_generalized.py` + `rl/training/graph_episode_setup.py` +
`rl/training/graph_hidden_placement.py` + `rl/training/graph_train.py` +
`rl/training/graph_rollout.py` + `rl/training/graph_benchmark_preflight.py`
(`a27a3b1`, integrated `f98b293`, PR #57).**

A THIRD EPISODE-DESIGN BUNDLE BESIDE `fixed_cell_v1` AND `generalized_v1`, and it is a
POPULATION contract and nothing else. It adds NO episode mechanism: the bounded-backoff
placement geometry, the certified FD eligibility physics, the post-FD completion-boundary
semantics, the event-conditioned continuation-reference arithmetic, the Task-4 selector /
persistence, the Task-5 quota / budget / preflight, the opt-in early-stopping rule, the
per-wake FD diagnostics and the MATCH-AOU backend seam are exactly the contracts above.
**NO OBSERVATION, ACTION, MASK, REWARD, PPO, GAE, CTDE, TRIGGER, EXECUTOR, SOLVER OR BLADE
SEMANTICS WERE CHANGED BY PR #57.** What changes is WHICH WORLDS a training episode is
drawn from, and WHEN the hidden load is decided.

**THE SELECTOR NOW HOLDS EXACTLY THREE DESIGNS, AND `generalized_v2` RESOLVES THE
IDENTICAL FOUR LOW-LEVEL POLICY IDS AS `generalized_v1`.** `GENERALIZED_V2 =
EpisodeDesign(design="generalized_v2", hidden_policy=HIDDEN_POLICY_BOUNDED_BACKOFF_V1,
eligibility_policy=FD_ELIGIBILITY_CERTIFIED_V1,
post_fd_wake_policy=POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
reference_policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1)` — field for field the
`GENERALIZED_V1` record, **and that identity is the point**: V2 reuses the reviewed episode
MECHANISMS exactly as they were locked and varies only the population. So a V2 episode runs
the certified both-severity FD eligibility walk, the completion-boundary post-FD wake, the
event-conditioned continuation reference and the bounded-backoff hidden placement, all
unchanged. Target destruction stays DETERMINISTIC at `probability = 1`
(`TARGET_DESTRUCTION_PROBABILITY = 1.0`); **`p(destroy) < 1` was NOT implemented here and
remains a separate future Grade-A research task.** The three predicates are the ones locked
above: `generalized` (V1 **or** V2), `generalized_v1_design` (EXACTLY V1) and
`route_relative_population` (EXACTLY V2).

**THE POPULATION IS RESOLVED IN TWO STAGES, ON TWO DISJOINT SEED DOMAINS, AND THE ORDER IS
THE CONTRACT.**

```text
  STAGE 1  PreSolveCardinality        A ~ U{2,3,4,5,6};  K | A ~ U{A, A+2}
    |      policy generalized_v2_pre_solve_uniform_v1
    |      domain generalized_v2_cardinality_v1
    |      drives the GENERATOR: the known-only world
    v
  [ the known-only MATCH-AOU solve, on p1_milp_v1, inside setup_episode ]
    |
    v
  STAGE 2  RouteRelativeHiddenLoad    R = |routed egos|;  H_requested ~ U{1, ..., R}
    |      policy route_relative_uniform_v2
    |      domain generalized_v2_hidden_load_v1
    |      drives the LOCKED bounded-backoff placement
    v
  RESOLVED EpisodeCardinality         (A, K, H), source generalized_v2_route_relative
```

- **STAGE 1 — `sample_generalized_v2_pre_solve_cardinality(*, episode_seed)`.** Exactly two
  index draws, in this order and no other: `A ~ Uniform(GENERALIZED_V2_AGENT_COUNTS)` =
  `{2, 3, 4, 5, 6}`, then `K = A + Uniform(GENERALIZED_V2_KNOWN_OFFSETS)` with
  `GENERALIZED_V2_KNOWN_OFFSETS = (0, 2)`, drawn CONDITIONAL on `A` — so `K ∈ {A, A + 2}`.
  Both are explicit `randrange` calls on a `random.Random` constructed from
  `derive_v2_cardinality_seed(episode_seed)`, so the number and order of draws is a stated
  contract a test can pin. **NO HIDDEN COUNT IS DRAWN HERE, AND NONE CAN BE** —
  `PreSolveCardinality` deliberately carries no hidden field, because at that point in the
  episode there is no honest number to put there and a placeholder would be read by every
  downstream consumer as a request. The constants are MIRRORED on
  `graph_episode_setup.GENERALIZED_V2_AGENT_COUNTS` / `GENERALIZED_V2_KNOWN_OFFSETS` so the
  sampler cannot draw a shape construction would refuse, and the mirror is test-enforced.
- **STAGE 2 — `resolve_route_relative_hidden_load(*, episode_seed, route_count)`.** `R` is
  the number of egos the KNOWN-ONLY allocation actually gave a route to, counted by
  `graph_hidden_placement.routed_ordinals(solution, agent_ordinals)` — which is defined
  BESIDE the bounded walk and shares its `_has_route` predicate, so the request can never
  exceed what the walk is able to attempt. `H_requested = 1 + randrange(R)`, one index draw
  on a `random.Random` built from `derive_hidden_load_seed(episode_seed)`.
  `RouteRelativeHiddenLoad` records `route_count` BESIDE `hidden_requested`, because a
  request of `1` means something entirely different at `R = 1` (the only possible request)
  than at `R = 6`, and a distribution of requests cannot be read without the supports they
  were drawn from. **`R >= 1` IS REQUIRED AND `R == 0` RAISES** — in the sampler's own
  argument check and in the construction path, which refuses a world whose known-only
  allocation routed nobody with a loud `RuntimeError` rather than repairing it into a
  zero-hidden world (a different population wearing this design's label). The dataclass also
  refuses any `H` outside `1 <= H <= R`.
- **`resolved_v2_cardinality(pre_solve, hidden_load)` BUILDS A THIRD, NEW OBJECT.** An
  ordinary `EpisodeCardinality` carrying `source = CARDINALITY_SOURCE_V2_ROUTE_RELATIVE`
  (`"generalized_v2_route_relative"`), so every downstream consumer that already knows how
  to read a requested cell — the scheduled-cell check, the construction-audit
  reconciliation, the per-episode record — reads a V2 episode with no special case.
  **NEITHER STAGE RECORD IS MUTATED**: all three are frozen, both stages remain readable
  beside the result, and a stage's record is written once and never revised.
  `CARDINALITY_SOURCE_V2_PRE_SOLVE` (`"generalized_v2_pre_solve"`) is the STAGE-1 label and
  can never label a resolved cell.

**DO NOT DESCRIBE `H` AS KNOWN BEFORE THE ALLOCATION.** Faking an up-front
`EpisodeCardinality` with a placeholder hidden count and correcting it afterwards would put
a number in a record that was never the request — precisely the class of defect the "the
REQUEST is never rewritten" rule exists to prevent. `episode_cardinality(cfg, seed, ...)`
therefore has **NO FOURTH SOURCE FOR V2 AND RAISES** on that path; the two-stage answer
comes from `v2_pre_solve_cardinality(cfg, seed)` before the solve and
`resolved_v2_cardinality` after it.

**WHY `H <= R` RATHER THAN `H <= A`, AND WHAT IT IS NOT.** The reviewed
`bounded_backoff_v1` policy realizes AT MOST ONE hidden target per ego route, so a request
above `R` is unsatisfiable BY CONSTRUCTION and its shortfall would say nothing about the
world — it would only re-measure the allocation. Bounding the request by `R` makes a
recorded shortfall mean what it is supposed to mean: the LOCKED geometry refused a route it
was offered. **IT IS DELIBERATELY NOT A PROMISE THAT `H_realized == H_requested`**: geometry
stays authoritative, a genuine geometric refusal is still RECORDED and never repaired, and
`R` is an upper bound on what a walk may ATTEMPT rather than a prediction of what it
realizes. **NOTHING ABOUT THE PLACEMENT GEOMETRY CHANGED** — the leg rule, the guaranteed
portion, the offset budget, the nearest-neighbour margin, the sensing guard, the independent
re-measurement, the ordinal-driven permutation and the per-candidate substreams are the
LOCKED B2 / bounded-backoff contract, reused and not reimplemented, and `routed_ordinals` is
PURE, READ-ONLY, consumes no randomness and predicts no geometry. **MULTIPLE HIDDEN TARGETS
ON ONE ROUTE REMAIN OUT OF SCOPE, UNIMPLEMENTED AND UNAPPROVED**, and must not be documented
as either. V2 avoids the known route-count cardinality mismatch by drawing `H` against the
REALIZED `R` — **not** by restoring the legacy MATCH-AOU redundant-stacking incentive, which
is the very thing `p1_milp_v1` removes.

**THE BACKEND IS DESIGN-CONSTRAINED, AND THE REFUSAL IS PRE-EXECUTION.**
`GENERALIZED_V2_REQUIRED_BACKEND = MATCH_AOU_BACKEND_P1_MILP_V1`. Backend selection stays
EXPLICIT with no `auto`, no fallback in either direction and no silent inference from
`episode_design`; `fixed_cell_v1` and `generalized_v1` each accept EITHER approved
objective; and **`generalized_v2` + `legacy_minlp_v1` is REFUSED by both
`TrainConfig.validate` and `RolloutConfig.validate` BEFORE anything executes**, never
overridden and never preferred on the run's behalf. The reason is the route count itself:
V2's hidden load is defined against the egos the known-only allocation routed, and the
legacy objective's EPSILON stacking incentive changes which allocations are optimal and
therefore which egos are routed — letting two objectives define that quantity would make one
design id mean two different population selectors.

**SETUP-SIDE SELECTION — `hidden_load_policy` IS A DIFFERENT QUESTION FROM `hidden_policy`,
AND THE HISTORICAL DEFAULT IS PRESERVED.** `HIDDEN_LOAD_POLICIES =
(HIDDEN_LOAD_POLICY_EXPLICIT_V1, HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2)` =
(`explicit_request_v1`, `route_relative_uniform_v2`), `DEFAULT_HIDDEN_LOAD_POLICY` is the
explicit one, and `resolve_hidden_load_policy` is the ONE validation site — an unknown id
RAISES rather than falling back, for the same reason an unknown design does. **Every pre-V2
construction call resolves `explicit_request_v1`, so `fixed_cell_v1` and `generalized_v1`
are unchanged by V2's existence**: their hidden load is still a number their caller computed
before `setup_episode` was entered. `setup_episode(..., hidden_load_policy=...,
hidden_load_seed=..., known_requested=..., population_recorder=...)` takes the
route-relative request as a COMPLETE set through `_resolve_route_relative_request`, which
REFUSES rather than guessing: the policy requires `hidden_policy == bounded_backoff_v1` (the
one-target-per-route rule is what makes `H <= R` the right bound); `n_hidden` must be ABSENT
(stating a count AND asking for it to be drawn are two contradictory requests);
`placement_rng` must be an explicit `random.Random`; `hidden_load_seed` must be an explicit
non-negative integer (`bool` rejected despite subclassing `int`); and `known_requested` must
be an explicit positive integer, **because V2's `K` is a CHOICE between `A` and `A + 2` and
construction cannot infer which one the schedule drew** — so it is told, and
`_require_generalized_v2_cardinality` refuses a world that disagrees rather than adopting
whatever it received. That check judges `A ∈ GENERALIZED_V2_AGENT_COUNTS` and `K ∈ {A + o}`
against the RAW pre-solve world inventory, BEFORE the solve, so an out-of-cell request costs
no solver call. Selecting a V2-only argument under the historical policy is likewise
REFUSED, never ignored.

**WHAT THE EPISODE CARRIES OUT, AND HOW A READER TELLS THE DESIGNS APART.**
`EpisodeContext` gains `hidden_load_policy` (defaulting to `explicit_request_v1`) and
`route_relative_load: Optional[RouteRelativeHiddenLoad]`; `_finish_context` takes both as
REQUIRED keywords and VERIFIES the pairing — a route-relative policy with no load, or a load
under the historical policy, RAISES. `split_meta` grows the V2-only keys
`hidden_load_policy`, `route_count_at_hidden_resolution` and `hidden_load` **only** when a
load was resolved, under the same discipline the generalized keys already follow: **a V1 or
fixed-cell record grows no new field, so the ABSENCE of these is how a reader tells that the
hidden count was STATED by the caller rather than RESOLVED against a route count.**
`ConstructionAudit.known_requested` is the SCHEDULE's `known_requested` under V2 (where `K`
is a choice) and stays `len(agents)` under V1 (where `K == A` is the rule); the realized
counts still come from the RAW pre-solve world snapshots and are VERIFIED, never trusted.

**THE V1 18-STRATUM BENCHMARK IS NEVER READ UNDER GENERALIZED-V2, AND NOTHING SUBSTITUTES
ANOTHER POPULATION.** Its strata are built from `A ∈ {2,3,4}` with a LOW/HIGH hidden load
defined against `A`, while V2 draws `A` from a wider set and defines its hidden load against
a realized route count — so a V1 manifest evaluated under V2 would report strata the
population never varied. `evaluate()` — the fixed held-out seed-band evaluator — still RAISES
under `route_relative_population`, because that band carries no stratum; the V1 loader and
the V2 loader each refuse the other's schema; and the V1 preflight path's
`_require_preflight_config` still refuses anything that is not EXACTLY `generalized_v1`.
**SINCE PR #59, GENERALIZED-V2 HAS ITS OWN SEPARATE EVALUATION CONSTRUCT** — the frozen
ten-cell benchmark contracted in the GENERALIZED-V2 BENCHMARK block at the end of this
section — so an evaluating V2 run is no longer refused: it must name a frozen V2 manifest and
one declared profile.

**THE APPROVED EARLY-STOPPING POLICY REMAINS `generalized_v1`-ONLY.** `validate()` refuses
`early_stopping` unless `design.generalized_v1_design`, so a `generalized_v2` run is REFUSED
as firmly as a fixed-cell one: the plateau contract was reviewed against the V1 population
and its training-reward trajectory, and extending it is a research decision nobody has
taken. **Do not generalize the stopping rule in documentation.**

**THE TRAINING ATTEMPT QUOTA APPLIES TO BOTH GENERALIZED DESIGNS.** Under `generalized_v2`
as under `generalized_v1`, `episodes_per_iteration` is a SUCCESSFUL-episode quota,
`generalized_max_attempts_per_iteration` is REQUIRED and never defaulted, an ordinary failed
attempt SPENDS its seed and its run-wide ordinal and is REPLACED by the next deterministic
attempt, and the budget bounds the run's MAXIMUM POSSIBLE training-attempt seed band.
`fuel_damage_mode = seeded_variable` is REQUIRED under both. The BENCHMARK reading of that
band applies under both too: an evaluating run's frozen manifest — V1 or, since PR #59, the
whole V2 ten-cell manifest — is verified held out from it.

**FAILURE PROVENANCE IS STAGE-AWARE, WRITE-ONCE AND NEVER RE-DERIVED.** A failed attempt is
still part of the ATTEMPTED population, so its ledger entry must state the identity it
ACTUALLY RECEIVED — and a V2 attempt can die in either stage.

- **`graph_episode_setup.RouteRelativePopulationRecorder` is the CALLER-OWNED, WRITE-ONCE
  carrier.** The caller constructs a FRESH one per attempt, hands it in, and reads it back
  whether `setup_episode` RETURNED or RAISED. `setup_episode` writes the frozen
  `RouteRelativeHiddenLoad` into it **the INSTANT the draw exists and BEFORE anything that
  can fail consumes it** — the bounded-backoff walk, the patch, the env-2 reload, the
  world-cardinality checks, the deferred reference solve — so from that line on an attempt
  that fails has still RECEIVED this population identity. A SECOND write RAISES (one attempt
  resolves exactly one stage-2 record), a non-empty recorder is REFUSED at entry (it would
  attribute another attempt's population to this one), and it fabricates nothing: an attempt
  that failed BEFORE stage 2 leaves `load` at `None`. **It is PURE PROVENANCE — setup writes
  it and reads nothing back, so the episode behaves identically whether one was supplied or
  not**, and a caller with no ledger (the diagnostic rollout) passes none. It is a
  caller-owned object rather than an attribute set on the exception, because wrapping would
  change the recorded `error_type` and therefore the failure's classification.
- **`graph_train._v2_failure_population(pre_solve, route_relative_load)`** emits the
  `generalized_v2_population` block of a FAILED attempt, KEYED OFF the stage-1 record so it
  appears **ONLY** on the V2 path. It states `stage_resolved` outright — `"pre_solve"` or
  `"route_relative"` — so a reader never infers the stage from which fields are `null`.
  Before stage 2 resolves it carries the pre-solve `A` / `K` identity with its policy, rng
  domain and derived seed, and every hidden-load field (`hidden_load_policy`,
  `hidden_load_rng_domain`, `hidden_load_derived_seed`, `route_count_at_hidden_resolution`,
  `hidden_load`) is `null`: **NOT a writer that forgot them, and NOT a record to be
  reconstructed later from the seed.** After stage 2 resolves it carries the EXACT draw
  VERBATIM from the frozen component record. **IT IS DELIBERATELY NEVER RE-DERIVED:** `H` is
  reproducible from the seed and `R`, so a post-hoc redraw would usually agree — and
  "usually" is exactly the property a ledger must not rest on. The ledger says what the
  attempt resolved, not what a replay would.
- **`graph_train._failure_cardinality(card, pre_solve, route_relative_load)`** answers which
  population identity a failed attempt received: a non-V2 attempt reports the cell its
  schedule resolved up front; a V2 attempt that died BEFORE stage 2 reports its stage-1
  half-cell; a V2 attempt that died AFTER stage 2 reports the RESOLVED cell, ASSEMBLED from
  the two recorded facts through `resolved_v2_cardinality` — an assembly, never a redraw.
- **HISTORICAL RECORDS ARE UNCHANGED.** A `fixed_cell_v1` or `generalized_v1` failure record
  grows NO key at all and keeps the shape every existing reader and preserved artifact
  already has, and the successful-outcome stream stays at episode-outcome schema
  **version 3** — the V2 `generalized_v2_population` block is added conditionally, not by a
  schema bump.

**KEYWORD OMISSION PRESERVES EVERY HISTORICAL CALL.** `_pre_solve_kwargs`,
`_v2_hidden_load_kwargs` and `_population_recorder_kwargs` return `{}` on every non-V2 run,
exactly as `_artifact_kwargs` / `_ctde_kwargs` / `_cardinality_kwargs` /
`_generalized_setup_kwargs` / `_backend_setup_kwargs` already do — so a `fixed_cell_v1` or
`generalized_v1` run calls `_run_one_episode` and `setup_episode` with EXACTLY the arguments
it did before this design existed, which is the stronger invariance claim.

**THE DIAGNOSTIC ROLLOUT HAS SELECTOR PARITY AND STAYS DIAGNOSTIC.** `RolloutConfig` mirrors
`episode_design`, exposes the same three ids, resolves the same four policy ids from the
same site, applies the SAME P1-backend refusal, samples the same two-stage training
population per seed and records the resolved cell — and it passes NO population recorder,
because it keeps no failure ledger. It builds no benchmark, evaluates no matched group and
makes no benchmark claim — and that is UNCHANGED by PR #59: the V2 matched benchmark
evaluator is `graph_train.evaluate_benchmark` → `_evaluate_v2_benchmark`, and the rollout
remains a diagnostic of the TRAINING population only.

**SUPPORTED GENERALIZED-V2 TRAINING CARDINALITY CURRENTLY STOPS AT `A <= 6`.**
`GENERALIZED_V2_AGENT_COUNTS = (2, 3, 4, 5, 6)` is the approved and enforced support, on
BOTH the sampler side and the construction side. **`A = 8` and `A = 10` are ENGINEERING
SCALING EVIDENCE ONLY** — they are not in the approved support, are not selectable through
this design, and **must not be described or implied as supported training cells.**

**WHAT IS EXPLICITLY NOT IN THIS DESIGN.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate
future Grade-A research task**. The frozen solver / BONMIN, the frozen
`match_aou_MINLP_solver.py` and the vendored BLADE engine are untouched. No actor, encoder,
`ActionHead`, PPO, GAE or critic architecture change; **no new `MetaAction`**; no change to
the action surface, the mask, terminal-on-last credit placement, `graph_reward`'s
`static_t0_v1` formula, the fuel-damage mechanism, the certified-FD physics, the
continuation-reference arithmetic, the B2 geometry, `DETECTION_KM`, the fixed-cell seed
formulas, the manifest SCHEMA or the no-communication boundary; no peer behaviour change and
no communication channel of any kind; no multi-hidden-per-route placement; and **no
GENERALIZED-V2 evaluation construct, benchmark, stratification or worlds-per-cell scale AS
PR #57 SHIPPED IT.** *(PR #57 shipped no evaluation construct; PR #59 added it afterwards — see
[§9](#9-generalized-v2-benchmark-and-evaluation).)*
**NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no design id, stage label, policy id,
rng domain, derived seed, route count, requested or realized cardinality and no recorder
field enters `GraphObservation` or `CentralGraphObservation` — a count of what is hidden,
and the size of the team, are exactly the privileged quantities an ego cannot sense ([`CLAUDE.md` §3](../../CLAUDE.md#3-architecture--the-load-bearing-invariants)).
**No repository preset selects `generalized_v2`**:
`configs/graph_train/final_cell_probe.json` remains the ONLY repository preset, is untouched
and is still `fixed_cell_v1`. **PR #57 PRODUCED NO SCIENTIFIC MEASUREMENT, NO V2 BENCHMARK
AND NO V2 POLICY-PERFORMANCE RESULT**.

## 9. GENERALIZED-V2 benchmark and evaluation

**GENERALIZED-V2 BENCHMARK — THE FROZEN TEN-CELL BENCHMARK AND EVALUATION CONSTRUCT —
`rl/training/graph_generalized.py` + `rl/training/graph_benchmark_preflight.py` +
`rl/training/graph_train.py` + `rl/training/graph_episode_setup.py` +
`rl/training/graph_hidden_placement.py` (`786e821`, integrated `ea8778d`, PR #59).**

A benchmark / evaluation MECHANISM over the UNCHANGED PR #57 population, and nothing else.
**THE POPULATION CONTRACT ABOVE IS NOT ALTERED BY ONE CLAUSE:** `A ∈ {2,3,4,5,6}`;
`D = K − A ∈ {0, 2}`, i.e. `K ∈ {A, A+2}`; the known-only P1 solve first; the REALIZED
routed-ego count `R`; then `H_requested ~ U{1..R}`; `p1_milp_v1` REQUIRED and
`legacy_minlp_v1` REFUSED; and bounded-backoff geometry plus every reused low-level
mechanism unchanged. **PR #57 remains the historical population implementation and did NOT
contain this construct.** `fixed_cell_v1` and `generalized_v1` are unchanged, and the V1
18-stratum manifest keeps its pre-task canonical identity (pinned by
`tests/test_graph_generalized_v2_benchmark.py::test_po1_a_v1_manifest_keeps_its_pre_task_canonical_identity`).
**PR #59 IS CODE: it produced no scientific manifest, selected no real benchmark seeds, ran
no training and took no measurement.**

**1. A SEPARATE SCHEMA, NOT THE V1 18-STRATUM ONE.** `V2_BENCHMARK_SCHEMA =
"generalized_v2_benchmark_manifest"`, `V2_BENCHMARK_SCHEMA_VERSION = 1`, carried by
`V2BenchmarkManifest` / `V2BenchmarkWorld`. `v2_manifest_from_record` refuses the V1 schema
and the V1 loader refuses the V2 one, BEFORE anything else is read;
`load_benchmark_manifest_for_design` is the design-aware reader, `train` calls
`load_v2_benchmark_manifest` under V2, and `evaluate_benchmark` refuses a V2 manifest under
any non-V2 design.

**2. TEN EXOGENOUS BASE CELLS — AND NOTHING ENDOGENOUS IS A STRATUM.**
`V2_BENCHMARK_BASE_CELLS` is BUILT as the product `A ∈ GENERALIZED_V2_AGENT_COUNTS` ×
`D ∈ GENERALIZED_V2_KNOWN_OFFSETS` — **exactly 10 cells**, canonical order `A` then `D` —
so the count cannot drift from the population. `V2_BENCHMARK_STRATIFICATION_FACTORS =
("agent_count", "known_offset")`. **There is NO V2 LOW/HIGH bucket**, and `R`,
`H_requested`, `H/R` and hidden realization (`V2_BENCHMARK_NON_STRATA`) are **REPORTING
DESCRIPTORS ONLY** — never strata, never quotas, never an acceptance balance. They are
outputs of the known-only allocation and of the locked geometry, and selecting on them would
build the comparator out of the solver's own output.

**3. SCALE AND PROFILES, FIXED BY THE CONSTRUCT.** `V2_BENCHMARK_WORLDS_PER_CELL = 12`, and a
manifest whose cell does not hold world ordinals exactly `0..11` is REFUSED
(`_require_well_formed_v2_worlds`). One COMPLETE manifest therefore holds **120 frozen world
groups**, each a matched CLEAN / MILD / SEVERE triad — **360 member episodes if the entire
manifest were evaluated.** `V2_BENCHMARK_PROFILES` = (`development`, `confirmatory`):
`development` is world ordinals `0..1` in every cell (**20 groups / 60 members**) and
`confirmatory` is `2..11` (**100 groups / 300 members**). They are **DISJOINT AND EXHAUSTIVE
over the one frozen manifest** (`_require_v2_profiles_partition`). A profile SELECTS which
frozen groups a run evaluates (`profile_worlds`, `profile_identity_record`); it never changes
the manifest identity. Unlike V1, the worlds-per-cell scale is NOT an operator choice — the V2
preflight REFUSES any `worlds_per_cell != 12` — while `benchmark_base_seed` and
`max_candidates_per_cell` (`>= 12`) remain REQUIRED operator inputs with no default.

**4. THE FROZEN V2 WORLD IDENTITY IS STRONG AND UUID-FREE, AND A MISMATCH ABORTS.**
`V2WorldIdentity` (`_V2_IDENTITY_FIELDS`) carries: `seed`; `agent_count` (`A`);
`known_count` (`K`, with `known_offset = D` derived); `match_aou_backend`; the realized
`route_count` (`R`); `allocation_fingerprint` — the UUID-free, structural known-only
allocation fingerprint (`v2_allocation_fingerprint`, schema
`generalized_v2_allocation_fingerprint_v1`); the ACTUAL `hidden_requested`;
`hidden_realized`; `known_realized`; the hidden `geometric_fingerprint`; the certified FD
`fd_selected_ordinal`; and `fd_certificate_fingerprint`. `V2WorldPreflight` (identity + the
frozen stage-2 hidden-load record + the construction audit) is **REQUIRED** on every V2
world, unlike V1, because `R`, the allocation fingerprint and the actual `H_requested` exist
only once the world has been built. A reconstruction that disagrees on ANY field is a
`BenchmarkIdentityError` (`require_v2_world_matches_manifest` against the frozen world,
`require_v2_matched_group_identity` across the group's completed members) and **ABORTS —
never a runtime substitution, regeneration or repair.**

**5. THE LOADER AUTHENTICATES BOTH THE BYTES AND THE POPULATION.** `v2_manifest_from_record`
runs the SAME four steps as the V1 loader — (1) a non-empty `manifest_id` and THIS schema /
version / design; (2) the STORED payload, exactly as found, hashes to that id; (3) semantic
parse and validation with the canonical world order never re-sorted; (4) the stored payload
EQUALS the canonical payload — and `manifest_id` is, as in V1, the hash of the canonical
PAYLOAD and NOT of the file bytes. Step 3 additionally builds every `V2WorldPreflight`,
whose `__post_init__` refuses any frozen state **production V2 could not actually produce**:
the stored hidden-load record is rebuilt as a real `RouteRelativeHiddenLoad` (so `R >= 1` and
`1 <= H_requested <= R` are that type's own invariants) and must be in canonical form; its
policy must be `HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2`; its rng domain must be
`V2_HIDDEN_LOAD_RNG_DOMAIN`; its derived seed must equal `derive_hidden_load_seed(world
seed)`; the record's `R` and `H_requested` must agree with the identity's;
`1 <= H_realized <= H_requested`; `known_realized == K`; and the hidden geometric
fingerprint must hold exactly `H_realized` placements. A violation is a
`BenchmarkManifestError`, a **self-consistently re-hashed impossible manifest is refused**,
and **nothing is repaired or normalized into validity.**

**6. THE V2 PREFLIGHT IS FAIL-CLOSED.** `run_benchmark_preflight` delegates a V2 config to
`_run_v2_benchmark_preflight` (policy `deterministic_per_cell_window_fail_closed_v2`), which
first requires EXACTLY `generalized_v2` on `p1_milp_v1` (`_require_v2_preflight_config`) and
complete Git provenance. `v2_cell_windows` gives cell `c` its OWN half-open window
`[base + c·M, base + (c+1)·M)`; `_scan_v2_cell` walks it in ascending seed order, attempts
each candidate EXACTLY once and stops at 12 acceptances; `probe_v2_world` builds the world
through the PRODUCTION two-stage route-relative path, FREEZES the ACTUAL
`ctx.route_relative_load` (never a redraw), and requires the three FD member plans'
identities to agree. **THE REPLACEMENT-ELIGIBLE SET IS CLOSED** (`V2_REJECTION_REASONS`,
recognized by TYPE and pipeline STAGE in `v2_rejection_reason`, on the wrapped
`EpisodeAttemptError`):

- `generation` + `TargetPlacementError` — strict generator target-placement refusal
  (`generator_target_placement_refused`);
- `setup` + `RouteRelativeNoRoutesError` — `R == 0` (`route_relative_no_routes`);
- `setup` + `BoundedBackoffExhaustedError` — bounded backoff realized zero targets
  (`bounded_backoff_zero_realized`);
- `setup` + a non-integrity `FuelDamageError` carrying `NO_FD_ELIGIBLE_EGO`.

**EVERYTHING ELSE ABORTS rather than spending another seed** — a generic
`HiddenPlacementError`, an unknown `RuntimeError`, any integrity / configuration / backend /
identity contradiction, and any other unclassified failure. **THE TWO TYPED CLASSIFICATIONS
ARE CLASSIFICATION ONLY, AND THE LOCKED LAYERS DID NOT OTHERWISE MOVE.**
`graph_episode_setup.RouteRelativeNoRoutesError(RuntimeError)` (`reason =
ROUTE_RELATIVE_NO_ROUTES`) is raised ONLY on the route-relative path, at both `R == 0` sites
(an empty known-only allocation, and an allocation that routed none of the scheduled egos),
with the same message text; the explicit-request (fixed-cell / V1) paths keep their plain
`RuntimeError`. `graph_hidden_placement.BoundedBackoffExhaustedError(HiddenPlacementError)`
(`reason = BOUNDED_BACKOFF_ZERO_REALIZED`) is raised ONLY at the ALREADY-EXISTING terminal
zero-realized branch of `place_hidden_targets_bounded`; **geometry, candidate ordering, RNG
draws, validation, backoff acceptance and historical `HiddenPlacementError` compatibility
are UNCHANGED** (it subclasses that type, so every existing catcher is unaffected), and every
other raise in that module stays a plain `HiddenPlacementError`. **The preflight stays
policy- and reward-BLIND**: no policy is built, no episode is run, and acceptance never reads
`R`, `H`, reward or actor behaviour. **A SHORT REALIZATION (`1 <= H_realized <
H_requested`) IS LEGITIMATE, ACCEPTED AND RECORDED, never retried.** A window exhausted before
12 acceptances writes a FAILED report, creates NO manifest and scans NO later cell — the same
complete-manifest rule as V1.

**7. EVALUATION — MATCHED, IDENTITY-VERIFIED, NEVER SUBSTITUTED.** `TrainConfig.validate`
requires an EVALUATING V2 run to name a `benchmark_manifest` AND a `benchmark_profile` in
`V2_BENCHMARK_PROFILES` (`--benchmark-profile`); a profile outside `generalized_v2`, or a
profile without a manifest, is refused. `evaluate()` still RAISES under V2 and points at
`evaluate_benchmark`, which dispatches to `_evaluate_v2_benchmark`: ONE deterministic round
over the declared profile, in which each frozen world is evaluated as a matched CLEAN / MILD /
SEVERE triad on its IDENTICAL frozen seed with disjoint artifact tags. Each member is REBUILT
through the real two-stage construction from `v2_benchmark_pre_solve_cardinality` (so `R`
and `H` are re-produced, never read off the manifest), with a fresh
`RouteRelativePopulationRecorder`, and its UUID-free identity is VERIFIED against the frozen
world BEFORE its reward or diagnostics reach a group or a summary. **A failed member is
recorded once with the population identity it actually received and is NEVER replaced** — no
other world, seed or preflight call takes its place; its group becomes INCOMPLETE, stays
VISIBLE, and contributes NO within-world delta. Identity and instrument failures
(`BenchmarkIdentityError`, `MeasurementIntegrityError`, `FuelDamageIntegrityError`, an
aborting `ReferenceIntegrityError`, `MatchAouBackendError`, `_VisualArtifactError`) ABORT.

**8. HELD-OUTNESS IS OVER THE WHOLE MANIFEST AGAINST THE MAXIMUM ATTEMPT BAND.**
`_require_benchmark_seeds_held_out` checks `manifest.seeds()` — every world seed of BOTH
profiles, whichever one the run evaluates — against `[base_seed, base_seed +
max_training_attempts)`, the run's MAXIMUM POSSIBLE training-attempt band and never the
successful-episode quota, at load time and before any run artifact or compute exists;
`_require_benchmark_tag_namespace` bounds the artifact-tag namespace against the manifest.
**Early stopping remains REFUSED under `generalized_v2`.**

**9. THE PRIMARY V2 BEHAVIOURAL ENDPOINT, AS IMPLEMENTED (`_v2_behaviour_summary`).** The
metric is the **SEVERE − MILD aggregate probability MASS on `SELF_PRESERVATION_ABORT` at the
immediate-FD wake of the certified ego** (`metric = severe_minus_mild_aggregate_abort_mass`,
`wake_kind = immediate_fuel_damage`). It is (1) **PAIRED WITHIN ONE FROZEN WORLD GROUP
FIRST**, then (2) **summarized per `(A, D)` base cell**, then (3) **MACRO-AVERAGED WITH EQUAL
WEIGHT OVER THE TEN BASE CELLS** (`macro_mean_over_base_cells`). A group is
METRIC-ELIGIBLE only when it is COMPLETE and its MILD and SEVERE members each carry EXACTLY
ONE immediate-FD wake with recorded diagnostics; ordinary and post-FD-boundary wakes are
filtered by their tagged kind, never by the selected action, and zero or several
immediate-FD wakes is a stated not-measurable reason. **The macro is `None` — undefined —
unless EVERY base cell has at least one metric-eligible group** (`macro_undefined_base_cells`
names the gaps), because a mean over fewer cells would silently re-weight the design; a pooled
over-groups mean is also reported and is NOT the primary endpoint. The AGGREGATE abort mass
(`aggregate_mass_is_not_selected_action_probability = true`) and the SELECTED joint-cell /
meta-action result are kept SEPARATE: the **directional switch** (`V2_SWITCH_DIRECTIONAL` —
MILD not abort AND SEVERE abort) and the **reverse switch** (`V2_SWITCH_REVERSE` — MILD abort
AND SEVERE not abort) are counted per cell and overall with rates whose denominator is
EXPLICITLY the metric-eligible groups (`rates_over` / `switch_rates_over =
"metric_eligible_groups"`). Undefined quantities are `None`, never `0`, and training rows
cannot reach the summary. A round's eval record carries `v2_behaviour`,
`v2_benchmark_groups` (canonical complete / incomplete / metric-eligible group-key lists with
digests) and the per-member `benchmark_v2` block, and `run_summary.json` carries the final V2
round's behaviour and groups. **REWARD IS DOWNSTREAM AND SECONDARY** (`_V2BenchmarkTally`:
per-cell reward means and within-world reward deltas over COMPLETE groups), never a substitute
for the primary endpoint. **THE CODE COMPUTES DESCRIPTIVE PER-ROUND SUMMARIES ONLY: no
significance test, confidence interval, decision threshold or other inference procedure is
implemented, and no scientific result is established by the construct.**

**WHAT IS EXPLICITLY NOT IN THIS TASK.** No scientific V2 manifest, no chosen benchmark seed
namespace, no preflight invocation, no frozen V2 seed population, no training run and no
evaluation result; **no benchmark manifest is committed or tracked in the repository**, and
**no repository preset selects `generalized_v2`**. No change to the V2 population, the V1
benchmark semantics, early stopping, the diagnostic rollout (still a TRAINING-population
diagnostic, never the matched benchmark evaluator), BLADE, the solvers, PPO, CTDE, the reward,
the observation or the action contracts; `p(destroy)` stays `1.0`; **supported V2 cardinality
stays `A <= 6`** and `A = 8` / `A = 10` remain ENGINEERING-ONLY, outside the V2 scientific
training and evaluation population. **NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no
cell key, profile, identity field, fingerprint, rejection slug or behavioural summary enters
`GraphObservation` or `CentralGraphObservation`.

## 10. Code routing

| Task | Files and symbols | Contract |
|---|---|---|
| run PPO training or plot a run | `rl/training/graph_train.py`: `TrainConfig`, `train`, `collect_provenance`, `_git_provenance`, `_iteration_outcome`, `build_run_summary`, `eval_episode_tag`, `plot_training` | §1; outputs in [artifacts and metrics](artifacts_metrics.md) |
| run a diagnostic rollout, or keep it at configuration parity with training | `rl/training/graph_rollout.py`: `RolloutConfig`, `run_rollout` | §1, §5, §8 |
| change how a run fails on an integrity fault (abort, never attrition) | `graph_train.py`: `MeasurementIntegrityError`, `EpisodeRosterError`, `_world_snapshot_ids`, `_episode_target_roster`, `_require_scheduled_cell`, `_ConditionTally.success(out, *, expected_cell)`, the abort re-raises in `_run_one_episode` and the attempt handlers | §2, §3; FD and backend faults in [construction §4](construction_fuel_damage.md#4-certified-fd-eligibility-live-certificate-check-and-post-fd-boundaries) and [reward and solvers §3](reward_solvers.md#3-match-aou-allocation-backends) |
| configure a run from a file, or record where a configuration came from | `configs/graph_train/final_cell_probe.json`; `graph_train.py`: `load_config_file`, `resolve_train_config`, `_effective_argv`, `_explicit_cli_dests`, `_CLI_FIELD_BY_DEST`, `config_source_record`, `_CONFIG_SOURCE_KINDS`, `write_run_config` | §4 |
| change the fixed training cell (target counts, geometry) | `graph_train.py`: `TrainConfig.num_agents` / `n_known` / `n_hidden` / `n_targets_emitted`, `build_variation_config`; `graph_rollout.py`: `RolloutConfig` | §1, §4 |
| change the FD training mixture, matched pair or triad evaluation, or FD reporting | `graph_train.py`: `TrainConfig.fuel_damage_parameters`, `reward_config`, `evaluate`, `eval_member_tag`, `_ConditionTally`, `_EVAL_PAIR_MEMBERS`, `_EVAL_TRIAD_MEMBERS`, `_EVAL_TRIAD_DELTAS`, `_scheduled_cell_probabilities` | §3; [construction §2–§3](construction_fuel_damage.md#2-fd-baseline-v1); [artifacts and metrics §3](artifacts_metrics.md#3-matched-triads-and-the-episode-outcome-stream) |
| select the episode population | `rl/training/graph_generalized.py`: `EPISODE_DESIGNS`, `EpisodeDesign` (`generalized`, `generalized_v1_design`, `route_relative_population`), `resolve_episode_design`; `graph_train.py`: `TrainConfig.episode_design`, `validate`, `_generalized_setup_kwargs`, `_cardinality_kwargs` | §5 |
| change the GENERALIZED-V1 training cardinality sampler | `graph_generalized.py`: `sample_generalized_cardinality`, `derive_cardinality_seed`, `CARDINALITY_RNG_DOMAIN`, `EpisodeCardinality`, `cardinality_sampler_record`; `graph_train.py`: `episode_cardinality`, `_scheduled_cell` | §5 |
| build, load or consume the GENERALIZED-V1 18-stratum manifest | `graph_generalized.py`: `BENCHMARK_STRATA`, `BenchmarkManifest`, `build_benchmark_manifest`, `manifest_from_record`, `load_benchmark_manifest`, `manifest_seed_overlap`, `WorldIdentity`, `require_world_matches_manifest`, `require_matched_group_identity`; `graph_train.py`: `evaluate_benchmark`, `_require_benchmark_seeds_held_out`, `_require_benchmark_tag_namespace` | §5 |
| change what `episodes_per_iteration` counts, the attempt budget or the held-out band | `graph_train.py`: `TRAINING_ATTEMPT_POLICIES`, `TrainConfig.training_attempt_policy` / `max_attempts_per_iteration` / `max_training_attempts`, `train_attempt_seed`, `train_seed`, `TrainingQuotaError`, `seed_bands` | §6 |
| select a benchmark population with the GENERALIZED-V1 preflight, or read a failed preflight | `rl/training/graph_benchmark_preflight.py`: `run_benchmark_preflight`, `cell_windows`, `probe_world`, `_scan_cell`, `CandidateOutcome`, `_rejection_reason`, `_require_preflight_config`, `PREFLIGHT_STATUSES`, `_failure_block`, `_build_report`, `BenchmarkPreflightError` | §6 |
| change when a GENERALIZED-V1 run stops training | `graph_train.py`: `EARLY_STOPPING_POLICIES`, `TrainConfig.early_stopping_enabled` / `early_stopping_earliest_stop_iterations`, `_EarlyStoppingMonitor`, `EarlyStoppingIntegrityError`, `_early_stopping_summary`, `TERMINATION_REASONS` | §7 |
| change the GENERALIZED-V2 two-stage population | `graph_generalized.py`: `GENERALIZED_V2_AGENT_COUNTS`, `GENERALIZED_V2_KNOWN_OFFSETS`, `GENERALIZED_V2_REQUIRED_BACKEND`, `PreSolveCardinality`, `RouteRelativeHiddenLoad`, `sample_generalized_v2_pre_solve_cardinality`, `resolve_route_relative_hidden_load`, `resolved_v2_cardinality`; `rl/training/graph_hidden_placement.py`: `routed_ordinals`; `rl/training/graph_episode_setup.py`: `_resolve_route_relative_request`, `_require_generalized_v2_cardinality`; `graph_train.py`: `v2_pre_solve_cardinality` | §8 |
| record the population identity of a failed GENERALIZED-V2 attempt | `graph_episode_setup.py`: `RouteRelativePopulationRecorder`; `graph_train.py`: `_v2_failure_population`, `_failure_cardinality`, `_population_recorder_kwargs` | §8 |
| change the GENERALIZED-V2 benchmark schema, cells, profiles or frozen identity | `graph_generalized.py`: `V2_BENCHMARK_BASE_CELLS`, `V2_BENCHMARK_WORLDS_PER_CELL`, `V2_BENCHMARK_PROFILES`, `V2WorldIdentity`, `V2WorldPreflight`, `V2BenchmarkManifest`, `build_v2_benchmark_manifest`, `v2_manifest_from_record`, `load_v2_benchmark_manifest`, `require_v2_world_matches_manifest` | §9 |
| change the GENERALIZED-V2 preflight or its typed world-level refusals | `graph_benchmark_preflight.py`: `_run_v2_benchmark_preflight`, `_require_v2_preflight_config`, `v2_cell_windows`, `_scan_v2_cell`, `probe_v2_world`, `v2_rejection_reason`, `V2_REJECTION_REASONS`; `graph_episode_setup.py`: `RouteRelativeNoRoutesError`; `graph_hidden_placement.py`: `BoundedBackoffExhaustedError` | §9 |
| evaluate a GENERALIZED-V2 run | `graph_train.py`: `TrainConfig.benchmark_profile`, `_evaluate_v2_benchmark`, `_V2BenchmarkTally`, `_v2_behaviour_summary`, `_v2_benchmark_member_identity`, `_observe_v2_world_identity` | §9; reading rules in [artifacts and metrics §6](artifacts_metrics.md#6-reading-preserved-artifacts) |

Every row except configuration and cell edits is a research-validity change
([`cc_review.md` §4](../workflows/cc_review.md#4-risk-and-verification)).

## 11. Known limitations and open items

- **Complete Git provenance is REQUIRED for a real training run (`1b48145`).** `train`
  raises before policy, generator, episode or optimizer work unless BOTH the full commit SHA
  and the clean/dirty verdict were determined, so a run cannot be launched from a checkout
  where `git` is unavailable, times out, or cannot read the index. A dirty tree is a
  hazard, not a blocker: it WARNS and runs. Consequence for tooling: anything driving
  `train` outside a working checkout must inject the verdict (the tests patch
  `_git_provenance`) rather than expect it to be optional.
- **Added enemy airbases are not seed-stable by id.** `ScenarioGenerator` mints a FRESH
  uuid for every red airbase it ADDS on each `generate()`, even at a fixed seed
  (geometry and utility identical, id different). The base template holds 3 red
  airbases, so at the locked `(6,6)` HALF the targets are minted per run. Consequence:
  `graph_rollout`'s `known_target_ids` and any id-keyed cross-run comparison are
  unreliable for added targets — compare by geometry fingerprint `(lat, lon, utility)`.
  Scenario `/currentScenario/id` and `/name` are likewise unseeded; template unit ids
  ARE stable.
