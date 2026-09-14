# Artifacts and metrics contract — visual artifacts, figures, outcome streams, persistence and per-wake diagnostics

> **Read this when** you change what a run writes or how it is summarized or plotted, or when
> you **review a completed run** or read preserved evidence: §6 lists the reading rules and
> known artifact defects every reviewer needs.
>
> **Status: normative technical contract.** Sections 1–5 and 7 were moved **verbatim** from
> `CLAUDE.md` at base `ae42cb01677f94868b2873008d87be677e31f0c8` (the visual-artifact, figure,
> measurement-surface, generalized-persistence and per-wake-diagnostics blocks of former §5,
> rows of former §6). Section 6 is new reading guidance written in this restructure and
> verified against code and preserved artifacts. Inside moved text a bare `§N` means that
> **former** `CLAUDE.md` section — resolve it with the
> [compatibility index](../../CLAUDE.md#8-compatibility-index-for-older-references). A
> statement inside a block that a run or result "does not exist" describes that block's own PR
> scope when it merged; current run and evidence state lives in the
> [handoff](../../graph_rl_project_handoff.md).
>
> Related: [training and benchmarks](training_benchmarks.md) ·
> [experiments workflow](../workflows/experiments.md) · [measurement history](../history/measurements.md).

## 1. Visual artifacts

- **Visual artifacts — the opt-in inspection surface (PR #10).** `TrainConfig.
  visual_artifacts` / `--visual-artifacts`, **OFF at both surfaces by default**. It is
  OBSERVATION, not measurement: nothing captured is ever read back into the pipeline.
  When enabled it selects EVERY scheduled `pre_update` / `train` / `post_update`
  attempt — there is deliberately no per-seed filter, which would be a second
  artifact-selection language beside the seed schedule — and stores one collision-free
  bundle per attempt under `<run_dir>/visual_artifacts/`. Each bundle holds:
  - `known_only_scenario.json` — the generator's known-only world copied **byte for
    byte** (never regenerated, normalized, reserialized or rebuilt from tasks); the
    original under `<run_dir>/scenarios` is untouched;
  - `executed_t0_scenario.json` — the AUTHORITATIVE executed world, serialized from
    `ctx.game.export_scenario()` on the **env-2** game, called EXACTLY ONCE and BEFORE
    `build_fuel_damage_controller`, `run_episode`, the top-of-tick fuel mutation, any
    policy decision and any `env.step`. Env-1, `build_patched_scenario` output, the
    placement audit, the beliefs and the oracle tasks are derived views and none of them
    substitutes for it;
  - the BLADE playback `.jsonl`, produced ONLY through the existing locked contract —
    armed by `setup_episode(recording_export_path=<attempt dir>)`, started / stepped /
    exported by `run_episode`. No recorder internal is called and no scenario name is
    mutated; the per-attempt directory is what keeps recordings apart. All chunks are
    listed if the recorder ever splits one;
  - `artifact_manifest.json` — the attempt's identity stated EXPLICITLY (phase,
    iteration, `updates_completed`, eval round / episode / pair-member ordinals, attempt
    ordinal, training episode index, exact seed, scheduled condition, exact
    `episode_tag`), plus target-count expectations vs observations. `status` is
    `incomplete` until the bundle is whole, and — since `36365f2` — `complete` only once
    the three files exist AND the observed world cardinality RECONCILES with the scheduled
    cell; a bundle may be read as full only when it is `complete`. See the
    roster/world-truth integrity contract below for the two corrections this claim
    depends on (`sync_recordings`, and `finalize`'s reconciliation).

  **Failure routing.** An artifact filesystem / serialization failure is INFRASTRUCTURE:
  it raises `_VisualArtifactError`, which the train and eval attempt handlers re-raise
  AHEAD of their broad `except Exception`. It therefore aborts the run loudly, is never
  written as a `generation` / `setup` / `run` / `reward` failure, never enters
  `skip_and_account_v1`, and cannot shrink a scientific denominator by masquerading as
  an episode failure. A NORMAL episode failure is unaffected — it stays in the existing
  stage taxonomy and leaves a clearly marked `incomplete` bundle holding whichever
  pre-failure artifacts were valid; no recording is ever fabricated, because the tick
  loop deliberately exports none when the loop raised. A directory collision raises
  rather than overwriting or merging two attempts.

  **OFF-path invariance.** With `visual_artifacts=False` no `visual_artifacts/` directory
  is created, no identity is constructed, no scenario is copied, `Game.export_scenario`
  is not called, and NEITHER keyword is passed at all — `_recording_kwargs` omits
  `recording_export_path` from the `setup_episode` call and `_artifact_kwargs` omits
  `artifacts` from the `_run_one_episode` call, so both are the pre-feature calls.
  Neither path adds an RNG object or draw (bundle names derive only from already-resolved
  schedule metadata), and seeds, scenario tags, scenario names, policy inference, PPO
  inputs, the solver, the reward, fuel-damage semantics, the failure taxonomy and BLADE
  are all unchanged. The resolved flag is recorded in `run_config.json` through the
  existing `asdict(cfg)` path and echoed in the startup header.

## 2. Figures and presentation invariants

- **Figures: `<run_dir>/plots/`, three files, one claim each.** The legacy single
  four-panel `training_plot.png` is RETIRED and is no longer written. `plot_training` and
  `plot_training_subprocess` now return the LIST of figures written (empty when there is
  nothing to plot or matplotlib is missing; matplotlib stays optional and never fails a
  run). `run_summary.json` carries `plots_dir` + `plot_paths`, and the legacy `plot_path`
  key survives as a documented ALIAS of the performance figure.
  - `training_performance.png` — training reward; held-out evaluation as TWO SEPARATE
    per-condition series; the matched-pair delta.
  - `policy_diagnostics.png` — meta-action mix and policy entropy over the TRAINING
    decisions only.
  - `measurement_health.png` — the denominators, titled as health and explicitly NOT
    performance: train `success_fraction` and `wake_fraction_of_successful`, eval episode
    `success_fraction`, eval `pair_success_fraction`, the absolute counts, and
    PER-CONDITION held-out completion (`eval_n_<condition>_attempted` / `_successful`).
- **The two presentation invariants.** (1) **Condition means vs the paired delta.**
  `eval_reward_mean_clean` and `eval_reward_mean_damaged` are each a mean over THAT
  condition's own SUCCESSFUL episodes, so when one condition fails more held-out seeds
  the two curves are not averages over the same completed seeds and their gap is NOT a
  within-seed effect — the panel title, both legend entries and the per-condition
  denominators say so. The ONLY within-seed comparison is `eval_paired_reward_delta`, over
  pairs whose BOTH members completed. Pooling the two conditions into one held-out curve —
  what the retired dashboard drew — averages across the very factor the cell was built to
  study, and is drawn only as an explicitly labelled fallback for pre-FD records carrying
  no per-condition means. (2) **The honest x-axis.** All three figures share ONE quantity,
  stamped on each: PPO updates completed BEFORE the measurement. Training points sit at
  `updates_completed_before` (the updates the policy that GENERATED those episodes had
  received) and eval points at `updates_completed`, so the untrained policy's first batch
  and its `pre_update` round share an origin. A batch or round with NO successful episode
  is DROPPED from a curve rather than drawn at 0 — the reward is oracle-normalized regret,
  so 0 is the OPTIMUM and plotting a total data loss there would invert its meaning; the
  gap is accounted for in `measurement_health.png`.

## 3. Matched triads and the episode-outcome stream

**FD-VARIABLE-SEVERITY-v1 measurement surface — matched TRIADS and the durable outcome
stream — `rl/training/graph_train.py`.**

- **The matched CLEAN / MILD / SEVERE TRIAD** (`_EVAL_TRIAD_MEMBERS`, in attempt order,
  each member a `(cell, mode)` pair). All three members use the SAME held-out seed —
  hence the same generated world, the same solved `A_init`, the same hidden geometry, and
  for the two damaged members the SAME deterministically selected ego — and differ ONLY in
  the fuel-damage event. That is what lets "did the actor respond DIFFERENTLY to a
  survivable loss than to an unsurvivable one?" be asked WITHIN one world rather than
  across worlds. The clean member reuses the existing `forced_clean` mode rather than
  needing a new one. Members carry DISTINCT artifact tags (`eval_member_tag`, slot
  `e·group_size + m`), and `TrainConfig.validate` sizes the tag namespace against the
  group size the run will really use. **A legacy run keeps its clean/damaged PAIR and
  evaluation NEVER silently becomes a triad** — only a `seeded_variable` run evaluates
  triads; `TrainConfig.eval_group_kind` reports `pair` or `triad` so a reader never has to
  count members.
- **A CELL IS A REPORTING LABEL, NOT A NEW CONDITION.** `_ConditionTally` stores per CELL
  — `clean` / `damaged` for a legacy run, `clean` / `mild` / `severe` for a variable one —
  and the clean/damaged keys are DERIVED by pooling (`cell_condition`). For a legacy run
  the cells ARE the conditions, the pooling is the identity, and every emitted key keeps
  exactly the value it had.
- **PRIMARY BEHAVIOURAL EVIDENCE: the severity-conditioned FD-WAKE META-ACTION RESPONSE**,
  not reward. It is tracked per cell with its OWN denominator — **FD WAKES**, which is at
  most the cell's successful-episode count and CAN be smaller (an event can fire without
  the policy ever being woken by it), so it is stored and reported separately rather than
  inferred. `_ConditionTally.success` counts EVERY successful episode of the cell but
  increments `fd_wakes[cell]` only on `wake_occurred`, so `fd_wakes[cell] <=
  successful(cell)` — with EQUALITY when every successful episode in that cell did produce
  an FD wake. The point of the separate denominator is that the two CAN diverge, never
  that they must.
  Rates are `None`, never `0.0`, on an empty wake population. **"Mild must always choose
  `PLAN_COMPLIANCE`" is NOT encoded anywhere as a correctness rule and must not be** —
  opportunistic engagement under a survivable loss can be rational; what is measured is
  whether the response DIFFERS, not whether it matches a prescribed label.
- **THE THREE WITHIN-SEED DELTAS** (`_EVAL_TRIAD_DELTAS`): `mild − clean`,
  `severe − clean`, `severe − mild`. **Every one is averaged over COMPLETE matched groups
  ONLY** — a triad needs ALL THREE members to succeed, a group with a failed member
  contributes to NONE of the deltas, is never repaired from its surviving members (a
  clean+mild pair inside a failed triad yields no mild−clean delta either), and is still
  visible in the attempt counts. The per-cell reward MEANS remain each over THAT cell's
  own successful subset, so — exactly as for the legacy pair (§5, the two presentation
  invariants) — the only within-seed claims are the deltas.
- **`episode_outcomes.jsonl` — ONE durable record per SUCCESSFUL attempt.** The
  per-iteration and per-round records are AGGREGATES, and an aggregate cannot be
  un-averaged: "how did the actor respond to MILD, episode by episode, and in which
  worlds?" is a per-episode question. Each record states its own identity, event and
  outcome once, appended and FLUSHED immediately, so a run killed mid-batch still accounts
  for every completed attempt. **It never duplicates the ledger** — failed attempts stay
  in `episode_failures.jsonl` and appear here NOT AT ALL, so the two files are disjoint by
  construction. **Missing is `null`, never `0`** (a clean episode has no fuel reading, an
  unfired event has no tick, an absent wake has no meta-action; a zero would read as a
  measurement). **`run_summary.json:/severity_response` is DERIVED FROM THIS FILE**
  (`_severity_response_from_outcomes`, with `severity_response_source` naming it), not
  from a separate in-memory aggregate — one metric path, so the summary cannot describe a
  run its own artifacts do not.
- **Everything the legacy reporting surface carried is preserved**: attempted /
  successful / failed counts per cell, per-cell reward means, RTB command yield (still
  from `FuelDamageOutcome.rtb_command_issued`, i.e. real Phase-2 COMMAND HISTORY, never
  the executor's `rtb_issued` latch), deaths and target coverage. The legacy
  damaged−clean delta key is `null` under a triad run, whose three named deltas are the
  complete statement of it.

## 4. Generalized persistence and aggregates

**PERSISTENCE — THE EXISTING CANONICAL ARTIFACTS NOW CARRY THE TASK-1/2/3 STRUCTURES.** No
new file was added: `episode_outcomes.jsonl` (schema version 2 **AS OF THIS TASK-4 LAYER — see
the version note below**) and `episode_failures.jsonl`
grew fields, and `run_summary.json` grew `episode_design` and `episode_design_policies`
(taken from the CONFIG rather than guessed from the records, so a run with zero completed
episodes still states its design) plus the derived `generalized` block. A SUCCESSFUL
attempt's outcome record now states, beside everything it already carried:

- **WHICH POPULATION** — `episode_design`, `generalized`, the four policy ids and
  `target_destruction_probability`, **on BOTH designs**, so a reader never has to infer the
  design from the ABSENCE of generalized keys;
- **CARDINALITY, REQUESTED AND REALIZED** — `agent_count` / `known_requested` /
  `hidden_requested` / `targets_requested` / `cardinality_source` beside `known_realized` /
  `hidden_realized` / `targets_realized` / `hidden_short_realized`, plus the whole
  `construction_audit`, the backoff `candidate_order`, the per-candidate rejection slugs and
  the id-free `hidden_geometric_fingerprint`;
- **FD ELIGIBILITY AND THE EVENT CERTIFICATE** — `fd_eligibility_policy`,
  `fd_post_fd_wake_policy`, the eligibility rng domain and derived seed, the candidate count,
  candidate order, considered ordinals, rejection slugs and `selected_ordinal`, the whole
  `fd_eligibility_audit` and `fd_certificate`, and the `fd_certificate_fingerprint`;
- **POST-FD ADAPTATION UNDER ITS OWN DENOMINATORS** — `post_fd_policy` / `armed` / `active` /
  `deactivation_reason` / `boundaries_confirmed` / `boundaries_with_remaining_mission` /
  `boundaries_terminal` / `boundary_wakes` / `boundary_ticks` / `boundary_meta_actions` (plus
  names), **never folded into** the immediate-wake pair `fd_wake_occurred` /
  `fd_wake_meta_action`, over which an approved measurement is reported;
- **THE REFERENCE DECOMPOSITION AND THE CONTINUATION-SOLVER AUDIT** — `reference_kind`,
  `reference_checkpoint_tick`, `u_achieved` / `u_oracle` / `u_ref` / `u_prefix` /
  `u_cont_ref` / `u_post` / `u_aircraft`, `reward_ratio`, `reward_penalty`, the allocated and
  candidate task counts, the continuation-agent count and its recorded
  `reference_excluded_agents`, `reference_solver_invoked` / `_accepted` / `_termination` /
  `_seconds`, and the whole `reference_record`;
- **SCORED VS UNSCORED COMPLETION** — `unique_completed_targets`, `scored_completed_targets`,
  `unscored_completed_targets`, `unscored_completed_target_ids`;
- **AIRCRAFT LOSS AND THE SELECTED EGO'S REAL RTB** — `n_dead` beside
  `fd_rtb_command_issued`, which is still real Phase-2 COMMAND HISTORY
  (`FuelDamageOutcome.rtb_command_issued`) and **never** the executor's `rtb_issued` latch;
- **BENCHMARK STRATUM AND WORLD IDENTITY** — `benchmark_manifest_id`, `benchmark_stratum`,
  `benchmark_group_key`, `benchmark_agent_count`, `benchmark_load_bucket`,
  `benchmark_world_ordinal` and the id-free `benchmark_world_identity`. These keys are
  present as `null` on every NON-benchmark episode (`_EMPTY_BENCHMARK_KEYS`), so ONE schema
  reads both and an absent key is never confused with a writer that forgot it.

**THE "MISSING IS `null`, NEVER `0`" RULE EXTENDS TO THE NEW BLOCKS, and there it is a
statement about the DESIGN rather than about the episode:** an `exact_v1` construction
genuinely has no backoff audit, a legacy plan has no certificate, a `single_wake_v1` run has
no post-FD adaptation, and a `static_t0_v1` episode has no reference object. `null` there
means "this design produces no such structure", never "it measured zero". Every FD, reference
and construction number is copied VERBATIM from the component's own frozen record and never
recomputed, so this stream cannot disagree with the aggregate that summarizes it.
`_reward_breakdown_record` pins an EXPLICIT field list rather than `asdict`, so a future
`EpisodeReward` field added for an internal purpose does not silently start appearing in a
scientific artifact.

**THE SCHEMA VERSION IN THIS PARAGRAPH IS THE TASK-4 ONE, AND THE CURRENT WRITER IS VERSION 3.**
"Schema version 2" was accurate when this layer landed, and it is preserved as that record.
The CURRENT writer is `graph_train._EPISODE_OUTCOME_VERSION = 3`, moved `2 → 3` by the LATER
per-wake FD policy diagnostics layer (`81a148f8`, PR #52 — §5, §7) to carry the per-wake
actor diagnostics block; **every generalized key this paragraph describes is unchanged by
that move**, and a v2 artifact stays truthfully readable (`_observed_artifact_schema`
reports what the RECORDS carry, never the writer's constant). **NEITHER PR #57 NOR THIS
DOCUMENTATION TASK CHANGED THE SCHEMA:** the GENERALIZED-V2 `generalized_v2_population`
block is added CONDITIONALLY at version 3, not by a bump.

**THE FAILURE LEDGER KEEPS THE SCHEDULED POPULATION IDENTITY, WHETHER OR NOT A WORLD WAS
BUILT.** A failed attempt records `agent_count` / `known_requested` / `hidden_requested` /
`cardinality_source` and the benchmark identity keys — the world it was SCHEDULED to build.
It records those because **a failed attempt leaves NO successful `episode_outcomes` record,
so nothing else in the artifacts can reconstruct which stratum and which requested load the
attempt belonged to**: without them a failed HIGH-load attempt would be invisible in the
requested-vs-realized distribution and a failed benchmark member would be missing from its
stratum's denominator, so both would silently shrink.

**A FAILED ATTEMPT MAY HAVE FAILED BEFORE *OR* AFTER WORLD CONSTRUCTION, AND THE STAGE
ATTRIBUTION IS UNCHANGED.** `skip_and_account_v1` accounts failures at four stages —
`generation`, `setup`, `run`, `reward` — and only the first two are necessarily
pre-construction: a `run`- or `reward`-stage failure occurs after `setup_episode` really
built a world and, at `reward` stage, after `run_episode` really executed one. **The ledger
therefore records the SCHEDULED cardinality, never a realized one** (there may be no
realized measurement it could safely report, and reading one out of a failed attempt is
exactly the reconstruction this contract refuses). *(SUPERSEDED, and corrected here: this
paragraph previously said those keys are recorded "precisely because it never built a
world". That is true of a `generation` / `setup` failure and FALSE of a `run` / `reward`
one; the stage taxonomy itself is unchanged.)*

The ledger also records `reference_fault_reason` — which, by the routing above, **can only
ever carry the attrition case**, since an aborting reference fault never reaches this ledger
at all. A failure is still recorded ONCE, still never retried and still never replaced.

**AGGREGATES — DERIVED FROM THE CANONICAL STREAMS, WITH EXPLICIT DENOMINATORS, AND JUDGING
NOTHING.** `run_summary.json:/generalized` is built by `_generalized_summary` from
`episode_outcomes.jsonl`, `episode_failures.jsonl` and the per-round eval records — ONE
metric path, exactly as `severity_response` already is — so the summary cannot describe a run
its own artifacts do not. The two streams are DISJOINT by construction, so `attempted ==
successful + failed` per bucket by construction rather than by assumption. It carries:
attempted / successful / failed by `agent_count` and by `hidden_requested`;
`cardinality_requested_vs_realized` as a HISTOGRAM per requested load with `n_short_realized`
and `short_realized_fraction`; the construction-backoff and FD-eligibility rejection tallies;
the selected-ordinal histogram; post-FD adaptation counts with
`rates_over = "post_fd_boundary_wakes"` (a rate over BOUNDARY WAKES, which is at most the
boundary count and can be smaller — a TERMINAL boundary correctly wakes nobody, so the two
are counted apart rather than one inferred from the other); the reference block (kinds,
`n_solver_invoked` vs `n_solver_accepted` reported SEPARATELY because a SKIPPED degenerate
solve is a legitimate zero reference and not a failure, terminations, solver seconds,
allocated and candidate task counts, and the scored-vs-unscored totals);
`reference_fault_attrition`; and the `benchmark` block, whose FINAL-round strata, base cells
and named deltas are reported with their own `_n` denominators and are kept STRICTLY APART
from `strata_attempt_totals_across_rounds`, flagged
`totals_across_rounds_are_repeated_measures: true` — every round re-measures the same frozen
manifest, so cross-round totals describe a TRAJECTORY and **are not independent worlds**.

**REQUESTED-VS-REALIZED IS REPORTED FOR INSPECTION, AND NO ACCEPTANCE THRESHOLD IS
INVENTED.** `measurement_health.png` gains a FOURTH panel — drawn only when a run has a
cardinality to show, so a fixed-cell run draws the three panels it always did, on the same
axes, unchanged — plotting requested vs realized hidden load as GROUPED COUNTS per requested
load: deliberately a distribution, never a mean, because a mean is exactly what would hide a
HIGH stratum whose realized load collapses toward 1. It belongs to MEASUREMENT HEALTH rather
than to a new figure because this is a DENOMINATOR question: a HIGH stratum that keeps
realizing one hidden target is not a HIGH stratum. **Neither `_generalized_summary` nor the
panel applies a pass/fail threshold, and neither returns a verdict.** Whether a future
concrete benchmark population is acceptable is a HUMAN / GPT SCIENTIFIC REVIEW decision taken
before any measurement; the code reports the shape and stops there.

## 5. Per-wake FD policy diagnostics

**GENERALIZED-V1 DURABLE PER-WAKE FD POLICY DIAGNOSTICS (MEASUREMENT HARDENING) —
`rl/action/graph_action.py` + `rl/training/graph_tick_loop.py` +
`rl/training/graph_train.py` (`81a148f8`, integrated `28eb8dad`, PR #52 — §7).**

A REPORTING layer, and nothing else. It adds NO episode mechanism: the bounded-backoff
geometry, the FD certification physics, the post-FD boundary semantics, the
continuation-reference arithmetic, the Task-4 selector / sampler / manifest / persistence,
the Task-5 quota, budget and preflight, and the opt-in early-stopping rule are exactly the
contracts above. What it changes is WHAT A COMPLETED RUN CAN BE ASKED AFTERWARDS.

**WHY IT EXISTS.** The R1 diagnostic replay had to reconstruct, OFFLINE from checkpoints,
what the actor saw and what its masked distribution looked like at each fuel-damage wake —
which is only possible while the checkpoints, the frozen manifest and the exact measured
code SHA still exist together. Recording it AT THE DECISION removes that dependency, and it
removes a worse hazard with it: an offline reconstruction can disagree with the decision it
claims to describe, and a report that disagrees with the run it describes is worse than no
report at all.

**TWO VERSIONED SCHEMAS, AND BOTH ARE ADDITIVE.**
`_EPISODE_OUTCOME_VERSION` moves `2 → 3`. **VERSION 3'S ADDITION IS ONE CONCEPTUAL PER-WAKE
DIAGNOSTICS BLOCK, AND THAT BLOCK IS REPRESENTED BY THREE SIBLING FIELDS ON THE OUTCOME
RECORD — NOT BY ONE FIELD**: `wake_diagnostics_schema_version` (the block's OWN version,
`_WAKE_DIAGNOSTICS_VERSION = 1`, so the diagnostics can version independently of the
record that carries them), `n_wake_decisions` (the count) and `wake_decisions` (the
per-wake records themselves). The writer emits all three fields together. Existing
reporting readers remain as implemented — they detect diagnostics from a list-valued
`wake_decisions` field and observe the diagnostics schema version separately — and
**this contract does NOT claim atomic three-field validation.**
**EVERY OTHER v2 FIELD IS UNCHANGED IN NAME, MEANING AND VALUE.** A
successful ZERO-WAKE episode records `[]` — a real, legitimate outcome of the
event-triggered design, and deliberately NOT `null`, which would read as "not recorded".

**`wake_decisions` IS DURABLE AND REPORTING-ONLY, AND THE SECOND HALF IS STRUCTURAL.**
**THE DURABLE DATA PATH HAS THREE DISTINCT STAGES AND THEY MUST NOT BE COLLAPSED INTO
"IT REACHES ALL THREE".** (i) **RAW PER-WAKE RECORDS → `episode_outcomes.jsonl`**,
which is where they are PERSISTED and is the record. (ii) **DERIVED REPORTING SUMMARIES
→ `run_summary.json`**, which does **NOT** carry the raw list: it carries what was
COMPUTED FROM that stream — `fd_policy_sensitivity` (via
`_fd_policy_sensitivity_from_outcomes`) and `observed_artifact_schema` (via
`_observed_artifact_schema`), beside the `wake_kinds` and `wake_diagnostics_source`
labels — the same ONE-METRIC-PATH discipline `severity_response` already follows, so
the summary cannot describe a run its own artifacts do not. (iii) **DERIVED PLOTTING
INPUT → the figures**, through `_fd_sensitivity_plot_data` and
`_plot_fd_policy_sensitivity`, from the same stream and never from a parallel in-memory
aggregate.
**REPORTING CONSUMERS READ IT TO PERSIST AND SUMMARIZE IT, BUT NO ACTING, MASK, BELIEF,
COMMAND, PPO/CTDE INPUT, ADVANTAGE, REWARD, OPTIMIZER, EARLY-STOPPING,
EVALUATION-SCHEDULING OR CHECKPOINT-CONTROL PATH READS IT BACK** — and neither does
failure classification. The claim is deliberately that of a bounded set of REPORTING
readers, never that no reader exists: persisting and summarizing a record IS reading it,
and a contract that denied it would be false in the one direction that matters. It is
built AFTER `sample_action` has already drawn, from the SAME `logits` and `mask` that
call acted on.

**THE THREE PROPERTIES THAT MAKE IT SAFE TO COMPUTE INSIDE A TRAINING RUN.**

- **NO RNG DRAW.** `graph_action.summarize_decision` samples nothing — no `dist.sample()`
  and no generator of any kind — so the torch RNG state is byte-identical either way and no
  later stochastic `sample_action` can be displaced by one draw. A diagnostic that perturbed
  the training stream would silently change the run it exists to describe.
- **NO GRADIENT.** `logits.detach()` is taken BEFORE the distribution is rebuilt, so no
  autograd node is created and the PPO graph is untouched.
- **NO CONTROL PATH.** The record is attached to `Transition.decision` and read only by
  `graph_train._wake_decision_records`, which RECOMPUTES NOTHING and SKIPS a transition that
  carries no such field rather than inventing one.

**THE PROBABILITIES ARE THE ACTOR'S OWN, FROM THE SHARED CONSTRUCTION SITE.** Every
probability, entropy and argmax comes from `graph_action._masked_dist` — the SAME single
site `sample_action` and `evaluate_action` already route through — evaluated on a DETACHED
copy of the same logits in their ORIGINAL dtype. It is deliberately NOT a second masked
softmax: an independent implementation, in another dtype or with another tie rule, could
report a distribution the actor never acted on. The joint argmax is literally
`torch.argmax(flat)` — the expression the deterministic branch of `sample_action` evaluates
— so an exact tie breaks identically, the row-major `flat = node*3 + meta` convention is the
shared one, the raw entropy is the same scalar the PPO entropy bonus receives, and the
top-two margin is differenced IN TORCH on the exact probabilities before any JSON conversion.
Conversion to plain builtins happens only after every quantity is final.

**TWO VIEWS OF ONE `k × 3` SURFACE, NAMED SO THEY CANNOT BE CONFUSED.** One meta-action owns
`k` cells, so its total mass is spread across them and the highest JOINT CELL is not
generally the argmax of the per-column SUM: a meta-action can hold the largest total mass
while every one of its cells sits below a rival's single best cell. Both are therefore
reported, with an explicit `joint_vs_aggregate_disagree` flag —
`selected_joint_cell_abort_fraction` is what deterministic evaluation ACTUALLY DOES, and
`aggregate_p_abort_mean` is the total MASS on the abort column and is **NEVER** the
probability of the selected action. Normalized joint entropy is `raw / log(valid cells)` and
is `None` — never `0.0` or `1.0` — when fewer than two cells are valid, because a single
valid cell has no spread to normalize. The record also carries the actor-INPUT summary only
the tick loop can see: graph shape, the ego's own `fuel_norm`, the `reachable_by_ego` vector,
the `dist_to_ego_norm` column, and how much of that column is SATURATED at the fixed
normalizer (`n_task_distance_clipped` / `fraction_task_distance_clipped`) — a property of the
NORMALIZER, not of the policy, unobservable in any aggregate, recorded per wake and left for
a reader to judge. **The normalizer and the feature are UNCHANGED.**

**THE THREE WAKE KINDS ARE DISJOINT, TAGGED AT THE TRIGGER, AND NEVER INFERRED FROM THE
ACTION.** `WAKE_KINDS` = (`ordinary`, `immediate_fuel_damage`, `post_fd_boundary`).
`Transition.wake_kind` is stamped in `run_episode` from the SAME `ego_fuel_damage` /
`ego_post_fd` flags that decide what `decide_triggers` is told — "the actor aborted" is not
evidence about why it was ASKED, so reconstructing the kind from the selected action is
exactly what this field exists to prevent. The separation is load-bearing rather than tidy:
an approved measurement is reported over the IMMEDIATE-FD population alone
(`FuelDamageOutcome.wake_occurred` / `wake_meta_action`), and a later completion-boundary
decision of the same ego has its OWN denominator (`PostFdAdaptationOutcome.boundary_wakes`),
so folding one into the other would silently change what an approved measurement means.
**THE ORDINARY CALL IS BYTE-UNCHANGED**: the `wake_kind` keyword is OMITTED entirely for an
ordinary wake, the same keyword-omission discipline `_artifact_kwargs` / `_ctde_kwargs` /
`_cardinality_kwargs` already use.

**THREE PHASE POPULATIONS, KEPT APART, AND POOLING IS NAMED WHERE IT HAPPENS.**
`_fd_policy_sensitivity_from_outcomes` gives `train`, `pre_update` and `post_update` their
OWN blocks under `by_phase` and NEVER averages them: training is a stochastic actor on a
sampled population and held-out evaluation is a deterministic actor on a frozen one, so a
rate whose denominator mixes them describes neither. A pooled view exists only under
`all_phases_pooled`, whose own name and note say what it did, and every scientific severity
quantity is taken over `_EVAL_PHASES` and never over `_ARTIFACT_PHASES`. Every rate carries
its explicit denominator, and an EMPTY population reports `None` — never `0.0`, which on a
rate is a measured value that would read as "the actor never aborted" when the truth is "no
wake of this kind occurred".

**MATCHED DELTAS USE THE FULL ROUND IDENTITY PLUS THE BENCHMARK GROUP IDENTITY.** A
`benchmark_group_key` alone is NOT an identity: the SAME frozen world group is re-measured in
EVERY evaluation round, so keying a matched table on it collapses a whole run into one bucket
and turns repeated measures of one world into what reads like independent worlds.
`_round_identity` is therefore `(evaluation_stage, updates_completed, eval_round_ordinal,
benchmark_manifest_id)`, and the group key identifies the WORLD *within* that round: a group
contributes a severe-minus-mild delta only when BOTH members are present IN THE SAME ROUND,
deltas are reported PER ROUND, and the cross-round pool is explicitly flagged
`totals_across_rounds_are_repeated_measures: true`. Target uuids are used nowhere — they are
not seed-stable labels (§8).

**THE FINAL EVALUATION ROUND IS SELECTED SEMANTICALLY, NEVER `eval_records[-1]`.**
`_select_final_eval_record` orders by the run's own monotone `eval_round_ordinal` and
CROSS-CHECKS it against `updates_completed`; `_final_eval_identity` is the SINGLE validator
of a complete round identity (`_FINAL_EVAL_IDENTITY_FIELDS`), strict on type as well as
presence (stage must be in `_EVAL_PHASES`; a `bool` is refused where an `int` is required,
since `True` would otherwise pass as update count 1). It REFUSES rather than guessing, with a
stated reason PERSISTED as `run_summary.json:/final_eval_selection`:
`no_evaluation_records`, `incomplete_evaluation_identity`,
`ambiguous_identity_duplicate_rounds`, `ambiguous_highest_round_ordinal` or
`contradictory_ordering_ordinal_vs_updates_completed`. **THE SUBSET IS THE DANGEROUS CASE,
WHICH IS WHY IT IS REFUSED RATHER THAN TOLERATED**: a digest missing its ordinal still
carries a stage and an update count, those two very often single out exactly one round, and
such a match would SUCCEED and produce a "final round" nobody validated — indistinguishable
from a correct selection in the artifact. `_select_final_matched_round` compares ALL THREE
fields through that SAME validator, so the two selectors cannot come to disagree about what a
valid identity is, and the ONE selected record drives `final_eval`, every `final_eval_*`
scalar and the sensitivity table's own final round — so the three cannot name different
rounds. `eval_group_kind` is a RUN-INVARIANT label and is read from the selected round when
one exists, with the positional expression surviving only as the fallback for a record set
the selector refused, and the config as the last resort.

**LEGACY v2 ARTIFACTS REMAIN TRUTHFUL AND READABLE.** `build_run_summary` runs on ANY run
directory, so `_observed_artifact_schema` reports the schema the RECORDS ACTUALLY CARRY —
`no_records` / `uniform` / `mixed`, never collapsed — and the writer's own constants appear
only under `_writer` names that cannot be mistaken for an observation. Stamping the current
constants on a legacy directory would tell a reader that a v2 artifact is v3 and that it
carries wake diagnostics it does not. `_fd_policy_sensitivity_from_outcomes` returns
`{"recorded": false, …}` with a note when no record carries the field — a truthful "not
recorded", never a table of fabricated zeros — and every reader treats an absent key as "not
recorded" rather than as zero.

**`fd_policy_sensitivity.png` IS OPTIONAL AND EVALUATION-ONLY, AND `_PLOT_FILENAMES` STILL
NAMES EXACTLY THE THREE REQUIRED FIGURES.** `_PLOT_FILENAMES` remains
(`training_performance.png`, `policy_diagnostics.png`, `measurement_health.png`) — the
REQUIRED set a shortfall against which is reported as "plots incomplete" — and the new figure
lives in the SEPARATE `_PLOT_OPTIONAL_FILENAMES`, so a run directory written before schema v3
legitimately has three figures and is NOT reported as broken. `_fd_sensitivity_plot_data` is
PURE (no matplotlib) so the scientific content is testable without rendering, and it admits
**EVALUATION records only** and **IMMEDIATE-FD wakes only**: a training row can enter no value
and no denominator on this figure, and ordinary and post-FD-boundary wakes appear on no panel
of it. `_plot_fd_policy_sensitivity` returns `None` on exactly that condition, and
`run_summary.json:/optional_plot_paths` is keyed off the FIGURE'S OWN predicate rather than
off the digest's broader `recorded` flag — which is true as soon as ANY record, a training one
included, carries diagnostics — so a train-only or evaluation-disabled run declares nothing
and never promises a file it will not write. The per-cell argmax-disagreement series is
emitted for EVERY cell that has immediate-FD wakes, never for "whichever cell sorts first"
(normally `clean`, which normally has none, so the series would silently be empty).
`policy_diagnostics.png`'s entropy panel changed its LABEL ONLY — the plotted series is
byte-unchanged — to say that it is the RAW joint entropy and therefore cardinality-dependent.

**WHAT IS EXPLICITLY NOT IN THIS TASK.** Target destruction stays DETERMINISTIC at
`probability = 1` — **`p(destroy) < 1` was NOT implemented here and remains a separate future
Grade-A research task**. The frozen solver / BONMIN and the vendored BLADE engine are
untouched. No actor, encoder, `ActionHead`, PPO, GAE or critic architecture change; **no new
`MetaAction`**; no change to the action surface, the mask, `sample_action`'s or
`evaluate_action`'s semantics, terminal-on-last credit placement, `graph_reward`'s
`static_t0_v1` formula, or the no-communication boundary; no scenario, world-construction,
reward, solver, fuel-damage, seed-formula, episode-design, cardinality-sampler, manifest,
preflight, attempt-policy, early-stopping or evaluation-schedule change; no peer behaviour
change and no communication channel of any kind. `evaluate`, `evaluate_benchmark` and
`save_checkpoint`'s payload are untouched, and checkpoint RESUME remains out of scope.
**NOTHING FROM THIS LAYER REACHES THE ACTING PATH:** no wake kind, probability, entropy,
clipping count, ownership label, schema version, selection reason or plot field enters
`GraphObservation` or `CentralGraphObservation`. **PR #52 PRODUCED NO SCIENTIFIC MEASUREMENT
AND DID NOT MODIFY THE R1 RUN, ITS ARTIFACTS OR ITS VERDICT** — R1 was measured at code SHA
`4af6c5aa5dd28072692bfda63282964b55010aae`, which PREDATES this layer, so **R1's own
artifacts are episode-outcome schema v2 and carry NO `wake_decisions`**; §7 owns the R1
record and §8 the phase state.

## 6. Reading preserved artifacts

These rules apply to every completed run directory and every evidence commit. They record
facts; none of them re-approves a measurement.

### 6.1 Known summary-label defect: `run_summary.json:/generalized/cardinality_sampler`

`graph_train._generalized_summary` builds `run_summary.json:/generalized` and writes that key
with the exact string `"cardinality_sampler": cardinality_sampler_record(),` — the
**GENERALIZED-V1** sampler record (`generalized_cardinality_uniform_v1`, `A ∈ {2, 3, 4}`) —
for every generalized run, **`generalized_v2` included**. The run's configuration block is
built correctly: `run_config.json:/episode_design/cardinality_sampler` selects
`generalized_v2_cardinality_sampler_record()` when `cfg.route_relative_population` is true.
Both preserved GENERALIZED-V2 development runs show exactly this split (their `run_config.json`
names `generalized_v2_pre_solve_uniform_v1`; their `run_summary.json` names
`generalized_cardinality_uniform_v1`), and PR #62's `artifact_sha256.txt` already records the
caveat.

- **It is a summary label only.** Episodes are sampled by the design's own sampler; read the
  sampler from `run_config.json:/episode_design/cardinality_sampler` and per-episode
  `cardinality_source` in `episode_outcomes.jsonl`, never from that summary key.
- **Never normalize archived bytes** to correct it. Fixing the code is a separate task that no
  documentation record authorizes.

### 6.2 Measured code SHA versus evidence-commit identity

A run's **measured code SHA** is `run_config.json:/provenance/git/commit` (with
`dirty = false`). An **evidence commit** is a later preservation commit whose tree adds copies
of the run's files; its SHA is not a measurement identity and must never be cited as one.
Record both, labelled, whenever evidence is cited. For PR #61 and PR #62 each evidence commit's
single parent is the measured code SHA `ae42cb01677f94868b2873008d87be677e31f0c8`.

### 6.3 Other standing reading rules

- **Generated uuids are not seed-stable**: compare worlds across runs by geometry
  fingerprints and ordinals, never by target or agent id
  ([training and benchmarks §11](training_benchmarks.md#11-known-limitations-and-open-items)).
- **Elapsed time is two distinct quantities**: the harness's `run_summary.json:run_seconds`
  excludes process start-up, `conda run` dispatch, imports and teardown, so it is never the
  external wall clock.
- **Schema versions are observed, not assumed**: read
  `run_summary.json:/observed_artifact_schema`. Runs measured before PR #52 (R1 included) carry
  episode-outcome schema v2 and no `wake_decisions`.
- **Sharded evidence streams**: an evidence commit may split `episode_outcomes.jsonl` into
  line-aligned shards; reconstruct by concatenating the shards in the order given by
  `episode_outcomes.index.json` and check the SHA-256 against `artifact_sha256.txt` before
  reading.
- **External artifacts named but not committed** (checkpoints, benchmark manifests) are
  identified by path and SHA-256 only; an unverified external file is not evidence of its own
  contents.

## 7. Code routing

| … | Go to |
|---|---|
| Change a FIGURE (or add one) | `rl/training/graph_train.py` (`plot_training`, `_plots_dir`, `_plot_training_performance`, `_plot_policy_diagnostics`, `_plot_measurement_health`, `_PLOT_FILENAMES`, `_PLOT_X_LABEL` / `_PLOT_X_SEMANTICS`, `_xy`, `plot_training_subprocess`). Figures go to `<run_dir>/plots/`; the two presentation invariants in §5 (condition means vs complete-pair delta, and the honest x-axis) are contractual |
| Read what an episode ACTUALLY did, per successful attempt (not an aggregate) | `rl/training/graph_train.py` (`_EPISODE_OUTCOMES_FILENAME` = `episode_outcomes.jsonl`, `_episode_outcome_record`, `_append_episode_outcome_record`, `_severity_response_from_outcomes` and the `severity_response` / `severity_response_source` / `episode_outcomes_recorded` keys of `run_summary.json`). SUCCESSFUL attempts only — failures stay in `episode_failures.jsonl` and the two streams are disjoint by construction. The severity-response table is DERIVED from this file, never from a parallel in-memory aggregate. |
| Record or read PER-WAKE ACTOR DIAGNOSTICS (why a wake happened, what the actor saw, what its masked distribution looked like) | `rl/action/graph_action.py` (`summarize_decision`, built on the SHARED `_masked_dist`) + `rl/training/graph_tick_loop.py` (`WAKE_KINDS` = `WAKE_KIND_ORDINARY` / `WAKE_KIND_IMMEDIATE_FD` / `WAKE_KIND_POST_FD_BOUNDARY`, `OWNERSHIP_EGO` / `OWNERSHIP_PEER` / `OWNERSHIP_UNASSIGNED`, `_node_ownership`, `_decision_record`, `Transition.wake_kind` / `.decision`, and `_wake_decision(..., wake_kind=...)` — OMITTED for an ordinary wake) + `rl/training/graph_train.py` (`_EPISODE_OUTCOME_VERSION = 3`, `_WAKE_DIAGNOSTICS_VERSION = 1`, `_wake_decision_records`, `_wake_diag_digest`, `_wake_population_block`, `_EVAL_PHASES`, `_fd_policy_sensitivity_from_outcomes`, `_observed_artifact_schema`, and the `fd_policy_sensitivity` / `observed_artifact_schema` / `wake_kinds` / `wake_diagnostics_source` keys of `run_summary.json`). **RESEARCH-VALIDITY / GRADE A**: it is DURABLE and REPORTING-ONLY, and the data path has THREE distinct stages — RAW per-wake records PERSISTED in `episode_outcomes.jsonl`, DERIVED reporting summaries in `run_summary.json` (`fd_policy_sensitivity` / `observed_artifact_schema`, computed FROM that stream rather than copied from it) and DERIVED plotting input for the figures — so reporting consumers read it to persist and summarize it, but no acting, mask, belief, command, PPO/CTDE input, advantage, reward, optimizer, early-stopping, evaluation-scheduling or checkpoint-control path reads it back; the probabilities come from the actor's OWN `_masked_dist` on a DETACHED copy of the same logits, so no second implementation can describe a distribution the actor never used; **no RNG draw, no gradient and no control path is added**; the three wake kinds are DISJOINT and tagged at the TRIGGER, never inferred from the selected action, because an approved measurement is reported over the immediate-FD population alone; `train` / `pre_update` / `post_update` are SEPARATE populations and pooling is named where it happens; and an empty population is `None`, never `0.0` (§5) |
| Ask "which evaluation round was FINAL?" (never `eval_records[-1]`) | `rl/training/graph_train.py` (`_FINAL_EVAL_IDENTITY_FIELDS` = `evaluation_stage` / `updates_completed` / `eval_round_ordinal`, `_final_eval_identity` — the ONE validator — `_select_final_eval_record`, `_select_final_matched_round`, `_INCOMPLETE_FINAL_EVAL_IDENTITY`, `_round_identity`, `_matched_rounds`, `_immediate_fd_abort_mass`, and `run_summary.json:/final_eval_selection`). **RESEARCH-VALIDITY / GRADE A**: the last row a file happens to hold is a fact about the WRITER, not about the run, so the round is chosen by the monotone `eval_round_ordinal` CROSS-CHECKED against `updates_completed` and the selector REFUSES with a stated, persisted reason rather than guessing; a PARTIAL identity is refused even when the remaining fields would have been unique, because such a match succeeds silently and is indistinguishable from a correct one; ALL THREE fields are always compared, never a subset; and matched severe-minus-mild deltas pair through the FULL round identity PLUS `benchmark_group_key`, so re-measuring one frozen world in every round yields independent per-round deltas and a cross-round pool explicitly flagged as REPEATED MEASURES (§5) |
| Change or read the OPTIONAL FD-policy-sensitivity figure | `rl/training/graph_train.py` (`_PLOT_FD_SENSITIVITY` = `fd_policy_sensitivity.png`, `_PLOT_OPTIONAL_FILENAMES`, `_FD_SENSITIVITY_SERIES_KEYS`, `_FD_DISAGREEMENT_KEY`, `_FD_CELL_ORDER`, the PURE `_fd_sensitivity_plot_data`, `_plot_fd_policy_sensitivity`, the `None`-filtering in `plot_training`, the optional-existence pass in `plot_training_subprocess`, and `run_summary.json:/optional_plot_paths`). **RESEARCH-VALIDITY / GRADE A**: `_PLOT_FILENAMES` still names EXACTLY the three REQUIRED figures and completeness is judged against that set alone, so a pre-v3 run directory keeps three figures and is NOT reported as broken; the figure is **HELD-OUT EVALUATION ONLY** and **IMMEDIATE-FD WAKES ONLY**, so no training row and no ordinary or post-FD-boundary wake enters a value or a denominator on it; SELECTED-joint-cell action and AGGREGATE column MASS are separate panels under names that cannot be confused (the mass is NEVER P(selected action)); raw and normalized joint entropy are distinguished; the argmax-disagreement series is emitted PER CELL rather than for whichever cell sorts first; and `optional_plot_paths` is keyed off the figure's OWN predicate so a run never declares a file it will not write (§5) |
| Capture per-attempt VISUAL ARTIFACTS (known-only scenario + executed t=0 scenario + BLADE playback + manifest) | `rl/training/graph_train.py` (`TrainConfig.visual_artifacts` and the `--visual-artifacts` flag, `_AttemptIdentity`, `_AttemptArtifacts` with `open` / `capture_known_only_scenario` / `capture_executed_t0_scenario` / `sync_recordings` / `finalize` (which reconciles expected vs observed world counts before it will say `complete`) / `to_manifest`, `_VisualArtifactError`, `_recording_kwargs`, `_artifact_kwargs`; consumed by `_run_one_episode(..., artifacts=...)` and wired from `train` / `evaluate(..., artifacts_root=...)`). OFF by default and OFF is byte-unchanged — see the §5 trainer contract. `graph_tick_loop`, `graph_episode_setup`, `PlaybackRecorder.py` and `Game.py` are NOT touched; recording is armed only through `setup_episode(recording_export_path=...)`. |
| Persist or AGGREGATE the generalized per-episode diagnostics | `rl/training/graph_train.py` (`_episode_outcome_record`, written at the CURRENT `_EPISODE_OUTCOME_VERSION = 3` — **version 2 was the version this Task-4 layer landed at, and the later per-wake FD diagnostics layer moved the writer `2 → 3`**; the generalized keys below are unchanged by that move, `_reward_breakdown_record`, `_failure_record`'s scheduled-cardinality + `reference_fault_reason` fields, `_EMPTY_BENCHMARK_KEYS`, `_backoff_rejections` / `_eligibility_rejections`, `_generalized_summary` and `run_summary.json:/generalized`, `_construction_record`, `seed_bands(..., benchmark=...)` / `EVAL_SEED_SOURCE_MANIFEST`, the `episode_design` block of `write_run_config`, and the FOURTH `measurement_health.png` panel in `_plot_measurement_health`). **RESEARCH-VALIDITY / GRADE A**: every aggregate is DERIVED from the canonical jsonl streams (ONE metric path), every denominator is explicit, the two streams stay DISJOINT, `null` never means `0`, cross-round benchmark totals are flagged REPEATED MEASURES, and requested-vs-realized is REPORTED for human/GPT inspection with **no automatic acceptance threshold** (§5, §8) |
| Read why/how a run stopped | `train_records.jsonl:/early_stopping_check` (the durable per-check history — one entry per DUE check, on the iteration it was computed from; ABSENT entirely on a run with the feature off) + `rl/training/graph_train.py` (`_EARLY_STOPPING_RECORD_KEY`, `_early_stopping_summary`, `TERMINATION_REASONS` = `TERMINATION_REASON_PLATEAU` / `TERMINATION_REASON_MAX_BUDGET` / `TERMINATION_REASON_DISABLED`) + `run_summary.json:/early_stopping` (`enabled`, `policy`, `metric`, the configured shape, `earliest_possible_stop_iterations`, `triggered`, `termination_reason`, the planned/actual pairs `planned_iterations` / `completed_iterations`, `planned_successful_episodes` / `actual_successful_episodes`, `planned_max_training_attempts` / `actual_training_attempts`, `stop_completed_iterations` / `stop_iteration_index`, `n_checks`, `checks`, `checks_source`). **RESEARCH-VALIDITY / GRADE A**: the summary is DERIVED from the durable records (ONE metric path), the block is present on EVERY run so `disabled_fixed_budget` STATES the fixed-budget contract rather than leaving it to be inferred, planned and actual are always reported as a PAIR, `null` never means `0`, and **a triggered stop records only that the configured plateau rule fired — never a convergence or optimality claim** (§5) |
