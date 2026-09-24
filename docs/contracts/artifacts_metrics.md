# Artifacts and metrics contract — visual artifacts, figures, outcome streams, persistence and per-wake diagnostics

> **Read this when** you change what a run writes or how it is summarized or plotted, or when
> you **review a completed run** or read preserved evidence: §6 lists the reading rules and
> known artifact defects every reviewer needs.
>
> **Status: normative, current technical contract** for the code on `main`. Lock history is in
> [`implementation.md`](../history/implementation.md), block provenance in
> [`documentation_migration.md`](../documentation_migration.md), and current run and evidence
> state in the [handoff](../../graph_rl_project_handoff.md).
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
  own successful subset, so — exactly as for the legacy pair ([§2](#2-figures-and-presentation-invariants), the two
  presentation invariants) — the only within-seed claims are the deltas.
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
per-wake FD policy diagnostics layer (`81a148f8`, PR #52 — [§5](#5-per-wake-fd-policy-diagnostics)) to carry the per-wake
actor diagnostics block; **every generalized key this paragraph describes is unchanged by
that move**, and a v2 artifact stays truthfully readable (`_observed_artifact_schema`
reports what the RECORDS carry, never the writer's constant). **PR #57 DID NOT CHANGE THE
SCHEMA:** the GENERALIZED-V2 `generalized_v2_population`
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
exactly the reconstruction this contract refuses).

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
`rl/training/graph_train.py` (`81a148f8`, integrated `28eb8dad`, PR #52).**

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

**EPISODE OUTCOME VERSION 4, WAKE DIAGNOSTICS VERSION 2 (2026-09-16).** The action
representation changed to `semantic_k_plus_2_logmeanexp_v1`
([policy and CTDE §2](policy_ctde.md#2-encoder-action-head-and-selection-stage-4)), so the
MEANING of every selected action and probability changed and the versions move rather than hide
it: `_EPISODE_OUTCOME_VERSION = 4` adds a top-level `action_representation_id`, and
`_WAKE_DIAGNOSTICS_VERSION = 2` is the semantic per-wake schema below. Wake diagnostics 1 (records
of episode-outcome v3) describe the historical node-indexed joint representation and carry no
representation id; readers label them `LEGACY_ACTION_REPRESENTATION_LABEL`
(`legacy_node_indexed_joint_k_x_3`, a reader label never written to an artifact). **No archived
artifact is rewritten.**

**CURRENT WRITER: EPISODE OUTCOME VERSION 5, WAKE DIAGNOSTICS VERSION 3 (2026-09-24).** The actor
observation changed to `actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1` (the ego row
gained `mission_fuel_slack_norm`,
[policy and CTDE §1](policy_ctde.md#1-graph-observation-stage-3)), so what a wake was decided ON
changed and the versions move: `_EPISODE_OUTCOME_VERSION = 5` adds a top-level
`actor_observation_id`, and `_WAKE_DIAGNOSTICS_VERSION = 3` is wake diagnostics 2 — every
semantic-action field unchanged in name and meaning — PLUS three per-wake fields:
`actor_observation_id`; `ego_mission_fuel_slack_norm`, the ego row's column 1 **exactly as the
encoder received it** (float32, read off the same `GraphObservation`); and
`mission_slack_audit`, the immutable PRE-ACTION audit the builder computed for that value
(`MissionSlackEstimate.as_record()`: the ego's position, `current_fuel`, `max_fuel`,
`speed_knots`, `fuel_rate`, home base, its own confirmed target ids, the remaining assignments
with levels and private target coordinates, the excluded confirmed assignments, the chosen route
order with per-leg km, the return leg, total distance, required fuel, the fuel slack and the
normalized slack). The audit is the SAME object the feature was taken from — never a second
computation — and it is reporting metadata, never a model input; no severity, condition or other
privileged label is in it. A hand-built observation with no audit records `null`. Records of
episode-outcome v4 / wake diagnostics 2 carry the one-column agent row (`fuel_norm` only), which
`graph_builder.LEGACY_ACTOR_OBSERVATION_LABEL` (`actor_graph_task6_agent1_fuel_norm`) names for
readers; that label is never written to an artifact.

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
probability, entropy and argmax comes from `graph_action._semantic_dist` — the SAME single
site `sample_action` and `evaluate_action` route through — evaluated on a DETACHED copy of
the same scores in their ORIGINAL dtype. It is deliberately NOT a second softmax: an
independent implementation, in another dtype or with another tie rule, could report a
distribution the actor never acted on. The deterministic leaf is literally
`torch.argmax(semantic_logits)` — the expression the deterministic branch of `sample_action`
evaluates — so an exact tie breaks identically, the raw entropy is the same scalar the PPO
entropy bonus receives, and the top-two margin is differenced IN TORCH on the exact
probabilities before any JSON conversion. Conversion to plain builtins happens only after
every quantity is final.

**THE SEMANTIC PER-WAKE SCHEMA (wake diagnostics 2).** Each wake record names
`action_representation_id` and carries: `n_task_nodes`, `n_semantic_leaves` (`k + 2`, the
action-space size), `n_valid_semantic_leaves`, `n_abort_legal_nodes`,
`n_engage_legal_leaves`; the source `source_scores` `[k, 3]` and `source_cell_legal`;
`semantic_leaves` (leaf, meta-action, nullable node, legal, score, probability) and the flat
`semantic_probabilities`; the selected `selected_leaf` / `selected_meta_action(_name)` /
nullable `selected_node` / `selected_action_probability`; `semantic_probability_per_meta_action`
(PLAN = the one PLAN leaf, ABORT = the one ABORT leaf, ENGAGE = the sum over legal ENGAGE leaves);
`deterministic_argmax_leaf` / `deterministic_argmax_meta_action(_name)`;
`top_two_semantic_leaves` and `top_two_probability_margin`; `semantic_entropy_raw` and
`semantic_entropy_normalized` (`raw / log(valid leaves)`, `None` — never `0.0` or `1.0` — with
fewer than two legal leaves). A global action's `selected_node_ownership` is `None`.
**There is NO `joint_vs_aggregate_disagree` field**: that quantity described the retired
alias geometry and is not a measurement of this representation, so it is not fabricated.

**HISTORICAL RECORDS (wake diagnostics 1) STAY READABLE UNDER THEIR OWN MEANING.** They carry
`aggregate_probability_per_meta_action`, `joint_*` and `joint_vs_aggregate_disagree`: one
meta-action owned `k` cells, so `selected_joint_cell_abort_fraction` was what deterministic
evaluation did and `aggregate_p_abort_mean` was the total abort-column MASS, **NEVER** the
probability of a selected action. `_wake_diag_digest` keeps those HISTORICAL keys computed from
historical records only, and adds REPRESENTATION-NEUTRAL keys (`selected_abort_fraction`,
`p_abort_mean`, `p_plan_mean`, `p_engage_mean`, `entropy_raw_mean`, `entropy_normalized_mean`,
`n_valid_actions_mean`) that read each wake under its own representation
(`_wake_meta_probability`: the semantic leaf, or the historical aggregate mass). A population
mixing representations reports those neutral means as `None` with
`mixed_action_representations: true`; `fd_policy_sensitivity.png` plots the neutral keys.
The record also carries the actor-INPUT summary only
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
not seed-stable labels ([runtime §2](runtime.md#2-episode-setup-stage-0)).

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

**WHAT IS EXPLICITLY NOT IN THIS TASK** (PR #52's scope, kept as recorded; the action
representation itself changed later, 2026-09-16). Target destruction stays DETERMINISTIC at
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
artifacts are episode-outcome schema v2 and carry NO `wake_decisions`**; the R1 record is in
[measurement history](../history/measurements.md#2-measurement-records).

### 5.1 Training credit diagnostics

**`train_credit_diagnostics.jsonl` — TRAINING-ONLY, APPEND-ONLY, OBSERVATIONAL (2026-09-16).**
Schema `graph_train_credit_diagnostics`, `_CREDIT_DIAGNOSTICS_VERSION = 1`, named in
`run_config.json:/training/credit_diagnostics`. `train` truncates it at run start and, after
EVERY productive update, writes ONE row per transition that update trained on — every wake kind,
not only fuel-damage wakes, so the batch context behind normalization stays readable.

- **THE VALUES THE UPDATE USED, NEVER A RECONSTRUCTION.** `PPOUpdater.update` /
  `CTDEUpdater.update(…, credit_sink=…)` hand the trainer a `CreditReport` holding the SAME
  `AdvantageBatch` / `CTDEAdvantageBatch` object the update consumed
  ([policy and CTDE §4](policy_ctde.md#4-phase-b-ctde)); `graph_train._credit_rows` copies
  numbers out of it. No actor or critic forward, GAE pass, baseline computation, RNG draw or
  gradient is added, and turning the sink on or off leaves selected actions, stored log-probs,
  advantages, optimizer state and parameters identical (tested).
- **COMMON FIELDS:** `schema`, `schema_version`, `action_representation_id`, `training_mode`,
  `iteration`, `updates_completed_before`, `episode_seed`, `episode_index`,
  `batch_transition_ordinal`, `ego_id`, `tick`, `wake_kind`, `selected_meta_action(_name)`,
  nullable `selected_node`, `stored_log_prob`, `episode_reward`, `transition_reward`,
  `raw_advantage`, `normalized_advantage`, `batch_raw_advantage_mean`,
  `batch_raw_advantage_std`, `adv_norm_eps`, `gamma`, `batch_n_transitions`,
  `batch_n_episodes`, `batch_n_episodes_with_wakes`.
- **ACTOR-ONLY FIELDS:** `return`, `actor_only_episode_baseline`, `ego_chain_ordinal`
  (position in the ego's chain, the actor-only credit structure); `transition_reward` is the
  transition's realized reward (`null` if unset).
- **CTDE FIELDS:** `value_old` (`V_old`), `td_residual` (`delta_t` from the same GAE pass),
  `value_target`, `gae_lambda`, `episode_decision_ordinal` (position in the episode's global
  decision sequence); `transition_reward` is the `r_t` the GAE pass consumed, and
  `raw_advantage` is the GAE advantage.
- **SCHEMA RULE: every key is present on every row; a key the row's training mode does not define
  is `null`** ("not defined for this mode"), never `0`.
- **`measurement_join` IS MEASUREMENT ONLY.** `cell`, `condition`, `severity`,
  `fd_selected_ego_id`, `fd_event_tick`, `is_fd_selected_ego` and `joined` come from a
  TRAINER-SIDE map keyed by `(episode_index, seed)` (`_credit_measurement_tags`), filled from the
  episode outcome and read only by `_persist_credit_diagnostics` AFTER the update. No such tag
  enters `GraphObservation`, `Transition`, `CentralGraphObservation`, a record or batch, the
  mask, the reward, PPO / GAE or an optimizer (tested structurally and by AST).
- **FAIL LOUD.** `_persist_credit_diagnostics` raises `CreditDiagnosticsError` — the run stops —
  when a productive update hands over no report or several, when the rows do not cover exactly
  the update's transitions, or when the file cannot be written.
- **NO CONTROL PATH READS IT BACK.** Early stopping, evaluation scheduling, checkpointing, the
  reward, failure classification and action selection never reference it; `build_run_summary`
  only OBSERVES its schema as `run_summary.json:/observed_credit_diagnostics` (row count, schema
  versions, representation ids, training modes).
- **NOT MEASURED BY THIS LAYER.** It is engineering instrumentation; no credit or
  severity-separation result exists until a future authorized run records and a review reads it.

### 5.2 CTDE actor-gradient diagnostics

**`train_actor_gradient_diagnostics.jsonl` — OPT-IN, OFF BY DEFAULT, CTDE ONLY, TRAINING-ONLY,
APPEND-ONLY, OBSERVATIONAL.** Enabled by `TrainConfig.actor_gradient_diagnostics` /
`--actor-gradient-diagnostics`; `TrainConfig.validate` refuses it unless
`training_mode = ctde`. Schema `graph_train_actor_gradient_diagnostics`,
`_ACTOR_GRADIENT_DIAGNOSTICS_VERSION = 1`. `run_config.json:/training/actor_gradient_diagnostics`
always records `enabled`, the artifact name, schema and version, the scope
(`ctde_updater_epoch_0`), the gradient and group-loss definitions, the four group names and the
`separation_contrast` definition. When
it is off, no file is created and the update receives no group ids. When it is on, `train`
truncates the file at run start and writes ONE record per productive update.

- **WHAT IS MEASURED.** At PPO epoch 0 of `CTDEUpdater.update`, after the per-transition policy
  losses and `actor_loss` exist and BEFORE the real `backward()`, clipping and optimizer steps,
  the updater differentiates with `torch.autograd.grad` on the retained graph (writing no
  `.grad`): for each group `G`, `sum(policy_loss_i for i in G) / n_transitions` — the group's real
  share of the batch-mean clipped surrogate, **before the entropy term and never re-normalized by
  the group's own size** — plus the total surrogate (`policy_loss`) and the total actual actor
  loss (`policy_loss - entropy_coeff * entropy_mean`). The group gradients therefore sum to the
  total surrogate gradient, and each record reports that reconstruction error.
- **LOCAL SEVERITY-SEPARATION PRESSURE.** At the same point, when the batch holds at least one
  `immediate_fd_severe` AND one `immediate_fd_mild` transition, the updater forms the
  differentiable contrast
  `mean P(ABORT | immediate_fd_severe) - mean P(ABORT | immediate_fd_mild)` — the semantic global
  ABORT-leaf probability built by `graph_action._semantic_dist` from the SAME epoch-0 logits the
  losses used (no forward, no detached rollout diagnostic) — and its actor-parameter gradient
  `h`. The trainer passes only the pair of opaque ids `(positive, negative)`
  (`_actor_gradient_contrast_ids`); the updater attaches no meaning to them. For every reported
  policy-gradient component `g` — each primary group, derived `fd` and `non_fd`, the total
  policy-surrogate gradient and the total actual actor-loss gradient — the record carries
  `separation_pressure = -dot(h, g)` and `separation_alignment = cosine(-g, h)`.
  **Interpretation:** a raw gradient-descent step `-lr * g` changes the contrast by
  `lr * separation_pressure` to first order, so **positive** pressure / alignment means that raw
  gradient component locally pushes toward LARGER SEVERE-minus-MILD ABORT separation and
  **negative** means it pushes AGAINST it. It is a **local first-order raw-gradient diagnostic,
  not a prediction of the actual Adam parameter step** (Adam rescales per parameter, clipping and
  later epochs intervene). Pressure is additive over components; alignment is not. The record
  also carries the contrast value (`separation_contrast`) and `separation_contrast_grad_norm`;
  `h` itself is never persisted.
- **GROUPS ARE MEASUREMENT METADATA RESOLVED TRAINER-SIDE.** `graph_train._actor_gradient_group`
  classifies each transition from `Transition.wake_kind` and the same `credit_tags` join the
  credit rows use (§5.1): `immediate_fuel_damage` wakes of the joined FD-selected ego split into
  `immediate_fd_mild` / `immediate_fd_severe` by the joined severity; `post_fd_boundary` wakes are
  `post_fd`; every other wake is `ordinary`. `_actor_gradient_group_ids` turns them into OPAQUE
  integers in the batch's transition order, and ONLY those integers cross into
  `CTDEUpdater.update(…, gradient_group_ids=…, gradient_sink=…)`; `graph_ppo` never sees a group
  name, severity or ego tag. An immediate-FD wake with no join, a non-selected ego or a severity
  other than MILD / SEVERE fails loud.
- **OBSERVATIONAL GUARANTEES (tested).** Diagnostic on versus off leaves actor and critic
  parameters, both optimizer states, the torch and numpy RNG states, the module forward count, the
  final `.grad`, the clip-norm calls and the optimizer step count identical; any labelling of the
  ids gives the identical update, advantages, evaluated logits and credit rows. No tag enters an
  actor or critic observation, GAE / advantages, a PPO loss, the reward, action selection or an
  optimizer.
- **RECORD FIELDS.** `schema`, `schema_version`, `action_representation_id`, `training_mode`,
  `iteration`, `updates_completed_before`, `epoch` (`0`), `gradient`, `group_loss`,
  `n_actor_parameters`, `batch_n_transitions`, `entropy_coeff`,
  `total_policy_surrogate_grad_norm`, `total_actor_loss_grad_norm`,
  `cosine_policy_surrogate_vs_actor_loss`; `separation_contrast_definition`,
  `separation_contrast`, `separation_contrast_grad_norm`,
  `total_policy_surrogate_separation_alignment` / `_pressure`,
  `total_actor_loss_separation_alignment` / `_pressure`; `groups.<group>` and `derived.fd` /
  `derived.non_fd` (`fd = immediate_fd_mild + immediate_fd_severe`,
  `non_fd = post_fd + ordinary`), each with `n_transitions`, `batch_fraction`, `grad_norm`,
  `cosine_vs_total`, `projection_on_total` (signed projection onto the total surrogate
  direction), `separation_alignment` and `separation_pressure`; `fd_grad_norm`, `non_fd_grad_norm`, `cosine_fd_vs_non_fd`,
  `projection_non_fd_on_fd` (signed projection of the non-FD gradient onto the FD direction);
  `reconstruction_error_norm`, `reconstruction_relative_error`. No full gradient vector is
  persisted.
- **UNDEFINED IS `null`, NEVER `0`.** An empty group has `n_transitions = 0` and `grad_norm = 0.0`
  with `null` cosine and projection; a cosine is `null` when either vector has zero norm, a
  projection when its reference direction has zero norm; `cosine_fd_vs_non_fd` and
  `projection_non_fd_on_fd` are `null` unless both derived groups are non-empty. **If either the
  MILD or the SEVERE immediate-FD group is absent, every contrast-dependent field is `null`**
  (the decomposition is still written normally); an empty component's separation fields are
  `null` too, and an alignment is `null` when `g` or `h` has zero norm. Records are
  written with `allow_nan = False`. The vector arithmetic is elementwise numpy only, because
  BLAS-backed numpy calls can abort next to torch on the local Windows stack.
- **FAIL LOUD.** `_persist_actor_gradient_diagnostics` raises `ActorGradientDiagnosticsError` —
  the run stops — when a productive update hands over no report or several, when the record does
  not cover the update's transitions, when the ids re-derived from the report's own batch do not
  match the ids the update used, when the report's contrast ids are not the trainer's, or when a
  contrast is present without both groups (or absent with both), or when the file cannot be
  written. `CTDEUpdater.update` raises `ValueError` when only one of `gradient_group_ids` /
  `gradient_sink` is given, the id count differs from the batch, or `gradient_contrast_ids` is
  given without group ids or is not two distinct ids.
- **COST AND SCOPE.** Up to seven extra `autograd.grad` passes per productive update, at epoch 0
  only (one per non-empty group, the two totals and the contrast), plus re-applying the semantic
  construction to the immediate-FD transitions' existing logits, over the retained epoch-0 graph;
  nothing on actor-only runs. Epochs 1+ are not decomposed, and version 1 has no
  action-conditioned subgroups.
- **NO CONTROL PATH READS IT BACK, AND IT MEASURES NOTHING BY ITSELF.** No stopping, evaluation,
  checkpoint, reward or selection path references it. It is engineering instrumentation: no
  gradient-pressure, separation-pressure or cancellation result exists until a future authorized
  run records it and a
  review reads it.

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
  episode-outcome schema v2 and no `wake_decisions`. Every preserved GENERALIZED-V2 run carries
  episode-outcome v3 / wake diagnostics 1 — the historical node-indexed action representation
  (`wake_action_representations_observed` = `legacy_node_indexed_joint_k_x_3`) — and no
  `train_credit_diagnostics.jsonl`. Episode-outcome v4 / wake diagnostics 2 is the semantic
  representation; never read a probability of one under the other's meaning. Episode-outcome v5 /
  wake diagnostics 3 is the same semantic representation decided on the two-column actor agent
  row (`actor_observation_id`); a v4 record's actor had no `mission_fuel_slack_norm` input.
- **Sharded evidence streams**: an evidence commit may split `episode_outcomes.jsonl` into
  line-aligned shards; reconstruct by concatenating the shards in the order given by
  `episode_outcomes.index.json` and check the SHA-256 against `artifact_sha256.txt` before
  reading.
- **External artifacts named but not committed** (checkpoints, benchmark manifests) are
  identified by path and SHA-256 only; an unverified external file is not evidence of its own
  contents.

## 7. Code routing

| Task | Files and symbols | Contract |
|---|---|---|
| change a figure | `rl/training/graph_train.py`: `plot_training`, `plot_training_subprocess`, `_plot_training_performance`, `_plot_policy_diagnostics`, `_plot_measurement_health`, `_PLOT_FILENAMES`, `_PLOT_X_LABEL` | §2 |
| change or read the optional FD-policy-sensitivity figure | `graph_train.py`: `_PLOT_FD_SENSITIVITY`, `_PLOT_OPTIONAL_FILENAMES`, `_fd_sensitivity_plot_data`, `_plot_fd_policy_sensitivity` | §5 |
| capture per-attempt visual artifacts | `graph_train.py`: `TrainConfig.visual_artifacts`, `_AttemptIdentity`, `_AttemptArtifacts` (`sync_recordings`, `finalize`), `_VisualArtifactError`, `_recording_kwargs`, `_artifact_kwargs` | §1 |
| read what an episode did, per successful attempt | `graph_train.py`: `_episode_outcome_record`, `_append_episode_outcome_record`, `_severity_response_from_outcomes`; `episode_outcomes.jsonl`; `run_summary.json:/severity_response` | §3 |
| record or read per-wake actor diagnostics | `rl/action/graph_action.py`: `summarize_decision`, `_semantic_dist`, `ACTION_REPRESENTATION_ID`; `rl/training/graph_tick_loop.py`: `WAKE_KINDS`, `_decision_record`, `_node_ownership`, `Transition.wake_kind` / `.decision`; `rl/observation/graph_builder.py`: `MissionSlackEstimate.as_record`, `ACTOR_OBSERVATION_ID`; `graph_train.py`: `_EPISODE_OUTCOME_VERSION`, `_WAKE_DIAGNOSTICS_VERSION`, `LEGACY_ACTION_REPRESENTATION_LABEL`, `_wake_action_representation`, `_wake_meta_probability`, `_wake_decision_records`, `_wake_diag_digest`, `_fd_policy_sensitivity_from_outcomes`, `_observed_artifact_schema` | §5 |
| record or read training credit diagnostics | `graph_train.py`: `_CREDIT_DIAGNOSTICS_FILENAME`, `_CREDIT_DIAGNOSTICS_SCHEMA`, `_CREDIT_DIAGNOSTICS_VERSION`, `CreditDiagnosticsError`, `_credit_measurement_tags`, `_credit_rows`, `_persist_credit_diagnostics`, `_observed_credit_diagnostics`; `rl/training/graph_ppo.py`: `CreditReport`; tests `tests/test_graph_semantic_action_credit.py` | §5.1 |
| record or read CTDE actor-gradient diagnostics | `graph_train.py`: `TrainConfig.actor_gradient_diagnostics`, `_ACTOR_GRADIENT_DIAGNOSTICS_FILENAME`, `_ACTOR_GRADIENT_DIAGNOSTICS_SCHEMA`, `_ACTOR_GRADIENT_DIAGNOSTICS_VERSION`, `_ACTOR_GRADIENT_GROUPS`, `ActorGradientDiagnosticsError`, `_actor_gradient_group`, `_actor_gradient_group_ids`, `_ACTOR_GRADIENT_CONTRAST`, `_actor_gradient_contrast_ids`, `_actor_gradient_record`, `_persist_actor_gradient_diagnostics`; `rl/training/graph_ppo.py`: `ActorGradientReport`, `GradientSink`, `_flat_actor_grad`; tests `tests/test_graph_ctde_actor_gradient_diagnostics.py` | §5.2 |
| select the final evaluation round (never `eval_records[-1]`) | `graph_train.py`: `_FINAL_EVAL_IDENTITY_FIELDS`, `_final_eval_identity`, `_select_final_eval_record`, `_select_final_matched_round`, `_round_identity`; `run_summary.json:/final_eval_selection` | §5 |
| persist or aggregate generalized per-episode data | `graph_train.py`: `_episode_outcome_record`, `_reward_breakdown_record`, `_failure_record`, `_EMPTY_BENCHMARK_KEYS`, `_generalized_summary`, `_construction_record`, `seed_bands`, `write_run_config` | §4; known label defect in [§6.1](#61-known-summary-label-defect-run_summaryjsongeneralizedcardinality_sampler) |
| read why or how a run stopped | `train_records.jsonl:/early_stopping_check`; `run_summary.json:/early_stopping`; `graph_train.py`: `_early_stopping_summary`, `TERMINATION_REASONS` | [training and benchmarks §7](training_benchmarks.md#7-early-stopping) |

A row that changes what a scientific artifact records, or how a summary is derived from it, is a
research-validity change ([`cc_review.md` §4](../workflows/cc_review.md#4-risk-and-verification)).
