"""graph_train.py -- the outer PPO Trainer (Phase A: actor-only). THE training loop.

This is the last piece of Phase A. Every component it drives is already built and
locked (CLAUDE.md sections 5 and 7); this module contributes NO pipeline logic of its
own -- it is the loop that wraps them into an actual training run:

    for iteration:
        for episode:
            generate -> setup_episode -> run_episode -> compute_episode_reward
                     -> PPOBuffer.add(EpisodeRecord.from_trajectory(...))
        diag = PPOUpdater.update(buffer);  buffer.clear()
        [log a scalar record | periodically evaluate | periodically checkpoint]

The per-episode body is the SAME skeleton as the diagnostic harness
(``graph_rollout.run_rollout``) -- one generator, one policy, per-episode reseed, env
closed in a ``finally``, a failed episode logged and skipped rather than aborting the
run. The ONLY additions are the buffer, the update, evaluation, and checkpointing.

RELATION TO graph_rollout
-------------------------
``graph_rollout`` is diagnostics (random weights, no learning); this module LEARNS.
They deliberately do not share code: the rollout is a locked, reviewed artifact and
factoring a common episode body out of it would be an edit to a locked file for no
functional gain. The duplication is ~30 lines and is called out here so it stays
visible.

WHY TRAINING ROLLOUTS ARE STOCHASTIC AND EVAL IS DETERMINISTIC
--------------------------------------------------------------
Training episodes always sample (``deterministic=False``): PPO's ratio
``exp(log_prob_new - log_prob_old)`` is only meaningful if the stored ``log_prob``
came from the distribution the action was actually drawn from. An argmax rollout
would put all mass on one cell, kill exploration, and make the surrogate degenerate.
So the trainer does NOT expose a "deterministic training" knob. Evaluation is the
mirror image: ``deterministic=True`` removes sampling noise so the learning curve
measures the policy, not the dice.

SEEDING SCHEDULE (this module owns it -- the two bands are DISJOINT BY VALIDATION)
---------------------------------------------------------------------------------
  * TRAINING: global episode index ``g = iteration * episodes_per_iteration + j``;
    episode seed ``base_seed + g``. This continues the rollout convention -- given the
    policy weights at that point, an episode is a pure function of its seed.
    GENERALIZED-V1 Task 5C GENERALIZES ``g`` WITHOUT CHANGING IT: under EITHER
    generalized design an ordinary failure is REPLACED rather than losing its slot, so
    ``g`` becomes the run-wide ATTEMPT ordinal -- monotone over the
    whole run, advanced by every attempt, successful or failed
    (:func:`train_attempt_seed`). When every slot is attempted exactly once, which is
    the whole of the fixed-cell policy, the two expressions are IDENTICAL and the
    historical loop still calls :func:`train_seed`. A failed seed is SPENT either way:
    nothing is ever retried at the same seed.
  * EVAL: a FIXED, disjoint band -- eval episode ``e`` always uses
    ``eval_base_seed + e``, the SAME E seeds on EVERY eval round. Evaluating the same
    held-out scenarios each round is what isolates policy improvement from scenario
    variance; a fresh eval sample each round would make the curve mostly noise.
    :meth:`TrainConfig.validate` REFUSES a config whose training band would reach into
    the eval band, so "held-out" is enforced, not hoped for. The band it checks is the
    MAXIMUM POSSIBLE attempt band (:attr:`TrainConfig.max_training_attempts`), which
    under replacement is wider than the successful-episode quota and on the fixed-cell
    path is exactly ``total_episodes``.
  * BOTH bands reseed global ``random`` AND torch at the head of every episode. That is
    what makes eval observationally pure with respect to training: the RNG state
    entering any episode depends only on that episode's seed, never on how many
    episodes, eval rounds, or updates ran before it. Eval additionally performs no
    update and holds no buffer, so it cannot touch the weights. Proven empirically in
    ``_selftest`` TEST 2 (train records byte-identical with eval on vs off).

EVAL EPISODE TAGS: ONE NAMESPACE PER ROUND (the seeds stay fixed)
-----------------------------------------------------------------
Eval scenarios are generated under a disjoint episode TAG namespace
(:func:`eval_episode_tag`, based at ``_EVAL_EPISODE_TAG_BASE``) so they can never
overwrite a training scenario artifact. Each eval ROUND additionally gets its own
sub-namespace, ``round_ordinal`` strides of ``_EVAL_ROUND_TAG_STRIDE`` above the base:
the ``pre_update`` round is ordinal 0 and every later ``post_update`` round takes the
next ordinal. Previously every round reused one fixed tag per episode index, which was
described here as "idempotent, not accumulating" -- and it was not: the rounds share
the seed band but NOT the policy, so round 2 silently overwrote the scenario JSON of
round 1 and a finished run could no longer show which world any but the last round had
actually run on.

The tag is NAMING ONLY. ``ScenarioGenerator.generate`` consumes ``episode`` after every
rng draw, in one step that sets ``scenario["name"]`` and the output filename; it is not
seed-derived and enters neither the geometry, the policy input, the action sampling nor
the reward. The held-out band is therefore untouched: eval episode ``e`` runs seed
``eval_base_seed + e`` on EVERY round, exactly as before -- only the file it is written
to now says which round wrote it.

PER-EPISODE OBSERVABILITY (what a run prints while it is running)
-----------------------------------------------------------------
Every successful attempt -- training and evaluation alike -- prints one ``OK`` block
immediately on return, before the next attempt starts (:func:`_format_episode_block`),
mirroring the pre-existing ``FAILED`` line so the two are never confused. The block
names the phase, the indices and the exact seed, the reward / wakes / ending / ticks /
dead / elapsed time, and the episode's TARGET ROSTER by BLADE ``name`` -- known and
hidden, listed against the subsets that were confirmed killed. Ids never appear: a
uuid tells a reader nothing, and generated target ids are not even seed-stable.

Confirmation counts are UNIQUE OVER TARGET ID. ``GraphPlanExecutor.done`` holds
``(ego_id, target_id)`` pairs, so ``EpisodeResult.confirmed_kills`` counts CONFIRMATIONS
and can exceed the number of targets in the world when two egos confirm the same kill.
That is correct for what it measures and is left untouched; this module simply stops
aggregating it. ``targets_confirmed_unique_mean`` / ``eval_targets_confirmed_unique_mean``
are the authoritative aggregates, ``target_confirmation_count_semantics`` states the
convention in the record itself, and ``kills_mean`` / ``eval_kills_mean`` survive as
compatibility aliases fed from the same corrected number.

The authoritative count is ``len(_unique_confirmed_target_ids(executor.done))`` and
NOTHING ELSE. It is never derived from how many targets the roster managed to name --
that dependency is what let a degraded roster report an episode with real confirmations
as a successful ``0/0``. The roster is required measurement structure, not a
best-effort label source: a structural failure (no beliefs, malformed task lists, t=0
beliefs that disagree, a missing world snapshot, or a confirmed target the roster does
not contain) raises :class:`EpisodeRosterError` -- a :class:`MeasurementIntegrityError`,
so it ABORTS the run rather than being accounted as a skipped episode -- while an
unresolvable NAME degrades to ``<unnamed target>`` and changes no id and no count.

WHICH TARGETS EXIST vs WHICH TARGETS WERE ALLOCATED
----------------------------------------------------
The roster's world comes from ``EpisodeContext.known_target_ids`` /
``executed_target_ids``: RAW snapshots ``setup_episode`` takes before either solve. It
must never come from ``ctx.oracle_tasks`` or from the beliefs, which are ALLOCATED-ONLY
by ``solve_and_normalize``'s contract and therefore omit any target the solver did not
select -- targets that are nonetheless in the world, sensible, attackable and
confirmable. Reading an allocation as an inventory is what made the long baseline
scientifically inconclusive: 143 of 800 training attempts were destroyed by a roster that
under-counted its own world, and the fault was booked as ordinary episode attrition.
``oracle_tasks`` / ``oracle_solution`` are unchanged and remain what the reward's oracle
denominator reads -- that is a question about allocation, and it was always right.

THE DIFFICULTY FACTOR (FD-BASELINE-v1) -- ONE FACTOR, MEASURED IN PAIRS
------------------------------------------------------------------------
The scenario cell is unchanged; what this module adds is the seeded, ego-local, one-shot
fuel-damage event of ``graph_fuel_damage`` plus the reward coefficient that gives it
teeth. Three consequences live here:

  * TRAINING draws the condition from the episode seed
    (``fuel_damage_mode = seeded_mixture``, ``P(damaged) = 0.5``), so a batch contains
    both conditions and every record reports its clean/damaged split next to the
    per-condition mean.
  * EVALUATION runs MATCHED PAIRS. Each held-out seed is attempted twice, forced clean
    and forced damaged, on the SAME seed -- therefore the same generated world, the same
    ``A_init`` and the same hidden geometry. The paired delta is averaged over pairs whose
    BOTH members completed and is reported with that pair denominator; an unpaired
    clean-vs-damaged comparison across different seeds would be confounded by scenario
    variance and is not computed anywhere.
  * THE REWARD FORMULA IS UNTOUCHED. ``compute_episode_reward`` is simply called with an
    explicit ``RewardConfig(aircraft_penalty_coeff=...)`` instead of falling through to
    the module default of ``0.0``, and the resolved coefficient is recorded in
    ``run_config.json`` and in every training record.

A damaged episode with no valid strict fuel window fails at the ``setup`` stage and is
skipped and accounted like any other -- never silently downgraded to a clean episode,
which would move the population a per-condition statistic is reported over.

SCENARIO SOURCE (the offline construction path: B1 generation + B3 setup seam)
-----------------------------------------------------------------------------
Episodes are built from an EXPLICIT reference cell -- ``num_agents``, ``n_known``,
``n_hidden`` and the requested geometry -- not from a ratio applied to a target count.
``build_variation_config`` is the only place that turns the config into a generator
request, and it asks for a KNOWN-ONLY world: exactly ``n_known`` targets, Layer-1
discovery-chain relocation OFF, and the geometry declared STRICT so the generator raises
instead of quietly weakening it.

The hidden half is built AFTER the known-only solve, inside ``setup_episode``'s
construction path (B3): solve A_init -> place route-relative hidden targets -> patch the
scenario JSON -> reload. This module therefore hands setup the ``n_hidden`` count and a
fresh per-episode ``random.Random(seed)``, and the world an episode really runs on holds
``n_known + n_hidden`` targets (:attr:`TrainConfig.n_targets_emitted`). ``split_tasks``
is not called at all on this path. The pre-B1 split surface (``partial_ratio``,
``num_red_airbases``, ``derived_split``, ``split_preview``) is retained and still tested,
but the construction path does not consult it.

RUN ARTIFACTS (what makes a run auditable after the fact)
---------------------------------------------------------
A run directory is the record; nothing about a run should have to be reconstructed from
a console scrollback:

  * ``run_config.json``        -- the resolved config PLUS a ``provenance`` block: code
                                 SHA and dirty state, the exact argv, interpreter and
                                 platform, targeted package versions/paths, the BONMIN
                                 executable and a bounded version probe, both seed bands
                                 as half-open ranges with their formulas, and the
                                 exact-cardinality policy id. Collected FIRST -- before
                                 the run directory exists, let alone the engine, the
                                 policy or the solver -- and INCOMPLETE Git provenance
                                 refuses the run outright.
  * ``train_records.jsonl``    -- one scalar record per iteration.
  * ``eval_records.jsonl``     -- one record per eval round, the first being the
                                 ``pre_update`` measurement of the initial policy.
  * ``episode_failures.jsonl`` -- append-only, flushed per record: every failed attempt
                                 with its phase, seed, pipeline stage and traceback.
  * ``train_credit_diagnostics.jsonl`` -- TRAINING-ONLY, append-only: one row per
                                 transition of every productive update, carrying the
                                 credit values that update ALREADY computed
                                 (``_credit_rows``). Observational; read back by nothing
                                 but the run summary's schema observation.
  * ``train_actor_gradient_diagnostics.jsonl`` -- OPT-IN (``--actor-gradient-diagnostics``),
                                 CTDE only, append-only: one record per productive
                                 update decomposing the epoch-0 policy-surrogate
                                 gradient by measurement group
                                 (``_actor_gradient_record``). Observational; read back
                                 by nothing.
  * ``run_summary.json``       -- derived from the three jsonl files at completion.
  * ``plots/``                 -- the three figures, derived from the jsonl files alone:
                                 ``training_performance.png`` (train reward, held-out
                                 clean vs damaged, matched-pair delta),
                                 ``policy_diagnostics.png`` (meta-action mix, entropy)
                                 and ``measurement_health.png`` (the denominators).
  * ``scenarios/``             -- the generated scenario JSON of every attempt.

JSON PRESETS (``--config <path>``)
----------------------------------
A run's shape can be declared in a JSON file instead of a command line, which is what
makes a bounded probe reproducible from the repository rather than from someone's shell
history. The file names :class:`TrainConfig` FIELDS directly (with the nested PPO knobs
under ``"ppo"``); ``TrainConfig`` stays the single source of truth, unknown keys are
refused, and an EXPLICIT command-line flag still wins over the file. The resolved config
and the preset it came from are both recorded in ``run_config.json``
(``/config_source``), so a finished run states which preset produced it. The repository
owns one preset today: ``configs/graph_train/final_cell_probe.json``, the bounded short
final-cell probe.

VISUAL ARTIFACTS (opt-in: ``--visual-artifacts`` / ``TrainConfig.visual_artifacts``)
------------------------------------------------------------------------------------
OFF by default, and off is byte-unchanged. When enabled, every scheduled ``pre_update`` /
``train`` / ``post_update`` attempt preserves an inspection bundle under
``<run_dir>/visual_artifacts/<attempt>/``: the exact generated KNOWN-ONLY scenario, the
AUTHORITATIVE executed t=0 scenario (``Game.export_scenario()`` off the env-2 game, taken
before the fuel-damage controller exists and therefore before the first tick), the BLADE
playback recording, and an ``artifact_manifest.json`` that states the attempt's phase,
iteration, update count, ordinals, exact seed, scheduled condition and scenario tag
explicitly. That is what makes a finished probe openable in PyCharm and in the BLADE
client instead of only readable as numbers.

It is OBSERVATION, not measurement: nothing captured is read back, no seed / tag /
scenario name / RNG draw changes, recording is armed only through the locked
``setup_episode(recording_export_path=...)`` seam and driven only by ``run_episode``, and
a capture failure raises ``_VisualArtifactError`` -- infrastructure, aborting the run --
rather than entering the scientific failure ledger. See section 3e.

EXACT-CARDINALITY FAILURES: SKIP AND ACCOUNT (``skip_and_account_v1``)
---------------------------------------------------------------------
B2's locked contract places one hidden target per non-empty ego route and B3 requires
``len(placements) == n_hidden`` exactly, so a solve that leaves an ego idle FAILS the
episode. That is a property of the cell, not a bug, and it is measurable: on the default
cell, seeds 2 and 8 of 0..11 produced only 2 usable routes.

The approved policy is to let those seeds fail and ACCOUNT for them. Every scheduled
seed is attempted at most once; a failed seed is never retried, never replaced by
another seed, and never shifts the band, so the training and held-out bands stay exactly
what the config declares. Failures never enter a PPO buffer or a reward aggregate, and
every one is recorded once in the ledger. Consequently every reward statistic in this
module describes the exact-cardinality-FEASIBLE SUBSET, and is always reported next to
``n_attempted`` / ``n_successful`` / ``n_failed`` -- a held-out mean without its
denominator is not a result.

Two things that are NOT failures, and are never counted as such: a successful zero-wake
episode (a real episode in which nothing was sensed) and a zero-wake iteration.
Conversely, an all-failed batch or eval round reports its reward as ``null``, never as
``0.0`` -- the reward is oracle-normalized regret, so 0 is the OPTIMUM, and rendering a
total data loss as a perfect score is the specific research bug this avoids.

SCOPE
-----
Checkpoints are SAVED here; loading / resuming a run is deliberately NOT implemented
(a separate, deferred task). This module is a LEAF harness: nothing pure imports it,
which is why it may import BLADE (like ``graph_rollout``) without touching the
import-purity guard.

PLOTTING RUNS IN A CHILD PROCESS (an environment constraint, not a preference)
------------------------------------------------------------------------------
On this Windows/Anaconda stack torch and matplotlib CANNOT share a process: each
links its own OpenMP runtime, and the second one to initialize aborts the process
outright ("OMP: Error #15 ... libiomp5md.dll already initialized" -> ``Fatal Python
error: Aborted``). This was verified in BOTH environments and in BOTH import orders,
so it is a property of the machine, not of this module. A hard abort cannot be caught
by ``try/except``.

Consequently:
  * :func:`plot_training` is the real renderer and imports matplotlib lazily. It is
    safe to call from any process that has NOT loaded torch.
  * :func:`plot_training_subprocess` is what a TORCH process (a finished training run,
    the selftest, a test) must call. It re-invokes this module's ``--plot`` CLI in a
    child process that only reads jsonl and draws the PNGs -- no tensor math at all --
    with ``KMP_DUPLICATE_LIB_OK=TRUE`` set for that child alone. The duplicate-OpenMP
    tolerance is therefore confined to a throwaway, numerics-free process; the training
    process itself never gets a second OpenMP runtime.
Either way, training NEVER depends on matplotlib: a missing matplotlib (or a failed
child) prints one notice and returns an EMPTY LIST of figures. Both functions return the
list of figure paths they wrote, because there are three of them (see PLOTS below).

Windows-safe: pathlib paths, ASCII-only console output (cp1255 console).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ...solvers.match_aou_backend import (
    DEFAULT_MATCH_AOU_BACKEND,
    MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
    MATCH_AOU_BACKEND_P1_MILP_V1,
    MATCH_AOU_BACKENDS,
    MatchAouBackendError,
    resolve_match_aou_backend,
)
from .graph_episode_setup import (
    RouteRelativePopulationRecorder,
    setup_episode,
    DETECTION_KM,
    MAX_SIM_TICKS,
)
from .graph_fuel_damage import (
    CONDITION_CLEAN,
    CONDITION_DAMAGED,
    CONDITIONS,
    SEVERITIES,
    SEVERITY_MILD,
    SEVERITY_SEVERE,
    FuelDamageIntegrityError,
    FuelDamageMode,
    FuelDamageOutcome,
    FuelDamageParameters,
    build_fuel_damage_controller,
    resolve_condition,
    resolve_severity,
)
from .graph_ppo import (
    CTDEBuffer,
    CTDEConfig,
    CTDEEpisodeRecord,
    CTDEUpdater,
    ActorGradientReport,
    CreditReport,
    EpisodeRecord,
    PPOBuffer,
    PPOConfig,
    PPOUpdater,
    build_central_critic,
)
from .graph_hidden_placement import geometric_fingerprint
from .graph_generalized import (
    CARDINALITY_RNG_DOMAIN,
    CARDINALITY_SAMPLER_POLICY,
    GENERALIZED_AGENT_COUNTS,
    BENCHMARK_CELLS,
    BENCHMARK_DELTAS,
    BENCHMARK_GROUP_SIZE,
    BENCHMARK_STRATA,
    BENCHMARK_STRATUM_KEYS,
    EPISODE_DESIGN_FIXED_CELL_V1,
    EPISODE_DESIGN_GENERALIZED_V1,
    EPISODE_DESIGNS,
    TARGET_DESTRUCTION_PROBABILITY,
    BenchmarkIdentityError,
    BenchmarkManifest,
    EPISODE_DESIGN_GENERALIZED_V2,
    GENERALIZED_V2_AGENT_COUNTS,
    GENERALIZED_V2_KNOWN_OFFSETS,
    GENERALIZED_V2_REQUIRED_BACKEND,
    HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
    PRE_SOLVE_CARDINALITY_POLICY_V2,
    V2_CARDINALITY_RNG_DOMAIN,
    V2_HIDDEN_LOAD_RNG_DOMAIN,
    PreSolveCardinality,
    RouteRelativeHiddenLoad,
    EpisodeCardinality,
    EpisodeDesign,
    WorldIdentity,
    cardinality_sampler_record,
    certificate_fingerprint,
    fixed_cell_cardinality,
    load_benchmark_manifest,
    manifest_seed_overlap,
    require_matched_group_identity,
    require_world_matches_manifest,
    resolve_episode_design,
    CARDINALITY_SOURCE_V2_PRE_SOLVE,
    CARDINALITY_SOURCE_V2_BENCHMARK_PRE_SOLVE,
    V2_BENCHMARK_BASE_CELL_KEYS,
    V2_BENCHMARK_PROFILES,
    BenchmarkManifestError,
    V2BenchmarkManifest,
    V2BenchmarkWorld,
    V2WorldIdentity,
    canonical_digest,
    generalized_v2_cardinality_sampler_record,
    load_v2_benchmark_manifest,
    require_v2_matched_group_identity,
    require_v2_world_matches_manifest,
    resolved_v2_cardinality,
    sample_generalized_cardinality,
    sample_generalized_v2_pre_solve_cardinality,
    v2_allocation_fingerprint,
)
from .graph_reward import (
    ReferenceIntegrityError,
    RewardConfig,
    compute_episode_reward,
    reference_fault_aborts,
)
from .graph_tick_loop import (
    WAKE_KIND_IMMEDIATE_FD,
    WAKE_KIND_ORDINARY,
    WAKE_KIND_POST_FD_BOUNDARY,
    WAKE_KINDS,
    build_policy,
    run_episode,
)
from ..observation.central_graph_builder import CentralStateRecorder
from ..action.graph_action import ACTION_REPRESENTATION_ID, MetaAction
from ...models import StepKind
from ...utils.blade_utils.scenario_generator import (
    ScenarioGenerator,
    VariationConfig,
)

# The base template every generated variation derives from -- the SAME scenario the
# rollout harness and the sibling selftests use. This file lives at
# src/match_aou/rl/training/, so parents[4] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BASE_SCENARIO = _REPO_ROOT / "data" / "scenarios" / "strike_training_4v5.json"

# The three meta-action columns, in enum order (0..2). Fixed key set for the counts.
_META_NAMES = [MetaAction(i).name for i in range(len(MetaAction))]
# The two meta-action names the FD-policy diagnostics report on by name. Bound to
# the enum rather than typed as string literals, so a rename cannot silently turn a
# reported rate into a lookup miss.
_PLAN_NAME = MetaAction.PLAN_COMPLIANCE.name
_ABORT_NAME = MetaAction.SELF_PRESERVATION_ABORT.name

# Eval scenarios are tagged from here up, so their filenames / scenario names can never
# collide with a training episode's global index g. Purely a NAMING offset: the tag is
# not seed-derived and does not enter scenario content.
_EVAL_EPISODE_TAG_BASE = 900_000

# Width of ONE eval round's tag sub-namespace (see `eval_episode_tag`). Round ordinal r
# owns [base + r*stride, base + (r+1)*stride), so no two rounds can name the same
# scenario file. `TrainConfig.validate` refuses an `eval_episodes` that would not fit.
_EVAL_ROUND_TAG_STRIDE = 1_000

# What a target-confirmation COUNT means in every record this module writes. Stated in
# the record rather than left to the reader because the executor's own `done` set is
# keyed `(ego_id, target_id)` -- a different, also-correct quantity that this module
# deliberately no longer aggregates.
_TARGET_CONFIRMATION_SEMANTICS = "unique_target_id"

# Shown instead of a target's BLADE name when the scenario cannot supply one. NEVER the
# target's id: ids are uuids, they are not seed-stable for generated targets, and a
# success block that printed one would be unreadable rather than merely incomplete.
_UNNAMED_TARGET = "<unnamed target>"

# Wall-clock fields -- excluded when two runs' records are compared for equality
# (see _selftest TEST 2: timing legitimately differs run to run).
_TIMING_KEYS = frozenset({
    "iteration_seconds", "episodes_seconds", "update_seconds", "eval_seconds",
})

# --- B4 auditability constants ------------------------------------------------
# The schema version of the `provenance` block in run_config.json. Bump it if a field
# is REMOVED or its meaning changes; adding a field does not require a bump.
_PROVENANCE_VERSION = 1

# The approved exact-cardinality failure policy, recorded verbatim in every run's
# provenance so an artifact states which policy produced it. SKIP AND ACCOUNT: every
# scheduled seed is attempted at most once, a failed seed is NEVER replaced, retried
# under another seed, or shifted, the seed bands never move, failures never enter a
# PPO buffer or a reward aggregate, and every failure is recorded once in the ledger.
_EXACT_CARDINALITY_POLICY = "skip_and_account_v1"

# The four pipeline stages one episode attempt can fail in, in execution order. An
# attempt is attributed to exactly one of them (see `EpisodeAttemptError`).
_PIPELINE_STAGES = ("generation", "setup", "run", "reward")

# The two evaluation stages. `pre_update` is the held-out measurement of the INITIAL
# policy (updates_completed == 0); `post_update` is every later round.
_EVAL_STAGE_PRE_UPDATE = "pre_update"
_EVAL_STAGE_POST_UPDATE = "post_update"

# Provenance sub-probes are BOUNDED: a hung `git` or `bonmin` must not stall a run
# that has not even built its policy yet.
_GIT_PROBE_TIMEOUT_S = 15.0
_BONMIN_PROBE_TIMEOUT_S = 10.0

# Third-party / project modules whose version + on-disk path are recorded. Deliberately
# a SHORT targeted list, not a `pip freeze`: these are the four whose identity can
# change a result (the engine fork in particular is vendored, so its PATH is the fact
# that matters, not a version string it does not carry).
_PROVENANCE_MODULES = ("torch", "gymnasium", "blade", "match_aou")

# --- FD-BASELINE-v1 constants -------------------------------------------------
# The two members of a matched evaluation pair, in the order they are attempted. Both
# members of a pair use the SAME held-out seed (so the same generator world and the same
# hidden-placement geometry) and differ ONLY in the fuel-damage condition -- that is what
# makes their reward difference attributable to the event rather than to the scenario.
_EVAL_PAIR_MEMBERS = (
    (CONDITION_CLEAN, FuelDamageMode.FORCED_CLEAN),
    (CONDITION_DAMAGED, FuelDamageMode.FORCED_DAMAGED),
)

# Eval scenario tags are allocated per GROUP MEMBER, not per seed: member m of held-out
# episode e takes tag slot `e * group_size + m` inside the round's namespace, so the
# members of one seed are written to distinct files and none overwrites another.
# `TrainConfig.validate` sizes the namespace against the group size the run will use.
_EVAL_PAIR_SIZE = len(_EVAL_PAIR_MEMBERS)

# --- FD-VARIABLE-SEVERITY-v1 constants ----------------------------------------
# The THREE members of a matched evaluation TRIAD, in the order they are attempted. All
# three use the SAME held-out seed -- hence the same generated world, the same solved
# A_init, the same hidden geometry and (for the two damaged members) the SAME selected
# ego -- and differ only in the fuel-damage event. A triad, rather than the legacy pair,
# is what lets "did the actor respond DIFFERENTLY to a survivable loss than to an
# unsurvivable one?" be asked within a single world instead of across worlds.
#
# A member is `(CELL, mode)`. The CELL is the label the member is reported under, and for
# a damaged member of a triad it IS the severity -- which is why the clean member reuses
# the existing `forced_clean` mode rather than needing a new one.
_EVAL_TRIAD_MEMBERS = (
    (CONDITION_CLEAN, FuelDamageMode.FORCED_CLEAN),
    (SEVERITY_MILD, FuelDamageMode.FORCED_MILD),
    (SEVERITY_SEVERE, FuelDamageMode.FORCED_SEVERE),
)

# The within-seed differences each design reports, as `(cell, reference_cell)` pairs.
# EVERY one of them is averaged over COMPLETE matched groups only -- a group with a
# failed member contributes to none of them, is never repaired with its surviving
# members, and is still visible in the attempt counts.
_EVAL_PAIR_DELTAS = ((CONDITION_DAMAGED, CONDITION_CLEAN),)
_EVAL_TRIAD_DELTAS = (
    (SEVERITY_MILD, CONDITION_CLEAN),
    (SEVERITY_SEVERE, CONDITION_CLEAN),
    (SEVERITY_SEVERE, SEVERITY_MILD),
)

# What a matched group is CALLED in a record, so a reader never has to count members.
_EVAL_GROUP_KIND_PAIR = "pair"
_EVAL_GROUP_KIND_TRIAD = "triad"

# The modes a TRAINING run may be configured with. The forced modes belong to an
# evaluation group member: setting one here would condition every training episode
# identically, which is a mixture of one.
_TRAINING_FUEL_DAMAGE_MODES = (
    FuelDamageMode.OFF, FuelDamageMode.SEEDED_MIXTURE, FuelDamageMode.SEEDED_VARIABLE,
)

# The durable per-SUCCESSFUL-ATTEMPT record stream (see `_episode_outcome_record`). One
# canonical file, deliberately not several overlapping ones: the aggregate per-iteration
# and per-round records cannot answer a distributional question ("how did the actor
# respond to MILD, episode by episode?"), and inventing a second stream per question is
# how two files start disagreeing. Failed attempts stay in `episode_failures.jsonl` --
# this stream never duplicates the ledger.
_EPISODE_OUTCOMES_FILENAME = "episode_outcomes.jsonl"
_EPISODE_OUTCOME_SCHEMA = "graph_train_episode_outcome"
# VERSION 2 adds the GENERALIZED-V1 blocks: the resolved episode design and its four
# policy ids, requested-vs-realized cardinality, the construction / FD-eligibility /
# post-FD-adaptation audits, the reward-reference decomposition, and (for a
# manifest-driven evaluation member) its benchmark stratum identity. Every added key is
# present on BOTH designs and states a real fact about the episode -- on the historical
# path the design is `fixed_cell_v1`, the reference policy is `static_t0_v1`, requested
# equals realized, and the structures the historical policies do not produce are `null`
# rather than a fabricated zero.
_EPISODE_OUTCOME_VERSION = 4
# VERSION 3 adds the PER-WAKE ACTOR DIAGNOSTICS (`wake_decisions`, versioned by
# `_WAKE_DIAGNOSTICS_VERSION`): for every recorded wake, why it happened, what the actor
# saw and what the masked distribution actually looked like. It exists because the
# R1 diagnostic replay had to reconstruct all of that OFFLINE from checkpoints, which is
# only possible while the checkpoints, the manifest and the exact code SHA all still
# exist together. It is REPORTING-ONLY and additive: a v2 record simply lacks the key,
# and every reader below treats an absent key as "not recorded" rather than as zero.
# VERSION 4 adds the top-level `action_representation_id` and carries wake diagnostics
# schema 2. The MEANING of the action a record describes changed (the semantic k + 2
# representation), so the version moves rather than hiding that change behind v3.
_WAKE_DIAGNOSTICS_VERSION = 2
# WAKE DIAGNOSTICS 1 described the retired node-indexed k x 3 joint distribution
# (`aggregate_probability_per_meta_action`, `joint_*`, `joint_vs_aggregate_disagree`).
# WAKE DIAGNOSTICS 2 describes the semantic k + 2 leaf distribution and names its
# representation on every wake (`action_representation_id`,
# `semantic_probability_per_meta_action`, `semantic_entropy_*`). The readers below keep
# BOTH readable and never read one schema's field under the other's meaning.

#: A READER label for historical wake records, which carry no representation id. It is
#: never written into any artifact.
LEGACY_ACTION_REPRESENTATION_LABEL = "legacy_node_indexed_joint_k_x_3"

# TRAINING CREDIT DIAGNOSTICS -- `train_credit_diagnostics.jsonl`. One row per transition
# of every productive update, built by `_credit_rows` from the `CreditReport` the updater
# handed its sink AFTER the update: the values the update already used, never a second
# forward, GAE pass or baseline. Schema rule: every key is present on every row; a key
# that the row's training mode does not define is `null` ("not defined for this mode"),
# never `0`. The `measurement_join` block is a TRAINER-SIDE join of measurement tags
# (cell / condition / severity / FD ego / FD tick) written after credit exists; no
# learning object carries them.
_CREDIT_DIAGNOSTICS_FILENAME = "train_credit_diagnostics.jsonl"
_CREDIT_DIAGNOSTICS_SCHEMA = "graph_train_credit_diagnostics"
_CREDIT_DIAGNOSTICS_VERSION = 1


class CreditDiagnosticsError(RuntimeError):
    """The credit-diagnostics artifact could not be produced or persisted.

    An INSTRUMENTATION / INFRASTRUCTURE integrity failure: an instrumented run that
    silently lost its credit rows would be scientifically incomplete, so the run stops.
    """

# ACTOR-GRADIENT DIAGNOSTICS -- `train_actor_gradient_diagnostics.jsonl`. OPT-IN, OFF by
# default, CTDE only. One record per productive update decomposing the EPOCH-0 PPO
# policy-surrogate gradient (before the entropy term) into four measurement groups. The
# groups are resolved HERE, trainer-side, from the same `credit_tags` join the credit rows
# use; `CTDEUpdater` receives only their opaque integer ids (the index into
# `_ACTOR_GRADIENT_GROUPS`) and never learns what a group means. Undefined cosines /
# projections are `null`, never `0`.
_ACTOR_GRADIENT_DIAGNOSTICS_FILENAME = "train_actor_gradient_diagnostics.jsonl"
_ACTOR_GRADIENT_DIAGNOSTICS_SCHEMA = "graph_train_actor_gradient_diagnostics"
_ACTOR_GRADIENT_DIAGNOSTICS_VERSION = 1
_ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_MILD = "immediate_fd_mild"
_ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_SEVERE = "immediate_fd_severe"
_ACTOR_GRADIENT_GROUP_POST_FD = "post_fd"
_ACTOR_GRADIENT_GROUP_ORDINARY = "ordinary"
_ACTOR_GRADIENT_GROUPS = (
    _ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_MILD,
    _ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_SEVERE,
    _ACTOR_GRADIENT_GROUP_POST_FD,
    _ACTOR_GRADIENT_GROUP_ORDINARY,
)
_ACTOR_GRADIENT_DERIVED = {
    "fd": (_ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_MILD,
           _ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_SEVERE),
    "non_fd": (_ACTOR_GRADIENT_GROUP_POST_FD, _ACTOR_GRADIENT_GROUP_ORDINARY),
}


class ActorGradientDiagnosticsError(RuntimeError):
    """The actor-gradient diagnostic could not be produced, verified or persisted.

    Like :class:`CreditDiagnosticsError`: an enabled diagnostic that silently dropped or
    misattributed a record would be scientifically incomplete, so the run stops.
    """

# Keys holding the full record lists inside a run summary. They are returned in-process
# but NOT persisted to run_summary.json -- the jsonl files are the record, and copying
# them into the summary would create a second, divergeable metric path.
_SUMMARY_RECORD_KEYS = (
    "train_records", "eval_records", "failure_records", "episode_outcome_records",
)

# --- VISUAL ARTIFACTS (opt-in, OFF by default) --------------------------------
# One directory per SELECTED attempt under `<run_dir>/visual_artifacts/`, holding the
# exact generated known-only scenario, the authoritative executed t=0 scenario, the BLADE
# playback recording, and a manifest stating the attempt's identity explicitly.
_VISUAL_ARTIFACTS_DIRNAME = "visual_artifacts"
_ARTIFACT_KNOWN_ONLY_SCENARIO = "known_only_scenario.json"
_ARTIFACT_EXECUTED_T0_SCENARIO = "executed_t0_scenario.json"
_ARTIFACT_MANIFEST = "artifact_manifest.json"

# The playback recorder writes `<scenario name> Recording <start> - <end>.jsonl` into the
# export path it was armed with. The manifest lists whatever it produced -- a plural, so a
# recorder that ever splits a long recording into chunks is recorded as chunks, not as a
# single file that silently lost its tail.
_ARTIFACT_RECORDING_GLOB = "*.jsonl"

_ARTIFACT_MANIFEST_SCHEMA = "final_cell_visual_artifacts"
_ARTIFACT_MANIFEST_VERSION = 1
_ARTIFACT_STATUS_INCOMPLETE = "incomplete"
_ARTIFACT_STATUS_COMPLETE = "complete"

# This module's own name for a scheduled TRAINING attempt. The other two artifact phases
# are the existing evaluation STAGE names, so a manifest's `phase` and an eval record's
# `evaluation_stage` are literally the same string.
_ARTIFACT_PHASE_TRAIN = "train"
_ARTIFACT_PHASES = (
    _EVAL_STAGE_PRE_UPDATE, _ARTIFACT_PHASE_TRAIN, _EVAL_STAGE_POST_UPDATE,
)
# The two EVALUATION phases, as a set apart from training. Held-out evaluation is
# deterministic on a frozen population while training is stochastic on a sampled one, so
# a rate whose denominator mixes them describes neither. Every scientific severity
# quantity below is taken over THIS tuple and never over `_ARTIFACT_PHASES`.
_EVAL_PHASES = (_EVAL_STAGE_PRE_UPDATE, _EVAL_STAGE_POST_UPDATE)

# --- PLOTS: one subdirectory, three semantically separate figures ---------------
# The legacy single `training_plot.png` dashboard is GONE. It put the stochastic
# training mean and the deterministic held-out mean on one axis, which invited reading
# them as one curve, and it pre-dated the matched-pair evaluation entirely -- so the
# held-out series it drew pooled the clean and damaged conditions and therefore answered
# no question about the difficulty factor. The three figures below separate what is
# being claimed: PERFORMANCE, policy DIAGNOSTICS, and the DENOMINATORS behind both.
_PLOTS_DIRNAME = "plots"
_PLOT_PERFORMANCE = "training_performance.png"
_PLOT_DIAGNOSTICS = "policy_diagnostics.png"
_PLOT_MEASUREMENT_HEALTH = "measurement_health.png"
_PLOT_FILENAMES = (_PLOT_PERFORMANCE, _PLOT_DIAGNOSTICS, _PLOT_MEASUREMENT_HEALTH)

# The FD-policy-sensitivity figure is OPTIONAL and is deliberately NOT in
# `_PLOT_FILENAMES`: that tuple is the REQUIRED set, and a shortfall against it is
# reported as "plots incomplete". This figure is drawn only from episode-outcome schema
# v3 `wake_decisions`, so a run directory written before that field existed legitimately
# has three figures and must not be reported as broken.
_PLOT_FD_SENSITIVITY = "fd_policy_sensitivity.png"
_PLOT_OPTIONAL_FILENAMES = (_PLOT_FD_SENSITIVITY,)

# Every figure shares ONE x-coordinate concept, stated on every figure so a reader never
# has to infer it: the policy state a measurement describes, counted in PPO updates
# COMPLETED BEFORE that measurement was taken. Training points therefore sit at
# `updates_completed_before` (the updates the policy that GENERATED those episodes had
# received) and eval points at `updates_completed` (0 for the pre-update round), which is
# what puts the untrained policy's training batch and its held-out measurement at the
# same origin. This is the honest placement, not the flattering one: it never credits a
# batch to an update that had not happened when the batch was collected.
_PLOT_X_LABEL = "PPO updates completed before the measurement"
_PLOT_X_SEMANTICS = (
    "x = PPO updates completed BEFORE the measurement (train: "
    "updates_completed_before; eval: updates_completed)"
)

# --- JSON PRESETS -------------------------------------------------------------
# A preset names TrainConfig FIELDS, not CLI flags: `TrainConfig` is the contract, and a
# second parallel naming scheme would be a second place for the two to drift apart.
# Nested PPO knobs live under this key, mirroring the dataclass.
_CONFIG_PPO_KEY = "ppo"

# The nested CTDE block, the sibling of `"ppo"`. Read only by a `ctde` run; a preset may
# still declare it under `actor_only` (it is simply unused), exactly as a preset may
# declare PPO knobs an iteration never reaches.
_CONFIG_CTDE_KEY = "ctde"

# --- THE TWO TRAINING MODES (Phase B) ----------------------------------------------
# `actor_only` is the DEFAULT and is the Phase-A reference path the approved long
# baseline was measured on: no central critic, no central observation, no value loss,
# and the episode-mean-baseline credit assignment of `compute_returns_and_advantages`.
# `ctde` adds a centralized value estimator during TRAINING only.
#
# These are two DISJOINT code paths, not one path with a coefficient. `actor_only` is
# never expressed as "ctde with value_coeff = 0": that would still build a critic, still
# capture central states and still replace the episode-mean baseline with a learned one,
# so it would not be the Phase-A path at all. Whichever mode a run selects, EXECUTION is
# identical and fully decentralized -- evaluation and inference are actor-only in both.
TRAINING_MODE_ACTOR_ONLY = "actor_only"
TRAINING_MODE_CTDE = "ctde"
TRAINING_MODES = (TRAINING_MODE_ACTOR_ONLY, TRAINING_MODE_CTDE)

# =============================================================================
# GENERALIZED-V1 Task 5C: WHAT `episodes_per_iteration` COUNTS
# =============================================================================
# TWO training ATTEMPT policies, selected by `episode_design` and by nothing else.
#
#   `scheduled_attempts_v1` (FIXED CELL, historical, unchanged) -- `episodes_per_iteration`
#       is a count of scheduled ATTEMPTS. Each is made exactly once, a failure is recorded
#       and the slot is simply lost, and the batch that reaches the updater is whatever
#       survived. Every approved measurement was taken under this policy; nothing about it
#       moves here.
#
#   `successful_quota_with_deterministic_replacement_v1` (GENERALIZED) --
#       `episodes_per_iteration` is a count of SUCCESSFUL episodes the batch must hold.
#       Ordinary attrition is REPLACED by the next deterministic attempt rather than
#       silently shrinking the batch: the generalized population is drawn from a sampler
#       whose worlds legitimately fail construction or FD certification, so a fixed
#       attempt count would make the PPO batch size a function of world attrition and the
#       learning curve of two arms incomparable. A failed attempt is still recorded ONCE,
#       still SPENDS its seed, and is never retried at that seed.
#
# The replacement is bounded: `generalized_max_attempts_per_iteration` is an explicit
# operator budget, and exhausting it ABORTS rather than updating on a partial batch.
TRAINING_ATTEMPT_POLICY_SCHEDULED = "scheduled_attempts_v1"
TRAINING_ATTEMPT_POLICY_QUOTA = "successful_quota_with_deterministic_replacement_v1"
TRAINING_ATTEMPT_POLICIES = (
    TRAINING_ATTEMPT_POLICY_SCHEDULED, TRAINING_ATTEMPT_POLICY_QUOTA,
)

# GENERALIZED-V1 EARLY STOPPING: one OPT-IN policy over TRAINING reward alone.
#
# `training_reward_plateau_v1` ends a run once its TRAINING reward has stopped
# improving. It is OFF by default -- a run that does not opt in is fixed-budget and
# behaves exactly as it always did -- and `validate` approves it for `generalized_v1`
# only.
#
# THE STOPPING SIGNAL IS `train_reward_mean` AND NOTHING ELSE. Not a benchmark or
# held-out reward, not a success or feasibility rate, not a PPO/CTDE diagnostic, not the
# critic, not a checkpoint, not any final-comparator result. That exclusion is the whole
# research-validity content of the feature: the frozen benchmark is the COMPARATOR two
# arms are judged by, so letting it decide when an arm stops training would let each arm
# pick its own stopping point on the very population the comparison is made over, and
# the measured difference would no longer be attributable to the training algorithm. The
# separation is MECHANICAL rather than a convention -- the decision is computed from the
# iteration's own training record, before any evaluation triggered at that same boundary
# can run (see `train`), and `_EarlyStoppingMonitor` is handed one number.
#
# ACTOR-ONLY AND CTDE SHARE THIS MECHANISM WITH NO MODE-SPECIFIC BRANCH. `training_mode`
# is not read anywhere in it. Two arms compared under this policy share
#     the same maximum budget + the same frozen stopping rule
#     + the same training-population contract,
# which is deliberately NOT "the same actual number of iterations": the actual count is
# an OUTCOME of the rule, and forcing the two to match would defeat the rule's purpose.
#
# IT DOES NOT ESTABLISH CONVERGENCE. A triggered stop records that the configured
# plateau rule fired -- never that a global optimum was reached, nor that the training
# reward has provably converged. Nothing here may be reported as a convergence claim.
EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU = "training_reward_plateau_v1"
EARLY_STOPPING_POLICIES = (EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU,)

# The ONE quantity the policy reads, named as a constant so "which metric stopped this
# run?" is a STATED fact in every record rather than something a reader infers. It is
# the same key `train_records.jsonl` already persists, and the monitor is handed the
# value read back OFF that record -- one metric path, not two.
EARLY_STOPPING_METRIC = "train_reward_mean"

# What a due check IS. The first one has no earlier best to improve on and only
# establishes one; every later one is a comparison against that best. They are named
# apart so `best_window_mean_before = null` is never read as a measured zero.
EARLY_STOPPING_CHECK_BASELINE = "baseline"
EARLY_STOPPING_CHECK_COMPARISON = "comparison"
EARLY_STOPPING_CHECK_KINDS = (
    EARLY_STOPPING_CHECK_BASELINE, EARLY_STOPPING_CHECK_COMPARISON,
)

# The key a DUE check is persisted under inside a training record. Added only on an
# iteration that really took a check, and never at all on a run with the feature off --
# ABSENT, not null, which is the discipline the CTDE critic diagnostics already follow.
_EARLY_STOPPING_RECORD_KEY = "early_stopping_check"

# WHY the training loop ended, recorded in `run_summary.json:/early_stopping`. Three
# unambiguous values, so "this run is short" is never left to be inferred by comparing
# two counts.
TERMINATION_REASON_PLATEAU = "training_reward_plateau"
TERMINATION_REASON_MAX_BUDGET = "maximum_budget_reached"
TERMINATION_REASON_DISABLED = "disabled_fixed_budget"
TERMINATION_REASONS = (
    TERMINATION_REASON_PLATEAU, TERMINATION_REASON_MAX_BUDGET,
    TERMINATION_REASON_DISABLED,
)

# JSON has no comments, so a preset may carry any number of underscore-prefixed keys as
# prose. Everything else must be a real field name -- an unrecognized key is REFUSED
# rather than ignored, because a typo that silently leaves a knob at its default is a
# run that measured something other than what its file says.
_CONFIG_COMMENT_PREFIX = "_"

# CLI dest -> TrainConfig field. The ONE mapping site behind both `main`'s config
# construction and the JSON-preset override precedence, so a flag cannot reach a
# different field than the preset key of the same name does.
_CLI_FIELD_BY_DEST = {
    "iterations": "n_iterations",
    "episodes": "episodes_per_iteration",
    "seed": "base_seed",
    "out": "output_dir",
    "checkpoint_every": "checkpoint_every",
    "eval_every": "eval_every",
    "eval_episodes": "eval_episodes",
    "eval_base_seed": "eval_base_seed",
    "num_agents": "num_agents",
    "n_known": "n_known",
    "n_hidden": "n_hidden",
    "min_target_distance_km": "min_target_distance_km",
    "min_known_separation_km": "min_known_separation_km",
    "num_red_airbases": "num_red_airbases",
    "partial_ratio": "partial_ratio",
    "stretch_target_ratio": "stretch_target_ratio",
    "fuel_damage_mode": "fuel_damage_mode",
    "fuel_damage_probability": "fuel_damage_probability",
    "fuel_damage_mild_probability": "fuel_damage_mild_probability",
    "fuel_damage_leg_progress": "fuel_damage_leg_progress",
    "fuel_damage_rtb_margin": "fuel_damage_rtb_margin",
    "aircraft_penalty_coeff": "aircraft_penalty_coeff",
    "visual_artifacts": "visual_artifacts",
    "actor_gradient_diagnostics": "actor_gradient_diagnostics",
    "training_mode": "training_mode",
    "episode_design": "episode_design",
    "match_aou_backend": "match_aou_backend",
    "benchmark_manifest": "benchmark_manifest",
    "benchmark_profile": "benchmark_profile",
    "generalized_max_attempts_per_iteration":
        "generalized_max_attempts_per_iteration",
    "early_stopping": "early_stopping",
    "early_stopping_min_iterations": "early_stopping_min_iterations",
    "early_stopping_window_iterations": "early_stopping_window_iterations",
    "early_stopping_patience_windows": "early_stopping_patience_windows",
    "early_stopping_min_delta": "early_stopping_min_delta",
}

# CLI dest -> PPOConfig field (the nested block).
_CLI_PPO_FIELD_BY_DEST = {
    "lr": "lr",
    "epochs": "n_epochs",
    "entropy_coeff": "entropy_coeff",
    "clip_ratio": "clip_ratio",
}

# Fields whose JSON form is a list but whose dataclass form is a tuple. `asdict` writes
# `num_red_airbases` as a list, so a preset copied out of a previous run's
# `run_config.json:/train_config` must load back into the same config it came from.
_CONFIG_TUPLE_FIELDS = ("num_red_airbases",)

# The THREE ways a resolved config can have come about, recorded verbatim in
# `run_config.json:/config_source`. `config_source` is ALWAYS a structured object --
# never `null` -- so a reader parses one shape and reads `resolved_from` to learn which
# case it is. "No preset" is then a STATED fact (`path: null`, empty field lists) rather
# than an absent key, which is indistinguishable from a writer that forgot to record it.
#
#   cli_defaults  : a COMMAND LINE with no `--config`. The values are the argparse
#                   defaults plus whatever flags were typed.
#   config_file   : a command line that named a JSON preset (`path` says which).
#   direct_config : a `TrainConfig` built IN PYTHON and handed straight to `train()`,
#                   with no command line and no preset involved at all. `_selftest`
#                   does exactly this, and so does any notebook or script that imports
#                   the trainer -- so this is a real repository path, not a hypothetical
#                   one. It is a SEPARATE value on purpose: labelling such a run
#                   `cli_defaults` would assert that a command line resolved it, which
#                   is precisely the kind of plausible-but-false provenance a run record
#                   exists to prevent. (A caller that DID resolve a real source passes
#                   it through `train(..., config_source=...)`; this value is only the
#                   fallback for one that did not.)
_CONFIG_SOURCE_CLI_DEFAULTS = "cli_defaults"
_CONFIG_SOURCE_FILE = "config_file"
_CONFIG_SOURCE_DIRECT = "direct_config"
_CONFIG_SOURCE_KINDS = (
    _CONFIG_SOURCE_CLI_DEFAULTS, _CONFIG_SOURCE_FILE, _CONFIG_SOURCE_DIRECT,
)



# =============================================================================
# 1. Config
# =============================================================================

def derived_split(n: int, partial_ratio: float) -> Tuple[int, int]:
    """Preview ``(known, hidden)`` for ``n`` targets at ``partial_ratio``.

    LEGACY SPLIT SURFACE. The offline scenario-construction path no longer derives its
    known/hidden counts from a ratio -- it states them outright as ``TrainConfig.n_known``
    / ``n_hidden`` -- so this function and everything built on it
    (:attr:`TrainConfig.split_preview`, the ``derived_split`` key in ``run_config.json``,
    the legacy hazard warnings) now describe a surface the construction path does not
    consult. They are kept, green and tested, because retiring the split is its own
    phase; do not repurpose them to mean the construction counts.

    A MIRROR of the authority, ``graph_episode_setup.split_tasks``, which computes
    ``num_partial = max(1, int(n * partial_ratio))`` and hides the rest. Nothing here
    decides anything -- ``split_tasks`` remains the only site that performs the split;
    this exists so the trainer can SHOW the researcher what that site will do before an
    episode is generated. The equivalence is TEST-ENFORCED
    (``tests/test_graph_train.py`` asserts this function's ``known`` against
    ``meta["known"]`` from a real ``split_tasks`` call over a grid of ``n`` x ratios),
    so the two arithmetics cannot silently diverge.

    ``int()`` TRUNCATES toward zero -- it does not round. At ``n = 6`` that makes
    ``1.0/3.0`` -> known 2 but the decimal ``0.333`` -> known 1: a different, hazardous
    config. The truncation is deliberately NOT "cleaned up" into rounding here, because
    mirroring the locked arithmetic exactly is the entire point.

    ``n`` is the resolved target count. Under ``include_sams=False`` (the baseline)
    facilities are forced to 0, so the enemy targets are exactly the red airbases and
    ``n == num_red_airbases``.

    Geometry can change WHICH tasks are known (isolated targets get pinned into the
    known set), never HOW MANY -- every ``split_tasks`` return path yields
    ``known == num_partial`` for ``n >= 2``.
    """
    n = int(n)
    if n < 2:                       # split_tasks' degenerate branch: nothing to hide
        return n, 0
    known = max(1, int(n * float(partial_ratio)))
    return known, n - known


def _format_split_preview(preview: List[Dict[str, int]]) -> str:
    """``[{n,known,hidden}, ...]`` -> ``"3/3   (n=6)"`` / ``"2/2   (n=4) ... 4/4   (n=8)"``."""
    return " ... ".join(
        "%d/%d   (n=%d)" % (p["known"], p["hidden"], p["n"]) for p in preview
    )


@dataclass
class TrainConfig:
    """Knobs for one PPO training run.

    ``n_iterations`` has NO default on purpose: how long to train is the one decision
    a caller must make explicitly (the CLI marks ``--iterations`` required).

    Attributes:
        n_iterations: number of PPO iterations (each = a batch of episodes + ONE
            :meth:`PPOUpdater.update`).
        episodes_per_iteration: episodes collected per iteration -- the PPO batch.
        base_seed: pins the initial policy weights (once, before ``build_policy``) and
            anchors the training seed band.
        output_dir: the run directory; defaults to ``training_output_<timestamp>``.
        ppo: the frozen :class:`PPOConfig` (never mutated mid-run).
        training_mode: ``actor_only`` (default, the Phase-A reference path) or ``ctde``
            (a centralized critic during TRAINING only). See :data:`TRAINING_MODES`.
        ctde: the frozen :class:`CTDEConfig`. Read ONLY when ``training_mode == "ctde"``.
        checkpoint_every: save a checkpoint every N iterations (and at the end).
        eval_every: run a deterministic eval round every N iterations (and at the end).
            ``<= 0`` disables evaluation entirely.
        eval_episodes: episodes per eval round. ``<= 0`` also disables evaluation.
        eval_base_seed: start of the FIXED, held-out eval seed band. Must sit beyond
            every training seed the run will reach (enforced by :meth:`validate`).
        num_agents: fleet size. Must be ``<= n_known`` -- more agents than targets is
            the forced-stacking cell that pinned every Phase-A episode at R = -1/3.
        n_known: targets the generator EMITS, all of them known at t=0.
        n_hidden: hidden targets ``setup_episode``'s construction path places against the
            solved routes and patches into the world. Passed to setup with a fresh
            per-episode rng; :attr:`n_targets_emitted` is the resulting world size.
        min_target_distance_km / min_known_separation_km: the requested construction
            geometry, declared STRICT to the generator (see
            :func:`build_variation_config`). 200 km keeps a target out of the
            ``DETECTION_KM`` bubble an ego sits in at wheels-up; 100 km keeps the known
            routes from collapsing onto each other now that Layer 1 is off.
        partial_ratio: LEGACY. Once the fraction of tasks known at t=0; the construction
            path never reaches ``split_tasks`` and derives nothing from this. It survives
            only to keep :func:`derived_split` / :attr:`split_preview` / the
            ``derived_split`` record green until the split surface is retired. The
            truncation note still applies to that legacy arithmetic: WRITE EXACT
            FRACTIONS (``1.0/3.0``, never ``0.333``).
        max_ticks: per-episode tick cap (``None`` -> the env's own ``MAX_SIM_TICKS``).
        include_sams / randomize_red_airbase_positions / stretch_target_ratio:
            generator knobs, live on the construction path.
        visual_artifacts: opt in to per-attempt inspection bundles (OFF by default). See
            :class:`_AttemptArtifacts`; it is an observation surface and changes nothing
            an episode measures.
        actor_gradient_diagnostics: opt in (OFF by default, ``ctde`` only) to the
            epoch-0 actor-gradient decomposition artifact
            ``train_actor_gradient_diagnostics.jsonl``. Observational; it changes no
            update.
        num_red_airbases: LEGACY, like ``partial_ratio`` -- the construction path emits
            ``n_known`` targets and never reads this.
    """

    n_iterations: int
    episodes_per_iteration: int = 8
    base_seed: int = 0
    output_dir: Union[str, Path] = ""       # "" -> training_output_<timestamp>
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # --- PHASE B: which TRAINING algorithm this run uses ---------------------------
    # `actor_only` (the DEFAULT) is the Phase-A reference path, byte-for-byte what the
    # approved long baseline was measured on. `ctde` adds a centralized critic during
    # training only. EXECUTION IS DECENTRALIZED IN BOTH: evaluation and inference run
    # the actor alone, on its own private observation, with no critic present. The
    # nested `ctde` block is read ONLY by a `ctde` run; under `actor_only` no critic and
    # no central observation is ever constructed (see `TRAINING_MODES`).
    training_mode: str = TRAINING_MODE_ACTOR_ONLY
    ctde: CTDEConfig = field(default_factory=CTDEConfig)

    checkpoint_every: int = 10
    eval_every: int = 5
    eval_episodes: int = 8
    eval_base_seed: int = 1_000_000

    max_ticks: Optional[int] = None

    # --- WHICH EPISODE POPULATION THIS RUN DRAWS FROM ------------------------------
    # ONE explicit selector over a CLOSED set of THREE designs, never inferred from the
    # values of `num_agents` / `n_hidden` (a run that reached a generalized bundle because
    # someone typed `--n-hidden 2` would be a design nobody chose).
    #
    # `fixed_cell_v1` is the DEFAULT and preserves the historical behaviour in full: the
    # exact cell below, `exact_v1` hidden cardinality, the legacy FD eligibility and
    # single post-FD wake, and the `static_t0_v1` reward reference -- the path every
    # approved measurement was taken on.
    #
    # `generalized_v1` selects the COMPLETE approved Task-1/2/3 bundle in one word, and
    # makes the construction cell PER-EPISODE: A is sampled from {2,3,4}, K == A, and
    # H_requested ~ Uniform({1..A}), from the sampler's own rng domain.
    #
    # `generalized_v2` selects the SAME four low-level policy ids and changes only the
    # POPULATION, in TWO STAGES: A from {2,3,4,5,6} and K from {A, A+2} BEFORE the
    # known-only solve, then H_requested against R -- the number of egos that allocation
    # actually routed -- AFTER it, which is why it cannot be drawn up front. It is valid
    # ONLY with `match_aou_backend = p1_milp_v1`: the objective that produces that route
    # count cannot be chosen separately from the design defined against it, so `validate`
    # REFUSES the legacy objective here rather than overriding it.
    #
    # Under EITHER generalized design the three fixed-cell fields below are NOT read for a
    # training episode, and `validate` says so loudly rather than ignoring them silently.
    episode_design: str = EPISODE_DESIGN_FIXED_CELL_V1

    # --- WHICH MATCH-AOU ALLOCATION OBJECTIVE THIS RUN SOLVES ---------------------
    # An EXPLICIT selector, deliberately NOT part of the `episode_design` bundle and NEVER
    # inferred -- not from the design, not from the task probabilities, not from which
    # solver happens to be installed, and not from anything else. `legacy_minlp_v1` is the
    # DEFAULT -- the frozen MINLP through BONMIN, the objective every approved measurement
    # was taken on.
    #
    # ITS VALID VALUE SET IS DESIGN-CONSTRAINED, WHICH IS NOT THE SAME AS BEING INFERRED.
    # `fixed_cell_v1` and `generalized_v1` accept EITHER approved objective.
    # `generalized_v2` is defined only against `p1_milp_v1`: it resolves its hidden load
    # from the number of non-empty routes the known-only allocation produced, so the
    # objective that produces that route count cannot be chosen separately from the design
    # that is defined against it. A `generalized_v2` run naming `legacy_minlp_v1` is
    # REFUSED by `validate()` before any compute -- refused, never silently overridden.
    # The run still has to STATE the objective it wants; nothing selects one on its behalf.
    #
    # `p1_milp_v1` selects the deterministic p = 1 MILP instead. IT IS NOT A TRANSPARENT
    # PERFORMANCE SWAP: it removes the legacy EPSILON stacking incentive, so it changes
    # which allocations are optimal, and because `A_init` is what route-relative hidden
    # placement predicts routes from, it can change the hidden geometry, episode
    # feasibility and what the policy learns. No equivalence is claimed, and there is no
    # `auto` and no fallback in either direction: one run uses one backend for every
    # MATCH-AOU solve of every episode, and an unknown id is REFUSED before any compute.
    match_aou_backend: str = DEFAULT_MATCH_AOU_BACKEND

    # The FROZEN 18-stratum benchmark this run EVALUATES on -- a path to a manifest
    # written by `graph_generalized.write_benchmark_manifest`. REQUIRED for a
    # `generalized_v1` run WITH evaluation enabled: the held-out seed band is a fixed-cell
    # construct, and letting that run fall back on it would evaluate an UNSTRATIFIED
    # population under a stratified label. To train `generalized_v1` without a benchmark,
    # disable evaluation explicitly.
    #
    # A `generalized_v2` run with evaluation enabled REQUIRES a manifest too, but a
    # DIFFERENT one: the frozen ten-cell `generalized_v2` benchmark. The loader is
    # design-aware and each schema refuses the other, so a V1 manifest can never be
    # evaluated under V2 (or the reverse). A `fixed_cell_v1` run refuses any manifest: it
    # would build every world from the fixed cell while reporting labels it never varied.
    benchmark_manifest: Optional[str] = None

    # Which frozen `generalized_v2` benchmark PROFILE this run evaluates: `development`
    # (world ordinals 0..1) or `confirmatory` (2..11). REQUIRED for an evaluating V2 run
    # and REFUSED on every other design, which has no profiles. It selects which frozen
    # groups are measured and nothing else -- the manifest identity, the training
    # population and the held-out check (always over the WHOLE manifest) do not move.
    benchmark_profile: Optional[str] = None

    # The GENERALIZED-only bounded ATTEMPT BUDGET per iteration. `None` on the
    # historical path, where it is REFUSED if set (a fixed-cell run must not silently
    # acquire replacement behaviour), and REQUIRED under BOTH generalized designs --
    # `generalized_v1` and `generalized_v2` alike -- where `episodes_per_iteration` stops
    # meaning "attempts" and starts meaning "SUCCESSFUL episodes the PPO/CTDE batch must
    # hold" (:data:`TRAINING_ATTEMPT_POLICY_QUOTA`). The quota is selected by
    # `TrainConfig.training_attempt_policy`, which reads `cfg.generalized` and nothing
    # else, so the two designs share it exactly.
    #
    # It has NO DEFAULT ON PURPOSE. How many attempts a generalized iteration needs
    # depends on the world-attrition rate of a population whose bounded runtime / solver
    # validation has not been done, so a number invented here would silently make that
    # scientific decision -- exactly as `build_benchmark_manifest` refuses to invent
    # `worlds_per_cell`. It must be >= `episodes_per_iteration`, and reaching it before
    # the quota is full ABORTS the run rather than updating on a partial batch.
    #
    # It also fixes the run's MAXIMUM POSSIBLE training-attempt seed band
    # (`max_training_attempts`), which is what every seed-band claim is made against on
    # BOTH generalized designs. On both it is additionally what a frozen benchmark
    # manifest of that design is verified to be held out from -- under `generalized_v2`
    # over EVERY manifest seed, whichever profile the run evaluates.
    generalized_max_attempts_per_iteration: Optional[int] = None

    # --- GENERALIZED-V1 EARLY STOPPING: opt-in, OFF by default --------------------
    # `early_stopping` selects `training_reward_plateau_v1`
    # (:data:`EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU`). OFF, the run is fixed-budget
    # and every control-flow decision below is exactly the one it always was: no check is
    # computed, no key is added to a training record, and the loop cannot exit early.
    #
    # ON, the run still declares its MAXIMUM budget through `n_iterations`, and every
    # held-out / seed-band claim is still made against that maximum
    # (:attr:`max_training_attempts`) rather than against wherever the run happens to
    # stop. Early stopping changes only what a run ACTUALLY consumes.
    #
    # The four parameters below are the approved defaults, and every one of them is a
    # COMPLETED-ITERATION count, never a zero-based index:
    #   min_iterations     = 100 : the first monitored check. With 8 successful episodes
    #                              per iteration, monitoring begins after 800 successful
    #                              episodes.
    #   window_iterations  = 25  : each check averages `train_reward_mean` over the most
    #                              recent 25 completed iterations. Checks fall every 25
    #                              iterations from `min_iterations`, so the monitored
    #                              windows are NON-OVERLAPPING and the first 75 completed
    #                              iterations sit outside every window.
    #   patience_windows   = 3   : consecutive non-improving windows before stopping, so
    #                              the earliest possible stop is 100 + 3 x 25 = 175
    #                              completed iterations (1400 successful episodes at 8
    #                              per iteration).
    #   min_delta          = 0.01: the smallest reward gain that counts as a MEANINGFUL
    #                              improvement. Below it the window is stale.
    # They are validated only when the feature is enabled -- the unused block may hold
    # any value on a fixed-budget run, exactly as the unused `ctde` block may.
    early_stopping: bool = False
    early_stopping_min_iterations: int = 100
    early_stopping_window_iterations: int = 25
    early_stopping_patience_windows: int = 3
    early_stopping_min_delta: float = 0.01

    # --- THE OFFLINE SCENARIO-CONSTRUCTION REFERENCE CELL (B1) ---------------------
    # Stated outright, never derived from a ratio. A CELL, NOT A LAW: a later phase
    # varies these per episode, so nothing downstream may hard-code them.
    #   num_agents = 3 <= n_known = 3   : one target per ego, no forced stacking.
    #   n_hidden   = 3                  : placed route-relative by setup_episode's
    #                                     construction path, one per ego route.
    #   min_target_distance_km = 200    : the old 50 km floor equalled DETECTION_KM, so
    #                                     the measured fixture put targets 58.8 / 63.2 km
    #                                     from launch -- discoverable seconds after
    #                                     wheels-up, which destroys the mid-route
    #                                     discovery event the phase studies.
    #   min_known_separation_km = 100   : Layer 1 is OFF on this path (it used to pull
    #                                     pairs to 13.7 / 28.9 km and flatten route
    #                                     diversity); this is what pushes them apart.
    num_agents: int = 3
    n_known: int = 3
    n_hidden: int = 3
    min_target_distance_km: float = 200.0
    min_known_separation_km: float = 100.0

    # --- generator knobs live on the construction path ---
    include_sams: bool = False
    randomize_red_airbase_positions: bool = True
    stretch_target_ratio: float = 0.5

    # --- FD-BASELINE-v1: THE difficulty factor of the final Phase-A baseline cell ----
    # The scenario cell above is UNCHANGED -- same counts, same geometry, same p = 1,
    # same weapon lethality, no SAMs. The only added difficulty is a seeded, ego-local,
    # one-shot fuel-damage event (`graph_fuel_damage`), which turns
    # SELF_PRESERVATION_ABORT from a never-correct action into a live alternative.
    #   fuel_damage_mode        : `seeded_mixture` -> the condition is a deterministic
    #                             function of the episode seed. Evaluation overrides it
    #                             per pair member and never uses this value.
    #   fuel_damage_probability : half of the scheduled TRAINING episodes are damaged.
    #   fuel_damage_leg_progress: the event fires at ~30% of the ego's first planned leg.
    #   fuel_damage_rtb_margin  : the engine's own 1.10 reserve, applied to both ends of
    #                             the strict window.
    #
    # FD-VARIABLE-SEVERITY-v1 is selected by setting `fuel_damage_mode` to
    # `seeded_variable` instead. It keeps every knob above -- the same P(damaged), the
    # same trigger point, the same reserve -- and adds ONE:
    #   fuel_damage_mild_probability : P(mild | damaged). With P(damaged) = 0.5 this is
    #                             the approved 0.50 clean / 0.25 mild / 0.25 severe
    #                             distribution. Ignored (but still recorded) by the
    #                             legacy modes, which have no severity.
    # A `seeded_variable` run evaluates each held-out seed as a clean / mild / severe
    # matched TRIAD; a legacy run keeps its clean / damaged matched PAIR.
    fuel_damage_mode: str = FuelDamageMode.SEEDED_MIXTURE
    fuel_damage_probability: float = 0.5
    fuel_damage_leg_progress: float = 0.30
    fuel_damage_rtb_margin: float = 1.10
    fuel_damage_mild_probability: float = 0.5

    # The death penalty coefficient `c`, ACTIVATED here (graph_reward's default is 0.0 and
    # its FORMULA is untouched). At 2.25 a lost airframe costs 2.25 max-utility targets,
    # so on a 6 x 80 cell flying the tank dry to reach one more target is decisively
    # net-negative and RTB strictly beats suicide-on-best. Passed as an explicit
    # `RewardConfig` at the reward call site rather than by mutating a shared default.
    aircraft_penalty_coeff: float = 2.25

    # --- VISUAL ARTIFACTS: opt-in inspection bundles, OFF by default --------------
    # Purely additive OBSERVATION. When True, every scheduled `pre_update` / `train` /
    # `post_update` attempt preserves the exact generated known-only scenario, the
    # authoritative executed t=0 scenario and the BLADE playback recording in its own
    # directory under `<run_dir>/visual_artifacts/`, so a finished run can be re-opened in
    # PyCharm and in the BLADE client. It selects EVERY scheduled attempt -- there is
    # deliberately no per-seed filter, which would be a second artifact-selection language
    # next to the seed schedule. It changes no seed, no scenario tag, no scenario name and
    # no episode outcome (see `_AttemptArtifacts`).
    visual_artifacts: bool = False

    # --- ACTOR-GRADIENT DIAGNOSTICS: opt-in, OFF by default, CTDE only -------------
    # Observational: extra `autograd.grad` calls on the epoch-0 graph of each productive
    # update (cost), no change to what the update does. See `_actor_gradient_record`.
    actor_gradient_diagnostics: bool = False

    # --- LEGACY split surface (see `derived_split`) -------------------------------
    # The Phase-A baseline cell, kept so `derived_split` / `split_preview` / the
    # `derived_split` record / the hazard warnings stay green and testable. The
    # construction path emits `n_known` targets and runs setup in construction mode, so
    # NEITHER of these reaches the generator or the split any more (`split_tasks` is not
    # called at all). Retiring them is a separate phase.
    num_red_airbases: Tuple[int, int] = (6, 6)
    partial_ratio: float = 0.5

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not str(self.output_dir):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"training_output_{stamp}"

    # ------------------------------------------------------------------
    @property
    def total_episodes(self) -> int:
        """Training episodes the run will collect (eval episodes are extra)."""
        return int(self.n_iterations) * int(self.episodes_per_iteration)

    @property
    def eval_enabled(self) -> bool:
        """True iff evaluation rounds run at all (both knobs must be positive)."""
        return self.eval_every > 0 and self.eval_episodes > 0

    # ------------------------------------------------------------------
    # GENERALIZED-V1 Task 5C: attempts vs SUCCESSFUL episodes.
    #
    # `total_episodes` above keeps its meaning under BOTH designs -- the training
    # episodes the run intends to COLLECT -- and the three properties below say how many
    # ATTEMPTS that collection may cost. On the fixed-cell path they collapse to exactly
    # the historical arithmetic (one attempt per scheduled episode), which is what keeps
    # every seed-band, tag-namespace and provenance statement byte-identical there.
    @property
    def training_attempt_policy(self) -> str:
        """Whether ``episodes_per_iteration`` counts ATTEMPTS or SUCCESSFUL episodes."""
        return (TRAINING_ATTEMPT_POLICY_QUOTA if self.generalized
                else TRAINING_ATTEMPT_POLICY_SCHEDULED)

    @property
    def max_attempts_per_iteration(self) -> int:
        """The most ordinary training attempts ONE iteration may make.

        Under the historical policy an iteration makes exactly
        ``episodes_per_iteration`` attempts, so this IS that number and no new field is
        consulted. Under the quota policy it is the operator's explicit budget, which
        ``validate`` has already required to be present and ``>= episodes_per_iteration``.
        """
        if not self.generalized:
            return int(self.episodes_per_iteration)
        budget = self.generalized_max_attempts_per_iteration
        if budget is None:
            raise ValueError(
                "episode_design=%r requires an explicit "
                "generalized_max_attempts_per_iteration; validate() refuses a run "
                "without one." % self.episode_design
            )
        return int(budget)

    @property
    def max_training_attempts(self) -> int:
        """The MAXIMUM number of training seeds this run can possibly consume.

        THE BOUND EVERY HELD-OUT CLAIM MUST BE MADE AGAINST. Under replacement a run may
        spend more seeds than it collects episodes, so ``total_episodes`` is no longer an
        upper bound on the training band and using it would let a benchmark world sit in
        a seed range the training loop can actually reach -- a held-out failure that
        produces entirely normal-looking numbers. On the fixed-cell path this EQUALS
        ``total_episodes``, so nothing about that path's bands or checks changes.
        """
        return int(self.n_iterations) * int(self.max_attempts_per_iteration)

    # ------------------------------------------------------------------
    @property
    def early_stopping_enabled(self) -> bool:
        """True iff this run may end before its maximum budget.

        THE ONE predicate behind every early-stopping branch in this module, so "is this
        run early-stopped?" has a single answer and cannot be re-derived (differently) at
        the loop, the header and the summary. It reads the opt-in flag and nothing else
        -- notably NOT ``training_mode``: the stopping mechanism is identical under
        ``actor_only`` and ``ctde``, which is what keeps two arms comparable.
        """
        return bool(self.early_stopping)

    @property
    def early_stopping_earliest_stop_iterations(self) -> int:
        """The FEWEST completed iterations at which a stop could possibly occur.

        ``min_iterations + patience_windows * window_iterations``: the baseline check,
        then that many consecutive non-improving windows. At the approved defaults this
        is ``100 + 3 * 25 = 175``. :meth:`validate` requires ``n_iterations`` to reach it,
        because a run too short to ever stop would record an ACTIVE stopping policy whose
        mechanism was structurally inert.
        """
        return (int(self.early_stopping_min_iterations)
                + int(self.early_stopping_patience_windows)
                * int(self.early_stopping_window_iterations))

    @property
    def ctde_enabled(self) -> bool:
        """True iff this run trains with the Phase-B centralized critic.

        THE ONE predicate behind every CTDE branch in this module, so "is this a CTDE
        run?" has a single answer and cannot be re-derived (differently) at three call
        sites. It reads the TRAINING mode and nothing else -- notably not
        ``ctde.value_coeff``, because a coefficient is not a mode (:data:`TRAINING_MODES`).

        It says nothing about EXECUTION: evaluation and inference are actor-only in both
        modes.
        """
        return str(self.training_mode) == TRAINING_MODE_CTDE

    @property
    def n_targets_emitted(self) -> int:
        """Targets an episode's world really holds: ``n_known + n_hidden``.

        The one place that distinguishes what the GENERATOR writes (``n_known``, a
        known-only world) from what the episode finally RUNS ON: ``setup_episode``'s
        construction path patches ``n_hidden`` route-relative targets into the scenario
        between the two solves, so the executed world is the sum. The startup header and
        ``run_config.json`` both report through this property.
        """
        return int(self.n_known) + int(self.n_hidden)

    # ------------------------------------------------------------------
    def fuel_damage_parameters(
        self, mode: Optional[str] = None
    ) -> FuelDamageParameters:
        """The ONE site that turns this config into a :class:`FuelDamageParameters`.

        ``mode`` overrides only the mode -- evaluation forces ``forced_clean`` /
        ``forced_damaged`` (or ``forced_mild`` / ``forced_severe``) per group member
        while keeping the threshold, the margin and the two probabilities identical to
        training, which is what makes an eval measurement describe the same event the
        training episodes contained.
        """
        design = self.design
        return FuelDamageParameters(
            mode=str(self.fuel_damage_mode if mode is None else mode),
            probability=float(self.fuel_damage_probability),
            leg_progress_threshold=float(self.fuel_damage_leg_progress),
            rtb_safety_margin=float(self.fuel_damage_rtb_margin),
            mild_probability=float(self.fuel_damage_mild_probability),
            # GENERALIZED-V1: the two FD policy seams, RESOLVED FROM THE ONE DESIGN
            # SELECTOR rather than exposed as two more independent knobs. Under
            # `fixed_cell_v1` these are exactly the values `FuelDamageParameters` would
            # have defaulted to, so the constructed object is identical to the one every
            # pre-Task-4 call site produced -- stated explicitly here so a record always
            # says which policies ran instead of leaving it to be inferred from a default.
            eligibility_policy=design.eligibility_policy,
            post_fd_wake_policy=design.post_fd_wake_policy,
        )

    # ------------------------------------------------------------------
    @property
    def design(self) -> EpisodeDesign:
        """THE ONE resolution site: this run's four low-level policy ids.

        Every consumer -- the FD parameters, the ``setup_episode`` keywords, the run
        config block and the per-episode record -- reads the bundle from HERE, so a run
        cannot resolve half a design. An unrecognized id RAISES rather than falling back
        on the historical bundle (:func:`resolve_episode_design`); ``validate`` refuses it
        before any compute, and this property refuses it again if one is reached anyway.
        """
        return resolve_episode_design(self.episode_design)

    @property
    def route_relative_population(self) -> bool:
        """True iff this run draws its worlds from the GENERALIZED-V2 population.

        The ONE predicate behind every two-stage branch, resolved from ``episode_design``
        and from nothing else -- so the pre-solve ``(A, K)`` draw, the post-solve ``H | R``
        draw, the required P1 backend and the route-relative provenance can never be
        reached one without the others.
        """
        return self.design.route_relative_population

    @property
    def generalized(self) -> bool:
        """True iff this run draws from a GENERALIZED population -- V1 OR V2.

        The two designs share every harness behaviour that keys off this predicate: the
        seeded-variable fuel-damage mixture, the successful-episode training quota and its
        bounded attempt budget, and the dynamic construction provenance. What separates
        them is WHICH population, which is :attr:`route_relative_population`'s question.
        """
        return self.design.generalized

    # ------------------------------------------------------------------
    # THE MATCHED-EVALUATION GROUP. Its shape is decided by the run's TRAINING mode and
    # by nothing else -- never by an individual member's forced mode, which would make
    # the question "how many members does a held-out seed get?" unanswerable from inside
    # one of them.
    @property
    def variable_severity(self) -> bool:
        """True iff this run uses FD-VARIABLE-SEVERITY-v1 rather than the legacy design."""
        return str(self.fuel_damage_mode) == FuelDamageMode.SEEDED_VARIABLE

    @property
    def eval_group_members(self) -> Tuple[Tuple[str, str], ...]:
        """The matched group's ``(cell, forced mode)`` members, in attempt order."""
        return (_EVAL_TRIAD_MEMBERS if self.variable_severity
                else _EVAL_PAIR_MEMBERS)

    @property
    def eval_group_size(self) -> int:
        """Attempts per held-out SEED (2 for a pair, 3 for a triad)."""
        return len(self.eval_group_members)

    @property
    def eval_group_kind(self) -> str:
        """``pair`` or ``triad`` -- what a record calls this run's matched group."""
        return (_EVAL_GROUP_KIND_TRIAD if self.variable_severity
                else _EVAL_GROUP_KIND_PAIR)

    @property
    def eval_group_deltas(self) -> Tuple[Tuple[str, str], ...]:
        """The within-seed ``(cell, reference cell)`` differences this run reports."""
        return _EVAL_TRIAD_DELTAS if self.variable_severity else _EVAL_PAIR_DELTAS

    @property
    def reported_cells(self) -> Tuple[str, ...]:
        """The labels episodes are reported under: clean/damaged, or clean/mild/severe.

        A CELL is a reporting label, not a new condition: ``mild`` and ``severe``
        episodes are both DAMAGED, and every clean/damaged count keeps exactly the
        meaning it had (:meth:`_ConditionTally.to_record` derives those by pooling).
        """
        return tuple(cell for cell, _mode in self.eval_group_members)

    def reward_config(self) -> RewardConfig:
        """The ONE site that turns this config into a :class:`RewardConfig`.

        The reward FORMULA is untouched (``graph_reward`` stays frozen); this only
        supplies the death-penalty coefficient the formula already accepted, instead of
        letting the call site fall through to the module default of ``0.0``.
        """
        return RewardConfig(aircraft_penalty_coeff=float(self.aircraft_penalty_coeff))

    # ------------------------------------------------------------------
    def _airbase_bounds(self) -> Tuple[int, int]:
        """``num_red_airbases`` as an inclusive ``(lo, hi)`` (a bare int -> ``(n, n)``)."""
        v = self.num_red_airbases
        if isinstance(v, (tuple, list)):
            lo = int(v[0])
            hi = int(v[-1])
        else:
            lo = hi = int(v)
        return lo, hi

    @property
    def split_preview(self) -> List[Dict[str, int]]:
        """The derived known/hidden split at each END of the ``num_red_airbases`` range.

        One entry for a fixed count, two for a range -- the generator samples ``n``
        uniformly inside it, so the two ends bracket every split the run can produce.
        Computed ONLY through :func:`derived_split` (the one arithmetic site).
        """
        lo, hi = self._airbase_bounds()
        out: List[Dict[str, int]] = []
        for n in ([lo] if lo == hi else [lo, hi]):
            known, hidden = derived_split(n, self.partial_ratio)
            out.append({"n": int(n), "known": int(known), "hidden": int(hidden)})
        return out

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Refuse a self-inconsistent config BEFORE any expensive work starts.

        The load-bearing check is seed-band disjointness: if the training band
        ``[base_seed, base_seed + total_episodes)`` reached into the eval band
        ``[eval_base_seed, eval_base_seed + eval_episodes)``, the "held-out" eval set
        would silently contain scenarios the policy had trained on and the learning
        curve would be measuring memorization. That is a research bug that produces
        plausible-looking numbers, so it fails LOUD here.

        The construction cell is checked here too, and ``num_agents > n_known`` RAISES:
        it is not a hazard a researcher probes, it is the forced-stacking configuration
        that made every Phase-A episode return the same reward.

        Scenario HAZARDS are additionally reported -- PRINTED, never raised. A
        researcher may deliberately probe a stalling or a pop-up-free cell, so refusing
        them would be wrong; the point is that none can be entered by ACCIDENT. The two
        legacy ones are evaluated at the LOW end of the ``num_red_airbases`` range --
        fewest targets is the worst case for each -- through :func:`derived_split`, the
        one split-arithmetic site, and they judge the LEGACY surface only.
        """
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if self.episodes_per_iteration < 1:
            raise ValueError(
                f"episodes_per_iteration must be >= 1, got {self.episodes_per_iteration}"
            )
        if not (0.0 < float(self.partial_ratio) <= 1.0):
            raise ValueError(f"partial_ratio must be in (0, 1], got {self.partial_ratio}")

        # An UNRECOGNIZED training mode raises rather than silently falling back to
        # `actor_only`: a run that quietly trained the Phase-A algorithm while its config
        # said `ctde` (or the reverse) would be a mislabelled measurement, which is worse
        # than a crash. `ctde_enabled` is the only reader of this field.
        if str(self.training_mode) not in TRAINING_MODES:
            raise ValueError(
                "training_mode must be one of %s, got %r"
                % (list(TRAINING_MODES), self.training_mode)
            )
        if self.ctde_enabled:
            if not (0.0 <= float(self.ctde.gae_lambda) <= 1.0):
                raise ValueError(
                    "ctde.gae_lambda must be in [0, 1], got %r" % (self.ctde.gae_lambda,)
                )
            if float(self.ctde.critic_lr) <= 0.0:
                raise ValueError(
                    "ctde.critic_lr must be > 0, got %r" % (self.ctde.critic_lr,)
                )
            # STRICTLY POSITIVE, and 0 is the case this exists to refuse. A run labelled
            # `ctde` with `value_coeff = 0` would build central observations and take
            # its advantages from the critic while NEVER TRAINING that critic -- so the
            # baseline would stay a frozen random function forever. That is neither the
            # `actor_only` reference algorithm nor the approved CTDE one, and it would
            # be recorded and read as CTDE. Rejected here, before any compute.
            # This is a VALIDITY bound, not a mode selector: `ctde_enabled` still reads
            # `training_mode` and nothing else.
            if float(self.ctde.value_coeff) <= 0.0:
                raise ValueError(
                    "ctde.value_coeff must be > 0 under training_mode='ctde', got %r. "
                    "A zero coefficient never trains the critic while still using its "
                    "advantages, which is neither training mode. To run without a "
                    "critic set training_mode='actor_only'."
                    % (self.ctde.value_coeff,)
                )

        # The actor-gradient diagnostic instruments `CTDEUpdater` only; accepting it on an
        # `actor_only` run would record an enabled diagnostic that can never write.
        if self.actor_gradient_diagnostics and not self.ctde_enabled:
            raise ValueError(
                "actor_gradient_diagnostics requires training_mode='ctde', got %r"
                % (self.training_mode,))

        # --- THE DESIGN SELECTOR, checked before anything reads it -----------------
        # An UNRECOGNIZED design raises rather than falling back on the historical
        # bundle: a run that quietly measured the fixed cell while its config said
        # `generalized_v1` is a mislabelled measurement, which is worse than a crash.
        if str(self.episode_design) not in EPISODE_DESIGNS:
            raise ValueError(
                "episode_design must be one of %s, got %r"
                % (list(EPISODE_DESIGNS), self.episode_design)
            )
        # --- WHICH MATCH-AOU objective, checked before anything solves anything -----
        # An EXPLICIT selector: it is never RESOLVED FROM `episode_design`, and an unknown
        # id raises here (as `MatchAouBackendError`, the stable backend-integrity type)
        # rather than falling back on the historical objective -- for the same reason an
        # unknown design does: a run that quietly solved a different objective than its
        # record claims is a mislabelled measurement.
        #
        # What the design DOES constrain is the valid VALUE SET. `fixed_cell_v1` and
        # `generalized_v1` accept either approved objective; the `generalized_v2` check
        # immediately below refuses a run that named the legacy one. That is a REFUSAL of
        # a contradictory request, not a selection made on the run's behalf.
        resolve_match_aou_backend(self.match_aou_backend)
        design = self.design
        if design.route_relative_population:
            # --- GENERALIZED-V2: the design is only defined with the P1 objective ------
            # NOT an override and NOT a fallback: refused before any episode executes,
            # because V2 resolves its hidden load from the number of NON-EMPTY ROUTES the
            # known-only allocation produced. Letting two different objectives define that
            # route count would make one design id mean two different population
            # selectors -- and the legacy objective's EPSILON stacking incentive in
            # particular changes which allocations are optimal, hence which egos are routed.
            if str(self.match_aou_backend) != GENERALIZED_V2_REQUIRED_BACKEND:
                raise ValueError(
                    "episode_design=%r requires match_aou_backend=%r; got %r. The "
                    "route-relative hidden load is defined against the route count the "
                    "known-only allocation produces, so the design and the objective that "
                    "produces it cannot be chosen independently. Refused rather than "
                    "silently overridden."
                    % (EPISODE_DESIGN_GENERALIZED_V2, GENERALIZED_V2_REQUIRED_BACKEND,
                       self.match_aou_backend)
                )
            # --- the GENERALIZED-V2 benchmark: its OWN ten-cell construct -------------
            # An evaluating V2 run must name a frozen V2 manifest AND exactly one declared
            # profile. It never falls back on the fixed held-out band, which carries no
            # stratum, and never reads the V1 18-stratum manifest -- that refusal happens
            # at LOAD time, where the schema is known (`load_v2_benchmark_manifest`).
            if self.eval_enabled:
                if not str(self.benchmark_manifest or ""):
                    raise ValueError(
                        "episode_design=%r with evaluation enabled requires "
                        "benchmark_manifest: a frozen %r benchmark manifest. The fixed "
                        "held-out seed band carries no stratum, so evaluating on it would "
                        "measure an UNSTRATIFIED population under a generalized label. "
                        "Point benchmark_manifest at a frozen V2 manifest, or disable "
                        "evaluation explicitly (eval_every=0 / eval_episodes=0)."
                        % (EPISODE_DESIGN_GENERALIZED_V2, EPISODE_DESIGN_GENERALIZED_V2)
                    )
                if str(self.benchmark_profile or "") not in V2_BENCHMARK_PROFILES:
                    raise ValueError(
                        "episode_design=%r with evaluation enabled requires "
                        "benchmark_profile to be one of %r, got %r: a run must declare "
                        "which frozen profile it measures."
                        % (EPISODE_DESIGN_GENERALIZED_V2, list(V2_BENCHMARK_PROFILES),
                           self.benchmark_profile)
                    )
        if self.benchmark_profile is not None:
            if not design.route_relative_population:
                raise ValueError(
                    "benchmark_profile is set but episode_design=%r: benchmark profiles "
                    "are a %r construct only." % (self.episode_design,
                                                  EPISODE_DESIGN_GENERALIZED_V2)
                )
            if str(self.benchmark_profile) not in V2_BENCHMARK_PROFILES:
                raise ValueError(
                    "benchmark_profile must be one of %r, got %r"
                    % (list(V2_BENCHMARK_PROFILES), self.benchmark_profile)
                )
            if not str(self.benchmark_manifest or ""):
                raise ValueError(
                    "benchmark_profile=%r is set but benchmark_manifest is not: a profile "
                    "selects groups FROM a frozen manifest." % (self.benchmark_profile,)
                )
        if design.generalized:
            # The approved generalized TRAINING mixture is 0.50 clean / 0.25 mild /
            # 0.25 severe, which is `seeded_variable` and nothing else. A generalized run
            # on `seeded_mixture` would make every damaged episode structurally SEVERE
            # while its records claimed the generalized design, and a generalized run on
            # `off` would carry no fuel-damage event for the certified eligibility policy
            # to certify -- neither is the approved design.
            if str(self.fuel_damage_mode) != FuelDamageMode.SEEDED_VARIABLE:
                raise ValueError(
                    "episode_design=%r requires fuel_damage_mode=%r (the approved "
                    "0.50 clean / 0.25 mild / 0.25 severe training mixture); got %r. "
                    "The certified eligibility policy certifies BOTH severities on one "
                    "ego, and the matched benchmark evaluates clean/mild/severe triads."
                    % (self.episode_design, FuelDamageMode.SEEDED_VARIABLE,
                       self.fuel_damage_mode)
                )
            # Evaluation MUST be the frozen stratified benchmark. The held-out seed band
            # is a FIXED-CELL construct: its seeds carry no stratum, so evaluating a
            # generalized run on it would measure an unstratified population and report
            # it under a stratified design. Refused rather than silently substituted.
            # V2 has already been refused above (it has no benchmark at all), so this is
            # the V1 rule and reaches only V1 runs.
            if (design.generalized_v1_design and self.eval_enabled
                    and not str(self.benchmark_manifest or "")):
                raise ValueError(
                    "episode_design=%r with evaluation enabled requires "
                    "benchmark_manifest: the fixed held-out seed band carries no "
                    "stratum, so evaluating on it would measure an UNSTRATIFIED "
                    "population under a stratified label. Point benchmark_manifest at a "
                    "frozen manifest, or disable evaluation explicitly "
                    "(eval_every=0 / eval_episodes=0)."
                    % EPISODE_DESIGN_GENERALIZED_V1
                )
            # The fixed-cell fields are NOT read on this path: a training episode's
            # cardinality is drawn per seed and a benchmark member's comes from its
            # stratum. Said out loud, because a config that carries `n_hidden = 3` and
            # produces worlds with 1..A hidden targets is otherwise confusing to read.
            if design.route_relative_population:
                print("[WARN] episode_design=%s: num_agents / n_known / n_hidden are NOT "
                      "read. A training episode's cardinality is SAMPLED in TWO STAGES: "
                      "A ~ U%s and K | A ~ U{A + o : o in %s} BEFORE the known-only solve, "
                      "then H_requested ~ U{1..R} AFTER it, where R is the number of egos "
                      "that solve routed. Proceeding."
                      % (EPISODE_DESIGN_GENERALIZED_V2,
                         list(GENERALIZED_V2_AGENT_COUNTS),
                         list(GENERALIZED_V2_KNOWN_OFFSETS)))
            else:
                print("[WARN] episode_design=%s: num_agents / n_known / n_hidden are NOT "
                      "read. A training episode's cardinality is SAMPLED per seed "
                      "(A ~ U{2,3,4}, K == A, H ~ U{1..A}); a benchmark member's comes "
                      "from its stratum. Proceeding." % EPISODE_DESIGN_GENERALIZED_V1)
            # THE BOUNDED ATTEMPT BUDGET (Task 5C). Under this design
            # `episodes_per_iteration` is a SUCCESSFUL-episode quota, so the loop must be
            # told how many attempts it may spend obtaining it. Required and never
            # defaulted: a number invented here would silently decide how much world
            # attrition the campaign tolerates, and it is also the bound every seed-band
            # claim is made against (`max_training_attempts`) -- which under
            # `generalized_v1` is additionally what the frozen benchmark is verified to be
            # held out from, while `generalized_v2` has no benchmark to hold out from.
            budget = self.generalized_max_attempts_per_iteration
            if budget is None:
                raise ValueError(
                    "episode_design=%r requires an explicit "
                    "generalized_max_attempts_per_iteration: under this design "
                    "episodes_per_iteration (%d) is a quota of SUCCESSFUL episodes, and "
                    "the loop needs a bounded attempt budget to obtain it. There is no "
                    "default -- the value decides how much world attrition the run "
                    "tolerates, and it sets the run's MAXIMUM POSSIBLE training-attempt "
                    "seed band, which every seed-band claim is made against."
                    % (self.episode_design, int(self.episodes_per_iteration))
                )
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise ValueError(
                    "generalized_max_attempts_per_iteration must be an int, got %r"
                    % (budget,)
                )
            if int(budget) < int(self.episodes_per_iteration):
                raise ValueError(
                    "generalized_max_attempts_per_iteration (%d) must be >= "
                    "episodes_per_iteration (%d): the budget is the most ATTEMPTS an "
                    "iteration may make to collect that many SUCCESSFUL episodes, so a "
                    "smaller budget could never fill the quota."
                    % (int(budget), int(self.episodes_per_iteration))
                )
        elif self.generalized_max_attempts_per_iteration is not None:
            raise ValueError(
                "generalized_max_attempts_per_iteration is set but episode_design=%r: "
                "the successful-episode quota and its deterministic replacement are a "
                "GENERALIZED behaviour (%r and %r). A fixed-cell run makes exactly "
                "episodes_per_iteration attempts per iteration and must not silently "
                "acquire replacement, which would change the population every approved "
                "measurement was taken over."
                % (self.episode_design, EPISODE_DESIGN_GENERALIZED_V1,
                   EPISODE_DESIGN_GENERALIZED_V2)
            )
        if not design.generalized and str(self.benchmark_manifest or ""):
            raise ValueError(
                "benchmark_manifest is set but episode_design=%r: the 18-stratum "
                "benchmark is defined for %r only, and a fixed-cell run evaluating it "
                "would build every world from the fixed cell while reporting stratum "
                "labels it never varied."
                % (self.episode_design, EPISODE_DESIGN_GENERALIZED_V1)
            )

        # --- GENERALIZED-V1 EARLY STOPPING: approved for THIS design only ---
        # Refused elsewhere rather than quietly tolerated. The approved stopping contract
        # is a GENERALIZED-V1 one, and a fixed-cell run that ended early would no longer
        # be the fixed-budget path every approved measurement (CLAUDE.md section 7) was
        # taken on -- while its records would carry the same schedule fields and read as
        # though it were. The parameters are checked ONLY when the feature is enabled:
        # an unused block may hold any value, exactly as the unused `ctde` block may.
        if self.early_stopping_enabled:
            # EXACTLY `generalized_v1`, not "any generalized design". The approved plateau
            # contract was reviewed against the V1 population and its training-reward
            # trajectory; extending it to the V2 population is a research decision nobody
            # has taken, and a V2 run that stopped early would carry an approved policy id
            # over a contract that was never approved for it.
            if not design.generalized_v1_design:
                raise ValueError(
                    "early_stopping is enabled but episode_design=%r: the "
                    "%r stopping policy is approved for %r only. A fixed-cell run must "
                    "stay fixed-budget -- that is the path every approved measurement "
                    "was taken on -- and no stopping rule has been approved for any other "
                    "population; either way a run that stopped early would silently be a "
                    "different training contract under the same record schema."
                    % (self.episode_design,
                       EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU,
                       EPISODE_DESIGN_GENERALIZED_V1)
                )
            # `bool` is an `int` subclass, so it is rejected explicitly (the same guard
            # `generalized_max_attempts_per_iteration` carries): `True` silently reading
            # as a one-iteration window is exactly the kind of value that produces a
            # plausible-looking wrong stop.
            for name in ("early_stopping_min_iterations",
                         "early_stopping_window_iterations",
                         "early_stopping_patience_windows"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("%s must be an int, got %r" % (name, value))
                if int(value) < 1:
                    raise ValueError("%s must be >= 1, got %r" % (name, value))
            if (int(self.early_stopping_min_iterations)
                    < int(self.early_stopping_window_iterations)):
                raise ValueError(
                    "early_stopping_min_iterations (%d) must be >= "
                    "early_stopping_window_iterations (%d): the first monitored check "
                    "averages a FULL window of completed iterations, so a smaller "
                    "minimum would take that first decision over a partial window."
                    % (int(self.early_stopping_min_iterations),
                       int(self.early_stopping_window_iterations))
                )
            delta = self.early_stopping_min_delta
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                raise ValueError(
                    "early_stopping_min_delta must be a number, got %r" % (delta,))
            if float(delta) < 0.0:
                raise ValueError(
                    "early_stopping_min_delta must be >= 0 (it is the smallest reward "
                    "GAIN that counts as an improvement), got %r" % (delta,))
            earliest = self.early_stopping_earliest_stop_iterations
            if int(self.n_iterations) < earliest:
                raise ValueError(
                    "n_iterations (%d) is shorter than the earliest possible stop (%d = "
                    "min_iterations %d + patience_windows %d x window_iterations %d): "
                    "this run could never reach a stopping decision, so it would record "
                    "an ACTIVE stopping policy whose mechanism was structurally inert. "
                    "Raise n_iterations, or shorten the policy."
                    % (int(self.n_iterations), earliest,
                       int(self.early_stopping_min_iterations),
                       int(self.early_stopping_patience_windows),
                       int(self.early_stopping_window_iterations))
                )

        # --- the construction cell: shape errors RAISE, before any compute ---
        if int(self.num_agents) < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if int(self.n_known) < 1:
            raise ValueError(
                f"n_known must be >= 1 (an episode needs a target), got {self.n_known}"
            )
        if int(self.n_hidden) < 0:
            raise ValueError(f"n_hidden must be >= 0, got {self.n_hidden}")
        if int(self.num_agents) > int(self.n_known):
            raise ValueError(
                "num_agents (%d) must be <= n_known (%d): more agents than targets "
                "forces the stacking cell in which several egos share one target, "
                "every episode returns the same reward, and there is no advantage "
                "signal to learn from."
                % (int(self.num_agents), int(self.n_known))
            )
        if float(self.min_target_distance_km) <= 0.0:
            raise ValueError(
                f"min_target_distance_km must be > 0, got {self.min_target_distance_km}"
            )
        if float(self.min_known_separation_km) < 0.0:
            raise ValueError(
                "min_known_separation_km must be >= 0 (0 disables the constraint), "
                f"got {self.min_known_separation_km}"
            )
        if bool(self.include_sams):
            raise ValueError(
                "include_sams=True is not supported on the construction path: hidden "
                "targets are patched in as enemy AIRBASES, and setup_episode refuses a "
                "world whose enemy units are not all airbases. Mixed SAM / facility / "
                "ship target semantics are a separate design task."
            )

        # --- FD-BASELINE-v1: the difficulty factor's own knobs ---
        # Shape errors RAISE (the parameter object owns the verdicts, so the trainer, the
        # rollout harness and the component itself cannot disagree about what is legal).
        # `fuel_damage_mode` is a TRAINING mode: the forced modes belong to an evaluation
        # pair member and would make every training episode identically conditioned.
        if self.fuel_damage_mode not in _TRAINING_FUEL_DAMAGE_MODES:
            raise ValueError(
                "fuel_damage_mode must be one of %r for a TRAINING run -- %r forces every "
                "training episode into one condition, which is an evaluation group "
                "member, not a mixture."
                % (list(_TRAINING_FUEL_DAMAGE_MODES), self.fuel_damage_mode)
            )
        self.fuel_damage_parameters().validate()
        if float(self.aircraft_penalty_coeff) < 0.0:
            raise ValueError(
                "aircraft_penalty_coeff must be >= 0 (it is a PENALTY subtracted from the "
                "reward numerator), got %r" % (self.aircraft_penalty_coeff,)
            )

        # --- FD hazard: a cell in which the added difficulty cannot be measured ---
        if self.fuel_damage_mode == FuelDamageMode.OFF:
            print("[WARN] fuel_damage_mode=off: FD-BASELINE-v1's difficulty factor is "
                  "DISABLED, so this run reproduces the easy pre-FD cell that was "
                  "learned in two updates. Proceeding.")
        elif float(self.fuel_damage_probability) in (0.0, 1.0):
            print("[WARN] fuel_damage_probability=%r: every training episode gets the "
                  "SAME condition, so the run carries no clean/damaged contrast to "
                  "learn the abort decision from. Proceeding."
                  % float(self.fuel_damage_probability))
        if (self.variable_severity
                and float(self.fuel_damage_mild_probability) in (0.0, 1.0)):
            # The whole point of the variable-severity design is that a damaged episode
            # is NOT reliably severe. At 0 or 1 it is again, and the actor can go back to
            # reading the event itself instead of its own fuel.
            print("[WARN] fuel_damage_mild_probability=%r: every DAMAGED training "
                  "episode gets the same severity, so the run carries no mild/severe "
                  "contrast and the variable-severity factor degenerates to a fixed one. "
                  "Proceeding." % float(self.fuel_damage_mild_probability))
        if float(self.aircraft_penalty_coeff) == 0.0:
            print("[WARN] aircraft_penalty_coeff=0.0: losing an aircraft costs NOTHING, "
                  "so flying the tank dry is never worse than aborting and the "
                  "fuel-damage event creates no decision. Proceeding.")

        # --- construction hazard: the bonmin symmetry stall is driven by n_known ---
        if int(self.n_known) < 3:
            print("[WARN] n_known=%d: fewer than 3 known tasks is the bonmin "
                  "branch-and-bound SYMMETRY-STALL region (~15 min per episode "
                  "observed instead of ~45 s). Proceeding."
                  % int(self.n_known))

        # --- legacy split-surface hazards: WARN, never raise (see the docstring) ---
        lo_n = self._airbase_bounds()[0]
        known, hidden = derived_split(lo_n, self.partial_ratio)
        if known < 3:
            print("[WARN] legacy split surface: n=%d targets, partial_ratio=%r -> "
                  "known/hidden = %d/%d: "
                  "known < 3 is the bonmin branch-and-bound SYMMETRY-STALL region "
                  "(~15 min per episode observed instead of ~45 s). Proceeding."
                  % (lo_n, self.partial_ratio, known, hidden))
        if hidden == 0:
            print("[WARN] legacy split surface: n=%d targets, partial_ratio=%r -> "
                  "known/hidden = %d/%d: "
                  "NO target is hidden, so no pop-up can occur, no "
                  "OPPORTUNISTIC_ENGAGEMENT is reachable, and the episode is a "
                  "degenerate learning target. Proceeding."
                  % (lo_n, self.partial_ratio, known, hidden))

        if not self.eval_enabled:
            return

        if str(self.benchmark_manifest or ""):
            # MANIFEST EVALUATION: the two legacy bounds below are about a schedule this
            # run does not execute, so they are NOT applied here -- and that is a
            # correctness decision, not a relaxation.
            #
            #   * the eval TAG-namespace bound is sized from `eval_episodes`, while a
            #     manifest round's size is the MANIFEST's member count;
            #   * the train/eval band overlap test compares the training band against
            #     `eval_base_seed .. + eval_episodes`, which this run never evaluates.
            #
            # Letting either stand in would be wrong in BOTH directions: an unused
            # configured band could falsely reject a properly held-out manifest, and it
            # could falsely validate one that contains a training seed. The real checks
            # need the manifest itself and therefore run at LOAD time, before the run
            # directory or any compute exists -- `_require_benchmark_seeds_held_out`
            # (the held-out claim) and `_require_benchmark_tag_namespace` (the naming
            # bound). The training-tag bound below is still checked, because benchmark
            # tags share that one namespace.
            # Bounded by the MAXIMUM POSSIBLE attempt count, not by the quota: a
            # training scenario is tagged by its global ATTEMPT ordinal, so replacements
            # push the training tag namespace up to `max_training_attempts`. Identical
            # to `total_episodes` on the fixed-cell path.
            if self.max_training_attempts > _EVAL_EPISODE_TAG_BASE:
                raise ValueError(
                    "maximum training attempts (%d) reaches the eval scenario-tag base "
                    "(%d): training and benchmark scenarios would collide by filename."
                    % (self.max_training_attempts, _EVAL_EPISODE_TAG_BASE)
                )
            return

        # --- the eval scenario-TAG namespace must stay disjoint (see eval_episode_tag)
        # These are artifact-NAMING bounds, not seed bounds, but a violation is the same
        # class of silent loss: one round's scenario JSON overwriting another's.
        # Each held-out seed is attempted ONCE PER MATCHED-GROUP MEMBER per round -- two
        # for the legacy clean/damaged pair, three for a clean/mild/severe triad -- and
        # each member needs its own tag so the worlds coexist as files. The size is taken
        # from THIS config's group, so a variable-severity run is sized for three.
        group_size = self.eval_group_size
        if int(self.eval_episodes) * group_size > _EVAL_ROUND_TAG_STRIDE:
            raise ValueError(
                "eval_episodes (%d) x %d matched %s members (%d tags) exceeds one eval "
                "round's scenario-tag namespace (%d): consecutive eval rounds would write "
                "over each other's scenario files. Raise _EVAL_ROUND_TAG_STRIDE or "
                "shorten the eval band."
                % (int(self.eval_episodes), group_size, self.eval_group_kind,
                   int(self.eval_episodes) * group_size, _EVAL_ROUND_TAG_STRIDE)
            )
        if self.max_training_attempts > _EVAL_EPISODE_TAG_BASE:
            raise ValueError(
                "maximum training attempts (%d) reaches the eval scenario-tag base "
                "(%d): training and eval scenarios would collide by filename."
                % (self.max_training_attempts, _EVAL_EPISODE_TAG_BASE)
            )

        train_lo = int(self.base_seed)
        # THE MAXIMUM POSSIBLE band, so a replacement can never reach an eval seed. On
        # the fixed-cell path -- the only design that reaches this legacy check, because
        # a generalized run with evaluation enabled must use a manifest and returned
        # above -- this is exactly `total_episodes` and the test is byte-unchanged.
        train_hi = train_lo + self.max_training_attempts     # exclusive
        eval_lo = int(self.eval_base_seed)
        eval_hi = eval_lo + int(self.eval_episodes)          # exclusive
        if train_lo < eval_hi and eval_lo < train_hi:        # half-open overlap test
            raise ValueError(
                "training and eval seed bands OVERLAP -- eval would not be held out: "
                f"train=[{train_lo}, {train_hi}) eval=[{eval_lo}, {eval_hi}). "
                "Raise eval_base_seed or shorten the run."
            )


# =============================================================================
# 2. The seeding schedule (pure functions -- unit-testable without a run)
# =============================================================================

# =============================================================================
# 1b. JSON presets -- a run's shape declared in a file, not in a shell history
# =============================================================================

def _config_field_names() -> Tuple[str, ...]:
    """Every :class:`TrainConfig` field name a preset may set."""
    return tuple(f.name for f in dataclass_fields(TrainConfig))


def _ppo_field_names() -> Tuple[str, ...]:
    """Every :class:`PPOConfig` field name a preset's ``"ppo"`` block may set."""
    return tuple(f.name for f in dataclass_fields(PPOConfig))


def _ctde_field_names() -> Tuple[str, ...]:
    """Every :class:`CTDEConfig` field name a preset's ``"ctde"`` block may set."""
    return tuple(f.name for f in dataclass_fields(CTDEConfig))


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON preset and return the :class:`TrainConfig` overrides it declares.

    STDLIB ONLY -- ``json``, no YAML and no new dependency. The file is a flat object of
    ``TrainConfig`` FIELD names, plus optional nested ``"ppo"`` / ``"ctde"`` objects of
    :class:`PPOConfig` / :class:`CTDEConfig` field names::

        {"_comment": "...", "n_iterations": 2, "base_seed": 0, "ppo": {"lr": 0.0003},
         "training_mode": "ctde", "ctde": {"gae_lambda": 0.95}}

    Naming FIELDS rather than CLI flags is deliberate: ``TrainConfig`` is the contract,
    and a second parallel naming scheme would be a second place for the two to drift.

    Three strictnesses, each because the failure it prevents is silent:

      * an UNRECOGNIZED key raises. A misspelled knob that is quietly ignored produces a
        run whose file says one thing and whose behaviour is another -- the config would
        stop describing the measurement;
      * ``"ppo"`` and ``"ctde"`` must be objects, and their keys are checked the same
        way, through ONE loop so a nested block cannot be added with weaker strictness;
      * a list becomes a tuple ONLY for the fields whose dataclass form is a tuple
        (:data:`_CONFIG_TUPLE_FIELDS`), so a preset copied out of a previous run's
        ``run_config.json:/train_config`` loads back into the config it came from.

    Keys beginning with ``_`` are ignored as comments (JSON has none of its own).

    Returns the override mapping ONLY -- it neither constructs nor validates a
    ``TrainConfig``. Resolution against the CLI happens in :func:`resolve_train_config`,
    the one site that knows what "explicit" means.
    """
    cfg_path = Path(path)
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise ValueError("config file not found: %s" % str(cfg_path))
    except json.JSONDecodeError as exc:
        raise ValueError("config file %s is not valid JSON: %s" % (str(cfg_path), exc))
    if not isinstance(raw, dict):
        raise ValueError(
            "config file %s must hold a JSON object of TrainConfig fields, got %s"
            % (str(cfg_path), type(raw).__name__)
        )

    known = set(_config_field_names())
    nested_known = {
        _CONFIG_PPO_KEY: (set(_ppo_field_names()), "PPOConfig"),
        _CONFIG_CTDE_KEY: (set(_ctde_field_names()), "CTDEConfig"),
    }
    values: Dict[str, Any] = {}
    for key, value in raw.items():
        if str(key).startswith(_CONFIG_COMMENT_PREFIX):
            continue
        if key in nested_known:
            # The nested blocks (`ppo`, `ctde`) are checked exactly like the flat keys:
            # object-shaped, and every field name real. One loop for both, so a future
            # third block cannot be added with weaker strictness than its siblings.
            block_known, block_what = nested_known[key]
            if not isinstance(value, dict):
                raise ValueError(
                    "config file %s: %r must be a JSON object of %s fields, got %s"
                    % (str(cfg_path), key, block_what, type(value).__name__)
                )
            block_values = {
                k: v for k, v in value.items()
                if not str(k).startswith(_CONFIG_COMMENT_PREFIX)
            }
            unknown = sorted(set(block_values) - block_known)
            if unknown:
                raise ValueError(
                    "config file %s: unknown %s field(s) %s; known fields are %s"
                    % (str(cfg_path), block_what, unknown, sorted(block_known))
                )
            values[key] = block_values
            continue
        if key not in known:
            raise ValueError(
                "config file %s: unknown TrainConfig field %r; known fields are %s"
                % (str(cfg_path), key, sorted(known))
            )
        if key in _CONFIG_TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        values[key] = value
    return values


def config_source_record(
    *,
    resolved_from: str,
    config_path: Optional[Union[str, Path]] = None,
    config_fields: Optional[List[str]] = None,
    cli_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The ONE construction site of ``run_config.json:/config_source``.

    Always returns a STRUCTURED object -- never ``null`` -- so a reader parses one shape
    and reads ``resolved_from`` to learn which of :data:`_CONFIG_SOURCE_KINDS` produced
    the run. ``null`` would have collapsed two different facts into one value: "this run
    used no preset" and "whoever wrote this file did not record where the config came
    from".

    ``resolved_from`` is REQUIRED and is never inferred. It was briefly derived from
    whether a ``config_path`` was present, which silently reported every direct
    ``train(cfg)`` call -- ``_selftest`` among them -- as ``cli_defaults``, i.e. as
    having been resolved by a command line that never existed. A provenance field that
    can be wrong in a plausible way is worse than one that is absent, so the caller now
    has to say which case it is.

    ``config_fields`` is what a preset supplied; ``cli_overrides`` is what an explicit
    flag then took back off it. An empty ``cli_overrides`` next to a non-empty
    ``config_fields`` is the statement that the run is the preset unmodified.

    The record is checked for INTERNAL consistency before it is returned: only
    ``config_file`` may carry a path, and it must carry one. A record claiming a preset
    it cannot name -- or naming a file while claiming it came from somewhere else -- is
    a defect in the writer, and it fails here rather than being written to disk.
    """
    if resolved_from not in _CONFIG_SOURCE_KINDS:
        raise ValueError(
            "resolved_from must be one of %s, got %r"
            % (list(_CONFIG_SOURCE_KINDS), resolved_from)
        )
    if resolved_from == _CONFIG_SOURCE_FILE and config_path is None:
        raise ValueError(
            "resolved_from=%r requires the config_path it was resolved from"
            % _CONFIG_SOURCE_FILE
        )
    if resolved_from != _CONFIG_SOURCE_FILE and config_path is not None:
        raise ValueError(
            "config_path is only meaningful for resolved_from=%r, got %r with path %s"
            % (_CONFIG_SOURCE_FILE, resolved_from, str(config_path))
        )
    return {
        "path": None if config_path is None else str(config_path),
        "absolute_path": (
            None if config_path is None else str(Path(config_path).resolve())
        ),
        "format": None if config_path is None else "json",
        "config_fields": sorted(config_fields or []),
        "cli_overrides": sorted(cli_overrides or []),
        "resolved_from": str(resolved_from),
    }


def _effective_argv(argv: Optional[List[str]]) -> List[str]:
    """The ONE argv vector a CLI invocation is resolved from.

    ``argparse`` falls back to ``sys.argv[1:]`` when it is handed ``None``, so a caller
    that passes ``None`` to one parse and ``[]`` to another is parsing TWO DIFFERENT
    command lines. That is not hypothetical here: ``main()`` is normally called with no
    argument at all (PyCharm, a terminal, ``python -m ...``), and the override-precedence
    pass would then have seen an EMPTY command line and concluded that the operator typed
    nothing -- letting a preset silently overwrite a flag that was really given. Resolving
    the vector ONCE, here, is what keeps both passes describing the same invocation.
    """
    return list(sys.argv[1:]) if argv is None else list(argv)


def _explicit_cli_dests(argv: Optional[List[str]]) -> set:
    """The set of argparse dests the caller ACTUALLY typed on the command line.

    A parsed namespace cannot answer this on its own -- a flag left out and a flag passed
    its own default value produce the identical value -- and the answer is exactly what
    the override precedence needs: an explicit flag must beat a preset, a default must
    not. Determined by re-parsing the same argv through a THROWAWAY copy of the parser
    whose defaults are all :data:`argparse.SUPPRESS`, which makes argparse omit the
    attribute entirely for anything that was not supplied. The real parser -- and its
    real defaults, which is what ``--help`` must keep showing -- is untouched.

    ``argv=None`` means the REAL command line (:func:`_effective_argv`), exactly as it
    does for ``parser.parse_args``. Reading it as an empty command line would make every
    ordinary invocation -- ``main()`` with no argument, which is how PyCharm and a
    terminal call it -- report that nothing was typed, and a preset would then override
    flags the operator really passed.
    """
    probe = _build_arg_parser()
    for action in probe._actions:      # argparse exposes no public equivalent
        action.default = argparse.SUPPRESS
    return set(vars(probe.parse_args(_effective_argv(argv))))


def resolve_train_config(
    args: argparse.Namespace,
    *,
    explicit: set,
    config_values: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[TrainConfig, Dict[str, Any]]:
    """Resolve dataclass defaults < JSON preset < EXPLICIT CLI flags into one config.

    Three layers, in that order, and only the middle one is new: with no preset this
    reproduces exactly what the CLI built before -- every mapped flag's argparse default,
    which is itself read off :class:`TrainConfig` (drift-guarded by test). A preset only
    chooses among values the command line could already have given.

    ``explicit`` is :func:`_explicit_cli_dests`. A flag in it wins over the preset; a
    flag absent from it does not, even though ``args`` carries a value for it.

    Returns ``(cfg, config_source)``. The second element is the audit record written into
    ``run_config.json`` (:func:`config_source_record`) -- which preset was read, which
    fields it supplied, and which of those a command-line flag then overrode -- so a
    finished run states what produced it instead of leaving a reader to compare numbers
    by eye. It is a structured object for a CLI-only run too, which then states
    ``resolved_from = "cli_defaults"`` and carries no path. This function is a CLI path
    by definition, so it never produces ``direct_config`` -- that value belongs to a
    caller that built a :class:`TrainConfig` in Python (see :func:`config_source_record`).
    """
    values = dict(config_values or {})
    ppo_values = dict(values.pop(_CONFIG_PPO_KEY, {}) or {})
    # The CTDE block has no CLI flags of its own, so it is a preset-only layer: there is
    # nothing for layer 3 to override, and the resolved object is simply the defaults
    # updated by whatever the preset declared.
    ctde_values = dict(values.pop(_CONFIG_CTDE_KEY, {}) or {})

    # Layer 1: the argparse defaults (== the dataclass defaults) for every mapped flag.
    kwargs: Dict[str, Any] = {}
    for dest, field_name in _CLI_FIELD_BY_DEST.items():
        kwargs[field_name] = getattr(args, dest)
    ppo_kwargs: Dict[str, Any] = {}
    for dest, field_name in _CLI_PPO_FIELD_BY_DEST.items():
        ppo_kwargs[field_name] = getattr(args, dest)
    # `--iterations` has no default: absent, it must come from the preset (or fail).
    if kwargs.get("n_iterations") is None:
        kwargs.pop("n_iterations", None)

    # Layer 2: the preset. It may also set fields no flag exposes (e.g. `max_ticks`).
    kwargs.update(values)
    ppo_kwargs.update(ppo_values)

    # Layer 3: explicit command-line flags, which beat the preset.
    overridden: List[str] = []
    for dest in sorted(explicit):
        if dest in _CLI_FIELD_BY_DEST:
            field_name = _CLI_FIELD_BY_DEST[dest]
            kwargs[field_name] = getattr(args, dest)
            if field_name in values:
                overridden.append(field_name)
        elif dest in _CLI_PPO_FIELD_BY_DEST:
            field_name = _CLI_PPO_FIELD_BY_DEST[dest]
            ppo_kwargs[field_name] = getattr(args, dest)
            if field_name in ppo_values:
                overridden.append("%s.%s" % (_CONFIG_PPO_KEY, field_name))

    if kwargs.get("n_iterations") is None:
        raise ValueError(
            "n_iterations is not set: pass --iterations, or declare it in the --config "
            "preset. How long to train is the one decision that is never defaulted."
        )

    cfg = TrainConfig(
        ppo=PPOConfig(**ppo_kwargs), ctde=CTDEConfig(**ctde_values), **kwargs
    )
    # This function is reached only from a COMMAND LINE, so the kind is one of the two
    # CLI values -- which of them is exactly whether a preset was named.
    config_source = config_source_record(
        resolved_from=(
            _CONFIG_SOURCE_CLI_DEFAULTS if config_path is None else _CONFIG_SOURCE_FILE
        ),
        config_path=config_path,
        config_fields=(
            list(values)
            + ["%s.%s" % (_CONFIG_PPO_KEY, k) for k in ppo_values]
            + ["%s.%s" % (_CONFIG_CTDE_KEY, k) for k in ctde_values]
        ),
        cli_overrides=overridden,
    )
    return cfg, config_source


def global_episode_index(cfg: TrainConfig, iteration: int, j: int) -> int:
    """``g = iteration * episodes_per_iteration + j`` -- the run-wide episode index."""
    return int(iteration) * int(cfg.episodes_per_iteration) + int(j)


def train_seed(cfg: TrainConfig, iteration: int, j: int) -> int:
    """Seed of training episode ``j`` of ``iteration``: ``base_seed + g``."""
    return int(cfg.base_seed) + global_episode_index(cfg, iteration, j)


def train_attempt_seed(cfg: TrainConfig, attempt_ordinal: int) -> int:
    """Seed of the run's ``attempt_ordinal``-th training ATTEMPT: ``base_seed + n``.

    THE ONE SEED FORMULA UNDER BOTH POLICIES, expressed over the quantity that is
    monotone in both: a run-wide attempt ordinal that advances on EVERY attempt, whether
    it succeeded or failed. That is what makes a replacement deterministic (it is simply
    the next ordinal), makes a failed seed SPENT (no later attempt can revisit it), and
    keeps every artifact tag unique even when an iteration takes more attempts than it
    collects episodes.

    It is a GENERALIZATION of :func:`train_seed`, not a competitor to it: when every
    scheduled slot is attempted exactly once -- which is the whole of the fixed-cell
    policy -- the run's ordinal at iteration ``i`` slot ``j`` is ``i *
    episodes_per_iteration + j``, so ``train_attempt_seed(cfg,
    global_episode_index(cfg, i, j)) == train_seed(cfg, i, j)`` identically. The
    historical loop therefore keeps calling :func:`train_seed` and is unchanged.
    """
    n = int(attempt_ordinal)
    if n < 0:
        raise ValueError("attempt_ordinal must be >= 0, got %d" % n)
    return int(cfg.base_seed) + n


def eval_seed(cfg: TrainConfig, e: int) -> int:
    """Seed of eval episode ``e``: ``eval_base_seed + e`` -- FIXED across rounds."""
    return int(cfg.eval_base_seed) + int(e)


def eval_episode_tag(*, round_ordinal: int, e: int) -> int:
    """Scenario TAG for eval episode ``e`` of eval round ``round_ordinal``.

    A NAME, NOT A SEED, and the distinction is the whole point of this function. The
    held-out band is fixed: eval episode ``e`` runs :func:`eval_seed` on every round, so
    the same world is re-measured as the policy changes. What must NOT be fixed is the
    FILE that world is written to -- with one tag per episode index, every round wrote
    ``episode_900000_scenario.json`` again and the earlier rounds' scenario artifacts
    were destroyed as the run progressed. A finished run could then no longer show which
    world any round but the last had actually run on.

    Round ordinal ``r`` therefore owns the half-open tag band
    ``[base + r*stride, base + (r+1)*stride)``: ``pre_update`` is ordinal 0 and each
    later ``post_update`` round takes the next ordinal, so the three tag sets (training,
    pre-update, post-update round k) are pairwise DISJOINT by construction.

    ``ScenarioGenerator.generate`` consumes ``episode`` only after every rng draw, in the
    single step that sets ``scenario["name"]`` and the output filename. Nothing here can
    reach seed derivation, the generated geometry, the policy input, action sampling or
    the reward.

    Raises:
        ValueError: on a negative ordinal, or on ``e`` outside one round's stride --
            which would let round ``r`` reach into round ``r+1``'s band and reintroduce
            exactly the overwrite this exists to prevent. :meth:`TrainConfig.validate`
            refuses such a config up front; this is the second, local guard.
    """
    r = int(round_ordinal)
    i = int(e)
    if r < 0:
        raise ValueError("round_ordinal must be >= 0, got %d" % r)
    if not (0 <= i < _EVAL_ROUND_TAG_STRIDE):
        raise ValueError(
            "eval episode index %d does not fit one round's tag namespace of %d: "
            "rounds would overwrite each other's scenario files."
            % (i, _EVAL_ROUND_TAG_STRIDE)
        )
    return _EVAL_EPISODE_TAG_BASE + r * _EVAL_ROUND_TAG_STRIDE + i


def eval_member_tag(
    *, round_ordinal: int, e: int, member: int, group_size: int = _EVAL_PAIR_SIZE
) -> int:
    """Scenario TAG for ONE member of held-out episode ``e``'s matched group.

    FD-BASELINE-v1 evaluates every held-out geometry TWICE per round -- once forced clean
    and once forced damaged -- on the SAME seed, so that the reward difference is
    attributable to the event and not to the world. FD-VARIABLE-SEVERITY-v1 evaluates it
    THREE times, adding a mild and a severe member in place of the single damaged one.
    Every member would otherwise be written to the same
    ``episode_<tag>_scenario.json`` and the later ones would destroy the first's
    artifact, which is the same silent loss :func:`eval_episode_tag` exists to prevent
    one level up.

    So each member takes its own slot inside the round's namespace: episode ``e``'s
    members occupy ``e * group_size + m``. The SEEDS are untouched -- every member runs
    :func:`eval_seed` of ``e`` -- and that is the entire point: identical geometry,
    disjoint artifacts. :meth:`TrainConfig.validate` sizes the namespace for the run's
    own group up front; :func:`eval_episode_tag`'s own range guard is the second line of
    defence.

    ``group_size`` defaults to the legacy PAIR width so an existing caller (and an
    existing record's tag arithmetic) is unchanged. A caller that evaluates triads passes
    3; the two layouts are different tag allocations of the same namespace and are never
    mixed within one run.
    """
    m = int(member)
    size = int(group_size)
    if size < 1:
        raise ValueError("eval group size must be >= 1, got %d" % size)
    if not (0 <= m < size):
        raise ValueError("eval group member must be in [0, %d), got %d" % (size, m))
    return eval_episode_tag(round_ordinal=round_ordinal, e=int(e) * size + m)


def cell_condition(cell: str) -> str:
    """The CONDITION a reporting cell belongs to -- ``clean`` or ``damaged``.

    A severity cell (``mild`` / ``severe``) IS a damaged episode, so this is the mapping
    that lets every clean/damaged count keep its existing meaning while the finer cells
    are reported alongside. One site, so pooling can never be spelled two ways.
    """
    return CONDITION_DAMAGED if str(cell) in SEVERITIES else str(cell)


def _outcome_cell(plan_record: Dict[str, Any]) -> str:
    """The cell a COMPLETED episode is reported under, from its own plan record.

    Read off the plan the episode really ran with rather than re-derived from the seed,
    so a successful attempt is always counted under the event it actually contained.
    """
    severity = (plan_record or {}).get("severity")
    return (str(severity) if severity
            else str((plan_record or {}).get("condition", CONDITION_CLEAN)))


def _delta_key(cell: str, reference: str) -> str:
    """The flat record key carrying the ``cell - reference`` within-seed difference."""
    return "eval_delta_%s_minus_%s" % (str(cell), str(reference))


# =============================================================================
# 3. Small helpers (stdlib only)
# =============================================================================

def _stats_or_none(values: List[float]) -> Dict[str, Optional[float]]:
    """(mean, min, max), or ``None`` for each on an EMPTY population.

    THE anti-false-zero helper. It replaced a zero-filling variant outright rather than
    sitting next to one, because two aggregators differing only in what they do with an
    empty list is exactly the pair that gets mixed up. The reward is oracle-normalized
    regret, so ``0.0`` is the perfect-information OPTIMUM -- the single best value an
    episode can report. A batch or an eval round in which every scheduled attempt FAILED
    has no reward population at all, and summarizing that absence as ``0.0`` would plot
    a total data loss as a perfect score.

    ``None`` (JSON ``null``) says "not measured" and cannot be confused with a number.
    A SUCCESSFUL zero-wake episode is a different thing entirely: it has a real reward
    and is part of the population here.
    """
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _fraction(numerator: int, denominator: int) -> Optional[float]:
    """``numerator / denominator``, or ``None`` when the denominator is 0.

    Same rule as :func:`_stats_or_none`: an undefined fraction is reported as missing,
    never as ``0.0`` (which would read as "0% succeeded" where nothing was attempted).
    """
    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _fmt_opt(value: Optional[float], spec: str = "%+.4f") -> str:
    """Format an optional number for the ASCII console: ``None`` -> ``"n/a"``."""
    return "n/a" if value is None else (spec % float(value))


# =============================================================================
# 3b. Episode-attempt failures: attribution + the durable ledger
# =============================================================================

class EpisodeAttemptError(RuntimeError):
    """One episode attempt failed, tagged with the PIPELINE STAGE it failed in.

    A thin attribution wrapper, not a new failure mode. ``_run_one_episode`` drives four
    stages in order -- ``generation`` -> ``setup`` -> ``run`` -> ``reward`` -- and the
    ledger is only useful if it says WHICH one broke: an exact-cardinality construction
    failure (``setup``) and an engine edge case (``run``) are different findings about
    the run, and a summary that lumped them together would hide the one B4 exists to
    account for.

    The original exception is preserved BOTH as ``__cause__`` (``raise ... from``) and as
    :attr:`original`, so the ledger records the real type and message rather than this
    wrapper's. Callers that only ever caught ``Exception`` are unaffected.
    """

    def __init__(self, stage: str, original: BaseException) -> None:
        super().__init__("%s failed: %s: %s"
                         % (stage, type(original).__name__, original))
        self.stage = str(stage)
        self.original = original


def _v2_failure_population(
    pre_solve: Optional[PreSolveCardinality],
    route_relative_load: Optional[RouteRelativeHiddenLoad],
) -> Dict[str, Any]:
    """The GENERALIZED-V2 population block of a FAILED attempt -- or ``{}`` elsewhere.

    A FAILED ATTEMPT IS STILL PART OF THE ATTEMPTED POPULATION, so its record has to state
    the identity it ACTUALLY RECEIVED. V2 resolves that identity in two stages, and an
    attempt can die in either, so the block is STAGE-AWARE rather than uniform:

      * ``pre_solve`` -- the attempt failed before the known-only solve produced a route
        count, so there was no hidden load to draw. ``route_count_at_hidden_resolution``
        and every hidden-load field are ``null``: NOT a writer that forgot them, and NOT a
        stage-2 record to be reconstructed later from the seed. Nothing is fabricated.
      * ``route_relative`` -- the attempt HAD resolved ``R`` and ``H_requested`` before it
        failed, so the exact draw it received is carried through VERBATIM from the frozen
        :class:`RouteRelativeHiddenLoad` the construction path produced.

    It is deliberately never re-derived. ``H`` is reproducible from the seed and ``R``, so
    a post-hoc redraw would usually agree -- and "usually" is exactly the property a ledger
    must not rest on. The ledger says what the attempt resolved, not what a replay would.

    KEYED OFF THE STAGE-1 RECORD, which only the V2 path ever produces, so the block is
    added ONLY there: a ``fixed_cell_v1`` or ``generalized_v1`` failure record grows no key
    at all and keeps the shape every existing reader and preserved artifact already has.
    Both halves of the two-stage identity are carried, so a Case-B record still names the
    stage-1 draw that produced its ``A`` and ``K``.
    """
    if pre_solve is None:
        return {}
    if str(pre_solve.source) not in (
            CARDINALITY_SOURCE_V2_PRE_SOLVE,
            CARDINALITY_SOURCE_V2_BENCHMARK_PRE_SOLVE):  # pragma: no cover
        # Defensive: only the V2 stage-1 sampler and the V2 benchmark base cell mint this
        # record, so a foreign source here would mean the block was about to describe a
        # population it did not come from. Refusing to emit is the truthful response.
        return {}
    load = route_relative_load
    return {
        "generalized_v2_population": {
            # WHICH STAGE this attempt reached before it failed. Stated outright so a
            # reader never has to infer it from which fields happen to be null.
            "stage_resolved": (
                "route_relative" if load is not None else "pre_solve"
            ),
            # Read off the stage-1 record rather than restated as constants: a training
            # draw carries the sampler's policy and domain, a benchmark base cell its own
            # policy and NO domain (nothing was drawn).
            "pre_solve_cardinality_policy": str(pre_solve.policy),
            "pre_solve_rng_domain": (
                None if pre_solve.rng_domain is None else str(pre_solve.rng_domain)
            ),
            "pre_solve_derived_seed": pre_solve.derived_seed,
            "agent_count": int(pre_solve.agent_count),
            "known_requested": int(pre_solve.known_count),
            # The EXACT stage-2 facts, or `null` because stage 2 never happened.
            "hidden_load_policy": None if load is None else str(load.policy),
            "hidden_load_rng_domain": None if load is None else str(load.rng_domain),
            "hidden_load_derived_seed": None if load is None else load.derived_seed,
            "route_count_at_hidden_resolution": (
                None if load is None else int(load.route_count)
            ),
            "hidden_load": None if load is None else load.to_record(),
        },
    }


def _failure_record(
    *,
    phase: str,
    evaluation_stage: Optional[str],
    updates_completed: int,
    iteration: Optional[int],
    attempt_ordinal: int,
    episode_index: Optional[int],
    eval_tag: Optional[str],
    seed: int,
    condition: str,
    exc: BaseException,
    cell: Optional[str] = None,
    cardinality: Optional[Union[EpisodeCardinality, PreSolveCardinality]] = None,
    pre_solve_cardinality: Optional[PreSolveCardinality] = None,
    route_relative_load: Optional[RouteRelativeHiddenLoad] = None,
    benchmark: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build ONE ledger record for a failed attempt (see :func:`_append_failure_record`).

    Every field a post-hoc audit needs to place the attempt exactly: which phase and
    (for eval) which stage, how much learning had happened when it was attempted, its
    position in the SCHEDULE, its identity, its exact seed, the stage it died in, and the
    original exception with its traceback.

    ``seed`` is the scheduled seed, recorded even though it produced nothing -- a failed
    seed stays part of the attempted population and must remain visible.

    ``condition`` is the SCHEDULED fuel-damage condition (``clean`` / ``damaged``). It is
    resolvable without a world -- :func:`resolve_condition` is a pure function of the seed
    and the mode -- which is exactly why it can be recorded for an attempt that never
    produced an episode. Without it, "failed counts by condition" would be unanswerable
    and a per-condition mean could quietly be taken over a different denominator than it
    appears to have.

    ``cell`` is the finer SCHEDULED reporting label (``clean`` / ``mild`` / ``severe``),
    beside -- never instead of -- ``condition``: the existing ``failures_by_condition``
    tally keeps meaning exactly what it always did, and the severity is added so a
    per-CELL denominator is complete rather than reconstructed.

    GENERALIZED-V1. ``cardinality`` and ``benchmark`` record the world this attempt was
    SCHEDULED to build, and they matter precisely because it never built one: without them
    a failed HIGH-load attempt would be invisible in the requested-vs-realized
    distribution and a failed benchmark member would be missing from its stratum's
    denominator, so both would silently shrink. A failure is still recorded ONCE, still
    never retried, and still never replaced.

    ``reference_fault_reason`` is the stable slug of a reference refusal that was
    ACCOUNTED as attrition (an unanswered solve). A reference fault that ABORTED never
    reaches this ledger at all -- that is the whole point of the routing -- so the field
    can only ever carry the attrition case.
    """
    original = getattr(exc, "original", exc)
    stage = getattr(exc, "stage", "unknown")
    return {
        "phase": str(phase),
        "evaluation_stage": evaluation_stage,
        "updates_completed": int(updates_completed),
        "iteration": None if iteration is None else int(iteration),
        "attempt_ordinal": int(attempt_ordinal),
        "episode_index": None if episode_index is None else int(episode_index),
        "eval_tag": eval_tag,
        "seed": int(seed),
        "condition": str(condition),
        "cell": None if cell is None else str(cell),
        "pipeline_stage": str(stage),
        "error_type": type(original).__name__,
        "error_message": str(original),
        "reference_fault_reason": getattr(original, "reason", None) if isinstance(
            original, ReferenceIntegrityError) else None,
        # The SCHEDULED world shape (never a realized one -- nothing was realized).
        #
        # Under GENERALIZED-V2 this is STAGE-AWARE, because the population identity itself
        # is resolved in two stages and an attempt can fail in either. `cardinality` is
        # whichever object the attempt REALLY got: the stage-1 half-cell when it died
        # before the known-only solve produced a route count, and the RESOLVED cell --
        # built from the actual stage-2 draw -- when it died after. `cardinality_source`
        # says which, so `hidden_requested = null` is read as "no hidden load was ever
        # drawn" rather than as a writer that forgot the field, and a resolved record is
        # never a post-hoc reconstruction of one.
        "agent_count": None if cardinality is None else int(cardinality.agent_count),
        "known_requested": None if cardinality is None else int(
            cardinality.known_count),
        "hidden_requested": (
            None if cardinality is None
            else getattr(cardinality, "hidden_requested", None)
        ),
        "cardinality_source": None if cardinality is None else str(cardinality.source),
        # V2-ONLY, and absent entirely on every other design (`_v2_failure_population`).
        **_v2_failure_population(pre_solve_cardinality, route_relative_load),
        **(benchmark or _EMPTY_BENCHMARK_KEYS),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _append_failure_record(path: Optional[Path], record: Dict[str, Any]) -> None:
    """Append ONE record to ``episode_failures.jsonl`` and flush it immediately.

    Opened in append mode per record and flushed before returning, so the ledger is
    durable at the moment of the failure: a run killed by the next episode still leaves a
    complete account of everything that failed before it. ``path=None`` disables the
    ledger (used by callers that have no run directory).
    """
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _episode_outcome_record(
    out: "_EpisodeOutcome",
    *,
    phase: str,
    iteration: Optional[int],
    updates_completed: int,
    updates_completed_before: Optional[int],
    attempt_ordinal: int,
    episode_index: Optional[int],
    eval_round_ordinal: Optional[int],
    eval_episode_index: Optional[int],
    eval_group_member: Optional[int],
    seed: int,
    episode_tag: int,
    fuel_damage_mode: str,
    design: EpisodeDesign,
    benchmark: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """ONE durable record per SUCCESSFUL attempt, for ``episode_outcomes.jsonl``.

    WHY THIS FILE EXISTS. The per-iteration and per-round records are AGGREGATES: they
    say a round's mild episodes averaged some reward and produced some abort rate, which
    is enough to plot a curve and not enough to inspect a distribution. The question this
    experiment is built to answer -- did the actor respond DIFFERENTLY to a survivable
    fuel loss than to an unsurvivable one, and in which worlds -- is per episode, and an
    aggregate cannot be un-averaged afterwards. So every completed attempt states its own
    identity, its own event and its own outcome, once, in one canonical stream.

    IT DOES NOT DUPLICATE THE LEDGER. A FAILED attempt is written to
    ``episode_failures.jsonl`` under ``skip_and_account_v1`` and appears here NOT AT ALL;
    the two files are disjoint by construction, so no attempt can be counted twice by
    reading both.

    MISSING IS ``null``, NEVER ``0``. A clean episode has no fuel reading, an event that
    did not fire has no tick, and a wake that did not happen has no meta-action -- each
    of those is an absence, and a zero would read as a measurement (an empty tank, tick
    zero, ``PLAN_COMPLIANCE``). Every FD number here is copied verbatim from the
    component's own frozen records (:meth:`FuelDamagePlan.to_record` /
    :meth:`FuelDamageOutcome.to_record`); nothing is recomputed, so this stream cannot
    disagree with the aggregate that summarizes it.

    GENERALIZED-V1 (schema version 2). The record additionally states the resolved
    episode DESIGN and its four policy ids, the REQUESTED-vs-REALIZED cardinality, and the
    per-episode structures Tasks 1-3 produce -- the construction audit, the FD eligibility
    audit and event certificate, the post-FD adaptation record and the reward-reference
    decomposition -- each carried WHOLE from the component's own frozen record rather than
    flattened into a second description of it. A member of a frozen benchmark also states
    its stratum, its matched world GROUP and the id-free identity of the world it built,
    so a stratum's denominator and a group's completeness are both readable per episode.

    ``design`` is a REQUIRED keyword: both production call sites resolve it before the
    episode runs, and an optional one would let a future call site omit the one field
    that says which population this episode was drawn from.

    THE SAME "MISSING IS null" RULE APPLIES TO THE NEW BLOCKS, and there it is a statement
    about the DESIGN rather than about the episode: an ``exact_v1`` construction genuinely
    has no backoff audit, a legacy plan has no certificate, a ``single_wake_v1`` run has
    no post-FD adaptation, and a ``static_t0_v1`` episode has no reference object.
    ``null`` there means "this design produces no such structure", never "it measured
    zero".
    """
    plan = out.fuel_damage_plan or {}
    outcome = out.fuel_damage_outcome or {}
    meta = outcome.get("wake_meta_action")
    card = out.cardinality
    breakdown = out.reward_breakdown or {}
    reference = out.reference or {}
    eligibility = plan.get("eligibility_audit") or {}
    post_fd = out.post_fd_adaptation or {}
    return {
        "schema": _EPISODE_OUTCOME_SCHEMA,
        "schema_version": _EPISODE_OUTCOME_VERSION,
        # The action semantics every selected action and probability below is stated in.
        "action_representation_id": ACTION_REPRESENTATION_ID,
        # --- WHICH POPULATION this episode was drawn from, and under which policies ---
        # Stated on BOTH designs rather than only on the generalized one: "this run was
        # the historical fixed cell" is a fact worth recording, and a reader must never
        # have to infer the design from the ABSENCE of generalized keys.
        "episode_design": str(design.design),
        "generalized": bool(design.generalized),
        "hidden_policy": str(design.hidden_policy),
        "eligibility_policy": str(design.eligibility_policy),
        "post_fd_wake_policy": str(design.post_fd_wake_policy),
        "reference_policy": str(design.reference_policy),
        "target_destruction_probability": float(TARGET_DESTRUCTION_PROBABILITY),
        # --- identity: which scheduled attempt this was ---
        "phase": str(phase),
        "iteration": None if iteration is None else int(iteration),
        "updates_completed": int(updates_completed),
        "updates_completed_before": (
            None if updates_completed_before is None else int(updates_completed_before)
        ),
        "attempt_ordinal": int(attempt_ordinal),
        "episode_index": None if episode_index is None else int(episode_index),
        "eval_round_ordinal": (
            None if eval_round_ordinal is None else int(eval_round_ordinal)
        ),
        "eval_episode_index": (
            None if eval_episode_index is None else int(eval_episode_index)
        ),
        "eval_group_member": (
            None if eval_group_member is None else int(eval_group_member)
        ),
        "seed": int(seed),
        "episode_tag": int(episode_tag),
        "fuel_damage_mode": str(fuel_damage_mode),
        # --- the cell this episode is reported under, and its two components ---
        "cell": _outcome_cell(plan),
        "condition": plan.get("condition"),
        "severity": plan.get("severity"),
        # --- the event: what was planned, and what really happened ---
        # Both derived seeds travel with the record so the draws that produced this
        # episode can be reproduced from the artifact alone, without the run's config.
        "fd_derived_seed": plan.get("derived_seed"),
        "fd_severity_derived_seed": plan.get("severity_derived_seed"),
        "fd_target_policy": plan.get("target_policy"),
        "fd_ego_id": plan.get("ego_id"),
        "fd_fired": outcome.get("fired"),
        "fd_event_tick": outcome.get("event_tick"),
        "fd_observed_progress": outcome.get("observed_progress"),
        "fd_fuel_before": outcome.get("fuel_before"),
        "fd_fuel_after": outcome.get("fuel_after"),
        "fd_damage_factor": outcome.get("damage_factor"),
        "fd_fuel_after_fraction_of_max": outcome.get("fuel_after_fraction_of_max"),
        "fd_max_fuel": outcome.get("max_fuel"),
        # The LIVE bounds the mutation was really validated against, and the margin that
        # physically separates mild from severe: positive => continuing stays feasible.
        "fd_live_rtb_fuel_floor": outcome.get("live_rtb_fuel_floor"),
        "fd_live_continue_fuel_requirement": outcome.get(
            "live_continue_fuel_requirement"),
        "fd_continuation_margin": outcome.get("continuation_margin"),
        "fd_live_band_low": outcome.get("live_band_low"),
        "fd_live_band_high": outcome.get("live_band_high"),
        # The PLANNED bounds, kept under their own names so a reader always knows which
        # window a number came from (see `_fuel_damage_lines`).
        "fd_planned_rtb_fuel_floor": plan.get("rtb_fuel_floor"),
        "fd_planned_continue_fuel_requirement": plan.get("continue_fuel_requirement"),
        # --- the behavioural measurement ---
        "fd_wake_occurred": outcome.get("wake_occurred"),
        "fd_wake_meta_action": meta,
        "fd_wake_meta_action_name": None if meta is None else MetaAction(int(meta)).name,
        # COMMAND HISTORY, not the executor's lifecycle latch.
        "fd_rtb_command_issued": out.selected_ego_rtb_issued,
        # --- the episode's outcome ---
        "reward": float(out.reward),
        "n_dead": int(out.n_dead),
        "targets_confirmed_unique": int(out.targets_confirmed_unique),
        "targets_total": int(out.targets_total),
        "target_confirmation_count_semantics": _TARGET_CONFIRMATION_SEMANTICS,
        "n_wakes": int(out.n_wakes),
        "ended": str(out.ended),
        "ticks": int(out.ticks),
        "seconds": float(out.seconds),
        # --- CONSTRUCTION: requested vs realized, and the backoff's own reasons -----
        # The REQUEST is never rewritten to match what was realized: the two travel side
        # by side so requested-vs-realized is readable as a DISTRIBUTION across episodes,
        # which is exactly the inspection a HIGH hidden-load stratum that quietly
        # collapses into the LOW one would otherwise escape (handoff 3l.6).
        "agent_count": None if card is None else int(card.agent_count),
        "known_requested": None if card is None else int(card.known_count),
        "hidden_requested": None if card is None else int(card.hidden_requested),
        "targets_requested": None if card is None else int(card.targets_requested),
        "cardinality_source": None if card is None else str(card.source),
        "known_realized": (
            None if out.world_identity is None
            else int(out.world_identity.known_realized)
        ),
        "hidden_realized": out.hidden_realized,
        "targets_realized": int(out.targets_total),
        "hidden_short_realized": (
            None if card is None or out.hidden_realized is None
            else bool(int(out.hidden_realized) < int(card.hidden_requested))
        ),
        # --- GENERALIZED-V2: the TWO-STAGE population, present only under that design -
        # A nested block added ONLY on the route-relative path, under the same discipline
        # every other design-specific structure follows: a `fixed_cell_v1` or
        # `generalized_v1` record grows NO new key, so the ABSENCE of this block is how a
        # reader tells the hidden count was stated up front rather than drawn against a
        # realized route count. Every number in it is copied VERBATIM from the stage
        # records the population layer and the construction path produced -- nothing is
        # recomputed here, so this stream cannot disagree with them.
        **({} if out.route_relative_load is None else {
            "generalized_v2_population": {
                "hidden_load_policy": str(out.route_relative_load.policy),
                "pre_solve_cardinality_policy": (
                    None if out.pre_solve_cardinality is None
                    else str(out.pre_solve_cardinality.policy)
                ),
                "pre_solve_rng_domain": (
                    None if (out.pre_solve_cardinality is None
                             or out.pre_solve_cardinality.rng_domain is None)
                    else str(out.pre_solve_cardinality.rng_domain)
                ),
                "pre_solve_derived_seed": (
                    None if out.pre_solve_cardinality is None
                    else out.pre_solve_cardinality.derived_seed
                ),
                "hidden_load_rng_domain": str(out.route_relative_load.rng_domain),
                "hidden_load_derived_seed": out.route_relative_load.derived_seed,
                "route_count_at_hidden_resolution": int(
                    out.route_relative_load.route_count),
                "pre_solve": (
                    None if out.pre_solve_cardinality is None
                    else out.pre_solve_cardinality.to_record()
                ),
                "hidden_load": out.route_relative_load.to_record(),
            },
        }),
        "construction_audit": out.construction_audit,
        "construction_backoff_candidate_order": (
            (out.construction_audit or {}).get("backoff", {}).get("candidate_order")
        ),
        "construction_backoff_rejections": _backoff_rejections(out.construction_audit),
        "hidden_geometric_fingerprint": (
            None if out.world_identity is None
            else [list(pair) for pair in out.world_identity.geometric_fingerprint]
        ),
        # --- FD ELIGIBILITY + the event CERTIFICATE --------------------------------
        "fd_eligibility_policy": plan.get("eligibility_policy"),
        "fd_post_fd_wake_policy": plan.get("post_fd_wake_policy"),
        "fd_eligibility_rng_domain": eligibility.get("rng_domain"),
        "fd_eligibility_derived_seed": eligibility.get("derived_seed"),
        "fd_eligibility_candidate_count": eligibility.get("candidate_count"),
        "fd_eligibility_candidate_order": eligibility.get("candidate_order"),
        "fd_eligibility_considered_ordinals": eligibility.get("considered_ordinals"),
        "fd_eligibility_rejections": _eligibility_rejections(eligibility),
        "fd_eligibility_selected_ordinal": eligibility.get("selected_ordinal"),
        "fd_eligibility_audit": plan.get("eligibility_audit"),
        "fd_certificate": plan.get("certificate"),
        "fd_certificate_fingerprint": (
            None if out.world_identity is None
            else out.world_identity.fd_certificate_fingerprint
        ),
        # --- POST-FD ADAPTATION, under its OWN denominators ------------------------
        # Never folded into the immediate-wake pair above: `fd_wake_occurred` /
        # `fd_wake_meta_action` keep meaning the IMMEDIATE fuel-damage wake, and an
        # approved measurement is reported over exactly those two.
        "post_fd_policy": post_fd.get("policy"),
        "post_fd_armed": post_fd.get("armed"),
        "post_fd_active": post_fd.get("active"),
        "post_fd_deactivation_reason": post_fd.get("deactivation_reason"),
        "post_fd_boundaries_confirmed": post_fd.get("boundaries_confirmed"),
        "post_fd_boundaries_with_remaining_mission": post_fd.get(
            "boundaries_with_remaining_mission"),
        "post_fd_boundaries_terminal": post_fd.get("boundaries_terminal"),
        "post_fd_boundary_wakes": post_fd.get("boundary_wakes"),
        "post_fd_boundary_ticks": post_fd.get("boundary_ticks"),
        "post_fd_boundary_meta_actions": post_fd.get("boundary_meta_actions"),
        "post_fd_boundary_meta_action_names": [
            MetaAction(int(m)).name
            for m in (post_fd.get("boundary_meta_actions") or [])
        ],
        "post_fd_adaptation": out.post_fd_adaptation,
        # --- THE REWARD-BEARING REFERENCE and the reward's decomposition -----------
        # `u_ref` is the denominator source under BOTH policies (on the static path it
        # EQUALS `u_oracle`), so a consumer asking "what was this normalized by?" reads
        # ONE field and is correct either way. `u_oracle` is `null` under the opt-in
        # policy because no static full-set optimum was ever solved -- `0.0` there would
        # fabricate a perfect oracle.
        "reference_kind": reference.get("kind") or breakdown.get("reference_kind"),
        "reference_checkpoint_tick": breakdown.get("checkpoint_tick"),
        "u_achieved": breakdown.get("u_achieved"),
        "u_oracle": breakdown.get("u_oracle"),
        "u_ref": breakdown.get("u_ref"),
        "u_prefix": breakdown.get("u_prefix"),
        "u_cont_ref": breakdown.get("u_cont_ref"),
        "u_post": breakdown.get("u_post"),
        "u_aircraft": breakdown.get("u_aircraft"),
        "reward_ratio": breakdown.get("ratio"),
        "reward_penalty": breakdown.get("penalty"),
        "unique_completed_targets": breakdown.get("unique_completed_targets"),
        "scored_completed_targets": breakdown.get("scored_completed_targets"),
        "unscored_completed_targets": breakdown.get("unscored_completed_targets"),
        "unscored_completed_target_ids": list(
            breakdown.get("unscored_completed_target_ids") or ()
        ),
        "reference_allocated_task_count": reference.get("allocated_task_count"),
        "reference_candidate_task_count": reference.get("candidate_task_count"),
        "reference_continuation_agent_count": (
            None if not reference
            else len(reference.get("continuation_agent_ids") or ())
        ),
        "reference_excluded_agents": reference.get("excluded_agents"),
        "reference_solver_invoked": reference.get("solver_invoked"),
        "reference_solver_accepted": reference.get("solver_accepted"),
        "reference_solver_termination": reference.get("solver_termination"),
        "reference_solver_seconds": reference.get("solver_seconds"),
        "reference_record": out.reference,
        # --- PER-WAKE ACTOR DIAGNOSTICS (schema v3) --------------------------------
        # One entry per recorded wake, in trajectory order, so future runs can be
        # diagnosed WITHOUT an offline checkpoint replay. A successful ZERO-WAKE
        # episode records `[]` -- a real, legitimate outcome of the event-triggered
        # design, and deliberately not `null`, which would read as "not recorded".
        "wake_diagnostics_schema_version": _WAKE_DIAGNOSTICS_VERSION,
        "n_wake_decisions": len(_wake_decision_records(out.trajectory)),
        "wake_decisions": _wake_decision_records(out.trajectory),
        # --- FROZEN-BENCHMARK identity (a manifest-driven eval member only) --------
        **(benchmark or _EMPTY_BENCHMARK_KEYS),
    }


#: The benchmark identity keys, present as ``null`` on every NON-benchmark episode so one
#: schema reads both. A key that appears on only some rows makes a jsonl stream awkward to
#: load into a table, and its ABSENCE is indistinguishable from a writer that forgot it.
_EMPTY_BENCHMARK_KEYS: Dict[str, Any] = {
    "benchmark_manifest_id": None,
    "benchmark_stratum": None,
    "benchmark_group_key": None,
    "benchmark_agent_count": None,
    "benchmark_load_bucket": None,
    "benchmark_world_ordinal": None,
    "benchmark_world_identity": None,
}


def _backoff_rejections(audit: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """The bounded-backoff candidates' REJECTION REASONS, in visit order.

    A flat list of stable slugs beside the full audit, so "why did this world realize
    fewer hidden targets than it requested?" can be tallied straight off the record
    without re-walking a nested structure. ``None`` -- not ``[]`` -- when there is no
    audit at all: no backoff ran, which is a different fact from a backoff that rejected
    nobody.
    """
    if not audit:
        return None
    candidates = (audit.get("backoff") or {}).get("candidates") or []
    return [
        str(c.get("reason")) for c in candidates
        if not c.get("accepted") and c.get("reason") is not None
    ]


def _eligibility_rejections(audit: Dict[str, Any]) -> Optional[List[str]]:
    """The certified FD-eligibility walk's REJECTION REASONS, in visit order.

    ``None`` when no certified walk ran (the legacy policy), for the same reason as
    :func:`_backoff_rejections`.
    """
    if not audit:
        return None
    return [
        str(c.get("reason")) for c in (audit.get("candidates") or [])
        if not c.get("accepted") and c.get("reason") is not None
    ]


def _append_episode_outcome_record(
    path: Optional[Path], record: Dict[str, Any]
) -> None:
    """Append ONE record to ``episode_outcomes.jsonl`` and flush it immediately.

    Same durability discipline as :func:`_append_failure_record`: opened in append mode
    per record and flushed before returning, so a run killed mid-batch still leaves a
    complete account of every attempt that had already completed. ``path=None`` disables
    the stream (used by callers that have no run directory).
    """
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _wake_decision_records(trajectory: List[Any]) -> List[Dict[str, Any]]:
    """The per-wake diagnostic records of one trajectory, in wake order.

    Reads ``Transition.decision``, which ``graph_tick_loop._wake_decision`` already
    built from the SAME logits and mask the actor acted on. It recomputes NOTHING: a
    second computation here could disagree with the decision that was really made, which
    is the whole failure mode this record exists to remove.

    A transition without the field (an older object, or a caller that constructed one by
    hand) is SKIPPED rather than filled in with invented values.
    """
    out: List[Dict[str, Any]] = []
    for tr in trajectory or ():
        rec = getattr(tr, "decision", None)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _wake_action_representation(record: Mapping[str, Any]) -> str:
    """The action representation ONE wake record is stated in.

    A wake-diagnostics-2 record names it; a historical record carries no id and is read
    under the reader label :data:`LEGACY_ACTION_REPRESENTATION_LABEL`.
    """
    rid = record.get("action_representation_id")
    return str(rid) if isinstance(rid, str) and rid else LEGACY_ACTION_REPRESENTATION_LABEL


def _num(value: Any) -> Optional[float]:
    """A finite-typed number as float, else ``None`` (``bool`` is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _wake_meta_probability(record: Mapping[str, Any], name: str) -> Optional[float]:
    """P(meta-action) of ONE wake, read under THAT record's own representation.

    Semantic records: the one PLAN / ABORT leaf, or the ENGAGE leaves' sum
    (``semantic_probability_per_meta_action``). Historical records: the aggregate column
    mass over the node-indexed aliases (``aggregate_probability_per_meta_action``). An
    unknown representation is not read at all.
    """
    rep = _wake_action_representation(record)
    if rep == ACTION_REPRESENTATION_ID:
        source = record.get("semantic_probability_per_meta_action")
    elif rep == LEGACY_ACTION_REPRESENTATION_LABEL:
        source = record.get("aggregate_probability_per_meta_action")
    else:
        return None
    return _num((source or {}).get(name)) if isinstance(source, Mapping) else None


def _wake_diag_digest(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a flat list of per-wake diagnostics. REPORTING-ONLY, version-aware.

    Every rate carries its own explicit denominator, and an EMPTY population reports
    ``None`` rather than ``0.0`` -- on a rate, 0 is a measured value and would read as
    "the actor never aborted" when the truth is "no wake of this kind occurred".

    REPRESENTATION-NEUTRAL KEYS (``selected_abort_fraction``, ``p_abort_mean``,
    ``p_plan_mean``, ``p_engage_mean``, ``entropy_raw_mean``,
    ``entropy_normalized_mean``, ``n_valid_actions_mean``) read each wake under its OWN
    representation (:func:`_wake_meta_probability`). Under the semantic representation
    ``P(ABORT)`` is directly the one ABORT leaf; under the historical one it was the
    aggregate mass over the abort aliases. A population mixing representations reports
    those means as ``None`` with ``mixed_action_representations: true`` -- never a mean
    across two action semantics.

    HISTORICAL KEYS (``selected_joint_cell_abort_fraction``, ``aggregate_p_*``,
    ``joint_*``, ``joint_vs_aggregate_disagreement_fraction``, ``n_valid_cells_mean``)
    are computed from HISTORICAL records only, keep their original meaning, and are
    ``None`` when none is present. No semantic record contributes to them.
    """
    n = len(records)
    counts = {name: 0 for name in _META_NAMES}
    own: Dict[str, int] = {}
    by_rep: Dict[str, int] = {}
    for r in records:
        nm = str(r.get("selected_meta_action_name") or "")
        if nm in counts:
            counts[nm] += 1
        o = str(r.get("selected_node_ownership") or "")
        if o:
            own[o] = own.get(o, 0) + 1
        rep = _wake_action_representation(r)
        by_rep[rep] = by_rep.get(rep, 0) + 1
    mixed = len(by_rep) > 1
    legacy = [r for r in records
              if _wake_action_representation(r) == LEGACY_ACTION_REPRESENTATION_LABEL]

    def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [v for v in (_num(r.get(key)) for r in rows) if v is not None]
        return (float(sum(vals)) / len(vals)) if vals else None

    def _neutral_mean(fn) -> Optional[float]:
        if mixed:
            return None
        vals = [v for v in (fn(r) for r in records) if v is not None]
        return (float(sum(vals)) / len(vals)) if vals else None

    def _is_semantic(r) -> bool:
        return _wake_action_representation(r) == ACTION_REPRESENTATION_ID

    def _entropy_raw(r):
        return _num(r.get("semantic_entropy_raw" if _is_semantic(r) else "joint_entropy_raw"))

    def _entropy_norm(r):
        return _num(r.get("semantic_entropy_normalized" if _is_semantic(r)
                          else "joint_entropy_normalized"))

    def _n_valid(r):
        return _num(r.get("n_valid_semantic_leaves" if _is_semantic(r) else "n_valid_cells"))

    n_norm_defined = 0 if mixed else sum(1 for r in records if _entropy_norm(r) is not None)
    legacy_counts = sum(1 for r in legacy
                        if str(r.get("selected_meta_action_name") or "") == _ABORT_NAME)
    legacy_agg = [r.get("aggregate_probability_per_meta_action") or {} for r in legacy]
    abort_mass = [v for v in (_num(a.get(_ABORT_NAME)) for a in legacy_agg) if v is not None]
    plan_mass = [v for v in (_num(a.get(_PLAN_NAME)) for a in legacy_agg) if v is not None]
    legacy_norm = [v for v in (_num(r.get("joint_entropy_normalized")) for r in legacy)
                   if v is not None]
    dis = [bool(r.get("joint_vs_aggregate_disagree")) for r in legacy
           if r.get("joint_vs_aggregate_disagree") is not None]
    return {
        "n_wakes": n,
        "selected_meta_action_counts": counts,
        "selected_node_ownership_counts": own,
        "n_wakes_by_action_representation": by_rep,
        "action_representation_ids_observed": sorted(by_rep),
        "mixed_action_representations": bool(mixed),
        # --- representation-neutral (each wake read under its own representation) ---
        "selected_abort_fraction": (float(counts[_ABORT_NAME]) / n) if n else None,
        "p_abort_mean": _neutral_mean(lambda r: _wake_meta_probability(r, _ABORT_NAME)),
        "p_plan_mean": _neutral_mean(lambda r: _wake_meta_probability(r, _PLAN_NAME)),
        "p_engage_mean": _neutral_mean(
            lambda r: _wake_meta_probability(r, MetaAction.OPPORTUNISTIC_ENGAGEMENT.name)),
        "entropy_raw_mean": _neutral_mean(_entropy_raw),
        # Normalization is UNDEFINED for a single valid action, so that wake contributes
        # to neither the mean nor its denominator, and the denominator is reported.
        "entropy_normalized_mean": _neutral_mean(_entropy_norm),
        "n_entropy_normalized_defined": n_norm_defined,
        "n_valid_actions_mean": _neutral_mean(_n_valid),
        "distance_clipping_fraction_mean": _mean(records, "fraction_task_distance_clipped"),
        # --- HISTORICAL node-indexed joint representation only ---
        "selected_joint_cell_abort_fraction": (
            float(legacy_counts) / len(legacy) if legacy else None),
        "aggregate_p_abort_mean": (
            (float(sum(abort_mass)) / len(abort_mass)) if abort_mass else None),
        "aggregate_p_plan_mean": (
            (float(sum(plan_mass)) / len(plan_mass)) if plan_mass else None),
        "joint_entropy_raw_mean": _mean(legacy, "joint_entropy_raw"),
        "joint_entropy_normalized_mean": (
            (float(sum(legacy_norm)) / len(legacy_norm)) if legacy_norm else None),
        "n_joint_entropy_normalized_defined": len(legacy_norm),
        "aggregate_meta_action_entropy_mean": _mean(legacy, "aggregate_meta_action_entropy"),
        "joint_vs_aggregate_disagreement_fraction": (
            (float(sum(1 for d in dis if d)) / len(dis)) if dis else None),
        "n_valid_cells_mean": _mean(legacy, "n_valid_cells"),
    }


def _empty_meta_counts() -> Dict[str, int]:
    """A zeroed count dict over the three fixed meta-action names."""
    return {name: 0 for name in _META_NAMES}


def _add_meta_action_counts(counts: Dict[str, int], trajectory: List[Any]) -> None:
    """Accumulate one trajectory's meta-actions into ``counts`` (in place)."""
    for tr in trajectory:
        counts[MetaAction(int(tr.meta_action)).name] += 1


def _meta_fractions(counts: Dict[str, int]) -> Dict[str, float]:
    """Normalize meta-action counts to fractions (all zeros on an empty batch)."""
    total = sum(counts.values())
    if total <= 0:
        return {name: 0.0 for name in _META_NAMES}
    return {name: counts[name] / total for name in _META_NAMES}


class _ConditionTally:
    """Per-CELL attempt accounting + FD event counters for ONE batch or eval round.

    ONE site behind every clean/damaged (and, under FD-VARIABLE-SEVERITY-v1, every
    mild/severe) number a record carries, so the training loop and the evaluation round
    cannot drift into counting the same thing two ways. It holds ATTEMPTS (which include
    failures, and are therefore the denominators) separately from the reward population
    (successes only) -- the distinction :func:`_stats_or_none` exists to protect, applied
    per cell.

    A CELL IS A REPORTING LABEL, NOT A NEW CONDITION. Storage is per cell -- ``clean`` /
    ``damaged`` for a legacy run, ``clean`` / ``mild`` / ``severe`` for a
    variable-severity one -- and the clean/damaged keys are DERIVED from it by pooling
    (:func:`cell_condition`). For a legacy run the cells ARE the conditions, so the
    pooling is the identity and every emitted key keeps exactly the value it had.

    The FD counters answer the questions an operator needs in order to trust a damaged
    batch at all: did the events actually fire (``events_applied``), did they actually
    wake the intended ego (``wakes``), and did anything come of it (``rtb_issued``,
    ``deaths``). A damaged round with zero applied events would produce clean-looking
    numbers under a damaged label, and these counters are what makes that visible.

    THE FD-WAKE META-ACTION MIX IS TRACKED PER CELL, and it is the PRIMARY behavioural
    measurement of the variable-severity experiment: the scientific question is not
    whether reward changed but whether the actor ABORTS DIFFERENTLY when a loss is
    survivable than when it is not. Its denominator is FD wakes in that cell, which is
    smaller than the cell's successful-episode count (an event can fire without the
    policy ever being woken by it), so it is stored and reported separately rather than
    inferred.

    ``rtb_issued`` counts EMITTED ``aircraft_return_to_base`` COMMANDS, taken from
    ``_EpisodeOutcome.selected_ego_rtb_issued`` which the fuel-damage controller derives
    from the Phase-2 command lists. It is deliberately not the executor's ``rtb_issued``
    latch, which is also set for a dead ego that emitted no command -- counting that
    would let one episode register as an RTB *and* a death.
    """

    def __init__(self, cells: Sequence[str] = CONDITIONS) -> None:
        # Every reported cell is present from the start, so a cell that saw no attempt
        # reports an explicit 0 / None instead of vanishing from the record.
        self.cells: Tuple[str, ...] = tuple(str(c) for c in cells)
        self.attempted: Dict[str, int] = {c: 0 for c in self.cells}
        self.failed: Dict[str, int] = {c: 0 for c in self.cells}
        self.rewards: Dict[str, List[float]] = {c: [] for c in self.cells}
        # FD-wake behaviour per cell: the meta-action the fuel-damage wake selected.
        self.fd_fired: Dict[str, int] = {c: 0 for c in self.cells}
        self.fd_wakes: Dict[str, int] = {c: 0 for c in self.cells}
        self.fd_meta: Dict[str, Dict[str, int]] = {
            c: _empty_meta_counts() for c in self.cells
        }
        self.events_applied = 0
        self.wakes = 0
        self.rtb_issued = 0
        self.deaths = 0

    def attempt(self, cell: str) -> None:
        """Count one SCHEDULED attempt, before it is known whether it will succeed.

        The schedule and this tally are built from the same
        :attr:`TrainConfig.reported_cells`, so an unknown cell here means the two were
        built from different configs -- a denominator that would never be reported.
        """
        if str(cell) not in self.attempted:
            raise MeasurementIntegrityError(
                "a scheduled attempt names cell %r, which this run does not report "
                "(cells: %r); its denominator would be invisible."
                % (cell, list(self.cells))
            )
        self.attempted[str(cell)] += 1

    def failure(self, cell: str) -> None:
        self.failed[str(cell)] = self.failed.get(str(cell), 0) + 1

    def success(self, out: "_EpisodeOutcome", *, expected_cell: str) -> str:
        """Fold one successful episode in; returns the CELL it was counted under.

        THE EXECUTED CELL MUST BE THE SCHEDULED ONE. ``expected_cell`` is the cell the
        SCHEDULE resolved before the episode was built -- the same value
        :meth:`attempt` counted the denominator under -- and ``cell`` is read from the
        plan the episode REALLY ran with. Requiring equality, rather than mere
        membership, is the whole guarantee: under FD-VARIABLE-SEVERITY-v1 a scheduled
        ``mild`` that executed as ``severe`` is a legal member of ``self.cells``, so a
        membership test accepts it and silently books the attempt in one cell and the
        reward in another. That corrupts BOTH denominators at once -- the scheduled cell
        reads as a failure that never happened, the executed cell as a success that was
        never scheduled -- and it is exactly the matched-group integrity fault the triad
        design exists to make measurable.

        ``expected_cell`` is a REQUIRED keyword, deliberately: an optional one would let
        a future call site skip the check by omission, which is the same class of defect
        one level up. Both production call sites know the scheduled cell before the
        episode runs and must state it here.

        THREE DISJOINT FAULTS, each named separately because they are different
        diagnoses:

          1. the SCHEDULE named a cell this tally does not report -- the schedule and the
             tally were built from different configs;
          2. the EXECUTION reports a cell this run does not report at all -- a legacy run
             that produced a severity, or a severity outside the declared set;
          3. both are reportable but they DISAGREE -- the matched-group fault above.

        All three are INFRASTRUCTURE and abort, exactly as a roster fault does. Every
        check runs BEFORE any state is mutated, so a rejected episode leaves the tally
        byte-unchanged and can never be half-counted.
        """
        plan = out.fuel_damage_plan or {}
        outcome = out.fuel_damage_outcome or {}
        cell = _outcome_cell(plan)
        expected = str(expected_cell)
        if expected not in self.rewards:
            raise MeasurementIntegrityError(
                "a successful episode was scheduled under cell %r, which this run does "
                "not report (cells: %r); the schedule and the tally disagree about what "
                "this run measures." % (expected, list(self.cells))
            )
        if cell not in self.rewards:
            raise MeasurementIntegrityError(
                "a successful episode reports cell %r, which this run does not report "
                "(cells: %r). The executed fuel-damage plan disagrees with the schedule "
                "that counted the attempt, so its reward would be missing from every "
                "per-cell mean while still inside the round's totals."
                % (cell, list(self.cells))
            )
        if cell != expected:
            raise MeasurementIntegrityError(
                "a successful episode was SCHEDULED as %r but EXECUTED as %r. The "
                "attempt was already counted in %r's denominator, so folding its reward "
                "into %r would report a failure that never happened in one cell and a "
                "success that was never scheduled in the other -- and, for a matched "
                "group, a within-seed delta between two members that are not the "
                "members the schedule paired. Cells: %r."
                % (expected, cell, expected, cell, list(self.cells))
            )
        self.rewards[cell].append(float(out.reward))
        if outcome.get("fired"):
            self.events_applied += 1
            self.fd_fired[cell] = self.fd_fired.get(cell, 0) + 1
        if outcome.get("wake_occurred"):
            self.wakes += 1
            self.fd_wakes[cell] = self.fd_wakes.get(cell, 0) + 1
            meta = outcome.get("wake_meta_action")
            if meta is not None:
                bucket = self.fd_meta.setdefault(cell, _empty_meta_counts())
                name = MetaAction(int(meta)).name
                bucket[name] = bucket.get(name, 0) + 1
        if out.selected_ego_rtb_issued:
            self.rtb_issued += 1
        self.deaths += int(out.n_dead)
        return cell

    # ---- per-cell reads ---------------------------------------------------------
    def successful(self, cell: str) -> int:
        return len(self.rewards.get(str(cell), []))

    def mean(self, cell: str) -> Optional[float]:
        """Mean reward over that cell's SUCCESSFUL episodes, ``None`` if none."""
        return _stats_or_none(self.rewards.get(str(cell), []))["mean"]

    # ---- condition reads, DERIVED by pooling the cells that belong to them -------
    def _cells_of(self, condition: str) -> Tuple[str, ...]:
        return tuple(c for c in self.cells if cell_condition(c) == str(condition))

    def condition_attempted(self, condition: str) -> int:
        return sum(int(self.attempted.get(c, 0)) for c in self._cells_of(condition))

    def condition_failed(self, condition: str) -> int:
        return sum(int(self.failed.get(c, 0)) for c in self._cells_of(condition))

    def condition_rewards(self, condition: str) -> List[float]:
        pooled: List[float] = []
        for c in self._cells_of(condition):
            pooled.extend(self.rewards.get(c, []))
        return pooled

    def to_record(self, prefix: str = "") -> Dict[str, Any]:
        """The tally as flat scalars. ``prefix`` namespaces the eval copy of the keys."""
        out: Dict[str, Any] = {}
        # The clean/damaged keys, ALWAYS emitted and always meaning the same thing. For a
        # variable-severity run `damaged` pools mild and severe, which is the truthful
        # reading of "how many damaged episodes were there" -- and the finer cells are
        # emitted below rather than instead.
        for condition in CONDITIONS:
            pooled = self.condition_rewards(condition)
            out["%sn_%s_attempted" % (prefix, condition)] = self.condition_attempted(
                condition)
            out["%sn_%s_successful" % (prefix, condition)] = len(pooled)
            out["%sn_%s_failed" % (prefix, condition)] = self.condition_failed(condition)
            out["%sreward_mean_%s" % (prefix, condition)] = _stats_or_none(pooled)["mean"]
        # The SEVERITY cells, only when this run has them -- a legacy record must not
        # sprout mild/severe keys it could never populate.
        for cell in self.cells:
            if cell in CONDITIONS:
                continue
            out["%sn_%s_attempted" % (prefix, cell)] = int(self.attempted.get(cell, 0))
            out["%sn_%s_successful" % (prefix, cell)] = self.successful(cell)
            out["%sn_%s_failed" % (prefix, cell)] = int(self.failed.get(cell, 0))
            out["%sreward_mean_%s" % (prefix, cell)] = self.mean(cell)
        # PER-CELL FD-wake behaviour, for every DAMAGED cell of this run (the legacy
        # `damaged` cell included, so a legacy run gains the same measurement). Counts
        # are exact; the rates carry their own denominator and are `None` -- never 0.0 --
        # when there was no FD wake to take a rate over.
        for cell in self.cells:
            if cell_condition(cell) != CONDITION_DAMAGED:
                continue
            counts = self.fd_meta.get(cell, _empty_meta_counts())
            denom = int(self.fd_wakes.get(cell, 0))
            out["%sn_%s_fd_fired" % (prefix, cell)] = int(self.fd_fired.get(cell, 0))
            out["%sn_%s_fd_wakes" % (prefix, cell)] = denom
            out["%sfd_meta_action_counts_%s" % (prefix, cell)] = dict(counts)
            out["%sfd_meta_action_rates_%s" % (prefix, cell)] = {
                name: _fraction(int(counts.get(name, 0)), denom) for name in _META_NAMES
            }
        out["%sfuel_damage_events_applied" % prefix] = int(self.events_applied)
        out["%sfuel_damage_wakes" % prefix] = int(self.wakes)
        out["%sfuel_damage_rtb_issued" % prefix] = int(self.rtb_issued)
        out["%sdeaths" % prefix] = int(self.deaths)
        return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a jsonl file into a list of dicts; missing file -> empty list."""
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# =============================================================================
# 3c. Provenance -- what code, on what machine, with which seeds
# =============================================================================

def _truncate(text: str, *, max_lines: int = 3, max_chars: int = 400) -> str:
    """First few lines of a probe's output, hard-capped -- provenance, not a transcript."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    joined = " | ".join(lines[:max_lines])
    return joined if len(joined) <= max_chars else joined[:max_chars] + "..."


def _probe_command(
    args: List[str],
    *,
    timeout: float,
    cwd: Union[str, Path, None] = None,
) -> Tuple[int, str, str]:
    """Run a short provenance probe and return ``(returncode, stdout, stderr)``.

    Output is captured as BYTES and decoded leniently, NOT via ``text=True``. This is
    not defensive styling -- it is a measured failure on this stack. ``bonmin -v`` under
    ``nlp_env`` emits byte ``0x81`` (a cp1252/cp1255 byte), and with ``text=True``
    subprocess decodes on a READER THREAD: the ``UnicodeDecodeError`` kills that thread,
    prints a traceback to stderr, and hands back an EMPTY stdout with returncode 0. The
    probe would then have reported ``ok`` with no output -- a silent loss of the exact
    fact it exists to record, plus a spurious traceback on every real run.

    ``stdin`` is closed so a probe can never sit waiting for input, and ``timeout`` is
    the caller's hard bound. Raises only what the caller already handles
    (``OSError`` / ``SubprocessError``, including ``TimeoutExpired``).
    """
    proc = subprocess.run(
        args, capture_output=True, cwd=None if cwd is None else str(cwd),
        stdin=subprocess.DEVNULL, timeout=timeout,
    )
    return (
        proc.returncode,
        (proc.stdout or b"").decode("utf-8", errors="replace"),
        (proc.stderr or b"").decode("utf-8", errors="replace"),
    )


def _git_provenance(repo_root: Union[str, Path]) -> Dict[str, Any]:
    """The code identity of a run: full commit SHA + whether the tree was dirty.

    THE load-bearing provenance field. A research result is attributable only if the
    exact code that produced it can be named, and "dirty" is half of that statement: a
    SHA plus uncommitted edits describes a tree that exists nowhere but that machine.

    ``available=True`` therefore means BOTH facts were determined -- the full HEAD SHA
    *and* the clean/dirty verdict. A SHA on its own is not attribution: it names a
    commit the run may or may not have actually executed, and reporting that as
    available provenance is the failure this guards. When the status probe fails the
    recovered ``commit`` is still returned (it is useful for debugging) but
    ``available`` stays ``False``, ``dirty`` stays ``None``, and ``reason`` says why.

    Every failure mode is reported EXPLICITLY rather than omitted -- no repository, no
    ``git`` on PATH, a timeout -- because a silently absent key is indistinguishable from
    a key nobody thought to collect. ``available`` is the single flag a reader checks;
    ``reason`` says why when it is ``False``.

    ``repo_root`` is a parameter (not always :data:`_REPO_ROOT`) so this is testable
    against a directory whose Git state is CHOSEN by the test rather than inherited from
    whatever the developer's checkout happens to look like.
    """
    root = Path(repo_root)
    info: Dict[str, Any] = {
        "repo_root": str(root),
        "available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "dirty_path_count": None,
        "reason": None,
    }

    # The last transport-level error, kept out of `info` so a LATER optional probe
    # (the branch name) cannot overwrite the reason an earlier required one failed.
    errors: List[str] = []

    def _git(args: List[str]) -> Optional[Tuple[int, str, str]]:
        try:
            return _probe_command(["git"] + args, timeout=_GIT_PROBE_TIMEOUT_S,
                                  cwd=root)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append("git %s: %s: %s" % (args[0], type(exc).__name__, exc))
            return None

    head = _git(["rev-parse", "HEAD"])
    if head is None:
        info["reason"] = errors[-1]
        return info
    if head[0] != 0:
        info["reason"] = _truncate(head[2] or "git rev-parse HEAD failed")
        return info
    info["commit"] = head[1].strip()

    # REQUIRED, not best-effort: without the clean/dirty verdict the commit alone does
    # not describe what ran, so `available` must not be set until this succeeds.
    status = _git(["status", "--porcelain"])
    if status is None:
        info["reason"] = errors[-1]
        return info
    if status[0] != 0:
        info["reason"] = _truncate(status[2] or "git status --porcelain failed")
        return info
    changed = [line for line in status[1].splitlines() if line.strip()]
    info["dirty"] = bool(changed)
    info["dirty_path_count"] = len(changed)
    info["available"] = True          # both required facts are now known

    # The branch name is a convenience, not part of attribution -- its failure must not
    # demote provenance that is already complete.
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch is not None and branch[0] == 0:
        info["branch"] = branch[1].strip()
    return info


def _module_provenance(name: str) -> Dict[str, Any]:
    """Version + on-disk path of one importable module, with failures made explicit.

    Both halves are recorded because neither alone identifies the code. The vendored
    BLADE fork carries no version string at all, so its PATH is the fact that matters
    (it is what proves an editable install resolved to the fork in this repository and
    not to some other copy); conversely a wheel's version is the fact and its path is
    noise. A module that cannot be imported yields ``available: false`` plus the error --
    never a missing key.
    """
    out: Dict[str, Any] = {
        "available": False, "version": None, "path": None, "error": None,
    }
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any import failure is just "unavailable"
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    out["available"] = True
    out["path"] = getattr(module, "__file__", None)
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(name)
        except Exception:  # noqa: BLE001 - no distribution metadata is normal here
            version = None
    out["version"] = None if version is None else str(version)
    return out


def _bonmin_provenance() -> Dict[str, Any]:
    """Where BONMIN resolves from, plus a BOUNDED version probe.

    The solver is part of a result's identity -- every episode's ``A_init`` and oracle
    come out of it -- and it is also the one dependency this project has repeatedly
    found in the WRONG environment (the base env has no ``bonmin`` at all and fails
    silently). Recording the resolved executable path makes "which solver produced this
    run" answerable after the fact instead of inferred from which shell was used.

    The probe is bounded three ways: ``stdin`` is closed so it can never sit waiting for
    input, it is killed after :data:`_BONMIN_PROBE_TIMEOUT_S`, and its output is
    truncated. It is a VERSION probe -- it solves nothing. ``probe`` is always one of
    ``not_found`` / ``ok`` / ``rc=<n>`` / ``timeout`` / ``error``.

    The output is decoded leniently by :func:`_probe_command`; this binary is one of the
    measured non-UTF-8 emitters that made that necessary.
    """
    executable = shutil.which("bonmin")
    out: Dict[str, Any] = {
        "executable": executable,
        "available": executable is not None,
        "probe": "not_found" if executable is None else None,
        "probe_output": None,
    }
    if executable is None:
        return out
    try:
        returncode, stdout, stderr = _probe_command(
            [executable, "-v"], timeout=_BONMIN_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        out["probe"] = "timeout"
        return out
    except (OSError, subprocess.SubprocessError) as exc:
        out["probe"] = "error"
        out["probe_output"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    out["probe"] = "ok" if returncode == 0 else "rc=%d" % returncode
    out["probe_output"] = _truncate(stdout + "\n" + stderr)
    return out


#: How a run's EVALUATION seeds were really obtained.
#:
#: Only the MANIFEST value is ever WRITTEN, and deliberately so: a fixed-cell run's
#: provenance block keeps exactly its historical shape (adding a source label there would
#: change an artifact every existing reader and every preserved run already has), so the
#: historical source is identified by the ABSENCE of `benchmark_evaluation` rather than by
#: a new key. The generalized block states its source outright, because a provenance block
#: that names a schedule the run did not execute is worse than one that names none: a
#: reviewer re-checking "was this held out?" would re-derive the wrong band.
EVAL_SEED_SOURCE_BAND: str = "eval_seed_band"
EVAL_SEED_SOURCE_MANIFEST: str = "benchmark_manifest"


def seed_bands(
    cfg: TrainConfig, *, benchmark: Optional[BenchmarkManifest] = None
) -> Dict[str, Any]:
    """The run's seed schedule as HALF-OPEN ranges, plus the derivation formulas.

    Recorded rather than left implicit because "held out" is a claim about intervals, and
    this block is the artifact that lets a reviewer re-check that claim without
    re-deriving the arithmetic. Half-open is stated in the payload (``stop`` EXCLUSIVE)
    so an off-by-one reading is not available.

    THE TRAINING HALF IS THE SAME UNDER BOTH DESIGNS, and is computed from the same pure
    functions the loop uses (:func:`global_episode_index`, :func:`train_seed`), so it
    cannot describe a schedule other than the one that runs.

    THE EVALUATION HALF STATES ITS ACTUAL SOURCE. ``benchmark`` omitted (every
    fixed-cell run, and every direct pre-Task-4 caller) leaves this function's output
    BYTE-UNCHANGED: the configured band and ``eval_seed = eval_base_seed + e``, which is
    exactly the schedule such a run executes. Supplied, the run's evaluation seeds come
    from the FROZEN MANIFEST instead, and the block says so: the manifest's identity, how
    many world seeds it holds, an ordered-seed digest and the seeds themselves, plus the
    verified held-out result against the training band. The configured band is then
    reported under ``unused_eval_band`` -- retained so a reader can see it was configured
    and NOT executed, rather than deleted (which would leave a reader wondering) or left
    in ``eval_band`` (which would assert a schedule that never ran).
    """
    train_start = int(cfg.base_seed)
    # THE MAXIMUM POSSIBLE training band. On the fixed-cell path this is exactly
    # `total_episodes` and the block below is byte-unchanged; under the generalized quota
    # policy a run may SPEND more seeds than it collects episodes, and the band a
    # held-out claim is made against must cover every seed the loop can reach.
    train_stop = train_start + int(cfg.max_training_attempts)
    band: Dict[str, Any] = {
        "train_band": {
            "start": train_start,
            "stop": train_stop,
            "half_open": True,
            "count": int(cfg.max_training_attempts),
        },
        "train_seed_formula":
            "train_seed = base_seed + (iteration * episodes_per_iteration + j)",
        "eval_enabled": bool(cfg.eval_enabled),
        "eval_band": None,
        "eval_seed_formula": "eval_seed = eval_base_seed + e",
        "eval_band_is_fixed_across_rounds": True,
    }
    if cfg.generalized:
        # ADDED ONLY on the generalized path, exactly as `benchmark_evaluation` is: the
        # historical block must keep the shape every existing reader and every preserved
        # run artifact already has, and a reader tells the two apart by the PRESENCE of
        # this key. It says what `train_band` now counts, so "count" is never misread as
        # the number of episodes the run collected.
        band["train_seed_formula"] = (
            "train_seed = base_seed + global_attempt_ordinal "
            "(monotone over the whole run; every attempt spends one, successful or not)"
        )
        band["train_attempt_policy"] = {
            "policy": cfg.training_attempt_policy,
            "successful_episodes_required_per_iteration":
                int(cfg.episodes_per_iteration),
            "max_attempts_per_iteration": int(cfg.max_attempts_per_iteration),
            "n_iterations": int(cfg.n_iterations),
            "successful_episode_quota_total": int(cfg.total_episodes),
            "max_possible_training_attempts": int(cfg.max_training_attempts),
            "train_band_counts": "maximum_possible_attempts",
            "note": (
                "episodes_per_iteration is a quota of SUCCESSFUL episodes; ordinary "
                "attrition is replaced by the next deterministic attempt seed, so the "
                "run may consume up to max_possible_training_attempts seeds"
            ),
        }
    if benchmark is not None:
        # The seeds this run REALLY evaluates. `eval_band` / `eval_seed_formula` are
        # emptied rather than left carrying the legacy band, so nothing in this block can
        # be read as the executed schedule.
        seeds = benchmark.seeds()
        overlap = manifest_seed_overlap(
            benchmark, start=train_start, stop=train_stop)
        band["eval_seed_source"] = EVAL_SEED_SOURCE_MANIFEST
        band["eval_band"] = None
        band["eval_seed_formula"] = None
        band["benchmark_evaluation"] = {
            "manifest_id": str(benchmark.manifest_id),
            "n_world_seeds": len(seeds),
            "n_member_episodes_per_round": int(benchmark.n_members),
            # Ordered digest AND the list itself: the digest is the compact re-check, and
            # the list is small enough to be worth reading directly (one seed per matched
            # world GROUP, not per episode).
            "seed_list_sha256": benchmark.seed_digest(),
            "seeds": [int(x) for x in seeds],
            "held_out_against_train_band": {
                "start": train_start, "stop": train_stop, "half_open": True,
            },
            "held_out_overlap_count": len(overlap),
            "held_out_verified": not overlap,
        }
        if isinstance(benchmark, V2BenchmarkManifest):
            # V2-ONLY keys. The held-out check above ran over EVERY manifest seed -- both
            # profiles -- and the round size is the declared profile's, not the manifest's.
            profile_record = benchmark.profile_identity_record(str(cfg.benchmark_profile))
            band["benchmark_evaluation"]["held_out_checked_over"] = (
                "entire_manifest_all_profiles")
            band["benchmark_evaluation"]["n_member_episodes_per_round"] = int(
                profile_record["n_members"])
            band["benchmark_evaluation"]["evaluation_profile"] = profile_record
        band["unused_legacy_eval_band"] = {
            "start": int(cfg.eval_base_seed),
            "stop": int(cfg.eval_base_seed) + int(cfg.eval_episodes),
            "half_open": True,
            "count": int(cfg.eval_episodes),
            "executed": False,
            "note": ("configured but NOT executed: this run's evaluation seeds come "
                     "from the frozen benchmark manifest"),
        }
        return band
    # THE HISTORICAL BLOCK, BYTE-UNCHANGED. No key is added on this path -- not even the
    # source label -- because a fixed-cell run's provenance must keep exactly the shape
    # every existing reader and every preserved run artifact already has. A reader tells
    # the two apart by the PRESENCE of `benchmark_evaluation`, which is a fact about the
    # generalized block rather than a new field on this one.
    if cfg.eval_enabled:
        eval_start = int(cfg.eval_base_seed)
        band["eval_band"] = {
            "start": eval_start,
            "stop": eval_start + int(cfg.eval_episodes),
            "half_open": True,
            "count": int(cfg.eval_episodes),
        }
    return band


def collect_provenance(
    cfg: TrainConfig,
    *,
    argv: Optional[List[str]] = None,
    repo_root: Union[str, Path, None] = None,
    benchmark: Optional[BenchmarkManifest] = None,
) -> Dict[str, Any]:
    """Everything needed to attribute a run, collected BEFORE any solver-heavy work.

    Written into ``run_config.json`` as its ``provenance`` block rather than into a
    competing manifest: a run already records its config there, and two files that both
    claim to describe a run are two files that can disagree.

    Ordering matters twice over. This runs at the very top of :func:`train`, before the
    policy, the generator, the engine or bonmin are touched, so a run that dies in its
    first episode still leaves a complete statement of what was attempted and with what.
    It also runs before the run DIRECTORY is created: ``output_dir`` may sit inside the
    repository, and files this run creates are untracked, so collecting afterwards would
    let the run's own scenarios and ledger be reported as pre-existing dirty source
    state. Provenance must describe the tree as it was BEFORE the run touched it.

    Nothing here is a ``pip freeze``: the package list is the four modules whose identity
    can change a result (:data:`_PROVENANCE_MODULES`). Everything that could not be
    determined is present with an explicit ``null`` / ``available: false`` / ``reason``,
    because an omitted key and an unavailable fact are indistinguishable to a reader.

    ``argv`` and ``repo_root`` are injectable so this is testable without depending on
    how the test process happened to be invoked or on the developer checkout's live Git
    state. The complete resolved :class:`TrainConfig` is NOT duplicated here -- it is the
    top-level ``train_config`` key of the same file, named by ``train_config_location``.
    """
    return {
        "provenance_version": _PROVENANCE_VERSION,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "exact_cardinality_policy": _EXACT_CARDINALITY_POLICY,
        "git": _git_provenance(_REPO_ROOT if repo_root is None else repo_root),
        "invocation": {
            "argv": [str(a) for a in (sys.argv if argv is None else argv)],
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "packages": {name: _module_provenance(name) for name in _PROVENANCE_MODULES},
        # WHICH allocation objective this run solved, beside where BONMIN resolved from.
        # The backend is part of a result's identity in exactly the way the solver
        # executable is: `A_init` and every reference come out of it, and the two
        # objectives do NOT generally share an optimal allocation. BONMIN's own probe is
        # still recorded on both backends -- a P1 run does not invoke it, and its absence
        # or presence remains a fact about the environment worth keeping.
        "solver": {
            "bonmin": _bonmin_provenance(),
            "match_aou_backend": resolve_match_aou_backend(cfg.match_aou_backend),
        },
        # `benchmark` omitted (every fixed-cell run) leaves this block byte-unchanged;
        # supplied, the evaluation half states the MANIFEST as its actual seed source
        # instead of a band this run never evaluates.
        "seeds": seed_bands(cfg, benchmark=benchmark),
        "train_config_location": "run_config.json:/train_config",
    }


def _difficulty_factor_name(cfg: TrainConfig) -> str:
    """The run's difficulty-factor IDENTIFIER, for provenance and for the header.

    Two designs share one mechanism, so one name would make an artifact ambiguous about
    which experiment produced it. A ``seeded_variable`` run is
    ``fuel_damage_variable_severity_v1``; everything else keeps the merged, measured
    ``fuel_damage_baseline_v1`` exactly as it was.
    """
    return ("fuel_damage_variable_severity_v1" if cfg.variable_severity
            else "fuel_damage_baseline_v1")


def _scheduled_cell_probabilities(cfg: TrainConfig) -> Dict[str, float]:
    """The scheduled clean/mild/severe distribution as three explicit numbers.

    ``P(damaged)`` and ``P(mild | damaged)`` are the knobs, but the thing a reader wants
    to check against the approved design is the flat 0.50 / 0.25 / 0.25. Recording the
    product rather than leaving it to be multiplied is what makes a mis-set conditional
    visible in the artifact instead of only in the results.
    """
    p_damaged = float(cfg.fuel_damage_probability)
    p_mild = float(cfg.fuel_damage_mild_probability)
    return {
        CONDITION_CLEAN: 1.0 - p_damaged,
        SEVERITY_MILD: p_damaged * p_mild,
        SEVERITY_SEVERE: p_damaged * (1.0 - p_mild),
    }


def _construction_record(cfg: TrainConfig) -> Dict[str, Any]:
    """The ``construction`` block of ``run_config.json``, per design.

    ``fixed_cell_v1`` returns exactly the historical block: the configured cell IS the
    executed one there, and every existing reader keeps resolving.

    EITHER generalized design returns a block that cannot be misread as a fixed cell.
    The geometry half is unchanged (it really is configured and really is applied to every
    generated world); the CARDINALITY half says it is dynamic, names the two sources it
    comes from, and carries the configured-but-unused counts under
    ``unused_fixed_cell_config`` so a reader can see what was configured AND that it was
    not executed.
    """
    geometry = {
        "min_target_distance_km": float(cfg.min_target_distance_km),
        "min_known_separation_km": float(cfg.min_known_separation_km),
        "detection_km": float(DETECTION_KM),
        "ensure_discovery_chain": False,
        "strict_geometry": True,
        "setup_mode": "construction",
    }
    if not cfg.generalized:
        return {
            "num_agents": int(cfg.num_agents),
            "n_known": int(cfg.n_known),
            "n_hidden": int(cfg.n_hidden),
            "n_targets_generated": int(cfg.n_known),
            "n_targets_emitted": cfg.n_targets_emitted,
            **geometry,
        }
    return {
        "cardinality_source": (
            # GENERALIZED-V2 evaluates nothing (`validate` refuses both a manifest and
            # evaluation itself), so naming a manifest here would describe a population
            # half of which this run never builds.
            "per_episode_two_stage_sampler" if cfg.route_relative_population
            else "per_episode_sampler_and_benchmark_manifest"
        ),
        "fixed_cell_config_used": False,
        "training_cardinality": (
            {
                "source": "per_episode_two_stage_sampler",
                "policy": PRE_SOLVE_CARDINALITY_POLICY_V2,
                "rng_domain": V2_CARDINALITY_RNG_DOMAIN,
                "hidden_load_policy": HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
                "hidden_load_rng_domain": V2_HIDDEN_LOAD_RNG_DOMAIN,
                "agent_counts": [int(a) for a in GENERALIZED_V2_AGENT_COUNTS],
                "known_offsets": [int(o) for o in GENERALIZED_V2_KNOWN_OFFSETS],
                "rule": (
                    "A ~ Uniform(agent_counts); K | A ~ Uniform({A + o}); THEN, after the "
                    "known-only solve, H_requested ~ Uniform({1..R}) where R is the number "
                    "of egos that solve routed"
                ),
            } if cfg.route_relative_population else {
                "source": "per_episode_sampler",
                "policy": CARDINALITY_SAMPLER_POLICY,
                "rng_domain": CARDINALITY_RNG_DOMAIN,
                "agent_counts": [int(a) for a in GENERALIZED_AGENT_COUNTS],
                "rule":
                    "A ~ Uniform(agent_counts); K == A; H_requested ~ Uniform({1..A})",
            }
        ),
        "evaluation_cardinality": (
            {
                "source": None,
                "note": ("this design defines no evaluation construct; `validate` "
                         "refuses a benchmark manifest and refuses evaluation outright"),
            } if cfg.route_relative_population else {
                "source": "benchmark_manifest",
                "note": ("a benchmark member's cell is its frozen stratum's, never a "
                         "function of its seed"),
            }
        ),
        # The counts are REALIZED per episode and may fall short of the request under
        # bounded backoff, so no single number here could describe the run.
        "n_targets_emitted": None,
        "realized_cardinality_recorded_in": _EPISODE_OUTCOMES_FILENAME,
        "unused_fixed_cell_config": {
            "num_agents": int(cfg.num_agents),
            "n_known": int(cfg.n_known),
            "n_hidden": int(cfg.n_hidden),
            "executed": False,
            "note": ("configured but NOT read on this path: the cell is sampled per "
                     "episode and taken from the manifest for benchmark members"),
        },
        **geometry,
    }


def write_run_config(
    run_dir: Path,
    cfg: TrainConfig,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    config_source: Optional[Dict[str, Any]] = None,
    benchmark: Optional[BenchmarkManifest] = None,
) -> Path:
    """Write ``run_dir/run_config.json`` -- the full resolved config of THIS run.

    Without this the run directory recorded no scenario config at all (the header only
    echoed the PPO knobs), so once the scenario knobs became CLI-settable two runs could
    no longer be compared after the fact. Contents:

      * ``train_config``  -- ``asdict(cfg)``, including the nested :class:`PPOConfig`;
      * ``construction``  -- the resolved reference cell and geometry, plus the
        GENERATED/EXECUTED distinction: the generator writes ``n_known`` targets, and
        ``setup_episode``'s construction path patches ``n_hidden`` route-relative
        targets in between the two solves, so the executed world holds
        ``n_targets_emitted == n_known + n_hidden``;
      * ``derived_split`` -- LEGACY. :attr:`TrainConfig.split_preview`, kept for
        continuity with pre-B1 runs; the construction path does not consult it;
      * ``config_source`` -- WHERE the resolved config came from: the JSON preset
        path (absolute and as typed), the fields that preset supplied, and the fields an
        explicit CLI flag then overrode. ALWAYS a structured object, never ``null``, and
        ``resolved_from`` names which of :data:`_CONFIG_SOURCE_KINDS` applies, so "no
        preset" is a stated fact rather than an absent key. Omitting the argument means
        the caller handed in a :class:`TrainConfig` DIRECTLY, which is recorded as
        ``direct_config`` -- NOT as ``cli_defaults``, which would claim a command line
        that never ran. This is what makes "what produced this run?" answerable from the
        run directory instead of by comparing numbers by eye;
      * ``base_scenario`` -- the template filename every variation derives from;
      * ``provenance``    -- :func:`collect_provenance`: code SHA + dirty state,
        invocation, interpreter, platform, targeted package versions and paths, the
        BONMIN executable and its bounded version probe, the two seed bands with their
        formulas, and the exact-cardinality policy identifier. Collected here rather
        than in a separate manifest so a run is described by ONE file.

    ``provenance`` may be passed in (already collected, so a caller can inspect it and
    warn before writing); omitted, it is collected now.

    ``default=str`` covers ``output_dir`` when it is a ``Path``.
    """
    payload = {
        "train_config": asdict(cfg),
        "provenance": (
            collect_provenance(cfg, benchmark=benchmark) if provenance is None
            else provenance
        ),
        # THE CONSTRUCTION CELL. Under `fixed_cell_v1` this is the resolved, executed
        # cell and the block is byte-unchanged. Under EITHER generalized design the three
        # count fields are NOT read by anything -- training cardinality is sampled per episode
        # and benchmark cardinality comes from the manifest -- so writing them in the
        # same shape would let the artifact be read as "this run executed 3/3/3", which
        # is a plausible-looking false statement about the population. The generalized
        # block therefore states that the cell is DYNAMIC, names where each half really
        # comes from, and keeps the configured numbers only under an explicitly-labelled
        # UNUSED sub-block. Per-episode REALIZED cardinalities are deliberately not
        # duplicated here; they belong in `episode_outcomes.jsonl`.
        "construction": _construction_record(cfg),
        # PHASE B: WHICH TRAINING ALGORITHM RAN, stated rather than inferred. A reader
        # must not have to deduce it from whether a `ctde` block happens to be present
        # (it always is -- it has dataclass defaults), so `mode` and `ctde_enabled` say
        # it outright and `ctde` is `null` when the run did not use one.
        # `execution` is recorded as a CONSTANT because it is one: evaluation and
        # inference are actor-only decentralized in BOTH modes, and no CTDE run may be
        # read as having changed how actions are taken.
        "training": {
            "mode": str(cfg.training_mode),
            "ctde_enabled": bool(cfg.ctde_enabled),
            "ctde": asdict(cfg.ctde) if cfg.ctde_enabled else None,
            "execution": "decentralized_actor_only",
            # The action semantics this run samples, stores, re-scores and reports in.
            "action_representation_id": ACTION_REPRESENTATION_ID,
            # The training-only credit artifact every productive update writes.
            "credit_diagnostics": {
                "artifact": _CREDIT_DIAGNOSTICS_FILENAME,
                "schema": _CREDIT_DIAGNOSTICS_SCHEMA,
                "schema_version": _CREDIT_DIAGNOSTICS_VERSION,
            },
            # The OPT-IN actor-gradient diagnostic, stated whether or not it is on.
            "actor_gradient_diagnostics": {
                "enabled": bool(cfg.actor_gradient_diagnostics),
                "artifact": _ACTOR_GRADIENT_DIAGNOSTICS_FILENAME,
                "schema": _ACTOR_GRADIENT_DIAGNOSTICS_SCHEMA,
                "schema_version": _ACTOR_GRADIENT_DIAGNOSTICS_VERSION,
                "scope": "ctde_updater_epoch_0",
                "gradient": "ppo_policy_surrogate_before_entropy",
                "group_loss": "sum_over_group_div_total_batch_transitions",
                "groups": list(_ACTOR_GRADIENT_GROUPS),
            },
            # GENERALIZED-V1 Task 5C: WHAT `episodes_per_iteration` COUNTS, and the
            # bounded budget behind it. Recorded on BOTH designs and stated positively:
            # the fixed-cell entry says the historical contract out loud (one attempt per
            # scheduled episode, no replacement) rather than leaving it to be inferred.
            # `max_possible_training_attempts` is the number every held-out claim in this
            # run's provenance is made against.
            "attempt_policy": {
                "policy": cfg.training_attempt_policy,
                "episodes_per_iteration_counts": (
                    "successful_episodes" if cfg.generalized
                    else "scheduled_attempts"
                ),
                "successful_episodes_required_per_iteration":
                    int(cfg.episodes_per_iteration),
                "max_attempts_per_iteration": int(cfg.max_attempts_per_iteration),
                "generalized_max_attempts_per_iteration": (
                    None if cfg.generalized_max_attempts_per_iteration is None
                    else int(cfg.generalized_max_attempts_per_iteration)
                ),
                "successful_episode_quota_total": int(cfg.total_episodes),
                "max_possible_training_attempts": int(cfg.max_training_attempts),
                "replacement": bool(cfg.generalized),
            },
        },
        # FD-BASELINE-v1: the run's ONE difficulty factor and the reward coefficient that
        # gives it teeth, recorded next to the cell they modify. `resolved_*` names the
        # exact objects the run built, so a reader never has to re-derive them from the
        # flat config fields above.
        "difficulty": {
            "factor": _difficulty_factor_name(cfg),
            "fuel_damage": cfg.fuel_damage_parameters().to_record(),
            # The AUTHORITATIVE description of this run's matched evaluation group.
            "eval_group_kind": cfg.eval_group_kind,
            "eval_group_cells": list(cfg.reported_cells),
            "eval_group_modes": [mode for _cell, mode in cfg.eval_group_members],
            "eval_group_members_per_seed": cfg.eval_group_size,
            "eval_group_deltas": [
                _delta_key(cell, ref) for cell, ref in cfg.eval_group_deltas
            ],
            # The scheduled three-way distribution, written out so a reader never has to
            # multiply two conditionals to learn what the run actually sampled. `null`
            # under a legacy mode, which has no severity to distribute.
            "scheduled_cell_probabilities": (
                _scheduled_cell_probabilities(cfg) if cfg.variable_severity else None
            ),
            # LEGACY KEY, kept so an existing reader still resolves: the member CELLS.
            # `eval_group_cells` is the authoritative name.
            "eval_pair_conditions": list(cfg.reported_cells),
            "eval_pair_members_per_seed": cfg.eval_group_size,
            "reward": {
                "aircraft_penalty_coeff": float(cfg.reward_config().aircraft_penalty_coeff),
                "regret_epsilon": float(cfg.reward_config().regret_epsilon),
                "formula_changed": False,
            },
        },
        # WHICH POPULATION this run draws from, stated rather than inferred. The four low-level policy ids are RESOLVED here from the one design
        # selector, so a reader never has to know the bundle to check what ran, and
        # `p(destroy)` is recorded explicitly to state -- rather than imply -- that the
        # redesign did not touch it.
        "episode_design": {
            **cfg.design.to_record(),
            "target_destruction_probability": float(TARGET_DESTRUCTION_PROBABILITY),
            # The sampler is the TRAINING population's rule. Recorded on both designs:
            # under `fixed_cell_v1` it is `null`, which is the truthful statement that
            # the cell was configured rather than sampled.
            "cardinality_sampler": (
                None if not cfg.generalized
                else (generalized_v2_cardinality_sampler_record()
                      if cfg.route_relative_population
                      else cardinality_sampler_record())
            ),
            "fixed_cell": (
                None if cfg.generalized else {
                    "num_agents": int(cfg.num_agents),
                    "n_known": int(cfg.n_known),
                    "n_hidden": int(cfg.n_hidden),
                }
            ),
            # The FROZEN benchmark this run evaluates on -- its content hash included, so
            # "the two arms ran the same benchmark" is a checkable claim rather than an
            # assertion. `null` when no manifest is involved.
            "benchmark_manifest": (
                None if benchmark is None else {
                    "path": str(cfg.benchmark_manifest),
                    "absolute_path": str(Path(str(cfg.benchmark_manifest)).resolve()),
                    **benchmark.identity_record(),
                    # V2-ONLY: which frozen profile this run evaluates.
                    **({"evaluation_profile": benchmark.profile_identity_record(
                        str(cfg.benchmark_profile))}
                       if isinstance(benchmark, V2BenchmarkManifest) else {}),
                }
            ),
        },
        "derived_split": cfg.split_preview,
        # Never `null`, and never MISLABELLED: an omitted source means a caller built
        # this config in Python, which is a third provenance -- not a command line that
        # happened to use no preset.
        "config_source": (
            config_source_record(resolved_from=_CONFIG_SOURCE_DIRECT)
            if config_source is None else config_source
        ),
        "base_scenario": _BASE_SCENARIO.name,
    }
    path = Path(run_dir) / "run_config.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def episode_cardinality(
    cfg: TrainConfig, seed: int, *, benchmark_cardinality: Optional[
        EpisodeCardinality] = None
) -> EpisodeCardinality:
    """THE ONE site that answers "what cardinality is this scheduled attempt?".

    Three sources, and a record always says which one it was:

      * a BENCHMARK member states its own -- it comes from the frozen manifest stratum,
        so it is passed in and used verbatim (an evaluation member's world shape is not a
        function of its seed, it is a function of the stratum it was frozen into);
      * a ``generalized_v1`` TRAINING episode SAMPLES one from its seed, through the
        sampler's private rng domain (:func:`sample_generalized_cardinality`);
      * a ``fixed_cell_v1`` episode reads the configured cell verbatim -- no draw, no
        seed, no derivation, so the historical path resolves exactly what it always did.

    The sampler is never consulted on the fixed-cell path, so its rng domain is not even
    touched there, and the benchmark cardinality always wins where it is supplied -- a
    manifest member whose shape was re-derived from its seed would silently leave the
    stratum it was frozen into.

    ``generalized_v2`` HAS NO FOURTH SOURCE HERE, and that is the point: its hidden load is
    a function of the route count the known-only solve produces, so there is nothing
    honest to return before ``setup_episode`` has run. It is REFUSED rather than answered
    with a placeholder, because a placeholder would become the "requested" cardinality
    every downstream record reports. Its callers use
    :func:`v2_pre_solve_cardinality` before the solve and
    ``graph_generalized.resolved_v2_cardinality`` after it.
    """
    if benchmark_cardinality is not None:
        return benchmark_cardinality
    if cfg.route_relative_population:
        # GENERALIZED-V2 has NO single-stage answer: its hidden load is a function of the
        # route count the known-only solve produces, which does not exist yet. Refused
        # rather than answered with a placeholder, because a placeholder here would become
        # the "requested" cardinality every downstream record reports.
        raise ValueError(
            "episode_cardinality: episode_design=%r resolves its cardinality in TWO "
            "STAGES -- use sample_generalized_v2_pre_solve_cardinality() before the "
            "known-only solve and resolved_v2_cardinality() after it. There is no "
            "up-front hidden count to return." % (cfg.episode_design,)
        )
    if cfg.generalized:
        return sample_generalized_cardinality(episode_seed=int(seed))
    return fixed_cell_cardinality(
        agent_count=int(cfg.num_agents),
        known_count=int(cfg.n_known),
        hidden_requested=int(cfg.n_hidden),
    )


def _cardinality_kwargs(
    cardinality: Optional[EpisodeCardinality],
) -> Dict[str, Any]:
    """``_run_one_episode``'s cardinality keyword -- or NOTHING on the fixed-cell path.

    The SAME keyword-omission rule as :func:`_artifact_kwargs` and :func:`_ctde_kwargs`,
    for the same reason: a ``fixed_cell_v1`` run must call ``_run_one_episode`` with
    exactly the arguments it did before Task 4 existed, not with a new keyword carrying a
    value it could have derived itself. That is also the stronger invariance claim.
    """
    if cardinality is None:
        return {}
    return {"cardinality": cardinality}


def v2_pre_solve_cardinality(
    cfg: TrainConfig, seed: int
) -> Optional[PreSolveCardinality]:
    """THE ONE site that answers "what is this attempt's GENERALIZED-V2 stage-1 cell?".

    Sampled from the population layer's own rng domain and from the episode seed alone, so
    it cannot move -- and cannot be moved by -- the fuel-damage draws, the hidden-placement
    stream, the V1 cardinality sampler or torch's action sampling.

    ``None`` on every other design, which is what makes :func:`_pre_solve_kwargs` able to
    omit the keyword entirely there. Called ONCE per attempt by the caller, which then
    reuses the object for both the episode and (if the attempt fails) its ledger entry --
    so the ledger can never describe a different stage-1 draw than the one that ran.
    """
    if not cfg.route_relative_population:
        return None
    return sample_generalized_v2_pre_solve_cardinality(episode_seed=int(seed))


def _population_recorder_kwargs(
    recorder: Optional[RouteRelativePopulationRecorder],
) -> Dict[str, Any]:
    """``_run_one_episode``'s GENERALIZED-V2 provenance-carrier keyword -- or NOTHING.

    The SAME keyword-omission rule as every other optional seam here, so a non-V2 attempt
    is called exactly as it was before this carrier existed.
    """
    if recorder is None:
        return {}
    return {"population_recorder": recorder}


def _failure_cardinality(
    card: Optional[EpisodeCardinality],
    pre_solve: Optional[PreSolveCardinality],
    route_relative_load: Optional[RouteRelativeHiddenLoad],
) -> Optional[Union[EpisodeCardinality, PreSolveCardinality]]:
    """The population identity a FAILED attempt actually received.

    Three cases, and the third is the one this exists for:

      * a non-V2 attempt reports the cell its schedule resolved up front (``card``, or
        ``None`` on the fixed-cell path where the configured cell is already in the run
        config);
      * a V2 attempt that died BEFORE the known-only solve produced a route count reports
        its stage-1 half-cell -- there is no hidden load, and inventing one would put a
        number in the ledger that nothing ever drew;
      * a V2 attempt that died AFTER stage 2 reports the RESOLVED cell, assembled from the
        stage-1 draw and the ACTUAL :class:`RouteRelativeHiddenLoad` the construction path
        produced. That is an assembly of two recorded facts, never a redraw: ``H`` is
        reproducible from the seed and ``R``, and a ledger that leaned on that
        reproducibility would be reporting a replay rather than the attempt.
    """
    if card is not None:
        return card
    if pre_solve is not None and route_relative_load is not None:
        return resolved_v2_cardinality(pre_solve, route_relative_load)
    return pre_solve


def _pre_solve_kwargs(
    pre_solve: Optional[PreSolveCardinality],
) -> Dict[str, Any]:
    """``_run_one_episode``'s GENERALIZED-V2 stage-1 keyword -- or NOTHING.

    The SAME keyword-omission rule as :func:`_artifact_kwargs`, :func:`_ctde_kwargs` and
    :func:`_cardinality_kwargs`, for the same reason: every non-V2 run must call
    ``_run_one_episode`` with exactly the arguments it did before this design existed.
    """
    if pre_solve is None:
        return {}
    return {"pre_solve_cardinality": pre_solve}


def _generalized_setup_kwargs(cfg: TrainConfig) -> Dict[str, Any]:
    """``setup_episode``'s two GENERALIZED policy keywords -- or NOTHING.

    The same two keywords under BOTH generalized designs, because they resolve to the same
    four low-level policy ids; only the POPULATION differs between V1 and V2.

    Same rule again, one level down: on the historical path ``setup_episode`` is called
    with exactly its pre-Task-4 argument list, so the default ``exact_v1`` /
    ``static_t0_v1`` resolution happens inside setup as it always has.
    """
    if not cfg.generalized:
        return {}
    design = cfg.design
    return {
        "hidden_policy": design.hidden_policy,
        "reference_policy": design.reference_policy,
    }


def _v2_hidden_load_kwargs(
    cfg: TrainConfig,
    seed: int,
    pre_solve: Optional[PreSolveCardinality],
    *,
    recorder: Optional[RouteRelativePopulationRecorder] = None,
) -> Dict[str, Any]:
    """``setup_episode``'s GENERALIZED-V2 hidden-load keywords -- or NOTHING.

    Same rule again, one level down: a non-V2 run calls ``setup_episode`` with exactly its
    pre-V2 argument list, so the historical ``explicit_request_v1`` hidden-load policy is
    resolved inside setup as it always has been.

    The three POLICY keywords travel together and are produced at one site, because
    ``setup_episode`` refuses a half-supplied route-relative request: the policy without
    the seed cannot draw, and the seed without ``known_requested`` cannot refuse a world
    whose known cardinality disagrees with the schedule.

    ``recorder`` is the caller's write-once carrier for the stage-2 draw and is added only
    when one was supplied. It is PURE PROVENANCE -- setup writes it and reads nothing back
    -- and it exists so a caller that keeps a failure ledger can state the population an
    attempt really received even when ``setup_episode`` RAISES after resolving it. A
    caller with no ledger (the diagnostic rollout) simply passes none.
    """
    if pre_solve is None or not cfg.route_relative_population:
        return {}
    kwargs: Dict[str, Any] = {
        "hidden_load_policy": HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
        "hidden_load_seed": int(seed),
        "known_requested": int(pre_solve.known_count),
    }
    if recorder is not None:
        kwargs["population_recorder"] = recorder
    return kwargs


def _backend_setup_kwargs(cfg: TrainConfig) -> Dict[str, Any]:
    """``setup_episode``'s MATCH-AOU backend keyword -- or NOTHING, on the default.

    The same keyword-OMISSION discipline as ``_artifact_kwargs`` / ``_ctde_kwargs`` /
    ``_cardinality_kwargs`` / :func:`_generalized_setup_kwargs`: on ``legacy_minlp_v1``
    ``setup_episode`` is called with EXACTLY its pre-integration argument list and
    resolves its own historical default, which is the stronger invariance claim.

    It is a SEPARATE helper from :func:`_generalized_setup_kwargs` on purpose: the backend
    is its own EXPLICIT selector and not a member of the generalized policy bundle.
    ``fixed_cell_v1`` and ``generalized_v1`` may each run under EITHER approved objective,
    so folding the backend into the bundle would make it unreachable on the fixed-cell path.

    ``generalized_v2`` narrows the valid VALUE SET to ``p1_milp_v1`` and
    ``TrainConfig.validate`` REFUSES anything else on that design -- but the run still
    STATES the objective, and this helper simply resolves whatever was stated. It selects
    nothing on the run's behalf, and there is no ``auto`` and no fallback in either
    direction.
    """
    backend = resolve_match_aou_backend(cfg.match_aou_backend)
    if backend == MATCH_AOU_BACKEND_LEGACY_MINLP_V1:
        return {}
    return {"match_aou_backend": backend}


def build_variation_config(
    cfg: TrainConfig,
    seed: int,
    *,
    cardinality: Optional[Union[EpisodeCardinality, PreSolveCardinality]] = None,
) -> VariationConfig:
    """The ONE site that turns a :class:`TrainConfig` into the generator's input.

    ``cardinality`` overrides the CELL for this one episode and nothing else: the fleet
    size and the known-target count come from it, while the geometry, the discovery-chain
    switch, the strictness and the detection radius stay exactly as configured. Omitted
    (the historical call), the configured fixed cell is used and this function is
    byte-unchanged.

    It accepts a GENERALIZED-V2 :class:`PreSolveCardinality` as well as a resolved
    :class:`EpisodeCardinality`, and reads exactly the two fields both carry. That is not a
    convenience: the generator emits the KNOWN-ONLY world, so ``A`` and ``K`` are all it
    ever needed, and a V2 episode genuinely has no hidden count at this point.

    This is the B1 construction request, and every part of it is deliberate:

      * ``num_aircraft=num_agents`` / ``num_red_airbases=n_known`` -- the explicit cell.
        ``n_hidden`` is absent ON PURPOSE: the generator emits KNOWN targets only, and
        hidden targets are placed relative to SOLVED routes inside ``setup_episode``,
        which cannot happen inside the generator.
      * ``ensure_discovery_chain=False`` -- Layer 1 exists to guarantee that a hidden
        target has a known neighbour within ``DETECTION_KM``. On the construction path
        discovery is guaranteed by placement instead, and leaving Layer 1 on would pull
        the known targets into <= ``DETECTION_KM`` pairs and flatten exactly the route
        diversity B2 places against.
      * ``strict_geometry=True`` -- the 200/100 geometry is a premise, not a
        preference: the generator must raise rather than quietly weaken it.
      * ``detection_km=DETECTION_KM`` -- the single-radius invariant. Layer 1 is off, so
        this only pins the radius the rest of the pipeline agrees on; there is no
        second sensing radius anywhere.

    ``VariationConfig`` is a dataclass, so a test can compare the whole request for
    equality instead of re-listing the fields.
    """
    cell = cardinality if cardinality is not None else fixed_cell_cardinality(
        agent_count=int(cfg.num_agents),
        known_count=int(cfg.n_known),
        hidden_requested=int(cfg.n_hidden),
    )
    return VariationConfig(
        include_sams=cfg.include_sams,
        num_aircraft=int(cell.agent_count),
        num_red_airbases=int(cell.known_count),
        randomize_red_airbase_positions=cfg.randomize_red_airbase_positions,
        stretch_target_ratio=float(cfg.stretch_target_ratio),
        min_target_distance_km=float(cfg.min_target_distance_km),
        min_target_separation_km=float(cfg.min_known_separation_km),
        ensure_discovery_chain=False,
        strict_geometry=True,
        detection_km=DETECTION_KM,
        seed=int(seed),
    )


def _build_generator(scen_dir: Path) -> ScenarioGenerator:
    """ONE generator for the whole run (training AND eval reuse this instance).

    The time-feasibility cap is computed once over the full pool; ``generate`` never
    mutates state that affects a later generation (it resets its own stats snapshot and
    derives everything else from the per-call ``VariationConfig`` + its own
    ``random.Random(seed)``), so instance reuse cannot couple episodes.
    """
    gen = ScenarioGenerator(
        base_scenario_path=str(_BASE_SCENARIO),
        output_dir=str(scen_dir),
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    return gen


# =============================================================================
# 3d. Per-episode observability -- the target roster and the OK block
# =============================================================================
#
# Everything in this section is READ-ONLY with respect to the pipeline. It inspects an
# `EpisodeContext` and the executor's confirmed-kill set and formats text; it never
# mutates a belief, a solution, an executor, a reward or a scenario.
#
# TWO KINDS OF FAILURE, DELIBERATELY TREATED DIFFERENTLY -- the distinction is what
# keeps the metric honest:
#
#   * A DISPLAY failure -- one target's BLADE name cannot be resolved -- is nonfatal.
#     The target keeps its id, its place in the roster and its contribution to every
#     count; only the printed text degrades, to `_UNNAMED_TARGET`.
#   * A STRUCTURAL failure -- absent or malformed beliefs, t=0 beliefs that disagree, a
#     missing world snapshot, or a roster inconsistent with what the executor confirmed
#     -- raises `EpisodeRosterError`, a `MeasurementIntegrityError`, and ABORTS the run.
#     It is not an episode failure and never enters `skip_and_account_v1`.
#
# The second rule exists because of a real defect in the first version of this section:
# it swallowed every structural exception and returned an empty roster, and the
# authoritative count was derived from the names it had managed to classify. A degraded
# roster therefore turned an episode with real confirmations into a SUCCESSFUL
# `0/0` measurement, and that false zero flowed straight into
# `targets_confirmed_unique_mean` and its aliases. A research metric must never depend on
# whether a name diagnostic worked: the authoritative count is
# `len(_unique_confirmed_target_ids(executor.done))` and nothing else.
#
# WHERE THE WORLD COMES FROM -- the correction the long baseline forced.
#
# This section used to answer "which targets does this episode contain?" with
# `ctx.beliefs` (known) and `ctx.oracle_tasks` (executed). Both are ALLOCATIONS, not
# inventories: `solve_and_normalize` returns an allocated-only task list by contract, so
# any target the solver left unselected is absent from them while still sitting in the
# world the executor flies through, senses, attacks and confirms. The roster therefore
# under-counted the world by exactly the unselected targets, and then failed the episode
# for the discrepancy it had itself introduced -- as an accounted `setup` failure, which
# is why 143 of the long baseline's 800 training attempts disappeared while every
# preserved `executed_t0_scenario.json` held the full six-target world and 11 `complete`
# manifests reported `3 known / 2 hidden / 5 total` against an authoritative 3 + 3 = 6.
#
# The world now comes from `EpisodeContext.known_target_ids` / `executed_target_ids` --
# raw snapshots taken BEFORE either solve (see `graph_episode_setup`). Beliefs are still
# checked, but as a SUBSET constraint rather than as the known-world denominator, and
# `ctx.oracle_tasks` is not read here at all. It is unchanged and still correct for the
# reward's oracle denominator, which is a question about allocation.

def _ascii(text: Any) -> str:
    """Render a value for the cp1255 Windows console -- non-ASCII becomes ``?``.

    Target names come out of a scenario JSON, which is not this module's to constrain.
    A stray non-ASCII byte in one target's name must not take down a training run with a
    ``UnicodeEncodeError`` from ``print``.
    """
    return str(text).encode("ascii", errors="replace").decode("ascii")


def _format_names(names: Tuple[str, ...]) -> str:
    """A name list as a compact JSON array: ``["Enemy Airbase #1", ...]``, ``[]`` if empty."""
    return json.dumps([_ascii(n) for n in names])


class MeasurementIntegrityError(RuntimeError):
    """The measurement itself is unsound -- INFRASTRUCTURE, never a scientific outcome.

    Sibling of :class:`_VisualArtifactError` and routed the same way: it names no pipeline
    stage because it did not happen in one, it must never be appended to the failure
    ledger, never counted against a condition, never entered into
    ``skip_and_account_v1`` -- and it ABORTS the run.

    That routing is the correction. A data-integrity fault is not a property of the
    episode; it is a property of the instrument, so every episode it touches is suspect
    and the ones it does not touch cannot be trusted to be unaffected either. Accounting
    it as a skipped ``setup`` attempt does the maximally wrong thing: the run continues,
    the defect is invisible in the console, and the only trace is a shrinking scientific
    denominator that reads as ordinary episode attrition. The long baseline did exactly
    that -- 143 training attempts removed by a roster defect, over 83 iterations, while
    the run reported itself healthy and reconciled.
    """


class EpisodeRosterError(MeasurementIntegrityError):
    """The episode's target roster could not be built, or does not describe what ran.

    A STRUCTURAL failure of the measurement, not a display problem, and specifically not
    an episode failure: the roster is a t=0 statement about the world ``setup_episode``
    produced, so a roster that is missing, self-contradictory, or inconsistent with what
    the executor confirmed means the instrument is misreading the world -- see
    :class:`MeasurementIntegrityError` for why that aborts instead of being accounted.

    The name is retained from before the routing change so an audit trail keeps reading:
    what changed is where it goes, not what it means.
    """


class TrainingQuotaError(MeasurementIntegrityError):
    """A generalized iteration exhausted its attempt budget before filling its quota.

    GENERALIZED-V1 Task 5C. Under
    :data:`TRAINING_ATTEMPT_POLICY_QUOTA` ``episodes_per_iteration`` is a count of
    SUCCESSFUL episodes, so an iteration that runs out of
    ``generalized_max_attempts_per_iteration`` has NOT produced the batch its config
    declares. The only two things it could do instead are both wrong: updating on the
    partial batch would make the PPO/CTDE batch size a silent function of world
    attrition -- the very coupling the quota exists to remove -- and raising the budget
    on the fly would change the population mid-run.

    A :class:`MeasurementIntegrityError` deliberately, for the same reason the roster and
    certificate faults are: what it reports is that the INSTRUMENT cannot produce the
    population the run declared, so proceeding would yield a measurement whose batch
    composition nobody chose. Subclassing also means every existing abort re-raise
    already routes it correctly rather than letting a future handler account it as
    episode attrition.

    It is NOT a verdict on the worlds: an exhausted budget says the attrition rate is
    higher than the operator planned for, which is a scheduling fact to be inspected --
    never a reason to retry a spent seed.
    """


class EarlyStoppingIntegrityError(MeasurementIntegrityError):
    """A monitored early-stopping window holds an iteration with no training reward.

    GENERALIZED-V1 early stopping. ``train_reward_mean`` is ``None`` only when EVERY
    attempt of an iteration failed, and under
    :data:`TRAINING_ATTEMPT_POLICY_QUOTA` -- the only policy this feature is approved
    for -- that cannot happen: an iteration either fills its quota of SUCCESSFUL
    episodes or raises :class:`TrainingQuotaError` first. So a ``None`` inside a
    monitored window means the instrument contradicts its own attempt contract.

    The two alternatives are both wrong. Averaging the window's remaining values would
    fabricate a window mean over a population nobody chose; treating the missing value
    as ``0.0`` would insert the ORACLE OPTIMUM into a plateau test and could stop a run
    by declaring a total data loss the best window it ever saw. A
    :class:`MeasurementIntegrityError` deliberately, for the same reason the roster,
    certificate and quota faults are: proceeding would produce convergence evidence
    about a run whose own records cannot support it.
    """


def _task_target_id(task: Any) -> Optional[str]:
    """The target id a task attacks -- first ATTACK step, else ``steps[0]``.

    The builder's canonical form, duplicated here rather than imported: the builder's
    version is private to a locked layer, and ``graph_rollout`` already carries its own
    copy for the same reason. Returns ``None`` for a step-less or target-less task
    instead of raising.
    """
    steps = getattr(task, "steps", None) or []
    step = next(
        (s for s in steps if getattr(s, "step_kind", None) == StepKind.ATTACK),
        steps[0] if steps else None,
    )
    if step is None:
        return None
    target_id = getattr(step, "target_id", None)
    return None if target_id is None else str(target_id)


def _unique_confirmed_target_ids(done: Any) -> set:
    """``{target_id}`` from the executor's ``(ego_id, target_id)`` confirmed-kill set.

    THE deduplication this task exists for. ``GraphPlanExecutor.done`` records one entry
    per (ego, target) CONFIRMATION, so two egos that both close on the same wreck put two
    entries in it and ``EpisodeResult.confirmed_kills == len(done)`` can exceed the number
    of targets in the world. Reported as "kills" that reads as an impossible result --
    the approved first probe surfaced exactly that.

    The executor's set is correct for what it measures and is NOT changed (nor is the
    reward, which has always deduplicated the same way in
    ``graph_reward.realized_utility``). This is the one place that converts it into a
    count of TARGETS, and every aggregate and every printed count goes through here.
    """
    return {str(target_id) for _ego_id, target_id in (done or set())}


@dataclass(frozen=True)
class _TargetRoster:
    """The episode's target roster, snapshotted at t=0 and resolved to BLADE names.

    Ordered, not set-valued: known targets keep A_init's positional order and hidden
    targets keep the oracle task order, so two runs of the same seed print the same
    lists in the same order.

    ``known_ids`` / ``hidden_ids`` are kept because the confirmation split is computed by
    id; they are never printed (see :data:`_UNNAMED_TARGET`). They are UNIQUE within each
    half, DISJOINT across the halves, and together they cover the executed world exactly
    -- :func:`_episode_target_roster` refuses to build anything else. That is what lets
    the printed name subsets be reconciled against the authoritative count instead of
    substituting for it.

    A name is a LABEL for its id, never a stand-in: ``known_names[i]`` describes
    ``known_ids[i]``, and an unresolvable name becomes :data:`_UNNAMED_TARGET` without
    the id leaving the roster or any count changing.
    """

    known_ids: Tuple[str, ...]
    known_names: Tuple[str, ...]
    hidden_ids: Tuple[str, ...]
    hidden_names: Tuple[str, ...]

    @property
    def total(self) -> int:
        """Targets in the executed world: the denominator of every confirmation count."""
        return len(self.known_ids) + len(self.hidden_ids)

    def confirmed(self, confirmed_ids: set) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Split a unique confirmed-target-id set into ``(known_names, hidden_names)``.

        A PRESENTATION of the authoritative count, never its source. The caller has
        already computed ``len(confirmed_ids)``; this only says which names sit behind it,
        in roster order.

        Every confirmed id must therefore land in exactly one half. An id the roster does
        not contain means the roster does not describe the world that ran -- the executor
        confirmed a target the t=0 snapshot never listed -- so it RAISES rather than
        quietly dropping the target from the printed subsets, which would leave a block
        whose names no longer add up to its own total.

        Raises:
            EpisodeRosterError: if any confirmed id is outside the roster.
        """
        known = tuple(
            name for tid, name in zip(self.known_ids, self.known_names)
            if tid in confirmed_ids
        )
        hidden = tuple(
            name for tid, name in zip(self.hidden_ids, self.hidden_names)
            if tid in confirmed_ids
        )
        if len(known) + len(hidden) != len(confirmed_ids):
            unknown = sorted(
                set(confirmed_ids) - set(self.known_ids) - set(self.hidden_ids)
            )
            raise EpisodeRosterError(
                "%d confirmed target id(s) are not in the episode's executed roster "
                "(%d known + %d hidden): the t=0 snapshot does not describe the world "
                "that ran. First: %s"
                % (len(unknown), len(self.known_ids), len(self.hidden_ids),
                   ", ".join(unknown[:3]) or "<none>")
            )
        return known, hidden


def _resolve_target_name(ctx: Any, target_id: str) -> str:
    """One target's BLADE ``name`` via ``scenario.get_target``; never raises, never a uuid.

    A DISPLAY lookup, and the only nonfatal degradation in this section: an unresolvable
    name yields :data:`_UNNAMED_TARGET` while the target keeps its id, its roster slot and
    its full contribution to every count and denominator. Losing an episode because one
    label could not be rendered would be the wrong trade; losing a COUNT because of it
    would be worse, and cannot happen -- no count is computed from names.

    CALL BEFORE THE EPISODE RUNS. ``get_target`` scans the LIVE scenario, and a killed
    unit is removed from it -- resolving names afterwards would silently blank out
    exactly the targets the block is reporting as confirmed.
    """
    try:
        target = ctx.game.current_scenario.get_target(str(target_id))
    except Exception:  # noqa: BLE001 - a display lookup must never cost an episode
        return _UNNAMED_TARGET
    name = getattr(target, "name", None) if target is not None else None
    return _UNNAMED_TARGET if not name else str(name)


def _ordered_target_ids(tasks: Any, what: str) -> List[str]:
    """Target ids of ``tasks`` in order, DEDUPLICATED, or raise on a malformed task.

    Deduplication is normalization, not tolerance: the roster is a statement about
    TARGETS, and two tasks may legitimately name one target, so first use wins and order
    is preserved. A task that names NO target is different -- it means the structure this
    roster is derived from is not what it is assumed to be, and the count derived from it
    would be quietly short.

    Raises:
        EpisodeRosterError: on a non-iterable task list or a task with no target.
    """
    try:
        task_list = list(tasks)
    except TypeError as exc:
        raise EpisodeRosterError(
            "%s is not a task list (%s)" % (what, type(tasks).__name__)
        ) from exc
    ids: List[str] = []
    for index, task in enumerate(task_list):
        target_id = _task_target_id(task)
        if target_id is None:
            raise EpisodeRosterError(
                "%s task %d names no target: the roster cannot be derived from it"
                % (what, index)
            )
        ids.append(target_id)
    return list(dict.fromkeys(ids))


def _world_snapshot_ids(ctx: Any, attribute: str, what: str) -> List[str]:
    """One of the context's RAW t=0 world snapshots, validated as a list of target ids.

    ``EpisodeContext.known_target_ids`` / ``executed_target_ids`` are captured before
    either solve, so they state what the world CONTAINS rather than what a solver
    ALLOCATED (``graph_episode_setup``). Validated rather than trusted, because reading a
    world inventory off something that is not one is exactly the defect being closed: the
    attribute must exist, be a non-string sequence, and hold only non-empty ids.

    Deduplicated with first occurrence winning, matching :func:`_ordered_target_ids`, so
    both sides of the subset checks below are normalized the same way.

    Raises:
        EpisodeRosterError: if the snapshot is absent, malformed, or empty.
    """
    raw = getattr(ctx, attribute, None)
    if raw is None:
        raise EpisodeRosterError(
            "the episode context carries no %s (%s): the t=0 world inventory is "
            "unknown, and an allocated-only task list is not a substitute for it"
            % (what, attribute)
        )
    if isinstance(raw, (str, bytes)):
        raise EpisodeRosterError(
            "%s (%s) is a %s, not a sequence of target ids"
            % (what, attribute, type(raw).__name__)
        )
    try:
        values = list(raw)
    except TypeError as exc:
        raise EpisodeRosterError(
            "%s (%s) is not iterable (%s)" % (what, attribute, type(raw).__name__)
        ) from exc
    ids: List[str] = []
    for position, value in enumerate(values):
        target_id = "" if value is None else str(value)
        if not target_id:
            raise EpisodeRosterError(
                "%s (%s) entry %d is an empty target id: the snapshot would be silently "
                "short by one target" % (what, attribute, position)
            )
        ids.append(target_id)
    unique = list(dict.fromkeys(ids))
    if not unique:
        raise EpisodeRosterError(
            "%s (%s) is empty, so the episode's world is unknown" % (what, attribute)
        )
    return unique


def _episode_target_roster(ctx: Any) -> _TargetRoster:
    """Snapshot the known / hidden target roster of a freshly set-up episode.

    CALL AFTER ``setup_episode`` AND BEFORE ``run_episode``, because the NAMES are read
    out of the live scenario, which loses units as they are killed, and because the belief
    agreement check below is a t=0 statement (the N beliefs are byte-equal only then and
    legitimately DIVERGE per ego afterwards -- that divergence is the no-communication
    guarantee, not a defect).

      * KNOWN    -- ``ctx.known_target_ids``: every raw known-world target, captured
        BEFORE the known solve filtered it down to the allocated ones.
      * EXECUTED -- ``ctx.executed_target_ids``: every raw target in the authoritative
        environment, captured BEFORE the oracle solve filtered it.
      * HIDDEN   -- executed minus known, in executed-world order. Derived by SUBTRACTION
        rather than from ``ctx.placements``, which is deliberately id-free.

    ``ctx.oracle_tasks`` IS NOT READ HERE, and must not be reintroduced. It is an
    ALLOCATION over the executed world -- correct, and unchanged, for the reward's oracle
    denominator, and short of the world by whatever the oracle did not select. Reading it
    as the executed-world inventory is the defect this function was corrected for.

    The beliefs are still checked, in the role they can actually play. They are
    allocated-only too, so they are a SUBSET of the known world rather than its
    denominator; a belief naming a target the known-world snapshot does not contain is a
    real structural defect -- the egos were planned against something the world does not
    hold -- and raises.

    REQUIRED MEASUREMENT STRUCTURE, not a best-effort diagnostic. Every structural problem
    raises :class:`EpisodeRosterError` -- a :class:`MeasurementIntegrityError`, so it
    ABORTS the run rather than being accounted as a skipped episode. Only name RESOLUTION
    degrades (:func:`_resolve_target_name`), and it changes no id and no count.
    """
    beliefs_map = getattr(ctx, "beliefs", None) or {}
    if not beliefs_map:
        raise EpisodeRosterError(
            "the episode context carries no beliefs, so the t=0 belief agreement cannot "
            "be established"
        )

    # Compared BEFORE deduplication so a divergence in order is caught too. This is a
    # cheap invariant check (<= ~4 egos x ~9 tasks) on the guarantee that every belief is
    # minted from one A_init.
    per_ego = [
        (str(ego_id), _ordered_target_ids(getattr(belief, "tasks", None),
                                          "belief of ego %s" % ego_id))
        for ego_id, belief in beliefs_map.items()
    ]
    belief_ids = per_ego[0][1]
    for ego_id, ids in per_ego[1:]:
        if ids != belief_ids:
            raise EpisodeRosterError(
                "the t=0 beliefs disagree on the planned target set (ego %s vs ego %s): "
                "all beliefs are minted from one A_init, so this is a real defect and "
                "not something to report as one ego's view"
                % (per_ego[0][0], ego_id)
            )

    known_ids = _world_snapshot_ids(ctx, "known_target_ids", "the known-world snapshot")
    executed_ids = _world_snapshot_ids(
        ctx, "executed_target_ids", "the executed-world snapshot"
    )

    # A SUBSET check and never an equality one: a known target the solver did not select
    # is legitimately in no belief, and that is not a defect in either direction.
    known_set = set(known_ids)
    planned_outside = [tid for tid in belief_ids if tid not in known_set]
    if planned_outside:
        raise EpisodeRosterError(
            "%d t=0 belief target(s) are absent from the known-world snapshot (first: "
            "%s): the egos were planned against a target the world does not hold"
            % (len(planned_outside), ", ".join(planned_outside[:3]))
        )

    unmatched = [tid for tid in known_ids if tid not in set(executed_ids)]
    if unmatched:
        raise EpisodeRosterError(
            "%d t=0 known target(s) are absent from the executed world (first: %s): the "
            "roster would not cover what runs"
            % (len(unmatched), ", ".join(unmatched[:3]))
        )

    hidden_ids = [tid for tid in executed_ids if tid not in known_set]

    # known + hidden now partitions the executed target set exactly: unique within each
    # half (both deduplicated), disjoint (hidden excludes known), and complete (known is
    # a subset of executed, hidden is the rest).
    return _TargetRoster(
        known_ids=tuple(known_ids),
        known_names=tuple(_resolve_target_name(ctx, t) for t in known_ids),
        hidden_ids=tuple(hidden_ids),
        hidden_names=tuple(_resolve_target_name(ctx, t) for t in hidden_ids),
    )


def _fuel_damage_lines(out: "_EpisodeOutcome") -> List[str]:
    """The FD-BASELINE-v1 half of the per-episode ``OK`` block.

    Two lines for a damaged episode, one for a clean one -- the difficulty factor is the
    thing this baseline is about, so an operator watching a run must be able to see, per
    episode and without opening an artifact, whether the event fired, what it did to the
    tank, which window it landed in, and what the policy did about it.

    Every number is printed straight from the component's own records
    (:meth:`FuelDamagePlan.to_record` / :meth:`FuelDamageOutcome.to_record`); nothing is
    recomputed here, so the block cannot disagree with the record it summarizes. Missing
    values print as ``n/a`` rather than as ``0`` -- an event that never fired has no fuel
    reading, and a zero would read as an empty tank.
    """
    plan = out.fuel_damage_plan or {}
    outcome = out.fuel_damage_outcome or {}
    condition = str(plan.get("condition", CONDITION_CLEAN))
    if condition != CONDITION_DAMAGED:
        return ["  fuel_damage=clean ego=none"]

    meta = outcome.get("wake_meta_action")
    meta_name = "n/a" if meta is None else MetaAction(int(meta)).name
    rtb = out.selected_ego_rtb_issued
    severity = plan.get("severity")
    return [
        "  fuel_damage=%s ego=%s fired=%s tick=%s progress=%s"
        % (_ascii(severity) if severity else "damaged",
           _ascii(plan.get("ego_id")), outcome.get("fired"),
           _fmt_opt(outcome.get("event_tick"), "%d"),
           _fmt_opt(outcome.get("observed_progress"), "%.3f")),
        # `continue_margin` is the sign that says which severity this PHYSICALLY was:
        # positive means the ego could still finish its route and get home, negative
        # means it could not. Printed next to the fuel so a reader can check the label
        # against the physics rather than trusting it.
        "  fuel_before=%s fuel_after=%s factor=%s fuel_after/max=%s continue_margin=%s"
        % (_fmt_opt(outcome.get("fuel_before"), "%.1f"),
           _fmt_opt(outcome.get("fuel_after"), "%.1f"),
           _fmt_opt(outcome.get("damage_factor"), "%.4f"),
           _fmt_opt(outcome.get("fuel_after_fraction_of_max"), "%.4f"),
           _fmt_opt(outcome.get("continuation_margin"), "%+.1f")),
        # PLANNED and LIVE bounds side by side, never merged: the planned pair is the
        # preflight window, the live pair is what the mutation was really validated
        # against, and reporting one under the other's name would hide the difference the
        # live re-check exists to catch.
        "  planned_rtb_floor=%s planned_continue_req=%s | live_rtb_floor=%s "
        "live_continue_req=%s"
        % (_fmt_opt(plan.get("rtb_fuel_floor"), "%.1f"),
           _fmt_opt(plan.get("continue_fuel_requirement"), "%.1f"),
           _fmt_opt(outcome.get("live_rtb_fuel_floor"), "%.1f"),
           _fmt_opt(outcome.get("live_continue_fuel_requirement"), "%.1f")),
        # `rtb_command=` names it a COMMAND on purpose: it is True only if
        # `aircraft_return_to_base('<ego>')` was really emitted, never the executor latch.
        "  fd_wake=%s fd_meta=%s rtb_command=%s"
        % (outcome.get("wake_occurred"), meta_name,
           "n/a" if rtb is None else rtb),
    ]


def _format_episode_block(header: str, out: "_EpisodeOutcome") -> str:
    """The multi-line ``OK`` block for ONE completed episode.

    ``OK`` means the attempt COMPLETED -- generation, setup, run and reward all returned.
    It is not a verdict on the episode: ``ended`` still reports ``done`` /
    ``terminated`` / ``truncated``, and a successful zero-wake episode prints ``OK`` too.
    It exists so a completed attempt is never mistaken for the ``FAILED`` line, which is
    unchanged.

    Pure and separate from ``print`` so a test can assert on the exact text.
    """
    return "\n".join([
        "%s OK" % header,
        "  reward=%+.4f wakes=%d targets_confirmed_unique=%d/%d"
        % (out.reward, out.n_wakes, out.targets_confirmed_unique, out.targets_total),
        "  known_targets=%s" % _format_names(out.known_target_names),
        "  known_confirmed=%s" % _format_names(out.known_confirmed_names),
        "  hidden_targets=%s" % _format_names(out.hidden_target_names),
        "  hidden_confirmed=%s" % _format_names(out.hidden_confirmed_names),
    ] + _fuel_damage_lines(out) + [
        "  ended=%s ticks=%d dead=%d elapsed=%.1fs"
        % (_ascii(out.ended), out.ticks, out.n_dead, out.seconds),
    ])


# =============================================================================
# 3e. Visual artifacts -- one inspectable bundle per selected attempt
# =============================================================================
#
# WHAT THIS IS FOR. A finished scientific probe is a directory of numbers. To LOOK at an
# episode afterwards -- in PyCharm and in the BLADE client -- three files have to survive
# it, and each of them exists for only a moment inside `_run_one_episode`:
#
#   1. the exact generated KNOWN-ONLY scenario (three targets), which the construction
#      path immediately supersedes;
#   2. the AUTHORITATIVE EXECUTED t=0 scenario (six targets), which exists only as the
#      live env-2 game object -- `build_patched_scenario`'s intermediate JSON, the
#      placement audit, the beliefs and the oracle tasks are all DERIVED views and none of
#      them is what the engine actually loaded;
#   3. the BLADE playback recording of the run.
#
# WHAT THIS IS NOT. It is not a second measurement path. Nothing here is read back into
# the pipeline: the copies are writes, `Game.export_scenario()` is a read-only serializer
# the engine already exposes, and the recording is produced by the LOCKED tick-loop
# contract (armed by `setup_episode(recording_export_path=...)`, started / stepped /
# exported by `run_episode`) which is proven observationally pure. This layer calls no
# recorder internals and holds no randomness -- every name it derives comes from metadata
# the schedule had already resolved -- so an artifact-enabled attempt runs the same
# episode as the disabled one.
#
# FAILURES ARE INFRASTRUCTURE, NOT SCIENCE. A full disk or an unserializable export says
# nothing about the cell; counting it as an episode failure would put it in
# `episode_failures.jsonl` under a pipeline stage it did not happen in, and silently move
# the denominator every per-condition statistic is reported over. So it raises
# `_VisualArtifactError`, which the train and eval attempt handlers re-raise BEFORE their
# broad `except Exception`, and the run stops loudly. A normal EPISODE failure is
# unaffected: it stays in `skip_and_account_v1` and simply leaves an `incomplete` bundle
# holding whichever pre-failure artifacts were valid.

class _VisualArtifactError(RuntimeError):
    """A visual-artifact capture failed -- an INFRASTRUCTURE failure, never a scientific one.

    Deliberately NOT an :class:`EpisodeAttemptError`: it carries no pipeline stage,
    because it did not happen in one. It must never be appended to the failure ledger,
    never counted against a condition, and never skipped -- it aborts the run.
    """


@dataclass(frozen=True)
class _AttemptIdentity:
    """Exactly which scheduled attempt a bundle belongs to.

    Frozen and complete: the manifest must place an attempt WITHOUT anyone reading the
    console in order, so every discriminator the schedule used is carried explicitly
    rather than being implied by directory order or by a compact name.

    Attributes:
        phase: ``pre_update`` / ``train`` / ``post_update`` (:data:`_ARTIFACT_PHASES`).
        iteration: the zero-based training iteration, or ``None`` for the pre-update
            round (no training iteration has happened yet).
        updates_completed: PPO updates that had really run when the attempt started --
            the learning axis, and what stops a post-update bundle from being read as
            "iteration 0".
        eval_round_ordinal / eval_episode_index / eval_pair_member: the evaluation
            coordinates; ``None`` on a training attempt. The member is the matched
            clean/damaged slot, so the two members of one held-out seed are distinct.
        attempt_ordinal: the position in the phase's schedule (``j`` for training,
            ``e * 2 + member`` for evaluation) -- the same ordinal the failure ledger
            records.
        episode_index: the run-wide training episode index ``g``; ``None`` for evaluation.
        seed: the exact episode seed.
        condition: the SCHEDULED fuel-damage condition (``clean`` / ``damaged``).
        severity: the SCHEDULED severity (``mild`` / ``severe``) under
            FD-VARIABLE-SEVERITY-v1; ``None`` for a clean member and for every attempt of
            a legacy run. Carried because the two DAMAGED members of a matched triad
            share a condition and would otherwise be distinguishable only by their tag --
            a bundle has to be able to say which severity it holds.
        episode_tag: the exact scenario tag the generator was called with -- the link from
            this bundle back to the run's own ``scenarios/episode_<tag>_scenario.json``.
    """

    phase: str
    iteration: Optional[int]
    updates_completed: int
    eval_round_ordinal: Optional[int]
    eval_episode_index: Optional[int]
    eval_pair_member: Optional[int]
    attempt_ordinal: int
    episode_index: Optional[int]
    seed: int
    condition: str
    episode_tag: int
    # Defaulted, and last, so every existing construction site stays valid: a legacy run
    # has no severity to state, and being forced to pass `None` everywhere would add a
    # field to the schedule rather than to the record.
    severity: Optional[str] = None

    def __post_init__(self) -> None:
        if self.phase not in _ARTIFACT_PHASES:
            raise ValueError(
                "artifact phase must be one of %r, got %r"
                % (list(_ARTIFACT_PHASES), self.phase)
            )
        if self.phase == _ARTIFACT_PHASE_TRAIN:
            missing = [name for name in ("iteration", "episode_index")
                       if getattr(self, name) is None]
        else:
            missing = [name for name in ("eval_round_ordinal", "eval_episode_index",
                                         "eval_pair_member")
                       if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "a %r attempt identity is missing %s: a bundle that cannot say which "
                "attempt it is cannot be told apart from another one"
                % (self.phase, ", ".join(missing))
            )

    @property
    def directory_name(self) -> str:
        """The bundle's directory name -- compact, sortable, and unique by construction.

        Uniqueness comes from ``episode_tag``, which is already globally unique across a
        run: training attempts are tagged by ``g`` and every eval round/member owns its
        own disjoint tag slot (:func:`eval_member_tag`). The remaining fields are there to
        be READABLE at a glance; the manifest, not this name, is the record.
        """
        # The severity, when there is one, REPLACES the condition in the name rather than
        # being appended to it: `mild` and `severe` already say `damaged`, and the two
        # damaged members of a triad must not both read as `..._damaged_...`.
        label = str(self.severity or self.condition)
        if self.phase == _ARTIFACT_PHASE_TRAIN:
            return "train_iter%04d_ep%06d_seed%d_%s_tag%06d" % (
                int(self.iteration), int(self.episode_index), int(self.seed),
                label, int(self.episode_tag),
            )
        return "%s_r%03d_e%03d_m%d_seed%d_%s_tag%06d" % (
            str(self.phase), int(self.eval_round_ordinal),
            int(self.eval_episode_index), int(self.eval_pair_member),
            int(self.seed), label, int(self.episode_tag),
        )

    def to_record(self) -> Dict[str, Any]:
        """Every field, explicitly, for the manifest."""
        return {
            "phase": str(self.phase),
            "iteration": None if self.iteration is None else int(self.iteration),
            "updates_completed": int(self.updates_completed),
            "eval_round_ordinal": (
                None if self.eval_round_ordinal is None
                else int(self.eval_round_ordinal)),
            "eval_episode_index": (
                None if self.eval_episode_index is None
                else int(self.eval_episode_index)),
            "eval_pair_member": (
                None if self.eval_pair_member is None else int(self.eval_pair_member)),
            "attempt_ordinal": int(self.attempt_ordinal),
            "episode_index": (
                None if self.episode_index is None else int(self.episode_index)),
            "seed": int(self.seed),
            "condition": str(self.condition),
            "severity": None if self.severity is None else str(self.severity),
            "episode_tag": int(self.episode_tag),
        }


class _AttemptArtifacts:
    """The bundle of ONE selected attempt: its directory, its files and its manifest.

    Lifecycle, in the order ``_run_one_episode`` drives it:

      ``open()``                       -- claim the directory (a collision RAISES) and
                                          write the ``incomplete`` manifest;
      ``capture_known_only_scenario()``-- copy the generator's file BYTES;
      ``capture_executed_t0_scenario()``- serialize ``Game.export_scenario()`` from the
                                          authoritative env-2 game, before the fuel-damage
                                          controller exists and before the first tick;
      (the tick loop writes the playback recording into this same directory)
      ``sync_recordings()``            -- immediately after a COMPLETED ``run_episode``,
                                          list the playback chunks that really exist and
                                          record them while the manifest is still
                                          ``incomplete``;
      ``finalize()``                   -- require the whole bundle, reconcile expected
                                          against observed target counts, and mark the
                                          manifest ``complete``.

    ``sync_recordings`` is split out of ``finalize`` because the two answer different
    questions and an attempt can die between them. The long baseline left 17 real playback
    files whose manifests never listed them: the episode had completed and exported its
    recording, and a later validation failure meant ``finalize`` was never reached, so the
    only record of the file was the file. An ``incomplete`` manifest is allowed to say the
    attempt did not finish; it is not allowed to be silent about an artifact it holds.

    The manifest is rewritten after every step, so an attempt that dies mid-way leaves a
    truthful ``incomplete`` record of exactly what had been captured -- never a fabricated
    one. Only a ``complete`` manifest whose files exist may be read as a full bundle.
    """

    def __init__(self, *, root: Union[str, Path], identity: _AttemptIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self.directory = self.root / identity.directory_name
        self._known_only: Optional[str] = None
        self._executed_t0: Optional[str] = None
        self._recordings: Tuple[str, ...] = ()
        self._targets: Dict[str, Any] = {}
        self._status = _ARTIFACT_STATUS_INCOMPLETE

    # ------------------------------------------------------------------
    @property
    def recording_export_path(self) -> str:
        """Where the tick loop's recorder writes -- this bundle's own directory.

        Handed to ``setup_episode(recording_export_path=...)``, which is the ONLY way
        recording is armed. Nothing here touches the recorder itself.
        """
        return str(self.directory)

    def open(self) -> "_AttemptArtifacts":
        """Create the attempt directory and write the initial ``incomplete`` manifest.

        ``exist_ok=False`` on purpose: two attempts sharing a directory would interleave
        two episodes' scenarios and recordings into one unreadable bundle. The tag makes
        that impossible by construction, so a collision means an assumption broke and it
        must fail LOUDLY rather than merge or overwrite.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise _VisualArtifactError(
                "visual artifacts: %s already exists -- two attempts would share one "
                "bundle. Nothing was overwritten." % str(self.directory)
            ) from exc
        except OSError as exc:
            raise _VisualArtifactError(
                "visual artifacts: could not create %s: %s"
                % (str(self.directory), exc)
            ) from exc
        self._write_manifest()
        return self

    def capture_known_only_scenario(self, scenario_path: Union[str, Path]) -> None:
        """Preserve the generator's known-only scenario as EXACT BYTES.

        A copy, not a re-serialization: normalizing, reformatting or rebuilding it from
        tasks would produce a file that is not the one the run generated. The original
        under ``<run_dir>/scenarios`` is left untouched.
        """
        try:
            payload = Path(str(scenario_path)).read_bytes()
            (self.directory / _ARTIFACT_KNOWN_ONLY_SCENARIO).write_bytes(payload)
        except OSError as exc:
            raise _VisualArtifactError(
                "visual artifacts: could not preserve the known-only scenario %s in %s: %s"
                % (str(scenario_path), str(self.directory), exc)
            ) from exc
        self._known_only = _ARTIFACT_KNOWN_ONLY_SCENARIO
        self._write_manifest()

    def capture_executed_t0_scenario(self, game: Any) -> None:
        """Serialize the AUTHORITATIVE executed world at t=0, straight off the engine.

        ``Game.export_scenario()`` is the client-loadable wrapper the engine already
        exposes, and env-2 is the sole runtime source of truth (B3), so this is the only
        thing that is the six-target world the episode really runs. Called EXACTLY ONCE
        per bundle, before the fuel-damage controller is built and therefore before the
        top-of-tick mutation, any policy decision and any ``env.step``.

        Read-only: the returned object is serialized and dropped. It is never modified and
        never fed back into execution.
        """
        try:
            exported = game.export_scenario()
        except Exception as exc:  # noqa: BLE001 - an artifact read must not be a stage
            raise _VisualArtifactError(
                "visual artifacts: Game.export_scenario() failed for %s: %s: %s"
                % (str(self.directory), type(exc).__name__, exc)
            ) from exc
        try:
            with open(self.directory / _ARTIFACT_EXECUTED_T0_SCENARIO, "w",
                      encoding="utf-8") as fh:
                json.dump(exported, fh, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            raise _VisualArtifactError(
                "visual artifacts: could not write the executed t=0 scenario in %s: %s"
                % (str(self.directory), exc)
            ) from exc
        self._executed_t0 = _ARTIFACT_EXECUTED_T0_SCENARIO
        self._write_manifest()

    def sync_recordings(self) -> Tuple[str, ...]:
        """List the playback chunks the completed run really wrote, into an INCOMPLETE manifest.

        Called immediately after ``run_episode`` returns, before any measurement is
        validated, so the manifest names the artifact from the first moment the artifact
        exists. DISCOVERY ONLY -- nothing is created, renamed or fabricated here: the file
        set is whatever the recorder produced, and the recorder is the tick loop's, driven
        through the locked ``setup_episode(recording_export_path=...)`` contract.

        The recording is REQUIRED and never fabricated: the tick-loop contract exports one
        on every completed run and none when the loop raised, so a COMPLETED run with no
        playback file means recording was not really armed -- an infrastructure fault, not
        a quiet omission.

        Returns:
            The discovered chunk names, sorted.

        Raises:
            _VisualArtifactError: if the directory cannot be listed, or holds no playback.
        """
        try:
            recordings = sorted(
                p.name for p in self.directory.glob(_ARTIFACT_RECORDING_GLOB)
            )
        except OSError as exc:
            raise _VisualArtifactError(
                "visual artifacts: could not list %s: %s" % (str(self.directory), exc)
            ) from exc
        if not recordings:
            raise _VisualArtifactError(
                "visual artifacts: the episode completed but no BLADE playback %s was "
                "written to %s -- recording was not armed on the executed environment."
                % (_ARTIFACT_RECORDING_GLOB, str(self.directory))
            )
        self._recordings = tuple(recordings)
        # Written while the status is still `incomplete`: if the attempt dies during the
        # measurement validation that follows, this is the truthful record of a real file.
        self._write_manifest()
        return self._recordings

    def finalize(self, *, expected: Dict[str, int], observed: Dict[str, int]) -> None:
        """Require the whole bundle, reconcile the target counts, and mark it ``complete``.

        ``complete`` is a CLAIM -- that this bundle holds the three artifacts and that they
        describe the world the schedule asked for. It is therefore refused, loudly, when
        the observed cardinality differs from the expected one: the long baseline shipped
        11 `complete` manifests reporting ``3 known / 2 hidden / 5 total`` while their own
        authoritative ``executed_t0_scenario.json`` held 3 + 3 = 6, and a manifest that
        certifies a world its own files contradict is worse than no manifest.

        On a mismatch the observed counts are still WRITTEN, and the status stays
        ``incomplete``: the point is to record what was seen, not to hide it. The raise is
        a :class:`_VisualArtifactError` -- infrastructure / data integrity -- so it aborts
        the run and can never be booked as a scientific episode failure.

        The playback comes from :meth:`sync_recordings`, which must have run first; this
        method never discovers one of its own and never fabricates one.
        """
        missing = [name for name, value in (
            (_ARTIFACT_KNOWN_ONLY_SCENARIO, self._known_only),
            (_ARTIFACT_EXECUTED_T0_SCENARIO, self._executed_t0),
            (_ARTIFACT_RECORDING_GLOB, self._recordings or None),
        ) if value is None]
        if missing:
            raise _VisualArtifactError(
                "visual artifacts: %s was never captured for %s"
                % (", ".join(missing), str(self.directory))
            )
        expected_counts = {k: int(v) for k, v in dict(expected).items()}
        observed_counts = {k: int(v) for k, v in dict(observed).items()}
        self._targets = {"expected": expected_counts, "observed": observed_counts}
        mismatched = [
            "%s expected %d, observed %r" % (key, value, observed_counts.get(key))
            for key, value in expected_counts.items()
            if observed_counts.get(key) != value
        ]
        if mismatched:
            # Recorded, then refused: the manifest stays `incomplete` and now says why.
            self._write_manifest()
            raise _VisualArtifactError(
                "visual artifacts: %s cannot be marked complete -- the observed world "
                "contradicts the scheduled cell (%s). The bundle is left incomplete."
                % (str(self.directory), "; ".join(mismatched))
            )
        self._status = _ARTIFACT_STATUS_COMPLETE
        self._write_manifest()

    # ------------------------------------------------------------------
    def to_manifest(self) -> Dict[str, Any]:
        """The manifest as a dict (the same object that is written to disk)."""
        return {
            "schema": _ARTIFACT_MANIFEST_SCHEMA,
            "version": _ARTIFACT_MANIFEST_VERSION,
            "status": str(self._status),
            "identity": self.identity.to_record(),
            # Restated at the top level as well: an operator matching a bundle against
            # `<run_dir>/scenarios/episode_<tag>_scenario.json` should not have to know
            # where inside the identity block the tag lives.
            "source_episode_tag": int(self.identity.episode_tag),
            "known_only_scenario": self._known_only,
            "executed_t0_scenario": self._executed_t0,
            "playback_recordings": list(self._recordings),
            "targets": dict(self._targets),
        }

    def _write_manifest(self) -> None:
        try:
            with open(self.directory / _ARTIFACT_MANIFEST, "w", encoding="utf-8") as fh:
                json.dump(self.to_manifest(), fh, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            raise _VisualArtifactError(
                "visual artifacts: could not write %s in %s: %s"
                % (_ARTIFACT_MANIFEST, str(self.directory), exc)
            ) from exc


@dataclass(frozen=True)
class _ScheduledCell:
    """The world the roster is REQUIRED to describe, and where each number came from.

    Built by :func:`_scheduled_cell`, which is the only site allowed to decide what
    "hidden" should be -- see there for why the answer differs by design.
    """

    n_known: int
    n_hidden: int
    n_targets_executed: int
    hidden_requested: int
    realized_short: bool


def _scheduled_cell(
    cardinality: EpisodeCardinality, construction_audit: Any
) -> _ScheduledCell:
    """What the roster must contain, given the schedule and the construction's own audit.

    THE TWO DESIGNS DIFFER IN EXACTLY ONE PLACE -- what "hidden" means:

      * ``exact_v1`` construction is EXACT by contract, so the expected hidden count IS
        the requested one and any disagreement is a measurement fault. There is no audit
        on that path (``construction_audit is None``), which is precisely how a reader
        tells which policy ran.
      * ``bounded_backoff_v1`` may legitimately realize FEWER hidden targets than were
        requested (handoff 3l.1), so the expected count is the audit's ``hidden_realized``
        -- the construction's own statement of what it really built. The REQUEST is not
        rewritten to match it: it travels beside it so requested-vs-realized stays
        readable as a distribution, and a short realization is a recorded outcome rather
        than a failure.

    The audit is VERIFIED rather than trusted where it exists: it must agree with the
    schedule about ``A`` and about ``H_requested``, because an audit that disagrees with
    the schedule that produced it means the two describe different episodes.

    Raises:
        EpisodeRosterError: the construction audit contradicts the schedule.
    """
    known = int(cardinality.known_count)
    requested = int(cardinality.hidden_requested)
    if construction_audit is None:
        return _ScheduledCell(
            n_known=known,
            n_hidden=requested,
            n_targets_executed=known + requested,
            hidden_requested=requested,
            realized_short=False,
        )
    audit_agents = int(getattr(construction_audit, "agent_count", -1))
    audit_requested = int(getattr(construction_audit, "hidden_requested", -1))
    audit_realized = int(getattr(construction_audit, "hidden_realized", -1))
    wrong = []
    if audit_agents != int(cardinality.agent_count):
        wrong.append("agent count: audit %d, scheduled %d"
                     % (audit_agents, int(cardinality.agent_count)))
    if audit_requested != requested:
        wrong.append("hidden requested: audit %d, scheduled %d"
                     % (audit_requested, requested))
    if wrong:
        raise EpisodeRosterError(
            "the construction audit contradicts the schedule that produced it (%s): the "
            "episode and its accounting describe different worlds"
            % "; ".join(wrong)
        )
    return _ScheduledCell(
        n_known=known,
        n_hidden=audit_realized,
        n_targets_executed=known + audit_realized,
        hidden_requested=requested,
        realized_short=audit_realized < requested,
    )


def _require_scheduled_cell(
    roster: "_TargetRoster", expected: _ScheduledCell
) -> None:
    """The roster must describe the cell the schedule asked for, or the run ABORTS.

    ``expected`` states the cell exactly -- how many known targets, how many constructed
    ones, and how many the executed world holds -- and ``setup_episode``'s construction
    path already enforces its own cardinality LOUDLY on its own side (exact
    ``len(placements) == n_hidden`` under ``exact_v1``, a realized-count reconciliation
    under ``bounded_backoff_v1``, a known-target-loss check and a world cardinality
    check). So a roster that disagrees is not a scenario that came out differently; it is
    this module measuring the world wrongly, and every number derived from it --
    confirmation counts, denominators, manifest target blocks -- is suspect.

    Raised as an :class:`EpisodeRosterError` for that reason: a measurement-integrity
    fault, not a scientific episode outcome. Checked BEFORE the fuel-damage plan and
    before ``run_episode``, so nothing is paid for and no partial measurement exists.

    A generalized world that realized FEWER hidden targets than requested is NOT a
    disagreement -- :func:`_scheduled_cell` already resolved the expectation against the
    construction's own audit, so the shortfall is accounted, not rejected.
    """
    checks = (
        ("known targets", len(roster.known_ids), int(expected.n_known)),
        ("hidden targets", len(roster.hidden_ids), int(expected.n_hidden)),
        ("executed targets", int(roster.total), int(expected.n_targets_executed)),
    )
    wrong = ["%s: observed %d, scheduled %d" % (what, got, want)
             for what, got, want in checks if got != want]
    if wrong:
        raise EpisodeRosterError(
            "the t=0 roster does not describe the scheduled cell (%s): the episode was "
            "measured against a world the configuration did not ask for"
            % "; ".join(wrong)
        )


def _reward_breakdown_record(ep_reward: Any) -> Dict[str, Any]:
    """The ``EpisodeReward`` fields this run PERSISTS, read duck-typed.

    An explicit field list rather than ``asdict``, for two reasons. It pins exactly which
    of the reward's fields become a durable record -- so a future field added to
    ``EpisodeReward`` for an internal purpose does not silently start appearing in a
    scientific artifact -- and it reads through ``getattr``, which is how this module
    already consumes every optional structure it does not own. That keeps the record
    writable from the lightweight reward stubs the test suite drives the trainer with,
    without weakening anything: a REAL ``EpisodeReward`` supplies every field below.

    ``u_oracle`` and the checkpoint fields stay ``None`` where the reward left them
    ``None`` -- on a normalized regret scale ``0`` is the OPTIMUM, so a coerced zero
    would read as a perfect measurement rather than an absent one.
    """
    fields = (
        "u_achieved", "u_oracle", "u_ref", "u_aircraft", "n_lost", "ratio", "penalty",
        "reward", "reference_policy", "reference_kind", "checkpoint_tick",
        "u_prefix", "u_cont_ref", "u_post", "unique_completed_targets",
        "scored_completed_targets", "unscored_completed_targets",
    )
    record: Dict[str, Any] = {
        name: getattr(ep_reward, name, None) for name in fields
    }
    record["unscored_completed_target_ids"] = [
        str(t) for t in (getattr(ep_reward, "unscored_completed_target_ids", ()) or ())
    ]
    return record


def _observe_world_identity(
    ctx: Any, *, roster: "_TargetRoster", fd_plan_record: Dict[str, Any]
) -> WorldIdentity:
    """The ID-FREE identity of the world an episode really built.

    This is what makes "the three members of a matched benchmark group ran the SAME
    world" checkable rather than assumed, and what a frozen manifest's preflight is
    compared against. Every component is deliberately chosen to be reproducible from the
    SEED, so it can be compared BETWEEN runs:

      * the realized cardinality, taken from the RAW pre-solve world snapshots through the
        roster -- never from an allocated-only task list (the Stage-0 world-inventory
        contract);
      * ``geometric_fingerprint(ctx.placements)`` -- COORDINATES ONLY. Generated target
        uuids are not seed-derived (``CLAUDE.md`` section 8), so an id-keyed comparison
        would report two runs of the same seed as different worlds;
      * the certified damaged ego's ORDINAL, for exactly the same reason -- its uuid is
        not seed-derived, its position in the scheduled agent sequence is;
      * a content hash of the FD event certificate with the ego uuid REMOVED, which pins
        the event tick, position, fuel and both severity bands in one comparable scalar.

    ``None`` components are truthful absences: a legacy (uncertified) plan has no
    certificate and no eligibility ordinal, and fabricating either would make "these two
    worlds certified the same event" answerable where it is not.
    """
    audit = (fd_plan_record or {}).get("eligibility_audit") or {}
    selected_ordinal = audit.get("selected_ordinal")
    return WorldIdentity(
        hidden_realized=len(roster.hidden_ids),
        known_realized=len(roster.known_ids),
        geometric_fingerprint=geometric_fingerprint(getattr(ctx, "placements", ())),
        fd_selected_ordinal=(
            None if selected_ordinal is None else int(selected_ordinal)
        ),
        fd_certificate_fingerprint=certificate_fingerprint(
            (fd_plan_record or {}).get("certificate")
        ),
    )


def _v2_allocation_fingerprint(ctx: Any) -> str:
    """The UUID-free known-only allocation fingerprint of a freshly set-up V2 context.

    CALL AFTER ``setup_episode`` AND BEFORE ``run_episode``: the beliefs are byte-equal to
    ``A_init``'s task list only at t=0 (the roster has already verified that agreement).
    Task indices are mapped to positions in the RAW known-world inventory, so the
    fingerprint never rests on a generated uuid.

    Raises:
        MeasurementIntegrityError: the context cannot state its own allocation -- an
            instrument contradiction, never an episode outcome.
    """
    agent_ids = [str(a) for a in (getattr(ctx, "agent_ids", None) or ())]
    beliefs = getattr(ctx, "beliefs", None) or {}
    if not agent_ids or agent_ids[0] not in beliefs:
        raise MeasurementIntegrityError(
            "the V2 allocation fingerprint needs the t=0 belief of the first scheduled "
            "ego, and the context carries none")
    belief_target_ids = [_task_target_id(t) for t in beliefs[agent_ids[0]].tasks]
    if any(tid is None for tid in belief_target_ids):
        raise MeasurementIntegrityError(
            "an allocated known task names no target, so the V2 allocation fingerprint "
            "cannot be stated")
    try:
        return v2_allocation_fingerprint(
            getattr(ctx, "a_init", None) or {},
            agent_ids=agent_ids,
            belief_target_ids=[str(t) for t in belief_target_ids],
            known_target_ids=[str(t) for t in (getattr(ctx, "known_target_ids", ()) or ())],
        )
    except ValueError as exc:
        raise MeasurementIntegrityError(
            "the known-only allocation contradicts the known-world inventory (%s)" % exc
        ) from exc


def _observe_v2_world_identity(
    *,
    seed: int,
    pre_solve: PreSolveCardinality,
    route_relative_load: Optional[RouteRelativeHiddenLoad],
    match_aou_backend: str,
    allocation_fingerprint: Optional[str],
    world_identity: Optional[WorldIdentity],
) -> V2WorldIdentity:
    """ONE assembly site for a V2 world's frozen-comparable identity.

    Used by the preflight (which freezes it) and by the evaluation round (which verifies a
    reconstruction against it), so the two cannot come to describe a world differently.
    Every input is a RECORDED fact -- the stage-2 draw is read off the construction
    path's own record and is never re-derived from the seed.

    Raises:
        MeasurementIntegrityError: a component the V2 identity requires is absent.
    """
    missing = [name for name, value in (
        ("route_relative_load", route_relative_load),
        ("allocation_fingerprint", allocation_fingerprint),
        ("world_identity", world_identity)) if value is None]
    if missing:
        raise MeasurementIntegrityError(
            "a generalized_v2 benchmark world cannot state its frozen identity: %s absent"
            % ", ".join(missing))
    return V2WorldIdentity(
        seed=int(seed),
        agent_count=int(pre_solve.agent_count),
        known_count=int(pre_solve.known_count),
        match_aou_backend=resolve_match_aou_backend(match_aou_backend),
        route_count=int(route_relative_load.route_count),          # type: ignore[union-attr]
        allocation_fingerprint=str(allocation_fingerprint),
        hidden_requested=int(route_relative_load.hidden_requested),  # type: ignore[union-attr]
        hidden_realized=int(world_identity.hidden_realized),        # type: ignore[union-attr]
        known_realized=int(world_identity.known_realized),          # type: ignore[union-attr]
        geometric_fingerprint=world_identity.geometric_fingerprint,  # type: ignore[union-attr]
        fd_selected_ordinal=world_identity.fd_selected_ordinal,     # type: ignore[union-attr]
        fd_certificate_fingerprint=world_identity.fd_certificate_fingerprint,  # type: ignore[union-attr]
    )


def _recording_kwargs(artifacts: Optional[_AttemptArtifacts]) -> Dict[str, Any]:
    """``setup_episode``'s recording keyword -- or NOTHING at all when artifacts are off.

    Not ``{"recording_export_path": None}``: the OFF path must call ``setup_episode``
    exactly as it did before this feature existed, so the keyword is absent rather than
    present-and-empty.
    """
    if artifacts is None:
        return {}
    return {"recording_export_path": artifacts.recording_export_path}


def _artifact_kwargs(artifacts: Optional[_AttemptArtifacts]) -> Dict[str, Any]:
    """``_run_one_episode``'s artifact keyword -- or NOTHING when the run did not opt in.

    Same rule as :func:`_recording_kwargs`, one level up: a run with artifacts off calls
    ``_run_one_episode`` with exactly the arguments it did before this feature existed,
    rather than with a new keyword carrying ``None``.
    """
    if artifacts is None:
        return {}
    return {"artifacts": artifacts}


def _ctde_kwargs(recorder: Optional[CentralStateRecorder]) -> Dict[str, Any]:
    """``_run_one_episode``'s CTDE keyword -- or NOTHING on an ``actor_only`` run.

    The SAME rule as :func:`_artifact_kwargs`, and for the same reason: an
    ``actor_only`` run must call ``_run_one_episode`` with exactly the arguments it did
    before Phase B existed, not with a new keyword carrying ``None``. That is also the
    stronger invariance claim -- "the actor-only call is byte-unchanged" rather than
    "the actor-only call passes a falsy value" -- and it is what keeps existing callers
    that stub ``_run_one_episode`` with a fixed signature working.
    """
    if recorder is None:
        return {}
    return {"central_recorder": recorder}


def _central_kwargs(recorder: Optional[CentralStateRecorder]) -> Dict[str, Any]:
    """``run_episode``'s CTDE keyword -- or NOTHING when there is no recorder.

    :func:`_ctde_kwargs` one level down: on an ``actor_only`` run the tick loop is
    called exactly as it was before Phase B, so no central state is constructed and the
    episode is byte-identical to the Phase-A one.
    """
    if recorder is None:
        return {}
    return {"central": recorder}


# =============================================================================
# 4. One episode (shared by training and evaluation)
# =============================================================================

@dataclass
class _EpisodeOutcome:
    """What one finished episode hands back after its env is closed.

    ``trajectory`` survives the env close by construction: a ``Transition`` holds a
    ``GraphObservation`` (numpy arrays + id strings) and detached floats -- no BLADE
    handle -- so the buffer can outlive the episode it came from.

    The observability fields are the same story: plain ints and name STRINGS, resolved
    while the env was still open, so a block can be printed (and an aggregate taken)
    after the environment is gone.

    NONE OF THEM HAS A DEFAULT, deliberately. They used to default to an empty roster so
    that a caller could build an outcome without one -- which is precisely how a degraded
    roster produced a SUCCESSFUL ``0/0`` measurement. An outcome now cannot be
    constructed without stating what was measured, so the false zero is not expressible
    here at all.

    TWO COUNTS, DELIBERATELY BOTH PRESENT:
      * ``confirmed_kills`` is ``EpisodeResult.confirmed_kills`` verbatim -- the number
        of ``(ego_id, target_id)`` CONFIRMATIONS in ``GraphPlanExecutor.done``. It is
        kept so this record stays a faithful mirror of what the tick loop reported, and
        it is what could exceed the world's target count when two egos confirm one kill.
      * ``targets_confirmed_unique`` is ``len(_unique_confirmed_target_ids(done))`` --
        TARGETS, deduplicated over ego -- out of ``targets_total``. It is the only one
        printed or aggregated, and it is computed DIRECTLY from the id set, never from
        how many names the roster managed to classify.
    """

    trajectory: List[Any]
    reward: float
    ticks: int
    ended: str
    n_wakes: int
    confirmed_kills: int
    n_dead: int
    seconds: float

    # --- observability (see the class docstring; no defaults, on purpose) ---
    targets_confirmed_unique: int
    targets_total: int
    known_target_names: Tuple[str, ...]
    hidden_target_names: Tuple[str, ...]
    known_confirmed_names: Tuple[str, ...]
    hidden_confirmed_names: Tuple[str, ...]

    # --- FD-BASELINE-v1 (no defaults either, for the same reason) ---
    # `fuel_damage_plan` / `fuel_damage_outcome` are the component's own frozen records,
    # carried whole rather than flattened into a dozen fields: they are what the component
    # is contracted to expose, and copying a subset here would create a second, drifting
    # description of the same event.
    fuel_damage_plan: Dict[str, Any]
    fuel_damage_outcome: Dict[str, Any]
    # Did the SELECTED ego return to base? Read off `executor.rtb_issued` after the run.
    # `None` on a clean episode -- there is no selected ego, and `False` would read as
    # "the ego did not RTB", a claim about an ego that does not exist.
    selected_ego_rtb_issued: Optional[bool]

    # --- GENERALIZED-V1 (Task 4): the per-episode diagnostics Tasks 1-3 produce -----
    # These DO carry defaults, and the reason is different from the fields above rather
    # than a relaxation of them. The fields above are what an episode MEASURED, and a
    # missing one is the false-zero this class exists to make inexpressible. These are
    # STRUCTURES a design may or may not produce at all: an `exact_v1` construction has
    # no backoff audit, a legacy fuel-damage plan has no certificate, a `single_wake_v1`
    # run has no post-FD adaptation, and a `static_t0_v1` episode has no reference
    # object. `None` here means "this design produced no such structure", which is a
    # truthful statement and not an absent measurement.
    #
    # `_run_one_episode` populates every one of them on BOTH designs wherever the
    # structure exists, so a `None` in a real run's record is a fact about the design.
    cardinality: Optional[EpisodeCardinality] = None
    """The REQUESTED cardinality this attempt was scheduled with (never rewritten)."""

    pre_solve_cardinality: Optional[PreSolveCardinality] = None
    """GENERALIZED-V2 STAGE 1: the ``(A, K)`` draw made BEFORE the known-only solve.

    Kept beside the resolved cardinality rather than folded into it, so "what was requested
    before anything was solved" stays separately readable from "what the route count turned
    that into". ``None`` on every other design, where the cell was complete up front."""

    route_relative_load: Optional[RouteRelativeHiddenLoad] = None
    """GENERALIZED-V2 STAGE 2: the route count ``R`` and the ``H | R`` it produced.

    ``None`` on every other design -- that ABSENCE is how a reader tells the hidden count
    was stated by the caller rather than drawn against a realized route count."""

    hidden_realized: Optional[int] = None
    """How many hidden targets the construction really built (<= requested)."""

    construction_audit: Optional[Dict[str, Any]] = None
    """``ConstructionAudit.as_dict()`` -- GENERALIZED-V1 bounded backoff only."""

    post_fd_adaptation: Optional[Dict[str, Any]] = None
    """``PostFdAdaptationOutcome.to_record()`` -- completion-boundary policy only."""

    reference: Optional[Dict[str, Any]] = None
    """``EpisodeReference.to_record()`` -- event-conditioned reference policy only."""

    reward_breakdown: Optional[Dict[str, Any]] = None
    """The ``EpisodeReward`` decomposition, carried whole rather than re-derived."""

    world_identity: Optional[WorldIdentity] = None
    """The id-free identity of the world this attempt really built (benchmark checks)."""

    allocation_fingerprint: Optional[str] = None
    """GENERALIZED-V2 only: the UUID-free fingerprint of the known-only allocation
    (``graph_generalized.v2_allocation_fingerprint``), taken at t=0 before any tick. It is
    what lets a V2 benchmark member prove it was built on the frozen allocation. ``None``
    on every other design. Not written into any record by itself."""


def _run_one_episode(
    policy: Any,
    gen: ScenarioGenerator,
    cfg: TrainConfig,
    *,
    seed: int,
    episode_tag: int,
    deterministic: bool,
    fuel_damage_mode: Optional[str] = None,
    artifacts: Optional[_AttemptArtifacts] = None,
    central_recorder: Optional[CentralStateRecorder] = None,
    cardinality: Optional[EpisodeCardinality] = None,
    pre_solve_cardinality: Optional[PreSolveCardinality] = None,
    population_recorder: Optional[RouteRelativePopulationRecorder] = None,
) -> _EpisodeOutcome:
    """Generate -> setup -> run -> reward for ONE episode; always closes its env.

    Identical in structure for training and evaluation -- the only differences are
    ``deterministic`` and which seed / tag band the caller draws from, which is what
    makes the two paths comparable.

    The reseed of global ``random`` + torch happens HERE, at the episode head, so an
    episode's RNG state is a pure function of ``seed`` regardless of what ran before it
    (the generator has its own ``random.Random(seed)``, and action sampling draws from
    torch's global RNG). Hidden placement does NOT ride on global ``random``: it gets its
    own explicit ``random.Random(seed)``, so it is reproducible even if a future change
    adds or removes a global-random consumer earlier in the episode.

    Raises :class:`EpisodeAttemptError` -- the original exception wrapped with the
    PIPELINE STAGE it came from, and preserved as ``__cause__`` / ``.original``. Every
    stage is wrapped, so a ledger entry can always name where an attempt died
    (:data:`_PIPELINE_STAGES`); ``setup`` in particular is where an exact-cardinality
    construction failure lands. The caller decides whether a failure aborts (it does
    not: see :func:`train`), and this function NEVER retries or substitutes a seed.

    OBSERVABILITY is collected inside this function because it is the only scope that
    still holds the context: the target roster is snapshotted between ``setup_episode``
    and ``run_episode`` (t=0 beliefs, and a scenario that has not lost a unit yet), the
    unique confirmed-target set is read off the executor after the run, and both survive
    the ``finally`` that closes the env because they are ints and strings.

    The roster is REQUIRED measurement structure, read from the context's RAW t=0 world
    snapshots (``known_target_ids`` / ``executed_target_ids``) and never from the
    allocated-only ``oracle_tasks``. A structural failure -- it cannot be built, it does
    not describe the scheduled cell, or it does not account for every confirmed target --
    raises :class:`EpisodeRosterError`, a :class:`MeasurementIntegrityError`. That is
    INFRASTRUCTURE: it is NOT wrapped as an ``EpisodeAttemptError``, never reaches
    ``episode_failures.jsonl``, never enters ``skip_and_account_v1``, cannot shrink a
    scientific denominator, and ABORTS the run. Only NAME resolution degrades, and it
    changes nothing but the printed text.

    THE POST-RUN ORDER IS PART OF THE CONTRACT. Once ``run_episode`` returns: the playback
    is synchronized into the manifest, the unique confirmed-target ids are computed, they
    are validated against the executed-world roster, and only then is the reward computed
    and a successful outcome produced. A confirmed target outside that snapshot aborts as
    data integrity -- it is never a post-hoc ``setup`` failure, and the recording it
    already wrote is never left unlisted.

    FD-BASELINE-v1. ``fuel_damage_mode`` overrides ``cfg.fuel_damage_mode`` for this one
    attempt; evaluation passes a forced mode per matched-pair member and training passes
    nothing. The damage plan is prepared BETWEEN setup and run -- it needs the solved
    ``a_init``, the t=0 beliefs and an aircraft that has not burned a tick, and it must
    exist before the first tick can fire it. A DAMAGED episode with no valid strict fuel
    window raises there and is attributed to ``setup``, so ``skip_and_account_v1`` records
    it once; it is never silently downgraded to a clean episode, which would change the
    population every per-condition statistic is reported over. A FORCED-CLEAN member
    computes no window at all and therefore cannot fail for this reason -- the two members
    of a pair fail independently or not at all.

    VISUAL ARTIFACTS. ``artifacts`` is ``None`` unless the run opted in, and on that OFF
    path this function is byte-unchanged: no directory is created, no scenario is copied,
    ``Game.export_scenario`` is not called and ``setup_episode`` receives no recording
    keyword at all (:func:`_recording_kwargs`). When a bundle IS supplied it is opened
    before generation, the generated known-only scenario is copied into it, recording is
    armed on the returned env-2, the authoritative executed t=0 scenario is exported once
    -- before the fuel-damage controller exists, hence before the top-of-tick mutation,
    any policy decision and any ``env.step`` -- and the bundle is completed after the
    reward. A capture failure raises :class:`_VisualArtifactError`, NOT an
    :class:`EpisodeAttemptError`: it is infrastructure, it belongs in no pipeline stage,
    and the callers re-raise it instead of accounting it as a failed episode.

    PHASE-B CTDE. ``central_recorder`` is ``None`` unless the run's ``training_mode`` is
    ``ctde`` AND this is a TRAINING attempt; on that ``None`` path this function is
    byte-unchanged and ``run_episode`` receives no CTDE keyword at all
    (:func:`_central_kwargs`), so no central state is ever built. When supplied it is
    filled during the run with ONE central state per actor decision, aligned 1:1 with
    ``EpisodeResult.trajectory``, and the CALLER reads ``recorder.samples`` afterwards.
    It is deliberately caller-owned rather than returned on :class:`_EpisodeOutcome`:
    privileged state is not part of what an episode REPORTS, and evaluation must be
    unable to obtain it -- ``evaluate`` never constructs one.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    fd_params = cfg.fuel_damage_parameters(fuel_damage_mode)
    # The cell this attempt was SCHEDULED with.
    #
    # ONE STAGE on every historical path: supplied by a generalized-V1 caller (sampled for
    # training, taken from the frozen stratum for a benchmark member) and omitted on the
    # fixed-cell path, where it resolves to the configured cell and every call below is
    # exactly the pre-Task-4 one.
    #
    # TWO STAGES under GENERALIZED-V2, and `cell` is deliberately `None` until the second
    # one completes. Its hidden load is drawn against the number of egos the known-only
    # solve routes, so no honest requested cardinality exists before `setup_episode`
    # returns -- and a placeholder would be indistinguishable from a request in every
    # record it reached. `pre_solve_cardinality` carries the half that DOES exist (`A` and
    # `K`), which is exactly what the generator needs.
    cell: Optional[EpisodeCardinality] = (
        None if pre_solve_cardinality is not None
        else episode_cardinality(cfg, seed, benchmark_cardinality=cardinality)
    )

    # Claimed BEFORE generation: the known-only scenario is the first artifact, and a
    # directory collision must be discovered before an episode is paid for.
    if artifacts is not None:
        artifacts.open()

    t0 = time.perf_counter()
    try:
        var = build_variation_config(
            cfg, seed,
            # The V2 pre-solve cell where there is one; otherwise the historical call.
            **({"cardinality": pre_solve_cardinality}
               if pre_solve_cardinality is not None
               else _cardinality_kwargs(cardinality)),
        )
        scenario_path = gen.generate(episode=int(episode_tag), config=var)
    except Exception as exc:
        raise EpisodeAttemptError("generation", exc) from exc

    if artifacts is not None:
        artifacts.capture_known_only_scenario(scenario_path)

    ctx = None
    try:
        try:
            ctx = setup_episode(
                scenario_path.read_text(encoding="utf-8"),
                # CONSTRUCTION PATH: the generated world is known-only, and setup builds
                # the hidden half from the solved routes (solve -> place -> patch ->
                # reload). `cfg.partial_ratio` is the legacy split surface and is
                # deliberately NOT passed -- `split_tasks` never runs here.
                # OMITTED under GENERALIZED-V2, whose hidden-load policy resolves the
                # count inside setup once the known-only solve has produced a route count.
                **({} if pre_solve_cardinality is not None
                   else {"n_hidden": int(cell.hidden_requested)}),   # type: ignore[union-attr]
                placement_rng=random.Random(seed),
                # GENERALIZED-V2: the route-relative hidden-load policy, its episode seed
                # and the scheduled known count. Absent entirely on every other design.
                **_v2_hidden_load_kwargs(cfg, seed, pre_solve_cardinality,
                                         recorder=population_recorder),
                # GENERALIZED-V1: the hidden-cardinality and reward-reference policies,
                # resolved from the ONE design selector. Absent entirely on the
                # historical path, where setup resolves its own `exact_v1` /
                # `static_t0_v1` defaults exactly as it always has.
                **_generalized_setup_kwargs(cfg),
                # WHICH MATCH-AOU objective every solve of THIS episode uses. Absent
                # entirely on the historical backend, where setup resolves its own
                # `legacy_minlp_v1` default exactly as it always has.
                **_backend_setup_kwargs(cfg),
                # Recording is ARMED here or nowhere; the tick loop drives it. Absent
                # entirely when artifacts are off.
                **_recording_kwargs(artifacts),
            )
        except MatchAouBackendError:
            # BACKEND / CONFIGURATION fault, not an episode outcome: the selected backend
            # could not be reached, or was handed a problem outside its contract. Re-raised
            # AHEAD of the broad handler so it is never wrapped as an accounted `setup`
            # failure, never entered into `skip_and_account_v1`, never replaced by the next
            # training seed -- and never answered by silently solving the other objective.
            raise
        except Exception as exc:
            raise EpisodeAttemptError("setup", exc) from exc

        # The authoritative six-target world, taken from env-2 -- the sole runtime source
        # of truth -- while nothing has run yet. Deliberately the FIRST thing after setup
        # returns, so no later step can be suspected of having moved it.
        if artifacts is not None:
            artifacts.capture_executed_t0_scenario(ctx.game)

        # The roster is snapshotted HERE -- after setup, before a single tick -- because
        # its inputs are t=0 facts: the N beliefs are byte-equal only now, and the live
        # scenario still holds every target it is about to lose to a kill. The WORLD half
        # comes off the context's raw pre-solve snapshots, so it is not affected by what
        # either solver selected.
        # NOT wrapped as an `EpisodeAttemptError`: a roster fault is a
        # `MeasurementIntegrityError`, so it propagates and the run stops. Accounting it
        # as a skipped `setup` attempt is what let the long baseline lose 143 training
        # attempts to a measurement defect while reporting itself reconciled. An
        # UNEXPECTED exception raised inside the roster code is normalized into the same
        # loud path, with its cause preserved -- an unforeseen internal error is still a
        # roster that could not be established, and must not fall through to the broad
        # episode handler below.
        # GENERALIZED-V2 STAGE 2 IS NOW A FACT ABOUT THE CONTEXT, so the requested cell
        # can finally be stated. Built as a NEW object from the two stage records -- the
        # pre-solve draw and the route-relative load setup performed -- rather than by
        # writing a hidden count into the stage-1 object, so neither stage's record is ever
        # revised. A context that declared the policy without producing the record is a
        # measurement-integrity fault, not an episode outcome: the population this episode
        # belongs to would be unstateable.
        if pre_solve_cardinality is not None:
            load = getattr(ctx, "route_relative_load", None)
            if load is None:
                raise MeasurementIntegrityError(
                    "the route-relative hidden-load policy was selected but the episode "
                    "context carries no record of the route count its hidden load was "
                    "drawn against, so this episode's requested cardinality cannot be "
                    "stated"
                )
            cell = resolved_v2_cardinality(pre_solve_cardinality, load)

        try:
            roster = _episode_target_roster(ctx)
            # DUCK-TYPED, like every other optional context field this module reads
            # (`graph_reward.uses_event_conditioned_reference` does the same): an ABSENT
            # audit resolves to the historical exact-cardinality expectation, which is
            # the only direction an unknown context may ever be read in.
            scheduled = _scheduled_cell(
                cell, getattr(ctx, "construction_audit", None))   # type: ignore[arg-type]
            _require_scheduled_cell(roster, scheduled)
        except MeasurementIntegrityError:
            raise
        except Exception as exc:
            raise EpisodeRosterError(
                "the t=0 target roster could not be established (%s: %s)"
                % (type(exc).__name__, exc)
            ) from exc

        # GENERALIZED-V2 ONLY: the UUID-free fingerprint of the known-only allocation,
        # taken HERE while the beliefs are still the t=0 plan. A V2 benchmark member is
        # verified against its frozen value; nothing else reads it, and no other design
        # computes it.
        allocation_fingerprint: Optional[str] = (
            _v2_allocation_fingerprint(ctx) if pre_solve_cardinality is not None
            else None
        )

        # The damage plan is a t=0 fact about the context setup produced -- the solved
        # routes, the untouched fuel -- so a plan that cannot be built (no eligible ego,
        # no valid strict window) is a `setup` finding, accounted exactly like an
        # exact-cardinality construction failure and never repaired into a clean episode.
        try:
            fuel_damage = build_fuel_damage_controller(
                ctx, episode_seed=int(seed), params=fd_params
            )
        except FuelDamageIntegrityError:
            # INSTRUMENT failure, not science -- see the run-stage re-raise below. An
            # ineligible candidate is NOT this exception; it is an ordinary
            # `FuelDamageError` and falls through to the accounted `setup` stage.
            raise
        except Exception as exc:
            raise EpisodeAttemptError("setup", exc) from exc

        try:
            result = run_episode(
                policy, ctx,
                deterministic=deterministic,
                max_ticks=cfg.max_ticks,
                fuel_damage=fuel_damage,
                # Absent entirely on an actor_only run (`_central_kwargs`), so the tick
                # loop is called exactly as it was before Phase B.
                **_central_kwargs(central_recorder),
            )
        except ReferenceIntegrityError as exc:
            # GENERALIZED-V1 Task 4: the reference layer states WHY it refused, as a
            # stable slug, and the routing reads that slug and NOTHING ELSE -- never the
            # message text. An INSTRUMENT contradiction (a policy that requires a
            # reference produced none, nothing retained to solve from, an unknown kind, a
            # record whose arithmetic does not close) implicates every episode the layer
            # touched, so it is re-raised here and ABORTS, exactly as a roster or
            # certificate fault does. A solver that was ASKED and did not ANSWER is a
            # fact about this one attempt: it falls through to the broad handler below and
            # is accounted as ordinary `run`-stage attrition, spent once and never retried.
            if reference_fault_aborts(exc):
                raise
            raise EpisodeAttemptError("run", exc) from exc
        except MatchAouBackendError:
            # A DEFERRED reference solve (the clean t=0 build, or the fuel-damage
            # continuation checkpoint) hit the same backend fault the setup solve can. It
            # is routed identically and for the same reason: the instrument is configured
            # against a domain the selected backend does not model, which implicates every
            # episode it touched -- so it aborts instead of being spent as `run` attrition.
            raise
        except FuelDamageIntegrityError:
            # GENERALIZED-V1 (handoff 3l.3): under the certified eligibility policy this
            # world was proven FD-capable BEFORE a tick was paid for, so a live event
            # state that contradicts its own certificate is a fault in the INSTRUMENT,
            # not an outcome of the experiment. Re-raised ahead of the broad handler so
            # it is never wrapped as an `EpisodeAttemptError("run", ...)`, never written
            # to `episode_failures.jsonl`, never counted against a condition tally and
            # never entered into `skip_and_account_v1` -- the same routing
            # `MeasurementIntegrityError` and `_VisualArtifactError` already have, and
            # for the same reason: a defect that silently shrinks a scientific
            # denominator is worse than one that stops the run.
            raise
        except Exception as exc:
            raise EpisodeAttemptError("run", exc) from exc

        # ORDER MATTERS FROM HERE. The playback is synchronized first, then the world is
        # validated, and only a world that validated is allowed to produce a reward and a
        # successful outcome. The long baseline ran it the other way round: 17 episodes
        # completed, exported a playback and computed a reward, and were then failed on a
        # confirmed id -- leaving a real recording no manifest listed, and booking a
        # measurement fault as a post-hoc `setup` episode failure.
        if artifacts is not None:
            artifacts.sync_recordings()

        # THE AUTHORITATIVE COUNT. It is `len()` of the deduplicated id set taken straight
        # off the executor -- a target both egos confirmed is ONE target here -- and it is
        # NOT derived from how many of those ids the roster managed to name.
        # `result.confirmed_kills` below still reports the raw (ego, target) confirmation
        # count, unchanged.
        confirmed_ids = _unique_confirmed_target_ids(
            getattr(ctx.executor, "done", None)
        )
        targets_confirmed_unique = len(confirmed_ids)

        # A confirmed target outside the AUTHORITATIVE executed-world snapshot means the
        # executor and the roster are describing different worlds. That is data integrity,
        # not an episode outcome: it aborts, and it is never written to the ledger.
        try:
            known_confirmed, hidden_confirmed = roster.confirmed(confirmed_ids)
        except MeasurementIntegrityError:
            raise
        except Exception as exc:
            raise EpisodeRosterError(
                "the confirmed targets could not be reconciled against the t=0 roster "
                "(%s: %s)" % (type(exc).__name__, exc)
            ) from exc

        try:
            # EXPLICIT RewardConfig: `graph_reward`'s own default is c = 0.0, so without
            # this the death penalty FD-BASELINE-v1 depends on would silently be off.
            # The formula is unchanged -- only the coefficient it already accepted, and
            # it still reads the SAME allocated-only `ctx.oracle_tasks` /
            # `ctx.oracle_solution` it always has. Computed only once the measurement is
            # known to be sound.
            ep_reward = compute_episode_reward(ctx, result, cfg.reward_config())
        except ReferenceIntegrityError as exc:
            # The same two-way routing as the run stage above, and the same reason: an
            # instrument contradiction aborts, an unanswered solve is accounted.
            if reference_fault_aborts(exc):
                raise
            raise EpisodeAttemptError("reward", exc) from exc
        except MatchAouBackendError:
            # `plan_value` refuses to score a plan under an objective that does not
            # describe it (a multi-step task or p != 1 reaching a P1-selected run). Same
            # routing as the setup and run stages: a configuration fault, never attrition.
            raise
        except Exception as exc:
            raise EpisodeAttemptError("reward", exc) from exc

        # COMMAND HISTORY, from the controller's read of what `run_episode` actually
        # emitted -- NOT `executor.rtb_issued`. That field is a lifecycle latch which the
        # executor also sets True for a DEAD ego, precisely because no command was (or
        # could be) emitted; reading it would report an ego that flew its plan into the
        # ground as both an RTB and a death.
        fd_outcome = fuel_damage.outcome
        selected_ego_rtb = fd_outcome.rtb_command_issued

        # The bundle is COMPLETE only now: its playback was synchronized above, its world
        # validated, its reward computed. The observed target counts are the roster's --
        # the same numbers the OK block prints -- and `finalize` refuses to certify a
        # bundle whose observed cell contradicts the scheduled one.
        if artifacts is not None:
            artifacts.finalize(
                expected={
                    # The RESOLVED cell -- which on the generalized path already accounts
                    # for a legitimately short hidden realization, so a bundle is never
                    # left `incomplete` for a shortfall the design permits.
                    "n_known": int(scheduled.n_known),
                    "n_hidden": int(scheduled.n_hidden),
                    "n_targets_executed": int(scheduled.n_targets_executed),
                },
                observed={
                    "n_known": len(roster.known_names),
                    "n_hidden": len(roster.hidden_names),
                    "n_targets_executed": int(roster.total),
                    "targets_confirmed_unique": int(targets_confirmed_unique),
                },
            )

        # The GENERALIZED-V1 per-episode diagnostics, read from the structures Tasks 1-3
        # already produce and carried WHOLE rather than flattened -- a subset copied here
        # would be a second, drifting description of the same event. Every one is `None`
        # when the design that produces it did not run.
        construction_audit = getattr(ctx, "construction_audit", None)
        # Duck-typed for the same reason the context fields above are: this module
        # consumes the fuel-damage controller through its published surface, and an
        # absent post-FD record simply means the wake policy that produces one did not
        # run. `None` is the truthful reading of that, not a missing measurement.
        post_fd = getattr(fuel_damage, "post_fd_outcome", None)
        reference = getattr(result, "reference", None)
        fd_plan_record = fuel_damage.plan.to_record()
        world_identity = _observe_world_identity(
            ctx, roster=roster, fd_plan_record=fd_plan_record
        )

        return _EpisodeOutcome(
            trajectory=list(result.trajectory),
            reward=float(ep_reward.reward),
            ticks=int(result.ticks),
            ended=str(result.ended),
            n_wakes=int(result.n_wakes),
            confirmed_kills=int(result.confirmed_kills),
            n_dead=int(result.n_dead),
            seconds=time.perf_counter() - t0,
            targets_confirmed_unique=targets_confirmed_unique,
            targets_total=roster.total,
            known_target_names=roster.known_names,
            hidden_target_names=roster.hidden_names,
            known_confirmed_names=known_confirmed,
            hidden_confirmed_names=hidden_confirmed,
            fuel_damage_plan=fd_plan_record,
            fuel_damage_outcome=fd_outcome.to_record(),
            selected_ego_rtb_issued=selected_ego_rtb,
            cardinality=cell,
            route_relative_load=(
                None if pre_solve_cardinality is None
                else getattr(ctx, "route_relative_load", None)
            ),
            pre_solve_cardinality=pre_solve_cardinality,
            hidden_realized=int(scheduled.n_hidden),
            construction_audit=(
                None if construction_audit is None else construction_audit.as_dict()
            ),
            post_fd_adaptation=(None if post_fd is None else post_fd.to_record()),
            reference=(None if reference is None else reference.to_record()),
            reward_breakdown=_reward_breakdown_record(ep_reward),
            world_identity=world_identity,
            allocation_fingerprint=allocation_fingerprint,
        )
    finally:
        if ctx is not None:
            try:
                ctx.env.close()
            except Exception:
                pass


# =============================================================================
# 5. Evaluation -- deterministic, no buffer, no update
# =============================================================================

def evaluate(
    policy: Any,
    gen: ScenarioGenerator,
    cfg: TrainConfig,
    *,
    iteration: Optional[int],
    stage: str = _EVAL_STAGE_POST_UPDATE,
    updates_completed: int = 0,
    round_ordinal: int = 0,
    failures_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    artifacts_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run ``cfg.eval_episodes`` deterministic episodes on the FIXED eval seed band.

    Touches NO optimizer, NO buffer and no weights -- ``run_episode`` is inference-only
    under its own ``torch.no_grad``. The same seeds are used on every round, so
    round-to-round differences in the returned mean are attributable to the policy.

    REFUSED under GENERALIZED-V2. ``TrainConfig.validate`` already refuses to build such
    a run with evaluation enabled, so this is a second lock on a door that is already
    bolted -- and it is here because ``evaluate`` is a public entry point: the fixed
    held-out band carries no stratum and its seeds are not drawn from the V2 population, so
    a round taken on it would report an unstratified measurement under a generalized label.

    ``stage`` is ``pre_update`` for the ONE round measured on the initial policy before
    any training episode or optimizer step, and ``post_update`` for every later round;
    ``updates_completed`` is how many PPO updates had actually run when the round
    started (0 for ``pre_update``). Together they are what stops a post-update result
    from being read as "iteration 0" -- and ``iteration`` is ``None`` for the pre-update
    round precisely because no training iteration has happened yet.

    SKIP AND ACCOUNT. Each of the ``cfg.eval_episodes`` scheduled seeds is attempted
    exactly ONCE; a failure is recorded in the ledger and skipped, never retried and
    never replaced by another seed, so the band stays the declared held-out band. The
    returned aggregates therefore describe the exact-cardinality-FEASIBLE SUBSET, which
    is why they are reported next to ``n_attempted`` / ``n_successful`` / ``n_failed``
    and are ``None`` (not ``0.0``) when that subset is empty.

    ``round_ordinal`` names this round's SCENARIO-TAG namespace (:func:`eval_member_tag`)
    and nothing else. The seeds are unchanged -- episode ``e`` is ``eval_seed(cfg, e)`` on
    every round -- so successive rounds re-measure the same held-out worlds; they just
    stop overwriting each other's scenario JSON while doing it. ``pre_update`` is
    ordinal 0 and each later ``post_update`` round takes the next.

    MATCHED GROUPS. Every held-out seed is attempted once per group MEMBER, and the
    members share EVERYTHING except the event: the same ``eval_seed``, hence the same
    generator world, the same solved ``A_init``, the same hidden-placement geometry (the
    placement rng is derived from the episode seed, not from the mode) and -- for the
    damaged members -- the same selected ego (the severity draw lives in its own rng
    domain, so it cannot move the ego selection). Their reward differences are therefore
    attributable to the fuel-damage event rather than to scenario variance, which an
    unmatched comparison across different seeds could never claim. Only the artifact TAGS
    differ (:func:`eval_member_tag`), so the worlds coexist as files.

      * FD-BASELINE-v1 -> a PAIR: ``forced_clean`` and ``forced_damaged``.
      * FD-VARIABLE-SEVERITY-v1 -> a TRIAD: ``forced_clean``, ``forced_mild`` and
        ``forced_severe``. The extra member is what makes "did the actor respond
        DIFFERENTLY to a survivable loss than to an unsurvivable one?" a within-seed
        question instead of a between-worlds one.

    The shape comes from ``cfg`` (:attr:`TrainConfig.eval_group_members`), so a legacy
    run keeps its pair and only a ``seeded_variable`` run evaluates triads. Evaluation
    never silently becomes a triad.

    Three denominators are reported and none substitutes for another: ``n_attempted``
    counts EPISODE attempts (group size per seed), the per-cell
    ``n_<cell>_attempted`` keys split that by reporting cell, and
    ``n_groups_successful`` counts the seeds where EVERY member completed. Every
    within-seed delta is averaged over that last population ALONE -- a group with a
    failed member contributes to no delta, is never repaired with its surviving members,
    and is still visible in the attempt counts.

    ``artifacts_root`` is the visual-artifact switch for this round: ``None`` (the
    default) captures nothing, and a path makes every scheduled member preserve its
    bundle under it. ``train`` passes it only when :attr:`TrainConfig.visual_artifacts` is
    set. The identities carry this round's ordinal and each member's pair slot, so the two
    members of a held-out seed -- and the same seed across two rounds -- can never share a
    bundle.

    Returns a scalar-only record (also written to ``eval_records.jsonl``), plus one
    printed ``OK`` block per successful episode.
    """
    rewards: List[float] = []
    unique_confirmed: List[float] = []
    wakes: List[float] = []
    meta_counts = _empty_meta_counts()
    ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
    if cfg.route_relative_population:
        raise ValueError(
            "evaluate() is not defined for episode_design=%r: the fixed held-out seed-band "
            "evaluator is a fixed-cell construct whose seeds carry no stratum and are not "
            "drawn from this design's population. Evaluate this design through "
            "evaluate_benchmark() with its frozen generalized_v2 benchmark manifest and a "
            "declared benchmark_profile." % (cfg.episode_design,)
        )
    members = cfg.eval_group_members
    group_size = cfg.eval_group_size
    tally = _ConditionTally(cfg.reported_cells)
    # One list per declared within-seed difference, each filled ONLY from complete
    # groups. Pre-seeded with every declared key so a delta this run reports is present
    # (as `None`) even when no group completed -- a missing key and a missing measurement
    # are different things to a reader.
    group_deltas: Dict[Tuple[str, str], List[float]] = {
        pair: [] for pair in cfg.eval_group_deltas
    }
    n_failed = 0
    n_groups = int(cfg.eval_episodes)
    n_groups_successful = 0
    n_attempted = n_groups * group_size
    t0 = time.perf_counter()

    for e in range(n_groups):
        seed = eval_seed(cfg, e)
        # Every member of this group, keyed by CELL; a member that failed is simply
        # absent, which is what makes the "all succeeded" test below a membership test
        # rather than a sentinel comparison.
        member_rewards: Dict[str, float] = {}

        for member, (cell, mode) in enumerate(members):
            tag = eval_member_tag(round_ordinal=round_ordinal, e=e, member=member,
                                  group_size=group_size)
            condition = cell_condition(cell)
            tally.attempt(cell)
            artifacts = None
            if artifacts_root is not None:
                artifacts = _AttemptArtifacts(
                    root=artifacts_root,
                    identity=_AttemptIdentity(
                        phase=str(stage),
                        iteration=iteration,
                        updates_completed=int(updates_completed),
                        eval_round_ordinal=int(round_ordinal),
                        eval_episode_index=int(e),
                        eval_pair_member=int(member),
                        attempt_ordinal=e * group_size + member,
                        episode_index=None,
                        seed=int(seed),
                        condition=str(condition),
                        severity=(str(cell) if cell in SEVERITIES else None),
                        episode_tag=int(tag),
                    ),
                )
            try:
                out = _run_one_episode(
                    policy, gen, cfg,
                    seed=seed,
                    episode_tag=tag,
                    deterministic=True,
                    fuel_damage_mode=mode,
                    **_artifact_kwargs(artifacts),
                )
            except (_VisualArtifactError, MeasurementIntegrityError,
                    FuelDamageIntegrityError, ReferenceIntegrityError,
                    MatchAouBackendError):
                # INFRASTRUCTURE / DATA INTEGRITY, not science: none names a pipeline
                # stage, none may enter the ledger or a condition tally, and none may be
                # skipped. A `ReferenceIntegrityError` reaching here has ALREADY been
                # classified by `_run_one_episode`: the attrition case (an unanswered
                # solve) was wrapped as an `EpisodeAttemptError` there, so only the
                # instrument-contradiction case can arrive unwrapped. Re-raised ahead of the broad handler so the run stops loudly
                # instead of recording a scientific failure that never happened.
                # `FuelDamageIntegrityError` joins them under GENERALIZED-V1: a world
                # CERTIFIED FD-capable that then contradicts its own certificate is an
                # instrument fault, never attrition (handoff 3l.3).
                raise
            except Exception as exc:  # an eval failure must not abort training either
                n_failed += 1
                tally.failure(cell)
                _append_failure_record(failures_path, _failure_record(
                    phase="eval",
                    evaluation_stage=stage,
                    updates_completed=updates_completed,
                    iteration=iteration,
                    attempt_ordinal=e * group_size + member,
                    episode_index=None,
                    eval_tag="eval_e%d_%s_tag%d" % (e, cell, tag),
                    seed=seed,
                    # The ledger keeps naming the CONDITION, so `failures_by_condition`
                    # means what it always did; the finer cell is now a field of its own
                    # beside it (it was previously only inside the tag string).
                    condition=condition,
                    cell=cell,
                    exc=exc,
                ))
                print("  [eval %s e%d %s] FAILED (seed=%d): %s: %s"
                      % (stage, e, cell, seed, type(exc).__name__, exc))
                traceback.print_exc()
                continue
            # Printed BEFORE the next attempt starts, so a long eval round is readable
            # while it runs rather than only in the round's summary line.
            print(_format_episode_block(
                "[eval stage=%s ep=%d %s seed=%d]"
                % (_ascii(stage), e, cell, seed), out
            ))
            # `cell` is THIS member's scheduled cell, from the matched-group schedule a
            # few lines above. Passing it is what makes the guard a scheduled-vs-executed
            # comparison rather than a membership test, and it runs BEFORE the member
            # reward is recorded -- so a mismatched member can never enter a matched
            # group, and therefore never enter a within-seed delta.
            member_rewards[tally.success(out, expected_cell=cell)] = out.reward
            _append_episode_outcome_record(outcomes_path, _episode_outcome_record(
                out,
                phase=str(stage),
                iteration=iteration,
                updates_completed=int(updates_completed),
                updates_completed_before=int(updates_completed),
                attempt_ordinal=e * group_size + member,
                episode_index=None,
                eval_round_ordinal=int(round_ordinal),
                eval_episode_index=int(e),
                eval_group_member=int(member),
                seed=int(seed),
                episode_tag=int(tag),
                fuel_damage_mode=str(mode),
                design=cfg.design,
            ))
            rewards.append(out.reward)
            unique_confirmed.append(float(out.targets_confirmed_unique))
            wakes.append(float(out.n_wakes))
            _add_meta_action_counts(meta_counts, out.trajectory)
            if out.ended in ended_counts:
                ended_counts[out.ended] += 1

        # A COMPLETE group only. A partial group is not a matched measurement, and
        # filling the gap with the surviving members would report a within-seed
        # difference that was never measured. The test is over EVERY declared member, so
        # a triad needs all three -- a clean+mild pair inside a failed triad yields no
        # mild-minus-clean delta either, because the group it belonged to is incomplete.
        if all(cell in member_rewards for cell, _mode in members):
            n_groups_successful += 1
            for pair in group_deltas:
                cell, reference = pair
                group_deltas[pair].append(
                    member_rewards[cell] - member_rewards[reference]
                )

    n_successful = len(rewards)
    episodes_with_wakes = sum(1 for w in wakes if w > 0)
    r = _stats_or_none(rewards)
    # ONE arithmetic site behind BOTH the authoritative key and its legacy alias, so the
    # two can never drift apart and the alias can never revert to the (ego,target) count.
    unique_confirmed_mean = _stats_or_none(unique_confirmed)["mean"]
    # Every declared within-seed difference as a FLAT key, so a plot or a notebook can
    # read one by name without decoding a nested structure. `eval_delta_keys` names them,
    # so a reader does not have to know the design to find them.
    delta_record: Dict[str, Any] = {}
    for (cell, reference), values in group_deltas.items():
        stats = _stats_or_none(values)
        key = _delta_key(cell, reference)
        delta_record[key] = stats["mean"]
        delta_record["%s_min" % key] = stats["min"]
        delta_record["%s_max" % key] = stats["max"]
    legacy_delta_key = _delta_key(CONDITION_DAMAGED, CONDITION_CLEAN)
    return {
        "evaluation_stage": str(stage),
        "updates_completed": int(updates_completed),
        "iteration": None if iteration is None else int(iteration),
        # Which scenario-tag namespace this round's worlds were written under -- the
        # link from a record back to the `episode_<tag>_scenario.json` files it ran on.
        "eval_round_ordinal": int(round_ordinal),
        "episode_tag_start": eval_member_tag(round_ordinal=round_ordinal, e=0, member=0,
                                             group_size=group_size),
        # --- attempt accounting: the AUTHORITATIVE names ---
        # `n_attempted` counts EPISODE attempts, which is `n_groups * group_size` since
        # every held-out seed is run once per matched-group member.
        "n_attempted": n_attempted,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "success_fraction": _fraction(n_successful, n_attempted),
        "episodes_with_wakes": int(episodes_with_wakes),
        "wake_fraction_of_successful": _fraction(episodes_with_wakes, n_successful),
        # --- MATCHED-GROUP accounting, with its OWN denominator ---
        # The AUTHORITATIVE generic names. `eval_group_kind` / `eval_group_size` say
        # which design produced them, so "how many complete groups" never has to be
        # inferred from a member count.
        "eval_group_kind": str(cfg.eval_group_kind),
        "eval_group_size": int(group_size),
        "eval_group_cells": list(cfg.reported_cells),
        "n_groups_attempted": n_groups,
        "n_groups_successful": int(n_groups_successful),
        "group_success_fraction": _fraction(n_groups_successful, n_groups),
        # Every within-seed difference this design declares, over COMPLETE groups only.
        # None (never 0.0) when no group completed: 0.0 would say "the event changed
        # nothing", which is a measurement, not an absence of one.
        "eval_delta_keys": [_delta_key(c, r_) for c, r_ in cfg.eval_group_deltas],
        "eval_delta_over": "groups_with_all_members_successful",
        **delta_record,
        # --- LEGACY ALIASES, kept so every existing reader still resolves ---
        # `n_pairs_*` are the same quantity as `n_groups_*` (complete matched groups);
        # for a legacy run they are literally pairs, and `eval_group_kind` says when they
        # are not. `eval_paired_reward_delta` is the damaged-minus-clean difference and
        # is therefore `None` under a TRIAD, where there is no single damaged member to
        # difference against -- the three named deltas above carry that round's result.
        "n_pairs_attempted": n_groups,
        "n_pairs_successful": int(n_groups_successful),
        "pair_success_fraction": _fraction(n_groups_successful, n_groups),
        "eval_paired_reward_delta": delta_record.get(legacy_delta_key),
        "eval_paired_reward_delta_min": delta_record.get("%s_min" % legacy_delta_key),
        "eval_paired_reward_delta_max": delta_record.get("%s_max" % legacy_delta_key),
        "paired_delta_over": "pairs_with_both_members_successful",
        # --- aggregates over the SUCCESSFUL subset only (None when it is empty) ---
        # `eval_reward_mean` spans EVERY cell; the per-cell means below are the ones to
        # read when the question is about the difficulty factor.
        "eval_reward_mean": r["mean"],
        "eval_reward_min": r["min"],
        "eval_reward_max": r["max"],
        # AUTHORITATIVE: mean number of distinct TARGETS confirmed killed per successful
        # episode. Bounded by the world's target count by construction.
        "eval_targets_confirmed_unique_mean": unique_confirmed_mean,
        "target_confirmation_count_semantics": _TARGET_CONFIRMATION_SEMANTICS,
        "eval_wakes_mean": _stats_or_none(wakes)["mean"],
        "aggregates_over": "successful_episodes",
        "meta_action_counts": dict(meta_counts),
        # Per-cell attempt counts, per-cell reward means, the per-cell FD-wake
        # meta-action mix, and the FD event counters (applied / wakes / RTBs / deaths) --
        # all through the ONE tally site.
        **tally.to_record(prefix="eval_"),
        "meta_action_fractions": _meta_fractions(meta_counts),
        "ended_counts": dict(ended_counts),
        "seed_band": {
            # SEEDS, not attempts: the band is `eval_episodes` wide however many times
            # each of its seeds is run. Multiplying the attempts (a matched pair or
            # triad) must not look like a widened held-out band.
            "start": int(cfg.eval_base_seed),
            "stop": int(cfg.eval_base_seed) + n_groups,
            "half_open": True,
        },
        # --- compatibility names kept so pre-B4 readers still parse a record ---
        "n_episodes": n_attempted,
        "n_ok": n_successful,
        # ALIAS of `eval_targets_confirmed_unique_mean`, not a second measurement. It
        # used to average `len(executor.done)` -- (ego, target) CONFIRMATIONS, which can
        # exceed the number of targets in the world -- and now carries the corrected
        # unique-target count under its old name.
        "eval_kills_mean": unique_confirmed_mean,
        "eval_seconds": time.perf_counter() - t0,
    }


# =============================================================================
# 5b. GENERALIZED-V1: the FROZEN 18-stratum benchmark evaluation round
# =============================================================================

class _BenchmarkTally:
    """Per-STRATUM accounting for ONE manifest-driven evaluation round.

    A SEPARATE structure from :class:`_ConditionTally` rather than a widened one, because
    it answers a different question with a different denominator. ``_ConditionTally``
    answers "how did the CLEAN / MILD / SEVERE cells do in this round" and is kept beside
    this one, unchanged, so a benchmark round still reports every number a legacy round
    does. This answers "how did each of the EIGHTEEN REQUESTED STRATA do", which a
    cell-level tally cannot: a mean over ``severe`` pools six different team-size / load
    combinations, and pooling is exactly what the stratification exists to avoid.

    EVERY STRATUM IS PRESENT FROM THE START, including one that saw no attempt, so a
    stratum with an empty denominator reports an explicit 0 / ``None`` rather than
    vanishing from the record -- an absent key and a measured zero must never look alike.

    REQUESTED-VS-REALIZED IS TRACKED HERE, NOT INFERRED. Each successful attempt records
    the hidden count its world REALLY built beside the one its stratum ASKED for, so the
    distribution a human must inspect before any measurement (handoff 3l.6) is readable
    straight off the round. NOTHING here decides whether a HIGH stratum has "degenerated"
    -- this module deliberately invents no threshold for that; it reports the distribution
    and the research review decides.
    """

    def __init__(self) -> None:
        self.strata: Tuple[str, ...] = BENCHMARK_STRATUM_KEYS
        self.attempted: Dict[str, int] = {k: 0 for k in self.strata}
        self.failed: Dict[str, int] = {k: 0 for k in self.strata}
        self.rewards: Dict[str, List[float]] = {k: [] for k in self.strata}
        self.fd_wakes: Dict[str, int] = {k: 0 for k in self.strata}
        self.fd_fired: Dict[str, int] = {k: 0 for k in self.strata}
        self.fd_meta: Dict[str, Dict[str, int]] = {
            k: _empty_meta_counts() for k in self.strata
        }
        self.deaths: Dict[str, int] = {k: 0 for k in self.strata}
        self.rtb: Dict[str, int] = {k: 0 for k in self.strata}
        self.targets: Dict[str, List[float]] = {k: [] for k in self.strata}
        # REQUESTED vs REALIZED hidden cardinality, per stratum. `realized` is a
        # histogram keyed by the realized count, so a HIGH stratum that keeps realizing 1
        # is visible as a distribution rather than only as a mean.
        self.hidden_requested: Dict[str, int] = {k: 0 for k in self.strata}
        self.hidden_realized: Dict[str, List[int]] = {k: [] for k in self.strata}
        self.short_realized: Dict[str, int] = {k: 0 for k in self.strata}
        # Complete matched GROUPS and their within-group deltas, per BASE CELL and
        # overall. A group contributes to a delta only when ALL THREE members completed.
        self.groups_attempted: Dict[str, int] = {}
        self.groups_successful: Dict[str, int] = {}
        self.deltas: Dict[Tuple[str, str], List[float]] = {
            pair: [] for pair in BENCHMARK_DELTAS
        }
        self.cell_deltas: Dict[str, Dict[Tuple[str, str], List[float]]] = {}

    # ---- attempts ---------------------------------------------------------------
    def attempt(self, stratum: str) -> None:
        if str(stratum) not in self.attempted:
            raise MeasurementIntegrityError(
                "a scheduled benchmark attempt names stratum %r, which is not one of "
                "the %d requested strata; its denominator would be invisible."
                % (stratum, len(self.strata))
            )
        self.attempted[str(stratum)] += 1

    def failure(self, stratum: str) -> None:
        self.failed[str(stratum)] = self.failed.get(str(stratum), 0) + 1

    def success(self, stratum: str, out: "_EpisodeOutcome", *, requested: int) -> None:
        """Fold one successful benchmark member in, under its own stratum."""
        key = str(stratum)
        if key not in self.rewards:
            raise MeasurementIntegrityError(
                "a successful benchmark episode reports stratum %r, which this round "
                "does not report (%d strata)." % (key, len(self.strata))
            )
        outcome = out.fuel_damage_outcome or {}
        self.rewards[key].append(float(out.reward))
        self.deaths[key] += int(out.n_dead)
        self.targets[key].append(float(out.targets_confirmed_unique))
        if out.selected_ego_rtb_issued:
            self.rtb[key] += 1
        if outcome.get("fired"):
            self.fd_fired[key] += 1
        if outcome.get("wake_occurred"):
            self.fd_wakes[key] += 1
            meta = outcome.get("wake_meta_action")
            if meta is not None:
                self.fd_meta[key][MetaAction(int(meta)).name] += 1
        self.hidden_requested[key] += int(requested)
        realized = int(out.hidden_realized or 0)
        self.hidden_realized[key].append(realized)
        if realized < int(requested):
            self.short_realized[key] += 1

    # ---- matched groups ---------------------------------------------------------
    def group(
        self, base_cell: str, member_rewards: Dict[str, float], *, complete: bool
    ) -> None:
        """Record one matched world group, and its deltas ONLY when it is complete."""
        self.groups_attempted[base_cell] = self.groups_attempted.get(base_cell, 0) + 1
        cell_deltas = self.cell_deltas.setdefault(
            base_cell, {pair: [] for pair in BENCHMARK_DELTAS}
        )
        if not complete:
            # A partial group is NOT a matched measurement, and filling the gap with its
            # surviving members would report a within-world difference nobody measured.
            # It stays visible in `groups_attempted` and in the per-stratum counts.
            return
        self.groups_successful[base_cell] = self.groups_successful.get(base_cell, 0) + 1
        for pair in BENCHMARK_DELTAS:
            cell, reference = pair
            delta = member_rewards[cell] - member_rewards[reference]
            self.deltas[pair].append(delta)
            cell_deltas[pair].append(delta)

    # ---- the record -------------------------------------------------------------
    def to_record(self) -> Dict[str, Any]:
        """The round's stratified accounting, every metric beside its own denominator."""
        strata: Dict[str, Any] = {}
        for key in self.strata:
            rewards = self.rewards[key]
            realized = self.hidden_realized[key]
            denom = int(self.fd_wakes[key])
            counts = self.fd_meta[key]
            strata[key] = {
                "n_attempted": int(self.attempted[key]),
                "n_successful": len(rewards),
                "n_failed": int(self.failed[key]),
                "success_fraction": _fraction(len(rewards), int(self.attempted[key])),
                "reward_mean": _stats_or_none(rewards)["mean"],
                "reward_min": _stats_or_none(rewards)["min"],
                "reward_max": _stats_or_none(rewards)["max"],
                "targets_confirmed_unique_mean": _stats_or_none(
                    self.targets[key])["mean"],
                "n_deaths": int(self.deaths[key]),
                "n_rtb_command_issued": int(self.rtb[key]),
                "n_fd_fired": int(self.fd_fired[key]),
                "n_fd_wakes": denom,
                "fd_meta_action_counts": dict(counts),
                "fd_meta_action_rates": {
                    name: _fraction(int(counts.get(name, 0)), denom)
                    for name in _META_NAMES
                },
                "fd_rates_over": "fd_wakes",
                # REQUESTED vs REALIZED, the distribution a human inspects before any
                # scientific measurement. `hidden_realized_histogram` is the honest
                # shape; the mean is a convenience beside it, never instead of it.
                "hidden_requested_total": int(self.hidden_requested[key]),
                "hidden_realized_total": int(sum(realized)),
                "hidden_realized_mean": _stats_or_none(
                    [float(r) for r in realized])["mean"],
                "hidden_realized_histogram": {
                    str(v): realized.count(v) for v in sorted(set(realized))
                },
                "n_short_realized": int(self.short_realized[key]),
                "short_realized_fraction": _fraction(
                    int(self.short_realized[key]), len(realized)),
                "aggregates_over": "successful_episodes",
            }
        base_cells: Dict[str, Any] = {}
        for cell_key in sorted(
            set(self.groups_attempted) | set(self.cell_deltas)
        ):
            attempted = int(self.groups_attempted.get(cell_key, 0))
            successful = int(self.groups_successful.get(cell_key, 0))
            entry: Dict[str, Any] = {
                "n_groups_attempted": attempted,
                "n_groups_successful": successful,
                "group_success_fraction": _fraction(successful, attempted),
            }
            for pair, values in self.cell_deltas.get(cell_key, {}).items():
                entry[_delta_key(*pair)] = _stats_or_none(values)["mean"]
            base_cells[cell_key] = entry
        record: Dict[str, Any] = {
            "benchmark_strata": strata,
            "benchmark_stratum_keys": list(self.strata),
            "benchmark_base_cells": base_cells,
            "benchmark_delta_keys": [_delta_key(c, r) for c, r in BENCHMARK_DELTAS],
            "benchmark_delta_over": "world_groups_with_all_members_successful",
        }
        for pair, values in self.deltas.items():
            stats = _stats_or_none(values)
            key = _delta_key(*pair)
            record[key] = stats["mean"]
            record["%s_min" % key] = stats["min"]
            record["%s_max" % key] = stats["max"]
            record["%s_n" % key] = len(values)
        return record


def _benchmark_member_identity(
    manifest: BenchmarkManifest, world: Any, cell: str,
    identity: Optional[WorldIdentity],
) -> Dict[str, Any]:
    """The frozen-benchmark identity keys one member's records carry."""
    return {
        "benchmark_manifest_id": str(manifest.manifest_id),
        "benchmark_stratum": world.stratum_key(cell),
        "benchmark_group_key": world.key,
        "benchmark_agent_count": int(world.agent_count),
        "benchmark_load_bucket": str(world.load_bucket),
        "benchmark_world_ordinal": int(world.world_ordinal),
        "benchmark_world_identity": (
            None if identity is None else identity.to_record()
        ),
    }


def evaluate_benchmark(
    policy: Any,
    gen: ScenarioGenerator,
    cfg: TrainConfig,
    manifest: BenchmarkManifest,
    *,
    iteration: Optional[int],
    stage: str = _EVAL_STAGE_POST_UPDATE,
    updates_completed: int = 0,
    round_ordinal: int = 0,
    failures_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    artifacts_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """ONE deterministic round over the FROZEN benchmark of this run's design.

    DESIGN-AWARE DISPATCH. Under GENERALIZED-V2 the round is delegated, before anything
    else happens, to :func:`_evaluate_v2_benchmark`, which reconstructs each member through
    the two-stage route-relative construction and verifies the frozen V2 identity. The
    body below is the GENERALIZED-V1 18-stratum round and is unchanged.

    THE STRUCTURE IS THE SAME AS :func:`evaluate`'S -- matched groups on one world, one
    attempt per member, skip-and-account on failure, deltas over COMPLETE groups only --
    and the two differences are exactly the two the stratified design needs:

      * the population comes from the FROZEN MANIFEST, not from the held-out seed band.
        A benchmark member's world SHAPE is its stratum's, not a function of its seed, so
        the cardinality is taken from the manifest and never re-derived;
      * accounting is per STRATUM as well as per cell. A mean over ``severe`` pools six
        team-size / load combinations, which is the pooling the stratification exists to
        avoid, so every stratified metric carries its own stratum denominator.

    MATCHED WORLD GROUPS ARE VERIFIED, NOT ASSUMED. The three members share one seed and
    one requested cardinality, and under the certified eligibility policy -- whose walk
    depends on the episode seed ALONE -- they certify the same ego at the same event
    point. That is a claim, so it is CHECKED: each completed member's id-free world
    identity is compared against the others', and (when the manifest was preflighted)
    against the frozen record. A disagreement is a
    :class:`~match_aou.rl.training.graph_generalized.BenchmarkIdentityError` and ABORTS --
    two members that built different worlds would make their delta a between-worlds
    comparison wearing a within-world label.

    NO SUBSTITUTION, EVER. A failed member is recorded once and skipped; its group becomes
    incomplete and contributes to NO delta; no other world, seed or stratum takes its
    place; and the manifest is never regenerated to route around it.

    ``evaluate`` is left completely untouched by this function -- a ``fixed_cell_v1`` run
    still runs exactly the round it always did.
    """
    rewards: List[float] = []
    unique_confirmed: List[float] = []
    wakes: List[float] = []
    meta_counts = _empty_meta_counts()
    ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
    if cfg.route_relative_population:
        return _evaluate_v2_benchmark(
            policy, gen, cfg, manifest,
            iteration=iteration, stage=stage, updates_completed=updates_completed,
            round_ordinal=round_ordinal, failures_path=failures_path,
            outcomes_path=outcomes_path, artifacts_root=artifacts_root,
        )
    if isinstance(manifest, V2BenchmarkManifest):
        raise BenchmarkManifestError(
            "a generalized_v2 benchmark manifest cannot be evaluated under "
            "episode_design=%r" % (cfg.episode_design,))
    tally = _ConditionTally(BENCHMARK_CELLS)
    bench = _BenchmarkTally()
    n_failed = 0
    n_groups_successful = 0
    n_groups = manifest.n_worlds
    n_attempted = manifest.n_members
    t0 = time.perf_counter()

    for w, world in enumerate(manifest.worlds):
        cardinality = world.cardinality()
        member_rewards: Dict[str, float] = {}
        identities: Dict[str, WorldIdentity] = {}

        for member, (cell, mode) in enumerate(world.members()):
            stratum = world.stratum_key(cell)
            tag = eval_member_tag(round_ordinal=round_ordinal, e=w, member=member,
                                  group_size=BENCHMARK_GROUP_SIZE)
            condition = cell_condition(cell)
            tally.attempt(cell)
            bench.attempt(stratum)
            artifacts = None
            if artifacts_root is not None:
                artifacts = _AttemptArtifacts(
                    root=artifacts_root,
                    identity=_AttemptIdentity(
                        phase=str(stage),
                        iteration=iteration,
                        updates_completed=int(updates_completed),
                        eval_round_ordinal=int(round_ordinal),
                        eval_episode_index=int(w),
                        eval_pair_member=int(member),
                        attempt_ordinal=w * BENCHMARK_GROUP_SIZE + member,
                        episode_index=None,
                        seed=int(world.seed),
                        condition=str(condition),
                        severity=(str(cell) if cell in SEVERITIES else None),
                        episode_tag=int(tag),
                    ),
                )
            try:
                out = _run_one_episode(
                    policy, gen, cfg,
                    seed=int(world.seed),
                    episode_tag=tag,
                    deterministic=True,
                    fuel_damage_mode=mode,
                    cardinality=cardinality,
                    **_artifact_kwargs(artifacts),
                )
            except (_VisualArtifactError, MeasurementIntegrityError,
                    FuelDamageIntegrityError, BenchmarkIdentityError,
                    ReferenceIntegrityError, MatchAouBackendError):
                # INFRASTRUCTURE / DATA INTEGRITY, not science: none names a pipeline
                # stage, none may enter the ledger or a tally, and none may be skipped.
                # A `ReferenceIntegrityError` reaching HERE has already been classified
                # by `_run_one_episode` -- the attrition case was wrapped as an
                # `EpisodeAttemptError` there, so only the instrument-contradiction case
                # can arrive unwrapped, and it aborts.
                raise
            except Exception as exc:
                n_failed += 1
                tally.failure(cell)
                bench.failure(stratum)
                _append_failure_record(failures_path, _failure_record(
                    phase="eval",
                    evaluation_stage=stage,
                    updates_completed=updates_completed,
                    iteration=iteration,
                    attempt_ordinal=w * BENCHMARK_GROUP_SIZE + member,
                    episode_index=None,
                    eval_tag="benchmark_%s_%s_tag%d" % (world.key, cell, tag),
                    seed=int(world.seed),
                    condition=condition,
                    cell=cell,
                    cardinality=cardinality,
                    benchmark=_benchmark_member_identity(manifest, world, cell, None),
                    exc=exc,
                ))
                print("  [bench %s %s %s] FAILED (seed=%d): %s: %s"
                      % (stage, world.key, cell, world.seed,
                         type(exc).__name__, exc))
                traceback.print_exc()
                continue

            print(_format_episode_block(
                "[bench stage=%s %s %s seed=%d]"
                % (_ascii(stage), world.key, cell, world.seed), out
            ))
            # THE WORLD THIS MEMBER REALLY BUILT, checked against the frozen manifest
            # BEFORE its reward is allowed anywhere near a stratum or a group.
            identity = out.world_identity
            if identity is not None:
                require_world_matches_manifest(world, identity)
                identities[cell] = identity
            member_rewards[tally.success(out, expected_cell=cell)] = out.reward
            bench.success(stratum, out,
                          requested=int(cardinality.hidden_requested))
            _append_episode_outcome_record(outcomes_path, _episode_outcome_record(
                out,
                phase=str(stage),
                iteration=iteration,
                updates_completed=int(updates_completed),
                updates_completed_before=int(updates_completed),
                attempt_ordinal=w * BENCHMARK_GROUP_SIZE + member,
                episode_index=None,
                eval_round_ordinal=int(round_ordinal),
                eval_episode_index=int(w),
                eval_group_member=int(member),
                seed=int(world.seed),
                episode_tag=int(tag),
                fuel_damage_mode=str(mode),
                design=cfg.design,
                benchmark=_benchmark_member_identity(
                    manifest, world, cell, identity),
            ))
            rewards.append(out.reward)
            unique_confirmed.append(float(out.targets_confirmed_unique))
            wakes.append(float(out.n_wakes))
            _add_meta_action_counts(meta_counts, out.trajectory)
            if out.ended in ended_counts:
                ended_counts[out.ended] += 1

        # THE MATCHED-GROUP CHECK, over the members that completed. Runs BEFORE the
        # deltas, so two members that built different worlds can never be differenced.
        require_matched_group_identity(world, identities)
        complete = all(cell in member_rewards for cell, _mode in world.members())
        bench.group(world.base_cell_key, member_rewards, complete=complete)
        if complete:
            n_groups_successful += 1

    n_successful = len(rewards)
    episodes_with_wakes = sum(1 for x in wakes if x > 0)
    r = _stats_or_none(rewards)
    unique_confirmed_mean = _stats_or_none(unique_confirmed)["mean"]
    return {
        "evaluation_stage": str(stage),
        "updates_completed": int(updates_completed),
        "iteration": None if iteration is None else int(iteration),
        "eval_round_ordinal": int(round_ordinal),
        "episode_tag_start": eval_member_tag(
            round_ordinal=round_ordinal, e=0, member=0,
            group_size=BENCHMARK_GROUP_SIZE),
        # --- WHICH frozen population this round measured -------------------------
        "eval_population": "benchmark_manifest",
        "benchmark_manifest_id": str(manifest.manifest_id),
        "benchmark_label": manifest.label,
        "benchmark_n_worlds": int(manifest.n_worlds),
        "benchmark_n_members": int(manifest.n_members),
        "benchmark_n_strata": len(BENCHMARK_STRATA),
        # --- attempt accounting ---------------------------------------------------
        "n_attempted": n_attempted,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "success_fraction": _fraction(n_successful, n_attempted),
        "episodes_with_wakes": int(episodes_with_wakes),
        "wake_fraction_of_successful": _fraction(episodes_with_wakes, n_successful),
        # --- matched WORLD GROUPS, with their own denominator ---------------------
        "eval_group_kind": _EVAL_GROUP_KIND_TRIAD,
        "eval_group_size": BENCHMARK_GROUP_SIZE,
        "eval_group_cells": list(BENCHMARK_CELLS),
        "n_groups_attempted": n_groups,
        "n_groups_successful": int(n_groups_successful),
        "group_success_fraction": _fraction(n_groups_successful, n_groups),
        "eval_delta_keys": [_delta_key(c, r_) for c, r_ in BENCHMARK_DELTAS],
        "eval_delta_over": "world_groups_with_all_members_successful",
        # LEGACY ALIASES, so every existing reader of an eval record still resolves.
        "n_pairs_attempted": n_groups,
        "n_pairs_successful": int(n_groups_successful),
        "pair_success_fraction": _fraction(n_groups_successful, n_groups),
        "eval_paired_reward_delta": None,
        "paired_delta_over": "world_groups_with_all_members_successful",
        # --- aggregates over the SUCCESSFUL subset only ---------------------------
        "eval_reward_mean": r["mean"],
        "eval_reward_min": r["min"],
        "eval_reward_max": r["max"],
        "eval_targets_confirmed_unique_mean": unique_confirmed_mean,
        "target_confirmation_count_semantics": _TARGET_CONFIRMATION_SEMANTICS,
        "eval_wakes_mean": _stats_or_none(wakes)["mean"],
        "aggregates_over": "successful_episodes",
        "meta_action_counts": dict(meta_counts),
        "meta_action_fractions": _meta_fractions(meta_counts),
        "ended_counts": dict(ended_counts),
        # The per-CELL numbers a legacy round also reports, so the two record shapes stay
        # readable side by side -- and the per-STRATUM ones beside them.
        **tally.to_record(prefix="eval_"),
        **bench.to_record(),
        "n_episodes": n_attempted,
        "n_ok": n_successful,
        "eval_kills_mean": unique_confirmed_mean,
        "eval_seconds": time.perf_counter() - t0,
    }


# =============================================================================
# 5c. GENERALIZED-V2: the frozen ten-cell benchmark evaluation round
# =============================================================================

class _V2BenchmarkTally:
    """Per-BASE-CELL member accounting for one V2 round. REPORTING, and SECONDARY.

    The V2 strata are the ten exogenous ``(A, K-A)`` base cells; this carries each cell's
    member attempts / successes / failures and its within-world REWARD deltas over
    COMPLETE groups. Reward is downstream of the primary behavioural measurement
    (:func:`_v2_behaviour_summary`) and is never a substitute for it.
    """

    def __init__(self) -> None:
        cells = V2_BENCHMARK_BASE_CELL_KEYS
        self.attempted = {bc: {c: 0 for c in BENCHMARK_CELLS} for bc in cells}
        self.failed = {bc: {c: 0 for c in BENCHMARK_CELLS} for bc in cells}
        self.rewards: Dict[str, Dict[str, List[float]]] = {
            bc: {c: [] for c in BENCHMARK_CELLS} for bc in cells}
        self.groups_attempted = {bc: 0 for bc in cells}
        self.groups_complete = {bc: 0 for bc in cells}
        self.deltas: Dict[Tuple[str, str], List[float]] = {p: [] for p in BENCHMARK_DELTAS}
        self.cell_deltas = {bc: {p: [] for p in BENCHMARK_DELTAS} for bc in cells}

    def _require(self, base_cell: str) -> str:
        if str(base_cell) not in self.attempted:
            raise MeasurementIntegrityError(
                "a generalized_v2 benchmark member names base cell %r, which is not one of "
                "the %d base cells" % (base_cell, len(self.attempted)))
        return str(base_cell)

    def attempt(self, base_cell: str, cell: str) -> None:
        self.attempted[self._require(base_cell)][str(cell)] += 1

    def failure(self, base_cell: str, cell: str) -> None:
        self.failed[self._require(base_cell)][str(cell)] += 1

    def success(self, base_cell: str, cell: str, reward: float) -> None:
        self.rewards[self._require(base_cell)][str(cell)].append(float(reward))

    def group(self, base_cell: str, member_rewards: Dict[str, float], *,
              complete: bool) -> None:
        key = self._require(base_cell)
        self.groups_attempted[key] += 1
        if not complete:
            return
        self.groups_complete[key] += 1
        for pair in BENCHMARK_DELTAS:
            delta = member_rewards[pair[0]] - member_rewards[pair[1]]
            self.deltas[pair].append(delta)
            self.cell_deltas[key][pair].append(delta)

    def to_record(self) -> Dict[str, Any]:
        cells: Dict[str, Any] = {}
        for bc in V2_BENCHMARK_BASE_CELL_KEYS:
            cells[bc] = {
                "n_groups_attempted": int(self.groups_attempted[bc]),
                "n_groups_complete": int(self.groups_complete[bc]),
                "members": {
                    c: {
                        "n_attempted": int(self.attempted[bc][c]),
                        "n_successful": len(self.rewards[bc][c]),
                        "n_failed": int(self.failed[bc][c]),
                        "reward_mean": _stats_or_none(self.rewards[bc][c])["mean"],
                    } for c in BENCHMARK_CELLS
                },
                "reward_deltas": {
                    _delta_key(*p): _stats_or_none(v)["mean"]
                    for p, v in self.cell_deltas[bc].items()
                },
                "reward_delta_n": len(self.cell_deltas[bc][BENCHMARK_DELTAS[0]]),
                "reward_deltas_over": "complete_matched_world_groups",
            }
        record: Dict[str, Any] = {"v2_benchmark_base_cells": cells}
        for pair, values in self.deltas.items():
            stats = _stats_or_none(values)
            key = _delta_key(*pair)
            record[key] = stats["mean"]
            record["%s_min" % key] = stats["min"]
            record["%s_max" % key] = stats["max"]
            record["%s_n" % key] = len(values)
        return record


# The two DISJOINT selected-action switch definitions of the V2 behavioural summary.
V2_SWITCH_DIRECTIONAL: str = "mild_not_abort_and_severe_abort"
V2_SWITCH_REVERSE: str = "mild_abort_and_severe_not_abort"


def _v2_immediate_fd_member(
    decisions: Optional[Sequence[Mapping[str, Any]]], cell: str
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """(P(ABORT), selected meta-action name, not-measurable reason) of ONE member.

    ONLY immediate-fuel-damage wakes are read -- ordinary and post-FD-boundary wakes are
    filtered out by their tagged kind, never by the selected action. Exactly one such
    wake is required; zero or several is a stated, non-measurable reason, never a guess.

    ``P(ABORT)`` is read under the wake's OWN representation
    (:func:`_wake_meta_probability`): the one semantic ABORT leaf, or -- for a historical
    record -- the aggregate mass over the abort aliases. The concept is unchanged.
    """
    fd = [d for d in (decisions or ())
          if str(d.get("wake_kind") or "") == WAKE_KIND_IMMEDIATE_FD]
    if not fd:
        return None, None, "%s_no_immediate_fd_wake" % cell
    if len(fd) > 1:
        return None, None, "%s_multiple_immediate_fd_wakes" % cell
    p_abort = _wake_meta_probability(fd[0], _ABORT_NAME)
    selected = fd[0].get("selected_meta_action_name")
    if p_abort is None or selected is None:
        return None, None, "%s_immediate_fd_diagnostics_unrecorded" % cell
    return p_abort, str(selected), None


def _v2_fd_representation(decisions: Optional[Sequence[Mapping[str, Any]]]) -> Optional[str]:
    """The representation of a member's single immediate-FD wake, or ``None``."""
    fd = [d for d in (decisions or ())
          if str(d.get("wake_kind") or "") == WAKE_KIND_IMMEDIATE_FD]
    return _wake_action_representation(fd[0]) if len(fd) == 1 else None


def _v2_behaviour_summary(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """THE PRIMARY V2 behavioural summary, over ONE evaluation round. PURE.

    ``groups`` is one entry per SCHEDULED matched world group of the round:
    ``{"group_key", "base_cell", "complete", "member_decisions": {cell: [wake records]}}``,
    where the wake records are the per-wake diagnostics of THAT round's evaluation members
    (training rows never reach this function).

    For each COMPLETE group whose MILD and SEVERE members each have exactly one
    immediate-FD wake, the metric is PAIRED FIRST, per world group:

        delta = P(ABORT | SEVERE) - P(ABORT | MILD)

    and only then aggregated -- per base cell, as an EQUAL-WEIGHT macro mean over the ten
    base cells, and pooled. ``P(ABORT)`` is the SEMANTIC abort probability: under the
    semantic representation it is directly the one ABORT leaf; a historical record states
    it as the aggregate mass over the node-indexed abort aliases. A pair whose two
    members are stated in DIFFERENT representations is not measurable. The selected
    meta-action directional switch (MILD != ABORT and SEVERE == ABORT) and its reverse
    are counted over the same metric-eligible groups, with their denominators.

    An incomplete group contributes to no delta and no rate and is listed; a complete group
    that is not measurable is listed with its reason. Every undefined quantity is
    ``None``, never ``0``. The macro mean is ``None`` unless EVERY base cell has at least
    one metric-eligible group -- a mean over fewer cells would silently re-weight the
    design.
    """
    per_cell: Dict[str, Dict[str, Any]] = {
        bc: {"n_groups_attempted": 0, "n_groups_complete": 0, "deltas": [],
             "n_directional": 0, "n_reverse": 0}
        for bc in V2_BENCHMARK_BASE_CELL_KEYS
    }
    rows: List[Dict[str, Any]] = []
    reasons: List[str] = []
    reps_eligible: set = set()
    for g in groups:
        bc = str(g["base_cell"])
        slot = per_cell.setdefault(bc, {"n_groups_attempted": 0, "n_groups_complete": 0,
                                        "deltas": [], "n_directional": 0,
                                        "n_reverse": 0})
        slot["n_groups_attempted"] += 1
        row: Dict[str, Any] = {
            "group_key": str(g["group_key"]), "base_cell": bc,
            "complete": bool(g["complete"]), "metric_eligible": False,
            "not_measurable_reason": None,
            "p_abort_mild": None, "p_abort_severe": None,
            "severe_minus_mild_abort_mass": None,
            "mild_selected_meta_action": None, "severe_selected_meta_action": None,
            V2_SWITCH_DIRECTIONAL: None, V2_SWITCH_REVERSE: None,
        }
        if not g["complete"]:
            row["not_measurable_reason"] = "incomplete_group"
            rows.append(row)
            continue
        slot["n_groups_complete"] += 1
        decisions = g.get("member_decisions") or {}
        mild, mild_sel, mild_why = _v2_immediate_fd_member(
            decisions.get(SEVERITY_MILD), SEVERITY_MILD)
        severe, severe_sel, severe_why = _v2_immediate_fd_member(
            decisions.get(SEVERITY_SEVERE), SEVERITY_SEVERE)
        why = mild_why or severe_why
        pair_reps = {_v2_fd_representation(decisions.get(SEVERITY_MILD)),
                     _v2_fd_representation(decisions.get(SEVERITY_SEVERE))}
        if why is None and len(pair_reps) != 1:
            why = "mixed_action_representations"
        if why is not None:
            row["not_measurable_reason"] = why
            reasons.append(why)
            rows.append(row)
            continue
        reps_eligible.update(pair_reps)
        delta = float(severe) - float(mild)           # type: ignore[arg-type]
        directional = mild_sel != _ABORT_NAME and severe_sel == _ABORT_NAME
        reverse = mild_sel == _ABORT_NAME and severe_sel != _ABORT_NAME
        row.update({
            "metric_eligible": True, "p_abort_mild": mild, "p_abort_severe": severe,
            "severe_minus_mild_abort_mass": delta,
            "mild_selected_meta_action": mild_sel,
            "severe_selected_meta_action": severe_sel,
            V2_SWITCH_DIRECTIONAL: bool(directional), V2_SWITCH_REVERSE: bool(reverse),
        })
        slot["deltas"].append(delta)
        slot["n_directional"] += int(directional)
        slot["n_reverse"] += int(reverse)
        rows.append(row)

    by_cell: Dict[str, Any] = {}
    cell_means: Dict[str, Optional[float]] = {}
    all_deltas: List[float] = []
    n_dir = n_rev = 0
    for bc, slot in per_cell.items():
        n_eligible = len(slot["deltas"])
        mean = _stats_or_none(slot["deltas"])["mean"]
        cell_means[bc] = mean
        all_deltas.extend(slot["deltas"])
        n_dir += slot["n_directional"]
        n_rev += slot["n_reverse"]
        by_cell[bc] = {
            "n_groups_attempted": int(slot["n_groups_attempted"]),
            "n_groups_complete": int(slot["n_groups_complete"]),
            "n_groups_metric_eligible": n_eligible,
            "severe_minus_mild_abort_mass_mean": mean,
            "directional_switch_count": int(slot["n_directional"]),
            "directional_switch_rate": _fraction(slot["n_directional"], n_eligible),
            "reverse_switch_count": int(slot["n_reverse"]),
            "reverse_switch_rate": _fraction(slot["n_reverse"], n_eligible),
            "rates_over": "metric_eligible_groups",
        }
    undefined = [bc for bc in V2_BENCHMARK_BASE_CELL_KEYS if cell_means.get(bc) is None]
    defined = [cell_means[bc] for bc in V2_BENCHMARK_BASE_CELL_KEYS
               if cell_means.get(bc) is not None]
    n_complete = sum(1 for r in rows if r["complete"])
    n_eligible_total = len(all_deltas)
    reps_sorted = sorted(str(r) for r in reps_eligible)
    return {
        "metric": "severe_minus_mild_aggregate_abort_mass",
        "wake_kind": WAKE_KIND_IMMEDIATE_FD,
        "abort_meta_action": _ABORT_NAME,
        # WHICH action semantics the eligible pairs' P(ABORT) is stated in. The metric
        # concept -- SEVERE minus MILD semantic abort probability -- is the same in both;
        # only the historical representation had to express it as alias mass.
        "action_representation_ids_observed": reps_sorted,
        "abort_probability_definition": {
            ACTION_REPRESENTATION_ID: "the one semantic SELF_PRESERVATION_ABORT leaf",
            LEGACY_ACTION_REPRESENTATION_LABEL:
                "aggregate mass over the node-indexed abort cells",
        },
        # True only for the historical representation, where the aggregate mass was not
        # the probability of any single selectable action; False under the semantic one,
        # where it IS the ABORT action's probability; None when no eligible pair exists.
        "aggregate_mass_is_not_selected_action_probability": (
            True if reps_sorted == [LEGACY_ACTION_REPRESENTATION_LABEL]
            else False if reps_sorted == [ACTION_REPRESENTATION_ID]
            else None),
        "pairing": "per_complete_matched_world_group_then_aggregated",
        "eligibility": ("complete group whose MILD and SEVERE members each have exactly "
                        "one immediate-fuel-damage wake with recorded diagnostics"),
        "n_groups_attempted": len(rows),
        "n_groups_complete": n_complete,
        "n_groups_incomplete": len(rows) - n_complete,
        "n_groups_metric_eligible": n_eligible_total,
        "n_groups_complete_not_measurable": n_complete - n_eligible_total,
        "not_measurable_reasons": _tally_slugs(reasons),
        "by_base_cell": by_cell,
        "macro_mean_over_base_cells": (
            float(sum(defined)) / len(defined)
            if defined and not undefined else None),
        "macro_n_base_cells_required": len(V2_BENCHMARK_BASE_CELL_KEYS),
        "macro_n_base_cells_defined": len(defined),
        "macro_undefined_base_cells": undefined,
        "pooled_mean_over_groups": _stats_or_none(all_deltas)["mean"],
        "pooled_n_groups": n_eligible_total,
        "directional_switch_definition": V2_SWITCH_DIRECTIONAL,
        "directional_switch_count": n_dir,
        "directional_switch_rate": _fraction(n_dir, n_eligible_total),
        "reverse_switch_definition": V2_SWITCH_REVERSE,
        "reverse_switch_count": n_rev,
        "reverse_switch_rate": _fraction(n_rev, n_eligible_total),
        "switch_rates_over": "metric_eligible_groups",
        "groups": rows,
    }


def _v2_group_population(
    manifest: V2BenchmarkManifest, profile: str,
    groups: Sequence[Mapping[str, Any]], behaviour: Mapping[str, Any],
) -> Dict[str, Any]:
    """The CANONICAL group population one V2 round measured, with digests.

    Two runs quoting the same ``manifest_id`` are comparable only if they measured the
    same profile AND the same complete / incomplete / metric-eligible groups; the ordered
    key lists and their digests make that a checkable claim. The frozen-population
    histograms (``R``, ``H_requested``, ``H_realized``) are REPORTED descriptors of the
    evaluated worlds -- never strata and never quotas.
    """
    def _keys(pred) -> List[str]:
        return [str(g["group_key"]) for g in groups if pred(g)]

    eligible = {str(r["group_key"]) for r in behaviour.get("groups") or ()
                if r.get("metric_eligible")}
    evaluated = _keys(lambda g: True)
    complete = _keys(lambda g: g["complete"])
    incomplete = _keys(lambda g: not g["complete"])
    eligible_keys = [k for k in evaluated if k in eligible]
    frozen = [w.preflight.identity for w in manifest.profile_worlds(profile)]
    return {
        "benchmark_profile_identity": manifest.profile_identity_record(profile),
        "evaluated_group_keys": evaluated,
        "evaluated_group_keys_sha256": canonical_digest({"group_keys": evaluated}),
        "complete_group_keys": complete,
        "complete_group_keys_sha256": canonical_digest({"group_keys": complete}),
        "incomplete_group_keys": incomplete,
        "incomplete_group_keys_sha256": canonical_digest({"group_keys": incomplete}),
        "metric_eligible_group_keys": eligible_keys,
        "metric_eligible_group_keys_sha256": canonical_digest(
            {"group_keys": eligible_keys}),
        "frozen_population": {
            "route_count_histogram": _histogram([i.route_count for i in frozen]),
            "hidden_requested_histogram": _histogram(
                [i.hidden_requested for i in frozen]),
            "hidden_requested_given_route_count": _tally_slugs(
                ["R%d:H%d" % (i.route_count, i.hidden_requested) for i in frozen]),
            "hidden_realized_histogram": _histogram([i.hidden_realized for i in frozen]),
            "n_hidden_short_realized": sum(
                1 for i in frozen if i.hidden_realized < i.hidden_requested),
            "descriptors_not_strata": True,
        },
    }


def _v2_benchmark_member_identity(
    manifest: V2BenchmarkManifest, world: V2BenchmarkWorld, cell: str, profile: str,
    identity: Optional[V2WorldIdentity],
) -> Dict[str, Any]:
    """The frozen-benchmark identity keys one V2 member's records carry.

    The shared benchmark keys keep their V1 names so every reader resolves them; under V2
    the stratum IS the base cell and there is no load bucket (``null``). The V2-only
    facts travel in a nested ``benchmark_v2`` block that V1 records never carry.
    """
    frozen = world.preflight.identity
    return {
        "benchmark_manifest_id": str(manifest.manifest_id),
        "benchmark_stratum": world.base_cell_key,
        "benchmark_group_key": world.key,
        "benchmark_agent_count": int(world.agent_count),
        "benchmark_load_bucket": None,
        "benchmark_world_ordinal": int(world.world_ordinal),
        "benchmark_world_identity": None if identity is None else identity.to_record(),
        "benchmark_v2": {
            "design": EPISODE_DESIGN_GENERALIZED_V2,
            "profile": str(profile),
            "base_cell": world.base_cell_key,
            "agent_count": int(world.agent_count),
            "known_count": int(world.known_count),
            "known_offset": int(world.known_offset),
            "member_cell": str(cell),
            "frozen_route_count": int(frozen.route_count),
            "frozen_hidden_requested": int(frozen.hidden_requested),
            "frozen_hidden_realized": int(frozen.hidden_realized),
            "frozen_identity": frozen.to_record(),
            "reconstructed_identity_verified": identity is not None,
        },
    }


def _evaluate_v2_benchmark(
    policy: Any,
    gen: ScenarioGenerator,
    cfg: TrainConfig,
    manifest: Any,
    *,
    iteration: Optional[int],
    stage: str,
    updates_completed: int,
    round_ordinal: int,
    failures_path: Optional[Path],
    outcomes_path: Optional[Path],
    artifacts_root: Optional[Path],
) -> Dict[str, Any]:
    """ONE deterministic round over the FROZEN GENERALIZED-V2 benchmark, one profile.

    Each scheduled world of the declared profile is evaluated as a matched CLEAN / MILD /
    SEVERE triad on its identical frozen seed, with disjoint artifact tags. Every member
    is rebuilt through the PRODUCTION two-stage construction: ``(A, K)`` from the base
    cell, then the known-only P1 solve, the ACTUAL route count, the ACTUAL route-relative
    ``H`` and bounded backoff. Its UUID-free V2 identity is then VERIFIED against the
    frozen preflight and across the group's completed members; any disagreement is a
    :class:`BenchmarkIdentityError` and ABORTS.

    NO RUNTIME SUBSTITUTION, EVER, and no preflight call: a failed member is recorded once
    with the population identity it actually received, its group becomes incomplete, and
    no other world or seed takes its place.
    """
    if not isinstance(manifest, V2BenchmarkManifest):
        raise BenchmarkManifestError(
            "episode_design=%r evaluates a frozen generalized_v2 benchmark manifest, got "
            "%s" % (cfg.episode_design, type(manifest).__name__))
    profile = str(cfg.benchmark_profile or "")
    worlds = manifest.profile_worlds(profile)        # an unknown profile RAISES
    rewards: List[float] = []
    unique_confirmed: List[float] = []
    wakes: List[float] = []
    meta_counts = _empty_meta_counts()
    ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
    tally = _ConditionTally(BENCHMARK_CELLS)
    bench = _V2BenchmarkTally()
    group_obs: List[Dict[str, Any]] = []
    n_failed = 0
    n_groups_successful = 0
    n_groups = len(worlds)
    n_attempted = n_groups * BENCHMARK_GROUP_SIZE
    t0 = time.perf_counter()

    for w, world in enumerate(worlds):
        pre = world.pre_solve_cardinality()
        member_rewards: Dict[str, float] = {}
        identities: Dict[str, V2WorldIdentity] = {}
        member_decisions: Dict[str, List[Dict[str, Any]]] = {}

        for member, (cell, mode) in enumerate(world.members()):
            tag = eval_member_tag(round_ordinal=round_ordinal, e=w, member=member,
                                  group_size=BENCHMARK_GROUP_SIZE)
            condition = cell_condition(cell)
            tally.attempt(cell)
            bench.attempt(world.base_cell_key, cell)
            artifacts = None
            if artifacts_root is not None:
                artifacts = _AttemptArtifacts(
                    root=artifacts_root,
                    identity=_AttemptIdentity(
                        phase=str(stage),
                        iteration=iteration,
                        updates_completed=int(updates_completed),
                        eval_round_ordinal=int(round_ordinal),
                        eval_episode_index=int(w),
                        eval_pair_member=int(member),
                        attempt_ordinal=w * BENCHMARK_GROUP_SIZE + member,
                        episode_index=None,
                        seed=int(world.seed),
                        condition=str(condition),
                        severity=(str(cell) if cell in SEVERITIES else None),
                        episode_tag=int(tag),
                    ),
                )
            # A FRESH write-once carrier per member, so a member that fails after stage 2
            # still reports the population identity it really received.
            recorder = RouteRelativePopulationRecorder()
            try:
                out = _run_one_episode(
                    policy, gen, cfg,
                    seed=int(world.seed),
                    episode_tag=tag,
                    deterministic=True,
                    fuel_damage_mode=mode,
                    pre_solve_cardinality=pre,
                    population_recorder=recorder,
                    **_artifact_kwargs(artifacts),
                )
            except (_VisualArtifactError, MeasurementIntegrityError,
                    FuelDamageIntegrityError, BenchmarkIdentityError,
                    ReferenceIntegrityError, MatchAouBackendError):
                raise
            except Exception as exc:
                n_failed += 1
                tally.failure(cell)
                bench.failure(world.base_cell_key, cell)
                load = recorder.load
                _append_failure_record(failures_path, _failure_record(
                    phase="eval",
                    evaluation_stage=stage,
                    updates_completed=updates_completed,
                    iteration=iteration,
                    attempt_ordinal=w * BENCHMARK_GROUP_SIZE + member,
                    episode_index=None,
                    eval_tag="benchmark_v2_%s_%s_tag%d" % (world.key, cell, tag),
                    seed=int(world.seed),
                    condition=condition,
                    cell=cell,
                    cardinality=_failure_cardinality(None, pre, load),
                    pre_solve_cardinality=pre,
                    route_relative_load=load,
                    benchmark=_v2_benchmark_member_identity(
                        manifest, world, cell, profile, None),
                    exc=exc,
                ))
                print("  [bench-v2 %s %s %s] FAILED (seed=%d): %s: %s"
                      % (stage, world.key, cell, world.seed, type(exc).__name__, exc))
                traceback.print_exc()
                continue

            print(_format_episode_block(
                "[bench-v2 stage=%s %s %s seed=%d]"
                % (_ascii(stage), world.key, cell, world.seed), out
            ))
            # THE RECONSTRUCTED V2 WORLD, verified against the frozen manifest BEFORE its
            # reward or its diagnostics are allowed anywhere near a group or a summary.
            identity = _observe_v2_world_identity(
                seed=int(world.seed), pre_solve=pre,
                route_relative_load=out.route_relative_load,
                match_aou_backend=cfg.match_aou_backend,
                allocation_fingerprint=out.allocation_fingerprint,
                world_identity=out.world_identity,
            )
            require_v2_world_matches_manifest(world, identity)
            identities[cell] = identity
            member_rewards[tally.success(out, expected_cell=cell)] = out.reward
            bench.success(world.base_cell_key, cell, out.reward)
            member_decisions[cell] = _wake_decision_records(out.trajectory)
            _append_episode_outcome_record(outcomes_path, _episode_outcome_record(
                out,
                phase=str(stage),
                iteration=iteration,
                updates_completed=int(updates_completed),
                updates_completed_before=int(updates_completed),
                attempt_ordinal=w * BENCHMARK_GROUP_SIZE + member,
                episode_index=None,
                eval_round_ordinal=int(round_ordinal),
                eval_episode_index=int(w),
                eval_group_member=int(member),
                seed=int(world.seed),
                episode_tag=int(tag),
                fuel_damage_mode=str(mode),
                design=cfg.design,
                benchmark=_v2_benchmark_member_identity(
                    manifest, world, cell, profile, identity),
            ))
            rewards.append(out.reward)
            unique_confirmed.append(float(out.targets_confirmed_unique))
            wakes.append(float(out.n_wakes))
            _add_meta_action_counts(meta_counts, out.trajectory)
            if out.ended in ended_counts:
                ended_counts[out.ended] += 1

        require_v2_matched_group_identity(world, identities)
        complete = all(cell in member_rewards for cell, _mode in world.members())
        bench.group(world.base_cell_key, member_rewards, complete=complete)
        if complete:
            n_groups_successful += 1
        group_obs.append({
            "group_key": world.key,
            "base_cell": world.base_cell_key,
            "complete": complete,
            "member_decisions": {c: member_decisions.get(c)
                                 for c in (SEVERITY_MILD, SEVERITY_SEVERE)},
        })

    behaviour = _v2_behaviour_summary(group_obs)
    n_successful = len(rewards)
    episodes_with_wakes = sum(1 for x in wakes if x > 0)
    r = _stats_or_none(rewards)
    unique_confirmed_mean = _stats_or_none(unique_confirmed)["mean"]
    return {
        "evaluation_stage": str(stage),
        "updates_completed": int(updates_completed),
        "iteration": None if iteration is None else int(iteration),
        "eval_round_ordinal": int(round_ordinal),
        "episode_tag_start": eval_member_tag(
            round_ordinal=round_ordinal, e=0, member=0,
            group_size=BENCHMARK_GROUP_SIZE),
        "eval_population": "benchmark_manifest",
        "benchmark_design": EPISODE_DESIGN_GENERALIZED_V2,
        "benchmark_manifest_id": str(manifest.manifest_id),
        "benchmark_label": manifest.label,
        "benchmark_profile": profile,
        "benchmark_n_worlds": int(n_groups),
        "benchmark_n_members": int(n_attempted),
        "benchmark_manifest_n_worlds": int(manifest.n_worlds),
        "benchmark_n_base_cells": len(V2_BENCHMARK_BASE_CELL_KEYS),
        "n_attempted": n_attempted,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "success_fraction": _fraction(n_successful, n_attempted),
        "episodes_with_wakes": int(episodes_with_wakes),
        "wake_fraction_of_successful": _fraction(episodes_with_wakes, n_successful),
        "eval_group_kind": _EVAL_GROUP_KIND_TRIAD,
        "eval_group_size": BENCHMARK_GROUP_SIZE,
        "eval_group_cells": list(BENCHMARK_CELLS),
        "n_groups_attempted": n_groups,
        "n_groups_successful": int(n_groups_successful),
        "group_success_fraction": _fraction(n_groups_successful, n_groups),
        "eval_delta_keys": [_delta_key(c, r_) for c, r_ in BENCHMARK_DELTAS],
        "eval_delta_over": "world_groups_with_all_members_successful",
        "n_pairs_attempted": n_groups,
        "n_pairs_successful": int(n_groups_successful),
        "pair_success_fraction": _fraction(n_groups_successful, n_groups),
        "eval_paired_reward_delta": None,
        "paired_delta_over": "world_groups_with_all_members_successful",
        "eval_reward_mean": r["mean"],
        "eval_reward_min": r["min"],
        "eval_reward_max": r["max"],
        "eval_targets_confirmed_unique_mean": unique_confirmed_mean,
        "target_confirmation_count_semantics": _TARGET_CONFIRMATION_SEMANTICS,
        "eval_wakes_mean": _stats_or_none(wakes)["mean"],
        "aggregates_over": "successful_episodes",
        "meta_action_counts": dict(meta_counts),
        "meta_action_fractions": _meta_fractions(meta_counts),
        "ended_counts": dict(ended_counts),
        **tally.to_record(prefix="eval_"),
        **bench.to_record(),
        "v2_benchmark_groups": _v2_group_population(manifest, profile, group_obs,
                                                    behaviour),
        # THE PRIMARY V2 behavioural measurement. Reward deltas above are SECONDARY.
        "v2_behaviour": behaviour,
        "n_episodes": n_attempted,
        "n_ok": n_successful,
        "eval_kills_mean": unique_confirmed_mean,
        "eval_seconds": time.perf_counter() - t0,
    }


def _require_benchmark_seeds_held_out(
    manifest: BenchmarkManifest, cfg: TrainConfig
) -> None:
    """The benchmark's ACTUAL seeds must lie outside every reachable TRAINING seed.

    THE BAND IS THE MAXIMUM POSSIBLE ATTEMPT BAND (``cfg.max_training_attempts``), which
    under the generalized quota policy is wider than the successful-episode quota because
    ordinary attrition is replaced by further attempts. On the fixed-cell path the two
    are the same number and this check is byte-unchanged.

    THIS IS THE HELD-OUT CHECK FOR A MANIFEST-DRIVEN RUN, and the legacy one cannot serve
    as it. ``TrainConfig.validate`` compares the training band against
    ``eval_base_seed .. eval_base_seed + eval_episodes`` -- exactly the right test when
    that band IS the evaluation schedule, and simply not this run's schedule at all when
    the seeds come from a frozen manifest. Leaving the legacy check to stand in for this
    one would be wrong in both directions: it could reject a perfectly held-out manifest
    because an unused configured band happened to overlap, and it could validate a
    manifest that contains a training seed.

    A collision is a HELD-OUT FAILURE, not attrition: an evaluation world the policy
    trained on measures memorization while reporting generalization, and the numbers it
    produces look entirely normal. So it REFUSES the run, names every offending seed, and
    offers no repair -- no retry, no seed replacement, no manifest rewrite, because each
    of those silently changes the population a result is reported over.

    Called after the manifest is loaded and BEFORE the run directory, the provenance, the
    policy, the generator or any solver work exists, so a refused run costs nothing and
    leaves nothing behind.

    Raises:
        ValueError: one or more manifest world seeds lie inside the training band.
    """
    train_lo = int(cfg.base_seed)
    # THE MAXIMUM POSSIBLE ATTEMPT BAND, not the successful-episode quota. Under the
    # generalized replacement policy an iteration may spend up to
    # `generalized_max_attempts_per_iteration` seeds to collect `episodes_per_iteration`
    # SUCCESSFUL episodes, so `total_episodes` is no longer an upper bound on which seeds
    # the training loop can reach. Checking against the quota would leave a corridor of
    # seeds a run with ordinary attrition really does train on while its manifest was
    # certified held out -- exactly the silent failure this function exists to prevent.
    # Identical to `total_episodes` on the fixed-cell path.
    train_hi = train_lo + int(cfg.max_training_attempts)      # exclusive
    overlap = manifest_seed_overlap(manifest, start=train_lo, stop=train_hi)
    if overlap:
        raise ValueError(
            "benchmark manifest %s is NOT held out: %d of its %d world seed(s) lie "
            "inside this run's MAXIMUM POSSIBLE training attempt band [%d, %d) -- %s. "
            "The policy would be evaluated on worlds it trained on, which measures "
            "memorization while reporting generalization. Refused: no seed is replaced, "
            "retried or rewritten. Move the training band (base_seed / n_iterations x "
            "the per-iteration attempt budget) or freeze a manifest outside it."
            % (manifest.manifest_id[:16], len(overlap), manifest.n_worlds,
               train_lo, train_hi,
               ", ".join(str(s) for s in overlap[:8])
               + (" ..." if len(overlap) > 8 else ""))
        )


def _require_benchmark_tag_namespace(
    manifest: BenchmarkManifest, cfg: TrainConfig
) -> None:
    """The benchmark's scenario-tag namespace must fit one eval round, or REFUSE.

    An artifact-NAMING bound, not a seed bound -- but a violation is the same class of
    silent loss the held-out band's own check exists to prevent: one round's scenario
    JSON overwriting another's. It is checked against the MANIFEST's member count rather
    than ``eval_episodes``, because a manifest-driven round's size is the manifest's, and
    ``TrainConfig.validate`` cannot know it (it holds a path, not a population).
    """
    if manifest.n_members > _EVAL_ROUND_TAG_STRIDE:
        raise ValueError(
            "the benchmark manifest schedules %d member episode(s) per round, which "
            "exceeds one eval round's scenario-tag namespace (%d): consecutive rounds "
            "would write over each other's scenario files. Shorten the manifest or "
            "raise _EVAL_ROUND_TAG_STRIDE."
            % (manifest.n_members, _EVAL_ROUND_TAG_STRIDE)
        )


# =============================================================================
# 6. Checkpointing (SAVE only -- resume is deliberately out of scope)
# =============================================================================

def _print_eval_pair_line(ev: Dict[str, Any]) -> None:
    """One (or two) lines summarizing an eval round's MATCHED-GROUP result.

    Printed next to every eval summary because the round's headline
    ``eval_reward_mean`` spans every cell and therefore answers no question about the
    difficulty factor. Every delta appears with its own denominator -- a delta over 1 of
    4 complete groups is a different claim from the same number over 4 of 4, and the two
    must never be printed as if they were the same.

    A legacy PAIR round prints the same quantities it always did, with the delta now
    NAMED (``damaged_minus_clean``) rather than labelled ``delta`` -- the one wording
    change, and it is the wording that stops being ambiguous the moment a design has more
    than one delta. A TRIAD round prints the three cell means, its three named deltas,
    and on a second line the FD-wake ABORT RATE per severity -- the primary behavioural
    measurement, and the one an operator most wants to watch while a run is still going.
    """
    cells = list(ev.get("eval_group_cells") or list(CONDITIONS))
    kind = str(ev.get("eval_group_kind", _EVAL_GROUP_KIND_PAIR))
    means = " | ".join(
        "%s R=%s" % (cell, _fmt_opt(ev.get("eval_reward_mean_%s" % cell)))
        for cell in cells
    )
    deltas = " ".join(
        "%s=%s" % (key.replace("eval_delta_", ""), _fmt_opt(ev.get(key)))
        for key in (ev.get("eval_delta_keys") or [])
    )
    print("            fd %ss: %s | %s over %s/%s %s(s) | applied=%s wakes=%s rtb=%s "
          "dead=%s"
          % (kind, means, deltas,
             ev.get("n_groups_successful", ev.get("n_pairs_successful")),
             ev.get("n_groups_attempted", ev.get("n_pairs_attempted")), kind,
             ev.get("eval_fuel_damage_events_applied"),
             ev.get("eval_fuel_damage_wakes"),
             ev.get("eval_fuel_damage_rtb_issued"),
             ev.get("eval_deaths")))
    # The severity-response line, only when there are severities to compare. Abort RATE
    # travels with the FD-wake count it is a rate over, because a 100% abort rate over
    # one wake and over eight are different findings.
    severity_cells = [c for c in cells if c in SEVERITIES]
    if severity_cells:
        abort = MetaAction.SELF_PRESERVATION_ABORT.name
        comply = MetaAction.PLAN_COMPLIANCE.name
        print("            fd response: %s"
              % " | ".join(
                  "%s abort=%s comply=%s over %s fd-wake(s)"
                  % (cell,
                     _fmt_opt((ev.get("eval_fd_meta_action_rates_%s" % cell) or {})
                              .get(abort), "%.2f"),
                     _fmt_opt((ev.get("eval_fd_meta_action_rates_%s" % cell) or {})
                              .get(comply), "%.2f"),
                     ev.get("eval_n_%s_fd_wakes" % cell))
                  for cell in severity_cells))


def save_checkpoint(
    policy: Any,
    updater: Union[PPOUpdater, CTDEUpdater],
    iteration: int,
    ckpt_dir: Path,
    critic: Optional[Any] = None,
) -> Path:
    """Save encoder + head + optimizer state (and provenance) to ``ckpt_iter<NNNN>.pt``.

    The optimizer's state_dict is included because Adam's moment estimates ARE training
    state -- a checkpoint without them could not faithfully continue a run. The
    ``PPOConfig`` is stored as a plain dict (not the dataclass) so a loader never needs
    to unpickle a project class.

    THE ACTOR-ONLY PAYLOAD. With ``critic is None`` -- which is every ``actor_only`` run
    -- the saved object holds the five historical keys (``iteration`` / ``encoder`` /
    ``head`` / ``optimizer`` / ``ppo_config``) PLUS ``action_representation_id``. The
    encoder / head tensor shapes did not change, so a historical checkpoint would still
    LOAD into them -- but its weights were trained under the retired node-indexed action
    representation, and semantic compatibility is INTENTIONALLY broken. The id is what
    makes a new checkpoint self-describing; a historical one (no id) remains evidence of
    the old representation. No migration and no warm-start conversion exist.

    A CTDE run saves the ACTUAL CTDE training state, which is strictly more: the same
    six keys (``encoder`` / ``head`` / ``optimizer`` are the ACTOR's), plus
    ``training_mode`` and the critic's own ``critic_encoder`` / ``value_head`` /
    ``critic_optimizer`` / ``ctde_config``. There is deliberately NO second
    "actor export" file -- the actor portion of this one payload is already sufficient
    for later inference, precisely because the actor's keys did not move.

    There is intentionally NO loader here: restoring a run is a separate, deferred task
    (it needs decisions about the seed schedule and the scenario stream that saving
    does not). ``tests/test_graph_train.py`` proves the saved payload round-trips.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ("ckpt_iter%04d.pt" % int(iteration))
    payload: Dict[str, Any] = {
        "iteration": int(iteration),
        "encoder": policy.encoder.state_dict(),
        "head": policy.head.state_dict(),
        "optimizer": updater.optimizer.state_dict(),
        "ppo_config": asdict(updater.cfg),
        "action_representation_id": ACTION_REPRESENTATION_ID,
    }
    if critic is not None:
        payload["training_mode"] = TRAINING_MODE_CTDE
        payload["critic_encoder"] = critic.encoder.state_dict()
        payload["value_head"] = critic.value_head.state_dict()
        payload["critic_optimizer"] = updater.critic_optimizer.state_dict()
        payload["ctde_config"] = asdict(updater.ctde_cfg)
    torch.save(payload, path)
    return path


def _credit_measurement_tags(out: Any) -> Dict[str, Any]:
    """MEASUREMENT-ONLY tags of one successful training episode, for the credit join.

    Read off the episode OUTCOME the trainer already holds, and stored in a trainer-side
    map keyed by episode identity. They never enter a ``Transition``, a record, a batch,
    an observation, the reward or an optimizer: the credit rows are joined to them only
    after the update has already produced its credit values.
    """
    plan = out.fuel_damage_plan or {}
    outcome = out.fuel_damage_outcome or {}
    return {
        "cell": _outcome_cell(plan),
        "condition": plan.get("condition"),
        "severity": plan.get("severity"),
        "fd_selected_ego_id": plan.get("ego_id"),
        "fd_event_tick": outcome.get("event_tick"),
    }


def _credit_rows(
    report: CreditReport,
    *,
    iteration: int,
    updates_completed_before: int,
    measurement_tags: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """One JSON row per transition of ONE productive update. REPORTING-ONLY.

    Every credit number is COPIED out of ``report.batch`` -- the object the update
    consumed -- and never recomputed: no forward pass, no GAE pass, no baseline. Keys the
    row's training mode does not define are ``None`` (never ``0``). The
    ``measurement_join`` block is looked up by ``(episode_index, seed)`` AFTER the
    credit values exist; the credit computation never saw it.
    """
    batch = report.batch
    records = list(report.records)
    cfg = report.cfg
    ctde = report.training_mode == TRAINING_MODE_CTDE
    n_with_wakes = sum(1 for rec in records if rec.has_wakes)
    rows: List[Dict[str, Any]] = []
    for i, tr in enumerate(batch.transitions):
        rec = records[int(batch.record_positions[i])]
        tags = measurement_tags.get((int(rec.episode_index), int(rec.seed)))
        fd_ego = None if tags is None else tags.get("fd_selected_ego_id")
        meta = int(tr.meta_action)
        rows.append({
            "schema": _CREDIT_DIAGNOSTICS_SCHEMA,
            "schema_version": _CREDIT_DIAGNOSTICS_VERSION,
            "action_representation_id": ACTION_REPRESENTATION_ID,
            "training_mode": str(report.training_mode),
            "iteration": int(iteration),
            "updates_completed_before": int(updates_completed_before),
            "episode_seed": int(rec.seed),
            "episode_index": int(rec.episode_index),
            "batch_transition_ordinal": int(i),
            "episode_decision_ordinal": (
                int(batch.decision_ordinals[i]) if ctde else None),
            "ego_chain_ordinal": None if ctde else int(batch.chain_ordinals[i]),
            "ego_id": str(tr.ego_id),
            "tick": int(tr.tick),
            "wake_kind": str(tr.wake_kind),
            "selected_meta_action": meta,
            "selected_meta_action_name": MetaAction(meta).name,
            "selected_node": None if tr.node_v is None else int(tr.node_v),
            "stored_log_prob": float(tr.log_prob),
            "episode_reward": float(rec.episode_reward),
            "transition_reward": (
                float(batch.rewards[i]) if ctde
                else (None if tr.reward is None else float(tr.reward))),
            "raw_advantage": float(batch.raw_advantages[i]),
            "normalized_advantage": float(batch.advantages[i]),
            "batch_raw_advantage_mean": float(batch.adv_mean_raw),
            "batch_raw_advantage_std": float(batch.adv_std_raw),
            "adv_norm_eps": float(cfg.adv_norm_eps),
            "gamma": float(cfg.gamma),
            "batch_n_transitions": int(batch.n_transitions),
            "batch_n_episodes": int(batch.n_episodes),
            "batch_n_episodes_with_wakes": int(n_with_wakes),
            # --- actor_only credit (null under ctde) ---
            "return": None if ctde else float(batch.returns[i]),
            "actor_only_episode_baseline": None if ctde else float(batch.baseline),
            # --- ctde credit (null under actor_only) ---
            "value_old": float(batch.values[i]) if ctde else None,
            "td_residual": float(batch.td_residuals[i]) if ctde else None,
            "value_target": float(batch.value_targets[i]) if ctde else None,
            "gae_lambda": (float(report.ctde_cfg.gae_lambda)
                           if ctde and report.ctde_cfg is not None else None),
            # --- measurement-only join (never an input to anything above) ---
            "measurement_join": {
                "joined": tags is not None,
                "cell": None if tags is None else tags.get("cell"),
                "condition": None if tags is None else tags.get("condition"),
                "severity": None if tags is None else tags.get("severity"),
                "fd_selected_ego_id": fd_ego,
                "fd_event_tick": None if tags is None else tags.get("fd_event_tick"),
                "is_fd_selected_ego": (
                    None if fd_ego is None else str(tr.ego_id) == str(fd_ego)),
            },
        })
    return rows


def _persist_credit_diagnostics(
    path: Path,
    reports: Sequence[CreditReport],
    diag: Mapping[str, Any],
    *,
    iteration: int,
    updates_completed_before: int,
    measurement_tags: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> int:
    """Write one update's credit rows, or STOP the run. Returns the row count.

    Fails LOUD (:class:`CreditDiagnosticsError`) when a productive update handed over no
    report or several, when the rows do not cover exactly the update's transitions, or
    when the file cannot be written -- each would leave an instrumented run silently
    incomplete.
    """
    productive = (int(diag.get("n_epochs_run", 0)) > 0
                  and int(diag.get("n_transitions", 0)) > 0)
    if len(reports) > 1:
        raise CreditDiagnosticsError(
            "an update handed %d credit reports; exactly one is expected" % len(reports))
    if not reports:
        if productive:
            raise CreditDiagnosticsError(
                "a productive update (%d transition(s)) produced no credit report; the "
                "train_credit_diagnostics artifact would be incomplete"
                % int(diag["n_transitions"]))
        return 0
    rows = _credit_rows(reports[0], iteration=iteration,
                        updates_completed_before=updates_completed_before,
                        measurement_tags=measurement_tags)
    if len(rows) != int(diag.get("n_transitions", -1)):
        raise CreditDiagnosticsError(
            "credit rows (%d) do not cover the update's %s transition(s)"
            % (len(rows), diag.get("n_transitions")))
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
    except (OSError, TypeError, ValueError) as exc:
        raise CreditDiagnosticsError(
            "could not persist %s: %s" % (_CREDIT_DIAGNOSTICS_FILENAME, exc)) from exc
    return len(rows)


def _actor_gradient_group(tr: Any, tags: Optional[Mapping[str, Any]]) -> str:
    """The MEASUREMENT group of one transition. Trainer-side only; fails LOUD.

    ``immediate_fuel_damage`` wakes split by the joined severity and must belong to the
    joined FD-selected ego; ``post_fd_boundary`` wakes are ``post_fd``; every other wake
    is ``ordinary``. An immediate-FD wake that cannot be attributed raises rather than
    being filed under a group it may not belong to.
    """
    kind = str(tr.wake_kind)
    if kind == WAKE_KIND_POST_FD_BOUNDARY:
        return _ACTOR_GRADIENT_GROUP_POST_FD
    if kind != WAKE_KIND_IMMEDIATE_FD:
        return _ACTOR_GRADIENT_GROUP_ORDINARY
    if tags is None:
        raise ActorGradientDiagnosticsError(
            "an immediate fuel-damage transition has no measurement join")
    if str(tags.get("fd_selected_ego_id")) != str(tr.ego_id):
        raise ActorGradientDiagnosticsError(
            "an immediate fuel-damage transition of ego %r is not the joined FD-selected "
            "ego %r" % (tr.ego_id, tags.get("fd_selected_ego_id")))
    severity = tags.get("severity")
    if severity == SEVERITY_MILD:
        return _ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_MILD
    if severity == SEVERITY_SEVERE:
        return _ACTOR_GRADIENT_GROUP_IMMEDIATE_FD_SEVERE
    raise ActorGradientDiagnosticsError(
        "an immediate fuel-damage transition has unattributable severity %r" % (severity,))


def _actor_gradient_group_ids(
    records: Sequence[CTDEEpisodeRecord],
    measurement_tags: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> List[int]:
    """OPAQUE group ids, one per transition, in the order ``compute_ctde_advantages``
    flattens ``records`` (record order, then each record's decisions; zero-wake records
    contribute nothing). Only these integers cross into the updater."""
    return [
        _ACTOR_GRADIENT_GROUPS.index(_actor_gradient_group(
            tr, measurement_tags.get((int(rec.episode_index), int(rec.seed)))))
        for rec in records for tr in rec.transitions
    ]


# Vector math for the gradient record is ELEMENTWISE numpy only (no `np.dot` /
# `np.linalg`): BLAS-backed calls initialize a second OpenMP runtime next to torch's on
# this Windows stack and abort the process (see PLOTTING RUNS IN A CHILD PROCESS above).
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def _norm(v: np.ndarray) -> float:
    return math.sqrt(_dot(v, v))


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Cosine, or ``None`` when either vector has zero norm (undefined, never 0)."""
    na, nb = _norm(a), _norm(b)
    return None if na == 0.0 or nb == 0.0 else _dot(a, b) / (na * nb)


def _projection(v: np.ndarray, onto: np.ndarray) -> Optional[float]:
    """Signed length of ``v`` along ``onto``'s direction; ``None`` if ``onto`` is zero."""
    n = _norm(onto)
    return None if n == 0.0 else _dot(v, onto) / n


def _actor_gradient_record(
    report: ActorGradientReport,
    *,
    iteration: int,
    updates_completed_before: int,
    measurement_tags: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    """ONE JSON record from ONE :class:`ActorGradientReport`. REPORTING-ONLY.

    The group ids the updater used are RE-DERIVED from the report's own batch and
    records and must match exactly, so a misaligned id list fails LOUD instead of
    misattributing gradient. Empty groups carry ``n_transitions = 0``, norm ``0.0`` and
    ``null`` cosine / projection. No full gradient vector is persisted.
    """
    batch = report.batch
    records = list(report.records)
    n = int(batch.n_transitions)
    expected = []
    for i, tr in enumerate(batch.transitions):
        rec = records[int(batch.record_positions[i])]
        expected.append(_ACTOR_GRADIENT_GROUPS.index(_actor_gradient_group(
            tr, measurement_tags.get((int(rec.episode_index), int(rec.seed))))))
    if list(report.group_ids) != expected or len(expected) != n:
        raise ActorGradientDiagnosticsError(
            "the update's gradient group ids are not aligned with its batch transitions")
    counts = {name: int(report.group_counts.get(gid, 0))
              for gid, name in enumerate(_ACTOR_GRADIENT_GROUPS)}
    if (set(report.group_counts) - set(range(len(_ACTOR_GRADIENT_GROUPS)))
            or sum(counts.values()) != n):
        raise ActorGradientDiagnosticsError(
            "gradient group counts do not partition the batch's %d transitions" % n)
    total = report.total_policy_surrogate_grad
    actor_total = report.total_actor_loss_grad
    zero = np.zeros_like(total)
    grads = {name: report.group_policy_surrogate_grads.get(gid, zero)
             for gid, name in enumerate(_ACTOR_GRADIENT_GROUPS)}

    def block(v: np.ndarray, count: int) -> Dict[str, Any]:
        defined = count > 0
        return {
            "n_transitions": int(count),
            "batch_fraction": float(count) / n,
            "grad_norm": _norm(v),
            "cosine_vs_total": _cosine(v, total) if defined else None,
            "projection_on_total": _projection(v, total) if defined else None,
        }

    derived_vecs = {name: sum((grads[m] for m in members), zero)
                    for name, members in _ACTOR_GRADIENT_DERIVED.items()}
    derived_counts = {name: sum(counts[m] for m in members)
                      for name, members in _ACTOR_GRADIENT_DERIVED.items()}
    fd, non_fd = derived_vecs["fd"], derived_vecs["non_fd"]
    both = derived_counts["fd"] > 0 and derived_counts["non_fd"] > 0
    residual = total - sum(grads.values(), zero)
    total_norm = _norm(total)
    residual_norm = _norm(residual)
    return {
        "schema": _ACTOR_GRADIENT_DIAGNOSTICS_SCHEMA,
        "schema_version": _ACTOR_GRADIENT_DIAGNOSTICS_VERSION,
        "action_representation_id": ACTION_REPRESENTATION_ID,
        "training_mode": TRAINING_MODE_CTDE,
        "iteration": int(iteration),
        "updates_completed_before": int(updates_completed_before),
        "epoch": int(report.epoch),
        "gradient": "ppo_policy_surrogate_before_entropy",
        "group_loss": "sum_over_group_div_total_batch_transitions",
        "n_actor_parameters": int(total.size),
        "batch_n_transitions": n,
        "entropy_coeff": float(report.entropy_coeff),
        "total_policy_surrogate_grad_norm": total_norm,
        "total_actor_loss_grad_norm": _norm(actor_total),
        "cosine_policy_surrogate_vs_actor_loss": _cosine(total, actor_total),
        "groups": {name: block(grads[name], counts[name])
                   for name in _ACTOR_GRADIENT_GROUPS},
        "derived": {name: block(derived_vecs[name], derived_counts[name])
                    for name in _ACTOR_GRADIENT_DERIVED},
        "fd_grad_norm": _norm(fd),
        "non_fd_grad_norm": _norm(non_fd),
        "cosine_fd_vs_non_fd": _cosine(fd, non_fd) if both else None,
        "projection_non_fd_on_fd": _projection(non_fd, fd) if both else None,
        "reconstruction_error_norm": residual_norm,
        "reconstruction_relative_error": (
            None if total_norm == 0.0 else residual_norm / total_norm),
    }


def _persist_actor_gradient_diagnostics(
    path: Path,
    reports: Sequence[ActorGradientReport],
    diag: Mapping[str, Any],
    *,
    iteration: int,
    updates_completed_before: int,
    measurement_tags: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> int:
    """Write one update's gradient record, or STOP the run. Returns the record count.

    Fails LOUD (:class:`ActorGradientDiagnosticsError`) when a productive update handed
    over no report or several, when the record does not cover the update's transitions
    or its group ids are misaligned, or when the file cannot be written.
    """
    productive = (int(diag.get("n_epochs_run", 0)) > 0
                  and int(diag.get("n_transitions", 0)) > 0)
    if len(reports) > 1:
        raise ActorGradientDiagnosticsError(
            "an update handed %d gradient reports; exactly one is expected" % len(reports))
    if not reports:
        if productive:
            raise ActorGradientDiagnosticsError(
                "a productive update (%d transition(s)) produced no gradient report"
                % int(diag["n_transitions"]))
        return 0
    record = _actor_gradient_record(reports[0], iteration=iteration,
                                    updates_completed_before=updates_completed_before,
                                    measurement_tags=measurement_tags)
    if record["batch_n_transitions"] != int(diag.get("n_transitions", -1)):
        raise ActorGradientDiagnosticsError(
            "the gradient record covers %d transition(s), the update %s"
            % (record["batch_n_transitions"], diag.get("n_transitions")))
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, allow_nan=False) + "\n")
            fh.flush()
    except (OSError, TypeError, ValueError) as exc:
        raise ActorGradientDiagnosticsError(
            "could not persist %s: %s" % (_ACTOR_GRADIENT_DIAGNOSTICS_FILENAME, exc)) from exc
    return 1


def _observed_credit_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the credit artifact actually carries -- OBSERVED, never asserted."""
    versions = sorted({int(r["schema_version"]) for r in rows
                       if isinstance(r.get("schema_version"), int)
                       and not isinstance(r.get("schema_version"), bool)})
    reps = sorted({str(r["action_representation_id"]) for r in rows
                   if isinstance(r.get("action_representation_id"), str)})
    return {
        "source": _CREDIT_DIAGNOSTICS_FILENAME,
        "recorded": bool(rows),
        "n_rows": len(rows),
        "schema_versions_observed": versions,
        "action_representation_ids_observed": reps,
        "training_modes_observed": sorted({str(r.get("training_mode")) for r in rows}),
        "schema_version_writer": _CREDIT_DIAGNOSTICS_VERSION,
    }


# =============================================================================
# 7. The training loop
# =============================================================================

class _EarlyStoppingMonitor:
    """The ``training_reward_plateau_v1`` state machine, over TRAINING reward alone.

    PURE: it is handed one number per completed iteration and holds no reference to the
    policy, the critic, the buffer, the updater, an evaluation record, the benchmark
    manifest or the config. That is not a stylistic choice -- it is the comparator
    isolation the policy exists to guarantee, expressed as a type signature: there is no
    channel through which a held-out or benchmark quantity could reach the decision even
    by accident (:data:`EARLY_STOPPING_METRIC`).

    THE STATE MACHINE, in COMPLETED-ITERATION counts (never zero-based indices):

      * checks fall at ``min_iterations`` and every ``window_iterations`` afterwards, so
        with the approved defaults they are 100, 125, 150, 175, ...;
      * each check averages the most recent ``window_iterations`` values, so consecutive
        monitored windows do NOT overlap and the first ``min_iterations -
        window_iterations`` completed iterations (75 at the defaults) fall outside every
        window;
      * the FIRST check only establishes the best window mean and cannot stop
        (``patience_windows >= 1``);
      * a later window is a MEANINGFUL improvement iff
        ``window_mean >= best_window_mean + min_delta``. It then becomes the new best and
        resets the stale counter; otherwise the stale counter increments;
      * the run stops when ``stale_windows >= patience_windows`` -- 175 completed
        iterations at the earliest, on the approved defaults.

    A missing ``train_reward_mean`` inside a monitored window RAISES
    (:class:`EarlyStoppingIntegrityError`) instead of being skipped or read as ``0.0``.
    A ``None`` outside every monitored window is not consumed by the rule and is
    therefore not judged here -- what is refused is fabricating a window mean, never an
    iteration the mechanism never reads.

    It decides WHETHER to stop and records WHY; it never touches the loop, the records or
    the artifacts. The caller owns all of that.
    """

    def __init__(
        self,
        *,
        min_iterations: int,
        window_iterations: int,
        patience_windows: int,
        min_delta: float,
    ) -> None:
        self.policy = EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU
        self.metric = EARLY_STOPPING_METRIC
        self.min_iterations = int(min_iterations)
        self.window_iterations = int(window_iterations)
        self.patience_windows = int(patience_windows)
        self.min_delta = float(min_delta)
        # (completed_iteration, train_reward_mean) for the most recent window, kept as a
        # list of PAIRS so a refusal can name the offending iteration rather than only
        # its position.
        self._window: List[Tuple[int, Optional[float]]] = []
        self._best: Optional[float] = None
        self._stale = 0
        self.checks: List[Dict[str, Any]] = []
        self.stop_triggered = False
        self.stop_completed_iterations: Optional[int] = None

    # ------------------------------------------------------------------
    def is_due(self, completed_iterations: int) -> bool:
        """True iff a check falls at this COMPLETED-ITERATION count."""
        count = int(completed_iterations)
        if count < self.min_iterations:
            return False
        return (count - self.min_iterations) % self.window_iterations == 0

    def observe(
        self, *, completed_iterations: int, train_reward_mean: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """Record one completed iteration; return its check record, or ``None``.

        ``None`` means no check was due at this count -- not that a check passed. The
        returned dict carries everything needed to reconstruct the decision, and
        ``stop_triggered`` is the caller's signal to leave the loop.
        """
        count = int(completed_iterations)
        self._window.append(
            (count, None if train_reward_mean is None else float(train_reward_mean))
        )
        if len(self._window) > self.window_iterations:
            self._window = self._window[-self.window_iterations:]
        if not self.is_due(count):
            return None

        if len(self._window) < self.window_iterations:
            # Unreachable while `validate` requires min_iterations >= window_iterations;
            # kept so a partial window can never be averaged as though it were full.
            raise EarlyStoppingIntegrityError(
                "early stopping: the window due at %d completed iteration(s) holds only "
                "%d of %d values. A partial window is not the quantity the policy is "
                "defined over, so this stops rather than averaging a shorter one."
                % (count, len(self._window), self.window_iterations)
            )
        missing = [i for i, value in self._window if value is None]
        if missing:
            raise EarlyStoppingIntegrityError(
                "early stopping: the window due at %d completed iteration(s) contains "
                "%d iteration(s) with no %s (%s). Under the successful-episode quota an "
                "iteration cannot complete without one, so this is an instrument "
                "contradiction. Skipping the value would average a window over a "
                "population nobody chose, and reading it as 0.0 would insert the oracle "
                "OPTIMUM into a plateau test -- both fabricate convergence evidence."
                % (count, len(missing), self.metric,
                   ", ".join("iteration %d" % i for i in missing))
            )

        values = [float(value) for _i, value in self._window]
        window_mean = sum(values) / len(values)
        stale_before = self._stale
        best_before = self._best
        if best_before is None:
            # THE BASELINE. There is no earlier best to improve on, so it sets one and
            # cannot stop; `best_window_mean_before` stays null so a reader is never left
            # comparing against a fabricated zero.
            kind = EARLY_STOPPING_CHECK_BASELINE
            improvement: Optional[float] = None
            meaningful = True
        else:
            kind = EARLY_STOPPING_CHECK_COMPARISON
            improvement = window_mean - best_before
            meaningful = window_mean >= best_before + self.min_delta
        if meaningful:
            self._best = window_mean
            self._stale = 0
        else:
            self._stale = stale_before + 1
        stop = self._stale >= self.patience_windows
        if stop:
            self.stop_triggered = True
            self.stop_completed_iterations = count

        check = {
            "policy": self.policy,
            "metric": self.metric,
            "check_kind": kind,
            # Both forms, so neither a reader nor a later aggregate has to convert one
            # into the other and risk an off-by-one between them.
            "completed_iterations": count,
            "iteration": count - 1,
            "min_iterations": self.min_iterations,
            "window_iterations": self.window_iterations,
            "window_first_completed_iteration": self._window[0][0],
            "window_last_completed_iteration": self._window[-1][0],
            "window_mean": window_mean,
            "best_window_mean_before": best_before,
            "best_window_mean_after": self._best,
            "min_delta": self.min_delta,
            "improvement": improvement,
            "meaningful_improvement": bool(meaningful),
            "stale_windows_before": stale_before,
            "stale_windows": self._stale,
            "patience_windows": self.patience_windows,
            "stop_triggered": bool(stop),
        }
        self.checks.append(check)
        return check


def train(
    cfg: TrainConfig,
    *,
    config_source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run ``cfg.n_iterations`` PPO iterations end-to-end; returns an aggregate summary.

    Per iteration: collect ``episodes_per_iteration`` stochastic episodes into a fresh
    :class:`PPOBuffer`, run ONE :meth:`PPOUpdater.update`, clear the buffer, and append
    ONE scalar record to ``train_records.jsonl``.

    Two failure modes are NORMAL and are logged rather than raised:
      * a failed EPISODE (solver hiccup, engine edge case, an exact-cardinality
        construction failure) is recorded in ``episode_failures.jsonl`` with its
        pipeline stage and traceback, counted, and skipped -- it never enters the
        buffer, so it cannot distort the baseline, and the run continues. Its seed is
        NOT retried and NOT replaced (:data:`_EXACT_CARDINALITY_POLICY`);
      * a ZERO-WAKE iteration (no ego woke in any episode) yields an empty batch, and
        ``update`` documents that as a clean no-op with ``n_epochs_run == 0``. It is
        logged like any other iteration -- an iteration in which nothing was sensed is
        a legitimate outcome of the event-triggered design, not an error to swallow. A
        successful zero-wake episode is a REAL episode and is counted as one; only a
        raised attempt counts as failed.

    Those two are DISJOINT states (:func:`_iteration_outcome`) even though both end with
    ``n_epochs_run == 0``, and both the console flag and the summary counters keep them
    apart: an all-failed batch measured nothing, a zero-wake batch measured episodes in
    which nobody sensed anything.

    PROVENANCE IS A PRECONDITION, not a log line. It is collected before this function
    creates the run directory (so the run's own artifacts cannot register as dirty
    source state), and a run whose Git provenance is INCOMPLETE -- no SHA, or a SHA
    without a clean/dirty verdict -- raises before the policy, the generator or any
    episode exists. The attempted ``run_config.json`` is written first so the refusal is
    inspectable. A dirty tree only WARNS: that is a hazard a researcher may choose.

    EVALUATION TIMING. When evaluation is enabled, ONE ``pre_update`` round runs after
    the initial policy is built and before the first training episode, the first buffer
    insert and the first optimizer step -- ``updates_completed == 0``. That is the
    held-out measurement of the UNTRAINED policy, and without it there is nothing to
    compare a trained curve against. ``updates_completed`` counts updates that actually
    ran epochs, so a zero-wake iteration (a no-op update) does not inflate it.

    Each eval round is given the NEXT scenario-tag namespace (``eval_round_ordinal`` ->
    :func:`eval_episode_tag`), so every round's generated worlds survive the run instead
    of the next round overwriting them. The held-out SEEDS are untouched.

    EARLY STOPPING (opt-in, ``generalized_v1`` only). With ``early_stopping`` off this
    loop is fixed-budget and nothing below changes. On, ONE check may fall at the end of
    an iteration, and its ORDERING is the contract: the iteration's training record is
    built, the check is computed FROM THAT RECORD'S ``train_reward_mean`` and attached to
    it, the record is flushed -- and only then, if the plateau rule fired, the loop exits
    BEFORE the periodic evaluation and checkpoint branch for that same boundary. So no
    held-out or benchmark measurement can influence the decision, and the final
    evaluation is strictly post-decision. Finalization then uses the ACTUAL last
    completed iteration, which is the maximum-budget one on a run that was not stopped.

    Every successful attempt prints one ``OK`` block on return
    (:func:`_format_episode_block`), before the next attempt starts.

    The updater (hence its Adam moments) is built ONCE for the whole run.

    ``config_source`` is the audit record from :func:`resolve_train_config` (which JSON
    preset produced this config, and what the command line overrode). It is recorded in
    ``run_config.json`` and read by nothing -- ``cfg`` is the config; this only says
    where it came from. Omitted -- as every DIRECT caller does, ``_selftest`` included --
    the run records ``resolved_from = "direct_config"``, which is the truthful statement
    that no command line and no preset were involved (:func:`config_source_record`).
    """
    cfg.validate()

    # THE FROZEN BENCHMARK, loaded and VERIFIED before anything is created and before a
    # second of compute is spent: a manifest that fails its own content hash, is out of
    # canonical order, or is missing a stratum must cost nothing and leave nothing
    # behind. `validate` has already refused a generalized run that named none, and a
    # fixed-cell run that named one, so reaching here with a path means the design asked
    # for it. `None` -> this run evaluates the historical held-out band (or not at all).
    benchmark: Optional[Union[BenchmarkManifest, V2BenchmarkManifest]] = None
    if cfg.generalized and cfg.eval_enabled:
        # DESIGN-AWARE: V2 reads ONLY the ten-cell V2 schema (which refuses a V1 manifest
        # by schema), V1 makes exactly its historical call. Under V2 the held-out check
        # below runs over the WHOLE manifest -- both profiles -- whichever one this run
        # evaluates.
        if cfg.route_relative_population:
            benchmark = load_v2_benchmark_manifest(str(cfg.benchmark_manifest))
        else:
            benchmark = load_benchmark_manifest(str(cfg.benchmark_manifest))
        # THE held-out check for this run's real evaluation seeds. Deliberately here --
        # after the manifest is known, before the run directory, the provenance, the
        # policy, the generator or any solver work exists.
        _require_benchmark_seeds_held_out(benchmark, cfg)
        _require_benchmark_tag_namespace(benchmark, cfg)

    # PROVENANCE FIRST -- before this run creates ANYTHING. Not merely before the
    # engine, the policy, the generator or bonmin: before the run directory itself.
    # `output_dir` may point inside the repository, and a directory this run created is
    # untracked, so collecting after `mkdir` would let the run's own scenarios and
    # ledger show up as pre-existing dirty SOURCE state -- provenance contaminated by
    # the act of recording it.
    provenance = collect_provenance(cfg, benchmark=benchmark)
    git_info = provenance["git"]

    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scen_dir = run_dir / "scenarios"
    scen_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    # The visual-artifact switch, resolved ONCE. `None` disables capture everywhere --
    # `_run_one_episode` and `evaluate` both read it as the single on/off signal, so the
    # OFF path never constructs an identity or touches the filesystem.
    artifacts_root = (
        run_dir / _VISUAL_ARTIFACTS_DIRNAME if cfg.visual_artifacts else None
    )
    train_records_path = run_dir / "train_records.jsonl"
    eval_records_path = run_dir / "eval_records.jsonl"
    failures_path = run_dir / "episode_failures.jsonl"
    # The durable per-SUCCESSFUL-ATTEMPT stream. Disjoint from the failure ledger by
    # construction: an attempt appears in exactly one of the two files.
    outcomes_path = run_dir / _EPISODE_OUTCOMES_FILENAME
    # The TRAINING-ONLY credit stream: the credit values each productive update used.
    credit_path = run_dir / _CREDIT_DIAGNOSTICS_FILENAME
    # The OPT-IN actor-gradient stream; `None` (no file at all) when it is off.
    gradient_path = (run_dir / _ACTOR_GRADIENT_DIAGNOSTICS_FILENAME
                     if cfg.actor_gradient_diagnostics else None)

    # Written BEFORE the completeness gate below, so a refused run still leaves an
    # inspectable record of what was attempted and why it was refused.
    run_config_path = write_run_config(run_dir, cfg, provenance=provenance,
                                       config_source=config_source,
                                       benchmark=benchmark)

    if not git_info["available"]:
        raise RuntimeError(
            "provenance: complete Git provenance is UNAVAILABLE (%s). A training run "
            "must be attributable to an exact code state -- a commit SHA without a "
            "clean/dirty verdict does not say what actually ran -- so this stops here "
            "rather than spending compute on records nobody can tie to a revision. The "
            "attempted run_config.json was written to %s."
            % (git_info["reason"], str(run_config_path))
        )
    if git_info["dirty"]:
        # A dirty tree is a RESEARCH HAZARD, not an error: a researcher may deliberately
        # run an uncommitted experiment. It is reported loudly and never normalized.
        print("[WARN] provenance: the working tree is DIRTY at %s (%s uncommitted "
              "path(s)). The exact code that produced this run exists only on this "
              "machine." % (git_info["commit"], git_info["dirty_path_count"]))

    # Truncate the ledger and the outcome stream: they describe THIS run, and appending
    # to a previous run's records in a reused directory would silently corrupt the
    # accounting. After the gate, so a refused run never destroys an earlier run's files.
    for append_only_path in (failures_path, outcomes_path, credit_path, gradient_path):
        if append_only_path is None:
            continue
        with open(append_only_path, "w", encoding="utf-8"):
            pass

    # Match the rollout/selftest PlaybackRecorder override (harmless when recording is
    # off, which it always is here). Lazy import: engine boundary.
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    # ONE policy (weights pinned by base_seed) and ONE updater for the whole run --
    # rebuilding the updater per iteration would silently discard Adam's moments.
    torch.manual_seed(cfg.base_seed)
    policy = build_policy()
    # PHASE B. The critic and the CTDE updater exist ONLY on a `ctde` run. On an
    # `actor_only` run `critic` stays None, `PPOUpdater` is built exactly as before, and
    # nothing below constructs a central observation -- which is what makes "actor_only
    # is unchanged" a structural fact rather than a claim about a coefficient. Like the
    # policy, the critic is built ONCE: it carries training state (its weights AND its
    # Adam moments) across every iteration.
    critic = build_central_critic() if cfg.ctde_enabled else None
    updater: Union[PPOUpdater, CTDEUpdater]
    if critic is not None:
        updater = CTDEUpdater(policy, critic, cfg.ppo, cfg.ctde)
    else:
        updater = PPOUpdater(policy, cfg.ppo)
    gen = _build_generator(scen_dir)

    print("=" * 78)
    print("graph_train: %d iteration(s) x %d episode(s) = %d training episodes"
          % (cfg.n_iterations, cfg.episodes_per_iteration, cfg.total_episodes))
    print("base_seed=%d  train seeds [%d, %d)"
          % (cfg.base_seed, cfg.base_seed,
             cfg.base_seed + cfg.max_training_attempts))
    if cfg.generalized:
        # ECHOED BEFORE COMPUTE, because it changes what the line above means: the band
        # is the MAXIMUM POSSIBLE attempt band, and `episodes_per_iteration` is a quota
        # of SUCCESSFUL episodes rather than a count of attempts.
        print("training attempts: %s -- %d SUCCESSFUL episode(s) per iteration, at most "
              "%d attempt(s) each (%d seeds max for the run). An ordinary failure is "
              "recorded, SPENDS its seed, and is replaced by the next attempt; "
              "exhausting the budget ABORTS rather than updating on a partial batch."
              % (cfg.training_attempt_policy, cfg.episodes_per_iteration,
                 cfg.max_attempts_per_iteration, cfg.max_training_attempts))
    # THE STOPPING RULE, echoed before compute is spent -- and only when it is armed, so
    # a fixed-budget run's console is unchanged. `n_iterations` is stated as the MAXIMUM
    # it now is, next to the earliest point the rule could possibly fire.
    if cfg.early_stopping_enabled:
        print("early stopping: %s on %s -- min_iterations=%d window=%d patience=%d "
              "min_delta=%.4f"
              % (EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU, EARLY_STOPPING_METRIC,
                 cfg.early_stopping_min_iterations,
                 cfg.early_stopping_window_iterations,
                 cfg.early_stopping_patience_windows,
                 float(cfg.early_stopping_min_delta)))
        print("          n_iterations=%d is the MAXIMUM budget; the earliest possible "
              "stop is %d completed iteration(s). The decision reads TRAINING reward "
              "only -- no held-out, benchmark, critic or PPO diagnostic enters it, and "
              "a stop is NOT a convergence claim."
              % (cfg.n_iterations, cfg.early_stopping_earliest_stop_iterations))
    if cfg.eval_enabled:
        print("eval: every %d iter, %d held-out seed(s) x %d matched %s member(s) "
              "(%s) = %d episode(s), FIXED seeds [%d, %d)"
              % (cfg.eval_every, cfg.eval_episodes, cfg.eval_group_size,
                 cfg.eval_group_kind, "/".join(cfg.reported_cells),
                 cfg.eval_episodes * cfg.eval_group_size, cfg.eval_base_seed,
                 cfg.eval_base_seed + cfg.eval_episodes))
    else:
        print("eval: DISABLED")
    print("ppo: %s" % (asdict(cfg.ppo),))
    # The training algorithm, echoed BEFORE any compute is spent, next to the same
    # standing reminder the record carries: CTDE changes TRAINING only.
    if cfg.ctde_enabled:
        print("training_mode: %s (centralized critic during TRAINING only; "
              "evaluation and inference stay decentralized actor-only)"
              % TRAINING_MODE_CTDE)
        print("ctde: %s" % (asdict(cfg.ctde),))
    else:
        print("training_mode: %s (no critic, no central observation)"
              % TRAINING_MODE_ACTOR_ONLY)
    # The construction cell as the run will really build it. This is the standing
    # defence against a config that reads plausibly and generates something else: the
    # operator sees the EMITTED target count, not a derived one, before compute is spent.
    # WHICH POPULATION, echoed before compute is spent -- an operator who meant to run
    # the generalized design and typed the fixed cell (or the reverse) sees it here, not
    # in the results.
    design = cfg.design
    print("episode_design: %s  [hidden=%s  fd_eligibility=%s  post_fd_wake=%s  "
          "reference=%s]"
          % (design.design, design.hidden_policy, design.eligibility_policy,
             design.post_fd_wake_policy, design.reference_policy))
    # WHICH allocation objective, echoed for the same reason as the design above, and
    # stated as a SEPARATE EXPLICIT selector whose valid VALUE SET the design constrains --
    # so an operator can neither read one off the other nor believe one was chosen for
    # them. Both branches say the same rule, because a P1 run may be any design and a
    # legacy run may be either historical one. A P1 run additionally says outright that it
    # is not the objective the approved measurements were taken on.
    backend = resolve_match_aou_backend(cfg.match_aou_backend)
    if backend == MATCH_AOU_BACKEND_P1_MILP_V1:
        print("match_aou_backend: %s  (deterministic p = 1 MILP, no EPSILON; STATED by "
              "this run, never inferred -- fixed_cell_v1 and generalized_v1 accept either "
              "approved objective, and generalized_v2 is defined ONLY against this one. "
              "NOT the historical objective: it removes the legacy stacking incentive, so "
              "allocations -- and therefore hidden geometry and feasibility -- can differ. "
              "BONMIN is not invoked.)" % backend)
    else:
        print("match_aou_backend: %s  (frozen MINLP through BONMIN -- the historical "
              "objective; STATED by this run, never inferred. Accepted by fixed_cell_v1 "
              "and generalized_v1; generalized_v2 requires p1_milp_v1 and REFUSES this "
              "one.)" % backend)
    if cfg.route_relative_population:
        print("scenario (GENERALIZED-V2): the cell is SAMPLED PER EPISODE in TWO STAGES")
        print("          stage 1 (before the known-only solve): A ~ U{%s}, "
              "K | A ~ U{A+%s}"
              % (",".join(str(a) for a in GENERALIZED_V2_AGENT_COUNTS),
                 ", A+".join(str(o) for o in GENERALIZED_V2_KNOWN_OFFSETS)))
        print("          stage 2 (after it): H_requested ~ U{1..R}, R = the number of "
              "egos that solve routed")
        print("          policies=%s / %s  rng_domains=%s / %s (two OWN seed domains: "
              "neither can move, or be moved by, the fuel-damage or placement streams)"
              % (PRE_SOLVE_CARDINALITY_POLICY_V2, HIDDEN_LOAD_POLICY_ROUTE_RELATIVE_V2,
                 V2_CARDINALITY_RNG_DOMAIN, V2_HIDDEN_LOAD_RNG_DOMAIN))
        print("          num_agents / n_known / n_hidden are NOT read on this path")
        print("          bounded backoff may realize FEWER hidden targets than "
              "requested; that is a RECORDED outcome, never a retry")
        print("          evaluation (when enabled) is the frozen ten-cell %s benchmark, "
              "one declared profile" % EPISODE_DESIGN_GENERALIZED_V2)
    elif cfg.generalized:
        print("scenario (GENERALIZED-V1): the cell is SAMPLED PER EPISODE -- "
              "A ~ U{%s}, K == A, H_requested ~ U{1..A}"
              % ",".join(str(a) for a in GENERALIZED_AGENT_COUNTS))
        print("          sampler=%s rng_domain=%s (its OWN seed domain: it cannot move, "
              "and cannot be moved by, the fuel-damage or placement streams)"
              % (CARDINALITY_SAMPLER_POLICY, CARDINALITY_RNG_DOMAIN))
        print("          num_agents / n_known / n_hidden are NOT read on this path")
        print("          bounded backoff may realize FEWER hidden targets than "
              "requested; that is a RECORDED outcome, never a retry")
    else:
        print("scenario (construction): num_agents=%d  n_known=%d + n_hidden=%d "
              "-> %d target(s) in the executed world"
              % (cfg.num_agents, cfg.n_known, cfg.n_hidden, cfg.n_targets_emitted))
        print("          the generator writes the %d known target(s); setup_episode "
              "places the %d hidden one(s) route-relative and patches them in "
              "(split_tasks NOT run)" % (cfg.n_known, cfg.n_hidden))
    if isinstance(benchmark, V2BenchmarkManifest):
        chosen = benchmark.profile_worlds(str(cfg.benchmark_profile))
        print("benchmark (%s): %s  profile=%s  %d of %d world group(s) x %d member(s) = "
              "%d episode(s)/round, %d base cells (A x K-A)"
              % (EPISODE_DESIGN_GENERALIZED_V2, benchmark.manifest_id[:16],
                 cfg.benchmark_profile, len(chosen), benchmark.n_worlds,
                 BENCHMARK_GROUP_SIZE, len(chosen) * BENCHMARK_GROUP_SIZE,
                 len(V2_BENCHMARK_BASE_CELL_KEYS)))
    elif benchmark is not None:
        print("benchmark: %s  %d world group(s) x %d member(s) = %d episode(s)/round, "
              "%d requested strata"
              % (benchmark.manifest_id[:16], benchmark.n_worlds, BENCHMARK_GROUP_SIZE,
                 benchmark.n_members, len(BENCHMARK_STRATA)))
    if benchmark is not None:
        print("          worlds per base cell: %s"
              % benchmark.worlds_per_base_cell())
        print("          a failed member is recorded and SKIPPED; its group contributes "
              "to NO within-world delta and is never substituted")
    print("          geometry: min_target_distance=%.1f km  min_known_separation=%.1f km"
          "  detection=%.1f km  discovery_chain=OFF  strict=ON"
          % (cfg.min_target_distance_km, cfg.min_known_separation_km, DETECTION_KM))
    print("          stretch=%s  sams=%s" % (cfg.stretch_target_ratio, cfg.include_sams))
    # The ONE difficulty factor, echoed before compute is spent -- an operator who meant
    # to run the hard cell and typed the easy one sees it here, not in the results.
    print("difficulty (%s): fuel_damage mode=%s p(damaged)=%.2f "
          "leg_progress=%.2f rtb_margin=%.2f"
          % (_difficulty_factor_name(cfg), cfg.fuel_damage_mode,
             cfg.fuel_damage_probability, cfg.fuel_damage_leg_progress,
             cfg.fuel_damage_rtb_margin))
    if cfg.variable_severity:
        # The three-way distribution, stated as the numbers a reader will look for
        # rather than as the two conditionals it is stored as.
        p_damaged = float(cfg.fuel_damage_probability)
        p_mild = float(cfg.fuel_damage_mild_probability)
        print("          severity: p(mild|damaged)=%.2f -> clean %.2f / mild %.2f / "
              "severe %.2f. MILD leaves continuation+RTB feasible; SEVERE does not. "
              "The policy is told NEITHER -- only its own live fuel changes."
              % (p_mild, 1.0 - p_damaged, p_damaged * p_mild,
                 p_damaged * (1.0 - p_mild)))
    print("          reward aircraft_penalty_coeff=%.2f (graph_reward formula UNCHANGED); "
          "eval uses matched %s (%s) on the SAME seed"
          % (cfg.aircraft_penalty_coeff, cfg.eval_group_kind + "s",
             " / ".join("forced_%s" % c for c in cfg.reported_cells)))
    print("legacy split surface (NOT used by the construction path): "
          "num_red_airbases=%r partial_ratio=%s -> known/hidden = %s"
          % (cfg.num_red_airbases, cfg.partial_ratio,
             _format_split_preview(cfg.split_preview)))
    if artifacts_root is None:
        print("visual artifacts: DISABLED")
    else:
        print("visual artifacts: ENABLED for every scheduled pre_update / train / "
              "post_update attempt -> %s" % str(artifacts_root))
        print("          per attempt: the generated known-only scenario, the executed "
              "t=0 scenario, the BLADE playback and a manifest")
    print("run_dir=%s" % str(run_dir))
    print("config:  %s" % str(run_config_path))
    print("code:    %s%s"
          % (git_info["commit"] or "UNKNOWN",
             "  [DIRTY]" if git_info["dirty"] else ""))
    print("policy:  exact-cardinality failures = %s (a failed seed is skipped and "
          "accounted, never replaced)" % _EXACT_CARDINALITY_POLICY)
    print("=" * 78)

    def _eval_round(
        *, iteration: Optional[int], stage: str, updates: int, ordinal: int
    ) -> Dict[str, Any]:
        """ONE dispatch site for "run an evaluation round", so the three call sites below
        (pre-update, periodic and final) cannot drift into evaluating different
        populations. A ``generalized_v1`` run measures the FROZEN 18-stratum benchmark; a
        ``fixed_cell_v1`` run measures the historical held-out band, through exactly the
        call it always made. A ``generalized_v2`` run never reaches here at all --
        ``validate`` refuses evaluation on that design, which defines no evaluation
        construct."""
        common = dict(
            iteration=iteration, stage=stage, updates_completed=updates,
            round_ordinal=ordinal, failures_path=failures_path,
            outcomes_path=outcomes_path, artifacts_root=artifacts_root,
        )
        if benchmark is not None:
            return evaluate_benchmark(policy, gen, cfg, benchmark, **common)
        return evaluate(policy, gen, cfg, **common)

    train_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    last_eval_iteration = -1
    last_ckpt_iteration = -1
    # The ACTUAL last completed iteration, which finalization is keyed on. It is
    # `n_iterations - 1` on a run that spends its whole budget -- so nothing about the
    # fixed-budget path moves -- and the real stopping point on a run that ended early,
    # where `n_iterations - 1` would name an iteration that never ran.
    last_completed_iteration = -1
    # Which eval round is next, and therefore which SCENARIO-TAG namespace it writes into
    # (see `eval_episode_tag`). 0 is the pre-update round; every later round takes the
    # next ordinal so no round can overwrite an earlier round's scenario artifacts. It
    # names files only -- the held-out seeds are the same on every round.
    eval_round_ordinal = 0
    # Updates that actually ran epochs. This is the learning-curve x-axis: it is the
    # amount of LEARNING behind a measurement, which an iteration counter is not (a
    # zero-wake iteration completes but performs no gradient step).
    updates_completed = 0
    t_run = time.perf_counter()

    with open(train_records_path, "w", encoding="utf-8") as train_fh, \
            open(eval_records_path, "w", encoding="utf-8") as eval_fh:

        # ---- PRE-UPDATE held-out measurement of the INITIAL policy ----
        # Deliberately here: after build_policy/PPOUpdater, before the first training
        # episode, the first PPOBuffer insert and the first optimizer step. Anything
        # measured later is a trained policy, and a curve without this point has no
        # origin to be compared against.
        if cfg.eval_enabled:
            ev = _eval_round(iteration=None, stage=_EVAL_STAGE_PRE_UPDATE,
                             updates=0, ordinal=eval_round_ordinal)
            eval_round_ordinal += 1
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval PRE-UPDATE, updates_completed=0] mean=%s ok=%d/%d  %5.1fs"
                  % (_fmt_opt(ev["eval_reward_mean"]), ev["n_successful"],
                     ev["n_attempted"], ev["eval_seconds"]))
            _print_eval_pair_line(ev)

        # GENERALIZED-V1 Task 5C: the attempt policy, resolved ONCE for the whole run,
        # and the ONE monotone run-wide attempt ordinal it is expressed over. The
        # ordinal advances on EVERY training attempt of the run -- successful or failed,
        # in every iteration -- which is what makes a spent seed unrecoverable, a
        # replacement deterministic, and every training artifact tag unique.
        quota_policy = cfg.generalized
        quota = int(cfg.episodes_per_iteration)
        attempt_budget = int(cfg.max_attempts_per_iteration)
        global_attempt_ordinal = 0
        # GENERALIZED-V1 EARLY STOPPING, resolved ONCE for the whole run. `None` is the
        # fixed-budget path: no check is computed, no record key is added and the loop
        # cannot exit early. There is deliberately no `training_mode` branch here or
        # anywhere below it -- actor-only and CTDE stop by the identical rule.
        monitor = (
            _EarlyStoppingMonitor(
                min_iterations=cfg.early_stopping_min_iterations,
                window_iterations=cfg.early_stopping_window_iterations,
                patience_windows=cfg.early_stopping_patience_windows,
                min_delta=cfg.early_stopping_min_delta,
            )
            if cfg.early_stopping_enabled else None
        )

        for iteration in range(cfg.n_iterations):
            t_iter = time.perf_counter()
            # One buffer kind per training mode. They are separate classes rather than
            # one buffer with a flag because they hold different things: the actor-only
            # buffer stores PER-EGO chains (the Phase-A credit structure), the CTDE
            # buffer stores the episode's GLOBAL decision sequence beside its central
            # states.
            buf: Union[PPOBuffer, CTDEBuffer] = (
                CTDEBuffer() if cfg.ctde_enabled else PPOBuffer()
            )
            meta_counts = _empty_meta_counts()
            ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
            rewards: List[float] = []
            unique_confirmed: List[float] = []
            ticks: List[float] = []
            # The batch is tallied under the SAME cells evaluation reports, so a
            # training record and an eval record of one run split the damaged half the
            # same way and can be read side by side.
            tally = _ConditionTally(cfg.reported_cells)
            n_failed_iter = 0
            # THE ATTEMPT COUNT IS MEASURED, NOT ASSUMED (Task 5C). Under the historical
            # policy every scheduled slot is attempted exactly once, so this ends the
            # loop at `episodes_per_iteration` and the record is byte-identical; under
            # the quota policy it is the real number of attempts the iteration spent.
            n_attempted_iter = 0
            # How much learning stands behind the episodes collected BELOW -- they are
            # generated by the policy as it is now, before this iteration's update.
            updates_before = updates_completed
            # MEASUREMENT-ONLY tags per successful episode, keyed by (episode_index,
            # seed). A TRAINER-SIDE map: it never enters the buffer, a Transition, an
            # observation, the reward or the update, and is joined to credit rows only
            # after the update has produced them (`_credit_rows`).
            credit_tags: Dict[Tuple[int, int], Dict[str, Any]] = {}

            # ---- collect the batch ----
            t_eps = time.perf_counter()
            while True:
                # THE ONE PLACE THE TWO ATTEMPT POLICIES DIFFER (Task 5C).
                #
                # HISTORICAL (`scheduled_attempts_v1`): exactly `episodes_per_iteration`
                # attempts, whatever they produce. Byte-identical to the `for j in
                # range(...)` this replaced -- the loop still makes one attempt per slot,
                # in order, and a failure still simply loses its slot.
                #
                # QUOTA (`successful_quota_with_deterministic_replacement_v1`): keep
                # attempting until the batch holds `episodes_per_iteration` SUCCESSFUL
                # episodes, bounded by the operator's explicit attempt budget. Exhausting
                # that budget ABORTS -- it never updates on the partial batch, because a
                # PPO/CTDE batch whose size is a silent function of world attrition is
                # the exact coupling the quota exists to remove.
                if quota_policy:
                    if len(rewards) >= quota:
                        break
                    if n_attempted_iter >= attempt_budget:
                        raise TrainingQuotaError(
                            "iteration %d exhausted its attempt budget: %d attempt(s) "
                            "produced only %d of the %d SUCCESSFUL episode(s) "
                            "episodes_per_iteration requires (%d failed). The run stops "
                            "rather than updating on a partial batch -- no seed is "
                            "retried, no failure is reclassified, and no budget is "
                            "raised mid-run. Every attempt is recorded: inspect "
                            "episode_failures.jsonl, then either raise "
                            "generalized_max_attempts_per_iteration or investigate the "
                            "world-attrition rate."
                            % (iteration, n_attempted_iter, len(rewards), quota,
                               n_failed_iter)
                        )
                elif n_attempted_iter >= quota:
                    break

                # This attempt's position INSIDE the iteration. Under the historical
                # policy it is the scheduled slot; under the quota policy it is the
                # attempt index, which is what keeps a replacement's artifact identity
                # distinct from the failed attempt it replaces.
                j = n_attempted_iter
                if quota_policy:
                    # ONE monotone run-wide ordinal, advanced by EVERY attempt. A failed
                    # seed is therefore SPENT (nothing can revisit it) and a replacement
                    # is simply the next ordinal -- deterministic, and unique across the
                    # whole run, so no replacement can overwrite the artifacts of the
                    # attempt it replaced.
                    g = global_attempt_ordinal
                    seed = train_attempt_seed(cfg, g)
                else:
                    # THE HISTORICAL FORMULAS, still called. The run-wide counter is
                    # VERIFIED to agree with them rather than assumed to: on this path
                    # every slot is attempted exactly once, so the two must coincide, and
                    # a divergence would mean the seed schedule had silently moved.
                    g = global_episode_index(cfg, iteration, j)
                    seed = train_seed(cfg, iteration, j)
                    if g != global_attempt_ordinal:
                        raise MeasurementIntegrityError(
                            "training attempt ordinal drift at iteration %d slot %d: "
                            "the historical schedule says %d and the run-wide attempt "
                            "counter says %d. The seed schedule is the run's identity, "
                            "so this stops rather than measuring an unknown population."
                            % (iteration, j, g, global_attempt_ordinal)
                        )
                # SPENT BEFORE THE ATTEMPT IS MADE, so a `continue` out of the failure
                # handler below can never re-use this ordinal or this seed.
                n_attempted_iter += 1
                global_attempt_ordinal += 1
                # The SCHEDULED cell, resolved from the seed and the mode alone. Known
                # before the episode is built and still known if it never builds, which
                # is what lets a failure be accounted under its own cell. Under a legacy
                # mode the cell IS the condition; under `seeded_variable` a damaged
                # episode's cell is its severity.
                fd_params = cfg.fuel_damage_parameters()
                condition = resolve_condition(episode_seed=seed, params=fd_params)
                severity = resolve_severity(episode_seed=seed, params=fd_params)
                cell = str(severity) if severity else str(condition)
                tally.attempt(cell)
                # GENERALIZED-V1: this episode's world SHAPE, drawn from the sampler's
                # own rng domain and from the episode seed alone -- so it cannot move,
                # and cannot be moved by, the fuel-damage draws resolved just above, the
                # hidden-placement stream, or torch's action sampling. `None` on the
                # historical path, where `_run_one_episode` receives no such keyword at
                # all and resolves the configured fixed cell exactly as it always did.
                card = (
                    sample_generalized_cardinality(episode_seed=seed)
                    if cfg.generalized and not cfg.route_relative_population
                    else None
                )
                # GENERALIZED-V2: the STAGE-1 half of the same question. Drawn once, here,
                # and reused for the episode AND for the ledger entry if the attempt fails,
                # so the two can never describe different draws. `None` on every other
                # design, where `_pre_solve_kwargs` passes no keyword at all.
                pre_card = v2_pre_solve_cardinality(cfg, seed)
                # GENERALIZED-V2: a FRESH write-once carrier per attempt. Setup fills it
                # the instant stage 2 resolves, so this loop can state the population the
                # attempt really received even when `setup_episode` RAISES afterwards --
                # and "fresh per attempt" is what stops one attempt's identity being
                # attributed to the next. `None` on every other design, where
                # `_population_recorder_kwargs` passes no keyword at all.
                population_recorder = (
                    RouteRelativePopulationRecorder()
                    if cfg.route_relative_population else None
                )
                artifacts = None
                if artifacts_root is not None:
                    artifacts = _AttemptArtifacts(
                        root=artifacts_root,
                        identity=_AttemptIdentity(
                            phase=_ARTIFACT_PHASE_TRAIN,
                            iteration=int(iteration),
                            updates_completed=int(updates_before),
                            eval_round_ordinal=None,
                            eval_episode_index=None,
                            eval_pair_member=None,
                            attempt_ordinal=int(j),
                            episode_index=int(g),
                            seed=int(seed),
                            condition=str(condition),
                            severity=severity,
                            episode_tag=int(g),
                        ),
                    )
                # A FRESH recorder per attempt, on a CTDE run only. Per attempt because
                # its samples belong to exactly one episode's decision sequence, and a
                # reused one would splice two episodes' states into a single GAE chain.
                # `None` on an actor_only run -> no keyword is passed at all.
                central_recorder = (
                    CentralStateRecorder() if cfg.ctde_enabled else None
                )
                try:
                    out = _run_one_episode(
                        policy, gen, cfg,
                        seed=seed, episode_tag=g, deterministic=False,
                        **_artifact_kwargs(artifacts),
                        # Absent entirely unless this is a CTDE run (`_ctde_kwargs`).
                        **_ctde_kwargs(central_recorder),
                        # Absent entirely on the fixed-cell path (`_cardinality_kwargs`).
                        **_cardinality_kwargs(card),
                        # Absent entirely except on GENERALIZED-V2 (`_pre_solve_kwargs`).
                        **_pre_solve_kwargs(pre_card),
                        # Likewise absent except on GENERALIZED-V2: the write-once carrier
                        # that keeps this attempt's stage-2 draw readable after a failure.
                        **_population_recorder_kwargs(population_recorder),
                    )
                except (_VisualArtifactError, MeasurementIntegrityError,
                        FuelDamageIntegrityError, BenchmarkIdentityError,
                        ReferenceIntegrityError, MatchAouBackendError):
                    # INFRASTRUCTURE / DATA INTEGRITY, not science. Re-raised ahead of the
                    # broad handler so none can be written to the ledger as a
                    # `generation` / `setup` / `run` / `reward` failure, enter
                    # `skip_and_account_v1`, or shrink a scientific denominator by
                    # masquerading as an episode failure. The run stops. That routing is
                    # the long baseline's lesson: a roster defect accounted as a `setup`
                    # failure removed 143 training attempts in silence.
                    # `FuelDamageIntegrityError` (GENERALIZED-V1, handoff 3l.3) is routed
                    # identically: a CERTIFIED world that contradicts its own certificate
                    # is an instrument fault, never attrition.
                    raise
                except Exception as exc:  # never abort the run on one episode
                    # SKIP AND ACCOUNT: record it and move to the NEXT scheduled seed.
                    # This seed is spent -- no retry, no substitute, no shift of the
                    # band. `j` continues, so the schedule is untouched.
                    n_failed_iter += 1
                    tally.failure(cell)
                    _append_failure_record(failures_path, _failure_record(
                        phase="train",
                        evaluation_stage=None,
                        updates_completed=updates_before,
                        iteration=iteration,
                        attempt_ordinal=j,
                        episode_index=g,
                        eval_tag=None,
                        seed=seed,
                        # The ledger keeps naming the CONDITION, so
                        # `failures_by_condition` means what it always did; the finer
                        # cell and the scheduled world shape are added beside it so a
                        # per-cell and a per-cardinality denominator stay complete.
                        condition=condition,
                        cell=cell,
                        # The population identity this attempt ACTUALLY RECEIVED, which
                        # under GENERALIZED-V2 depends on how far it got: the stage-1
                        # half-cell if it died before the known-only solve produced a route
                        # count, and the RESOLVED cell -- built from the stage-2 draw the
                        # construction path really made -- if it died after. A failed
                        # attempt stays part of the attempted population, so its record
                        # must state the identity it ran under rather than the last one
                        # this loop happened to hold.
                        cardinality=_failure_cardinality(
                            card, pre_card,
                            None if population_recorder is None
                            else population_recorder.load),
                        # BOTH halves of the V2 identity travel with the record, so a
                        # Case-B entry still names the stage-1 draw behind its A and K.
                        pre_solve_cardinality=pre_card,
                        route_relative_load=(
                            None if population_recorder is None
                            else population_recorder.load),
                        exc=exc,
                    ))
                    print("  [iter %d ep %d] FAILED (seed=%d, cell=%s, stage=%s): %s: %s"
                          % (iteration, g, seed, cell,
                             getattr(exc, "stage", "unknown"),
                             type(getattr(exc, "original", exc)).__name__,
                             getattr(exc, "original", exc)))
                    traceback.print_exc()
                    continue

                # Printed BEFORE the next attempt starts: a batch of long episodes is
                # then readable as it runs, and a completed attempt is visibly distinct
                # from the FAILED line above.
                print(_format_episode_block(
                    "[train iter=%d ep=%d seed=%d]" % (iteration, g, seed), out
                ))

                # `cell` is this attempt's scheduled cell, resolved from the seed before
                # the episode was built and already counted by `tally.attempt(cell)`.
                # The guard runs FIRST, so a mismatched episode reaches neither the
                # durable outcome stream nor the PPO buffer below.
                tally.success(out, expected_cell=cell)
                _append_episode_outcome_record(outcomes_path, _episode_outcome_record(
                    out,
                    phase=_ARTIFACT_PHASE_TRAIN,
                    iteration=int(iteration),
                    updates_completed=int(updates_completed),
                    updates_completed_before=int(updates_before),
                    attempt_ordinal=int(j),
                    episode_index=int(g),
                    eval_round_ordinal=None,
                    eval_episode_index=None,
                    eval_group_member=None,
                    seed=int(seed),
                    episode_tag=int(g),
                    fuel_damage_mode=str(cfg.fuel_damage_mode),
                    design=cfg.design,
                ))
                # The SAME episode, recorded under the credit structure its training
                # mode uses. CTDE keeps the GLOBAL decision order beside the central
                # states captured during the run; `CTDEEpisodeRecord` validates the 1:1
                # alignment on construction, so a drifted capture seam fails LOUD here
                # rather than silently mispairing a value with a decision.
                if cfg.ctde_enabled:
                    buf.add(CTDEEpisodeRecord.from_episode(
                        out.trajectory,
                        central_recorder.samples if central_recorder else [],
                        out.reward, seed=seed, episode_index=g,
                    ))
                else:
                    buf.add(EpisodeRecord.from_trajectory(
                        out.trajectory, out.reward, seed=seed, episode_index=g,
                    ))
                credit_tags[(int(g), int(seed))] = _credit_measurement_tags(out)
                rewards.append(out.reward)
                unique_confirmed.append(float(out.targets_confirmed_unique))
                ticks.append(float(out.ticks))
                _add_meta_action_counts(meta_counts, out.trajectory)
                if out.ended in ended_counts:
                    ended_counts[out.ended] += 1
            episodes_seconds = time.perf_counter() - t_eps
            n_successful_iter = len(rewards)

            # ---- ONE update over the batch (empty batch -> documented no-op) ----
            t_upd = time.perf_counter()
            # The credit sink only COLLECTS the report the update hands it after its last
            # epoch; the rows are built and written below, outside the update.
            credit_reports: List[CreditReport] = []
            # OPT-IN gradient decomposition: only OPAQUE integer group ids cross into the
            # updater; the tags that resolve them stay here.
            gradient_reports: List[ActorGradientReport] = []
            gradient_kwargs: Dict[str, Any] = (
                {"gradient_group_ids": _actor_gradient_group_ids(buf.records, credit_tags),
                 "gradient_sink": gradient_reports.append}
                if gradient_path is not None else {}
            )
            diag = updater.update(buf, credit_sink=credit_reports.append,
                                  **gradient_kwargs)
            update_seconds = time.perf_counter() - t_upd
            _persist_credit_diagnostics(
                credit_path, credit_reports, diag,
                iteration=int(iteration),
                updates_completed_before=int(updates_before),
                measurement_tags=credit_tags,
            )
            if gradient_path is not None:
                _persist_actor_gradient_diagnostics(
                    gradient_path, gradient_reports, diag,
                    iteration=int(iteration),
                    updates_completed_before=int(updates_before),
                    measurement_tags=credit_tags,
                )
            buf.clear()
            if int(diag["n_epochs_run"]) > 0:
                updates_completed += 1

            # The batch mean over the SUCCESSFUL episodes -- or None when every
            # scheduled attempt failed. Never 0.0: 0 is the oracle optimum (see
            # `_stats_or_none`), and `diag["baseline"]` is defined as 0.0 on an empty
            # batch, which is exactly the value that must not reach a record.
            train_reward_mean = (
                float(diag["baseline"]) if n_successful_iter > 0 else None
            )
            # ONE arithmetic site behind BOTH the authoritative key and its legacy alias.
            unique_confirmed_mean = _stats_or_none(unique_confirmed)["mean"]

            # ---- the per-iteration SCALAR record (no per_epoch lists, no tensors) ----
            record = {
                "iteration": iteration,
                # --- attempt accounting: the AUTHORITATIVE names ---
                "n_attempted": n_attempted_iter,
                "n_successful": n_successful_iter,
                "n_failed": n_failed_iter,
                "success_fraction": _fraction(n_successful_iter, n_attempted_iter),
                "wake_fraction_of_successful": _fraction(
                    int(diag["episodes_with_wakes"]), n_successful_iter
                ),
                # --- where this measurement sits on the learning axis ---
                # `updates_completed_before` is the x of the TRAINING curve: it is how
                # many updates the policy that GENERATED these episodes had received.
                # It makes iteration 0 land at x=0, alongside the pre-update eval.
                "updates_completed_before": int(updates_before),
                "updates_completed": int(updates_completed),
                # --- compatibility names kept so pre-B4 readers still parse a record --
                "episodes_per_iteration": cfg.episodes_per_iteration,
                "n_failed_episodes": n_failed_iter,
                # GENERALIZED-V1 Task 5C: WHAT `episodes_per_iteration` COUNTED here, and
                # what the iteration really spent to obtain it. Present on BOTH designs
                # so a reader never has to infer the policy from an absence -- on the
                # fixed-cell path it truthfully states the historical contract, where the
                # quota IS the attempt count and no replacement exists.
                "training_attempt_policy": cfg.training_attempt_policy,
                "successful_episodes_required": quota,
                "max_attempts_per_iteration": attempt_budget,
                # The attempts spent BEYOND the quota. Under the quota policy a completed
                # iteration has `n_successful == quota`, so this equals `n_failed` by
                # construction; it is recorded under its own name because "how many
                # replacements did this iteration need" is the question the number
                # answers. Always 0 under the historical policy, which replaces nothing.
                "n_replacement_attempts": max(0, n_attempted_iter - quota),
                # The training learning-curve value IS diag["baseline"]: the mean
                # episode R over the iteration's SUCCESSFUL episodes, zero-wake episodes
                # included. Recorded from the update, never recomputed a second way.
                "train_reward_mean": train_reward_mean,
                "aggregates_over": "successful_episodes",
                "baseline": train_reward_mean,
                "policy_loss": float(diag["policy_loss"]),
                "total_loss": float(diag["total_loss"]),
                "entropy": float(diag["entropy"]),
                "mean_ratio": float(diag["mean_ratio"]),
                "clip_fraction": float(diag["clip_fraction"]),
                "approx_kl": float(diag["approx_kl"]),
                "max_ratio_dev": float(diag["max_ratio_dev"]),
                "grad_norm": float(diag["grad_norm"]),
                "adv_std_raw": float(diag["adv_std_raw"]),
                "n_transitions": int(diag["n_transitions"]),
                "n_episodes": int(diag["n_episodes"]),
                "episodes_with_wakes": int(diag["episodes_with_wakes"]),
                "n_epochs_run": int(diag["n_epochs_run"]),
                "meta_action_counts": dict(meta_counts),
                "meta_action_fractions": _meta_fractions(meta_counts),
                "ended_counts": dict(ended_counts),
                "reward_min": _stats_or_none(rewards)["min"],
                "reward_max": _stats_or_none(rewards)["max"],
                # AUTHORITATIVE: mean number of distinct TARGETS confirmed killed per
                # successful episode, deduplicated over ego.
                "targets_confirmed_unique_mean": unique_confirmed_mean,
                "target_confirmation_count_semantics":
                    _TARGET_CONFIRMATION_SEMANTICS,
                # ALIAS of the key above, not a second measurement. It used to average
                # `len(executor.done)` -- (ego, target) CONFIRMATIONS, which can exceed
                # the number of targets in the world.
                "kills_mean": unique_confirmed_mean,
                "ticks_mean": _stats_or_none(ticks)["mean"],
                # --- FD-BASELINE-v1: per-condition accounting + event counters ---
                # The scheduled mixture is deterministic per seed, so `n_clean_attempted`
                # + `n_damaged_attempted` == `n_attempted` by construction; the per-
                # condition means are over that condition's SUCCESSFUL episodes and are
                # None (never 0.0) when it had none.
                **tally.to_record(),
                "fuel_damage_mode": str(cfg.fuel_damage_mode),
                "aircraft_penalty_coeff": float(cfg.aircraft_penalty_coeff),
                "iteration_seconds": time.perf_counter() - t_iter,
                "episodes_seconds": episodes_seconds,
                "update_seconds": update_seconds,
            }
            # PHASE-B CTDE: the CRITIC's own four diagnostics, added ONLY on a `ctde`
            # run. `CTDEUpdater.update` already computed them; they are copied straight
            # out of `diag` and NEVER recomputed here, so the record cannot describe a
            # critic the update did not have. An `actor_only` record is byte-unchanged
            # (the keys are absent, not null) -- its updater has no critic to describe,
            # and a nullable key would invite reading "no critic" as "a critic that
            # scored 0". Every actor-side key above, `train_reward_mean` and `baseline`
            # included, keeps exactly its existing meaning in both modes.
            if cfg.ctde_enabled:
                record.update({
                    "value_loss": float(diag["value_loss"]),
                    "value_mean": float(diag["value_mean"]),
                    "value_target_mean": float(diag["value_target_mean"]),
                    "critic_grad_norm": float(diag["critic_grad_norm"]),
                })
            # ---- EARLY STOPPING: decided here, from TRAINING reward alone ----
            # DELIBERATELY at this point and no other. The record is complete, so the
            # decision reads the SAME `train_reward_mean` the artifact persists (one
            # metric path); and this is still BEFORE the periodic evaluation branch
            # below, so nothing measured on the frozen benchmark -- the comparator the
            # two arms are judged by -- can reach the decision that ends an arm's
            # training. A due check is attached to the record it was computed from, so
            # every decision is reconstructable from `train_records.jsonl` alone.
            stop_early = False
            early_stopping_check: Optional[Dict[str, Any]] = None
            if monitor is not None:
                early_stopping_check = monitor.observe(
                    completed_iterations=iteration + 1,
                    train_reward_mean=record["train_reward_mean"],
                )
                if early_stopping_check is not None:
                    record[_EARLY_STOPPING_RECORD_KEY] = early_stopping_check
                    stop_early = bool(early_stopping_check["stop_triggered"])

            train_records.append(record)
            train_fh.write(json.dumps(record) + "\n")
            train_fh.flush()
            # The iteration is now COMPLETE and durably recorded, whatever happens next.
            last_completed_iteration = iteration

            # Exactly ONE of the three states, never two at once: an all-failed batch
            # used to print both flags and so read as "episodes ran, nobody woke".
            outcome = _iteration_outcome(record)
            if outcome == "all_failed":
                flag = ("  [ALL %d ATTEMPTS FAILED: no episode completed, nothing "
                        "measured]" % record["n_attempted"])
            elif outcome == "zero_wake":
                flag = "  [ZERO-WAKE: episodes ran, no ego woke; update skipped]"
            else:
                flag = ""
            print("[iter %3d] R=%s ok=%d/%d trans=%3d wake_eps=%d/%d loss=%+.4f "
                  "ent=%.3f kl=%+.4f clip=%.2f gn=%.3f  %5.1fs%s"
                  % (iteration, _fmt_opt(record["train_reward_mean"]),
                     record["n_successful"], record["n_attempted"],
                     record["n_transitions"], record["episodes_with_wakes"],
                     record["n_episodes"],
                     record["total_loss"], record["entropy"], record["approx_kl"],
                     record["clip_fraction"], record["grad_norm"],
                     record["iteration_seconds"], flag))
            # The difficulty factor's own line: the batch's split by CELL, each cell's
            # conditional mean, and whether the scheduled events actually happened. For a
            # legacy run the cells are clean/damaged and this is the line it always was;
            # for a variable-severity run the damaged half is shown as mild and severe,
            # which is the split the run exists to measure.
            print("           fd: %s | applied=%d wakes=%d rtb=%d dead=%d"
                  % (" | ".join(
                         "%s %d/%d R=%s"
                         % (cell, record["n_%s_successful" % cell],
                            record["n_%s_attempted" % cell],
                            _fmt_opt(record["reward_mean_%s" % cell]))
                         for cell in cfg.reported_cells),
                     record["fuel_damage_events_applied"],
                     record["fuel_damage_wakes"], record["fuel_damage_rtb_issued"],
                     record["deaths"]))

            # ---- EARLY STOP: leave BEFORE this boundary's periodic eval/checkpoint --
            # Exiting here rather than after the branch below is what makes the final
            # evaluation strictly POST-DECISION, and it is also what keeps the
            # finalization single: a stopping boundary that is also an eval/checkpoint
            # boundary produces exactly one final evaluation and one final checkpoint,
            # never a periodic pair plus a duplicate final pair.
            if stop_early:
                print("  [EARLY STOP @iter %d, %d completed iteration(s)] %s: "
                      "window_mean=%s vs best=%s over %d stale window(s) (min_delta=%s). "
                      "The MAXIMUM budget was %d iteration(s). This records that the "
                      "configured plateau rule fired -- it is NOT a convergence claim."
                      % (iteration, iteration + 1,
                         EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU,
                         _fmt_opt(early_stopping_check["window_mean"]),
                         _fmt_opt(early_stopping_check["best_window_mean_before"]),
                         early_stopping_check["stale_windows"],
                         early_stopping_check["min_delta"], cfg.n_iterations))
                break

            # ---- periodic eval ----
            if cfg.eval_enabled and ((iteration + 1) % cfg.eval_every == 0):
                ev = _eval_round(iteration=iteration,
                                 stage=_EVAL_STAGE_POST_UPDATE,
                                 updates=updates_completed,
                                 ordinal=eval_round_ordinal)
                eval_round_ordinal += 1
                eval_records.append(ev)
                eval_fh.write(json.dumps(ev) + "\n")
                eval_fh.flush()
                last_eval_iteration = iteration
                print("  [eval @iter %d, updates=%d] mean=%s min=%s max=%s "
                      "targets_unique=%s ok=%d/%d  %5.1fs"
                      % (iteration, ev["updates_completed"],
                         _fmt_opt(ev["eval_reward_mean"]),
                         _fmt_opt(ev["eval_reward_min"]),
                         _fmt_opt(ev["eval_reward_max"]),
                         _fmt_opt(ev["eval_targets_confirmed_unique_mean"], "%.1f"),
                         ev["n_successful"], ev["n_attempted"], ev["eval_seconds"]))
                _print_eval_pair_line(ev)

            # ---- periodic checkpoint ----
            if cfg.checkpoint_every > 0 and ((iteration + 1) % cfg.checkpoint_every == 0):
                path = save_checkpoint(
                    policy, updater, iteration, ckpt_dir, critic=critic
                )
                last_ckpt_iteration = iteration
                print("  [ckpt @iter %d] %s" % (iteration, path.name))

        # ---- final eval + final checkpoint (skipped if this iteration just did one) ----
        # THE ACTUAL LAST COMPLETED ITERATION, not `n_iterations - 1`. On a run that
        # spent its whole budget the two are the same number and this is byte-unchanged;
        # on an early-stopped run `n_iterations - 1` would name an iteration that never
        # ran, so the final evaluation and the final checkpoint would both be labelled
        # with a point the policy never reached. The fallback is unreachable
        # (`validate` requires `n_iterations >= 1`, so the loop body always runs at least
        # once) and exists only so finalization can never address a negative iteration.
        final_iteration = (
            last_completed_iteration if last_completed_iteration >= 0
            else cfg.n_iterations - 1
        )
        if cfg.eval_enabled and last_eval_iteration != final_iteration:
            ev = _eval_round(iteration=final_iteration,
                             stage=_EVAL_STAGE_POST_UPDATE,
                             updates=updates_completed,
                             ordinal=eval_round_ordinal)
            eval_round_ordinal += 1
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval @iter %d, final, updates=%d] mean=%s ok=%d/%d  %5.1fs"
                  % (final_iteration, ev["updates_completed"],
                     _fmt_opt(ev["eval_reward_mean"]), ev["n_successful"],
                     ev["n_attempted"], ev["eval_seconds"]))
            _print_eval_pair_line(ev)

    if last_ckpt_iteration != final_iteration:
        # SAVE-only, exactly as before (`save_checkpoint` has no loader and this task
        # added none) -- and at the iteration the run really reached.
        path = save_checkpoint(
            policy, updater, final_iteration, ckpt_dir, critic=critic
        )
        print("  [ckpt @iter %d, final] %s" % (final_iteration, path.name))

    # Re-read the jsonl artifacts rather than summarizing the in-memory lists: the
    # summary must be DERIVED from what was durably recorded, so it cannot describe a
    # run the files do not. This also makes `build_run_summary` usable on any run
    # directory, which is what the tests exercise.
    summary = build_run_summary(run_dir, cfg=cfg,
                                run_seconds=time.perf_counter() - t_run)
    summary_path = write_run_summary(run_dir, summary)
    print("  [summary] %s" % str(summary_path))
    _print_summary(summary)
    return summary


# =============================================================================
# 8. Aggregate + print
# =============================================================================

def _iteration_outcome(record: Dict[str, Any]) -> str:
    """Classify one training iteration: ``all_failed`` / ``zero_wake`` / ``productive``.

    THE THREE STATES ARE DISJOINT, and keeping them so is the point of this function.
    They were previously conflated because both an all-failed batch and a zero-wake
    batch end with ``n_epochs_run == 0``, so an iteration in which NO episode completed
    was being counted as "zero-wake" -- a claim that episodes ran and nobody sensed
    anything. They are opposite findings:

      * ``all_failed``  -- not one scheduled attempt produced an episode. Nothing was
        measured; this is a DATA-YIELD failure (on this cell, an exact-cardinality
        construction failure).
      * ``zero_wake``   -- episodes really ran and really finished, and no ego ever
        woke. That is a legitimate outcome of the event-triggered design and a
        statement about the POLICY's world, not about the pipeline.
      * ``productive``  -- at least one successful episode carried at least one wake.

    Judged on episode counts rather than on ``n_epochs_run`` so the classification is a
    property of the collected batch, independent of what the updater did with it.
    Pre-B4 records fall back to their old field names.
    """
    attempted = int(
        record.get("n_attempted", record.get("episodes_per_iteration", 0)) or 0
    )
    successful = int(record.get("n_successful", record.get("n_episodes", 0)) or 0)
    wake_bearing = int(record.get("episodes_with_wakes", 0) or 0)
    if attempted > 0 and successful == 0:
        return "all_failed"
    if successful > 0 and wake_bearing == 0:
        return "zero_wake"
    return "productive"


def _count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    """Group-count records by one string field (missing -> ``"unknown"``)."""
    out: Dict[str, int] = {}
    for rec in records:
        name = str(rec.get(key) or "unknown")
        out[name] = out.get(name, 0) + 1
    return out


def _sum_field(records: List[Dict[str, Any]], key: str, fallback: str) -> int:
    """Sum an integer field over records, falling back to a pre-B4 field name."""
    return sum(int(rec.get(key, rec.get(fallback, 0)) or 0) for rec in records)


# The COMPLETE identity a final-evaluation selection is allowed to rely on. All three
# are required: the stage says which kind of round it was, `updates_completed` says which
# policy state it measured, and the ordinal says which round it was. Two of the three do
# not identify a round -- two rounds can share an update count -- and none of them can be
# inferred from a record's POSITION in a file, which is a fact about the writer.
_FINAL_EVAL_IDENTITY_FIELDS = (
    "evaluation_stage", "updates_completed", "eval_round_ordinal",
)


# The ONE refusal reason for a final-evaluation identity that is not complete. Shared by
# `_select_final_eval_record` (over the eval RECORDS) and `_select_final_matched_round`
# (over the digest), so "incomplete" cannot come to mean two different things.
_INCOMPLETE_FINAL_EVAL_IDENTITY = "incomplete_evaluation_identity"


def _final_eval_identity(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The complete round identity of ONE eval record, or ``None`` if incomplete.

    Strict on TYPE as well as presence: a stage outside :data:`_EVAL_PHASES` is not an
    evaluation round, and a ``bool`` is rejected where an ``int`` is required (``True``
    would otherwise pass as update count 1). Returning ``None`` rather than a partially
    filled dict is what lets the caller refuse instead of ordering on a guess.
    """
    stage = record.get("evaluation_stage")
    updates = record.get("updates_completed")
    ordinal = record.get("eval_round_ordinal")
    if not isinstance(stage, str) or stage not in _EVAL_PHASES:
        return None
    if not isinstance(updates, int) or isinstance(updates, bool):
        return None
    if not isinstance(ordinal, int) or isinstance(ordinal, bool):
        return None
    return {
        "evaluation_stage": str(stage),
        "updates_completed": int(updates),
        "eval_round_ordinal": int(ordinal),
    }


def _select_final_eval_record(
    eval_records: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Select THE final evaluation round SEMANTICALLY. Never ``eval_records[-1]``.

    ``eval_records[-1]`` is the last row a file happens to hold. That is a property of
    the writer, not of the run: re-reading the same rounds in another order, or a reader
    that concatenates two artifacts, would silently relabel a different round as "final"
    and every ``final_eval_*`` number in the summary with it. Nothing about such a
    summary would look wrong.

    The round is therefore chosen by the highest ``eval_round_ordinal`` -- the run's own
    monotone round counter -- with ``updates_completed`` used as a CROSS-CHECK rather
    than as a second ordering: the selected round must also hold the maximum update
    count, because a run whose latest round measured an earlier policy state is an
    artifact whose two orderings contradict each other.

    It REFUSES, with an explicit reason, rather than guessing:

    * ``no_evaluation_records`` -- nothing to select from (evaluation disabled, or none
      completed). Not an error, and not an empty round either.
    * ``incomplete_evaluation_identity`` -- some record does not state all three of
      :data:`_FINAL_EVAL_IDENTITY_FIELDS`. An unorderable row in the population makes
      "the last round" unanswerable, so the answer is refused rather than taken from the
      rows that happen to be well-formed.
    * ``ambiguous_identity_duplicate_rounds`` -- two records claim the SAME identity.
    * ``ambiguous_highest_round_ordinal`` -- two records share the highest ordinal, so
      the tie could only be broken by position.
    * ``contradictory_ordering_ordinal_vs_updates_completed`` -- the highest-ordinal
      round does not carry the highest update count.

    Returns ``(record_or_None, selection_record)``. The selection record is PERSISTED in
    the summary, so a refusal is visible as a stated reason rather than as a field that
    is quietly ``None``.
    """
    rows = list(eval_records or ())
    if not rows:
        return None, {"selected": False, "reason": "no_evaluation_records",
                      "n_eval_records": 0}

    idents = [_final_eval_identity(r) for r in rows]
    n_missing = sum(1 for i in idents if i is None)
    if n_missing:
        return None, {
            "selected": False,
            "reason": _INCOMPLETE_FINAL_EVAL_IDENTITY,
            "required_fields": list(_FINAL_EVAL_IDENTITY_FIELDS),
            "n_eval_records": len(rows),
            "n_records_missing_identity": n_missing,
        }

    keys = [(i["evaluation_stage"], i["updates_completed"], i["eval_round_ordinal"])
            for i in idents]
    n_duplicated = len(keys) - len(set(keys))
    if n_duplicated:
        return None, {
            "selected": False,
            "reason": "ambiguous_identity_duplicate_rounds",
            "n_eval_records": len(rows),
            "n_duplicate_identity_records": n_duplicated,
        }

    max_ordinal = max(i["eval_round_ordinal"] for i in idents)
    finalists = [n for n, i in enumerate(idents)
                 if i["eval_round_ordinal"] == max_ordinal]
    if len(finalists) != 1:
        return None, {
            "selected": False,
            "reason": "ambiguous_highest_round_ordinal",
            "n_eval_records": len(rows),
            "highest_round_ordinal": max_ordinal,
            "n_candidates": len(finalists),
        }

    best = finalists[0]
    max_updates = max(i["updates_completed"] for i in idents)
    if idents[best]["updates_completed"] != max_updates:
        return None, {
            "selected": False,
            "reason": "contradictory_ordering_ordinal_vs_updates_completed",
            "n_eval_records": len(rows),
            "highest_round_ordinal_identity": idents[best],
            "max_updates_completed": max_updates,
        }

    return rows[best], {
        "selected": True,
        "reason": "selected_by_validated_identity",
        "ordered_by": ("highest eval_round_ordinal, cross-checked against "
                       "updates_completed"),
        "n_eval_records": len(rows),
        "identity": idents[best],
    }


def _eval_digest(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The reportable core of one eval round -- ALWAYS carrying its denominator.

    A held-out mean is not interpretable without the population it was taken over: a
    "-0.12" over 2 of 8 feasible seeds is a different claim from the same number over
    8 of 8. The two are therefore inseparable in every place this appears.
    """
    if record is None:
        return None
    return {
        "evaluation_stage": record.get("evaluation_stage"),
        "updates_completed": record.get("updates_completed"),
        # The round's own ordinal, so this digest states a COMPLETE round identity --
        # stage + update count + ordinal. Without it a consumer can only find the round
        # again by position in the file, which is a fact about the writer.
        "eval_round_ordinal": record.get("eval_round_ordinal"),
        "iteration": record.get("iteration"),
        "n_attempted": record.get("n_attempted", record.get("n_episodes")),
        "n_successful": record.get("n_successful", record.get("n_ok")),
        "n_failed": record.get("n_failed"),
        "success_fraction": record.get("success_fraction"),
        "eval_reward_mean": record.get("eval_reward_mean"),
        "eval_reward_min": record.get("eval_reward_min"),
        "eval_reward_max": record.get("eval_reward_max"),
        "aggregates_over": "successful_episodes",
        # FD-BASELINE-v1: the conditional means and the paired delta travel WITH their
        # own denominator, for the same reason the round mean does.
        "eval_reward_mean_clean": record.get("eval_reward_mean_clean"),
        "eval_reward_mean_damaged": record.get("eval_reward_mean_damaged"),
        "n_clean_successful": record.get("eval_n_clean_successful"),
        "n_damaged_successful": record.get("eval_n_damaged_successful"),
        "eval_paired_reward_delta": record.get("eval_paired_reward_delta"),
        "n_pairs_successful": record.get("n_pairs_successful"),
        "n_pairs_attempted": record.get("n_pairs_attempted"),
        "paired_delta_over": record.get("paired_delta_over"),
        # --- the matched GROUP, generically: every cell mean and every declared delta,
        # each with its own denominator. Under a legacy PAIR round these repeat what the
        # keys above say; under a TRIAD they are the only complete statement of it.
        "eval_group_kind": record.get("eval_group_kind"),
        "eval_group_cells": record.get("eval_group_cells"),
        "n_groups_successful": record.get("n_groups_successful"),
        "n_groups_attempted": record.get("n_groups_attempted"),
        "eval_delta_over": record.get("eval_delta_over"),
        "cell_reward_means": {
            cell: record.get("eval_reward_mean_%s" % cell)
            for cell in (record.get("eval_group_cells") or list(CONDITIONS))
        },
        "cell_successful": {
            cell: record.get("eval_n_%s_successful" % cell)
            for cell in (record.get("eval_group_cells") or list(CONDITIONS))
        },
        "group_deltas": {
            key: record.get(key) for key in (record.get("eval_delta_keys") or [])
        },
        # THE PRIMARY BEHAVIOURAL MEASUREMENT: what the FD wake chose, per damaged cell,
        # over the FD wakes that actually happened in it.
        "fd_wake_meta_action_counts": {
            cell: record.get("eval_fd_meta_action_counts_%s" % cell)
            for cell in (record.get("eval_group_cells") or list(CONDITIONS))
            if record.get("eval_fd_meta_action_counts_%s" % cell) is not None
        },
        "fd_wakes_by_cell": {
            cell: record.get("eval_n_%s_fd_wakes" % cell)
            for cell in (record.get("eval_group_cells") or list(CONDITIONS))
            if record.get("eval_n_%s_fd_wakes" % cell) is not None
        },
    }


def _wake_population_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split ONE already-selected population of outcome records by wake kind and cell.

    It takes the population as an ARGUMENT and selects nothing itself, so the caller --
    and only the caller -- decides which phase, stage or round the block describes. That
    is deliberate: the defect this replaces was a block that quietly pooled training and
    held-out evaluation rows under a name that mentioned neither.
    """
    by_kind: Dict[str, List[Dict[str, Any]]] = {k: [] for k in WAKE_KINDS}
    by_kind_cell: Dict[str, Dict[str, List[Dict[str, Any]]]] = {k: {} for k in WAKE_KINDS}
    n_by_kind: Dict[str, int] = {k: 0 for k in WAKE_KINDS}
    for rec in rows:
        cell = str(rec.get("cell", CONDITION_CLEAN))
        for d in rec.get("wake_decisions") or ():
            kind = str(d.get("wake_kind") or WAKE_KIND_ORDINARY)
            by_kind.setdefault(kind, []).append(d)
            by_kind_cell.setdefault(kind, {}).setdefault(cell, []).append(d)
            n_by_kind[kind] = n_by_kind.get(kind, 0) + 1
    return {
        "n_episode_records": len(rows),
        "n_wakes_by_kind": {k: n_by_kind.get(k, 0) for k in by_kind},
        "by_wake_kind": {k: _wake_diag_digest(v) for k, v in by_kind.items()},
        "by_wake_kind_and_cell": {
            k: {c: _wake_diag_digest(v) for c, v in cells.items()}
            for k, cells in by_kind_cell.items()},
    }


def _round_identity(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The FULL evaluation identity of one outcome record, as plain builtins.

    A benchmark ``group_key`` alone is NOT an identity: the SAME frozen world group is
    re-measured in EVERY evaluation round, so keying a matched table on it collapses
    every round of a run into one bucket and silently turns repeated measures of one
    world into what reads like independent worlds. The identity is therefore the
    evaluation STAGE, the update count, the round ordinal and the manifest the round was
    drawn from; the group key identifies the WORLD *within* that round.
    """
    return {
        "evaluation_stage": (None if rec.get("phase") is None else str(rec.get("phase"))),
        "updates_completed": (None if rec.get("updates_completed") is None
                              else int(rec.get("updates_completed"))),
        "eval_round_ordinal": (None if rec.get("eval_round_ordinal") is None
                               else int(rec.get("eval_round_ordinal"))),
        "benchmark_manifest_id": (None if rec.get("benchmark_manifest_id") is None
                                  else str(rec.get("benchmark_manifest_id"))),
    }


def _immediate_fd_abort_mass(rec: Dict[str, Any]) -> Optional[float]:
    """Mean P(ABORT) over this record's IMMEDIATE-FD wakes, or ``None``.

    Each wake is read under its OWN representation (:func:`_wake_meta_probability`): the
    semantic ABORT leaf, or the historical aggregate abort mass. ``None`` -- never
    ``0.0`` -- when the record has no immediate-FD wake at all: on a probability 0 is a
    measured value and would read as "the actor put no weight on aborting" when the
    truth is "this episode was never asked".
    """
    mass = [p for p in (_wake_meta_probability(d, _ABORT_NAME)
                        for d in rec.get("wake_decisions") or ()
                        if str(d.get("wake_kind") or "") == WAKE_KIND_IMMEDIATE_FD)
            if p is not None]
    return (float(sum(mass)) / len(mass)) if mass else None


def _matched_rounds(eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-ROUND matched severe-minus-mild aggregate P(ABORT). EVALUATION rows only.

    Pairing uses the FULL evaluation identity (:func:`_round_identity`) plus
    ``benchmark_group_key``: a group contributes a delta only when its MILD and SEVERE
    members are present IN THE SAME ROUND. Two rounds that measure the same frozen world
    therefore produce two independent deltas rather than one mixed number, and the order
    the records happen to appear in the file cannot change either of them.
    """
    rounds: Dict[Any, Dict[str, Any]] = {}
    for rec in eval_rows:
        gkey = rec.get("benchmark_group_key")
        if not gkey:
            continue
        mass = _immediate_fd_abort_mass(rec)
        if mass is None:
            continue
        ident = _round_identity(rec)
        key = (ident["evaluation_stage"], ident["updates_completed"],
               ident["eval_round_ordinal"], ident["benchmark_manifest_id"])
        slot = rounds.setdefault(key, {"identity": ident, "groups": {}})
        slot["groups"].setdefault(str(gkey), {})[
            str(rec.get("cell", ""))] = mass

    out: List[Dict[str, Any]] = []
    for key in sorted(rounds, key=lambda t: tuple(
            (v is None, v) for v in t)):
        slot = rounds[key]
        deltas = [v[SEVERITY_SEVERE] - v[SEVERITY_MILD]
                  for v in slot["groups"].values()
                  if SEVERITY_MILD in v and SEVERITY_SEVERE in v]
        rec_out = dict(slot["identity"])
        rec_out.update({
            "n_groups_with_immediate_fd_wakes": len(slot["groups"]),
            "n": len(deltas),
            "mean": (float(sum(deltas)) / len(deltas)) if deltas else None,
            "min": (min(deltas) if deltas else None),
            "max": (max(deltas) if deltas else None),
        })
        out.append(rec_out)
    return out


def _select_final_matched_round(
    rounds: List[Dict[str, Any]],
    final_eval: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Pick the round the FINAL evaluation digest names, BY COMPLETE VALIDATED IDENTITY.

    Never by file order. ``eval_records[-1]`` is the last row a file happens to hold,
    which is a statement about the writer, not about which round is final. So the round
    is matched against the digest's own identity -- and against ALL THREE of
    :data:`_FINAL_EVAL_IDENTITY_FIELDS`, never a subset of whichever ones happen to be
    present.

    THE SUBSET IS THE DANGEROUS CASE, which is why it is refused rather than tolerated.
    A digest missing its ordinal still carries a stage and an update count, and those two
    very often single out exactly one round -- so a subset match SUCCEEDS, returns a
    round, and produces a "final round" that was selected on an identity nobody
    validated. That is indistinguishable from a correct selection in the artifact. A
    partial identity therefore refuses with :data:`_INCOMPLETE_FINAL_EVAL_IDENTITY`
    even when the remaining fields would have been unique.

    :func:`_final_eval_identity` is the SINGLE validator, shared with
    :func:`_select_final_eval_record`: one definition of "a complete round identity",
    one set of type rules (stage in :data:`_EVAL_PHASES`, ``int`` update count and
    ordinal, ``bool`` rejected), so the two selectors cannot come to disagree about what
    a valid identity is.
    """
    if not final_eval:
        return None, {"selected": False, "reason": "no_final_evaluation_digest_provided"}
    requested = {k: final_eval.get(k) for k in _FINAL_EVAL_IDENTITY_FIELDS}
    identity = _final_eval_identity(final_eval)
    if identity is None:
        return None, {
            "selected": False,
            "reason": _INCOMPLETE_FINAL_EVAL_IDENTITY,
            "required_fields": list(_FINAL_EVAL_IDENTITY_FIELDS),
            "requested_identity": requested,
        }
    # ALL THREE fields, always. `identity` is already normalized by the validator, so
    # this compares like for like without re-coercing anything here.
    hits = [r for r in rounds
            if all(r.get(k) == identity[k] for k in _FINAL_EVAL_IDENTITY_FIELDS)]
    if len(hits) == 1:
        return hits[0], {"selected": True, "reason": "matched_by_validated_identity",
                         "requested_identity": requested,
                         "matched_identity": identity, "n_candidates": 1}
    return None, {
        "selected": False,
        "reason": ("no_matched_round_with_that_identity" if not hits
                   else "ambiguous_identity_matched_several_rounds"),
        "requested_identity": requested,
        "n_candidates": len(hits),
    }


def _fd_policy_sensitivity_from_outcomes(
    outcome_records: List[Dict[str, Any]],
    *,
    final_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The FD-policy-sensitivity digest, DERIVED FROM THE DURABLE PER-ATTEMPT STREAM.

    ONE metric path, exactly as :func:`_severity_response_from_outcomes` already is: the
    summary reads ``episode_outcomes.jsonl`` rather than a parallel in-memory aggregate,
    so it cannot describe a run its own artifacts do not.

    **THREE POPULATIONS, KEPT APART BY PHASE.** ``train``, ``pre_update`` and
    ``post_update`` each get their OWN block under ``by_phase`` and are never averaged
    together. The distinction is not cosmetic: training episodes are drawn by a
    stochastic actor from a sampled population, held-out evaluation is deterministic on a
    frozen one, and a rate whose denominator mixes the two describes neither. A pooled
    view is still available, but only under ``all_phases_pooled`` -- a name that states
    what it did.

    **THREE WAKE KINDS, ALSO KEPT APART** -- ``ordinary``, ``immediate_fuel_damage`` and
    ``post_fd_boundary`` -- because an approved measurement is reported over the
    immediate-FD population alone, and a boundary decision folded into it would change
    what that measurement means. Within each, the digest is additionally split by the
    episode's reporting CELL (clean / mild / severe, or clean / damaged on a legacy run).

    **MATCHED PAIRING USES THE FULL EVALUATION IDENTITY**, never ``benchmark_group_key``
    alone and never a target uuid: the same frozen world group is re-measured every
    round, so the round is part of what identifies a measurement. Deltas are reported
    PER ROUND, the cross-round pool is explicitly flagged as repeated measures, and the
    final round is selected by validated identity rather than by file order.

    **SCHEMA VERSIONS ARE OBSERVED, NOT ASSERTED.** The version reported is the one the
    records actually carry, so a legacy artifact cannot be described by this writer's
    current constants.

    Returns ``{"recorded": False, ...}`` when no record carries the v3 field -- a
    truthful "not recorded", never a table of fabricated zeros.
    """
    rows = list(outcome_records or [])
    with_diag = [r for r in rows if isinstance(r.get("wake_decisions"), list)]
    observed_versions = sorted({
        int(r["wake_diagnostics_schema_version"]) for r in with_diag
        if isinstance(r.get("wake_diagnostics_schema_version"), int)
        and not isinstance(r.get("wake_diagnostics_schema_version"), bool)})
    if not with_diag:
        return {
            "recorded": False,
            "source": _EPISODE_OUTCOMES_FILENAME,
            "wake_diagnostics_schema_version_writer": _WAKE_DIAGNOSTICS_VERSION,
            "wake_diagnostics_schema_versions_observed": [],
            "wake_diagnostics_schema_version_observed": None,
            "note": ("no episode_outcomes record carries `wake_decisions`; this run "
                     "predates episode-outcome schema v3"),
        }

    by_phase: Dict[str, Dict[str, Any]] = {}
    phases_seen = list(_ARTIFACT_PHASES) + sorted(
        {str(r.get("phase")) for r in with_diag
         if str(r.get("phase")) not in _ARTIFACT_PHASES})
    for phase in phases_seen:
        by_phase[phase] = _wake_population_block(
            [r for r in with_diag if str(r.get("phase")) == phase])

    eval_rows = [r for r in with_diag if str(r.get("phase")) in _EVAL_PHASES]
    rounds = _matched_rounds(eval_rows)
    final_round, selection = _select_final_matched_round(rounds, final_eval)
    pooled = [d for r in rounds for d in ([r["mean"]] if r["n"] else [])]
    all_deltas_n = sum(int(r["n"]) for r in rounds)

    return {
        "recorded": True,
        "source": _EPISODE_OUTCOMES_FILENAME,
        "wake_diagnostics_schema_version_writer": _WAKE_DIAGNOSTICS_VERSION,
        "wake_diagnostics_schema_versions_observed": observed_versions,
        "wake_diagnostics_schema_version_observed": (
            observed_versions[0] if len(observed_versions) == 1 else None),
        "n_episode_records_with_diagnostics": len(with_diag),
        "phases": list(_ARTIFACT_PHASES),
        "evaluation_phases": list(_EVAL_PHASES),
        "populations_note": (
            "train / pre_update / post_update are SEPARATE populations and are never "
            "averaged together; `all_phases_pooled` is the only pooled view and says so "
            "in its own name"),
        "by_phase": by_phase,
        "all_phases_pooled": {
            "note": ("TRAINING AND HELD-OUT EVALUATION POOLED -- a mixed denominator, "
                     "kept only as a coarse run-wide count. No scientific severity "
                     "claim is made from it; use `by_phase` instead"),
            **_wake_population_block(with_diag),
        },
        "matched_severe_minus_mild_aggregate_p_abort": {
            "population": "evaluation records only (%s)" % ", ".join(_EVAL_PHASES),
            "pairing": ("full evaluation identity (evaluation stage, updates_completed, "
                        "eval round ordinal, benchmark manifest id) + "
                        "benchmark_group_key"),
            "over": "world_groups_with_both_mild_and_severe_present_in_the_same_round",
            "n_rounds": len(rounds),
            "by_round": rounds,
            "pooled_across_rounds": {
                "n_rounds_with_deltas": len(pooled),
                "n_group_deltas": all_deltas_n,
                "mean_of_round_means": (
                    (float(sum(pooled)) / len(pooled)) if pooled else None),
                "min_round_mean": (min(pooled) if pooled else None),
                "max_round_mean": (max(pooled) if pooled else None),
                "totals_across_rounds_are_repeated_measures": True,
            },
            "final_round": final_round,
            "final_round_selection": selection,
        },
        "metric_semantics": {
            "selected_abort_fraction":
                "fraction of wakes whose SELECTED meta-action was SELF_PRESERVATION_ABORT",
            "p_abort_mean":
                "mean P(ABORT), each wake read under its own action representation: the "
                "one semantic ABORT leaf (%s), or the aggregate mass over the "
                "node-indexed abort cells (historical records). None over a population "
                "that mixes representations" % ACTION_REPRESENTATION_ID,
            "entropy_normalized_mean":
                "policy entropy divided by log(valid action count), per representation "
                "(semantic leaves, or historical joint cells); undefined (and excluded) "
                "when fewer than two actions are valid",
            "selected_joint_cell_abort_fraction":
                "HISTORICAL records only: fraction of wakes whose SELECTED joint "
                "(node, meta) cell was an abort",
            "aggregate_p_abort_mean":
                "HISTORICAL records only: mean TOTAL probability mass on the abort "
                "column, summed over its k cells. NOT the probability of the selected "
                "action",
            "joint_entropy_raw_mean":
                "HISTORICAL records only: entropy of the joint (node, meta-action) "
                "distribution, in nats, cardinality-dependent",
            "joint_entropy_normalized_mean":
                "HISTORICAL records only: the same entropy divided by log(valid cell "
                "count); undefined (and excluded) when fewer than two cells are valid",
        },
    }


def _observed_artifact_schema(
    outcome_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """What schema the DURABLE records actually carry -- OBSERVED, never asserted.

    Reporting this writer's own ``_EPISODE_OUTCOME_VERSION`` as a run directory's schema
    is false whenever the two differ, and it differs exactly when it matters: summarizing
    a LEGACY run written before the current writer existed. A reader would then be told
    a v2 artifact is v3 and that wake diagnostics are present when the file has none.

    Three states are reported explicitly and never collapsed:
    ``no_records`` (nothing to observe -- the observed version is ``None``, not a
    default), ``uniform`` (every record agrees), and ``mixed`` (a run directory whose
    records disagree, which is a fact about the artifact and must not be averaged away).
    The writer's own constants are still reported, under names that say so.
    """
    rows = list(outcome_records or [])
    versions = sorted({
        int(r["schema_version"]) for r in rows
        if isinstance(r.get("schema_version"), int)
        and not isinstance(r.get("schema_version"), bool)})
    n_unversioned = sum(
        1 for r in rows
        if not isinstance(r.get("schema_version"), int)
        or isinstance(r.get("schema_version"), bool))
    with_diag = [r for r in rows if isinstance(r.get("wake_decisions"), list)]
    diag_versions = sorted({
        int(r["wake_diagnostics_schema_version"]) for r in with_diag
        if isinstance(r.get("wake_diagnostics_schema_version"), int)
        and not isinstance(r.get("wake_diagnostics_schema_version"), bool)})
    # The action representation the records' WAKES are stated in, observed per wake --
    # a historical wake (no id) is reported under the reader label, never as the
    # current representation.
    wake_reps = sorted({_wake_action_representation(d) for r in with_diag
                        for d in (r.get("wake_decisions") or ())
                        if isinstance(d, Mapping)})
    if not rows:
        state = "no_records"
    elif len(versions) == 1 and n_unversioned == 0:
        state = "uniform"
    else:
        state = "mixed"
    return {
        "state": state,
        "source": _EPISODE_OUTCOMES_FILENAME,
        "n_episode_outcome_records": len(rows),
        "n_records_without_schema_version": n_unversioned,
        "episode_outcome_schema_versions_observed": versions,
        "episode_outcome_schema_version_observed": (
            versions[0] if state == "uniform" else None),
        "n_records_with_wake_decisions": len(with_diag),
        "wake_diagnostics_recorded": bool(with_diag),
        "wake_diagnostics_schema_versions_observed": diag_versions,
        "wake_diagnostics_schema_version_observed": (
            diag_versions[0] if len(diag_versions) == 1 else None),
        "wake_action_representations_observed": wake_reps,
        # The CURRENT writer, reported under names that cannot be read as an
        # observation of the artifact.
        "episode_outcome_schema_version_writer": _EPISODE_OUTCOME_VERSION,
        "wake_diagnostics_schema_version_writer": _WAKE_DIAGNOSTICS_VERSION,
        "action_representation_id_writer": ACTION_REPRESENTATION_ID,
    }


def _severity_response_from_outcomes(
    outcome_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The severity-response table, DERIVED FROM THE DURABLE PER-ATTEMPT STREAM.

    The run summary must not claim anything an artifact does not already state, so this
    reads ``episode_outcomes.jsonl`` rather than an in-memory aggregate: a summary that
    could describe a run its own files do not is the failure mode the whole one-metric-
    path discipline exists to prevent (:func:`build_run_summary`).

    For every phase and every reporting cell it reports, over SUCCESSFUL attempts:
    how many episodes, how many fired an event, how many produced an FD WAKE, and the
    meta-action that wake chose. The rates are over FD WAKES -- the only population in
    which the actor was actually asked -- and are ``None``, never ``0.0``, when that
    population is empty.
    """
    by_phase: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rec in outcome_records:
        phase = str(rec.get("phase", "unknown"))
        cell = str(rec.get("cell", CONDITION_CLEAN))
        bucket = by_phase.setdefault(phase, {}).setdefault(cell, {
            "n_episodes": 0, "n_fd_fired": 0, "n_fd_wakes": 0,
            "meta_action_counts": _empty_meta_counts(),
            "rewards": [],
            "n_dead": 0, "n_rtb_command": 0,
        })
        bucket["n_episodes"] += 1
        bucket["rewards"].append(float(rec.get("reward", 0.0)))
        bucket["n_dead"] += int(rec.get("n_dead", 0) or 0)
        if rec.get("fd_fired"):
            bucket["n_fd_fired"] += 1
        if rec.get("fd_rtb_command_issued"):
            bucket["n_rtb_command"] += 1
        if rec.get("fd_wake_occurred"):
            bucket["n_fd_wakes"] += 1
            name = rec.get("fd_wake_meta_action_name")
            if name in bucket["meta_action_counts"]:
                bucket["meta_action_counts"][name] += 1
    out: Dict[str, Any] = {}
    for phase, cells in by_phase.items():
        out[phase] = {}
        for cell, bucket in cells.items():
            denom = int(bucket["n_fd_wakes"])
            out[phase][cell] = {
                "n_episodes": int(bucket["n_episodes"]),
                "n_fd_fired": int(bucket["n_fd_fired"]),
                "n_fd_wakes": denom,
                "n_rtb_command_issued": int(bucket["n_rtb_command"]),
                "n_dead": int(bucket["n_dead"]),
                "reward_mean": _stats_or_none(bucket["rewards"])["mean"],
                "meta_action_counts": dict(bucket["meta_action_counts"]),
                "meta_action_rates": {
                    name: _fraction(int(bucket["meta_action_counts"][name]), denom)
                    for name in _META_NAMES
                },
                "rates_over": "fd_wakes",
            }
    return out


def _tally_slugs(values: Sequence[Any]) -> Dict[str, int]:
    """Count stable slugs into a plain dict (sorted, so a record is diff-stable)."""
    counts: Dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def _histogram(values: Sequence[Any]) -> Dict[str, int]:
    """A count-by-value histogram keyed by the value's string form."""
    return _tally_slugs(values)


def _early_stopping_summary(
    train_records: List[Dict[str, Any]],
    *,
    cfg: Optional[TrainConfig],
    train_attempted: int,
    train_successful: int,
) -> Dict[str, Any]:
    """The run's stopping block: what was configured, what fired, and what it cost.

    DERIVED FROM THE DURABLE RECORDS. The check history is read back out of
    ``train_records.jsonl`` -- the same dicts :func:`train` attached to the iterations
    they were computed on -- rather than from a parallel in-memory tally, so this cannot
    describe a decision the artifacts do not contain. ``cfg`` supplies only the
    CONFIGURED shape and the PLANNED budget, which the records cannot know, exactly as it
    does for ``episode_design``; every planned/actual pair is reported side by side so
    "this run is short" never has to be inferred by comparing two numbers from different
    files.

    ``termination_reason`` is one of :data:`TERMINATION_REASONS` and describes what the
    RECORDS show: ``training_reward_plateau`` when a check fired,
    ``maximum_budget_reached`` for an enabled run that did not, and
    ``disabled_fixed_budget`` when the feature was off. It is ``None`` only when neither
    the records nor a config can say (a summary built over a pre-feature run directory
    with no ``cfg``).

    A TRIGGERED STOP IS NOT A CONVERGENCE CLAIM. What is recorded is that the configured
    plateau rule fired on ``train_reward_mean``; nothing here asserts an optimum.
    """
    checks = [
        r[_EARLY_STOPPING_RECORD_KEY] for r in train_records
        if isinstance(r.get(_EARLY_STOPPING_RECORD_KEY), dict)
    ]
    stop_check = next(
        (c for c in checks if bool(c.get("stop_triggered"))), None
    )
    triggered = stop_check is not None

    if cfg is None:
        # No config: a check history proves the feature was on, but its ABSENCE does not
        # prove it was off (a short enabled run reaches no check), so `None` is the
        # truthful answer there rather than a guessed `False`.
        enabled: Optional[bool] = True if checks else None
    else:
        enabled = bool(cfg.early_stopping_enabled)

    if enabled is False:
        reason: Optional[str] = TERMINATION_REASON_DISABLED
    elif triggered:
        reason = TERMINATION_REASON_PLATEAU
    elif enabled is True:
        reason = TERMINATION_REASON_MAX_BUDGET
    else:
        reason = None

    configured = enabled is True and cfg is not None
    return {
        "enabled": enabled,
        # The policy and its metric are STATED, never left to be inferred from the
        # presence of a check: a run that stopped and a run that never reached a check
        # must describe the same rule.
        "policy": (
            EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU if enabled
            else None
        ),
        "metric": EARLY_STOPPING_METRIC if enabled else None,
        "min_iterations": (
            int(cfg.early_stopping_min_iterations) if configured else None),
        "window_iterations": (
            int(cfg.early_stopping_window_iterations) if configured else None),
        "patience_windows": (
            int(cfg.early_stopping_patience_windows) if configured else None),
        "min_delta": (
            float(cfg.early_stopping_min_delta) if configured else None),
        "earliest_possible_stop_iterations": (
            int(cfg.early_stopping_earliest_stop_iterations) if configured else None),
        "triggered": bool(triggered),
        "termination_reason": reason,
        # --- planned vs actual, always as a PAIR ---
        "planned_iterations": None if cfg is None else int(cfg.n_iterations),
        "completed_iterations": len(train_records),
        "planned_successful_episodes": None if cfg is None else int(cfg.total_episodes),
        "actual_successful_episodes": int(train_successful),
        "planned_max_training_attempts": (
            None if cfg is None else int(cfg.max_training_attempts)),
        "actual_training_attempts": int(train_attempted),
        # Where it stopped, in both forms. `null` -- never 0 -- when nothing fired: 0 is
        # a real completed-iteration count and a real iteration index.
        "stop_completed_iterations": (
            int(stop_check["completed_iterations"]) if stop_check else None),
        "stop_iteration_index": (
            int(stop_check["iteration"]) if stop_check else None),
        # THE DURABLE CHECK HISTORY, verbatim: every due check, in order, each carrying
        # its window, its comparison and its stale count, so each decision can be
        # re-derived without re-running anything.
        "n_checks": len(checks),
        "checks": checks,
        "checks_source": "train_records.jsonl",
    }


def _generalized_summary(
    outcome_records: List[Dict[str, Any]],
    failure_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The GENERALIZED-V1 roll-up, DERIVED FROM THE CANONICAL jsonl STREAMS.

    ONE metric path, exactly as ``_severity_response_from_outcomes`` already is: every
    number here is read from ``episode_outcomes.jsonl`` (successful attempts),
    ``episode_failures.jsonl`` (failed ones) and the per-round eval records, never from a
    parallel in-memory aggregate -- so the summary cannot describe a run its own artifacts
    do not.

    EVERY DENOMINATOR IS EXPLICIT, and the two streams are DISJOINT by construction, so
    ``attempted == successful + failed`` per bucket rather than by assumption. A failed
    attempt still carries the world it was SCHEDULED to build (the ledger records its
    cardinality precisely because it never built one), which is what keeps a HIGH-load
    stratum's denominator complete instead of quietly shrinking to the attempts that
    happened to succeed.

    THE PHASE A BUCKET IS TAKEN OVER IS PART OF ITS NAME. ``train_by_agent_count`` and
    ``train_by_hidden_requested`` are the TRAINING population and are therefore derived
    from the training-phase rows of both streams alone; held-out evaluation attempts --
    which are scheduled independently and, under a frozen benchmark, re-measure the same
    worlds every round -- belong to the ``benchmark`` block and never to these two.

    THE REQUESTED-VS-REALIZED DISTRIBUTION IS REPORTED, NOT JUDGED. ``hidden_realized``
    is emitted as a HISTOGRAM per requested load, so a HIGH stratum that keeps realizing
    one hidden target is visible as a shape rather than hidden inside a mean. This
    function deliberately invents NO threshold for "systematic degeneration" and returns
    no verdict: the handoff requires that distribution to be INSPECTED by the research
    review before any measurement, and a threshold here would pre-empt that decision
    (handoff 3l.6).
    """
    successes = [r for r in outcome_records if r.get("generalized")]
    failures = [r for r in failure_records if r.get("agent_count") is not None]
    if not successes and not failures:
        return {}

    # THE `train_by_*` AGGREGATES ARE TRAINING-POPULATION ACCOUNTING, so they are taken
    # over the TRAINING phase alone. Both canonical streams mix phases by design -- an
    # outcome row carries `train` / `pre_update` / `post_update` and a failure row
    # carries `train` / `eval` -- so a bucket built from the unfiltered streams would
    # silently fold held-out evaluation attempts into a denominator whose NAME says
    # training. That is a research-validity fault rather than a cosmetic one: the two
    # populations are scheduled independently (a benchmark round re-measures the same
    # frozen worlds every round), so their sum describes nothing. Filtering here, at the
    # ONE derived site, leaves the canonical streams and every other block untouched.
    train_successes = [r for r in successes
                       if str(r.get("phase")) == _ARTIFACT_PHASE_TRAIN]
    train_failures = [r for r in failures
                      if str(r.get("phase")) == _ARTIFACT_PHASE_TRAIN]

    def _by(
        key: str,
        rows_successful: List[Dict[str, Any]],
        rows_failed: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """attempted / successful / failed, bucketed by one scheduled field.

        The two populations are passed in EXPLICITLY rather than closed over, so the
        phase a bucket is taken over is stated at the call site and cannot drift back to
        "whatever was in scope".
        """
        buckets: Dict[str, Dict[str, int]] = {}
        for rec in rows_successful:
            b = buckets.setdefault(str(rec.get(key)), {"successful": 0, "failed": 0})
            b["successful"] += 1
        for rec in rows_failed:
            b = buckets.setdefault(str(rec.get(key)), {"successful": 0, "failed": 0})
            b["failed"] += 1
        return {
            name: {
                "n_attempted": b["successful"] + b["failed"],
                "n_successful": b["successful"],
                "n_failed": b["failed"],
                "success_fraction": _fraction(
                    b["successful"], b["successful"] + b["failed"]),
            }
            for name, b in sorted(buckets.items())
        }

    # --- requested vs realized hidden cardinality, per REQUESTED load ------------
    realized_by_request: Dict[str, List[int]] = {}
    for rec in successes:
        requested = rec.get("hidden_requested")
        realized = rec.get("hidden_realized")
        if requested is None or realized is None:
            continue
        realized_by_request.setdefault(str(int(requested)), []).append(int(realized))
    cardinality = {
        request: {
            "n_successful": len(values),
            "hidden_realized_histogram": _histogram(values),
            "hidden_realized_mean": _stats_or_none([float(v) for v in values])["mean"],
            "n_short_realized": sum(1 for v in values if v < int(request)),
            "short_realized_fraction": _fraction(
                sum(1 for v in values if v < int(request)), len(values)),
        }
        for request, values in sorted(realized_by_request.items())
    }

    # --- reference-solve health, over the episodes that produced a reference -----
    ref_rows = [r for r in successes if r.get("reference_kind") is not None]
    invoked = [r for r in ref_rows if r.get("reference_solver_invoked")]
    seconds = [float(r["reference_solver_seconds"]) for r in invoked
               if r.get("reference_solver_seconds") is not None]
    allocated = [float(r["reference_allocated_task_count"]) for r in ref_rows
                 if r.get("reference_allocated_task_count") is not None]
    candidates = [float(r["reference_candidate_task_count"]) for r in ref_rows
                  if r.get("reference_candidate_task_count") is not None]
    scored = [int(r["scored_completed_targets"]) for r in successes
              if r.get("scored_completed_targets") is not None]
    unscored = [int(r["unscored_completed_targets"]) for r in successes
                if r.get("unscored_completed_targets") is not None]

    # --- post-FD completion-boundary adaptation ---------------------------------
    boundary_rows = [r for r in successes if r.get("post_fd_armed")]
    boundary_meta: List[str] = []
    for rec in boundary_rows:
        boundary_meta.extend(rec.get("post_fd_boundary_meta_action_names") or [])

    # --- the frozen benchmark, from the LAST round and across every round -------
    bench_rounds = [r for r in eval_records if r.get("benchmark_strata")]
    benchmark: Optional[Dict[str, Any]] = None
    if bench_rounds:
        final = bench_rounds[-1]
        totals: Dict[str, Dict[str, int]] = {}
        for round_record in bench_rounds:
            for key, entry in (round_record.get("benchmark_strata") or {}).items():
                acc = totals.setdefault(
                    key, {"n_attempted": 0, "n_successful": 0, "n_failed": 0,
                          "n_short_realized": 0, "n_fd_wakes": 0})
                for field_name in acc:
                    acc[field_name] += int(entry.get(field_name, 0) or 0)
        benchmark = {
            "manifest_id": final.get("benchmark_manifest_id"),
            "n_worlds": final.get("benchmark_n_worlds"),
            "n_members_per_round": final.get("benchmark_n_members"),
            "n_strata": final.get("benchmark_n_strata"),
            "n_rounds": len(bench_rounds),
            # The FINAL round is the clean statistical unit for the finished policy; the
            # cross-round totals describe the TRAJECTORY and are NOT independent worlds
            # (every round re-measures the same frozen manifest), which is why they are
            # reported under their own name and never pooled into the final round's.
            "final_round_strata": final.get("benchmark_strata"),
            "final_round_base_cells": final.get("benchmark_base_cells"),
            "final_round_deltas": {
                key: final.get(key) for key in (final.get("benchmark_delta_keys") or [])
            },
            "final_round_delta_n": {
                key: final.get("%s_n" % key)
                for key in (final.get("benchmark_delta_keys") or [])
            },
            "final_round_groups_successful": final.get("n_groups_successful"),
            "final_round_groups_attempted": final.get("n_groups_attempted"),
            "strata_attempt_totals_across_rounds": totals,
            "totals_across_rounds_are_repeated_measures": True,
        }

    # --- the frozen GENERALIZED-V2 benchmark, added ONLY when V2 rounds exist ---
    v2_rounds = sorted(
        [r for r in eval_records if r.get("v2_behaviour") is not None],
        key=lambda r: (int(r.get("updates_completed") or 0),
                       int(r.get("eval_round_ordinal") or 0)))
    v2_block: Optional[Dict[str, Any]] = None
    if v2_rounds:
        final_v2 = v2_rounds[-1]
        v2_block = {
            "manifest_id": final_v2.get("benchmark_manifest_id"),
            "profiles": _tally_slugs([r.get("benchmark_profile") for r in v2_rounds]),
            "n_rounds": len(v2_rounds),
            "final_round_identity": {
                "evaluation_stage": final_v2.get("evaluation_stage"),
                "updates_completed": final_v2.get("updates_completed"),
                "eval_round_ordinal": final_v2.get("eval_round_ordinal"),
                "benchmark_profile": final_v2.get("benchmark_profile"),
            },
            "final_round_behaviour": final_v2.get("v2_behaviour"),
            "final_round_groups": final_v2.get("v2_benchmark_groups"),
            "totals_across_rounds_are_repeated_measures": True,
        }

    summary = {
        "episode_design": (
            successes[0].get("episode_design") if successes
            else EPISODE_DESIGN_GENERALIZED_V1
        ),
        "cardinality_sampler": cardinality_sampler_record(),
        "train_by_agent_count": _by(
            "agent_count", train_successes, train_failures),
        "train_by_hidden_requested": _by(
            "hidden_requested", train_successes, train_failures),
        "cardinality_requested_vs_realized": cardinality,
        "construction_backoff_rejections": _tally_slugs(
            [reason for rec in successes
             for reason in (rec.get("construction_backoff_rejections") or [])]
        ),
        "fd_eligibility_rejections": _tally_slugs(
            [reason for rec in successes
             for reason in (rec.get("fd_eligibility_rejections") or [])]
        ),
        "fd_eligibility_selected_ordinals": _histogram(
            [rec.get("fd_eligibility_selected_ordinal") for rec in successes]
        ),
        "post_fd_adaptation": {
            "n_episodes_armed": len(boundary_rows),
            "n_boundaries_confirmed": sum(
                int(r.get("post_fd_boundaries_confirmed") or 0)
                for r in boundary_rows),
            "n_boundaries_with_remaining_mission": sum(
                int(r.get("post_fd_boundaries_with_remaining_mission") or 0)
                for r in boundary_rows),
            "n_boundaries_terminal": sum(
                int(r.get("post_fd_boundaries_terminal") or 0)
                for r in boundary_rows),
            "n_boundary_wakes": sum(
                int(r.get("post_fd_boundary_wakes") or 0) for r in boundary_rows),
            # A rate over BOUNDARY WAKES, which is at most the boundary count and can be
            # smaller: a TERMINAL boundary correctly wakes nobody, so the two are
            # counted apart rather than one inferred from the other.
            "boundary_meta_action_counts": _tally_slugs(boundary_meta),
            "boundary_meta_action_rates": {
                name: _fraction(boundary_meta.count(name), len(boundary_meta))
                for name in _META_NAMES
            },
            "rates_over": "post_fd_boundary_wakes",
            "deactivation_reasons": _tally_slugs(
                [r.get("post_fd_deactivation_reason") for r in boundary_rows]),
        },
        "reference": {
            "n_episodes_with_reference": len(ref_rows),
            "kinds": _tally_slugs([r.get("reference_kind") for r in ref_rows]),
            "n_solver_invoked": len(invoked),
            # A SKIPPED degenerate solve is not a failure: it is a legitimate zero
            # reference that costs no solver call at all -- under either backend --
            # which is why invoked/accepted are reported separately from the episode
            # count.
            "n_solver_accepted": sum(
                1 for r in ref_rows if r.get("reference_solver_accepted")),
            "terminations": _tally_slugs(
                [r.get("reference_solver_termination") for r in ref_rows]),
            "solver_seconds_mean": _stats_or_none(seconds)["mean"],
            "solver_seconds_max": _stats_or_none(seconds)["max"],
            "allocated_task_count_mean": _stats_or_none(allocated)["mean"],
            "candidate_task_count_mean": _stats_or_none(candidates)["mean"],
            "n_scored_completed_targets": sum(scored),
            "n_unscored_completed_targets": sum(unscored),
            "scored_vs_unscored_over": "successful_episodes_with_a_reference",
        },
        # An accounted reference refusal (an unanswered solve). An ABORTING one never
        # reaches the ledger at all, so a non-empty tally here is always attrition.
        "reference_fault_attrition": _tally_slugs(
            [r.get("reference_fault_reason") for r in failure_records]),
        "benchmark": benchmark,
    }
    if v2_block is not None:
        # V2-ONLY key: a V1 or fixed-cell summary keeps exactly the shape it always had.
        summary["v2_benchmark"] = v2_block
    return summary


def _summarize(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    failure_records: List[Dict[str, Any]],
    *,
    outcome_records: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional[TrainConfig] = None,
    run_dir: Path,
    run_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate the three record streams into one summary dict.

    PURE: every number here is derived from the records passed in, so the summary cannot
    describe a run the durable artifacts do not. ``cfg`` supplies only the SCHEDULED
    shape (how many episodes were planned), which the records themselves cannot know.

    Attempt accounting is the point. Training and evaluation are reported as
    attempted / successful / failed rather than as a single count, the failures are
    grouped by phase, by pipeline stage and by exception type, and
    ``accounting_reconciled`` cross-checks the per-record failure counts against the
    independent ledger -- if those two ever disagree, some failure was double-counted or
    lost, and the summary says so instead of quietly presenting a plausible total.

    Reward aggregates skip ``None`` entries (all-failed batches / rounds) rather than
    treating them as ``0.0``; ``n_iterations_without_reward`` reports how many were
    skipped, so a mean is never silently taken over a smaller population than it appears.
    """
    means = [r.get("train_reward_mean", r.get("baseline")) for r in train_records]
    measured = [float(m) for m in means if m is not None]

    meta_totals = _empty_meta_counts()
    for r in train_records:
        for name in _META_NAMES:
            meta_totals[name] += int(r.get("meta_action_counts", {}).get(name, 0))

    train_attempted = _sum_field(train_records, "n_attempted", "episodes_per_iteration")
    train_ok = _sum_field(train_records, "n_successful", "n_episodes")
    train_failed = _sum_field(train_records, "n_failed", "n_failed_episodes")
    eval_attempted = _sum_field(eval_records, "n_attempted", "n_episodes")
    eval_ok = _sum_field(eval_records, "n_successful", "n_ok")
    eval_failed = _sum_field(eval_records, "n_failed", "n_failed")

    by_phase = _count_by(failure_records, "phase")
    ledger_train = by_phase.get("train", 0)
    ledger_eval = by_phase.get("eval", 0)

    pre_update = next(
        (r for r in eval_records
         if r.get("evaluation_stage") == _EVAL_STAGE_PRE_UPDATE),
        None,
    )
    eval_means = [
        float(r["eval_reward_mean"]) for r in eval_records
        if r.get("eval_reward_mean") is not None
    ]
    wake_episodes = sum(
        int(r.get("episodes_with_wakes", 0) or 0) for r in train_records
    )
    outcomes = [_iteration_outcome(r) for r in train_records]

    # --- FD-BASELINE-v1 roll-up ---------------------------------------------------
    # Counts sum; MEANS do not. A per-condition mean is only ever taken over that
    # condition's successful episodes, and averaging per-iteration means would silently
    # weight a 1-episode iteration like an 8-episode one -- so the summary reports the
    # per-condition counts (which are exact) and leaves the conditional means to the
    # per-iteration records and to the eval digests, where their denominators travel
    # with them.
    fd_totals: Dict[str, int] = {}
    for key in ("fuel_damage_events_applied", "fuel_damage_wakes",
                "fuel_damage_rtb_issued", "deaths"):
        fd_totals["train_%s" % key] = sum(
            int(r.get(key, 0) or 0) for r in train_records
        )
        fd_totals["eval_%s" % key] = sum(
            int(r.get("eval_%s" % key, 0) or 0) for r in eval_records
        )
    # Cells, not only conditions: the two severities are summed as well when this run
    # has them, so a variable-severity run's per-cell yield is in the summary and not
    # only in the per-round records. Reading a key that a legacy record never wrote
    # simply sums zero, which is the correct total for a cell that never existed.
    reported_cells = (
        tuple(cfg.reported_cells) if cfg is not None else tuple(CONDITIONS)
    )
    for cell in tuple(CONDITIONS) + tuple(
        c for c in reported_cells if c not in CONDITIONS
    ):
        for suffix in ("attempted", "successful", "failed"):
            fd_totals["train_%s_%s" % (cell, suffix)] = sum(
                int(r.get("n_%s_%s" % (cell, suffix), 0) or 0)
                for r in train_records
            )
            fd_totals["eval_%s_%s" % (cell, suffix)] = sum(
                int(r.get("eval_n_%s_%s" % (cell, suffix), 0) or 0)
                for r in eval_records
            )
    # Complete matched GROUPS. `pairs` is the legacy name of the same quantity, so both
    # are emitted from the SAME sum rather than counted twice.
    groups_attempted = sum(
        int(r.get("n_groups_attempted", r.get("n_pairs_attempted", 0)) or 0)
        for r in eval_records
    )
    groups_successful = sum(
        int(r.get("n_groups_successful", r.get("n_pairs_successful", 0)) or 0)
        for r in eval_records
    )
    fd_totals["eval_groups_attempted"] = groups_attempted
    fd_totals["eval_groups_successful"] = groups_successful
    fd_totals["eval_pairs_attempted"] = groups_attempted
    fd_totals["eval_pairs_successful"] = groups_successful

    # The severity-response table, derived from the DURABLE per-attempt stream so the
    # summary states nothing the artifacts do not.
    outcome_rows = list(outcome_records or [])
    severity_response = _severity_response_from_outcomes(outcome_rows)
    observed_artifact_schema = _observed_artifact_schema(outcome_rows)
    # The final EVALUATION round is identified ONCE, here, and the same digest is both
    # reported as `final_eval` below and handed to the sensitivity table -- so the two
    # cannot name different rounds, and the sensitivity table's "final round" is chosen
    # by that validated identity rather than by which record the file happens to end on.
    # THE final round is chosen SEMANTICALLY, once, here -- never `eval_records[-1]`.
    # The same selected record drives `final_eval`, every `final_eval_*` scalar below,
    # and the FD-sensitivity table's own final-round pick, so the three cannot name
    # different rounds; and a refusal is persisted as a stated reason.
    final_eval_record, final_eval_selection = _select_final_eval_record(eval_records)
    final_eval_digest = _eval_digest(final_eval_record)
    fd_policy_sensitivity = _fd_policy_sensitivity_from_outcomes(
        outcome_rows, final_eval=final_eval_digest)
    # THE SAME ELIGIBILITY RULE THE FIGURE ITSELF APPLIES. `_plot_fd_policy_sensitivity`
    # returns `None` on exactly this condition, so declaring the optional figure from
    # `fd_policy_sensitivity["recorded"]` -- which is true as soon as ANY record, a
    # TRAINING one included, carries wake diagnostics -- would promise a file a
    # train-only or evaluation-disabled run never writes.
    fd_sensitivity_plot_data = _fd_sensitivity_plot_data(outcome_rows)

    run_path = Path(run_dir)
    summary: Dict[str, Any] = {
        # --- shape ---
        "n_iterations": len(train_records),
        "n_iterations_scheduled": None if cfg is None else int(cfg.n_iterations),
        "episodes_per_iteration": (
            None if cfg is None else int(cfg.episodes_per_iteration)
        ),
        "exact_cardinality_policy": _EXACT_CARDINALITY_POLICY,
        "updates_completed": (
            int(train_records[-1].get("updates_completed", 0)) if train_records else 0
        ),
        # --- training attempt accounting ---
        "train_episodes_attempted": train_attempted,
        "train_episodes_successful": train_ok,
        "train_episodes_failed": train_failed,
        "train_success_fraction": _fraction(train_ok, train_attempted),
        "train_episodes_with_wakes": wake_episodes,
        "train_wake_fraction_of_successful": _fraction(wake_episodes, train_ok),
        "train_zero_wake_episodes": max(train_ok - wake_episodes, 0),
        # GENERALIZED-V1 Task 5C: WHAT `episodes_per_iteration` COUNTED, and how many
        # attempts the run really spent. DERIVED from `train_records.jsonl` like every
        # other number here -- one metric path, so the summary cannot describe a run its
        # own artifacts do not. Present on BOTH designs and truthful on each: a
        # fixed-cell run reports `scheduled_attempts_v1` with zero replacements, which is
        # the historical contract stated rather than left to be inferred.
        "training_attempt_policy": (
            str(train_records[-1].get(
                "training_attempt_policy", TRAINING_ATTEMPT_POLICY_SCHEDULED))
            if train_records else None
        ),
        "train_replacement_attempts": _sum_field(
            train_records, "n_replacement_attempts", "__absent__"),
        "train_iterations_at_full_quota": sum(
            1 for r in train_records
            if int(r.get("n_successful", 0) or 0)
            >= int(r.get("successful_episodes_required",
                         r.get("episodes_per_iteration", 0)) or 0)
        ),
        # GENERALIZED-V1 EARLY STOPPING: whether the run ended at its maximum budget or
        # because the configured plateau rule fired, with the planned/actual pairs and
        # the durable check history behind it. Present on every run: a fixed-budget run
        # reports `disabled_fixed_budget`, which states the contract rather than leaving
        # it to be inferred from an absent key.
        "early_stopping": _early_stopping_summary(
            train_records, cfg=cfg,
            train_attempted=train_attempted, train_successful=train_ok,
        ),
        # DISJOINT by construction (see `_iteration_outcome`): an iteration in which
        # every attempt failed is a data-yield failure, NOT an iteration in which
        # episodes ran and nobody woke.
        "n_zero_wake_iterations": outcomes.count("zero_wake"),
        "n_all_failed_iterations": outcomes.count("all_failed"),
        "n_productive_iterations": outcomes.count("productive"),
        "total_transitions": sum(
            int(r.get("n_transitions", 0)) for r in train_records
        ),
        # --- evaluation attempt accounting ---
        "eval_episodes_attempted": eval_attempted,
        "eval_episodes_successful": eval_ok,
        "eval_episodes_failed": eval_failed,
        "eval_success_fraction": _fraction(eval_ok, eval_attempted),
        "n_eval_rounds": len(eval_records),
        # --- failures, grouped ---
        "failures_recorded": len(failure_records),
        "failures_by_phase": by_phase,
        "failures_by_pipeline_stage": _count_by(failure_records, "pipeline_stage"),
        "failures_by_error_type": _count_by(failure_records, "error_type"),
        "failures_by_condition": _count_by(failure_records, "condition"),
        # --- the difficulty factor's own accounting ---
        "difficulty_factor": (
            "fuel_damage_baseline_v1" if cfg is None else _difficulty_factor_name(cfg)
        ),
        "fuel_damage_mode": (
            None if cfg is None else str(cfg.fuel_damage_mode)
        ),
        "fuel_damage_mild_probability": (
            None if cfg is None or not cfg.variable_severity
            else float(cfg.fuel_damage_mild_probability)
        ),
        # The RECORDS are authoritative (they describe what ran), the config is the
        # fallback for a run whose records predate the field, and `None` only when
        # neither can say -- never a guessed default that a reader could not distinguish
        # from a measured one.
        # A RUN-INVARIANT (`pair` or `triad`), identical in every round, so this is a
        # label rather than a measurement. It is still read from the SELECTED round
        # whenever one exists, so it cannot depend on record order; the positional
        # expression survives only as the fallback for a record set the selector
        # refused, and the config remains the last resort.
        "eval_group_kind": (
            ((final_eval_record or (eval_records[-1] if eval_records else None) or {})
             .get("eval_group_kind"))
            or (None if cfg is None else cfg.eval_group_kind)
        ),
        "eval_group_cells": list(reported_cells),
        "aircraft_penalty_coeff": (
            None if cfg is None else float(cfg.aircraft_penalty_coeff)
        ),
        "fuel_damage_totals": fd_totals,
        # THE PRIMARY BEHAVIOURAL MEASUREMENT of the variable-severity design, derived
        # from `episode_outcomes.jsonl`: per phase and per cell, what the fuel-damage
        # wake chose, over the FD wakes it is a rate of. Empty for a run with no durable
        # outcome stream (a pre-feature run directory), never fabricated.
        "severity_response": severity_response,
        "severity_response_source": _EPISODE_OUTCOMES_FILENAME,
        "episode_outcomes_recorded": len(outcome_rows),
        # PER-WAKE ACTOR DIAGNOSTICS (episode-outcome schema v3), derived from the SAME
        # durable stream. `recorded: false` -- never a table of zeros -- for a run
        # directory written before the field existed.
        #
        # THE SCHEMA REPORTED HERE IS OBSERVED FROM THE RECORDS, NEVER THIS WRITER'S OWN
        # CONSTANTS. `build_run_summary` runs on ANY run directory, including one written
        # by an older writer, so stamping the current constants would tell a reader that
        # a legacy v2 artifact is v3 and that it carries wake diagnostics it does not.
        # The writer's constants are still reported, inside the block, under `_writer`
        # names that cannot be mistaken for an observation.
        "observed_artifact_schema": observed_artifact_schema,
        "wake_diagnostics_source": _EPISODE_OUTCOMES_FILENAME,
        "wake_kinds": list(WAKE_KINDS),
        "fd_policy_sensitivity": fd_policy_sensitivity,
        # WHICH POPULATION this run drew from. Recorded on BOTH designs, and taken from
        # the config rather than guessed from the records, so a run with zero completed
        # episodes still states its design.
        "episode_design": (
            None if cfg is None else str(cfg.episode_design)
        ),
        "episode_design_policies": (
            None if cfg is None else cfg.design.to_record()
        ),
        # The GENERALIZED-V1 roll-up, DERIVED from the same two durable streams the
        # severity table is (one metric path). `{}` -- never fabricated content -- for a
        # run whose records carry no generalized episode.
        "generalized": _generalized_summary(
            outcome_rows, failure_records, eval_records),
        # The held-out numbers the factor is measured by, taken from the LAST round and
        # always carrying their group denominator. `final_eval_paired_reward_delta` is
        # the LEGACY damaged-minus-clean key and is `null` for a triad run, whose three
        # named deltas are in `final_eval_group_deltas`.
        "final_eval_paired_reward_delta": (
            final_eval_record.get("eval_paired_reward_delta")
            if final_eval_record else None
        ),
        "final_eval_group_deltas": (
            {key: final_eval_record.get(key)
             for key in (final_eval_record.get("eval_delta_keys") or [])}
            if final_eval_record else None
        ),
        "final_eval_groups_successful": (
            final_eval_record.get("n_groups_successful",
                                  final_eval_record.get("n_pairs_successful"))
            if final_eval_record else None
        ),
        "final_eval_groups_attempted": (
            final_eval_record.get("n_groups_attempted",
                                  final_eval_record.get("n_pairs_attempted"))
            if final_eval_record else None
        ),
        "final_eval_pairs_successful": (
            final_eval_record.get("n_pairs_successful")
            if final_eval_record else None
        ),
        "final_eval_pairs_attempted": (
            final_eval_record.get("n_pairs_attempted")
            if final_eval_record else None
        ),
        "accounting_reconciled": (
            ledger_train == train_failed and ledger_eval == eval_failed
        ),
        # --- held-out results, each with its denominator ---
        "initial_pre_update_eval": _eval_digest(pre_update),
        "final_eval": final_eval_digest,
        # WHICH round `final_eval` and every `final_eval_*` field above came from, or
        # WHY none could be chosen. Persisted so a refusal reads as a stated reason
        # rather than as a set of quietly null fields.
        "final_eval_selection": final_eval_selection,
        "eval_reward_best": max(eval_means) if eval_means else None,
        # --- training reward over the MEASURED iterations only ---
        "train_reward_first": measured[0] if measured else None,
        "train_reward_last": measured[-1] if measured else None,
        "train_reward_mean": _stats_or_none(measured)["mean"],
        "n_iterations_without_reward": len(means) - len(measured),
        "aggregates_over": "successful_episodes",
        "meta_action_totals": meta_totals,
        # --- artifacts ---
        "run_seconds": run_seconds,
        "run_dir": str(run_path),
        "train_records_path": str(run_path / "train_records.jsonl"),
        "eval_records_path": str(run_path / "eval_records.jsonl"),
        "failures_path": str(run_path / "episode_failures.jsonl"),
        "episode_outcomes_path": str(run_path / _EPISODE_OUTCOMES_FILENAME),
        "run_config_path": str(run_path / "run_config.json"),
        "run_summary_path": str(run_path / "run_summary.json"),
        # Figures live under `<run_dir>/plots/`, one claim per file. Listed by NAME so a
        # reader (or a notebook) can resolve a specific figure without knowing the
        # layout, and so the summary states which figures a run is supposed to have.
        "plots_dir": str(_plots_dir(run_path)),
        "plot_paths": {
            name: str(_plots_dir(run_path) / name) for name in _PLOT_FILENAMES
        },
        # The OPTIONAL figures. Keyed off whether this run's DATA supports the figure,
        # not off whether the file exists yet: `build_run_summary` runs BEFORE
        # `plot_training_subprocess`, so an existence test here would always be empty.
        # That is also exactly the convention `plot_paths` above already follows -- it
        # DECLARES where a figure belongs rather than stat-ing for it. Empty (never
        # null-valued) for a run whose data does not support the figure, so a reader
        # never mistakes "not applicable" for "failed to render".
        #
        # THE PREDICATE IS THE FIGURE'S OWN: `_fd_sensitivity_plot_data(...)["recorded"]`
        # is precisely the condition `_plot_fd_policy_sensitivity` returns `None` on, so
        # a v3 TRAIN-ONLY or evaluation-disabled run declares nothing. Using the digest's
        # broader `recorded` flag instead would declare a figure that is never written.
        "optional_plot_paths": (
            {_PLOT_FD_SENSITIVITY: str(_plots_dir(run_path) / _PLOT_FD_SENSITIVITY)}
            if fd_sensitivity_plot_data.get("recorded") else {}
        ),
    }
    # Pre-B4 names, kept so an existing reader of a summary still resolves.
    summary["total_train_episodes"] = train_attempted
    summary["n_failed_episodes"] = train_failed
    summary["train_baseline_first"] = summary["train_reward_first"]
    summary["train_baseline_last"] = summary["train_reward_last"]
    summary["train_baseline_mean"] = summary["train_reward_mean"]
    summary["eval_reward_first"] = (eval_means[0] if eval_means else None)
    summary["eval_reward_last"] = (eval_means[-1] if eval_means else None)
    # ALIAS, not a fourth figure: the retired single `training_plot.png` dashboard is
    # gone, and an existing reader of `plot_path` is pointed at the figure that carries
    # the run's performance claim. `plot_paths` is the authoritative list.
    summary["plot_path"] = summary["plot_paths"][_PLOT_PERFORMANCE]
    # The full record streams, for in-process callers ONLY (see _SUMMARY_RECORD_KEYS:
    # they are stripped before the summary is written, because the jsonl files are the
    # record and a copy of them inside the summary could diverge from it).
    summary["train_records"] = train_records
    summary["eval_records"] = eval_records
    summary["failure_records"] = failure_records
    summary["episode_outcome_records"] = outcome_rows
    return summary


def build_run_summary(
    run_dir: Union[str, Path],
    *,
    cfg: Optional[TrainConfig] = None,
    run_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Read a run directory's three jsonl artifacts and summarize them.

    The ONE metric path: :func:`train` calls this on the files it has just written
    rather than aggregating its own in-memory lists, so ``run_summary.json`` can only
    ever state what the durable records state. It works on any run directory -- finished
    or in progress -- which is also what makes it testable from synthetic fixtures with
    no training involved.
    """
    run_path = Path(run_dir)
    summary = _summarize(
        _read_jsonl(run_path / "train_records.jsonl"),
        _read_jsonl(run_path / "eval_records.jsonl"),
        _read_jsonl(run_path / "episode_failures.jsonl"),
        # The per-attempt stream is read here too, so every aggregate derived from it is
        # derived from the FILE -- a missing file is simply an empty population, never a
        # fabricated one.
        outcome_records=_read_jsonl(run_path / _EPISODE_OUTCOMES_FILENAME),
        cfg=cfg,
        run_dir=run_path,
        run_seconds=run_seconds,
    )
    # The credit artifact's SCHEMA is observed here and nothing else is read from it:
    # no stopping, evaluation, checkpoint or reward path consumes its values.
    summary["observed_credit_diagnostics"] = _observed_credit_diagnostics(
        _read_jsonl(run_path / _CREDIT_DIAGNOSTICS_FILENAME))
    return summary


def write_run_summary(run_dir: Union[str, Path], summary: Dict[str, Any]) -> Path:
    """Persist ``run_dir/run_summary.json`` (without the embedded record lists).

    The record lists are stripped (:data:`_SUMMARY_RECORD_KEYS`) so the summary stays a
    SUMMARY: the jsonl files remain the single record, and there is no second copy of
    them that could drift out of agreement with the first.
    """
    payload = {k: v for k, v in summary.items() if k not in _SUMMARY_RECORD_KEYS}
    path = Path(run_dir) / "run_summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def _print_summary(s: Dict[str, Any]) -> None:
    """Print the run summary as an ASCII table (no unicode -- cp1255 console)."""
    print("-" * 78)
    print("TRAINING SUMMARY (%d iteration(s) recorded, %s update(s) completed)"
          % (s["n_iterations"], s["updates_completed"]))
    print("-" * 78)
    print("train eps:  attempted=%d  ok=%d  failed=%d  success=%s"
          % (s["train_episodes_attempted"], s["train_episodes_successful"],
             s["train_episodes_failed"],
             _fmt_opt(s["train_success_fraction"], "%.3f")))
    print("            transitions=%d  wake-bearing=%d  zero-wake eps=%d"
          % (s["total_transitions"], s["train_episodes_with_wakes"],
             s["train_zero_wake_episodes"]))
    # Printed only under the quota policy, so a fixed-cell run's console is unchanged.
    # There the line would say nothing anyway: one attempt per scheduled episode, zero
    # replacements, by contract.
    if s.get("training_attempt_policy") == TRAINING_ATTEMPT_POLICY_QUOTA:
        print("            attempt policy=%s  replacements=%d  iterations at full "
              "quota=%d/%d"
              % (s["training_attempt_policy"], s["train_replacement_attempts"],
                 s["train_iterations_at_full_quota"], s["n_iterations"]))
    print("iterations: productive=%d  zero-wake=%d  all-failed=%d   (disjoint: an "
          "all-failed iteration measured nothing)"
          % (s["n_productive_iterations"], s["n_zero_wake_iterations"],
             s["n_all_failed_iterations"]))
    print("train R:    first=%s  last=%s  mean=%s   (over SUCCESSFUL episodes; "
          "%d iteration(s) had none)"
          % (_fmt_opt(s["train_reward_first"]), _fmt_opt(s["train_reward_last"]),
             _fmt_opt(s["train_reward_mean"]), s["n_iterations_without_reward"]))
    mt = s["meta_action_totals"]
    print("meta-acts:  PLAN_COMPLIANCE=%d  OPPORTUNISTIC_ENGAGEMENT=%d  "
          "SELF_PRESERVATION_ABORT=%d"
          % (mt["PLAN_COMPLIANCE"], mt["OPPORTUNISTIC_ENGAGEMENT"],
             mt["SELF_PRESERVATION_ABORT"]))
    if s["n_eval_rounds"]:
        print("eval eps:   attempted=%d  ok=%d  failed=%d  success=%s"
              % (s["eval_episodes_attempted"], s["eval_episodes_successful"],
                 s["eval_episodes_failed"],
                 _fmt_opt(s["eval_success_fraction"], "%.3f")))
        for label, digest in (("pre-update ", s["initial_pre_update_eval"]),
                              ("final      ", s["final_eval"])):
            if digest is None:
                print("eval R:     %s (not recorded)" % label)
                continue
            print("eval R:     %s R=%s over %s/%s seed(s)  [updates=%s]"
                  % (label, _fmt_opt(digest["eval_reward_mean"]),
                     digest["n_successful"], digest["n_attempted"],
                     digest["updates_completed"]))
        print("            rounds=%d  best=%s"
              % (s["n_eval_rounds"], _fmt_opt(s["eval_reward_best"])))
        for label, digest in (("pre-update ", s["initial_pre_update_eval"]),
                              ("final      ", s["final_eval"])):
            if digest is None:
                continue
            cells = digest.get("eval_group_cells") or list(CONDITIONS)
            means = digest.get("cell_reward_means") or {}
            oks = digest.get("cell_successful") or {}
            deltas = digest.get("group_deltas") or {}
            print("eval fd:    %s %s | %s over %s/%s %s(s)"
                  % (label,
                     " | ".join("%s R=%s (%s ok)"
                                % (c, _fmt_opt(means.get(c)), oks.get(c))
                                for c in cells),
                     " ".join("%s=%s" % (k.replace("eval_delta_", ""), _fmt_opt(v))
                              for k, v in deltas.items()) or "no delta",
                     digest.get("n_groups_successful"),
                     digest.get("n_groups_attempted"),
                     digest.get("eval_group_kind") or _EVAL_GROUP_KIND_PAIR))
    else:
        print("eval R:     (disabled)")
    fd = s["fuel_damage_totals"]
    cells = list(s.get("eval_group_cells") or CONDITIONS)
    print("fuel dmg:   mode=%s  penalty_c=%s  factor=%s"
          % (s["fuel_damage_mode"], s["aircraft_penalty_coeff"],
             s.get("difficulty_factor")))
    for phase in ("train", "eval"):
        print("            %-6s %s, events=%d wakes=%d rtb=%d dead=%d%s"
              % (phase + ":",
                 ", ".join("%s %d/%d ok"
                           % (c, fd.get("%s_%s_successful" % (phase, c), 0),
                              fd.get("%s_%s_attempted" % (phase, c), 0))
                           for c in cells),
                 fd["%s_fuel_damage_events_applied" % phase],
                 fd["%s_fuel_damage_wakes" % phase],
                 fd["%s_fuel_damage_rtb_issued" % phase],
                 fd["%s_deaths" % phase],
                 ("  %s %d/%d" % (s.get("eval_group_kind") or _EVAL_GROUP_KIND_PAIR,
                                  fd["eval_groups_successful"],
                                  fd["eval_groups_attempted"])
                  if phase == "eval" else "")))
    # THE PRIMARY BEHAVIOURAL MEASUREMENT, printed only when the run has severities to
    # compare -- for a legacy run there is one damaged cell and no comparison to make.
    response = s.get("severity_response") or {}
    if any(c in SEVERITIES for c in cells):
        abort = MetaAction.SELF_PRESERVATION_ABORT.name
        for phase in sorted(response):
            per_cell = response[phase]
            severities = [c for c in cells if c in SEVERITIES and c in per_cell]
            if not severities:
                continue
            print("            severity response [%s]: %s   (rates over FD WAKES)"
                  % (phase,
                     " | ".join(
                         "%s abort=%s over %d wake(s)"
                         % (c, _fmt_opt(per_cell[c]["meta_action_rates"][abort], "%.2f"),
                            per_cell[c]["n_fd_wakes"])
                         for c in severities)))
    print("failures:   %d recorded  by phase=%s  by stage=%s%s"
          % (s["failures_recorded"], s["failures_by_phase"],
             s["failures_by_pipeline_stage"],
             "" if s["accounting_reconciled"]
             else "   [!] LEDGER DISAGREES WITH THE RECORD COUNTS"))
    if s["run_seconds"] is not None:
        print("timing:     total=%.1fs" % s["run_seconds"])
    print("plots:      %s" % s["plots_dir"])
    print("records:    %s" % s["train_records_path"])
    print("            %s" % s["eval_records_path"])
    print("            %s" % s["failures_path"])
    print("            %s  (%d successful attempt(s))"
          % (s.get("episode_outcomes_path"), s.get("episode_outcomes_recorded", 0)))
    print("            %s" % s["run_summary_path"])
    print("-" * 78)


# =============================================================================
# 9. Plotting (lazy matplotlib -- training never hard-depends on it)
# =============================================================================

def _xy(
    records: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    x_fallback: str = "iteration",
) -> Tuple[List[float], List[float]]:
    """Paired (x, y) series, DROPPING points whose y is missing.

    A ``None`` reward means "this batch or round produced no measurement at all"
    (:func:`_stats_or_none`). Such a point is omitted from the curve rather than drawn:
    plotting it as 0 would show a total data loss AT THE ORACLE OPTIMUM, and plotting it
    as some other number would invent one. Its attempts are still visible --
    ``measurement_health.png`` shows the success fraction that caused the gap.
    """
    xs: List[float] = []
    ys: List[float] = []
    for rec in records:
        y = rec.get(y_key)
        if y is None:
            continue
        x = rec.get(x_key, rec.get(x_fallback))
        if x is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def _xy_first(
    records: List[Dict[str, Any]], x_key: str, *y_keys: str
) -> Tuple[List[float], List[float]]:
    """:func:`_xy` over the FIRST of ``y_keys`` that yields any point.

    The matched-group keys were renamed from ``*_pairs_*`` to the design-neutral
    ``*_groups_*`` when triads arrived, and both are written by every current run. A run
    directory produced BEFORE that -- the preserved Phase-A baseline among them -- carries
    only the legacy names, and `--plot <run_dir>` must keep drawing its complete-pair
    coverage rather than silently losing a series that the records do contain.
    """
    for key in y_keys:
        xs, ys = _xy(records, x_key, key)
        if ys:
            return xs, ys
    return [], []


def _record_cells(eval_records: List[Dict[str, Any]]) -> List[str]:
    """The reporting CELLS a run's eval records use, taken from the records themselves.

    A figure is drawn from jsonl alone -- ``--plot <run_dir>`` has no ``TrainConfig`` --
    so the design has to be read off the file. The LAST round is authoritative (a run
    does not change design mid-flight), and a record that predates the field falls back
    to the legacy clean/damaged pair, which is what such a file actually contains.
    """
    for rec in reversed(eval_records or []):
        cells = rec.get("eval_group_cells")
        if cells:
            return [str(c) for c in cells]
    return list(CONDITIONS)


def _record_delta_keys(eval_records: List[Dict[str, Any]]) -> List[str]:
    """The within-seed delta KEYS a run's eval records carry, from the records.

    Same rule as :func:`_record_cells`; the fallback is the legacy
    ``eval_paired_reward_delta``, so a pre-severity run still plots the one delta it has.
    """
    for rec in reversed(eval_records or []):
        keys = rec.get("eval_delta_keys")
        if keys:
            return [str(k) for k in keys]
    return ["eval_paired_reward_delta"]


# One colour per reporting cell, fixed so the same cell reads the same way on every
# figure of every run. The ordering is deliberate: clean is the reference (green),
# severe is the case the factor exists to create (red), and mild sits between them.
_CELL_STYLE = {
    CONDITION_CLEAN: ("tab:green", "o"),
    CONDITION_DAMAGED: ("tab:red", "s"),
    SEVERITY_MILD: ("tab:orange", "^"),
    SEVERITY_SEVERE: ("tab:red", "s"),
}

# Distinct styles for however many within-seed deltas a design declares (one for a pair,
# three for a triad), by position rather than by name -- a delta is identified by its
# legend entry, which spells out the two cells it differences.
_DELTA_STYLE = (
    ("tab:purple", "D"), ("tab:orange", "^"), ("tab:brown", "v"), ("tab:cyan", "P"),
)


def _plots_dir(run_dir: Union[str, Path]) -> Path:
    """``<run_dir>/plots`` -- the ONE place a figure is ever written.

    Figures are derived, regenerable and (unlike the jsonl records) not evidence, so they
    live in their own subdirectory instead of sitting next to the run's scientific
    artifacts. A run root then holds records, scenarios, checkpoints, optional visual
    artifacts and plots as five clearly separate things.
    """
    return Path(run_dir) / _PLOTS_DIRNAME


def _annotate_x_semantics(fig: Any) -> None:
    """Stamp the shared x-axis meaning onto a figure, in the figure itself.

    Every figure in this module uses the same x-coordinate, and it is NOT the iteration
    index (see :data:`_PLOT_X_SEMANTICS`). Stating it on the image rather than only in a
    docstring is the point: a PNG travels out of the run directory -- into a slide, a
    message, a thesis -- and has to keep carrying what its axis means.
    """
    fig.text(0.005, 0.005, _PLOT_X_SEMANTICS, fontsize=7, color="0.35", ha="left")


def _plot_training_performance(
    plt: Any,
    plots_dir: Path,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
) -> Path:
    """PERFORMANCE only: training reward, held-out clean vs damaged, matched delta.

    Three panels, deliberately NOT one:

      1. TRAINING reward (``train_reward_mean``) -- the stochastic policy on the
         training seed band, averaged over that batch's SUCCESSFUL episodes only.
      2. HELD-OUT matched evaluation, ONE SERIES PER REPORTING CELL -- clean/damaged for
         a legacy run, clean/mild/severe for a variable-severity one. Every member of a
         matched group runs the same fixed held-out seed -- the same generated world, the
         same A_init, the same hidden geometry, the same selected ego -- and they differ
         only in the fuel-damage event. Pooling them into a single "eval reward" curve,
         which is what the retired dashboard drew, averages across the very factor the
         cell was built to study, so that pooled series is NOT drawn here as the held-out
         signal. It appears only as an explicitly labelled fallback for pre-FD records
         that carry no per-cell means at all.

         WHAT THESE SERIES ARE NOT: a within-seed comparison. Each is a mean over ITS OWN
         cell's SUCCESSFUL episodes, and different cells can fail a different number of
         held-out seeds, so the curves are not necessarily averages over the same
         completed seeds. Their vertical gaps are therefore suggestive, not measurements.
         The panel title and every legend entry say so, and ``measurement_health.png``
         carries the per-cell completion counts that make the asymmetry inspectable.
      3. The MATCHED within-seed DELTAS, over groups whose EVERY member completed --
         ``damaged - clean`` for a legacy pair, and ``mild - clean`` / ``severe - clean``
         / ``severe - mild`` for a triad. These are the numbers that isolate the
         difficulty factor and the ONLY within-seed comparisons on this figure, with 0
         marked. An incomplete group contributes to none of them, which is exactly why
         they stay valid when panel 2's populations differ. For the variable-severity
         design, ``severe - mild`` is the one that answers the experiment's question
         directly: it differences two DAMAGED runs of the same world, so it cannot be
         explained by the world at all.

    Panels 1 and 2 mark ``R = 0``: the reward is oracle-normalized regret, so 0 is the
    perfect-information optimum -- a ceiling, not an arbitrary gridline. That is also
    why a batch or round with no successful episode is DROPPED from a curve rather than
    drawn at 0 (see :func:`_xy`): plotting a total data loss at the optimum would invert
    its meaning. The denominators behind every point live in ``measurement_health.png``.
    """
    curve_x, curve_y = _xy(
        train_records, "updates_completed_before", "train_reward_mean"
    )
    if not curve_y:  # pre-B4 records carry the value under its old name
        curve_x, curve_y = _xy(train_records, "updates_completed_before", "baseline")
    # The CELLS and the DELTAS are read off the records themselves, so a pair round draws
    # two series and one delta while a triad round draws three and three -- without this
    # function having to know which design produced the file it is plotting.
    cells = _record_cells(eval_records)
    delta_keys = _record_delta_keys(eval_records)
    kind = str((eval_records[-1].get("eval_group_kind") if eval_records else None)
               or _EVAL_GROUP_KIND_PAIR)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Panel 1: TRAINING reward (stochastic policy, training seeds) ---
    ax = axes[0]
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4",
               label="oracle optimum (R = 0)")
    if curve_y:
        ax.plot(curve_x, curve_y, color="tab:blue", linewidth=1.6,
                marker=".", markersize=5, label="train mean R (stochastic)")
    ax.set_ylabel("episode reward R")
    ax.set_title("TRAINING reward -- regret vs oracle, 0 = optimum "
                 "(SUCCESSFUL episodes only)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 2: HELD-OUT matched evaluation, one series per CELL ---
    ax = axes[1]
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4",
               label="oracle optimum (R = 0)")
    drew_any = False
    for cell in cells:
        color, marker = _CELL_STYLE.get(cell, ("tab:blue", "o"))
        xs, ys = _xy(eval_records, "updates_completed", "eval_reward_mean_%s" % cell)
        if ys:
            drew_any = True
            ax.plot(xs, ys, color=color, linewidth=2.2, marker=marker, markersize=5,
                    label="held-out %s -- mean over SUCCESSFUL forced_%s episodes"
                          % (cell.upper(), cell))
    if not drew_any:
        # Pre-FD records have no per-cell means. Drawing the pooled mean is then the
        # only held-out information that exists -- labelled as pooled, so it can never
        # be mistaken for a per-cell measurement.
        pooled_x, pooled_y = _xy(eval_records, "updates_completed", "eval_reward_mean")
        if pooled_y:
            ax.plot(pooled_x, pooled_y, color="0.35", linewidth=1.8, linestyle=":",
                    marker="o", markersize=4,
                    label="held-out mean R -- ALL CONDITIONS POOLED (legacy records)")
    ax.set_ylabel("episode reward R")
    ax.set_title("HELD-OUT BY %s -- each mean over THAT cell's successful episodes"
                 % ("SEVERITY" if any(c in SEVERITIES for c in cells)
                    else "CONDITION"), fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 3: the matched within-seed delta(s) ---
    ax = axes[2]
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4",
               label="no measured effect (delta = 0)")
    for i, key in enumerate(delta_keys):
        color, marker = _DELTA_STYLE[i % len(_DELTA_STYLE)]
        # A pre-severity run carries the damaged-minus-clean difference only under the
        # legacy key, so an old run directory still plots its one delta.
        xs, ys = _xy_first(
            eval_records, "updates_completed", key,
            *(("eval_paired_reward_delta",)
              if key == _delta_key(CONDITION_DAMAGED, CONDITION_CLEAN) else ()),
        )
        if ys:
            ax.plot(xs, ys, color=color, linewidth=2.0, marker=marker, markersize=5,
                    label="mean(%s) over COMPLETE %ss"
                          % (key.replace("eval_delta_", "").replace("_minus_", " - "),
                             kind))
    ax.set_ylabel("matched reward delta")
    ax.set_xlabel(_PLOT_X_LABEL)
    ax.set_title("MATCHED-%s fuel-damage delta(s) -- the WITHIN-SEED comparison, "
                 "COMPLETE %ss only (denominators: %s)"
                 % (kind.upper(), kind, _PLOT_MEASUREMENT_HEALTH), fontsize=11)
    # Upper right: a damaging event makes the delta negative, so the top of this panel
    # is the half that stays empty in the case the figure exists to show.
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _annotate_x_semantics(fig)
    out_path = plots_dir / _PLOT_PERFORMANCE
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_policy_diagnostics(
    plt: Any,
    plots_dir: Path,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
) -> Path:
    """DIAGNOSTICS: what the policy DID -- overall mix, entropy, and the FD response.

    No panel here is a performance claim; they describe the policy's BEHAVIOUR. The mix
    says which meta-actions were sampled and entropy is its collapse detector -- a mix
    that flattens onto PLAN_COMPLIANCE (always legal, the easy local optimum) while
    entropy falls is the failure the first two panels exist to make visible before a
    reward curve is over-read.

    Panel 3 is THE PRIMARY BEHAVIOURAL MEASUREMENT of FD-VARIABLE-SEVERITY-v1: the
    fraction of held-out FUEL-DAMAGE WAKES that chose ``SELF_PRESERVATION_ABORT``, drawn
    as one series PER DAMAGED CELL. The experiment's question is not whether reward moved
    but whether the actor aborts differently when a fuel loss is SURVIVABLE than when it
    is not -- two series that track each other say it learned "damage => abort" and never
    read its gauge; two that separate say it did. A legacy run has one damaged cell and
    therefore one series, which is still a real measurement (how often the event produced
    an abort at all) and is drawn the same way.

    ITS DENOMINATOR IS FD WAKES, NOT EPISODES, and that distinction is load-bearing: an
    event can fire without the policy ever being woken by it, so dividing by episodes
    would silently deflate every rate. The counts behind these fractions are in
    ``measurement_health.png`` and in ``run_summary.json:/severity_response``.

    Each training point is one batch, placed at the updates its GENERATING policy had
    received -- not at an iteration index -- so the panels line up with the performance
    figure. The titles say "batch" rather than "iteration" for exactly that reason.
    """
    train_x = [
        float(r.get("updates_completed_before", r.get("iteration", 0)))
        for r in train_records
    ]
    entropies = [float(r.get("entropy", 0.0)) for r in train_records]
    fractions = {
        name: [float(r.get("meta_action_fractions", {}).get(name, 0.0))
               for r in train_records]
        for name in _META_NAMES
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    for name, color in zip(_META_NAMES, ("tab:green", "tab:orange", "tab:purple")):
        ax.plot(train_x, fractions[name], color=color, linewidth=1.6,
                marker=".", markersize=4, label=name)
    ax.set_ylabel("fraction of decisions")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Meta-action mix per TRAINING batch (decisions sampled while "
                 "collecting that batch)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(train_x, entropies, color="tab:brown", linewidth=1.6,
            marker=".", markersize=4)
    # LABEL ONLY -- the plotted series is the record's `entropy` unchanged. Under the
    # semantic representation it is the RAW entropy of the k + 2 semantic-leaf
    # distribution (a historical run's records hold the joint k x 3 cell entropy); either
    # way it grows with the number of valid actions and is NOT comparable across episodes
    # of different task-node counts. The normalized form lives on
    # `fd_policy_sensitivity.png`.
    ax.set_ylabel("raw policy entropy (nats)")
    ax.set_title("RAW policy entropy per TRAINING batch (semantic leaves; historical runs: "
                 "joint cells) -- CARDINALITY-DEPENDENT, so not comparable across "
                 "differing task-node counts (collapse detector for the mix above; the "
                 "normalized form is on %s)" % _PLOT_FD_SENSITIVITY, fontsize=10)
    ax.grid(alpha=0.25)

    # --- Panel 3: the FD-wake severity response (the primary behavioural measurement)
    ax = axes[2]
    abort = MetaAction.SELF_PRESERVATION_ABORT.name
    damaged_cells = [c for c in _record_cells(eval_records)
                     if cell_condition(c) == CONDITION_DAMAGED]
    drew_any = False
    for cell in damaged_cells:
        color, marker = _CELL_STYLE.get(cell, ("tab:red", "s"))
        xs: List[float] = []
        ys: List[float] = []
        for rec in eval_records:
            rates = rec.get("eval_fd_meta_action_rates_%s" % cell) or {}
            rate = rates.get(abort)
            x = rec.get("updates_completed")
            # A round in which the cell had NO fd wake reports `None` and is DROPPED,
            # not drawn at 0: 0.0 would claim the actor was asked and chose not to
            # abort, which is a measurement, and there was none.
            if rate is None or x is None:
                continue
            xs.append(float(x))
            ys.append(float(rate))
        if ys:
            drew_any = True
            ax.plot(xs, ys, color=color, linewidth=2.0, marker=marker, markersize=5,
                    label="held-out %s: SELF_PRESERVATION_ABORT rate" % cell.upper())
    ax.set_ylabel("fraction of FD wakes")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(_PLOT_X_LABEL)
    ax.set_title("HELD-OUT FUEL-DAMAGE RESPONSE -- abort rate per damaged cell, over "
                 "FD WAKES (not episodes)%s"
                 % ("" if drew_any else "  [no FD wake recorded]"), fontsize=11)
    if drew_any:
        ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _annotate_x_semantics(fig)
    out_path = plots_dir / _PLOT_DIAGNOSTICS
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# The four per-cell series `fd_policy_sensitivity.png` draws, plus the disagreement
# series, named once so the figure and its test read the SAME keys.
_FD_SENSITIVITY_SERIES_KEYS = (
    "selected_abort_fraction",
    "p_abort_mean",
    "entropy_normalized_mean",
    "distance_clipping_fraction_mean",
)
_FD_DISAGREEMENT_KEY = "joint_vs_aggregate_disagreement_fraction"
# Presentation order only. Membership is decided by which cells the DATA holds, so a
# cell missing from this tuple still gets its series and no series depends on a cell's
# position in it.
_FD_CELL_ORDER = (CONDITION_CLEAN, SEVERITY_MILD, SEVERITY_SEVERE, CONDITION_DAMAGED)


def _fd_sensitivity_plot_data(
    outcome_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The plot data behind ``fd_policy_sensitivity.png``. PURE -- no matplotlib.

    Extracted from the figure so the scientific content can be TESTED without rendering
    anything, and so the two population rules below are enforced in one readable place
    rather than inside five plotting closures.

    **EVALUATION RECORDS ONLY.** Every series here is taken over ``_EVAL_PHASES``
    (``pre_update`` / ``post_update``). A TRAINING row can enter no value and no
    denominator on this figure: training is a stochastic actor on a sampled population
    and held-out evaluation is a deterministic actor on a frozen one, so a severity
    curve that mixed them would answer neither question while looking like it answered
    both.

    **IMMEDIATE-FD WAKES ONLY**, for the same reason the digest keeps the kinds apart:
    an approved measurement is reported over that population alone.

    **THE DISAGREEMENT SERIES IS PER CELL AND ORDER-INDEPENDENT.** It is emitted for
    EVERY cell that actually has immediate-FD wakes -- so mild and severe each get their
    own -- and it is never taken from "whichever cell happens to sort first", which is
    normally ``clean`` and normally has no immediate-FD wake at all, so the series would
    silently be empty.
    """
    rows = [r for r in (outcome_records or [])
            if isinstance(r.get("wake_decisions"), list)
            and str(r.get("phase")) in _EVAL_PHASES]
    if not rows:
        return {"recorded": False, "evaluation_phases": list(_EVAL_PHASES),
                "cells": [], "series": {}, "disagreement_series": {}, "matched": {}}

    present = {str(r.get("cell", "")) for r in rows}
    cells = ([c for c in _FD_CELL_ORDER if c in present]
             + sorted(present - set(_FD_CELL_ORDER) - {""}))

    # cell -> x (updates completed) -> the immediate-FD decisions measured at that x
    buckets: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for rec in rows:
        x = rec.get("updates_completed")
        if x is None:
            continue
        cell = str(rec.get("cell", ""))
        for d in rec["wake_decisions"]:
            if str(d.get("wake_kind") or WAKE_KIND_ORDINARY) == WAKE_KIND_IMMEDIATE_FD:
                buckets.setdefault(cell, {}).setdefault(int(x), []).append(d)

    def _series(cell: str, key: str) -> Dict[str, List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for x in sorted(buckets.get(cell, {})):
            v = _wake_diag_digest(buckets[cell][x]).get(key)
            if v is not None:
                xs.append(float(x))
                ys.append(float(v))
        return {"x": xs, "y": ys}

    series = {key: {c: _series(c, key) for c in cells}
              for key in _FD_SENSITIVITY_SERIES_KEYS}
    # EVERY cell with immediate-FD wakes, not `cells[0]`.
    disagreement = {c: _series(c, _FD_DISAGREEMENT_KEY)
                    for c in cells if buckets.get(c)}

    # MATCHED severe-minus-mild, per x, from the SAME round-scoped pairing the digest
    # uses -- so the figure and `run_summary.json` cannot disagree about a delta.
    matched_points: Dict[int, List[float]] = {}
    per_round: Dict[Any, Dict[str, Dict[str, float]]] = {}
    for rec in rows:
        gkey = rec.get("benchmark_group_key")
        x = rec.get("updates_completed")
        if not gkey or x is None:
            continue
        mass = _immediate_fd_abort_mass(rec)
        if mass is None:
            continue
        ident = _round_identity(rec)
        key = (ident["evaluation_stage"], ident["updates_completed"],
               ident["eval_round_ordinal"], ident["benchmark_manifest_id"])
        per_round.setdefault(key, {}).setdefault(str(gkey), {})[
            str(rec.get("cell", ""))] = mass
    for key, groups in per_round.items():
        x = key[1]
        for v in groups.values():
            if SEVERITY_MILD in v and SEVERITY_SEVERE in v:
                matched_points.setdefault(int(x), []).append(
                    v[SEVERITY_SEVERE] - v[SEVERITY_MILD])

    xs_m = sorted(matched_points)
    return {
        "recorded": True,
        "evaluation_phases": list(_EVAL_PHASES),
        "wake_kind": WAKE_KIND_IMMEDIATE_FD,
        "cells": cells,
        "series": series,
        "disagreement_series": disagreement,
        "matched": {
            "x": [float(x) for x in xs_m],
            "mean": [float(sum(matched_points[x])) / len(matched_points[x])
                     for x in xs_m],
            "points": {int(x): list(matched_points[x]) for x in xs_m},
        },
    }


def _plot_fd_policy_sensitivity(
    plt: Any,
    plots_dir: Path,
    outcome_records: List[Dict[str, Any]],
) -> Optional[Path]:
    """FD-POLICY SENSITIVITY -- five panels, derived from ``episode_outcomes.jsonl``.

    THE FIGURE THE R1 DIAGNOSTIC REPLAY HAD TO RECONSTRUCT OFFLINE. It answers, from
    durable artifacts alone: did the actor's SELECTED action, and the shape of the
    distribution behind it, respond to fuel-damage severity?

    **HELD-OUT EVALUATION ONLY.** Every series comes from :func:`_fd_sensitivity_plot_data`,
    which admits ``pre_update`` / ``post_update`` records and nothing else, so no training
    row can enter a value or a denominator here.

    Every panel names its own quantity, and each wake is read under its own action
    representation (:func:`_wake_diag_digest`):

    * the SELECTED meta-action -- what deterministic evaluation does;
    * P(ABORT) -- the one semantic ABORT leaf, or for a historical record the aggregate
      mass over its node-indexed abort cells;
    * NORMALIZED policy entropy, divided by ``log(valid action count)``; the historical
      joint-vs-aggregate argmax disagreement is drawn only where historical records exist.

    Populations are never pooled: post-FD boundary wakes and ordinary wakes are separate
    populations and appear on no panel here, and the matched panel pairs mild against
    severe through the FULL evaluation identity -- stage, update count, round ordinal,
    manifest and frozen benchmark group -- rather than through target uuids (which are
    not seed-stable labels) or through a group key alone (which pools every round).

    Returns the written path, or ``None`` when no EVALUATION record carries
    ``wake_decisions`` -- a legacy run keeps its three figures and gets no fabricated
    fourth.
    """
    data = _fd_sensitivity_plot_data(outcome_records)
    if not data.get("recorded"):
        return None

    cells = data["cells"]
    colors = {CONDITION_CLEAN: "#2ca02c", SEVERITY_MILD: "#1f77b4",
              SEVERITY_SEVERE: "#d62728", CONDITION_DAMAGED: "#ff7f0e"}

    def _draw(ax: Any, key: str) -> None:
        for c in cells:
            xy = data["series"][key].get(c) or {"x": [], "y": []}
            if xy["x"]:
                ax.plot(xy["x"], xy["y"], marker="o", label=c, color=colors.get(c))

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.8))

    ax = axes[0]
    _draw(ax, "selected_abort_fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SELECTED meta-action\nfraction of immediate-FD wakes selecting "
                 "SELF_PRESERVATION_ABORT", fontsize=9)
    ax.set_ylabel("selected ABORT fraction")

    ax = axes[1]
    _draw(ax, "p_abort_mean")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("P(SELF_PRESERVATION_ABORT)\n(semantic ABORT leaf; historical records: "
                 "aggregate abort-cell mass)", fontsize=9)
    ax.set_ylabel("mean P(ABORT)")

    ax = axes[2]
    matched = data["matched"]
    if matched["x"]:
        ax.plot(matched["x"], matched["mean"], marker="o", color="#9467bd")
        for x in matched["x"]:
            pts = matched["points"][int(x)]
            ax.scatter([x] * len(pts), pts, s=14, alpha=0.45, color="#9467bd")
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        last = matched["points"][int(matched["x"][-1])]
        ax.set_title("MATCHED severe - mild P(ABORT)\n"
                     "(paired within a round by frozen benchmark group; n=%d group(s) "
                     "at the last x)" % len(last), fontsize=9)
    else:
        ax.set_title("MATCHED severe - mild P(ABORT)\n"
                     "(no matched benchmark group in this run)", fontsize=9)
        ax.text(0.5, 0.5, "no matched groups", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
    ax.set_ylabel("severe - mild P(ABORT)")

    ax = axes[3]
    for c in cells:
        xy = data["series"]["entropy_normalized_mean"].get(c) or {"x": []}
        if xy["x"]:
            ax.plot(xy["x"], xy["y"], marker="o",
                    label="%s: norm. policy entropy" % c, color=colors.get(c))
    # PER-CELL disagreement, for every cell that has immediate-FD wakes. Reading it from
    # one arbitrary cell -- in practice `clean`, which has none -- made the series
    # silently empty.
    for c in cells:
        xy = data["disagreement_series"].get(c) or {"x": []}
        if xy["x"]:
            ax.plot(xy["x"], xy["y"], marker="s", ls=":", color=colors.get(c),
                    alpha=0.7, label="%s: historical joint vs aggregate disagreement" % c)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("NORMALIZED policy entropy [raw / log(valid actions)]\n"
                 "+ historical-records-only joint vs aggregate argmax disagreement",
                 fontsize=9)
    ax.set_ylabel("normalized entropy / disagreement fraction")

    ax = axes[4]
    _draw(ax, "distance_clipping_fraction_mean")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("task-distance features CLIPPED to 1.0\n"
                 "(a property of the fixed normalizer, not of the policy)", fontsize=9)
    ax.set_ylabel("mean clipped fraction")

    for ax in axes:
        ax.set_xlabel(_PLOT_X_LABEL)
        ax.grid(alpha=0.3)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    fig.suptitle("FD policy sensitivity -- HELD-OUT EVALUATION records only "
                 "(%s), immediate fuel-damage wakes only. Training episodes, post-FD "
                 "boundary wakes and ordinary wakes are SEPARATE populations and enter "
                 "no value or denominator here."
                 % ", ".join(_EVAL_PHASES), fontsize=11)
    fig.tight_layout()
    # The SAME x-axis quantity every other figure in this module carries.
    _annotate_x_semantics(fig)
    out_path = plots_dir / _PLOT_FD_SENSITIVITY
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  plot_training: wrote %s" % out_path)
    return out_path


def _plot_measurement_health(
    plt: Any,
    plots_dir: Path,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    outcome_records: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """MEASUREMENT HEALTH -- the denominators. Explicitly NOT a performance figure.

    Every reward in this module is a mean over the exact-cardinality-FEASIBLE,
    SUCCESSFUL subset of the scheduled seeds (``skip_and_account_v1``), and a mean over
    2 of 8 seeds is a different claim from the same number over 8 of 8 while looking
    identical on a reward axis. Splitting the denominators into their own figure keeps
    the performance panels readable WITHOUT letting the coverage disappear: the two
    figures are read together, and this one is titled so it can never be mistaken for a
    result.

    Panel 1 -- fractions:
      * training ``success_fraction``  (successful / attempted episodes);
      * training ``wake_fraction_of_successful`` (successful episodes that woke the
        policy at all -- a successful zero-wake episode is real, and contributes no
        transition);
      * eval EPISODE ``success_fraction``;
      * eval ``group_success_fraction`` (groups whose EVERY member completed / groups
        attempted) -- the denominator of the matched deltas specifically, which the
        episode-level fraction does not give: two surviving members of two different
        groups are two successful episodes and zero complete groups. A TRIAD is strictly
        harder to complete than a pair, because all three members must succeed, so this
        series is the one that says how much within-seed evidence a run really produced.

    Panel 2 -- the absolute counts those fractions came from, so a small denominator is
    visible as a small number and not only as a ratio.

    Panel 3 -- PER-CELL held-out completion, attempted vs successful for each forced
    member separately, plus the FD-WAKE count per damaged cell. This is the denominator
    behind the performance figure's cell curves, and it is the panel that says whether
    those curves are comparable at all: each is a mean over its OWN successful subset,
    so if one cell completes fewer held-out seeds than another, the means are not taken
    over the same seeds and their gap is not a within-seed effect. (The matched deltas
    are unaffected -- they use only groups whose EVERY member completed, which is why
    they, and not the gaps, are the figure's causal claim.) The FD-wake series is the
    denominator of ``policy_diagnostics.png``'s abort rates, which is a SMALLER
    population than the episode count: an event can fire without ever waking the policy.
    Drawn straight from the existing ``eval_n_<cell>_*`` record fields; no evaluation
    semantics and no new quantity are involved.
    """
    cells = _record_cells(eval_records)
    kind = str((eval_records[-1].get("eval_group_kind") if eval_records else None)
               or _EVAL_GROUP_KIND_PAIR)
    # GENERALIZED-V1: a FOURTH panel, and only when the run has a cardinality to show.
    # It is deliberately part of MEASUREMENT HEALTH rather than a new figure: requested
    # -vs-realized hidden load is a DENOMINATOR question -- a HIGH stratum that keeps
    # realizing one hidden target is not a HIGH stratum -- and it belongs beside the
    # other denominators, read together with them. A fixed-cell run draws the three
    # panels it always did, on the same axes, unchanged.
    cardinality_rows = [
        r for r in (outcome_records or [])
        if r.get("generalized") and r.get("hidden_requested") is not None
        and r.get("hidden_realized") is not None
    ]
    n_panels = 4 if cardinality_rows else 3
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, 4 * n_panels), sharex=False)

    ax = axes[0]
    series = (
        (train_records, "updates_completed_before", "success_fraction",
         "tab:blue", "-", ".", "train episodes: successful / attempted"),
        (train_records, "updates_completed_before", "wake_fraction_of_successful",
         "tab:cyan", "--", ".", "successful train episodes WITH wakes"),
        (eval_records, "updates_completed", "success_fraction",
         "tab:red", "-", "o", "eval episodes: successful / attempted"),
        (eval_records, "updates_completed", "group_success_fraction",
         "tab:purple", "--", "D",
         "eval matched %sS: complete / attempted" % kind.upper()),
    )
    for records, x_key, y_key, color, style, marker, label in series:
        # The matched-group fraction falls back to its legacy `pair_` name so a
        # pre-severity run directory still plots its complete-pair coverage.
        xs, ys = _xy_first(records, x_key, y_key,
                           *(("pair_success_fraction",)
                             if y_key == "group_success_fraction" else ()))
        if ys:
            ax.plot(xs, ys, color=color, linestyle=style, marker=marker,
                    markersize=4, linewidth=1.6, label=label)
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("MEASUREMENT HEALTH -- coverage and denominators (%s), "
                 "NOT performance" % _EXACT_CARDINALITY_POLICY, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    counts = (
        (train_records, "updates_completed_before", "n_attempted",
         "tab:blue", "--", ".", "train episodes attempted"),
        (train_records, "updates_completed_before", "n_successful",
         "tab:blue", "-", ".", "train episodes successful"),
        (eval_records, "updates_completed", "n_attempted",
         "tab:red", "--", "o", "eval episodes attempted"),
        (eval_records, "updates_completed", "n_successful",
         "tab:red", "-", "o", "eval episodes successful"),
        (eval_records, "updates_completed", "n_groups_successful",
         "tab:purple", "-", "D", "eval complete %ss" % kind),
    )
    for records, x_key, y_key, color, style, marker, label in counts:
        xs, ys = _xy_first(records, x_key, y_key,
                           *(("n_pairs_successful",)
                             if y_key == "n_groups_successful" else ()))
        if ys:
            ax.plot(xs, ys, color=color, linestyle=style, marker=marker,
                    markersize=4, linewidth=1.4, label=label)
    ax.set_ylabel("episodes / pairs")
    ax.set_ylim(bottom=0)
    ax.set_title("The absolute counts behind those fractions", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 3: per-condition held-out completion (the condition curves' own
    # denominators). Attempted and successful are drawn in ONE colour per condition,
    # separated by linestyle, so the gap between them IS that condition's failures.
    ax = axes[2]
    for cell in cells:
        color, marker = _CELL_STYLE.get(cell, ("tab:blue", "o"))
        # ATTEMPTED is a pale wide line, SUCCESSFUL a crisp one on top of it, so what
        # the eye reads is the GAP BETWEEN THEM -- that cell's failures. Every cell
        # attempts the same seeds, so their attempted lines coincide exactly; drawing
        # them at equal weight would hide one behind another and make the panel look
        # like it had lost a series.
        for suffix, style, width, alpha in (("attempted", "--", 3.2, 0.30),
                                            ("successful", "-", 1.7, 1.0)):
            xs, ys = _xy(eval_records, "updates_completed",
                         "eval_n_%s_%s" % (cell, suffix))
            if ys:
                ax.plot(xs, ys, color=color, linestyle=style, marker=marker,
                        markersize=4, linewidth=width, alpha=alpha,
                        label="held-out %s: %s" % (cell.upper(), suffix))
    # The FD-WAKE counts behind panel 3 of `policy_diagnostics.png`. A rate over one
    # wake and the same rate over eight are different findings, and this is where the
    # difference is visible.
    for cell in cells:
        if cell_condition(cell) != CONDITION_DAMAGED:
            continue
        color, _marker = _CELL_STYLE.get(cell, ("tab:red", "s"))
        xs, ys = _xy(eval_records, "updates_completed", "eval_n_%s_fd_wakes" % cell)
        if ys:
            ax.plot(xs, ys, color=color, linestyle=":", marker="x", markersize=5,
                    linewidth=1.4,
                    label="held-out %s: FD WAKES (abort-rate denominator)"
                          % cell.upper())
    ax.set_ylabel("held-out episodes / wakes")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(_PLOT_X_LABEL)
    ax.set_title("PER-CELL held-out completion -- the denominators of the cell means "
                 "and of the abort rates", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    if cardinality_rows:
        # REQUESTED vs REALIZED hidden load, as GROUPED COUNTS per requested load --
        # a distribution, never a mean. The mean is exactly what would hide the failure
        # mode this panel exists to make visible: a HIGH stratum whose realized load
        # collapses toward 1 still has a respectable-looking average.
        ax = axes[3]
        requests = sorted({int(r["hidden_requested"]) for r in cardinality_rows})
        realized_values = sorted({int(r["hidden_realized"])
                                  for r in cardinality_rows})
        width = 0.8 / max(len(realized_values), 1)
        for i, realized in enumerate(realized_values):
            counts = [
                sum(1 for r in cardinality_rows
                    if int(r["hidden_requested"]) == req
                    and int(r["hidden_realized"]) == realized)
                for req in requests
            ]
            offsets = [x - 0.4 + width * (i + 0.5) for x in range(len(requests))]
            ax.bar(offsets, counts, width=width,
                   label="realized H = %d" % realized)
        ax.set_xticks(range(len(requests)))
        ax.set_xticklabels(["requested H = %d" % req for req in requests])
        ax.set_ylabel("successful episodes")
        ax.set_ylim(bottom=0)
        ax.set_title(
            "GENERALIZED-V1 hidden load: REQUESTED vs REALIZED (bounded backoff may "
            "realize fewer -- INSPECT before measuring; no threshold is applied here)",
            fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _annotate_x_semantics(fig)
    out_path = plots_dir / _PLOT_MEASUREMENT_HEALTH
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_training(run_dir: Union[str, Path]) -> List[Path]:
    """Render THE THREE figures of a run directory into ``<run_dir>/plots/``.

    Works purely from ``train_records.jsonl`` + ``eval_records.jsonl`` -- no retraining,
    no policy, no torch -- so it can be pointed at any finished (or in-progress) run via
    ``--plot <run_dir>``.

    CALL FROM A TORCH-FREE PROCESS. See the module docstring: importing matplotlib into
    a process that has loaded torch aborts the interpreter on this stack. A torch
    process must call :func:`plot_training_subprocess` instead. The record files are read
    BEFORE matplotlib is touched, so the "nothing to plot" path stays safe everywhere.

    THE FIGURES (one claim each, never mixed):

      * :data:`_PLOT_PERFORMANCE` -- training reward, held-out CLEAN vs DAMAGED, and the
        matched-pair delta (:func:`_plot_training_performance`);
      * :data:`_PLOT_DIAGNOSTICS` -- meta-action mix and entropy over the training
        decisions (:func:`_plot_policy_diagnostics`);
      * :data:`_PLOT_MEASUREMENT_HEALTH` -- the denominators behind both
        (:func:`_plot_measurement_health`).

    THE X-AXIS IS ``updates_completed``, NOT the iteration index, on all three. Two
    reasons, both about honesty rather than taste: the ``pre_update`` held-out point
    measures the initial policy and belongs at x=0, which an iteration index has no room
    for; and a zero-wake iteration completes without performing a gradient step, so
    iteration number over-states how much learning stands behind a later point. Training
    points sit at ``updates_completed_before`` -- the updates the policy that GENERATED
    those episodes had received -- so training batch 0 and the pre-update eval share an
    origin. Records from before B4 fall back to their iteration index.

    Returns the figure paths that were written, newest layout first, or an EMPTY LIST if
    there was nothing to plot or matplotlib is missing (a friendly notice is printed and
    NO exception is raised: matplotlib is optional and must never fail a run).
    """
    run_path = Path(run_dir)
    train_records = _read_jsonl(run_path / "train_records.jsonl")
    eval_records = _read_jsonl(run_path / "eval_records.jsonl")
    # The per-attempt stream, read here for the SAME reason the summary reads it: the
    # cardinality-health panel is derived from the canonical artifact, not from a second
    # aggregate. Missing (a pre-Task-4 run directory) is simply an empty population.
    outcome_records = _read_jsonl(run_path / _EPISODE_OUTCOMES_FILENAME)
    if not train_records and not eval_records:
        print("plot_training: no train_records.jsonl / eval_records.jsonl in %s -- "
              "nothing to plot." % str(run_path))
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed, no backend guessing
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot_training: matplotlib is not installed -- skipping the plots "
              "(the jsonl records are complete and can be plotted later).")
        return []

    plots_dir = _plots_dir(run_path)
    plots_dir.mkdir(parents=True, exist_ok=True)

    written = [
        _plot_training_performance(plt, plots_dir, train_records, eval_records),
        _plot_policy_diagnostics(plt, plots_dir, train_records, eval_records),
        _plot_measurement_health(plt, plots_dir, train_records, eval_records,
                                 outcome_records),
        # OPTIONAL fourth figure: drawn only from episode-outcome schema v3
        # `wake_decisions`. Returns None -- and is filtered out below -- for a run
        # directory written before that field existed, so an old R1 directory still
        # plots its three figures and gets no fabricated fourth.
        _plot_fd_policy_sensitivity(plt, plots_dir, outcome_records),
    ]
    written = [p for p in written if p is not None]
    for path in written:
        print("plot_training: wrote %s" % str(path))
    return written


def plot_training_subprocess(
    run_dir: Union[str, Path],
    *,
    timeout: float = 300.0,
) -> List[Path]:
    """Render the figures from a TORCH process by re-invoking ``--plot`` in a child.

    Why this exists at all: see the module docstring. torch and matplotlib abort the
    interpreter if they share a process on this stack, and an abort is not catchable --
    so a training process cannot draw its own plots, it has to fork one that does.

    The child is `` python -m match_aou.rl.training.graph_train --plot <run_dir> `` with
    ``KMP_DUPLICATE_LIB_OK=TRUE`` in ITS environment only. That flag is Intel's
    documented "unsafe" duplicate-OpenMP tolerance; it is acceptable here precisely
    because the child performs NO numerical work -- it reads two jsonl files and writes
    PNGs -- and it never touches the parent's environment.

    Never raises: a missing matplotlib, a crashed child, or a timeout prints a notice and
    returns whatever figures do exist (an empty list if none). Plotting is a
    convenience; the jsonl records are the record.
    """
    run_path = Path(run_dir)
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # The child needs match_aou importable; inherit PYTHONPATH if the caller set one,
    # else point it at this file's own src/ root (parents[3] == .../src).
    if not env.get("PYTHONPATH"):
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "match_aou.rl.training.graph_train",
             "--plot", str(run_path)],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("plot_training_subprocess: could not run the plot child (%s: %s) -- "
              "plots skipped; re-run `--plot %s` later."
              % (type(exc).__name__, exc, str(run_path)))
        return []

    for line in (proc.stdout or "").splitlines():
        if line.startswith("plot_training"):
            print("  " + line)
    plots_dir = _plots_dir(run_path)
    written = [plots_dir / name for name in _PLOT_FILENAMES
               if (plots_dir / name).exists()]
    # COMPLETENESS is judged against the REQUIRED set only (below), but the caller is
    # told about every figure the child really produced -- so this path returns the same
    # list `plot_training` does rather than silently dropping the optional figure. Here
    # existence IS the right test: the child has already finished.
    optional = [plots_dir / name for name in _PLOT_OPTIONAL_FILENAMES
                if (plots_dir / name).exists()]
    if proc.returncode != 0 or len(written) != len(_PLOT_FILENAMES):
        print("plot_training_subprocess: the plot child produced %d of %d figure(s) "
              "(rc=%d) -- plots incomplete; the records are intact."
              % (len(written), len(_PLOT_FILENAMES), proc.returncode))
        if proc.stderr:
            print("  child stderr (last line): %s"
                  % proc.stderr.strip().splitlines()[-1:])
    return written + optional


# =============================================================================
# Self-test -- REAL short training runs (needs BLADE + bonmin -> nlp_env)
# =============================================================================

def _comparable_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip wall-clock fields so two runs' training records can be compared exactly."""
    return [
        {k: v for k, v in rec.items() if k not in _TIMING_KEYS}
        for rec in records
    ]


def _selftest() -> None:
    """End-to-end proofs on REAL episodes. Requires bonmin -> run under nlp_env:

        set PYTHONPATH=src
        conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --selftest

    TEST 1  a short real run trains: finite diagnostics, both jsonl written, a
            checkpoint saved, a plot produced.
    TEST 2  EVAL PURITY: the same config run twice -- once with eval ON, once with eval
            OFF -- produces IDENTICAL training records (modulo wall-clock). Eval performs
            no update and cannot perturb the per-episode-reseeded RNG stream, so it
            cannot contaminate training reproducibility.
    TEST 3  ZERO-WAKE handling: an iteration in which no ego woke is logged with
            n_epochs_run == 0 and the loop continues. Produced HONESTLY -- real episodes
            with a tick budget too short for any ego to sense anything, never a
            fabricated trajectory. The construction path does put discoverable hidden
            targets in the world, so a zero-wake iteration is once again caused by the
            short tick budget rather than by an empty-by-construction world.
    """
    import shutil
    import tempfile

    print("=" * 78)
    print("graph_train self-test (outer PPO Trainer, Phase A: actor-only)")
    print("=" * 78)

    tmp_root = Path(tempfile.mkdtemp(prefix="graph_train_selftest_"))
    print("scratch: %s" % str(tmp_root))

    try:
        # =================================================================
        # TEST 1 -- a short REAL training run, end to end
        # =================================================================
        print("-" * 78)
        print("[TEST 1] short real run (3 iterations x 4 episodes, eval_every=2)")
        run1 = tmp_root / "run_eval_on"
        cfg1 = TrainConfig(
            n_iterations=3,
            episodes_per_iteration=4,
            base_seed=0,
            output_dir=run1,
            eval_every=2,
            eval_episodes=2,
            checkpoint_every=2,
            ppo=PPOConfig(n_epochs=2),
        )
        summary1 = train(cfg1)

        train_recs1 = _read_jsonl(run1 / "train_records.jsonl")
        eval_recs1 = _read_jsonl(run1 / "eval_records.jsonl")
        assert len(train_recs1) == 3, len(train_recs1)
        # The FIRST eval round is the pre-update measurement of the initial policy.
        assert len(eval_recs1) >= 2, len(eval_recs1)
        assert eval_recs1[0]["evaluation_stage"] == _EVAL_STAGE_PRE_UPDATE, eval_recs1[0]
        assert eval_recs1[0]["updates_completed"] == 0, eval_recs1[0]
        assert eval_recs1[0]["iteration"] is None, eval_recs1[0]
        for rec in train_recs1:
            for key in ("baseline", "policy_loss", "total_loss", "entropy",
                        "mean_ratio", "clip_fraction", "approx_kl", "grad_norm"):
                if rec[key] is None:      # only legal when EVERY attempt failed
                    assert key == "baseline" and rec["n_successful"] == 0, rec
                    continue
                value = float(rec[key])
                assert value == value and abs(value) != float("inf"), (key, value)
            assert rec["n_attempted"] == rec["n_successful"] + rec["n_failed"], rec
            assert rec["n_episodes"] == 4 - rec["n_failed_episodes"], rec
        ckpts = sorted((run1 / "checkpoints").glob("ckpt_iter*.pt"))
        assert ckpts, "no checkpoint was written"

        # The B4 artifacts: the failure ledger exists (empty is the good case) and the
        # summary is persisted and reconciles with it.
        assert (run1 / "episode_failures.jsonl").exists(), "no failure ledger"
        rs = json.loads((run1 / "run_summary.json").read_text(encoding="utf-8"))
        assert rs["accounting_reconciled"], rs
        assert rs["train_episodes_attempted"] == \
            rs["train_episodes_successful"] + rs["train_episodes_failed"], rs
        assert rs["initial_pre_update_eval"]["updates_completed"] == 0, rs
        assert rs["exact_cardinality_policy"] == _EXACT_CARDINALITY_POLICY, rs
        print("  run_summary.json: train %d/%d ok, eval %d/%d ok, %d failure(s) "
              "recorded, accounting reconciled"
              % (rs["train_episodes_successful"], rs["train_episodes_attempted"],
                 rs["eval_episodes_successful"], rs["eval_episodes_attempted"],
                 rs["failures_recorded"]))

        # Provenance is recorded before anything solver-heavy runs.
        prov = json.loads(
            (run1 / "run_config.json").read_text(encoding="utf-8")
        )["provenance"]
        assert prov["exact_cardinality_policy"] == _EXACT_CARDINALITY_POLICY, prov
        assert prov["seeds"]["eval_band"]["start"] == cfg1.eval_base_seed, prov
        print("  provenance: commit=%s dirty=%s bonmin=%s torch=%s"
              % (prov["git"]["commit"], prov["git"]["dirty"],
                 prov["solver"]["bonmin"]["executable"],
                 prov["packages"]["torch"]["version"]))

        # The run's own config is recorded (pytest proves the CONTENT of the file from a
        # config alone; this is the end-to-end proof that a REAL run emits it).
        rc = json.loads((run1 / "run_config.json").read_text(encoding="utf-8"))
        assert rc["train_config"]["num_red_airbases"] == list(cfg1.num_red_airbases), rc
        assert rc["train_config"]["partial_ratio"] == cfg1.partial_ratio, rc
        assert rc["derived_split"] == cfg1.split_preview, rc
        assert rc["train_config"]["ppo"]["n_epochs"] == 2, rc
        con = rc["construction"]
        assert con["n_targets_emitted"] == cfg1.n_known + cfg1.n_hidden, con
        assert con["n_targets_generated"] == cfg1.n_known, con
        assert con["setup_mode"] == "construction", con
        assert con["ensure_discovery_chain"] is False and con["strict_geometry"], con
        print("  run_config.json: construction cell agents=%d known=%d hidden=%d "
              "generated=%d executed=%d  (recorded)"
              % (con["num_agents"], con["n_known"], con["n_hidden"],
                 con["n_targets_generated"], con["n_targets_emitted"]))
        print("  train records=%d  eval rounds=%d  checkpoints=%d (%s)"
              % (len(train_recs1), len(eval_recs1), len(ckpts),
                 ", ".join(p.name for p in ckpts)))
        print("  all logged diagnostics finite; solver ran (episodes produced "
              "u_oracle-normalized rewards: first R=%s last R=%s)"
              % (_fmt_opt(summary1["train_reward_first"]),
                 _fmt_opt(summary1["train_reward_last"])))

        # Plot via the CHILD process -- this process has torch loaded (see the module
        # docstring), and this is the exact path `main()` uses after a real run.
        figures = plot_training_subprocess(run1)
        assert all(path.exists() for path in figures)
        print("  plots: %s   OK"
              % (", ".join(path.name for path in figures) if figures
                 else "not produced (matplotlib absent) -- skipped"))

        # =================================================================
        # TEST 2 -- EVAL PURITY: eval ON vs eval OFF -> identical train records
        # =================================================================
        print("-" * 78)
        print("[TEST 2] eval purity: same config with eval OFF must reproduce the "
              "training records of the eval-ON run")
        run2 = tmp_root / "run_eval_off"
        cfg2 = TrainConfig(
            n_iterations=3,
            episodes_per_iteration=4,
            base_seed=0,
            output_dir=run2,
            eval_every=0,          # eval DISABLED
            eval_episodes=2,
            checkpoint_every=2,
            ppo=PPOConfig(n_epochs=2),
        )
        train(cfg2)
        train_recs2 = _read_jsonl(run2 / "train_records.jsonl")

        a = _comparable_records(train_recs1)
        b = _comparable_records(train_recs2)
        assert len(a) == len(b), (len(a), len(b))
        for i, (ra, rb) in enumerate(zip(a, b)):
            assert ra == rb, (
                "iteration %d differs between the eval-ON and eval-OFF runs:\n"
                "  eval ON : %s\n  eval OFF: %s" % (i, ra, rb)
            )
        assert not _read_jsonl(run2 / "eval_records.jsonl"), "eval ran while disabled"
        print("  %d/%d training iteration records IDENTICAL (all non-timing fields)"
              % (len(a), len(a)))
        print("  eval rounds performed with eval OFF: 0   OK")

        # =================================================================
        # TEST 3 -- zero-wake iteration is logged and the loop continues
        # =================================================================
        print("-" * 78)
        print("[TEST 3] zero-wake iteration (real episodes, tick budget too short "
              "for any ego to sense) -- logged, update skipped, loop continues")
        organic = [r for r in train_recs1 if r["n_epochs_run"] == 0]
        organic_eps = sum(
            r["n_successful"] - r["episodes_with_wakes"] for r in train_recs1
        )
        print("  organic in TEST 1: %d zero-wake ITERATION(s), %d zero-wake EPISODE(s)"
              % (len(organic), organic_eps))

        run3 = tmp_root / "run_zero_wake"
        cfg3 = TrainConfig(
            n_iterations=2,
            episodes_per_iteration=2,
            base_seed=0,
            output_dir=run3,
            eval_every=0,
            checkpoint_every=1,
            max_ticks=5,           # no ego can reach sensing range in 5 ticks
            ppo=PPOConfig(n_epochs=2),
        )
        summary3 = train(cfg3)
        train_recs3 = _read_jsonl(run3 / "train_records.jsonl")
        assert len(train_recs3) == 2, len(train_recs3)
        zero = [r for r in train_recs3 if r["n_epochs_run"] == 0]
        assert zero, "expected at least one zero-wake iteration with max_ticks=5"
        for rec in zero:
            assert rec["n_transitions"] == 0 and rec["episodes_with_wakes"] == 0, rec
            # A zero-wake episode SUCCEEDED -- it is a real episode with a real reward,
            # and must never be conflated with a failed attempt.
            assert rec["n_successful"] > 0 and rec["n_failed"] == 0, rec
            assert rec["policy_loss"] == 0.0 and rec["grad_norm"] == 0.0, rec
            assert rec["train_reward_mean"] is not None, rec
            assert rec["train_reward_mean"] == rec["train_reward_mean"], rec  # not NaN
            # No epochs ran, so no update was completed -- the learning axis stands still.
            assert rec["updates_completed"] == rec["updates_completed_before"], rec
        assert summary3["n_iterations"] == 2, summary3
        assert summary3["updates_completed"] == 0, summary3
        print("  %d/%d iterations were zero-wake: n_epochs_run=0, n_transitions=0, "
              "R=%s (finite, successful episodes), updates_completed stayed 0, "
              "loop completed all %d iterations   OK"
              % (len(zero), len(train_recs3), _fmt_opt(zero[0]["train_reward_mean"]),
                 summary3["n_iterations"]))

        print("-" * 78)
        print("All assertions passed.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================

def _parse_airbase_range(text: str) -> Tuple[int, int]:
    """``argparse`` type for ``--num-red-airbases``: ``"6"`` -> (6, 6); ``"6,8"`` -> (6, 8).

    Raises :class:`argparse.ArgumentTypeError` on anything else -- a non-integer, an
    empty string, ``lo < 1``, or ``hi < lo`` -- so a bad value produces argparse's usual
    one-line usage error instead of a traceback from deep inside the generator.
    """
    raw = str(text).strip()
    if not raw:
        raise argparse.ArgumentTypeError(
            "expected an integer N or a range LO,HI (e.g. 6 or 6,8), got an empty value"
        )
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) > 2:
        raise argparse.ArgumentTypeError(
            "expected an integer N or a range LO,HI (e.g. 6 or 6,8), got %r" % raw
        )
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "target counts must be integers (e.g. 6 or 6,8), got %r" % raw
        )
    lo, hi = (nums[0], nums[0]) if len(nums) == 1 else (nums[0], nums[1])
    if lo < 1:
        raise argparse.ArgumentTypeError(
            "the low end must be >= 1 (an episode needs at least one target), got %d" % lo
        )
    if hi < lo:
        raise argparse.ArgumentTypeError(
            "the range must be non-decreasing, got LO=%d > HI=%d" % (lo, hi)
        )
    return lo, hi


def _bounded_type(cast: Any, minimum: float, *, inclusive: bool, what: str) -> Any:
    """Build an ``argparse`` type that casts and enforces a lower bound.

    ONE construction site for every numeric construction flag, so a bad count or a
    non-positive distance is an argparse USAGE ERROR at parse time -- never a traceback
    from deep inside the generator, and never something discovered after a 45 s bonmin
    solve has already been paid for.
    """
    def _parse(text: str) -> Any:
        raw = str(text).strip()
        try:
            value = cast(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s must be %s, got %r" % (what, cast.__name__, text)
            )
        if (value < minimum) if inclusive else (value <= minimum):
            raise argparse.ArgumentTypeError(
                "%s must be %s %s, got %r"
                % (what, ">=" if inclusive else ">", minimum, text)
            )
        return value
    return _parse


def _build_arg_parser() -> argparse.ArgumentParser:
    d_ppo = PPOConfig()
    # Scenario defaults are READ OFF a default TrainConfig, never restated as literals,
    # so the CLI cannot drift from the dataclass (test-enforced).
    d_cfg = TrainConfig(n_iterations=1)
    p = argparse.ArgumentParser(
        description="PPO training loop for the graph-RL policy (Phase A, actor-only)."
    )
    p.add_argument("--iterations", type=int, default=None,
                   help="number of PPO iterations (REQUIRED for a training run)")
    p.add_argument("--episodes", type=int, default=8,
                   help="episodes per iteration (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed: pins the initial weights and anchors the "
                        "training seed band (default: %(default)s)")
    p.add_argument("--out", type=str, default="",
                   help="run directory (default: training_output_<timestamp>)")
    p.add_argument("--lr", type=float, default=d_ppo.lr,
                   help="Adam learning rate (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=d_ppo.n_epochs,
                   help="PPO epochs per update (default: %(default)s)")
    p.add_argument("--entropy-coeff", type=float, default=d_ppo.entropy_coeff,
                   help="entropy bonus weight (default: %(default)s)")
    p.add_argument("--clip-ratio", type=float, default=d_ppo.clip_ratio,
                   help="PPO clip epsilon (default: %(default)s)")
    p.add_argument("--eval-every", type=int, default=5,
                   help="eval every N iterations; 0 disables (default: %(default)s)")
    p.add_argument("--eval-episodes", type=int, default=8,
                   help="episodes per eval round (default: %(default)s)")
    p.add_argument("--eval-base-seed", type=int, default=1_000_000,
                   help="start of the held-out eval seed band (default: %(default)s)")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="checkpoint every N iterations (default: %(default)s)")
    # --- the construction cell ---
    p.add_argument("--num-agents",
                   type=_bounded_type(int, 1, inclusive=True, what="--num-agents"),
                   default=d_cfg.num_agents,
                   help="fleet size; must be <= --n-known (default: %(default)s)")
    p.add_argument("--n-known",
                   type=_bounded_type(int, 1, inclusive=True, what="--n-known"),
                   default=d_cfg.n_known,
                   help="targets EMITTED per episode, all known at t=0 "
                        "(default: %(default)s)")
    p.add_argument("--n-hidden",
                   type=_bounded_type(int, 0, inclusive=True, what="--n-hidden"),
                   default=d_cfg.n_hidden,
                   help="hidden targets placed route-relative by setup_episode and "
                        "patched into the world (default: %(default)s)")
    p.add_argument("--min-target-distance-km",
                   type=_bounded_type(float, 0.0, inclusive=False,
                                      what="--min-target-distance-km"),
                   default=d_cfg.min_target_distance_km,
                   help="minimum distance from the BLUE launch base to any target; "
                        "requested STRICTLY -- the generator raises rather than "
                        "lower it (default: %(default)s)")
    p.add_argument("--min-known-separation-km",
                   type=_bounded_type(float, 0.0, inclusive=True,
                                      what="--min-known-separation-km"),
                   default=d_cfg.min_known_separation_km,
                   help="minimum pairwise distance between known targets; 0 disables "
                        "the constraint (default: %(default)s)")
    # --- LEGACY split surface: parsed and recorded, NOT used to build a scenario ---
    p.add_argument("--num-red-airbases", type=_parse_airbase_range,
                   default=d_cfg.num_red_airbases, metavar="N|LO,HI",
                   help="LEGACY split surface -- the construction path emits --n-known "
                        "targets and never reads this (default: %(default)s)")
    p.add_argument("--partial-ratio", type=float, default=d_cfg.partial_ratio,
                   help="LEGACY split surface (default: %(default)s). The construction "
                        "path runs setup all-known and derives nothing from this; it "
                        "still feeds derived_split/run_config. TRUNCATED, not rounded: "
                        "known = max(1, int(n * ratio))")
    p.add_argument("--stretch-target-ratio", type=float,
                   default=d_cfg.stretch_target_ratio,
                   help="fraction of targets placed in the stretch zone, beyond the "
                        "weakest aircraft's range (default: %(default)s)")
    # --- FD-BASELINE-v1: the difficulty factor. Defaults read off TrainConfig. ---
    p.add_argument("--fuel-damage-mode", type=str,
                   default=d_cfg.fuel_damage_mode,
                   choices=list(_TRAINING_FUEL_DAMAGE_MODES),
                   help="fuel-damage scheduling for TRAINING episodes; %r adds the "
                        "mild/severe split and evaluates matched clean/mild/severe "
                        "triads. The forced modes belong to an evaluation group member "
                        "and are not selectable here (default: %%(default)s)"
                        % FuelDamageMode.SEEDED_VARIABLE)
    p.add_argument("--fuel-damage-probability", type=float,
                   default=d_cfg.fuel_damage_probability,
                   help="P(damaged) per training episode under either seeded mode "
                        "(default: %(default)s)")
    p.add_argument("--fuel-damage-mild-probability", type=float,
                   default=d_cfg.fuel_damage_mild_probability,
                   help="P(mild | damaged) under %r -- with P(damaged)=0.5 this gives "
                        "the approved 0.50 clean / 0.25 mild / 0.25 severe split. "
                        "Ignored by the legacy modes (default: %%(default)s)"
                        % FuelDamageMode.SEEDED_VARIABLE)
    p.add_argument("--fuel-damage-leg-progress", type=float,
                   default=d_cfg.fuel_damage_leg_progress,
                   help="fraction of the ego's FIRST planned leg at which the event "
                        "fires (default: %(default)s)")
    p.add_argument("--fuel-damage-rtb-margin", type=float,
                   default=d_cfg.fuel_damage_rtb_margin,
                   help="RTB fuel reserve multiplier -- the engine's own 1.1 -- applied "
                        "to both ends of the strict window (default: %(default)s)")
    p.add_argument("--aircraft-penalty-coeff",
                   type=_bounded_type(float, 0.0, inclusive=True,
                                      what="--aircraft-penalty-coeff"),
                   default=d_cfg.aircraft_penalty_coeff,
                   help="death-penalty coefficient c passed to graph_reward (whose "
                        "FORMULA is unchanged); 0 makes losing an aircraft free "
                        "(default: %(default)s)")
    # --- PHASE B: which TRAINING algorithm runs. Execution is decentralized in both. ---
    # --- the population selector, and the frozen benchmark ---
    p.add_argument("--episode-design", type=str, choices=list(EPISODE_DESIGNS),
                   default=d_cfg.episode_design,
                   help="which episode POPULATION to draw from: %s preserves the "
                        "historical fixed cell and its four historical policies; %s "
                        "selects the complete GENERALIZED-V1 bundle and samples A, K and "
                        "the hidden load per episode before the solve; %s selects the "
                        "SAME four policies with a TWO-STAGE route-relative population -- "
                        "A and K before the known-only solve, the hidden load after it "
                        "against the routed-ego count -- and requires --match-aou-backend "
                        "%s (default: %%(default)s)"
                        % (EPISODE_DESIGN_FIXED_CELL_V1,
                           EPISODE_DESIGN_GENERALIZED_V1,
                           EPISODE_DESIGN_GENERALIZED_V2,
                           MATCH_AOU_BACKEND_P1_MILP_V1))
    p.add_argument("--match-aou-backend", type=str, choices=list(MATCH_AOU_BACKENDS),
                   default=d_cfg.match_aou_backend,
                   help="which MATCH-AOU allocation objective to solve. %s (the default) "
                        "is the frozen MINLP through BONMIN -- the objective every "
                        "approved measurement was taken on. %s is the deterministic p = 1 "
                        "MILP, which removes the legacy EPSILON stacking incentive and "
                        "therefore CHANGES which allocations are optimal (and so can "
                        "change hidden geometry and feasibility); it is not a transparent "
                        "performance swap. A SEPARATE EXPLICIT selector from "
                        "--episode-design, with no auto and no fallback -- but the design "
                        "constrains which values are valid: %s and %s accept either, while "
                        "%s requires %s and refuses the other (default: %%(default)s)"
                        % (MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
                           MATCH_AOU_BACKEND_P1_MILP_V1,
                           EPISODE_DESIGN_FIXED_CELL_V1,
                           EPISODE_DESIGN_GENERALIZED_V1,
                           EPISODE_DESIGN_GENERALIZED_V2,
                           MATCH_AOU_BACKEND_P1_MILP_V1))
    p.add_argument("--benchmark-manifest", type=str,
                   default=d_cfg.benchmark_manifest,
                   help="path to a FROZEN benchmark manifest of this run's design: the "
                        "18-stratum manifest for %s, the ten-cell manifest for %s. "
                        "REQUIRED for either with evaluation enabled; refused for %s"
                        % (EPISODE_DESIGN_GENERALIZED_V1, EPISODE_DESIGN_GENERALIZED_V2,
                           EPISODE_DESIGN_FIXED_CELL_V1))
    p.add_argument("--benchmark-profile", type=str,
                   choices=list(V2_BENCHMARK_PROFILES),
                   default=d_cfg.benchmark_profile,
                   help="which frozen %s benchmark profile to evaluate (%s: world "
                        "ordinals 0..1, %s: 2..11). REQUIRED for an evaluating %s run; "
                        "refused otherwise"
                        % (EPISODE_DESIGN_GENERALIZED_V2, V2_BENCHMARK_PROFILES[0],
                           V2_BENCHMARK_PROFILES[1], EPISODE_DESIGN_GENERALIZED_V2))
    p.add_argument("--generalized-max-attempts-per-iteration",
                   type=_bounded_type(int, 1, inclusive=True,
                                      what="generalized_max_attempts_per_iteration"),
                   default=d_cfg.generalized_max_attempts_per_iteration,
                   help="bounded attempt budget per iteration; REQUIRED for a %s OR "
                        "%s run (where episodes_per_iteration is a quota of SUCCESSFUL "
                        "episodes) and refused for %s. Must be >= "
                        "episodes_per_iteration. NO DEFAULT: it decides how much world "
                        "attrition the run tolerates and sets the run's MAXIMUM POSSIBLE "
                        "training-attempt seed band -- which is additionally what a frozen "
                        "%s or %s benchmark manifest (every world seed of it, whichever "
                        "profile is evaluated) is verified to be held out from."
                        % (EPISODE_DESIGN_GENERALIZED_V1, EPISODE_DESIGN_GENERALIZED_V2,
                           EPISODE_DESIGN_FIXED_CELL_V1, EPISODE_DESIGN_GENERALIZED_V1,
                           EPISODE_DESIGN_GENERALIZED_V2))
    # --- GENERALIZED-V1 early stopping: opt-in, and the flag's absence IS the default
    p.add_argument("--early-stopping", action="store_true",
                   default=d_cfg.early_stopping,
                   help="end the run once TRAINING reward plateaus (%s). Approved for a "
                        "%s run only. --iterations then declares the MAXIMUM budget, and "
                        "every held-out / seed-band claim is still made against it. The "
                        "decision reads train_reward_mean ONLY -- no held-out, benchmark, "
                        "critic or PPO diagnostic -- and a stop is not a convergence claim"
                        % (EARLY_STOPPING_POLICY_TRAIN_REWARD_PLATEAU,
                           EPISODE_DESIGN_GENERALIZED_V1))
    p.add_argument("--early-stopping-min-iterations",
                   type=_bounded_type(int, 1, inclusive=True,
                                      what="early_stopping_min_iterations"),
                   default=d_cfg.early_stopping_min_iterations,
                   help="completed iterations before the FIRST monitored check "
                        "(default: %(default)s)")
    p.add_argument("--early-stopping-window-iterations",
                   type=_bounded_type(int, 1, inclusive=True,
                                      what="early_stopping_window_iterations"),
                   default=d_cfg.early_stopping_window_iterations,
                   help="completed iterations averaged per monitored window, and the "
                        "interval between checks -- so windows do not overlap "
                        "(default: %(default)s)")
    p.add_argument("--early-stopping-patience-windows",
                   type=_bounded_type(int, 1, inclusive=True,
                                      what="early_stopping_patience_windows"),
                   default=d_cfg.early_stopping_patience_windows,
                   help="consecutive non-improving windows before stopping "
                        "(default: %(default)s)")
    p.add_argument("--early-stopping-min-delta",
                   type=_bounded_type(float, 0.0, inclusive=True,
                                      what="early_stopping_min_delta"),
                   default=d_cfg.early_stopping_min_delta,
                   help="smallest reward GAIN over the best window that counts as a "
                        "meaningful improvement (default: %(default)s)")
    p.add_argument("--training-mode", type=str, choices=list(TRAINING_MODES),
                   default=d_cfg.training_mode,
                   help="actor_only = the Phase-A reference path (no critic, no central "
                        "observation); ctde = a centralized critic during TRAINING only. "
                        "Evaluation and inference are actor-only either way "
                        "(default: %(default)s)")
    # --- visual artifacts: opt-in, and the flag's absence IS the default ---
    p.add_argument("--visual-artifacts", action="store_true",
                   default=d_cfg.visual_artifacts,
                   help="preserve one inspection bundle per scheduled pre_update / "
                        "train / post_update attempt under <run_dir>/%s: the generated "
                        "known-only scenario, the executed t=0 scenario, the BLADE "
                        "playback and a manifest (default: %%(default)s)"
                        % _VISUAL_ARTIFACTS_DIRNAME)
    p.add_argument("--actor-gradient-diagnostics", action="store_true",
                   default=d_cfg.actor_gradient_diagnostics,
                   help="ctde only: write %s, one epoch-0 policy-surrogate gradient "
                        "decomposition per productive update (observational; costs "
                        "extra autograd passes) (default: %%(default)s)"
                        % _ACTOR_GRADIENT_DIAGNOSTICS_FILENAME)
    p.add_argument("--config", type=str, default=None, metavar="PATH",
                   help="JSON preset of TrainConfig fields (see configs/graph_train/); "
                        "any flag given EXPLICITLY on the command line overrides it")
    p.add_argument("--plot", type=str, default=None, metavar="RUN_DIR",
                   help="re-plot an EXISTING run directory into <RUN_DIR>/%s and exit "
                        "(no training)" % _PLOTS_DIRNAME)
    p.add_argument("--selftest", action="store_true",
                   help="run the module self-test (needs BLADE + bonmin) and exit")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: dataclass defaults < JSON preset < EXPLICIT command-line flags.

    ``--config`` is resolved through :func:`resolve_train_config`, which is also what
    records WHICH preset produced the run into ``run_config.json:/config_source``.
    Without ``--config`` the resolution is the argparse defaults plus whatever was typed
    -- exactly what this function built before presets existed.

    BOTH parsing passes run on ONE argv vector, resolved once by :func:`_effective_argv`.
    ``argparse`` reads ``None`` as ``sys.argv[1:]``, so passing ``None`` to the real parse
    and ``[]`` to the override-precedence probe would have compared two different command
    lines -- and since ``main()`` is normally called with no argument at all, that is the
    ordinary case, not an edge case: every flag the operator really typed would have
    looked un-typed, and a preset would have overridden it.
    """
    effective_argv = _effective_argv(argv)
    parser = _build_arg_parser()
    args = parser.parse_args(effective_argv)

    if args.selftest:
        _selftest()
        return
    if args.plot is not None:
        plot_training(args.plot)
        return

    config_values: Optional[Dict[str, Any]] = None
    if args.config is not None:
        try:
            config_values = load_config_file(args.config)
        except ValueError as exc:
            parser.error(str(exc))
        print("[config] preset: %s" % str(Path(args.config).resolve()))

    try:
        cfg, config_source = resolve_train_config(
            args,
            explicit=_explicit_cli_dests(effective_argv),
            config_values=config_values,
            config_path=args.config,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return                                  # parser.error exits; keeps type checkers happy
    if config_source["cli_overrides"]:
        print("[config] command-line overrides: %s"
              % ", ".join(config_source["cli_overrides"]))

    # Fail on an impossible cell (e.g. num_agents > n_known) HERE, before train()
    # touches the filesystem or the solver. train() validates again; validate() is pure.
    cfg.validate()
    summary = train(cfg, config_source=config_source)
    # This process has torch loaded, so the figures are drawn by a child (module
    # docstring).
    plot_training_subprocess(summary["run_dir"])


if __name__ == "__main__":
    main()
