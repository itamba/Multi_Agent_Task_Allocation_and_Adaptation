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
  * EVAL: a FIXED, disjoint band -- eval episode ``e`` always uses
    ``eval_base_seed + e``, the SAME E seeds on EVERY eval round. Evaluating the same
    held-out scenarios each round is what isolates policy improvement from scenario
    variance; a fresh eval sample each round would make the curve mostly noise.
    :meth:`TrainConfig.validate` REFUSES a config whose training band would reach into
    the eval band, so "held-out" is enforced, not hoped for.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .graph_episode_setup import (
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
    EpisodeRecord,
    PPOBuffer,
    PPOConfig,
    PPOUpdater,
    build_central_critic,
)
from .graph_reward import RewardConfig, compute_episode_reward
from .graph_tick_loop import build_policy, run_episode
from ..observation.central_graph_builder import CentralStateRecorder
from ..action.graph_action import MetaAction
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
_EPISODE_OUTCOME_VERSION = 1

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
    "training_mode": "training_mode",
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
        return FuelDamageParameters(
            mode=str(self.fuel_damage_mode if mode is None else mode),
            probability=float(self.fuel_damage_probability),
            leg_progress_threshold=float(self.fuel_damage_leg_progress),
            rtb_safety_margin=float(self.fuel_damage_rtb_margin),
            mild_probability=float(self.fuel_damage_mild_probability),
        )

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
            if float(self.ctde.value_coeff) < 0.0:
                raise ValueError(
                    "ctde.value_coeff must be >= 0, got %r" % (self.ctde.value_coeff,)
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
        if self.total_episodes > _EVAL_EPISODE_TAG_BASE:
            raise ValueError(
                "total training episodes (%d) reaches the eval scenario-tag base (%d): "
                "training and eval scenarios would collide by filename."
                % (self.total_episodes, _EVAL_EPISODE_TAG_BASE)
            )

        train_lo = int(self.base_seed)
        train_hi = train_lo + self.total_episodes            # exclusive
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
        "pipeline_stage": str(stage),
        "error_type": type(original).__name__,
        "error_message": str(original),
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
    """
    plan = out.fuel_damage_plan or {}
    outcome = out.fuel_damage_outcome or {}
    meta = outcome.get("wake_meta_action")
    return {
        "schema": _EPISODE_OUTCOME_SCHEMA,
        "schema_version": _EPISODE_OUTCOME_VERSION,
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
    }


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


def seed_bands(cfg: TrainConfig) -> Dict[str, Any]:
    """The run's two seed bands as HALF-OPEN ranges, plus the derivation formulas.

    Recorded rather than left implicit because "held out" is a claim about these two
    intervals: :meth:`TrainConfig.validate` refuses a config whose bands overlap, and
    this block is the artifact that lets a reviewer re-check that refusal without
    re-deriving the arithmetic. Half-open is stated in the payload
    (``stop`` EXCLUSIVE) so an off-by-one reading is not available.

    Computed from the same three pure functions the loop itself uses
    (:func:`global_episode_index`, :func:`train_seed`, :func:`eval_seed`), so this
    cannot describe a schedule other than the one that runs.
    """
    train_start = int(cfg.base_seed)
    band: Dict[str, Any] = {
        "train_band": {
            "start": train_start,
            "stop": train_start + int(cfg.total_episodes),
            "half_open": True,
            "count": int(cfg.total_episodes),
        },
        "train_seed_formula":
            "train_seed = base_seed + (iteration * episodes_per_iteration + j)",
        "eval_enabled": bool(cfg.eval_enabled),
        "eval_band": None,
        "eval_seed_formula": "eval_seed = eval_base_seed + e",
        "eval_band_is_fixed_across_rounds": True,
    }
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
        "solver": {"bonmin": _bonmin_provenance()},
        "seeds": seed_bands(cfg),
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


def write_run_config(
    run_dir: Path,
    cfg: TrainConfig,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    config_source: Optional[Dict[str, Any]] = None,
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
            collect_provenance(cfg) if provenance is None else provenance
        ),
        "construction": {
            "num_agents": int(cfg.num_agents),
            "n_known": int(cfg.n_known),
            "n_hidden": int(cfg.n_hidden),
            "n_targets_generated": int(cfg.n_known),
            "n_targets_emitted": cfg.n_targets_emitted,
            "min_target_distance_km": float(cfg.min_target_distance_km),
            "min_known_separation_km": float(cfg.min_known_separation_km),
            "detection_km": float(DETECTION_KM),
            "ensure_discovery_chain": False,
            "strict_geometry": True,
            "setup_mode": "construction",
        },
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


def build_variation_config(cfg: TrainConfig, seed: int) -> VariationConfig:
    """The ONE site that turns a :class:`TrainConfig` into the generator's input.

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
    return VariationConfig(
        include_sams=cfg.include_sams,
        num_aircraft=int(cfg.num_agents),
        num_red_airbases=int(cfg.n_known),
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


def _require_scheduled_cell(roster: "_TargetRoster", cfg: TrainConfig) -> None:
    """The roster must describe the cell the schedule asked for, or the run ABORTS.

    ``TrainConfig`` states the cell exactly -- ``n_known`` known targets, ``n_hidden``
    constructed ones, ``n_targets_emitted`` in the executed world -- and
    ``setup_episode``'s construction path already enforces that cardinality LOUDLY on its
    own side (exact ``len(placements) == n_hidden``, a known-target-loss check and a world
    cardinality check). So a roster that disagrees is not a scenario that came out
    differently; it is this module measuring the world wrongly, and every number derived
    from it -- confirmation counts, denominators, manifest target blocks -- is suspect.

    Raised as an :class:`EpisodeRosterError` for that reason: a measurement-integrity
    fault, not a scientific episode outcome. Checked BEFORE the fuel-damage plan and
    before ``run_episode``, so nothing is paid for and no partial measurement exists.
    """
    checks = (
        ("known targets", len(roster.known_ids), int(cfg.n_known)),
        ("hidden targets", len(roster.hidden_ids), int(cfg.n_hidden)),
        ("executed targets", int(roster.total), int(cfg.n_targets_emitted)),
    )
    wrong = ["%s: observed %d, scheduled %d" % (what, got, want)
             for what, got, want in checks if got != want]
    if wrong:
        raise EpisodeRosterError(
            "the t=0 roster does not describe the scheduled cell (%s): the episode was "
            "measured against a world the configuration did not ask for"
            % "; ".join(wrong)
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

    # Claimed BEFORE generation: the known-only scenario is the first artifact, and a
    # directory collision must be discovered before an episode is paid for.
    if artifacts is not None:
        artifacts.open()

    t0 = time.perf_counter()
    try:
        var = build_variation_config(cfg, seed)
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
                n_hidden=int(cfg.n_hidden),
                placement_rng=random.Random(seed),
                # Recording is ARMED here or nowhere; the tick loop drives it. Absent
                # entirely when artifacts are off.
                **_recording_kwargs(artifacts),
            )
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
        try:
            roster = _episode_target_roster(ctx)
            _require_scheduled_cell(roster, cfg)
        except MeasurementIntegrityError:
            raise
        except Exception as exc:
            raise EpisodeRosterError(
                "the t=0 target roster could not be established (%s: %s)"
                % (type(exc).__name__, exc)
            ) from exc

        # The damage plan is a t=0 fact about the context setup produced -- the solved
        # routes, the untouched fuel -- so a plan that cannot be built (no eligible ego,
        # no valid strict window) is a `setup` finding, accounted exactly like an
        # exact-cardinality construction failure and never repaired into a clean episode.
        try:
            fuel_damage = build_fuel_damage_controller(
                ctx, episode_seed=int(seed), params=fd_params
            )
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
                    "n_known": int(cfg.n_known),
                    "n_hidden": int(cfg.n_hidden),
                    "n_targets_executed": int(cfg.n_targets_emitted),
                },
                observed={
                    "n_known": len(roster.known_names),
                    "n_hidden": len(roster.hidden_names),
                    "n_targets_executed": int(roster.total),
                    "targets_confirmed_unique": int(targets_confirmed_unique),
                },
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
            fuel_damage_plan=fuel_damage.plan.to_record(),
            fuel_damage_outcome=fd_outcome.to_record(),
            selected_ego_rtb_issued=selected_ego_rtb,
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
            except (_VisualArtifactError, MeasurementIntegrityError):
                # INFRASTRUCTURE / DATA INTEGRITY, not science: neither names a pipeline
                # stage, neither may enter the ledger or a condition tally, and neither
                # may be skipped. Re-raised ahead of the broad handler so the run stops
                # loudly instead of recording a scientific failure that never happened.
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
                    # means what it always did; the finer cell is in the tag.
                    condition=condition,
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

    THE ACTOR-ONLY PAYLOAD IS UNCHANGED. With ``critic is None`` -- which is every
    ``actor_only`` run -- the saved object holds EXACTLY the five keys it always held
    (``iteration`` / ``encoder`` / ``head`` / ``optimizer`` / ``ppo_config``), with the
    same meanings. Nothing was renamed and nothing was added, not even a mode label: a
    Phase-A checkpoint must stay readable by anything that could read one before.

    A CTDE run saves the ACTUAL CTDE training state, which is strictly more: the same
    five keys (``encoder`` / ``head`` / ``optimizer`` are the ACTOR's), plus
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
    }
    if critic is not None:
        payload["training_mode"] = TRAINING_MODE_CTDE
        payload["critic_encoder"] = critic.encoder.state_dict()
        payload["value_head"] = critic.value_head.state_dict()
        payload["critic_optimizer"] = updater.critic_optimizer.state_dict()
        payload["ctde_config"] = asdict(updater.ctde_cfg)
    torch.save(payload, path)
    return path


# =============================================================================
# 7. The training loop
# =============================================================================

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

    # PROVENANCE FIRST -- before this run creates ANYTHING. Not merely before the
    # engine, the policy, the generator or bonmin: before the run directory itself.
    # `output_dir` may point inside the repository, and a directory this run created is
    # untracked, so collecting after `mkdir` would let the run's own scenarios and
    # ledger show up as pre-existing dirty SOURCE state -- provenance contaminated by
    # the act of recording it.
    provenance = collect_provenance(cfg)
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

    # Written BEFORE the completeness gate below, so a refused run still leaves an
    # inspectable record of what was attempted and why it was refused.
    run_config_path = write_run_config(run_dir, cfg, provenance=provenance,
                                       config_source=config_source)

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
    for append_only_path in (failures_path, outcomes_path):
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
          % (cfg.base_seed, cfg.base_seed, cfg.base_seed + cfg.total_episodes))
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
    print("scenario (construction): num_agents=%d  n_known=%d + n_hidden=%d "
          "-> %d target(s) in the executed world"
          % (cfg.num_agents, cfg.n_known, cfg.n_hidden, cfg.n_targets_emitted))
    print("          the generator writes the %d known target(s); setup_episode places "
          "the %d hidden one(s) route-relative and patches them in (split_tasks NOT run)"
          % (cfg.n_known, cfg.n_hidden))
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

    train_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    last_eval_iteration = -1
    last_ckpt_iteration = -1
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
            ev = evaluate(policy, gen, cfg, iteration=None,
                          stage=_EVAL_STAGE_PRE_UPDATE, updates_completed=0,
                          round_ordinal=eval_round_ordinal,
                          failures_path=failures_path,
                          outcomes_path=outcomes_path,
                          artifacts_root=artifacts_root)
            eval_round_ordinal += 1
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval PRE-UPDATE, updates_completed=0] mean=%s ok=%d/%d  %5.1fs"
                  % (_fmt_opt(ev["eval_reward_mean"]), ev["n_successful"],
                     ev["n_attempted"], ev["eval_seconds"]))
            _print_eval_pair_line(ev)

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
            n_attempted_iter = int(cfg.episodes_per_iteration)
            # How much learning stands behind the episodes collected BELOW -- they are
            # generated by the policy as it is now, before this iteration's update.
            updates_before = updates_completed

            # ---- collect the batch ----
            t_eps = time.perf_counter()
            for j in range(cfg.episodes_per_iteration):
                g = global_episode_index(cfg, iteration, j)
                seed = train_seed(cfg, iteration, j)
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
                    )
                except (_VisualArtifactError, MeasurementIntegrityError):
                    # INFRASTRUCTURE / DATA INTEGRITY, not science. Re-raised ahead of the
                    # broad handler so neither can be written to the ledger as a
                    # `generation` / `setup` / `run` / `reward` failure, enter
                    # `skip_and_account_v1`, or shrink a scientific denominator by
                    # masquerading as an episode failure. The run stops. That routing is
                    # the long baseline's lesson: a roster defect accounted as a `setup`
                    # failure removed 143 training attempts in silence.
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
                        # `failures_by_condition` means what it always did.
                        condition=condition,
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
            diag = updater.update(buf)
            update_seconds = time.perf_counter() - t_upd
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
            train_records.append(record)
            train_fh.write(json.dumps(record) + "\n")
            train_fh.flush()

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

            # ---- periodic eval ----
            if cfg.eval_enabled and ((iteration + 1) % cfg.eval_every == 0):
                ev = evaluate(policy, gen, cfg, iteration=iteration,
                              stage=_EVAL_STAGE_POST_UPDATE,
                              updates_completed=updates_completed,
                              round_ordinal=eval_round_ordinal,
                              failures_path=failures_path,
                              outcomes_path=outcomes_path,
                              artifacts_root=artifacts_root)
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
        final_iteration = cfg.n_iterations - 1
        if cfg.eval_enabled and last_eval_iteration != final_iteration:
            ev = evaluate(policy, gen, cfg, iteration=final_iteration,
                          stage=_EVAL_STAGE_POST_UPDATE,
                          updates_completed=updates_completed,
                          round_ordinal=eval_round_ordinal,
                          failures_path=failures_path,
                          outcomes_path=outcomes_path,
                          artifacts_root=artifacts_root)
            eval_round_ordinal += 1
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval @iter %d, final, updates=%d] mean=%s ok=%d/%d  %5.1fs"
                  % (final_iteration, ev["updates_completed"],
                     _fmt_opt(ev["eval_reward_mean"]), ev["n_successful"],
                     ev["n_attempted"], ev["eval_seconds"]))
            _print_eval_pair_line(ev)

    if last_ckpt_iteration != cfg.n_iterations - 1:
        path = save_checkpoint(
            policy, updater, cfg.n_iterations - 1, ckpt_dir, critic=critic
        )
        print("  [ckpt @iter %d, final] %s" % (cfg.n_iterations - 1, path.name))

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
        "eval_group_kind": (
            (eval_records[-1].get("eval_group_kind") if eval_records else None)
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
        # The held-out numbers the factor is measured by, taken from the LAST round and
        # always carrying their group denominator. `final_eval_paired_reward_delta` is
        # the LEGACY damaged-minus-clean key and is `null` for a triad run, whose three
        # named deltas are in `final_eval_group_deltas`.
        "final_eval_paired_reward_delta": (
            eval_records[-1].get("eval_paired_reward_delta") if eval_records else None
        ),
        "final_eval_group_deltas": (
            {key: eval_records[-1].get(key)
             for key in (eval_records[-1].get("eval_delta_keys") or [])}
            if eval_records else None
        ),
        "final_eval_groups_successful": (
            eval_records[-1].get("n_groups_successful",
                                 eval_records[-1].get("n_pairs_successful"))
            if eval_records else None
        ),
        "final_eval_groups_attempted": (
            eval_records[-1].get("n_groups_attempted",
                                 eval_records[-1].get("n_pairs_attempted"))
            if eval_records else None
        ),
        "final_eval_pairs_successful": (
            eval_records[-1].get("n_pairs_successful") if eval_records else None
        ),
        "final_eval_pairs_attempted": (
            eval_records[-1].get("n_pairs_attempted") if eval_records else None
        ),
        "accounting_reconciled": (
            ledger_train == train_failed and ledger_eval == eval_failed
        ),
        # --- held-out results, each with its denominator ---
        "initial_pre_update_eval": _eval_digest(pre_update),
        "final_eval": _eval_digest(eval_records[-1] if eval_records else None),
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
    return _summarize(
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
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Policy entropy per TRAINING batch (collapse detector for the mix "
                 "above)", fontsize=11)
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


def _plot_measurement_health(
    plt: Any,
    plots_dir: Path,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
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
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

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
        _plot_measurement_health(plt, plots_dir, train_records, eval_records),
    ]
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
    if proc.returncode != 0 or len(written) != len(_PLOT_FILENAMES):
        print("plot_training_subprocess: the plot child produced %d of %d figure(s) "
              "(rc=%d) -- plots incomplete; the records are intact."
              % (len(written), len(_PLOT_FILENAMES), proc.returncode))
        if proc.stderr:
            print("  child stderr (last line): %s"
                  % proc.stderr.strip().splitlines()[-1:])
    return written


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
