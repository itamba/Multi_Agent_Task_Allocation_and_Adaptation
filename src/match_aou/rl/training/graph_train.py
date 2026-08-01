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
beliefs that disagree, or a confirmed target the roster does not contain) fails the
attempt at the ``setup`` stage and is skipped and accounted like any other, while an
unresolvable NAME degrades to ``<unnamed target>`` and changes no id and no count.

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
  * ``training_plot.png``      -- 4 panels, derived from the jsonl files alone.

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
    child process that only reads jsonl and draws a PNG -- no tensor math whatsoever --
    with ``KMP_DUPLICATE_LIB_OK=TRUE`` set for that child alone. The duplicate-OpenMP
    tolerance is therefore confined to a throwaway, numerics-free process; the training
    process itself never gets a second OpenMP runtime.
Either way, training NEVER depends on matplotlib: a missing matplotlib (or a failed
child) prints one notice and returns ``None``.

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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .graph_episode_setup import (
    setup_episode,
    DETECTION_KM,
    MAX_SIM_TICKS,
)
from .graph_ppo import EpisodeRecord, PPOBuffer, PPOConfig, PPOUpdater
from .graph_reward import compute_episode_reward
from .graph_tick_loop import build_policy, run_episode
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

# Keys holding the full record lists inside a run summary. They are returned in-process
# but NOT persisted to run_summary.json -- the jsonl files are the record, and copying
# them into the summary would create a second, divergeable metric path.
_SUMMARY_RECORD_KEYS = ("train_records", "eval_records", "failure_records")



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
        num_red_airbases: LEGACY, like ``partial_ratio`` -- the construction path emits
            ``n_known`` targets and never reads this.
    """

    n_iterations: int
    episodes_per_iteration: int = 8
    base_seed: int = 0
    output_dir: Union[str, Path] = ""       # "" -> training_output_<timestamp>
    ppo: PPOConfig = field(default_factory=PPOConfig)

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
        if int(self.eval_episodes) > _EVAL_ROUND_TAG_STRIDE:
            raise ValueError(
                "eval_episodes (%d) exceeds one eval round's scenario-tag namespace "
                "(%d): consecutive eval rounds would write over each other's scenario "
                "files. Raise _EVAL_ROUND_TAG_STRIDE or shorten the eval band."
                % (int(self.eval_episodes), _EVAL_ROUND_TAG_STRIDE)
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
    exc: BaseException,
) -> Dict[str, Any]:
    """Build ONE ledger record for a failed attempt (see :func:`_append_failure_record`).

    Every field a post-hoc audit needs to place the attempt exactly: which phase and
    (for eval) which stage, how much learning had happened when it was attempted, its
    position in the SCHEDULE, its identity, its exact seed, the stage it died in, and the
    original exception with its traceback.

    ``seed`` is the scheduled seed, recorded even though it produced nothing -- a failed
    seed stays part of the attempted population and must remain visible.
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


def write_run_config(
    run_dir: Path,
    cfg: TrainConfig,
    *,
    provenance: Optional[Dict[str, Any]] = None,
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
        "derived_split": cfg.split_preview,
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
#   * A STRUCTURAL failure -- absent or malformed beliefs / oracle tasks, t=0 beliefs
#     that disagree, or a roster that does not cover the executed world -- FAILS THE
#     ATTEMPT through the existing `skip_and_account_v1` machinery, attributed to the
#     `setup` stage.
#
# The second rule exists because of a real defect in the first version of this section:
# it swallowed every structural exception and returned an empty roster, and the
# authoritative count was derived from the names it had managed to classify. A degraded
# roster therefore turned an episode with real confirmations into a SUCCESSFUL
# `0/0` measurement, and that false zero flowed straight into
# `targets_confirmed_unique_mean` and its aliases. A research metric must never depend on
# whether a name diagnostic worked: the authoritative count is now
# `len(_unique_confirmed_target_ids(executor.done))` and nothing else, and a roster that
# cannot describe the world that ran is a failed attempt, never a measured zero.

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


class EpisodeRosterError(RuntimeError):
    """The episode's target roster could not be built, or does not describe what ran.

    A STRUCTURAL failure of the measurement, not a display problem. It is raised, wrapped
    as an ``EpisodeAttemptError("setup", ...)`` and accounted like any other failed
    attempt, because the alternative -- reporting the episode as a successful zero -- is
    the exact false-zero defect this class exists to prevent. It is NOT a new pipeline
    stage: the roster is a t=0 fact about the context ``setup_episode`` produced, so a
    roster that is missing, self-contradictory, or too small to cover the executed world
    is a ``setup`` finding, and it appears in ``episode_failures.jsonl`` under its own
    ``error_type`` so an audit can tell it apart from an exact-cardinality failure.
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


def _episode_target_roster(ctx: Any) -> _TargetRoster:
    """Snapshot the known / hidden target roster of a freshly set-up episode.

    CALL AFTER ``setup_episode`` AND BEFORE ``run_episode``, for two independent reasons:
    the names are read out of the live scenario, which loses units as they are killed;
    and the known set is read from ``ctx.beliefs``, which are byte-equal at t=0 and
    legitimately DIVERGE per ego afterwards (that divergence is the no-communication
    guarantee, not a defect).

      * KNOWN   -- the t=0 belief task list, in A_init's positional order.
      * FULL    -- ``ctx.oracle_tasks``: the oracle is an independent solve over ALL
        env-2 targets (B3), so this is the executed world, hidden half included.
      * HIDDEN  -- full minus known, in oracle order. Derived by SUBTRACTION rather than
        from ``ctx.placements``, which is deliberately id-free.

    REQUIRED MEASUREMENT STRUCTURE, not a best-effort diagnostic. Every structural
    problem raises :class:`EpisodeRosterError`, which the caller accounts as a failed
    ``setup`` attempt: no beliefs, a malformed belief or oracle task list, t=0 beliefs
    that disagree (they are all minted from one A_init, so a disagreement is a real
    no-communication defect and must never be averaged over or reported as one ego's
    view), no oracle tasks, or a known target the executed world does not contain. The
    earlier version swallowed all of these and returned an empty roster; that turned a
    structural failure into a successful ``0/0`` measurement.

    Only name RESOLUTION degrades (:func:`_resolve_target_name`), and it changes no id
    and no count.
    """
    beliefs_map = getattr(ctx, "beliefs", None) or {}
    if not beliefs_map:
        raise EpisodeRosterError(
            "the episode context carries no beliefs, so the t=0 known target set "
            "cannot be established"
        )

    # Compared BEFORE deduplication so a divergence in order is caught too. This is a
    # cheap invariant check (<= ~4 egos x ~9 tasks) on the guarantee that every belief is
    # minted from one A_init.
    per_ego = [
        (str(ego_id), _ordered_target_ids(getattr(belief, "tasks", None),
                                          "belief of ego %s" % ego_id))
        for ego_id, belief in beliefs_map.items()
    ]
    known_ids = per_ego[0][1]
    for ego_id, ids in per_ego[1:]:
        if ids != known_ids:
            raise EpisodeRosterError(
                "the t=0 beliefs disagree on the known target set (ego %s vs ego %s): "
                "all beliefs are minted from one A_init, so this is a real defect and "
                "not something to report as one ego's view"
                % (per_ego[0][0], ego_id)
            )

    executed_ids = _ordered_target_ids(getattr(ctx, "oracle_tasks", None),
                                       "oracle")
    if not executed_ids:
        raise EpisodeRosterError(
            "the oracle names no targets, so the executed world is unknown"
        )

    executed_set = set(executed_ids)
    unmatched = [tid for tid in known_ids if tid not in executed_set]
    if unmatched:
        raise EpisodeRosterError(
            "%d t=0 known target(s) are absent from the executed world: the roster "
            "would not cover what runs" % len(unmatched)
        )

    known_set = set(known_ids)
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
        "  ended=%s ticks=%d dead=%d elapsed=%.1fs"
        % (_ascii(out.ended), out.ticks, out.n_dead, out.seconds),
    ])


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


def _run_one_episode(
    policy: Any,
    gen: ScenarioGenerator,
    cfg: TrainConfig,
    *,
    seed: int,
    episode_tag: int,
    deterministic: bool,
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

    The roster is REQUIRED measurement structure. A structural failure -- it cannot be
    built, or it does not account for every confirmed target -- raises
    :class:`EpisodeRosterError` and is wrapped as a ``setup`` attempt failure, so it is
    skipped and accounted like any other. No new pipeline stage is introduced, and no
    such attempt reaches a reward aggregate. Only NAME resolution degrades, and it
    changes nothing but the printed text.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.perf_counter()
    try:
        var = build_variation_config(cfg, seed)
        scenario_path = gen.generate(episode=int(episode_tag), config=var)
    except Exception as exc:
        raise EpisodeAttemptError("generation", exc) from exc

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
            )
        except Exception as exc:
            raise EpisodeAttemptError("setup", exc) from exc

        # The roster is snapshotted HERE -- after setup, before a single tick -- because
        # both of its sources are t=0 facts: the N beliefs are byte-equal only now, and
        # the live scenario still holds every target it is about to lose to a kill.
        # Read-only, but REQUIRED: a roster that cannot be established describes a
        # context `setup_episode` should not have produced, so it fails the attempt at
        # the `setup` stage rather than degrading into a successful zero measurement.
        try:
            roster = _episode_target_roster(ctx)
        except Exception as exc:
            raise EpisodeAttemptError("setup", exc) from exc

        try:
            result = run_episode(
                policy, ctx,
                deterministic=deterministic,
                max_ticks=cfg.max_ticks,
            )
        except Exception as exc:
            raise EpisodeAttemptError("run", exc) from exc
        try:
            ep_reward = compute_episode_reward(ctx, result)
        except Exception as exc:
            raise EpisodeAttemptError("reward", exc) from exc

        # THE AUTHORITATIVE COUNT, and the one line the review finding is about. It is
        # `len()` of the deduplicated id set taken straight off the executor -- a target
        # both egos confirmed is ONE target here -- and it is NOT derived from how many
        # of those ids the roster managed to name. `result.confirmed_kills` below still
        # reports the raw (ego, target) confirmation count, unchanged.
        confirmed_ids = _unique_confirmed_target_ids(
            getattr(ctx.executor, "done", None)
        )
        targets_confirmed_unique = len(confirmed_ids)

        # The name subsets are a PRESENTATION of that number. If they cannot account for
        # every confirmed id, the roster does not describe the world that ran, and the
        # attempt fails as a `setup` finding instead of printing a block whose names
        # contradict its own total.
        try:
            known_confirmed, hidden_confirmed = roster.confirmed(confirmed_ids)
        except Exception as exc:
            raise EpisodeAttemptError("setup", exc) from exc

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

    ``round_ordinal`` names this round's SCENARIO-TAG namespace (:func:`eval_episode_tag`)
    and nothing else. The seeds are unchanged -- episode ``e`` is ``eval_seed(cfg, e)`` on
    every round -- so successive rounds re-measure the same held-out worlds; they just
    stop overwriting each other's scenario JSON while doing it. ``pre_update`` is
    ordinal 0 and each later ``post_update`` round takes the next.

    Returns a scalar-only record (also written to ``eval_records.jsonl``), plus one
    printed ``OK`` block per successful episode.
    """
    rewards: List[float] = []
    unique_confirmed: List[float] = []
    wakes: List[float] = []
    meta_counts = _empty_meta_counts()
    ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
    n_failed = 0
    n_attempted = int(cfg.eval_episodes)
    t0 = time.perf_counter()

    for e in range(cfg.eval_episodes):
        seed = eval_seed(cfg, e)
        tag = eval_episode_tag(round_ordinal=round_ordinal, e=e)
        try:
            out = _run_one_episode(
                policy, gen, cfg,
                seed=seed,
                episode_tag=tag,
                deterministic=True,
            )
        except Exception as exc:  # an eval failure must not abort training either
            n_failed += 1
            _append_failure_record(failures_path, _failure_record(
                phase="eval",
                evaluation_stage=stage,
                updates_completed=updates_completed,
                iteration=iteration,
                attempt_ordinal=e,
                episode_index=None,
                eval_tag="eval_e%d_tag%d" % (e, tag),
                seed=seed,
                exc=exc,
            ))
            print("  [eval %s e%d] FAILED (seed=%d): %s: %s"
                  % (stage, e, seed, type(exc).__name__, exc))
            traceback.print_exc()
            continue
        # Printed BEFORE the next attempt starts, so a long eval round is readable while
        # it runs rather than only in the round's summary line.
        print(_format_episode_block(
            "[eval stage=%s ep=%d seed=%d]" % (_ascii(stage), e, seed), out
        ))
        rewards.append(out.reward)
        unique_confirmed.append(float(out.targets_confirmed_unique))
        wakes.append(float(out.n_wakes))
        _add_meta_action_counts(meta_counts, out.trajectory)
        if out.ended in ended_counts:
            ended_counts[out.ended] += 1

    n_successful = len(rewards)
    episodes_with_wakes = sum(1 for w in wakes if w > 0)
    r = _stats_or_none(rewards)
    # ONE arithmetic site behind BOTH the authoritative key and its legacy alias, so the
    # two can never drift apart and the alias can never revert to the (ego,target) count.
    unique_confirmed_mean = _stats_or_none(unique_confirmed)["mean"]
    return {
        "evaluation_stage": str(stage),
        "updates_completed": int(updates_completed),
        "iteration": None if iteration is None else int(iteration),
        # Which scenario-tag namespace this round's worlds were written under -- the
        # link from a record back to the `episode_<tag>_scenario.json` files it ran on.
        "eval_round_ordinal": int(round_ordinal),
        "episode_tag_start": eval_episode_tag(round_ordinal=round_ordinal, e=0),
        # --- attempt accounting: the AUTHORITATIVE names ---
        "n_attempted": n_attempted,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "success_fraction": _fraction(n_successful, n_attempted),
        "episodes_with_wakes": int(episodes_with_wakes),
        "wake_fraction_of_successful": _fraction(episodes_with_wakes, n_successful),
        # --- aggregates over the SUCCESSFUL subset only (None when it is empty) ---
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
        "meta_action_fractions": _meta_fractions(meta_counts),
        "ended_counts": dict(ended_counts),
        "seed_band": {
            "start": int(cfg.eval_base_seed),
            "stop": int(cfg.eval_base_seed) + n_attempted,
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

def save_checkpoint(
    policy: Any,
    updater: PPOUpdater,
    iteration: int,
    ckpt_dir: Path,
) -> Path:
    """Save encoder + head + optimizer state (and provenance) to ``ckpt_iter<NNNN>.pt``.

    The optimizer's state_dict is included because Adam's moment estimates ARE training
    state -- a checkpoint without them could not faithfully continue a run. The
    ``PPOConfig`` is stored as a plain dict (not the dataclass) so a loader never needs
    to unpickle a project class.

    There is intentionally NO loader here: restoring a run is a separate, deferred task
    (it needs decisions about the seed schedule and the scenario stream that saving
    does not). ``tests/test_graph_train.py`` proves the saved payload round-trips.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ("ckpt_iter%04d.pt" % int(iteration))
    torch.save(
        {
            "iteration": int(iteration),
            "encoder": policy.encoder.state_dict(),
            "head": policy.head.state_dict(),
            "optimizer": updater.optimizer.state_dict(),
            "ppo_config": asdict(updater.cfg),
        },
        path,
    )
    return path


# =============================================================================
# 7. The training loop
# =============================================================================

def train(cfg: TrainConfig) -> Dict[str, Any]:
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
    train_records_path = run_dir / "train_records.jsonl"
    eval_records_path = run_dir / "eval_records.jsonl"
    failures_path = run_dir / "episode_failures.jsonl"

    # Written BEFORE the completeness gate below, so a refused run still leaves an
    # inspectable record of what was attempted and why it was refused.
    run_config_path = write_run_config(run_dir, cfg, provenance=provenance)

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

    # Truncate the ledger: it describes THIS run, and appending to a previous run's
    # failures in a reused directory would silently corrupt the accounting. After the
    # gate, so a refused run never destroys an earlier run's ledger.
    with open(failures_path, "w", encoding="utf-8"):
        pass

    # Match the rollout/selftest PlaybackRecorder override (harmless when recording is
    # off, which it always is here). Lazy import: engine boundary.
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024

    # ONE policy (weights pinned by base_seed) and ONE updater for the whole run --
    # rebuilding the updater per iteration would silently discard Adam's moments.
    torch.manual_seed(cfg.base_seed)
    policy = build_policy()
    updater = PPOUpdater(policy, cfg.ppo)
    gen = _build_generator(scen_dir)

    print("=" * 78)
    print("graph_train: %d iteration(s) x %d episode(s) = %d training episodes"
          % (cfg.n_iterations, cfg.episodes_per_iteration, cfg.total_episodes))
    print("base_seed=%d  train seeds [%d, %d)"
          % (cfg.base_seed, cfg.base_seed, cfg.base_seed + cfg.total_episodes))
    if cfg.eval_enabled:
        print("eval: every %d iter, %d episode(s), FIXED seeds [%d, %d)"
              % (cfg.eval_every, cfg.eval_episodes, cfg.eval_base_seed,
                 cfg.eval_base_seed + cfg.eval_episodes))
    else:
        print("eval: DISABLED")
    print("ppo: %s" % (asdict(cfg.ppo),))
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
    print("legacy split surface (NOT used by the construction path): "
          "num_red_airbases=%r partial_ratio=%s -> known/hidden = %s"
          % (cfg.num_red_airbases, cfg.partial_ratio,
             _format_split_preview(cfg.split_preview)))
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
                          failures_path=failures_path)
            eval_round_ordinal += 1
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval PRE-UPDATE, updates_completed=0] mean=%s ok=%d/%d  %5.1fs"
                  % (_fmt_opt(ev["eval_reward_mean"]), ev["n_successful"],
                     ev["n_attempted"], ev["eval_seconds"]))

        for iteration in range(cfg.n_iterations):
            t_iter = time.perf_counter()
            buf = PPOBuffer()
            meta_counts = _empty_meta_counts()
            ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
            rewards: List[float] = []
            unique_confirmed: List[float] = []
            ticks: List[float] = []
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
                try:
                    out = _run_one_episode(
                        policy, gen, cfg,
                        seed=seed, episode_tag=g, deterministic=False,
                    )
                except Exception as exc:  # never abort the run on one episode
                    # SKIP AND ACCOUNT: record it and move to the NEXT scheduled seed.
                    # This seed is spent -- no retry, no substitute, no shift of the
                    # band. `j` continues, so the schedule is untouched.
                    n_failed_iter += 1
                    _append_failure_record(failures_path, _failure_record(
                        phase="train",
                        evaluation_stage=None,
                        updates_completed=updates_before,
                        iteration=iteration,
                        attempt_ordinal=j,
                        episode_index=g,
                        eval_tag=None,
                        seed=seed,
                        exc=exc,
                    ))
                    print("  [iter %d ep %d] FAILED (seed=%d, stage=%s): %s: %s"
                          % (iteration, g, seed, getattr(exc, "stage", "unknown"),
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

            # ---- periodic eval ----
            if cfg.eval_enabled and ((iteration + 1) % cfg.eval_every == 0):
                ev = evaluate(policy, gen, cfg, iteration=iteration,
                              stage=_EVAL_STAGE_POST_UPDATE,
                              updates_completed=updates_completed,
                              round_ordinal=eval_round_ordinal,
                              failures_path=failures_path)
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

            # ---- periodic checkpoint ----
            if cfg.checkpoint_every > 0 and ((iteration + 1) % cfg.checkpoint_every == 0):
                path = save_checkpoint(policy, updater, iteration, ckpt_dir)
                last_ckpt_iteration = iteration
                print("  [ckpt @iter %d] %s" % (iteration, path.name))

        # ---- final eval + final checkpoint (skipped if this iteration just did one) ----
        final_iteration = cfg.n_iterations - 1
        if cfg.eval_enabled and last_eval_iteration != final_iteration:
            ev = evaluate(policy, gen, cfg, iteration=final_iteration,
                          stage=_EVAL_STAGE_POST_UPDATE,
                          updates_completed=updates_completed,
                          round_ordinal=eval_round_ordinal,
                          failures_path=failures_path)
            eval_round_ordinal += 1
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval @iter %d, final, updates=%d] mean=%s ok=%d/%d  %5.1fs"
                  % (final_iteration, ev["updates_completed"],
                     _fmt_opt(ev["eval_reward_mean"]), ev["n_successful"],
                     ev["n_attempted"], ev["eval_seconds"]))

    if last_ckpt_iteration != cfg.n_iterations - 1:
        path = save_checkpoint(policy, updater, cfg.n_iterations - 1, ckpt_dir)
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
    }


def _summarize(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    failure_records: List[Dict[str, Any]],
    *,
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
        "run_config_path": str(run_path / "run_config.json"),
        "run_summary_path": str(run_path / "run_summary.json"),
        "plot_path": str(run_path / "training_plot.png"),
    }
    # Pre-B4 names, kept so an existing reader of a summary still resolves.
    summary["total_train_episodes"] = train_attempted
    summary["n_failed_episodes"] = train_failed
    summary["train_baseline_first"] = summary["train_reward_first"]
    summary["train_baseline_last"] = summary["train_reward_last"]
    summary["train_baseline_mean"] = summary["train_reward_mean"]
    summary["eval_reward_first"] = (eval_means[0] if eval_means else None)
    summary["eval_reward_last"] = (eval_means[-1] if eval_means else None)
    # The full record streams, for in-process callers ONLY (see _SUMMARY_RECORD_KEYS:
    # they are stripped before the summary is written, because the jsonl files are the
    # record and a copy of them inside the summary could diverge from it).
    summary["train_records"] = train_records
    summary["eval_records"] = eval_records
    summary["failure_records"] = failure_records
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
    else:
        print("eval R:     (disabled)")
    print("failures:   %d recorded  by phase=%s  by stage=%s%s"
          % (s["failures_recorded"], s["failures_by_phase"],
             s["failures_by_pipeline_stage"],
             "" if s["accounting_reconciled"]
             else "   [!] LEDGER DISAGREES WITH THE RECORD COUNTS"))
    if s["run_seconds"] is not None:
        print("timing:     total=%.1fs" % s["run_seconds"])
    print("records:    %s" % s["train_records_path"])
    print("            %s" % s["eval_records_path"])
    print("            %s" % s["failures_path"])
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
    as some other number would invent one. Its attempts are still visible -- panel 4
    shows the success fraction that caused the gap.
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


def plot_training(run_dir: Union[str, Path]) -> Optional[Path]:
    """Render the 4-panel training figure from a run directory's jsonl files.

    Works purely from ``train_records.jsonl`` + ``eval_records.jsonl`` -- no retraining,
    no policy, no torch -- so it can be pointed at any finished (or in-progress) run via
    ``--plot <run_dir>``.

    CALL FROM A TORCH-FREE PROCESS. See the module docstring: importing matplotlib into
    a process that has loaded torch aborts the interpreter on this stack. A torch
    process must call :func:`plot_training_subprocess` instead. The record files are
    read BEFORE matplotlib is touched, so the "nothing to plot" path stays safe
    everywhere.

    THE X-AXIS IS ``updates_completed``, NOT the iteration index. Two reasons, both
    about honesty rather than taste: the ``pre_update`` held-out point measures the
    initial policy and belongs at x=0, which an iteration index has no room for; and a
    zero-wake iteration completes without performing a gradient step, so iteration
    number over-states how much learning stands behind a later point. Training points
    are placed at ``updates_completed_before`` -- the updates the policy that GENERATED
    those episodes had received -- so training iteration 0 and the pre-update eval sit
    at the same origin. Records from before B4 fall back to their iteration index.

    Panels (stacked, sharing the x-axis):
      1. LEARNING CURVE: per-iteration training mean R (faint) + eval mean R (bold),
         with a dashed reference at ``R = 0``. The reward is oracle-normalized regret,
         so 0 is the perfect-information optimum -- the ceiling, not an arbitrary
         gridline, which is exactly why a batch with NO successful episode is dropped
         from the curve instead of drawn at 0 (see :func:`_xy`).
      2. META-ACTION MIX: the fraction of each meta-action per iteration.
      3. POLICY ENTROPY per iteration -- the collapse detector for panel 2.
      4. DATA YIELD: training success fraction, the fraction of SUCCESSFUL training
         episodes that contained wakes, and eval success fraction at each eval point.
         Panel 1 without panel 4 is unreadable: a mean over 2 of 8 feasible seeds and a
         mean over 8 of 8 look identical there and are not the same claim.

    Returns the PNG path, or ``None`` if matplotlib is missing (a friendly notice is
    printed and NO exception is raised: matplotlib is optional).
    """
    run_path = Path(run_dir)
    train_records = _read_jsonl(run_path / "train_records.jsonl")
    eval_records = _read_jsonl(run_path / "eval_records.jsonl")
    if not train_records and not eval_records:
        print("plot_training: no train_records.jsonl / eval_records.jsonl in %s -- "
              "nothing to plot." % str(run_path))
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed, no backend guessing
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot_training: matplotlib is not installed -- skipping the plot "
              "(the jsonl records are complete and can be plotted later).")
        return None

    # Training points sit at the updates the GENERATING policy had received; eval points
    # at the updates completed when the round ran (0 for the pre-update round).
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

    curve_x, curve_y = _xy(
        train_records, "updates_completed_before", "train_reward_mean"
    )
    if not curve_y:  # pre-B4 records carry the value under its old name
        curve_x, curve_y = _xy(
            train_records, "updates_completed_before", "baseline"
        )
    eval_x, eval_y = _xy(eval_records, "updates_completed", "eval_reward_mean")

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # --- Panel 1: learning curve ---
    ax = axes[0]
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4",
               label="oracle optimum (R = 0)")
    if curve_y:
        ax.plot(curve_x, curve_y, color="tab:blue", alpha=0.35, linewidth=1.2,
                marker=".", markersize=4, label="train mean R (stochastic)")
    if eval_y:
        ax.plot(eval_x, eval_y, color="tab:red", linewidth=2.2,
                marker="o", markersize=5, label="eval mean R (deterministic)")
    ax.set_ylabel("episode reward R")
    ax.set_title("Learning curve -- oracle-normalized regret (0 = optimum); "
                 "means over SUCCESSFUL episodes only")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 2: meta-action mix ---
    ax = axes[1]
    for name, color in zip(_META_NAMES, ("tab:green", "tab:orange", "tab:purple")):
        ax.plot(train_x, fractions[name], color=color, linewidth=1.6,
                marker=".", markersize=4, label=name)
    ax.set_ylabel("fraction of decisions")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Meta-action mix per iteration")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 3: entropy ---
    ax = axes[2]
    ax.plot(train_x, entropies, color="tab:brown", linewidth=1.6,
            marker=".", markersize=4)
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Policy entropy per iteration (collapse detector)")
    ax.grid(alpha=0.25)

    # --- Panel 4: data yield (the denominator behind panel 1) ---
    ax = axes[3]
    ok_x, ok_y = _xy(train_records, "updates_completed_before", "success_fraction")
    if ok_y:
        ax.plot(ok_x, ok_y, color="tab:blue", linewidth=1.8, marker=".",
                markersize=5, label="train episodes: successful / attempted")
    wake_x, wake_y = _xy(
        train_records, "updates_completed_before", "wake_fraction_of_successful"
    )
    if wake_y:
        ax.plot(wake_x, wake_y, color="tab:cyan", linewidth=1.4, marker=".",
                markersize=4, linestyle="--",
                label="successful train episodes with wakes")
    ev_ok_x, ev_ok_y = _xy(eval_records, "updates_completed", "success_fraction")
    if ev_ok_y:
        ax.plot(ev_ok_x, ev_ok_y, color="tab:red", linewidth=1.8, marker="o",
                markersize=5, label="eval episodes: successful / attempted")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("PPO updates completed")
    ax.set_title("Data yield -- exact-cardinality feasibility (%s) and wake coverage"
                 % _EXACT_CARDINALITY_POLICY)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path = run_path / "training_plot.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("plot_training: wrote %s" % str(out_path))
    return out_path


def plot_training_subprocess(
    run_dir: Union[str, Path],
    *,
    timeout: float = 300.0,
) -> Optional[Path]:
    """Render the figure from a TORCH process by re-invoking ``--plot`` in a child.

    Why this exists at all: see the module docstring. torch and matplotlib abort the
    interpreter if they share a process on this stack, and an abort is not catchable --
    so a training process cannot draw its own plot, it has to fork one that does.

    The child is `` python -m match_aou.rl.training.graph_train --plot <run_dir> `` with
    ``KMP_DUPLICATE_LIB_OK=TRUE`` in ITS environment only. That flag is Intel's
    documented "unsafe" duplicate-OpenMP tolerance; it is acceptable here precisely
    because the child performs NO numerical work -- it reads two jsonl files and writes
    a PNG -- and it never touches the parent's environment.

    Never raises: a missing matplotlib, a crashed child, or a timeout prints a notice
    and returns ``None``. Plotting is a convenience; the jsonl records are the record.
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
              "plot skipped; re-run `--plot %s` later."
              % (type(exc).__name__, exc, str(run_path)))
        return None

    for line in (proc.stdout or "").splitlines():
        if line.startswith("plot_training"):
            print("  " + line)
    out_path = run_path / "training_plot.png"
    if proc.returncode != 0 or not out_path.exists():
        print("plot_training_subprocess: the plot child did not produce a figure "
              "(rc=%d) -- plot skipped; the records are intact."
              % proc.returncode)
        if proc.stderr:
            print("  child stderr (last line): %s"
                  % proc.stderr.strip().splitlines()[-1:])
        return None
    return out_path


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
        png = plot_training_subprocess(run1)
        assert png is None or png.exists()
        print("  plot: %s   OK"
              % (str(png) if png else "not produced (matplotlib absent) -- skipped"))

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
    p.add_argument("--plot", type=str, default=None, metavar="RUN_DIR",
                   help="plot an EXISTING run directory and exit (no training)")
    p.add_argument("--selftest", action="store_true",
                   help="run the module self-test (needs BLADE + bonmin) and exit")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        _selftest()
        return
    if args.plot is not None:
        plot_training(args.plot)
        return
    if args.iterations is None:
        parser.error("--iterations is required for a training run "
                     "(or pass --plot RUN_DIR / --selftest)")

    cfg = TrainConfig(
        n_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        base_seed=args.seed,
        output_dir=args.out,
        ppo=PPOConfig(
            clip_ratio=args.clip_ratio,
            entropy_coeff=args.entropy_coeff,
            lr=args.lr,
            n_epochs=args.epochs,
        ),
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_base_seed=args.eval_base_seed,
        num_agents=args.num_agents,
        n_known=args.n_known,
        n_hidden=args.n_hidden,
        min_target_distance_km=args.min_target_distance_km,
        min_known_separation_km=args.min_known_separation_km,
        num_red_airbases=args.num_red_airbases,
        partial_ratio=args.partial_ratio,
        stretch_target_ratio=args.stretch_target_ratio,
    )
    # Fail on an impossible cell (e.g. num_agents > n_known) HERE, before train()
    # touches the filesystem or the solver. train() validates again; validate() is pure.
    cfg.validate()
    summary = train(cfg)
    # This process has torch loaded, so the figure is drawn by a child (module docstring).
    plot_training_subprocess(summary["run_dir"])


if __name__ == "__main__":
    main()
