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

Eval scenarios are generated under a disjoint episode TAG namespace
(``_EVAL_EPISODE_TAG_BASE``) so they can never overwrite a training scenario artifact.
The tag only names the file and the scenario (it is not seed-derived and does not
affect scenario CONTENT), so a fixed tag per eval episode means each round rewrites
the same content -- idempotent, not accumulating.

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
import json
import os
import random
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

# Wall-clock fields -- excluded when two runs' records are compared for equality
# (see _selftest TEST 2: timing legitimately differs run to run).
_TIMING_KEYS = frozenset({
    "iteration_seconds", "episodes_seconds", "update_seconds", "eval_seconds",
})



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


# =============================================================================
# 3. Small helpers (stdlib only)
# =============================================================================

def _stats(values: List[float]) -> Dict[str, float]:
    """(mean, min, max) of a list, safe on empty (returns zeros)."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


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


def write_run_config(run_dir: Path, cfg: TrainConfig) -> Path:
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
      * ``base_scenario`` -- the template filename every variation derives from.

    ``default=str`` covers ``output_dir`` when it is a ``Path``.
    """
    payload = {
        "train_config": asdict(cfg),
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
# 4. One episode (shared by training and evaluation)
# =============================================================================

@dataclass
class _EpisodeOutcome:
    """What one finished episode hands back after its env is closed.

    ``trajectory`` survives the env close by construction: a ``Transition`` holds a
    ``GraphObservation`` (numpy arrays + id strings) and detached floats -- no BLADE
    handle -- so the buffer can outlive the episode it came from.
    """

    trajectory: List[Any]
    reward: float
    ticks: int
    ended: str
    n_wakes: int
    confirmed_kills: int
    n_dead: int
    seconds: float


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

    Raises whatever the pipeline raises -- the caller decides whether a failure aborts
    (it does not: see :func:`train`).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.perf_counter()
    var = build_variation_config(cfg, seed)
    scenario_path = gen.generate(episode=int(episode_tag), config=var)

    ctx = None
    try:
        ctx = setup_episode(
            scenario_path.read_text(encoding="utf-8"),
            # CONSTRUCTION PATH: the generated world is known-only, and setup builds the
            # hidden half from the solved routes (solve -> place -> patch -> reload).
            # `cfg.partial_ratio` is the legacy split surface and is deliberately NOT
            # passed -- `split_tasks` never runs here.
            n_hidden=int(cfg.n_hidden),
            placement_rng=random.Random(seed),
        )
        result = run_episode(
            policy, ctx,
            deterministic=deterministic,
            max_ticks=cfg.max_ticks,
        )
        ep_reward = compute_episode_reward(ctx, result)
        return _EpisodeOutcome(
            trajectory=list(result.trajectory),
            reward=float(ep_reward.reward),
            ticks=int(result.ticks),
            ended=str(result.ended),
            n_wakes=int(result.n_wakes),
            confirmed_kills=int(result.confirmed_kills),
            n_dead=int(result.n_dead),
            seconds=time.perf_counter() - t0,
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
    iteration: int,
) -> Dict[str, Any]:
    """Run ``cfg.eval_episodes`` deterministic episodes on the FIXED eval seed band.

    Touches NO optimizer, NO buffer and no weights -- ``run_episode`` is inference-only
    under its own ``torch.no_grad``. The same seeds are used on every round, so
    round-to-round differences in the returned mean are attributable to the policy.

    Returns a scalar-only record (also written to ``eval_records.jsonl``).
    """
    rewards: List[float] = []
    kills: List[float] = []
    wakes: List[float] = []
    meta_counts = _empty_meta_counts()
    ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
    n_failed = 0
    t0 = time.perf_counter()

    for e in range(cfg.eval_episodes):
        seed = eval_seed(cfg, e)
        try:
            out = _run_one_episode(
                policy, gen, cfg,
                seed=seed,
                episode_tag=_EVAL_EPISODE_TAG_BASE + e,
                deterministic=True,
            )
        except Exception as exc:  # an eval failure must not abort training either
            n_failed += 1
            print("  [eval e%d] FAILED (seed=%d): %s: %s"
                  % (e, seed, type(exc).__name__, exc))
            traceback.print_exc()
            continue
        rewards.append(out.reward)
        kills.append(float(out.confirmed_kills))
        wakes.append(float(out.n_wakes))
        _add_meta_action_counts(meta_counts, out.trajectory)
        if out.ended in ended_counts:
            ended_counts[out.ended] += 1

    r = _stats(rewards)
    return {
        "iteration": int(iteration),
        "n_episodes": int(cfg.eval_episodes),
        "n_ok": len(rewards),
        "n_failed": n_failed,
        "eval_reward_mean": r["mean"],
        "eval_reward_min": r["min"],
        "eval_reward_max": r["max"],
        "eval_kills_mean": _stats(kills)["mean"],
        "eval_wakes_mean": _stats(wakes)["mean"],
        "meta_action_counts": dict(meta_counts),
        "meta_action_fractions": _meta_fractions(meta_counts),
        "ended_counts": dict(ended_counts),
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
      * a failed EPISODE (solver hiccup, engine edge case) is logged with a traceback,
        counted, and skipped -- it never enters the buffer, so it cannot distort the
        baseline, and the run continues;
      * a ZERO-WAKE iteration (no ego woke in any episode) yields an empty batch, and
        ``update`` documents that as a clean no-op with ``n_epochs_run == 0``. It is
        logged like any other iteration -- an iteration in which nothing was sensed is
        a legitimate outcome of the event-triggered design, not an error to swallow.

    The updater (hence its Adam moments) is built ONCE for the whole run.
    """
    cfg.validate()

    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    scen_dir = run_dir / "scenarios"
    scen_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    train_records_path = run_dir / "train_records.jsonl"
    eval_records_path = run_dir / "eval_records.jsonl"
    run_config_path = write_run_config(run_dir, cfg)

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
    print("=" * 78)

    train_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    n_failed_total = 0
    last_eval_iteration = -1
    last_ckpt_iteration = -1
    t_run = time.perf_counter()

    with open(train_records_path, "w", encoding="utf-8") as train_fh, \
            open(eval_records_path, "w", encoding="utf-8") as eval_fh:

        for iteration in range(cfg.n_iterations):
            t_iter = time.perf_counter()
            buf = PPOBuffer()
            meta_counts = _empty_meta_counts()
            ended_counts = {"done": 0, "terminated": 0, "truncated": 0}
            rewards: List[float] = []
            kills: List[float] = []
            ticks: List[float] = []
            n_failed_iter = 0

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
                    n_failed_iter += 1
                    n_failed_total += 1
                    print("  [iter %d ep %d] FAILED (seed=%d): %s: %s"
                          % (iteration, g, seed, type(exc).__name__, exc))
                    traceback.print_exc()
                    continue

                buf.add(EpisodeRecord.from_trajectory(
                    out.trajectory, out.reward, seed=seed, episode_index=g,
                ))
                rewards.append(out.reward)
                kills.append(float(out.confirmed_kills))
                ticks.append(float(out.ticks))
                _add_meta_action_counts(meta_counts, out.trajectory)
                if out.ended in ended_counts:
                    ended_counts[out.ended] += 1
            episodes_seconds = time.perf_counter() - t_eps

            # ---- ONE update over the batch (empty batch -> documented no-op) ----
            t_upd = time.perf_counter()
            diag = updater.update(buf)
            update_seconds = time.perf_counter() - t_upd
            buf.clear()

            # ---- the per-iteration SCALAR record (no per_epoch lists, no tensors) ----
            record = {
                "iteration": iteration,
                "episodes_per_iteration": cfg.episodes_per_iteration,
                "n_failed_episodes": n_failed_iter,
                # The training learning-curve value IS diag["baseline"]: the mean
                # episode R over the iteration's episodes, zero-wake episodes included.
                # Recorded from the update, never recomputed a second way.
                "baseline": float(diag["baseline"]),
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
                "reward_min": _stats(rewards)["min"],
                "reward_max": _stats(rewards)["max"],
                "kills_mean": _stats(kills)["mean"],
                "ticks_mean": _stats(ticks)["mean"],
                "iteration_seconds": time.perf_counter() - t_iter,
                "episodes_seconds": episodes_seconds,
                "update_seconds": update_seconds,
            }
            train_records.append(record)
            train_fh.write(json.dumps(record) + "\n")
            train_fh.flush()

            flag = "" if record["n_epochs_run"] else "  [ZERO-WAKE: update skipped]"
            print("[iter %3d] R=%+.4f trans=%3d wake_eps=%d/%d loss=%+.4f ent=%.3f "
                  "kl=%+.4f clip=%.2f gn=%.3f  %5.1fs%s"
                  % (iteration, record["baseline"], record["n_transitions"],
                     record["episodes_with_wakes"], record["n_episodes"],
                     record["total_loss"], record["entropy"], record["approx_kl"],
                     record["clip_fraction"], record["grad_norm"],
                     record["iteration_seconds"], flag))

            # ---- periodic eval ----
            if cfg.eval_enabled and ((iteration + 1) % cfg.eval_every == 0):
                ev = evaluate(policy, gen, cfg, iteration=iteration)
                eval_records.append(ev)
                eval_fh.write(json.dumps(ev) + "\n")
                eval_fh.flush()
                last_eval_iteration = iteration
                print("  [eval @iter %d] mean=%+.4f min=%+.4f max=%+.4f "
                      "kills=%.1f ok=%d/%d  %5.1fs"
                      % (iteration, ev["eval_reward_mean"], ev["eval_reward_min"],
                         ev["eval_reward_max"], ev["eval_kills_mean"],
                         ev["n_ok"], ev["n_episodes"], ev["eval_seconds"]))

            # ---- periodic checkpoint ----
            if cfg.checkpoint_every > 0 and ((iteration + 1) % cfg.checkpoint_every == 0):
                path = save_checkpoint(policy, updater, iteration, ckpt_dir)
                last_ckpt_iteration = iteration
                print("  [ckpt @iter %d] %s" % (iteration, path.name))

        # ---- final eval + final checkpoint (skipped if this iteration just did one) ----
        final_iteration = cfg.n_iterations - 1
        if cfg.eval_enabled and last_eval_iteration != final_iteration:
            ev = evaluate(policy, gen, cfg, iteration=final_iteration)
            eval_records.append(ev)
            eval_fh.write(json.dumps(ev) + "\n")
            eval_fh.flush()
            print("  [eval @iter %d, final] mean=%+.4f ok=%d/%d  %5.1fs"
                  % (final_iteration, ev["eval_reward_mean"], ev["n_ok"],
                     ev["n_episodes"], ev["eval_seconds"]))

    if last_ckpt_iteration != cfg.n_iterations - 1:
        path = save_checkpoint(policy, updater, cfg.n_iterations - 1, ckpt_dir)
        print("  [ckpt @iter %d, final] %s" % (cfg.n_iterations - 1, path.name))

    summary = _summarize(
        cfg, train_records, eval_records, n_failed_total,
        run_dir, train_records_path, eval_records_path,
        time.perf_counter() - t_run,
    )
    _print_summary(summary)
    return summary


# =============================================================================
# 8. Aggregate + print
# =============================================================================

def _summarize(
    cfg: TrainConfig,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    n_failed_total: int,
    run_dir: Path,
    train_records_path: Path,
    eval_records_path: Path,
    run_seconds: float,
) -> Dict[str, Any]:
    """Aggregate the run into a summary dict (scalars + the record lists)."""
    baselines = [r["baseline"] for r in train_records]
    eval_means = [r["eval_reward_mean"] for r in eval_records]
    meta_totals = _empty_meta_counts()
    for r in train_records:
        for name in _META_NAMES:
            meta_totals[name] += int(r["meta_action_counts"].get(name, 0))

    return {
        "n_iterations": cfg.n_iterations,
        "episodes_per_iteration": cfg.episodes_per_iteration,
        "total_train_episodes": cfg.total_episodes,
        "n_failed_episodes": n_failed_total,
        "train_baseline_first": baselines[0] if baselines else 0.0,
        "train_baseline_last": baselines[-1] if baselines else 0.0,
        "train_baseline_mean": _stats(baselines)["mean"],
        "n_zero_wake_iterations": sum(
            1 for r in train_records if r["n_epochs_run"] == 0
        ),
        "total_transitions": sum(int(r["n_transitions"]) for r in train_records),
        "meta_action_totals": meta_totals,
        "n_eval_rounds": len(eval_records),
        "eval_reward_first": eval_means[0] if eval_means else 0.0,
        "eval_reward_last": eval_means[-1] if eval_means else 0.0,
        "eval_reward_best": max(eval_means) if eval_means else 0.0,
        "run_seconds": run_seconds,
        "run_dir": str(run_dir),
        "train_records_path": str(train_records_path),
        "eval_records_path": str(eval_records_path),
        "train_records": train_records,
        "eval_records": eval_records,
    }


def _print_summary(s: Dict[str, Any]) -> None:
    """Print the run summary as an ASCII table (no unicode -- cp1255 console)."""
    print("-" * 78)
    print("TRAINING SUMMARY (%d iteration(s) x %d episode(s))"
          % (s["n_iterations"], s["episodes_per_iteration"]))
    print("-" * 78)
    print("episodes:   train=%d  failed=%d  transitions=%d"
          % (s["total_train_episodes"], s["n_failed_episodes"],
             s["total_transitions"]))
    print("train R:    first=%+.4f  last=%+.4f  mean=%+.4f"
          % (s["train_baseline_first"], s["train_baseline_last"],
             s["train_baseline_mean"]))
    print("zero-wake:  %d iteration(s) with no transitions (update skipped)"
          % s["n_zero_wake_iterations"])
    mt = s["meta_action_totals"]
    print("meta-acts:  PLAN_COMPLIANCE=%d  OPPORTUNISTIC_ENGAGEMENT=%d  "
          "SELF_PRESERVATION_ABORT=%d"
          % (mt["PLAN_COMPLIANCE"], mt["OPPORTUNISTIC_ENGAGEMENT"],
             mt["SELF_PRESERVATION_ABORT"]))
    if s["n_eval_rounds"]:
        print("eval R:     rounds=%d  first=%+.4f  last=%+.4f  best=%+.4f"
              % (s["n_eval_rounds"], s["eval_reward_first"],
                 s["eval_reward_last"], s["eval_reward_best"]))
    else:
        print("eval R:     (disabled)")
    print("timing:     total=%.1fs" % s["run_seconds"])
    print("records:    %s" % s["train_records_path"])
    print("            %s" % s["eval_records_path"])
    print("-" * 78)


# =============================================================================
# 9. Plotting (lazy matplotlib -- training never hard-depends on it)
# =============================================================================

def plot_training(run_dir: Union[str, Path]) -> Optional[Path]:
    """Render the 3-panel training figure from a run directory's jsonl files.

    Works purely from ``train_records.jsonl`` + ``eval_records.jsonl`` -- no retraining,
    no policy, no torch -- so it can be pointed at any finished (or in-progress) run via
    ``--plot <run_dir>``.

    CALL FROM A TORCH-FREE PROCESS. See the module docstring: importing matplotlib into
    a process that has loaded torch aborts the interpreter on this stack. A torch
    process must call :func:`plot_training_subprocess` instead. The record files are
    read BEFORE matplotlib is touched, so the "nothing to plot" path stays safe
    everywhere.

    Panels (stacked, sharing the iteration x-axis):
      1. LEARNING CURVE: per-iteration training ``baseline`` (faint) + eval mean reward
         (bold), with a dashed reference at ``R = 0``. The reward is oracle-normalized
         regret, so 0 is the perfect-information optimum -- the ceiling, not an
         arbitrary gridline.
      2. META-ACTION MIX: the fraction of each meta-action per iteration.
      3. POLICY ENTROPY per iteration -- the collapse detector for panel 2.

    Returns the PNG path, or ``None`` if matplotlib is missing (a friendly notice is
    printed and NO exception is raised: matplotlib is optional).
    """
    run_path = Path(run_dir)
    train_records = _read_jsonl(run_path / "train_records.jsonl")
    if not train_records:
        print("plot_training: no train_records.jsonl in %s -- nothing to plot."
              % str(run_path))
        return None
    eval_records = _read_jsonl(run_path / "eval_records.jsonl")

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed, no backend guessing
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot_training: matplotlib is not installed -- skipping the plot "
              "(the jsonl records are complete and can be plotted later).")
        return None

    iters = [int(r["iteration"]) for r in train_records]
    baselines = [float(r["baseline"]) for r in train_records]
    entropies = [float(r["entropy"]) for r in train_records]
    fractions = {
        name: [float(r.get("meta_action_fractions", {}).get(name, 0.0))
               for r in train_records]
        for name in _META_NAMES
    }
    eval_iters = [int(r["iteration"]) for r in eval_records]
    eval_means = [float(r["eval_reward_mean"]) for r in eval_records]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # --- Panel 1: learning curve ---
    ax = axes[0]
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4",
               label="oracle optimum (R = 0)")
    ax.plot(iters, baselines, color="tab:blue", alpha=0.35, linewidth=1.2,
            marker=".", markersize=4, label="train mean R (stochastic)")
    if eval_iters:
        ax.plot(eval_iters, eval_means, color="tab:red", linewidth=2.2,
                marker="o", markersize=5, label="eval mean R (deterministic)")
    ax.set_ylabel("episode reward R")
    ax.set_title("Learning curve -- oracle-normalized regret (0 = optimum)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 2: meta-action mix ---
    ax = axes[1]
    for name, color in zip(_META_NAMES, ("tab:green", "tab:orange", "tab:purple")):
        ax.plot(iters, fractions[name], color=color, linewidth=1.6,
                marker=".", markersize=4, label=name)
    ax.set_ylabel("fraction of decisions")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Meta-action mix per iteration")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- Panel 3: entropy ---
    ax = axes[2]
    ax.plot(iters, entropies, color="tab:brown", linewidth=1.6,
            marker=".", markersize=4)
    ax.set_ylabel("policy entropy (nats)")
    ax.set_xlabel("PPO iteration")
    ax.set_title("Policy entropy per iteration (collapse detector)")
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
        assert len(eval_recs1) >= 1, len(eval_recs1)
        for rec in train_recs1:
            for key in ("baseline", "policy_loss", "total_loss", "entropy",
                        "mean_ratio", "clip_fraction", "approx_kl", "grad_norm"):
                value = float(rec[key])
                assert value == value and abs(value) != float("inf"), (key, value)
            assert rec["n_episodes"] == 4 - rec["n_failed_episodes"], rec
        ckpts = sorted((run1 / "checkpoints").glob("ckpt_iter*.pt"))
        assert ckpts, "no checkpoint was written"

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
              "u_oracle-normalized rewards: first R=%+.4f last R=%+.4f)"
              % (summary1["train_baseline_first"], summary1["train_baseline_last"]))

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
            r["n_episodes"] - r["episodes_with_wakes"] for r in train_recs1
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
            assert rec["n_episodes"] > 0, rec   # episodes DID run; nobody woke
            assert rec["policy_loss"] == 0.0 and rec["grad_norm"] == 0.0, rec
            assert rec["baseline"] == rec["baseline"], rec  # not NaN
        assert summary3["n_iterations"] == 2, summary3
        print("  %d/%d iterations were zero-wake: n_epochs_run=0, n_transitions=0, "
              "baseline=%+.4f (finite), loop completed all %d iterations   OK"
              % (len(zero), len(train_recs3), zero[0]["baseline"],
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
