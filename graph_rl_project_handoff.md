# Multi-Agent Graph RL — B3 Closure / B4 Handoff

**Supersedes all earlier handoffs.**

Written 2026-07-29. **The offline scenario-construction phase is COMPLETE: B1, B2 and B3 are
all CLOSED / MERGED / LOCKED.** **B4 — training-run planning and preparation — is the next
task.** The commit that lands this document changes documents only — no code, no test delta.

This handoff describes the next phase only. Technical contracts and frozen layers live in
`CLAUDE.md`; code and tests remain decisive. Where a fact is already in `CLAUDE.md` this
document cross-references it instead of restating it — a duplicated fact is a fact that will
drift.

## 0. How this workspace reads the repository

- Repository access is **capability-aware**, not shared. The two orchestrators see different
  things and must resolve facts through what each actually has:
  - **GPT** resolves repository facts through GitHub first — branches, PRs, files, and exact
    SHAs. That access is a **search** interface, not a filesystem: it cannot list files,
    count occurrences, prove that something is ABSENT, or run `git`.
  - **Claude** resolves facts through a synchronized mounted snapshot of `main` plus the Git
    evidence CC reports. It is the weaker of the two views: like GPT's it is a **search**
    interface — it cannot list files, count occurrences, prove that something is ABSENT, or
    run `git` — and in addition it cannot produce a diff and can lag the true `main` head.
    Task branches and PRs must not be assumed accessible.
  - Either orchestrator asks the user ONE focused question only when its own available
    access cannot establish the required fact.
- Every repository claim must be tied to an **explicit full SHA**. At that SHA, code and
  tests are authoritative, followed by `CLAUDE.md` and this handoff read at the SAME SHA.
  Project Sources, memory, chat summaries, and pasted reports are **not** evidence of current
  repository state.
- Cite code by **file + symbol or exact string**, never by line number. This is the same
  anchoring rule surgical OLD→NEW edits already require, so recon and dispatch share one
  convention.
- Documents can lag code, and lag each other. Observed at the `384845b` baseline: the code
  carried all three scenario-construction preconditions while `CLAUDE.md` still showed the
  pre-migration §1 and a `PENDING` §7 entry.
- `CLAUDE.md` §1 now defines the grade scale as a **trust policy** — C trusted with no
  review, B the orchestrator reads the changed files, A the orchestrator approves the exact
  full reviewed SHA, with line-by-line review when cross-ego isolation or a §5 locked layer
  is touched: GPT reads the exact GitHub `base...candidate` comparison, and Claude receives
  focused changed hunks or targeted evidence from CC. No full diff is pasted into chat. This
  handoff declares only a grade per task; the definition, and the Grade-A routing default
  between the two orchestrators, belong there.

## 1. Current state

- **B1 — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `d6758ac1899621b2ceebcb63afb5e8577184cd91`, merged by
  `bd087c3c18b96f1fe847b4987c73f394a43249c1` (PR #2). `CLAUDE.md` §7 records the lock.
- **B2 — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `e22aee359e06591bdb179ef06a566db90f83a558`, merged by
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3). `CLAUDE.md` §7 records the lock.
  **The pure placement layer is now CONSUMED** by construction-mode `setup_episode`.
- **B3 — CLOSED / MERGED / LOCKED.** Reviewed code SHA
  `dd14ab418c71e3bd615f1198d0c612502642d29b`, integrated into `main` by merge commit
  `14224531db9deb700f6e397203177eb8c701c6cc` (PR #4, merged). The merged tree is
  byte-identical to the approved one (`git diff --quiet dd14ab4 1422453` succeeds).
  `CLAUDE.md` §7 records the lock under `dd14ab4` and §5 carries the full contract; that
  entry is the technical authority, so §2 below stays short.
- **The offline scenario-construction phase is therefore COMPLETE.** The solve → place →
  patch → reload seam is IMPLEMENTED, not future work, and the default world really contains
  hidden targets.
- No active implementation task, candidate, or PR. Ownership of B3 is RELEASED. As with B1
  and B2, the reviewed B3 SHA is the last reviewed **code baseline**, NOT the base SHA for
  the next task — docs-only commits advance `main` further, so B4 must resolve the current
  full `main` SHA immediately before dispatch and declare THAT as its base.
- **No training run has ever been performed, and none is authorized by this document.** What
  changed is WHY: B4 was previously blocked because the world contained nothing to discover.
  That blocker is gone. B4 is now blocked only on its own planning and run preparation (§4).
- The old task branches `task/b1-generator-configuration`,
  `task/b2-route-relative-hidden-placement` and `task/b3-setup-seam` have all been deleted
  remotely after verification. `main`, `flat-final` and the `pre-cleanup` tag are untouched.

## 2. What B3 delivered

Full contract in `CLAUDE.md` §5 (Episode-setup) and the §7 `dd14ab4` lock entry. In brief:

`setup_episode` now has TWO EXPLICIT PATHS, selected by whether the
`(n_hidden, placement_rng)` **pair** was supplied — never inferred from `partial_ratio`:

- **legacy split path** (both omitted): unchanged, one world, `split_tasks` masks the hidden
  half. Retained and still tested;
- **construction path** (both supplied): known-only env-1 → solve `A_init` → locked B2
  placement → patch the scenario JSON → close env-1 → env-2 reload → re-extract → full
  oracle solve → beliefs/executor built from **env-2 objects only**.

Supplying exactly one half raises before any BLADE object exists. `n_hidden=0` is a legal
construction probe that never calls `split_tasks`. Cardinality is exact
(`len(placements) == n_hidden`, no padding or truncation), the cell is airbase-only
(`include_sams=True` rejected by both configs' `validate()`), env-1 closes on every path,
`_build_env` owns cleanup until it returns successfully (the review fix — a failing
`env.reset()` closes the environment it had already created and re-raises the original
exception unchanged), agent IDs must survive the reload, and known tasks are re-materialized
from env-2 **by target id in A_init's positional order**. No env-1 `Agent` or `Task` object
reaches the returned context. `EpisodeContext.placements` is the id-free audit and
construction `split_meta` reports truthful `known/hidden/partial/full` plus a
coordinates-only `geometric_fingerprint`.

`graph_train._run_one_episode` and `graph_rollout.run_rollout` both drive the construction
path with `n_hidden=cfg.n_hidden` and a fresh per-episode `random.Random(seed)`, and report
real emitted counts (`n_targets_emitted == n_known + n_hidden`).

**Evidence at the lock.** Base suite **118 passed, 4 skipped** (102 → 121 collected), import
purity 12/12, `git diff --check` clean; **19/19** `tests/test_graph_setup_seam.py` checks
under `nlp_env`, plus the placement, train, import-purity and legacy `graph_episode_setup`
runners. Private-sensing isolation is proven through the INTEGRATED setup/tick seam — a
setup-constructed hidden world target sensed only by ego A enters only ego A's belief and
executor slice via the unmodified `run_episode` Phase-1 chain, with every peer byte-unchanged
and the target in NO belief at t=0. Live seed-0 reference rollout: 3 agents,
**3 known + 3 hidden = 6 full targets**, `U_oracle = 479.99968` (the frozen-EPSILON form of a
raw 480), **4 organic wakes**, `ended=done`, reward `-0.3333`, both bonmin solves successful,
no `CRASH`/`Traceback`.

That is ONE reference episode at one seed — a feasibility measurement, **not a baseline**.

## 3. Design facts B4 depends on

Route prediction, segment geometry, the scenario/data seam and reproducibility are all
IMPLEMENTED and LOCKED across B2 (`e22aee3`) and B3 (`dd14ab4`). They are contracts now, not
requirements to satisfy — `CLAUDE.md` §3 (no-communication invariants, the single
`DETECTION_KM = 50` radius, launch point == the BLUE airbase), §5 (the layer contracts) and
§2 (frozen BLADE engine, frozen solver) remain in force and are NOT restated here.

Three facts B4 planning must hold onto:

### Route cardinality — an OPEN policy decision, not a bug

B2's locked contract is **one placement per non-empty ego route**, and B3 requires
`len(placements) == n_hidden` exactly. When bonmin leaves an ego unassigned there are fewer
routes than `n_hidden`, `setup_episode` raises, and the harness logs the traceback, counts
the episode failed, and continues.

Measured on the default cell over seeds 0–11: **10/12 produced 3 usable ego routes; seeds 2
and 8 produced only 2**, so those two episodes fail loudly and are skipped. Seed 0 (the
reference) is clean.

**B4 must decide explicitly what a skipped seed means for a training batch** — accept the
~17% loss, reseed past it, relax the cell, or commission the general
`n_hidden != usable ego routes` distribution policy that B2 named as a separate design task.
Do **not** resolve it by weakening the cardinality check, the B2 geometry, or the loud
failure. Recorded in `CLAUDE.md` §8.

### 480 vs 479.99968

The six airbase targets sum to exactly `6 × 80 = 480` raw utility, but `graph_reward.plan_value`
is bit-faithful to `MatchAou._add_objective` and carries the frozen `EPSILON = 1e-6`, giving
`U_oracle = 479.99968` for the measured seed-0 allocation. Both are correct for their own
operand. Do not "fix" either, and do not compare one against the other.

### Solver stall and timeout

Solves with `known ≤ 2` can stall for roughly 15 minutes against a typical 45 s
(`CLAUDE.md` §8). The locked cell is clear of it and both configs WARN, but **a solve timeout
is required before any low-known or n-randomized configuration enters a long run**.

## 4. Work sequence

### B1 — generator and configuration — **CLOSED / MERGED / LOCKED**

`d6758ac1899621b2ceebcb63afb5e8577184cd91`, merged by `bd087c3c…` (PR #2). Explicit
`num_agents` / `n_known` / `n_hidden` cell, strict great-circle geometry
(`min_target_distance_km = 200`, `min_known_separation_km = 100`), known-only emission with
Layer 1 disabled on the construction path, `RolloutConfig` aligned with `TrainConfig`.

### B2 — route-relative hidden-target placement — **CLOSED / MERGED / LOCKED**

`e22aee359e06591bdb179ef06a566db90f83a558`, merged by `8db9428…` (PR #3). The PURE geometry
layer `rl/training/graph_hidden_placement.py`. Closed geometry: `f ~ Uniform[0.60, 0.85]`
over `G = L − D`; `guard_km = 10`; leg-1 cap `D − guard`; later-leg origin uncertainty
`(1 − s/L)·D`; strict `gap > 2·D` with equality rejected; uniform choice among eligible later
legs with valid leg 1 as fallback; loud failure plus independent re-validation. One placement
per non-empty ego route. `CLAUDE.md` §7 is the authority.

### B3 — setup seam — **CLOSED / MERGED / LOCKED**

`dd14ab418c71e3bd615f1198d0c612502642d29b`, merged by
`14224531db9deb700f6e397203177eb8c701c6cc` (PR #4). See §2 above and `CLAUDE.md` §5 / §7.

### B4 — training-run planning and preparation — **NEXT TASK**

The construction phase is closed, so the world now poses a real learning problem. B4 is
planning and preparation FIRST; the run itself is a later, separately authorized step.

Required before any long run:

1. **Fresh exact-SHA initialization** — resolve the current `main` HEAD after this
   documentation PR lands and declare THAT as the base; select the transport mode per
   `CLAUDE.md` §1.
2. **Provenance manifest** — add or verify a manifest carrying the code SHA, the exact
   command and resolved config, the seed bands, the environment, and the solver versions.
   `run_config.json` already records the resolved config and construction cell; decide
   whether the manifest extends it or sits beside it.
3. **Decide the exact-cardinality policy** (§3) — what a failed/skipped seed means for a
   training batch. This is a real decision with a measured ~2/12 incidence; do not let it be
   discovered mid-run.
4. **Re-measure iteration-0 held-out performance** on the post-B3 world. The old
   `95c09dd` numbers are pre-launch-point-fix AND pre-construction; nothing about learning
   headroom may be inherited. Read saturation carefully — near-ceiling held-out performance
   **at iteration 0** means the cell has no headroom and must be revisited before compute is
   spent, whereas saturation **after training** is expected and intended in this phase.
5. **Watch the first two iterations** before committing to a long run — the loop has never
   run to completion. Scale: two bonmin solves per episode at roughly 45 s each, so a few
   hundred episodes is 10–15 hours of wall clock.
6. **Do not invent the final training command or config in a documentation task.**

Difficulty returns later through `p < 1`, `fuel_damage`, and targets that shoot back — not by
weakening this phase's scenario.

## 5. Closed decisions

- Offline construction only; no live world mutation and no injection scheduler.
- Patch-and-reload, never regenerate after solving.
- Route prediction is required and supports `num_agents < n_known`.
- One sensing/arrival/attack/kill-confirmation radius: `DETECTION_KM = 50`.
- Acceptance for the construction phase was structural correctness and feasibility, not final
  learnability.
- `round_trip_cost` and `graph_reward` remain unchanged and frozen. The flat `R ≈ −1/3` was
  scenario degeneracy, never a reward bug.
- Random fleet class mix is acceptable; use relative placement parameters.
- **B1 reference cell (`d6758ac`):** fleet 3, `n_known = 3`, `n_hidden = 3` — a cell, not a
  law; a later phase varies the counts per episode.
- **B1 construction geometry (`d6758ac`):** `min_target_distance_km = 200` km and
  `min_known_separation_km = 100` km, both enforced as a TRUE great-circle floor under
  `strict_geometry`.
- **`RolloutConfig` follows `TrainConfig`'s construction surface (`d6758ac`)** —
  field-for-field, validated the same way, compared by an anti-drift test.
- **B2 placement geometry and eligibility (`e22aee3`):** the fraction range, guard and offset
  caps, the strict `gap > 2·D` tie margin, uniform-among-eligible-later-legs with leg-1
  fallback, one placement per non-empty ego route, an explicit `random.Random`, and id-free
  geometric fingerprints.
- **B2 stays pure (`e22aee3`):** the placement layer takes `detection_km` through
  `PlacementParameters` and never imports `graph_episode_setup`. B3 consumes it; the import
  direction is one-way and must stay that way.
- **B3 path selection and ownership (`dd14ab4`):** the `(n_hidden, placement_rng)` pair
  selects construction mode explicitly and is never inferred; env-1 is temporary and always
  closed; `_build_env` owns cleanup until it returns; env-2 is the sole runtime source of
  truth; agent IDs must survive reload; known tasks are re-materialized by target id in
  A_init order; cardinality is exact; the cell is airbase-only.
- **The legacy split surface is RETAINED, not retired** (`split_tasks`, `partial_ratio`,
  `derived_split`, `split_preview`, `num_red_airbases`, Layer 1). The construction path
  simply does not consult it. Retiring it is still a separate, later phase.

## 6. Out of scope

Do not mix B4 with:

- retiring `split_tasks`, `derived_split`, `split_preview`, or Layer 1;
- legacy import cleanup or deletion;
- changes to `round_trip_cost`, the solver, reward, geometry, cardinality, or the
  reachability mask;
- centralized critic, dense reward, `p < 1`, ETA/peer-dropout work, or fuel damage;
- the README rewrite.

## 7. Documentation duties, with their trigger

| Trigger | Duty |
|---|---|
| B1 lands — **DONE** | update `CLAUDE.md` §6's "change the training scenario cell" row |
| B1's parameterization closed — **DONE** | retire the `min_target_distance_km` item in `CLAUDE.md` §8 |
| B2 lands — **DONE** | add the `CLAUDE.md` §6 placement row and the §7 `e22aee3` lock |
| B3 lands — **DONE** | rewrite `CLAUDE.md` §4's pipeline diagram for the two paths, document the §5 setup contract, add the §7 `dd14ab4` lock, mark the `split_meta["outcome"]` §8 item legacy-path-only, and replace the pre-B3 zero-headroom item — all completed in this documentation lock commit |
| B4's first real run completes | record the measured iteration-0 and final held-out numbers, and whatever the exact-cardinality policy turned out to be in practice |

## 8. Next action

B1, B2 and B3 are closed and the offline scenario-construction phase is complete. **B4 —
training-run planning and preparation — is next**, and B3 ownership is released. The next
orchestrator:

1. performs fresh exact-SHA initialization — resolve the current `main` HEAD (this
   documentation commit's merge, not the B3 reviewed or merge SHA) and declare it as the
   base SHA, and select the transport mode per `CLAUDE.md` §1;
2. conducts focused B4 recon — `graph_train`'s loop and seeding schedule, `run_config.json`,
   the checkpoint surface (save-only; resume is still deferred), and the measured
   route-cardinality incidence;
3. designs the provenance manifest and the exact-cardinality policy (§4);
4. dispatches a short instrumented probe — NOT a long run — to measure iteration-0 held-out
   performance on the post-B3 world;
5. only then proposes the long run, with the first two iterations watched before commitment.

**No training run is authorized by this document.**
