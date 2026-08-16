# Multi-Agent Graph RL — Harness Closure / Final-Cell Probe Handoff

**Supersedes all earlier handoffs.**

Written 2026-08-11; updated 2026-08-14 for the final-cell PROBE HARNESS closure,
2026-08-15 to record the FIRST EXECUTED bounded short probe and the three
research-validity defects it exposed (§3d), 2026-08-16 to record the CLOSURE of ALL
THREE of those defects — **Defect A, ego-global `SELF_PRESERVATION_ABORT`, merged through
PR #17**, **Defect B, the attack-confirmation wait derived from the salvo about to fly,
merged through PR #19**, and **Defect C, physical RTB completion, merged through
PR #21** — and 2026-08-16 again to record the **CORRECTED-CELL SHORT-PROBE RERUN**
(§3e). **The three-defect CODE correction is COMPLETE, the rerun is EXECUTED,
INDEPENDENTLY REVIEWED and VALID, and the long-baseline validity gate is PASSED. No long
baseline has been run.**
B1–B4, the first real post-B3 instrumented probe, the B4 observability follow-up (PR #7),
**FD-BASELINE-v1** (PR #8), **FINAL-CELL-VISUAL-ARTIFACTS** (PR #10), the repository
code-hygiene cleanup (PR #11), the documentation hygiene (PR #12) and now the
**FINAL-CELL PROBE HARNESS** (PR #14) are all CLOSED. None of them changes research
behaviour: the solver, BLADE, reward, fuel-damage semantics, PPO math, seed schedules,
scenario construction and matched-pair evaluation are exactly as their own locks left them.

Baseline **difficulty selection is finished**, the **inspection surface is in place**, and
the **operator harness a probe is driven from — preset, run layout and figures — is
merged**. The bounded short probe has now been run TWICE. The FIRST run
(`training_output_20260815_173029`, from clean `main` at
`238062d7d284334432d9c39d7543fb0bbf39ea7c`) passed every mechanical harness and accounting
check **and** exposed three research-validity defects (§3d). They were corrected in
sequence — A, then B, then C, as separate reviewed tasks, with the policy recorded in
`CLAUDE.md` §8: **all three are CLOSED and MERGED** (PR #17, PR #19 and PR #21). The SAME
bounded probe shape was then rerun ONCE from the corrected `main`
(`training_output_20260816_162130`, exact code SHA
`900ff0b24898eccfa2e35d2db05c4e0229c64ce3`) and independently reviewed:
**VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS** (§3e). **The validity gate is PASSED.**
The corrected cell therefore has real PERFORMANCE evidence at SHORT-PROBE scale — bounded,
24 scheduled attempts, never a baseline. **No long baseline has been run**, and no
long-baseline result may be pre-claimed.

This handoff is volatile and deliberately thin. Technical contracts live in `CLAUDE.md`;
code and tests remain decisive. Where a fact is already in `CLAUDE.md` this document
cross-references it rather than duplicating it.

## 0. How this workspace reads the repository

- Repository access is capability-aware, not shared:
  - GPT resolves repository facts through GitHub first — branches, PRs, files and exact
    SHAs. GitHub is a search interface, not a filesystem.
  - Claude resolves facts through a synchronized mounted snapshot of `main` plus Git
    evidence CC reports. It can lag and cannot prove a historical diff on its own.
  - Either orchestrator asks one focused question only when its available access cannot
    establish a blocking fact.
- Tie every repository claim to an explicit full SHA. Read code, tests, `CLAUDE.md` and
  this handoff at that same SHA. Project Sources, memory, chat summaries and pasted
  narratives are not evidence of current repository state.
- Cite code by file + symbol or exact string, never by line number.
- `CLAUDE.md` §1 owns grading, transport, candidate review, fixes and status reporting.

## 1. Current state

- **CURRENT `main`:** `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, committed
  `2026-08-16T15:26:55+03:00` (the PR #22 merge, which integrated the Defect-C
  documentation lock). This is also the exact code SHA the corrected short-probe rerun was
  measured at. Any later work still performs its own fresh exact-SHA initialization
  against the repository rather than trusting this line.
- **CORRECTED-CELL SHORT-PROBE RERUN — EXECUTED / INDEPENDENTLY REVIEWED / VALID.**
  Run `training_output_20260816_162130`, from a clean checkout at exact code SHA
  `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, ONE invocation of the reviewed preset with
  no typed override. Verdict **VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS**, so the
  long-baseline validity gate is **PASSED**. §3e records its provenance, accounting,
  matched-pair denominators, failures, rewards, PPO evidence, event/RTB/death outcomes,
  artifact completeness and playback witnesses; `CLAUDE.md` §7 owns the authoritative
  measurement record and the evidence hashes, and §8 owns the gate. **It changed no
  tracked file** — it is a run of merged code, not a candidate. **No long baseline has
  been run.**
- **DEFECT C (RTB ISSUANCE is not physical RTB COMPLETION) — CLOSED / APPROVED /
  MERGED.** Approved candidate `ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b`, integrated by
  merge commit `0de9f21eb9e8904f06f836f4ecd010bc46c788b6` (PR #21). The candidate was
  merged with a MERGE COMMIT and preserved as its SECOND PARENT (integration parents, in
  order: `6e97940733d2c7cf8c4ffc7033180c65f644ae17` then `ea62e4e…`); candidate and
  integration share the IDENTICAL tree `6d05cc5ea9af0f6bdcd4a2d6865767bcbe525ebe`, and
  the candidate→integration comparison contains ZERO changed files. Implementation fixed
  base `6e97940733d2c7cf8c4ffc7033180c65f644ae17`. Grade A under `GPT_GITHUB`, mode
  BUILD, exactly SIX files (`src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `src/match_aou/rl/training/graph_tick_loop.py`,
  `src/match_aou/rl/training/graph_episode_setup.py`, `tools/graph_executor_smoke.py`,
  `tests/test_graph_setup_seam.py`, `tests/test_graph_fuel_damage.py`). The first
  candidate `5a0809df1a490df6ff266343788655d32fcefd81` was corrected by the NEW CHILD
  COMMIT `ea62e4e…` on the same branch and PR, with no amend, rebase, force-push or
  history rewrite. **The approved semantics:** `is_done(observation)` requires the LIVE
  post-step observation and has no observation-free default; assignment completion still
  comes from executor semantic state (`plans`, ego-private task resolution and the
  proximity-confirmed `done` set), while the PHYSICAL half comes from `_physical_state`
  (airborne / landed / removed) and `_note_dead` reconciles a newly observed death
  idempotently into `executor.dead` — for every ego, before the global verdict — so a
  death on the ride home reaches `EpisodeResult.n_dead` and the UNCHANGED reward formula
  charges the airframe really lost. `rtb_issued` keeps its ONE job as the single-issue
  BLADE-toggle guard and is never survival, landing or completion; an ego whose latch is
  set leaves Phase 1 entirely while peers continue normally and Phase 2 still runs every
  tick. The `add_return_to_base=False` contract is preserved and BLADE is unchanged.
  `CLAUDE.md` §4, §5 (Execution, Stage 1, and the tick loop), §6 and §7 own the
  authoritative contract, routing and lock. **This closes DEFECT C ONLY.** No probe,
  training run, rollout or baseline was run for it.
- **DEFECT B (the attack-confirmation wait derived from the salvo about to fly) —
  CLOSED / APPROVED / MERGED.** Approved candidate
  `39a16f2e5e1a3302d545c11b072e037e9702dffe`, integrated by merge commit
  `60a82d17398e9d14be1c2684cc72fafd020e0d9b` (PR #19). The candidate was merged with a
  MERGE COMMIT and preserved as its SECOND PARENT; candidate and integration share the
  IDENTICAL tree `ee86f0782ac50ee8bd0ee2fe634393a9cfc53a66`, and the
  candidate→integration comparison contains ZERO changed files. Implementation fixed base
  `cefda78b18ea2daeda5014bab9a75a0945ef8e37`. Grade A under `GPT_GITHUB`, mode SURGICAL,
  exactly TWO files (`src/match_aou/utils/blade_utils/blade_graph_executor.py`,
  `tests/test_graph_setup_seam.py`). The first candidate
  `45a0352312ae308df76a506a8e2e9907a9531a43` had its RUNTIME IMPLEMENTATION ACCEPTED and
  received REQUEST-FIXES on three DOCUMENTATION-accuracy findings; the correction landed as
  a NEW CHILD COMMIT on the same branch and PR, with no amend, rebase, force-push or history
  rewrite, and its runtime-relevant executor token stream is unchanged. `CLAUDE.md` §5
  (Execution, Stage 1), §6 and §7 own the authoritative contract, routing and lock.
  **This closes DEFECT B ONLY.** No probe, training run, rollout or baseline was run for
  it.
- **DEFECT A (ego-global `SELF_PRESERVATION_ABORT`) — CLOSED / APPROVED / MERGED.**
  Approved candidate `d56fda636ab5ec1a5cce6076f07acac5556d10cb`, integrated by merge
  commit `f094e0b32e5e67b79757edbfe4e73c1fe01b0a87` (PR #17). The candidate was merged
  with a MERGE COMMIT and preserved as its SECOND PARENT; candidate and integration share
  the IDENTICAL tree `70e5af2446f0a1b0674eb10819c9451753260560`, and the
  candidate→integration comparison contains ZERO changed files. Grade A under
  `GPT_GITHUB`, mode SURGICAL, exactly five files. The first candidate
  `c306455085de408c7bf383135c27e600ff3f1428` received REQUEST-FIXES for three
  documentation inaccuracies; the correction landed as a NEW CHILD COMMIT on the same
  branch and PR, with no amend, rebase, force-push or history rewrite. `CLAUDE.md` §5
  (Stage 4 SELECTION, Stage 5 EFFECT) and §7 own the authoritative contract and lock.
  **That closed DEFECT A ONLY.** No probe, training run, rollout or baseline was run for
  it.
- **B1 — CLOSED / MERGED / LOCKED.**
  `d6758ac1899621b2ceebcb63afb5e8577184cd91`, merged by
  `bd087c3c18b96f1fe847b4987c73f394a43249c1` (PR #2).
- **B2 — CLOSED / MERGED / LOCKED.**
  `e22aee359e06591bdb179ef06a566db90f83a558`, merged by
  `8db9428147b77e9432e7ad6b085dc5898c9062bb` (PR #3).
- **B3 — CLOSED / MERGED / LOCKED.**
  `dd14ab418c71e3bd615f1198d0c612502642d29b`, merged by
  `14224531db9deb700f6e397203177eb8c701c6cc` (PR #4).
- **B4 preparation — CLOSED / MERGED / LOCKED.**
  `1b48145f4ba6ed542c27ab6ed7a9ea3e6f6ab12c`, merged by
  `ba936606deada050ed9298600ee9041fc330af6c` (PR #6).
- **First real post-B3 instrumented probe — CLOSED / REVIEWED MEASUREMENT of the
  PRE-FD cell.** Ran from the clean exact code SHA
  `a3f0838616990987bcb8a51665fa75d84edf5952`. §2 records what it measured and what it
  does NOT establish.
- **B4 per-episode observability follow-up — CLOSED / MERGED / LOCKED.**
  `211e12e49b676637362d42effdb80988dd0e55eb`, merged by
  `ffb95a6ee90df45b2d89802b321dcadcbc272821` (PR #7). Candidate
  `24241690572a7a5264e24348db5e9412b41bc47a` received REQUEST-FIXES; the approved
  correction was a new commit, never rewritten.
- **FD-BASELINE-v1 — CLOSED / MERGED / LOCKED, and it remains so.** Approved candidate
  `a8669f450708c2508753c49ab16fd1028b29607d`, integrated by
  `1cecb0ac99f839d47ffeea12c8871aec77e66640` (PR #8); the merged tree was independently
  verified identical to the approved candidate tree. The FIRST candidate
  `1cf53fcee3ee05b3466c8391cbc6bb04420a0985` received REQUEST-FIXES; the correction landed
  as a NEW CHILD COMMIT on the same branch and PR, with no amend, rebase, force-push or
  history rewrite. §3 summarizes the factor; `CLAUDE.md` §5 and §7 own the authoritative
  contract and lock. Nothing in the visual-artifact work below changed it.
- **FINAL-CELL-VISUAL-ARTIFACTS — CLOSED / MERGED / REVIEWED.** Approved candidate
  `24d1835f31d2e6aac04b418308a8753c392ac951`, integrated by
  `771f2107211fb3f984b64482b799613260e19aca` (PR #10); the merged tree was verified
  byte-identical to the approved candidate tree (`git diff --quiet 24d1835 771f210`).
  Grade A under `GPT_GITHUB`, mode SURGICAL, exactly two files
  (`src/match_aou/rl/training/graph_train.py`, `tests/test_graph_train.py`). §3b
  summarizes it; `CLAUDE.md` §5, §6 and §7 own the contract, routing and lock.
- **Repository code-hygiene cleanup — CLOSED / APPROVED / MERGED.** Approved candidate
  `2a3f89cf2d027581308493a98767ae658107d6d1`, integrated by
  `6e2757dd30100f429d492f4d23fd8b5f57cf4fac` (PR #11). Grade A under `GPT_GITHUB`, mode
  SURGICAL. The retired `blade_executor_minimal.py` is **removed from current `main`**, and
  its one live helper `nearest_neighbor_order` now lives in
  `src/match_aou/utils/scheduling_utils.py`, imported by BOTH `GraphPlanExecutor._eligible`
  and `graph_hidden_placement.predict_route` — so online execution and offline route
  prediction still share exactly ONE implementation. Verified no behaviour change: placement
  geometry and executor eligibility were byte-identical to the pre-merge base over seed and
  world sweeps, full suite 216 passed / 4 skipped, import purity 12/12. No training,
  rollout, BONMIN solve or scientific probe was run. `CLAUDE.md` §7 owns the lock.

- **FINAL-CELL PROBE HARNESS — CLOSED / MERGED / APPROVED.** Approved candidate
  `61e539ed62fcf1e3fe25a83d213cae06f5afa98e`, integrated by
  `a5f389a2af328640e19db51d3277a33167c08f25` (PR #14); GitHub verification established
  that candidate…merge has ZERO changed files, so the merged tree is exactly the reviewed
  tree. Grade A under `GPT_GITHUB`, mode SURGICAL — **the grade was corrected from B to A
  during review** because `graph_train.py` belongs to the §5 locked trainer contract, with
  no implementation redo required. Four files: `src/match_aou/rl/training/graph_train.py`,
  `tests/test_graph_train.py`, `README.md`, and the new
  `configs/graph_train/final_cell_probe.json`. TWO REQUEST-FIXES rounds landed as new child
  commits on the same branch and PR (`4238e0e` → `de51883` → `61e539e`) with no amend,
  rebase, force-push or history rewrite. §3c summarizes it; `CLAUDE.md` §5, §6 and §7 own
  the contract, routing and lock.

- **No scientific probe or training run had been performed on the merged final cell at the
  time of the locks above.** Every test behind every one of these locks is solver-free and
  drives the pipeline through stubbed engine seams. The locks certify implementation; they
  say nothing about how the cell behaves. **The harness lock in particular measured
  nothing**: it certifies that a probe can be configured, run and read, not that the cell
  learns anything. That measurement gap is now PARTLY closed — and only partly — by the
  executed short probe recorded in the next bullet and detailed in §3d.
- **FIRST BOUNDED SHORT PROBE OF THE FINAL FUEL-DAMAGE CELL — EXECUTED / REVIEWED /
  SCIENTIFICALLY INCONCLUSIVE.** Run `training_output_20260815_173029`, executed from a
  clean checkout at exact code SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`. It confirms
  HARNESS AND ACCOUNTING OPERABILITY and nothing more: the process exited normally,
  `run_summary.json` reported `accounting_reconciled=true`, training accounting was 8
  attempted / 6 successful / 2 `setup` failures, evaluation was 16/16 successful with 4/4
  complete matched pairs in BOTH the `pre_update` and the `post_update` round, and two
  productive PPO updates completed. It ALSO exposed three research-validity defects in
  merged, previously locked behaviour — abort semantics, premature attack re-fire, and
  episode termination on RTB ISSUANCE rather than RTB COMPLETION. **Its post-update reward
  improvement is therefore NOT scientific evidence about the fuel-damage cell**. §3d records
  the run state, the three defects and the decided direction. **All three defects are now
  CORRECTED and MERGED** (see the Defect-A, Defect-B and Defect-C bullets at the top of
  this section and `CLAUDE.md` §7). That does NOT rehabilitate this run: it was executed at
  `238062d…`, before any of the three corrections existed, so its numbers remain historical
  evidence about the OLD behaviour and are not evidence about current `main`. The
  corrected-cell measurement is the SEPARATE rerun in §3e, and it is that rerun — never
  this one — that passed the validity gate.
- **Repository documentation hygiene — CLOSED / APPROVED / MERGED.** Approved candidate
  `52064c2d306df7c8447d159df20e6e189a59bf85`, integrated by
  `5f78904e3af1e2e47386c9b0e01ddbaa273724f5` (PR #12); the approved candidate tree was
  verified identical to the integration tree. Grade C, documents and dead data only —
  `README.md` replaced, `docs/BLADE_API_DOCUMENTATION.md` audited against the vendored
  fork, four obsolete scenario JSONs and two dead utility symbols removed. It changed no
  research behaviour. The FIRST candidate `6302847bdc8b5e40313763b4b167af85dd0a462e`
  received REQUEST-FIXES; the correction landed as a new child commit, never rewritten.
  `CLAUDE.md` §7 owns the lock.
- **All PRIOR repository-hygiene work is CLOSED, and this document is the phase's final
  closure record.** Both hygiene tasks are merged (PR #11 code, PR #12 documentation) and
  their branches `task/repo-code-hygiene` and `task/repo-doc-hygiene` have been **deleted**
  locally and remotely by safe deletion, each verified an ancestor of `main` first; every
  reviewed candidate tip stays reachable on GitHub through `refs/pull/<n>/head`. The earlier
  implementation branch `task/final-cell-visual-artifacts` was likewise verified and
  deleted. `flat-final` (`4d44c3454a5561a6cb9d7aed593d59a40068d6d7`) and the `pre-cleanup`
  tag (peeling to `561b7cb7f2d873e584a8c0dabe71df8050f1b4ed`) are untouched.
  **Ownership is RELEASED once this closure record is integrated into `main`.** No further
  repository-hygiene task follows it. **Live branch and PR state — including whatever
  delivered this record — must be resolved from GitHub, never from this document**, and the
  next orchestrator performs fresh exact-SHA initialization: **resolve the current full
  `main` SHA from the repository** rather than reusing any SHA named here.

## 2. Historical probe — evidence about the EASY PRE-FD cell only

Measured code: exact clean SHA `a3f0838616990987bcb8a51665fa75d84edf5952`. Shape: two
iterations × four scheduled train attempts, seeds `[0,8)`, with the same fixed four
held-out seeds `[1000000,1000004)` before training and after two completed updates.

- Provenance complete (exact SHA, `dirty=false`, Windows / `nlp_env`, vendored BLADE,
  BONMIN); the run completed normally in 79.21 s.
- `pre_update`, `updates_completed=0`: reward mean `-0.4999997395829586`, **4/4**.
- Training **7/8** successful, all with wakes; **24 transitions**; two productive
  iterations and two PPO updates. The single failure was train seed 2 at `setup` — B2
  produced two placements for three requested hidden targets — attempted once, recorded
  once, never retried or replaced.
- `post_update`, `updates_completed=2`: reward mean `5.000007394910353e-7`, **4/4**.
- `run_summary.json:accounting_reconciled=true`; all six durable artifacts existed.
- Evidence SHA-256 for the six artifacts is recorded in `CLAUDE.md` §7 under `a3f0838`.

**That cell contained NO difficulty factor.** It is preserved as historical evidence that
the pipeline collects data and updates, and that the easy cell had headroom the loop could
close. It is **not** a baseline, and it is **not** evidence about the fuel-damage cell —
those numbers must never be reused as the new cell's expected behaviour, and the probe in
§4 must not be judged against them.

## 3. What PR #8 closed — FD-BASELINE-v1

Authoritative contract: `CLAUDE.md` §5 ("FD-BASELINE-v1 — the difficulty factor") and §4
for the tick placement. Lock, fix chain and verification: `CLAUDE.md` §7 under `a8669f4`.
Summarized here only far enough to hand over.

**The selected factor and its research role.** The pre-FD cell was learned in a two-update
probe: the static plan already sat close to the oracle, and the only adaptation on offer
was engaging a pop-up the ego happened to fly past. FD-BASELINE-v1 adds exactly ONE
attributable source of adaptation difficulty — a seeded, ego-local, one-shot fuel-damage
event partway along an ego's first planned leg, sized so that flying home stays feasible
while completing the route and returning does not. That turns `SELF_PRESERVATION_ABORT`
from a never-correct action into a live alternative, and the explicit
`RewardConfig(aircraft_penalty_coeff=2.25)` is what gives losing the airframe a cost.

**The scenario cell is otherwise unchanged**: 3 agents, 3 known + 3 route-relative hidden
airbase targets, 200 km / 100 km geometry, `DETECTION_KM = 50`, `include_sams=False`,
`probability = 1`, unchanged BLADE weapon lethality, frozen solver, unchanged PPO, and the
`graph_reward` formula untouched.

**Deferred, NOT bundled.** `probability < 1`, enemy fire / SAMs and dense (per-wake) reward
were considered and **not selected**. Each remains a separate research change with its own
semantics, observability and proof obligations, and none may be enabled implicitly — see
`CLAUDE.md` §8. In particular `p` remains 1, so the reward operand scale is not reopened.

**Verification at the approved head** (`CLAUDE.md` §7 has the full statement): full suite
192 passed / 4 skipped, 35 fuel-damage tests, 73 graph-train tests, import purity 12/12,
`graph_trigger` selftest green, `git diff --check` clean.

## 3b. What PR #10 closed — FINAL-CELL-VISUAL-ARTIFACTS

Authoritative contract: `CLAUDE.md` §5 ("Visual artifacts — the opt-in inspection
surface"); routing in §6; lock and verification in §7 under `24d1835`. Summarized here
only far enough to hand over.

**What it is for.** A finished probe is otherwise a directory of numbers. With
`TrainConfig.visual_artifacts` / `--visual-artifacts` enabled, every scheduled
`pre_update` / `train` / `post_update` attempt keeps one collision-free bundle under
`<run_dir>/visual_artifacts/`: the byte-identical generated known-only scenario, the
authoritative executed t=0 scenario taken from `ctx.game.export_scenario()` on the env-2
game before the fuel-damage controller exists, the BLADE playback produced through the
existing `setup_episode(recording_export_path=...)` + `run_episode` contract, and an
`artifact_manifest.json` stating phase, iteration, `updates_completed`, ordinals, exact
seed, scheduled condition and exact scenario tag. A whole bundle is `complete`; a bundle
left behind by a failed attempt is clearly marked `incomplete`.

**What it is not.** It is observation, not measurement. Nothing captured is read back into
the pipeline. **OFF by default, and OFF is byte-unchanged** — no directory, no identity, no
copy, no `Game.export_scenario` call, and neither the `recording_export_path` nor the
`artifacts` keyword is passed at all. No seed, scenario tag, scenario name, RNG draw,
policy inference, PPO input, solver input, reward, fuel-damage semantic, failure taxonomy
or BLADE behaviour changed.

**Failure routing.** An artifact filesystem / serialization failure is INFRASTRUCTURE:
`_VisualArtifactError` is re-raised ahead of the broad episode handlers, aborts the run
loudly, and never enters `skip_and_account_v1` or `episode_failures.jsonl`, so it cannot
shrink a scientific denominator by masquerading as an episode failure.

**Verification at the approved head** (`CLAUDE.md` §7 has the full statement):
`tests/test_graph_train.py` 89 passed in BOTH environments, setup-seam + fuel-damage
regressions 66 passed / 4 skipped, import purity 12/12, full suite 208 passed / 4 skipped,
`git diff --check` clean, plus three mutation checks confirming the load-bearing tests
falsify.

**No live BLADE/BONMIN probe, training run, rollout, artifact-generating smoke or
scientific baseline was performed** for either PR #8 or PR #10.

## 3c. What PR #14 closed — the FINAL-CELL PROBE HARNESS

Operator-facing only. `CLAUDE.md` §5 ("Experiment harness") owns the contract; this is the
short form.

- **`--config <path>`** loads a JSON preset of `TrainConfig` FIELDS (nested PPO knobs under
  `"ppo"`). Precedence: dataclass / CLI defaults < preset < EXPLICITLY typed CLI flags.
  `TrainConfig` stays the configuration authority; an unrecognized key raises.
- **`configs/graph_train/final_cell_probe.json`** is the repository's ONLY preset and is
  the bounded short probe: 2 scheduled training iterations × 4 scheduled attempts, base
  seed 0, `eval_every` 2, four fixed held-out seeds from 1_000_000 → one `pre_update` and
  one `post_update` matched round, the final 3-agent / 3-known / 3-hidden cell,
  FD-BASELINE-v1, visual artifacts ENABLED. **Two scheduled iterations do NOT imply two
  productive PPO updates** — `updates_completed` may be 0, 1 or 2, since a zero-wake
  iteration runs no epochs.
- **`run_config.json:/config_source`** is always a structured object with exactly three
  truthful `resolved_from` kinds: `config_file`, `cli_defaults`, `direct_config`.
- **Figures moved to `<run_dir>/plots/`** and the single four-panel `training_plot.png` is
  retired: `training_performance.png`, `policy_diagnostics.png`, `measurement_health.png`.
  Training reward is separated from held-out evaluation; clean and damaged held-out means
  are shown separately and are each over THAT condition's own successful episodes; the
  within-seed comparison is the matched-pair delta over COMPLETE pairs; measurement health
  preserves the episode, pair, wake and per-condition denominators; and the shared x
  quantity is PPO updates completed before the measurement.
- **Run organization is unchanged in kind**: everything still lives under ONE run root —
  records, `scenarios/`, `checkpoints/`, optional `visual_artifacts/`, and now `plots/`.
- **It ran no probe.** No BONMIN solve, BLADE episode, training run, rollout or selftest
  was executed for this PR; every test is solver-free, and the figures were rendered from
  SYNTHETIC records.

## 3d. The executed bounded short probe — operability CONFIRMED, scientific reading BLOCKED

Run identifier `training_output_20260815_173029`, executed from a clean checkout at exact
code SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`, in the merged preset's shape: 2
scheduled training iterations × 4 scheduled attempts, plus the fixed held-out matched
`pre_update` / `post_update` rounds.

**What it establishes — harness and accounting OPERABILITY only.**

- the process exited normally;
- `run_summary.json` reported `accounting_reconciled=true`;
- training accounting: **8 attempted, 6 successful, 2 `setup` failures**;
- evaluation accounting: **16/16 successful**, with **4/4 complete matched pairs in the
  `pre_update` round and 4/4 in the `post_update` round**;
- **two productive PPO updates** completed.

These facts say the instrument runs and accounts for itself. They do **not** authorize the
long baseline, because the same run exposed the three research-validity defects below.

**Defect A — `SELF_PRESERVATION_ABORT` WAS node-indexed, not an ego-global abort.**
**STATUS: CLOSED / APPROVED / MERGED through PR #17** — approved candidate
`d56fda636ab5ec1a5cce6076f07acac5556d10cb`, integrated by
`f094e0b32e5e67b79757edbfe4e73c1fe01b0a87`, identical tree
`70e5af2446f0a1b0674eb10819c9451753260560` and a zero-file candidate→integration
comparison. The observations below are preserved as HISTORICAL EVIDENCE about the probe's
own SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`; they describe behaviour that no longer
exists on current `main`.

- At this SHA `graph_effect.apply_meta_action` removes only the assignment(s) whose
  `task_idx == node_v`, so SPA aborts ONE task rather than the ego's mission.
- Probe playback showed a fuel-damaged KC-135 selecting SPA while its existing BLADE route
  continued and further assignments remained.
- **The decided behaviour — now IMPLEMENTED and MERGED:** an **ego-global mission
  abort**. Selecting `SELF_PRESERVATION_ABORT` on ANY legal cell clears ALL of that ego's
  remaining assignments, so the executor reaches its empty-plan RTB path. It was **NOT
  implemented at the SHA above**; it IS implemented on current `main` (`f094e0b`).
- The existing `k × 3` action-head structure was **not** redesigned: the action surface,
  `NUM_META_ACTIONS`, the mask rules and the sampled/stored/PPO-re-scored `(node, meta)`
  cell are all unchanged, and only the EFFECT of the abort cell became ego-global. The
  merged implementation is proven end to end — pure effect layer, the real
  `graph_tick_loop._wake_decision` chain, and a solver-free REAL-BLADE test in which the
  stale route is replaced by the ride home (`CLAUDE.md` §7).
- Execution-seam fact that narrows the diagnosis: `graph_tick_loop._wake_decision` already
  resyncs the edited ego plan before Phase 2, and an actually EMPTY plan should make
  `GraphPlanExecutor` emit `aircraft_return_to_base`, whose BLADE handling replaces the
  stale route with the home-base route. The observed stale route is therefore currently
  explained by SPA not emptying the plan — **not** by evidence of a missing resync call.

**Defect B — premature re-fire exhausts weapons.**
**STATUS: CLOSED / APPROVED / MERGED through PR #19** — approved candidate
`39a16f2e5e1a3302d545c11b072e037e9702dffe`, integrated by
`60a82d17398e9d14be1c2684cc72fafd020e0d9b`, identical tree
`ee86f0782ac50ee8bd0ee2fe634393a9cfc53a66` and a zero-file candidate→integration
comparison. The observations below are preserved as HISTORICAL EVIDENCE about the probe's
own SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`; they describe behaviour that no longer
exists on current `main`.

- In the `post_update` damaged eval seed `1000003`, B-2 Spirit #698 engaged its
  route-relative hidden targets successfully but reached the final known target
  `Floridistan AFB #4067` with **zero onboard weapons**, and then remained over it until
  fuel exhaustion.
- **Artifact RECONSTRUCTION of the sequence — read as a reconstruction, not as a
  controlled measurement:** at approximately t=5140 the final 2 AIM-120 launched at Hidden
  Airbase #003; at approximately t=5240 2 AIM-9 launched at Hidden Airbase #001 from about
  47.2 km; at approximately t=5300, before that slower AIM-9 salvo resolved, the fixed
  60-tick confirmation cooldown expired and a redundant second salvo consumed the final
  2 AGM-65 — and the AIM-9 salvo killed the target in that same engine update, leaving the
  B-2 with no weapons for the final known target. The distances and tick indices here are
  inferred from the run's artifacts; **the merged fix's own real-BLADE proof is a separate,
  controlled construction and neither of its two arms is a rerun of this episode.**
- Code anchors: `GraphPlanExecutor.kill_confirm_ticks`,
  `GraphPlanExecutor._command_for_ego`, `Game.handle_aircraft_attack`,
  `weaponEngagement.launch_weapon`.
- **The decided direction — now IMPLEMENTED and MERGED:** not raising the constant blindly,
  but DERIVING a conservative confirmation wait from the ACTUAL auto-selected live weapon
  and the CURRENT engagement distance, with the configured `kill_confirm_ticks` kept as its
  FLOOR and FALLBACK. Current lethality, the two-argument attack command and FROZEN BLADE
  behaviour are preserved, and the probabilistic-miss / weapons-exhaustion redesign stayed
  OUT of scope. `CLAUDE.md` §5 (Execution, Stage 1) owns the contract; §7 owns the lock.
- **What the merged real-BLADE proof measured**, both engagements inside the single
  `DETECTION_KM = 50` envelope and at the production default `kill_confirm_ticks = 60`:
  at **~47.2 km** — the distance reconstructed from this probe's artifacts — the
  conservative bound is 62 and the derived wait 63, so the flat 60 was ALREADY below the
  bound, and the control arm escaped a redundant salvo by exactly ONE tick (real
  confirmation on call 60). At **~49.0 km** — a CONTROL ARM in the same envelope, far
  enough out that the one-tick escape is gone (bound 64, derived wait 65, real confirmation
  on call 62 against a re-fire on call 61) — the flat-60 arm DOES exhibit the premature
  re-fire and loses the reserve, while the derived wait fires exactly once and keeps it.
  The 49.0 km arm demonstrates the SAME MECHANISM inside the same envelope; it is **not**
  a rerun of the probe world above. **Neither arm is a scientific probe result.**

**Defect C — RTB ISSUANCE is not physical RTB COMPLETION.**
**STATUS: CLOSED / APPROVED / MERGED through PR #21** — approved candidate
`ea62e4e33eb8d17b773d9742aa8dfd577fe3d98b`, integrated by
`0de9f21eb9e8904f06f836f4ecd010bc46c788b6`, identical tree
`6d05cc5ea9af0f6bdcd4a2d6865767bcbe525ebe` and a zero-file candidate→integration
comparison. The observations below are preserved as HISTORICAL EVIDENCE about the probe's
own SHA `238062d7d284334432d9c39d7543fb0bbf39ea7c`; they describe behaviour that no longer
exists on current `main`.

- At this SHA `GraphPlanExecutor.is_done()` treated the `rtb_issued` lifecycle latch as
  RTB-resolved, and `run_episode` stopped when `executor.is_done()` became true — so an
  episode could end immediately after an RTB command while the aircraft was still
  airborne.
- In the `post_update` damaged eval seed `1000000`, the damaged KC-135 completed its work
  and an RTB command was eventually issued while it no longer had enough fuel to physically
  reach home; the episode nevertheless ended before the resulting fuel exhaustion / death
  could occur, recorded `dead=0`, and contributed reward 0.
- **That observation is INFERRED FROM THIS RUN'S ARTIFACTS. It is NOT one of the merged
  fix's proof tests**, which are a separate, controlled real-BLADE construction and are
  not a rerun of this episode.
- **The decided direction — now IMPLEMENTED and MERGED:** separate **"RTB command
  issued"** from **"RTB physically resolved"**. `is_done(observation)` now requires the
  LIVE post-step observation; assignment completion still comes from executor semantic
  state, the physical half from `_physical_state` (airborne / landed / removed), and
  `_note_dead` reconciles a newly observed death into `executor.dead` for every ego before
  the verdict, so a burn-out on the ride home reaches `EpisodeResult.n_dead` and the
  UNCHANGED reward formula charges it. The single-issue RTB toggle protection is
  preserved, an ego that has committed to return leaves Phase 1 while peers continue, and
  BLADE stays FROZEN. `CLAUDE.md` §4 and §5 own the contract; §6 the routing; §7 the lock.
- **What the merged proof measured**, against the real engine: an ego ordered home from
  120 km out with fuel to spare keeps ticking past the order and the episode ends only
  once BLADE has put it back in an airbase inventory (`dead=0`); given HALF the fuel the
  engine itself says the trip needs, the same construction removes it mid-return, counts
  it dead and the unchanged reward path charges the airframe. The death is pinned by a
  DIRECT CAUSAL WITNESS on `Game.remove_aircraft`: exactly ONE recorded removal of that
  ego, at `current_fuel <= 0`, with no replacement airframe in any inventory. **Neither
  arm is a scientific probe result.**

**Scientific interpretation.**

- The first probe is USEFUL and did its job: it successfully exposed these defects.
- Its post-update reward improvement **must NOT** be treated as scientific evidence for the
  fuel-damage cell, because episode termination (Defect C) and abort semantics (Defect A)
  can distort the measured airframe penalty — precisely the quantity FD-BASELINE-v1 exists
  to make real. **Closing all three defects does NOT rehabilitate this run:** it was
  executed at `238062d…`, before any of the corrections existed, so its numbers remain
  historical evidence about the OLD behaviour and are not evidence about current `main`.
- All three fixes were reviewed and merged, their documentation/lock duty closed, and the
  SAME bounded short-probe shape was then rerun ONCE from the corrected `main`. **That
  rerun is recorded in §3e and it is the corrected-cell measurement**; closing the three
  defects never constituted one by itself.
- **The long baseline was BLOCKED at the time this section was written. It is no longer
  blocked on the rerun** — §3e passed the validity gate. It remains subject to its own
  separately authorized recon and execution, and no long-baseline result may be
  pre-claimed.

## 3e. The corrected-cell short-probe RERUN — EXECUTED / REVIEWED / VALID

Run identifier `training_output_20260816_162130`, executed from a clean checkout at exact
code SHA `900ff0b24898eccfa2e35d2db05c4e0229c64ce3` (`2026-08-16T15:26:55+03:00`), in the
merged preset's shape. `CLAUDE.md` §7 owns the authoritative measurement record, including
the six evidence SHA-256 hashes; §8 owns the gate. Summarized here far enough to hand over.

**Exact provenance and configuration.** ONE invocation, native exit code **0**:
`conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train --config configs/graph_train/final_cell_probe.json`.
Tracked checkout clean before and after; **no code, configuration, preset or
research-semantic change accompanied the run**. Preset blob
`3c85e5bdc780600fe1ee528b3e35fc71591fe4b7`, `config_source.resolved_from = config_file`,
`cli_overrides = []` — the reviewed preset ran with NO typed override, so there is no
deviation to report. Provenance complete: `git.available=true`, exact SHA, `branch=main`,
`dirty=false`, Windows / `nlp_env`, vendored BLADE, BONMIN available and probed `ok`.
`difficulty.factor = fuel_damage_baseline_v1`, `aircraft_penalty_coeff = 2.25`,
`reward.formula_changed = false`. **Elapsed time is TWO DISTINCT QUANTITIES, never
merged:** the harness's own `run_summary.json:run_seconds = 204.50847799982876`, and the
externally measured invocation wall clock of **223.117 s** (the preserved probe runner's
`timing.txt`: `PROBE_START_UTC = 2026-08-16T13:21:24.1839202Z` → `PROBE_END_UTC =
2026-08-16T13:25:07.3008671Z`). The harness figure excludes process start-up, `conda run`
dispatch, imports and teardown, so it is NOT the wall clock.

**Verdict: VALID MEASUREMENT / CORRECTED SHORT-PROBE PASS**, judged by the §4 validity gate
and not by whether reward improved — complete clean provenance; `accounting_reconciled =
true`; no INFRASTRUCTURE failure; **4/4 complete matched pairs in BOTH rounds**; complete
PPO-update and artifact evidence.

**Accounting — `skip_and_account_v1`, every denominator explicit.** **24 scheduled
attempts, 22 successful, 2 failed, both at `setup`.** Clean **11 attempted / 10 successful
/ 1 failed**; damaged **13 attempted / 12 successful / 1 failed**. Training 8 / 6 / 2 with
all six successes producing wakes; evaluation **16/16 successful**. No retry, no
substitution, no seed-band shift. The two failures:

- train seed 2, **damaged**, `setup`, `RuntimeError` — B2 produced only two placements for
  `n_hidden=3` (the static solve left one ego without a non-empty route);
- train seed 4, **clean**, `setup`, `EpisodeRosterError` — one t=0 known target was absent
  from the executed world.

**Fuel-damage event yield — mind the denominator.** The event fired and woke its selected
ego in **12/12 successfully completed damaged episodes**, which is **12/13 SCHEDULED
damaged attempts**, because the damaged seed-2 attempt failed at `setup` before an event
could exist. It must never be stated as 12/12 scheduled damaged attempts.

**Matched held-out evaluation** — same fixed band, each seed run `forced_clean` and
`forced_damaged`, 8 attempted / 8 successful per round:

| Round | Pairs | Clean mean | Damaged mean | Paired delta | Eval deaths | Unique targets |
|---|---|---|---|---|---|---|
| `pre_update` (`updates_completed=0`) | 4/4 | `-0.4999997395829586` /4 | `-0.8749999192707323` /4 | `-0.37500017968777366` over 4/4 | 4 | 3.00 |
| `post_update` (`updates_completed=2`) | 4/4 | `-0.12499955989518505` /4 | `-0.4583330529509838` /4 | `-0.33333349305579874` over 4/4 | 0 | 4.25 |

Held-out OVERALL mean moved `-0.6874998294268455` → `-0.2916663064230844`, **each over 8/8
completed eval episodes**. Post-update damaged **real RTB command yield 4/4**. Training
produced **26 transitions** across 6 successful episodes, **two productive iterations** and
`updates_completed = 2`. The deterministic post-update eval meta-action mix was
`PLAN_COMPLIANCE 0 / OPPORTUNISTIC_ENGAGEMENT 18 / SELF_PRESERVATION_ABORT 13` (the
`pre_update` round was `52 / 0 / 0`). **All of these are SHORT-PROBE OBSERVATIONS, not
estimates of converged policy performance**, and the two per-condition means are each over
their own successful subset — the only within-seed claim is the paired delta.

**Artifact completeness**, reported alongside the scientific denominators and never in
place of one: **24 bundles and 24 manifests; 22 `complete`** (matching the 22 successful
attempts) **and 2 `incomplete`** (matching the two `setup` failures). Every complete bundle
holds its known-only scenario, executed t=0 scenario and BLADE playback; executed-world
cardinality reconciles at **3 known / 3 hidden / 6 total** on every one; **neither
incomplete bundle fabricated a playback**. The complete scientific artifact is preserved.

**Playback witnesses and corroborating run evidence.** All of the following concerns the
attempt preserved as
`visual_artifacts/post_update_r001_e003_m1_seed1000003_damaged_tag901007`, and **two
evidence sources are involved which are deliberately not merged.** The BLADE **playback
JSON** proves PHYSICAL state only — position, fuel, `rtb`, route, weapon inventory, airbase
membership — is sampled every ten ticks (offsets below are from the recording's first
frame), and records **no per-wake meta-action label**; the jsonl records persist per-round
and per-iteration meta-action AGGREGATES only. The **preserved console transcript** of the
per-episode `OK` blocks (`probe_console.log`, SHA-256
`97bf45d56a3b224ef0ebe5a362bb7415b73e88520d192c891e347cb2412f31c4`, re-verified read-only
before citation) is the only artifact recording a SELECTED ACTION LABEL, and it does so for
the fuel-damage wake specifically.

**Defect A — KC-135R Stratotanker #76** (ego `0a14f756-13f2-4c78-8aa8-446da245aee5`, the id
the playback binds to that name). *Console transcript, this exact attempt:* `[eval
stage=post_update ep=3 damaged seed=1000003] OK` records `fired=True tick=269
progress=0.300`, `fuel_before=203494.4 fuel_after=70026.7 factor=0.3441` and `fd_wake=True
fd_meta=SELF_PRESERVATION_ABORT rtb_command=True` — **the action label is an attribution
from that record, not something visible in playback.** *Playback, independently:* fuel falls
from `203578.18` at the T+260 sample to `70017.43` at T+270 (sampled values, hence not
identical to the console's exact event-time pair), `rtb` flips `False → True` on that frame,
the route is replaced by a route to base at once, the aircraft lands ≈ T+540, and it never
resumes the abandoned assignment queue. The two sources agree, and the physical signature is
**consistent with the merged EGO-GLOBAL abort semantics**.

**Defect B — B-2 Spirit #698** (playback evidence alone; no action label claimed). It fires
**one salvo per target** — Floridistan AFB #1794 ≈ T+2320 (AIM-120 `4 → 2`) and Hidden
Airbase #003 ≈ T+5140 (AIM-120 `2 → 0`) — each BLADE two-argument attack launching two
physical AIM-120 missiles, with no redundant second salvo and no flat-timeout re-fire loop.
Salvo quantity, lethality and ammunition management are unchanged.

**Defect C — the same B-2** (playback, corroborated by the transcript's terminal line). It
enters RTB ≈ T+5240, the episode keeps ticking while it physically flies home, it lands
≈ T+7705, and only then does the episode finish; the console block for the same attempt
independently reports `ended=done ticks=7705 dead=0`. That is physical completion, not
command issuance.

**Deferred research hypothesis — NOT a defect and NOT a proven action attribution.** Hidden
Airbase #001 lies close to Hidden Airbase #003; the sampled playback puts the B-2 at
**50.07 km** from Hidden #001 at T+5230 while not yet in RTB, against a 50 km detection
threshold, with RTB visible on the next sampled frame at T+5240. Playback is sampled every
ten ticks and **nothing preserved records an ORGANIC wake's selected meta-action** — the
jsonl records carry aggregates only, and the console transcript labels the fuel-damage wake
alone, which is a different ego in a different episode phase — so it is **plausible but not
proven** that the B-2 crossed the threshold between samples and selected
`SELF_PRESERVATION_ABORT`. Treat possible over-conservatism as a future research hypothesis
about policy calibration, relevant to a later variable-FD-severity experiment. **It opens no
defect, changes no reward, retunes no policy, and neither invalidates nor blocks this probe
or the long baseline.**

## 4. Next task — the short-probe sequence is COMPLETE; the LONG BASELINE is next

Start with fresh exact-SHA initialization against the current `main`. **This documentation
task neither authorizes nor runs anything; it records state only, and it does NOT authorize
CC to run training.**

**Task 0 — Defect A: DONE.** Ego-global `SELF_PRESERVATION_ABORT` is merged and locked
(approved `d56fda6`, integrated `f094e0b`, PR #17 — §1, §3d, `CLAUDE.md` §5 and §7). It is
listed here only so the sequence is unambiguous; nothing about it is outstanding.

**Task 1 — Defect B: DONE.** The evidence-derived attack-confirmation wait is merged and
locked (approved `39a16f2`, integrated `60a82d1`, PR #19 — §1, §3d, `CLAUDE.md` §5, §6
and §7). The wait is DERIVED per salvo from the acting ego's live auto-selected weapon and
the current engagement distance, with the configured `kill_confirm_ticks` kept as its FLOOR
and FALLBACK; lethality and the FROZEN vendored engine are unchanged. Nothing about it is
outstanding.

**Task 2 — Defect C: DONE.** Physical RTB completion is merged and locked (approved
`ea62e4e`, integrated `0de9f21`, PR #21 — §1, §3d, `CLAUDE.md` §4, §5, §6 and §7).
`is_done(observation)` requires the LIVE post-step observation; assignment completion still
comes from executor semantic state while the physical half comes from `_physical_state`,
`_note_dead` reconciles a death on the ride home into `executor.dead` before the verdict,
`rtb_issued` remains only the single-issue toggle guard, a returning ego leaves Phase 1
while peers continue, and the reward formula and FROZEN BLADE are unchanged. Nothing about
it is outstanding. **The three-defect CODE correction is COMPLETE.**

**Task 3 — RERUN the bounded short probe on the corrected cell: DONE / REVIEWED / VALID.**
Executed ONCE as `training_output_20260816_162130` from a clean checkout at exact code SHA
`900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, through the merged repository preset with no
typed override, and independently reviewed as **VALID MEASUREMENT / CORRECTED SHORT-PROBE
PASS**. §3e records it in full and `CLAUDE.md` §7 owns the authoritative measurement record
and evidence hashes. **All three defect corrections are now operationally witnessed in real
playback**, not only in proof tests. The validity gate below is what it was judged against,
and it PASSED. Nothing about this task is outstanding.

**Task 4 — the LONG BASELINE. NEXT, and separately authorized.** It becomes the next
research task **once this documentation record is merged into `main` and its branch
`task/corrected-short-probe-doc-lock` is retired**. It begins with **fresh live-`main`
resolution** and derives its **exact execution contract from the newly merged documents** —
`CLAUDE.md` §5 and §8 plus this handoff at that resolved SHA. **Its shape, duration, seed
schedule, iteration count, evaluation cadence and CLI invocation are deliberately NOT
stated here**: fixing them in a documentation task would invent a research contract that no
recon has produced. That derivation, and the authorization to run anything, belong to the
long-baseline session itself. **No long-baseline result may be pre-claimed** — not reward
improvement, not productive-update yield, not survival, not anything else.

**What the corrected rerun already fixed about the probe's shape** is that the shape is no
longer a question for the SHORT probe: it is the merged repository preset
`configs/graph_train/final_cell_probe.json` (PR #14, §3c), run through `--config` —
2 scheduled training iterations, 4 scheduled attempts each, base seed 0, four fixed
held-out seeds from 1_000_000 with matched forced-clean / forced-damaged members in a
`pre_update` and a `post_update` round, the final 3-agent / 3-known / 3-hidden cell,
FD-BASELINE-v1, and visual artifacts enabled. **That is the SHORT-probe shape and it must
not be assumed to be the long baseline's.**

**Execution discipline (unchanged, and what the corrected rerun satisfied).** Run from a
clean checkout at an exact resolved `main` SHA, with COMPLETE Git provenance (`train`
refuses otherwise — `CLAUDE.md` §8), on the merged cell as the configuration configures it:
no ad-hoc knob changes, no second difficulty factor, no retry of a failed seed, no band
shift. An explicitly typed CLI flag overrides a preset, so anything typed beyond `--config`
is a deliberate deviation and must be reported as one; `run_config.json:/config_source`
records exactly which fields a preset supplied and which a flag overrode.

**Report, all with explicit denominators:**

- complete provenance and the exact resolved configuration;
- scheduled **clean vs damaged** attempt populations and their successes/failures;
- **matched-pair yield** and the paired reward delta over pairs whose BOTH members
  completed, next to its pair denominator;
- **failures by pipeline stage** (`generation` / `setup` / `run` / `reward`), including any
  `setup` planning refusal and any `run`-stage live-window refusal;
- per-episode **event fired / wake / real RTB command / death** outcomes;
- reward headroom, and whether the PPO updates were productive;
- **artifact completeness** — how many selected attempts produced a `complete` bundle and
  how many an `incomplete` one, reported ALONGSIDE the scientific denominators and never
  in place of one.

**What makes a run VALID — as opposed to favourable.** This is the gate the corrected
short-probe rerun was judged by and passed, and it carries forward. A run counts as a valid
measurement when ALL of:

- Git provenance is COMPLETE and the checkout was clean;
- `run_summary.json:accounting_reconciled` is true — the ledger and the record counts
  agree;
- no INFRASTRUCTURE failure occurred (a `_VisualArtifactError` or any crash outside the
  `generation` / `setup` / `run` / `reward` episode taxonomy aborts the run and is not a
  scientific result);
- **at least one COMPLETED matched pair exists in BOTH the `pre_update` and the
  `post_update` round** — a pair counts only when both its members completed.

If either round yields no completed matched pair, or accounting / data integrity fails, the
run is **INCONCLUSIVE**.

**A negative result is still a valid result.** No reward improvement, or zero productive
PPO updates (`updates_completed = 0`), is a valid NEGATIVE SCIENTIFIC OBSERVATION about the
cell — not a technical failure, and not grounds to re-run, re-tune or re-seed.
Productive-update yield is one of the quantities being measured.

**Interpretation rules survive unchanged:** a held-out mean is never read without its
denominator; an all-failed batch reports `null`, never `0.0`; an empty successful-pair
population is `null` too; and the held-out per-condition means are each over their own
successful subset, so the within-seed claim is the matched-pair delta alone (`CLAUDE.md`
§5). **Do not pre-claim any result**, and do not reuse §2's or §3e's numbers as the long
baseline's expectation — §2 measured a different, easier cell, and §3e is a bounded
24-attempt probe.

## 5. Closed decisions

- Offline construction only: solve → place → patch → reload.
- Route prediction is required and supports `num_agents < n_known`.
- One sensing/arrival/attack/kill-confirmation radius: `DETECTION_KM = 50`.
- `round_trip_cost` and the current p=1 `graph_reward` FORMULA remain frozen.
- B1 reference cell: 3 agents, 3 known, 3 hidden; strict 200 km launch-point distance,
  100 km known-target separation and 0.5 stretch ratio. A reference cell, not a law.
- B2: one placement per non-empty ego route, explicit `random.Random`, id-free geometric
  fingerprints and one-way placement-layer imports.
- B3: explicit construction-path selection; env-2 is the runtime source of truth;
  ordered agent IDs survive reload; exact cardinality; airbase-only cell.
- B4: complete provenance precondition, `skip_and_account_v1`, fixed held-out band,
  explicit denominators, six run artifacts, true pre-update evaluation and disjoint
  all-failed / zero-wake / productive states.
- PR #7: per-episode `OK` blocks, direct unique-target-id counts, accounted structural
  roster failures and disjoint per-round eval artifact namespaces.
- PR #8 (FD-BASELINE-v1): fuel damage is the ONE selected difficulty factor; deterministic
  private RNG domain; matched forced-clean / forced-damaged evaluation pairs on the same
  held-out seed; the strict window validated twice (planned, then live before mutation);
  RTB measured from real emitted command history, never from `GraphPlanExecutor.rtb_issued`;
  explicit `aircraft_penalty_coeff = 2.25` with the reward formula unchanged.
- PR #10 (FINAL-CELL-VISUAL-ARTIFACTS): artifact capture is opt-in and OFF by default; it
  selects every scheduled attempt rather than a per-seed subset; the executed t=0 snapshot
  comes from env-2 before the controller and the run; recording is armed only through the
  existing setup/tick-loop contract; artifact failures are infrastructure and stay outside
  the scientific ledger.
- The legacy split surface remains retained, not retired.

## 6. Out of scope for the next task

- **any training run driven from THIS documentation task** — it authorizes none. The long
  baseline is the next research task, and it starts only after this record is merged, its
  branch is retired, live `main` is freshly resolved and its own contract is derived there
  (§4 Task 4);
- **inventing the long baseline's shape, duration, seed schedule or CLI invocation in a
  documentation task** — those are recon outputs, not editorial choices;
- selecting or enabling a SECOND difficulty factor (`probability < 1`, hostile fire/SAMs,
  dense reward) — each is its own research change;
- **reopening Defects A, B or C, or acting on the §3e over-safety hypothesis** — the three
  defects are closed, approved, merged and now operationally witnessed, and the hypothesis
  is a future research question about policy calibration, not a defect, a reward change or
  a retune;
- reworking the merged FD-BASELINE-v1 mechanism, the merged visual-artifact surface or
  the merged probe harness (preset, `--config` precedence, `config_source`, run layout,
  the three figures), or their reviewed research decisions — a run RUNS what is merged.
  The §3d validity correction was the ONE authorized exception, it was scoped to those
  three defects alone (abort semantics, the confirmation wait, RTB completion), and it is
  now CLOSED; it never was a licence to retune the cell, the reward, the seeds or the
  harness;
- extending artifact capture (per-seed filters, new artifact kinds, artifact-derived
  metrics) — a run uses what is merged;
- **modifying, repackaging, moving or deleting any preserved scientific artifact**, in
  particular `training_output_20260815_173029` and `training_output_20260816_162130`;
- checkpoint loading/resume;
- centralized critic / CTDE;
- low-known-cell solver timeout unless the chosen cell needs `known ≤ 2`;
- ETA/peer-dropout, reachability-model and legacy-split retirement work;
- further repository/documentation hygiene — the README rewrite and the BLADE fork
  documentation audit were closed by PR #11's follow-up hygiene task, and neither a probe
  nor the long baseline needs or may reopen them.

## 7. Documentation duties

| Trigger | Duty |
|---|---|
| B1–B4 preparation lands — **DONE** | Contracts and locks recorded in `CLAUDE.md` |
| First real post-B3 probe completes — **DONE** | Exact code SHA, denominators, yield, failure stage, transitions and pre/post held-out measurements recorded |
| PR #7 observability follow-up lands — **DONE** | Unique-target semantics, per-episode output, eval artifact preservation and fix-chain lock recorded |
| Selected baseline-difficulty factors land — **DONE for FD-BASELINE-v1** | Contract in `CLAUDE.md` §5, tick placement in §4, routing in §6, lock + fix chain in §7, selection closure and deferrals in §8 — recorded without pre-claiming any result |
| Repository code-hygiene + documentation alignment lands — **DONE for PR #11 and its follow-up** | Helper ownership, retired-executor removal and the lock recorded in `CLAUDE.md` §5–§7; README replaced, BLADE fork documentation audited, obsolete scenarios and dead utility symbols removed |
| Visual-artifact support lands — **DONE for PR #10** | Contract in `CLAUDE.md` §5, routing in §6, lock in §7, and the §8 note that the bounded probe MAY enable it — recorded without pre-claiming any result |
| Probe harness lands — **DONE for PR #14** | Preset, `--config` precedence, three-kind `config_source` and the `plots/` figures recorded as contracts in `CLAUDE.md` §5, routed in §6, locked in §7 with the two-round fix chain; the retired four-panel `training_plot.png` removed from the contracts — recorded without pre-claiming any result |
| FIRST final-cell short probe completes — **DONE for `training_output_20260815_173029`** | Run identity, exact code SHA, accounting and denominators, and the three research-validity defects it exposed recorded in §3d — as findings only, with no scientific claim about the cell and no long-baseline authorization |
| §3d validity correction lands — **DONE: ALL THREE DEFECTS RECORDED and their locks integrated (PR #18, PR #20, PR #22)** | Record each corrected contract in `CLAUDE.md` §5–§7 with its own lock and fix chain, ONE DEFECT AT A TIME and only once that defect is merged, never in advance — a DOCUMENTATION-RECORDING rule, not a constraint on how the fixes are broken into tasks (`CLAUDE.md` §8 owns the sequential-defect policy, which is historical workflow context). **Defect A — DONE:** the ego-global `SELF_PRESERVATION_ABORT` selection/effect contracts are in `CLAUDE.md` §5 (Stages 4 and 5), the lock, fix chain and evidence in §7, and the defect state in §8. **Defect B — DONE:** the derived attack-confirmation wait contract is in `CLAUDE.md` §5 (Execution, Stage 1), routed in §6, with the lock, append-only fix chain and evidence in §7 and the defect state in §8. **Defect C — DONE:** the physical-completion contract is in `CLAUDE.md` §4 (the terminal loop) and §5 (Execution, Stage 1, and the tick loop), routed in §6, with the lock, append-only fix chain and evidence in §7 and the defect state in §8. The three-defect correction is COMPLETE: every lock is integrated into `main`, and the corrected-cell rerun in the row below has since witnessed all three operationally |
| Probe RERUN completes on the corrected cell — **DONE for `training_output_20260816_162130`** | Exact config, provenance, denominators, clean/damaged and matched-pair populations, failures by stage, event/wake/RTB/death outcomes, reward headroom, update evidence, artifact completeness and playback witnesses recorded in §3e, with the authoritative measurement record and evidence hashes in `CLAUDE.md` §7 and the gate in §8. Verdict VALID — the long-baseline validity gate is PASSED |
| LONG BASELINE completes — **NEXT MEASUREMENT TRIGGER; NOT RUN** | Record its exact derived contract and resolved configuration, complete provenance, every denominator, clean/damaged and matched-pair populations, failures by stage, event/wake/RTB/death outcomes, reward headroom, productive-update yield and artifact completeness — judged by the §4 validity gate, never by whether reward improved. No result may be pre-claimed |

## 8. Next action

Implementation for the final Phase-A baseline cell is COMPLETE and locked, its inspection
surface is merged, repository hygiene is CLOSED (PR #11 code, PR #12 documentation), the
**probe harness is CLOSED** (PR #14), the bounded short probe was **EXECUTED a first time**
(§3d), **ALL THREE of the defects it exposed are CLOSED / APPROVED / MERGED** — **Defect A,
ego-global `SELF_PRESERVATION_ABORT`** (approved `d56fda6`, integrated `f094e0b`, PR #17,
identical tree `70e5af2…`), **Defect B, the attack-confirmation wait derived from the salvo
about to fly** (approved `39a16f2`, integrated `60a82d1`, PR #19, identical tree
`ee86f07…`) and **Defect C, physical RTB completion** (approved `ea62e4e`, integrated
`0de9f21`, PR #21, identical tree `6d05cc5…`) — and the **CORRECTED-CELL SHORT-PROBE RERUN
has been EXECUTED, INDEPENDENTLY REVIEWED and judged VALID** (`training_output_20260816_162130`
at exact code SHA `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`, §3e, `CLAUDE.md` §7).
**The three-defect CODE correction is COMPLETE and the long-baseline validity gate is
PASSED.** Current `main` is `900ff0b24898eccfa2e35d2db05c4e0229c64ce3`
(`2026-08-16T15:26:55+03:00`).

**No active CODE candidate exists**, and the state below is written to be valid on BOTH
sides of this record's own integration. **While this record is published and under review**
the sole active candidate of any kind is the documentation/lock task itself — branch
`task/corrected-short-probe-doc-lock` and its draft PR — and no other candidate should be
claimed. **Once this record is integrated into `main` and that branch is retired, no active
candidate remains** and ownership is RELEASED for the LONG BASELINE. The integrating
merge's SHA is deliberately NOT named here: it does not exist while this is written, and
inventing it would be a false provenance claim. **GitHub remains authoritative for live
branch and PR state — resolve it there, never from this document.**

**The next task is the LONG BASELINE** (§4 Task 4). It starts only after this record is
merged and its branch retired, it begins with **fresh live-`main` resolution**, and it
**derives its exact execution contract — shape, duration, seed schedule, evaluation cadence
and CLI invocation — from the newly merged documents at that resolved SHA**. None of those
values is stated anywhere in this documentation task, deliberately: they are recon outputs,
not editorial choices. It is judged by the §4 validity gate rather than by whether reward
improved, and **no result may be pre-claimed** — not reward improvement, productive-update
yield, survival or anything else.

**What evidence exists, stated precisely.** The corrected cell now has BOTH implementation
evidence (real-BLADE and BONMIN-backed proof tests behind three locks) AND real
**SHORT-PROBE performance evidence** with every denominator explicit (§3e). That evidence
is bounded — 24 scheduled attempts, 22 successful, two evaluation rounds — and it is **not
a baseline and not an estimate of converged policy performance**. **NO LONG BASELINE HAS
BEEN RUN.** The §3e over-safety observation is a deferred **research hypothesis** about
policy calibration, not a defect and not a semantic change; it opens nothing and blocks
nothing.

Resolve live branch and PR state from GitHub; this document does not track it.

**This document authorizes neither an implementation nor a training run.**
