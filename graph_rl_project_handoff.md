# Multi-Agent Graph RL — Harness Closure / Final-Cell Probe Handoff

**Supersedes all earlier handoffs.**

Written 2026-08-11; updated 2026-08-14 for the final-cell PROBE HARNESS closure, and
2026-08-15 to record the FIRST EXECUTED bounded short probe and the three
research-validity defects it exposed (§3d).
B1–B4, the first real post-B3 instrumented probe, the B4 observability follow-up (PR #7),
**FD-BASELINE-v1** (PR #8), **FINAL-CELL-VISUAL-ARTIFACTS** (PR #10), the repository
code-hygiene cleanup (PR #11), the documentation hygiene (PR #12) and now the
**FINAL-CELL PROBE HARNESS** (PR #14) are all CLOSED. None of them changes research
behaviour: the solver, BLADE, reward, fuel-damage semantics, PPO math, seed schedules,
scenario construction and matched-pair evaluation are exactly as their own locks left them.

Baseline **difficulty selection is finished**, the **inspection surface is in place**, and
the **operator harness a probe is driven from — preset, run layout and figures — is
merged**. The bounded short probe HAS NOW BEEN RUN ONCE
(`training_output_20260815_173029`, from clean `main` at
`238062d7d284334432d9c39d7543fb0bbf39ea7c`). It passed every mechanical harness and
accounting check **and** exposed three research-validity defects (§3d). The next task is
therefore no longer the probe itself but a **Grade-A research-validity correction** of
those defects, after which the SAME bounded probe shape is rerun once and reviewed. **A
long baseline remains UNAUTHORIZED.**

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
  improvement is therefore NOT final scientific evidence about the fuel-damage cell**, and
  **a long baseline stays BLOCKED**. §3d records the run state, the three defects and the
  decided direction; the correction is a FUTURE Grade-A task and is **not implemented at
  the SHA above**.
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

**Defect A — `SELF_PRESERVATION_ABORT` is node-indexed, not an ego-global abort.**

- At this SHA `graph_effect.apply_meta_action` removes only the assignment(s) whose
  `task_idx == node_v`, so SPA aborts ONE task rather than the ego's mission.
- Probe playback showed a fuel-damaged KC-135 selecting SPA while its existing BLADE route
  continued and further assignments remained.
- **User decision for the NEXT CODE TASK:** the desired behaviour is an **ego-global
  mission abort** — selecting `SELF_PRESERVATION_ABORT` must clear ALL of that ego's
  remaining assignments, so the executor reaches its empty-plan RTB path. **This is a
  decision about future work; it is NOT implemented at the SHA above.**
- The existing `k × 3` action-head structure is **not** authorized for redesign by this
  record. The next code task determines the minimal contract-safe implementation and tests
  it end to end.
- Execution-seam fact that narrows the diagnosis: `graph_tick_loop._wake_decision` already
  resyncs the edited ego plan before Phase 2, and an actually EMPTY plan should make
  `GraphPlanExecutor` emit `aircraft_return_to_base`, whose BLADE handling replaces the
  stale route with the home-base route. The observed stale route is therefore currently
  explained by SPA not emptying the plan — **not** by evidence of a missing resync call.

**Defect B — premature re-fire exhausts weapons.**

- In the `post_update` damaged eval seed `1000003`, B-2 Spirit #698 engaged its
  route-relative hidden targets successfully but reached the final known target
  `Floridistan AFB #4067` with **zero onboard weapons**, and then remained over it until
  fuel exhaustion.
- Artifact reconstruction of the sequence: at approximately t=5140 the final 2 AIM-120
  launched at Hidden Airbase #003; at approximately t=5240 2 AIM-9 launched at Hidden
  Airbase #001 from about 47.2 km; at approximately t=5300, before that slower AIM-9 salvo
  resolved, the fixed 60-tick confirmation cooldown expired and a redundant second salvo
  consumed the final 2 AGM-65 — and the AIM-9 salvo killed the target in that same engine
  update, leaving the B-2 with no weapons for the final known target.
- Code anchors: `GraphPlanExecutor.kill_confirm_ticks`,
  `GraphPlanExecutor._command_for_ego`, `Game.handle_aircraft_attack`,
  `weaponEngagement.launch_weapon`.
- **Design direction for the next code task:** do NOT merely raise the constant blindly.
  Derive a conservative confirmation wait from the ACTUAL auto-selected live weapon and the
  current engagement distance, while preserving current lethality and frozen BLADE
  behaviour. Unrelated future probabilistic-miss / weapons-exhaustion redesign stays OUT of
  scope unless the evidence requires it.

**Defect C — RTB ISSUANCE is not physical RTB COMPLETION.**

- `GraphPlanExecutor.is_done()` currently treats the `rtb_issued` lifecycle latch as
  RTB-resolved, and `run_episode` stops when `executor.is_done()` becomes true — so an
  episode may end immediately after an RTB command while the aircraft is still airborne.
- In the `post_update` damaged eval seed `1000000`, the damaged KC-135 completed its work
  and an RTB command was eventually issued while it no longer had enough fuel to physically
  reach home; the episode nevertheless ended before the resulting fuel exhaustion / death
  could occur, recorded `dead=0`, and contributed reward 0.
- The next code task must separate **"RTB command issued"** from **"RTB physically
  resolved"**: a non-dead ego must actually be back in an airbase / landed before episode
  completion. The single-issue RTB toggle protection is preserved.

**Scientific interpretation.**

- The first probe is USEFUL and did its job: it successfully exposed these defects.
- Its post-update reward improvement **must NOT** be treated as final scientific evidence
  for the fuel-damage cell, because episode termination (Defect C) and abort semantics
  (Defect A) can distort the measured airframe penalty — precisely the quantity
  FD-BASELINE-v1 exists to make real.
- **The long baseline remains BLOCKED / UNAUTHORIZED.**
- After the fixes are reviewed and merged, rerun the SAME bounded short-probe shape ONCE
  from the new clean exact `main` and perform a fresh artifact review before any long run.

## 4. Next tasks — the Grade-A validity correction, then a RERUN of the same bounded probe

Start with fresh exact-SHA initialization against the current `main`. **This documentation
task neither authorizes nor runs anything; it records state only.**

**Task 1 — the Grade-A research-validity correction (NEXT; no active code candidate
exists).** Correct the three §3d defects: ego-global `SELF_PRESERVATION_ABORT`, an
evidence-derived confirmation wait, and RTB COMPLETION rather than RTB issuance as the
episode-completion condition. Expected implementation mode is **BUILD**, or SURGICAL only
if recon proves the contract changes truly remain narrowly local. It touches §5-locked
layers, so it is **Grade A** with declared proof obligations, and it must be dispatched by
the next GPT orchestrator after exact-SHA initialization and task-focused recon. Nothing
here pre-decides its design beyond the directions §3d records.

**Task 2 — RERUN the same bounded short probe, ONCE, after Task 1 is reviewed and
merged**, from the new clean exact `main`, followed by a fresh artifact review. The shape,
execution discipline, reporting duties and validity gate below are UNCHANGED and are what
gets rerun. The first run's numbers are not its expectation, exactly as §2's are not.

**The probe's shape is no longer "to be confirmed" — it is the merged repository preset**
`configs/graph_train/final_cell_probe.json` (PR #14, §3c), run through `--config`:

- 2 scheduled training iterations;
- 4 scheduled training attempts per iteration;
- base seed 0;
- four fixed held-out seeds from 1_000_000, with matched forced-clean and forced-damaged
  members run BEFORE training (`pre_update`) and AFTER training (`post_update`) on the
  same seeds;
- the final 3-agent / 3-known / 3-hidden cell and FD-BASELINE-v1, unchanged;
- visual artifacts ENABLED — the flag selects every scheduled attempt, so a successful one
  yields a `complete` bundle and a failed one a clearly marked `incomplete` bundle;
- live console output (the per-episode `OK` blocks and the per-iteration / per-round
  lines).

The OPERATIONAL procedure — the local PyCharm / CLI invocation and the environment
(`nlp_env`) — is no longer an open question either: the first run (§3d) exercised it end to
end. What the RERUN still requires is confirmation of a clean checkout at the exact
resolved post-correction `main` SHA. The configuration itself is a reviewed file in the
repository, not a decision to be re-made.

**Execution discipline.** Run it ONCE, from a clean checkout at an exact resolved `main`
SHA, with COMPLETE Git provenance (`train` refuses otherwise — `CLAUDE.md` §8), on the
merged cell as the preset configures it: no ad-hoc knob changes, no second difficulty
factor, no retry of a failed seed, no band shift. An explicitly typed CLI flag overrides
the preset, so anything typed beyond `--config` is a deliberate deviation and must be
reported as one; `run_config.json:/config_source` records exactly which fields the preset
supplied and which a flag overrode.

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

**What makes the probe VALID — as opposed to favourable.** The probe is an OPERATIONAL
and SCIENTIFIC VALIDATION of the merged harness and cell. It is *not* a requirement to
demonstrate reward improvement. A run counts as a valid measurement when ALL of:

- Git provenance is COMPLETE and the checkout was clean;
- `run_summary.json:accounting_reconciled` is true — the ledger and the record counts
  agree;
- no INFRASTRUCTURE failure occurred (a `_VisualArtifactError` or any crash outside the
  `generation` / `setup` / `run` / `reward` episode taxonomy aborts the run and is not a
  scientific result);
- **at least one COMPLETED matched pair exists in BOTH the `pre_update` and the
  `post_update` round** — a pair counts only when both its members completed.

If either round yields no completed matched pair, or accounting / data integrity fails,
the probe is **INCONCLUSIVE** and the long baseline stays blocked.

**A negative result is still a valid result.** No reward improvement, or zero productive
PPO updates (`updates_completed = 0`), is a valid NEGATIVE SCIENTIFIC OBSERVATION about
the cell — not a technical probe failure, and not grounds to re-run, re-tune or re-seed.
Productive-update yield is one of the quantities being measured.

**A long baseline remains UNAUTHORIZED until the short probe has been executed and
reviewed.** Interpretation rules survive unchanged: a held-out mean is never read without
its denominator; an all-failed batch reports `null`, never `0.0`; an empty successful-pair
population is `null` too; and the held-out per-condition means are each over their own
successful subset, so the within-seed claim is the matched-pair delta alone (`CLAUDE.md`
§5). **Do not pre-claim any probe result**, and do not reuse the §2 numbers as its
expectation.

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

- a long training run before the §3d correction is merged and the probe RERUN has been
  executed and reviewed;
- selecting or enabling a SECOND difficulty factor (`probability < 1`, hostile fire/SAMs,
  dense reward) — each is its own research change;
- reworking the merged FD-BASELINE-v1 mechanism, the merged visual-artifact surface or
  the merged probe harness (preset, `--config` precedence, `config_source`, run layout,
  the three figures), or their reviewed research decisions — the probe RUNS what is merged.
  The §3d validity correction is the ONE authorized exception, and it is scoped to those
  three defects alone (abort semantics, the confirmation wait, RTB completion); it is not a
  licence to retune the cell, the reward, the seeds or the harness;
- extending artifact capture (per-seed filters, new artifact kinds, artifact-derived
  metrics) — the probe uses what is merged;
- checkpoint loading/resume;
- centralized critic / CTDE;
- low-known-cell solver timeout unless the chosen cell needs `known ≤ 2`;
- ETA/peer-dropout, reachability-model and legacy-split retirement work;
- further repository/documentation hygiene — the README rewrite and the BLADE fork
  documentation audit were closed by PR #11's follow-up hygiene task, and the probe
  neither needs nor may reopen them.

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
| §3d validity correction lands | Record the corrected `SELF_PRESERVATION_ABORT`, confirmation-wait and RTB-completion contracts in `CLAUDE.md` §5–§7 with their lock and fix chain — only once merged, never in advance |
| Probe RERUN completes on the corrected cell — **NEXT MEASUREMENT TRIGGER** | Record exact config, provenance, denominators, clean/damaged and matched-pair populations, failures by stage, event/wake/RTB/death outcomes, reward headroom, update evidence and artifact completeness before authorizing a long baseline |

## 8. Next action

Implementation for the final Phase-A baseline cell is COMPLETE and locked, its inspection
surface is merged, repository hygiene is CLOSED (PR #11 code, PR #12 documentation), the
**probe harness is CLOSED** (PR #14), and the bounded short probe has now been **EXECUTED
ONCE** (§3d). **No active code candidate exists**, and **ownership is RELEASED once this
record is integrated into `main`**.

The next task is the **Grade-A research-validity correction** of the three §3d defects —
ego-global `SELF_PRESERVATION_ABORT`, an evidence-derived confirmation wait, and RTB
COMPLETION rather than RTB issuance as the episode-completion condition. Expected
implementation mode is BUILD, or SURGICAL only if recon proves the contract changes truly
remain narrowly local. It must be dispatched by the next GPT orchestrator after fresh
exact-SHA initialization against the current `main` and task-focused recon; none of it is
implemented at `238062d7d284334432d9c39d7543fb0bbf39ea7c`.

Once that correction is reviewed and merged, the SAME bounded short probe (§4) is rerun
ONCE from the new clean exact `main` and reviewed afresh, judged by the §4 validity gate
rather than by whether reward improved. **A long baseline remains UNAUTHORIZED until that
rerun has been executed and reviewed**, and no result may be pre-claimed for it.

Resolve live branch and PR state from GitHub; this document does not track it.

**This document authorizes neither an implementation nor a training run.**
