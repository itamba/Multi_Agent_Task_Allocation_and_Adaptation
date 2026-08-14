# Multi-Agent Graph RL — Harness Closure / Final-Cell Probe Handoff

**Supersedes all earlier handoffs.**

Written 2026-08-11; updated 2026-08-14 for the final-cell PROBE HARNESS closure below.
B1–B4, the first real post-B3 instrumented probe, the B4 observability follow-up (PR #7),
**FD-BASELINE-v1** (PR #8), **FINAL-CELL-VISUAL-ARTIFACTS** (PR #10), the repository
code-hygiene cleanup (PR #11), the documentation hygiene (PR #12) and now the
**FINAL-CELL PROBE HARNESS** (PR #14) are all CLOSED. None of them changes research
behaviour: the solver, BLADE, reward, fuel-damage semantics, PPO math, seed schedules,
scenario construction and matched-pair evaluation are exactly as their own locks left them.

Baseline **difficulty selection is finished**, the **inspection surface is in place**, and
the **operator harness a probe is driven from — preset, run layout and figures — is now
merged**. What has NOT happened is any measurement of the resulting cell. The next task is
a bounded short scientific probe of the merged fuel-damage configuration, before any long
baseline.

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

- **No scientific probe or training run has been performed on the merged final cell.**
  Every test behind every one of these locks is solver-free and drives the pipeline through
  stubbed engine seams. The locks certify implementation; they say nothing about how the
  cell behaves. **The harness lock in particular measured nothing**: it certifies that a
  probe can be configured, run and read, not that the cell learns anything.
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

## 4. Next task — a bounded SHORT SCIENTIFIC PROBE of the merged fuel-damage cell

Start with fresh exact-SHA initialization against the new `main`. **This documentation
task neither authorizes nor runs the probe.**

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

What still has to be closed before the run is the OPERATIONAL procedure only — the exact
local PyCharm / CLI invocation, the environment (`nlp_env`), and confirmation of a clean
checkout at an exact resolved `main` SHA. The configuration itself is now a reviewed file
in the repository, not a decision to be re-made.

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

- a long training run before the final-cell probe passes;
- selecting or enabling a SECOND difficulty factor (`probability < 1`, hostile fire/SAMs,
  dense reward) — each is its own research change;
- reworking the merged FD-BASELINE-v1 mechanism, the merged visual-artifact surface or
  the merged probe harness (preset, `--config` precedence, `config_source`, run layout,
  the three figures), or their reviewed research decisions — the probe RUNS what is
  merged;
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
| Final-cell short probe completes — **NEXT MEASUREMENT TRIGGER** | Record exact config, provenance, denominators, clean/damaged and matched-pair populations, failures by stage, event/wake/RTB/death outcomes, reward headroom, update evidence and artifact completeness before authorizing a long baseline |

## 8. Next action

Implementation for the final Phase-A baseline cell is COMPLETE and locked, its inspection
surface is merged, repository hygiene is CLOSED (PR #11 code, PR #12 documentation), and
the **probe harness is CLOSED** (PR #14). **No active scientific candidate exists**, no
implementation task remains before the probe, and **ownership is RELEASED once this
closure record is integrated into `main`** — the orchestrator and user own the probe's
preparation and run from that point.

The next orchestrator performs fresh exact-SHA initialization against the current `main`,
closes the probe's remaining OPERATIONAL procedure per §4 (the configuration itself is now
the merged preset), and only then runs it once, from a clean checkout of the final
post-documentation `main` SHA. The next task is still the bounded SHORT SCIENTIFIC PROBE
of the merged fuel-damage cell described in §4 — unchanged by the harness work — judged by
the §4 validity gate rather than by whether reward improved, and **a long baseline remains
UNAUTHORIZED until the short probe has been executed and reviewed** (§4).

Resolve live branch and PR state from GitHub; this document does not track it.

**This document authorizes neither an implementation nor a training run.**
