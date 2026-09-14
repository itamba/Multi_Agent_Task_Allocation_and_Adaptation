# Documentation migration record — 2026-09 restructure

> **Status: review record for the documentation restructure on branch
> `docs/project-guidance-restructure`.** It is not a contract and not a current-state document.
> Every source location below refers to base **`ae42cb01677f94868b2873008d87be677e31f0c8`**; the
> pre-restructure files remain readable there with `git show ae42cb0:CLAUDE.md` and
> `git show ae42cb0:graph_rl_project_handoff.md`. The restructure was performed under the
> user-approved documentation packet transferred to CC on 2026-09-14 ("the packet").

## 0. Method and grouping

**Method.** Normative contract text and historical records were moved **verbatim** by an
assembler that copied exact base line ranges into the destinations and recorded coverage: no
base line was copied twice, and every non-blank `CLAUDE.md` line not copied is accounted for in
§1.1 (headings, two rewritten paragraphs and five phase-state bullets consolidated into history).
Current state, procedures and orientation were written by hand and verified against code, test
bodies and preserved artifacts (§2). No source, test, config, preset, evidence or frozen-engine
file changed.

**Grouping** — fewer files where the material is read together, split only where a distinct task
can skip the rest:

| Destination | Why it is its own file |
|---|---|
| `CLAUDE.md` | the only always-read normative file: scope, language, authority, permission boundaries, frozen layers, invariants, reading table, compatibility index |
| `graph_rl_project_handoff.md` | the only always-read current-state file |
| `docs/workflows/cc_review.md` | read by every implementation task; no technical content |
| `docs/workflows/experiments.md` | read only when planning, reviewing or preserving runs |
| `docs/workflows/environments_cleanup.md` | environment use and cleanup share the ref / artifact registry; both are occasional |
| `docs/contracts/runtime.md` | setup, executor, triggers and tick ordering change together |
| `docs/contracts/construction_fuel_damage.md` | hidden placement and fuel damage share the certified-world contracts |
| `docs/contracts/policy_ctde.md` | the acting path and the training-only critic are reviewed against the same no-communication boundary |
| `docs/contracts/reward_solvers.md` | reward, reference and solver objectives are inseparable |
| `docs/contracts/training_benchmarks.md` | trainer, designs, quotas, early stopping and benchmarks; the largest contract, but its blocks cross-reference one another continuously |
| `docs/contracts/artifacts_metrics.md` | what a run writes and how a reviewer reads it; run review needs it without the trainer internals |
| `docs/history/implementation.md` | locks, merged-PR ledger, closure narratives |
| `docs/history/measurements.md` | run registry and every measurement record |
| `docs/history/decisions.md` | decision log, design and ordering records, superseded procedure |

## 1. Source-block map

Actions: **retained** (kept in a normative destination, verbatim unless noted) · **moved** (verbatim
to a normative contract or procedure) · **moved to history** (verbatim to `docs/history/`) ·
**consolidated duplicate** (the same facts already live in the named destination; the duplicate
wording stays readable at the base) · **corrected** (rewritten because it was stale or superseded).

### 1.1 `CLAUDE.md`

| Source block (opening string) | Base lines | Action | Destination |
|---|---|---|---|
| `# CLAUDE.md` / "Repository guidance for Claude Code. This is the **Multi-Agent GRAPH RL** project" | 1–8 | corrected (same scope; full ref SHAs) | `CLAUDE.md` intro |
| "## 1. Communication & workflow (read first)" — "**User speaks Hebrew.**" … "**Output discipline:**" | 12–94 | language, scope and minimal-file rules retained; transport, grade, status-block and fix-chain rules corrected (§3); full text moved to history | `CLAUDE.md` §1, `cc_review.md`, `decisions.md` §7 |
| "**Environment — TWO VALIDATED EXECUTION CONTEXTS, and BOTH are CURRENT.**" … "**What that cluster smoke IS and IS NOT.**" | 95–136 | moved; the three load-bearing rules also retained in brief | `environments_cleanup.md` §1; `CLAUDE.md` §1 |
| "### 🛑 BLADE engine (vendored Panopticon fork) — FROZEN" (incl. the load-bearing `Game.py` edits, the live-list mutation paragraph and "### 🛑 MATCH-AOU solver — FROZEN") | 141–168 | retained | `CLAUDE.md` §2 |
| "### 🛑 The BUILT graph layers are stable & reviewed" | 170–177 | corrected (names every locked layer, points to the contracts) | `CLAUDE.md` §2 |
| "**The Phase-B CTDE training layer is BUILT / REVIEWED / MERGED**" | 179–196 | moved, plus a current-state note | `policy_ctde.md` §4.1 |
| "Everything derives from **NO-COMMUNICATION**" … "Launch point == the BLUE airbase" | 201–221 | retained | `CLAUDE.md` §3 |
| "`setup_episode` has TWO explicit paths" (the pipeline, §4) | 227–422 | moved | `runtime.md` §1 |
| "**Episode-setup (Stage 0)**" through "**NO env-1 `Agent` or `Task` object may enter the returned context**" | 427–520 | moved | `runtime.md` §2 |
| "**GENERALIZED-V1 HIDDEN CARDINALITY — TWO EXPLICIT POLICIES ON ONE CONSTRUCTION SEAM**" | 521–626 | moved | `construction_fuel_damage.md` §1 |
| "**Execution (Stage 1)**" (derived wait, confirmed-kill reconciliation, physical completion) | 627–766 | moved | `runtime.md` §3 |
| "**Trigger (Stage 2)**" | 767–782 | moved | `runtime.md` §4 |
| "**Build (Stage 3)**" | 783–785 | moved | `policy_ctde.md` §1 |
| "**Encode + decide (Stage 4)**" / "**SELECTION CONTRACT (locked by Defect A**" | 786–789 | moved | `policy_ctde.md` §2 |
| "**Effect (Stage 5)**" | 790–792 | moved | `policy_ctde.md` §3 |
| "**Resync (Stage 6)**" | 793–794 | moved | `runtime.md` §5 |
| "**Reward (Stage 7)**" | 795–797 | moved | `reward_solvers.md` §1 |
| "**The two-phase tick (Stages 2–6)**" | 798–800 | moved | `runtime.md` §5 |
| "**Trainer + run auditability (B4)**" through "**Per-round eval scenario preservation (PR #7).**" | 801–864 | moved | `training_benchmarks.md` §1 |
| "- **Visual artifacts — the opt-in inspection surface (PR #10).**" | 865–917 | moved | `artifacts_metrics.md` §1 |
| "**Roster / world-truth integrity (PR #24)**" | 918–977 | moved | `training_benchmarks.md` §2 |
| "**Experiment harness: JSON presets, run layout and the three figures (PR #14)**" — presets, short-probe preset, `config_source` | 978–1023 | moved | `training_benchmarks.md` §4 |
| "- **Figures: `<run_dir>/plots/`, three files, one claim each.**" and "- **The two presentation invariants.**" | 1024–1055 | moved | `artifacts_metrics.md` §2 |
| "**FD-BASELINE-v1 — the LEGACY difficulty factor**" | 1056–1152 | moved | `construction_fuel_damage.md` §2 |
| "**FD-VARIABLE-SEVERITY-v1 — the mild/severe extension of the SAME event**" | 1153–1243 | moved | `construction_fuel_damage.md` §3 |
| "**FD-VARIABLE-SEVERITY-v1 measurement surface — matched TRIADS and the durable outcome**" | 1244–1305 | moved | `artifacts_metrics.md` §3 |
| "**GENERALIZED-V1 CERTIFIED FD ELIGIBILITY + POST-FD COMPLETION-BOUNDARY ADAPTATION**" (incl. the live certificate check and post-FD boundaries) | 1306–1605 | moved | `construction_fuel_damage.md` §4 |
| "**GENERALIZED-V1 EVENT-CONDITIONED MATCH-AOU CONTINUATION REFERENCE + REWARD CHECKPOINT**" | 1606–1937 | moved | `reward_solvers.md` §2 |
| "**GENERALIZED-V1 EPISODE-DESIGN SELECTOR, TRAINING CARDINALITY SAMPLER, FROZEN STRATIFIED**" through "**THE CONSTRUCTION PROVENANCE IS HONEST ABOUT A DYNAMIC CELL.**" | 1938–2271 | moved | `training_benchmarks.md` §5 |
| "**PERSISTENCE — THE EXISTING CANONICAL ARTIFACTS NOW CARRY THE TASK-1/2/3 STRUCTURES.**" through "**REQUESTED-VS-REALIZED IS REPORTED FOR INSPECTION**" | 2272–2393 | moved | `artifacts_metrics.md` §4 |
| "**THE DIAGNOSTIC ROLLOUT HAS SELECTOR PARITY AND STAYS DIAGNOSTIC.**" through the Task-4 "**WHAT IS EXPLICITLY NOT IN THIS TASK.**" | 2394–2453 | moved | `training_benchmarks.md` §5 |
| "**GENERALIZED-V1 TASK 5 — SUMMARY-POPULATION CORRECTION**" | 2454–2770 | moved | `training_benchmarks.md` §6 |
| "**GENERALIZED-V1 OPT-IN TRAINING-REWARD EARLY STOPPING**" | 2771–2990 | moved | `training_benchmarks.md` §7 |
| "**GENERALIZED-V1 DURABLE PER-WAKE FD POLICY DIAGNOSTICS (MEASUREMENT HARDENING)**" | 2991–3189 | moved | `artifacts_metrics.md` §5 |
| "**PHASE-B CTDE — the TRAINING-ONLY centralized critic**" | 3190–3370 | moved | `policy_ctde.md` §4 |
| "**SCHEDULED CELL vs EXECUTED CELL — a measurement-integrity abort**" | 3371–3392 | moved | `training_benchmarks.md` §3 |
| "**THE MATCH-AOU ALLOCATION BACKEND — AN EXPLICIT SELECTOR OVER TWO NON-INTERCHANGEABLE**" | 3393–3552 | moved | `reward_solvers.md` §3 |
| "**GENERALIZED-V2 — THE TWO-STAGE ROUTE-RELATIVE POPULATION**" | 3553–3840 | moved | `training_benchmarks.md` §8 |
| "**GENERALIZED-V2 BENCHMARK — THE FROZEN TEN-CELL BENCHMARK AND EVALUATION CONSTRUCT**" | 3841–4024 | moved | `training_benchmarks.md` §9 |
| "## 6. File map" — the 72 "I want to…" rows | 4031–4102 | moved, split by topic (runtime 17, construction 7, policy 9, reward 8, training 23, artifacts 8) | each contract's "Code routing" section |
| "> Shared domain infra (`scenario_generator`, `scenario_factory`…" | 4104 | moved | `runtime.md` §6 |
| "> **Hash convention:**" | 4110 | moved to history; live procedure corrected (§3 row 7) | `implementation.md` §1; `cc_review.md` §5 |
| §7 implementation and lock entries — "`777bd85` — executor per-ego private task lists" … "`786e821` — **GENERALIZED-V2 BENCHMARK**" (excluding the seven measurement entries) | 4112–4425, 4453–4832, 5092–5145, 5261–5321, 5527–6046, 6135–6193, 6341–6619 | moved to history | `implementation.md` §2 |
| §7 measurement entries — "`a3f0838` — **First real post-B3 instrumented probe**", "**CORRECTED-CELL BOUNDED SHORT PROBE**", "**FIRST LONG BASELINE**", "**PHASE-A LONG BASELINE (RERUN)**", "**FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE**", "**GENERALIZED-V1 TASK 5A / TASK 5B**", "**GENERALIZED-V1 ACTOR-ONLY R1 LONG RUN**" | 4426–4452, 4833–5091, 5146–5260, 5322–5526, 6047–6134, 6194–6340 | moved to history | `measurements.md` §2 |
| "- **PHASE-STATE CORRECTION (LATEST) — THE GENERALIZED-V2 POPULATION (PR #57) AND ITS FROZEN" | 6623–6682 | consolidated duplicate and corrected: the construct is in `training_benchmarks.md` §9; "no V2 manifest / run / result exists" and "the next action is the V2 preflight" are superseded by the evidence of PR #61 / #62; unchanged facts and the unauthorized list are current in the handoff | handoff §1, §4–§7; `decisions.md` §1 |
| "- **PHASE-STATE CORRECTION (SUPERSEDED AS CURRENT STATE BY THE BULLET ABOVE, AND PRESERVED" (V2 population) | 6684–6780 | consolidated duplicate | `decisions.md` §1 (2026-09-12 rows); `training_benchmarks.md` §8; `measurements.md` §6.2; handoff §4 |
| "- **PHASE-STATE CORRECTION (SUPERSEDED AS CURRENT STATE BY THE BULLET ABOVE, AND" (P1 / certified FD) | 6782–6846 | consolidated duplicate | `decisions.md` §1 (2026-09-06 rows); `measurements.md` §6.1; `reward_solvers.md` §3; `construction_fuel_damage.md` §4 |
| "- **PHASE-STATE CORRECTION — THE ACTOR-ONLY R1 LONG RUN IS NO LONGER PENDING" | 6848–6901 | consolidated duplicate | `measurements.md` §2 (R1 record); `decisions.md` §1 (2026-09-05 rows) |
| "- **GENERALIZED-V1 — THE ACTIVE PHASE. TASKS 1, 2, 3 AND 4 ARE ALL IMPLEMENTED" | 6903–7121 | consolidated duplicate; its active rules (one shared manifest per comparison, human inspection of requested-vs-realized, validity before performance) are carried in `experiments.md` §2–§3 | contracts; `decisions.md` §1; `experiments.md` |
| "- **PHASE A IS CLOSED. A SCIENTIFICALLY VALID LONG-BASELINE MEASUREMENT**" | 7123–7288 | moved to history | `measurements.md` §3 |
| "- **The three research-validity defects the first short probe exposed**" and "- **A FOURTH, SEPARATE defect — the ROSTER read an ALLOCATION**" | 7289–7366 | moved to history | `implementation.md` §4 |
| "- **Complete Git provenance is REQUIRED for a real training run (`1b48145`).**" | 7367–7373 | moved | `training_benchmarks.md` §11 |
| "- **RESEARCH ORDERING — the 2026-08-22 PARALLEL arrangement, NOW FULLY TRAVERSED.**" | 7374–7435 | moved to history | `decisions.md` §4 |
| "- **THE AUTHORIZED MEASUREMENT CONTRACT — the bounded actor-only FD-VARIABLE-SEVERITY-v1**" | 7436–7473 | moved to history | `measurements.md` §3 |
| "- **Centralized critic / value head (CTDE) — PHASE B.**" (comparison specification) | 7474–7547 | moved to history; binding rules carried as active procedure | `decisions.md` §4; `experiments.md` §3.3 |
| "- **Baseline difficulty selection — CLOSED for the current cell by FD-BASELINE-v1**" | 7548–7576 | moved to history; the deferrals carried as active prohibitions | `decisions.md` §4; `experiments.md` §5; handoff §6 |
| "- **Solver 2:1 stacking**" | 7577 | moved | `reward_solvers.md` §5 |
| "- **Peer-dropout as a deterministic pre-build trigger**" | 7578 | moved | `runtime.md` §7 |
| "- **`reachable_by_ego` marginal-detour model:**" | 7579 | moved | `policy_ctde.md` §6 |
| "- **`assigned_to_peer` as a task-feature column**" | 7580 | moved | `runtime.md` §7 |
| "- **`setup_episode` does not guard `split_meta[\"outcome\"]`**" | 7581–7593 | moved | `runtime.md` §7 |
| "- **Exact-cardinality construction failures — RESOLVED as `skip_and_account_v1`**" | 7594–7616 | moved | `construction_fuel_damage.md` §6 |
| "- **Added enemy airbases are not seed-stable by id.**" | 7617–7624 | moved | `training_benchmarks.md` §11 |
| "- **bonmin needs a solve timeout at `known ≤ 2`.**" | 7625–7629 | moved | `reward_solvers.md` §5 |
| "- **Single-radius invariant (§3) — CLOSED.**" | 7630–7633 | moved | `runtime.md` §7 |
| "- **`RolloutConfig`/`TrainConfig` construction-default divergence — CLOSED by B1**" | 7634–7639 | moved to history | `implementation.md` §4 |
| "- **`min_target_distance_km` — RESOLVED for the construction path by B1**" | 7640–7652 | moved | `construction_fuel_damage.md` §6 |
| "- **Post-B3 headroom exists**" | 7653–7669 | moved to history | `measurements.md` §3 |
| "- **Raw utility 480 vs reward-side `U_oracle = 479.99968`**" | 7670–7675 | moved | `reward_solvers.md` §5 |
| "- **`match_aou.*` inherits `pyomo` from the ROOT package**" | 7676–7687 | moved | `runtime.md` §7 |
| "> **Repository hygiene / documentation alignment: CLOSED**" | 7688–7723 | moved to history | `implementation.md` §4 |
| "> **Flat-path cleanup phase: CLOSED.**" | 7725 | moved to history | `implementation.md` §4 |
| Section headings "## 1."–"## 8." and the file-map header row | 1, 12, 139, 199, 225, 425, 4027, 4029–4030, 4108, 6621 | replaced by the new structure; numbering kept discoverable | `CLAUDE.md` §8 compatibility index |

### 1.2 `graph_rl_project_handoff.md`

| Source block (opening string) | Base lines | Action | Destination |
|---|---|---|---|
| Title "# Multi-Agent Graph RL — THE GENERALIZED-V2 POPULATION (PR #57…" and "**Supersedes all earlier handoffs.**" | 1–3 | corrected | handoff title and status note |
| "Written 2026-08-11; updated 2026-08-14 …" (dated revision chain) | 5–136 | consolidated duplicate of the decision log; the pre-log dates preserved | `decisions.md` §1, §1.1 |
| "**THE LIVE STATE (2026-09-13, GENERALIZED-V2 BENCHMARK POST-MERGE DOCUMENTATION / LOCK)**" | 138–222 | corrected: current state rewritten (V2 development evidence supersedes "no V2 manifest / training run / result"; PR #60 merged); PR #59 facts consolidated | handoff §1–§7; `implementation.md` §2–§3; `training_benchmarks.md` §9 |
| "**THE PRECEDING LIVE STATE (2026-09-12…**", "(2026-09-06…", "**THE LIVE STATE (2026-09-05…**", "(2026-09-02, FINAL EARLY-STOPPING…", "(2026-09-01)", "(2026-08-31)", "(2026-08-30)", "(2026-08-26)" | 224–829 | consolidated duplicates of the dated decision-log rows, the implementation ledger and the measurement records | `decisions.md` §1; `implementation.md` §2–§3.1; `measurements.md` |
| "**THE STATE OF THE CLOSED PHASES, STATED PLAINLY.**" and "Baseline **difficulty selection is finished**" | 831–907 | consolidated duplicate | `measurements.md` §1–§3; `implementation.md` §2 |
| "**AND THE REPOSITORY WAS THEN CLOSED FOR HANDOFF — HISTORICAL (2026-08-23)**" | 909–925 | consolidated duplicate | `decisions.md` §1 (2026-08-23 rows); `implementation.md` §3 (PR #32, #33) |
| "This handoff is volatile and deliberately thin." | 927–929 | corrected | handoff status note |
| "## 0. How this workspace reads the repository" | 931–944 | corrected (`GPT_GITHUB` only; the mounted-main description retired) | `cc_review.md` §1–§3, §8 |
| "## 1. Current state" — supersession notes, "**LIVE PHASE (2026-09-02)**" and its Task 1–5, 5A/5B, R1, "four facts", early-stopping / resume / cluster sub-bullets | 946–1256 | consolidated duplicate; current facts corrected into the handoff | handoff; `implementation.md` §2–§3; `measurements.md` §2 |
| "- **A PREVIOUSLY EXECUTED OLD-CONTRACT CTDE MEASUREMENT IS OUT OF SCOPE FOR THIS PHASE.**" | 1257–1272 | consolidated; rule retained as active | `experiments.md` §3.3; handoff §4, §6; `decisions.md` §1 (undated row) |
| "- **BASE of THIS FINAL EARLY-STOPPING HANDOFF-STABILIZATION record:**" and the "HISTORICAL BASE of …" bullets | 1273–1340 | consolidated duplicate (base identities are merges in the ledger) | `implementation.md` §3, §3.1 |
| "- **PHASE-B CTDE CODE — CLOSED**" through "- **Repository code-hygiene cleanup — CLOSED**", "- **FINAL-CELL PROBE HARNESS**", "- **No scientific probe or training run had been performed**", "- **FIRST BOUNDED SHORT PROBE**", "- **Repository documentation hygiene**" | 1341–1641 | consolidated duplicates of the lock and measurement entries; reference roles retained | `implementation.md` §2; `measurements.md`; `environments_cleanup.md` §4.2 |
| "- **All PRIOR repository-hygiene work is CLOSED**" and "- **POST-MERGE REPOSITORY CLEANUP — DONE.**" | 1642–1653, 1795–1822 | moved to history (rewritten as a list) | `environments_cleanup.md` §4.7 |
| "- **VOLATILE STATE — WHAT IS DONE, WHAT IS WRITABLE, AND WHO OWNS IT (updated 2026-08-25).**" (writable state, references, cleanup-eligible branches, CTDE gate) | 1654–1794 | consolidated: ownership rules → procedure; refs → registry; gate → history | `cc_review.md` §2–§3; `environments_cleanup.md` §4.1–§4.7; `implementation.md` §2 (`a6f3aa9` entry); `decisions.md` §4 |
| "## 2. Historical probe — evidence about the EASY PRE-FD cell only" | 1824–1845 | consolidated duplicate | `measurements.md` §1, §2 (`a3f0838` record) |
| "## 3. What PR #8 closed", "## 3b. What PR #10 closed", "## 3c. What PR #14 closed" | 1847–1942 | consolidated duplicates | `implementation.md` §2; `construction_fuel_damage.md` §2; `artifacts_metrics.md` §1–§2; `training_benchmarks.md` §4 |
| "## 3d. The executed bounded short probe" (defect observations, reconstructions, code anchors) | 1946–2091 | moved to history | `measurements.md` §4 |
| "## 3e. The corrected-cell short-probe RERUN", "## 3f. The FIRST executed LONG BASELINE", "## 3h. The Phase-A LONG BASELINE (RERUN)", "## 3j. The FD-VARIABLE-SEVERITY-v1 ACTOR-ONLY BASELINE" | 2093–2309, 2357–2429, 2517–2602 | consolidated duplicates of the §7 measurement records, which carry the same tables, witnesses, caveats and supersessions; the playback-export caveat is also current | `measurements.md` §2; handoff §5 item 5 |
| "## 3g. What PR #24 closed", "## 3i. What PR #27 closed", "## 3k. What PR #30 closed" | 2311–2355, 2431–2515, 2604–2663 | consolidated duplicates | `implementation.md` §2; `training_benchmarks.md` §2; `construction_fuel_damage.md` §3; `artifacts_metrics.md` §3; `policy_ctde.md` §4 |
| "## 3l. GENERALIZED-V1 — the ACTIVE phase: APPROVED DESIGN" through "### 3l.9 What the redesign explicitly does NOT touch" | 2667–3019 | moved to history | `decisions.md` §3 |
| "## 3m." intro, "### 3m.1 The implementation stack", "### 3m.2 Writable ownership" | 3021–3113 | consolidated | `implementation.md` §3, §3.1; `cc_review.md` §2–§3 |
| "### 3m.3 Engineering validation — Task 5A and Task 5B" | 3115–3157 | consolidated duplicate | `measurements.md` §2 (Task 5A / 5B record); `experiments.md` §1 |
| "### 3m.4 The first GENERALIZED-V1 ACTOR-ONLY long run (R1)" — "**THE FROZEN PLAN**" table | 3170–3193 | moved to history | `measurements.md` §5 |
| "### 3m.4" — state, "FOUR DIFFERENT FACTS", review requirement, early-stopping and cluster notes | 3159–3169, 3194–3252 | consolidated (superseded by the R1 record) | `measurements.md` §2; `training_benchmarks.md` §7; `environments_cleanup.md` §3 |
| "### 3m.5 The integration sequence — PERFORMED" | 3254–3289 | consolidated; the exact-base re-review rule retained as procedure | `implementation.md` §3.1; `cc_review.md` §3 |
| "### 3m.6 BGU cluster execution environment — VALIDATED / READY (VOLATILE OPERATIONS)" | 3293–3381 | moved (dated observation) | `environments_cleanup.md` §3 |
| "### 3m.7 Opt-in training-reward early stopping" | 3383–3479 | consolidated | `training_benchmarks.md` §7; `implementation.md` §3.1 |
| "## 3n." — "### 3n.1 R1", "### 3n.2 The diagnostic replay", "### 3n.3 PR #52" | 3481–3614 | consolidated duplicates | `measurements.md` §2 (R1 record); `implementation.md` §2 (`81a148f` entry); `artifacts_metrics.md` §5, §6.3 |
| "## 3o." — "### 3o.1 PR #54", "### 3o.2 PR #55", "### 3o.3 The pre-existing frozen-BLADE behaviour" | 3617–3713 | consolidated duplicates | `reward_solvers.md` §3; `construction_fuel_damage.md` §4; `CLAUDE.md` §2; `implementation.md` §2 |
| "### 3o.4 The ABORTED P1 arm — `ABORTED / DO NOT RESUME`" | 3717–3745 | moved to history; status retained as current | `measurements.md` §6.1; handoff §4, §6 |
| "### 3o.5 R1 is untouched", "### 3p.3 R1 is untouched" | 3747–3755, 3847–3855 | consolidated | handoff §4; `experiments.md` §3.3 |
| "## 3p." intro and "### 3p.1 PR #57" | 3758–3801 | consolidated duplicate | `implementation.md` §2 (`a27a3b1` entry); `training_benchmarks.md` §8 |
| "### 3p.2 The MINIMAL research provenance — why V2 exists" | 3805–3845 | moved to history; the no-causal-inference rule retained as active | `measurements.md` §6.2; `experiments.md` §3.3 |
| "### 3p.4 The V2 evaluation / benchmark design question" | 3859–3890 | moved to history | `decisions.md` §5 |
| "## 3q." — "### 3q.1" … "### 3q.5 Writable ownership" | 3892–3961 | consolidated; 3q.3–3q.5 current-state claims corrected (superseded by the V2 development evidence; PR #60 merged) | `implementation.md` §2 (`786e821` entry); `training_benchmarks.md` §9; handoff |
| "## 4. Current work" — supersession notes, current and preceding records, ordering, ownership, Tasks 0–8, Task 10 | 3964–4261, 4298–4463 | consolidated; Task 10's pre-measurement inspection and comparator rules retained as procedure | `decisions.md` §1, §4; `measurements.md`; `experiments.md` §2, §3.2–§3.3 |
| "**Task 9 — THE FIRST CONTROLLED ACTOR-ONLY vs CTDE SCIENTIFIC COMPARISON ON THE OLD FIXED CELL**" | 4213–4260 | consolidated duplicate of `CLAUDE.md` 7474–7547; binding rules retained | `decisions.md` §4; `experiments.md` §3.3 |
| "**What makes a run VALID — carried forward unchanged**" and the negative-result paragraph | 4262–4278 | moved | `experiments.md` §3.1 |
| "**Interpretation rules survive unchanged:**" | 4280–4296 | moved | `experiments.md` §3.2 |
| "## 5. Closed decisions" | 4466–4575 | moved to history | `decisions.md` §2 |
| "## 6. Out of scope for the current work" (bullets and the "FUTURE DESIGN TOPIC" paragraph) | 4577–4747 | consolidated and retained as active prohibitions; stale "current work" framing dropped | `experiments.md` §5; handoff §6; `environments_cleanup.md` §4; `decisions.md` §4 |
| "## 7. Documentation duties" (table) | 4749–4789 | consolidated duplicate: every DONE row's content is the corresponding lock or measurement entry; the forward rule corrected | `implementation.md`; `measurements.md`; `cc_review.md` §5 |
| "## 8. Next action" — the 2026-09-13, 2026-09-12, 2026-09-06 and 2026-09-05 blocks, the R1-review-era block and its ownership text | 4791–5123 | corrected (current next actions rewritten); history consolidated | handoff §5–§6; `decisions.md` §1 |
| "**FUTURE CAMPAIGN ITEMS REMAIN OPEN, UNDECIDED AND NOT AUTHORIZED**" (the five items) | 5124–5153 | moved to history; the items remain open | `decisions.md` §6; handoff §6 |
| "## 8." — scale/authorization restatement, "**THE REST OF THIS SECTION IS THE PRESERVED RECORD OF THE CLOSED PHASES**", "**THE NEXT REPOSITORY ACTION, RESTATED ONCE AND ONLY ONCE**" | 5154–5367 | consolidated duplicate | `measurements.md`; `decisions.md` §1 |
| "### 9.1 How a receiving orchestrator initializes — MANDATORY, IN THIS ORDER" | 5371–5394 | corrected (procedure) | `cc_review.md` §8 |
| "### 9.2 Decision log — append one dated entry per MATERIAL change" | 5398–5445 | moved to history; two rows appended | `decisions.md` §1 |

### 1.3 Other documents

| Document and block | Action | Destination |
|---|---|---|
| `README.md` intro ("A MATCH-AOU MINLP solver produces one optimal allocation offline") and §1 oracle bullet | corrected: solver-neutral; reference depends on the design | `README.md` intro, §1 |
| `README.md` §2 diagram ("MATCH-AOU solve ──> oracle") and "**Policy.**" paragraph | corrected: reference solve; actor-only default with optional training-only CTDE | `README.md` §2 |
| `README.md` §3 rows "**Event-triggered**" and "**Actor-only PPO**" | corrected: post-FD boundary wake; decentralized execution with optional CTDE | `README.md` §3 |
| `README.md` §4 "## 4. Current experiment cell" (including "**No long baseline has been run on this cell**") | corrected: the three episode designs, the two backends, where results live; the old statement marked historical | `README.md` §4 |
| `README.md` §5 layout | corrected: docs tree, `environment.cluster.yml`, solver modules, `central_graph_builder.py`, `graph_generalized.py`, `graph_benchmark_preflight.py`, backend benchmark tool | `README.md` §5 |
| `README.md` §6 environment | corrected: two execution contexts, solver requirements, `PYTHONNOUSERSITE=1` | `README.md` §6 |
| `README.md` §7 entry points, preset description and options table | corrected: preflight entry point; the preset described as a fixed-cell probe; design, backend, training-mode, budget, benchmark and early-stopping options | `README.md` §7 |
| `README.md` §8 outputs, plots and the "attempted at most once" paragraph | corrected: `episode_outcomes.jsonl`, optional FD-sensitivity figure, generalized quota accounting, reference-normalized reward | `README.md` §8 |
| `README.md` §9 documentation map | corrected | `README.md` §9 |
| `docs/BLADE_API_DOCUMENTATION.md` §13 "**Fuel burns every tick for every airborne aircraft**, including ones with no route." | corrected narrowly | same location |
| `environment.cluster.yml`, `requirements.txt` | unchanged (byte-identical); their `CLAUDE.md §1` / `§2` references resolve through the compatibility index | `CLAUDE.md` §8 |

## 2. Changed technical claims

| Claim | Where | Code symbol | Test or evidence anchor |
|---|---|---|---|
| An airborne aircraft is not guaranteed a movement-and-burn visit every tick, because landing and fuel exhaustion remove entries from the live list being iterated | BLADE doc §13 | `Game.update_all_aircraft_position` (`aircraft.current_fuel -= aircraft.fuel_rate / 3600`), `Game.land_aicraft` → `Game.remove_aircraft`, the `current_fuel <= 0` branch | `tests/test_graph_fuel_damage.py::test_g1_11b_the_absolute_outer_tick_is_diagnostic_never_binding`; `CLAUDE.md` §2 |
| `run_summary.json:/generalized/cardinality_sampler` names the V1 sampler for V2 runs; `run_config.json` names the correct V2 sampler | `artifacts_metrics.md` §6.1; handoff §4; `measurements.md` §7 | `graph_train._generalized_summary` exact string `"cardinality_sampler": cardinality_sampler_record(),`; `graph_train.write_run_config` selecting `generalized_v2_cardinality_sampler_record()` when `cfg.route_relative_population` | both `run_config.json` / `run_summary.json` pairs at PR #61 `1375a88…` and PR #62 `b2bbe7a…`; PR #62 `artifact_sha256.txt`. No test covers the summary key (gap §4) |
| Each evidence commit's single parent is the measured SHA, and its tree adds only `research_evidence/generalized_v2/…` | `artifacts_metrics.md` §6.2; handoff §3; `measurements.md` §7 | `run_config.json:/provenance/git/commit` | `git log -1 --format=%P` of both heads = `ae42cb01677f94868b2873008d87be677e31f0c8`; `git show --name-only` of both heads |
| V2 development arms' frozen contract and accounting (design, backend, profile, manifest id, schedule, seeds, cadence, 3008/3000/8, 960/960/0, 16 rounds, 375 updates, 3960 records) | handoff §4; `measurements.md` §7; `decisions.md` §1 | `TrainConfig` fields; `run_summary.json` keys `train_episodes_attempted` … `episode_outcomes_recorded` | the committed `run_config.json`, `run_summary.json` and `artifact_sha256.txt` at both evidence heads |
| Verdict provenance: CTDE verdict attributed to PR #62's own text; no actor-only verdict recorded; no GitHub review record for either PR | handoff §4; `policy_ctde.md` §4.1; `measurements.md` §7 | — | `gh pr view 61 / 62` bodies; PR #62 `artifact_sha256.txt` header |
| CTDE is implemented and opt-in; `actor_only` is the default; evaluation is actor-only | `README.md` §2, §3, §7 | `TrainConfig.training_mode = TRAINING_MODE_ACTOR_ONLY`, `TRAINING_MODES`, `--training-mode` | `tests/test_graph_ctde.py::test_actor_only_never_constructs_a_critic_or_a_central_state`, `::test_the_poison_is_live_a_ctde_run_hits_it` |
| Three episode designs; `fixed_cell_v1` is the default; `generalized_v2` requires `p1_milp_v1` and refuses `legacy_minlp_v1` | `README.md` §4, §7 | `graph_generalized.EPISODE_DESIGNS`, `EPISODE_DESIGN_FIXED_CELL_V1`, `TrainConfig.episode_design`, `GENERALIZED_V2_REQUIRED_BACKEND` checked in `TrainConfig.validate` | `tests/test_graph_generalized_v2.py::test_po2_the_design_requires_the_p1_objective_on_both_harnesses`, `::test_po2_the_backend_contract_is_stated_as_design_constrained_not_independent` |
| Two MATCH-AOU backends, `legacy_minlp_v1` default, explicit selection with no fallback; the P1 backend uses SciPy `milp` (HiGHS) | `README.md` §4, §6; `environments_cleanup.md` §2 | `solvers.match_aou_backend.MATCH_AOU_BACKENDS`, `DEFAULT_MATCH_AOU_BACKEND`; `match_aou_p1_milp_solver` (`from scipy.optimize import Bounds, LinearConstraint, milp`) | `tests/test_match_aou_backend_integration.py::test_po1_every_default_is_the_historical_legacy_backend`; `tests/test_match_aou_p1_milp_solver.py` |
| The V2 benchmark profiles are `development` and `confirmatory`; the preflight is a module entry point | `README.md` §4, §7 | `graph_generalized.V2_BENCHMARK_PROFILES`; `graph_benchmark_preflight.main` and its `__main__` guard | `tests/test_graph_generalized_v2_benchmark.py::test_po3_the_v2_preflight_config_requires_p1_and_twelve_worlds` |
| On generalized designs `episodes_per_iteration` is a successful-episode quota and a failed attempt spends its seed and is replaced | `README.md` §7, §8 | `TrainConfig.training_attempt_policy`, `graph_train.train_attempt_seed` | `tests/test_graph_train.py::test_task5c_a_generalized_iteration_fills_its_successful_quota`, `::test_task5c_a_failed_seed_is_spent_and_recorded_once` |
| Early stopping is opt-in and approved for `generalized_v1` only | `README.md` §7 | `TrainConfig.early_stopping`, `TrainConfig.validate` (`design.generalized_v1_design`) | `tests/test_graph_train.py::test_es_disabled_is_fixed_budget_and_adds_no_record_noise`, `::test_es_the_monitor_reads_training_reward_and_nothing_else` |
| Runs write `episode_outcomes.jsonl` and may write the optional `fd_policy_sensitivity.png` | `README.md` §8 | `graph_train._EPISODE_OUTCOMES_FILENAME`, `_PLOT_FD_SENSITIVITY` | `artifacts_metrics.md` §3, §5 contracts |
| The diagnostic rollout accepts `--episode-design`, `--fuel-damage-mode` and `--match-aou-backend` | `README.md` §7 | `graph_rollout` argument parser | — |
| Statements inside moved contract blocks that something "does not exist" describe that block's PR scope at merge | every contract header | — | reading rule, not a behaviour claim |

## 3. Procedural supersessions

All are user-approved in the packet ("Approved procedure changes").

| # | Former procedure (source) | Now | Where |
|---|---|---|---|
| 1 | Transport `CLAUDE_MOUNTED_MAIN`: direct push to `main`, reviewed after the push, with Grade-A routing exceptions (`CLAUDE.md` §1; handoff §0) | retired; `GPT_GITHUB` branch plus draft PR is the only live transport | `cc_review.md` §3 |
| 2 | "one focused commit per task" (`CLAUDE.md` §1) | several focused commits are allowed | `cc_review.md` §3 |
| 3 | a universal long status block ending every task (`CLAUDE.md` §1) | proportional reporting with a short required list | `cc_review.md` §6 |
| 4 | Grade C "trusted, no review, no lock ceremony" (`CLAUDE.md` §1) | every grade receives exact-candidate review; historical grade labels keep their meaning | `cc_review.md` §4 |
| 5 | fix chain: new commits, never amend, rebase or force-push (`CLAUDE.md` §1) | retained, extended explicitly to squash, and binding once review begins | `cc_review.md` §3 |
| 6 | separate documentation-lock and post-merge closure PRs after code merges (the practice recorded throughout the handoff) | code and the documentation it makes stale change in the same branch and PR; no separate post-merge documentation task is required | `cc_review.md` §5; `CLAUDE.md` §1 |
| 7 | the hash convention requiring each lock's SHA to be recorded by the next commit (`CLAUDE.md` §7) | integration SHAs are recorded only when materially needed; a document never names its own SHA; no task is opened only to record one | `cc_review.md` §5; `implementation.md` §1 |
| 8 | the handoff stacked dated supersession blocks above one another (handoff preamble, §1, §4, §8) | the handoff is one compact snapshot whose stale lines are replaced; history lives in `docs/history/` | handoff status note |
| 9 | "packet or user must state which transport mode applies" (`CLAUDE.md` §1) | superseded by row 1: no mode choice remains | `cc_review.md` §3 |
| 10 | the receiving-orchestrator protocol (handoff §9.1) | retained | `cc_review.md` §8 |
| 11 | output discipline and scope discipline (`CLAUDE.md` §1) | retained | `cc_review.md` §6–§7; `CLAUDE.md` §1 |
| 12 | no required checklist for documentation-only tasks | link and anchor resolution, symbol existence, contradiction search, `git diff --check`, unchanged non-Markdown blobs | `cc_review.md` §9 |

## 4. Outstanding gaps

Each gap is scoped to what the repository and the accessible GitHub state can show on
2026-09-14.

- **Research decision.** The preceding research chat's final decision on the GENERALIZED-V2
  development arms is not available: whether the actor-only arm was reviewed, what comparison
  conclusion (if any) was drawn, and whether confirmatory-profile use is authorized or has
  occurred.
- **Verdicts.** No verdict is recorded for the V2 actor-only development arm. The CTDE verdict
  is attributed to PR #62's own text; GitHub holds no review or comment record for PR #61 or PR
  #62, and this restructure does not re-approve either measurement or either evidence package.
- **Benchmark preflight.** Its authorization and review record are absent. The external
  manifest file was inspected neither by GPT nor here; its identity is recorded as PR #62 states
  it, and the global status of confirmatory data is not inferred.
- **Unrecorded identities.** The R1 run directory, the fresh deterministic-P1 arm's artifacts
  and the old fixed-cell CTDE measurement's identity are not recorded in the repository.
- **Ownership episodes.** Who removed the branches the former handoff listed as
  cleanup-eligible, and when, is not recorded; the roles of three local worktrees are not
  recorded.
- **Old numbering inside moved text.** Moved contract and history blocks still cite former
  section numbers (`§5`, `§8`, `§3l`, …) and carry PR-scoped present-tense statements. They are
  resolved by the compatibility index and by each document's reading note rather than rewritten,
  so no requirement was changed while moving it. Source-code comments that cite
  `CLAUDE.md` section numbers were not edited (out of scope) and resolve through the same index.
- **A stale pointer that predates this restructure.** `graph_hidden_placement.py` cites
  `graph_rl_project_handoff.md, "Route prediction"`; no such heading existed at the base either.
  The matching text is the closed decision "Route prediction is required and supports
  `num_agents < n_known`" (`decisions.md` §2).
- **Test coverage.** No test pins the `generalized.cardinality_sampler` summary label.
- **This PR's number** is not written into the handoff table, to avoid a self-referential
  update cycle; resolve it on GitHub from the branch name.

## 5. Sizes and reading routes

**Before** (base): `CLAUDE.md` 684,752 bytes and the handoff 531,370 bytes were both mandatory for
every task — **1,216,122 bytes** before any task-specific reading. `README.md` was 22,808 bytes and
the BLADE doc 25,322 bytes.

**After** (this candidate):

| File | Bytes |
|---|---:|
| `CLAUDE.md` | 24,381 |
| `graph_rl_project_handoff.md` | 9,439 |
| `README.md` | 25,402 |
| `docs/BLADE_API_DOCUMENTATION.md` | 26,902 |
| `docs/workflows/cc_review.md` | 7,255 |
| `docs/workflows/experiments.md` | 11,275 |
| `docs/workflows/environments_cleanup.md` | 16,968 |
| `docs/contracts/runtime.md` | 49,627 |
| `docs/contracts/construction_fuel_damage.md` | 58,855 |
| `docs/contracts/policy_ctde.md` | 27,517 |
| `docs/contracts/reward_solvers.md` | 51,345 |
| `docs/contracts/training_benchmarks.md` | 152,859 |
| `docs/contracts/artifacts_metrics.md` | 50,201 |
| `docs/history/implementation.md` | 159,551 |
| `docs/history/measurements.md` | 112,087 |
| `docs/history/decisions.md` | 134,078 |

The mandatory core is now **33,820 bytes** (`CLAUDE.md` plus the handoff).

| Reading route | Documents | Bytes |
|---|---|---:|
| Focused implementation (for example an executor change) | core + `cc_review.md` + `runtime.md` | 90,702 |
| Experiment planning | core + `experiments.md` + `training_benchmarks.md` + `artifacts_metrics.md` + `environments_cleanup.md` | 265,123 |
| Run review | core + `experiments.md` + `artifacts_metrics.md` (+ the one relevant record in `measurements.md`) | 95,296 (+ record) |
| Evidence preservation | core + `experiments.md` + `environments_cleanup.md` | 62,063 |
| Routine cleanup | core + `environments_cleanup.md` + `cc_review.md` | 58,043 |

Across the four pre-existing documents the base held 1,264,252 bytes; the sixteen files above
hold 917,742 bytes, and this record adds its own size on top. The reduction comes from
consolidated duplicates, whose wording stays readable at the base; no unique fact was dropped.

## 6. Checks performed

- Coverage: no base line copied twice; the non-blank `CLAUDE.md` lines not copied are exactly the
  title and section headings, the file-map header row (4029–4030), the rewritten lines 3–8 and
  170–177, and the consolidated phase-state bullets within 6623–7121, all accounted for in §1.1.
- Internal links and GitHub-style anchors across `CLAUDE.md`, `README.md`, the handoff and every
  file under `docs/` resolve.
- Every symbol named in §2 exists at the base.
- A search of the normative documents for stale current-state wording (mounted-main transport,
  one-commit rule, sole-writable claims, "no long baseline", "no V2 training run") finds only
  explicitly historical mentions.
- `git diff --check` is clean, and the diff against the base touches only the declared Markdown
  files.
